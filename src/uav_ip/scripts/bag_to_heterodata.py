#!/usr/bin/env python3
"""
bag_to_heterodata.py — Convierte Rosbags (.mcap) a HeteroData para Instant Policy.

Genera grafos heterogeneos de PyTorch Geometric utilizando nubes de puntos
LiDAR 360° sinteticas generadas por observacion privilegiada (synthetic_lidar.py),
deprecando la extraccion de nubes pesadas PointCloud2 del sensor.

Formato del Grafo (HeteroData):
  - data['scene'].pos: Tensor (T, 2048, 3) con nubes LiDAR 360° en marco egocentrico FLU escaladas (s=0.1)
  - data['agent'].x:   Tensor (T, 3) con posiciones del dron escaladas (s=0.1)
  - data['agent'].y:   Tensor (T, 3) con acciones objetivo / setpoints escalados (s=0.1)
  - data['agent', 'temporal', 'agent'].edge_index: Conexiones temporales t -> t+1

Uso:
    python3 bag_to_heterodata.py --bag-dir /ruta/al/rosbag --waypoints /ruta/a/waypoints.yaml --output data.pt
"""

import argparse
import math
import sys
from pathlib import Path
from typing import Optional, Dict, Any, List

import numpy as np
import yaml

# Asegurar inclusion del directorio de paquetes de usuario (~/.local/lib/python3.X/site-packages)
# por si el entorno ROS 2 o WSL2 desactivo o no heredo ENABLE_USER_SITE
user_site = Path.home() / ".local" / "lib" / f"python{sys.version_info.major}.{sys.version_info.minor}" / "site-packages"
if user_site.is_dir() and str(user_site) not in sys.path:
    sys.path.insert(0, str(user_site))

# Importar synthetic_lidar
try:
    from synthetic_lidar import (
        SyntheticLidarGenerator,
        px4_odometry_to_gazebo_enu,
        ned_to_gazebo_enu,
        yaw_ned_to_gazebo_enu,
        quaternion_to_rot_matrix,
        fps_numpy
    )
except ImportError:
    # Si se ejecuta desde otra carpeta, buscar en el mismo directorio del script
    sys.path.append(str(Path(__file__).resolve().parent))
    from synthetic_lidar import (
        SyntheticLidarGenerator,
        px4_odometry_to_gazebo_enu,
        ned_to_gazebo_enu,
        yaw_ned_to_gazebo_enu,
        quaternion_to_rot_matrix,
        fps_numpy
    )

# Librerias opcionales de rosbags (soporta rosbags >= 0.10, 0.11 y versiones legadas)
_rosbags_err = None
AnyReader = None
Reader = None
_typestore = None
deserialize_cdr = None

try:
    from rosbags.highlevel import AnyReader
except ImportError as e:
    _rosbags_err = e

try:
    from rosbags.typesys import Stores, get_typestore
    _typestore = get_typestore(getattr(Stores, "ROS2_JAZZY", Stores.LATEST))
except ImportError:
    pass

try:
    from rosbags.rosbag2 import Reader
except ImportError as e:
    if _rosbags_err is None:
        _rosbags_err = e

try:
    from rosbags.serde import deserialize_cdr
except ImportError:
    pass

if AnyReader is None and Reader is None:
    try:
        import rosbags
    except ImportError as e:
        _rosbags_err = e

try:
    import torch
except ImportError:
    torch = None

try:
    from torch_geometric.data import HeteroData
except ImportError:
    HeteroData = None

# Topicos
TOPIC_ODOM = "/fmu/out/vehicle_odometry"
TOPIC_LOCAL_POS = "/fmu/out/vehicle_local_position_v1"
TOPIC_SETPOINT = "/fmu/in/trajectory_setpoint"
TOPIC_PCL_DEPRECATED = "/depth_camera/points"


def find_waypoints_file(bag_path: Path, explicit_path: Optional[Path] = None) -> Optional[Path]:
    """Busca waypoints.yaml en la ruta especificada o en directorios adyacentes."""
    if explicit_path and explicit_path.is_file():
        return explicit_path

    # Candidatos automaticos tanto si bag_path es directorio como archivo
    candidates = []
    if bag_path.is_dir():
        candidates.append(bag_path / "waypoints.yaml")
    candidates.append(bag_path.parent / "waypoints.yaml")
    candidates.append(bag_path.parent.parent / "waypoints.yaml")

    for c in candidates:
        if c.is_file():
            return c
    return None


def parse_bag_telemetry(bag_path: Path) -> Dict[str, List[Dict[str, Any]]]:
    """
    Parsea los mensajes de telemetria de PX4 desde el archivo .mcap.
    Soporta rosbags 0.11+ (AnyReader / Typestore) y versiones anteriores.
    """
    if AnyReader is None and Reader is None:
        detail = f": {_rosbags_err}" if _rosbags_err else ""
        raise ImportError(f"No se pudo cargar la libreria 'rosbags'{detail}. Ejecuta: pip install --user --break-system-packages rosbags")

    print(f"Leyendo rosbag desde: {bag_path}")
    data = {
        "odometry": [],
        "local_pos": [],
        "setpoints": []
    }

    bag_p = Path(bag_path)

    # Preferir AnyReader si esta disponible (gestiona esquemas MCAP y tipados automaticamente)
    if AnyReader is not None:
        # AnyReader acepta una lista de rutas (archivos o directorios)
        if bag_p.is_file():
            target_paths = [bag_p]
        elif bag_p.is_dir():
            mcaps = list(bag_p.glob("*.mcap"))
            target_paths = mcaps if mcaps else [bag_p]
        else:
            target_paths = [bag_p]

        with AnyReader(target_paths, default_typestore=_typestore) as reader:
            for connection, timestamp, rawdata in reader.messages():
                topic = connection.topic

                if topic == TOPIC_ODOM:
                    msg = reader.deserialize(rawdata, connection.msgtype)
                    pos = list(msg.position) if hasattr(msg, "position") else [0.0, 0.0, 0.0]
                    q = list(msg.q) if hasattr(msg, "q") else [1.0, 0.0, 0.0, 0.0]
                    vel = list(msg.velocity) if hasattr(msg, "velocity") else [0.0, 0.0, 0.0]
                    data["odometry"].append({
                        "timestamp": timestamp,
                        "position": pos,
                        "q": q,
                        "velocity": vel
                    })

                elif topic == TOPIC_LOCAL_POS:
                    msg = reader.deserialize(rawdata, connection.msgtype)
                    x = float(msg.x) if hasattr(msg, "x") else 0.0
                    y = float(msg.y) if hasattr(msg, "y") else 0.0
                    z = float(msg.z) if hasattr(msg, "z") else 0.0
                    heading = float(msg.heading) if hasattr(msg, "heading") else 0.0
                    data["local_pos"].append({
                        "timestamp": timestamp,
                        "x": x,
                        "y": y,
                        "z": z,
                        "heading": heading
                    })

                elif topic == TOPIC_SETPOINT:
                    msg = reader.deserialize(rawdata, connection.msgtype)
                    sp_pos = list(msg.position) if hasattr(msg, "position") else [0.0, 0.0, 0.0]
                    sp_yaw = float(msg.yaw) if hasattr(msg, "yaw") else 0.0
                    data["setpoints"].append({
                        "timestamp": timestamp,
                        "position": sp_pos,
                        "yaw": sp_yaw
                    })
    else:
        # Fallback con Reader clasico
        deserializer = _typestore.deserialize_cdr if _typestore is not None else deserialize_cdr
        if deserializer is None:
            raise ImportError("No se encontro un deserializador CDR en rosbags. Actualiza rosbags.")

        with Reader(bag_path) as reader:
            for connection, timestamp, rawdata in reader.messages():
                topic = connection.topic

                if topic == TOPIC_ODOM:
                    msg = deserializer(rawdata, connection.msgtype)
                    pos = list(msg.position) if hasattr(msg, "position") else [0.0, 0.0, 0.0]
                    q = list(msg.q) if hasattr(msg, "q") else [1.0, 0.0, 0.0, 0.0]
                    vel = list(msg.velocity) if hasattr(msg, "velocity") else [0.0, 0.0, 0.0]
                    data["odometry"].append({
                        "timestamp": timestamp,
                        "position": pos,
                        "q": q,
                        "velocity": vel
                    })

                elif topic == TOPIC_LOCAL_POS:
                    msg = deserializer(rawdata, connection.msgtype)
                    x = float(msg.x) if hasattr(msg, "x") else 0.0
                    y = float(msg.y) if hasattr(msg, "y") else 0.0
                    z = float(msg.z) if hasattr(msg, "z") else 0.0
                    heading = float(msg.heading) if hasattr(msg, "heading") else 0.0
                    data["local_pos"].append({
                        "timestamp": timestamp,
                        "x": x,
                        "y": y,
                        "z": z,
                        "heading": heading
                    })

                elif topic == TOPIC_SETPOINT:
                    msg = deserializer(rawdata, connection.msgtype)
                    sp_pos = list(msg.position) if hasattr(msg, "position") else [0.0, 0.0, 0.0]
                    sp_yaw = float(msg.yaw) if hasattr(msg, "yaw") else 0.0
                    data["setpoints"].append({
                        "timestamp": timestamp,
                        "position": sp_pos,
                        "yaw": sp_yaw
                    })

    print(f"  Mensajes leidos: {len(data['odometry'])} odom, "
          f"{len(data['local_pos'])} local_pos, {len(data['setpoints'])} setpoints.")
    return data


def find_flight_trim_indices(
    odom_list: List[Dict[str, Any]],
    using_local_pos: bool = False,
    trim_start: str = "takeoff_exit",
    min_takeoff_alt: float = 1.0,
    liftoff_alt: float = 0.4,
    touchdown_alt: float = 0.38
) -> tuple[int, int]:
    """
    Detecta los indices de inicio y fin de la ventana util de vuelo:
    - start_idx: Cuando el dron completa el despegue e inicia el avance hacia el circuito
                 (o cuando despega si trim_start='liftoff').
    - end_idx: Cuando el dron aterriza (touchdown) en la plataforma final.
    """
    N = len(odom_list)
    if N == 0:
        return 0, 0

    z_enu = np.zeros(N, dtype=np.float64)
    dist_xy = np.zeros(N, dtype=np.float64)

    for i, item in enumerate(odom_list):
        if not using_local_pos:
            pos = item["position"]  # NED
            z = -float(pos[2])      # ENU
            x = float(pos[1])
            y = float(pos[0])
        else:
            z = float(item["z"])
            x = float(item["y"])
            y = float(item["x"])
        z_enu[i] = z
        dist_xy[i] = math.sqrt(x * x + y * y)

    # 1. Inicio de vuelo util
    if trim_start == "none":
        start_idx = 0
    elif trim_start == "liftoff":
        idx_candidates = np.where(z_enu >= liftoff_alt)[0]
        start_idx = int(idx_candidates[0]) if len(idx_candidates) > 0 else 0
    else:  # "takeoff_exit" (default)
        at_alt = np.where(z_enu >= min_takeoff_alt)[0]
        if len(at_alt) > 0:
            first_alt = int(at_alt[0])
            moving = np.where(dist_xy[first_alt:] >= 0.25)[0]
            if len(moving) > 0:
                start_idx = int(first_alt + moving[0])
            else:
                start_idx = first_alt
        else:
            start_idx = 0

    # 2. Fin de vuelo util (touchdown)
    if trim_start == "none":
        end_idx = N - 1
    else:
        max_dist_idx = int(np.argmax(dist_xy))
        after_circuit = np.where((np.arange(N) >= max_dist_idx) & (z_enu <= touchdown_alt))[0]
        if len(after_circuit) > 0:
            end_idx = min(N - 1, int(after_circuit[0] + 10))
        else:
            end_idx = N - 1

    if end_idx <= start_idx:
        end_idx = N - 1
        start_idx = 0

    return start_idx, end_idx


def build_heterodata_from_synthetic_lidar(
    parsed_data: Dict[str, Any],
    waypoints_file: Path,
    output_path: Path,
    scale_factor: float = 0.1,
    r_max: float = 10.0,
    fps_points: int = 2048,
    subsample_step: int = 5,
    trim_idle: bool = True,
    trim_start: str = "takeoff_exit",
    min_takeoff_alt: float = 1.0
) -> bool:
    """
    Construye el HeteroData usando SyntheticLidarGenerator para sintetizar las
    nubes de puntos LiDAR 360° en cada instante temporal de la trayectoria.
    """
    # Cargar waypoints y landmarks (con soporte tanto para 'landmarks' como 'hoops')
    with open(waypoints_file, "r") as f:
        wp_cfg = yaml.safe_load(f)

    raw_lms = wp_cfg.get("landmarks") or wp_cfg.get("hoops") or []
    landmarks = []
    for lm in raw_lms:
        lm_c = dict(lm)
        if "id" not in lm_c and "name" in lm_c:
            lm_c["id"] = lm_c["name"]
        if "type" not in lm_c:
            lm_c["type"] = "square_gate"
        landmarks.append(lm_c)
    landing = wp_cfg.get("landing", None)

    print(f"Cargados {len(landmarks)} landmarks desde {waypoints_file.name}")

    lidar_gen = SyntheticLidarGenerator(
        r_max=r_max,
        scale_factor=scale_factor,
        target_points=fps_points
    )

    odom_list = parsed_data.get("odometry", [])
    if not odom_list:
        # Fallback a local_pos si odometry estuviera vacia
        odom_list = parsed_data.get("local_pos", [])
        using_local_pos = True
    else:
        using_local_pos = False

    if not odom_list:
        print("ERROR: No se encontraron mensajes de odometria ni posicion en el rosbag.")
        return False

    # Recorte de momentos estaticos en tierra (espera de armado pre-despegue y post-aterrizaje)
    if trim_idle:
        start_idx, end_idx = find_flight_trim_indices(
            odom_list,
            using_local_pos=using_local_pos,
            trim_start=trim_start,
            min_takeoff_alt=min_takeoff_alt
        )
        total_raw = len(odom_list)
        odom_list = odom_list[start_idx:end_idx + 1]
        print(f"  Recorte de vuelo util aplicado: [{start_idx}:{end_idx}] "
              f"({len(odom_list)} de {total_raw} muestras conservadas).")

    # Submuestreo temporal uniforme para no saturar con 250Hz nativos de PX4
    sampled_indices = list(range(0, len(odom_list), max(1, subsample_step)))
    T = len(sampled_indices)
    print(f"Generando nubes LiDAR sinteticas 360° para T={T} instantes temporales...")

    agent_positions = []
    agent_rotations = []
    scene_pointclouds = []

    for idx in sampled_indices:
        item = odom_list[idx]
        if not using_local_pos:
            pos_ned = item["position"]
            q_ned = item["q"]
            p_enu, R_enu = px4_odometry_to_gazebo_enu(pos_ned, q_ned)
            drone_pose = {
                "x": float(p_enu[0]),
                "y": float(p_enu[1]),
                "z": float(p_enu[2]),
                "R": R_enu,
                "q_ned": [float(q_ned[0]), float(q_ned[1]), float(q_ned[2]), float(q_ned[3])]
            }
        else:
            p_ned = [item["x"], item["y"], item["z"]]
            p_enu = ned_to_gazebo_enu(p_ned)
            yaw_enu = yaw_ned_to_gazebo_enu(item["heading"])
            drone_pose = {
                "x": float(p_enu[0]),
                "y": float(p_enu[1]),
                "z": float(p_enu[2]),
                "yaw": float(yaw_enu)
            }

        # Generar nube LiDAR 360° con escala s = 0.1
        cloud_scaled = lidar_gen.generate_pointcloud(drone_pose, landmarks, landing)
        scene_pointclouds.append(cloud_scaled)

        # Posicion del agente en ENU escalada por s = 0.1
        agent_positions.append(p_enu * scale_factor)

    agent_positions = np.array(agent_positions, dtype=np.float32)       # (T, 3)
    scene_pointclouds = np.array(scene_pointclouds, dtype=np.float32)   # (T, 2048, 3)

    # Procesar acciones / setpoints con alineacion temporal exacta
    setpoints = parsed_data.get("setpoints", [])
    if setpoints:
        sp_timestamps = np.array([sp["timestamp"] for sp in setpoints], dtype=np.int64)
        sp_positions = np.array([ned_to_gazebo_enu(sp["position"]) * scale_factor for sp in setpoints], dtype=np.float32)

        target_actions = np.zeros((T, 3), dtype=np.float32)
        for i, idx in enumerate(sampled_indices):
            t = odom_list[idx]["timestamp"]
            sp_idx = np.searchsorted(sp_timestamps, t, side="right") - 1
            sp_idx = int(np.clip(sp_idx, 0, len(sp_positions) - 1))
            target_actions[i] = sp_positions[sp_idx]
    else:
        # Si no hay setpoints, usar deltas de la odometria real t+1 - t como targets
        if T > 1:
            deltas = np.zeros_like(agent_positions)
            deltas[:-1] = agent_positions[1:] - agent_positions[:-1]
            deltas[-1] = deltas[-2]
            target_actions = deltas
        else:
            target_actions = np.zeros_like(agent_positions)

    # Guardar datos
    output_path.parent.mkdir(parents=True, exist_ok=True)

    if HeteroData is not None and torch is not None:
        data = HeteroData()
        data["agent"].x = torch.from_numpy(agent_positions)
        data["agent"].y = torch.from_numpy(target_actions)
        data["scene"].pos = torch.from_numpy(scene_pointclouds)

        # Aristas temporales
        if T > 1:
            edges_src = torch.arange(0, T - 1, dtype=torch.long)
            edges_dst = torch.arange(1, T, dtype=torch.long)
            data["agent", "temporal", "agent"].edge_index = torch.stack([edges_src, edges_dst], dim=0)
        else:
            data["agent", "temporal", "agent"].edge_index = torch.empty((2, 0), dtype=torch.long)

        torch.save(data, output_path)
        print(f"Grafo HeteroData guardado exitosamente en: {output_path}")
    else:
        # Fallback a numpy comprimido si torch_geometric no esta en el entorno
        npz_path = output_path.with_suffix(".npz")
        np.savez_compressed(
            npz_path,
            agent_x=agent_positions,
            agent_y=target_actions,
            scene_pos=scene_pointclouds
        )
        print(f"Torch/HeteroData no disponible. Datos guardados en formato NumPy: {npz_path}")

    return True


def main():
    parser = argparse.ArgumentParser(
        description="Convierte Rosbags a HeteroData usando LiDAR Sintetico 360°"
    )
    parser.add_argument(
        "--bag-dir", "--bag-path",
        dest="bag_path",
        type=Path,
        required=True,
        help="Directorio o archivo del rosbag (.mcap)"
    )
    parser.add_argument(
        "--waypoints",
        type=Path,
        default=None,
        help="Ruta explicita al archivo waypoints.yaml (opcional, se auto-detecta si no se pasa)"
    )
    parser.add_argument(
        "--output", "--output-dir",
        dest="output",
        type=Path,
        default=None,
        help="Ruta del archivo de salida (.pt)"
    )
    parser.add_argument(
        "--scale",
        type=float,
        default=0.1,
        help="Factor de escala metrica s (default: 0.1)"
    )
    parser.add_argument(
        "--fps-points",
        type=int,
        default=2048,
        help="Numero de puntos por observacion LiDAR (default: 2048)"
    )
    parser.add_argument(
        "--r-max",
        type=float,
        default=10.0,
        help="Alcance maximo del LiDAR en metros (default: 10.0)"
    )
    parser.add_argument(
        "--subsample-step",
        type=int,
        default=5,
        help="Paso de submuestreo temporal de telemetria (default: 5)"
    )
    parser.add_argument(
        "--no-trim-idle",
        action="store_true",
        help="Desactiva el recorte automatico de momentos estaticos en tierra (pre-despegue y post-aterrizaje)"
    )
    parser.add_argument(
        "--trim-start",
        choices=["takeoff_exit", "liftoff", "none"],
        default="takeoff_exit",
        help="Criterio de inicio de recorte: 'takeoff_exit' (al salir de despegue hacia circuito) o 'liftoff' (al despegar del suelo)"
    )

    args = parser.parse_args()

    # Buscar waypoints.yaml
    wp_file = find_waypoints_file(args.bag_path, args.waypoints)
    if not wp_file or not wp_file.is_file():
        print(f"ERROR: No se encontro waypoints.yaml para el rosbag {args.bag_path}.")
        print("Especifica la ruta explicitamente con --waypoints /ruta/a/waypoints.yaml")
        sys.exit(1)

    # Determinar ruta de salida
    if args.output is None:
        out_file = Path("dataset_processed") / (args.bag_path.name + ".pt")
    elif args.output.is_dir() or args.output.suffix == "":
        out_file = args.output / (args.bag_path.name + ".pt")
    else:
        out_file = args.output

    parsed = parse_bag_telemetry(args.bag_path)
    build_heterodata_from_synthetic_lidar(
        parsed,
        wp_file,
        out_file,
        scale_factor=args.scale,
        r_max=args.r_max,
        fps_points=args.fps_points,
        subsample_step=args.subsample_step,
        trim_idle=not args.no_trim_idle,
        trim_start=args.trim_start
    )


if __name__ == "__main__":
    main()
