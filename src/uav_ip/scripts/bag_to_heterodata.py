#!/usr/bin/env python3
"""
bag_to_heterodata.py — Convierte Rosbags (.mcap) a HeteroData para Instant Policy.

Genera grafos heterogeneos de PyTorch Geometric utilizando nubes de puntos
LiDAR 360° sinteticas en marco egocentrico estabilizado por gravedad (Horizontal FLU)
y acciones relativas SE(3) en horizonte T=8 sobre la trayectoria ejecutada,
junto con extraccion geometrica de L=10 keyframes de contexto y 6 nodos keypoint del UAV.

Formato del Grafo (HeteroData):
  - data['scene'].pos:             Tensor (T_steps, 2048, 3) nubes LiDAR 360° en Horizontal FLU (s=0.1)
  - data['agent'].x / pos_enu:     Tensor (T_steps, 3) posiciones del dron en ENU escaladas (s=0.1)
  - data['agent'].yaw_enu:         Tensor (T_steps, 1) rumbo yaw en ENU [rad]
  - data['agent'].rot_enu:         Tensor (T_steps, 3, 3) matrices R_z(yaw_t) estabilizadas
  - data['agent'].pose_se3:        Tensor (T_steps, 4, 4) poses homogeneas T_{WE}^t en SE(3)
  - data['agent'].q_ned:           Tensor (T_steps, 4) cuaternion completo PX4 para auditoria
  - data['agent'].flight_state:    Tensor (T_steps, 1) estado binario s_t in {0, 1} (1=vuelo, 0=aterrizaje)
  - data['agent'].keypoints_body:  Tensor (6, 3) 6 keypoints canonicos del UAV en marco local (s=0.1)
  - data['agent'].y:               Tensor (T_steps, 8, 5) horizonte local [dx_flu, dy_flu, dz_flu, dyaw, s_{t+k}]
  - data['agent'].action_se3:      Tensor (T_steps, 8, 4, 4) transformaciones T_{EA}^{(t,k)} = (T_{WE}^t)^{-1} T_{WE}^{t+k}
  - data['agent'].keyframe_indices:Tensor (10,) indices de L=10 keyframes geometricos de contexto
  - data['agent'].setpoint_raw:    Tensor (T_steps, 3) setpoint crudo de PX4 (solo diagnostico)
  - data['agent', 'temporal', 'agent'].edge_index: Conexiones temporales t -> t+1

Uso:
    python3 bag_to_heterodata.py --bag-dir /ruta/al/rosbag --waypoints /ruta/a/waypoints.yaml --output data.pt
"""

import argparse
import math
import sys
from pathlib import Path
from typing import Optional, Dict, Any, List, Tuple

import numpy as np
import yaml

# Asegurar inclusion del directorio de paquetes de usuario (~/.local/lib/python3.X/site-packages)
user_site = Path.home() / ".local" / "lib" / f"python{sys.version_info.major}.{sys.version_info.minor}" / "site-packages"
if user_site.is_dir() and str(user_site) not in sys.path:
    sys.path.insert(0, str(user_site))

# Inyectar raiz del paquete uav_ip (src/uav_ip) para importar la fuente unica uav_ip.synthetic_lidar
_pkg_root = Path(__file__).resolve().parent.parent
if (_pkg_root / "uav_ip").is_dir() and str(_pkg_root) not in sys.path:
    sys.path.insert(0, str(_pkg_root))

try:
    from uav_ip.synthetic_lidar import (
        SyntheticLidarGenerator,
        px4_odometry_to_gazebo_enu,
        extract_yaw_from_rot_matrix,
        ned_to_gazebo_enu,
        yaw_ned_to_gazebo_enu,
        quaternion_to_rot_matrix,
        rot_z,
        fps_numpy,
    )
except ImportError:
    sys.path.append(str(Path(__file__).resolve().parent))
    from synthetic_lidar import (
        SyntheticLidarGenerator,
        px4_odometry_to_gazebo_enu,
        extract_yaw_from_rot_matrix,
        ned_to_gazebo_enu,
        yaw_ned_to_gazebo_enu,
        quaternion_to_rot_matrix,
        rot_z,
        fps_numpy,
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


def _wrap_angle(angle: float) -> float:
    """Envuelve un angulo en radianes al rango [-pi, pi]."""
    return float(math.atan2(math.sin(angle), math.cos(angle)))


def get_uav_keypoints_body(
    scale_factor: float = 0.1,
    arm_length: float = 0.25,
    z_height: float = 0.15
) -> np.ndarray:
    """
    Retorna los 6 nodos keypoint canonicos {p_g^i}_{i=1}^6 del UAV en el marco
    local estabilizado (Horizontal FLU) escalados por scale_factor (s = 0.1):
      0: Centro de masa (0, 0, 0)
      1: Proa   +X (+arm_length, 0, 0)
      2: Popa   -X (-arm_length, 0, 0)
      3: Babor  +Y (0, +arm_length, 0)
      4: Estribor -Y (0, -arm_length, 0)
      5: Cenit  +Z (0, 0, +z_height)
    Permite aplicar directamente el estimador SVD de Instant Policy (Sec. 3.2 y 3.3).
    """
    kp_m = np.array([
        [0.0,         0.0,         0.0],
        [arm_length,  0.0,         0.0],
        [-arm_length, 0.0,         0.0],
        [0.0,         arm_length,  0.0],
        [0.0,        -arm_length,  0.0],
        [0.0,         0.0,         z_height],
    ], dtype=np.float32)
    return (kp_m * float(scale_factor)).astype(np.float32)


def compute_flight_state(
    pos_enu_m: np.ndarray,
    landing: Optional[Dict[str, Any]] = None,
    landing_xy_radius: float = 1.2,
    landing_z_threshold: float = 0.85
) -> np.ndarray:
    """
    Calcula el estado binario de vuelo s_t in {0.0, 1.0} de forma (T_steps, 1):
      - s_t = 1.0: Vuelo activo en el circuito.
      - s_t = 0.0: Descenso final / touchdown sobre la plataforma de aterrizaje.
    """
    T = len(pos_enu_m)
    state = np.ones((T, 1), dtype=np.float32)
    if T == 0:
        return state

    if landing is not None and "x" in landing and "y" in landing:
        lx, ly = float(landing["x"]), float(landing["y"])
        dist_xy = np.hypot(pos_enu_m[:, 0] - lx, pos_enu_m[:, 1] - ly)
        is_landing = (dist_xy <= landing_xy_radius) & (pos_enu_m[:, 2] <= landing_z_threshold)
        if np.any(is_landing):
            first_land = int(np.argmax(is_landing))
            state[first_land:, 0] = 0.0
            return state

    # Fallback si no hay diccionario landing o el dron descendio al final de la ruta
    max_x_idx = int(np.argmax(pos_enu_m[:, 0]))
    low_tail = np.where((np.arange(T) >= max_x_idx) & (pos_enu_m[:, 2] <= landing_z_threshold))[0]
    if len(low_tail) > 0:
        state[int(low_tail[0]):, 0] = 0.0
    elif T >= 3 and pos_enu_m[-1, 2] < pos_enu_m[-2, 2]:
        state[-1, 0] = 0.0

    return state


def compute_se3_action_horizon(
    pos_enu_scaled: np.ndarray,
    yaw_enu: np.ndarray,
    flight_state: np.ndarray,
    horizon: int = 8,
    action_stride: int = 1
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Calcula las poses homogeneas del agente T_{WE}^t in SE(3) y el horizonte de
    T=8 transformaciones relativas futuras sobre la trayectoria realmente volada:
        T_{EA}^{(t, k)} = (T_{WE}^t)^{-1} T_{WE}^{t + k * stride},   k = 1, ..., horizon

    Retorna:
      - pose_se3:     (T_steps, 4, 4) matrices T_{WE}^t en marco global ENU escalado.
      - action_se3:   (T_steps, horizon, 4, 4) matrices relativas T_{EA}^{(t, k)} en Horizontal FLU.
      - action_local: (T_steps, horizon, 5) vectores [dx_flu, dy_flu, dz_flu, dyaw, s_{t+k}].
    """
    T = len(pos_enu_scaled)
    yaw_flat = np.asarray(yaw_enu, dtype=np.float64).reshape(-1)
    s_flat = np.asarray(flight_state, dtype=np.float32).reshape(-1)

    pose_se3 = np.zeros((T, 4, 4), dtype=np.float32)
    action_se3 = np.zeros((T, horizon, 4, 4), dtype=np.float32)
    action_local = np.zeros((T, horizon, 5), dtype=np.float32)

    for t in range(T):
        R_t = rot_z(float(yaw_flat[t]))
        p_t = pos_enu_scaled[t].astype(np.float64)
        T_we = np.eye(4, dtype=np.float64)
        T_we[:3, :3] = R_t
        T_we[:3, 3] = p_t
        pose_se3[t] = T_we.astype(np.float32)

        for k in range(1, horizon + 1):
            tk = min(T - 1, t + k * max(1, action_stride))
            p_tk = pos_enu_scaled[tk].astype(np.float64)
            dp_world = p_tk - p_t
            # Proyeccion al marco local estabilizado del dron en el instante t: R_t^T * dp_world
            dp_flu = R_t.T @ dp_world
            dyaw = _wrap_angle(float(yaw_flat[tk] - yaw_flat[t]))
            R_rel = rot_z(dyaw)

            T_ea = np.eye(4, dtype=np.float32)
            T_ea[:3, :3] = R_rel.astype(np.float32)
            T_ea[:3, 3] = dp_flu.astype(np.float32)

            action_se3[t, k - 1] = T_ea
            action_local[t, k - 1, 0:3] = dp_flu.astype(np.float32)
            action_local[t, k - 1, 3] = float(dyaw)
            action_local[t, k - 1, 4] = float(s_flat[tk])

    return pose_se3, action_se3, action_local


def extract_trajectory_keyframes(
    pos_enu_scaled: np.ndarray,
    yaw_enu: np.ndarray,
    flight_state: np.ndarray,
    num_keyframes: int = 10
) -> np.ndarray:
    """
    Selecciona geometricamente los L = num_keyframes (default: 10) indices temporales
    mas informativos de una demostracion de vuelo para construir el grafo de contexto:
      1. Inicio (t = 0) y fin (t = T - 1) del vuelo.
      2. Transicion del estado binario de vuelo s_t (inicio del descenso de aterrizaje).
      3. Maximos locales de curvatura geometrica 3D kappa(t) y velocidad angular |dyaw(t)|
         (capturando automaticamente los vertices de slalom, overfly y los cuadrantes de orbit).
      4. Cobertura uniforme por longitud de arco para completar exactamente L keyframes ordenados.
    """
    T = len(pos_enu_scaled)
    if T == 0:
        return np.zeros(num_keyframes, dtype=np.int64)
    if T <= num_keyframes:
        idx = np.linspace(0, T - 1, num_keyframes)
        return np.round(idx).astype(np.int64)

    pos = np.asarray(pos_enu_scaled, dtype=np.float64)
    yaw_flat = np.asarray(yaw_enu, dtype=np.float64).reshape(-1)
    s_flat = np.asarray(flight_state, dtype=np.float32).reshape(-1)

    # Longitud de arco acumulada
    seg_dists = np.linalg.norm(np.diff(pos, axis=0), axis=1)
    arc_len = np.concatenate([[0.0], np.cumsum(seg_dists)])
    total_len = float(arc_len[-1]) if arc_len[-1] > 1e-8 else 1.0

    # Curvatura discreta 3D (cambio direccional de tangente) + tasa de cambio de yaw
    curvature = np.zeros(T, dtype=np.float64)
    yaw_rate = np.zeros(T, dtype=np.float64)
    for t in range(1, T - 1):
        v1 = pos[t] - pos[t - 1]
        v2 = pos[t + 1] - pos[t]
        n1 = np.linalg.norm(v1)
        n2 = np.linalg.norm(v2)
        if n1 > 1e-6 and n2 > 1e-6:
            cos_ang = np.clip(np.dot(v1, v2) / (n1 * n2), -1.0, 1.0)
            curvature[t] = math.acos(float(cos_ang)) / (0.5 * (n1 + n2) + 1e-6)
        yaw_rate[t] = abs(_wrap_angle(float(yaw_flat[t + 1] - yaw_flat[t - 1])))

    c_max = float(np.max(curvature)) if np.max(curvature) > 1e-8 else 1.0
    y_max = float(np.max(yaw_rate)) if np.max(yaw_rate) > 1e-8 else 1.0
    salience = (curvature / c_max) + 0.5 * (yaw_rate / y_max)

    selected: List[int] = [0, T - 1]

    # Ancla de transicion de estado de vuelo (inicio de aterrizaje)
    land_transitions = np.where(s_flat < 0.5)[0]
    if len(land_transitions) > 0:
        t_land = int(land_transitions[0])
        if t_land not in selected:
            selected.append(t_land)

    # Maximos locales de saliencia geometrica con supresion de no-maximos
    min_sep = max(1, T // (2 * num_keyframes))
    local_peaks = []
    for t in range(1, T - 1):
        if salience[t] >= salience[t - 1] and salience[t] >= salience[t + 1] and salience[t] > 0.15:
            local_peaks.append((float(salience[t]), t))
    local_peaks.sort(key=lambda x: x[0], reverse=True)

    # Reservar hasta la mitad de los keyframes para picos de alta curvatura, dejando resto para cobertura de arco
    max_peak_slots = max(2, num_keyframes - 3)
    for _, t_peak in local_peaks:
        if len(selected) >= max_peak_slots:
            break
        if all(abs(t_peak - s_idx) >= min_sep for s_idx in selected):
            selected.append(int(t_peak))

    # Completar hasta exactamente num_keyframes maximizando distancia minima en longitud de arco (1D FPS)
    while len(selected) < num_keyframes:
        best_t = -1
        best_dist = -1.0
        for candidate in range(T):
            if candidate in selected:
                continue
            # Distancia combinada en longitud de arco + indice temporal normalizado
            min_d = min(
                abs(arc_len[candidate] - arc_len[s_idx]) / total_len
                + 0.25 * abs(candidate - s_idx) / float(T)
                for s_idx in selected
            )
            if min_d > best_dist:
                best_dist = min_d
                best_t = candidate
        if best_t < 0:
            break
        selected.append(int(best_t))

    selected_sorted = sorted(selected[:num_keyframes])
    return np.array(selected_sorted, dtype=np.int64)


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
    min_takeoff_alt: float = 1.0,
    action_horizon: int = 8,
    action_stride: int = 1,
    num_keyframes: int = 10,
    gravity_aligned: bool = True
) -> bool:
    """
    Construye el HeteroData usando SyntheticLidarGenerator en marco Horizontal FLU
    y calcula las acciones relativas SE(3) en horizonte T=8 y L=10 keyframes geometricos.
    """
    with open(waypoints_file, "r") as f:
        wp_cfg = yaml.safe_load(f)

    ep_seed = int(wp_cfg.get("seed", 42))
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
        target_points=fps_points,
        gravity_aligned=gravity_aligned
    )

    odom_list = parsed_data.get("odometry", [])
    if not odom_list:
        odom_list = parsed_data.get("local_pos", [])
        using_local_pos = True
    else:
        using_local_pos = False

    if not odom_list:
        print("ERROR: No se encontraron mensajes de odometria ni posicion en el rosbag.")
        return False

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

    sampled_indices = list(range(0, len(odom_list), max(1, subsample_step)))
    T = len(sampled_indices)
    print(f"Generando nubes LiDAR sinteticas 360° (gravity_aligned={gravity_aligned}) para T={T} instantes...")

    pos_enu_m_list = []
    yaw_enu_list = []
    rot_enu_list = []
    q_ned_list = []
    scene_pointclouds = []

    for step_i, idx in enumerate(sampled_indices):
        item = odom_list[idx]
        if not using_local_pos:
            pos_ned = item["position"]
            q_ned = [float(item["q"][0]), float(item["q"][1]), float(item["q"][2]), float(item["q"][3])]
            p_enu, R_full = px4_odometry_to_gazebo_enu(pos_ned, q_ned, gravity_aligned=False)
            yaw_val = extract_yaw_from_rot_matrix(R_full)
            R_stab = rot_z(yaw_val)
            drone_pose = {
                "x": float(p_enu[0]),
                "y": float(p_enu[1]),
                "z": float(p_enu[2]),
                "yaw": float(yaw_val),
                "R": R_stab if gravity_aligned else R_full,
                "q_ned": q_ned,
            }
        else:
            p_ned = [item["x"], item["y"], item["z"]]
            p_enu = ned_to_gazebo_enu(p_ned)
            yaw_val = _wrap_angle(yaw_ned_to_gazebo_enu(item["heading"]))
            R_stab = rot_z(yaw_val)
            half_yaw_ned = 0.5 * float(item["heading"])
            q_ned = [math.cos(half_yaw_ned), 0.0, 0.0, math.sin(half_yaw_ned)]
            drone_pose = {
                "x": float(p_enu[0]),
                "y": float(p_enu[1]),
                "z": float(p_enu[2]),
                "yaw": float(yaw_val),
            }

        # Semilla determinista por episodio y paso temporal
        step_seed = (ep_seed * 10000 + step_i) % (2**31 - 1)
        cloud_scaled = lidar_gen.generate_pointcloud(
            drone_pose, landmarks, landing, seed=step_seed, gravity_aligned=gravity_aligned
        )
        scene_pointclouds.append(cloud_scaled)

        pos_enu_m_list.append(p_enu)
        yaw_enu_list.append([yaw_val])
        rot_enu_list.append(R_stab)
        q_ned_list.append(q_ned)

    pos_enu_m = np.array(pos_enu_m_list, dtype=np.float64)                  # (T, 3) en metros
    agent_positions = (pos_enu_m * scale_factor).astype(np.float32)         # (T, 3) escalado s=0.1
    agent_yaw = np.array(yaw_enu_list, dtype=np.float32)                    # (T, 1) rad
    agent_rot = np.array(rot_enu_list, dtype=np.float32)                    # (T, 3, 3) R_z(yaw)
    agent_q_ned = np.array(q_ned_list, dtype=np.float32)                    # (T, 4)
    scene_pointclouds = np.array(scene_pointclouds, dtype=np.float32)       # (T, 2048, 3)

    # 1. Estado binario de vuelo s_t in {0, 1} y 6 nodos keypoint canonicos
    flight_state = compute_flight_state(pos_enu_m, landing=landing)         # (T, 1)
    keypoints_body = get_uav_keypoints_body(scale_factor=scale_factor)      # (6, 3)

    # 2. Poses SE(3) y horizonte de acciones relativas T=8 sobre la trayectoria ejecutada
    pose_se3, action_se3, action_local = compute_se3_action_horizon(
        agent_positions,
        agent_yaw,
        flight_state,
        horizon=action_horizon,
        action_stride=action_stride,
    )

    # 3. Extraccion geometrica de L=10 keyframes de contexto
    keyframe_indices = extract_trajectory_keyframes(
        agent_positions,
        agent_yaw,
        flight_state,
        num_keyframes=num_keyframes,
    )

    # 4. Setpoints crudos de PX4 (conservados exclusivamente para diagnostico del controlador)
    setpoints = parsed_data.get("setpoints", [])
    setpoint_raw = np.zeros((T, 3), dtype=np.float32)
    if setpoints:
        sp_timestamps = np.array([sp["timestamp"] for sp in setpoints], dtype=np.int64)
        sp_positions = np.array(
            [ned_to_gazebo_enu(sp["position"]) * scale_factor for sp in setpoints],
            dtype=np.float32,
        )
        for i, idx in enumerate(sampled_indices):
            t = odom_list[idx]["timestamp"]
            sp_idx = np.searchsorted(sp_timestamps, t, side="right") - 1
            sp_idx = int(np.clip(sp_idx, 0, len(sp_positions) - 1))
            setpoint_raw[i] = sp_positions[sp_idx]
    else:
        setpoint_raw[:] = agent_positions

    output_path.parent.mkdir(parents=True, exist_ok=True)

    if HeteroData is not None and torch is not None:
        data = HeteroData()
        data["scene"].pos = torch.from_numpy(scene_pointclouds)

        data["agent"].x = torch.from_numpy(agent_positions)
        data["agent"].pos_enu = torch.from_numpy(agent_positions)
        data["agent"].yaw_enu = torch.from_numpy(agent_yaw)
        data["agent"].rot_enu = torch.from_numpy(agent_rot)
        data["agent"].pose_se3 = torch.from_numpy(pose_se3)
        data["agent"].q_ned = torch.from_numpy(agent_q_ned)
        data["agent"].flight_state = torch.from_numpy(flight_state)
        data["agent"].keypoints_body = torch.from_numpy(keypoints_body)

        data["agent"].y = torch.from_numpy(action_local)
        data["agent"].action_se3 = torch.from_numpy(action_se3)
        data["agent"].keyframe_indices = torch.from_numpy(keyframe_indices)
        data["agent"].setpoint_raw = torch.from_numpy(setpoint_raw)

        # Aristas temporales t -> t+1
        if T > 1:
            edges_src = torch.arange(0, T - 1, dtype=torch.long)
            edges_dst = torch.arange(1, T, dtype=torch.long)
            data["agent", "temporal", "agent"].edge_index = torch.stack([edges_src, edges_dst], dim=0)
        else:
            data["agent", "temporal", "agent"].edge_index = torch.empty((2, 0), dtype=torch.long)

        torch.save(data, output_path)
        print(f"Grafo HeteroData SE(3) guardado exitosamente en: {output_path}")
        print(f"  Keyframes (L={len(keyframe_indices)}): {keyframe_indices.tolist()}")
    else:
        npz_path = output_path.with_suffix(".npz")
        np.savez_compressed(
            npz_path,
            scene_pos=scene_pointclouds,
            agent_x=agent_positions,
            pos_enu=agent_positions,
            yaw_enu=agent_yaw,
            rot_enu=agent_rot,
            pose_se3=pose_se3,
            q_ned=agent_q_ned,
            flight_state=flight_state,
            keypoints_body=keypoints_body,
            agent_y=action_local,
            action_se3=action_se3,
            keyframe_indices=keyframe_indices,
            setpoint_raw=setpoint_raw,
        )
        print(f"Torch/HeteroData no disponible. Datos SE(3) guardados en formato NumPy: {npz_path}")

    return True


def main():
    parser = argparse.ArgumentParser(
        description="Convierte Rosbags a HeteroData SE(3) usando LiDAR Sintetico 360°"
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
        "--action-horizon",
        type=int,
        default=8,
        help="Horizonte de prediccion de acciones relativas SE(3) (default: 8)"
    )
    parser.add_argument(
        "--action-stride",
        type=int,
        default=1,
        help="Salto temporal entre pasos del horizonte de accion (default: 1)"
    )
    parser.add_argument(
        "--num-keyframes",
        type=int,
        default=10,
        help="Numero L de keyframes geometricos de contexto (default: 10)"
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

    wp_file = find_waypoints_file(args.bag_path, args.waypoints)
    if not wp_file or not wp_file.is_file():
        print(f"ERROR: No se encontro waypoints.yaml para el rosbag {args.bag_path}.")
        print("Especifica la ruta explicitamente con --waypoints /ruta/a/waypoints.yaml")
        sys.exit(1)

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
        trim_start=args.trim_start,
        action_horizon=args.action_horizon,
        action_stride=args.action_stride,
        num_keyframes=args.num_keyframes,
    )


if __name__ == "__main__":
    main()
