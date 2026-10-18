#!/usr/bin/env python3
r"""
bag_to_heterodata.py — Convierte Rosbags (.mcap) a HeteroData para Instant Policy.

Genera grafos heterogeneos de PyTorch Geometric utilizando nubes de puntos
LiDAR 360° sinteticas en marco egocentrico estabilizado por gravedad (Horizontal FLU),
acciones relativas SE(3) en horizonte T=8 sobre la trayectoria ejecutada,
extraccion geometrica de L=10 keyframes de contexto, 6 nodos keypoint del UAV,
aumento de datos sintetico para recuperacion (30% Tube Augmentation) y
procesamiento masivo por lotes en paralelo (multiprocessing).

Formato del Grafo (HeteroData):
  Nodos Nominales ('scene' y 'agent'):
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
  - data['agent'].aug_mask:        Tensor (T_steps,) mascara binaria de instantes con Tube Augmentation
  - data['agent'].setpoint_raw:    Tensor (T_steps, 3) setpoint crudo de PX4 (solo diagnostico)
  - data['agent', 'temporal', 'agent'].edge_index: Conexiones temporales t -> t+1

  Nodos de Recuperacion ('aug' — 30% Tube Augmentation, Apendices D y E de Instant Policy):
  - data['aug'].timestep_idx:      Tensor (T_aug,) indice temporal nominal t in [0, T_steps-1]
  - data['aug'].scene_pos:         Tensor (T_aug, 2048, 3) nube LiDAR 360° evaluada en la pose perturbada \tilde{T}_{WE}^t
  - data['aug'].pos_enu:           Tensor (T_aug, 3) posicion perturbada \tilde{p}_t en ENU escalada (s=0.1)
  - data['aug'].yaw_enu:           Tensor (T_aug, 1) yaw perturbado \tilde{\psi}_t en ENU [rad]
  - data['aug'].rot_enu:           Tensor (T_aug, 3, 3) rotacion perturbada R_z(\tilde{\psi}_t)
  - data['aug'].pose_se3:          Tensor (T_aug, 4, 4) pose homogenea perturbada \tilde{T}_{WE}^t in SE(3)
  - data['aug'].delta_pos_m:       Tensor (T_aug, 3) desviacion inyectada \delta p en metros
  - data['aug'].delta_yaw:         Tensor (T_aug, 1) desviacion inyectada \delta \psi en radianes
  - data['aug'].flight_state_in:   Tensor (T_aug, 1) estado binario de entrada \tilde{s}_t (con 10% flip)
  - data['aug'].y:                 Tensor (T_aug, 8, 5) accion correctiva [\Delta\tilde{x}, \Delta\tilde{y}, \Delta\tilde{z}, \Delta\tilde{\psi}, s_{t+k}]
  - data['aug'].action_se3:        Tensor (T_aug, 8, 4, 4) transformaciones correctivas \tilde{T}_{EA}^{(t,k)} = (\tilde{T}_{WE}^t)^{-1} T_{WE}^{t+k}
  - data['aug'].target_pos_enu:    Tensor (T_aug, 8, 3) posiciones nominales futuras p_{t+1..t+8} en ENU escalado

Uso:
  # Procesar un episodio individual:
  python3 bag_to_heterodata.py --bag-dir datasets/task_0000/episode_00/rosbag --output data.pt

  # Procesar todo el dataset en paralelo (multiproceso):
  python3 bag_to_heterodata.py --dataset-dir datasets/ --workers 8
"""

import argparse
import json
import math
import multiprocessing as mp
import os
import sys
import time
from pathlib import Path
from typing import Optional, Dict, Any, List, Tuple

# Evitar contencion de hilos BLAS/OpenMP cuando se ejecuta con multiprocessing.Pool
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")

import numpy as np
import yaml

# Asegurar inclusion de ~/venv_ip y ~/.local en sys.path para ejecucion transparente
_py_ver = f"python{sys.version_info.major}.{sys.version_info.minor}"
for _candidate_site in [
    Path.home() / "venv_ip" / "lib" / _py_ver / "site-packages",
    Path.home() / ".local" / "lib" / _py_ver / "site-packages",
]:
    if _candidate_site.is_dir() and str(_candidate_site) not in sys.path:
        sys.path.insert(0, str(_candidate_site))

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

    candidates = []
    if bag_path.is_dir():
        candidates.append(bag_path / "waypoints.yaml")
    candidates.append(bag_path.parent / "waypoints.yaml")
    candidates.append(bag_path.parent.parent / "waypoints.yaml")

    for c in candidates:
        if c.is_file():
            return c
    return None


def parse_bag_telemetry(bag_path: Path, verbose: bool = True) -> Dict[str, List[Dict[str, Any]]]:
    """
    Parsea los mensajes de telemetria de PX4 desde el archivo .mcap.
    Soporta rosbags 0.11+ (AnyReader / Typestore) y versiones anteriores.
    """
    if AnyReader is None and Reader is None:
        detail = f": {_rosbags_err}" if _rosbags_err else ""
        raise ImportError(
            f"No se pudo cargar la libreria 'rosbags'{detail}. "
            "Ejecuta: bash src/uav_ip/scripts/setup_venv_ip.sh"
        )

    if verbose:
        print(f"Leyendo rosbag desde: {bag_path}")
    data = {
        "odometry": [],
        "local_pos": [],
        "setpoints": []
    }

    bag_p = Path(bag_path)

    if AnyReader is not None:
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

    if verbose:
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


def compute_tube_augmentation(
    pos_enu_m: np.ndarray,
    yaw_enu: np.ndarray,
    flight_state: np.ndarray,
    landmarks: List[Dict[str, Any]],
    landing: Optional[Dict[str, Any]],
    lidar_gen: SyntheticLidarGenerator,
    ep_seed: int = 42,
    scale_factor: float = 0.1,
    horizon: int = 8,
    action_stride: int = 1,
    tube_aug_prob: float = 0.30,
    tube_sigma_pos: float = 0.25,
    tube_sigma_yaw_deg: float = 5.0,
    state_flip_prob: float = 0.10,
    max_pos_dev: float = 0.60,
    max_yaw_dev_deg: float = 15.0,
    min_safe_alt: float = 0.20,
    gravity_aligned: bool = True,
) -> Dict[str, np.ndarray]:
    r"""
    Aumento de Datos Sintetico para Recuperacion (30% Tube Augmentation, Instant Policy Apendices D y E).

    Para una fraccion p_aug = tube_aug_prob (default: 0.30) de los instantes t de la trayectoria:
      1. Muestrea una perturbacion local de posicion \delta p ~ N(0, \sigma_p^2 I) (\sigma_p = 0.25 m)
         y orientacion \delta \psi ~ N(0, \sigma_\psi^2) (\sigma_\psi = 5 deg), y con probabilidad
         state_flip_prob = 0.10 invierte el estado binario de entrada \tilde{s}_t = 1 - s_t.
      2. Sintetiza analiticamente la observacion LiDAR 360° en la pose perturbada \tilde{T}_{WE}^t.
      3. Calcula el horizonte de acciones correctivas hacia las poses futuras nominales:
             \tilde{T}_{EA}^{(t, k)} = (\tilde{T}_{WE}^t)^{-1} T_{WE}^{t + k * stride},   k = 1, ..., horizon
         garantizando que el flujo de accion apunte convergentemente de regreso a la trayectoria experta.
    """
    T = len(pos_enu_m)
    fps_pts = int(lidar_gen.target_points)
    if T == 0 or tube_aug_prob <= 0.0:
        return {
            "aug_mask": np.zeros(T, dtype=np.uint8),
            "timestep_idx": np.zeros((0,), dtype=np.int64),
            "scene_pos": np.zeros((0, fps_pts, 3), dtype=np.float32),
            "pos_enu": np.zeros((0, 3), dtype=np.float32),
            "yaw_enu": np.zeros((0, 1), dtype=np.float32),
            "rot_enu": np.zeros((0, 3, 3), dtype=np.float32),
            "pose_se3": np.zeros((0, 4, 4), dtype=np.float32),
            "delta_pos_m": np.zeros((0, 3), dtype=np.float32),
            "delta_yaw": np.zeros((0, 1), dtype=np.float32),
            "flight_state_in": np.zeros((0, 1), dtype=np.float32),
            "y": np.zeros((0, horizon, 5), dtype=np.float32),
            "action_se3": np.zeros((0, horizon, 4, 4), dtype=np.float32),
            "target_pos_enu": np.zeros((0, horizon, 3), dtype=np.float32),
        }

    rng = np.random.RandomState((int(ep_seed) * 9973 + 17) % (2**31 - 1))
    T_aug = min(T, max(1, int(round(float(tube_aug_prob) * T))))

    # Muestreo estratificado sin reemplazo a lo largo de [0, T-1] para cubrir todo el circuito
    if T_aug == T:
        aug_indices = np.arange(T, dtype=np.int64)
    else:
        edges = np.linspace(0, T, T_aug + 1)
        picked = []
        for b in range(T_aug):
            lo = int(math.floor(edges[b]))
            hi = max(lo + 1, int(math.floor(edges[b + 1])))
            hi = min(T, hi)
            c = int(rng.randint(lo, hi))
            if c in picked:
                for alt_c in range(lo, hi):
                    if alt_c not in picked:
                        c = alt_c
                        break
            picked.append(c)
        aug_indices = np.array(sorted(set(picked)), dtype=np.int64)
        while len(aug_indices) < T_aug:
            remaining = [idx for idx in range(T) if idx not in aug_indices]
            if not remaining:
                break
            extra_c = int(remaining[rng.randint(0, len(remaining))])
            aug_indices = np.array(sorted(list(aug_indices) + [extra_c]), dtype=np.int64)

    T_aug = len(aug_indices)
    aug_mask = np.zeros(T, dtype=np.uint8)
    aug_mask[aug_indices] = 1

    yaw_flat = np.asarray(yaw_enu, dtype=np.float64).reshape(-1)
    s_flat = np.asarray(flight_state, dtype=np.float32).reshape(-1)
    pos_scaled = (pos_enu_m * float(scale_factor)).astype(np.float64)

    sigma_yaw_rad = math.radians(float(tube_sigma_yaw_deg))
    max_yaw_rad = math.radians(float(max_yaw_dev_deg))

    aug_scene_pos = np.zeros((T_aug, fps_pts, 3), dtype=np.float32)
    aug_pos_enu = np.zeros((T_aug, 3), dtype=np.float32)
    aug_yaw_enu = np.zeros((T_aug, 1), dtype=np.float32)
    aug_rot_enu = np.zeros((T_aug, 3, 3), dtype=np.float32)
    aug_pose_se3 = np.zeros((T_aug, 4, 4), dtype=np.float32)
    aug_delta_pos_m = np.zeros((T_aug, 3), dtype=np.float32)
    aug_delta_yaw = np.zeros((T_aug, 1), dtype=np.float32)
    aug_flight_state_in = np.zeros((T_aug, 1), dtype=np.float32)
    aug_action_local = np.zeros((T_aug, horizon, 5), dtype=np.float32)
    aug_action_se3 = np.zeros((T_aug, horizon, 4, 4), dtype=np.float32)
    aug_target_pos_enu = np.zeros((T_aug, horizon, 3), dtype=np.float32)

    for i, t_idx in enumerate(aug_indices):
        t = int(t_idx)
        p_nom_m = pos_enu_m[t].astype(np.float64)
        yaw_nom = float(yaw_flat[t])

        # Perturbacion gaussiana acotada en posicion (m) y yaw (rad)
        dp_m = rng.normal(0.0, float(tube_sigma_pos), size=3)
        dp_m = np.clip(dp_m, -float(max_pos_dev), float(max_pos_dev))
        # Garantizar una desviacion transversal/vertical minima no trivial (>= 5 cm) para entrenamiento util
        if np.linalg.norm(dp_m) < 0.05:
            dp_m[1] = 0.15 if (i % 2 == 0) else -0.15

        tilde_p_m = p_nom_m + dp_m
        if tilde_p_m[2] < min_safe_alt:
            tilde_p_m[2] = float(min_safe_alt)
            dp_m[2] = tilde_p_m[2] - p_nom_m[2]

        dyaw_pert = float(np.clip(rng.normal(0.0, sigma_yaw_rad), -max_yaw_rad, max_yaw_rad))
        tilde_yaw = _wrap_angle(yaw_nom + dyaw_pert)
        tilde_R = rot_z(tilde_yaw)
        tilde_p_scaled = tilde_p_m * float(scale_factor)

        T_we_pert = np.eye(4, dtype=np.float64)
        T_we_pert[:3, :3] = tilde_R
        T_we_pert[:3, 3] = tilde_p_scaled

        # Perturbacion del 10% del estado binario de entrada s_t (Apendice D/E)
        s_nom = float(s_flat[t])
        s_in = (1.0 - s_nom) if (rng.uniform(0.0, 1.0) < float(state_flip_prob)) else s_nom

        # Sintesis analitica del LiDAR 360° desde la pose perturbada \tilde{T}_{WE}^t
        aug_seed = (int(ep_seed) * 20000 + t + 5000) % (2**31 - 1)
        tilde_cloud = lidar_gen.generate_pointcloud(
            {
                "x": float(tilde_p_m[0]),
                "y": float(tilde_p_m[1]),
                "z": float(tilde_p_m[2]),
                "yaw": float(tilde_yaw),
            },
            landmarks,
            landing,
            seed=aug_seed,
            gravity_aligned=gravity_aligned,
        )

        aug_scene_pos[i] = tilde_cloud
        aug_pos_enu[i] = tilde_p_scaled.astype(np.float32)
        aug_yaw_enu[i, 0] = float(tilde_yaw)
        aug_rot_enu[i] = tilde_R.astype(np.float32)
        aug_pose_se3[i] = T_we_pert.astype(np.float32)
        aug_delta_pos_m[i] = dp_m.astype(np.float32)
        aug_delta_yaw[i, 0] = float(dyaw_pert)
        aug_flight_state_in[i, 0] = float(s_in)

        # Accion correctiva relativa \tilde{T}_{EA}^{(t, k)} = (\tilde{T}_{WE}^t)^{-1} T_{WE}^{t+k}
        for k in range(1, horizon + 1):
            tk = min(T - 1, t + k * max(1, action_stride))
            p_tk = pos_scaled[tk]
            dp_world = p_tk - tilde_p_scaled
            dp_flu = tilde_R.T @ dp_world
            dyaw_corr = _wrap_angle(float(yaw_flat[tk] - tilde_yaw))
            R_rel_corr = rot_z(dyaw_corr)

            T_ea_corr = np.eye(4, dtype=np.float32)
            T_ea_corr[:3, :3] = R_rel_corr.astype(np.float32)
            T_ea_corr[:3, 3] = dp_flu.astype(np.float32)

            aug_action_se3[i, k - 1] = T_ea_corr
            aug_action_local[i, k - 1, 0:3] = dp_flu.astype(np.float32)
            aug_action_local[i, k - 1, 3] = float(dyaw_corr)
            aug_action_local[i, k - 1, 4] = float(s_flat[tk])
            aug_target_pos_enu[i, k - 1] = p_tk.astype(np.float32)

    return {
        "aug_mask": aug_mask,
        "timestep_idx": aug_indices,
        "scene_pos": aug_scene_pos,
        "pos_enu": aug_pos_enu,
        "yaw_enu": aug_yaw_enu,
        "rot_enu": aug_rot_enu,
        "pose_se3": aug_pose_se3,
        "delta_pos_m": aug_delta_pos_m,
        "delta_yaw": aug_delta_yaw,
        "flight_state_in": aug_flight_state_in,
        "y": aug_action_local,
        "action_se3": aug_action_se3,
        "target_pos_enu": aug_target_pos_enu,
    }


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

    seg_dists = np.linalg.norm(np.diff(pos, axis=0), axis=1)
    arc_len = np.concatenate([[0.0], np.cumsum(seg_dists)])
    total_len = float(arc_len[-1]) if arc_len[-1] > 1e-8 else 1.0

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

    land_transitions = np.where(s_flat < 0.5)[0]
    if len(land_transitions) > 0:
        t_land = int(land_transitions[0])
        if t_land not in selected:
            selected.append(t_land)

    min_sep = max(1, T // (2 * num_keyframes))
    local_peaks = []
    for t in range(1, T - 1):
        if salience[t] >= salience[t - 1] and salience[t] >= salience[t + 1] and salience[t] > 0.15:
            local_peaks.append((float(salience[t]), t))
    local_peaks.sort(key=lambda x: x[0], reverse=True)

    max_peak_slots = max(2, num_keyframes - 3)
    for _, t_peak in local_peaks:
        if len(selected) >= max_peak_slots:
            break
        if all(abs(t_peak - s_idx) >= min_sep for s_idx in selected):
            selected.append(int(t_peak))

    while len(selected) < num_keyframes:
        best_t = -1
        best_dist = -1.0
        for candidate in range(T):
            if candidate in selected:
                continue
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


def _extract_episode_seed(waypoints_file: Path, wp_cfg: Dict[str, Any]) -> int:
    """Obtiene la semilla determinista del episodio desde episode_meta.json o waypoints.yaml."""
    if "seed" in wp_cfg:
        return int(wp_cfg["seed"])
    meta_file = waypoints_file.parent / "episode_meta.json"
    if meta_file.is_file():
        try:
            with open(meta_file, "r") as f:
                meta = json.load(f)
            if meta.get("episode_seed") is not None:
                return int(meta["episode_seed"])
            if meta.get("task_seed") is not None:
                return int(meta["task_seed"])
        except Exception:
            pass
    return 42


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
    gravity_aligned: bool = True,
    tube_aug_prob: float = 0.30,
    tube_sigma_pos: float = 0.25,
    tube_sigma_yaw_deg: float = 5.0,
    state_flip_prob: float = 0.10,
    verbose: bool = True,
) -> Dict[str, Any]:
    """
    Construye el HeteroData usando SyntheticLidarGenerator en marco Horizontal FLU,
    calcula las acciones relativas SE(3) en horizonte T=8, extrae L=10 keyframes geometricos
    e inyecta el 30% de Tube Augmentation para recuperacion en vuelo cerrado.
    """
    with open(waypoints_file, "r") as f:
        wp_cfg = yaml.safe_load(f)

    ep_seed = _extract_episode_seed(waypoints_file, wp_cfg)
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

    if verbose:
        print(f"Cargados {len(landmarks)} landmarks desde {waypoints_file.name} (ep_seed={ep_seed})")

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
        if verbose:
            print("ERROR: No se encontraron mensajes de odometria ni posicion en el rosbag.")
        return {"ok": False, "T": 0, "T_aug": 0, "saved_path": None}

    if trim_idle:
        start_idx, end_idx = find_flight_trim_indices(
            odom_list,
            using_local_pos=using_local_pos,
            trim_start=trim_start,
            min_takeoff_alt=min_takeoff_alt
        )
        total_raw = len(odom_list)
        odom_list = odom_list[start_idx:end_idx + 1]
        if verbose:
            print(f"  Recorte de vuelo util aplicado: [{start_idx}:{end_idx}] "
                  f"({len(odom_list)} de {total_raw} muestras conservadas).")

    sampled_indices = list(range(0, len(odom_list), max(1, subsample_step)))
    T = len(sampled_indices)
    if verbose:
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

    # 4. Aumento de datos sintetico para recuperacion (30% Tube Augmentation)
    aug_dict = compute_tube_augmentation(
        pos_enu_m=pos_enu_m,
        yaw_enu=agent_yaw,
        flight_state=flight_state,
        landmarks=landmarks,
        landing=landing,
        lidar_gen=lidar_gen,
        ep_seed=ep_seed,
        scale_factor=scale_factor,
        horizon=action_horizon,
        action_stride=action_stride,
        tube_aug_prob=tube_aug_prob,
        tube_sigma_pos=tube_sigma_pos,
        tube_sigma_yaw_deg=tube_sigma_yaw_deg,
        state_flip_prob=state_flip_prob,
        gravity_aligned=gravity_aligned,
    )
    T_aug = len(aug_dict["timestep_idx"])

    # 5. Setpoints crudos de PX4 (conservados exclusivamente para diagnostico del controlador)
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
        data["agent"].aug_mask = torch.from_numpy(aug_dict["aug_mask"])
        data["agent"].setpoint_raw = torch.from_numpy(setpoint_raw)

        # Nodos de recuperacion (30% Tube Augmentation)
        data["aug"].timestep_idx = torch.from_numpy(aug_dict["timestep_idx"])
        data["aug"].scene_pos = torch.from_numpy(aug_dict["scene_pos"])
        data["aug"].pos_enu = torch.from_numpy(aug_dict["pos_enu"])
        data["aug"].yaw_enu = torch.from_numpy(aug_dict["yaw_enu"])
        data["aug"].rot_enu = torch.from_numpy(aug_dict["rot_enu"])
        data["aug"].pose_se3 = torch.from_numpy(aug_dict["pose_se3"])
        data["aug"].delta_pos_m = torch.from_numpy(aug_dict["delta_pos_m"])
        data["aug"].delta_yaw = torch.from_numpy(aug_dict["delta_yaw"])
        data["aug"].flight_state_in = torch.from_numpy(aug_dict["flight_state_in"])
        data["aug"].y = torch.from_numpy(aug_dict["y"])
        data["aug"].action_se3 = torch.from_numpy(aug_dict["action_se3"])
        data["aug"].target_pos_enu = torch.from_numpy(aug_dict["target_pos_enu"])

        # Aristas temporales t -> t+1
        if T > 1:
            edges_src = torch.arange(0, T - 1, dtype=torch.long)
            edges_dst = torch.arange(1, T, dtype=torch.long)
            data["agent", "temporal", "agent"].edge_index = torch.stack([edges_src, edges_dst], dim=0)
        else:
            data["agent", "temporal", "agent"].edge_index = torch.empty((2, 0), dtype=torch.long)

        saved_path = output_path.with_suffix(".pt")
        torch.save(data, saved_path)
        if verbose:
            print(f"Grafo HeteroData SE(3) guardado exitosamente en: {saved_path}")
            print(f"  Keyframes (L={len(keyframe_indices)}): {keyframe_indices.tolist()}")
            print(f"  Tube Augmentation: {T_aug}/{T} muestras de recuperacion ({100.0 * T_aug / max(1, T):.1f}%)")
    else:
        saved_path = output_path.with_suffix(".npz")
        np.savez_compressed(
            saved_path,
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
            aug_mask=aug_dict["aug_mask"],
            setpoint_raw=setpoint_raw,
            aug_timestep_idx=aug_dict["timestep_idx"],
            aug_scene_pos=aug_dict["scene_pos"],
            aug_pos_enu=aug_dict["pos_enu"],
            aug_yaw_enu=aug_dict["yaw_enu"],
            aug_rot_enu=aug_dict["rot_enu"],
            aug_pose_se3=aug_dict["pose_se3"],
            aug_delta_pos_m=aug_dict["delta_pos_m"],
            aug_delta_yaw=aug_dict["delta_yaw"],
            aug_flight_state_in=aug_dict["flight_state_in"],
            aug_y=aug_dict["y"],
            aug_action_se3=aug_dict["action_se3"],
            aug_target_pos_enu=aug_dict["target_pos_enu"],
        )
        if verbose:
            print(f"Torch/HeteroData no disponible. Datos SE(3) guardados en formato NumPy: {saved_path}")
            print(f"  Tube Augmentation: {T_aug}/{T} muestras de recuperacion ({100.0 * T_aug / max(1, T):.1f}%)")

    return {"ok": True, "T": T, "T_aug": T_aug, "saved_path": saved_path}


def discover_dataset_episodes(
    dataset_dir: Path,
    output_root: Optional[Path] = None,
    include_failed: bool = False,
) -> List[Dict[str, Any]]:
    """
    Descubre recursivamente todos los episodios validos (task_XXXX/episode_YY)
    dentro de dataset_dir que contienen un rosbag (.mcap) y waypoints.yaml.
    """
    dataset_dir = Path(dataset_dir).resolve()
    if not dataset_dir.is_dir():
        return []

    candidate_ep_dirs: List[Path] = []
    # Caso 1: dataset_dir contiene task_XXXX/episode_YY
    for task_dir in sorted(dataset_dir.glob("task_*")):
        if task_dir.is_dir():
            for ep_dir in sorted(task_dir.glob("episode_*")):
                if ep_dir.is_dir():
                    candidate_ep_dirs.append(ep_dir)

    # Caso 2: dataset_dir es directamente un task_XXXX con episode_YY
    if not candidate_ep_dirs:
        for ep_dir in sorted(dataset_dir.glob("episode_*")):
            if ep_dir.is_dir():
                candidate_ep_dirs.append(ep_dir)

    # Caso 3: busqueda general por directorios que contengan subdirectorio rosbag
    if not candidate_ep_dirs:
        for rb_dir in sorted(dataset_dir.rglob("rosbag")):
            if rb_dir.is_dir():
                candidate_ep_dirs.append(rb_dir.parent)

    episodes = []
    for ep_dir in candidate_ep_dirs:
        meta_file = ep_dir / "episode_meta.json"
        if meta_file.is_file() and not include_failed:
            try:
                with open(meta_file, "r") as f:
                    meta = json.load(f)
                status = str(meta.get("status", "success")).lower()
                if status not in ("success", "ok"):
                    continue
            except Exception:
                pass

        bag_dir = ep_dir / "rosbag"
        if not bag_dir.is_dir():
            mcaps = list(ep_dir.glob("*.mcap"))
            if not mcaps:
                continue
            bag_dir = ep_dir
        else:
            mcaps = list(bag_dir.glob("*.mcap"))
            if not mcaps:
                continue

        wp_file = find_waypoints_file(bag_dir, ep_dir / "waypoints.yaml")
        if wp_file is None or not wp_file.is_file():
            continue

        if output_root is None:
            out_file = ep_dir / "heterodata.pt"
        else:
            rel = ep_dir.relative_to(dataset_dir)
            out_file = Path(output_root).resolve() / rel / "heterodata.pt"

        try:
            label = str(ep_dir.relative_to(dataset_dir))
        except ValueError:
            label = ep_dir.name

        episodes.append({
            "label": label,
            "ep_dir": ep_dir,
            "bag_path": bag_dir,
            "wp_file": wp_file,
            "out_file": out_file,
        })

    return episodes


def _process_episode_worker(job: Dict[str, Any]) -> Dict[str, Any]:
    """Funcion worker para procesar un episodio completo dentro de multiprocessing.Pool."""
    t0 = time.perf_counter()
    label = job["label"]
    bag_path: Path = job["bag_path"]
    wp_file: Path = job["wp_file"]
    out_file: Path = job["out_file"]

    if job.get("skip_existing", False):
        expected_ext = ".pt" if (HeteroData is not None and torch is not None) else ".npz"
        candidate_existing = out_file.with_suffix(expected_ext)
        if candidate_existing.is_file():
            return {
                "label": label,
                "status": "skipped",
                "elapsed_s": time.perf_counter() - t0,
                "T": 0,
                "T_aug": 0,
                "saved_path": str(candidate_existing),
                "error": None,
            }

    try:
        parsed = parse_bag_telemetry(bag_path, verbose=False)
        res = build_heterodata_from_synthetic_lidar(
            parsed_data=parsed,
            waypoints_file=wp_file,
            output_path=out_file,
            scale_factor=job["scale"],
            r_max=job["r_max"],
            fps_points=job["fps_points"],
            subsample_step=job["subsample_step"],
            trim_idle=job["trim_idle"],
            trim_start=job["trim_start"],
            action_horizon=job["action_horizon"],
            action_stride=job["action_stride"],
            num_keyframes=job["num_keyframes"],
            gravity_aligned=job["gravity_aligned"],
            tube_aug_prob=job["tube_aug_prob"],
            tube_sigma_pos=job["tube_sigma_pos"],
            tube_sigma_yaw_deg=job["tube_sigma_yaw_deg"],
            state_flip_prob=job["state_flip_prob"],
            verbose=False,
        )
        elapsed = time.perf_counter() - t0
        if not res.get("ok", False):
            return {
                "label": label,
                "status": "error",
                "elapsed_s": elapsed,
                "T": 0,
                "T_aug": 0,
                "saved_path": None,
                "error": "No odometry messages found",
            }
        return {
            "label": label,
            "status": "ok",
            "elapsed_s": elapsed,
            "T": int(res["T"]),
            "T_aug": int(res["T_aug"]),
            "saved_path": str(res["saved_path"]),
            "error": None,
        }
    except Exception as e:
        return {
            "label": label,
            "status": "error",
            "elapsed_s": time.perf_counter() - t0,
            "T": 0,
            "T_aug": 0,
            "saved_path": None,
            "error": str(e),
        }


def process_dataset_directory(
    dataset_dir: Path,
    output_root: Optional[Path] = None,
    workers: int = 0,
    scale_factor: float = 0.1,
    r_max: float = 10.0,
    fps_points: int = 2048,
    subsample_step: int = 5,
    trim_idle: bool = True,
    trim_start: str = "takeoff_exit",
    action_horizon: int = 8,
    action_stride: int = 1,
    num_keyframes: int = 10,
    gravity_aligned: bool = True,
    tube_aug_prob: float = 0.30,
    tube_sigma_pos: float = 0.25,
    tube_sigma_yaw_deg: float = 5.0,
    state_flip_prob: float = 0.10,
    skip_existing: bool = False,
    include_failed: bool = False,
) -> Dict[str, Any]:
    """
    Procesa en lote y en paralelo todos los episodios de un directorio de dataset.
    """
    episodes = discover_dataset_episodes(
        dataset_dir, output_root=output_root, include_failed=include_failed
    )
    if not episodes:
        print(f"ERROR: No se encontraron episodios validos en {dataset_dir}")
        return {"total": 0, "ok": 0, "skipped": 0, "errors": 0, "elapsed_s": 0.0}

    cpu_cores = os.cpu_count() or 4
    num_workers = cpu_cores if workers <= 0 else min(int(workers), cpu_cores)
    num_workers = min(num_workers, len(episodes))

    # Pre-compilar el kernel C SIMD una sola vez en el proceso padre antes de abrir el Pool
    try:
        from uav_ip.synthetic_lidar import _get_fps_c_kernel
        _get_fps_c_kernel()
    except Exception:
        pass

    jobs = []
    for ep in episodes:
        job = dict(ep)
        job.update({
            "scale": scale_factor,
            "r_max": r_max,
            "fps_points": fps_points,
            "subsample_step": subsample_step,
            "trim_idle": trim_idle,
            "trim_start": trim_start,
            "action_horizon": action_horizon,
            "action_stride": action_stride,
            "num_keyframes": num_keyframes,
            "gravity_aligned": gravity_aligned,
            "tube_aug_prob": tube_aug_prob,
            "tube_sigma_pos": tube_sigma_pos,
            "tube_sigma_yaw_deg": tube_sigma_yaw_deg,
            "state_flip_prob": state_flip_prob,
            "skip_existing": skip_existing,
        })
        jobs.append(job)

    fmt_name = "HeteroData (.pt)" if (HeteroData is not None and torch is not None) else "NumPy (.npz)"
    print("=" * 72)
    print(f" Procesamiento Masivo Multiproceso -> {fmt_name}")
    print(f"   Directorio dataset : {Path(dataset_dir).resolve()}")
    print(f"   Episodios hallados : {len(jobs)}")
    print(f"   Workers paralelos  : {num_workers} (de {cpu_cores} nucleos disponibles)")
    print(f"   Tube Augmentation  : {tube_aug_prob * 100:.0f}% (sigma_p={tube_sigma_pos}m, sigma_yaw={tube_sigma_yaw_deg} deg)")
    print("=" * 72)

    t_start = time.perf_counter()
    results = []

    if num_workers == 1:
        for idx, job in enumerate(jobs, 1):
            res = _process_episode_worker(job)
            results.append(res)
            status_tag = res["status"].upper()
            print(
                f"  [{idx:3d}/{len(jobs):3d}] {res['label']:28s} | {status_tag:7s} | "
                f"T={res['T']:3d} (+{res['T_aug']:2d} aug) | {res['elapsed_s']*1000:6.1f} ms"
            )
    else:
        with mp.Pool(processes=num_workers) as pool:
            for idx, res in enumerate(pool.imap_unordered(_process_episode_worker, jobs), 1):
                results.append(res)
                status_tag = res["status"].upper()
                if res["status"] == "error":
                    print(f"  [{idx:3d}/{len(jobs):3d}] {res['label']:28s} | ERROR: {res['error']}")
                else:
                    print(
                        f"  [{idx:3d}/{len(jobs):3d}] {res['label']:28s} | {status_tag:7s} | "
                        f"T={res['T']:3d} (+{res['T_aug']:2d} aug) | {res['elapsed_s']*1000:6.1f} ms"
                    )

    total_elapsed = time.perf_counter() - t_start
    n_ok = sum(1 for r in results if r["status"] == "ok")
    n_skip = sum(1 for r in results if r["status"] == "skipped")
    n_err = sum(1 for r in results if r["status"] == "error")
    total_T = sum(r["T"] for r in results)
    total_aug = sum(r["T_aug"] for r in results)
    eps_per_sec = len(jobs) / max(1e-6, total_elapsed)

    print("-" * 72)
    print(f" Resumen: {n_ok} exitosos, {n_skip} omitidos, {n_err} errores.")
    print(f" Muestras totales: {total_T} nominales + {total_aug} tube-augmented = {total_T + total_aug} estados.")
    print(f" Tiempo total: {total_elapsed:.2f} s ({eps_per_sec:.2f} episodios/s, "
          f"estimado 100 episodios: {100.0 / eps_per_sec:.1f} s).")
    print("=" * 72)

    return {
        "total": len(jobs),
        "ok": n_ok,
        "skipped": n_skip,
        "errors": n_err,
        "total_T": total_T,
        "total_T_aug": total_aug,
        "elapsed_s": total_elapsed,
        "results": results,
    }


def main():
    parser = argparse.ArgumentParser(
        description="Convierte Rosbags a HeteroData SE(3) usando LiDAR Sintetico 360° y Tube Augmentation"
    )
    src_group = parser.add_mutually_exclusive_group(required=True)
    src_group.add_argument(
        "--bag-dir", "--bag-path",
        dest="bag_path",
        type=Path,
        help="Directorio o archivo de un rosbag individual (.mcap)"
    )
    src_group.add_argument(
        "--dataset-dir",
        dest="dataset_dir",
        type=Path,
        help="Directorio raiz del dataset (ej. datasets/) para procesamiento por lotes multiproceso"
    )

    parser.add_argument(
        "--waypoints",
        type=Path,
        default=None,
        help="Ruta explicita al archivo waypoints.yaml (opcional en modo --bag-dir)"
    )
    parser.add_argument(
        "--output", "--output-dir",
        dest="output",
        type=Path,
        default=None,
        help="Ruta del archivo de salida (.pt) en modo --bag-dir o carpeta raiz en modo --dataset-dir"
    )
    parser.add_argument(
        "--workers", "-j",
        type=int,
        default=0,
        help="Numero de procesos paralelos en modo --dataset-dir (default: 0 = todos los nucleos CPU)"
    )
    parser.add_argument(
        "--skip-existing",
        action="store_true",
        help="Omitir episodios que ya posean su archivo heterodata generado"
    )
    parser.add_argument(
        "--include-failed",
        action="store_true",
        help="Procesar tambien episodios cuyo episode_meta.json no tenga status='success'"
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
        "--tube-aug-prob",
        type=float,
        default=0.30,
        help="Fraccion de instantes con Tube Augmentation de recuperacion (default: 0.30)"
    )
    parser.add_argument(
        "--tube-sigma-pos",
        type=float,
        default=0.25,
        help="Desviacion estandar de posicion en metros para Tube Augmentation (default: 0.25)"
    )
    parser.add_argument(
        "--tube-sigma-yaw",
        type=float,
        default=5.0,
        help="Desviacion estandar de yaw en grados para Tube Augmentation (default: 5.0)"
    )
    parser.add_argument(
        "--state-flip-prob",
        type=float,
        default=0.10,
        help="Probabilidad de perturbar el estado binario de entrada en Tube Augmentation (default: 0.10)"
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

    if args.dataset_dir is not None:
        summary = process_dataset_directory(
            dataset_dir=args.dataset_dir,
            output_root=args.output,
            workers=args.workers,
            scale_factor=args.scale,
            r_max=args.r_max,
            fps_points=args.fps_points,
            subsample_step=args.subsample_step,
            trim_idle=not args.no_trim_idle,
            trim_start=args.trim_start,
            action_horizon=args.action_horizon,
            action_stride=args.action_stride,
            num_keyframes=args.num_keyframes,
            tube_aug_prob=args.tube_aug_prob,
            tube_sigma_pos=args.tube_sigma_pos,
            tube_sigma_yaw_deg=args.tube_sigma_yaw,
            state_flip_prob=args.state_flip_prob,
            skip_existing=args.skip_existing,
            include_failed=args.include_failed,
        )
        if summary["errors"] > 0 or summary["total"] == 0:
            sys.exit(1)
        return

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
    res = build_heterodata_from_synthetic_lidar(
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
        tube_aug_prob=args.tube_aug_prob,
        tube_sigma_pos=args.tube_sigma_pos,
        tube_sigma_yaw_deg=args.tube_sigma_yaw,
        state_flip_prob=args.state_flip_prob,
    )
    if not res.get("ok", False):
        sys.exit(1)


if __name__ == "__main__":
    main()
