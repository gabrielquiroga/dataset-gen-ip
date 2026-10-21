#!/usr/bin/env python3
"""
validate_episode.py — Verificador Fisico e Independiente de Demostraciones UAV.

Audita y certifica que un episodio grabado en PX4 SITL + Gazebo Harmonic sea una
demostracion fisicamente valida antes de ser aceptada en el dataset de Instant Policy.

Separa explicitamente cuatro dimensiones de estado (Atendiendo Revision Tecnica, Punto 2):
  1. pilot_exit_code:   Codigo de salida del proceso del controlador experto.
  2. recording_status:  Integridad del archivo .mcap, presencia de topicos obligatorios,
                        continuidad temporal del reloj de simulacion, ausencia de NaNs
                        y estabilidad del estimador EKF2 (sin resets en vuelo).
  3. mission_status:    Verificacion geometrica independiente de cada primitiva ordenada
                        en waypoints.yaml (traverse, slalom left/right, overfly, orbit)
                        más despegue y aterrizaje confirmado sobre la plataforma.
  4. validation_status: Veredicto final ('accepted' o 'rejected') junto con la lista
                        detallada de causas de rechazo ('rejection_reasons') y metricas
                        continuas de calidad ('metrics').

Soporta lectura dual:
  - Lector binario nativo en Python puro (struct CDR sobre MCAP sin compresion) para
    operar sin dependencias externas en el entorno ROS 2 del sistema.
  - Fallback automatico a la libreria `rosbags` (en ~/venv_ip) para bolsas comprimidas.
"""

import argparse
import json
import math
import struct
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
import yaml

# Inyectar ~/venv_ip y ~/.local en sys.path por si se requiere fallback a rosbags
_py_ver = f"python{sys.version_info.major}.{sys.version_info.minor}"
for _cand in [
    Path.home() / "venv_ip" / "lib" / _py_ver / "site-packages",
    Path.home() / ".local" / "lib" / _py_ver / "site-packages",
]:
    if _cand.is_dir() and str(_cand) not in sys.path:
        sys.path.insert(0, str(_cand))

from uav_ip.synthetic_lidar import (
    extract_yaw_from_rot_matrix,
    ned_to_gazebo_enu,
    px4_odometry_to_gazebo_enu,
    yaw_ned_to_gazebo_enu,
)

# ============================================================================
# Constantes Geometricas y Umbrales Fisicos de Validacion
# ============================================================================

# Geometria nominal de obstaculos (sincronizada con world_generator.py y synthetic_lidar.py)
GATE_HALF_WIDTH = 0.70          # Pilares izquierdo/derecho centrados en y = ±0.70 m
GATE_HALF_HEIGHT = 0.75         # Vigas superior/inferior centradas en z = ±0.75 m
GATE_BAR_THICKNESS = 0.10       # Espesor de las 4 barras (luz libre: 1.30m ancho x 1.40m alto)
CIRCULAR_MAJOR_RADIUS = 0.75    # Radio mayor del toroide (m)
CIRCULAR_TUBE_RADIUS = 0.05     # Radio del tubo del toroide (luz libre radial: 0.70 m)
POLE_RADIUS = 0.10              # Radio del cilindro vertical (m)
POLE_HEIGHT = 3.00              # Altura total del poste apoyado en z=0 (m)
LANDING_PAD_HALF_SIZE = 1.00    # Plataforma cuadrada de 2.0m x 2.0m
LANDING_PAD_TOP_Z = 0.10        # Cota superior de plataforma_llegada (centro z=0.05, alto 0.10)

# Envolvente de seguridad del cuadricoptero gz_x500
UAV_COLLISION_RADIUS_XY = 0.18  # Radio minimo admisible a superficies solidas de obstaculos (m)
UAV_SAFE_POLE_CENTER_DIST = POLE_RADIUS + 0.25  # Distancia 2D minima al eje de un poste (0.35 m)

# Criterios de exito por primitiva
TRAVERSE_MAX_ABS_Y = 0.48       # Maximo desvio lateral al cruzar el plano del aro (m) (luz libre ±0.65m)
TRAVERSE_MAX_ABS_Z = 0.52       # Maximo desvio vertical al cruzar el plano del aro (m) (luz libre ±0.70m)
TRAVERSE_CIRC_MAX_R = 0.50      # Maximo radio al cruzar aro circular (m) (luz libre 0.70m)

SLALOM_MIN_LATERAL = 0.42       # Separacion lateral minima en el cruce x_loc=0 (m)
SLALOM_MAX_LATERAL = 2.80       # Separacion lateral maxima para no aceptar excursiones ajenas (m)

OVERFLY_MIN_ALT_M = 3.25        # Altitud minima absoluta al cruzar sobre un poste de 3.0m (m)
OVERFLY_MAX_LAT_M = 0.90        # Desvio lateral maximo permitido al sobrevolar el poste (m)

ORBIT_MIN_SWEEP_DEG = 210.0     # Barrido angular neto minimo requerido para la orbita de 270° (grados)
ORBIT_MAX_RADIUS_M = 3.20       # Radio maximo de vecindad para computar la orbita alrededor del poste (m)

LANDING_MAX_XY_ERR_M = 0.85     # Distancia horizontal maxima al centro de la plataforma al aterrizar (m)
LANDING_MAX_FINAL_Z_M = 0.32    # Altitud ENU maxima al finalizar el aterrizaje (m)
LANDING_MAX_FINAL_SPEED = 0.40  # Velocidad maxima residual al corte de grabacion (m/s)
TAKEOFF_MIN_ALT_M = 0.95        # Altitud minima que debe alcanzar el vuelo (m)
MIN_ACTIVE_GROUND_Z_M = 0.15    # Altitud minima en vuelo activo fuera de despegue/aterrizaje (m)


# ============================================================================
# Lector Nativo MCAP + CDR (Zero-Dependency) y Fallback a rosbags
# ============================================================================

def _read_mcap_str(buf: bytes, offset: int) -> Tuple[str, int]:
    length = struct.unpack_from("<I", buf, offset)[0]
    end = offset + 4 + length
    return buf[offset + 4:end].decode("utf-8", errors="replace"), end


def _parse_mcap_native(mcap_path: Path) -> Optional[Dict[str, List[Dict[str, Any]]]]:
    """
    Lee directamente un archivo .mcap sin compresion (formato por defecto de ros2 bag record)
    y deserializa los mensajes CDR de tamano fijo de px4_msgs usando `struct` de la biblioteca
    estandar. Si encuentra chunks comprimidos (zstd/lz4), retorna None para delegar a `rosbags`.
    """
    buf = mcap_path.read_bytes()
    if len(buf) < 16 or buf[:8] != b"\x89MCAP0\r\n":
        return None

    channels: Dict[int, str] = {}
    telemetry: Dict[str, List[Dict[str, Any]]] = {
        "odometry": [],
        "local_pos": [],
        "land_detected": [],
        "vehicle_status": [],
        "setpoints": [],
    }
    compressed_chunk_found = False

    def _walk_records(b: bytes, start: int, stop: int) -> None:
        nonlocal compressed_chunk_found
        o = start
        while o + 9 <= stop:
            op = b[o]
            rlen = struct.unpack_from("<Q", b, o + 1)[0]
            rec_start = o + 9
            rec_end = rec_start + rlen
            if rec_end > stop:
                break

            if op == 0x04:  # Channel record
                cid = struct.unpack_from("<H", b, rec_start)[0]
                topic, _ = _read_mcap_str(b, rec_start + 4)
                channels[cid] = topic

            elif op == 0x05:  # Message record
                cid = struct.unpack_from("<H", b, rec_start)[0]
                log_time = struct.unpack_from("<Q", b, rec_start + 6)[0]
                raw = b[rec_start + 22:rec_end]
                topic = channels.get(cid, "")

                if topic == "/fmu/out/vehicle_odometry" and len(raw) >= 118:
                    ts_us, ts_sample_us = struct.unpack_from("<QQ", raw, 4)
                    pose_frame = int(raw[20])
                    px, py, pz = struct.unpack_from("<fff", raw, 24)
                    qw, qx, qy, qz = struct.unpack_from("<ffff", raw, 36)
                    vel_frame = int(raw[52])
                    vx, vy, vz = struct.unpack_from("<fff", raw, 56)
                    wx, wy, wz = struct.unpack_from("<fff", raw, 68)
                    reset_counter = int(raw[116])
                    quality = int(struct.unpack_from("<b", raw, 117)[0])
                    telemetry["odometry"].append({
                        "timestamp": int(log_time),
                        "timestamp_us": int(ts_us),
                        "timestamp_sample_us": int(ts_sample_us),
                        "pose_frame": pose_frame,
                        "position": [float(px), float(py), float(pz)],
                        "q": [float(qw), float(qx), float(qy), float(qz)],
                        "velocity_frame": vel_frame,
                        "velocity": [float(vx), float(vy), float(vz)],
                        "angular_velocity": [float(wx), float(wy), float(wz)],
                        "reset_counter": reset_counter,
                        "quality": quality,
                    })

                elif topic == "/fmu/out/vehicle_local_position_v1" and len(raw) >= 120:
                    ts_us, ts_sample_us = struct.unpack_from("<QQ", raw, 4)
                    xy_valid = bool(raw[20])
                    z_valid = bool(raw[21])
                    x, y, z = struct.unpack_from("<fff", raw, 24)
                    vx, vy, vz = struct.unpack_from("<fff", raw, 48)
                    heading = float(struct.unpack_from("<f", raw, 88)[0])
                    telemetry["local_pos"].append({
                        "timestamp": int(log_time),
                        "timestamp_us": int(ts_us),
                        "timestamp_sample_us": int(ts_sample_us),
                        "x": float(x),
                        "y": float(y),
                        "z": float(z),
                        "vx": float(vx),
                        "vy": float(vy),
                        "vz": float(vz),
                        "heading": heading,
                        "xy_valid": xy_valid,
                        "z_valid": z_valid,
                    })

                elif topic == "/fmu/out/vehicle_land_detected" and len(raw) >= 16:
                    ts_us = struct.unpack_from("<Q", raw, 4)[0]
                    freefall, ground_contact, maybe_landed, landed = struct.unpack_from("<BBBB", raw, 12)
                    at_rest = bool(raw[21]) if len(raw) >= 22 else bool(landed)
                    telemetry["land_detected"].append({
                        "timestamp": int(log_time),
                        "timestamp_us": int(ts_us),
                        "freefall": bool(freefall),
                        "ground_contact": bool(ground_contact),
                        "maybe_landed": bool(maybe_landed),
                        "landed": bool(landed),
                        "at_rest": at_rest,
                    })

                elif topic == "/fmu/out/vehicle_status_v4" and len(raw) >= 40:
                    ts_us = struct.unpack_from("<Q", raw, 4)[0]
                    arming_state = int(raw[28])
                    nav_state = int(raw[41]) if len(raw) > 41 else 0
                    telemetry["vehicle_status"].append({
                        "timestamp": int(log_time),
                        "timestamp_us": int(ts_us),
                        "arming_state": arming_state,
                        "nav_state": nav_state,
                    })

                elif topic == "/fmu/in/trajectory_setpoint" and len(raw) >= 68:
                    ts_us = struct.unpack_from("<Q", raw, 4)[0]
                    spx, spy, spz = struct.unpack_from("<fff", raw, 12)
                    svx, svy, svz = struct.unpack_from("<fff", raw, 24)
                    sp_yaw, sp_yawspeed = struct.unpack_from("<ff", raw, 60)
                    telemetry["setpoints"].append({
                        "timestamp": int(log_time),
                        "timestamp_us": int(ts_us),
                        "position": [float(spx), float(spy), float(spz)],
                        "velocity": [float(svx), float(svy), float(svz)],
                        "yaw": float(sp_yaw),
                        "yawspeed": float(sp_yawspeed),
                    })

            elif op == 0x06:  # Chunk record
                comp, p = _read_mcap_str(b, rec_start + 28)
                clen = struct.unpack_from("<Q", b, p)[0]
                if comp == "":
                    _walk_records(b, p + 8, p + 8 + clen)
                else:
                    compressed_chunk_found = True
                    return

            o = rec_end

    _walk_records(buf, 8, max(8, len(buf) - 8))
    if compressed_chunk_found:
        return None
    return telemetry


def load_episode_telemetry(bag_path: Path) -> Dict[str, List[Dict[str, Any]]]:
    """
    Carga la telemetria completa de un episodio desde un directorio `rosbag/` o archivo `.mcap`.
    Intenta primero el parser nativo en Python puro y recurre a `rosbags` si hay compresion.
    """
    bag_p = Path(bag_path)
    if bag_p.is_dir():
        mcaps = sorted(bag_p.glob("*.mcap"))
        if not mcaps and (bag_p / "rosbag").is_dir():
            mcaps = sorted((bag_p / "rosbag").glob("*.mcap"))
    elif bag_p.is_file() and bag_p.suffix == ".mcap":
        mcaps = [bag_p]
    else:
        mcaps = []

    if not mcaps:
        raise FileNotFoundError(f"No se encontraron archivos .mcap en {bag_path}")

    # 1. Intentar lectura directa sin dependencias externas
    merged: Dict[str, List[Dict[str, Any]]] = {
        "odometry": [],
        "local_pos": [],
        "land_detected": [],
        "vehicle_status": [],
        "setpoints": [],
    }
    native_ok = True
    for mp in mcaps:
        if mp.stat().st_size == 0:
            raise ValueError(f"Archivo .mcap vacio (0 bytes): {mp}")
        res = _parse_mcap_native(mp)
        if res is None:
            native_ok = False
            break
        for k in merged:
            merged[k].extend(res[k])

    if native_ok:
        return merged

    # 2. Fallback a rosbags (para chunks comprimidos zstd/lz4)
    from rosbags.highlevel import AnyReader
    from rosbags.typesys import Stores, get_typestore

    typestore = get_typestore(getattr(Stores, "ROS2_JAZZY", Stores.LATEST))
    merged = {k: [] for k in merged}
    with AnyReader(mcaps, default_typestore=typestore) as reader:
        for connection, timestamp, rawdata in reader.messages():
            topic = connection.topic
            msg = reader.deserialize(rawdata, connection.msgtype)
            if topic == "/fmu/out/vehicle_odometry":
                merged["odometry"].append({
                    "timestamp": int(timestamp),
                    "timestamp_us": int(getattr(msg, "timestamp", 0)),
                    "timestamp_sample_us": int(getattr(msg, "timestamp_sample", 0)),
                    "pose_frame": int(getattr(msg, "pose_frame", 1)),
                    "position": [float(x) for x in msg.position],
                    "q": [float(x) for x in msg.q],
                    "velocity_frame": int(getattr(msg, "velocity_frame", 1)),
                    "velocity": [float(x) for x in msg.velocity],
                    "angular_velocity": [float(x) for x in getattr(msg, "angular_velocity", [0.0, 0.0, 0.0])],
                    "reset_counter": int(getattr(msg, "reset_counter", 0)),
                    "quality": int(getattr(msg, "quality", 0)),
                })
            elif topic == "/fmu/out/vehicle_local_position_v1":
                merged["local_pos"].append({
                    "timestamp": int(timestamp),
                    "timestamp_us": int(getattr(msg, "timestamp", 0)),
                    "x": float(msg.x),
                    "y": float(msg.y),
                    "z": float(msg.z),
                    "vx": float(getattr(msg, "vx", 0.0)),
                    "vy": float(getattr(msg, "vy", 0.0)),
                    "vz": float(getattr(msg, "vz", 0.0)),
                    "heading": float(getattr(msg, "heading", 0.0)),
                    "xy_valid": bool(getattr(msg, "xy_valid", True)),
                    "z_valid": bool(getattr(msg, "z_valid", True)),
                })
            elif topic == "/fmu/out/vehicle_land_detected":
                merged["land_detected"].append({
                    "timestamp": int(timestamp),
                    "timestamp_us": int(getattr(msg, "timestamp", 0)),
                    "freefall": bool(getattr(msg, "freefall", False)),
                    "ground_contact": bool(getattr(msg, "ground_contact", False)),
                    "maybe_landed": bool(getattr(msg, "maybe_landed", False)),
                    "landed": bool(getattr(msg, "landed", False)),
                    "at_rest": bool(getattr(msg, "at_rest", False)),
                })
            elif topic == "/fmu/out/vehicle_status_v4":
                merged["vehicle_status"].append({
                    "timestamp": int(timestamp),
                    "timestamp_us": int(getattr(msg, "timestamp", 0)),
                    "arming_state": int(getattr(msg, "arming_state", 0)),
                    "nav_state": int(getattr(msg, "nav_state", 0)),
                })
            elif topic == "/fmu/in/trajectory_setpoint":
                merged["setpoints"].append({
                    "timestamp": int(timestamp),
                    "timestamp_us": int(getattr(msg, "timestamp", 0)),
                    "position": [float(x) for x in msg.position],
                    "velocity": [float(x) for x in getattr(msg, "velocity", [float("nan")] * 3)],
                    "yaw": float(getattr(msg, "yaw", 0.0)),
                })
    return merged


# ============================================================================
# Geometria Analitica de Distancia a Obstaculos y Verificacion de Primitivas
# ============================================================================

def _world_to_landmark_local(
    pos_enu: np.ndarray,
    lm_x: float,
    lm_y: float,
    lm_z: float,
    lm_yaw: float,
) -> np.ndarray:
    """
    Transforma puntos `pos_enu` (N, 3) en Gazebo ENU al marco local FLU del landmark:
      +x_loc: direccion de avance / normal del landmark (cos(yaw), sin(yaw))
      +y_loc: flanco izquierdo del landmark (-sin(yaw), cos(yaw))
      +z_loc: vertical hacia arriba (z - lm_z)
    """
    dx = pos_enu[:, 0] - float(lm_x)
    dy = pos_enu[:, 1] - float(lm_y)
    dz = pos_enu[:, 2] - float(lm_z)
    c = math.cos(float(lm_yaw))
    s = math.sin(float(lm_yaw))
    x_loc = dx * c + dy * s
    y_loc = -dx * s + dy * c
    return np.column_stack([x_loc, y_loc, dz])


def _points_to_aabb_distance(
    pts_local: np.ndarray,
    box_center: Tuple[float, float, float],
    box_size: Tuple[float, float, float],
) -> np.ndarray:
    """Calcula la distancia euclidea exacta desde N puntos (N, 3) a una caja 3D alineada."""
    c = np.asarray(box_center, dtype=np.float64)
    half = 0.5 * np.asarray(box_size, dtype=np.float64)
    excess = np.maximum(0.0, np.abs(pts_local - c) - half)
    return np.linalg.norm(excess, axis=1)


def compute_min_distance_to_landmark(
    pos_enu: np.ndarray,
    landmark: Dict[str, Any],
) -> float:
    """
    Calcula la distancia minima euclidea (en metros) desde la trayectoria 3D `pos_enu` (N, 3)
    hasta la superficie solida de un landmark ('gate', 'circular_gate', o 'pole').
    """
    if len(pos_enu) == 0:
        return float("inf")

    lm_type = str(landmark.get("type", "gate")).lower()
    cx = float(landmark["x"])
    cy = float(landmark["y"])
    cz = float(landmark.get("z", 1.5))
    yaw = float(landmark.get("yaw", 0.0))

    if "pole" in lm_type or "cyl" in lm_type:
        # Poste cilindrico vertical en (cx, cy), z in [0, POLE_HEIGHT], radio POLE_RADIUS
        r_xy = np.hypot(pos_enu[:, 0] - cx, pos_enu[:, 1] - cy)
        d_rad = np.maximum(0.0, r_xy - POLE_RADIUS)
        d_z_top = np.maximum(0.0, pos_enu[:, 2] - POLE_HEIGHT)
        d_z_bot = np.maximum(0.0, 0.0 - pos_enu[:, 2])
        d_z = np.maximum(d_z_top, d_z_bot)
        dists = np.hypot(d_rad, d_z)
        return float(np.min(dists))

    pts_loc = _world_to_landmark_local(pos_enu, cx, cy, cz, yaw)

    if "circ" in lm_type or "torus" in lm_type:
        # Toroide en el plano Y-Z local con radio mayor R=0.75 y tubo r=0.05
        r_yz = np.hypot(pts_loc[:, 1], pts_loc[:, 2])
        d_tube_center = np.hypot(r_yz - CIRCULAR_MAJOR_RADIUS, pts_loc[:, 0])
        dists = np.maximum(0.0, d_tube_center - CIRCULAR_TUBE_RADIUS)
        return float(np.min(dists))

    # Puerta cuadrada: 4 barras identicas a world_generator._create_gate_element()
    bars = [
        ((0.0, 0.0, 0.75), (0.10, 1.50, 0.10)),
        ((0.0, 0.0, -0.75), (0.10, 1.50, 0.10)),
        ((0.0, 0.70, 0.0), (0.10, 0.10, 1.40)),
        ((0.0, -0.70, 0.0), (0.10, 0.10, 1.40)),
    ]
    bar_dists = [_points_to_aabb_distance(pts_loc, b_c, b_s) for b_c, b_s in bars]
    min_per_step = np.minimum.reduce(bar_dists)
    return float(np.min(min_per_step))


def _verify_single_primitive(
    pos_enu: np.ndarray,
    yaw_enu: np.ndarray,
    prim: Dict[str, Any],
    landmark: Dict[str, Any],
) -> Dict[str, Any]:
    """
    Verifica geometricamente una primitiva individual sobre la trayectoria medida.
    Retorna un diccionario con `passed` (bool), `reason` (Optional[str]) y metricas especificas.
    """
    p_type = str(prim.get("type", "traverse")).lower()
    target_id = str(prim.get("target", landmark.get("id", "unknown")))
    lm_type = str(landmark.get("type", "gate")).lower()
    cx = float(landmark["x"])
    cy = float(landmark["y"])
    cz = float(landmark.get("z", 1.5))
    lm_yaw = float(landmark.get("yaw", 0.0))

    pts_loc = _world_to_landmark_local(pos_enu, cx, cy, cz, lm_yaw)
    x_loc = pts_loc[:, 0]
    y_loc = pts_loc[:, 1]
    z_loc = pts_loc[:, 2]
    dist_xy = np.hypot(x_loc, y_loc)
    min_surf_dist = compute_min_distance_to_landmark(pos_enu, landmark)

    result: Dict[str, Any] = {
        "target": target_id,
        "primitive": p_type,
        "landmark_type": lm_type,
        "min_surface_clearance_m": round(min_surf_dist, 4),
        "passed": False,
        "reason": None,
    }

    # 1. Comprobar colision directa contra la estructura del landmark
    if min_surf_dist < UAV_COLLISION_RADIUS_XY:
        result["reason"] = (
            f"collision_with_{target_id} (min_clearance={min_surf_dist:.3f}m < {UAV_COLLISION_RADIUS_XY:.2f}m)"
        )
        return result

    # 2. Evaluar segun el tipo de primitiva
    if p_type == "traverse":
        # Buscar cruces del plano x_loc = 0 en direccion +x_loc dentro de la vecindad del aro
        cross_candidates = np.where(
            (x_loc[:-1] <= 0.0) & (x_loc[1:] > 0.0) & (dist_xy[:-1] < 2.5)
        )[0]
        if len(cross_candidates) == 0:
            result["reason"] = f"traverse_{target_id}_no_plane_crossing"
            return result

        # Elegir el cruce mas cercano al centro del aro
        best_cross = None
        best_err = float("inf")
        for idx in cross_candidates:
            dx_step = float(x_loc[idx + 1] - x_loc[idx])
            alpha = float(-x_loc[idx] / dx_step) if abs(dx_step) > 1e-9 else 0.5
            alpha = max(0.0, min(1.0, alpha))
            y_c = float(y_loc[idx] + alpha * (y_loc[idx + 1] - y_loc[idx]))
            z_c = float(z_loc[idx] + alpha * (z_loc[idx + 1] - z_loc[idx]))
            r_c = math.hypot(y_c, z_c)
            if r_c < best_err:
                best_err = r_c
                best_cross = (y_c, z_c, r_c)

        assert best_cross is not None
        y_c, z_c, r_c = best_cross
        result["cross_y_m"] = round(y_c, 4)
        result["cross_z_m"] = round(z_c, 4)
        result["cross_radial_err_m"] = round(r_c, 4)

        if "circ" in lm_type or "torus" in lm_type:
            if r_c > TRAVERSE_CIRC_MAX_R:
                result["reason"] = (
                    f"traverse_{target_id}_outside_circular_aperture (r={r_c:.3f}m > {TRAVERSE_CIRC_MAX_R:.2f}m)"
                )
                return result
        else:
            if abs(y_c) > TRAVERSE_MAX_ABS_Y or abs(z_c) > TRAVERSE_MAX_ABS_Z:
                result["reason"] = (
                    f"traverse_{target_id}_outside_gate_aperture "
                    f"(|y|={abs(y_c):.3f}m vs {TRAVERSE_MAX_ABS_Y:.2f}m, "
                    f"|z|={abs(z_c):.3f}m vs {TRAVERSE_MAX_ABS_Z:.2f}m)"
                )
                return result

        result["passed"] = True
        return result

    elif p_type == "slalom":
        side = str(prim.get("side", "right")).lower()
        result["side"] = side
        min_center_dist_2d = float(np.min(dist_xy))
        result["min_pole_center_dist_2d_m"] = round(min_center_dist_2d, 4)

        if min_center_dist_2d < UAV_SAFE_POLE_CENTER_DIST:
            result["reason"] = (
                f"slalom_{target_id}_too_close_to_pole "
                f"(min_dist_2d={min_center_dist_2d:.3f}m < {UAV_SAFE_POLE_CENTER_DIST:.2f}m)"
            )
            return result

        # Verificar que el dron pasa de antes del poste (x_loc < -0.5) a despues (x_loc > +0.5)
        cross_indices = np.where(
            (x_loc[:-1] <= 0.0) & (x_loc[1:] > 0.0) & (np.abs(y_loc[:-1]) <= SLALOM_MAX_LATERAL + 0.5)
        )[0]
        if len(cross_indices) == 0:
            result["reason"] = f"slalom_{target_id}_no_longitudinal_pass"
            return result

        idx = int(cross_indices[0])
        dx_step = float(x_loc[idx + 1] - x_loc[idx])
        alpha = float(-x_loc[idx] / dx_step) if abs(dx_step) > 1e-9 else 0.5
        y_c = float(y_loc[idx] + alpha * (y_loc[idx + 1] - y_loc[idx]))
        result["cross_y_m"] = round(y_c, 4)

        # En el marco local FLU del landmark: +y_loc es IZQUIERDA ('left'), -y_loc es DERECHA ('right')
        if side == "left":
            if not (SLALOM_MIN_LATERAL <= y_c <= SLALOM_MAX_LATERAL):
                result["reason"] = (
                    f"slalom_{target_id}_wrong_flank_or_offset_left "
                    f"(y_cross={y_c:+.3f}m not in [+{SLALOM_MIN_LATERAL:.2f}, +{SLALOM_MAX_LATERAL:.2f}]m)"
                )
                return result
        else:  # 'right'
            if not (-SLALOM_MAX_LATERAL <= y_c <= -SLALOM_MIN_LATERAL):
                result["reason"] = (
                    f"slalom_{target_id}_wrong_flank_or_offset_right "
                    f"(y_cross={y_c:+.3f}m not in [-{SLALOM_MAX_LATERAL:.2f}, -{SLALOM_MIN_LATERAL:.2f}]m)"
                )
                return result

        result["passed"] = True
        return result

    elif p_type == "overfly":
        # Verificar cruce longitudinal sobre la huella del poste y cota superior
        cross_indices = np.where((x_loc[:-1] <= 0.0) & (x_loc[1:] > 0.0))[0]
        if len(cross_indices) == 0:
            result["reason"] = f"overfly_{target_id}_no_longitudinal_pass"
            return result

        idx = int(cross_indices[0])
        dx_step = float(x_loc[idx + 1] - x_loc[idx])
        alpha = float(-x_loc[idx] / dx_step) if abs(dx_step) > 1e-9 else 0.5
        y_c = float(y_loc[idx] + alpha * (y_loc[idx + 1] - y_loc[idx]))
        z_world_c = float(pos_enu[idx, 2] + alpha * (pos_enu[idx + 1, 2] - pos_enu[idx, 2]))

        # Ademas verificar todos los puntos dentro del cilindro vertical de radio 0.50m sobre el poste
        over_mask = dist_xy <= 0.50
        min_z_over = float(np.min(pos_enu[over_mask, 2])) if np.any(over_mask) else z_world_c

        result["cross_y_m"] = round(y_c, 4)
        result["cross_z_world_m"] = round(z_world_c, 4)
        result["min_z_over_pole_m"] = round(min_z_over, 4)

        if abs(y_c) > OVERFLY_MAX_LAT_M:
            result["reason"] = (
                f"overfly_{target_id}_excessive_lateral_offset (|y|={abs(y_c):.3f}m > {OVERFLY_MAX_LAT_M:.2f}m)"
            )
            return result

        if min_z_over < OVERFLY_MIN_ALT_M or z_world_c < OVERFLY_MIN_ALT_M:
            result["reason"] = (
                f"overfly_{target_id}_insufficient_clearance_above_pole "
                f"(min_z={min(min_z_over, z_world_c):.3f}m < {OVERFLY_MIN_ALT_M:.2f}m)"
            )
            return result

        result["passed"] = True
        return result

    elif p_type == "orbit":
        min_center_dist_2d = float(np.min(dist_xy))
        result["min_pole_center_dist_2d_m"] = round(min_center_dist_2d, 4)
        if min_center_dist_2d < UAV_SAFE_POLE_CENTER_DIST:
            result["reason"] = (
                f"orbit_{target_id}_pole_collision_or_shortcut "
                f"(min_dist_2d={min_center_dist_2d:.3f}m < {UAV_SAFE_POLE_CENTER_DIST:.2f}m)"
            )
            return result

        near_indices = np.where(dist_xy <= ORBIT_MAX_RADIUS_M)[0]
        if len(near_indices) < 10:
            result["reason"] = f"orbit_{target_id}_never_entered_orbit_radius"
            return result

        # Tomar el segmento contiguo principal dentro de ORBIT_MAX_RADIUS_M
        splits = np.where(np.diff(near_indices) > 5)[0]
        segments = np.split(near_indices, splits + 1)
        main_seg = max(segments, key=len)

        angles = np.unwrap(np.arctan2(y_loc[main_seg], x_loc[main_seg]))
        # Progreso angular monotono maximo en sentido antihorario (de -pi a +pi/2 = +270 deg)
        # antes de la salida tangencial hacia el siguiente landmark en +X
        running_min = np.minimum.accumulate(angles)
        ccw_progression_deg = math.degrees(float(np.max(angles - running_min)))
        max_span_deg = math.degrees(float(np.max(angles) - np.min(angles)))
        mean_r = float(np.mean(dist_xy[main_seg]))

        # Error de orientacion centripeta durante el arco orbital (excluyendo cola de salida x_loc > 1.2)
        arc_mask = dist_xy[main_seg] <= 2.2
        arc_seg = main_seg[arc_mask] if np.any(arc_mask) else main_seg
        desired_yaw = np.arctan2(cy - pos_enu[arc_seg, 1], cx - pos_enu[arc_seg, 0])
        yaw_seg = yaw_enu[arc_seg].reshape(-1)
        yaw_diff = np.arctan2(np.sin(yaw_seg - desired_yaw), np.cos(yaw_seg - desired_yaw))
        mean_yaw_err_deg = math.degrees(float(np.mean(np.abs(yaw_diff))))

        result["ccw_sweep_deg"] = round(ccw_progression_deg, 2)
        result["angular_span_deg"] = round(max_span_deg, 2)
        result["mean_orbit_radius_m"] = round(mean_r, 4)
        result["mean_centripetal_yaw_err_deg"] = round(mean_yaw_err_deg, 2)

        if ccw_progression_deg < ORBIT_MIN_SWEEP_DEG:
            result["reason"] = (
                f"orbit_{target_id}_incomplete_angular_sweep "
                f"(ccw_sweep={ccw_progression_deg:.1f}deg < {ORBIT_MIN_SWEEP_DEG:.1f}deg)"
            )
            return result

        result["passed"] = True
        return result

    else:
        result["reason"] = f"unsupported_primitive_type_{p_type}"
        return result


def reconstruct_simulation_clock(
    ts_px4_s: np.ndarray,
    ts_bag_s: np.ndarray,
) -> Tuple[np.ndarray, int]:
    """
    Reconstruye el reloj monotono de simulacion a partir de `msg.timestamp` de PX4 (`ts_px4_s`)
    y el tiempo de grabacion del bag (`ts_bag_s`).

    En PX4 SITL + MicroXRCEAgent, `uxrce_dds_client` suma un `time_offset` al reloj de arranque
    de PX4 (`hrt_absolute_time`). Cuando el filtro `Timesync` actualiza ese offset en vuelo
    (1 a 4 muestras por episodio), `ts_px4_s` exhibe un salto artificial de ~0.2s o cae por
    una muestra al tiempo de boot sin offset, mientras `ts_bag_s` avanza suavemente (~8 ms).
    Esta funcion detecta y corrige esos saltos de offset de XRCE-DDS preservando los `dt`
    reales de simulacion.
    """
    N = len(ts_px4_s)
    if N == 0:
        return np.zeros(0, dtype=np.float64), 0
    if N == 1:
        return np.zeros(1, dtype=np.float64), 0

    dt_px4 = np.diff(ts_px4_s)
    dt_bag = np.diff(ts_bag_s)

    # Un salto de offset de Timesync ocurre cuando dt_px4 difiere de dt_bag en > 50 ms
    # mientras que el bag recibio el mensaje con un intervalo normal (0 <= dt_bag <= 100 ms).
    timesync_jump_mask = (np.abs(dt_px4 - dt_bag) > 0.050) & (dt_bag >= 0.0) & (dt_bag <= 0.100)
    num_timesync_jumps = int(np.sum(timesync_jump_mask))

    dt_sim = np.where(timesync_jump_mask, dt_bag, dt_px4)
    ts_sim_rel = np.concatenate([[0.0], np.cumsum(dt_sim)])
    return ts_sim_rel, num_timesync_jumps


# ============================================================================
# Funcion Principal de Validacion de un Episodio
# ============================================================================

def validate_episode_data(
    telemetry: Dict[str, List[Dict[str, Any]]],
    waypoints_cfg: Dict[str, Any],
    pilot_exit_code: Optional[int] = 0,
) -> Dict[str, Any]:
    """
    Ejecuta la validacion fisica, geometrica y de grabacion completa sobre datos en memoria.

    Args:
        telemetry: Diccionario retornado por `load_episode_telemetry()`.
        waypoints_cfg: Diccionario cargado desde `waypoints.yaml`.
        pilot_exit_code: Codigo de salida reportado por el piloto (default: 0).

    Returns:
        Diccionario con:
          - `pilot_exit_code`: int
          - `recording_status`: 'valid' | 'invalid'
          - `mission_status`: 'valid' | 'invalid'
          - `validation_status`: 'accepted' | 'rejected'
          - `rejection_reasons`: List[str]
          - `primitive_results`: List[Dict]
          - `metrics`: Dict[str, float]
    """
    rejection_reasons: List[str] = []

    if pilot_exit_code is not None and int(pilot_exit_code) != 0:
        rejection_reasons.append(f"pilot_non_zero_exit_code ({pilot_exit_code})")

    odom_list = telemetry.get("odometry", [])
    local_pos_list = telemetry.get("local_pos", [])
    land_list = telemetry.get("land_detected", [])

    using_local_pos = False
    if not odom_list and local_pos_list:
        odom_list = local_pos_list
        using_local_pos = True

    if len(odom_list) < 30:
        rejection_reasons.append(f"insufficient_odometry_messages ({len(odom_list)} < 30)")
        return {
            "pilot_exit_code": pilot_exit_code,
            "recording_status": "invalid",
            "mission_status": "invalid",
            "validation_status": "rejected",
            "rejection_reasons": rejection_reasons,
            "primitive_results": [],
            "metrics": {"num_odom_messages": len(odom_list)},
        }

    # 1. Extraer series temporales y validar ausencia de NaNs / cuaterniones corruptos
    N = len(odom_list)
    pos_enu = np.zeros((N, 3), dtype=np.float64)
    vel_enu = np.zeros((N, 3), dtype=np.float64)
    yaw_enu = np.zeros(N, dtype=np.float64)
    ts_px4_raw_s = np.zeros(N, dtype=np.float64)
    ts_bag_raw_s = np.zeros(N, dtype=np.float64)
    reset_counters = np.zeros(N, dtype=np.int64)
    nan_count = 0

    for i, item in enumerate(odom_list):
        t_us = item.get("timestamp_us", 0)
        t_bag_ns = item.get("timestamp", 0)
        ts_bag_raw_s[i] = (float(t_bag_ns) * 1e-9) if t_bag_ns > 0 else (float(i) * 0.01)
        ts_px4_raw_s[i] = (float(t_us) * 1e-6) if t_us > 0 else ts_bag_raw_s[i]

        if not using_local_pos:
            p_ned = item["position"]
            q_ned = item["q"]
            v_ned = item.get("velocity", [0.0, 0.0, 0.0])
            if (
                not np.all(np.isfinite(p_ned))
                or not np.all(np.isfinite(q_ned))
                or not np.all(np.isfinite(v_ned))
            ):
                nan_count += 1
                continue
            q_norm = math.sqrt(sum(float(c) * float(c) for c in q_ned))
            if q_norm < 0.90 or q_norm > 1.10:
                nan_count += 1
                continue
            p_e, R_full = px4_odometry_to_gazebo_enu(p_ned, q_ned, gravity_aligned=False)
            pos_enu[i] = p_e
            vel_enu[i] = ned_to_gazebo_enu(v_ned)
            yaw_enu[i] = extract_yaw_from_rot_matrix(R_full)
            reset_counters[i] = int(item.get("reset_counter", 0))
        else:
            p_ned = [item["x"], item["y"], item["z"]]
            v_ned = [item.get("vx", 0.0), item.get("vy", 0.0), item.get("vz", 0.0)]
            h_ned = item.get("heading", 0.0)
            if not np.all(np.isfinite(p_ned)) or not math.isfinite(h_ned):
                nan_count += 1
                continue
            pos_enu[i] = ned_to_gazebo_enu(p_ned)
            vel_enu[i] = ned_to_gazebo_enu(v_ned)
            yaw_enu[i] = yaw_ned_to_gazebo_enu(h_ned)

    if nan_count > 0:
        rejection_reasons.append(f"corrupted_nan_or_invalid_quaternion_samples ({nan_count})")

    # 2. Reconstruccion y verificacion de continuidad temporal del reloj de simulacion
    ts_sim_s, timesync_offset_jumps = reconstruct_simulation_clock(ts_px4_raw_s, ts_bag_raw_s)
    dt_arr = np.diff(ts_sim_s)
    pos_dt = dt_arr[dt_arr > 0]
    duration_s = float(ts_sim_s[-1] - ts_sim_s[0]) if N > 1 else 0.0
    dt_median_ms = float(np.median(pos_dt) * 1000.0) if len(pos_dt) > 0 else 0.0
    dt_p95_ms = float(np.percentile(pos_dt, 95) * 1000.0) if len(pos_dt) > 0 else 0.0
    max_dt_gap_s = float(np.max(dt_arr)) if len(dt_arr) > 0 else 0.0
    min_dt_s = float(np.min(dt_arr)) if len(dt_arr) > 0 else 0.0

    if duration_s < 3.0:
        rejection_reasons.append(f"flight_duration_too_short ({duration_s:.2f}s < 3.0s)")
    if min_dt_s < -0.05:
        rejection_reasons.append(f"non_monotonic_simulation_clock (min_dt={min_dt_s:.3f}s)")
    if max_dt_gap_s > 0.60:
        rejection_reasons.append(f"telemetry_dropout_gap (max_dt_gap={max_dt_gap_s:.3f}s > 0.60s)")

    # 3. Ventana de vuelo activo y chequeo de resets del estimador EKF2 en vuelo
    airborne_indices = np.where(pos_enu[:, 2] >= 0.45)[0]
    max_alt_m = float(np.max(pos_enu[:, 2]))
    if max_alt_m < TAKEOFF_MIN_ALT_M or len(airborne_indices) < 10:
        rejection_reasons.append(f"takeoff_failed (max_alt={max_alt_m:.2f}m < {TAKEOFF_MIN_ALT_M:.2f}m)")
        in_flight_resets = 0
    else:
        i_takeoff = int(airborne_indices[0])
        i_touchdown = int(airborne_indices[-1])
        rc_flight = reset_counters[i_takeoff:i_touchdown + 1]
        in_flight_resets = int(np.sum(np.diff(rc_flight) != 0)) if len(rc_flight) > 1 else 0
        if in_flight_resets > 0:
            rejection_reasons.append(f"ekf2_estimator_reset_in_flight (resets={in_flight_resets})")

        # Verificar que entre el fin del ascenso inicial y el inicio del descenso final no haya impacto contra el suelo
        cruise_indices = np.where(pos_enu[:, 2] >= 1.0)[0]
        if len(cruise_indices) >= 2:
            c_start = int(cruise_indices[0])
            c_end = int(cruise_indices[-1])
            min_cruise_z = float(np.min(pos_enu[c_start:c_end + 1, 2]))
            if min_cruise_z < MIN_ACTIVE_GROUND_Z_M:
                rejection_reasons.append(
                    f"unexpected_ground_contact_mid_mission (min_cruise_z={min_cruise_z:.2f}m)"
                )

    recording_status = (
        "invalid"
        if any(
            r.startswith((
                "insufficient_",
                "corrupted_",
                "flight_duration_",
                "non_monotonic_",
                "telemetry_dropout_",
                "ekf2_estimator_reset_",
            ))
            for r in rejection_reasons
        )
        else "valid"
    )

    # 4. Verificar cada primitiva en orden de aparicion en el circuito
    raw_lms = waypoints_cfg.get("landmarks") or waypoints_cfg.get("hoops") or []
    landmarks_list: List[Dict[str, Any]] = []
    for lm in raw_lms:
        lm_c = dict(lm)
        if "id" not in lm_c and "name" in lm_c:
            lm_c["id"] = lm_c["name"]
        if "type" not in lm_c:
            lm_c["type"] = "gate"
        landmarks_list.append(lm_c)

    landmarks_map = {str(lm["id"]): lm for lm in landmarks_list}
    primitives = waypoints_cfg.get("primitives") or []
    if not primitives and landmarks_list:
        sorted_lms = sorted(landmarks_list, key=lambda l: float(l.get("x", 0.0)))
        primitives = [{"type": "traverse", "target": lm["id"]} for lm in sorted_lms]

    primitives_sorted = sorted(
        primitives,
        key=lambda p: float(landmarks_map.get(str(p.get("target")), {}).get("x", 0.0)),
    )

    primitive_results: List[Dict[str, Any]] = []
    for prim in primitives_sorted:
        tid = str(prim.get("target", ""))
        if tid not in landmarks_map:
            rejection_reasons.append(f"missing_landmark_definition ({tid})")
            continue
        p_res = _verify_single_primitive(pos_enu, yaw_enu, prim, landmarks_map[tid])
        primitive_results.append(p_res)
        if not p_res["passed"] and p_res["reason"]:
            rejection_reasons.append(str(p_res["reason"]))

    # 5. Verificar colision global contra CUALQUIER landmark del mundo (incluso no objetivo actual)
    min_global_clearance = float("inf")
    for lm in landmarks_list:
        d_lm = compute_min_distance_to_landmark(pos_enu, lm)
        min_global_clearance = min(min_global_clearance, d_lm)
        if d_lm < UAV_COLLISION_RADIUS_XY:
            reason_col = f"collision_with_{lm['id']} (min_clearance={d_lm:.3f}m)"
            if not any(r.startswith(f"collision_with_{lm['id']}") for r in rejection_reasons):
                rejection_reasons.append(reason_col)

    # 6. Verificar aterrizaje final sobre plataforma_llegada
    landing_cfg = waypoints_cfg.get("landing")
    final_pos = pos_enu[-1]
    final_speed = float(np.linalg.norm(vel_enu[-1]))
    landed_flag_confirmed = any(
        bool(item.get("landed", False)) or bool(item.get("ground_contact", False))
        for item in land_list[-10:]
    ) if land_list else False

    landing_xy_err = 0.0
    if landing_cfg and "x" in landing_cfg and "y" in landing_cfg:
        lx, ly = float(landing_cfg["x"]), float(landing_cfg["y"])
        # Medir distancia horizontal en los instantes finales de baja altura
        low_tail_idx = np.where(pos_enu[N // 2:, 2] <= 0.45)[0]
        if len(low_tail_idx) > 0:
            tail_pts = pos_enu[N // 2 + low_tail_idx]
            landing_xy_err = float(np.min(np.hypot(tail_pts[:, 0] - lx, tail_pts[:, 1] - ly)))
        else:
            landing_xy_err = float(math.hypot(final_pos[0] - lx, final_pos[1] - ly))

        if landing_xy_err > LANDING_MAX_XY_ERR_M:
            rejection_reasons.append(
                f"landing_off_pad (xy_err={landing_xy_err:.3f}m > {LANDING_MAX_XY_ERR_M:.2f}m)"
            )

    if final_pos[2] > LANDING_MAX_FINAL_Z_M:
        rejection_reasons.append(
            f"landing_incomplete_altitude (final_z={final_pos[2]:.3f}m > {LANDING_MAX_FINAL_Z_M:.2f}m)"
        )

    if final_speed > LANDING_MAX_FINAL_SPEED and not landed_flag_confirmed:
        rejection_reasons.append(
            f"landing_not_at_rest (final_speed={final_speed:.3f}m/s > {LANDING_MAX_FINAL_SPEED:.2f}m/s)"
        )

    if land_list and not landed_flag_confirmed and final_pos[2] > 0.22:
        rejection_reasons.append("px4_land_detector_not_triggered")

    # 7. Calcular metricas continuas de calidad del piloto experto (Sugerencia 3.4)
    seg_lengths = np.linalg.norm(np.diff(pos_enu, axis=0), axis=1)
    path_length_m = float(np.sum(seg_lengths))
    speeds = np.linalg.norm(vel_enu, axis=1)
    mean_speed_mps = float(np.mean(speeds))
    max_speed_mps = float(np.max(speeds))

    # Estimar aceleracion y jerk suavizados sobre submuestreo a ~10 Hz para evitar ruido de cuantizacion
    step_10hz = max(1, int(round(100.0 / max(1.0, dt_median_ms)))) if dt_median_ms > 0 else 5
    vel_sub = vel_enu[::step_10hz]
    ts_sub = ts_sim_s[::step_10hz]
    if len(ts_sub) >= 4:
        dt_sub = np.maximum(1e-3, np.diff(ts_sub))[:, None]
        acc_sub = np.diff(vel_sub, axis=0) / dt_sub
        acc_norms = np.linalg.norm(acc_sub, axis=1)
        max_acc_mps2 = float(np.percentile(acc_norms, 98))
        dt_sub2 = np.maximum(1e-3, np.diff(ts_sub[1:]))[:, None]
        jerk_sub = np.diff(acc_sub, axis=0) / dt_sub2
        rms_jerk_mps3 = float(np.sqrt(np.mean(np.sum(jerk_sub * jerk_sub, axis=1))))
    else:
        max_acc_mps2 = 0.0
        rms_jerk_mps3 = 0.0

    mission_status = "valid" if len(rejection_reasons) == 0 else "invalid"
    validation_status = "accepted" if (recording_status == "valid" and mission_status == "valid") else "rejected"

    metrics = {
        "num_odom_messages": int(N),
        "duration_s": round(duration_s, 3),
        "dt_median_ms": round(dt_median_ms, 3),
        "dt_p95_ms": round(dt_p95_ms, 3),
        "max_dt_gap_s": round(max_dt_gap_s, 4),
        "timesync_offset_jumps": int(timesync_offset_jumps),
        "in_flight_estimator_resets": int(in_flight_estimator_resets) if "in_flight_estimator_resets" in locals() else int(in_flight_resets),
        "path_length_m": round(path_length_m, 3),
        "max_altitude_m": round(max_alt_m, 3),
        "mean_speed_mps": round(mean_speed_mps, 3),
        "max_speed_mps": round(max_speed_mps, 3),
        "p98_acc_mps2": round(max_acc_mps2, 3),
        "rms_jerk_mps3": round(rms_jerk_mps3, 3),
        "min_obstacle_clearance_m": round(
            min_global_clearance if math.isfinite(min_global_clearance) else -1.0, 4
        ),
        "landing_xy_error_m": round(landing_xy_err, 4),
        "final_altitude_m": round(float(final_pos[2]), 4),
        "final_speed_mps": round(final_speed, 4),
        "landed_detector_confirmed": bool(landed_flag_confirmed),
    }

    return {
        "pilot_exit_code": pilot_exit_code,
        "recording_status": recording_status,
        "mission_status": mission_status,
        "validation_status": validation_status,
        "rejection_reasons": rejection_reasons,
        "primitive_results": primitive_results,
        "metrics": metrics,
    }


def validate_episode_directory(
    ep_dir: Path,
    pilot_exit_code: Optional[int] = None,
    update_meta: bool = False,
) -> Dict[str, Any]:
    """
    Valida un directorio de episodio completo (`datasets/task_XXXX/episode_YY`).
    Si `update_meta=True`, persiste los campos de verificacion independiente dentro de `episode_meta.json`.
    """
    ep_dir = Path(ep_dir).resolve()
    wp_file = ep_dir / "waypoints.yaml"
    if not wp_file.is_file():
         parent_wp = ep_dir.parent / "waypoints.yaml"
         wp_file = parent_wp if parent_wp.is_file() else wp_file

    meta_file = ep_dir / "episode_meta.json"
    existing_meta: Dict[str, Any] = {}
    if meta_file.is_file():
        try:
            with open(meta_file, "r") as f:
                existing_meta = json.load(f)
        except Exception:
            pass

    if pilot_exit_code is None:
        if "pilot_exit_code" in existing_meta:
            pilot_exit_code = existing_meta["pilot_exit_code"]
        else:
            st = str(existing_meta.get("status", "success"))
            pilot_exit_code = 0 if st == "success" else -1

    if not wp_file.is_file():
        report = {
            "pilot_exit_code": pilot_exit_code,
            "recording_status": "invalid",
            "mission_status": "invalid",
            "validation_status": "rejected",
            "rejection_reasons": ["missing_waypoints_yaml"],
            "primitive_results": [],
            "metrics": {},
        }
    else:
        try:
            with open(wp_file, "r") as f:
                wp_cfg = yaml.safe_load(f) or {}
            telemetry = load_episode_telemetry(ep_dir)
            report = validate_episode_data(telemetry, wp_cfg, pilot_exit_code=pilot_exit_code)
        except Exception as e:
            report = {
                "pilot_exit_code": pilot_exit_code,
                "recording_status": "invalid",
                "mission_status": "invalid",
                "validation_status": "rejected",
                "rejection_reasons": [f"bag_read_error: {e}"],
                "primitive_results": [],
                "metrics": {},
            }

    if update_meta and meta_file.is_file():
        existing_meta["pilot_exit_code"] = report["pilot_exit_code"]
        existing_meta["recording_status"] = report["recording_status"]
        existing_meta["mission_status"] = report["mission_status"]
        existing_meta["validation_status"] = report["validation_status"]
        existing_meta["rejection_reasons"] = report["rejection_reasons"]
        existing_meta["validation_metrics"] = report["metrics"]
        existing_meta["primitive_validation"] = report["primitive_results"]
        if report["validation_status"] != "accepted" and existing_meta.get("status") == "success":
            existing_meta["status"] = "rejected"
        with open(meta_file, "w") as f:
            json.dump(existing_meta, f, indent=2)

    return report


# ============================================================================
# CLI
# ============================================================================

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Validador fisico e independiente de demostraciones de vuelo para UAV Instant Policy."
    )
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument(
        "--episode-dir",
        type=Path,
        help="Directorio de un episodio individual (ej. datasets/task_0000/episode_00)",
    )
    group.add_argument(
        "--dataset-dir",
        type=Path,
        help="Directorio raiz del dataset (ej. datasets/) para validar todos los episodios",
    )
    parser.add_argument(
        "--update-meta",
        action="store_true",
        help="Actualiza episode_meta.json con los estados y metricas de validacion fisica",
    )
    parser.add_argument(
        "--strict",
        action="store_true",
        help="Finaliza con exit code 1 si algun episodio resulta rechazado",
    )
    args = parser.parse_args()

    if args.episode_dir is not None:
        rep = validate_episode_directory(args.episode_dir, update_meta=args.update_meta)
        print(json.dumps(rep, indent=2))
        if args.strict and rep["validation_status"] != "accepted":
            sys.exit(1)
        return

    dataset_dir = Path(args.dataset_dir).resolve()
    ep_dirs = sorted(dataset_dir.glob("task_*/episode_*"))
    if not ep_dirs:
        print(f"[ERROR] No se encontraron episodios en {dataset_dir}")
        sys.exit(1)

    print("=" * 88)
    print(f" VALIDACIÓN FÍSICA INDEPENDIENTE DE EPISODIOS — {dataset_dir}")
    print("=" * 88)

    n_accepted = 0
    n_rejected = 0
    for ep_dir in ep_dirs:
        label = f"{ep_dir.parent.name}/{ep_dir.name}"
        rep = validate_episode_directory(ep_dir, update_meta=args.update_meta)
        status = rep["validation_status"].upper()
        m = rep.get("metrics", {})
        prim_summary = ", ".join(
            f"{p['primitive']}({'OK' if p['passed'] else 'FAIL'})"
            for p in rep.get("primitive_results", [])
        )
        if rep["validation_status"] == "accepted":
            n_accepted += 1
            print(
                f"  [ACCEPTED] {label:22s} | T={m.get('duration_s', 0):5.1f}s | "
                f"dt={m.get('dt_median_ms', 0):4.1f}ms | "
                f"min_clear={m.get('min_obstacle_clearance_m', 0):4.2f}m | "
                f"land_err={m.get('landing_xy_error_m', 0):4.2f}m | {prim_summary}"
            )
        else:
            n_rejected += 1
            reasons = "; ".join(rep.get("rejection_reasons", []))
            print(f"  [REJECTED] {label:22s} | CAUSAS: {reasons}")

    print("-" * 88)
    print(
        f" Total: {len(ep_dirs)} episodios | Aceptados: {n_accepted} "
        f"({100.0 * n_accepted / len(ep_dirs):.1f}%) | Rechazados: {n_rejected}"
    )
    print("=" * 88)

    if args.strict and n_rejected > 0:
        sys.exit(1)


if __name__ == "__main__":
    main()
