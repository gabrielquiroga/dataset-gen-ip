#!/usr/bin/env python3
"""
validate_episode.py — Verificador Geometrico, Cinemático y de Grabacion de Demostraciones UAV.

Audita un episodio grabado en PX4 SITL + Gazebo Harmonic bajo hipotesis explicitas de
geometria de obstaculos estaticos, envolvente del vehiculo y continuidad de telemetria
antes de aceptarlo en el dataset de Instant Policy.

Separa explicitamente cuatro dimensiones de estado:
  1. pilot_exit_code:   Codigo de salida del proceso del controlador experto.
  2. recording_status:  Integridad estructural del archivo .mcap (header/footer), presencia
                        de topicos obligatorios, validez de marcos (NED), continuidad temporal
                        del reloj de simulacion, ausencia de NaNs y cero resets de EKF2 en vuelo.
  3. mission_status:    Verificacion geometrica independiente de cada primitiva en orden
                        temporal estricto (traverse, slalom left/right, overfly, orbit),
                        margen libre de colision sobre segmentos subdivididos y aterrizaje
                        sincronizado en la ventana final sobre la plataforma.
  4. validation_status: Veredicto final ('accepted' o 'rejected') junto con la lista
                        detallada de causas de rechazo ('rejection_reasons') y metricas
                        continuas ('metrics').

Estrategia de lectura de bolsas MCAP:
  - Prioriza la libreria mantenida `rosbags` (disponible en ~/venv_ip) tras verificar la
    integridad estructural del contenedor MCAP (magic inicial/final y registro Footer).
  - Incluye un deserializador nativo CDR verificable (`_parse_mcap_native`) como fallback
    cuando se ejecuta en el Python del sistema ROS 2 sin `rosbags`.
"""

import argparse
import json
import math
import os
import struct
import sys
import tempfile
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
import yaml

# Inyectar ~/venv_ip y ~/.local en sys.path para disponer de `rosbags` como lector primario
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
# Constantes Geometricas y Umbrales de Validacion (Sincronizados con Documentacion)
# ============================================================================

VALIDATOR_VERSION: str = "1.1.0"

SUPPORTED_LANDMARK_TYPES = frozenset({"gate", "square_gate", "circular_gate", "pole"})
GATE_LANDMARK_TYPES = frozenset({"gate", "square_gate", "circular_gate"})
POLE_LANDMARK_TYPES = frozenset({"pole"})
SUPPORTED_PRIMITIVE_TYPES = frozenset({"traverse", "slalom", "overfly", "orbit"})
SUPPORTED_SLALOM_SIDES = frozenset({"left", "right"})

# 1. Geometria nominal de obstaculos (identica a world_generator.py y synthetic_lidar.py)
GATE_HALF_WIDTH: float = 0.70           # Pilares izquierdo/derecho centrados en y = ±0.70 m
GATE_HALF_HEIGHT: float = 0.75          # Vigas superior/inferior centradas en z = ±0.75 m
GATE_BAR_THICKNESS: float = 0.10        # Espesor de las 4 barras (semiespesor = 0.05 m)
SQUARE_GATE_CLEAR_HALF_Y: float = GATE_HALF_WIDTH - 0.5 * GATE_BAR_THICKNESS   # 0.65 m (luz libre horizontal: 1.30 m)
SQUARE_GATE_CLEAR_HALF_Z: float = GATE_HALF_HEIGHT - 0.5 * GATE_BAR_THICKNESS  # 0.70 m (luz libre vertical: 1.40 m)

CIRCULAR_MAJOR_RADIUS: float = 0.75     # Radio mayor del anillo circular (m)
CIRCULAR_TUBE_RADIUS: float = 0.05      # Radio de cada cilindro del anillo (m)
CIRCULAR_NUM_SEGMENTS: int = 16         # Segmentos cilindricos en world_generator.py
# Correccion conservadora por cuerda poligonal de 16 segmentos: R * (1 - cos(pi/16)) ≈ 0.0144 m
CIRCULAR_POLY_SAGITTA: float = CIRCULAR_MAJOR_RADIUS * (1.0 - math.cos(math.pi / CIRCULAR_NUM_SEGMENTS))
CIRCULAR_GATE_CLEAR_RADIUS: float = CIRCULAR_MAJOR_RADIUS - CIRCULAR_POLY_SAGITTA - CIRCULAR_TUBE_RADIUS  # ~0.6856 m

POLE_RADIUS: float = 0.10               # Radio del poste cilindrico vertical (m)
POLE_HEIGHT: float = 3.00               # Altura total del poste apoyado en z=0 (m)
LANDING_PAD_HALF_SIZE: float = 1.00     # Plataforma cuadrada de 2.0m x 2.0m

# 2. Envolvente geometrica del cuadricoptero gz_x500
UAV_PHYSICAL_RADIUS_XY: float = 0.23    # Radio fisico de estructura/brazos proyectado (m)
UAV_SAFETY_MARGIN_XY: float = 0.05      # Margen adicional de seguridad geometrica (m)
UAV_COLLISION_RADIUS_XY: float = UAV_PHYSICAL_RADIUS_XY + UAV_SAFETY_MARGIN_XY  # 0.28 m
UAV_SAFE_POLE_CENTER_DIST: float = POLE_RADIUS + UAV_COLLISION_RADIUS_XY        # 0.38 m

# Subdivisión espacial maxima para evaluar colisiones sobre segmentos entre muestras
COLLISION_SEGMENT_MAX_STEP_M: float = 0.02

# 3. Criterios de exito por primitiva derivados de la abertura libre menos UAV_COLLISION_RADIUS_XY
TRAVERSE_MAX_ABS_Y: float = round(SQUARE_GATE_CLEAR_HALF_Y - UAV_COLLISION_RADIUS_XY, 4)      # 0.37 m
TRAVERSE_MAX_ABS_Z: float = round(SQUARE_GATE_CLEAR_HALF_Z - UAV_COLLISION_RADIUS_XY, 4)      # 0.42 m
TRAVERSE_CIRC_MAX_R: float = round(CIRCULAR_GATE_CLEAR_RADIUS - UAV_COLLISION_RADIUS_XY, 4)   # 0.4056 m

SLALOM_MIN_LATERAL: float = UAV_SAFE_POLE_CENTER_DIST + 0.07  # 0.45 m
SLALOM_MAX_LATERAL: float = 2.60                              # 2.60 m

OVERFLY_MIN_ALT_M: float = POLE_HEIGHT + 0.30                 # 3.30 m
OVERFLY_MAX_LAT_M: float = 0.85                               # 0.85 m

# En orbit (arco nominal 270° a R=1.5m con esfera de aceptacion final 0.60m -> ~23.6°):
# exigimos >= 240.0° de barrido antihorario acumulado y error medio de yaw centripeto <= 45.0°
ORBIT_MIN_SWEEP_DEG: float = 240.0
ORBIT_MAX_RADIUS_M: float = 3.20
ORBIT_MAX_MEAN_YAW_ERR_DEG: float = 45.0

# 4. Criterios de despegue, continuidad temporal y ventana sincronizada de aterrizaje
TAKEOFF_MIN_ALT_M: float = 0.95
MIN_ACTIVE_GROUND_Z_M: float = 0.15
MAX_ALLOWED_DT_GAP_S: float = 0.15      # Hueco temporal maximo admisible entre muestras (150 ms)
MAX_ALLOWED_STEP_DIST_M: float = 0.35   # Paso espacial maximo admisible entre muestras consecutivas (m)

LANDING_MAX_XY_ERR_M: float = 0.75      # Error radial maximo en la ventana final de reposo (dentro del pad ±1.0m)
LANDING_MAX_FINAL_Z_M: float = 0.25     # Altitud maxima en la ventana final de reposo (m)
LANDING_MAX_FINAL_SPEED: float = 0.25   # Velocidad maxima en la ventana final de reposo (m/s)
LANDING_MAX_DETECTOR_AGE_S: float = 2.0 # Antigüedad maxima del ultimo mensaje VehicleLandDetected (s)

MCAP_MAGIC = b"\x89MCAP0\r\n"

EXPECTED_TOPIC_SCHEMAS = {
    "/fmu/out/vehicle_odometry": "px4_msgs/msg/VehicleOdometry",
    "/fmu/out/vehicle_local_position_v1": "px4_msgs/msg/VehicleLocalPosition",
    "/fmu/out/vehicle_land_detected": "px4_msgs/msg/VehicleLandDetected",
    "/fmu/out/vehicle_status_v4": "px4_msgs/msg/VehicleStatus",
    "/fmu/in/trajectory_setpoint": "px4_msgs/msg/TrajectorySetpoint",
}


def get_effective_validation_parameters() -> Dict[str, Any]:
    """Retorna un diccionario serializable con la version del validador y parametros efectivos."""
    return {
        "validator_version": VALIDATOR_VERSION,
        "uav_physical_radius_xy_m": UAV_PHYSICAL_RADIUS_XY,
        "uav_safety_margin_xy_m": UAV_SAFETY_MARGIN_XY,
        "uav_collision_radius_xy_m": UAV_COLLISION_RADIUS_XY,
        "collision_segment_max_step_m": COLLISION_SEGMENT_MAX_STEP_M,
        "traverse_max_abs_y_m": TRAVERSE_MAX_ABS_Y,
        "traverse_max_abs_z_m": TRAVERSE_MAX_ABS_Z,
        "traverse_circ_max_r_m": TRAVERSE_CIRC_MAX_R,
        "slalom_min_lateral_m": SLALOM_MIN_LATERAL,
        "slalom_max_lateral_m": SLALOM_MAX_LATERAL,
        "overfly_min_alt_m": OVERFLY_MIN_ALT_M,
        "overfly_max_lat_m": OVERFLY_MAX_LAT_M,
        "orbit_min_sweep_deg": ORBIT_MIN_SWEEP_DEG,
        "orbit_max_radius_m": ORBIT_MAX_RADIUS_M,
        "orbit_max_mean_yaw_err_deg": ORBIT_MAX_MEAN_YAW_ERR_DEG,
        "takeoff_min_alt_m": TAKEOFF_MIN_ALT_M,
        "min_active_ground_z_m": MIN_ACTIVE_GROUND_Z_M,
        "max_allowed_dt_gap_s": MAX_ALLOWED_DT_GAP_S,
        "max_allowed_step_dist_m": MAX_ALLOWED_STEP_DIST_M,
        "landing_max_xy_err_m": LANDING_MAX_XY_ERR_M,
        "landing_max_final_z_m": LANDING_MAX_FINAL_Z_M,
        "landing_max_final_speed_mps": LANDING_MAX_FINAL_SPEED,
        "landing_max_detector_age_s": LANDING_MAX_DETECTOR_AGE_S,
    }



# ============================================================================
# Verificacion de Integridad MCAP y Lectura de Telemetria (rosbags + Fallback CDR)
# ============================================================================

def _read_mcap_str(buf: bytes, offset: int) -> Tuple[str, int]:
    if offset + 4 > len(buf):
        raise ValueError("Truncated MCAP string length")
    length = struct.unpack_from("<I", buf, offset)[0]
    end = offset + 4 + length
    if end > len(buf):
        raise ValueError("Truncated MCAP string payload")
    return buf[offset + 4:end].decode("utf-8", errors="replace"), end


def verify_mcap_integrity(mcap_path: Path) -> bytes:
    """
    Verifica que el archivo `.mcap` no este vacio ni truncado:
      1. Comienza y termina con los 8 bytes magicos `\\x89MCAP0\\r\\n`.
      2. Contiene el registro `Footer` (`opcode = 0x02`, `record_len = 20`) inmediatamente
         antes del magic final.
    Retorna los bytes del archivo o lanza `ValueError` si el archivo esta truncado/corrupto.
    """
    if not mcap_path.is_file():
        raise FileNotFoundError(f"Archivo MCAP no encontrado: {mcap_path}")
    buf = mcap_path.read_bytes()
    if len(buf) < 45:
        raise ValueError(f"truncated_or_empty_mcap ({mcap_path.name}: {len(buf)} bytes)")
    if buf[:8] != MCAP_MAGIC:
        raise ValueError(f"invalid_mcap_header_magic ({mcap_path.name})")
    if buf[-8:] != MCAP_MAGIC:
        raise ValueError(f"truncated_mcap_missing_trailer_magic ({mcap_path.name})")

    footer_pos = len(buf) - 8 - 29
    footer_op = buf[footer_pos]
    footer_len = struct.unpack_from("<Q", buf, footer_pos + 1)[0]
    if footer_op != 0x02 or footer_len != 20:
        raise ValueError(f"corrupted_mcap_footer_record ({mcap_path.name})")
    return buf


def _parse_mcap_native(mcap_path: Path, buf: Optional[bytes] = None) -> Optional[Dict[str, List[Dict[str, Any]]]]:
    """
    Deserializador nativo MCAP + CDR (little-endian XCDR1) verificado contra las definiciones
    de `src/px4_msgs/msg/*.msg`:
      - Valida integridad estructural del archivo MCAP y de cada registro/chunk (rechaza truncados).
      - Verifica el nombre del esquema (`px4_msgs/msg/...`) y `encoding == 'cdr'` de cada canal.
      - Verifica el encabezado de encapsulamiento CDR little-endian (`0x00, 0x01`).
      - Si encuentra chunks comprimidos (zstd/lz4), retorna `None` para delegar a `rosbags`.
    """
    if buf is None:
        buf = verify_mcap_integrity(mcap_path)

    schemas: Dict[int, Tuple[str, str]] = {}
    channels: Dict[int, Tuple[str, str, str]] = {}
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
        while o < stop:
            if o + 9 > stop:
                raise ValueError(f"truncated_mcap_record_header at offset {o}")
            op = b[o]
            rlen = struct.unpack_from("<Q", b, o + 1)[0]
            rec_start = o + 9
            rec_end = rec_start + rlen
            if rec_end > stop:
                raise ValueError(f"truncated_mcap_record_body at offset {o} (expected {rec_end} <= {stop})")

            if op == 0x03:  # Schema record
                sid = struct.unpack_from("<H", b, rec_start)[0]
                sname, next_p = _read_mcap_str(b, rec_start + 2)
                senc, _ = _read_mcap_str(b, next_p)
                schemas[sid] = (sname, senc)

            elif op == 0x04:  # Channel record
                cid, sid = struct.unpack_from("<HH", b, rec_start)
                topic, next_p = _read_mcap_str(b, rec_start + 4)
                msg_enc, _ = _read_mcap_str(b, next_p)
                sname = schemas.get(sid, ("", ""))[0]
                channels[cid] = (topic, sname, msg_enc)

            elif op == 0x05:  # Message record
                if rlen < 22:
                    raise ValueError("corrupted_mcap_message_record_too_short")
                cid = struct.unpack_from("<H", b, rec_start)[0]
                log_time = struct.unpack_from("<Q", b, rec_start + 6)[0]
                raw = b[rec_start + 22:rec_end]
                topic, sname, msg_enc = channels.get(cid, ("", "", ""))

                if topic in EXPECTED_TOPIC_SCHEMAS:
                    expected_schema = EXPECTED_TOPIC_SCHEMAS[topic]
                    if sname and sname != expected_schema:
                        raise ValueError(f"schema_mismatch_for_{topic} ({sname} != {expected_schema})")
                    if msg_enc and msg_enc != "cdr":
                        raise ValueError(f"unsupported_message_encoding ({msg_enc})")
                    if len(raw) < 4 or raw[0] != 0x00 or raw[1] != 0x01:
                        raise ValueError(f"unsupported_or_corrupted_cdr_header on {topic}")

                if topic == "/fmu/out/vehicle_odometry":
                    if len(raw) < 118:
                        raise ValueError(f"truncated_cdr_payload_vehicle_odometry ({len(raw)} < 118)")
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

                elif topic == "/fmu/out/vehicle_local_position_v1":
                    # Layout CDR exacto segun src/px4_msgs/msg/VehicleLocalPosition.msg:
                    # 4..20: timestamp, timestamp_sample (2x uint64)
                    # 20..24: xy_valid, z_valid, v_xy_valid, v_z_valid (4x bool)
                    # 24..36: x, y, z (3x float32)
                    # 36..44: delta_xy (2x float32), 44: xy_reset_counter, 45..48: pad, 48..52: delta_z, 52: z_reset_counter, 53..56: pad
                    # 56..72: vx, vy, vz, z_deriv (4x float32)
                    # 72..80: delta_vxy, 80: vxy_reset_counter, 81..84: pad, 84..88: delta_vz, 88: vz_reset_counter, 89..92: pad
                    # 92..104: ax, ay, az (3x float32)
                    # 104..108: heading (float32)
                    if len(raw) < 108:
                        raise ValueError(f"truncated_cdr_payload_vehicle_local_position ({len(raw)} < 108)")
                    ts_us, ts_sample_us = struct.unpack_from("<QQ", raw, 4)
                    xy_valid = bool(raw[20])
                    z_valid = bool(raw[21])
                    v_xy_valid = bool(raw[22])
                    v_z_valid = bool(raw[23])
                    x, y, z = struct.unpack_from("<fff", raw, 24)
                    vx, vy, vz = struct.unpack_from("<fff", raw, 56)
                    heading = float(struct.unpack_from("<f", raw, 104)[0])
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
                        "v_xy_valid": v_xy_valid,
                        "v_z_valid": v_z_valid,
                    })

                elif topic == "/fmu/out/vehicle_land_detected":
                    # Layout CDR exacto segun src/px4_msgs/msg/VehicleLandDetected.msg:
                    # 4..12: timestamp (uint64)
                    # 12..24: 12 bools (12=freefall, 13=ground_contact, 14=maybe_landed, 15=landed, ..., 23=at_rest)
                    if len(raw) < 24:
                        raise ValueError(f"truncated_cdr_payload_vehicle_land_detected ({len(raw)} < 24)")
                    ts_us = struct.unpack_from("<Q", raw, 4)[0]
                    freefall, ground_contact, maybe_landed, landed = struct.unpack_from("<BBBB", raw, 12)
                    at_rest = bool(raw[23])
                    telemetry["land_detected"].append({
                        "timestamp": int(log_time),
                        "timestamp_us": int(ts_us),
                        "freefall": bool(freefall),
                        "ground_contact": bool(ground_contact),
                        "maybe_landed": bool(maybe_landed),
                        "landed": bool(landed),
                        "at_rest": at_rest,
                    })

                elif topic == "/fmu/out/vehicle_status_v4":
                    # Layout CDR exacto segun src/px4_msgs/msg/VehicleStatus.msg:
                    # 4..28: timestamp, armed_time, takeoff_time (3x uint64)
                    # 28..31: arming_state, latest_arming_reason, latest_disarming_reason (3x uint8)
                    # 31..36: pad (5 bytes para alinear uint64 a data_pos=32)
                    # 36..44: nav_state_timestamp (uint64)
                    # 44: nav_state_user_intention (uint8)
                    # 45: nav_state (uint8)
                    # ... 62: failsafe (bool)
                    if len(raw) < 63:
                        raise ValueError(f"truncated_cdr_payload_vehicle_status ({len(raw)} < 63)")
                    ts_us = struct.unpack_from("<Q", raw, 4)[0]
                    arming_state = int(raw[28])
                    nav_state = int(raw[45])
                    failsafe = bool(raw[62])
                    telemetry["vehicle_status"].append({
                        "timestamp": int(log_time),
                        "timestamp_us": int(ts_us),
                        "arming_state": arming_state,
                        "nav_state": nav_state,
                        "failsafe": failsafe,
                    })

                elif topic == "/fmu/in/trajectory_setpoint":
                    if len(raw) < 68:
                        raise ValueError(f"truncated_cdr_payload_trajectory_setpoint ({len(raw)} < 68)")
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
                if p + 8 + clen > rec_end:
                    raise ValueError("truncated_mcap_chunk_records")
                if comp == "":
                    _walk_records(b, p + 8, p + 8 + clen)
                else:
                    compressed_chunk_found = True
                    return

            o = rec_end

    _walk_records(buf, 8, len(buf) - 8)
    if compressed_chunk_found:
        return None
    return telemetry


def _parse_with_rosbags(mcaps: List[Path]) -> Dict[str, List[Dict[str, Any]]]:
    """Lee una lista de archivos `.mcap` utilizando la biblioteca mantenida `rosbags`."""
    from rosbags.highlevel import AnyReader
    from rosbags.typesys import Stores, get_typestore

    typestore = get_typestore(getattr(Stores, "ROS2_JAZZY", Stores.LATEST))
    merged: Dict[str, List[Dict[str, Any]]] = {
        "odometry": [],
        "local_pos": [],
        "land_detected": [],
        "vehicle_status": [],
        "setpoints": [],
    }
    with AnyReader(mcaps, default_typestore=typestore) as reader:
        for connection, timestamp, rawdata in reader.messages():
            topic = connection.topic
            msg = reader.deserialize(rawdata, connection.msgtype)
            if topic == "/fmu/out/vehicle_odometry":
                merged["odometry"].append({
                    "timestamp": int(timestamp),
                    "timestamp_us": int(getattr(msg, "timestamp", 0)),
                    "timestamp_sample_us": int(getattr(msg, "timestamp_sample", getattr(msg, "timestamp", 0))),
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
                    "timestamp_sample_us": int(getattr(msg, "timestamp_sample", getattr(msg, "timestamp", 0))),
                    "x": float(msg.x),
                    "y": float(msg.y),
                    "z": float(msg.z),
                    "vx": float(getattr(msg, "vx", 0.0)),
                    "vy": float(getattr(msg, "vy", 0.0)),
                    "vz": float(getattr(msg, "vz", 0.0)),
                    "heading": float(getattr(msg, "heading", 0.0)),
                    "xy_valid": bool(getattr(msg, "xy_valid", True)),
                    "z_valid": bool(getattr(msg, "z_valid", True)),
                    "v_xy_valid": bool(getattr(msg, "v_xy_valid", True)),
                    "v_z_valid": bool(getattr(msg, "v_z_valid", True)),
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
                    "failsafe": bool(getattr(msg, "failsafe", False)),
                })
            elif topic == "/fmu/in/trajectory_setpoint":
                merged["setpoints"].append({
                    "timestamp": int(timestamp),
                    "timestamp_us": int(getattr(msg, "timestamp", 0)),
                    "position": [float(x) for x in msg.position],
                    "velocity": [float(x) for x in getattr(msg, "velocity", [float("nan")] * 3)],
                    "yaw": float(getattr(msg, "yaw", 0.0)),
                    "yawspeed": float(getattr(msg, "yawspeed", float("nan"))),
                })
    return merged


def load_episode_telemetry(bag_path: Path) -> Dict[str, List[Dict[str, Any]]]:
    """
    Carga la telemetria completa de un episodio desde un directorio `rosbag/` o archivo `.mcap`.
    1. Verifica siempre la integridad estructural de todos los archivos `.mcap` (magic + Footer).
    2. Prioriza `rosbags` cuando esta disponible en el entorno Python activo.
    3. Utiliza `_parse_mcap_native` como fallback verificado si `rosbags` no esta instalado.
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

    raw_buffers: Dict[Path, bytes] = {}
    for mp in mcaps:
        raw_buffers[mp] = verify_mcap_integrity(mp)

    # 1. Intentar con `rosbags` como lector primario mantenido
    try:
        import rosbags.highlevel  # noqa: F401
        has_rosbags = True
    except ImportError:
        has_rosbags = False

    if has_rosbags:
        try:
            return _parse_with_rosbags(mcaps)
        except Exception:
            # Si rosbags falla por falta de registro de tipos custom de px4_msgs en su typestore
            # por defecto, delegar al parser nativo CDR con validacion estricta de esquema.
            pass

    # 2. Fallback al lector nativo MCAP + CDR con validacion de esquema y longitud
    merged: Dict[str, List[Dict[str, Any]]] = {
        "odometry": [],
        "local_pos": [],
        "land_detected": [],
        "vehicle_status": [],
        "setpoints": [],
    }
    for mp in mcaps:
        res = _parse_mcap_native(mp, buf=raw_buffers[mp])
        if res is None:
            raise RuntimeError(
                f"El archivo {mp} contiene chunks comprimidos y `rosbags` no pudo deserializarlo."
            )
        for k in merged:
            merged[k].extend(res[k])
    return merged


# ============================================================================
# Geometria Analitica de Distancia a Obstaculos (con Subdivisión de Segmentos)
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
    """Calcula la distancia euclidea desde N puntos (N, 3) a una caja 3D alineada."""
    c = np.asarray(box_center, dtype=np.float64)
    half = 0.5 * np.asarray(box_size, dtype=np.float64)
    excess = np.maximum(0.0, np.abs(pts_local - c) - half)
    return np.linalg.norm(excess, axis=1)


def _subdivide_trajectory_segments(
    pos_enu: np.ndarray,
    max_step_m: float = COLLISION_SEGMENT_MAX_STEP_M,
) -> np.ndarray:
    """
    Subdivide linealmente todo segmento entre muestras consecutivas cuya longitud supere
    `max_step_m` (default: 0.02 m) para evitar que un obstaculo delgado quede oculto
    entre dos muestras discretas.
    """
    if len(pos_enu) <= 1 or max_step_m <= 0.0:
        return pos_enu

    diffs = np.diff(pos_enu, axis=0)
    seg_lens = np.linalg.norm(diffs, axis=1)
    if np.all(seg_lens <= max_step_m):
        return pos_enu

    chunks: List[np.ndarray] = []
    for i in range(len(diffs)):
        L = float(seg_lens[i])
        if L <= max_step_m:
            chunks.append(pos_enu[i:i + 1])
        else:
            n_sub = int(math.ceil(L / max_step_m))
            alphas = np.linspace(0.0, 1.0, n_sub, endpoint=False, dtype=np.float64)[:, None]
            seg_pts = pos_enu[i] + alphas * diffs[i]
            chunks.append(seg_pts)
    chunks.append(pos_enu[-1:])
    return np.vstack(chunks)


def compute_min_distance_to_landmark(
    pos_enu: np.ndarray,
    landmark: Dict[str, Any],
    max_step_m: float = COLLISION_SEGMENT_MAX_STEP_M,
) -> float:
    """
    Calcula la distancia minima euclidea (en metros) desde la trayectoria poligonal 3D
    subdividida a paso `<= max_step_m` hasta la superficie solida de un landmark
    ('gate'/'square_gate', 'circular_gate', o 'pole').
    """
    if len(pos_enu) == 0:
        return float("inf")

    pts_dense = _subdivide_trajectory_segments(pos_enu, max_step_m=max_step_m)
    lm_type = str(landmark.get("type", "gate")).lower()
    cx = float(landmark["x"])
    cy = float(landmark["y"])
    cz = float(landmark.get("z", 1.5))
    yaw = float(landmark.get("yaw", 0.0))

    if "pole" in lm_type or "cyl" in lm_type:
        # Poste cilindrico vertical en (cx, cy), z in [0, POLE_HEIGHT], radio POLE_RADIUS
        r_xy = np.hypot(pts_dense[:, 0] - cx, pts_dense[:, 1] - cy)
        d_rad = np.maximum(0.0, r_xy - POLE_RADIUS)
        d_z_top = np.maximum(0.0, pts_dense[:, 2] - POLE_HEIGHT)
        d_z_bot = np.maximum(0.0, 0.0 - pts_dense[:, 2])
        d_z = np.maximum(d_z_top, d_z_bot)
        dists = np.hypot(d_rad, d_z)
        return float(np.min(dists))

    pts_loc = _world_to_landmark_local(pts_dense, cx, cy, cz, yaw)

    if "circ" in lm_type or "torus" in lm_type:
        # Anillo circular poligonal de 16 segmentos: descontamos CIRCULAR_POLY_SAGITTA
        # del radio libre del toroide ideal como cota inferior conservadora.
        r_yz = np.hypot(pts_loc[:, 1], pts_loc[:, 2])
        d_tube_center = np.hypot(r_yz - CIRCULAR_MAJOR_RADIUS, pts_loc[:, 0])
        dists = np.maximum(0.0, d_tube_center - CIRCULAR_TUBE_RADIUS - CIRCULAR_POLY_SAGITTA)
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
    Retorna un diccionario con `passed` (bool), `reason` (Optional[str]), `event_index` (Optional[int])
    y metricas especificas de la maniobra.
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
        "event_index": None,
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
        best_idx = int(cross_candidates[0])
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
                best_idx = int(idx)

        assert best_cross is not None
        y_c, z_c, r_c = best_cross
        result["event_index"] = best_idx
        result["cross_y_m"] = round(y_c, 4)
        result["cross_z_m"] = round(z_c, 4)
        result["cross_radial_err_m"] = round(r_c, 4)

        if "circ" in lm_type or "torus" in lm_type:
            if r_c > TRAVERSE_CIRC_MAX_R:
                result["reason"] = (
                    f"traverse_{target_id}_outside_circular_aperture (r={r_c:.3f}m > {TRAVERSE_CIRC_MAX_R:.3f}m)"
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

        # Verificar que el dron pasa de antes del poste (x_loc <= 0) a despues (x_loc > 0)
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
        result["event_index"] = idx
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
        cross_indices = np.where((x_loc[:-1] <= 0.0) & (x_loc[1:] > 0.0) & ( np.abs(y_loc[:-1]) <= OVERFLY_MAX_LAT_M + 1.0 ))[0]
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

        result["event_index"] = idx
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
        result["event_index"] = int(main_seg[len(main_seg) // 2])

        angles = np.unwrap(np.arctan2(y_loc[main_seg], x_loc[main_seg]))
        # Progreso angular monotono maximo en sentido antihorario (de -pi a +pi/2 = +270 deg)
        running_min = np.minimum.accumulate(angles)
        ccw_progression_deg = math.degrees(float(np.max(angles - running_min)))
        max_span_deg = math.degrees(float(np.max(angles) - np.min(angles)))
        mean_r = float(np.mean(dist_xy[main_seg]))

        # Error de orientacion centripeta durante el arco orbital (excluyendo cola de salida)
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

        if mean_yaw_err_deg > ORBIT_MAX_MEAN_YAW_ERR_DEG:
            result["reason"] = (
                f"orbit_{target_id}_excessive_centripetal_yaw_error "
                f"(mean_yaw_err={mean_yaw_err_deg:.1f}deg > {ORBIT_MAX_MEAN_YAW_ERR_DEG:.1f}deg)"
            )
            return result

        result["passed"] = True
        return result

    else:
        result["reason"] = f"unsupported_primitive_type_{p_type}"
        return result


def _detect_timesync_and_clock_anomalies(
    ts_px4_s: np.ndarray,
    ts_bag_s: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Clasifica cada transicion temporal `i -> i+1` entre:
      1. `verified_timesync_mask`: recalibracion verificable del offset `Timesync` de `uxrce_dds_client`:
         - Patron A: muestra aislada `k` en tiempo crudo de boot (`< 1e6` s) rodeada por muestras en
           epoca UNIX (`> 1e8` s) con `0 < dt_bag <= 0.10` s.
         - Patron B: escalon de resincronizacion que corrige un retardo acumulado (`lag_before >= 0.15` s)
           entre el reloj de PX4 y el reloj UNIX del host, dejando `lag_after in [-0.02, +0.05]` s
           con `0 < dt_bag <= 0.10` s.
      2. `unexplained_backstep_mask`: retrocesos o estancamientos (`dt_px4 <= 0`) que NO corresponden
         al Patron A de `Timesync`.
    Retorna `(dt_reconstructed, verified_timesync_mask, unexplained_backstep_mask)`.
    """
    N = len(ts_px4_s)
    if N <= 1:
        empty_b = np.zeros(0, dtype=bool)
        return np.zeros(0, dtype=np.float64), empty_b, empty_b

    dt_px4 = np.diff(ts_px4_s)
    dt_bag = np.diff(ts_bag_s)
    M = len(dt_px4)

    verified_timesync_mask = np.zeros(M, dtype=bool)

    # Patron A: muestra aislada en tiempo de boot (< 1e6) entre muestras UNIX (> 1e8)
    for k in range(1, N - 1):
        if (
            ts_px4_s[k] < 1e6
            and ts_px4_s[k - 1] > 1e8
            and ts_px4_s[k + 1] > 1e8
            and 0.0 < dt_bag[k - 1] <= 0.100
            and 0.0 < dt_bag[k] <= 0.100
        ):
            verified_timesync_mask[k - 1] = True
            verified_timesync_mask[k] = True

    # Patron B: escalon de resincronizacion de offset que cierra un lag_before >= 0.15s a ~0s
    for i in range(M):
        if verified_timesync_mask[i]:
            continue
        if (
            ts_px4_s[i] > 1e8
            and ts_px4_s[i + 1] > 1e8
            and 0.0 < dt_bag[i] <= 0.100
            and abs(dt_px4[i] - dt_bag[i]) > 0.050
        ):
            lag_before = float(ts_bag_s[i] - ts_px4_s[i])
            lag_after = float(ts_bag_s[i + 1] - ts_px4_s[i + 1])
            if lag_before >= 0.150 and -0.020 <= lag_after <= 0.050:
                verified_timesync_mask[i] = True

    unexplained_backstep_mask = (dt_px4 <= 0.0) & (~verified_timesync_mask)

    # Reconstruir dt_sim:
    # - Los saltos verificados de Timesync se reemplazan por dt_bag.
    # - Los huecos hacia adelante NO explicados por Timesync (dt_px4 > 0) se conservan intactos
    #   para que nunca se oculte una congelacion de la fuente entregada en rafaga.
    # - Si existe un retroceso/estancamiento (dt_px4 <= 0), se sustituye por max(1e-6, dt_bag)
    #   para garantizar que la serie devuelta sea estrictamente monotona, pero queda registrado en
    #   `unexplained_backstep_mask` para que `validate_episode_data` rechace la grabacion.
    safe_dt_bag = np.maximum(1e-6, dt_bag)
    replace_mask = verified_timesync_mask | unexplained_backstep_mask
    dt_sim = np.where(replace_mask, safe_dt_bag, dt_px4)
    return dt_sim, verified_timesync_mask, unexplained_backstep_mask


def reconstruct_simulation_clock(
    ts_px4_s: np.ndarray,
    ts_bag_s: np.ndarray,
) -> Tuple[np.ndarray, int]:
    """
    Reconstruye una serie temporal estrictamente monotona `ts_sim_rel` a partir del reloj de
    PX4 (`ts_px4_s`) y el reloj de recepcion del bag (`ts_bag_s`), sin ocultar huecos hacia
    adelante no explicados por `Timesync`.

    Retorna `(ts_sim_rel, num_repaired_steps)`.
    """
    N = len(ts_px4_s)
    if N == 0:
        return np.zeros(0, dtype=np.float64), 0
    if N == 1:
        return np.zeros(1, dtype=np.float64), 0

    dt_sim, verified_mask, backstep_mask = _detect_timesync_and_clock_anomalies(ts_px4_s, ts_bag_s)
    num_repaired = int(np.sum(verified_mask | backstep_mask))
    ts_sim_rel = np.concatenate([[0.0], np.cumsum(dt_sim)])
    return ts_sim_rel, num_repaired


# ============================================================================
# Validacion Estructural y Semantica de Configuracion (`waypoints.yaml`)
# ============================================================================

def _is_finite_number(val: Any) -> bool:
    """Retorna True si y solo si `val` es un escalar numerico real finito (excluyendo bool)."""
    if isinstance(val, bool) or not isinstance(val, (int, float, np.number)):
        return False
    return bool(math.isfinite(float(val)))


def validate_waypoints_config(
    waypoints_cfg: Any,
    *,
    allow_legacy_hoops: bool = False,
) -> Dict[str, Any]:
    """
    Valida estructural y semanticamente el diccionario de configuracion `waypoints.yaml`.
    Exige para la campana actual:
      - Objeto raiz de tipo diccionario no vacio.
      - Lista no vacia de landmarks con IDs presentes, no vacios y unicos.
      - Tipos geometricos soportados (`gate`, `square_gate`, `circular_gate`, `pole`).
      - Coordenadas (`x`, `y`, `z`) y orientacion (`yaw`) finitas.
      - Lista no vacia de primitivas con tipos soportados (`traverse`, `slalom`, `overfly`, `orbit`).
      - Referencias `target` a landmarks existentes y compatibilidad entre primitiva y tipo de landmark.
      - `side` explicito y valido (`left` o `right`) en `slalom`.
      - Plataforma de aterrizaje (`landing`) con coordenadas finitas.
    """
    errors: List[str] = []
    legacy_normalized = False

    if not isinstance(waypoints_cfg, dict) or len(waypoints_cfg) == 0:
        return {
            "valid": False,
            "configuration_status": "invalid",
            "errors": ["invalid_configuration:waypoints_root_must_be_non_empty_dict"],
            "normalized_config": None,
            "legacy_hoops_normalized": False,
        }

    # Validar start_pose si esta presente
    if "start_pose" in waypoints_cfg and waypoints_cfg["start_pose"] is not None:
        sp = waypoints_cfg["start_pose"]
        if not isinstance(sp, dict):
            errors.append("invalid_configuration:invalid_start_pose_not_dict")
        else:
            for coord_k in ("x", "y", "z", "yaw"):
                if coord_k in sp and not _is_finite_number(sp[coord_k]):
                    errors.append(f"invalid_configuration:non_finite_start_pose ({coord_k})")

    # 1. Validar lista de landmarks
    raw_lms = None
    if "landmarks" in waypoints_cfg:
        raw_lms = waypoints_cfg["landmarks"]
    elif allow_legacy_hoops and "hoops" in waypoints_cfg:
        raw_lms = waypoints_cfg["hoops"]
        legacy_normalized = True

    if not isinstance(raw_lms, list) or len(raw_lms) == 0:
        errors.append("invalid_configuration:missing_or_empty_landmarks")
        raw_lms = []

    landmarks_list: List[Dict[str, Any]] = []
    landmarks_map: Dict[str, Dict[str, Any]] = {}

    for idx, lm in enumerate(raw_lms):
        if not isinstance(lm, dict):
            errors.append(f"invalid_configuration:landmark_entry_not_dict (index={idx})")
            continue

        raw_id = lm.get("id")
        if raw_id is None and legacy_normalized and "name" in lm:
            raw_id = lm.get("name")

        if not isinstance(raw_id, str) or not raw_id.strip():
            errors.append(f"invalid_configuration:missing_or_empty_landmark_id (index={idx})")
            lm_id = f"__invalid_id_{idx}__"
        else:
            lm_id = raw_id.strip()
            if lm_id in landmarks_map:
                errors.append(f"invalid_configuration:duplicate_landmark_id ({lm_id})")

        raw_type = lm.get("type")
        if raw_type is None and legacy_normalized:
            raw_type = "gate"

        if not isinstance(raw_type, str) or raw_type.strip().lower() not in SUPPORTED_LANDMARK_TYPES:
            errors.append(f"invalid_configuration:unsupported_landmark_type ({lm_id}:{raw_type})")
            lm_type = str(raw_type).strip().lower() if raw_type is not None else "unknown"
        else:
            lm_type = raw_type.strip().lower()

        pose_ok = True
        for field in ("x", "y", "z", "yaw"):
            if field not in lm or not _is_finite_number(lm[field]):
                errors.append(f"invalid_configuration:non_finite_landmark_pose ({lm_id}.{field})")
                pose_ok = False

        if pose_ok:
            norm_lm = {
                "id": lm_id,
                "type": lm_type,
                "x": float(lm["x"]),
                "y": float(lm["y"]),
                "z": float(lm["z"]),
                "yaw": float(lm["yaw"]),
            }
            landmarks_list.append(norm_lm)
            if lm_id not in landmarks_map and not lm_id.startswith("__invalid_id_"):
                landmarks_map[lm_id] = norm_lm
        else:
            if lm_id not in landmarks_map and not lm_id.startswith("__invalid_id_"):
                landmarks_map[lm_id] = {"id": lm_id, "type": lm_type, "x": 0.0, "y": 0.0, "z": 1.5, "yaw": 0.0}

    # 2. Validar lista de primitivas
    raw_prims = None
    if "primitives" in waypoints_cfg:
        raw_prims = waypoints_cfg["primitives"]
    elif legacy_normalized and landmarks_list:
        sorted_lms = sorted(landmarks_list, key=lambda item: float(item["x"]))
        raw_prims = [{"type": "traverse", "target": item["id"]} for item in sorted_lms]

    if not isinstance(raw_prims, list) or len(raw_prims) == 0:
        errors.append("invalid_configuration:missing_or_empty_primitives")
        raw_prims = []

    primitives_list: List[Dict[str, Any]] = []
    for idx, prim in enumerate(raw_prims):
        if not isinstance(prim, dict):
            errors.append(f"invalid_configuration:primitive_entry_not_dict (index={idx})")
            continue

        raw_ptype = prim.get("type")
        if not isinstance(raw_ptype, str) or raw_ptype.strip().lower() not in SUPPORTED_PRIMITIVE_TYPES:
            errors.append(f"invalid_configuration:unsupported_primitive_type (index={idx}:{raw_ptype})")
            p_type = str(raw_ptype).strip().lower() if raw_ptype is not None else "unknown"
        else:
            p_type = raw_ptype.strip().lower()

        raw_target = prim.get("target")
        if not isinstance(raw_target, str) or not raw_target.strip():
            errors.append(f"invalid_configuration:missing_primitive_target (index={idx})")
            target_id = ""
        else:
            target_id = raw_target.strip()
            if target_id not in landmarks_map:
                errors.append(f"invalid_configuration:unknown_primitive_target ({target_id})")
            else:
                target_lm_type = landmarks_map[target_id]["type"]
                if p_type == "traverse" and target_lm_type not in GATE_LANDMARK_TYPES:
                    errors.append(
                        f"invalid_configuration:incompatible_primitive_landmark ({p_type} requires gate, got {target_lm_type} on {target_id})"
                    )
                elif p_type in {"slalom", "overfly", "orbit"} and target_lm_type not in POLE_LANDMARK_TYPES:
                    errors.append(
                        f"invalid_configuration:incompatible_primitive_landmark ({p_type} requires pole, got {target_lm_type} on {target_id})"
                    )

        norm_prim: Dict[str, Any] = {"type": p_type, "target": target_id}
        if p_type == "slalom":
            raw_side = prim.get("side")
            if not isinstance(raw_side, str) or raw_side.strip().lower() not in SUPPORTED_SLALOM_SIDES:
                errors.append(f"invalid_configuration:invalid_or_missing_slalom_side ({target_id}:{raw_side})")
            else:
                norm_prim["side"] = raw_side.strip().lower()

        primitives_list.append(norm_prim)

    # 3. Validar plataforma de aterrizaje (`landing`)
    raw_landing = waypoints_cfg.get("landing")
    norm_landing: Optional[Dict[str, float]] = None
    if not isinstance(raw_landing, dict):
        errors.append("invalid_configuration:missing_or_invalid_landing")
    else:
        landing_ok = True
        for req_coord in ("x", "y"):
            if req_coord not in raw_landing or not _is_finite_number(raw_landing[req_coord]):
                errors.append(f"invalid_configuration:non_finite_landing_coordinate ({req_coord})")
                landing_ok = False
        for opt_coord in ("z", "yaw"):
            if opt_coord in raw_landing and not _is_finite_number(raw_landing[opt_coord]):
                errors.append(f"invalid_configuration:non_finite_landing_coordinate ({opt_coord})")
                landing_ok = False
        if landing_ok:
            norm_landing = {
                "x": float(raw_landing["x"]),
                "y": float(raw_landing["y"]),
                "z": float(raw_landing.get("z", 0.05)),
            }

    is_valid = len(errors) == 0
    normalized_cfg = (
        {
            "landmarks": landmarks_list,
            "primitives": primitives_list,
            "landing": norm_landing,
            "legacy_hoops_normalized": legacy_normalized,
        }
        if is_valid
        else None
    )

    return {
        "valid": is_valid,
        "configuration_status": "valid" if is_valid else "invalid",
        "errors": errors,
        "normalized_config": normalized_cfg,
        "legacy_hoops_normalized": legacy_normalized,
    }


def _infer_pilot_exit_code_from_legacy_status(
    raw_status: Any,
) -> Tuple[Optional[int], str, List[str]]:
    """
    Tabla explicita de conversion de estados historicos de `data_collector.py` cuando
    `pilot_exit_code` no fue registrado directamente. Nunca inventa exito por defecto.
    Retorna `(pilot_exit_code, pilot_status_source, extra_rejection_reasons)`.
    """
    if not isinstance(raw_status, str) or not raw_status.strip():
        return None, "unknown", []

    st = raw_status.strip()
    if st == "success":
        return 0, "inferred_from_legacy_status", []
    if st == "timeout":
        return None, "inferred_from_legacy_status", ["pilot_timeout_status"]
    if st.startswith("pilot_exit_"):
        suffix = st[len("pilot_exit_"):]
        try:
            code = int(suffix)
            return code, "inferred_from_legacy_status", []
        except ValueError:
            return None, "unknown", [f"unrecognized_pilot_status ({st})"]
    return None, "unknown", []


def _atomic_write_json(target_path: Path, data: Dict[str, Any]) -> None:
    """Escribe un archivo JSON de forma atomica mediante archivo temporal en el mismo directorio y os.replace."""
    target_path = Path(target_path)
    parent_dir = target_path.parent
    fd, tmp_name = tempfile.mkstemp(
        prefix=f".{target_path.name}.",
        suffix=".tmp",
        dir=str(parent_dir),
        text=True,
    )
    tmp_path = Path(tmp_name)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2)
            f.write("\n")
            f.flush()
            os.fsync(f.fileno())
        os.replace(str(tmp_path), str(target_path))
    except Exception:
        if tmp_path.exists():
            try:
                tmp_path.unlink()
            except OSError:
                pass
        raise


def _find_episode_mcap_files(bag_path: Path) -> List[Path]:
    """Localiza todos los archivos `.mcap` asociados a un directorio de episodio o ruta directa."""
    bag_p = Path(bag_path)
    if bag_p.is_dir():
        mcaps = sorted(bag_p.glob("*.mcap"))
        if not mcaps and (bag_p / "rosbag").is_dir():
            mcaps = sorted((bag_p / "rosbag").glob("*.mcap"))
        return mcaps
    elif bag_p.is_file() and bag_p.suffix == ".mcap":
        return [bag_p]
    return []


def discover_episode_dirs(dataset_dir: Path) -> List[Path]:
    """
    Descubre todas las carpetas `task_*/episode_*` dentro de `dataset_dir`
    sin omitir aquellas que carezcan de grabacion `.mcap`.
    """
    ds_dir = Path(dataset_dir).resolve()
    if not ds_dir.is_dir():
        return []
    return sorted(p for p in ds_dir.glob("task_*/episode_*") if p.is_dir())


# ============================================================================
# Funcion Principal de Validacion de un Episodio
# ============================================================================

def validate_episode_data(
    telemetry: Dict[str, List[Dict[str, Any]]],
    waypoints_cfg: Dict[str, Any],
    pilot_exit_code: Optional[int] = None,
    *,
    pilot_status_source: str = "explicit_argument",
    allow_legacy_hoops: bool = False,
) -> Dict[str, Any]:
    """
    Ejecuta la validacion de configuracion, grabacion y geometria de mision sobre datos en memoria.
    Si falta evidencia (ej. `pilot_exit_code is None`) o la configuracion es invalida, rechaza el episodio.
    """
    rejection_reasons: List[str] = []

    if pilot_exit_code is None:
        rejection_reasons.append("unknown_pilot_exit_code")
    elif isinstance(pilot_exit_code, bool) or not isinstance(pilot_exit_code, (int, np.integer)):
        rejection_reasons.append(f"invalid_pilot_exit_code ({pilot_exit_code})")
    elif int(pilot_exit_code) != 0:
        rejection_reasons.append(f"pilot_non_zero_exit_code ({pilot_exit_code})")

    # 0. Validar configuracion `waypoints.yaml` antes de evaluar el vuelo
    cfg_check = validate_waypoints_config(waypoints_cfg, allow_legacy_hoops=allow_legacy_hoops)
    if not cfg_check["valid"]:
        rejection_reasons.extend(cfg_check["errors"])
        return {
            "pilot_exit_code": pilot_exit_code,
            "pilot_status_source": pilot_status_source,
            "configuration_status": "invalid",
            "recording_status": "invalid",
            "mission_status": "invalid",
            "validation_status": "rejected",
            "rejection_reasons": rejection_reasons,
            "primitive_results": [],
            "metrics": {},
        }

    norm_cfg = cfg_check["normalized_config"]
    assert norm_cfg is not None
    configuration_status = "valid"

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
            "pilot_status_source": pilot_status_source,
            "configuration_status": configuration_status,
            "recording_status": "invalid",
            "mission_status": "invalid",
            "validation_status": "rejected",
            "rejection_reasons": rejection_reasons,
            "primitive_results": [],
            "metrics": {"num_odom_messages": len(odom_list)},
        }

    # 1. Extraer series temporales y validar marcos (NED), flags de validez, NaNs y cuaterniones
    N = len(odom_list)
    pos_enu = np.zeros((N, 3), dtype=np.float64)
    vel_enu = np.zeros((N, 3), dtype=np.float64)
    yaw_enu = np.zeros(N, dtype=np.float64)
    ts_px4_raw_s = np.zeros(N, dtype=np.float64)
    ts_bag_raw_s = np.zeros(N, dtype=np.float64)
    reset_counters = np.zeros(N, dtype=np.int64)
    nan_count = 0
    invalid_frame_count = 0

    for i, item in enumerate(odom_list):
        # Preferir timestamp_sample_us (instante real de integracion EKF2) sobre timestamp_us
        t_us = item.get("timestamp_sample_us") or item.get("timestamp_us", 0)
        t_bag_ns = item.get("timestamp", 0)
        ts_bag_raw_s[i] = (float(t_bag_ns) * 1e-9) if t_bag_ns > 0 else (float(i) * 0.01)
        ts_px4_raw_s[i] = (float(t_us) * 1e-6) if t_us > 0 else ts_bag_raw_s[i]

        if not using_local_pos:
            pose_frame = int(item.get("pose_frame", 1))
            vel_frame = int(item.get("velocity_frame", 1))
            if pose_frame != 1 or vel_frame != 1:
                invalid_frame_count += 1

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
            if (
                not bool(item.get("xy_valid", True))
                or not bool(item.get("z_valid", True))
                or not bool(item.get("v_xy_valid", True))
                or not bool(item.get("v_z_valid", True))
            ):
                invalid_frame_count += 1

            p_ned = [item["x"], item["y"], item["z"]]
            v_ned = [item.get("vx", 0.0), item.get("vy", 0.0), item.get("vz", 0.0)]
            h_ned = item.get("heading", 0.0)
            if (
                not np.all(np.isfinite(p_ned))
                or not np.all(np.isfinite(v_ned))
                or not math.isfinite(h_ned)
            ):
                nan_count += 1
                continue
            pos_enu[i] = ned_to_gazebo_enu(p_ned)
            vel_enu[i] = ned_to_gazebo_enu(v_ned)
            yaw_enu[i] = yaw_ned_to_gazebo_enu(h_ned)

    if nan_count > 0:
        rejection_reasons.append(f"corrupted_nan_or_invalid_quaternion_samples ({nan_count})")
    if invalid_frame_count > 0:
        rejection_reasons.append(f"invalid_or_non_ned_coordinate_frame ({invalid_frame_count})")

    # 2. Reconstruccion y verificacion estricta de continuidad temporal del reloj
    dt_sim, verified_ts_mask, unexplained_backstep_mask = _detect_timesync_and_clock_anomalies(
        ts_px4_raw_s, ts_bag_raw_s
    )
    ts_sim_s = np.concatenate([[0.0], np.cumsum(dt_sim)])
    timesync_offset_jumps = int(np.sum(verified_ts_mask))
    unexplained_backsteps = int(np.sum(unexplained_backstep_mask))

    pos_dt = dt_sim[dt_sim > 0]
    duration_s = float(ts_sim_s[-1] - ts_sim_s[0]) if N > 1 else 0.0
    dt_median_ms = float(np.median(pos_dt) * 1000.0) if len(pos_dt) > 0 else 0.0
    dt_p95_ms = float(np.percentile(pos_dt, 95) * 1000.0) if len(pos_dt) > 0 else 0.0
    max_dt_gap_s = float(np.max(dt_sim)) if len(dt_sim) > 0 else 0.0
    min_dt_s = float(np.min(dt_sim)) if len(dt_sim) > 0 else 0.0

    seg_lengths = np.linalg.norm(np.diff(pos_enu, axis=0), axis=1)
    max_step_dist_m = float(np.max(seg_lengths)) if len(seg_lengths) > 0 else 0.0

    if duration_s < 3.0:
        rejection_reasons.append(f"flight_duration_too_short ({duration_s:.2f}s < 3.0s)")
    if unexplained_backsteps > 0 or min_dt_s <= 0.0:
        rejection_reasons.append(
            f"non_monotonic_or_stalled_simulation_clock (unexplained_backsteps={unexplained_backsteps}, min_dt={min_dt_s:.4f}s)"
        )
    if max_dt_gap_s > MAX_ALLOWED_DT_GAP_S:
        rejection_reasons.append(
            f"telemetry_dropout_gap (max_dt_gap={max_dt_gap_s:.3f}s > {MAX_ALLOWED_DT_GAP_S:.2f}s)"
        )
    if max_step_dist_m > MAX_ALLOWED_STEP_DIST_M:
        rejection_reasons.append(
            f"excessive_spatial_step_between_samples (max_step={max_step_dist_m:.3f}m > {MAX_ALLOWED_STEP_DIST_M:.2f}m)"
        )

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
                "invalid_or_non_ned_",
                "flight_duration_",
                "non_monotonic_",
                "telemetry_dropout_",
                "excessive_spatial_step_",
                "ekf2_estimator_reset_",
            ))
            for r in rejection_reasons
        )
        else "valid"
    )

    # 4. Verificar cada primitiva y exigir orden temporal estricto de ejecucion
    landmarks_list = norm_cfg["landmarks"]
    landmarks_map = {str(lm["id"]): lm for lm in landmarks_list}
    primitives_sorted = norm_cfg["primitives"]

    primitive_results: List[Dict[str, Any]] = []
    for prim in primitives_sorted:
        tid = str(prim["target"])
        p_res = _verify_single_primitive(pos_enu, yaw_enu, prim, landmarks_map[tid])
        primitive_results.append(p_res)
        if not p_res["passed"] and p_res["reason"]:
            rejection_reasons.append(str(p_res["reason"]))

    # Verificar que los eventos de las primitivas ocurran en orden temporal estrictamente creciente
    for k in range(1, len(primitive_results)):
        prev_res = primitive_results[k - 1]
        curr_res = primitive_results[k]
        idx_prev = prev_res.get("event_index")
        idx_curr = curr_res.get("event_index")
        if idx_prev is not None and idx_curr is not None and int(idx_curr) <= int(idx_prev):
            rejection_reasons.append(
                f"primitive_order_violation ({prev_res['target']}@{idx_prev} >= {curr_res['target']}@{idx_curr})"
            )

    # 5. Verificar colision global (con subdivision de segmentos) contra CUALQUIER landmark del mundo
    min_global_clearance = float("inf")
    for lm in landmarks_list:
        d_lm = compute_min_distance_to_landmark(pos_enu, lm)
        min_global_clearance = min(min_global_clearance, d_lm)
        if d_lm < UAV_COLLISION_RADIUS_XY:
            reason_col = f"collision_with_{lm['id']} (min_clearance={d_lm:.3f}m < {UAV_COLLISION_RADIUS_XY:.2f}m)"
            if not any(r.startswith(f"collision_with_{lm['id']}") for r in rejection_reasons):
                rejection_reasons.append(reason_col)

    # 6. Verificar aterrizaje final sincronizado en la ventana terminal del episodio
    landing_cfg = norm_cfg["landing"]
    tail_k = min(25, N)
    final_window_pos = pos_enu[-tail_k:]
    final_window_vel = vel_enu[-tail_k:]
    final_pos = pos_enu[-1]
    final_window_max_z = float(np.max(final_window_pos[:, 2]))
    final_window_max_speed = float(np.max(np.linalg.norm(final_window_vel, axis=1)))
    final_speed = float(np.linalg.norm(vel_enu[-1]))

    if not land_list:
        rejection_reasons.append("missing_land_detected_messages")
        landed_flag_confirmed = False
    else:
        last_ld_ts_s = float(land_list[-1].get("timestamp", 0)) * 1e-9
        ld_age_s = abs(float(ts_bag_raw_s[-1]) - last_ld_ts_s) if last_ld_ts_s > 0 else 0.0
        if ld_age_s > LANDING_MAX_DETECTOR_AGE_S:
            rejection_reasons.append(
                f"stale_land_detector_message (age={ld_age_s:.2f}s > {LANDING_MAX_DETECTOR_AGE_S:.2f}s)"
            )
        # Exigir explicitamente `landed == True` (etapa 3 de VehicleLandDetected, nunca solo `ground_contact`)
        # con persistencia en las muestras finales del detector
        if len(land_list) >= 2:
            landed_flag_confirmed = bool(land_list[-1].get("landed", False)) and bool(
                land_list[-2].get("landed", False)
            )
        else:
            landed_flag_confirmed = bool(land_list[-1].get("landed", False))

        if not landed_flag_confirmed:
            rejection_reasons.append("px4_land_detector_not_confirmed")

    lx, ly = float(landing_cfg["x"]), float(landing_cfg["y"])
    landing_xy_err = float(
        np.max(np.hypot(final_window_pos[:, 0] - lx, final_window_pos[:, 1] - ly))
    )
    if landing_xy_err > LANDING_MAX_XY_ERR_M:
        rejection_reasons.append(
            f"landing_off_pad (xy_err={landing_xy_err:.3f}m > {LANDING_MAX_XY_ERR_M:.2f}m)"
        )

    if final_window_max_z > LANDING_MAX_FINAL_Z_M:
        rejection_reasons.append(
            f"landing_incomplete_altitude (final_window_max_z={final_window_max_z:.3f}m > {LANDING_MAX_FINAL_Z_M:.2f}m)"
        )

    if final_window_max_speed > LANDING_MAX_FINAL_SPEED:
        rejection_reasons.append(
            f"landing_not_at_rest (final_window_max_speed={final_window_max_speed:.3f}m/s > {LANDING_MAX_FINAL_SPEED:.2f}m/s)"
        )

    # 7. Calcular metricas continuas de calidad del piloto experto
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
    validation_status = (
        "accepted"
        if (
            configuration_status == "valid"
            and recording_status == "valid"
            and mission_status == "valid"
            and pilot_exit_code == 0
            and len(rejection_reasons) == 0
        )
        else "rejected"
    )

    metrics = {
        "num_odom_messages": int(N),
        "duration_s": round(duration_s, 3),
        "dt_median_ms": round(dt_median_ms, 3),
        "dt_p95_ms": round(dt_p95_ms, 3),
        "max_dt_gap_s": round(max_dt_gap_s, 4),
        "max_step_dist_m": round(max_step_dist_m, 4),
        "timesync_offset_jumps": int(timesync_offset_jumps),
        "unexplained_clock_backsteps": int(unexplained_backsteps),
        "in_flight_estimator_resets": int(in_flight_resets),
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
        "pilot_status_source": pilot_status_source,
        "configuration_status": configuration_status,
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
    - No inventa exito cuando falta `episode_meta.json` o `pilot_exit_code`.
    - Valida `waypoints.yaml` antes de evaluar la mision.
    - Reporta `missing_bag` cuando el episodio carece de archivo `.mcap`.
    - Si `update_meta=True`, preserva la procedencia original del piloto y escribe
      `episode_meta.json` de manera atomica sin sobrescribir JSONs corruptos.
    """
    ep_dir = Path(ep_dir).resolve()
    wp_file = ep_dir / "waypoints.yaml"
    if not wp_file.is_file():
        parent_wp = ep_dir.parent / "waypoints.yaml"
        wp_file = parent_wp if parent_wp.is_file() else wp_file

    meta_file = ep_dir / "episode_meta.json"
    existing_meta: Dict[str, Any] = {}
    meta_corrupted = False
    meta_corruption_reason: Optional[str] = None

    if meta_file.is_file():
        try:
            with open(meta_file, "r", encoding="utf-8") as f:
                loaded_meta = json.load(f)
            if not isinstance(loaded_meta, dict):
                meta_corrupted = True
                meta_corruption_reason = "corrupted_episode_meta_json (root_not_dict)"
            else:
                existing_meta = loaded_meta
        except Exception as e:
            meta_corrupted = True
            meta_corruption_reason = f"corrupted_episode_meta_json ({type(e).__name__})"

    # Resolver pilot_exit_code y procedencia sin defaults que fabriquen exito
    extra_meta_reasons: List[str] = []
    if meta_corrupted and meta_corruption_reason is not None:
        extra_meta_reasons.append(meta_corruption_reason)

    if pilot_exit_code is not None:
        resolved_exit_code: Optional[int] = int(pilot_exit_code)
        pilot_status_source = "explicit_argument"
    elif not meta_corrupted and meta_file.is_file():
        if "original_pilot_exit_code" in existing_meta and existing_meta["original_pilot_exit_code"] is not None:
            resolved_exit_code = int(existing_meta["original_pilot_exit_code"])
            pilot_status_source = str(existing_meta.get("pilot_status_source", "recorded_exit_code"))
        elif "pilot_exit_code" in existing_meta and existing_meta["pilot_exit_code"] is not None:
            resolved_exit_code = int(existing_meta["pilot_exit_code"])
            pilot_status_source = str(existing_meta.get("pilot_status_source", "recorded_exit_code"))
        else:
            raw_status = existing_meta.get("original_status", existing_meta.get("status"))
            resolved_exit_code, pilot_status_source, status_reasons = _infer_pilot_exit_code_from_legacy_status(
                raw_status
            )
            extra_meta_reasons.extend(status_reasons)
    else:
        resolved_exit_code = None
        pilot_status_source = "unknown"

    # 1. Validar presencia y estructura de waypoints.yaml
    wp_cfg: Any = None
    cfg_errors: List[str] = []
    if not wp_file.is_file():
        cfg_errors.append("invalid_configuration:missing_waypoints_yaml")
    else:
        try:
            with open(wp_file, "r", encoding="utf-8") as f:
                wp_cfg = yaml.safe_load(f)
        except Exception as e:
            cfg_errors.append(f"invalid_configuration:malformed_waypoints_yaml ({type(e).__name__})")

    if not cfg_errors:
        cfg_check = validate_waypoints_config(wp_cfg)
        if not cfg_check["valid"]:
            cfg_errors.extend(cfg_check["errors"])

    # 2. Verificar presencia de grabacion MCAP (`missing_bag`)
    mcap_files = _find_episode_mcap_files(ep_dir)
    has_bag = len(mcap_files) > 0

    if cfg_errors or not has_bag:
        reasons: List[str] = list(extra_meta_reasons)
        if resolved_exit_code is None:
            if "unknown_pilot_exit_code" not in reasons:
                reasons.append("unknown_pilot_exit_code")
        elif int(resolved_exit_code) != 0:
            reasons.append(f"pilot_non_zero_exit_code ({resolved_exit_code})")

        reasons.extend(cfg_errors)
        if not has_bag:
            reasons.append("missing_bag")

        report: Dict[str, Any] = {
            "pilot_exit_code": resolved_exit_code,
            "pilot_status_source": pilot_status_source,
            "configuration_status": "invalid" if cfg_errors else "valid",
            "recording_status": "invalid" if (not has_bag or meta_corrupted) else "not_evaluated",
            "mission_status": "invalid",
            "validation_status": "rejected",
            "rejection_reasons": reasons,
            "primitive_results": [],
            "metrics": {},
        }
    else:
        try:
            telemetry = load_episode_telemetry(ep_dir)
            report = validate_episode_data(
                telemetry,
                wp_cfg,
                pilot_exit_code=resolved_exit_code,
                pilot_status_source=pilot_status_source,
            )
            if extra_meta_reasons:
                merged_reasons = list(extra_meta_reasons) + [
                    r for r in report["rejection_reasons"] if r not in extra_meta_reasons
                ]
                report["rejection_reasons"] = merged_reasons
                report["validation_status"] = "rejected"
                if meta_corrupted:
                    report["recording_status"] = "invalid"
        except Exception as e:
            reasons = list(extra_meta_reasons)
            if resolved_exit_code is None and "unknown_pilot_exit_code" not in reasons:
                reasons.append("unknown_pilot_exit_code")
            elif resolved_exit_code is not None and int(resolved_exit_code) != 0:
                reasons.append(f"pilot_non_zero_exit_code ({resolved_exit_code})")
            reasons.append(f"corrupted_bag_or_read_error ({type(e).__name__}: {e})")
            report = {
                "pilot_exit_code": resolved_exit_code,
                "pilot_status_source": pilot_status_source,
                "configuration_status": "valid",
                "recording_status": "invalid",
                "mission_status": "invalid",
                "validation_status": "rejected",
                "rejection_reasons": reasons,
                "primitive_results": [],
                "metrics": {},
            }

    # 3. Actualizar episode_meta.json de forma atomica preservando procedencia (nunca sobrescribir JSON corrupto)
    if update_meta and not meta_corrupted and meta_file.parent.is_dir():
        if "status" in existing_meta and "original_status" not in existing_meta:
            existing_meta["original_status"] = existing_meta["status"]
        if (
            "pilot_exit_code" in existing_meta
            and existing_meta["pilot_exit_code"] is not None
            and "original_pilot_exit_code" not in existing_meta
        ):
            existing_meta["original_pilot_exit_code"] = existing_meta["pilot_exit_code"]
        if (
            "rejection_reasons" in existing_meta
            and existing_meta["rejection_reasons"]
            and "original_rejection_reasons" not in existing_meta
        ):
            existing_meta["original_rejection_reasons"] = list(existing_meta["rejection_reasons"])

        if existing_meta.get("pilot_exit_code") is None:
            existing_meta["pilot_exit_code"] = report["pilot_exit_code"]

        existing_meta["pilot_status_source"] = report["pilot_status_source"]
        existing_meta["configuration_status"] = report["configuration_status"]
        existing_meta["recording_status"] = report["recording_status"]
        existing_meta["mission_status"] = report["mission_status"]
        existing_meta["validation_status"] = report["validation_status"]
        existing_meta["rejection_reasons"] = report["rejection_reasons"]
        existing_meta["validation_metrics"] = report["metrics"]
        existing_meta["primitive_validation"] = report["primitive_results"]
        existing_meta["validator_version"] = VALIDATOR_VERSION
        existing_meta["validation_parameters"] = get_effective_validation_parameters()
        existing_meta["status"] = (
            "success" if report["validation_status"] == "accepted" else "rejected"
        )
        _atomic_write_json(meta_file, existing_meta)

    return report


# ============================================================================
# CLI para Auditar un Episodio o Todo un Directorio de Dataset
# ============================================================================

def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser(
        description="Validador fisico, geometrico y de grabacion independiente para episodios UAV."
    )
    target_group = parser.add_mutually_exclusive_group(required=True)
    target_group.add_argument(
        "--episode-dir",
        type=Path,
        default=None,
        help="Ruta a un directorio de episodio individual (ej. datasets/task_0001/episode_00).",
    )
    target_group.add_argument(
        "--dataset-dir",
        type=Path,
        default=None,
        help="Ruta al directorio raiz del dataset (ej. datasets/) para validar todos los episodios.",
    )
    parser.add_argument(
        "--update-meta",
        action="store_true",
        help="Actualiza `episode_meta.json` con los campos de validacion fisica y metricas.",
    )
    parser.add_argument(
        "--json",
        action="store_true",
        help="Imprime el reporte completo en formato JSON.",
    )
    parser.add_argument(
        "--strict",
        action="store_true",
        help="Retorna codigo de salida 1 si al menos un episodio del dataset es rechazado.",
    )
    args = parser.parse_args(argv)

    if args.episode_dir is not None:
        ep_path = Path(args.episode_dir)
        if not ep_path.exists() or not ep_path.is_dir():
            print(f"Error: el directorio de episodio no existe: {args.episode_dir}", file=sys.stderr)
            return 2

        rep = validate_episode_directory(ep_path, update_meta=args.update_meta)
        if args.json:
            print(json.dumps(rep, indent=2))
        else:
            status_tag = "ACCEPTED" if rep["validation_status"] == "accepted" else "REJECTED"
            print(f"[{status_tag}] {args.episode_dir}")
            print(
                f"  config={rep['configuration_status']} | recording={rep['recording_status']} | "
                f"mission={rep['mission_status']} | "
                f"min_clear={rep['metrics'].get('min_obstacle_clearance_m', -1.0):.3f}m | "
                f"land_err={rep['metrics'].get('landing_xy_error_m', -1.0):.3f}m"
            )
            if rep["rejection_reasons"]:
                print(f"  reasons: {', '.join(rep['rejection_reasons'])}")
        return 0 if rep["validation_status"] == "accepted" else 1

    # Modo --dataset-dir
    ds_path = Path(args.dataset_dir)
    if not ds_path.exists() or not ds_path.is_dir():
        print(f"Error: el directorio de dataset no existe: {args.dataset_dir}", file=sys.stderr)
        return 2

    ds_dir = ds_path.resolve()
    ep_dirs = discover_episode_dirs(ds_dir)
    if len(ep_dirs) == 0:
        print(
            f"Error: dataset vacio (no se encontraron carpetas task_*/episode_* en {ds_dir})",
            file=sys.stderr,
        )
        return 2

    summary_list = []
    accepted_count = 0
    for ep in ep_dirs:
        rep = validate_episode_directory(ep, update_meta=args.update_meta)
        label = f"{ep.parent.name}/{ep.name}"
        if rep["validation_status"] == "accepted":
            accepted_count += 1
        summary_list.append({"episode": label, **rep})

    if args.json:
        print(json.dumps({
            "total_episodes": len(summary_list),
            "accepted": accepted_count,
            "rejected": len(summary_list) - accepted_count,
            "episodes": summary_list,
        }, indent=2))
    else:
        print("=" * 88)
        print(f" VALIDACIÓN FÍSICA INDEPENDIENTE DE EPISODIOS — {ds_dir}")
        print("=" * 88)
        for item in summary_list:
            st = "ACCEPTED" if item["validation_status"] == "accepted" else "REJECTED"
            m = item.get("metrics", {})
            prims_str = ", ".join(
                f"{p['primitive']}({'OK' if p['passed'] else 'FAIL'})"
                for p in item.get("primitive_results", [])
            )
            print(
                f"  [{st:8s}] {item['episode']:22s} | "
                f"T={m.get('duration_s', 0):5.1f}s | "
                f"dt={m.get('dt_median_ms', 0):4.1f}ms | "
                f"min_clear={m.get('min_obstacle_clearance_m', -1):4.2f}m | "
                f"land_err={m.get('landing_xy_error_m', -1):4.2f}m | "
                f"{prims_str}"
            )
            if item["rejection_reasons"]:
                print(f"             -> MOTIVOS: {'; '.join(item['rejection_reasons'])}")
        print("-" * 88)
        total = len(summary_list)
        pct = (100.0 * accepted_count / total) if total > 0 else 0.0
        print(f" Total: {total} episodios | Aceptados: {accepted_count} ({pct:.1f}%) | Rechazados: {total - accepted_count}")
        print("=" * 88)

    if args.strict and accepted_count < len(summary_list):
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())

