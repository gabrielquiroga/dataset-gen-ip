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
import zlib
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
from uav_ip.validation_geometry import (
    CIRCULAR_GATE_CLEAR_RADIUS,
    CIRCULAR_MAJOR_RADIUS,
    CIRCULAR_NUM_SEGMENTS,
    CIRCULAR_POLY_SAGITTA,
    CIRCULAR_TUBE_RADIUS,
    COLLISION_SEGMENT_MAX_STEP_M,
    DISCRETIZATION_MARGIN_M,
    GATE_BAR_THICKNESS,
    GATE_HALF_HEIGHT,
    GATE_HALF_WIDTH,
    LANDING_MAX_XY_ERR_M,
    LANDING_PAD_HALF_SIZE,
    ORBIT_MAX_MEAN_YAW_ERR_DEG,
    ORBIT_MAX_RADIUS_M,
    ORBIT_MIN_SWEEP_DEG,
    OVERFLY_MAX_LAT_M,
    OVERFLY_MIN_ALT_M,
    PILOT_BUFFER_M,
    PILOT_TRAVERSE_CIRC_MAX_R,
    PILOT_TRAVERSE_SQUARE_MAX_Y,
    PILOT_TRAVERSE_SQUARE_MAX_Z,
    POLE_HEIGHT,
    POLE_RADIUS,
    PROTECTED_VOLUME_MODEL,
    PROTECTED_VOLUME_REFERENCE,
    SLALOM_MAX_LATERAL,
    SLALOM_MIN_LATERAL,
    SQUARE_GATE_CLEAR_HALF_Y,
    SQUARE_GATE_CLEAR_HALF_Z,
    TRAVERSE_CIRC_MAX_R,
    TRAVERSE_MAX_ABS_Y,
    TRAVERSE_MAX_ABS_Z,
    UAV_COLLISION_RADIUS_XY,
    UAV_PHYSICAL_RADIUS_XY,
    UAV_SAFE_POLE_CENTER_DIST,
    UAV_SAFETY_MARGIN_XY,
    X500_FULL_ROTOR_TIP_RADIUS_M,
    X500_SDF_MEASUREMENT,
    compute_landmark_clearance,
    derive_validation_tolerances,
    points_to_aabb_distance as _points_to_aabb_distance,
    subdivide_trajectory_segments as _subdivide_trajectory_segments,
    world_to_landmark_local as _world_to_landmark_local,
)

# ============================================================================
# Constantes Geometricas y Umbrales de Validacion (Sincronizados con Documentacion)
# ============================================================================

VALIDATOR_VERSION: str = "1.2.0"

SUPPORTED_LANDMARK_TYPES = frozenset({"gate", "square_gate", "circular_gate", "pole"})
GATE_LANDMARK_TYPES = frozenset({"gate", "square_gate", "circular_gate"})
POLE_LANDMARK_TYPES = frozenset({"pole"})
SUPPORTED_PRIMITIVE_TYPES = frozenset({"traverse", "slalom", "overfly", "orbit"})
SUPPORTED_SLALOM_SIDES = frozenset({"left", "right"})

# Criterios de despegue, continuidad temporal y ventana sincronizada de aterrizaje
TAKEOFF_MIN_ALT_M: float = 0.95
MIN_ACTIVE_GROUND_Z_M: float = 0.15
MAX_ALLOWED_DT_GAP_S: float = 0.15      # Hueco temporal maximo admisible entre muestras (150 ms)
MAX_ALLOWED_STEP_DIST_M: float = 0.35   # Paso espacial maximo admisible entre muestras consecutivas (m)
TIMESYNC_MATCH_WINDOW_S: float = 0.05   # Ventana maxima de coincidencia temporal entre salto de odometria y TimesyncStatus (s)
TIMESYNC_MATCH_TOL_S: float = 0.05      # Tolerancia maxima de magnitud entre salto de odometria y delta estimated_offset (s)

LANDING_WINDOW_S: float = 0.50          # Duracion de la ventana temporal final de aterrizaje (s, parametro de ingenieria)
LANDING_MAX_FINAL_Z_M: float = 0.25     # Altitud maxima en la ventana final de reposo (m)
LANDING_MAX_FINAL_SPEED: float = 0.25   # Velocidad maxima en la ventana final de reposo (m/s)
LANDING_MAX_DETECTOR_AGE_S: float = 1.5 # Antigüedad maxima del ultimo mensaje VehicleLandDetected al cierre (s)
LANDING_MAX_DETECTOR_INTERVAL_S: float = 1.5  # Intervalo maximo admisible sin evidencia entre mensajes del detector (s)

MCAP_MAGIC = b"\x89MCAP0\r\n"

EXPECTED_TOPIC_SCHEMAS = {
    "/fmu/out/vehicle_odometry": "px4_msgs/msg/VehicleOdometry",
    "/fmu/out/vehicle_local_position_v1": "px4_msgs/msg/VehicleLocalPosition",
    "/fmu/out/vehicle_land_detected": "px4_msgs/msg/VehicleLandDetected",
    "/fmu/out/vehicle_status_v4": "px4_msgs/msg/VehicleStatus",
    "/fmu/in/trajectory_setpoint": "px4_msgs/msg/TrajectorySetpoint",
    "/fmu/out/timesync_status": "px4_msgs/msg/TimesyncStatus",
}

EXPECTED_SCHEMA_REQUIRED_TOKENS: Dict[str, Tuple[str, ...]] = {
    "px4_msgs/msg/VehicleOdometry": (
        "uint64 timestamp",
        "uint64 timestamp_sample",
        "uint8 pose_frame",
        "float32[3] position",
        "float32[4] q",
        "uint8 velocity_frame",
        "float32[3] velocity",
        "uint8 reset_counter",
    ),
    "px4_msgs/msg/VehicleLocalPosition": (
        "uint64 timestamp",
        "uint64 timestamp_sample",
        "bool xy_valid",
        "bool z_valid",
        "float32 x",
        "float32 y",
        "float32 z",
        "uint8 xy_reset_counter",
        "uint8 z_reset_counter",
        "uint8 vxy_reset_counter",
        "uint8 vz_reset_counter",
        "float32 heading",
    ),
    "px4_msgs/msg/VehicleLandDetected": (
        "uint64 timestamp",
        "bool freefall",
        "bool ground_contact",
        "bool maybe_landed",
        "bool landed",
        "bool at_rest",
    ),
    "px4_msgs/msg/VehicleStatus": (
        "uint64 timestamp",
        "uint8 arming_state",
        "uint8 nav_state",
        "bool failsafe",
    ),
    "px4_msgs/msg/TrajectorySetpoint": (
        "uint64 timestamp",
        "float32[3] position",
        "float32[3] velocity",
        "float32 yaw",
        "float32 yawspeed",
    ),
    "px4_msgs/msg/TimesyncStatus": (
        "uint64 timestamp",
        "uint8 source_protocol",
        "uint64 remote_timestamp",
        "int64 observed_offset",
        "int64 estimated_offset",
        "uint32 round_trip_time",
    ),
}


def get_effective_validation_parameters() -> Dict[str, Any]:
    """Retorna un diccionario serializable con la version del validador y parametros efectivos."""
    tols = derive_validation_tolerances(
        uav_physical_radius_m=UAV_PHYSICAL_RADIUS_XY,
        uav_safety_margin_m=UAV_SAFETY_MARGIN_XY,
        pilot_buffer_m=PILOT_BUFFER_M,
    )
    return {
        "validator_version": VALIDATOR_VERSION,
        "protected_volume_model": PROTECTED_VOLUME_MODEL,
        "protected_volume_reference": PROTECTED_VOLUME_REFERENCE,
        "x500_sdf_commit": X500_SDF_MEASUREMENT["px4_git_commit"],
        "x500_full_rotor_tip_radius_xy_m": X500_FULL_ROTOR_TIP_RADIUS_M,
        "uav_physical_radius_xy_m": tols["uav_physical_radius_xy_m"],
        "uav_safety_margin_xy_m": tols["uav_safety_margin_xy_m"],
        "uav_collision_radius_xy_m": tols["uav_collision_radius_xy_m"],
        "collision_segment_max_step_m": COLLISION_SEGMENT_MAX_STEP_M,
        "discretization_margin_m": DISCRETIZATION_MARGIN_M,
        "traverse_max_abs_y_m": tols["traverse_max_abs_y_m"],
        "traverse_max_abs_z_m": tols["traverse_max_abs_z_m"],
        "traverse_circ_max_r_m": tols["traverse_circ_max_r_m"],
        "pilot_traverse_square_max_y_m": tols["pilot_traverse_square_max_y_m"],
        "pilot_traverse_square_max_z_m": tols["pilot_traverse_square_max_z_m"],
        "pilot_traverse_circ_max_r_m": tols["pilot_traverse_circ_max_r_m"],
        "slalom_min_lateral_m": tols["slalom_min_lateral_m"],
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
        "timesync_match_window_s": TIMESYNC_MATCH_WINDOW_S,
        "timesync_match_tol_s": TIMESYNC_MATCH_TOL_S,
        "landing_window_s": LANDING_WINDOW_S,
        "landing_max_xy_err_m": tols["landing_max_xy_err_m"],
        "landing_max_final_z_m": LANDING_MAX_FINAL_Z_M,
        "landing_max_final_speed_mps": LANDING_MAX_FINAL_SPEED,
        "landing_max_detector_age_s": LANDING_MAX_DETECTOR_AGE_S,
        "landing_max_detector_interval_s": LANDING_MAX_DETECTOR_INTERVAL_S,
    }


# ============================================================================
# Verificacion de Integridad MCAP y Lectura de Telemetria (rosbags + Fallback CDR)
# ============================================================================

class TelemetryBundle(dict):
    """Diccionario de telemetria con metadatos del lector (`reader_info`) adjuntos."""

    def __init__(self, *args: Any, reader_info: Optional[Dict[str, Any]] = None, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        self.reader_info: Dict[str, Any] = reader_info or {}


def _read_mcap_str(buf: bytes, offset: int, limit: Optional[int] = None) -> Tuple[str, int]:
    bound = len(buf) if limit is None else min(len(buf), limit)
    if offset + 4 > bound:
        raise ValueError("Truncated MCAP string length")
    length = struct.unpack_from("<I", buf, offset)[0]
    end = offset + 4 + length
    if end > bound:
        raise ValueError("Truncated MCAP string payload")
    return buf[offset + 4:end].decode("utf-8", errors="replace"), end


def _read_mcap_bytes(buf: bytes, offset: int, limit: Optional[int] = None) -> Tuple[bytes, int]:
    bound = len(buf) if limit is None else min(len(buf), limit)
    if offset + 4 > bound:
        raise ValueError("Truncated MCAP byte-array length")
    length = struct.unpack_from("<I", buf, offset)[0]
    end = offset + 4 + length
    if end > bound:
        raise ValueError("Truncated MCAP byte-array payload")
    return buf[offset + 4:end], end


def _verify_mcap_record_stream(buf: bytes, start: int, stop: int, mcap_name: str) -> None:
    """
    Recorre el flujo completo de registros MCAP entre `start` y `stop`, verificando:
      - Que cada cabecera y cuerpo de registro quepa íntegramente en `[start, stop]`.
      - En registros `Chunk` (`op = 0x06`), que su longitud interna sea consistente y,
        cuando `compression == ""` y `uncompressed_crc != 0`, que el CRC32 de los
        registros coincida exactamente con `zlib.crc32`.
      - Que el flujo alcance exactamente `stop` sin bytes huérfanos ni truncamiento intermedio.
    """
    o = start
    while o < stop:
        if o + 9 > stop:
            raise ValueError(f"truncated_mcap_record_header ({mcap_name} at offset {o})")
        op = buf[o]
        rlen = struct.unpack_from("<Q", buf, o + 1)[0]
        rec_start = o + 9
        rec_end = rec_start + rlen
        if rec_end > stop:
            raise ValueError(
                f"truncated_mcap_record_body ({mcap_name} op=0x{op:02x} at offset {o}: {rec_end} > {stop})"
            )

        if op == 0x06:  # Chunk record
            if rlen < 36:
                raise ValueError(f"corrupted_mcap_chunk_header_too_short ({mcap_name} at offset {o})")
            uncompressed_size = struct.unpack_from("<Q", buf, rec_start + 16)[0]
            uncompressed_crc = struct.unpack_from("<I", buf, rec_start + 24)[0]
            comp, p = _read_mcap_str(buf, rec_start + 28, limit=rec_end)
            if p + 8 > rec_end:
                raise ValueError(f"truncated_mcap_chunk_records_len ({mcap_name} at offset {o})")
            clen = struct.unpack_from("<Q", buf, p)[0]
            chunk_data_start = p + 8
            chunk_data_end = chunk_data_start + clen
            if chunk_data_end != rec_end:
                raise ValueError(
                    f"corrupted_mcap_chunk_records_bounds ({mcap_name} at offset {o}: {chunk_data_end} != {rec_end})"
                )
            if comp == "":
                if uncompressed_size != clen:
                    raise ValueError(
                        f"corrupted_mcap_chunk_uncompressed_size ({mcap_name} at offset {o}: {uncompressed_size} != {clen})"
                    )
                chunk_bytes = buf[chunk_data_start:chunk_data_end]
                if uncompressed_crc != 0:
                    computed_crc = zlib.crc32(chunk_bytes) & 0xFFFFFFFF
                    if computed_crc != uncompressed_crc:
                        raise ValueError(
                            f"corrupted_mcap_chunk_crc32 ({mcap_name} at offset {o}: "
                            f"expected 0x{uncompressed_crc:08x}, got 0x{computed_crc:08x})"
                        )
                _verify_mcap_record_stream(buf, chunk_data_start, chunk_data_end, mcap_name)

        o = rec_end


def verify_mcap_integrity(mcap_path: Path) -> bytes:
    """
    Verifica que el archivo `.mcap` no este vacio, truncado ni corrupto a mitad de archivo:
      1. Comienza y termina con los 8 bytes magicos `\\x89MCAP0\\r\\n`.
      2. Contiene el registro `Footer` (`opcode = 0x02`, `record_len = 20`) inmediatamente
         antes del magic final y verifica `summary_crc` si está presente (`!= 0`).
      3. Recorre la totalidad del flujo de registros de principio a fin y valida tamaños y
         `uncompressed_crc` (`zlib.crc32`) en todos los chunks no comprimidos.
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

    summary_start, summary_offset_start, summary_crc = struct.unpack_from("<QQI", buf, footer_pos + 9)
    if summary_crc != 0 and summary_start != 0:
        if summary_start < 8 or summary_start > footer_pos:
            raise ValueError(f"corrupted_mcap_summary_start ({mcap_path.name})")
        summary_region = buf[summary_start : footer_pos + 9 + 16]
        calc_sum_crc = zlib.crc32(summary_region) & 0xFFFFFFFF
        if calc_sum_crc != summary_crc:
            raise ValueError(
                f"corrupted_mcap_summary_crc32 ({mcap_path.name}: expected 0x{summary_crc:08x}, got 0x{calc_sum_crc:08x})"
            )

    _verify_mcap_record_stream(buf, 8, len(buf) - 8, mcap_path.name)
    return buf


def _verify_schema_definition_compatibility(schema_name: str, schema_encoding: str, schema_data: bytes) -> None:
    """
    Si el registro Schema del MCAP incluye el texto de definicion del mensaje (`ros2msg`/`ros2idl`),
    verifica que contenga los campos requeridos por el layout CDR utilizado por el validador.
    """
    req_tokens = EXPECTED_SCHEMA_REQUIRED_TOKENS.get(schema_name)
    if not req_tokens:
        return
    if schema_encoding and schema_encoding not in ("ros2msg", "ros2idl", "cdr", ""):
        raise ValueError(f"unsupported_schema_encoding ({schema_name}: {schema_encoding})")
    if not schema_data:
        return
    text = schema_data.decode("utf-8", errors="replace")
    normalized = " ".join(text.split())
    for tok in req_tokens:
        if tok not in normalized:
            raise ValueError(
                f"incompatible_schema_definition ({schema_name} missing required field '{tok}')"
            )


def _parse_mcap_native(
    mcap_path: Path,
    buf: Optional[bytes] = None,
) -> Optional[Tuple[Dict[str, List[Dict[str, Any]]], List[str]]]:
    """
    Deserializador nativo MCAP + CDR (little-endian XCDR1) verificado contra las definiciones
    de `src/px4_msgs/msg/*.msg`:
      - Valida integridad estructural del archivo MCAP y de cada registro/chunk (rechaza truncados y CRCs inválidos).
      - Exige que todo canal de telemetría relevante tenga un `Schema` válido y compatible con `EXPECTED_TOPIC_SCHEMAS`.
      - Verifica `encoding == 'cdr'` y el encabezado de encapsulamiento CDR little-endian (`0x00, 0x01`).
      - Extrae todos los contadores de reset de `VehicleLocalPosition` (`xy_reset_counter`, `z_reset_counter`,
        `vxy_reset_counter`, `vz_reset_counter`, `heading_reset_counter`).
      - Si encuentra chunks comprimidos (zstd/lz4), retorna `None` para que el llamador lo rechace si no hay `rosbags`.
    """
    if buf is None:
        buf = verify_mcap_integrity(mcap_path)

    schemas: Dict[int, Tuple[str, str, bytes]] = {}
    channels: Dict[int, Tuple[str, str, str, int]] = {}
    verified_schemas: set[str] = set()
    telemetry: Dict[str, List[Dict[str, Any]]] = {
        "odometry": [],
        "local_pos": [],
        "land_detected": [],
        "vehicle_status": [],
        "setpoints": [],
        "timesync_status": [],
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
                if rlen < 10:
                    raise ValueError("corrupted_mcap_schema_record_too_short")
                sid = struct.unpack_from("<H", b, rec_start)[0]
                sname, next_p = _read_mcap_str(b, rec_start + 2, limit=rec_end)
                senc, next_p = _read_mcap_str(b, next_p, limit=rec_end)
                sdata, _ = _read_mcap_bytes(b, next_p, limit=rec_end)
                _verify_schema_definition_compatibility(sname, senc, sdata)
                schemas[sid] = (sname, senc, sdata)

            elif op == 0x04:  # Channel record
                if rlen < 12:
                    raise ValueError("corrupted_mcap_channel_record_too_short")
                cid, sid = struct.unpack_from("<HH", b, rec_start)
                topic, next_p = _read_mcap_str(b, rec_start + 4, limit=rec_end)
                msg_enc, _ = _read_mcap_str(b, next_p, limit=rec_end)
                if topic in EXPECTED_TOPIC_SCHEMAS:
                    if sid == 0 or sid not in schemas:
                        raise ValueError(f"missing_schema_for_channel ({topic}, schema_id={sid})")
                    sname, senc, sdata = schemas[sid]
                    expected_schema = EXPECTED_TOPIC_SCHEMAS[topic]
                    if sname != expected_schema:
                        raise ValueError(f"schema_mismatch_for_{topic} ({sname!r} != {expected_schema!r})")
                    if msg_enc != "cdr":
                        raise ValueError(f"unsupported_message_encoding ({topic}: {msg_enc!r})")
                    _verify_schema_definition_compatibility(sname, senc, sdata)
                    verified_schemas.add(sname)
                else:
                    sname = schemas.get(sid, ("", "", b""))[0]
                channels[cid] = (topic, sname, msg_enc, sid)

            elif op == 0x05:  # Message record
                if rlen < 22:
                    raise ValueError("corrupted_mcap_message_record_too_short")
                cid = struct.unpack_from("<H", b, rec_start)[0]
                if cid not in channels:
                    raise ValueError(f"unknown_mcap_channel_id ({cid})")
                log_time = struct.unpack_from("<Q", b, rec_start + 6)[0]
                raw = b[rec_start + 22:rec_end]
                topic, sname, msg_enc, sid = channels[cid]

                if topic in EXPECTED_TOPIC_SCHEMAS:
                    expected_schema = EXPECTED_TOPIC_SCHEMAS[topic]
                    if sname != expected_schema:
                        raise ValueError(f"schema_mismatch_for_{topic} ({sname!r} != {expected_schema!r})")
                    if msg_enc != "cdr":
                        raise ValueError(f"unsupported_message_encoding ({msg_enc!r})")
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
                    # 108..120: heading_var, unaided_heading, delta_heading (3x float32), 120: heading_reset_counter (uint8)
                    if len(raw) < 108:
                        raise ValueError(f"truncated_cdr_payload_vehicle_local_position ({len(raw)} < 108)")
                    ts_us, ts_sample_us = struct.unpack_from("<QQ", raw, 4)
                    xy_valid = bool(raw[20])
                    z_valid = bool(raw[21])
                    v_xy_valid = bool(raw[22])
                    v_z_valid = bool(raw[23])
                    x, y, z = struct.unpack_from("<fff", raw, 24)
                    xy_reset_counter = int(raw[44])
                    z_reset_counter = int(raw[52])
                    vx, vy, vz = struct.unpack_from("<fff", raw, 56)
                    vxy_reset_counter = int(raw[80])
                    vz_reset_counter = int(raw[88])
                    heading = float(struct.unpack_from("<f", raw, 104)[0])
                    lp_entry: Dict[str, Any] = {
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
                        "xy_reset_counter": xy_reset_counter,
                        "z_reset_counter": z_reset_counter,
                        "vxy_reset_counter": vxy_reset_counter,
                        "vz_reset_counter": vz_reset_counter,
                    }
                    if len(raw) >= 121:
                        lp_entry["heading_reset_counter"] = int(raw[120])
                    telemetry["local_pos"].append(lp_entry)

                elif topic == "/fmu/out/vehicle_land_detected":
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

                elif topic == "/fmu/out/timesync_status":
                    # Layout CDR exacto segun src/px4_msgs/msg/TimesyncStatus.msg:
                    # 4..12: timestamp (uint64)
                    # 12..13: source_protocol (uint8), 13..20: padding a 8 bytes
                    # 20..28: remote_timestamp (uint64)
                    # 28..36: observed_offset (int64)
                    # 36..44: estimated_offset (int64)
                    # 44..48: round_trip_time (uint32)
                    if len(raw) < 48:
                        raise ValueError(f"truncated_cdr_payload_timesync_status ({len(raw)} < 48)")
                    ts_us = struct.unpack_from("<Q", raw, 4)[0]
                    source_protocol = int(raw[12])
                    remote_ts_us, obs_offset_us, est_offset_us, rtt_us = struct.unpack_from("<QqqI", raw, 20)
                    telemetry["timesync_status"].append({
                        "timestamp": int(log_time),
                        "timestamp_us": int(ts_us),
                        "source_protocol": source_protocol,
                        "remote_timestamp": int(remote_ts_us),
                        "remote_timestamp_us": int(remote_ts_us),
                        "observed_offset": int(obs_offset_us),
                        "observed_offset_us": int(obs_offset_us),
                        "estimated_offset": int(est_offset_us),
                        "estimated_offset_us": int(est_offset_us),
                        "round_trip_time": int(rtt_us),
                        "round_trip_time_us": int(rtt_us),
                    })

            elif op == 0x06:  # Chunk record
                comp, p = _read_mcap_str(b, rec_start + 28, limit=rec_end)
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
    return telemetry, sorted(verified_schemas)


def _parse_with_rosbags(mcaps: List[Path]) -> Tuple[Dict[str, List[Dict[str, Any]]], List[str], str]:
    """Lee una lista de archivos `.mcap` utilizando la biblioteca mantenida `rosbags`."""
    import rosbags
    from rosbags.highlevel import AnyReader
    from rosbags.typesys import Stores, get_typestore

    typestore = get_typestore(getattr(Stores, "ROS2_JAZZY", Stores.LATEST))
    # Si el archivo MCAP incluye definiciones de tipos custom (ej. px4_msgs), registrarlas en el typestore
    from rosbags.typesys import get_types_from_msg

    merged: Dict[str, List[Dict[str, Any]]] = {
        "odometry": [],
        "local_pos": [],
        "land_detected": [],
        "vehicle_status": [],
        "setpoints": [],
        "timesync_status": [],
    }
    verified_schemas: set[str] = set()
    with AnyReader(mcaps, default_typestore=typestore) as reader:
        for conn in reader.connections:
            if conn.topic in EXPECTED_TOPIC_SCHEMAS:
                expected_schema = EXPECTED_TOPIC_SCHEMAS[conn.topic]
                if conn.msgtype != expected_schema:
                    raise ValueError(f"schema_mismatch_for_{conn.topic} ({conn.msgtype!r} != {expected_schema!r})")
                verified_schemas.add(conn.msgtype)
            if conn.msgdef and getattr(conn.msgdef, "data", None) and conn.msgtype not in typestore.types:
                typestore.register(get_types_from_msg(conn.msgdef.data, conn.msgtype))

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
                lp_item: Dict[str, Any] = {
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
                }
                for rc_field in (
                    "xy_reset_counter",
                    "z_reset_counter",
                    "vxy_reset_counter",
                    "vz_reset_counter",
                    "heading_reset_counter",
                ):
                    if hasattr(msg, rc_field):
                        lp_item[rc_field] = int(getattr(msg, rc_field))
                merged["local_pos"].append(lp_item)
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
            elif topic == "/fmu/out/timesync_status":
                remote_ts = int(getattr(msg, "remote_timestamp", 0))
                obs_offset = int(getattr(msg, "observed_offset", 0))
                est_offset = int(getattr(msg, "estimated_offset", 0))
                merged["timesync_status"].append({
                    "timestamp": int(timestamp),
                    "timestamp_us": int(getattr(msg, "timestamp", 0)),
                    "source_protocol": int(getattr(msg, "source_protocol", 0)),
                    "remote_timestamp": remote_ts,
                    "remote_timestamp_us": remote_ts,
                    "observed_offset": obs_offset,
                    "observed_offset_us": obs_offset,
                    "estimated_offset": est_offset,
                    "estimated_offset_us": est_offset,
                    "round_trip_time": int(getattr(msg, "round_trip_time", 0)),
                })
    rb_ver = str(getattr(rosbags, "__version__", "unknown"))
    return merged, sorted(verified_schemas), rb_ver


def load_episode_telemetry(
    bag_path: Path,
    *,
    backend: str = "auto",
) -> TelemetryBundle:
    """
    Carga la telemetria completa de un episodio desde un directorio `rosbag/` o archivo `.mcap`.
    1. Verifica siempre la integridad estructural completa de todos los archivos `.mcap`
       (header magic, trailer magic, Footer record, flujo completo de registros y CRC32 de chunks).
    2. Política sin degradación silenciosa (`C5`):
       - Si `backend == 'rosbags'`, exige que `rosbags` esté instalado (si falta, lanza
         `RuntimeError('missing_reader_dependency:rosbags')`) y propaga cualquier fallo de
         deserialización sin caer a `_parse_mcap_native`.
       - Si `backend == 'auto'` y `rosbags` está instalado, usa `rosbags` y NUNCA captura
         excepciones para caer silenciosamente a `_parse_mcap_native`: cualquier fallo de
         `rosbags` se propaga como `ValueError('rosbags_deserialization_error: ...')`.
       - Solo cuando `backend == 'native'` o cuando `backend == 'auto'` y `rosbags` no está
         instalado (`ImportError`) se utiliza `_parse_mcap_native` con verificación de esquema/CRC.
    """
    if backend not in ("auto", "rosbags", "native"):
        raise ValueError(f"unsupported_telemetry_backend ({backend})")

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

    integrity_checks = [
        "header_magic",
        "footer_record",
        "trailer_magic",
        "record_stream_bounds",
        "uncompressed_chunk_crc32",
    ]

    if backend in ("auto", "rosbags"):
        try:
            import rosbags.highlevel  # noqa: F401
            has_rosbags = True
        except ImportError as exc:
            has_rosbags = False
            if backend == "rosbags":
                raise RuntimeError("missing_reader_dependency:rosbags") from exc

        if has_rosbags:
            try:
                merged_rb, schemas_rb, rb_ver = _parse_with_rosbags(mcaps)
            except Exception as exc:
                raise ValueError(f"rosbags_deserialization_error ({type(exc).__name__}: {exc})") from exc
            r_info = {
                "backend": "rosbags",
                "backend_version": rb_ver,
                "integrity_checks": integrity_checks,
                "schema_revision": "ros2_msgdef_xcdr1",
                "schemas_verified": schemas_rb,
            }
            return TelemetryBundle(merged_rb, reader_info=r_info)

    # Lector nativo MCAP + CDR con verificación de esquema, CRC32 y longitud de payload
    merged: Dict[str, List[Dict[str, Any]]] = {
        "odometry": [],
        "local_pos": [],
        "land_detected": [],
        "vehicle_status": [],
        "setpoints": [],
        "timesync_status": [],
    }
    all_schemas: set[str] = set()
    for mp in mcaps:
        res = _parse_mcap_native(mp, buf=raw_buffers[mp])
        if res is None:
            raise RuntimeError(
                f"compressed_mcap_requires_rosbags ({mp.name} contiene chunks comprimidos y rosbags no está instalado)"
            )
        tel_part, schemas_part = res
        all_schemas.update(schemas_part)
        for k in merged:
            merged[k].extend(tel_part[k])

    r_info = {
        "backend": "native_cdr",
        "backend_version": VALIDATOR_VERSION,
        "integrity_checks": integrity_checks,
        "schema_revision": "px4_msgs_xcdr1_little_endian",
        "schemas_verified": sorted(all_schemas),
    }
    return TelemetryBundle(merged, reader_info=r_info)


# ============================================================================
# Geometria Analitica de Distancia a Obstaculos (con Cota Inferior de Discretizacion)
# ============================================================================

def compute_min_distance_to_landmark(
    pos_enu: np.ndarray,
    landmark: Dict[str, Any],
    max_step_m: float = COLLISION_SEGMENT_MAX_STEP_M,
    *,
    return_lower_bound: bool = False,
) -> float:
    """
    Calcula la distancia mínima euclídea (en metros) desde la trayectoria poligonal 3D
    subdividida a paso `<= max_step_m` hasta la superficie sólida de un landmark
    ('gate'/'square_gate', 'circular_gate' con sus 16 cilindros finitos reales, o 'pole').
    Si `return_lower_bound=True`, retorna la cota inferior conservadora de discretización
    `max(0.0, d_sampled - max_step_m / 2.0)`.
    """
    clearance = compute_landmark_clearance(pos_enu, landmark, max_step_m=max_step_m)
    if return_lower_bound:
        return float(clearance["lower_bound_distance_m"])
    return float(clearance["sampled_min_distance_m"])


def _verify_single_primitive(
    pos_enu: np.ndarray,
    yaw_enu: np.ndarray,
    prim: Dict[str, Any],
    landmark: Dict[str, Any],
    *,
    start_from_index: int = 0,
    ts_sim_s: Optional[np.ndarray] = None,
) -> Dict[str, Any]:
    """
    Verifica geometricamente una primitiva individual sobre la trayectoria medida
    buscando su ejecucion a partir de `start_from_index` (inclusive).
    Retorna un diccionario con:
      - `passed` (bool), `reason` (Optional[str])
      - `start_index`, `event_index`, `completion_index` (Optional[int], indices globales)
      - `start_time_s`, `event_time_s`, `completion_time_s` (Optional[float])
      - `min_surface_clearance_sampled_m`, `discretization_margin_m`, `min_surface_clearance_m` (cota inferior)
      - metricas especificas de la maniobra.
    """
    p_type = str(prim.get("type", "traverse")).lower()
    target_id = str(prim.get("target", landmark.get("id", "unknown")))
    lm_type = str(landmark.get("type", "gate")).lower()
    cx = float(landmark["x"])
    cy = float(landmark["y"])
    cz = float(landmark.get("z", 1.5))
    lm_yaw = float(landmark.get("yaw", 0.0))

    N_total = len(pos_enu)
    start_idx_clamped = max(0, min(int(start_from_index), N_total))

    clearance_info = compute_landmark_clearance(
        pos_enu, landmark, max_step_m=COLLISION_SEGMENT_MAX_STEP_M
    )
    sampled_surf_dist = float(clearance_info["sampled_min_distance_m"])
    disc_margin = float(clearance_info["discretization_margin_m"])
    lower_bound_surf_dist = float(clearance_info["lower_bound_distance_m"])

    def _idx_to_time(idx_val: Optional[int]) -> Optional[float]:
        if idx_val is None or ts_sim_s is None or idx_val < 0 or idx_val >= len(ts_sim_s):
            return None
        t_v = float(ts_sim_s[idx_val])
        return round(t_v, 4) if math.isfinite(t_v) else None

    result: Dict[str, Any] = {
        "target": target_id,
        "primitive": p_type,
        "landmark_type": lm_type,
        "search_start_index": int(start_idx_clamped),
        "min_surface_clearance_sampled_m": round(sampled_surf_dist, 4),
        "discretization_margin_m": round(disc_margin, 4),
        "min_surface_clearance_m": round(lower_bound_surf_dist, 4),
        "start_index": None,
        "event_index": None,
        "completion_index": None,
        "start_time_s": None,
        "event_time_s": None,
        "completion_time_s": None,
        "passed": False,
        "reason": None,
    }

    # 1. Comprobar colision contra la estructura del landmark usando la cota inferior conservadora
    # d_inferior = max(0, d_muestreada - h/2) < UAV_COLLISION_RADIUS_XY
    if lower_bound_surf_dist < UAV_COLLISION_RADIUS_XY:
        result["reason"] = (
            f"collision_with_{target_id} "
            f"(lower_bound_clearance={lower_bound_surf_dist:.4f}m [sampled={sampled_surf_dist:.4f}m, h/2={disc_margin:.3f}m] "
            f"< {UAV_COLLISION_RADIUS_XY:.2f}m)"
        )
        return result

    if N_total - start_idx_clamped < 2:
        result["reason"] = f"{p_type}_{target_id}_no_samples_after_previous_primitive"
        return result

    pts_loc = _world_to_landmark_local(pos_enu, cx, cy, cz, lm_yaw)
    x_loc = pts_loc[:, 0]
    y_loc = pts_loc[:, 1]
    z_loc = pts_loc[:, 2]
    dist_xy = np.hypot(x_loc, y_loc)

    # Recorte progresivo a partir de `start_idx_clamped`
    x_sub = x_loc[start_idx_clamped:]
    y_sub = y_loc[start_idx_clamped:]
    z_sub = z_loc[start_idx_clamped:]
    dist_xy_sub = dist_xy[start_idx_clamped:]

    # 2. Evaluar segun el tipo de primitiva en la ventana [start_idx_clamped:]
    if p_type == "traverse":
        cross_rel = np.where(
            (x_sub[:-1] <= 0.0) & (x_sub[1:] > 0.0) & (dist_xy_sub[:-1] < 2.5)
        )[0]
        if len(cross_rel) == 0:
            result["reason"] = f"traverse_{target_id}_no_plane_crossing"
            return result

        # Tomar el primer cruce cronologico a partir de `start_from_index`
        idx = int(start_idx_clamped + int(cross_rel[0]))
        dx_step = float(x_loc[idx + 1] - x_loc[idx])
        alpha = float(-x_loc[idx] / dx_step) if abs(dx_step) > 1e-9 else 0.5
        alpha = max(0.0, min(1.0, alpha))
        y_c = float(y_loc[idx] + alpha * (y_loc[idx + 1] - y_loc[idx]))
        z_c = float(z_loc[idx] + alpha * (z_loc[idx + 1] - z_loc[idx]))
        r_c = math.hypot(y_c, z_c)

        start_i = idx
        event_i = idx
        comp_i = min(N_total - 1, idx + 1)
        result["start_index"] = start_i
        result["event_index"] = event_i
        result["completion_index"] = comp_i
        result["start_time_s"] = _idx_to_time(start_i)
        result["event_time_s"] = _idx_to_time(event_i)
        result["completion_time_s"] = _idx_to_time(comp_i)
        result["cross_y_m"] = round(y_c, 4)
        result["cross_z_m"] = round(z_c, 4)
        result["cross_radial_err_m"] = round(r_c, 4)

        if "circ" in lm_type or "torus" in lm_type:
            if r_c > TRAVERSE_CIRC_MAX_R:
                result["reason"] = (
                    f"traverse_{target_id}_outside_circular_aperture (r={r_c:.3f}m > {TRAVERSE_CIRC_MAX_R:.4f}m)"
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

        cross_rel = np.where(
            (x_sub[:-1] <= 0.0) & (x_sub[1:] > 0.0) & (np.abs(y_sub[:-1]) <= SLALOM_MAX_LATERAL + 0.5)
        )[0]
        if len(cross_rel) == 0:
            result["reason"] = f"slalom_{target_id}_no_longitudinal_pass"
            return result

        idx = int(start_idx_clamped + int(cross_rel[0]))
        dx_step = float(x_loc[idx + 1] - x_loc[idx])
        alpha = float(-x_loc[idx] / dx_step) if abs(dx_step) > 1e-9 else 0.5
        y_c = float(y_loc[idx] + alpha * (y_loc[idx + 1] - y_loc[idx]))
        start_i = idx
        event_i = idx
        comp_i = min(N_total - 1, idx + 1)
        result["start_index"] = start_i
        result["event_index"] = event_i
        result["completion_index"] = comp_i
        result["start_time_s"] = _idx_to_time(start_i)
        result["event_time_s"] = _idx_to_time(event_i)
        result["completion_time_s"] = _idx_to_time(comp_i)
        result["cross_y_m"] = round(y_c, 4)

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
        cross_rel = np.where(
            (x_sub[:-1] <= 0.0) & (x_sub[1:] > 0.0) & (np.abs(y_sub[:-1]) <= OVERFLY_MAX_LAT_M + 1.0)
        )[0]
        if len(cross_rel) == 0:
            result["reason"] = f"overfly_{target_id}_no_longitudinal_pass"
            return result

        idx = int(start_idx_clamped + int(cross_rel[0]))
        dx_step = float(x_loc[idx + 1] - x_loc[idx])
        alpha = float(-x_loc[idx] / dx_step) if abs(dx_step) > 1e-9 else 0.5
        y_c = float(y_loc[idx] + alpha * (y_loc[idx + 1] - y_loc[idx]))
        z_world_c = float(pos_enu[idx, 2] + alpha * (pos_enu[idx + 1, 2] - pos_enu[idx, 2]))

        over_mask_sub = dist_xy_sub <= 0.50
        pos_sub = pos_enu[start_idx_clamped:]
        min_z_over = float(np.min(pos_sub[over_mask_sub, 2])) if np.any(over_mask_sub) else z_world_c

        start_i = idx
        event_i = idx
        comp_i = min(N_total - 1, idx + 1)
        result["start_index"] = start_i
        result["event_index"] = event_i
        result["completion_index"] = comp_i
        result["start_time_s"] = _idx_to_time(start_i)
        result["event_time_s"] = _idx_to_time(event_i)
        result["completion_time_s"] = _idx_to_time(comp_i)
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

        near_rel = np.where(dist_xy_sub <= ORBIT_MAX_RADIUS_M)[0]
        if len(near_rel) < 10:
            result["reason"] = f"orbit_{target_id}_never_entered_orbit_radius"
            return result

        near_indices = near_rel + start_idx_clamped
        splits = np.where(np.diff(near_indices) > 5)[0]
        segments = [seg for seg in np.split(near_indices, splits + 1) if len(seg) >= 10]
        if not segments:
            result["reason"] = f"orbit_{target_id}_never_entered_orbit_radius"
            return result

        # Elegir el primer segmento contiguo significativo a partir de start_from_index
        main_seg = segments[0]
        angles = np.unwrap(np.arctan2(y_loc[main_seg], x_loc[main_seg]))
        running_min = np.minimum.accumulate(angles)
        ccw_prog_deg_arr = np.degrees(angles - running_min)
        ccw_progression_deg = float(np.max(ccw_prog_deg_arr))
        max_span_deg = math.degrees(float(np.max(angles) - np.min(angles)))
        mean_r = float(np.mean(dist_xy[main_seg]))

        start_i = int(main_seg[0])
        reached_rel = np.where(ccw_prog_deg_arr >= ORBIT_MIN_SWEEP_DEG)[0]
        if len(reached_rel) > 0:
            reach_k = int(reached_rel[0])
            comp_i = int(main_seg[reach_k])
            event_i = int(main_seg[reach_k // 2])
            active_arc_seg = main_seg[: reach_k + 1]
        else:
            comp_i = int(main_seg[-1])
            event_i = int(main_seg[len(main_seg) // 2])
            active_arc_seg = main_seg

        result["start_index"] = start_i
        result["event_index"] = event_i
        result["completion_index"] = comp_i
        result["start_time_s"] = _idx_to_time(start_i)
        result["event_time_s"] = _idx_to_time(event_i)
        result["completion_time_s"] = _idx_to_time(comp_i)

        # Error de orientacion centripeta durante el barrido orbital activo
        arc_mask = dist_xy[active_arc_seg] <= 2.2
        arc_seg = active_arc_seg[arc_mask] if np.any(arc_mask) else active_arc_seg
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


class ClockReconstructionResult(dict):
    """
    Resultado estructurado de `reconstruct_simulation_clock()`.
    Hereda de `dict` para serializacion directa e inspeccion por clave, y expone acceso por atributo.
    """

    def __getattr__(self, name: str) -> Any:
        try:
            return self[name]
        except KeyError as exc:
            raise AttributeError(f"'ClockReconstructionResult' object has no attribute '{name}'") from exc


def _extract_timesync_offset_steps(
    timesync_status: Optional[Sequence[Dict[str, Any]]],
) -> List[Dict[str, Any]]:
    """
    Extrae escalones de offset significativos (|Δestimated_offset| > 0.01 s) a partir de la
    secuencia de mensajes `/fmu/out/timesync_status`.
    Soporta tanto diccionarios con campos en microsegundos/nanosegundos (`estimated_offset_us`,
    `estimated_offset`, `timestamp`, `timestamp_us`) como en segundos (`estimated_offset_s`,
    `timestamp_bag_s`, `timestamp_px4_s`).
    """
    if not timesync_status or len(timesync_status) < 2:
        return []

    raw_offsets: List[float] = []
    has_explicit_s = False
    has_explicit_us = False
    for item in timesync_status:
        if "estimated_offset_s" in item:
            raw_offsets.append(float(item["estimated_offset_s"]))
            has_explicit_s = True
        elif "estimated_offset_us" in item:
            raw_offsets.append(float(item["estimated_offset_us"]) * 1e-6)
            has_explicit_us = True
        elif "estimated_offset" in item:
            raw_offsets.append(float(item["estimated_offset"]))
        else:
            raw_offsets.append(0.0)

    if not has_explicit_s and not has_explicit_us:
        max_abs = max((abs(v) for v in raw_offsets), default=0.0)
        if max_abs > 50.0:
            raw_offsets = [v * 1e-6 for v in raw_offsets]

    events: List[Dict[str, Any]] = []
    for m in range(1, len(timesync_status)):
        delta_offset_s = float(raw_offsets[m] - raw_offsets[m - 1])
        if abs(delta_offset_s) <= 0.010:
            continue

        cur = timesync_status[m]
        prev = timesync_status[m - 1]

        t_bag_s: Optional[float] = None
        if "timestamp_bag_s" in cur:
            t_bag_s = float(cur["timestamp_bag_s"])
        elif "timestamp" in cur and cur["timestamp"] is not None:
            val_b = float(cur["timestamp"])
            t_bag_s = val_b * 1e-9 if val_b > 1e12 else val_b

        prev_bag_s: Optional[float] = None
        if "timestamp_bag_s" in prev:
            prev_bag_s = float(prev["timestamp_bag_s"])
        elif "timestamp" in prev and prev["timestamp"] is not None:
            val_pb = float(prev["timestamp"])
            prev_bag_s = val_pb * 1e-9 if val_pb > 1e12 else val_pb

        t_px4_s: Optional[float] = None
        if "timestamp_px4_s" in cur:
            t_px4_s = float(cur["timestamp_px4_s"])
        elif "timestamp_us" in cur and cur["timestamp_us"] is not None:
            val_p = float(cur["timestamp_us"])
            t_px4_s = val_p * 1e-6 if val_p > 1e11 else val_p

        events.append({
            "msg_index": int(m),
            "t_bag_s": t_bag_s,
            "prev_t_bag_s": prev_bag_s,
            "t_px4_s": t_px4_s,
            "delta_offset_s": delta_offset_s,
            "used": False,
        })
    return events


def reconstruct_simulation_clock(
    ts_px4_s: np.ndarray,
    ts_bag_s: np.ndarray,
    timesync_status: Optional[Sequence[Dict[str, Any]]] = None,
) -> ClockReconstructionResult:
    """
    Analiza y reconstruye de forma condicionada el reloj de simulacion a partir del reloj de
    PX4 (`ts_px4_s`) y el reloj de recepcion del bag (`ts_bag_s`):
      - Patron A (muestra aislada en tiempo de arranque `< 1e6` s entre muestras UNIX `> 1e8` s):
        reparacion condicionada que verifica la consistencia del intervalo entre los vecinos UNIX
        validos (`raw_px4[k+1] - raw_px4[k-1]`) y reparte dicho intervalo proporcionalmente a
        `dt_bag`, preservando exactamente el span entre vecinos sin borrar huecos reales.
      - Patron B (escalon de offset UNIX entre muestras `> 1e8` s):
        * Si `timesync_status` está disponible y un salto en `VehicleOdometry.timestamp` coincide
          dentro de `±TIMESYNC_MATCH_WINDOW_S` (0.05 s) con un cambio en `TimesyncStatus.estimated_offset`
          de idéntica magnitud (`|Δt_odom - Δestimated_offset| <= TIMESYNC_MATCH_TOL_S`), se valida
          formalmente como reajuste de offset de red y se repara `reconstructed_time_s` aplicando
          el diferencial de offset.
        * En ausencia de correlación analítica con `timesync_status`, se marca como ambiguo
          (`ambiguous_clock_step_pattern_b`) y NO se repara automaticamente, conservando el
          intervalo original `dt_px4` para nunca comprimir una entrega en rafaga tras congelacion.
      - Retrocesos o duplicados (`dt <= 0`) no explicados por un Patron A valido o desambiguacion
        de `timesync_status` quedan visibles en `reconstructed_time_s` y en `backstep_or_duplicate_mask`,
        marcando `usable_for_validation=False`.
      - Relojes ausentes, no finitos o `<= 0` provocan rechazo inmediato (`usable_for_validation=False`).
    """
    raw_px4 = np.asarray(ts_px4_s, dtype=np.float64)
    raw_bag = np.asarray(ts_bag_s, dtype=np.float64)

    if raw_px4.ndim != 1 or raw_bag.ndim != 1 or len(raw_px4) != len(raw_bag):
        empty_b = np.zeros(0, dtype=bool)
        return ClockReconstructionResult(
            raw_time_s=raw_px4,
            received_time_s=raw_bag,
            reconstructed_time_s=None,
            repaired_step_mask=empty_b,
            backstep_or_duplicate_mask=empty_b,
            ambiguous_step_mask=empty_b,
            repaired_events=[],
            num_repaired_steps=0,
            num_backsteps_or_duplicates=0,
            num_ambiguous_steps=0,
            max_raw_dt_s=0.0,
            min_raw_dt_s=0.0,
            max_reconstructed_dt_s=0.0,
            min_reconstructed_dt_s=0.0,
            usable_for_validation=False,
            reasons=["clock_series_length_or_dimension_mismatch"],
        )

    N = len(raw_px4)
    M = max(0, N - 1)
    empty_b = np.zeros(M, dtype=bool)

    reasons: List[str] = []
    if N < 2:
        reasons.append("insufficient_clock_samples")
        return ClockReconstructionResult(
            raw_time_s=raw_px4.copy(),
            received_time_s=raw_bag.copy(),
            reconstructed_time_s=np.zeros(N, dtype=np.float64) if N == 1 else None,
            repaired_step_mask=empty_b,
            backstep_or_duplicate_mask=empty_b,
            ambiguous_step_mask=empty_b,
            repaired_events=[],
            num_repaired_steps=0,
            num_backsteps_or_duplicates=0,
            num_ambiguous_steps=0,
            max_raw_dt_s=0.0,
            min_raw_dt_s=0.0,
            max_reconstructed_dt_s=0.0,
            min_reconstructed_dt_s=0.0,
            usable_for_validation=False,
            reasons=reasons,
        )

    px4_valid = bool(np.all(np.isfinite(raw_px4)) and np.all(raw_px4 > 0.0))
    bag_valid = bool(np.all(np.isfinite(raw_bag)) and np.all(raw_bag > 0.0))
    if not px4_valid:
        reasons.append("missing_or_non_finite_px4_clock")
    if not bag_valid:
        reasons.append("missing_or_non_finite_bag_clock")

    if not px4_valid or not bag_valid:
        return ClockReconstructionResult(
            raw_time_s=raw_px4.copy(),
            received_time_s=raw_bag.copy(),
            reconstructed_time_s=None,
            repaired_step_mask=empty_b,
            backstep_or_duplicate_mask=empty_b,
            ambiguous_step_mask=np.ones(M, dtype=bool),
            repaired_events=[],
            num_repaired_steps=0,
            num_backsteps_or_duplicates=0,
            num_ambiguous_steps=int(M),
            max_raw_dt_s=0.0,
            min_raw_dt_s=0.0,
            max_reconstructed_dt_s=0.0,
            min_reconstructed_dt_s=0.0,
            usable_for_validation=False,
            reasons=reasons,
        )

    dt_px4 = np.diff(raw_px4)
    dt_bag = np.diff(raw_bag)
    dt_reconstructed = dt_px4.copy()

    repaired_step_mask = np.zeros(M, dtype=bool)
    ambiguous_step_mask = np.zeros(M, dtype=bool)
    repaired_events: List[Dict[str, Any]] = []

    if np.any(dt_bag <= 0.0):
        ambiguous_step_mask |= (dt_bag <= 0.0)
        reasons.append("non_monotonic_bag_reception_clock")

    # 1. Patron A: muestra aislada en tiempo de arranque (< 1e6) rodeada por muestras UNIX (> 1e8)
    for k in range(1, N - 1):
        if raw_px4[k] < 1e6 and raw_px4[k - 1] > 1e8 and raw_px4[k + 1] > 1e8:
            neighbor_px4_span = float(raw_px4[k + 1] - raw_px4[k - 1])
            dt_b0 = float(dt_bag[k - 1])
            dt_b1 = float(dt_bag[k])
            neighbor_bag_span = dt_b0 + dt_b1

            # Preservar siempre el intervalo real entre los vecinos UNIX validos (k-1 y k+1)
            # cuando dicho intervalo y los pasos de recepcion son positivos.
            if neighbor_px4_span > 0.0 and dt_b0 > 0.0 and dt_b1 > 0.0:
                w0 = dt_b0 / neighbor_bag_span
                rep_dt0 = neighbor_px4_span * w0
                rep_dt1 = neighbor_px4_span - rep_dt0
                dt_reconstructed[k - 1] = rep_dt0
                dt_reconstructed[k] = rep_dt1
            else:
                rep_dt0 = float(dt_px4[k - 1])
                rep_dt1 = float(dt_px4[k])

            # Solo aceptar como reparacion valida si los intervalos de recepcion son normales (<= 0.10 s),
            # el span entre vecinos UNIX concuerda con la recepcion (|span_px4 - span_bag| <= 0.05 s)
            # y ninguno de los subpasos reconstruidos excede MAX_ALLOWED_DT_GAP_S.
            if (
                neighbor_px4_span > 0.0
                and 0.0 < dt_b0 <= 0.100
                and 0.0 < dt_b1 <= 0.100
                and abs(neighbor_px4_span - neighbor_bag_span) <= 0.050
                and max(rep_dt0, rep_dt1) <= MAX_ALLOWED_DT_GAP_S
            ):
                repaired_step_mask[k - 1] = True
                repaired_step_mask[k] = True
                repaired_events.append(
                    {
                        "sample_index": int(k),
                        "step_indices": [int(k - 1), int(k)],
                        "raw_sample_time_s": float(raw_px4[k]),
                        "neighbor_px4_times_s": [float(raw_px4[k - 1]), float(raw_px4[k + 1])],
                        "neighbor_px4_span_s": round(neighbor_px4_span, 6),
                        "original_px4_dts_s": [float(dt_px4[k - 1]), float(dt_px4[k])],
                        "neighbor_bag_dts_s": [round(dt_b0, 6), round(dt_b1, 6)],
                        "repaired_dts_s": [round(rep_dt0, 6), round(rep_dt1, 6)],
                        "method": "pattern_a_proportional_neighbor_span",
                    }
                )
            else:
                ambiguous_step_mask[k - 1] = True
                ambiguous_step_mask[k] = True
                if "ambiguous_clock_step_pattern_a" not in reasons:
                    reasons.append("ambiguous_clock_step_pattern_a")

    # 2. Patron B y desambiguacion analitica mediante /fmu/out/timesync_status:
    # Si un salto entre muestras consecutivas > 1e8 s coincide en ±0.05 s con un escalon
    # en TimesyncStatus.estimated_offset de identica magnitud (dentro de ±0.05 s), se valida
    # como reajuste de offset de red y se repara aplicando el diferencial de offset.
    # De lo contrario, se conserva dt_px4[i] y se rechaza como ambiguo o dropout.
    ts_events = _extract_timesync_offset_steps(timesync_status)
    for i in range(M):
        if repaired_step_mask[i] or ambiguous_step_mask[i]:
            continue
        if (
            raw_px4[i] > 1e8
            and raw_px4[i + 1] > 1e8
            and 0.0 < dt_bag[i] <= 0.100
            and abs(dt_px4[i] - dt_bag[i]) > 0.050
        ):
            odom_step_excess = float(dt_px4[i] - dt_bag[i])
            matched_ev: Optional[Dict[str, Any]] = None
            matched_sign: float = 1.0

            for ev in ts_events:
                if ev["used"]:
                    continue
                time_ok = False
                if ev["t_bag_s"] is not None:
                    tb = float(ev["t_bag_s"])
                    if (raw_bag[i] - TIMESYNC_MATCH_WINDOW_S) <= tb <= (raw_bag[i + 1] + TIMESYNC_MATCH_WINDOW_S):
                        time_ok = True
                if not time_ok and ev["t_px4_s"] is not None:
                    tp = float(ev["t_px4_s"])
                    p_lo = min(float(raw_px4[i]), float(raw_px4[i + 1])) - TIMESYNC_MATCH_WINDOW_S
                    p_hi = max(float(raw_px4[i]), float(raw_px4[i + 1])) + TIMESYNC_MATCH_WINDOW_S
                    if p_lo <= tp <= p_hi:
                        time_ok = True
                if not time_ok:
                    continue

                d_off = float(ev["delta_offset_s"])
                for sgn in (1.0, -1.0):
                    eff_off = sgn * d_off
                    if (
                        abs(odom_step_excess - eff_off) <= TIMESYNC_MATCH_TOL_S
                        or abs(float(dt_px4[i]) - eff_off) <= TIMESYNC_MATCH_TOL_S
                    ):
                        matched_ev = ev
                        matched_sign = sgn
                        break
                if matched_ev is not None:
                    break

            if matched_ev is not None:
                eff_off = matched_sign * float(matched_ev["delta_offset_s"])
                candidate_dt = float(dt_px4[i]) - eff_off
                if candidate_dt < 1e-4 or abs(candidate_dt - float(dt_bag[i])) > float(dt_bag[i]):
                    rep_dt = float(dt_bag[i])
                else:
                    rep_dt = candidate_dt

                if 0.0 < rep_dt <= MAX_ALLOWED_DT_GAP_S:
                    matched_ev["used"] = True
                    dt_reconstructed[i] = rep_dt
                    repaired_step_mask[i] = True
                    repaired_events.append(
                        {
                            "step_index": int(i),
                            "step_indices": [int(i)],
                            "original_px4_dt_s": float(dt_px4[i]),
                            "bag_dt_s": round(float(dt_bag[i]), 6),
                            "timesync_msg_index": int(matched_ev["msg_index"]),
                            "timesync_delta_offset_s": round(float(eff_off), 6),
                            "repaired_dt_s": round(float(rep_dt), 6),
                            "method": "pattern_b_timesync_status_disambiguated",
                        }
                    )
                    continue

            lag_before = float(raw_bag[i] - raw_px4[i])
            lag_after = float(raw_bag[i + 1] - raw_px4[i + 1])
            if lag_before >= 0.150 and -0.020 <= lag_after <= 0.050:
                ambiguous_step_mask[i] = True
                if "ambiguous_clock_step_pattern_b" not in reasons:
                    reasons.append("ambiguous_clock_step_pattern_b")

    # 3. Retrocesos o duplicados quedan visibles en `reconstructed_time_s` y en su mascara
    backstep_or_duplicate_mask = dt_reconstructed <= 0.0
    if np.any(backstep_or_duplicate_mask):
        reasons.append("clock_backstep_or_duplicate_timestamp")

    reconstructed_time_s = np.concatenate([[0.0], np.cumsum(dt_reconstructed)])
    max_raw_dt_s = float(np.max(dt_px4))
    min_raw_dt_s = float(np.min(dt_px4))
    max_reconstructed_dt_s = float(np.max(dt_reconstructed))
    min_reconstructed_dt_s = float(np.min(dt_reconstructed))

    if max_reconstructed_dt_s > MAX_ALLOWED_DT_GAP_S:
        reasons.append("telemetry_dropout_gap")

    usable_for_validation = bool(
        len(reasons) == 0
        and not np.any(ambiguous_step_mask)
        and not np.any(backstep_or_duplicate_mask)
        and min_reconstructed_dt_s > 0.0
        and max_reconstructed_dt_s <= MAX_ALLOWED_DT_GAP_S
    )

    return ClockReconstructionResult(
        raw_time_s=raw_px4.copy(),
        received_time_s=raw_bag.copy(),
        reconstructed_time_s=reconstructed_time_s,
        repaired_step_mask=repaired_step_mask,
        backstep_or_duplicate_mask=backstep_or_duplicate_mask,
        ambiguous_step_mask=ambiguous_step_mask,
        repaired_events=repaired_events,
        num_repaired_steps=int(np.sum(repaired_step_mask)),
        num_backsteps_or_duplicates=int(np.sum(backstep_or_duplicate_mask)),
        num_ambiguous_steps=int(np.sum(ambiguous_step_mask)),
        max_raw_dt_s=max_raw_dt_s,
        min_raw_dt_s=min_raw_dt_s,
        max_reconstructed_dt_s=max_reconstructed_dt_s,
        min_reconstructed_dt_s=min_reconstructed_dt_s,
        usable_for_validation=usable_for_validation,
        reasons=reasons,
    )


def _validate_landing_window(
    pos_enu: np.ndarray,
    vel_enu: np.ndarray,
    ts_bag_raw_s: np.ndarray,
    land_list: List[Dict[str, Any]],
    landing_cfg: Dict[str, Any],
    *,
    landing_window_s: float = LANDING_WINDOW_S,
) -> Dict[str, Any]:
    """
    Evalua el aterrizaje final sobre una ventana temporal comun `[t_end - landing_window_s, t_end]`
    usando el tiempo de recepcion (`ts_bag_raw_s` y `land_detected.timestamp`) para alinear ambos topicos:
      1. Cobertura temporal completa de odometria durante `landing_window_s` sin huecos `> MAX_ALLOWED_DT_GAP_S`.
      2. Posicion XY dentro del pad, velocidad bajo `LANDING_MAX_FINAL_SPEED` y altura compatible
         con reposo (`0 <= z <= LANDING_MAX_FINAL_Z_M`) en TODAS las muestras de la ventana.
      3. Persistencia real de `VehicleLandDetected.landed == True` (nunca `ground_contact`):
         - Todos los timestamps del detector presentes, finitos y estrictamente positivos (`> 0`).
         - Al menos 2 mensajes distintos con `landed == True` en la racha final contigua.
         - Ningun mensaje con `landed == False` dentro de `[t_win_start, t_end]`.
         - Vigencia continua durante toda la ventana `[t_win_start, t_end]` con mensaje activo al
           inicio de la ventana y sin intervalos sin evidencia mayores a `LANDING_MAX_DETECTOR_INTERVAL_S`,
           ni antiguedad final mayor a `LANDING_MAX_DETECTOR_AGE_S`.
    """
    reasons: List[str] = []
    N = len(pos_enu)
    lx, ly = float(landing_cfg["x"]), float(landing_cfg["y"])

    if (
        N < 2
        or landing_window_s <= 0.0
        or not np.all(np.isfinite(ts_bag_raw_s))
        or np.any(ts_bag_raw_s <= 0.0)
    ):
        reasons.append("landing_window_unavailable_invalid_clock")
        final_pos = pos_enu[-1] if N > 0 else np.zeros(3, dtype=np.float64)
        final_speed = float(np.linalg.norm(vel_enu[-1])) if N > 0 else 0.0
        xy_err = float(math.hypot(float(final_pos[0]) - lx, float(final_pos[1]) - ly)) if N > 0 else -1.0
        return {
            "reasons": reasons,
            "landed_detector_confirmed": False,
            "landing_xy_error_m": xy_err,
            "final_window_max_z_m": float(final_pos[2]) if N > 0 else -1.0,
            "final_window_max_speed_mps": final_speed,
            "final_altitude_m": float(final_pos[2]) if N > 0 else -1.0,
            "final_speed_mps": final_speed,
            "window_sample_count": 0,
            "window_duration_s": 0.0,
        }

    t_end = float(ts_bag_raw_s[-1])
    t_start_ep = float(ts_bag_raw_s[0])
    t_win_start = t_end - float(landing_window_s)

    if (t_end - t_start_ep) + 1e-6 < float(landing_window_s):
        reasons.append(
            f"insufficient_landing_window_coverage (episode_span={t_end - t_start_ep:.3f}s < {landing_window_s:.2f}s)"
        )

    win_indices = np.where(ts_bag_raw_s >= t_win_start - 1e-9)[0]
    if len(win_indices) < 2:
        reasons.append(f"insufficient_odometry_in_landing_window (samples={len(win_indices)})")
        win_indices = np.arange(max(0, N - 2), N)
    else:
        first_win_idx = int(win_indices[0])
        if first_win_idx > 0:
            gap_at_win_start = float(ts_bag_raw_s[first_win_idx] - t_win_start)
            prev_gap = float(ts_bag_raw_s[first_win_idx] - ts_bag_raw_s[first_win_idx - 1])
            if gap_at_win_start > MAX_ALLOWED_DT_GAP_S or prev_gap > MAX_ALLOWED_DT_GAP_S:
                reasons.append(
                    f"insufficient_landing_window_coverage (start_gap={max(gap_at_win_start, prev_gap):.3f}s)"
                )
        elif (float(ts_bag_raw_s[0]) - t_win_start) > 1e-6:
            reasons.append("insufficient_landing_window_coverage (missing_samples_at_window_start)")

        win_dts = np.diff(ts_bag_raw_s[win_indices])
        if len(win_dts) > 0 and float(np.max(win_dts)) > MAX_ALLOWED_DT_GAP_S:
            reasons.append(
                f"insufficient_landing_window_coverage (max_win_dt={float(np.max(win_dts)):.3f}s > {MAX_ALLOWED_DT_GAP_S:.2f}s)"
            )

    final_window_pos = pos_enu[win_indices]
    final_window_vel = vel_enu[win_indices]
    final_pos = pos_enu[-1]
    final_speed = float(np.linalg.norm(vel_enu[-1]))
    final_window_max_z = float(np.max(final_window_pos[:, 2]))
    final_window_min_z = float(np.min(final_window_pos[:, 2]))
    final_window_max_speed = float(np.max(np.linalg.norm(final_window_vel, axis=1)))
    landing_xy_err = float(
        np.max(np.hypot(final_window_pos[:, 0] - lx, final_window_pos[:, 1] - ly))
    )

    # Evaluar detector de aterrizaje en la misma ventana temporal [t_win_start, t_end]
    landed_flag_confirmed = False
    if not land_list:
        reasons.append("missing_land_detected_messages")
    else:
        ld_times_s: List[float] = []
        ld_flags: List[bool] = []
        invalid_ld_ts = False
        for msg in land_list:
            if not isinstance(msg, dict):
                invalid_ld_ts = True
                break
            raw_ld_ts = msg.get("timestamp")
            if not _is_finite_number(raw_ld_ts) or float(raw_ld_ts) <= 0.0:
                invalid_ld_ts = True
                break
            ld_times_s.append(float(raw_ld_ts) * 1e-9)
            # Nunca usar ground_contact como sustituto de landed
            ld_flags.append(bool(msg.get("landed", False)))

        if invalid_ld_ts:
            reasons.append("invalid_land_detector_timestamp")
        else:
            # Verificar orden temporal no decreciente
            non_decreasing = all(
                ld_times_s[k + 1] >= ld_times_s[k] for k in range(len(ld_times_s) - 1)
            )
            if not non_decreasing:
                reasons.append("non_monotonic_land_detector_timestamps")
            else:
                last_ld_ts_s = ld_times_s[-1]
                ld_age_s = t_end - last_ld_ts_s
                if ld_age_s > LANDING_MAX_DETECTOR_AGE_S or last_ld_ts_s < t_win_start - 1e-6:
                    reasons.append(
                        f"stale_land_detector_message (age={ld_age_s:.2f}s > {LANDING_MAX_DETECTOR_AGE_S:.2f}s)"
                    )

                # Si existe algun mensaje con landed=False dentro de [t_win_start, t_end], invalida persistencia
                false_in_window = any(
                    (t_win_start - 1e-6 <= t_m <= t_end + 1e-6) and (not flag)
                    for t_m, flag in zip(ld_times_s, ld_flags)
                )

                # Extraer la racha final contigua de mensajes con landed == True
                true_run_times: List[float] = []
                for t_m, flag in zip(reversed(ld_times_s), reversed(ld_flags)):
                    if flag:
                        true_run_times.append(t_m)
                    else:
                        break
                true_run_times.reverse()

                # Exigir al menos 2 mensajes distintos con landed == True
                distinct_true_Count = len(true_run_times) >= 2 and (
                    true_run_times[-1] > true_run_times[0] + 1e-9
                )

                if not false_in_window and distinct_true_Count:
                    # Para cubrir toda la ventana [t_win_start, t_end] mediante regla explicita de vigencia:
                    # 1. Debe existir un mensaje landed=True en o antes de t_win_start (vigente al entrar a la ventana)
                    #    cuya edad en t_win_start no supere LANDING_MAX_DETECTOR_INTERVAL_S.
                    # 2. Ningun intervalo entre mensajes consecutivos de la racha que cubren la ventana
                    #    debe superar LANDING_MAX_DETECTOR_INTERVAL_S.
                    # 3. La edad del ultimo mensaje en t_end no debe superar LANDING_MAX_DETECTOR_AGE_S.
                    prior_or_at_start_indices = [
                        idx for idx, t_m in enumerate(true_run_times) if t_m <= t_win_start + 1e-6
                    ]
                    if prior_or_at_start_indices:
                        anchor_idx = prior_or_at_start_indices[-1]
                        anchor_age_at_win_start = max(0.0, t_win_start - true_run_times[anchor_idx])
                        relevant_times = true_run_times[anchor_idx:]
                        max_internal_gap = (
                            max(
                                relevant_times[k + 1] - relevant_times[k]
                                for k in range(len(relevant_times) - 1)
                            )
                            if len(relevant_times) >= 2
                            else 0.0
                        )
                        end_age = max(0.0, t_end - relevant_times[-1])
                        if (
                            anchor_age_at_win_start <= LANDING_MAX_DETECTOR_INTERVAL_S
                            and max_internal_gap <= LANDING_MAX_DETECTOR_INTERVAL_S
                            and end_age <= LANDING_MAX_DETECTOR_AGE_S
                            and relevant_times[-1] >= t_win_start - 1e-6
                        ):
                            landed_flag_confirmed = True

                if not landed_flag_confirmed:
                    reasons.append("px4_land_detector_not_confirmed")

    if landing_xy_err > LANDING_MAX_XY_ERR_M:
        reasons.append(
            f"landing_off_pad (xy_err={landing_xy_err:.3f}m > {LANDING_MAX_XY_ERR_M:.2f}m)"
        )

    if final_window_max_z > LANDING_MAX_FINAL_Z_M or final_window_min_z < -0.10:
        reasons.append(
            f"landing_incomplete_altitude (final_window_max_z={final_window_max_z:.3f}m > {LANDING_MAX_FINAL_Z_M:.2f}m)"
        )

    if final_window_max_speed > LANDING_MAX_FINAL_SPEED:
        reasons.append(
            f"landing_not_at_rest (final_window_max_speed={final_window_max_speed:.3f}m/s > {LANDING_MAX_FINAL_SPEED:.2f}m/s)"
        )

    return {
        "reasons": reasons,
        "landed_detector_confirmed": landed_flag_confirmed,
        "landing_xy_error_m": landing_xy_err,
        "final_window_max_z_m": final_window_max_z,
        "final_window_max_speed_mps": final_window_max_speed,
        "final_altitude_m": float(final_pos[2]),
        "final_speed_mps": final_speed,
        "window_sample_count": int(len(win_indices)),
        "window_duration_s": float(ts_bag_raw_s[win_indices[-1]] - ts_bag_raw_s[win_indices[0]]),
    }


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
    landing_window_s: float = LANDING_WINDOW_S,
) -> Dict[str, Any]:
    """
    Ejecuta la validacion de configuracion, grabacion y geometria de mision sobre datos en memoria.
    Si falta evidencia (ej. `pilot_exit_code is None`, `VehicleOdometry` ausente, relojes ausentes)
    o la configuracion es invalida, rechaza el episodio sin fabricar valores por defecto.
    """
    rejection_reasons: List[str] = []
    reader_info: Dict[str, Any] = (
        dict(getattr(telemetry, "reader_info", {}))
        if hasattr(telemetry, "reader_info")
        else dict(telemetry.get("_reader_info", {}))
        if isinstance(telemetry, dict) and isinstance(telemetry.get("_reader_info"), dict)
        else {"backend": "in_memory_dict"}
    )

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
            "reader_info": reader_info,
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

    # Politica de odometria (C6): VehicleOdometry es obligatorio para aceptacion de campana.
    # Si falta VehicleOdometry pero existe VehicleLocalPosition, se rechaza el episodio con
    # `missing_required_vehicle_odometry` y solo se usa `local_pos` con fines de diagnostico.
    using_local_pos = False
    telemetry_source = "vehicle_odometry"
    if not odom_list:
        rejection_reasons.append("missing_required_vehicle_odometry")
        if local_pos_list:
            odom_list = local_pos_list
            using_local_pos = True
            telemetry_source = "vehicle_local_position_diagnostic"
        else:
            telemetry_source = "missing"

    if len(odom_list) < 30:
        rejection_reasons.append(f"insufficient_odometry_messages ({len(odom_list)} < 30)")
        return {
            "pilot_exit_code": pilot_exit_code,
            "pilot_status_source": pilot_status_source,
            "reader_info": reader_info,
            "configuration_status": configuration_status,
            "recording_status": "invalid",
            "mission_status": "invalid",
            "validation_status": "rejected",
            "rejection_reasons": rejection_reasons,
            "primitive_results": [],
            "metrics": {
                "num_odom_messages": len(odom_list),
                "telemetry_source": telemetry_source,
                "estimator_resets_verified": False,
                "in_flight_estimator_resets": None,
            },
        }

    # 1. Extraer series temporales y validar marcos (NED), flags de validez, NaNs y cuaterniones
    # Nunca fabricar timestamps ausentes con `i * 0.01` ni sustituir un reloj ausente por el otro.
    N = len(odom_list)
    pos_enu = np.zeros((N, 3), dtype=np.float64)
    vel_enu = np.zeros((N, 3), dtype=np.float64)
    yaw_enu = np.zeros(N, dtype=np.float64)
    ts_px4_raw_s = np.full(N, np.nan, dtype=np.float64)
    ts_bag_raw_s = np.full(N, np.nan, dtype=np.float64)
    reset_counters = np.zeros(N, dtype=np.int64)
    local_pos_rc_fields = ("xy_reset_counter", "z_reset_counter", "vxy_reset_counter", "vz_reset_counter")
    local_pos_rc_matrix = np.zeros((N, len(local_pos_rc_fields)), dtype=np.int64)
    local_pos_heading_rc = np.zeros(N, dtype=np.int64)
    local_pos_all_rc_present = True
    local_pos_heading_rc_present = True
    nan_count = 0
    invalid_frame_count = 0
    missing_px4_ts_count = 0
    missing_bag_ts_count = 0

    for i, item in enumerate(odom_list):
        # Preferir timestamp_sample_us (instante real de integracion EKF2) sobre timestamp_us
        raw_t_sample = item.get("timestamp_sample_us")
        if _is_finite_number(raw_t_sample) and float(raw_t_sample) > 0.0:
            raw_t_us = float(raw_t_sample)
        else:
            raw_t_pub = item.get("timestamp_us")
            raw_t_us = float(raw_t_pub) if (_is_finite_number(raw_t_pub) and float(raw_t_pub) > 0.0) else np.nan

        if math.isfinite(raw_t_us) and raw_t_us > 0.0:
            ts_px4_raw_s[i] = raw_t_us * 1e-6
        else:
            missing_px4_ts_count += 1

        raw_t_bag = item.get("timestamp")
        if _is_finite_number(raw_t_bag) and float(raw_t_bag) > 0.0:
            ts_bag_raw_s[i] = float(raw_t_bag) * 1e-9
        else:
            missing_bag_ts_count += 1

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

            for col_idx, rc_name in enumerate(local_pos_rc_fields):
                if rc_name in item and _is_finite_number(item[rc_name]):
                    local_pos_rc_matrix[i, col_idx] = int(item[rc_name])
                else:
                    local_pos_all_rc_present = False

            if "heading_reset_counter" in item and _is_finite_number(item["heading_reset_counter"]):
                local_pos_heading_rc[i] = int(item["heading_reset_counter"])
            else:
                local_pos_heading_rc_present = False

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
    if missing_px4_ts_count > 0:
        rejection_reasons.append(f"missing_or_invalid_px4_timestamps ({missing_px4_ts_count})")
    if missing_bag_ts_count > 0:
        rejection_reasons.append(f"missing_or_invalid_bag_timestamps ({missing_bag_ts_count})")

    # 2. Reconstruccion condicionada y verificacion estricta del reloj
    clock_res = reconstruct_simulation_clock(
        ts_px4_raw_s,
        ts_bag_raw_s,
        timesync_status=telemetry.get("timesync_status"),
    )
    ts_sim_s = clock_res["reconstructed_time_s"]
    timesync_offset_jumps = int(clock_res["num_repaired_steps"])
    unexplained_backsteps = int(clock_res["num_backsteps_or_duplicates"])
    ambiguous_clock_steps = int(clock_res["num_ambiguous_steps"])

    seg_lengths = np.linalg.norm(np.diff(pos_enu, axis=0), axis=1)
    max_step_dist_m = float(np.max(seg_lengths)) if len(seg_lengths) > 0 else 0.0

    if ts_sim_s is not None and len(ts_sim_s) > 1:
        dt_sim = np.diff(ts_sim_s)
        pos_dt = dt_sim[dt_sim > 0]
        duration_s = float(ts_sim_s[-1] - ts_sim_s[0])
        dt_median_ms = float(np.median(pos_dt) * 1000.0) if len(pos_dt) > 0 else 0.0
        dt_p95_ms = float(np.percentile(pos_dt, 95) * 1000.0) if len(pos_dt) > 0 else 0.0
        max_dt_gap_s = float(clock_res["max_reconstructed_dt_s"])
        min_dt_s = float(clock_res["min_reconstructed_dt_s"])

        if "ambiguous_clock_step_pattern_b" in clock_res["reasons"]:
            rejection_reasons.append(
                f"ambiguous_clock_step_pattern_b (ambiguous_steps={ambiguous_clock_steps}, max_dt={max_dt_gap_s:.3f}s)"
            )
        if "ambiguous_clock_step_pattern_a" in clock_res["reasons"]:
            rejection_reasons.append(
                f"ambiguous_clock_step_pattern_a (ambiguous_steps={ambiguous_clock_steps}, max_dt={max_dt_gap_s:.3f}s)"
            )
        if "non_monotonic_bag_reception_clock" in clock_res["reasons"]:
            rejection_reasons.append("non_monotonic_bag_reception_clock")

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
    else:
        duration_s = 0.0
        dt_median_ms = 0.0
        dt_p95_ms = 0.0
        max_dt_gap_s = -1.0
        min_dt_s = -1.0

    if max_step_dist_m > MAX_ALLOWED_STEP_DIST_M:
        rejection_reasons.append(
            f"excessive_spatial_step_between_samples (max_step={max_step_dist_m:.3f}m > {MAX_ALLOWED_STEP_DIST_M:.2f}m)"
        )

    # 3. Ventana de vuelo activo y chequeo de resets del estimador EKF2 en vuelo (C6)
    airborne_indices = np.where(pos_enu[:, 2] >= 0.45)[0]
    max_alt_m = float(np.max(pos_enu[:, 2]))
    in_flight_resets: Optional[int] = 0
    estimator_resets_verified: bool = True

    if max_alt_m < TAKEOFF_MIN_ALT_M or len(airborne_indices) < 10:
        rejection_reasons.append(f"takeoff_failed (max_alt={max_alt_m:.2f}m < {TAKEOFF_MIN_ALT_M:.2f}m)")
        if using_local_pos and not local_pos_all_rc_present:
            in_flight_resets = None
            estimator_resets_verified = False
            rejection_reasons.append("unverified_local_position_reset_counters")
    else:
        i_takeoff = int(airborne_indices[0])
        i_touchdown = int(airborne_indices[-1])
        if not using_local_pos:
            rc_flight = reset_counters[i_takeoff:i_touchdown + 1]
            in_flight_resets = int(np.sum(np.diff(rc_flight) != 0)) if len(rc_flight) > 1 else 0
            estimator_resets_verified = True
            if in_flight_resets > 0:
                rejection_reasons.append(f"ekf2_estimator_reset_in_flight (resets={in_flight_resets})")
        else:
            if local_pos_all_rc_present:
                rc_mat_flight = local_pos_rc_matrix[i_takeoff:i_touchdown + 1]
                resets_4 = int(np.sum(np.diff(rc_mat_flight, axis=0) != 0)) if len(rc_mat_flight) > 1 else 0
                resets_hdg = (
                    int(np.sum(np.diff(local_pos_heading_rc[i_takeoff:i_touchdown + 1]) != 0))
                    if (local_pos_heading_rc_present and (i_touchdown > i_takeoff))
                    else 0
                )
                in_flight_resets = resets_4 + resets_hdg
                estimator_resets_verified = True
                if in_flight_resets > 0:
                    rejection_reasons.append(f"ekf2_estimator_reset_in_flight (resets={in_flight_resets})")
            else:
                # Nunca reportar 0 resets si los contadores de local_pos no fueron verificados
                in_flight_resets = None
                estimator_resets_verified = False
                rejection_reasons.append("unverified_local_position_reset_counters")

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
                "missing_required_vehicle_odometry",
                "unverified_local_position_reset_counters",
                "insufficient_",
                "corrupted_",
                "invalid_or_non_ned_",
                "missing_or_invalid_px4_timestamps",
                "missing_or_invalid_bag_timestamps",
                "ambiguous_clock_",
                "flight_duration_",
                "non_monotonic_",
                "telemetry_dropout_",
                "excessive_spatial_step_",
                "ekf2_estimator_reset_",
            ))
            for r in rejection_reasons
        )
        or not bool(clock_res["usable_for_validation"])
        else "valid"
    )

    # 4. Verificar cada primitiva progresivamente y exigir orden temporal estricto (C9)
    landmarks_list = norm_cfg["landmarks"]
    landmarks_map = {str(lm["id"]): lm for lm in landmarks_list}
    primitives_sorted = norm_cfg["primitives"]

    primitive_results: List[Dict[str, Any]] = []
    prev_completion_idx: int = 0
    prev_target_label: Optional[str] = None

    for k, prim in enumerate(primitives_sorted):
        tid = str(prim["target"])
        lm_obj = landmarks_map[tid]

        # Busqueda progresiva a partir de `prev_completion_idx`
        p_res = _verify_single_primitive(
            pos_enu,
            yaw_enu,
            prim,
            lm_obj,
            start_from_index=prev_completion_idx,
            ts_sim_s=ts_sim_s,
        )

        # Comprobar si esta primitiva fue ejecutada prematuramente ANTES de que terminara la anterior
        if k > 0 and prev_completion_idx > 0:
            early_res = _verify_single_primitive(
                pos_enu,
                yaw_enu,
                prim,
                lm_obj,
                start_from_index=0,
                ts_sim_s=ts_sim_s,
            )
            early_start = early_res.get("start_index")
            if early_start is not None and int(early_start) < prev_completion_idx:
                order_msg = (
                    f"primitive_order_violation "
                    f"({prev_target_label}_completion@{prev_completion_idx} > {tid}_start@{early_start})"
                )
                p_res["premature_start_index"] = int(early_start)
                p_res["passed"] = False
                if p_res.get("reason") is None:
                    p_res["reason"] = order_msg
                if order_msg not in rejection_reasons:
                    rejection_reasons.append(order_msg)

        primitive_results.append(p_res)
        if not p_res["passed"] and p_res["reason"]:
            if str(p_res["reason"]) not in rejection_reasons:
                rejection_reasons.append(str(p_res["reason"]))

        if p_res.get("completion_index") is not None:
            prev_completion_idx = int(p_res["completion_index"])
            prev_target_label = tid
        elif prev_target_label is None:
            prev_target_label = tid

    # Verificacion adicional por pares de que start_index[k] >= completion_index[k-1] y event_index creciente
    for k in range(1, len(primitive_results)):
        prev_res = primitive_results[k - 1]
        curr_res = primitive_results[k]
        comp_prev = prev_res.get("completion_index")
        start_curr = curr_res.get("start_index")
        idx_prev = prev_res.get("event_index")
        idx_curr = curr_res.get("event_index")
        if comp_prev is not None and start_curr is not None and int(start_curr) < int(comp_prev):
            msg = (
                f"primitive_order_violation "
                f"({prev_res['target']}_completion@{comp_prev} > {curr_res['target']}_start@{start_curr})"
            )
            if msg not in rejection_reasons:
                rejection_reasons.append(msg)
        elif idx_prev is not None and idx_curr is not None and int(idx_curr) <= int(idx_prev):
            msg = f"primitive_order_violation ({prev_res['target']}@{idx_prev} >= {curr_res['target']}@{idx_curr})"
            if msg not in rejection_reasons:
                rejection_reasons.append(msg)

    # 5. Verificar colision global usando la cota inferior de discretizacion contra CUALQUIER landmark del mundo (C8)
    min_global_sampled_clearance = float("inf")
    min_global_lower_bound_clearance = float("inf")
    for lm in landmarks_list:
        c_lm = compute_landmark_clearance(pos_enu, lm, max_step_m=COLLISION_SEGMENT_MAX_STEP_M)
        d_sampled = float(c_lm["sampled_min_distance_m"])
        d_lower = float(c_lm["lower_bound_distance_m"])
        min_global_sampled_clearance = min(min_global_sampled_clearance, d_sampled)
        min_global_lower_bound_clearance = min(min_global_lower_bound_clearance, d_lower)
        if d_lower < UAV_COLLISION_RADIUS_XY:
            reason_col = (
                f"collision_with_{lm['id']} "
                f"(lower_bound_clearance={d_lower:.4f}m [sampled={d_sampled:.4f}m, h/2={DISCRETIZATION_MARGIN_M:.3f}m] "
                f"< {UAV_COLLISION_RADIUS_XY:.2f}m)"
            )
            if not any(r.startswith(f"collision_with_{lm['id']}") for r in rejection_reasons):
                rejection_reasons.append(reason_col)

    # 6. Verificar aterrizaje final sincronizado en la ventana temporal comun `[t_end - landing_window_s, t_end]`
    landing_cfg = norm_cfg["landing"]
    landing_res = _validate_landing_window(
        pos_enu,
        vel_enu,
        ts_bag_raw_s,
        land_list,
        landing_cfg,
        landing_window_s=landing_window_s,
    )
    rejection_reasons.extend(landing_res["reasons"])
    landed_flag_confirmed = bool(landing_res["landed_detector_confirmed"])
    landing_xy_err = float(landing_res["landing_xy_error_m"])

    # 7. Calcular metricas continuas de calidad del piloto experto
    path_length_m = float(np.sum(seg_lengths))
    speeds = np.linalg.norm(vel_enu, axis=1)
    mean_speed_mps = float(np.mean(speeds))
    max_speed_mps = float(np.max(speeds))

    # Estimar aceleracion y jerk suavizados sobre submuestreo a ~10 Hz para evitar ruido de cuantizacion
    if ts_sim_s is not None and dt_median_ms > 0.0:
        step_10hz = max(1, int(round(100.0 / max(1.0, dt_median_ms))))
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
    else:
        max_acc_mps2 = 0.0
        rms_jerk_mps3 = 0.0

    mission_reasons = [
        r
        for r in rejection_reasons
        if not r.startswith((
            "unknown_pilot_exit_code",
            "invalid_pilot_exit_code",
            "pilot_non_zero_exit_code",
            "pilot_timeout_status",
            "unrecognized_pilot_status",
            "missing_required_vehicle_odometry",
            "unverified_local_position_reset_counters",
            "insufficient_odometry_messages",
            "corrupted_",
            "invalid_or_non_ned_",
            "missing_or_invalid_px4_timestamps",
            "missing_or_invalid_bag_timestamps",
            "ambiguous_clock_",
            "flight_duration_",
            "non_monotonic_",
            "telemetry_dropout_",
            "excessive_spatial_step_",
            "ekf2_estimator_reset_",
        ))
    ]
    mission_status = "valid" if len(mission_reasons) == 0 else "invalid"
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
        "telemetry_source": telemetry_source,
        "num_odom_messages": int(N),
        "duration_s": round(duration_s, 3),
        "dt_median_ms": round(dt_median_ms, 3),
        "dt_p95_ms": round(dt_p95_ms, 3),
        "max_dt_gap_s": round(max_dt_gap_s, 4),
        "max_step_dist_m": round(max_step_dist_m, 4),
        "timesync_offset_jumps": int(timesync_offset_jumps),
        "unexplained_clock_backsteps": int(unexplained_backsteps),
        "ambiguous_clock_steps": int(ambiguous_clock_steps),
        "clock_usable_for_validation": bool(clock_res["usable_for_validation"]),
        "estimator_resets_verified": bool(estimator_resets_verified),
        "in_flight_estimator_resets": (
            int(in_flight_resets) if in_flight_resets is not None else None
        ),
        "path_length_m": round(path_length_m, 3),
        "max_altitude_m": round(max_alt_m, 3),
        "mean_speed_mps": round(mean_speed_mps, 3),
        "max_speed_mps": round(max_speed_mps, 3),
        "p98_acc_mps2": round(max_acc_mps2, 3),
        "rms_jerk_mps3": round(rms_jerk_mps3, 3),
        "min_obstacle_clearance_sampled_m": round(
            min_global_sampled_clearance if math.isfinite(min_global_sampled_clearance) else -1.0, 4
        ),
        "discretization_step_h_m": round(COLLISION_SEGMENT_MAX_STEP_M, 4),
        "discretization_margin_m": round(DISCRETIZATION_MARGIN_M, 4),
        "min_obstacle_clearance_m": round(
            min_global_lower_bound_clearance if math.isfinite(min_global_lower_bound_clearance) else -1.0, 4
        ),
        "landing_xy_error_m": round(landing_xy_err, 4),
        "final_altitude_m": round(float(landing_res["final_altitude_m"]), 4),
        "final_speed_mps": round(float(landing_res["final_speed_mps"]), 4),
        "landed_detector_confirmed": bool(landed_flag_confirmed),
        "landing_window_s": round(float(landing_window_s), 3),
        "landing_window_sample_count": int(landing_res["window_sample_count"]),
        "landing_window_duration_s": round(float(landing_res["window_duration_s"]), 4),
    }

    return {
        "pilot_exit_code": pilot_exit_code,
        "pilot_status_source": pilot_status_source,
        "reader_info": reader_info,
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
    backend: str = "auto",
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
            "reader_info": {"backend": "none"},
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
            telemetry = load_episode_telemetry(ep_dir, backend=backend)
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
                "reader_info": {"backend": "error", "error": f"{type(e).__name__}: {e}"},
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
        existing_meta["reader_info"] = report.get("reader_info", {})
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
    parser.add_argument(
        "--backend",
        choices=["auto", "rosbags", "native"],
        default="auto",
        help="Backend de lectura de telemetria MCAP ('auto', 'rosbags', o 'native' para diagnostico).",
    )
    args = parser.parse_args(argv)

    if args.episode_dir is not None:
        ep_path = Path(args.episode_dir)
        if not ep_path.exists() or not ep_path.is_dir():
            print(f"Error: el directorio de episodio no existe: {args.episode_dir}", file=sys.stderr)
            return 2

        rep = validate_episode_directory(ep_path, update_meta=args.update_meta, backend=args.backend)
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
        rep = validate_episode_directory(ep, update_meta=args.update_meta, backend=args.backend)
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

