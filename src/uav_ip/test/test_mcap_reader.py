#!/usr/bin/env python3
"""
Pruebas modulares del lector MCAP, verificacion de integridad/esquema y politica de odometria (Bloque 3: C5 + C6).
"""

import struct
import types
import zlib
from pathlib import Path
from typing import Any, Dict, List

import numpy as np
import pytest

from uav_ip.synthetic_lidar import (
    ned_to_gazebo_enu as gazebo_enu_to_ned,
    yaw_ned_to_gazebo_enu as yaw_gazebo_enu_to_ned,
)
from uav_ip.validate_episode import (
    MCAP_MAGIC,
    _parse_mcap_native,
    load_episode_telemetry,
    validate_episode_data,
    validate_episode_directory,
    verify_mcap_integrity,
)


def _mcap_str(s: str) -> bytes:
    b = s.encode("utf-8")
    return struct.pack("<I", len(b)) + b


def _mcap_bytes(data: bytes) -> bytes:
    return struct.pack("<I", len(data)) + data


def _mcap_record(op: int, body: bytes) -> bytes:
    return struct.pack("<BQ", op, len(body)) + body


def _pack_odom_cdr(
    ts_us: int,
    pos_ned: List[float],
    q_wxyz: List[float],
    vel_ned: List[float],
    *,
    pose_frame: int = 1,
    vel_frame: int = 1,
    reset_counter: int = 1,
) -> bytes:
    buf = bytearray(120)
    buf[0] = 0x00
    buf[1] = 0x01
    struct.pack_into("<QQ", buf, 4, int(ts_us), int(ts_us))
    buf[20] = int(pose_frame)
    struct.pack_into("<fff", buf, 24, float(pos_ned[0]), float(pos_ned[1]), float(pos_ned[2]))
    struct.pack_into("<ffff", buf, 36, float(q_wxyz[0]), float(q_wxyz[1]), float(q_wxyz[2]), float(q_wxyz[3]))
    buf[52] = int(vel_frame)
    struct.pack_into("<fff", buf, 56, float(vel_ned[0]), float(vel_ned[1]), float(vel_ned[2]))
    struct.pack_into("<fff", buf, 68, 0.0, 0.0, 0.0)
    buf[116] = int(reset_counter)
    struct.pack_into("<b", buf, 117, 100)
    return bytes(buf)


def _pack_local_pos_cdr(
    ts_us: int,
    pos_ned: List[float],
    vel_ned: List[float],
    heading: float,
    *,
    xy_rc: int = 1,
    z_rc: int = 1,
    vxy_rc: int = 1,
    vz_rc: int = 1,
    hdg_rc: int = 1,
) -> bytes:
    buf = bytearray(124)
    buf[0] = 0x00
    buf[1] = 0x01
    struct.pack_into("<QQ", buf, 4, int(ts_us), int(ts_us))
    buf[20] = 1
    buf[21] = 1
    buf[22] = 1
    buf[23] = 1
    struct.pack_into("<fff", buf, 24, float(pos_ned[0]), float(pos_ned[1]), float(pos_ned[2]))
    buf[44] = int(xy_rc)
    buf[52] = int(z_rc)
    struct.pack_into("<fff", buf, 56, float(vel_ned[0]), float(vel_ned[1]), float(vel_ned[2]))
    buf[80] = int(vxy_rc)
    buf[88] = int(vz_rc)
    struct.pack_into("<f", buf, 104, float(heading))
    buf[120] = int(hdg_rc)
    return bytes(buf)


def _pack_land_detected_cdr(ts_us: int, *, landed: bool, ground_contact: bool = False) -> bytes:
    buf = bytearray(24)
    buf[0] = 0x00
    buf[1] = 0x01
    struct.pack_into("<Q", buf, 4, int(ts_us))
    buf[12] = 0
    buf[13] = 1 if ground_contact else 0
    buf[14] = 1 if landed else 0
    buf[15] = 1 if landed else 0
    buf[23] = 1 if landed else 0
    return bytes(buf)


def _build_valid_mcap_with_chunk(
    *,
    odom_schema_name: str = "px4_msgs/msg/VehicleOdometry",
    odom_schema_data: bytes = (
        b"uint64 timestamp\nuint64 timestamp_sample\nuint8 pose_frame\n"
        b"float32[3] position\nfloat32[4] q\nuint8 velocity_frame\n"
        b"float32[3] velocity\nuint8 reset_counter\n"
    ),
    corrupt_chunk_crc: bool = False,
) -> bytes:
    header_rec = _mcap_record(0x01, _mcap_str("ros2") + _mcap_str("uav_ip_test"))
    schema_odom = _mcap_record(
        0x03,
        struct.pack("<H", 1) + _mcap_str(odom_schema_name) + _mcap_str("ros2msg") + _mcap_bytes(odom_schema_data),
    )
    schema_lp = _mcap_record(
        0x03,
        struct.pack("<H", 2)
        + _mcap_str("px4_msgs/msg/VehicleLocalPosition")
        + _mcap_str("ros2msg")
        + _mcap_bytes(
            b"uint64 timestamp\nuint64 timestamp_sample\nbool xy_valid\nbool z_valid\n"
            b"float32 x\nfloat32 y\nfloat32 z\nuint8 xy_reset_counter\nuint8 z_reset_counter\n"
            b"uint8 vxy_reset_counter\nuint8 vz_reset_counter\nfloat32 heading\n"
        ),
    )
    schema_ld = _mcap_record(
        0x03,
        struct.pack("<H", 3)
        + _mcap_str("px4_msgs/msg/VehicleLandDetected")
        + _mcap_str("ros2msg")
        + _mcap_bytes(
            b"uint64 timestamp\nbool freefall\nbool ground_contact\nbool maybe_landed\nbool landed\nbool at_rest\n"
        ),
    )

    chan_odom = _mcap_record(0x04, struct.pack("<HH", 1, 1) + _mcap_str("/fmu/out/vehicle_odometry") + _mcap_str("cdr") + struct.pack("<I", 0))
    chan_lp = _mcap_record(0x04, struct.pack("<HH", 2, 2) + _mcap_str("/fmu/out/vehicle_local_position_v1") + _mcap_str("cdr") + struct.pack("<I", 0))
    chan_ld = _mcap_record(0x04, struct.pack("<HH", 3, 3) + _mcap_str("/fmu/out/vehicle_land_detected") + _mcap_str("cdr") + struct.pack("<I", 0))

    raw_odom = _pack_odom_cdr(1_000_000, [0.0, 4.0, -1.5], [1.0, 0.0, 0.0, 0.0], [0.0, 1.0, 0.0])
    raw_lp = _pack_local_pos_cdr(1_000_000, [0.0, 4.0, -1.5], [0.0, 1.0, 0.0], 1.57, xy_rc=2, z_rc=3, vxy_rc=4, vz_rc=5, hdg_rc=6)
    raw_ld = _pack_land_detected_cdr(1_000_000, landed=True)

    msg_odom = _mcap_record(0x05, struct.pack("<HIQQ", 1, 0, 1_000_000_000, 1_000_000_000) + raw_odom)
    msg_lp = _mcap_record(0x05, struct.pack("<HIQQ", 2, 0, 1_000_000_000, 1_000_000_000) + raw_lp)
    msg_ld = _mcap_record(0x05, struct.pack("<HIQQ", 3, 0, 1_000_000_000, 1_000_000_000) + raw_ld)

    chunk_inner = schema_odom + schema_lp + schema_ld + chan_odom + chan_lp + chan_ld + msg_odom + msg_lp + msg_ld
    crc = zlib.crc32(chunk_inner) & 0xFFFFFFFF
    if corrupt_chunk_crc:
        crc ^= 0xDEADBEEF
    chunk_body = (
        struct.pack("<QQQI", 1_000_000_000, 1_000_000_000, len(chunk_inner), crc)
        + _mcap_str("")
        + struct.pack("<Q", len(chunk_inner))
        + chunk_inner
    )
    chunk_rec = _mcap_record(0x06, chunk_body)
    footer_rec = _mcap_record(0x02, struct.pack("<QQI", 0, 0, 0))
    return MCAP_MAGIC + header_rec + chunk_rec + footer_rec + MCAP_MAGIC


def test_mcap_integrity_rejects_truncation_and_midfile_chunk_crc_corruption(tmp_path: Path):
    """
    C5: `verify_mcap_integrity()` debe aceptar un MCAP íntegro con chunk CRC32 válido y rechazar:
      - Archivos truncados sin trailer magic.
      - Archivos con corrupción de un byte en mitad de un chunk (CRC32 inválido) aunque el Footer esté intacto.
      - Archivos con un registro truncado a mitad de archivo aunque se les añada Footer + magic final.
    """
    valid_bytes = _build_valid_mcap_with_chunk()
    p_ok = tmp_path / "valid.mcap"
    p_ok.write_bytes(valid_bytes)
    assert verify_mcap_integrity(p_ok) == valid_bytes

    # 1. Truncado al final
    p_trunc = tmp_path / "truncated.mcap"
    p_trunc.write_bytes(valid_bytes[:-12])
    with pytest.raises(ValueError, match="truncated_mcap_missing_trailer_magic"):
        verify_mcap_integrity(p_trunc)

    # 2. Corrupcion de CRC32 en un chunk intermedio con Footer intacto
    bad_crc_bytes = _build_valid_mcap_with_chunk(corrupt_chunk_crc=True)
    p_crc = tmp_path / "bad_crc.mcap"
    p_crc.write_bytes(bad_crc_bytes)
    with pytest.raises(ValueError, match="corrupted_mcap_chunk_crc32"):
        verify_mcap_integrity(p_crc)

    # 3. Registro truncado en medio del flujo pero con Footer + magic añadidos al final
    footer_and_magic = _mcap_record(0x02, struct.pack("<QQI", 0, 0, 0)) + MCAP_MAGIC
    # Un registro message que declara longitud 500 pero el archivo termina antes del footer
    fake_mid = MCAP_MAGIC + struct.pack("<BQ", 0x05, 500) + b"\x00" * 20 + footer_and_magic
    p_mid = tmp_path / "mid_trunc.mcap"
    p_mid.write_bytes(fake_mid)
    with pytest.raises(ValueError, match="truncated_mcap_record_body"):
        verify_mcap_integrity(p_mid)


def test_native_mcap_parser_schema_verification_and_local_pos_reset_counters(tmp_path: Path):
    """
    C5 + C6: `_parse_mcap_native()` debe:
      - Rechazar esquemas con nombre incorrecto (`schema_mismatch_for_...`).
      - Rechazar definiciones de esquema incompatibles cuando faltan campos requeridos (`incompatible_schema_definition`).
      - Extraer todos los contadores de reset de `VehicleLocalPosition` (`xy_reset_counter`, `z_reset_counter`,
        `vxy_reset_counter`, `vz_reset_counter`, `heading_reset_counter`).
    """
    p_ok = tmp_path / "ok.mcap"
    p_ok.write_bytes(_build_valid_mcap_with_chunk())
    res = _parse_mcap_native(p_ok)
    assert res is not None
    tel, schemas = res
    assert "px4_msgs/msg/VehicleOdometry" in schemas
    assert len(tel["local_pos"]) == 1
    lp = tel["local_pos"][0]
    assert lp["xy_reset_counter"] == 2
    assert lp["z_reset_counter"] == 3
    assert lp["vxy_reset_counter"] == 4
    assert lp["vz_reset_counter"] == 5
    assert lp["heading_reset_counter"] == 6

    # Esquema con nombre incompatible
    p_bad_name = tmp_path / "bad_schema_name.mcap"
    p_bad_name.write_bytes(_build_valid_mcap_with_chunk(odom_schema_name="px4_msgs/msg/WrongOdometry"))
    with pytest.raises(ValueError, match="schema_mismatch_for_/fmu/out/vehicle_odometry"):
        _parse_mcap_native(p_bad_name)

    # Esquema con texto de definicion incompatible (sin float32[3] position)
    p_bad_def = tmp_path / "bad_schema_def.mcap"
    p_bad_def.write_bytes(_build_valid_mcap_with_chunk(odom_schema_data=b"uint64 timestamp\nfloat64[3] wrong_pos\n"))
    with pytest.raises(ValueError, match="incompatible_schema_definition"):
        _parse_mcap_native(p_bad_def)


def test_load_episode_telemetry_never_silently_falls_back_when_rosbags_fails(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    """
    C5: Si `rosbags` está disponible y lanza un error de deserialización, `load_episode_telemetry()`
    NUNCA debe capturar el error para caer silenciosamente a `_parse_mcap_native`; debe propagar
    `ValueError('rosbags_deserialization_error ...')`.
    Y si se solicita `backend='rosbags'` sin `rosbags` instalado, debe lanzar `RuntimeError('missing_reader_dependency:rosbags')`.
    """
    p_ok = tmp_path / "ok.mcap"
    p_ok.write_bytes(_build_valid_mcap_with_chunk())

    # 1. Simular que rosbags.highlevel está instalado pero _parse_with_rosbags falla
    fake_rb = types.ModuleType("rosbags")
    fake_rb_hl = types.ModuleType("rosbags.highlevel")
    monkeypatch.setitem( __import__("sys").modules, "rosbags", fake_rb)
    monkeypatch.setitem( __import__("sys").modules, "rosbags.highlevel", fake_rb_hl)
    monkeypatch.setattr(
        "uav_ip.validate_episode._parse_with_rosbags",
        lambda mcaps: (_ for _ in ()).throw(RuntimeError("simulated_cdr_corruption_in_rosbags")),
    )

    with pytest.raises(ValueError, match="rosbags_deserialization_error"):
        load_episode_telemetry(p_ok, backend="auto")

    # 2. Simular que rosbags NO está instalado y se pide backend="rosbags" explícitamente
    monkeypatch.delitem(__import__("sys").modules, "rosbags", raising=False)
    monkeypatch.delitem(__import__("sys").modules, "rosbags.highlevel", raising=False)
    orig_import = __import__("builtins").__import__

    def _no_rosbags_import(name: str, *args: Any, **kwargs: Any) -> Any:
        if name == "rosbags" or name.startswith("rosbags."):
            raise ImportError("No module named 'rosbags'")
        return orig_import(name, *args, **kwargs)

    monkeypatch.setattr("builtins.__import__", _no_rosbags_import)
    with pytest.raises(RuntimeError, match="missing_reader_dependency:rosbags"):
        load_episode_telemetry(p_ok, backend="rosbags")

    # 3. Con backend="native", carga y expone reader_info completo
    bundle = load_episode_telemetry(p_ok, backend="native")
    assert bundle.reader_info["backend"] == "native_cdr"
    assert "uncompressed_chunk_crc32" in bundle.reader_info["integrity_checks"]
    assert "px4_msgs/msg/VehicleOdometry" in bundle.reader_info["schemas_verified"]


def test_real_rosbag2_serialized_mcap_fixture_parsed_accurately():
    """
    C5: Valida que `_parse_mcap_native()` lee y verifica un archivo `.mcap` real serializado
    por `rosbag2_storage_mcap` en el dataset del repositorio (`datasets/task_0001/episode_00/rosbag/rosbag_0.mcap`),
    confirmando compatibilidad con payloads XCDR1 reales producidos por un serializador externo mantenido.
    """
    repo_root = Path(__file__).resolve().parents[3]
    real_mcap = repo_root / "datasets" / "task_0001" / "episode_00" / "rosbag" / "rosbag_0.mcap"
    if not real_mcap.is_file():
        pytest.skip("No se encontró datasets/task_0001/episode_00/rosbag/rosbag_0.mcap")

    bundle = load_episode_telemetry(real_mcap, backend="native")
    assert bundle.reader_info["backend"] == "native_cdr"
    assert len(bundle["odometry"]) > 100
    assert len(bundle["local_pos"]) > 100
    assert len(bundle["land_detected"]) > 5
    # Verificar que los contadores de reset de local_pos fueron extraídos del binario real
    lp0 = bundle["local_pos"][0]
    for k in ("xy_reset_counter", "z_reset_counter", "vxy_reset_counter", "vz_reset_counter", "heading_reset_counter"):
        assert k in lp0


def test_odometry_policy_rejects_local_pos_only_and_never_fabricates_zero_resets():
    """
    C6:
      1. Un episodio sin `VehicleOdometry` (solo con `VehicleLocalPosition`) es RECHAZADO siempre
         para campaña con `missing_required_vehicle_odometry` y `recording_status == 'invalid'`.
      2. Si `local_pos` no trae sus contadores de reset, `in_flight_estimator_resets` es `None`
         (nunca `0`), `estimator_resets_verified` es `False` y se añade `unverified_local_position_reset_counters`.
      3. Si `local_pos` trae sus contadores de reset y ocurre un salto en cualquiera de ellos
         (`xy_reset_counter`, `z_reset_counter`, `vxy_reset_counter`, `vz_reset_counter`, `heading_reset_counter`),
         se detecta y rechaza con `ekf2_estimator_reset_in_flight`.
    """
    waypoints_cfg = {
        "landmarks": [{"id": "gate_1", "type": "square_gate", "x": 4.0, "y": 0.0, "z": 1.5, "yaw": 0.0}],
        "primitives": [{"type": "traverse", "target": "gate_1"}],
        "landing": {"x": 8.0, "y": 0.0, "z": 0.05},
    }

    N = 120
    dt_s = 0.05
    t0_s = 1_700_000_000.0
    local_pos_without_rc: List[Dict[str, Any]] = []
    local_pos_with_reset: List[Dict[str, Any]] = []
    land_msgs: List[Dict[str, Any]] = []

    for i in range(N):
        t_s = t0_s + i * dt_s
        ts_us = int(round(t_s * 1e6))
        ts_ns = int(round(t_s * 1e9))
        u = min(1.0, i / 80.0)
        x_e = 8.0 * u
        z_e = 1.5 if i < 80 else (0.05 if i >= 95 else 1.5 * (95 - i) / 15.0)
        vx_e = 1.0 if i < 80 else 0.0
        p_ned = gazebo_enu_to_ned([x_e, 0.0, z_e])
        v_ned = gazebo_enu_to_ned([vx_e, 0.0, 0.0])
        h_ned = float(yaw_gazebo_enu_to_ned(0.0))

        base_item = {
            "timestamp": ts_ns,
            "timestamp_us": ts_us,
            "timestamp_sample_us": ts_us,
            "x": float(p_ned[0]),
            "y": float(p_ned[1]),
            "z": float(p_ned[2]),
            "vx": float(v_ned[0]),
            "vy": float(v_ned[1]),
            "vz": float(v_ned[2]),
            "heading": h_ned,
            "xy_valid": True,
            "z_valid": True,
            "v_xy_valid": True,
            "v_z_valid": True,
        }
        local_pos_without_rc.append(dict(base_item))

        with_rc = dict(base_item)
        with_rc["xy_reset_counter"] = 1
        with_rc["z_reset_counter"] = 1
        with_rc["vxy_reset_counter"] = 1 if i < 40 else 2  # Salto en vxy_reset_counter a mitad de vuelo
        with_rc["vz_reset_counter"] = 1
        with_rc["heading_reset_counter"] = 1
        local_pos_with_reset.append(with_rc)

        if i % 5 == 0 or i == N - 1:
            is_landed = i >= 95
            land_msgs.append(
                {
                    "timestamp": ts_ns,
                    "timestamp_us": ts_us,
                    "freefall": False,
                    "ground_contact": is_landed,
                    "maybe_landed": is_landed,
                    "landed": is_landed,
                    "at_rest": is_landed,
                }
            )

    # Caso A: solo local_pos sin contadores de reset
    rep_no_rc = validate_episode_data(
        {"odometry": [], "local_pos": local_pos_without_rc, "land_detected": land_msgs},
        waypoints_cfg,
        pilot_exit_code=0,
    )
    assert rep_no_rc["validation_status"] == "rejected"
    assert rep_no_rc["recording_status"] == "invalid"
    assert "missing_required_vehicle_odometry" in rep_no_rc["rejection_reasons"]
    assert "unverified_local_position_reset_counters" in rep_no_rc["rejection_reasons"]
    assert rep_no_rc["metrics"]["estimator_resets_verified"] is False
    assert rep_no_rc["metrics"]["in_flight_estimator_resets"] is None

    # Caso B: solo local_pos con salto en vxy_reset_counter
    rep_with_rc = validate_episode_data(
        {"odometry": [], "local_pos": local_pos_with_reset, "land_detected": land_msgs},
        waypoints_cfg,
        pilot_exit_code=0,
    )
    assert rep_with_rc["validation_status"] == "rejected"
    assert "missing_required_vehicle_odometry" in rep_with_rc["rejection_reasons"]
    assert any(r.startswith("ekf2_estimator_reset_in_flight") for r in rep_with_rc["rejection_reasons"])
    assert rep_with_rc["metrics"]["estimator_resets_verified"] is True
    assert rep_with_rc["metrics"]["in_flight_estimator_resets"] == 1


def test_native_mcap_parser_deserializes_timesync_status(tmp_path: Path):
    """
    Fase 2 (Tarea 2.5.4): Verifica que `_parse_mcap_native` y `verify_mcap_integrity`
    reconocen el esquema `px4_msgs/msg/TimesyncStatus` en `/fmu/out/timesync_status`
    y deserializan correctamente `timestamp`, `estimated_offset`, `observed_offset` y `round_trip_time`.
    """
    buf = bytearray(48)
    buf[0] = 0x00
    buf[1] = 0x01
    ts_us = 1_790_000_000_000_000
    remote_ts_us = 15_000_000
    observed_offset_us = -350_000
    estimated_offset_us = -350_000
    rtt_us = 1200
    struct.pack_into("<Q", buf, 4, ts_us)
    buf[12] = 2  # source_protocol
    struct.pack_into("<QqqI", buf, 20, remote_ts_us, observed_offset_us, estimated_offset_us, rtt_us)

    header_rec = _mcap_record(0x01, _mcap_str("ros2") + _mcap_str("uav_ip_test"))
    schema_ts = _mcap_record(
        0x03,
        struct.pack("<H", 1)
        + _mcap_str("px4_msgs/msg/TimesyncStatus")
        + _mcap_str("ros2msg")
        + _mcap_bytes(
            b"uint64 timestamp\nuint8 source_protocol\nuint64 remote_timestamp\n"
            b"int64 observed_offset\nint64 estimated_offset\nuint32 round_trip_time\n"
        ),
    )
    chan_ts = _mcap_record(
        0x04,
        struct.pack("<HH", 1, 1) + _mcap_str("/fmu/out/timesync_status") + _mcap_str("cdr") + struct.pack("<I", 0),
    )
    log_ns = ts_us * 1000
    msg_ts = _mcap_record(0x05, struct.pack("<HIQQ", 1, 0, log_ns, log_ns) + bytes(buf))
    footer_rec = _mcap_record(0x02, struct.pack("<QQI", 0, 0, 0))

    raw_mcap = MCAP_MAGIC + header_rec + schema_ts + chan_ts + msg_ts + footer_rec + MCAP_MAGIC
    mcap_path = tmp_path / "timesync_test.mcap"
    mcap_path.write_bytes(raw_mcap)

    assert verify_mcap_integrity(mcap_path) == raw_mcap

    res = _parse_mcap_native(mcap_path)
    assert res is not None
    parsed, verified_schemas = res
    assert "px4_msgs/msg/TimesyncStatus" in verified_schemas
    assert len(parsed["timesync_status"]) == 1
    item = parsed["timesync_status"][0]
    assert item["timestamp_us"] == ts_us
    assert item["remote_timestamp_us"] == remote_ts_us
    assert item["observed_offset_us"] == observed_offset_us
    assert item["estimated_offset_us"] == estimated_offset_us
    assert item["round_trip_time_us"] == rtt_us

