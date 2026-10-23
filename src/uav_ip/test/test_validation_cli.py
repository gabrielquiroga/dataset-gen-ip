import json
import math
import struct
import subprocess
import sys
from pathlib import Path

import numpy as np
import yaml

REPO_ROOT = Path(__file__).resolve().parent.parent.parent.parent
PKG_ROOT = Path(__file__).resolve().parent.parent
if str(PKG_ROOT) not in sys.path:
    sys.path.insert(0, str(PKG_ROOT))

WRAPPER_SCRIPT = PKG_ROOT / "scripts" / "validate_episode.py"


def _make_mcap_record(opcode: int, payload: bytes) -> bytes:
    return bytes([opcode]) + struct.pack("<Q", len(payload)) + payload


def _make_mcap_str(s: str) -> bytes:
    b = s.encode("utf-8")
    return struct.pack("<I", len(b)) + b


def _write_synthetic_valid_mcap(mcap_path: Path, dt_s: float = 0.05) -> None:
    """
    Escribe un archivo `.mcap` autocontenido con una trayectoria valida que atraviesa
    un `square_gate` en X=5.0 y aterriza en reposo sobre la plataforma en X=10.0.
    """
    from uav_ip.validate_episode import MCAP_MAGIC

    n_flight = 140
    n_rest = 30
    x_flight = np.linspace(0.0, 10.0, n_flight)
    y_flight = np.zeros(n_flight)
    z_flight = np.full(n_flight, 1.5)
    z_flight[:15] = np.linspace(0.15, 1.5, 15)
    z_flight[-25:] = np.linspace(1.5, 0.08, 25)

    x_arr = np.concatenate([x_flight, np.full(n_rest, 10.0)])
    y_arr = np.concatenate([y_flight, np.zeros(n_rest)])
    z_arr = np.concatenate([z_flight, np.full(n_rest, 0.08)])
    pos_enu = np.column_stack([x_arr, y_arr, z_arr])
    n_total = len(pos_enu)

    vel_enu = np.zeros_like(pos_enu)
    vel_enu[1:] = np.diff(pos_enu, axis=0) / dt_s
    vel_enu[0] = vel_enu[1]
    vel_enu[n_flight:] = 0.0

    header_rec = _make_mcap_record(0x01, _make_mcap_str("ros2") + _make_mcap_str("test"))

    def make_schema(sid: int, name: str) -> bytes:
        payload = (
            struct.pack("<H", sid)
            + _make_mcap_str(name)
            + _make_mcap_str("ros2msg")
            + struct.pack("<I", 0)
        )
        return _make_mcap_record(0x03, payload)

    def make_channel(cid: int, sid: int, topic: str) -> bytes:
        payload = (
            struct.pack("<HH", cid, sid)
            + _make_mcap_str(topic)
            + _make_mcap_str("cdr")
            + struct.pack("<I", 0)
        )
        return _make_mcap_record(0x04, payload)

    def make_msg(cid: int, seq: int, log_time: int, cdr_data: bytes) -> bytes:
        hdr = struct.pack("<HIQQ", cid, seq, log_time, log_time)
        return _make_mcap_record(0x05, hdr + cdr_data)

    schemas = (
        make_schema(1, "px4_msgs/msg/VehicleOdometry")
        + make_schema(2, "px4_msgs/msg/VehicleLandDetected")
        + make_schema(3, "px4_msgs/msg/VehicleStatus")
    )
    channels = (
        make_channel(1, 1, "/fmu/out/vehicle_odometry")
        + make_channel(2, 2, "/fmu/out/vehicle_land_detected")
        + make_channel(3, 3, "/fmu/out/vehicle_status_v4")
    )

    msg_chunks = []
    yaw_ned = math.pi / 2.0  # yaw_enu = 0.0 -> yaw_ned = pi/2
    qw = math.cos(yaw_ned * 0.5)
    qz = math.sin(yaw_ned * 0.5)

    for i in range(n_total):
        t_us = int(round((100.0 + i * dt_s) * 1e6))
        t_ns = t_us * 1000
        # ENU -> NED: [y, x, -z]
        px_ned, py_ned, pz_ned = float(pos_enu[i, 1]), float(pos_enu[i, 0]), float(-pos_enu[i, 2])
        vx_ned, vy_ned, vz_ned = float(vel_enu[i, 1]), float(vel_enu[i, 0]), float(-vel_enu[i, 2])

        odom_buf = bytearray(120)
        odom_buf[0:4] = b"\x00\x01\x00\x00"
        struct.pack_into("<QQ", odom_buf, 4, t_us, t_us)
        odom_buf[20] = 1  # pose_frame = NED
        struct.pack_into("<fff", odom_buf, 24, px_ned, py_ned, pz_ned)
        struct.pack_into("<ffff", odom_buf, 36, qw, 0.0, 0.0, qz)
        odom_buf[52] = 1  # velocity_frame = NED
        struct.pack_into("<fff", odom_buf, 56, vx_ned, vy_ned, vz_ned)
        struct.pack_into("<fff", odom_buf, 68, 0.0, 0.0, 0.0)
        odom_buf[116] = 0  # reset_counter
        struct.pack_into("<b", odom_buf, 117, 100)
        msg_chunks.append(make_msg(1, i + 1, t_ns, bytes(odom_buf)))

        is_landed = 1 if i >= n_flight else 0
        land_buf = bytearray(28)
        land_buf[0:4] = b"\x00\x01\x00\x00"
        struct.pack_into("<Q", land_buf, 4, t_us)
        land_buf[12] = 0
        land_buf[13] = is_landed
        land_buf[14] = is_landed
        land_buf[15] = is_landed
        land_buf[23] = is_landed
        msg_chunks.append(make_msg(2, i + 1, t_ns, bytes(land_buf)))

    last_t_us = int(round((100.0 + (n_total - 1) * dt_s) * 1e6))
    stat_buf = bytearray(92)
    stat_buf[0:4] = b"\x00\x01\x00\x00"
    struct.pack_into("<Q", stat_buf, 4, last_t_us)
    stat_buf[28] = 2
    stat_buf[45] = 14
    stat_buf[62] = 0
    msg_chunks.append(make_msg(3, 1, last_t_us * 1000, bytes(stat_buf)))

    footer_rec = _make_mcap_record(0x02, struct.pack("<QQI", 0, 0, 0))
    full_mcap = (
        MCAP_MAGIC
        + header_rec
        + schemas
        + channels
        + b"".join(msg_chunks)
        + footer_rec
        + MCAP_MAGIC
    )
    mcap_path.parent.mkdir(parents=True, exist_ok=True)
    mcap_path.write_bytes(full_mcap)


def _create_episode_fixture(ep_dir: Path, *, with_bag: bool = True, pilot_exit_code: int = 0) -> Path:
    ep_dir.mkdir(parents=True, exist_ok=True)
    waypoints_cfg = {
        "start_pose": {"x": 0.0, "y": 0.0, "z": 1.5, "yaw": 0.0},
        "landmarks": [
            {"id": "landmark_1", "type": "square_gate", "x": 5.0, "y": 0.0, "z": 1.5, "yaw": 0.0}
        ],
        "primitives": [{"type": "traverse", "target": "landmark_1"}],
        "landing": {"x": 10.0, "y": 0.0, "z": 0.05},
    }
    (ep_dir / "waypoints.yaml").write_text(yaml.safe_dump(waypoints_cfg), encoding="utf-8")
    meta = {
        "task_id": ep_dir.parent.name,
        "episode_id": ep_dir.name,
        "status": "success" if pilot_exit_code == 0 else f"pilot_exit_{pilot_exit_code}",
        "pilot_exit_code": pilot_exit_code,
    }
    (ep_dir / "episode_meta.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
    if with_bag:
        _write_synthetic_valid_mcap(ep_dir / "rosbag" / "rosbag_0.mcap")
    else:
        (ep_dir / "rosbag").mkdir(parents=True, exist_ok=True)
    return ep_dir


def test_cli_wrapper_propagates_exit_code_1_on_rejected_episode(tmp_path):
    """C1: El wrapper real termina con codigo 1 ante un episodio rechazado."""
    ep_dir = _create_episode_fixture(
        tmp_path / "task_0000" / "episode_00",
        with_bag=True,
        pilot_exit_code=1,
    )
    res = subprocess.run(
        [sys.executable, str(WRAPPER_SCRIPT), "--episode-dir", str(ep_dir), "--json"],
        capture_output=True,
        text=True,
        check=False,
    )
    assert res.returncode == 1, f"Expected exit code 1, got {res.returncode}. Output: {res.stdout}"
    payload = json.loads(res.stdout)
    assert payload["validation_status"] == "rejected"
    assert any("pilot_non_zero_exit_code" in r for r in payload["rejection_reasons"])


def test_cli_wrapper_returns_0_on_accepted_episode(tmp_path):
    """C1: Episodio individual aceptado termina con codigo 0 en el wrapper real."""
    ep_dir = _create_episode_fixture(
        tmp_path / "task_0000" / "episode_00",
        with_bag=True,
        pilot_exit_code=0,
    )
    res = subprocess.run(
        [sys.executable, str(WRAPPER_SCRIPT), "--episode-dir", str(ep_dir), "--json"],
        capture_output=True,
        text=True,
        check=False,
    )
    assert res.returncode == 0, f"Expected exit code 0, got {res.returncode}. Stdout: {res.stdout} Stderr: {res.stderr}"
    payload = json.loads(res.stdout)
    assert payload["validation_status"] == "accepted"
    assert payload["configuration_status"] == "valid"


def test_cli_nonexistent_paths_return_code_2(tmp_path):
    """C1: Un directorio inexistente termina con codigo 2."""
    missing_ep = tmp_path / "does_not_exist_ep"
    res_ep = subprocess.run(
        [sys.executable, str(WRAPPER_SCRIPT), "--episode-dir", str(missing_ep)],
        capture_output=True,
        text=True,
        check=False,
    )
    assert res_ep.returncode == 2

    missing_ds = tmp_path / "does_not_exist_ds"
    res_ds = subprocess.run(
        [sys.executable, str(WRAPPER_SCRIPT), "--dataset-dir", str(missing_ds)],
        capture_output=True,
        text=True,
        check=False,
    )
    assert res_ds.returncode == 2


def test_cli_empty_dataset_returns_code_2_with_and_without_strict(tmp_path):
    """C1: Un directorio de dataset vacio termina con codigo 2 con y sin --strict."""
    empty_ds = tmp_path / "empty_dataset"
    empty_ds.mkdir(parents=True)

    res_no_strict = subprocess.run(
        [sys.executable, str(WRAPPER_SCRIPT), "--dataset-dir", str(empty_ds)],
        capture_output=True,
        text=True,
        check=False,
    )
    assert res_no_strict.returncode == 2

    res_strict = subprocess.run(
        [sys.executable, str(WRAPPER_SCRIPT), "--dataset-dir", str(empty_ds), "--strict"],
        capture_output=True,
        text=True,
        check=False,
    )
    assert res_strict.returncode == 2


def test_cli_mutually_exclusive_arguments_return_code_2(tmp_path):
    """C1: Pasar ambos argumentos (--episode-dir y --dataset-dir) o ninguno produce codigo 2."""
    res_both = subprocess.run(
        [
            sys.executable,
            str(WRAPPER_SCRIPT),
            "--episode-dir",
            str(tmp_path),
            "--dataset-dir",
            str(tmp_path),
        ],
        capture_output=True,
        text=True,
        check=False,
    )
    assert res_both.returncode == 2

    res_neither = subprocess.run(
        [sys.executable, str(WRAPPER_SCRIPT)],
        capture_output=True,
        text=True,
        check=False,
    )
    assert res_neither.returncode == 2


def test_cli_dataset_discovers_episode_without_bag_and_reports_missing_bag(tmp_path):
    """
    C1:
    - Un episodio sin bag no se omite; aparece en el informe como rechazado con 'missing_bag'.
    - El total reportado coincide con las carpetas de episodios descubiertas.
    - Sin --strict termina con 0 mostrando los rechazos; con --strict termina con 1.
    """
    ds_dir = tmp_path / "datasets"
    _create_episode_fixture(ds_dir / "task_0000" / "episode_00", with_bag=True, pilot_exit_code=0)
    _create_episode_fixture(ds_dir / "task_0000" / "episode_01", with_bag=False, pilot_exit_code=0)

    res_non_strict = subprocess.run(
        [sys.executable, str(WRAPPER_SCRIPT), "--dataset-dir", str(ds_dir), "--json"],
        capture_output=True,
        text=True,
        check=False,
    )
    assert res_non_strict.returncode == 0
    report = json.loads(res_non_strict.stdout)
    assert report["total_episodes"] == 2
    assert report["accepted"] == 1
    assert report["rejected"] == 1

    by_ep = {item["episode"]: item for item in report["episodes"]}
    assert "task_0000/episode_00" in by_ep
    assert "task_0000/episode_01" in by_ep
    assert by_ep["task_0000/episode_00"]["validation_status"] == "accepted"
    assert by_ep["task_0000/episode_01"]["validation_status"] == "rejected"
    assert "missing_bag" in by_ep["task_0000/episode_01"]["rejection_reasons"]

    res_strict = subprocess.run(
        [sys.executable, str(WRAPPER_SCRIPT), "--dataset-dir", str(ds_dir), "--strict", "--json"],
        capture_output=True,
        text=True,
        check=False,
    )
    assert res_strict.returncode == 1
