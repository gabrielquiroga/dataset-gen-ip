#!/usr/bin/env python3
"""
kinematic_pseudo_generator.py — Emulador Cinemático Directo a HeteroData (.pt) para Instant Policy UAV.

Genera demostraciones sintéticas mediante interpolación cinemática suave (splines cúbicos Hermite/PCHIP
libres de oscilaciones de Runge) evaluadas a 50 Hz (y submuestreadas a 10 Hz) sobre circuitos generados
por `task_generator.py` y `world_generator.py`, sin ejecutar PX4 SITL ni Gazebo.

Características clave:
  1. Comparte exactamente las mismas reglas semánticas (`--catalog-version {v1,v2}`), geometría de
     protección (`validation_geometry.py`), validador automático (`validate_episode.py`), LiDAR
     sintético 360° (`synthetic_lidar.py`) y empaquetador `HeteroData` (`bag_to_heterodata.py`).
  2. Soporta `--exclude-signatures-from <dir>` para garantizar cero fuga de firmas semánticas
     respecto a un conjunto de evaluación grabado en el simulador físico (`datasets_sitl_eval`).
  3. Registra explícitamente `"source_domain": "kinematic_spline"` en `episode_meta.json` y exporta
     `task_config.yaml`, `waypoints.yaml`, `validation_report.json` y `heterodata.pt` listos para
     ser auditados en 3 capas por `audit_dataset.py`.
"""

import argparse
import json
import math
import multiprocessing as mp
import os
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple

import numpy as np
from scipy.interpolate import PchipInterpolator
import yaml

# Asegurar inclusion de ~/venv_ip y ~/.local en sys.path
_py_ver = f"python{sys.version_info.major}.{sys.version_info.minor}"
for _candidate_site in [
    Path.home() / "venv_ip" / "lib" / _py_ver / "site-packages",
    Path.home() / ".local" / "lib" / _py_ver / "site-packages",
]:
    if _candidate_site.is_dir() and str(_candidate_site) not in sys.path:
        sys.path.append(str(_candidate_site))

_script_dir = Path(__file__).resolve().parent
_pkg_root = _script_dir.parent
for _p in (_pkg_root, _script_dir):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

from uav_ip.validation_geometry import (
    get_overfly_altitude_offset,
    get_slalom_lateral_offset,
)
from uav_ip.validate_episode import validate_episode_data, validate_waypoints_config
from task_generator import generate_tasks, get_task_signature
from world_generator import generate_random_circuit, generate_start_pose
from bag_to_heterodata import (
    HeteroData,
    torch,
    build_heterodata_from_synthetic_lidar,
)


DEFAULT_CRUISE_SPEED_MPS: float = 1.65
DEFAULT_ORBIT_SPEED_MPS: float = 1.25
DEFAULT_DESCENT_SPEED_MPS: float = 0.85
TELEMETRY_RATE_HZ: float = 50.0


def _local_to_world(
    lx: float, ly: float, lz: float, lyaw: float, dx: float, dy: float, dz: float
) -> Tuple[float, float, float]:
    """Transforma un desplazamiento local (dx, dy, dz) en el marco del landmark a coordenadas ENU."""
    c = math.cos(lyaw)
    s = math.sin(lyaw)
    wx = lx + dx * c - dy * s
    wy = ly + dx * s + dy * c
    wz = lz + dz
    return float(wx), float(wy), float(wz)


def build_kinematic_control_points(
    waypoints_config: Dict[str, Any],
    cruise_speed_mps: float = DEFAULT_CRUISE_SPEED_MPS,
    orbit_speed_mps: float = DEFAULT_ORBIT_SPEED_MPS,
    descent_speed_mps: float = DEFAULT_DESCENT_SPEED_MPS,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Construye una secuencia densa de puntos de control (t_s, pos_enu, yaw_enu) que respeta
    estrictamente las restricciones geométricas de cada primitiva en `waypoints_config`.
    """
    landmarks_by_id = {
        str(lm["id"]): lm for lm in waypoints_config.get("landmarks", [])
    }
    primitives = waypoints_config.get("primitives", [])
    landing = waypoints_config.get("landing", {"x": 15.0, "y": 0.0, "z": 0.05})

    pts: List[Tuple[float, float, float]] = []
    seg_speeds: List[float] = []
    yaw_overrides: List[Optional[float]] = []

    def _append_pt(
        xyz: Tuple[float, float, float],
        speed: float = cruise_speed_mps,
        yaw_override: Optional[float] = None,
    ) -> None:
        if pts:
            dist = math.dist(pts[-1], xyz)
            if dist < 0.05:
                return
            seg_speeds.append(max(0.20, float(speed)))
        pts.append((float(xyz[0]), float(xyz[1]), float(xyz[2])))
        yaw_overrides.append(yaw_override)

    # Fase de despegue vertical (recortada luego por find_flight_trim_indices con takeoff_exit)
    _append_pt((0.0, 0.0, 0.08), speed=1.0, yaw_override=0.0)
    _append_pt((0.0, 0.0, 1.00), speed=1.2, yaw_override=0.0)
    _append_pt((0.0, 0.0, 1.80), speed=1.2, yaw_override=0.0)
    _append_pt((0.6, 0.0, 1.80), speed=cruise_speed_mps, yaw_override=0.0)

    for prim in primitives:
        ptype = str(prim["type"])
        target_id = str(prim.get("target") or prim.get("landmark_id"))
        lm = landmarks_by_id[target_id]
        lm_type = str(lm["type"])
        lx = float(lm["x"])
        ly = float(lm["y"])
        lz = float(lm["z"])
        lyaw = float(lm.get("yaw", 0.0))

        if ptype == "traverse":
            for dx in (-2.0, -1.0, -0.35, 0.0, 0.35, 1.0, 2.0):
                _append_pt(
                    _local_to_world(lx, ly, lz, lyaw, dx, 0.0, 0.0),
                    speed=cruise_speed_mps,
                    yaw_override=lyaw,
                )

        elif ptype == "overfly":
            dz = get_overfly_altitude_offset(lm_type)
            _append_pt(_local_to_world(lx, ly, lz, lyaw, -2.0, 0.0, dz * 0.65), speed=cruise_speed_mps, yaw_override=lyaw)
            _append_pt(_local_to_world(lx, ly, lz, lyaw, -0.8, 0.0, dz), speed=cruise_speed_mps, yaw_override=lyaw)
            _append_pt(_local_to_world(lx, ly, lz, lyaw, 0.0, 0.0, dz), speed=cruise_speed_mps, yaw_override=lyaw)
            _append_pt(_local_to_world(lx, ly, lz, lyaw, 0.8, 0.0, dz), speed=cruise_speed_mps, yaw_override=lyaw)
            _append_pt(_local_to_world(lx, ly, lz, lyaw, 2.0, 0.0, dz * 0.65), speed=cruise_speed_mps, yaw_override=lyaw)

        elif ptype == "slalom":
            side = str(prim.get("side") or prim.get("direction") or "left").lower()
            sign = 1.0 if side == "left" else -1.0
            dy = sign * get_slalom_lateral_offset(lm_type)
            _append_pt(_local_to_world(lx, ly, lz, lyaw, -2.0, 0.0, 0.0), speed=cruise_speed_mps)
            _append_pt(_local_to_world(lx, ly, lz, lyaw, -0.85, dy * 0.88, 0.0), speed=cruise_speed_mps)
            _append_pt(_local_to_world(lx, ly, lz, lyaw, 0.0, dy, 0.0), speed=cruise_speed_mps, yaw_override=lyaw)
            _append_pt(_local_to_world(lx, ly, lz, lyaw, 0.85, dy * 0.88, 0.0), speed=cruise_speed_mps)
            _append_pt(_local_to_world(lx, ly, lz, lyaw, 2.0, 0.0, 0.0), speed=cruise_speed_mps)

        elif ptype == "orbit":
            direction = str(prim.get("direction") or "ccw").lower()
            dir_sign = -1.0 if direction == "cw" else 1.0
            orbit_r = 1.50
            _append_pt(_local_to_world(lx, ly, lz, lyaw, -2.2, 0.0, 0.0), speed=cruise_speed_mps)
            num_arc_steps = 14
            d_alpha = math.radians(30.0)
            for k in range(num_arc_steps + 1):
                alpha = math.pi + dir_sign * k * d_alpha
                dx = orbit_r * math.cos(alpha)
                dy = orbit_r * math.sin(alpha)
                wx, wy, wz = _local_to_world(lx, ly, lz, lyaw, dx, dy, 0.0)
                centripetal_yaw = math.atan2(ly - wy, lx - wx)
                _append_pt((wx, wy, wz), speed=orbit_speed_mps, yaw_override=centripetal_yaw)
            _append_pt(
                _local_to_world(lx, ly, lz, lyaw, 2.0, -dir_sign * orbit_r * 0.8, 0.0),
                speed=cruise_speed_mps,
            )

    # Fase de aproximación y descenso vertical hacia la plataforma de aterrizaje
    land_x = float(landing.get("x", 15.0))
    land_y = float(landing.get("y", 0.0))
    land_z = float(landing.get("z", 0.05))
    last_yaw = 0.0
    if pts:
        dx_l = land_x - pts[-1][0]
        dy_l = land_y - pts[-1][1]
        if math.hypot(dx_l, dy_l) > 1e-3:
            last_yaw = math.atan2(dy_l, dx_l)

    _append_pt((land_x - 1.5 * math.cos(last_yaw), land_y - 1.5 * math.sin(last_yaw), 1.6), speed=cruise_speed_mps, yaw_override=last_yaw)
    _append_pt((land_x, land_y, 1.4), speed=cruise_speed_mps * 0.8, yaw_override=last_yaw)
    _append_pt((land_x, land_y, 0.7), speed=descent_speed_mps, yaw_override=last_yaw)
    _append_pt((land_x, land_y, land_z + 0.03), speed=descent_speed_mps * 0.6, yaw_override=last_yaw)

    pos_ctrl = np.asarray(pts, dtype=np.float64)
    diffs = np.diff(pos_ctrl, axis=0)
    dists = np.linalg.norm(diffs, axis=1)
    dt_segs = dists / np.asarray(seg_speeds, dtype=np.float64)
    t_ctrl = np.concatenate([[0.0], np.cumsum(dt_segs)])

    yaws_ctrl = np.zeros(len(pos_ctrl), dtype=np.float64)
    for i in range(len(pos_ctrl)):
        if yaw_overrides[i] is not None:
            yaws_ctrl[i] = float(yaw_overrides[i])
        else:
            i_prev = max(0, i - 1)
            i_next = min(len(pos_ctrl) - 1, i + 1)
            dx = pos_ctrl[i_next, 0] - pos_ctrl[i_prev, 0]
            dy = pos_ctrl[i_next, 1] - pos_ctrl[i_prev, 1]
            if math.hypot(dx, dy) > 1e-3:
                yaws_ctrl[i] = math.atan2(dy, dx)
            elif i > 0:
                yaws_ctrl[i] = yaws_ctrl[i - 1]

    yaws_ctrl_unwrapped = np.unwrap(yaws_ctrl)
    return t_ctrl, pos_ctrl, yaws_ctrl_unwrapped


def synthesize_kinematic_telemetry(
    waypoints_config: Dict[str, Any],
    sample_rate_hz: float = TELEMETRY_RATE_HZ,
    cruise_speed_mps: float = DEFAULT_CRUISE_SPEED_MPS,
) -> Dict[str, Any]:
    """
    Genera la trayectoria cinemática continua evaluada a `sample_rate_hz` (50 Hz) con una
    ventana de reposo final de 1.5 s sobre la plataforma de aterrizaje, y devuelve el paquete
    de telemetría en formato PX4 NED compatible con `validate_episode_data()` y
    `build_heterodata_from_synthetic_lidar()`.
    """
    t_ctrl, pos_ctrl, yaws_ctrl_unwrapped = build_kinematic_control_points(
        waypoints_config=waypoints_config,
        cruise_speed_mps=cruise_speed_mps,
    )

    flight_duration_s = float(t_ctrl[-1])
    dt = 1.0 / float(sample_rate_hz)
    t_flight = np.arange(0.0, flight_duration_s + 0.5 * dt, dt, dtype=np.float64)
    if t_flight[-1] > flight_duration_s:
        t_flight[-1] = flight_duration_s

    interp_x = PchipInterpolator(t_ctrl, pos_ctrl[:, 0])
    interp_y = PchipInterpolator(t_ctrl, pos_ctrl[:, 1])
    interp_z = PchipInterpolator(t_ctrl, pos_ctrl[:, 2])
    interp_yaw = PchipInterpolator(t_ctrl, yaws_ctrl_unwrapped)

    x_flight = interp_x(t_flight)
    y_flight = interp_y(t_flight)
    z_flight = interp_z(t_flight)
    yaw_flight = np.arctan2(np.sin(interp_yaw(t_flight)), np.cos(interp_yaw(t_flight)))

    # Añadir 1.5 s de reposo en tierra sobre la plataforma para cumplir la ventana de aterrizaje
    n_rest = max(10, int(round(1.50 / dt)))
    t_rest = flight_duration_s + np.arange(1, n_rest + 1, dtype=np.float64) * dt
    t_eval = np.concatenate([t_flight, t_rest])
    x_eval = np.concatenate([x_flight, np.full(n_rest, x_flight[-1])])
    y_eval = np.concatenate([y_flight, np.full(n_rest, y_flight[-1])])
    z_eval = np.concatenate([z_flight, np.full(n_rest, z_flight[-1])])
    yaw_eval = np.concatenate([yaw_flight, np.full(n_rest, yaw_flight[-1])])

    pos_enu = np.column_stack([x_eval, y_eval, z_eval])
    N = len(t_eval)
    vel_enu = np.zeros_like(pos_enu)
    if N > 1:
        vel_enu[1:] = np.diff(pos_enu, axis=0) / dt
        vel_enu[0] = vel_enu[1]
    vel_enu[len(t_flight):] = 0.0

    odom_msgs = []
    base_s = 100.0
    for i in range(N):
        t_us = int(round((base_s + t_eval[i]) * 1e6))
        t_ns = t_us * 1000
        p_ned = [float(pos_enu[i, 1]), float(pos_enu[i, 0]), float(-pos_enu[i, 2])]
        v_ned = [float(vel_enu[i, 1]), float(vel_enu[i, 0]), float(-vel_enu[i, 2])]
        yaw_ned = math.pi / 2.0 - float(yaw_eval[i])
        q_ned = [math.cos(yaw_ned * 0.5), 0.0, 0.0, math.sin(yaw_ned * 0.5)]
        odom_msgs.append({
            "timestamp": t_ns,
            "timestamp_us": t_us,
            "timestamp_sample_us": t_us,
            "pose_frame": 1,
            "velocity_frame": 1,
            "position": p_ned,
            "q": q_ned,
            "velocity": v_ned,
            "reset_counter": 0,
        })

    step_ld = max(1, int(round(0.20 / dt)))
    rest_start_idx = len(t_flight)
    land_seq = []
    for idx in range(0, N, step_ld):
        t_us_ld = int(round((base_s + t_eval[idx]) * 1e6))
        is_landed = bool(idx >= rest_start_idx)
        land_seq.append({
            "timestamp": t_us_ld * 1000,
            "timestamp_us": t_us_ld,
            "landed": is_landed,
            "ground_contact": is_landed,
            "maybe_landed": is_landed,
            "freefall": False,
            "at_rest": is_landed,
        })
    last_t_us = int(round((base_s + t_eval[-1]) * 1e6))
    if not land_seq or land_seq[-1]["timestamp_us"] != last_t_us:
        land_seq.append({
            "timestamp": last_t_us * 1000,
            "timestamp_us": last_t_us,
            "landed": True,
            "ground_contact": True,
            "maybe_landed": True,
            "freefall": False,
            "at_rest": True,
        })

    telemetry_bundle = {
        "odometry": odom_msgs,
        "local_pos": [],
        "land_detected": land_seq,
        "vehicle_status": [
            {
                "timestamp": last_t_us * 1000,
                "timestamp_us": last_t_us,
                "nav_state": 14,
                "arming_state": 2,
                "failsafe": False,
            }
        ],
        "setpoints": [],
        "timesync_status": [],
    }

    return {
        "telemetry": telemetry_bundle,
        "duration_s": float(t_eval[-1]),
    }


def _collect_excluded_signatures(exclude_sources: List[Path]) -> Set[str]:
    """Lee todas las firmas semánticas presentes en directorios o archivos de datasets existentes."""
    excluded: Set[str] = set()
    for d in exclude_sources:
        if not d:
            continue
        p = Path(d)
        if not p.exists():
            continue
        if p.is_file():
            try:
                if p.suffix in {".yaml", ".yml"}:
                    with open(p, "r", encoding="utf-8") as f:
                        cfg = yaml.safe_load(f)
                    if cfg:
                        excluded.add(get_task_signature(cfg))
                elif p.suffix == ".json":
                    with open(p, "r", encoding="utf-8") as f:
                        data = json.load(f)
                    if isinstance(data, list):
                        for item in data:
                            if isinstance(item, dict):
                                excluded.add(get_task_signature(item))
                    elif isinstance(data, dict):
                        for item in data.get("tasks", []):
                            if isinstance(item, dict):
                                excluded.add(get_task_signature(item))
                        for sig in data.get("signatures", []):
                            if isinstance(sig, str):
                                excluded.add(sig)
            except Exception:
                pass
        elif p.is_dir():
            for cfg_file in sorted(p.glob("task_*/task_config.yaml")):
                try:
                    with open(cfg_file, "r", encoding="utf-8") as f:
                        cfg = yaml.safe_load(f)
                    if cfg:
                        excluded.add(get_task_signature(cfg))
                except Exception:
                    pass
    return excluded


def _build_episode_waypoints_config(task_cfg: Dict[str, Any], seed: int) -> Dict[str, Any]:
    """Genera el diccionario `waypoints.yaml` para un episodio a partir de `task_config` y una semilla."""
    start_pose = generate_start_pose(seed=seed)
    hoops, landing = generate_random_circuit(
        landmarks_info=task_cfg["landmarks"],
        seed=seed,
        primitives=task_cfg["primitives"],
        start_pose=start_pose,
    )
    # Normalizar tipo 'gate' a 'square_gate' para validate_waypoints_config
    norm_hoops = []
    for h in hoops:
        hc = dict(h)
        if hc.get("type") == "gate":
            hc["type"] = "square_gate"
        norm_hoops.append(hc)

    landmarks_map = {lm["id"]: lm for lm in norm_hoops}
    sorted_primitives = sorted(
        task_cfg["primitives"],
        key=lambda p: float(landmarks_map.get(p.get("target"), {}).get("x", 0.0)),
    )
    return {
        "seed": int(seed),
        "start_pose": start_pose,
        "landmarks": norm_hoops,
        "primitives": sorted_primitives,
        "landing": landing,
    }


def _generate_task_worker(job: Dict[str, Any]) -> Dict[str, Any]:
    """Worker multiproceso que genera y valida todos los episodios de una tarea cinemática."""
    t0 = time.perf_counter()
    task_id: str = job["task_id"]
    task_cfg: Dict[str, Any] = job["task_cfg"]
    task_dir: Path = job["task_dir"]
    episodes_per_task: int = job["episodes_per_task"]
    base_seed: int = job["seed"]
    scale_factor: float = job["scale_factor"]
    fps_points: int = job["fps_points"]
    r_max: float = job["r_max"]
    action_horizon: int = job["action_horizon"]
    num_keyframes: int = job["num_keyframes"]
    tube_aug_prob: float = job["tube_aug_prob"]
    tube_sigma_pos: float = job["tube_sigma_pos"]
    tube_sigma_yaw_deg: float = job["tube_sigma_yaw_deg"]
    tube_decay: float = job["tube_decay"]
    state_flip_prob: float = job["state_flip_prob"]
    require_torch: bool = job["require_torch"]

    task_dir.mkdir(parents=True, exist_ok=True)
    task_cfg_path = task_dir / "task_config.yaml"
    with open(task_cfg_path, "w", encoding="utf-8") as f:
        yaml.safe_dump(task_cfg, f, sort_keys=False)

    accepted_count = 0
    total_T = 0
    total_T_aug = 0
    errors: List[str] = []

    for ep_idx in range(episodes_per_task):
        ep_name = f"episode_{ep_idx:02d}"
        ep_dir = task_dir / ep_name
        ep_dir.mkdir(parents=True, exist_ok=True)

        ep_accepted = False
        last_reasons: List[str] = []
        ep_seed = base_seed
        waypoints_cfg: Dict[str, Any] = {}
        synth: Dict[str, Any] = {}
        val_report: Dict[str, Any] = {}

        for attempt in range(12):
            ep_seed = base_seed + ep_idx * 1009 + attempt * 131
            try:
                waypoints_cfg = _build_episode_waypoints_config(task_cfg, seed=ep_seed)
            except Exception as exc:
                last_reasons = [f"circuit_generation_error:{exc}"]
                continue

            cfg_check = validate_waypoints_config(waypoints_cfg)
            if not cfg_check.get("valid", False):
                last_reasons = cfg_check.get("errors", ["invalid_waypoints_config"])
                continue

            synth = synthesize_kinematic_telemetry(waypoints_cfg)
            val_report = validate_episode_data(
                telemetry=synth["telemetry"],
                waypoints_cfg=waypoints_cfg,
                pilot_exit_code=0,
            )
            if val_report.get("validation_status") == "accepted":
                ep_accepted = True
                break
            last_reasons = val_report.get("rejection_reasons", ["kinematic_validation_failed"])

        if not ep_accepted:
            errors.append(f"{task_id}/{ep_name}: {last_reasons}")
            continue

        wp_file = ep_dir / "waypoints.yaml"
        with open(wp_file, "w", encoding="utf-8") as f:
            yaml.safe_dump(waypoints_cfg, f, sort_keys=False)

        val_file = ep_dir / "validation_report.json"
        with open(val_file, "w", encoding="utf-8") as f:
            json.dump(val_report, f, indent=2)

        meta = {
            "task_id": task_id,
            "episode_id": ep_name,
            "original_episode_id": ep_name,
            "attempt_index": 0,
            "seed": int(ep_seed),
            "status": "success",
            "validation_status": "accepted",
            "rejection_reasons": [],
            "duration_s": round(float(synth["duration_s"]), 3),
            "source_domain": "kinematic_spline",
            "catalog_version": str(job.get("catalog_version", "v2")),
            "metrics": val_report.get("metrics", {}),
        }
        with open(ep_dir / "episode_meta.json", "w", encoding="utf-8") as f:
            json.dump(meta, f, indent=2)

        out_pt = ep_dir / "heterodata.pt"
        res = build_heterodata_from_synthetic_lidar(
            parsed_data=synth["telemetry"],
            waypoints_file=wp_file,
            output_path=out_pt,
            scale_factor=scale_factor,
            r_max=r_max,
            fps_points=fps_points,
            subsample_step=5,
            trim_idle=True,
            trim_start="takeoff_exit",
            action_horizon=action_horizon,
            action_stride=1,
            num_keyframes=num_keyframes,
            gravity_aligned=True,
            tube_aug_prob=tube_aug_prob,
            tube_sigma_pos=tube_sigma_pos,
            tube_sigma_yaw_deg=tube_sigma_yaw_deg,
            tube_decay=tube_decay,
            state_flip_prob=state_flip_prob,
            require_torch=require_torch,
            verbose=False,
        )
        if not res.get("ok", False):
            errors.append(f"{task_id}/{ep_name}: build_heterodata failed")
            continue

        accepted_count += 1
        total_T += int(res["T"])
        total_T_aug += int(res["T_aug"])

    return {
        "task_id": task_id,
        "signature": get_task_signature(task_cfg),
        "accepted_episodes": accepted_count,
        "expected_episodes": episodes_per_task,
        "total_T": total_T,
        "total_T_aug": total_T_aug,
        "elapsed_s": time.perf_counter() - t0,
        "errors": errors,
    }


def generate_kinematic_dataset(
    output_dir: Path,
    num_tasks: int = 100,
    episodes_per_task: int = 5,
    catalog_version: str = "v2",
    exclude_signatures_from: Optional[List[Path]] = None,
    seed: int = 1000,
    workers: int = 0,
    scale_factor: float = 0.1,
    fps_points: int = 2048,
    r_max: float = 10.0,
    action_horizon: int = 8,
    num_keyframes: int = 10,
    tube_aug_prob: float = 0.30,
    tube_sigma_pos: float = 0.25,
    tube_sigma_yaw_deg: float = 5.0,
    tube_decay: float = 0.70,
    state_flip_prob: float = 0.10,
    require_torch: bool = False,
) -> Dict[str, Any]:
    """
    Genera un dataset completo de pseudo-demostraciones cinemáticas en `output_dir`.
    """
    if require_torch and (HeteroData is None or torch is None):
        raise RuntimeError(
            "PyTorch y/o torch_geometric no están disponibles (--require-torch activo). "
            "Ejecuta con ../venv_ip/bin/python3."
        )

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    excluded_sigs = _collect_excluded_signatures(exclude_signatures_from or [])

    oversample = max(num_tasks + len(excluded_sigs) + 25, int(num_tasks * 1.5))
    raw_tasks = generate_tasks(
        num_tasks=oversample,
        seed=seed,
        catalog_version=catalog_version,
    )

    selected_tasks: List[Dict[str, Any]] = []
    seen_sigs: Set[str] = set(excluded_sigs)
    for t_cfg in raw_tasks:
        sig = get_task_signature(t_cfg)
        if sig in seen_sigs:
            continue
        seen_sigs.add(sig)
        idx = len(selected_tasks)
        t_cfg = dict(t_cfg)
        t_cfg["task_id"] = f"task_{idx:04d}"
        selected_tasks.append(t_cfg)
        if len(selected_tasks) >= num_tasks:
            break

    if len(selected_tasks) < num_tasks:
        raise RuntimeError(
            f"Solo se pudieron generar {len(selected_tasks)} tareas únicas tras excluir "
            f"{len(excluded_sigs)} firmas."
        )

    try:
        from uav_ip.synthetic_lidar import _get_fps_c_kernel
        _get_fps_c_kernel()
    except Exception:
        pass

    cpu_cores = os.cpu_count() or 4
    num_workers = cpu_cores if workers <= 0 else min(int(workers), cpu_cores)
    num_workers = min(num_workers, len(selected_tasks))

    jobs = []
    for idx, t_cfg in enumerate(selected_tasks):
        tid = t_cfg["task_id"]
        jobs.append({
            "task_id": tid,
            "task_cfg": t_cfg,
            "task_dir": output_dir / tid,
            "episodes_per_task": int(episodes_per_task),
            "catalog_version": catalog_version,
            "seed": int(seed + idx * 10007),
            "scale_factor": scale_factor,
            "fps_points": fps_points,
            "r_max": r_max,
            "action_horizon": action_horizon,
            "num_keyframes": num_keyframes,
            "tube_aug_prob": tube_aug_prob,
            "tube_sigma_pos": tube_sigma_pos,
            "tube_sigma_yaw_deg": tube_sigma_yaw_deg,
            "tube_decay": tube_decay,
            "state_flip_prob": state_flip_prob,
            "require_torch": require_torch,
        })

    print("=" * 72)
    print(f" Generador Cinemático de Pseudo-Demostraciones (Catálogo {catalog_version})")
    print(f"   Directorio salida  : {output_dir.resolve()}")
    print(f"   Tareas x Episodios : {len(jobs)} x {episodes_per_task} = {len(jobs) * episodes_per_task} episodios")
    print(f"   Firmas excluidas   : {len(excluded_sigs)}")
    print(f"   Workers paralelos  : {num_workers}")
    print("=" * 72)

    t_start = time.perf_counter()
    results = []
    if num_workers == 1:
        for i, job in enumerate(jobs, 1):
            res = _generate_task_worker(job)
            results.append(res)
            print(
                f"  [{i:3d}/{len(jobs):3d}] {res['task_id']} | "
                f"{res['accepted_episodes']}/{res['expected_episodes']} eps | "
                f"T={res['total_T']} (+{res['total_T_aug']} aug) | {res['elapsed_s']:.2f}s"
            )
    else:
        with mp.Pool(processes=num_workers) as pool:
            for i, res in enumerate(pool.imap_unordered(_generate_task_worker, jobs), 1):
                results.append(res)
                print(
                    f"  [{i:3d}/{len(jobs):3d}] {res['task_id']} | "
                    f"{res['accepted_episodes']}/{res['expected_episodes']} eps | "
                    f"T={res['total_T']} (+{res['total_T_aug']} aug) | {res['elapsed_s']:.2f}s"
                )

    elapsed_total = time.perf_counter() - t_start
    total_eps = sum(r["accepted_episodes"] for r in results)
    total_T = sum(r["total_T"] for r in results)
    total_aug = sum(r["total_T_aug"] for r in results)
    all_errors = [err for r in results for err in r["errors"]]

    print("-" * 72)
    print(f" Completado en {elapsed_total:.2f}s: {total_eps}/{len(jobs) * episodes_per_task} episodios aceptados.")
    print(f" Estados generados: {total_T} nominales + {total_aug} tube-augmented = {total_T + total_aug}.")
    if all_errors:
        print(f" Advertencias/Errores ({len(all_errors)}): {all_errors[:5]}")
    print("=" * 72)

    return {
        "num_tasks": len(jobs),
        "episodes_per_task": episodes_per_task,
        "total_accepted_episodes": total_eps,
        "total_T": total_T,
        "total_T_aug": total_aug,
        "elapsed_s": elapsed_total,
        "errors": all_errors,
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Generador cinemático de pseudo-demostraciones SE(3) para Instant Policy UAV."
    )
    parser.add_argument("--output-dir", type=Path, default=Path("datasets_kinematic"), help="Directorio de salida")
    parser.add_argument("--num-tasks", type=int, default=100, help="Número de tareas únicas a generar (default: 100)")
    parser.add_argument("--episodes-per-task", type=int, default=5, help="Episodios por tarea (default: 5)")
    parser.add_argument(
        "--catalog-version",
        choices=["v1", "v2"],
        default="v2",
        help="Versión del catálogo combinatorio (default: v2)",
    )
    parser.add_argument(
        "--exclude-signatures-from", "--exclude-tasks-from",
        dest="exclude_signatures_from",
        type=Path,
        nargs="*",
        default=None,
        help="Uno o más directorios o archivos de datasets cuyas firmas semánticas deben excluirse (alias: --exclude-tasks-from)",
    )
    parser.add_argument("--seed", type=int, default=1000, help="Semilla aleatoria base (default: 1000)")
    parser.add_argument("--workers", "-j", type=int, default=0, help="Procesos paralelos (default: 0 = todos los núcleos)")
    parser.add_argument("--tube-decay", type=float, default=0.70, help="Factor gamma de Tube Augmentation (default: 0.70)")
    parser.add_argument("--require-torch", action="store_true", help="Exigir exportación en formato PyTorch Geometric (.pt)")
    args = parser.parse_args()

    summary = generate_kinematic_dataset(
        output_dir=args.output_dir,
        num_tasks=args.num_tasks,
        episodes_per_task=args.episodes_per_task,
        catalog_version=args.catalog_version,
        exclude_signatures_from=args.exclude_signatures_from,
        seed=args.seed,
        workers=args.workers,
        tube_decay=args.tube_decay,
        require_torch=args.require_torch,
    )
    if summary["errors"] or summary["total_accepted_episodes"] < args.num_tasks * args.episodes_per_task:
        sys.exit(1)


if __name__ == "__main__":
    main()
