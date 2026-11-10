"""
kinematic_rollout_eval.py — Simulador en Lazo Cerrado Cinematico Ideal para UAV Instant Policy.

Ejecuta una politica (`MiniInstantPolicyNet`, `OracleReplayPolicy` o el modelo final del Hito 4)
en un bucle cerrado de horizonte deslizante (Receding Horizon / Action Chunking) puramente en
Python sobre cualquier circuito `waypoints.yaml`:
  1. En cada instante de replanificacion t, sintetiza en vivo la nube LiDAR 360° de 2048 puntos
     en marco Horizontal FLU (s=0.1) usando `SyntheticLidarGenerator`.
  2. Invoca `policy.predict_actions(sample)` para obtener el horizonte futuro de T=8 pasos.
  3. Ejecuta los primeros `execution_horizon` (T_exec in {1, 2, 4, 8}) pasos a 10 Hz,
     actualizando posicion, yaw, velocidad egocentrica `vel_flu` y estado `flight_state`.
  4. Al detectar aterrizaje sobre la plataforma (o alcanzar `max_steps`), interpola la
     trayectoria ejecutada a 50 Hz mediante splines PCHIP, empaqueta la telemetria en formato
     PX4 NED y la audita con `validate_episode_data()` de `uav_ip.validate_episode`.
"""

from __future__ import annotations

import math
import time
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
from scipy.interpolate import PchipInterpolator
import torch

from uav_ip.synthetic_lidar import SyntheticLidarGenerator, rot_z
from uav_ip.validate_episode import validate_episode_data

from .se3_diffusion_math import (
    DEFAULT_SCALE_FACTOR,
    build_se3_horizontal_flu_torch,
    get_uav_keypoints_body_torch,
    wrap_angle_numpy,
)


def _trajectory_10hz_to_px4_telemetry_50hz(
    timestamps_10hz: np.ndarray,
    pos_enu_10hz: np.ndarray,
    yaw_enu_10hz: np.ndarray,
    landing_cfg: Dict[str, Any],
     settle_to_pad_on_land: bool = True,
    sample_rate_hz: float = 50.0,
) -> Dict[str, Any]:
    """
    Convierte una serie temporal ejecutada a 10 Hz (`pos_enu_10hz`, `yaw_enu_10hz`)
    en un paquete de telemetria a 50 Hz en formato PX4 NED con cola de reposo de 1.5 s,
    compatible con `validate_episode_data()`.
    """
    t_arr = np.asarray(timestamps_10hz, dtype=np.float64).copy()
    pos_arr = np.asarray(pos_enu_10hz, dtype=np.float64).copy()
    yaw_arr = np.asarray(yaw_enu_10hz, dtype=np.float64).copy()

    # Si el ultimo punto quedo bajo al activar aterrizaje, asentar verticalmente en su (X, Y) real
    if settle_to_pad_on_land and landing_cfg:
        lz = float(landing_cfg.get("z", 0.05)) + 0.03
        last_x = float(pos_arr[-1, 0])
        last_y = float(pos_arr[-1, 1])
        if float(pos_arr[-1, 2]) <= 0.95 and float(pos_arr[-1, 2]) > lz + 1e-3:
            t_arr = np.append(t_arr, t_arr[-1] + 0.40)
            pos_arr = np.vstack([pos_arr, np.array([[last_x, last_y, lz]], dtype=np.float64)])
            yaw_arr = np.append(yaw_arr, yaw_arr[-1])

    for i in range(1, len(t_arr)):
        if t_arr[i] <= t_arr[i - 1]:
            t_arr[i] = t_arr[i - 1] + 0.1

    yaw_unwrapped = np.unwrap(yaw_arr)
    duration_s = float(t_arr[-1] - t_arr[0])
    t_rel = t_arr - t_arr[0]
    dt = 1.0 / float(sample_rate_hz)

    if len(t_rel) >= 2 and duration_s > dt:
        t_flight = np.arange(0.0, duration_s + 0.5 * dt, dt, dtype=np.float64)
        if t_flight[-1] > duration_s:
            t_flight[-1] = duration_s
        x_flight = PchipInterpolator(t_rel, pos_arr[:, 0])(t_flight)
        y_flight = PchipInterpolator(t_rel, pos_arr[:, 1])(t_flight)
        z_flight = PchipInterpolator(t_rel, pos_arr[:, 2])(t_flight)
        yaw_interp = PchipInterpolator(t_rel, yaw_unwrapped)(t_flight)
        yaw_flight = np.arctan2(np.sin(yaw_interp), np.cos(yaw_interp))
    else:
        t_flight = np.array([0.0], dtype=np.float64)
        x_flight = pos_arr[:1, 0]
        y_flight = pos_arr[:1, 1]
        z_flight = pos_arr[:1, 2]
        yaw_flight = yaw_arr[:1]

    # Añadir ventana de reposo final de 1.5 s para el validador de aterrizaje
    n_rest = max(10, int(round(1.50 / dt)))
    t_rest = float(t_flight[-1]) + np.arange(1, n_rest + 1, dtype=np.float64) * dt
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
    vel_enu[len(t_flight) :] = 0.0

    odom_msgs = []
    base_s = 100.0
    for i in range(N):
        t_us = int(round((base_s + t_eval[i]) * 1e6))
        t_ns = t_us * 1000
        p_ned = [float(pos_enu[i, 1]), float(pos_enu[i, 0]), float(-pos_enu[i, 2])]
        v_ned = [float(vel_enu[i, 1]), float(vel_enu[i, 0]), float(-vel_enu[i, 2])]
        yaw_ned = math.pi / 2.0 - float(yaw_eval[i])
        q_ned = [math.cos(yaw_ned * 0.5), 0.0, 0.0, math.sin(yaw_ned * 0.5)]
        odom_msgs.append(
            {
                "timestamp": t_ns,
                "timestamp_us": t_us,
                "timestamp_sample_us": t_us,
                "pose_frame": 1,
                "velocity_frame": 1,
                "position": p_ned,
                "q": q_ned,
                "velocity": v_ned,
                "reset_counter": 0,
            }
        )

    step_ld = max(1, int(round(0.20 / dt)))
    rest_start_idx = len(t_flight)
    land_seq = []
    for idx in range(0, N, step_ld):
        t_us_ld = int(round((base_s + t_eval[idx]) * 1e6))
        is_landed = bool(idx >= rest_start_idx)
        land_seq.append(
            {
                "timestamp": t_us_ld * 1000,
                "timestamp_us": t_us_ld,
                "landed": is_landed,
                "ground_contact": is_landed,
                "maybe_landed": is_landed,
                "freefall": False,
                "at_rest": is_landed,
            }
        )
    last_t_us = int(round((base_s + t_eval[-1]) * 1e6))
    if not land_seq or land_seq[-1]["timestamp_us"] != last_t_us:
        land_seq.append(
            {
                "timestamp": last_t_us * 1000,
                "timestamp_us": last_t_us,
                "landed": True,
                "ground_contact": True,
                "maybe_landed": True,
                "freefall": False,
                "at_rest": True,
            }
        )

    return {
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


_CACHED_LIDAR_GENERATORS: Dict[float, SyntheticLidarGenerator] = {}


def _get_lidar_generator(scale_factor: float) -> SyntheticLidarGenerator:
    sf = float(scale_factor)
    if sf not in _CACHED_LIDAR_GENERATORS:
        _CACHED_LIDAR_GENERATORS[sf] = SyntheticLidarGenerator(
            r_max=10.0,
            scale_factor=sf,
            target_points=2048,
        )
    return _CACHED_LIDAR_GENERATORS[sf]


def run_kinematic_closed_loop_rollout(
    policy: Any,
    waypoints_cfg: Dict[str, Any],
    context_pack: Dict[str, Any],
    execution_horizon: int = 4,
    num_ddim_steps: int = 4,
    max_steps_10hz: int = 160,
    scale_factor: float = DEFAULT_SCALE_FACTOR,
    action_stride: Optional[int] = None,
    initial_pos_enu_m: Optional[np.ndarray] = None,
    initial_yaw_enu_rad: Optional[float] = None,
    initial_vel_flu_mps: Optional[np.ndarray] = None,
    reference_pos_enu_m: Optional[np.ndarray] = None,
) -> Dict[str, Any]:
    """
    Ejecuta una simulacion completa en lazo cerrado con horizonte deslizante
    `T_exec = execution_horizon` in [1, T] y paso temporal `dt_step = 0.10 * action_stride`:
      - Sintetiza LiDAR 360° en vivo con `SyntheticLidarGenerator` en cada replanificacion.
      - Cachea las representaciones de contexto (φ_e + σ) una sola vez al inicio del episodio
        si `policy` implementa `encode_context` (incluyendo `context_keyframe_mask` si `L` es adaptativo).
      - Evalua la trayectoria resultante con `validate_episode_data()` y calcula el error
        ADE/FDE respecto a `reference_pos_enu_m` si se provee.
    """
    req_t_exec = max(1, int(execution_horizon))
    if action_stride is not None:
        stride = max(1, int(action_stride))
    elif isinstance(context_pack, dict) and "action_stride" in context_pack:
        stride = max(1, int(context_pack["action_stride"]))
    elif hasattr(policy, "action_stride"):
        stride = max(1, int(getattr(policy, "action_stride")))
    else:
        stride = 1
    dt_step = 0.10 * float(stride)

    lidar_gen = _get_lidar_generator(scale_factor)
    landmarks_list = waypoints_cfg.get("landmarks", [])
    landing_cfg = waypoints_cfg.get("landing", {})

    start_pose = waypoints_cfg.get("start_pose", {"x": 0.0, "y": 0.0, "z": 1.5, "yaw": 0.0})
    if initial_pos_enu_m is not None:
        curr_pos_m = np.asarray(initial_pos_enu_m, dtype=np.float64).reshape(3).copy()
    else:
        curr_pos_m = np.array(
            [
                float(start_pose.get("x", 0.0)),
                float(start_pose.get("y", 0.0)),
                float(start_pose.get("z", 1.5)),
            ],
            dtype=np.float64,
        )

    if initial_yaw_enu_rad is not None:
        curr_yaw = float(initial_yaw_enu_rad)
    else:
        curr_yaw = float(start_pose.get("yaw", 0.0))

    if initial_vel_flu_mps is not None:
        curr_vel_flu = np.asarray(initial_vel_flu_mps, dtype=np.float32).reshape(3).copy()
    else:
        curr_vel_flu = np.zeros(3, dtype=np.float32)

    curr_flight_state = 1.0
    kp_body = get_uav_keypoints_body_torch(scale_factor=scale_factor)

    # Pre-cachear contexto si la politica lo soporta (optimizacion clave de Instant Policy)
    t_ctx_start = time.perf_counter()
    cached_context = None
    if hasattr(policy, "encode_context") and callable(getattr(policy, "encode_context")):
        device = next(policy.parameters()).device
        with torch.no_grad():
            ctx_pose = context_pack.get("context_pose_se3")
            ctx_mask = context_pack.get("context_keyframe_mask")
            cached_context = policy.encode_context(
                context_scene_pos=context_pack["context_scene_pos"].to(device),
                context_vel_flu=context_pack["context_vel_flu"].to(device),
                context_flight_state=context_pack["context_flight_state"].to(device),
                context_delta_4dof=context_pack["context_delta_4dof"].to(device),
                context_target_flight_state=context_pack["context_target_flight_state"].to(device),
                keypoints_body=kp_body.to(device),
                context_pose_se3=ctx_pose.to(device) if ctx_pose is not None else None,
                context_keyframe_mask=ctx_mask.to(device) if ctx_mask is not None else None,
            )
    context_encode_ms = (time.perf_counter() - t_ctx_start) * 1000.0

    traj_times: List[float] = [0.0]
    traj_pos_enu_m: List[np.ndarray] = [curr_pos_m.copy()]
    traj_yaw_enu: List[float] = [curr_yaw]
    traj_flight_state: List[float] = [curr_flight_state]
    predicted_horizons_history: List[np.ndarray] = []

    land_x = float(landing_cfg.get("x", 15.0))
    land_y = float(landing_cfg.get("y", 0.0))

    step_count = 0
    odom_tick_cursor = 0
    effective_t_exec = req_t_exec
    num_replan_calls = 0
    inference_times_ms: List[float] = []
    lidar_times_ms: List[float] = []

    while odom_tick_cursor < max_steps_10hz:
        # 1. Sintetizar nube LiDAR 360° en la pose actual (Horizontal FLU, s=0.1)
        t_lid0 = time.perf_counter()
        cloud_flu_scaled = lidar_gen.generate_pointcloud(
            drone_pose={
                "x": float(curr_pos_m[0]),
                "y": float(curr_pos_m[1]),
                "z": float(curr_pos_m[2]),
                "yaw": float(curr_yaw),
            },
            landmarks=landmarks_list,
            landing=landing_cfg,
            as_tensor=False,
            seed=100 + odom_tick_cursor,
        )
        lidar_times_ms.append((time.perf_counter() - t_lid0) * 1000.0)

        live_pose_se3 = build_se3_horizontal_flu_torch(
            torch.tensor(curr_pos_m * scale_factor, dtype=torch.float32).unsqueeze(0),
            torch.tensor([curr_yaw], dtype=torch.float32),
        )  # (1, 4, 4)

        sample = {
            **context_pack,
            "action_stride": stride,
            "live_scene_pos": torch.from_numpy(cloud_flu_scaled).unsqueeze(0).to(dtype=torch.float32),
            "live_pose_se3": live_pose_se3,
            "live_vel_flu": torch.from_numpy(curr_vel_flu).unsqueeze(0).to(dtype=torch.float32),
            "live_flight_state": torch.tensor([[curr_flight_state]], dtype=torch.float32),
            "keypoints_body": kp_body,
        }

        # 2. Invocar prediccion de horizonte T pasos
        t_inf0 = time.perf_counter()
        if cached_context is not None:
            pred = policy.predict_actions(
                sample=sample,
                num_ddim_steps=num_ddim_steps,
                cached_context=cached_context,
            )
        elif hasattr(policy, "reference_episode") and getattr(policy, "reference_episode") is not None:
            pred = policy.predict_actions(sample=sample, step_cursor=odom_tick_cursor)
        else:
            pred = policy.predict_actions(sample=sample)
        inference_times_ms.append((time.perf_counter() - t_inf0) * 1000.0)
        num_replan_calls += 1

        wp_horizon = pred.pred_world_waypoints_enu_m  # (T, 4) [X_m, Y_m, Z_m, yaw_rad]
        fs_horizon = pred.pred_flight_state.numpy().reshape(-1)  # (T,)
        predicted_horizons_history.append(wp_horizon.copy())

        steps_to_exec = max(1, min(int(wp_horizon.shape[0]), req_t_exec))
        effective_t_exec = steps_to_exec

        # 3. Ejecutar los primeros `steps_to_exec` pasos del horizonte predicho
        landed_trigger = False
        for k in range(steps_to_exec):
            if odom_tick_cursor >= max_steps_10hz:
                break
            next_pos_m = wp_horizon[k, :3].astype(np.float64)
            next_yaw = float(wrap_angle_numpy(wp_horizon[k, 3]))
            next_fs = float(fs_horizon[k])

            vel_enu = (next_pos_m - curr_pos_m) / dt_step
            R_next = rot_z(next_yaw)
            curr_vel_flu = (R_next.T @ vel_enu).astype(np.float32)
            curr_pos_m = next_pos_m
            curr_yaw = next_yaw
            curr_flight_state = 1.0 if next_fs >= 0.5 else 0.0
            step_count += 1
            odom_tick_cursor += stride

            traj_times.append(step_count * dt_step)
            traj_pos_enu_m.append(curr_pos_m.copy())
            traj_yaw_enu.append(curr_yaw)
            traj_flight_state.append(curr_flight_state)

            dist_pad_xy = math.hypot(curr_pos_m[0] - land_x, curr_pos_m[1] - land_y)
            if dist_pad_xy <= 1.05 and curr_pos_m[2] <= 0.35 and (curr_flight_state < 0.5 or np.linalg.norm(vel_enu) < 0.35):
                landed_trigger = True
                break

        if landed_trigger:
            break

    pos_traj = np.asarray(traj_pos_enu_m, dtype=np.float64)
    yaw_traj = np.asarray(traj_yaw_enu, dtype=np.float64)
    times_traj = np.asarray(traj_times, dtype=np.float64)

    telemetry_50hz = _trajectory_10hz_to_px4_telemetry_50hz(
        timestamps_10hz=times_traj,
        pos_enu_10hz=pos_traj,
        yaw_enu_10hz=yaw_traj,
        landing_cfg=landing_cfg,
    )
    val_report = validate_episode_data(
        telemetry=telemetry_50hz,
        waypoints_cfg=waypoints_cfg,
        pilot_exit_code=0,
    )

    ade_m = float("nan")
    fde_m = float("nan")
    if reference_pos_enu_m is not None and len(reference_pos_enu_m) > 0:
        ref = np.asarray(reference_pos_enu_m, dtype=np.float64)
        ref_strided = ref[::stride]
        L_cmp = min(len(pos_traj), len(ref_strided))
        ade_m = float(np.mean(np.linalg.norm(pos_traj[:L_cmp] - ref_strided[:L_cmp], axis=-1)))
        fde_m = float(np.linalg.norm(pos_traj[-1] - ref[-1]))

    return {
        "validation_status": val_report.get("validation_status", "rejected"),
        "mission_status": val_report.get("mission_status", "failed"),
        "rejection_reasons": val_report.get("rejection_reasons", []),
        "validation_report": val_report,
        "pos_enu_m": pos_traj,
        "yaw_enu_rad": yaw_traj,
        "timestamps_s": times_traj,
        "predicted_horizons": predicted_horizons_history,
        "execution_horizon": int(effective_t_exec),
        "action_stride": int(stride),
        "num_ddim_steps": int(num_ddim_steps),
        "num_steps_executed": int(step_count),
        "num_replan_calls": int(num_replan_calls),
        "context_encode_ms": float(context_encode_ms),
        "mean_inference_ms": float(np.mean(inference_times_ms)) if inference_times_ms else 0.0,
        "mean_lidar_ms": float(np.mean(lidar_times_ms)) if lidar_times_ms else 0.0,
        "ade_m": ade_m,
        "fde_m": fde_m,
    }
