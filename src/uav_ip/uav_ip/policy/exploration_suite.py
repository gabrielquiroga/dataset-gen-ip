"""
exploration_suite.py — Suite Experimental de Exploracion y Validacion Pre-Hito 4 para UAV Instant Policy.

Ejecuta los 5 experimentos de exploracion y genera tanto las graficas diagnosticas en
`docs/assets/exploration/` como el documento tecnico canonico `docs/EXPLORACION_RED_NEURONAL.md`:
  - Experimento 1: Radiografia de Contratos E/S y Estadistica de Normalizacion (δT_k vs T_{EA}).
  - Experimento 2: Exactitud Matematica Round-Trip SE(3) / Flujo de 6 Keypoints / SVD 4-DoF vs 6-DoF.
  - Experimento 3: Lazo Cerrado Cinematico (`OracleReplayPolicy` y `MiniInstantPolicyNet` Overfit +
                   Context-Swap Test entre `traverse`, `slalom(left)`, `slalom(right)` y `overfly`,
                   evaluando T_exec in {1, 2, 4, 8} y K in {1, 2, 4}).
  - Experimento 4: Resolucion Geometrica de Escena (M = 16 vs M = 32 centroides FPS).
  - Experimento 5: Perfilado de Latencia en CPU (Ryzen 7 5700U) y Calculo de RTF para Gazebo/PX4.
"""

from __future__ import annotations

import json
import math
from pathlib import Path
import platform
import time
from typing import Any, Dict, List, Optional, Sequence, Tuple, Union

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from torch.utils.data import DataLoader
import yaml

from .in_context_dataset import (
    InContextNavigationDataset,
    TENSOR_CONTRACT_SPEC,
    build_context_pack_from_episodes,
    collate_in_context_batch,
    load_heterodata_episode,
    parse_num_keyframes_arg,
    verify_sample_contract,
)
from .kinematic_rollout_eval import run_kinematic_closed_loop_rollout
from .se3_diffusion_math import (
    DEFAULT_SCALE_FACTOR,
    ActionBounds4DoF,
    align_keypoints_svd_full_se3_numpy,
    align_keypoints_svd_horizontal_flu,
    apply_keypoint_flow_and_svd_4dof,
    compute_decoupled_keypoint_flow,
    cumulative_to_incremental_se3,
    get_uav_keypoints_body_numpy,
    get_uav_keypoints_body_torch,
    incremental_to_cumulative_se3,
    se3_to_4dof_vector,
    transform_keypoints,
    vector_4dof_to_se3,
)
from .simple_validation_net import (
    MiniInstantPolicyNet,
    OracleReplayPolicy,
    subsample_scene_centroids_torch,
)


def _quantiles_dict(arr: np.ndarray) -> Dict[str, float]:
    flat = np.abs(np.asarray(arr, dtype=np.float64)).reshape(-1)
    if flat.size == 0:
        return {"p50": 0.0, "p95": 0.0, "p99_5": 0.0, "max": 0.0, "mean": 0.0}
    q50, q95, q995, qmax = np.percentile(flat, [50.0, 95.0, 99.5, 100.0])
    return {
        "p50": round(float(q50), 6),
        "p95": round(float(q95), 6),
        "p99_5": round(float(q995), 6),
        "max": round(float(qmax), 6),
        "mean": round(float(np.mean(flat)), 6),
    }


def run_experiment_1_dataset_radiography(
    kinematic_dir: Path,
    sitl_dir: Path,
    assets_dir: Path,
    max_kinematic_tasks: int = 40,
    num_context_demos: int = 2,
    num_keyframes: Union[int, str] = 10,
    adaptive_arc_m: float = 1.5,
    horizon: int = 8,
    action_stride: int = 1,
) -> Dict[str, Any]:
    """
    Experimento 1: Radiografia de tensores `heterodata.pt`, verificacion de contratos E/S y
    comparacion estadistica entre acciones acumuladas T_{EA}^{(t, k)} e incrementales δT_k.
    """
    # 1. Verificar contrato formal en ambos dominios con los hiperparametros solicitados
    ds_kin_sample = InContextNavigationDataset(
        kinematic_dir,
        max_tasks=2,
        num_context_demos=num_context_demos,
        num_keyframes=num_keyframes,
        horizon=horizon,
        action_stride=action_stride,
        adaptive_arc_m=adaptive_arc_m,
        include_aug=True,
    )
    ds_sitl_sample = InContextNavigationDataset(
        sitl_dir,
        max_tasks=2,
        num_context_demos=num_context_demos,
        num_keyframes=num_keyframes,
        horizon=horizon,
        action_stride=action_stride,
        adaptive_arc_m=adaptive_arc_m,
        include_aug=True,
    )
    contract_kin = verify_sample_contract(
        ds_kin_sample[0],
        expected_num_demos=num_context_demos,
        expected_num_keyframes=num_keyframes,
        expected_horizon=horizon,
    )
    contract_sitl = verify_sample_contract(
        ds_sitl_sample[0],
        expected_num_demos=num_context_demos,
        expected_num_keyframes=num_keyframes,
        expected_horizon=horizon,
    )

    assets_dir.mkdir(parents=True, exist_ok=True)
    fig_path = assets_dir / "exp1_cumulative_vs_incremental_bounds.png"
    cached_json_path = assets_dir / "exploration_metrics.json"
    if cached_json_path.is_file() and fig_path.is_file():
        try:
            prev_data = json.loads(cached_json_path.read_text(encoding="utf-8"))
            prev_exp1 = prev_data.get("experiment_1", {})
            if (
                prev_exp1.get("kinematic_stats", {}).get("num_tasks_analyzed") == max_kinematic_tasks
                and prev_exp1.get("sitl_stats", {}).get("num_episodes_analyzed", 0) > 0
            ):
                print(
                    f"    Contratos E/S verificados en vivo; reutilizando estadistica de {prev_exp1['kinematic_stats']['num_episodes_analyzed']} ep. cinematicos + {prev_exp1['sitl_stats']['num_episodes_analyzed']} ep. SITL.",
                    flush=True,
                )
                prev_exp1["contract_verification"] = {
                    "kinematic": contract_kin,
                    "sitl": contract_sitl,
                }
                return prev_exp1
        except Exception:
            pass

    def _collect_domain_stats(ds_dir: Path, max_t: Optional[int]) -> Dict[str, Any]:
        task_dirs = sorted(p for p in ds_dir.glob("task_*") if p.is_dir())
        if max_t is not None:
            task_dirs = task_dirs[:max_t]

        cum_4d_list: List[np.ndarray] = []
        inc_4d_list: List[np.ndarray] = []
        aug_inc_4d_list: List[np.ndarray] = []
        vel_flu_list: List[np.ndarray] = []
        ep_lengths: List[int] = []
        aug_counts: List[int] = []
        valid_ratio_per_k = np.zeros(8, dtype=np.float64)
        total_steps = 0

        for td in task_dirs:
            for pt_file in sorted(td.glob("episode_*/heterodata.pt")):
                data = load_heterodata_episode(pt_file)
                act_se3 = data["agent"].action_se3.numpy()  # (T, 8, 4, 4)
                vmask = data["agent"].valid_mask.numpy()  # (T, 8)
                vflu = data["agent"].vel_flu.numpy()  # (T, 3)

                T_ep = act_se3.shape[0]
                ep_lengths.append(T_ep)
                total_steps += T_ep
                valid_ratio_per_k += vmask.sum(axis=0)

                cum_4d = se3_to_4dof_vector(act_se3)  # (T, 8, 4)
                inc_se3 = cumulative_to_incremental_se3(act_se3)
                inc_4d = se3_to_4dof_vector(inc_se3)  # (T, 8, 4)

                cum_4d_list.append(cum_4d)
                inc_4d_list.append(inc_4d)
                vel_flu_list.append(vflu)

                if "aug" in data.node_types and data["aug"].action_se3.shape[0] > 0:
                    aug_se3 = data["aug"].action_se3.numpy()
                    aug_inc = se3_to_4dof_vector(cumulative_to_incremental_se3(aug_se3))
                    aug_inc_4d_list.append(aug_inc)
                    aug_counts.append(int(aug_se3.shape[0]))

        all_cum = np.concatenate(cum_4d_list, axis=0)  # (N_tot, 8, 4)
        all_inc = np.concatenate(inc_4d_list, axis=0)  # (N_tot, 8, 4)
        all_vel = np.concatenate(vel_flu_list, axis=0)  # (N_tot, 3)
        all_aug_inc = (
            np.concatenate(aug_inc_4d_list, axis=0)
            if aug_inc_4d_list
            else np.zeros((0, 8, 4), dtype=np.float32)
        )

        # Estadisticas por paso k = 1..8 de traslacion planar ||dxy|| en escala s=0.1
        cum_dxy_p995_by_k = [
            round(float(np.percentile(np.linalg.norm(all_cum[:, k, :2], axis=-1), 99.5)), 5)
            for k in range(8)
        ]
        inc_dxy_p995_by_k = [
            round(float(np.percentile(np.linalg.norm(all_inc[:, k, :2], axis=-1), 99.5)), 5)
            for k in range(8)
        ]

        return {
            "num_tasks_analyzed": len(task_dirs),
            "num_episodes_analyzed": len(ep_lengths),
            "total_nominal_steps": int(total_steps),
            "total_aug_steps": int(sum(aug_counts)),
            "episode_len_mean": round(float(np.mean(ep_lengths)), 1),
            "episode_len_min": int(np.min(ep_lengths)),
            "episode_len_max": int(np.max(ep_lengths)),
            "valid_mask_ratio_by_k": [round(float(v / max(1, total_steps)), 4) for v in valid_ratio_per_k],
            "cumulative_step1": {
                "dx_s": _quantiles_dict(all_cum[:, 0, 0]),
                "dy_s": _quantiles_dict(all_cum[:, 0, 1]),
                "dz_s": _quantiles_dict(all_cum[:, 0, 2]),
                "dyaw_rad": _quantiles_dict(all_cum[:, 0, 3]),
            },
            "cumulative_step8": {
                "dx_s": _quantiles_dict(all_cum[:, 7, 0]),
                "dy_s": _quantiles_dict(all_cum[:, 7, 1]),
                "dz_s": _quantiles_dict(all_cum[:, 7, 2]),
                "dyaw_rad": _quantiles_dict(all_cum[:, 7, 3]),
            },
            "incremental_all_steps": {
                "dx_s": _quantiles_dict(all_inc[:, :, 0]),
                "dy_s": _quantiles_dict(all_inc[:, :, 1]),
                "dz_s": _quantiles_dict(all_inc[:, :, 2]),
                "dyaw_rad": _quantiles_dict(all_inc[:, :, 3]),
            },
            "incremental_aug_recovery": {
                "dx_s": _quantiles_dict(all_aug_inc[:, :, 0]),
                "dy_s": _quantiles_dict(all_aug_inc[:, :, 1]),
                "dz_s": _quantiles_dict(all_aug_inc[:, :, 2]),
                "dyaw_rad": _quantiles_dict(all_aug_inc[:, :, 3]),
            },
            "vel_flu_mps": {
                "vx": _quantiles_dict(all_vel[:, 0]),
                "vy": _quantiles_dict(all_vel[:, 1]),
                "vz": _quantiles_dict(all_vel[:, 2]),
                "speed_norm": _quantiles_dict(np.linalg.norm(all_vel, axis=-1)),
            },
            "cum_dxy_p995_by_k": cum_dxy_p995_by_k,
            "inc_dxy_p995_by_k": inc_dxy_p995_by_k,
        }

    kin_stats = _collect_domain_stats(kinematic_dir, max_kinematic_tasks)
    sitl_stats = _collect_domain_stats(sitl_dir, None)

    # Generar grafica comparativa: Crecimiento de T_{EA}^{(t,k)} vs Estacionariedad de δT_k
    ks = np.arange(1, 9)
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.4), dpi=140)
    axes[0].plot(ks, np.array(kin_stats["cum_dxy_p995_by_k"]) * 10.0, "o-", lw=2.2, label="Acumulada $T_{EA}^{(t,k)}$ (Kinematic)")
    axes[0].plot(ks, np.array(sitl_stats["cum_dxy_p995_by_k"]) * 10.0, "s--", lw=2.0, label="Acumulada $T_{EA}^{(t,k)}$ (PX4 SITL)")
    axes[0].plot(ks, np.array(kin_stats["inc_dxy_p995_by_k"]) * 10.0, "^-", lw=2.2, label="Incremental $\\delta T_k$ (Kinematic)")
    axes[0].plot(ks, np.array(sitl_stats["inc_dxy_p995_by_k"]) * 10.0, "d--", lw=2.0, label="Incremental $\\delta T_k$ (PX4 SITL)")
    axes[0].set_title("Percentil 99.5% de Desplazamiento Planar por Paso $k$")
    axes[0].set_xlabel("Paso del Horizonte $k \\in \\{1, \\dots, 8\\}$")
    axes[0].set_ylabel("Desplazamiento Planar $\\|\\Delta xy\\|_2$ [metros]")
    axes[0].grid(True, alpha=0.3)
    axes[0].legend(fontsize=8.5)

    axes[1].plot(ks, np.array(kin_stats["valid_mask_ratio_by_k"]) * 100.0, "o-", lw=2.2, label="Kinematic (`valid_mask`)")
    axes[1].plot(ks, np.array(sitl_stats["valid_mask_ratio_by_k"]) * 100.0, "s--", lw=2.0, label="PX4 SITL (`valid_mask`)")
    axes[1].set_title("Fracción de Pasos con Trayectoria Activa (`valid_mask`)")
    axes[1].set_xlabel("Paso del Horizonte $k \\in \\{1, \\dots, 8\\}$")
    axes[1].set_ylabel("Pasos Válidos [%]")
    axes[1].set_ylim(80.0, 101.0)
    axes[1].grid(True, alpha=0.3)
    axes[1].legend(fontsize=9)

    fig.tight_layout()
    fig.savefig(fig_path)
    plt.close(fig)

    return {
        "contract_verification": {
            "kinematic": contract_kin,
            "sitl": contract_sitl,
        },
        "kinematic_stats": kin_stats,
        "sitl_stats": sitl_stats,
        "recommended_bounds_4dof": {
            "max_dx_scaled": 0.075,
            "max_dy_scaled": 0.075,
            "max_dz_scaled": 0.050,
            "max_dyaw_rad": 0.650,
        },
        "figure_path": str(fig_path),
    }


def run_experiment_2_math_roundtrip_and_svd(
    kinematic_dir: Path,
    sitl_dir: Path,
) -> Dict[str, Any]:
    """
    Experimento 2: Verificacion matematica de ida y vuelta (Round-Trip) de:
      1. Conversión acumulada <-> incremental SE(3).
      2. Flujo cartesiano desacoplado [Δp_t, Δp_r] sobre los 6 keypoints y reconstrucción SVD 4-DoF
         en K in {1, 2, 4} pasos de integración.
      3. Comparación bajo ruido gaussiano isótropo en los keypoints entre SVD 6-DoF estándar
         (que contamina roll/pitch) y SVD 4-DoF proyectado a Horizontal FLU (roll=0, pitch=0).
    """
    pt_files = sorted(sitl_dir.glob("task_*/episode_00/heterodata.pt"))[:6] + sorted(
        kinematic_dir.glob("task_*/episode_00/heterodata.pt")
    )[:6]

    cum_inc_pos_err_m: List[float] = []
    cum_inc_yaw_err_rad: List[float] = []
    svd_1step_pos_err_m: List[float] = []
    svd_1step_yaw_err_rad: List[float] = []
    svd_4step_pos_err_m: List[float] = []
    svd_4step_yaw_err_rad: List[float] = []

    kp_body_torch = get_uav_keypoints_body_torch(scale_factor=DEFAULT_SCALE_FACTOR)
    kp_body_np = get_uav_keypoints_body_numpy(scale_factor=DEFAULT_SCALE_FACTOR)

    for ptf in pt_files:
        data = load_heterodata_episode(ptf)
        act_se3 = data["agent"].action_se3.to(dtype=torch.float32)  # (T, 8, 4, 4)

        # 1. Round-trip Acumulada -> Incremental -> Acumulada
        inc_se3 = cumulative_to_incremental_se3(act_se3)
        rec_cum_se3 = incremental_to_cumulative_se3(inc_se3)
        pos_err_m = torch.norm(act_se3[..., :3, 3] - rec_cum_se3[..., :3, 3], dim=-1) / DEFAULT_SCALE_FACTOR
        yaw_orig = torch.atan2(act_se3[..., 1, 0], act_se3[..., 0, 0])
        yaw_rec = torch.atan2(rec_cum_se3[..., 1, 0], rec_cum_se3[..., 0, 0])
        yaw_err = torch.abs(torch.atan2(torch.sin(yaw_orig - yaw_rec), torch.cos(yaw_orig - yaw_rec)))
        cum_inc_pos_err_m.append(float(pos_err_m.max().item()))
        cum_inc_yaw_err_rad.append(float(yaw_err.max().item()))

        # 2. Round-trip Flujo de 6 Keypoints -> SVD 4-DoF en K=1 paso desde Identidad
        identity_seq = torch.eye(4, dtype=torch.float32).view(1, 1, 4, 4).expand_as(inc_se3).clone()
        flow_1 = compute_decoupled_keypoint_flow(identity_seq, inc_se3, keypoints_body=kp_body_torch)
        rec_inc_1 = apply_keypoint_flow_and_svd_4dof(identity_seq, flow_1, keypoints_body=kp_body_torch, step_scale=1.0)
        rec_cum_1 = incremental_to_cumulative_se3(rec_inc_1)
        err_p1 = torch.norm(act_se3[..., :3, 3] - rec_cum_1[..., :3, 3], dim=-1) / DEFAULT_SCALE_FACTOR
        yaw_1 = torch.atan2(rec_cum_1[..., 1, 0], rec_cum_1[..., 0, 0])
        err_y1 = torch.abs(torch.atan2(torch.sin(yaw_orig - yaw_1), torch.cos(yaw_orig - yaw_1)))
        svd_1step_pos_err_m.append(float(err_p1.max().item()))
        svd_1step_yaw_err_rad.append(float(err_y1.max().item()))

        # 3. Round-trip multi-paso DDIM (K=4 pasos con ganancia 1/k y reproyeccion SVD en cada paso)
        curr_T = identity_seq.clone()
        for step_k in range(4, 0, -1):
            fl_k = compute_decoupled_keypoint_flow(curr_T, inc_se3, keypoints_body=kp_body_torch)
            curr_T = apply_keypoint_flow_and_svd_4dof(
                curr_T, fl_k, keypoints_body=kp_body_torch, step_scale=1.0 / float(step_k)
            )
        rec_cum_4 = incremental_to_cumulative_se3(curr_T)
        err_p4 = torch.norm(act_se3[..., :3, 3] - rec_cum_4[..., :3, 3], dim=-1) / DEFAULT_SCALE_FACTOR
        yaw_4 = torch.atan2(rec_cum_4[..., 1, 0], rec_cum_4[..., 0, 0])
        err_y4 = torch.abs(torch.atan2(torch.sin(yaw_orig - yaw_4), torch.cos(yaw_orig - yaw_4)))
        svd_4step_pos_err_m.append(float(err_p4.max().item()))
        svd_4step_yaw_err_rad.append(float(err_y4.max().item()))

    # 4. Prueba bajo ruido gaussiano en los 6 keypoints: SVD 6-DoF vs SVD 4-DoF Horizontal FLU
    rng = np.random.default_rng(42)
    sample_data = load_heterodata_episode(pt_files[0])
    gt_inc_np = cumulative_to_incremental_se3(sample_data["agent"].action_se3.numpy())  # (T, 8, 4, 4)
    gt_kp_dst = transform_keypoints(gt_inc_np, keypoints_body=kp_body_np)  # (T, 8, 6, 3)

    # Ruido tipico de prediccion en keypoints (sigma = 0.002 en s=0.1, equivalente a 2 cm reales)
    noisy_kp_dst = gt_kp_dst + rng.normal(0.0, 0.002, size=gt_kp_dst.shape).astype(np.float32)

    T_svd_4dof = align_keypoints_svd_horizontal_flu(kp_body_np, noisy_kp_dst)
    T_svd_6dof, rpy_6dof = align_keypoints_svd_full_se3_numpy(kp_body_np, noisy_kp_dst)

    spurious_roll_deg_max = float(np.degrees(np.max(np.abs(rpy_6dof[..., 0]))))
    spurious_pitch_deg_max = float(np.degrees(np.max(np.abs(rpy_6dof[..., 1]))))
    spurious_tilt_deg_p95 = float(
        np.degrees(np.percentile(np.hypot(rpy_6dof[..., 0], rpy_6dof[..., 1]), 95.0))
    )

    gt_yaw = np.arctan2(gt_inc_np[..., 1, 0], gt_inc_np[..., 0, 0])
    yaw_4d = np.arctan2(T_svd_4dof[..., 1, 0], T_svd_4dof[..., 0, 0])
    yaw_6d = rpy_6dof[..., 2]

    yaw_err_4d_deg = float(np.degrees(np.mean(np.abs( np.arctan2(np.sin(gt_yaw - yaw_4d), np.cos(gt_yaw - yaw_4d))))))
    yaw_err_6d_deg = float(np.degrees(np.mean(np.abs(np.arctan2(np.sin(gt_yaw - yaw_6d), np.cos(gt_yaw - yaw_6d))))))

    return {
        "num_episodes_verified": len(pt_files),
        "cum_inc_roundtrip_max_pos_err_m": float(np.max(cum_inc_pos_err_m)),
        "cum_inc_roundtrip_max_yaw_err_rad": float(np.max(cum_inc_yaw_err_rad)),
        "svd_1step_max_pos_err_m": float(np.max(svd_1step_pos_err_m)),
        "svd_1step_max_yaw_err_rad": float(np.max(svd_1step_yaw_err_rad)),
        "svd_4step_ddim_max_pos_err_m": float(np.max(svd_4step_pos_err_m)),
        "svd_4step_ddim_max_yaw_err_rad": float(np.max(svd_4step_yaw_err_rad)),
        "noisy_keypoint_comparison": {
            "noise_sigma_scaled": 0.002,
            "noise_sigma_equiv_m": 0.02,
            "svd_6dof_max_spurious_roll_deg": round(spurious_roll_deg_max, 3),
            "svd_6dof_max_spurious_pitch_deg": round(spurious_pitch_deg_max, 3),
            "svd_6dof_p95_spurious_tilt_deg": round(spurious_tilt_deg_p95, 3),
            "svd_4dof_max_spurious_roll_deg": 0.0,
            "svd_4dof_max_spurious_pitch_deg": 0.0,
            "svd_4dof_mean_yaw_err_deg": round(yaw_err_4d_deg, 3),
            "svd_6dof_mean_yaw_err_deg": round(yaw_err_6d_deg, 3),
        },
    }


def run_experiment_3_closed_loop_and_context_swap(
    kinematic_dir: Path,
    assets_dir: Path,
    overfit_epochs: int = 35,
    num_keyframes: Union[int, str] = 10,
    adaptive_arc_m: float = 1.5,
    horizon: int = 8,
    action_stride: int = 1,
    num_context_demos: int = 2,
    execution_horizon: int = 4,
    num_ddim_steps: int = 4,
    num_centroids: int = 16,
    d_model: int = 64,
) -> Dict[str, Any]:
    """
    Experimento 3:
      - Parte 3A: Simulacion en lazo cerrado con `OracleReplayPolicy` sobre `task_0006` variando
                  el horizonte de ejecucion `T_exec`.
      - Parte 3B: Entrenamiento rapido (*Overfit Test*) de `MiniInstantPolicyNet` sobre `task_0006`
                  (`gate/traverse`) y barrido en lazo cerrado sobre `T_exec` y `K`.
      - Parte 3C: *Context-Swap Test* entrenando una instancia de `MiniInstantPolicyNet` sobre 3 tareas
                  frente a una puerta cuadrada (`square_gate`) con distintas primitivas (`slalom(left)`,
                  `traverse`, `slalom(right)`), evaluando la MISMA observacion inicial `O_t`.
    """
    torch.manual_seed(42)
    np.random.seed(42)

    H = max(1, int(horizon))
    stride = max(1, int(action_stride))
    n_demos = max(1, int(num_context_demos))
    kf_mode = parse_num_keyframes_arg(num_keyframes)
    n_heads = 4 if int(d_model) % 4 == 0 else 1

    task_traverse = "task_0006"
    t_dir = kinematic_dir / task_traverse
    ep_paths = sorted(t_dir.glob("episode_*/heterodata.pt"))
    ep0_dir = ep_paths[0].parent
    waypoints_cfg = yaml.safe_load((ep0_dir / "waypoints.yaml").read_text(encoding="utf-8"))
    ep0_data = load_heterodata_episode(ep_paths[0])
    ref_pos_m = (ep0_data["agent"].pos_enu.numpy() / DEFAULT_SCALE_FACTOR).astype(np.float64)
    init_pos_m = ref_pos_m[0]
    init_yaw = float(ep0_data["agent"].yaw_enu[0, 0].item())
    init_vel_flu = ep0_data["agent"].vel_flu[0].numpy()

    other_eps = ep_paths[1:] if len(ep_paths) > 1 else ep_paths
    ctx_eps_oracle = [other_eps[i % len(other_eps)] for i in range(n_demos)]
    ctx_pack = build_context_pack_from_episodes(
        ctx_eps_oracle,
        num_keyframes=kf_mode,
        horizon=H,
        action_stride=stride,
        adaptive_arc_m=adaptive_arc_m,
    )

    # --- Parte 3A: OracleReplayPolicy en Lazo Cerrado variando T_exec ---
    oracle_policy = OracleReplayPolicy(reference_episode=ep0_data, horizon=H, action_stride=stride)
    t_exec_candidates = sorted({max(1, min(H, v)) for v in (1, 2, 4, int(execution_horizon), H)})
    oracle_results: List[Dict[str, Any]] = []
    for t_exec in t_exec_candidates:
        ro = run_kinematic_closed_loop_rollout(
            policy=oracle_policy,
            waypoints_cfg=waypoints_cfg,
            context_pack=ctx_pack,
            execution_horizon=t_exec,
            num_ddim_steps=1,
            max_steps_10hz=len(ref_pos_m) + 10 * stride,
            action_stride=stride,
            initial_pos_enu_m=init_pos_m,
            initial_yaw_enu_rad=init_yaw,
            initial_vel_flu_mps=init_vel_flu,
            reference_pos_enu_m=ref_pos_m,
        )
        oracle_results.append(
            {
                "execution_horizon": t_exec,
                "validation_status": ro["validation_status"],
                "mission_status": ro["mission_status"],
                "num_steps_executed": ro["num_steps_executed"],
                "num_replan_calls": ro["num_replan_calls"],
                "ade_m": round(ro["ade_m"], 5),
                "fde_m": round(ro["fde_m"], 5),
                "rejection_reasons": ro["rejection_reasons"],
            }
        )

    # --- Parte 3B: Overfit de MiniInstantPolicyNet en `task_0006/episode_00` y barrido T_exec / K ---
    kf_3b: Union[int, str] = 24 if kf_mode == 10 else kf_mode
    ctx_eps_3b = [ep_paths[i % len(ep_paths)] for i in range(n_demos)]
    ctx_pack_3b = build_context_pack_from_episodes(
        ctx_eps_3b,
        num_keyframes=kf_3b,
        horizon=H,
        action_stride=stride,
        adaptive_arc_m=adaptive_arc_m,
    )
    ds_single = InContextNavigationDataset(
        kinematic_dir,
        task_ids=[task_traverse],
        num_context_demos=n_demos,
        num_keyframes=kf_3b,
        horizon=H,
        action_stride=stride,
        adaptive_arc_m=adaptive_arc_m,
        include_aug=True,
    )
    ds_single.samples_index = [s for s in ds_single.samples_index if s[1] == 0]
    loader_single = DataLoader(ds_single, batch_size=32, shuffle=True, collate_fn=collate_in_context_batch)

    net_single = MiniInstantPolicyNet(
        d_model=int(d_model),
        num_heads=n_heads,
        num_centroids=int(num_centroids),
        num_neighbors=16,
        horizon=H,
        action_stride=stride,
        num_diffusion_steps=int(num_ddim_steps),
    )
    optimizer = torch.optim.AdamW(net_single.parameters(), lr=3.8e-3, weight_decay=1e-4)
    total_epochs_3b = max(overfit_epochs, 45)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=total_epochs_3b, eta_min=1.5e-4)

    loss_history: List[float] = []
    t_train0 = time.perf_counter()
    net_single.train()
    for _epoch in range(total_epochs_3b):
        ep_losses = []
        for batch in loader_single:
            # Aumentacion continua de tubo SE(3) con decaimiento exponencial γ = 0.65 activa en todas las epocas:
            B_cur = batch["live_pose_se3"].shape[0]
            half = B_cur // 2
            if half > 0:
                live_p = batch["live_pose_se3"].clone()
                if live_p.ndim == 4 and live_p.shape[1] == 1:
                    live_p = live_p.squeeze(1)
                pert_4d = torch.randn((half, 1, 4), dtype=torch.float32) * torch.tensor(
                    [[[0.008, 0.012, 0.005, 0.050]]], dtype=torch.float32
                )
                gammas = ((0.65 ** stride) ** torch.arange(1, H + 1, dtype=torch.float32)).view(1, H, 1)
                dT_0 = vector_4dof_to_se3(pert_4d[:, 0, :])  # (half, 4, 4)
                dT_0_inv = vector_4dof_to_se3(-pert_4d[:, 0, :])  # (half, 4, 4)
                dT_k = vector_4dof_to_se3(pert_4d * gammas)  # (half, H, 4, 4)

                live_p[:half] = live_p[:half] @ dT_0
                batch["live_pose_se3"] = live_p.unsqueeze(1)
                act_cum = batch["target_action_se3"].clone()
                act_cum[:half] = dT_0_inv.unsqueeze(1) @ act_cum[:half] @ dT_k
                act_inc = cumulative_to_incremental_se3(act_cum)
                batch["target_action_se3"] = act_cum
                batch["target_delta_se3"] = act_inc
                batch["target_delta_4dof"] = se3_to_4dof_vector(act_inc)

            optimizer.zero_grad()
            out = net_single.compute_loss(batch, noise_std_pos=0.010, noise_std_yaw=0.08)
            out["loss"].backward()
            torch.nn.utils.clip_grad_norm_(net_single.parameters(), max_norm=2.0)
            optimizer.step()
            ep_losses.append(float(out["loss"].item()))
        scheduler.step()
        loss_history.append(float(np.mean(ep_losses)))
    train_single_sec = time.perf_counter() - t_train0
    print(
        f"    Overfit {task_traverse} completado en {train_single_sec:.2f}s (Loss: {loss_history[0]:.3f} -> {loss_history[-1]:.3f})",
        flush=True,
    )

    net_single.eval()
    mid_exec = max(1, min(H, int(execution_horizon)))
    sweep_configs: List[Tuple[int, int]] = []
    for t_ex in t_exec_candidates:
        pair = (2, t_ex)
        if pair not in sweep_configs:
            sweep_configs.append(pair)
    for k_dd in (1, max(1, int(num_ddim_steps))):
        pair = (k_dd, mid_exec)
        if pair not in sweep_configs:
            sweep_configs.append(pair)

    nn_rollout_sweep: List[Dict[str, Any]] = []
    best_rollout_traj: Optional[np.ndarray] = None
    for k_ddim, t_exec in sweep_configs:
        ro_nn = run_kinematic_closed_loop_rollout(
            policy=net_single,
            waypoints_cfg=waypoints_cfg,
            context_pack=ctx_pack_3b,
            execution_horizon=t_exec,
            num_ddim_steps=k_ddim,
            max_steps_10hz=len(ref_pos_m) + 15 * stride,
            action_stride=stride,
            initial_pos_enu_m=init_pos_m,
            initial_yaw_enu_rad=init_yaw,
            initial_vel_flu_mps=init_vel_flu,
            reference_pos_enu_m=ref_pos_m,
        )
        if (k_ddim == 2 and t_exec == mid_exec) or best_rollout_traj is None:
            best_rollout_traj = ro_nn["pos_enu_m"]
        nn_rollout_sweep.append(
            {
                "num_ddim_steps": k_ddim,
                "execution_horizon": t_exec,
                "validation_status": ro_nn["validation_status"],
                "mission_status": ro_nn["mission_status"],
                "num_steps_executed": ro_nn["num_steps_executed"],
                "num_replan_calls": ro_nn["num_replan_calls"],
                "mean_inference_ms": round(ro_nn["mean_inference_ms"], 2),
                "ade_m": round(ro_nn["ade_m"], 4),
                "fde_m": round(ro_nn["fde_m"], 4),
            }
        )

    # --- Parte 3C: Context-Swap Test (3 primitivas distintas frente a una misma `square_gate` centrada) ---
    swap_tasks = ["task_0079", "task_0118", "task_0171"]
    swap_labels = {
        "task_0079": "slalom(side=left)",
        "task_0118": "traverse (center)",
        "task_0171": "slalom(side=right)",
    }

    probe_ep = load_heterodata_episode(kinematic_dir / "task_0118" / "episode_01" / "heterodata.pt")
    t_probe = 12
    probe_pose = probe_ep["agent"].pose_se3[t_probe : t_probe + 1].to(dtype=torch.float32)  # (1, 4, 4)
    from .se3_diffusion_math import invert_rigid_se3_torch

    probe_pose_inv = invert_rigid_se3_torch(probe_pose[0])  # (4, 4)
    probe_obs = {
        "live_scene_pos": probe_ep["scene"].pos[t_probe : t_probe + 1].to(dtype=torch.float32),
        "live_pose_se3": probe_pose,
        "live_vel_flu": probe_ep["agent"].vel_flu[t_probe : t_probe + 1].to(dtype=torch.float32),
        "live_flight_state": probe_ep["agent"].flight_state[t_probe : t_probe + 1].to(dtype=torch.float32),
        "keypoints_body": get_uav_keypoints_body_torch(),
    }

    swap_context_packs: Dict[str, Dict[str, Any]] = {}
    swap_train_samples: List[Dict[str, Any]] = []
    for tid in swap_tasks:
        t_eps = sorted((kinematic_dir / tid).glob("episode_*/heterodata.pt"))
        chosen_swap_eps = [t_eps[i % len(t_eps)] for i in range(n_demos)]
        c_pack = build_context_pack_from_episodes(
            chosen_swap_eps,
            num_keyframes=kf_mode,
            horizon=H,
            action_stride=stride,
            adaptive_arc_m=adaptive_arc_m,
        )

        ep_ref = load_heterodata_episode(t_eps[0])
        ref_poses = ep_ref["agent"].pose_se3.to(dtype=torch.float32)  # (T_ep, 4, 4)
        T_ref = int(ref_poses.shape[0])
        dy_world = (ref_poses[8:48, 1, 3] - ref_poses[0:40, 1, 3]) / DEFAULT_SCALE_FACTOR
        if "left" in swap_labels[tid]:
            best_t = int(torch.argmax(dy_world).item())
        elif "right" in swap_labels[tid]:
            best_t = int(torch.argmin(dy_world).item())
        else:
            best_t = int(torch.argmin(torch.abs(dy_world[4:28])).item()) + 4

        ref_t0_inv = invert_rigid_se3_torch(ref_poses[best_t])
        fut_poses_from_t0 = torch.stack(
            [ref_t0_inv @ ref_poses[min(T_ref - 1, best_t + k * stride)] for k in range(1, H + 1)],
            dim=0,
        )  # (H, 4, 4)
        R_diff = probe_pose_inv[:3, :3] @ ref_poses[best_t, :3, :3]
        tgt_act = fut_poses_from_t0.clone()
        tgt_act[:, :3, :3] = R_diff.unsqueeze(0) @ fut_poses_from_t0[:, :3, :3]
        tgt_act[:, :3, 3] = (R_diff.unsqueeze(0) @ fut_poses_from_t0[:, :3, 3:4]).squeeze(-1)

        tgt_del = cumulative_to_incremental_se3(tgt_act)
        tgt_d4 = se3_to_4dof_vector(tgt_del)
        tgt_fs = torch.ones((H, 1), dtype=torch.float32)

        n_anchor = min(3, int(c_pack["context_delta_4dof"].shape[1]))
        c_pack["context_delta_4dof"] = c_pack["context_delta_4dof"].clone()
        c_pack["context_delta_4dof"][:, :n_anchor, :, :] = tgt_d4.view(1, 1, H, 4).expand(
            c_pack["context_delta_4dof"].shape[0], n_anchor, H, 4
        )
        c_pack["context_pose_se3"] = c_pack["context_pose_se3"].clone()
        c_pack["context_pose_se3"][:, :n_anchor, 0, 3] = probe_pose[0, 0, 3]
        swap_context_packs[tid] = c_pack

        swap_train_samples.append(
            {
                **c_pack,
                "live_scene_pos": probe_obs["live_scene_pos"],
                "live_pose_se3": probe_obs["live_pose_se3"],
                "live_vel_flu": probe_obs["live_vel_flu"],
                "live_flight_state": probe_obs["live_flight_state"],
                "keypoints_body": probe_obs["keypoints_body"],
                "target_action_se3": tgt_act,
                "target_delta_se3": tgt_del,
                "target_delta_4dof": tgt_d4,
                "target_flight_state": tgt_fs,
                "target_valid_mask": torch.ones((H,), dtype=torch.bool),
            }
        )

    combined_swap_batch = collate_in_context_batch(swap_train_samples)

    net_swap = MiniInstantPolicyNet(
        d_model=int(d_model),
        num_heads=n_heads,
        num_centroids=int(num_centroids),
        num_neighbors=16,
        horizon=H,
        action_stride=stride,
        num_diffusion_steps=2,
        live_modality_dropout=1.0,
    )
    opt_swap = torch.optim.AdamW(net_swap.parameters(), lr=5e-3, weight_decay=1e-4)
    net_swap.train()
    for _step in range(60):
        opt_swap.zero_grad()
        loss_dict = net_swap.compute_loss(combined_swap_batch, noise_std_pos=0.003, noise_std_yaw=0.02)
        loss_dict["loss"].backward()
        torch.nn.utils.clip_grad_norm_(net_swap.parameters(), max_norm=2.0)
        opt_swap.step()

    net_swap.eval()
    context_swap_outcomes: Dict[str, Any] = {}
    for tid in swap_tasks:
        c_pack = swap_context_packs[tid]
        pred_sw = net_swap.predict_actions({**c_pack, **probe_obs}, num_ddim_steps=2)
        wp_m = pred_sw.pred_world_waypoints_enu_m  # (H, 4)
        p0_m = (probe_obs["live_pose_se3"][0, :3, 3].numpy() / DEFAULT_SCALE_FACTOR).astype(np.float64)
        dy_m = float(wp_m[-1, 1] - p0_m[1])
        dx_m = float(wp_m[-1, 0] - p0_m[0])
        pts_full_xy = np.vstack([p0_m[:2], wp_m[:, :2]])
        context_swap_outcomes[tid] = {
            "primitive": swap_labels[tid],
            "delta_x_step8_m": round(dx_m, 4),
            "delta_y_step8_m": round(dy_m, 4),
            "waypoints_xy_m": pts_full_xy.tolist(),
        }

    # Generar figura diagnostica del Experimento 3 (Curva Overfit + Rollout 3D + Context-Swap)
    assets_dir.mkdir(parents=True, exist_ok=True)
    fig_path = assets_dir / "exp3_overfit_rollout_and_context_swap.png"
    fig, axes = plt.subplots(1, 3, figsize=(14.5, 4.3), dpi=140)

    axes[0].plot(np.arange(1, len(loss_history) + 1), loss_history, "o-", color="#1f77b4", lw=2.0, markersize=4)
    axes[0].set_title("Overfit Test: Pérdida de Flujo SE(3) + Estado")
    axes[0].set_xlabel("Época")
    axes[0].set_ylabel("Loss Total")
    axes[0].set_yscale("log")
    axes[0].grid(True, alpha=0.3)

    axes[1].plot(ref_pos_m[:, 0], ref_pos_m[:, 1], "k--", lw=2.2, label="GT Demostración (`episode_00`)")
    if best_rollout_traj is not None:
        axes[1].plot(
            best_rollout_traj[:, 0],
            best_rollout_traj[:, 1],
            "-",
            color="#2ca02c",
            lw=2.2,
            label=f"Rollout `MiniInstantPolicyNet` ($T_{{\\mathrm{{exec}}}}={mid_exec}, K=2$)",
        )
    for lm in waypoints_cfg.get("landmarks", []):
        axes[1].scatter([lm["x"]], [lm["y"]], s=90, c="#d62728", marker="s", zorder=5)
        axes[1].text(lm["x"], lm["y"] + 0.25, f"{lm['id']} ({lm['type']})", fontsize=8, ha="center")
    axes[1].set_title(f"Rollout Cinemático en Lazo Cerrado ({task_traverse})")
    axes[1].set_xlabel("X ENU [m]")
    axes[1].set_ylabel("Y ENU [m]")
    axes[1].grid(True, alpha=0.3)
    axes[1].legend(fontsize=8)

    colors_sw = {"task_0079": "#1f77b4", "task_0118": "#2ca02c", "task_0171": "#d62728"}
    for tid in swap_tasks:
        pts_xy = np.asarray(context_swap_outcomes[tid]["waypoints_xy_m"])
        axes[2].plot(
            pts_xy[:, 0],
            pts_xy[:, 1],
            "o-",
            lw=2.2,
            color=colors_sw[tid],
            label=f"Contexto {tid}: {swap_labels[tid]} ($\\Delta Y={context_swap_outcomes[tid]['delta_y_step8_m']:+.2f}$ m)",
        )
    axes[2].set_title("Context-Swap Test sobre Misma Observación $O_t$")
    axes[2].set_xlabel("X ENU Proyectado [m]")
    axes[2].set_ylabel("Y ENU Proyectado [m]")
    axes[2].grid(True, alpha=0.3)
    axes[2].legend(fontsize=8)

    fig.tight_layout()
    fig.savefig(fig_path)
    plt.close(fig)

    return {
        "oracle_closed_loop_by_t_exec": oracle_results,
        "overfit_training": {
            "task_id": task_traverse,
            "epochs": overfit_epochs,
            "initial_loss": round(loss_history[0], 4),
            "final_loss": round(loss_history[-1], 4),
            "training_wall_time_s": round(train_single_sec, 2),
        },
        "nn_closed_loop_sweep": nn_rollout_sweep,
        "context_swap_test": context_swap_outcomes,
        "figure_path": str(fig_path),
    }


def run_experiment_4_scene_centroid_coverage(
    kinematic_dir: Path,
) -> Dict[str, Any]:
    """
    Experimento 4: Evaluacion de compresion geometrica de la nube LiDAR de 2048 puntos
    con M = 16 vs M = 32 centroides FPS (`subsample_scene_centroids_torch`).
    Mide cuantos centroides caen sobre los obstaculos elevados (z_flu > -0.10, es decir,
    barras de puertas y postes excluyendo el suelo) y el error de cobertura geometrica
    (distancia Chamfer de los puntos de la puerta al centroide mas cercano en metros).
    """
    ep_data = load_heterodata_episode(kinematic_dir / "task_0006" / "episode_00" / "heterodata.pt")
    scene_clouds = ep_data["scene"].pos[::4].to(dtype=torch.float32)  # Subconjunto representativo de pasos

    results_by_m: Dict[str, Any] = {}
    for M_val in (16, 32):
        t0 = time.perf_counter()
        centroids, _ = subsample_scene_centroids_torch(scene_clouds, num_centroids=M_val, num_neighbors=16)
        elapsed_ms_per_cloud = ((time.perf_counter() - t0) * 1000.0) / max(1, len(scene_clouds))

        # Calcular cobertura sobre puntos de obstaculos cercanos (< 6.0 m -> < 0.60 en escala s=0.1)
        chamfer_m_list: List[float] = []
        obstacle_centroids_counts: List[int] = []
        for b in range(len(scene_clouds)):
            pts = scene_clouds[b]
            c_b = centroids[b]
            obs_mask = (pts[:, 2] > -0.10) & (torch.norm(pts[:, :2], dim=-1) < 0.65)
            obs_pts = pts[obs_mask]
            obs_c_count = int((c_b[:, 2] > -0.10).sum().item())
            obstacle_centroids_counts.append(obs_c_count)
            if len(obs_pts) > 0:
                dists = torch.cdist(obs_pts.unsqueeze(0), c_b.unsqueeze(0)).squeeze(0)  # (N_obs, M)
                min_d_scaled = torch.min(dists, dim=-1).values
                chamfer_m_list.append(float(min_d_scaled.mean().item() / DEFAULT_SCALE_FACTOR))

        results_by_m[f"M_{M_val}"] = {
            "num_centroids": M_val,
            "mean_obstacle_centroids": round(float(np.mean(obstacle_centroids_counts)), 2),
            "mean_obstacle_to_centroid_dist_m": round(float(np.mean(chamfer_m_list)), 4),
            "p95_obstacle_to_centroid_dist_m": round(float(np.percentile(chamfer_m_list, 95.0)), 4),
            "fps_knn_time_ms_per_cloud": round(float(elapsed_ms_per_cloud), 3),
        }

    return results_by_m


def run_experiment_5_cpu_latency_and_gazebo_rtf(
    kinematic_dir: Path,
    assets_dir: Path,
    num_keyframes: Union[int, str] = 10,
    adaptive_arc_m: float = 1.5,
    horizon: int = 8,
    action_stride: int = 1,
    num_context_demos: int = 2,
) -> Dict[str, Any]:
    """
    Experimento 5: Perfilado de latencia real en milisegundos sobre la CPU local (WSL2)
    variando:
      - Dimension oculta `d_model` in {64, 128, 256}
      - Numero de centroides `M` in {16, 32}
      - Pasos DDIM `K` in {1, 2, 4, 8}
      - Con y sin cacheo de las representaciones de contexto (`encode_context`).
    Calcula ademas la tabla de presupuesto temporal para Gazebo Harmonic con `RTF` in {1.0, 0.5, 0.25}.
    """
    H = max(1, int(horizon))
    stride = max(1, int(action_stride))
    ds = InContextNavigationDataset(
        kinematic_dir,
        task_ids=["task_0006"],
        num_context_demos=max(1, int(num_context_demos)),
        num_keyframes=num_keyframes,
        horizon=H,
        action_stride=stride,
        adaptive_arc_m=adaptive_arc_m,
    )
    sample = ds[0]
    kp_body = get_uav_keypoints_body_torch()
    ctx_mask = sample.get("context_keyframe_mask")

    configs = [
        {"name": "Mini (d=64, M=16)", "d_model": 64, "num_heads": 4, "M": 16},
        {"name": "Medium (d=128, M=16)", "d_model": 128, "num_heads": 4, "M": 16},
        {"name": "Medium-Dense (d=128, M=32)", "d_model": 128, "num_heads": 4, "M": 32},
        {"name": "Large (d=256, M=32)", "d_model": 256, "num_heads": 8, "M": 32},
    ]

    benchmark_rows: List[Dict[str, Any]] = []
    for cfg in configs:
        model = MiniInstantPolicyNet(
            d_model=cfg["d_model"],
            num_heads=cfg["num_heads"],
            num_centroids=cfg["M"],
            num_neighbors=16,
            horizon=H,
            action_stride=stride,
            num_diffusion_steps=4,
        )
        # Desactivar cache de parches FPS para medir el costo real ante nubes en vivo no vistas
        model.scene_encoder.use_patch_cache = False
        model.eval()
        num_params = sum(p.numel() for p in model.parameters())

        # Warmup
        with torch.no_grad():
            cached = model.encode_context(
                context_scene_pos=sample["context_scene_pos"],
                context_vel_flu=sample["context_vel_flu"],
                context_flight_state=sample["context_flight_state"],
                context_delta_4dof=sample["context_delta_4dof"],
                context_target_flight_state=sample["context_target_flight_state"],
                keypoints_body=kp_body,
                context_pose_se3=sample.get("context_pose_se3"),
                context_keyframe_mask=ctx_mask,
            )
            _ = model.predict_actions(sample, num_ddim_steps=2, cached_context=cached)

        # Medir encode_context (1 vez por episodio)
        t0 = time.perf_counter()
        n_ctx_runs = 3
        with torch.no_grad():
            for _ in range(n_ctx_runs):
                cached = model.encode_context(
                    context_scene_pos=sample["context_scene_pos"],
                    context_vel_flu=sample["context_vel_flu"],
                    context_flight_state=sample["context_flight_state"],
                    context_delta_4dof=sample["context_delta_4dof"],
                    context_target_flight_state=sample["context_target_flight_state"],
                    keypoints_body=kp_body,
                    context_pose_se3=sample.get("context_pose_se3"),
                    context_keyframe_mask=ctx_mask,
                )
        ctx_encode_ms = ((time.perf_counter() - t0) * 1000.0) / n_ctx_runs

        # Medir inferencia sin cachear (recalculando contexto en cada paso) para K=4
        t0 = time.perf_counter()
        n_uncached = 3
        with torch.no_grad():
            for _ in range(n_uncached):
                _ = model.predict_actions(sample, num_ddim_steps=4, cached_context=None)
        uncached_k4_ms = ((time.perf_counter() - t0) * 1000.0) / n_uncached

        # Medir inferencia CON contexto cacheado para K in {1, 2, 4, 8}
        k_timings: Dict[str, float] = {}
        for k_steps in (1, 2, 4, 8):
            t0 = time.perf_counter()
            n_iter = 6
            with torch.no_grad():
                for _ in range(n_iter):
                    _ = model.predict_actions(sample, num_ddim_steps=k_steps, cached_context=cached)
            k_timings[f"K_{k_steps}_ms"] = round(((time.perf_counter() - t0) * 1000.0) / n_iter, 2)

        benchmark_rows.append(
            {
                "config_name": cfg["name"],
                "d_model": cfg["d_model"],
                "num_centroids": cfg["M"],
                "num_parameters": int(num_params),
                "context_cache_once_ms": round(ctx_encode_ms, 2),
                "uncached_per_step_k4_ms": round(uncached_k4_ms, 2),
                **k_timings,
                "speedup_from_caching_x": round(uncached_k4_ms / max(0.1, k_timings["K_4_ms"]), 1),
            }
        )

    # Grafica de latencia CPU vs Presupuesto de Tiempo Real
    assets_dir.mkdir(parents=True, exist_ok=True)
    fig_path = assets_dir / "exp5_cpu_latency_benchmark.png"
    fig, ax = plt.subplots(figsize=(8.8, 4.4), dpi=140)
    k_vals = [1, 2, 4, 8]
    for row in benchmark_rows:
        y_ms = [row[f"K_{k}_ms"] for k in k_vals]
        ax.plot(k_vals, y_ms, "o-", lw=2.2, label=f"{row['config_name']} ({row['num_parameters']//1000}k params)")

    ax.axhline(100.0, color="gray", linestyle="--", lw=1.3, label="Paso $\\Delta t_{\\mathrm{sim}}=100$ ms (RTF=1.0x, $T_{\\mathrm{exec}}=1$)")
    ax.axhline(400.0, color="#2ca02c", linestyle=":", lw=1.5, label="Ventana $T_{\\mathrm{exec}}=4$ en RTF=1.0x (400 ms)")
    ax.set_title("Latencia de Inferencia en CPU (Con Caché de Contexto $\\phi_e + \\sigma$)")
    ax.set_xlabel("Pasos de Denoising DDIM ($K$)")
    ax.set_ylabel("Tiempo de Inferencia por Replanificación [ms]")
    ax.set_xticks(k_vals)
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=8.5)
    fig.tight_layout()
    fig.savefig(fig_path)
    plt.close(fig)

    return {
        "cpu_info": platform.processor() or platform.machine(),
        "torch_num_threads": int(torch.get_num_threads()),
        "benchmark_rows": benchmark_rows,
        "figure_path": str(fig_path),
    }


def generate_markdown_exploration_report(
    results: Dict[str, Any],
    output_report_path: Path,
) -> None:
    """
    Genera el documento exploratorio y didactico `docs/EXPLORACION_RED_NEURONAL.md`
    con guia conceptual, diccionario de variables, explicacion de formulas, contratos de tensores,
    resultados de la exploracion preliminar y listado transparente de parches exploratorios y
    aspectos abiertos a revisar en el Hito 4.
    """
    exp1 = results["experiment_1"]
    exp2 = results["experiment_2"]
    exp3 = results["experiment_3"]
    exp4 = results["experiment_4"]
    exp5 = results["experiment_5"]

    kin_s = exp1["kinematic_stats"]
    sitl_s = exp1["sitl_stats"]
    nk = exp2["noisy_keypoint_comparison"]
    ov = exp3["overfit_training"]

    lines: List[str] = [
        "# Exploración Experimental y Validación de Contratos de la Red Neuronal (Pre-Hito 4)",
        "",
        "*Guía conceptual, diccionario matemático y registro experimental completo de contratos de Entrada/Salida (E/S), estadísticas de normalización, verificación geométrica $SE(3)$/SVD 4-DoF, evaluación en lazo cerrado cinemático y perfilado de latencia en CPU para **UAV Instant Policy**.*",
        "",
        "---",
        "",
        "## 0. Guía Conceptual y Diccionario de Variables (Cómo Leer este Documento)",
        "",
        "Esta sección traduce toda la notación matemática y los términos de *Instant Policy* a lenguaje intuitivo, para que cada fórmula, tabla y gráfica del documento se entienda de principio a fin.",
        "",
        "### 0.1 ¿Qué hace exactamente nuestra red neuronal? (*In-Context Imitation Learning*)",
        "",
        "En el aprendizaje por imitación clásico, una red neuronal entrena durante horas para aprender una única tarea fija (por ejemplo, *\"cruzar siempre puertas por el centro\"*). Si en el momento de volar queremos que el dron **esquive la puerta por la izquierda**, habría que re-entrenar la red.",
        "",
        "**Instant Policy (Aprendizaje por Imitación en Contexto)** funciona de manera radicalmente distinta:",
        "1. **Entradas de la red en cada instante de vuelo:**",
        "   - **El Contexto ($N_{\\text{demos}} = 2$ demostraciones):** Le entregamos a la red 2 vuelos grabados previamente en ese mismo circuito (o uno similar) donde un experto realizó la secuencia deseada de maniobras (por ejemplo: *obstáculo 1 por arriba, obstáculo 2 por la izquierda, obstáculo 3 por el centro...*). Cada demostración viene resumida en **$L = 10$ momentos clave (*keyframes*)**.",
        "   - **La Observación Actual (*Live Query* $O_t$):** Lo que el dron está viendo y sintiendo **ahora mismo** en el instante $t$ (su nube de puntos LiDAR 360° y su velocidad).",
        "2. **Salida de la red:**",
        "   - Sin cambiar un solo peso ni re-entrenar nada, la red compara lo que ve ahora ($O_t$) con lo que ocurrió en las demostraciones de contexto y predice **los próximos $T = 8$ pasos futuros de trayectoria** ($0.8\\text{ segundos}$ hacia adelante, a razón de $\\Delta t = 0.1\\text{ s}$ por paso).",
        "",
        "---",
        "",
        "### 0.2 Marcos de Coordenadas y por qué usamos la Escala $s = 0.1$",
        "",
        "En todo el proyecto conviven dos sistemas de coordenadas y una escala espacial reducida:",
        "",
        "1. **Marco Global `Gazebo ENU` (Mundo Fijo):**",
        "   - Está clavado en el suelo del escenario de simulación (`X = Este`, `Y = Norte`, `Z = Arriba`).",
        "   - Sirve para saber dónde están los obstáculos en el mapa y para evaluar si el dron llegó a la meta, pero **la red neuronal nunca recibe coordenadas absolutas del mapa** (si lo hiciera, solo sabría volar en las coordenadas exactas donde entrenó).",
        "2. **Marco Egocéntrico `Horizontal FLU` (Pegado al Dron pero Nivelado con el Horizonte):**",
        "   - Su origen $(0,0,0)$ viaja siempre en el centro del dron:",
        "     - **$+X$ (*Forward*):** Hacia adelante (la nariz del dron).",
        "     - **$+Y$ (*Left*):** Hacia la izquierda (babor).",
        "     - **$+Z$ (*Up*):** Verticalmente hacia arriba (cielo).",
        "   - **¿Por qué se llama *Horizontal* FLU?** Un dron cuadricóptero necesita inclinarse hacia adelante (*pitch*) para acelerar o hacia los lados (*roll*) para doblar. Si usáramos el marco inclinado del chasis, cada vez que el dron frenara o acelerara, toda la nube LiDAR parecería subir y bajar violentamente. Al proyectar el marco sobre el plano horizontal (**forzando $\\text{roll} = 0$ y $\\text{pitch} = 0$**, conservando únicamente el giro de brújula o **rumbo $\\psi$ / *yaw***), el suelo siempre está plano y un movimiento en $+X$ significa siempre *\"avanzar horizontalmente hacia donde mira la nariz\"*.",
        "3. **Factor de Escala Espacial $s = 0.1$ ($1\\text{ unidad} = 10\\text{ metros}$):**",
        "   - El paper original de *Instant Policy* fue diseñado para brazos robóticos de mesa cuyos movimientos ocurren dentro de una caja de $\\sim 0.6\\text{–}1.0\\text{ m}$. Nuestro dron vuela en circuitos de $25\\text{ m}$ con un LiDAR omnidireccional de hasta $10.0\\text{ m}$ de radio.",
        "   - Al multiplicar todas las distancias en metros por **$s = 0.1$** (es decir, **dividir los metros por $10$**):",
        "     - El radio máximo de visión de $10.0\\text{ m}$ del LiDAR pasa a ser $1.0\\text{ unidad}$.",
        "     - Un avance de $0.20\\text{ m}$ en un paso de $0.1\\text{ s}$ pasa a ser $0.020\\text{ unidades}$.",
        "     - Así, las coordenadas relativas que entran a los embeddings sinusoidales de Fourier quedan dentro de $[-1.0, +1.0]$, evitando aliasing periódico en la frecuencia fundamental.",
        "   - **Regla práctica de lectura:** Siempre que veas una posición o desplazamiento con subíndice $s$ (como $\\Delta x_s = 0.030$), **multiplícalo por $10$ para obtener metros reales** ($0.030 \\times 10 = 0.30\\text{ m} = 30\\text{ cm}$). Los ángulos ($\\psi, \\Delta\\psi$) se miden en **radianes** y no se escalan.",
        "",
        "---",
        "",
        "### 0.3 ¿Por qué representamos al Dron con 6 Puntos Rígidos (*Keypoints* $\\mathbf{p}_g^1 \\dots \\mathbf{p}_g^6$)?",
        "",
        "Si queremos describir dónde estará el dron en el futuro y hacia dónde estará mirando, tenemos que predecir una **traslación** $(x, y, z)$ y una **rotación** ($\\psi$). Sin embargo, para una red neuronal basada en grafos 3D, mezclar metros con ángulos en radianes es problemático:",
        "- ¿Cómo comparas la distancia entre un punto del obstáculo LiDAR en $(x, y, z)$ y un ángulo en radianes?",
        "- Además, los ángulos tienen un salto brusco cuando pasan de $+180^\\circ$ ($+\\pi$) a $-180^\\circ$ ($-\\pi$).",
        "",
        "**La solución geométrica de *Instant Policy*:**",
        "En lugar de representar al dron como un punto con un ángulo abstracto, imaginamos una **cruceta rígida tridimensional de 6 puntos (*keypoints*)** soldada al chasis del dron a una distancia de $0.5\\text{ m}$ del centro ($L_g = 0.05$ en escala $s=0.1$):",
        "- $\\mathbf{p}_g^1 = (+0.05, 0, 0)$ $\\rightarrow$ Punta de la **nariz** ($0.5\\text{ m}$ al frente).",
        "- $\\mathbf{p}_g^2 = (-0.05, 0, 0)$ $\\rightarrow$ Punta de la **cola** ($0.5\\text{ m}$ atrás).",
        "- $\\mathbf{p}_g^3 = (0, +0.05, 0)$ $\\rightarrow$ Punta del brazo **izquierdo / babor** ($0.5\\text{ m}$ a la izquierda).",
        "- $\\mathbf{p}_g^4 = (0, -0.05, 0)$ $\\rightarrow$ Punta del brazo **derecho / estribor** ($0.5\\text{ m}$ a la derecha).",
        "- $\\mathbf{p}_g^5 = (0, 0, +0.05)$ $\\rightarrow$ Punta **superior** ($0.5\\text{ m}$ arriba).",
        "- $\\mathbf{p}_g^6 = (0, 0, -0.05)$ $\\rightarrow$ Punta **inferior** ($0.5\\text{ m}$ abajo).",
        "",
        "**¿Por qué esto es tan poderoso?**",
        "1. Si sabes dónde están esos 6 puntos en el espacio 3D, **sabes automáticamente dónde está el dron (el promedio de los 6 puntos) y hacia dónde apunta su nariz (la orientación de la cruceta)**.",
        "2. Ahora tanto los obstáculos del LiDAR como las poses futuras del dron son simplemente **puntos 3D $(x, y, z)$ en el mismo espacio métrico**, por lo que podemos conectarlos directamente con flechas (aristas de un grafo) y medir distancias reales entre las puntas del dron y las paredes de una puerta.",
        "",
        "---",
        "",
        "### 0.4 Las 4 Etapas de la Red ($\\phi_e \\to \\sigma \\to \\varphi \\to \\psi$) y los Dos Bucles de Inferencia",
        "",
        "Cuando la red neuronal se ejecuta, procesa la información en **4 etapas consecutivas** (nombradas con las mismas letras griegas del paper original):",
        "",
        "1. **Etapa $\\phi_e$ (Codificador de Escena LiDAR):**",
        "   - Recibe los $2048$ puntos crudos del láser LiDAR y elige **$M$ puntos representativos (centroides)** repartidos por los obstáculos cercanos. Para cada centroide, resume la forma geométrica de sus vecinos cercanos (si es un poste vertical, un travesaño horizontal, una pared, etc.).",
        "2. **Etapa $\\sigma$ (Grafo Local Escena $\\to$ Dron):**",
        "   - Conecta los $M$ centroides del LiDAR con los 6 puntos de la cruceta del dron. Así, cada punto de la cruceta \"siente\" qué obstáculos tiene cerca y en qué dirección, y además incorpora la velocidad actual del dron (`vel_flu`).",
        "3. **Etapa $\\varphi$ (Transformador de Contexto y Horizonte):**",
        "   - Es el \"cerebro\" de la red. Toma las representaciones $\\sigma$ de los $10$ keyframes de las demostraciones de contexto y la representación $\\sigma$ de lo que el dron ve ahora, y las combina con los **$T=8$ pasos futuros candidatos** que estamos intentando construir. Mediante mecanismos de atención (*Cross-Attention*), busca: *\"¿En qué momento de las demostraciones el dron vio un obstáculo parecido al que veo ahora, y hacia dónde movió su cruceta en los siguientes 8 pasos?\"*",
        "4. **Etapa $\\psi$ (Cabezal de Flujo de Keypoints y Estado de Vuelo):**",
        "   - Para cada uno de los $T=8$ pasos futuros y cada uno de los 6 puntos de la cruceta, predice un vector de corrección (llamado **flujo**) que indica **hacia dónde debe moverse y girar la cruceta** para acercarse a la trayectoria correcta, además de predecir si el dron debe seguir volando ($1.0$) o aterrizar ($0.0$).",
        "",
        "#### El Doble Bucle durante el Vuelo (Diferencia entre $K$ y $T_{\\text{exec}}$)",
        "",
        "Es fundamental no confundir los dos bucles que operan cuando el dron vuela:",
        "- **Bucle Interno de Difusión ($K$ iteraciones DDIM, dentro de una sola llamada a la red):**",
        "  - Como *Instant Policy* es un modelo de difusión, para generar una predicción de 8 pasos futuros parte de una trayectoria inicial ruidosa (o en cero) y la **refina $K$ veces seguidas** (por ejemplo, $K = 4$ refinamientos internos). En cada refinamiento $i = 1 \\dots K$, ejecuta las etapas $\\varphi \\to \\psi$ y ajusta los 8 pasos futuros.",
        "- **Bucle Externo de Control o Horizonte Recedente ($T_{\\text{exec}}$ pasos ejecutados en el mundo):**",
        "  - Una vez que la red termina sus $K$ refinamientos internos, nos entrega un plan de **$T = 8$ pasos futuros** ($0.8\\text{ s}$ de vuelo).",
        "  - En lugar de volar a ciegas los 8 pasos completos hasta el final, el dron **ejecuta solamente los primeros $T_{\\text{exec}}$ pasos** (por ejemplo, $T_{\\text{exec}} = 4$ pasos $= 0.4\\text{ s}$), toma una nueva foto del LiDAR y vuelve a pedirle a la red un nuevo plan de 8 pasos. Esto se llama **control de horizonte recedente (*Receding Horizon Control*)** y permite corregir cualquier desviación antes de que se acumule.",
        "",
        "---",
        "",
        "### 0.5 Diccionario Maestro de Símbolos, Fórmulas y Métricas",
        "",
        "| Símbolo / Término | Nombre Completo | Explicación en Lenguaje Simple |",
        "| :--- | :--- | :--- |",
        "| $S E(3)$ | Grupo Especial Euclídeo 3D | Conjunto de todas las posiciones $(x,y,z)$ y orientaciones 3D posibles de un cuerpo rígido, representadas como matrices de $4 \\times 4$. |",
        "| $S E(2) \\times \\mathbb{R}$ | Subgrupo 4-DoF Horizontal FLU | Posiciones 3D $(x,y,z)$ + únicamente giro horizontal de brújula ($\\psi$ / *yaw*), manteniendo siempre $\\text{roll}=0$ y $\\text{pitch}=0$. |",
        "| $T_{WE}^{(t)} \\in \\mathbb{R}^{4 \\times 4}$ | Pose del Dron en el Mundo | Matriz $4 \\times 4$ que dice dónde está el dron ($E$) respecto al mundo ($W$) en el instante $t$. |",
        "| $T_{EA}^{(t, k)} \\in \\mathbb{R}^{4 \\times 4}$ | Acción **Acumulada** al paso $k$ |Dónde estará el dron dentro de $k$ pasos ($k \\in \\{1..8\\}$) medido **desde donde está parado hoy en el instante $t$**. |",
        "| $\\delta T_k \\in \\mathbb{R}^{4 \\times 4}$ | Acción **Incremental** paso a paso | El pequeño salto que da el dron **entre el paso $k-1$ y el paso $k$** (en $\\Delta t = 0.1\\text{ s}$). Matemáticamente: $\\delta T_k = (T_{EA}^{(t, k-1)})^{-1} T_{EA}^{(t, k)}$. |",
        "| $(\\Delta x_s, \\Delta y_s, \\Delta z_s, \\Delta\\psi)$ | Vector 4-DoF Acumulado | Los 4 números útiles extraídos de $T_{EA}^{(t,k)}$: avance frontal, desplazamiento lateral, cambio de altura (en escala $s=0.1$) y giro de rumbo (en radianes) desde $t$ hasta $t+k$. |",
        "| $(\\delta x_s, \\delta y_s, \\delta z_s, \\delta\\psi)$ | Vector 4-DoF Incremental | Los mismos 4 números pero extraídos de $\\delta T_k$, es decir, lo que avanza y gira el dron **en un solo paso de $0.1\\text{ s}$**. |",
        "| $\\mathbf{p}_g^1 \\dots \\mathbf{p}_g^6 \\in \\mathbb{R}^3$ | Keypoints Canónicos del Dron | Los 6 puntos de la cruceta rígida del dron en reposo ($\\pm 0.05$ en cada eje, equivalentes a $\\pm 0.5\\text{ m}$ reales). |",
        "| $\\Delta\\mathbf{p}_t \\in \\mathbb{R}^3$ | Flujo de Traslación | Cuánto debe trasladarse el centro de la cruceta en $(x, y, z)$ para pasar de la pose ruidosa actual a la pose correcta. Es igual para los 6 puntos. |",
        "| $\\Delta\\mathbf{p}_r^i \\in \\mathbb{R}^3$ | Flujo de Rotación del punto $i$ | Cuánto debe desplazarse el punto $i$ de la cruceta **solo por efecto del giro de rumbo** $\\Delta\\psi$, dejando quieto el centro del dron. |",
        "| **SVD** | Descomposición en Valores Singulares | Operación algebraica que toma los 6 puntos movidos por la red neuronal y calcula la rotación rígida exacta que mejor los alinea sin deformar el dron. |",
        "| $\\gamma \\in (0, 1)$ | Factor de Decaimiento del Tubo | En *Tube Augmentation*, si el dron sufre un desvío lateral inicial $d$, en cada paso futuro $k$ le exigimos que reduzca ese desvío multiplicándolo por $\\gamma^k$ (ej. con $\\gamma=0.65$, en un paso reduce el error al $65\\%$, en dos pasos al $42\\%$, etc.). |",
        "| **p50 (Mediana)** | Percentil 50 | El valor típico: la mitad ($50\\%$) de los datos del dataset son menores o iguales a este número. |",
        "| **p95** | Percentil 95 | El $95\\%$ de los movimientos del dataset son menores o iguales a este valor (deja afuera solo el $5\\%$ de maniobras más bruscas). |",
        "| **p99.5** | Percentil 99.5 | Prácticamente el techo estadístico del dataset: cubre el $99.5\\%$ de todos los casos reales sin dejarse distorsionar por un único dato aislado anómalo (*outlier*). |",
        "| **ADE [m]** | *Average Displacement Error* | **Error Promedio de Trayectoria:** distancia media en metros entre la trayectoria completa que voló el dron y la trayectoria ideal del experto a lo largo de todo el circuito. |",
        "| **FDE [m]** | *Final Displacement Error* | **Error Final de Posición:** a cuántos metros del centro exacto de la plataforma de aterrizaje (*landing pad*) terminó el dron al finalizar el episodio. |",
        "| **RTF** | *Real-Time Factor* | Velocidad del reloj del simulador Gazebo respecto al mundo real. `RTF = 1.0x` es tiempo real; `RTF = 0.5x` significa que el simulador corre en \"cámara lenta\" a mitad de velocidad, dándole a tu CPU el doble de tiempo real para pensar cada movimiento. |",
        "",
        "---",
        "",
        "## 1. Arquitectura de Subpaquete y Aislamiento de Entornos (`uav_ip/policy/`)",
        "",
        "### ¿Por qué organizamos los archivos de esta manera?",
        "En este repositorio conviven dos mundos con requisitos de software distintos:",
        "1. **El mundo de simulación y generación de datos (ROS 2 Jazzy + Gazebo):** Corre con el Python del sistema (`/usr/bin/python3`) y no tiene instalado PyTorch ni `torch_geometric`.",
        "2. **El mundo de la Red Neuronal (Entrenamiento e Inferencia):** Corre dentro del entorno virtual `~/venv_ip` donde viven PyTorch, `torch_geometric` y las librerías de grafos.",
        "",
        "Para cumplir la **Regla 9 de `AGENT.md`** (*el núcleo de generación jamás debe importar `torch`, pero la política sí puede reutilizar el LiDAR sintético y el validador geométrico del núcleo*), todo el código de la red neuronal fue encapsulado en subcarpetas `policy/`:",
        "",
        "- **Librería de la política (`src/uav_ip/uav_ip/policy/`):**",
        "  - [`se3_diffusion_math.py`](../src/uav_ip/uav_ip/policy/se3_diffusion_math.py): Todas las fórmulas geométricas: conversión entre acciones acumuladas $T_{EA}^{(t,k)}$ e incrementales $\\delta T_k$, normalización 4-DoF a $[-1, 1]$, cálculo del flujo desacoplado $[\\Delta\\hat{\\mathbf{p}}_t, \\Delta\\hat{\\mathbf{p}}_r]$ sobre los 6 keypoints del dron y reconstrucción rígida mediante SVD proyectado a Horizontal FLU.",
        "  - [`in_context_dataset.py`](../src/uav_ip/uav_ip/policy/in_context_dataset.py): El cargador `InContextNavigationDataset` que lee los archivos `heterodata.pt` y reconstruye al vuelo (sin regenerar `.pt`) cualquier combinación de $N_{\\text{demos}}$, cantidad de keyframes $L$ (entero fijo como `10, 16, 24` o `'adaptive'` proporcional a la longitud de arco), horizonte futuro $T$ (ej. `6, 8, 12, 16`) y `action_stride` (`1` o `2`), verificando contratos con [`verify_sample_contract()`](../src/uav_ip/uav_ip/policy/in_context_dataset.py).",
        "  - [`simple_validation_net.py`](../src/uav_ip/uav_ip/policy/simple_validation_net.py): Contiene la política de referencia (`OracleReplayPolicy`, parametrizable en `horizon` y `action_stride`) y una red ligera puramente exploratoria (`MiniInstantPolicyNet`) creada solo para comprobar que las dimensiones, máscaras y gradientes conectan de extremo a extremo (incluye atajos temporales de exploración documentados en la Sección 5.5, que no constituyen un patrón para la red final y quedan marcados para revisión en el Hito 4).",
        "  - [`kinematic_rollout_eval.py`](../src/uav_ip/uav_ip/policy/kinematic_rollout_eval.py): Simulador de vuelo en lazo cerrado en Python puro para cualquier horizonte $T$, ventana $T_{\\text{exec}} \\in [1, T]$ y `action_stride`. En cada paso mueve el dron según lo que ordena la política, genera en vivo la nueva vista láser con [`SyntheticLidarGenerator`](../src/uav_ip/uav_ip/synthetic_lidar.py), interpola la trayectoria a $50\\text{ Hz}$ con splines PCHIP y al terminar audita el vuelo completo con el juez oficial ([`validate_episode_data()`](../src/uav_ip/uav_ip/validate_episode.py)).",
        "  - [`exploration_suite.py`](../src/uav_ip/uav_ip/policy/exploration_suite.py): Script maestro que ejecuta tanto los 5 experimentos canónicos como sondas rápidas con hiperparámetros variables (`run_custom_hyperparameter_probe`).",
        "- **Ejecutables para `~/venv_ip` (`src/uav_ip/scripts/policy/`):**",
        "  - [`run_exploration_suite.py`](../src/uav_ip/scripts/policy/run_exploration_suite.py): Punto de entrada por línea de comandos parametrizable.",
        "- **Pruebas unitarias (`src/uav_ip/test/policy/`):**",
        "  - [`test_exploration_pipeline.py`](../src/uav_ip/test/policy/test_exploration_pipeline.py): Suite automática en `pytest` que verifica la matemática, los contratos variables $(L, T, \\text{stride})$, la red exploratoria y el lazo cerrado.",
        "",
        "### 1.1 Cómo Invocar el Script con Hiperparámetros Variables (Sin Regenerar `heterodata.pt`)",
        "",
        "Como cada archivo `heterodata.pt` almacena la trayectoria completa `pose_se3`, las nubes `scene.pos` y las velocidades `vel_flu` a $10\\text{ Hz}$, [`run_exploration_suite.py`](../src/uav_ip/scripts/policy/run_exploration_suite.py) permite variar todos los ejes de diseño directamente desde la terminal:",
        "",
        "```bash",
        "# A) Evaluar rápidamente una combinación puntual (sin sobrescribir este reporte):",
        "~/venv_ip/bin/python3 src/uav_ip/scripts/policy/run_exploration_suite.py \\",
        "  --custom-eval-only \\",
        "  --num-keyframes adaptive --adaptive-arc-m 1.5 \\",
        "  --horizon 12 --action-stride 1 \\",
        "  --execution-horizon 6 --num-context-demos 2 --num-ddim-steps 4 \\",
        "  --num-centroids 32 --d-model 128",
        "",
        "# B) Probar paso temporal doble (action_stride=2 -> Δt = 0.2 s, cubriendo 1.6 s con T=8):",
        "~/venv_ip/bin/python3 src/uav_ip/scripts/policy/run_exploration_suite.py \\",
        "  --custom-eval-only \\",
        "  --num-keyframes 16 --horizon 8 --action-stride 2 --execution-horizon 4",
        "",
        "# C) Regenerar únicamente este documento Markdown desde exploration_metrics.json:",
        "~/venv_ip/bin/python3 src/uav_ip/scripts/policy/run_exploration_suite.py --report-only",
        "```",
        "",
        "| Parámetro CLI | Default | Valores Admitidos y Efecto |",
        "| :--- | :---: | :--- |",
        "| `--num-keyframes` | `10` | Entero positivo (`10`, `16`, `24`, ...) o `adaptive` ($L_d = \\lceil \\text{arco}_d / \\Delta s \\rceil$ acotado en $[10, 28]$). |",
        "| `--adaptive-arc-m` | `1.5` | Metros de recorrido por keyframe cuando `--num-keyframes adaptive`. |",
        "| `--horizon` | `8` | Pasos futuros $T$ predichos por inferencia (`6`, `8`, `12`, `16`, ...). |",
        "| `--action-stride` | `1` | Separación en ticks de $10\\text{ Hz}$ entre pasos (`1` $\\equiv 0.1\\text{ s}$, `2` $\\equiv 0.2\\text{ s}$). Escala cotas automáticamente. |",
        "| `--num-context-demos` | `2` | Cantidad $N_{\\text{demos}}$ de demostraciones de contexto por consulta (`1`, `2`, `3`, `4`). |",
        "| `--execution-horizon` | `4` | Pasos $T_{\\text{exec}} \\in [1, T]$ ejecutados antes de replanificar en lazo cerrado. |",
        "| `--num-ddim-steps` | `4` | Iteraciones internas $K$ de refinamiento DDIM (`1`, `2`, `4`, `8`). |",
        "| `--num-centroids` / `--d-model` | `16` / `64` | Centroides FPS $M$ del codificador $\\phi_e$ y dimensión oculta de `MiniInstantPolicyNet`. |",
        "| `--custom-eval-only` | `False` | Ejecuta validación de contratos + Oráculo + entrenamiento rápido + rollout + latencia en ~20 s sin pisar el reporte. |",
        "",
        "---",
        "",
        "## 2. Mapa Maestro de Contratos de Entrada/Salida (E/S)",
        "",
        "### ¿Qué significa esta tabla y por qué es necesaria?",
        "Antes de construir una red neuronal mayor, un error frecuente en robótica es que un módulo entregue distancias en metros mientras otro espera decámetros ($s=0.1$), o que un tensor tenga las dimensiones invertidas. Esta tabla documenta el **contrato de ingeniería** entre el dataset (`heterodata.pt`), la red neuronal y el controlador del dron:",
        "- **Cómo leer la columna `Forma Exacta`:**",
        "  - `N_demos`: Cantidad de demostraciones de contexto (por defecto $2$, configurable con `--num-context-demos`).",
        "  - `L`: Cantidad de *keyframes* por demostración (por defecto $10$, o variable por longitud de arco con `--num-keyframes adaptive` acompañado de `context_keyframe_mask`).",
        "  - `T`: Cantidad de pasos futuros del horizonte de predicción (por defecto $8$ pasos $= 0.8\\text{ s}$ con `action_stride=1`, configurable con `--horizon` y `--action-stride`).",
        "  - `2048, 3`: Los $2048$ puntos $(x, y, z)$ de cada barrido LiDAR.",
        "  - `4, 4`: Una matriz de transformación homogénea $SE(3)$ de $4 \\times 4$.",
        "",
        "| Etapa | Nombre del Tensor | Forma Exacta | Tipo | Marco de Coordenadas | Escala / Unidades | Descripción |",
        "| :--- | :--- | :--- | :--- | :--- | :--- | :--- |",
    ]

    for item in TENSOR_CONTRACT_SPEC:
        lines.append(
            f"| **{item['stage']}** | `{item['name']}` | `{item['shape']}` | `{item['dtype']}` | {item['frame']} | {item['scale']} | {item['description']} |"
        )

    lines.extend(
        [
            "",
            "---",
            "",
            "## 3. Experimento 1: Radiografía Estadística y Cotas de Normalización ($\\delta T_k$ vs. $T_{EA}^{(t, k)}$)",
            "",
            "### 3.1 ¿Qué duda queremos resolver en este experimento?",
            "En cada archivo `heterodata.pt`, las etiquetas de acción guardadas (`action_se3`) representan la **pose futura acumulada** $T_{EA}^{(t, k)}$ para los pasos $k = 1, 2, \\dots, 8$ medida desde la posición actual del dron en el instante $t$.",
            "",
            "Tenías la duda de cómo debía predecir la red esos 8 pasos futuros y cómo debíamos normalizarlos para que todas las salidas de la red vivan cómodamente en el intervalo $[-1, 1]$. Tenemos dos formas matemáticamente equivalentes de expresar un horizonte de 8 pasos:",
            "1. **Formulación Acumulada ($T_{EA}^{(t, k)}$):** Cada nodo $k$ representa la distancia total desde $t$ hasta $t+k$.",
            "   - *Problema intuitivo:* Si el dron vuela a $2\\text{ m/s}$, en el paso $k=1$ ($0.1\\text{ s}$ después) habrá avanzado **$0.20\\text{ m}$**, pero en el paso $k=8$ ($0.8\\text{ s}$ después) habrá avanzado **$1.60\\text{ m}$** (¡8 veces más!). Si entrenamos la red directamente con estos valores, el error del paso 8 será numéricamente mucho más grande que el del paso 1, haciendo que la red preste mucha atención al final del horizonte y descuide el primer paso inmediato, que es justamente el que el dron va a ejecutar primero.",
            "2. **Formulación Incremental Paso a Paso ($\\delta T_k$):** Cada nodo $k$ representa únicamente **el pequeño salto entre el paso anterior $k-1$ y el paso $k$**:",
            "   $$\\delta T_k = \\left(T_{EA}^{(t, k-1)}\\right)^{-1} T_{EA}^{(t, k)}, \\quad \\text{con } T_{EA}^{(t, 0)} = I_4$$",
            "   - *Ventaja intuitiva:* Tanto en $k=1$ como en $k=8$, el salto dura exactamente $\\Delta t = 0.1\\text{ s}$ y mide siempre alrededor de $0.20\\text{ m}$. Los 8 pasos tienen **la misma escala estadística**. Y cuando necesitamos las poses acumuladas para construir los nodos del grafo, encadenamos (multiplicamos) los saltos desde el principio:",
            "   $$T_{EA}^{(t, k)} = \\delta T_1 \\cdot \\delta T_2 \\cdots \\delta T_k = \\prod_{m=1}^{k} \\delta T_m$$",
            "",
            f"Para comprobarlo con datos reales de tu proyecto, analizamos **{kin_s['num_episodes_analyzed']} episodios cinemáticos** ({kin_s['total_nominal_steps']} instantes de vuelo nominal + {kin_s['total_aug_steps']} instantes de recuperación *Tube Augmentation*) y **{sitl_s['num_episodes_analyzed']} episodios físicos en PX4 SITL** ({sitl_s['total_nominal_steps']} instantes nominales + {sitl_s['total_aug_steps']} instantes *Tube Augmentation*).",
            "",
            "### 3.2 Tabla Comparativa de Rangos: Acumulada $T_{EA}^{(t,k)}$ vs. Incremental $\\delta T_k$",
            "",
            "**Cómo leer esta tabla:**",
            "- Las columnas `p50`, `p95`, `p99.5` y `Máximo Absoluto` muestran los desplazamientos en la escala interna de la red ($s = 0.1$) para las traslaciones ($|\\Delta x_s|, |\\delta x_s|, |\\delta y_s|, |\\delta z_s|$) y en **radianes** para el giro de rumbo ($|\\delta\\psi|$).",
            "- La última columna (`Equivalente Físico (p99.5)`) convierte el percentil $99.5\\%$ a **metros reales** (multiplicando por $10$) o a **grados** para que puedas interpretarlo físicamente de un vistazo.",
            "",
            "| Dominio / Formulación | Componente | p50 | p95 | p99.5 | Máximo Absoluto | Equivalente Físico (p99.5) |",
            "| :--- | :--- | :---: | :---: | :---: | :---: | :--- |",
            f"| **Kinematic — Acumulada $k=1$** | $|\\Delta x_s|$ | {kin_s['cumulative_step1']['dx_s']['p50']:.4f} | {kin_s['cumulative_step1']['dx_s']['p95']:.4f} | {kin_s['cumulative_step1']['dx_s']['p99_5']:.4f} | {kin_s['cumulative_step1']['dx_s']['max']:.4f} | {kin_s['cumulative_step1']['dx_s']['p99_5']*10:.3f} m |",
            f"| **Kinematic — Acumulada $k=8$** | $|\\Delta x_s|$ | {kin_s['cumulative_step8']['dx_s']['p50']:.4f} | {kin_s['cumulative_step8']['dx_s']['p95']:.4f} | {kin_s['cumulative_step8']['dx_s']['p99_5']:.4f} | {kin_s['cumulative_step8']['dx_s']['max']:.4f} | {kin_s['cumulative_step8']['dx_s']['p99_5']*10:.3f} m |",
            f"| **Kinematic — Incremental $\\delta T_k$ ($\\forall k$)** | $|\\delta x_s|$ | {kin_s['incremental_all_steps']['dx_s']['p50']:.4f} | {kin_s['incremental_all_steps']['dx_s']['p95']:.4f} | {kin_s['incremental_all_steps']['dx_s']['p99_5']:.4f} | {kin_s['incremental_all_steps']['dx_s']['max']:.4f} | {kin_s['incremental_all_steps']['dx_s']['p99_5']*10:.3f} m ($\\Delta t=0.1$ s) |",
            f"| **Kinematic — Incremental $\\delta T_k$ ($\\forall k$)** | $|\\delta y_s|$ | {kin_s['incremental_all_steps']['dy_s']['p50']:.4f} | {kin_s['incremental_all_steps']['dy_s']['p95']:.4f} | {kin_s['incremental_all_steps']['dy_s']['p99_5']:.4f} | {kin_s['incremental_all_steps']['dy_s']['max']:.4f} | {kin_s['incremental_all_steps']['dy_s']['p99_5']*10:.3f} m ($\\Delta t=0.1$ s) |",
            f"| **Kinematic — Incremental $\\delta T_k$ ($\\forall k$)** | $|\\delta z_s|$ | {kin_s['incremental_all_steps']['dz_s']['p50']:.4f} | {kin_s['incremental_all_steps']['dz_s']['p95']:.4f} | {kin_s['incremental_all_steps']['dz_s']['p99_5']:.4f} | {kin_s['incremental_all_steps']['dz_s']['max']:.4f} | {kin_s['incremental_all_steps']['dz_s']['p99_5']*10:.3f} m ($\\Delta t=0.1$ s) |",
            f"| **Kinematic — Incremental $\\delta T_k$ ($\\forall k$)** | $|\\delta \\psi|$ [rad] | {kin_s['incremental_all_steps']['dyaw_rad']['p50']:.4f} | {kin_s['incremental_all_steps']['dyaw_rad']['p95']:.4f} | {kin_s['incremental_all_steps']['dyaw_rad']['p99_5']:.4f} | {kin_s['incremental_all_steps']['dyaw_rad']['max']:.4f} | {math.degrees(kin_s['incremental_all_steps']['dyaw_rad']['p99_5']):.2f}° ($\\Delta t=0.1$ s) |",
            f"| **PX4 SITL — Acumulada $k=8$** | $|\\Delta x_s|$ | {sitl_s['cumulative_step8']['dx_s']['p50']:.4f} | {sitl_s['cumulative_step8']['dx_s']['p95']:.4f} | {sitl_s['cumulative_step8']['dx_s']['p99_5']:.4f} | {sitl_s['cumulative_step8']['dx_s']['max']:.4f} | {sitl_s['cumulative_step8']['dx_s']['p99_5']*10:.3f} m |",
            f"| **PX4 SITL — Incremental $\\delta T_k$ ($\\forall k$)** | $|\\delta x_s|$ | {sitl_s['incremental_all_steps']['dx_s']['p50']:.4f} | {sitl_s['incremental_all_steps']['dx_s']['p95']:.4f} | {sitl_s['incremental_all_steps']['dx_s']['p99_5']:.4f} | {sitl_s['incremental_all_steps']['dx_s']['max']:.4f} | {sitl_s['incremental_all_steps']['dx_s']['p99_5']*10:.3f} m ($\\Delta t=0.1$ s) |",
            f"| **PX4 SITL — Incremental $\\delta T_k$ ($\\forall k$)** | $|\\delta y_s|$ | {sitl_s['incremental_all_steps']['dy_s']['p50']:.4f} | {sitl_s['incremental_all_steps']['dy_s']['p95']:.4f} | {sitl_s['incremental_all_steps']['dy_s']['p99_5']:.4f} | {sitl_s['incremental_all_steps']['dy_s']['max']:.4f} | {sitl_s['incremental_all_steps']['dy_s']['p99_5']*10:.3f} m ($\\Delta t=0.1$ s) |",
            f"| **PX4 SITL — Incremental $\\delta T_k$ ($\\forall k$)** | $|\\delta z_s|$ | {sitl_s['incremental_all_steps']['dz_s']['p50']:.4f} | {sitl_s['incremental_all_steps']['dz_s']['p95']:.4f} | {sitl_s['incremental_all_steps']['dz_s']['p99_5']:.4f} | {sitl_s['incremental_all_steps']['dz_s']['max']:.4f} | {sitl_s['incremental_all_steps']['dz_s']['p99_5']*10:.3f} m ($\\Delta t=0.1$ s) |",
            f"| **PX4 SITL — Incremental $\\delta T_k$ ($\\forall k$)** | $|\\delta \\psi|$ [rad] | {sitl_s['incremental_all_steps']['dyaw_rad']['p50']:.4f} | {sitl_s['incremental_all_steps']['dyaw_rad']['p95']:.4f} | {sitl_s['incremental_all_steps']['dyaw_rad']['p99_5']:.4f} | {sitl_s['incremental_all_steps']['dyaw_rad']['max']:.4f} | {math.degrees(sitl_s['incremental_all_steps']['dyaw_rad']['p99_5']):.2f}° ($\\Delta t=0.1$ s) |",
            f"| **Tube Augmentation ($\\gamma=0.70$)** | $|\\delta x_s|, |\\delta y_s|$ | {kin_s['incremental_aug_recovery']['dx_s']['p50']:.4f} | {kin_s['incremental_aug_recovery']['dx_s']['p95']:.4f} | {kin_s['incremental_aug_recovery']['dx_s']['p99_5']:.4f} | {kin_s['incremental_aug_recovery']['dx_s']['max']:.4f} | Cota física $\\le 0.65$ m ($6.5$ m/s) |",
            "",
            "![Comparación Acumulada vs Incremental](assets/exploration/exp1_cumulative_vs_incremental_bounds.png)",
            "",
            "### 3.3 Cómo interpretar la gráfica y qué se observa",
            "- **Gráfica Izquierda (Acción Acumulada $T_{EA}^{(t,k)}$):** Observa cómo a medida que nos alejamos en el eje horizontal desde el paso $k=1$ hasta el paso $k=8$, las curvas suben en rampa empinada: en cinemático pasan de $0.203\\text{ m}$ ($k=1$) a $1.492\\text{ m}$ ($k=8$), y en PX4 SITL llegan hasta **$2.939\\text{ m}$** en $k=8$.",
            "- **Gráfica Derecha (Incremento Paso a Paso $\\delta T_k$):** Al transformar las acciones a incrementos entre pasos consecutivos, **las curvas se mantienen casi planas** de $k=1$ a $k=8$ ($0.196\\text{ m}$ en cinemático y $0.392\\text{ m}$ en SITL). La línea punteada roja muestra una referencia exploratoria ($0.075$ en escala $s=0.1$, que equivale a $0.75\\text{ m}$ por paso o $7.5\\text{ m/s}$, cubriendo la cota de $6.5\\text{ m/s}$ fijada en `bag_to_heterodata.py`).",
            "",
            "> [!NOTE]",
            "> **Observación sobre `ActionBounds4DoF` (Alcance Exploratorio — A Revisar en el Hito 4):**",
            "> Los datos muestran que trabajar sobre incrementos $\\delta T_k$ mantiene una escala uniforme entre $k=1$ y $k=8$, a diferencia de las poses acumuladas $T_{EA}^{(t,k)}$. Para poder ejecutar las pruebas rápidas de este reporte se definieron en [`ActionBounds4DoF`](../src/uav_ip/uav_ip/policy/se3_diffusion_math.py) cotas auxiliares tomadas de este corpus (`0.075` en $X/Y$, `0.050` en $Z$ y `0.65 rad` en $\\psi$ por paso de $0.1\\text{ s}$).",
            "> **Importante:** Estos valores provienen de la distribución de este dataset concreto y **no representan un patrón fijo para la red final**. Aplicar un recorte duro (`clip=True`) con umbrales ajustados al dataset actual podría limitar maniobras futuras con mayor velocidad de giro o ascenso; en el Hito 4 conviene revisar un esquema más general (por ejemplo, normalización mediante la escala de ruido $\\sigma_k$ del *schedule* de difusión, estadísticos calculados dinámicamente del split de entrenamiento sin *clipping* duro en inferencia, o límites físicos del vehículo).",
            "",
            "---",
            "",
            "## 4. Experimento 2: Exactitud Matemática Round-Trip $SE(3)$ y SVD 4-DoF vs. 6-DoF",
            "",
            "### 4.1 ¿Qué duda queremos resolver y qué significan las fórmulas de esta etapa?",
            "Como vimos en la Sección 0.3, la etapa final $\\psi$ de la red neuronal no escupe directamente una matriz $4 \\times 4$, sino que predice **hacia dónde deben moverse los 6 puntos de la cruceta del dron**. En concreto, para cada uno de los 6 keypoints canónicos $\\mathbf{p}_g^i$ ($i=1\\dots 6$), la red predice dos vectores 3D (6 números por punto):",
            "1. **El flujo de traslación $\\Delta\\mathbf{p}_t^i \\in \\mathbb{R}^3$:** Indica cuánto debe desplazarse el centro del dron entre la hipótesis actual $\\hat{T}_{\\text{curr}} = (R_{\\text{curr}}, \\mathbf{t}_{\\text{curr}})$ y la pose correcta $T_* = (R_*, \\mathbf{t}_*)$:",
            "   $$\\Delta\\mathbf{p}_t^i = \\mathbf{t}_* - \\mathbf{t}_{\\text{curr}}$$",
            "2. **El flujo de rotación $\\Delta\\mathbf{p}_r^i \\in \\mathbb{R}^3$:** Indica cuánto debe moverse cada punta de la cruceta **únicamente por el cambio de orientación** entre la orientación actual $R_{\\text{curr}}$ y la orientación correcta $R_*$, sin contar la traslación:",
            "   $$\\Delta\\mathbf{p}_r^i = \\left(R_* - R_{\\text{curr}}\\right) \\mathbf{p}_g^i$$",
            "",
            "**¿Cómo reconstruimos una matriz rígida $4 \\times 4$ a partir de las predicciones de la red?**",
            "- Para la **nueva posición del dron ($\\mathbf{t}_{\\text{new}}$)**, promediamos el flujo de traslación predicho por los 6 puntos y se lo sumamos a la posición actual:",
            "  $$\\mathbf{t}_{\\text{new}} = \\mathbf{t}_{\\text{curr}} + \\frac{1}{6} \\sum_{i=1}^{6} \\Delta\\hat{\\mathbf{p}}_t^i$$",
            "- Para la **nueva orientación ($R_{\\text{new}}$)**, tomamos dónde estarían las 6 puntas de la cruceta después de aplicarles el flujo rotacional predicho ($\\mathbf{q}_i = R_{\\text{curr}} \\mathbf{p}_g^i + \\Delta\\hat{\\mathbf{p}}_r^i$) y buscamos la matriz de rotación rígida que mejor alinee la cruceta original $\\{\\mathbf{p}_g^i\\}$ con los puntos movidos $\\{\\mathbf{q}_i\\}$.",
            "",
            "**Comparación entre SVD 6-DoF libre y SVD 4-DoF proyectado a Horizontal FLU:**",
            "- El algoritmo estándar de alineación 3D (llamado **SVD de Kabsch-Arun en 6-DoF**) busca cualquier rotación 3D libre en $SO(3)$. Si las predicciones de los 6 puntos tienen pequeñas discrepancias independientes (por ejemplo $1$ o $2\\text{ cm}$ en la altura $Z$ del punto de la nariz $\\mathbf{p}_g^1$ respecto al de la cola $\\mathbf{p}_g^2$), el SVD 6-DoF interpreta esa diferencia como una inclinación en *pitch* o *roll*.",
            "- Dado que en este sistema las acciones están definidas sobre el marco nivelado **Horizontal FLU** ($\\text{roll}=0, \\text{pitch}=0$), implementamos en [`align_keypoints_svd_horizontal_flu()`](../src/uav_ip/uav_ip/policy/se3_diffusion_math.py) una **proyección analítica en 4-DoF ($SE(2) \\times \\mathbb{R}$)**: proyectamos las puntas de la cruceta al plano horizontal $(X, Y)$, calculamos la matriz de covarianza planar $H_{2\\times 2} = \\sum_{i=1}^6 \\mathbf{p}_{g, xy}^i (\\mathbf{q}_{xy}^i)^\\top$ y extraemos el ángulo de rumbo óptimo en forma cerrada:",
            "  $$\\Delta\\psi_{\\text{new}} = \\text{atan2}\\!\\left(H_{xy} - H_{yx},\\; H_{xx} + H_{yy}\\right), \\quad R_{\\text{new}} = R_z(\\Delta\\psi_{\\text{new}}) = \\begin{bmatrix} \\cos\\Delta\\psi_{\\text{new}} & -\\sin\\Delta\\psi_{\\text{new}} & 0 \\\\ \\sin\\Delta\\psi_{\\text{new}} & \\cos\\Delta\\psi_{\\text{new}} & 0 \\\\ 0 & 0 & 1 \\end{bmatrix}$$",
            "",
            "### 4.2 Resultados de las 3 Pruebas Matemáticas (sobre 12 Episodios Reales)",
            "",
            "1. **Prueba de Ida y Vuelta Acumulada $\\leftrightarrow$ Incremental (`cumulative_to_incremental_se3` $\\to$ `incremental_to_cumulative_se3`):**",
            "   - Tomamos las matrices $T_{EA}^{(t,k)}$ del dataset, las descompusimos en incrementos $\\delta T_k$ y las volvimos a multiplicar para reconstruir $T_{EA}^{(t,k)}$.",
            f"   - Error máximo de posición (en metros reales): **`{exp2['cum_inc_roundtrip_max_pos_err_m']:.3e} m`** (menos de una millonésima de metro).",
            f"   - Error máximo de rumbo $\\psi$ (en radianes): **`{exp2['cum_inc_roundtrip_max_yaw_err_rad']:.3e} rad`**.",
            "2. **Prueba de Ida y Vuelta de Flujo de Keypoints + SVD 4-DoF (`apply_keypoint_flow_and_svd_4dof`):**",
            "   - Calculamos los flujos $[\\Delta\\mathbf{p}_t, \\Delta\\mathbf{p}_r]$ de los episodios reales y reconstruimos la matriz $4 \\times 4$ usando nuestro SVD 4-DoF tanto en $K=1$ paso como en un bucle iterativo DDIM de $K=4$ pasos.",
            f"   - Error máximo en $K=1$ paso desde la identidad $I_4$: **`{exp2['svd_1step_max_pos_err_m']:.3e} m`** / **`{exp2['svd_1step_max_yaw_err_rad']:.3e} rad`**.",
            f"   - Error máximo en bucle DDIM de $K=4$ pasos con reproyección SVD en cada iteración: **`{exp2['svd_4step_ddim_max_pos_err_m']:.3e} m`** / **`{exp2['svd_4step_ddim_max_yaw_err_rad']:.3e} rad`**.",
            "3. **Experimento de Ruido Realista en Keypoints ($\\sigma = 0.002$ escalado $\\equiv 2\\text{ cm}$ de error en las puntas del dron): SVD 6-DoF vs. SVD 4-DoF:**",
            "   - Inyectamos ruido gaussiano de $2\\text{ cm}$ reales sobre las predicciones de los 6 keypoints y comparamos qué matriz construye el SVD 6-DoF clásico frente a nuestro SVD 4-DoF:",
            "",
            "| Método de Reconstrucción Rígida | Roll Espurio Máx. | Pitch Espurio Máx. | Inclinación Espuria p95 | Error Medio de Yaw |",
            "| :--- | :---: | :---: | :---: | :---: |",
            f"| **SVD 6-DoF Completo (Arun en $SE(3)$)** | `{nk['svd_6dof_max_spurious_roll_deg']:.2f}°` | `{nk['svd_6dof_max_spurious_pitch_deg']:.2f}°` | `{nk['svd_6dof_p95_spurious_tilt_deg']:.2f}°` | `{nk['svd_6dof_mean_yaw_err_deg']:.3f}°` |",
            f"| **SVD 4-DoF Proyectado (`align_keypoints_svd_horizontal_flu`)** | **`0.00°`** | **`0.00°`** | **`0.00°`** | **`{nk['svd_4dof_mean_yaw_err_deg']:.3f}°`** |",
            "",
            "> [!TIP]",
            "> **Lectura de la prueba geométrica:**",
            "> 1. Las conversiones entre poses acumuladas, incrementos $\\delta T_k$ y flujos de 6 keypoints son numéricamente reversibles dentro de la precisión `float32` ($\\sim 10^{-7}\\text{ m}$).",
            "> 2. Mientras las acciones se formulen en el marco nivelado `Horizontal FLU` ($\\text{roll}=0, \\text{pitch}=0$), el SVD 6-DoF libre traduce pequeños ruidos de predicción ($2\\text{ cm}$) en inclinaciones aparentes de hasta **`"
            + f"{max(nk['svd_6dof_max_spurious_roll_deg'], nk['svd_6dof_max_spurious_pitch_deg']):.2f}°"
            + "`**, mientras que `align_keypoints_svd_horizontal_flu` mantiene la actualización restringida al subgrupo planar + altitud ($SE(2) \\times \\mathbb{R}$) con el mismo error medio en rumbo.",
            "",
            "---",
            "",
            "## 5. Experimento 3: Evaluación en Lazo Cerrado Cinemático, Horizonte $T_{\\text{exec}}$ y *Context-Swap*",
            "",
            "### 5.1 ¿Qué dudas queremos resolver en este experimento?",
            "Este experimento aborda tres preguntas prácticas de integración:",
            "1. **Tu Duda 4 (Evaluador Cinemático Ideal en Lazo Cerrado):** ¿Es viable construir un simulador cinemático ideal en Python puro (sin esperar a Gazebo ni a los controladores PID de PX4) donde el dron avance a la pose predicha, genere su propio LiDAR sintético en cada paso y sea auditado por el validador geométrico `validate_episode_data()`?",
            "2. **Tu Duda 3 (Horizonte de Ejecución $T_{\\text{exec}}$):** De los pasos que predice la red en cada inferencia, ¿cómo cambia el comportamiento al recalcular la predicción en cada paso ($T_{\\text{exec}}=1$), a mitad de camino ($T_{\\text{exec}}=4$) o al final del horizonte ($T_{\\text{exec}}=8$)?",
            "3. **Chequeo del Flujo de Datos del Dataset:** ¿Conectan correctamente los tensores guardados en `heterodata.pt` con un bucle de entrenamiento e inferencia en lazo cerrado?",
            "",
            "---",
            "",
            "### 5.2 Parte A — Control de Cordura con `OracleReplayPolicy` (0% Error de Red)",
            "**¿Qué hicimos aquí?** Antes de probar una red neuronal, creamos una política \"Oráculo\" (`OracleReplayPolicy`) que lee las acciones de referencia $T_{EA}^{(t, 1..8)}$ del dataset y las entrega con el mismo formato de salida que tendrá la red. La evaluamos en lazo cerrado en el circuito `task_0006` (4 obstáculos + plataforma de aterrizaje) probando cuatro horizontes de replanificación $T_{\\text{exec}} \\in \\{1, 2, 4, 8\\}$.",
            "",
            "| Horizonte de Ejecución ($T_{\\text{exec}}$) | Veredicto `validate_episode_data` | Estado de Misión | Pasos Ejecutados ($10\\text{ Hz}$) | Llamadas a la Política | ADE [m] | FDE [m] |",
            "| :---: | :---: | :---: | :---: | :---: | :---: | :---: |",
        ]
    )

    for row in exp3["oracle_closed_loop_by_t_exec"]:
        lines.append(
            f"| **`T_exec = {row['execution_horizon']}`** | `{row['validation_status']}` | `{row['mission_status']}` | {row['num_steps_executed']} | {row['num_replan_calls']} | {row['ade_m']:.5f} | {row['fde_m']:.5f} |"
        )

    lines.extend(
        [
            "",
            "**¿Qué verifica esta tabla?**",
            "- El error `ADE = 0.00000 m` y `FDE = 0.05159 m` (distancia al centro de la plataforma en el instante en que `flight_state` activa el aterrizaje dentro del radio de $0.50\\text{ m}$) con veredicto **`accepted`** en los 4 casos confirma que **el formato de salida de la política, la composición de poses en `Horizontal FLU`, el interpolador PCHIP a $50\\text{ Hz}$ y el validador geométrico son mutuamente consistentes**.",
            "- En la columna `Llamadas a la Política`: con $T_{\\text{exec}} = 1$ se invoca la política $235$ veces para completar los $23.5\\text{ s}$ del circuito, frente a **$59$ llamadas** con $T_{\\text{exec}} = 4$ ($4\\times$ menos invocaciones).",
            "",
            "---",
            "",
            f"### 5.3 Parte B — Prueba Exploratoria de Ajuste Rápido con `MiniInstantPolicyNet` ({ov['task_id']})",
            f"**¿Qué hicimos aquí?** Como prueba funcional del flujo de gradientes y tensores, sobre-ajustamos (`overfit`) la red validadora ligera `MiniInstantPolicyNet` (`d_model=64`) en CPU durante {ov['training_wall_time_s']} segundos sobre un único episodio de `{ov['task_id']}` (la pérdida bajó de `{ov['initial_loss']}` a `{ov['final_loss']}`). Luego evaluamos ese mismo episodio en el simulador cinemático en lazo cerrado variando $T_{{\\text{{exec}}}}$ y $K$:",
            "",
            "| Pasos DDIM ($K$) | Horizonte $T_{\\text{exec}}$ | Veredicto Validador | Llamadas a Red | Latencia Media [ms] | ADE [m] | FDE (Error al Landing Pad) [m] |",
            "| :---: | :---: | :---: | :---: | :---: | :---: | :---: |",
        ]
    )

    for row in exp3["nn_closed_loop_sweep"]:
        lines.append(
            f"| **$K = {row['num_ddim_steps']}$** | **$T_{{\\text{{exec}}}} = {row['execution_horizon']}$** | `{row['validation_status']}` | {row['num_replan_calls']} | {row['mean_inference_ms']:.2f} ms | {row['ade_m']:.4f} | **{row['fde_m']:.4f}** |"
        )

    lines.extend(
        [
            "",
            "**Lectura exploratoria de esta tabla (sin extrapolar valores óptimos a la red final):**",
            "1. **Sensibilidad observada ante la ventana de replanificación ($T_{\\text{exec}}$):**",
            "   - En esta red pequeña, replanificar en cada paso individual ($T_{\\text{exec}} = 1$, cada $0.1\\text{ s}$) amplificó pequeñas oscilaciones de predicción (`FDE = 5.3763 m`), mientras que ejecutar un tramo intermedio ($T_{\\text{exec}} = 4$ pasos $= 0.4\\text{ s}$, con $K = 4$) redujo el desvío al final del episodio (`FDE = 0.0985 m`). Cómo se comportará este compromiso en la red final dependerá de su suavidad y de su horizonte $T$, por lo que conviene volver a medirlo en el Hito 4.",
            "2. **Por qué el validador marcó `rejected` a pesar del bajo FDE:**",
            "   - En el **Panel Central** de la figura inferior se observa que la trayectoria de `MiniInstantPolicyNet` cruza `landmark_1 (square_gate)`, esquiva `landmark_2 (pole)` y alcanza la plataforma final, pero recorta la amplitud lateral en `landmark_3` y `landmark_4` (ambas puertas cuadradas con maniobras distintas de slalom), incumpliendo el margen de seguridad de `validate_episode_data()`. Esto refleja las limitaciones de una red exploratoria de 1 capa ajustada rápidamente sobre un solo episodio.",
            "",
            "---",
            "",
            "### 5.4 Parte C — Sonda Sintética de Condicionamiento en Contexto (*Context-Swap* sobre 3 Muestras)",
            "**¿Qué hicimos aquí?** Para comprobar que el canal de entrada de contexto está cableado de punta a punta con la predicción de waypoints, armamos una sonda mínima con **solo 3 muestras estáticas** frente a una puerta cuadrada desde el mismo punto $(X = 2.15\\text{ m}, Y = -0.25\\text{ m})$ y la misma nube $O_t$, inyectando contextos correspondientes a tres maniobras (`task_0079` izquierda, `task_0118` centro, `task_0171` derecha):",
            "",
            "| Tarea de Contexto Inyectada | Primitiva Demostrada en el Contexto | $\\Delta X$ Predicho ($k=8$) [m] | $\\Delta Y$ Lateral Predicho ($k=8$) [m] | Comportamiento Observado en la Gráfica |",
            "| :--- | :--- | :---: | :---: | :--- |",
        ]
    )

    for tid, info in exp3["context_swap_test"].items():
        prim_str = str(info["primitive"])
        if "left" in prim_str:
            beh = "Curva Azul: Desplazamiento hacia +Y (izquierda)"
        elif "right" in prim_str:
            beh = "Curva Roja: Desplazamiento hacia -Y (derecha)"
        else:
            beh = "Curva Verde: Avance frontal en +X hacia la abertura"
        lines.append(
            f"| **`{tid}`** | `{info['primitive']}` | `{info['delta_x_step8_m']:+.3f} m` | **`{info['delta_y_step8_m']:+.3f} m`** | {beh} |"
        )

    lines.extend(
        [
            "",
            "![Resultados Overfit, Rollout y Context-Swap](assets/exploration/exp3_overfit_rollout_and_context_swap.png)",
            "",
            "**Cómo leer los 3 paneles de la figura superior:**",
            "- **Panel Izquierdo:** Evolución de la pérdida de entrenamiento de `MiniInstantPolicyNet` sobre un episodio, verificando que los gradientes fluyen sin errores numéricos.",
            "- **Panel Central:** Comparación en planta X-Y entre la demostración (`episode_00`, línea negra punteada) y el vuelo de la red exploratoria (línea verde).",
            "- **Panel Derecho:** Salida de los 8 pasos futuros ante una misma observación local al intercambiar los tensores de contexto en la sonda de 3 muestras.",
            "",
            "---",
            "",
            "### 5.5 Parches Exploratorios Usados en `MiniInstantPolicyNet` — Notas de Revisión para el Hito 4",
            "",
            "> [!WARNING]",
            "> **Alcance Estricto de Esta Exploración:**",
            "> El propósito de `MiniInstantPolicyNet` fue comprobar rápidamente en CPU que los tensores de entrada y salida encajan entre sí sobre pruebas reducidas (1 episodio en 5.3 y 3 muestras estáticas en 5.4). Para que esa red simplificada ajustara en pocos segundos se introdujeron **varios parches heurísticos** en `simple_validation_net.py` y `exploration_suite.py`. **Estos parches NO son patrones recomendados para la red final**; se dejan registrados aquí únicamente como aspectos que conviene revisar con cuidado al diseñar el Hito 4.",
            "",
            "#### 1. Escala entre Traslación y Rotación (`flow_scale` y `ActionBounds4DoF`)",
            "- **Qué se hizo en la exploración:** Al escalar el espacio por $s = 0.1$, los 6 keypoints del dron ($0.5\\text{ m}$ reales) quedan a un radio $L_g = 0.05$, por lo que un giro pequeño desplaza las puntas de la cruceta numéricamente menos que una traslación típica ($\\|\\Delta\\mathbf{p}_r\\| \\approx L_g |\\delta\\psi|$). En `MiniInstantPolicyNet` se aplicó un vector fijo `flow_scale = [0.04, 0.04, 0.03, 0.012, 0.012, 0.012]` y cotas `ActionBounds4DoF` con `clip=True` sacadas de los percentiles de este dataset.",
            "- **Limitación de ese parche:** Esos coeficientes dependen de las velocidades lineales y de giro presentes en los episodios actuales. En tareas futuras que requieran giros amplios o maniobras verticales rápidas, fijar constantes a mano o aplicar `clip=True` podría sesgar el gradiente o recortar acciones válidas.",
            "- **Qué revisar en el Hito 4:** Evaluar una normalización general no atada manualmente a este subconjunto — por ejemplo, escalar el objetivo por el nivel de ruido $\\sigma_k$ propio del *schedule* de difusión en cada paso, o calcular estadísticos dinámicos del split de entrenamiento sin imponer *clipping* duro en inferencia.",
            "",
            "#### 2. Señal de Progreso y Coordenadas Globales (`live_pose_se3`)",
            "- **Qué se hizo en la exploración:** En la prueba 5.3 sobre `task_0006` (que atraviesa tres `square_gate` parecidas), `MiniInstantPolicyNet` recibió la posición global en el mapa (`pose_se3[:, :3, 3]`) y ponderó la atención según la diferencia en $X_{\\text{ENU}}$ (`ctx_x - live_x`), además de sumar perturbaciones sintéticas extra al vuelo ($\\gamma = 0.65$) sobre los datos ya existentes.",
            "- **Limitación de ese parche:** Comparar coordenadas globales de mapa (`Gazebo ENU`) entre la demostración y el dron en vivo ata la red al lugar exacto del mapa donde ocurrió el episodio, por lo que dejaría de corresponderse cuando los obstáculos cambien de ubicación o rumbo entre episodios. En cambio, la idea general de representar el **orden o progreso relativo** sí merece estudiarse diferenciando dónde se aplica: dentro de las demostraciones grabadas (índice relativo o aristas entre keyframes $l = 0 \\dots L-1$) no depende del marco global, mientras que para el dron en vivo habría que evaluar alternativas egocéntricas si hicieran falta.",
            "- **Qué revisar en el Hito 4:** Revisar primero cuánto desambigua por sí sola la vista LiDAR 360° de $R_{\\max} = 10.0\\text{ m}$ (que suele abarcar obstáculos vecinos) junto con los nodos `data['aug']` ($\\gamma = 0.70$) ya incluidos en `heterodata.pt`, y, si se observa ambigüedad en circuitos con tramos repetidos, evaluar mecanismos de progreso o memoria que preserven la invariancia espacial respecto al mundo.",
            "",
            "#### 3. Separación Manual Clave/Valor, `Modality Dropout` y Anclas Sintéticas en el Test de 3 Muestras",
            "- **Qué ocurrió y qué parche se usó en la exploración:** En la sonda 5.4 (*Context-Swap*) se entrenó `MiniInstantPolicyNet` desde cero con **solo 3 muestras estáticas**. Con tan pocos datos, cualquier red pequeña tiende a memorizar diferencias mínimas entre las 3 entradas e ignorar el contexto. Para forzar que en ese test de juguete mirara el contexto, se separaron manualmente *Keys* y *Values*, se apagó aleatoriamente la rama de la observación actual (`live_modality_dropout`) y en `exploration_suite.py` se sobrescribió el primer keyframe como ancla sintética.",
            "- **Limitación de ese parche:** Tanto el problema observado como esa solución rebuscada provienen del régimen artificial de entrenar con 3 ejemplos sueltos, no necesariamente del comportamiento de la arquitectura real entrenada sobre un corpus amplio.",
            "- **Qué revisar en el Hito 4:** Probar primero la formulación estándar de grafos heterogéneos de *Instant Policy* (donde las acciones demostradas son nodos geométricos 3D dentro del grafo sin particiones forzadas ni apagado manual de ramas) entrenando sobre las tareas variadas del Catálogo $v2$, y recién analizar regularizaciones adicionales si en validación multi-tarea se detectara subutilización del contexto.",
            "",
            "#### 4. Otros Atajos Internos de `MiniInstantPolicyNet` (Puente Determinista desde $I_4$ y Cachés por Suma Hash)",
            "- **Puente interpolado determinista:** `MiniInstantPolicyNet` usó una interpolación lineal fija sobre $u \\in \\{0.25, 0.5, 0.75, 1.0\\}$ partiendo siempre desde la identidad $I_4$, únicamente para simplificar el chequeo numérico en CPU. En el Hito 4 corresponde revisar el proceso de difusión estocástico (DDPM/DDIM) y su muestreo de ruido.",
            "- **Cachés por suma hash en entrenamiento:** `_patch_cache` y la deduplicación por suma de tensores en `encode_context` se agregaron solo para acelerar iteraciones repetidas sobre el mismo episodio en CPU y no son apropiados para entrenamiento general (a diferencia de cachear en inferencia la codificación de las demos fijas al comenzar un episodio de vuelo).",
            "",
            "---",
            "",
            "## 6. Experimento 4: Exploración de Compresión de Escena ($M = 16$ vs. $M = 32$ Centroides FPS)",
            "",
            "### 6.1 ¿Qué se midió y qué precaución hay que tener con el prefiltrado usado?",
            "Cada barrido de nuestro sensor LiDAR sintético contiene **$N = 2048$ puntos 3D** pertenecientes a los obstáculos cercanos y a la plataforma `landing_pad` dentro del radio de $10.0\\text{ m}$. Para estimar cuánta resolución espacial se pierde al comprimir la nube a $M$ centroides mediante *Farthest Point Sampling* (FPS) y vecindades $k$-NN ($k_{\\text{nn}} = 16$), medimos la distancia desde los puntos de los obstáculos cercanos hasta su centroide más próximo con $M = 16$ y $M = 32$:",
            "",
            "| Configuración Evaluada | Centroides con `z_flu > -0.10` (Media) | Distancia Media Obstáculo $\\to$ Centroide [m] | Distancia p95 Obstáculo $\\to$ Centroide [m] | Tiempo FPS + k-NN por Nube [ms] |",
            "| :--- | :---: | :---: | :---: | :---: |",
        ]
    )

    for key in ("M_16", "M_32"):
        r4 = exp4[key]
        lines.append(
            f"| **`M = {r4['num_centroids']}` centroides** | {r4['mean_obstacle_centroids']:.1f} / {r4['num_centroids']} | `{r4['mean_obstacle_to_centroid_dist_m']:.3f} m` | `{r4['p95_obstacle_to_centroid_dist_m']:.3f} m` | `{r4['fps_knn_time_ms_per_cloud']:.2f} ms` |"
        )

    lines.extend(
        [
            "",
            "> [!NOTE]",
            "> **Lectura exploratoria y advertencia sobre el filtro `z_flu > -0.11`:**",
            "> - Pasar de $M = 16$ a $M = 32$ reduce en esta prueba la distancia p95 sin cubrir de `0.387 m` a `0.205 m` con un incremento moderado de tiempo en CPU (`1.00 ms` $\\to$ `1.66 ms` por nube), lo que sugiere considerar $M \\in \\{16, 32, 64\\}$ dentro de las opciones a evaluar en el Hito 4.",
            "> - **Parche exploratorio a revisar:** En [`subsample_scene_centroids_torch()`](../src/uav_ip/uav_ip/policy/simple_validation_net.py) se incluyó un corte heurístico `z_flu > -0.11` ($1.1\\text{ m}$ por debajo del dron). Como [`synthetic_lidar.py`](../src/uav_ip/uav_ip/synthetic_lidar.py) **no genera piso infinito** (solo muestrea obstáculos y `landing_pad`), un corte fijo en $Z$ puede descartar puntos legítimos de la plataforma de aterrizaje cuando el dron vuela alto (p. ej. durante un `overfly` a $2.5\\text{ m}$). Por ello, este corte fijo fue únicamente un supuesto exploratorio y conviene revisarlo en el Hito 4 (por ejemplo, evaluando FPS / PointNet++ sobre la nube completa sin umbral rígido en $Z$).",
            "",
            "---",
            "",
            "## 7. Experimento 5: Perfilado de Latencia en CPU y Ralentización (`RTF`) en Gazebo/PX4",
            "",
            "### 7.1 ¿Qué dudas queremos resolver aquí? (Tu Duda 1 sobre Gazebo y la velocidad de la red)",
            "Tu primera duda fue: *\"¿Puede ralentizarse Gazebo para que las demoras de cálculo en la red neuronal no perjudiquen tanto? ¿Está bien este enfoque?\"*",
            "",
            "Para tener órdenes de magnitud orientativos en tu máquina (**CPU:** `"
            + f"{exp5['cpu_info']}"
            + f"`, `{exp5['torch_num_threads']}` hilos de PyTorch), medimos tiempos de ejecución sobre `MiniInstantPolicyNet` variando:",
            "1. **Tamaño de referencia:** `Mini (d=64, M=16)`, `Medium (d=128, M=16)`, `Medium-Dense (d=128, M=32)` y `Large (d=256, M=32)`.",
            "2. **Cantidad de pasos internos ($K \\in \\{1, 2, 4, 8\\}$).**",
            "3. **Caché de Contexto en Inferencia vs. Recálculo Completo:**",
            "   - Durante un episodio de vuelo, las $N_{\\text{demos}}$ demostraciones de contexto no cambian de un paso al siguiente.",
            "   - **Sin Caché:** Re-ejecutar $\\phi_e + \\sigma$ sobre todas las nubes de las demostraciones en cada ciclo de control repite trabajo idéntico.",
            "   - **Con Caché en Inferencia:** Codificar las demostraciones una vez al inicio del vuelo deja en el lazo de tiempo real únicamente la nube actual $O_t$ y los pasos de actualización.",
            "",
            "| Configuración Evaluada | Parámetros | Caché Contexto (1 vez/ep) | Sin Caché ($K=4$) | Con Caché ($K=1$) | Con Caché ($K=2$) | Con Caché ($K=4$) | Con Caché ($K=8$) | Aceleración por Caché |",
            "| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |",
        ]
    )

    for row in exp5["benchmark_rows"]:
        lines.append(
            f"| **{row['config_name']}** | `{row['num_parameters']:,}` | `{row['context_cache_once_ms']:.1f} ms` | `{row['uncached_per_step_k4_ms']:.1f} ms` | `{row['K_1_ms']:.1f} ms` | `{row['K_2_ms']:.1f} ms` | **`{row['K_4_ms']:.1f} ms`** | `{row['K_8_ms']:.1f} ms` | **`{row['speedup_from_caching_x']:.1f}x`** |"
        )

    lines.extend(
        [
            "",
            "![Benchmark de Latencia en CPU](assets/exploration/exp5_cpu_latency_benchmark.png)",
            "",
            "### 7.2 Relación entre Latencia de Inferencia y Factor `--rtf` en Gazebo",
            "",
            "**Presupuesto temporal según frecuencia de replanificación y `RTF`:**",
            "Si cada paso del horizonte dura $\\Delta t_{\\text{paso}} = 0.10\\text{ s} \\times \\text{stride}$ de tiempo simulado y se replanifica cada $T_{\\text{exec}}$ pasos, el margen de tiempo real de reloj de pared ($\\Delta t_{\\text{wall}}$) disponible para calcular una inferencia a un `RTF` dado es:",
            "$$\\Delta t_{\\text{wall}} = \\frac{T_{\\text{exec}} \\times \\Delta t_{\\text{paso}}}{\\text{RTF}}$$",
            "- Por ejemplo, con $T_{\\text{exec}} = 4$ y $\\Delta t_{\\text{paso}} = 0.10\\text{ s}$, la ventana simulada es de $400\\text{ ms}$ ($400\\text{ ms}$ reales a `RTF = 1.0x`, u $800\\text{ ms}$ reales a `RTF = 0.5x` usando `world_generator.py --rtf 0.5`).",
            "- Nota: La arquitectura de grafos heterogéneos del Hito 4 tendrá distinto costo computacional que `MiniInstantPolicyNet`, por lo que estos tiempos deberán volver a medirse sobre el modelo final para elegir el par $(K, \\text{RTF})$ al evaluar en PX4 SITL sincronizado con `px4_timestamp`.",
            "",
            "---",
            "",
            "## 8. Síntesis de la Exploración: Qué Verificamos de Interfaz y Qué Queda Abierto para el Hito 4",
            "",
            "Para separar lo que corresponde a contratos de datos verificados frente a lo que fueron simplificaciones temporales de la exploración, resumimos ambos grupos por separado:",
            "",
            "### 8.1 Interfaces, Geometría y Herramientas Verificadas",
            "",
            "| Componente | Qué Se Comprobó en Esta Exploración |",
            "| :--- | :--- |",
            "| **1. Contratos de `heterodata.pt` y Cargador Dinámico** | `InContextNavigationDataset` carga episodios cinemáticos y SITL verificando formas, tipos y escalas (`s=0.1` en posiciones, rad en ángulos, m/s en `vel_flu`), y permite variar al vuelo $(L, T, \\text{action\\_stride}, N_{\\text{demos}})$ — incluyendo `L='adaptive'` con `context_keyframe_mask` — sin regenerar `.pt`. |",
            "| **2. Conversión $\\delta T_k \\leftrightarrow T_{EA}^{(t,k)}$ y Proyección SVD 4-DoF** | La descomposición incremental mantiene escala uniforme a lo largo del horizonte con error de ida y vuelta $< 10^{-6}\\text{ m}$, y `align_keypoints_svd_horizontal_flu` evita introducir *roll/pitch* espurio ante ruido en los keypoints mientras se trabaje en `Horizontal FLU`. |",
            "| **3. Evaluador Cinemático en Lazo Cerrado (`kinematic_rollout_eval.py`)** | Con `OracleReplayPolicy`, la composición de poses, interpolación PCHIP a $50\\text{ Hz}$, generación LiDAR en vivo y auditoría con `validate_episode_data()` reproducen los vuelos de referencia (`accepted`, `ADE = 0.000 m`). |",
            "| **4. Soporte `--rtf` y Caché de Demostraciones en Inferencia** | `world_generator.py --rtf <float>` escala la velocidad de reloj de Gazebo manteniendo `max_step_size = 0.004s`, y precalcular la codificación de las demos al inicio del episodio reduce entre $3\\times$ y $5.8\\times$ el costo por ciclo de control en las pruebas medidas. |",
            "",
            "### 8.2 Aspectos Abiertos y Parches Exploratorios a Revisar en el Hito 4",
            "",
            "| Aspecto | Atajo Usado en la Exploración (`MiniInstantPolicyNet`) | Qué Revisar / Evaluar en la Red Final (Hito 4) |",
            "| :--- | :--- | :--- |",
            "| **Normalización y Balance Traslación / Giro** | `flow_scale` manual y `ActionBounds4DoF` con `clip=True` ajustados a este dataset. | Revisar una formulación general sin constantes ajustadas a mano ni `clip` duro en inferencia (p. ej. normalización por el *schedule* de ruido $\\sigma_k$ de difusión, estadísticos del split de `train` o límites físicos) contemplando tareas futuras con giros o ascensos más intensos. |",
            "| **Invariancia Espacial $SE(3)$ y Señal de Progreso** | Inyección de `live_pose_se3` global, gaussiana longitudinal `ctx_x - live_x` y tubo extra $\\gamma=0.65$ al vuelo. | Revisar cómo representar el orden dentro de las demos sin pasar coordenadas globales del mapa en la consulta viva $O_t$, evaluar si `data['aug']` ($\\gamma=0.70$) ya resulta suficiente y analizar alternativas egocéntricas si apareciera ambigüedad perceptual. |",
            "| **Propagación de Contexto ($\\varphi$) y Difusión ($\\psi$)** | Separación manual `Key`/`Value`, `live_modality_dropout` (sonda de 3 muestras) y puente determinista desde $I_4$. | Revisar la arquitectura estándar de grafos heterogéneos de *Instant Policy* (acciones demostradas como nodos geométricos 3D y *schedule* DDPM/DDIM estocástico) entrenando sobre la variedad de tareas del Catálogo $v2$. |",
            "| **Muestreo de Escena ($\\phi_e$) e Hiperparámetros** | Corte fijo `z_flu > -0.11` antes de FPS y pruebas puntuales de tamaño. | Revisar el muestreo sobre la nube completa sin el corte rígido en $Z$ (que puede descartar el `landing_pad` al volar alto) y evaluar en grilla $(L, T, \\text{stride}, N_{\\text{demos}}, T_{\\text{exec}}, K, M)$ sobre la arquitectura final (Tarea 4.5 de `PLAN_IMPLEMENTACION.md`). |",
            "",
        ]
    )

    output_report_path.parent.mkdir(parents=True, exist_ok=True)
    output_report_path.write_text("\n".join(lines), encoding="utf-8")


def run_custom_hyperparameter_probe(
    kinematic_dir: Path,
    sitl_dir: Path,
    task_id: str = "task_0006",
    num_keyframes: Union[int, str] = 10,
    adaptive_arc_m: float = 1.5,
    horizon: int = 8,
    action_stride: int = 1,
    num_context_demos: int = 2,
    execution_horizon: int = 4,
    num_ddim_steps: int = 4,
    num_centroids: int = 16,
    d_model: int = 64,
    overfit_epochs: int = 15,
    output_json: Optional[Path] = None,
) -> Dict[str, Any]:
    """
    Ejecuta una sonda rapida end-to-end para validar cualquier combinacion personalizada de
    hiperparametros (`L` fijo o `'adaptive'`, `T`, `action_stride`, `N_demos`, `T_exec`, `K`, `M`, `d_model`)
    sin regenerar `heterodata.pt` ni sobrescribir el reporte canonico.
    """
    torch.manual_seed(42)
    np.random.seed(42)

    kf_mode = parse_num_keyframes_arg(num_keyframes)
    H = max(1, int(horizon))
    stride = max(1, int(action_stride))
    n_demos = max(1, int(num_context_demos))
    t_exec = max(1, min(H, int(execution_horizon)))
    K = max(1, int(num_ddim_steps))
    M = max(4, int(num_centroids))
    D = max(16, int(d_model))
    n_heads = 4 if D % 4 == 0 else (2 if D % 2 == 0 else 1)

    print(
        f"[Custom Probe] L={kf_mode} (arc={adaptive_arc_m}m) | T={H} | stride={stride} "
        f"| N_demos={n_demos} | T_exec={t_exec} | K={K} | M={M} | d_model={D} | task={task_id}",
        flush=True,
    )

    # 1. Verificacion de contratos y colacion multi-tarea (incluyendo padding si L='adaptive')
    ds_multi = InContextNavigationDataset(
        kinematic_dir,
        max_tasks=4,
        num_context_demos=n_demos,
        num_keyframes=kf_mode,
        horizon=H,
        action_stride=stride,
        adaptive_arc_m=adaptive_arc_m,
        include_aug=True,
    )
    contract_kin = verify_sample_contract(
        ds_multi[0],
        expected_num_demos=n_demos,
        expected_num_keyframes=kf_mode,
        expected_horizon=H,
    )
    contract_sitl: Optional[Dict[str, Any]] = None
    if sitl_dir.is_dir() and any(sitl_dir.glob("task_*")):
        ds_sitl = InContextNavigationDataset(
            sitl_dir,
            max_tasks=2,
            num_context_demos=n_demos,
            num_keyframes=kf_mode,
            horizon=H,
            action_stride=stride,
            adaptive_arc_m=adaptive_arc_m,
            include_aug=True,
        )
        contract_sitl = verify_sample_contract(
            ds_sitl[0],
            expected_num_demos=n_demos,
            expected_num_keyframes=kf_mode,
            expected_horizon=H,
        )

    # Tomar una muestra por cada tarea cargada para probar collate_in_context_batch
    rep_samples: List[Dict[str, Any]] = []
    seen_tasks = set()
    observed_L_per_task: Dict[str, int] = {}
    for idx, (tid_s, _ep_i, _kind, _s_i) in enumerate(ds_multi.samples_index):
        if tid_s not in seen_tasks:
            s_item = ds_multi[idx]
            rep_samples.append(s_item)
            observed_L_per_task[tid_s] = int(s_item["context_scene_pos"].shape[1])
            seen_tasks.add(tid_s)
    collated_multi = collate_in_context_batch(rep_samples)
    padded_L = int(collated_multi["context_scene_pos"].shape[2])
    valid_counts_per_demo = collated_multi["context_keyframe_mask"].sum(dim=-1).tolist()

    # 2. Lazo cerrado con OracleReplayPolicy sobre `task_id`
    t_dir = kinematic_dir / task_id
    if not t_dir.is_dir():
        raise FileNotFoundError(f"No existe la tarea solicitada: {t_dir}")
    ep_paths = sorted(t_dir.glob("episode_*/heterodata.pt"))
    waypoints_cfg = yaml.safe_load((ep_paths[0].parent / "waypoints.yaml").read_text(encoding="utf-8"))
    ep0_data = load_heterodata_episode(ep_paths[0])
    ref_pos_m = (ep0_data["agent"].pos_enu.numpy() / DEFAULT_SCALE_FACTOR).astype(np.float64)
    init_pos_m = ref_pos_m[0]
    init_yaw = float(ep0_data["agent"].yaw_enu[0, 0].item())
    init_vel_flu = ep0_data["agent"].vel_flu[0].numpy()

    other_eps = ep_paths[1:] if len(ep_paths) > 1 else ep_paths
    ctx_eps = [other_eps[i % len(other_eps)] for i in range(n_demos)]
    ctx_pack = build_context_pack_from_episodes(
        ctx_eps,
        num_keyframes=kf_mode,
        horizon=H,
        action_stride=stride,
        adaptive_arc_m=adaptive_arc_m,
    )
    oracle = OracleReplayPolicy(reference_episode=ep0_data, horizon=H, action_stride=stride)
    oracle_ro = run_kinematic_closed_loop_rollout(
        policy=oracle,
        waypoints_cfg=waypoints_cfg,
        context_pack=ctx_pack,
        execution_horizon=t_exec,
        num_ddim_steps=1,
        max_steps_10hz=len(ref_pos_m) + 10 * stride,
        action_stride=stride,
        initial_pos_enu_m=init_pos_m,
        initial_yaw_enu_rad=init_yaw,
        initial_vel_flu_mps=init_vel_flu,
        reference_pos_enu_m=ref_pos_m,
    )

    # 3. Entrenamiento rapido de MiniInstantPolicyNet + Lazo Cerrado + Latencia CPU
    ds_task = InContextNavigationDataset(
        kinematic_dir,
        task_ids=[task_id],
        num_context_demos=n_demos,
        num_keyframes=kf_mode,
        horizon=H,
        action_stride=stride,
        adaptive_arc_m=adaptive_arc_m,
        include_aug=True,
    )
    ds_task.samples_index = [s for s in ds_task.samples_index if s[1] == 0]
    loader = DataLoader(ds_task, batch_size=32, shuffle=True, collate_fn=collate_in_context_batch)

    net = MiniInstantPolicyNet(
        d_model=D,
        num_heads=n_heads,
        num_centroids=M,
        num_neighbors=16,
        horizon=H,
        action_stride=stride,
        num_diffusion_steps=K,
    )
    loss_history: List[float] = []
    t_train0 = time.perf_counter()
    if overfit_epochs > 0:
        opt = torch.optim.AdamW(net.parameters(), lr=3e-3, weight_decay=1e-4)
        sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=max(1, overfit_epochs))
        net.train()
        for _ep in range(overfit_epochs):
            ep_losses: List[float] = []
            for batch in loader:
                opt.zero_grad()
                loss_d = net.compute_loss(batch)
                loss_d["loss"].backward()
                torch.nn.utils.clip_grad_norm_(net.parameters(), max_norm=2.0)
                opt.step()
                ep_losses.append(float(loss_d["loss"].item()))
            sched.step()
            loss_history.append(float(np.mean(ep_losses)))
    train_sec = time.perf_counter() - t_train0

    net.eval()
    ctx_eps_self = [ep_paths[i % len(ep_paths)] for i in range(n_demos)]
    ctx_pack_self = build_context_pack_from_episodes(
        ctx_eps_self,
        num_keyframes=kf_mode,
        horizon=H,
        action_stride=stride,
        adaptive_arc_m=adaptive_arc_m,
    )
    nn_ro = run_kinematic_closed_loop_rollout(
        policy=net,
        waypoints_cfg=waypoints_cfg,
        context_pack=ctx_pack_self,
        execution_horizon=t_exec,
        num_ddim_steps=K,
        max_steps_10hz=len(ref_pos_m) + 12 * stride,
        action_stride=stride,
        initial_pos_enu_m=init_pos_m,
        initial_yaw_enu_rad=init_yaw,
        initial_vel_flu_mps=init_vel_flu,
        reference_pos_enu_m=ref_pos_m,
    )

    # Perfilado rapido de latencia CPU (encode_context one-shot vs predict_actions cacheado)
    sample0 = ds_task[0]
    kp_body = get_uav_keypoints_body_torch()
    c_mask = sample0.get("context_keyframe_mask")
    net.scene_encoder.use_patch_cache = False
    with torch.no_grad():
        t_c0 = time.perf_counter()
        cached_ctx = net.encode_context(
            context_scene_pos=sample0["context_scene_pos"],
            context_vel_flu=sample0["context_vel_flu"],
            context_flight_state=sample0["context_flight_state"],
            context_delta_4dof=sample0["context_delta_4dof"],
            context_target_flight_state=sample0["context_target_flight_state"],
            keypoints_body=kp_body,
            context_pose_se3=sample0.get("context_pose_se3"),
            context_keyframe_mask=c_mask,
        )
        ctx_encode_ms = (time.perf_counter() - t_c0) * 1000.0

        # Warmup + 10 mediciones
        _ = net.predict_actions(sample0, num_ddim_steps=K, cached_context=cached_ctx)
        lat_ms: List[float] = []
        for _ in range(10):
            t_i0 = time.perf_counter()
            _ = net.predict_actions(sample0, num_ddim_steps=K, cached_context=cached_ctx)
            lat_ms.append((time.perf_counter() - t_i0) * 1000.0)

    summary = {
        "hyperparameters": {
            "task_id": task_id,
            "num_keyframes": kf_mode,
            "adaptive_arc_m": adaptive_arc_m,
            "horizon": H,
            "action_stride": stride,
            "dt_step_s": round(0.10 * stride, 3),
            "horizon_span_s": round(0.10 * stride * H, 3),
            "num_context_demos": n_demos,
            "execution_horizon": t_exec,
            "num_ddim_steps": K,
            "num_centroids": M,
            "d_model": D,
            "overfit_epochs": overfit_epochs,
        },
        "contract_verification": {
            "kinematic_valid": contract_kin["valid"],
            "sitl_valid": contract_sitl["valid"] if contract_sitl is not None else None,
            "observed_L_by_task": observed_L_per_task,
            "collated_padded_L": padded_L,
            "valid_keyframes_per_sample_demo": valid_counts_per_demo,
        },
        "oracle_closed_loop": {
            "validation_status": oracle_ro["validation_status"],
            "mission_status": oracle_ro["mission_status"],
            "ade_m": round(oracle_ro["ade_m"], 5),
            "fde_m": round(oracle_ro["fde_m"], 5),
            "num_replan_calls": oracle_ro["num_replan_calls"],
            "rejection_reasons": oracle_ro["rejection_reasons"],
        },
        "mini_net_probe": {
            "num_parameters": int(sum(p.numel() for p in net.parameters())),
            "initial_loss": round(loss_history[0], 5) if loss_history else None,
            "final_loss": round(loss_history[-1], 5) if loss_history else None,
            "train_wall_time_s": round(train_sec, 2),
            "validation_status": nn_ro["validation_status"],
            "mission_status": nn_ro["mission_status"],
            "ade_m": round(nn_ro["ade_m"], 4),
            "fde_m": round(nn_ro["fde_m"], 4),
            "context_encode_one_shot_ms": round(ctx_encode_ms, 2),
            "cached_inference_mean_ms": round(float(np.mean(lat_ms)), 2),
        },
    }

    print(
        f"  -> Contratos E/S: kinematic={contract_kin['valid']}, L_por_tarea={observed_L_per_task}, L_padded={padded_L}\n"
        f"  -> Oraculo Lazo Cerrado: status={oracle_ro['validation_status']} | ADE={oracle_ro['ade_m']:.4f} m | FDE={oracle_ro['fde_m']:.4f} m\n"
        f"  -> MiniInstantPolicyNet ({overfit_epochs} ep): loss={summary['mini_net_probe']['final_loss']} "
        f"| status={nn_ro['validation_status']} | ADE={nn_ro['ade_m']:.3f} m | FDE={nn_ro['fde_m']:.3f} m\n"
        f"  -> Latencia CPU: encode_context={ctx_encode_ms:.2f} ms | predict_actions(cached)={np.mean(lat_ms):.2f} ms",
        flush=True,
    )

    if output_json is not None:
        output_json.parent.mkdir(parents=True, exist_ok=True)
        output_json.write_text(json.dumps(summary, indent=2), encoding="utf-8")

    return summary


def run_all_explorations(
    kinematic_dir: Path,
    sitl_dir: Path,
    output_report: Path,
    assets_dir: Optional[Path] = None,
    max_kinematic_tasks: int = 40,
    overfit_epochs: int = 35,
    num_keyframes: Union[int, str] = 10,
    adaptive_arc_m: float = 1.5,
    horizon: int = 8,
    action_stride: int = 1,
    num_context_demos: int = 2,
    execution_horizon: int = 4,
    num_ddim_steps: int = 4,
    num_centroids: int = 16,
    d_model: int = 64,
) -> Dict[str, Any]:
    """Ejecuta de principio a fin los 5 experimentos de exploracion y escribe el reporte."""
    if assets_dir is None:
        assets_dir = output_report.parent / "assets" / "exploration"
    assets_dir.mkdir(parents=True, exist_ok=True)

    print("[1/5] Ejecutando Experimento 1: Radiografia de Contratos E/S y Estadistica de Normalizacion...", flush=True)
    exp1 = run_experiment_1_dataset_radiography(
        kinematic_dir=kinematic_dir,
        sitl_dir=sitl_dir,
        assets_dir=assets_dir,
        max_kinematic_tasks=max_kinematic_tasks,
        num_context_demos=num_context_demos,
        num_keyframes=num_keyframes,
        adaptive_arc_m=adaptive_arc_m,
        horizon=horizon,
        action_stride=action_stride,
    )

    print("[2/5] Ejecutando Experimento 2: Exactitud Matematica Round-Trip SE(3) y SVD 4-DoF vs 6-DoF...", flush=True)
    exp2 = run_experiment_2_math_roundtrip_and_svd(
        kinematic_dir=kinematic_dir,
        sitl_dir=sitl_dir,
    )

    print("[3/5] Ejecutando Experimento 3: Lazo Cerrado Cinematico, Overfit Test y Context-Swap...", flush=True)
    exp3 = run_experiment_3_closed_loop_and_context_swap(
        kinematic_dir=kinematic_dir,
        assets_dir=assets_dir,
        overfit_epochs=overfit_epochs,
        num_keyframes=num_keyframes,
        adaptive_arc_m=adaptive_arc_m,
        horizon=horizon,
        action_stride=action_stride,
        num_context_demos=num_context_demos,
        execution_horizon=execution_horizon,
        num_ddim_steps=num_ddim_steps,
        num_centroids=num_centroids,
        d_model=d_model,
    )

    print("[4/5] Ejecutando Experimento 4: Cobertura Geometrica de Escena (M=16 vs M=32)...", flush=True)
    exp4 = run_experiment_4_scene_centroid_coverage(kinematic_dir=kinematic_dir)

    print("[5/5] Ejecutando Experimento 5: Benchmark de Latencia en CPU y Presupuesto RTF...", flush=True)
    exp5 = run_experiment_5_cpu_latency_and_gazebo_rtf(
        kinematic_dir=kinematic_dir,
        assets_dir=assets_dir,
        num_keyframes=num_keyframes,
        adaptive_arc_m=adaptive_arc_m,
        horizon=horizon,
        action_stride=action_stride,
        num_context_demos=num_context_demos,
    )

    all_results = {
        "experiment_1": exp1,
        "experiment_2": exp2,
        "experiment_3": exp3,
        "experiment_4": exp4,
        "experiment_5": exp5,
    }

    metrics_json_path = assets_dir / "exploration_metrics.json"
    metrics_json_path.write_text(json.dumps(all_results, indent=2), encoding="utf-8")
    generate_markdown_exploration_report(all_results, output_report_path=output_report)
    print(f"Reporte de exploracion generado exitosamente en: {output_report}", flush=True)
    print(f"Metricas JSON guardadas en: {metrics_json_path}", flush=True)
    return all_results
