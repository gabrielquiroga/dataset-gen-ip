"""
in_context_dataset.py — Muestreador In-Context y Contratos de Tensores E/S para UAV Instant Policy.

Carga episodios `heterodata.pt` agrupados por tarea y construye muestras de entrenamiento
e inferencia parametrizables sin regenerar los archivos `.pt`:
  1. `N_demos` demostraciones de contexto muestreadas en `L` keyframes (entero fijo o `'adaptive'`
     proporcional a la longitud de arco mediante `adaptive_arc_m`).
  2. La observacion actual (Query O_t), proveniente tanto de pasos nominales (`data['agent']`)
     como de pasos perturbados de recuperacion (`data['aug']`, Tube Augmentation 30%).
  3. Las etiquetas objetivo para horizonte futuro arbitrario `T = horizon` y paso `action_stride`
     (tanto acumuladas `T_{EA}^{(t, 1..T)}` como incrementales `δT_{1..T}` y `s_{t+1..t+T}`).
"""

from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple, Union

import numpy as np
import torch
from torch.utils.data import Dataset
import yaml

from .se3_diffusion_math import (
    DEFAULT_SCALE_FACTOR,
    build_se3_horizontal_flu_torch,
    cumulative_to_incremental_se3,
    get_uav_keypoints_body_torch,
    invert_rigid_se3_torch,
    se3_to_4dof_vector,
    wrap_angle_torch,
)

# Especificacion formal de contratos de Entrada/Salida para documentacion y validacion
TENSOR_CONTRACT_SPEC: List[Dict[str, str]] = [
    {
        "stage": "Context Input",
        "name": "context_scene_pos",
        "shape": "(N_demos, L, 2048, 3)",
        "dtype": "float32",
        "frame": "Horizontal FLU (egocentrico, roll=pitch=0)",
        "scale": "s = 0.1 (1 unidad = 10 m)",
        "description": "Nubes LiDAR 360° en los L keyframes (fijo o adaptativo) de cada demostracion.",
    },
    {
        "stage": "Context Input",
        "name": "context_pose_se3",
        "shape": "(N_demos, L, 4, 4)",
        "dtype": "float32",
        "frame": "Gazebo ENU estabilizado",
        "scale": "Traslacion s = 0.1; Rotacion R_z(yaw) in SO(2)",
        "description": "Poses T_{WE}^{t_l} de los keyframes (usadas solo para aristas relativas intra-demo).",
    },
    {
        "stage": "Context Input",
        "name": "context_vel_flu",
        "shape": "(N_demos, L, 3)",
        "dtype": "float32",
        "frame": "Horizontal FLU",
        "scale": "m/s (fisico directo)",
        "description": "Velocidad lineal propioceptiva [vx_flu, vy_flu, vz_flu] en cada keyframe.",
    },
    {
        "stage": "Context Input",
        "name": "context_flight_state",
        "shape": "(N_demos, L, 1)",
        "dtype": "float32",
        "frame": "Escalar binario",
        "scale": "{0.0, 1.0}",
        "description": "Estado de vuelo en el keyframe (1.0 = vuelo activo, 0.0 = aterrizaje).",
    },
    {
        "stage": "Context Input",
        "name": "context_action_se3",
        "shape": "(N_demos, L, T, 4, 4)",
        "dtype": "float32",
        "frame": "Horizontal FLU relativo al keyframe l",
        "scale": "Traslacion s = 0.1; Rotacion R_z(dyaw)",
        "description": "Horizonte futuro acumulado T_{EA}^{(t_l, 1..T)} demostrado en cada keyframe.",
    },
    {
        "stage": "Context Input",
        "name": "context_delta_se3",
        "shape": "(N_demos, L, T, 4, 4)",
        "dtype": "float32",
        "frame": "Horizontal FLU paso a paso (k-1 -> k)",
        "scale": "Traslacion s = 0.1 (Δt = 0.1 s * stride)",
        "description": "Incrementos paso a paso δT_k = (T_{EA}^{k-1})^{-1} T_{EA}^k en cada keyframe.",
    },
    {
        "stage": "Context Input",
        "name": "context_keyframe_mask",
        "shape": "(N_demos, L)",
        "dtype": "bool",
        "frame": "Mascara booleana",
        "scale": "{True, False}",
        "description": "True en keyframes reales; False en slots de relleno cuando L es adaptativo.",
    },
    {
        "stage": "Live Query Input",
        "name": "live_scene_pos",
        "shape": "(1, 2048, 3)",
        "dtype": "float32",
        "frame": "Horizontal FLU (egocentrico, roll=pitch=0)",
        "scale": "s = 0.1 (1 unidad = 10 m)",
        "description": "Nube LiDAR 360° actual observada por el dron en el instante t.",
    },
    {
        "stage": "Live Query Input",
        "name": "live_pose_se3",
        "shape": "(1, 4, 4)",
        "dtype": "float32",
        "frame": "Gazebo ENU estabilizado",
        "scale": "Traslacion s = 0.1; Rotacion R_z(yaw_t)",
        "description": "Pose actual T_{WE}^t (para proyectar la salida al mundo; usada en MiniInstantPolicyNet solo como atajo exploratorio, a revisar en el Hito 4).",
    },
    {
        "stage": "Live Query Input",
        "name": "live_vel_flu",
        "shape": "(1, 3)",
        "dtype": "float32",
        "frame": "Horizontal FLU",
        "scale": "m/s (fisico directo)",
        "description": "Velocidad lineal actual del dron en su propio marco Horizontal FLU.",
    },
    {
        "stage": "Live Query Input",
        "name": "live_flight_state",
        "shape": "(1, 1)",
        "dtype": "float32",
        "frame": "Escalar binario",
        "scale": "{0.0, 1.0}",
        "description": "Estado binario actual s_t antes de predecir el horizonte futuro.",
    },
    {
        "stage": "Network Output",
        "name": "pred_action_se3",
        "shape": "(T, 4, 4)",
        "dtype": "float32",
        "frame": "Horizontal FLU relativo al instante t",
        "scale": "Traslacion s = 0.1; Rotacion R_z(dyaw_k)",
        "description": "Horizonte de T poses relativas acumuladas predichas T_hat_{EA}^{(t, 1..T)}.",
    },
    {
        "stage": "Network Output",
        "name": "pred_flight_state",
        "shape": "(T, 1)",
        "dtype": "float32",
        "frame": "Probabilidad escalar",
        "scale": "[0.0, 1.0]",
        "description": "Prediccion del estado de vuelo futuro s_hat_{t+1..t+T} (<0.5 activa aterrizaje).",
    },
    {
        "stage": "Controller Output",
        "name": "pred_world_waypoints_enu_m",
        "shape": "(T, 4)",
        "dtype": "float32",
        "frame": "Gazebo ENU global",
        "scale": "Metros reales (÷ s = × 10.0) y radianes [-pi, pi)",
        "description": "Waypoints globales [X_enu_m, Y_enu_m, Z_enu_m, yaw_enu_rad] para el spline.",
    },
]


def load_heterodata_episode(pt_path: Union[str, Path]) -> Any:
    """Carga un archivo `heterodata.pt` de forma compatible con PyTorch 2.x."""
    p = Path(pt_path)
    if not p.exists():
        raise FileNotFoundError(f"Archivo heterodata no encontrado: {p}")
    return torch.load(p, map_location="cpu", weights_only=False)


def parse_num_keyframes_arg(value: Union[int, str]) -> Union[int, str]:
    """Interpreta un parametro CLI/API para `num_keyframes` (entero positivo o `'adaptive'`)."""
    if isinstance(value, int):
        if value <= 0:
            return "adaptive"
        return int(value)
    text = str(value).strip().lower()
    if text in ("adaptive", "auto", "arc"):
        return "adaptive"
    parsed = int(text)
    return "adaptive" if parsed <= 0 else parsed


def select_geometric_keyframe_indices(
    pose_se3: torch.Tensor,
    flight_state: torch.Tensor,
    num_keyframes: Union[int, str] = 10,
    precomputed_indices: Optional[torch.Tensor] = None,
    adaptive_arc_m: float = 1.5,
    min_keyframes: int = 10,
    max_keyframes: int = 28,
    scale_factor: float = DEFAULT_SCALE_FACTOR,
) -> torch.Tensor:
    """
    Selecciona geometricamente los `L` indices de keyframe mas informativos de una demostracion:
      - Si `num_keyframes == 10` y `precomputed_indices` tiene 10 elementos validos, los reutiliza.
      - Si `num_keyframes == 'adaptive'` (o <= 0), calcula la longitud total de arco en metros reales
        y asigna `L = clip(ceil(longitud_arco_m / adaptive_arc_m), min_keyframes, max_keyframes)`.
      - Prioriza inicio (`0`), fin (`T-1`), transicion de aterrizaje (`s_t < 0.5`), maximos locales de
        curvatura 3D + velocidad angular de yaw y cobertura max-min por longitud de arco.
    """
    T = int(pose_se3.shape[0])
    mode = parse_num_keyframes_arg(num_keyframes)

    if T == 0:
        target_L = min_keyframes if mode == "adaptive" else int(mode)
        return torch.zeros((max(1, target_L),), dtype=torch.long)

    pos = pose_se3[:, :3, 3].detach().cpu().numpy().astype(np.float64)
    yaw_flat = np.arctan2(
        pose_se3[:, 1, 0].detach().cpu().numpy().astype(np.float64),
        pose_se3[:, 0, 0].detach().cpu().numpy().astype(np.float64),
    )
    s_flat = flight_state.detach().cpu().numpy().reshape(-1).astype(np.float32)

    seg_dists = np.linalg.norm(np.diff(pos, axis=0), axis=1) if T > 1 else np.zeros((0,), dtype=np.float64)
    arc_len = np.concatenate([[0.0], np.cumsum(seg_dists)])
    total_len_scaled = float(arc_len[-1]) if len(arc_len) > 0 and arc_len[-1] > 1e-8 else 1e-6
    total_len_m = total_len_scaled / max(1e-6, float(scale_factor))

    if mode == "adaptive":
        arc_step = max(0.25, float(adaptive_arc_m))
        target_L = int(math.ceil(total_len_m / arc_step))
        target_L = max(int(min_keyframes), min(int(max_keyframes), target_L))
    else:
        target_L = max(2, int(mode))

    if (
        mode == 10
        and precomputed_indices is not None
        and int(precomputed_indices.numel()) == 10
    ):
        return precomputed_indices.to(dtype=torch.long).clone()

    if T <= target_L:
        idx = np.round(np.linspace(0, T - 1, target_L)).astype(np.int64)
        return torch.from_numpy(idx).to(dtype=torch.long)

    curvature = np.zeros(T, dtype=np.float64)
    yaw_rate = np.zeros(T, dtype=np.float64)
    for t in range(1, T - 1):
        v1 = pos[t] - pos[t - 1]
        v2 = pos[t + 1] - pos[t]
        n1 = float(np.linalg.norm(v1))
        n2 = float(np.linalg.norm(v2))
        if n1 > 1e-6 and n2 > 1e-6:
            cos_ang = float(np.clip(np.dot(v1, v2) / (n1 * n2), -1.0, 1.0))
            curvature[t] = float(np.arccos(cos_ang)) / (0.5 * (n1 + n2) + 1e-6)
        dy = float(yaw_flat[t + 1] - yaw_flat[t - 1])
        dy_wrapped = float(np.arctan2(np.sin(dy), np.cos(dy)))
        yaw_rate[t] = abs(dy_wrapped)

    c_max = float(np.max(curvature)) if np.max(curvature) > 1e-8 else 1.0
    y_max = float(np.max(yaw_rate)) if np.max(yaw_rate) > 1e-8 else 1.0
    salience = (curvature / c_max) + 0.5 * (yaw_rate / y_max)

    selected: List[int] = [0, T - 1]
    land_transitions = np.where(s_flat < 0.5)[0]
    if len(land_transitions) > 0:
        t_land = int(land_transitions[0])
        if t_land not in selected:
            selected.append(t_land)

    min_sep = max(1, T // (2 * target_L))
    local_peaks = []
    for t in range(1, T - 1):
        if salience[t] >= salience[t - 1] and salience[t] >= salience[t + 1] and salience[t] > 0.15:
            local_peaks.append((float(salience[t]), t))
    local_peaks.sort(key=lambda x: x[0], reverse=True)

    max_peak_slots = max(2, target_L - 3)
    for _, t_peak in local_peaks:
        if len(selected) >= max_peak_slots:
            break
        if all(abs(t_peak - s_idx) >= min_sep for s_idx in selected):
            selected.append(int(t_peak))

    while len(selected) < target_L:
        best_t = -1
        best_dist = -1.0
        for candidate in range(T):
            if candidate in selected:
                continue
            min_d = min(
                abs(arc_len[candidate] - arc_len[s_idx]) / total_len_scaled
                + 0.25 * abs(candidate - s_idx) / float(T)
                for s_idx in selected
            )
            if min_d > best_dist:
                best_dist = min_d
                best_t = candidate
        if best_t < 0:
            break
        selected.append(int(best_t))

    selected_sorted = sorted(selected[:target_L])
    return torch.tensor(selected_sorted, dtype=torch.long)


def recompute_nominal_horizon_tensors(
    pose_se3: torch.Tensor,
    flight_state: torch.Tensor,
    step_indices: torch.Tensor,
    horizon: int = 8,
    action_stride: int = 1,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Calcula en memoria las matrices acumuladas `action_se3` (M_idx, H, 4, 4), el estado futuro
    `target_flight_state` (M_idx, H, 1) y la mascara `valid_mask` (M_idx, H) para cualquier
    horizonte `H = horizon` y paso `action_stride` directamente desde `pose_se3` (T_ep, 4, 4).
    """
    H = max(1, int(horizon))
    stride = max(1, int(action_stride))
    T_ep = int(pose_se3.shape[0])
    idx_base = step_indices.to(dtype=torch.long).reshape(-1)  # (M_idx,)
    k_offsets = torch.arange(1, H + 1, dtype=torch.long, device=idx_base.device) * stride  # (H,)
    raw_future_idx = idx_base[:, None] + k_offsets[None, :]  # (M_idx, H)
    valid_mask = raw_future_idx < T_ep
    clamped_idx = raw_future_idx.clamp(min=0, max=max(0, T_ep - 1))

    poses_t = pose_se3[idx_base].to(dtype=torch.float32)  # (M_idx, 4, 4)
    poses_tk = pose_se3[clamped_idx].to(dtype=torch.float32)  # (M_idx, H, 4, 4)
    poses_t_inv = invert_rigid_se3_torch(poses_t).unsqueeze(1)  # (M_idx, 1, 4, 4)
    action_se3 = poses_t_inv @ poses_tk  # (M_idx, H, 4, 4)
    target_fs = flight_state[clamped_idx].to(dtype=torch.float32).reshape(idx_base.shape[0], H, 1)
    return action_se3, target_fs, valid_mask


def recompute_aug_horizon_tensors(
    aug_pose_se3: torch.Tensor,
    nominal_timestep: int,
    nom_pose_se3: torch.Tensor,
    nom_flight_state: torch.Tensor,
    horizon: int = 8,
    action_stride: int = 1,
    tube_decay: float = 0.70,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Reconstruye en memoria el horizonte correctivo de Tube Augmentation para cualquier `horizon`
    y `action_stride` usando decaimiento exponencial `gamma^k` hacia la trayectoria nominal.
    """
    H = max(1, int(horizon))
    stride = max(1, int(action_stride))
    T_ep = int(nom_pose_se3.shape[0])
    t0 = min(max(0, int(nominal_timestep)), max(0, T_ep - 1))

    tilde_T = aug_pose_se3.reshape(4, 4).to(dtype=torch.float32)
    nom_T0 = nom_pose_se3[t0].to(dtype=torch.float32)

    dp_scaled = tilde_T[:3, 3] - nom_T0[:3, 3]  # (3,)
    tilde_yaw = torch.atan2(tilde_T[1, 0], tilde_T[0, 0])
    nom_yaw0 = torch.atan2(nom_T0[1, 0], nom_T0[0, 0])
    dyaw_pert = wrap_angle_torch(tilde_yaw - nom_yaw0)

    k_steps = torch.arange(1, H + 1, dtype=torch.long)
    raw_tk = t0 + k_steps * stride
    valid_mask = raw_tk < T_ep
    tk = raw_tk.clamp(min=0, max=max(0, T_ep - 1))

    nom_Tk = nom_pose_se3[tk].to(dtype=torch.float32)  # (H, 4, 4)
    nom_pk = nom_Tk[:, :3, 3]  # (H, 3)
    nom_yawk = torch.atan2(nom_Tk[:, 1, 0], nom_Tk[:, 0, 0])  # (H,)

    decays = (float(tube_decay) ** k_steps.to(dtype=torch.float32))  # (H,)
    target_pk = nom_pk + decays[:, None] * dp_scaled[None, :]  # (H, 3)
    target_yawk = wrap_angle_torch(nom_yawk + decays * dyaw_pert)  # (H,)

    target_world_se3 = build_se3_horizontal_flu_torch(target_pk, target_yawk)  # (H, 4, 4)
    tilde_inv = invert_rigid_se3_torch(tilde_T).unsqueeze(0)  # (1, 4, 4)
    action_se3 = tilde_inv @ target_world_se3  # (H, 4, 4)
    target_fs = nom_flight_state[tk].to(dtype=torch.float32).reshape(H, 1)
    return action_se3, target_fs, valid_mask


def extract_context_from_episode(
    data: Any,
    num_keyframes: Union[int, str] = 10,
    horizon: int = 8,
    action_stride: int = 1,
    adaptive_arc_m: float = 1.5,
    min_keyframes: int = 10,
    max_keyframes: int = 28,
) -> Dict[str, torch.Tensor]:
    """
    Extrae los tensores de contexto de un episodio `HeteroData` en sus `L` keyframes
    (fijos o adaptativos por longitud de arco) y horizonte futuro `T = horizon` con paso `action_stride`.
    """
    H = max(1, int(horizon))
    stride = max(1, int(action_stride))
    pre_kf = getattr(data["agent"], "keyframe_indices", None)
    kf_idx = select_geometric_keyframe_indices(
        pose_se3=data["agent"].pose_se3,
        flight_state=data["agent"].flight_state,
        num_keyframes=num_keyframes,
        precomputed_indices=pre_kf,
        adaptive_arc_m=adaptive_arc_m,
        min_keyframes=min_keyframes,
        max_keyframes=max_keyframes,
    )
    L = int(kf_idx.numel())

    scene_pos = data["scene"].pos[kf_idx].to(dtype=torch.float32)  # (L, 2048, 3)
    pose_se3 = data["agent"].pose_se3[kf_idx].to(dtype=torch.float32)  # (L, 4, 4)
    vel_flu = data["agent"].vel_flu[kf_idx].to(dtype=torch.float32)  # (L, 3)
    flight_state = data["agent"].flight_state[kf_idx].to(dtype=torch.float32)  # (L, 1)

    if H == 8 and stride == 1 and int(data["agent"].action_se3.shape[1]) == 8:
        action_se3 = data["agent"].action_se3[kf_idx].to(dtype=torch.float32)
        action_y = data["agent"].y[kf_idx].to(dtype=torch.float32)
        valid_mask = data["agent"].valid_mask[kf_idx].to(dtype=torch.bool)
        target_flight_state = action_y[..., 4:5]
    else:
        action_se3, target_flight_state, valid_mask = recompute_nominal_horizon_tensors(
            pose_se3=data["agent"].pose_se3,
            flight_state=data["agent"].flight_state,
            step_indices=kf_idx,
            horizon=H,
            action_stride=stride,
        )

    delta_se3 = cumulative_to_incremental_se3(action_se3)  # (L, H, 4, 4)
    delta_4dof = se3_to_4dof_vector(delta_se3)  # (L, H, 4)
    keyframe_mask = torch.ones((L,), dtype=torch.bool)

    return {
        "keyframe_indices": kf_idx,
        "keyframe_mask": keyframe_mask,
        "scene_pos": scene_pos,
        "pose_se3": pose_se3,
        "vel_flu": vel_flu,
        "flight_state": flight_state,
        "action_se3": action_se3,
        "delta_se3": delta_se3,
        "delta_4dof": delta_4dof,
        "target_flight_state": target_flight_state,
        "valid_mask": valid_mask,
    }


def _pad_context_demo_dict(d: Dict[str, torch.Tensor], target_L: int) -> Dict[str, torch.Tensor]:
    """Rellena un diccionario de contexto de longitud `L <= target_L` repitiendo el ultimo keyframe y marcando `False` en `keyframe_mask`."""
    L = int(d["scene_pos"].shape[0])
    if L == target_L:
        return d
    pad = target_L - L
    out: Dict[str, torch.Tensor] = {}
    for k, v in d.items():
        if k == "keyframe_mask":
            pad_mask = torch.zeros((pad,), dtype=torch.bool, device=v.device)
            out[k] = torch.cat([v, pad_mask], dim=0)
        else:
            last_slice = v[-1:].expand((pad,) + tuple(v.shape[1:])).clone()
            out[k] = torch.cat([v, last_slice], dim=0)
    return out


def _stack_context_dicts(ctx_dicts: Sequence[Dict[str, torch.Tensor]]) -> Dict[str, torch.Tensor]:
    """Apila `N_demos` diccionarios de contexto, aplicando padding automatico si difieren en `L`."""
    max_L = max(int(d["scene_pos"].shape[0]) for d in ctx_dicts)
    padded = [_pad_context_demo_dict(d, max_L) for d in ctx_dicts]
    return {
        "context_keyframe_indices": torch.stack([d["keyframe_indices"] for d in padded], dim=0),
        "context_keyframe_mask": torch.stack([d["keyframe_mask"] for d in padded], dim=0),
        "context_scene_pos": torch.stack([d["scene_pos"] for d in padded], dim=0),
        "context_pose_se3": torch.stack([d["pose_se3"] for d in padded], dim=0),
        "context_vel_flu": torch.stack([d["vel_flu"] for d in padded], dim=0),
        "context_flight_state": torch.stack([d["flight_state"] for d in padded], dim=0),
        "context_action_se3": torch.stack([d["action_se3"] for d in padded], dim=0),
        "context_delta_se3": torch.stack([d["delta_se3"] for d in padded], dim=0),
        "context_delta_4dof": torch.stack([d["delta_4dof"] for d in padded], dim=0),
        "context_target_flight_state": torch.stack([d["target_flight_state"] for d in padded], dim=0),
        "context_valid_mask": torch.stack([d["valid_mask"] for d in padded], dim=0),
    }


def build_context_pack_from_episodes(
    episode_items: Sequence[Union[str, Path, Any]],
    num_keyframes: Union[int, str] = 10,
    horizon: int = 8,
    action_stride: int = 1,
    adaptive_arc_m: float = 1.5,
    min_keyframes: int = 10,
    max_keyframes: int = 28,
) -> Dict[str, Any]:
    """
    Construye el paquete de contexto apilando `N_demos` episodios en sus `L = num_keyframes`
    keyframes (fijos o `'adaptive'`) y horizonte `T = horizon` con paso `action_stride`.
    """
    if not episode_items:
        raise ValueError("build_context_pack_from_episodes requiere al menos 1 episodio de contexto.")

    per_demo = []
    for item in episode_items:
        ep_data = load_heterodata_episode(item) if isinstance(item, (str, Path)) else item
        per_demo.append(
            extract_context_from_episode(
                ep_data,
                num_keyframes=num_keyframes,
                horizon=horizon,
                action_stride=action_stride,
                adaptive_arc_m=adaptive_arc_m,
                min_keyframes=min_keyframes,
                max_keyframes=max_keyframes,
            )
        )

    pack: Dict[str, Any] = _stack_context_dicts(per_demo)
    pack["action_stride"] = int(max(1, action_stride))
    return pack


def collate_in_context_batch(samples: Sequence[Dict[str, Any]]) -> Dict[str, Any]:
    """
    `collate_fn` compatible con PyTorch `DataLoader` tanto para `L` fijo como para `L` adaptativo
    (donde muestras de distintas tareas dentro del mismo batch pueden tener distinto numero de keyframes).
    """
    if not samples:
        raise ValueError("collate_in_context_batch recibio una lista vacia.")

    max_L = max(int(s["context_scene_pos"].shape[1]) for s in samples)
    ctx_tensor_keys = (
        "context_keyframe_indices",
        "context_keyframe_mask",
        "context_scene_pos",
        "context_pose_se3",
        "context_vel_flu",
        "context_flight_state",
        "context_action_se3",
        "context_delta_se3",
        "context_delta_4dof",
        "context_target_flight_state",
        "context_valid_mask",
    )

    batch: Dict[str, Any] = {}
    for key in samples[0].keys():
        vals = [s[key] for s in samples]
        if key in ctx_tensor_keys:
            padded_vals = []
            for v in vals:
                L_cur = int(v.shape[1])
                if L_cur == max_L:
                    padded_vals.append(v)
                else:
                    pad = max_L - L_cur
                    if key == "context_keyframe_mask":
                        z = torch.zeros((v.shape[0], pad), dtype=torch.bool, device=v.device)
                        padded_vals.append(torch.cat([v, z], dim=1))
                    else:
                        tail = v[:, -1:].expand((v.shape[0], pad) + tuple(v.shape[2:])).clone()
                        padded_vals.append(torch.cat([v, tail], dim=1))
            batch[key] = torch.stack(padded_vals, dim=0)
        elif isinstance(vals[0], torch.Tensor):
            batch[key] = torch.stack(vals, dim=0)
        elif isinstance(vals[0], (int, float)):
            batch[key] = torch.tensor(vals)
        else:
            batch[key] = vals
    return batch


def verify_sample_contract(
    sample: Dict[str, Any],
    expected_num_demos: Optional[int] = None,
    expected_num_keyframes: Optional[Union[int, str]] = 10,
    expected_num_points: int = 2048,
    expected_horizon: int = 8,
) -> Dict[str, Any]:
    """
    Verifica exhaustivamente las formas, tipos, finitud y ortogonalidad SE(3) (roll=0, pitch=0)
    de un diccionario `sample` de entrada/salida. Lanza `AssertionError` si se viola algun contrato.
    """
    ctx_scene = sample["context_scene_pos"]
    assert isinstance(ctx_scene, torch.Tensor) and ctx_scene.dtype == torch.float32
    assert ctx_scene.ndim == 4, f"Expected 4D context_scene_pos, got {ctx_scene.shape}"
    N_demos, L, P, three = ctx_scene.shape
    if expected_num_demos is not None:
        assert N_demos == expected_num_demos, f"Expected N_demos={expected_num_demos}, got {N_demos}"
    if expected_num_keyframes is not None and parse_num_keyframes_arg(expected_num_keyframes) != "adaptive":
        exp_L = int(parse_num_keyframes_arg(expected_num_keyframes))
        assert L == exp_L, f"Expected L={exp_L}, got {L}"
    else:
        assert L >= 2, f"Expected adaptive L >= 2, got {L}"
    assert P == expected_num_points and three == 3

    assert sample["context_pose_se3"].shape == (N_demos, L, 4, 4)
    assert sample["context_vel_flu"].shape == (N_demos, L, 3)
    assert sample["context_flight_state"].shape == (N_demos, L, 1)
    assert sample["context_action_se3"].shape == (N_demos, L, expected_horizon, 4, 4)
    assert sample["context_delta_se3"].shape == (N_demos, L, expected_horizon, 4, 4)
    assert sample["context_delta_4dof"].shape == (N_demos, L, expected_horizon, 4)
    assert sample["context_target_flight_state"].shape == (N_demos, L, expected_horizon, 1)
    assert sample["context_valid_mask"].shape == (N_demos, L, expected_horizon)
    if "context_keyframe_mask" in sample:
        assert sample["context_keyframe_mask"].shape == (N_demos, L)
        assert sample["context_keyframe_mask"].dtype == torch.bool

    assert sample["live_scene_pos"].shape == (1, expected_num_points, 3)
    assert sample["live_pose_se3"].shape == (1, 4, 4)
    assert sample["live_vel_flu"].shape == (1, 3)
    assert sample["live_flight_state"].shape == (1, 1)
    assert sample["keypoints_body"].shape == (6, 3)

    if "target_action_se3" in sample:
        assert sample["target_action_se3"].shape == (expected_horizon, 4, 4)
        assert sample["target_delta_se3"].shape == (expected_horizon, 4, 4)
        assert sample["target_delta_4dof"].shape == (expected_horizon, 4)
        assert sample["target_flight_state"].shape == (expected_horizon, 1)
        assert sample["target_valid_mask"].shape == (expected_horizon,)

    # Verificar que no existan NaNs ni infinitos en ningun tensor numerico
    for k, v in sample.items():
        if isinstance(v, torch.Tensor) and (v.is_floating_point() or v.dtype == torch.bool):
            assert torch.all(torch.isfinite(v)), f"Non-finite values found in tensor '{k}'"

    # Verificar estabilizacion por gravedad (roll=0, pitch=0 -> R[2, 2] == 1, R[2, :2] == 0)
    R_live = sample["live_pose_se3"][0, :3, :3]
    det_live = float(torch.linalg.det(R_live).item())
    assert abs(det_live - 1.0) < 1e-4, f"Invalid rotation determinant: {det_live}"
    assert float(torch.abs(R_live[2, 0]).item()) < 1e-5
    assert float(torch.abs(R_live[2, 1]).item()) < 1e-5
    assert abs(float(R_live[2, 2].item()) - 1.0) < 1e-5

    return {
        "valid": True,
        "num_demos": N_demos,
        "num_keyframes": L,
        "num_points": P,
        "horizon": expected_horizon,
    }


class InContextNavigationDataset(Dataset):
    """
    Dataset de PyTorch para entrenamiento y validacion de UAV Instant Policy.

    Soporta parametrizacion completa sin regenerar archivos `heterodata.pt`:
      - `num_context_demos`: N_demos >= 1
      - `num_keyframes`: entero fijo (ej. 10, 16, 24) o `'adaptive'` (proporcional a la longitud de arco)
      - `adaptive_arc_m`: metros por keyframe en modo `'adaptive'` (default: 1.5 m)
      - `horizon`: pasos futuros T del horizonte de accion (ej. 6, 8, 12, 16)
      - `action_stride`: separacion en ticks de 10 Hz entre pasos del horizonte (1 = 0.1 s, 2 = 0.2 s)
    """

    def __init__(
        self,
        dataset_dir: Union[str, Path],
        split: Optional[str] = None,
        task_ids: Optional[Sequence[str]] = None,
        num_context_demos: int = 2,
        num_keyframes: Union[int, str] = 10,
        horizon: int = 8,
        action_stride: int = 1,
        adaptive_arc_m: float = 1.5,
        min_keyframes: int = 10,
        max_keyframes: int = 28,
        tube_decay: float = 0.70,
        include_aug: bool = True,
        cache_episodes_in_ram: bool = True,
        max_tasks: Optional[int] = None,
    ) -> None:
        super().__init__()
        self.dataset_dir = Path(dataset_dir)
        self.split = split
        self.num_context_demos = max(1, int(num_context_demos))
        self.num_keyframes = parse_num_keyframes_arg(num_keyframes)
        self.horizon = max(1, int(horizon))
        self.action_stride = max(1, int(action_stride))
        self.adaptive_arc_m = float(adaptive_arc_m)
        self.min_keyframes = int(min_keyframes)
        self.max_keyframes = int(max_keyframes)
        self.tube_decay = float(tube_decay)
        self.include_aug = bool(include_aug)
        self.cache_episodes_in_ram = bool(cache_episodes_in_ram)

        selected_tasks = self._resolve_task_list(task_ids=task_ids, split=split, max_tasks=max_tasks)
        self.task_episodes: Dict[str, List[Path]] = {}
        self._episode_cache: Dict[Path, Any] = {}
        self._context_cache: Dict[Path, Dict[str, torch.Tensor]] = {}
        self.samples_index: List[Tuple[str, int, str, int]] = []
        # Cada tupla en samples_index es: (task_id, ep_idx_in_task, source_type ['nominal'|'aug'], local_step_idx)

        for tid in selected_tasks:
            t_dir = self.dataset_dir / tid
            if not t_dir.is_dir():
                continue
            ep_paths = sorted(t_dir.glob("episode_*/heterodata.pt"))
            if not ep_paths:
                continue
            self.task_episodes[tid] = ep_paths

            for ep_i, pt_path in enumerate(ep_paths):
                ep_data = self._get_episode_data(pt_path)
                T_nom = int(ep_data["agent"].pos_enu.shape[0])
                for t in range(T_nom):
                    self.samples_index.append((tid, ep_i, "nominal", t))

                if self.include_aug and "aug" in ep_data.node_types:
                    T_aug = int(ep_data["aug"].pos_enu.shape[0])
                    for j in range(T_aug):
                        self.samples_index.append((tid, ep_i, "aug", j))

    def _resolve_task_list(
        self,
        task_ids: Optional[Sequence[str]],
        split: Optional[str],
        max_tasks: Optional[int],
    ) -> List[str]:
        if task_ids is not None:
            tasks = list(task_ids)
        elif split is not None:
            split_map = {
                "train": "train_tasks.txt",
                "val": "val_tasks.txt",
                "test": "test_tasks.txt",
                "held_out": "test_tasks.txt",
            }
            fname = split_map.get(split, f"{split}_tasks.txt")
            split_file = self.dataset_dir / "splits" / fname
            if split_file.is_file():
                raise FileNotFoundError(
                    f"Archivo de split '{split}' no encontrado en: {split_file}"
                )
            split_data = json.loads(split_file.read_text(encoding="utf-8"))
            tasks = [str(t).strip() for t in split_data.get("tasks", []) if str(t).strip()]
        else:
            tasks = sorted(p.name for p in self.dataset_dir.glob("task_*") if p.is_dir())

        if max_tasks is not None:
            tasks = tasks[: int(max_tasks)]
        return tasks

    def _get_episode_data(self, pt_path: Path) -> Any:
        if self.cache_episodes_in_ram:
            if pt_path not in self._episode_cache:
                self._episode_cache[pt_path] = load_heterodata_episode(pt_path)
            return self._episode_cache[pt_path]
        return load_heterodata_episode(pt_path)

    def _get_episode_context(self, pt_path: Path) -> Dict[str, torch.Tensor]:
        if self.cache_episodes_in_ram:
            if pt_path not in self._context_cache:
                ep_data = self._get_episode_data(pt_path)
                self._context_cache[pt_path] = extract_context_from_episode(
                    ep_data,
                    num_keyframes=self.num_keyframes,
                    horizon=self.horizon,
                    action_stride=self.action_stride,
                    adaptive_arc_m=self.adaptive_arc_m,
                    min_keyframes=self.min_keyframes,
                    max_keyframes=self.max_keyframes,
                )
            return self._context_cache[pt_path]
        ep_data = self._get_episode_data(pt_path)
        return extract_context_from_episode(
            ep_data,
            num_keyframes=self.num_keyframes,
            horizon=self.horizon,
            action_stride=self.action_stride,
            adaptive_arc_m=self.adaptive_arc_m,
            min_keyframes=self.min_keyframes,
            max_keyframes=self.max_keyframes,
        )

    def __len__(self) -> int:
        return len(self.samples_index)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        task_id, ep_idx, source_type, step_idx = self.samples_index[idx]
        ep_paths = self.task_episodes[task_id]
        query_pt_path = ep_paths[ep_idx]
        ep_data = self._get_episode_data(query_pt_path)

        # Seleccionar episodios de contexto de la misma tarea (excluyendo query_pt_path si hay mas de 1)
        other_paths = [p for i, p in enumerate(ep_paths) if i != ep_idx]
        if not other_paths:
            other_paths = [query_pt_path]

        chosen_ctx_paths: List[Path] = []
        for k in range(self.num_context_demos):
            chosen_ctx_paths.append(other_paths[(ep_idx + k) % len(other_paths)])

        ctx_dicts = [self._get_episode_context(p) for p in chosen_ctx_paths]
        ctx_pack = _stack_context_dicts(ctx_dicts)

        keypoints_body = (
            ep_data["agent"].keypoints_body.to(dtype=torch.float32)
            if hasattr(ep_data["agent"], "keypoints_body")
            else get_uav_keypoints_body_torch(scale_factor=DEFAULT_SCALE_FACTOR)
        )

        use_stored_horizon = (
            self.horizon == 8
            and self.action_stride == 1
            and int(ep_data["agent"].action_se3.shape[1]) == 8
        )

        if source_type == "nominal":
            live_scene_pos = ep_data["scene"].pos[step_idx : step_idx + 1].to(dtype=torch.float32)
            live_pose_se3 = ep_data["agent"].pose_se3[step_idx : step_idx + 1].to(dtype=torch.float32)
            live_vel_flu = ep_data["agent"].vel_flu[step_idx : step_idx + 1].to(dtype=torch.float32)
            live_flight_state = ep_data["agent"].flight_state[step_idx : step_idx + 1].to(dtype=torch.float32)
            nominal_timestep = int(step_idx)
            if use_stored_horizon:
                target_action_se3 = ep_data["agent"].action_se3[step_idx].to(dtype=torch.float32)
                target_y = ep_data["agent"].y[step_idx].to(dtype=torch.float32)
                target_valid_mask = ep_data["agent"].valid_mask[step_idx].to(dtype=torch.bool)
                target_flight_state = target_y[:, 4:5]
            else:
                act_seq, fs_seq, vmask_seq = recompute_nominal_horizon_tensors(
                    pose_se3=ep_data["agent"].pose_se3,
                    flight_state=ep_data["agent"].flight_state,
                    step_indices=torch.tensor([step_idx], dtype=torch.long),
                    horizon=self.horizon,
                    action_stride=self.action_stride,
                )
                target_action_se3 = act_seq[0]
                target_flight_state = fs_seq[0]
                target_valid_mask = vmask_seq[0]
        else:
            live_scene_pos = ep_data["aug"].scene_pos[step_idx : step_idx + 1].to(dtype=torch.float32)
            live_pose_se3 = ep_data["aug"].pose_se3[step_idx : step_idx + 1].to(dtype=torch.float32)
            live_vel_flu = ep_data["aug"].vel_flu[step_idx : step_idx + 1].to(dtype=torch.float32)
            live_flight_state = ep_data["aug"].flight_state_in[step_idx : step_idx + 1].to(dtype=torch.float32)
            nominal_timestep = int(ep_data["aug"].timestep_idx[step_idx].item())
            if use_stored_horizon and int(ep_data["aug"].action_se3.shape[1]) == 8:
                target_action_se3 = ep_data["aug"].action_se3[step_idx].to(dtype=torch.float32)
                target_y = ep_data["aug"].y[step_idx].to(dtype=torch.float32)
                target_valid_mask = ep_data["aug"].valid_mask[step_idx].to(dtype=torch.bool)
                target_flight_state = target_y[:, 4:5]
            else:
                target_action_se3, target_flight_state, target_valid_mask = recompute_aug_horizon_tensors(
                    aug_pose_se3=live_pose_se3[0],
                    nominal_timestep=nominal_timestep,
                    nom_pose_se3=ep_data["agent"].pose_se3,
                    nom_flight_state=ep_data["agent"].flight_state,
                    horizon=self.horizon,
                    action_stride=self.action_stride,
                    tube_decay=self.tube_decay,
                )

        target_delta_se3 = cumulative_to_incremental_se3(target_action_se3)
        target_delta_4dof = se3_to_4dof_vector(target_delta_se3)

        return {
            "task_id": task_id,
            "episode_id": query_pt_path.parent.name,
            "source_type": source_type,
            "step_idx": int(step_idx),
            "nominal_timestep": nominal_timestep,
            "action_stride": self.action_stride,
            **ctx_pack,
            "live_scene_pos": live_scene_pos,
            "live_pose_se3": live_pose_se3,
            "live_vel_flu": live_vel_flu,
            "live_flight_state": live_flight_state,
            "keypoints_body": keypoints_body,
            "target_action_se3": target_action_se3,
            "target_delta_se3": target_delta_se3,
            "target_delta_4dof": target_delta_4dof,
            "target_flight_state": target_flight_state,
            "target_valid_mask": target_valid_mask,
        }
