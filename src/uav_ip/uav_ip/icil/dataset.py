"""Dataset In-Context, splits JSON, keyframes y horizonte incremental SE(3) 4-DoF."""

from __future__ import annotations

from collections import OrderedDict
import json
import math
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple, Union

import torch
from torch import Tensor
from torch.utils.data import Dataset

from uav_ip.icil.geometry import (
    cumulative_to_incremental,
    invert_se3,
    se3_to_vec4,
    vec4_to_se3,
    wrap_angle,
)

SPLIT_MAP: Dict[str, str] = {
    "train": "train",
    "val": "val",
    "held_out": "held_out",
    "test": "held_out",
}


def load_split_tasks(dataset_root: Union[Path, str], split: str) -> List[str]:
    """Carga estrictamente la lista de tareas desde splits/{train,val,held_out}_tasks.json."""
    if split not in SPLIT_MAP:
        raise ValueError(f"Split desconocido '{split}'. Opciones validas: {list(SPLIT_MAP.keys())}")

    canonical_split = SPLIT_MAP[split]
    split_file = Path(dataset_root) / "splits" / f"{canonical_split}_tasks.json"
    if not split_file.is_file():
        raise FileNotFoundError(f"No existe el archivo de split oficial: {split_file}")

    with open(split_file, "r", encoding="utf-8") as f:
        payload = json.load(f)

    tasks = payload.get("tasks", [])
    if not isinstance(tasks, list) or len(tasks) == 0:
        raise ValueError(f"El archivo de split {split_file} no contiene una lista valida en 'tasks'.")
    return [str(t) for t in tasks]


def select_keyframe_indices(
    pos_enu: Tensor,
    yaw_enu: Tensor,
    flight_state: Tensor,
    num_keyframes: Union[int, str] = 10,
    max_keyframes: int = 16,
    target_arc_step_m: float = 2.5,
) -> Tuple[Tensor, Tensor]:
    """
    Selecciona indices de keyframes ordenados temporalmente y su mascara booleana:
      - Si num_keyframes es int (L): retorna (indices (L,), mask (L,) todo True).
      - Si num_keyframes == 'adaptive': retorna (indices (max_keyframes,), mask (max_keyframes,))
        con True en los K <= max_keyframes frames seleccionados y padding repitiendo el ultimo indice.
    """
    T = int(pos_enu.shape[0])
    is_adaptive = isinstance(num_keyframes, str)
    if is_adaptive and num_keyframes != "adaptive":
        raise ValueError(f"num_keyframes debe ser un entero >= 2 o 'adaptive', recibido: {num_keyframes}")

    L_pad = int(max_keyframes) if is_adaptive else int(num_keyframes)
    if L_pad < 2:
        raise ValueError(f"El numero de keyframes debe ser >= 2, recibido: {L_pad}")

    if T == 0:
        return torch.zeros(L_pad, dtype=torch.long), torch.zeros(L_pad, dtype=torch.bool)

    pos = pos_enu.detach().to(dtype=torch.float64, device="cpu")
    yaw = yaw_enu.detach().to(dtype=torch.float64, device="cpu").reshape(-1)
    s_flat = flight_state.detach().to(dtype=torch.float32, device="cpu").reshape(-1)

    if T <= 2:
        idx = torch.linspace(0, T - 1, steps=L_pad).round().to(torch.long)
        mask = torch.ones(L_pad, dtype=torch.bool)
        if is_adaptive:
            mask[T:] = False
        return idx, mask

    seg_dists = torch.linalg.norm(pos[1:] - pos[:-1], dim=-1)
    arc_len = torch.cat([torch.zeros(1, dtype=torch.float64), torch.cumsum(seg_dists, dim=0)], dim=0)
    total_len = float(arc_len[-1].item()) if float(arc_len[-1].item()) > 1e-8 else 1.0
    mean_step = max(total_len / max(1, T - 1), 1e-12)

    curvature = torch.zeros(T, dtype=torch.float64)
    yaw_rate = torch.zeros(T, dtype=torch.float64)
    v1 = pos[1:-1] - pos[:-2]
    v2 = pos[2:] - pos[1:-1]
    n1 = torch.linalg.norm(v1, dim=-1)
    n2 = torch.linalg.norm(v2, dim=-1)
    valid_seg = (n1 > 1e-6 * mean_step) & (n2 > 1e-6 * mean_step)
    if valid_seg.any():
        cos_ang = torch.clamp((v1 * v2).sum(dim=-1) / (n1 * n2 + 1e-12), -1.0, 1.0)
        curv_inner = torch.acos(cos_ang) / (0.5 * (n1 + n2) + 1e-8 * mean_step)
        curvature[1:-1] = torch.where(valid_seg, curv_inner, torch.zeros_like(curv_inner))
    yaw_rate[1:-1] = wrap_angle(yaw[2:] - yaw[:-2]).abs()

    c_max = max(float(curvature.max().item()), 1e-8)
    y_max = max(float(yaw_rate.max().item()), 1e-8)
    salience = (curvature / c_max) + 0.5 * (yaw_rate / y_max)

    if is_adaptive:
        arc_slots = int(math.ceil(total_len / max(0.5, float(target_arc_step_m)))) + 2
        target_k = min(L_pad, max(4, min(T, arc_slots)))
    else:
        target_k = min(L_pad, T)

    selected: List[int] = [0, T - 1]
    land_idx = torch.nonzero(s_flat < 0.5, as_tuple=False).reshape(-1)
    if land_idx.numel() > 0:
        t_land = int(land_idx[0].item())
        if t_land not in selected:
            selected.append(t_land)

    min_sep = max(1, T // (2 * max(2, target_k)))
    local_peaks: List[Tuple[float, int]] = []
    for t in range(1, T - 1):
        sal_t = float(salience[t].item())
        if sal_t >= float(salience[t - 1].item()) and sal_t >= float(salience[t + 1].item()) and sal_t > 0.15:
            local_peaks.append((sal_t, t))
    local_peaks.sort(key=lambda item: item[0], reverse=True)

    max_peak_slots = max(2, target_k - 3)
    for _, t_peak in local_peaks:
        if len(selected) >= max_peak_slots:
            break
        if all(abs(t_peak - s_idx) >= min_sep for s_idx in selected):
            selected.append(int(t_peak))

    while len(selected) < target_k:
        best_t = -1
        best_dist = -1.0
        for candidate in range(T):
            if candidate in selected:
                continue
            min_d = min(
                abs(float(arc_len[candidate].item() - arc_len[s_idx].item())) / total_len
                + 0.25 * abs(candidate - s_idx) / float(T)
                for s_idx in selected
            )
            if min_d > best_dist:
                best_dist = min_d
                best_t = candidate
        if best_t < 0:
            break
        selected.append(int(best_t))

    selected_sorted = sorted(selected[:target_k])
    k_valid = len(selected_sorted)
    if k_valid < L_pad:
        padded = selected_sorted + [selected_sorted[-1]] * (L_pad - k_valid)
    else:
        padded = selected_sorted[:L_pad]

    indices = torch.tensor(padded, dtype=torch.long)
    mask = torch.ones(L_pad, dtype=torch.bool)
    if is_adaptive and k_valid < L_pad:
        mask[k_valid:] = False
    return indices, mask


def extract_nominal_horizon(
    ep: Dict[str, Any],
    t: int,
    horizon: int = 8,
    action_stride: int = 1,
) -> Tuple[Tensor, Tensor, Tensor]:
    """Extrae (gt_delta_se3, gt_flight_state, action_valid_mask) para el instante nominal t."""
    T_ep = int(ep["T_ep"])
    ks_raw = t + torch.arange(1, horizon + 1, dtype=torch.long) * max(1, int(action_stride))
    valid_mask = ks_raw < T_ep
    ks = torch.clamp(ks_raw, max=T_ep - 1)
    gt_flight_state = ep["flight_state"][ks]  # (T, 1)

    if int(action_stride) == 1 and horizon <= int(ep["action_se3"].shape[1]):
        cum_se3 = ep["action_se3"][t, :horizon]
    else:
        T_we_t_inv = invert_se3(ep["pose_se3"][t]).unsqueeze(0)  # (1, 4, 4)
        T_we_ks = ep["pose_se3"][ks]                             # (T, 4, 4)
        cum_se3 = torch.matmul(T_we_t_inv, T_we_ks)

    gt_delta_se3 = cumulative_to_incremental(cum_se3)
    return gt_delta_se3, gt_flight_state, valid_mask


def extract_aug_horizon(
    ep: Dict[str, Any],
    aug_i: int,
    horizon: int = 8,
    action_stride: int = 1,
    tube_decay: float = 0.70,
) -> Tuple[Tensor, Tensor, Tensor]:
    """Extrae (gt_delta_se3, gt_flight_state, action_valid_mask) para la muestra Tube Augmentation aug_i."""
    T_ep = int(ep["T_ep"])
    t = int(ep["aug_timestep_idx"][aug_i].item())
    ks_raw = t + torch.arange(1, horizon + 1, dtype=torch.long) * max(1, int(action_stride))
    valid_mask = ks_raw < T_ep
    ks = torch.clamp(ks_raw, max=T_ep - 1)
    gt_flight_state = ep["flight_state"][ks]  # (T, 1)

    if int(action_stride) == 1 and horizon <= int(ep["aug_action_se3"].shape[1]):
        cum_se3 = ep["aug_action_se3"][aug_i, :horizon]
    else:
        tilde_pose = ep["aug_pose_se3"][aug_i]
        tilde_pos = ep["aug_pos_enu"][aug_i]
        dp = tilde_pos - ep["pos_enu"][t]
        dyaw = ep["aug_delta_yaw"][aug_i, 0]
        gamma = max(0.0, min(1.0, float(tube_decay)))
        k_steps = torch.arange(1, horizon + 1, dtype=torch.float32)
        decays = (gamma ** k_steps) if gamma > 0.0 else torch.zeros_like(k_steps)

        target_pos = ep["pos_enu"][ks] + decays.unsqueeze(-1) * dp.unsqueeze(0)
        target_yaw = wrap_angle(ep["yaw_enu"][ks, 0] + decays * dyaw)
        target_vec4 = torch.cat([target_pos, target_yaw.unsqueeze(-1)], dim=-1)
        T_we_target = vec4_to_se3(target_vec4)
        cum_se3 = torch.matmul(invert_se3(tilde_pose).unsqueeze(0), T_we_target)

    gt_delta_se3 = cumulative_to_incremental(cum_se3)
    return gt_delta_se3, gt_flight_state, valid_mask


class InContextDataset(Dataset):
    """Dataset In-Context para Instant Policy UAV con keyframes de contexto y Tube Augmentation."""

    def __init__(
        self,
        dataset_root: Union[Path, str],
        split: str = "train",
        num_demos: int = 2,
        num_keyframes: Union[int, str] = 10,
        max_keyframes: int = 16,
        horizon: int = 8,
        action_stride: int = 1,
        include_aug: bool = True,
        tube_decay: float = 0.70,
        deterministic_demos: Optional[bool] = None,
        max_tasks: Optional[int] = None,
        max_cached_episodes: int = 200,
    ) -> None:
        super().__init__()
        self.dataset_root = Path(dataset_root)
        self.split = split
        self.num_demos = int(num_demos)
        self.num_keyframes = num_keyframes
        self.max_keyframes = int(max_keyframes)
        self.horizon = int(horizon)
        self.action_stride = int(action_stride)
        self.include_aug = bool(include_aug)
        self.tube_decay = float(tube_decay)
        self.deterministic_demos = (split != "train") if deterministic_demos is None else bool(deterministic_demos)
        self.max_cached_episodes = max(8, int(max_cached_episodes))

        all_tasks = load_split_tasks(self.dataset_root, split)
        self.task_ids = all_tasks[: int(max_tasks)] if max_tasks is not None else all_tasks

        self.episodes: List[Dict[str, Any]] = []
        self.task_to_episodes: Dict[str, List[int]] = {}
        self.samples: List[Tuple[str, int, bool, int]] = []
        self._cloud_cache: OrderedDict[int, Dict[str, Tensor]] = OrderedDict()

        for task_id in self.task_ids:
            task_dir = self.dataset_root / task_id
            if not task_dir.is_dir():
                continue
            ep_dirs = sorted(p for p in task_dir.iterdir() if p.is_dir() and p.name.startswith("episode_"))
            task_ep_indices: List[int] = []

            for ep_dir in ep_dirs:
                pt_path = ep_dir / "heterodata.pt"
                if not pt_path.is_file():
                    continue
                meta_path = ep_dir / "episode_meta.json"
                if meta_path.is_file():
                    with open(meta_path, "r", encoding="utf-8") as mf:
                        meta = json.load(mf)
                    if meta.get("validation_status", "accepted") != "accepted":
                        continue

                ep_idx = len(self.episodes)
                ep_record, clouds_bundle = self._load_episode_record(pt_path, task_id, ep_dir.name)
                self.episodes.append(ep_record)
                self._put_cloud_cache(ep_idx, clouds_bundle)
                task_ep_indices.append(ep_idx)

                for t in range(ep_record["T_ep"]):
                    self.samples.append((task_id, ep_idx, False, t))
                if self.include_aug:
                    for aug_i in range(ep_record["T_aug"]):
                        self.samples.append((task_id, ep_idx, True, aug_i))

            if task_ep_indices:
                self.task_to_episodes[task_id] = task_ep_indices

        if len(self.samples) == 0:
            raise RuntimeError(f"No se encontraron muestras validas en {self.dataset_root} para split='{split}'.")

    def _load_episode_record(
        self,
        pt_path: Path,
        task_id: str,
        episode_id: str,
    ) -> Tuple[Dict[str, Any], Dict[str, Tensor]]:
        data = torch.load(pt_path, map_location="cpu", weights_only=False)
        agent = data["agent"]
        scene_pos = data["scene"].pos.to(torch.float32)
        pos_enu = agent.pos_enu.to(torch.float32)
        yaw_enu = agent.yaw_enu.to(torch.float32)
        pose_se3 = agent.pose_se3.to(torch.float32)
        vel_flu = agent.vel_flu.to(torch.float32)
        flight_state = agent.flight_state.to(torch.float32)
        action_se3 = agent.action_se3.to(torch.float32)

        if (
            isinstance(self.num_keyframes, int)
            and hasattr(agent, "keyframe_indices")
            and int(agent.keyframe_indices.numel()) == int(self.num_keyframes)
        ):
            kf_idx = agent.keyframe_indices.to(torch.long)
            kf_mask = torch.ones(int(self.num_keyframes), dtype=torch.bool)
        else:
            kf_idx, kf_mask = select_keyframe_indices(
                pos_enu=pos_enu,
                yaw_enu=yaw_enu,
                flight_state=flight_state,
                num_keyframes=self.num_keyframes,
                max_keyframes=self.max_keyframes,
            )

        has_aug = "aug" in data.node_types and int(data["aug"].timestep_idx.numel()) > 0
        if has_aug:
            aug = data["aug"]
            aug_timestep_idx = aug.timestep_idx.to(torch.long)
            aug_scene_pos = aug.scene_pos.to(torch.float32)
            aug_pos_enu = aug.pos_enu.to(torch.float32)
            aug_pose_se3 = aug.pose_se3.to(torch.float32)
            aug_vel_flu = aug.vel_flu.to(torch.float32)
            aug_delta_yaw = aug.delta_yaw.to(torch.float32)
            aug_flight_state_in = aug.flight_state_in.to(torch.float32)
            aug_action_se3 = aug.action_se3.to(torch.float32)
        else:
            aug_timestep_idx = torch.zeros((0,), dtype=torch.long)
            aug_scene_pos = torch.zeros((0, scene_pos.shape[1], 3), dtype=torch.float32)
            aug_pos_enu = torch.zeros((0, 3), dtype=torch.float32)
            aug_pose_se3 = torch.zeros((0, 4, 4), dtype=torch.float32)
            aug_vel_flu = torch.zeros((0, 3), dtype=torch.float32)
            aug_delta_yaw = torch.zeros((0, 1), dtype=torch.float32)
            aug_flight_state_in = torch.zeros((0, 1), dtype=torch.float32)
            aug_action_se3 = torch.zeros((0, action_se3.shape[1], 4, 4), dtype=torch.float32)

        ep_record: Dict[str, Any] = {
            "pt_path": pt_path,
            "task_id": task_id,
            "episode_id": episode_id,
            "T_ep": int(pos_enu.shape[0]),
            "T_aug": int(aug_timestep_idx.numel()),
            "pos_enu": pos_enu,
            "yaw_enu": yaw_enu,
            "pose_se3": pose_se3,
            "vel_flu": vel_flu,
            "flight_state": flight_state,
            "action_se3": action_se3,
            "demo_cloud": scene_pos[kf_idx].clone(),
            "demo_pose_se3": pose_se3[kf_idx].clone(),
            "demo_vel_flu": vel_flu[kf_idx].clone(),
            "demo_flight_state": flight_state[kf_idx].clone(),
            "demo_mask": kf_mask.clone(),
            "aug_timestep_idx": aug_timestep_idx,
            "aug_pos_enu": aug_pos_enu,
            "aug_pose_se3": aug_pose_se3,
            "aug_vel_flu": aug_vel_flu,
            "aug_delta_yaw": aug_delta_yaw,
            "aug_flight_state_in": aug_flight_state_in,
            "aug_action_se3": aug_action_se3,
        }
        clouds_bundle = {
            "scene_pos": scene_pos,
            "aug_scene_pos": aug_scene_pos,
        }
        return ep_record, clouds_bundle

    def _put_cloud_cache(self, ep_idx: int, clouds_bundle: Dict[str, Tensor]) -> None:
        self._cloud_cache[ep_idx] = clouds_bundle
        self._cloud_cache.move_to_end(ep_idx)
        while len(self._cloud_cache) > self.max_cached_episodes:
            self._cloud_cache.popitem(last=False)

    def _get_episode_clouds(self, ep_idx: int) -> Dict[str, Tensor]:
        if ep_idx in self._cloud_cache:
            self._cloud_cache.move_to_end(ep_idx)
            return self._cloud_cache[ep_idx]

        pt_path = self.episodes[ep_idx]["pt_path"]
        data = torch.load(pt_path, map_location="cpu", weights_only=False)
        scene_pos = data["scene"].pos.to(torch.float32)
        if "aug" in data.node_types and int(data["aug"].timestep_idx.numel()) > 0:
            aug_scene_pos = data["aug"].scene_pos.to(torch.float32)
        else:
            aug_scene_pos = torch.zeros((0, scene_pos.shape[1], 3), dtype=torch.float32)
        bundle = {"scene_pos": scene_pos, "aug_scene_pos": aug_scene_pos}
        self._put_cloud_cache(ep_idx, bundle)
        return bundle

    def _select_demo_episode_indices(self, task_id: str, query_ep_idx: int, sample_idx: int) -> List[int]:
        all_eps = self.task_to_episodes[task_id]
        candidates = [idx for idx in all_eps if idx != query_ep_idx]
        if not candidates:
            candidates = list(all_eps)

        if self.deterministic_demos:
            return [candidates[(sample_idx + k) % len(candidates)] for k in range(self.num_demos)]

        if len(candidates) >= self.num_demos:
            perm = torch.randperm(len(candidates))[: self.num_demos].tolist()
            return [candidates[i] for i in perm]
        picks = torch.randint(0, len(candidates), size=(self.num_demos,)).tolist()
        return [candidates[i] for i in picks]

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> Dict[str, Tensor]:
        task_id, ep_idx, is_aug, step_i = self.samples[idx]
        ep = self.episodes[ep_idx]
        clouds = self._get_episode_clouds(ep_idx)

        if not is_aug:
            live_cloud = clouds["scene_pos"][step_i]
            live_vel_flu = ep["vel_flu"][step_i]
            live_flight_state = ep["flight_state"][step_i]
            gt_delta_se3, gt_flight_state, action_valid_mask = extract_nominal_horizon(
                ep, step_i, horizon=self.horizon, action_stride=self.action_stride
            )
        else:
            live_cloud = clouds["aug_scene_pos"][step_i]
            live_vel_flu = ep["aug_vel_flu"][step_i]
            live_flight_state = ep["aug_flight_state_in"][step_i]
            gt_delta_se3, gt_flight_state, action_valid_mask = extract_aug_horizon(
                ep,
                step_i,
                horizon=self.horizon,
                action_stride=self.action_stride,
                tube_decay=self.tube_decay,
            )

        demo_ep_indices = self._select_demo_episode_indices(task_id, ep_idx, idx)
        demo_clouds = torch.stack([self.episodes[d]["demo_cloud"] for d in demo_ep_indices], dim=0)
        demo_poses_se3 = torch.stack([self.episodes[d]["demo_pose_se3"] for d in demo_ep_indices], dim=0)
        demo_vel_flu = torch.stack([self.episodes[d]["demo_vel_flu"] for d in demo_ep_indices], dim=0)
        demo_flight_state = torch.stack([self.episodes[d]["demo_flight_state"] for d in demo_ep_indices], dim=0)
        demo_mask = torch.stack([self.episodes[d]["demo_mask"] for d in demo_ep_indices], dim=0)

        return {
            "live_cloud": live_cloud,
            "live_vel_flu": live_vel_flu,
            "live_flight_state": live_flight_state,
            "demo_clouds": demo_clouds,
            "demo_poses_se3": demo_poses_se3,
            "demo_vel_flu": demo_vel_flu,
            "demo_flight_state": demo_flight_state,
            "demo_mask": demo_mask,
            "gt_delta_se3": gt_delta_se3,
            "gt_flight_state": gt_flight_state,
            "action_valid_mask": action_valid_mask,
        }


def collate_icil_batch(batch: Sequence[Dict[str, Tensor]]) -> Dict[str, Tensor]:
    """Agrupa una lista de muestras de InContextDataset en un batch de tensores."""
    if len(batch) == 0:
        raise ValueError("collate_icil_batch recibio un batch vacio.")
    keys = batch[0].keys()
    return {k: torch.stack([sample[k] for sample in batch], dim=0) for k in keys}


def compute_action_statistics(dataset: InContextDataset) -> Dict[str, float]:
    """
    Calcula estadisticas fisicas (p99 y max) de las acciones incrementales delta_T_k
    en metros y radianes sobre todos los pasos validos del dataset.
    """
    vec4_chunks: List[Tensor] = []

    for ep in dataset.episodes:
        T_ep = int(ep["T_ep"])
        for t in range(T_ep):
            deltas, _, valid_mask = extract_nominal_horizon(
                ep, t, horizon=dataset.horizon, action_stride=dataset.action_stride
            )
            if valid_mask.any():
                vec4_chunks.append(se3_to_vec4(deltas[valid_mask]))

        if dataset.include_aug:
            for aug_i in range(int(ep["T_aug"])):
                deltas_aug, _, valid_mask_aug = extract_aug_horizon(
                    ep,
                    aug_i,
                    horizon=dataset.horizon,
                    action_stride=dataset.action_stride,
                    tube_decay=dataset.tube_decay,
                )
                if valid_mask_aug.any():
                    vec4_chunks.append(se3_to_vec4(deltas_aug[valid_mask_aug]))

    if not vec4_chunks:
        raise RuntimeError("No se encontraron pasos de accion validos para calcular estadisticas.")

    all_vec4 = torch.cat(vec4_chunks, dim=0)  # (N_valid, 4)
    trans_abs = all_vec4[:, :3].abs()
    trans_norm = torch.linalg.norm(all_vec4[:, :3], dim=-1)
    yaw_abs = all_vec4[:, 3].abs()

    def _q99(x: Tensor) -> float:
        return float(torch.quantile(x.to(torch.float64), 0.99).item())

    return {
        "num_valid_steps": float(all_vec4.shape[0]),
        "trans_norm_p99_m": _q99(trans_norm),
        "trans_norm_max_m": float(trans_norm.max().item()),
        "dx_abs_p99_m": _q99(trans_abs[:, 0]),
        "dy_abs_p99_m": _q99(trans_abs[:, 1]),
        "dz_abs_p99_m": _q99(trans_abs[:, 2]),
        "dx_abs_max_m": float(trans_abs[:, 0].max().item()),
        "dy_abs_max_m": float(trans_abs[:, 1].max().item()),
        "dz_abs_max_m": float(trans_abs[:, 2].max().item()),
        "yaw_abs_p99_rad": _q99(yaw_abs),
        "yaw_abs_max_rad": float(yaw_abs.max().item()),
    }