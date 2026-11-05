"""
in_context_dataset.py — Muestreador In-Context y Contratos de Tensores E/S para UAV Instant Policy.

Carga episodios `heterodata.pt` agrupados por tarea y construye muestras de entrenamiento
e inferencia que conectan:
  1. N_demos demostraciones de contexto muestreadas en sus L=10 `keyframe_indices`.
  2. La observacion actual (Query O_t), proveniente tanto de pasos nominales (`data['agent']`)
     como de pasos perturbados de recuperacion (`data['aug']`, Tube Augmentation 30%).
  3. Las etiquetas objetivo (tanto acumuladas `T_{EA}^{(t, 1..8)}` como incrementales `δT_{1..8}`
     y el estado de vuelo binario `s_{t+1..t+8}`).
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple, Union

import numpy as np
import torch
from torch.utils.data import Dataset
import yaml

from .se3_diffusion_math import (
    DEFAULT_SCALE_FACTOR,
    cumulative_to_incremental_se3,
    get_uav_keypoints_body_torch,
    se3_to_4dof_vector,
)

# Especificacion formal de contratos de Entrada/Salida para documentacion y validacion
TENSOR_CONTRACT_SPEC: List[Dict[str, str]] = [
    {
        "stage": "Context Input",
        "name": "context_scene_pos",
        "shape": "(N_demos, L=10, 2048, 3)",
        "dtype": "float32",
        "frame": "Horizontal FLU (egocentrico, roll=pitch=0)",
        "scale": "s = 0.1 (1 unidad = 10 m)",
        "description": "Nubes LiDAR 360° en los L=10 keyframes de cada demostracion de contexto.",
    },
    {
        "stage": "Context Input",
        "name": "context_pose_se3",
        "shape": "(N_demos, L=10, 4, 4)",
        "dtype": "float32",
        "frame": "Gazebo ENU estabilizado",
        "scale": "Traslacion s = 0.1; Rotacion R_z(yaw) in SO(2)",
        "description": "Poses T_{WE}^{t_l} de los keyframes (usadas solo para aristas relativas intra-demo).",
    },
    {
        "stage": "Context Input",
        "name": "context_vel_flu",
        "shape": "(N_demos, L=10, 3)",
        "dtype": "float32",
        "frame": "Horizontal FLU",
        "scale": "m/s (fisico directo)",
        "description": "Velocidad lineal propioceptiva [vx_flu, vy_flu, vz_flu] en cada keyframe.",
    },
    {
        "stage": "Context Input",
        "name": "context_flight_state",
        "shape": "(N_demos, L=10, 1)",
        "dtype": "float32",
        "frame": "Escalar binario",
        "scale": "{0.0, 1.0}",
        "description": "Estado de vuelo en el keyframe (1.0 = vuelo activo, 0.0 = aterrizaje).",
    },
    {
        "stage": "Context Input",
        "name": "context_action_se3",
        "shape": "(N_demos, L=10, T=8, 4, 4)",
        "dtype": "float32",
        "frame": "Horizontal FLU relativo al keyframe l",
        "scale": "Traslacion s = 0.1; Rotacion R_z(dyaw)",
        "description": "Horizonte futuro acumulado T_{EA}^{(t_l, 1..8)} demostrado en cada keyframe.",
    },
    {
        "stage": "Context Input",
        "name": "context_delta_se3",
        "shape": "(N_demos, L=10, T=8, 4, 4)",
        "dtype": "float32",
        "frame": "Horizontal FLU paso a paso (k-1 -> k)",
        "scale": "Traslacion s = 0.1 (Δt = 0.1 s)",
        "description": "Incrementos paso a paso δT_k = (T_{EA}^{k-1})^{-1} T_{EA}^k en cada keyframe.",
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
        "description": "Pose actual T_{WE}^t (usada para proyectar la salida al mundo; NO entra a la GNN).",
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
        "shape": "(T=8, 4, 4)",
        "dtype": "float32",
        "frame": "Horizontal FLU relativo al instante t",
        "scale": "Traslacion s = 0.1; Rotacion R_z(dyaw_k)",
        "description": "Horizonte de T=8 poses relativas acumuladas predichas T_hat_{EA}^{(t, 1..8)}.",
    },
    {
        "stage": "Network Output",
        "name": "pred_flight_state",
        "shape": "(T=8, 1)",
        "dtype": "float32",
        "frame": "Probabilidad escalar",
        "scale": "[0.0, 1.0]",
        "description": "Prediccion del estado de vuelo futuro s_hat_{t+1..t+8} (<0.5 activa aterrizaje).",
    },
    {
        "stage": "Controller Output",
        "name": "pred_world_waypoints_enu_m",
        "shape": "(T=8, 4)",
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


def extract_context_from_episode(
    data: Any,
    num_keyframes: int = 10,
) -> Dict[str, torch.Tensor]:
    """
    Extrae los tensores de contexto de un episodio `HeteroData` en sus `L = num_keyframes`
    indices de keyframe precalculados (`data['agent'].keyframe_indices`).
    """
    kf_idx = data["agent"].keyframe_indices.to(dtype=torch.long)
    L_actual = int(kf_idx.numel())
    T_total = int(data["scene"].pos.shape[0])
    if L_actual == 0 or num_keyframes > L_actual:
        kf_idx = torch.linspace(0, max(0, T_total - 1), steps=num_keyframes).round().to(dtype=torch.long)
    elif L_actual != num_keyframes:
        # Ajustar a exactamente `num_keyframes` indices manteniendo extremos
        lin = torch.linspace(0, L_actual - 1, steps=num_keyframes).round().to(dtype=torch.long)
        kf_idx = kf_idx[lin]

    scene_pos = data["scene"].pos[kf_idx].to(dtype=torch.float32)  # (L, 2048, 3)
    pose_se3 = data["agent"].pose_se3[kf_idx].to(dtype=torch.float32)  # (L, 4, 4)
    vel_flu = data["agent"].vel_flu[kf_idx].to(dtype=torch.float32)  # (L, 3)
    flight_state = data["agent"].flight_state[kf_idx].to(dtype=torch.float32)  # (L, 1)
    action_se3 = data["agent"].action_se3[kf_idx].to(dtype=torch.float32)  # (L, 8, 4, 4)
    action_y = data["agent"].y[kf_idx].to(dtype=torch.float32)  # (L, 8, 5)
    valid_mask = data["agent"].valid_mask[kf_idx].to(dtype=torch.bool)  # (L, 8)

    delta_se3 = cumulative_to_incremental_se3(action_se3)  # (L, 8, 4, 4)
    delta_4dof = se3_to_4dof_vector(delta_se3)  # (L, 8, 4)
    target_flight_state = action_y[..., 4:5]  # (L, 8, 1)

    return {
        "keyframe_indices": kf_idx,
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


def build_context_pack_from_episodes(
    episode_items: Sequence[Union[str, Path, Any]],
    num_keyframes: int = 10,
) -> Dict[str, torch.Tensor]:
    """
    Construye el paquete de contexto apilando `N_demos` episodios en sus `L = num_keyframes`
    keyframes. Cada elemento de `episode_items` puede ser una ruta a `.pt` o un `HeteroData` ya cargado.
    """
    if not episode_items:
        raise ValueError("build_context_pack_from_episodes requiere al menos 1 episodio de contexto.")

    per_demo = []
    for item in episode_items:
        ep_data = load_heterodata_episode(item) if isinstance(item, (str, Path)) else item
        per_demo.append(extract_context_from_episode(ep_data, num_keyframes=num_keyframes))

    return {
        "context_keyframe_indices": torch.stack([d["keyframe_indices"] for d in per_demo], dim=0),
        "context_scene_pos": torch.stack([d["scene_pos"] for d in per_demo], dim=0),
        "context_pose_se3": torch.stack([d["pose_se3"] for d in per_demo], dim=0),
        "context_vel_flu": torch.stack([d["vel_flu"] for d in per_demo], dim=0),
        "context_flight_state": torch.stack([d["flight_state"] for d in per_demo], dim=0),
        "context_action_se3": torch.stack([d["action_se3"] for d in per_demo], dim=0),
        "context_delta_se3": torch.stack([d["delta_se3"] for d in per_demo], dim=0),
        "context_delta_4dof": torch.stack([d["delta_4dof"] for d in per_demo], dim=0),
        "context_target_flight_state": torch.stack([d["target_flight_state"] for d in per_demo], dim=0),
        "context_valid_mask": torch.stack([d["valid_mask"] for d in per_demo], dim=0),
    }


def verify_sample_contract(
    sample: Dict[str, Any],
    expected_num_demos: Optional[int] = None,
    expected_num_keyframes: int = 10,
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
    assert L == expected_num_keyframes, f"Expected L={expected_num_keyframes}, got {L}"
    assert P == expected_num_points and three == 3

    assert sample["context_pose_se3"].shape == (N_demos, L, 4, 4)
    assert sample["context_vel_flu"].shape == (N_demos, L, 3)
    assert sample["context_flight_state"].shape == (N_demos, L, 1)
    assert sample["context_action_se3"].shape == (N_demos, L, expected_horizon, 4, 4)
    assert sample["context_delta_se3"].shape == (N_demos, L, expected_horizon, 4, 4)
    assert sample["context_delta_4dof"].shape == (N_demos, L, expected_horizon, 4)
    assert sample["context_target_flight_state"].shape == (N_demos, L, expected_horizon, 1)
    assert sample["context_valid_mask"].shape == (N_demos, L, expected_horizon)

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

    Para cada tarea descubierta en `dataset_dir`, agrupa sus episodios aceptados (`episode_XX/heterodata.pt`)
    y expone todas las muestras de consulta `O_t`:
      - Pasos nominales `t in [0, T-1]` de `data['agent']`.
      - Pasos perturbados de recuperacion `j in [0, T_aug-1]` de `data['aug']` (si `include_aug=True`).

    Al consultar `__getitem__(idx)`, empareja la muestra `O_t` del episodio `e_query` con `num_context_demos`
    episodios de contexto de la MISMA tarea (excluyendo `e_query` siempre que la tarea tenga >= 2 episodios).
    """

    def __init__(
        self,
        dataset_dir: Union[str, Path],
        split: Optional[str] = None,
        task_ids: Optional[Sequence[str]] = None,
        num_context_demos: int = 2,
        num_keyframes: int = 10,
        include_aug: bool = True,
        cache_episodes_in_ram: bool = True,
        max_tasks: Optional[int] = None,
    ) -> None:
        super().__init__()
        self.dataset_dir = Path(dataset_dir)
        self.split = split
        self.num_context_demos = int(num_context_demos)
        self.num_keyframes = int(num_keyframes)
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
                tasks = [
                    line.strip()
                    for line in split_file.read_text(encoding="utf-8").splitlines()
                    if line.strip()
                ]
            else:
                tasks = sorted(p.name for p in self.dataset_dir.glob("task_*") if p.is_dir())
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
                    ep_data, num_keyframes=self.num_keyframes
                )
            return self._context_cache[pt_path]
        ep_data = self._get_episode_data(pt_path)
        return extract_context_from_episode(ep_data, num_keyframes=self.num_keyframes)

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
        ctx_pack = {
            "context_keyframe_indices": torch.stack([d["keyframe_indices"] for d in ctx_dicts], dim=0),
            "context_scene_pos": torch.stack([d["scene_pos"] for d in ctx_dicts], dim=0),
            "context_pose_se3": torch.stack([d["pose_se3"] for d in ctx_dicts], dim=0),
            "context_vel_flu": torch.stack([d["vel_flu"] for d in ctx_dicts], dim=0),
            "context_flight_state": torch.stack([d["flight_state"] for d in ctx_dicts], dim=0),
            "context_action_se3": torch.stack([d["action_se3"] for d in ctx_dicts], dim=0),
            "context_delta_se3": torch.stack([d["delta_se3"] for d in ctx_dicts], dim=0),
            "context_delta_4dof": torch.stack([d["delta_4dof"] for d in ctx_dicts], dim=0),
            "context_target_flight_state": torch.stack([d["target_flight_state"] for d in ctx_dicts], dim=0),
            "context_valid_mask": torch.stack([d["valid_mask"] for d in ctx_dicts], dim=0),
        }

        keypoints_body = (
            ep_data["agent"].keypoints_body.to(dtype=torch.float32)
            if hasattr(ep_data["agent"], "keypoints_body")
            else get_uav_keypoints_body_torch(scale_factor=DEFAULT_SCALE_FACTOR)
        )

        if source_type == "nominal":
            live_scene_pos = ep_data["scene"].pos[step_idx : step_idx + 1].to(dtype=torch.float32)
            live_pose_se3 = ep_data["agent"].pose_se3[step_idx : step_idx + 1].to(dtype=torch.float32)
            live_vel_flu = ep_data["agent"].vel_flu[step_idx : step_idx + 1].to(dtype=torch.float32)
            live_flight_state = ep_data["agent"].flight_state[step_idx : step_idx + 1].to(dtype=torch.float32)
            target_action_se3 = ep_data["agent"].action_se3[step_idx].to(dtype=torch.float32)
            target_y = ep_data["agent"].y[step_idx].to(dtype=torch.float32)
            target_valid_mask = ep_data["agent"].valid_mask[step_idx].to(dtype=torch.bool)
            nominal_timestep = int(step_idx)
        else:
            live_scene_pos = ep_data["aug"].scene_pos[step_idx : step_idx + 1].to(dtype=torch.float32)
            live_pose_se3 = ep_data["aug"].pose_se3[step_idx : step_idx + 1].to(dtype=torch.float32)
            live_vel_flu = ep_data["aug"].vel_flu[step_idx : step_idx + 1].to(dtype=torch.float32)
            live_flight_state = ep_data["aug"].flight_state_in[step_idx : step_idx + 1].to(dtype=torch.float32)
            target_action_se3 = ep_data["aug"].action_se3[step_idx].to(dtype=torch.float32)
            target_y = ep_data["aug"].y[step_idx].to(dtype=torch.float32)
            target_valid_mask = ep_data["aug"].valid_mask[step_idx].to(dtype=torch.bool)
            nominal_timestep = int(ep_data["aug"].timestep_idx[step_idx].item())

        target_delta_se3 = cumulative_to_incremental_se3(target_action_se3)
        target_delta_4dof = se3_to_4dof_vector(target_delta_se3)
        target_flight_state = target_y[:, 4:5]

        return {
            "task_id": task_id,
            "episode_id": query_pt_path.parent.name,
            "source_type": source_type,
            "step_idx": int(step_idx),
            "nominal_timestep": nominal_timestep,
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
