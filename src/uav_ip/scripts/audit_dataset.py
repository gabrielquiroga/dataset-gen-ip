#!/usr/bin/env python3
"""
audit_dataset.py — Auditor Universal en 3 Capas de Datasets y Generador de Particiones Held-Out.

Inspecciona cualquier estructura de dataset (`datasets/task_XXXX/episode_YY[_attempt_ZZ]`):
  - Capa 1 (Vuelos crudos):
      * Tasa de aceptación por intento original (`episode_XX`) vs. tasa de aceptación tras reintentos
        por slot de episodio (`episode_XX` + `episode_XX_attempt_YY`).
      * Desglose exhaustivo de `rejection_reasons` en `episode_meta.json`.
      * Integridad semántica y variabilidad espacial 3D intra-tarea (`waypoints.yaml`).
  - Capa 2 (Tensores `.pt` / `.npz` de `HeteroData`):
      * Finitud estricta (cero NaNs / Infs en todos los tensores).
      * Consistencia dimensional: `(T, 2048, 3)`, `(T, 8, 4, 4)`, `(T, 8, 5)`, `valid_mask` `(T, 8)`.
      * Ortogonalidad estricta de matrices SO(3): `||R^T R - I||_max <= 1e-4` y `|det(R) - 1| <= 1e-4`.
      * Horizonte físico `T = 8` a 10 Hz (`Δt = 0.1 s`, cubriendo `0.8 s`) y cota esférica `||δp||_2 <= 0.60 m`.
  - Capa 3 (Particionado sin fuga de datos):
      * Exige un mínimo de episodios aceptados (`>= min_accepted_episodes`, por defecto 5) por tarea
        antes de incluirla en los splits `train_tasks.json`, `val_tasks.json` y `held_out_tasks.json`.
      * Particionado a nivel de firmas semánticas canónicas (`get_task_signature`) con cero data leakage.

Uso:
  python3 audit_dataset.py --dataset-dir datasets
  python3 audit_dataset.py --dataset-dir datasets --output-dir datasets/splits --min-accepted-episodes 5 --seed 42
"""

import argparse
import json
import math
import random
import re
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple

import numpy as np
import yaml

# Asegurar inclusion de ~/venv_ip y ~/.local en sys.path para poder cargar PyTorch/PyG al auditar .pt
_py_ver = f"python{sys.version_info.major}.{sys.version_info.minor}"
for _candidate_site in [
    Path.home() / "venv_ip" / "lib" / _py_ver / "site-packages",
    Path.home() / ".local" / "lib" / _py_ver / "site-packages",
]:
    if _candidate_site.is_dir() and str(_candidate_site) not in sys.path:
        sys.path.append(str(_candidate_site))

_pkg_root = Path(__file__).resolve().parent.parent
if (_pkg_root / "uav_ip").is_dir() and str(_pkg_root) not in sys.path:
    sys.path.insert(0, str(_pkg_root))

# Importar get_task_signature de task_generator
try:
    from task_generator import get_task_signature
except ImportError:
    script_dir = Path(__file__).resolve().parent
    if str(script_dir) not in sys.path:
        sys.path.append(str(script_dir))
    from task_generator import get_task_signature

try:
    import torch
except ImportError:
    torch = None


DEFAULT_MIN_ACCEPTED_EPISODES: int = 5
EXPECTED_HORIZON_STEPS: int = 8
EXPECTED_DT_S: float = 0.10
EXPECTED_HORIZON_DURATION_S: float = 0.80
EXPECTED_LIDAR_POINTS: int = 2048
SO3_ORTHOGONALITY_TOL: float = 1e-4
TUBE_AUG_SPHERE_RADIUS_M: float = 0.60


def _parse_episode_slot_and_attempt(dir_name: str, meta: Optional[Dict[str, Any]] = None) -> Tuple[str, int]:
    """
    Determina el identificador del slot base (`episode_XX`) y el índice de intento (`0` = original,
    `>= 1` = reintento) a partir del nombre de carpeta o `episode_meta.json`.
    """
    if meta is not None:
        if "original_episode_id" in meta and "attempt_index" in meta:
            return str(meta["original_episode_id"]), int(meta["attempt_index"])
    m = re.match(r"^(episode_\d+)_attempt_(\d+)$", dir_name)
    if m:
        return m.group(1), int(m.group(2))
    return dir_name, 0


def _is_meta_accepted(meta: Dict[str, Any]) -> bool:
    """Evalua si un `episode_meta.json` corresponde a un episodio validado y aceptado."""
    if "validation_status" in meta:
        return meta.get("validation_status") == "accepted"
    return str(meta.get("status", "")).lower() == "success"


def _check_so3_batch(rot_matrices: np.ndarray, label: str, tol: float = SO3_ORTHOGONALITY_TOL) -> List[str]:
    """
    Verifica ortogonalidad (`R^T R = I`) y determinante (`det(R) = +1`) para un lote de matrices 3x3.
    """
    errors: List[str] = []
    R = np.asarray(rot_matrices, dtype=np.float64).reshape(-1, 3, 3)
    if len(R) == 0:
        return errors
    if not np.all(np.isfinite(R)):
        errors.append(f"{label}:non_finite_so3_entries")
        return errors

    eye = np.eye(3, dtype=np.float64)[None, :, :]
    rt_r = np.matmul(np.transpose(R, (0, 2, 1)), R)
    max_orth_err = float(np.max(np.abs(rt_r - eye)))
    if max_orth_err > tol:
        errors.append(f"{label}:so3_orthogonality_violation (max_err={max_orth_err:.2e} > {tol:.1e})")

    dets = np.linalg.det(R)
    max_det_err = float(np.max(np.abs(dets - 1.0)))
    if max_det_err > tol:
        errors.append(f"{label}:so3_determinant_violation (max_det_err={max_det_err:.2e} > {tol:.1e})")

    return errors


def audit_heterodata_file(
    tensor_path: Path,
    expected_horizon: int = EXPECTED_HORIZON_STEPS,
    expected_points: int = EXPECTED_LIDAR_POINTS,
    expected_dt_s: float = EXPECTED_DT_S,
) -> Dict[str, Any]:
    """
    Capa 2: Audita un archivo `.pt` (`HeteroData`) o `.npz` verificando:
      1. Finitud estricta (cero NaNs / Infs en todos los tensores).
      2. Dimensiones exactas: `scene.pos` `(T, 2048, 3)`, `agent.pose_se3` `(T, 4, 4)`,
         `agent.action_se3` `(T, 8, 4, 4)`, `agent.y` `(T, 8, 5)`, `agent.valid_mask` `(T, 8)`.
      3. Ortogonalidad estricta en SO(3) (`R^T R = I`, `det(R) = +1`) en todas las poses y acciones.
      4. Horizonte temporal físico `T = 8` a 10 Hz (`Δt = 0.1 s` -> `0.8 s` de horizonte).
      5. Cota esférica euclídea `||δp||_2 <= 0.60 m` en los nodos `aug`.
    """
    path = Path(tensor_path)
    issues: List[str] = []
    if not path.is_file():
        return {
            "path": str(path),
            "valid": False,
            "T": 0,
            "T_aug": 0,
            "horizon": 0,
            "horizon_duration_s": None,
            "issues": ["missing_tensor_file"],
        }

    tensors_np: Dict[str, np.ndarray] = {}
    dt_s_val: Optional[float] = None

    try:
        if path.suffix == ".pt":
            if torch is None:
                return {
                    "path": str(path),
                    "valid": False,
                    "T": 0,
                    "T_aug": 0,
                    "horizon": 0,
                    "horizon_duration_s": None,
                    "issues": ["torch_not_installed_cannot_audit_pt"],
                }
            data = torch.load(path, map_location="cpu", weights_only=False)
            tensors_np["scene_pos"] = data["scene"].pos.detach().cpu().numpy()
            tensors_np["agent_pos_enu"] = data["agent"].pos_enu.detach().cpu().numpy()
            tensors_np["agent_yaw_enu"] = data["agent"].yaw_enu.detach().cpu().numpy()
            tensors_np["agent_rot_enu"] = data["agent"].rot_enu.detach().cpu().numpy()
            tensors_np["agent_pose_se3"] = data["agent"].pose_se3.detach().cpu().numpy()
            tensors_np["agent_action_se3"] = data["agent"].action_se3.detach().cpu().numpy()
            tensors_np["agent_y"] = data["agent"].y.detach().cpu().numpy()
            if hasattr(data["agent"], "valid_mask") and data["agent"].valid_mask is not None:
                tensors_np["valid_mask"] = data["agent"].valid_mask.detach().cpu().numpy()
            if hasattr(data["agent"], "timestamps_s") and data["agent"].timestamps_s is not None:
                tensors_np["timestamps_s"] = data["agent"].timestamps_s.detach().cpu().numpy()
            if hasattr(data["agent"], "dt_s") and data["agent"].dt_s is not None:
                dt_s_val = float(data["agent"].dt_s.detach().cpu().item())

            if "aug" in data.node_types and hasattr(data["aug"], "timestep_idx"):
                tensors_np["aug_timestep_idx"] = data["aug"].timestep_idx.detach().cpu().numpy()
                tensors_np["aug_scene_pos"] = data["aug"].scene_pos.detach().cpu().numpy()
                tensors_np["aug_rot_enu"] = data["aug"].rot_enu.detach().cpu().numpy()
                tensors_np["aug_pose_se3"] = data["aug"].pose_se3.detach().cpu().numpy()
                tensors_np["aug_action_se3"] = data["aug"].action_se3.detach().cpu().numpy()
                tensors_np["aug_delta_pos_m"] = data["aug"].delta_pos_m.detach().cpu().numpy()
                tensors_np["aug_y"] = data["aug"].y.detach().cpu().numpy()
        else:
            with np.load(path, allow_pickle=False) as npz:
                tensors_np["scene_pos"] = npz["scene_pos"]
                tensors_np["agent_pos_enu"] = npz["pos_enu"]
                tensors_np["agent_yaw_enu"] = npz["yaw_enu"]
                tensors_np["agent_rot_enu"] = npz["rot_enu"]
                tensors_np["agent_pose_se3"] = npz["pose_se3"]
                tensors_np["agent_action_se3"] = npz["action_se3"]
                tensors_np["agent_y"] = npz["agent_y"]
                if "valid_mask" in npz:
                    tensors_np["valid_mask"] = npz["valid_mask"]
                if "timestamps_s" in npz:
                    tensors_np["timestamps_s"] = npz["timestamps_s"]
                if "dt_s" in npz:
                    dt_s_val = float(npz["dt_s"])
                if "aug_timestep_idx" in npz:
                    tensors_np["aug_timestep_idx"] = npz["aug_timestep_idx"]
                    tensors_np["aug_scene_pos"] = npz["aug_scene_pos"]
                    tensors_np["aug_rot_enu"] = npz["aug_rot_enu"]
                    tensors_np["aug_pose_se3"] = npz["aug_pose_se3"]
                    tensors_np["aug_action_se3"] = npz["aug_action_se3"]
                    tensors_np["aug_delta_pos_m"] = npz["aug_delta_pos_m"]
                    tensors_np["aug_y"] = npz["aug_y"]
    except Exception as exc:
        return {
            "path": str(path),
            "valid": False,
            "T": 0,
            "T_aug": 0,
            "horizon": 0,
            "horizon_duration_s": None,
            "issues": [f"corrupt_or_unreadable_tensor_file ({type(exc).__name__}: {exc})"],
        }

    # 1. Finitud estricta (cero NaNs / Infs)
    for k_name, arr in tensors_np.items():
        if not np.all(np.isfinite(arr)):
            issues.append(f"non_finite_values_in_{k_name}")

    scene_pos = tensors_np["scene_pos"]
    pose_se3 = tensors_np["agent_pose_se3"]
    action_se3 = tensors_np["agent_action_se3"]
    agent_y = tensors_np["agent_y"]
    rot_enu = tensors_np["agent_rot_enu"]

    T = int(scene_pos.shape[0]) if scene_pos.ndim >= 1 else 0
    if T <= 0:
        issues.append("empty_trajectory_T_zero")

    # 2. Dimensiones exactas
    if scene_pos.shape != (T, expected_points, 3):
        issues.append(f"invalid_scene_pos_shape ({scene_pos.shape} != {(T, expected_points, 3)})")
    if pose_se3.shape != (T, 4, 4):
        issues.append(f"invalid_pose_se3_shape ({pose_se3.shape} != {(T, 4, 4)})")
    if rot_enu.shape != (T, 3, 3):
        issues.append(f"invalid_rot_enu_shape ({rot_enu.shape} != {(T, 3, 3)})")
    if action_se3.shape != (T, expected_horizon, 4, 4):
        issues.append(f"invalid_action_se3_shape ({action_se3.shape} != {(T, expected_horizon, 4, 4)})")
    if agent_y.shape != (T, expected_horizon, 5):
        issues.append(f"invalid_agent_y_shape ({agent_y.shape} != {(T, expected_horizon, 5)})")

    if "valid_mask" not in tensors_np:
        issues.append("missing_valid_mask_tensor")
    else:
        vm = tensors_np["valid_mask"]
        if vm.shape != (T, expected_horizon):
            issues.append(f"invalid_valid_mask_shape ({vm.shape} != {(T, expected_horizon)})")
        elif vm.dtype != np.bool_:
            issues.append(f"invalid_valid_mask_dtype ({vm.dtype} != bool)")
        elif T > 0:
            # Verificar que los ultimos min(T, expected_horizon - 1) pasos reflejen el horizonte terminal
            expected_vm = np.zeros((T, expected_horizon), dtype=bool)
            for t_idx in range(T):
                for k_idx in range(1, expected_horizon + 1):
                    expected_vm[t_idx, k_idx - 1] = bool((t_idx + k_idx) < T)
            if not np.array_equal(vm, expected_vm):
                issues.append("valid_mask_mismatch_with_terminal_horizon")

    # 3. Ortogonalidad estricta en SO(3)
    if rot_enu.shape == (T, 3, 3):
        issues.extend(_check_so3_batch(rot_enu, "agent.rot_enu"))
    if pose_se3.shape == (T, 4, 4):
        issues.extend(_check_so3_batch(pose_se3[:, :3, :3], "agent.pose_se3"))
        bottom_row = pose_se3[:, 3, :]
        expected_bottom = np.tile(np.array([0.0, 0.0, 0.0, 1.0]), (T, 1))
        if T > 0 and float(np.max(np.abs(bottom_row - expected_bottom))) > 1e-5:
            issues.append("agent.pose_se3:invalid_homogeneous_bottom_row")
    if action_se3.shape == (T, expected_horizon, 4, 4):
        issues.extend(_check_so3_batch(action_se3[:, :, :3, :3], "agent.action_se3"))

    # 4. Horizonte temporal físico (T=8 a 10 Hz -> 0.8 s)
    horizon_duration_s: Optional[float] = None
    if "timestamps_s" in tensors_np and len(tensors_np["timestamps_s"]) >= 2:
        ts_arr = np.asarray(tensors_np["timestamps_s"], dtype=np.float64)
        dts = np.diff(ts_arr)
        # El ultimo punto de la grilla puede coincidir con el final exacto de la trayectoria
        interior_dts = dts[:-1] if len(dts) > 1 else dts
        median_dt = float(np.median(interior_dts))
        horizon_duration_s = float(expected_horizon * median_dt)
        if abs(median_dt - expected_dt_s) > 0.015:
            issues.append(
                f"non_10hz_temporal_grid (median_dt={median_dt:.4f}s != {expected_dt_s:.2f}s, "
                f"horizon={horizon_duration_s:.3f}s)"
            )
    elif dt_s_val is not None:
        horizon_duration_s = float(expected_horizon * dt_s_val)
        if abs(dt_s_val - expected_dt_s) > 0.015:
            issues.append(f"non_10hz_dt_s ({dt_s_val:.4f}s != {expected_dt_s:.2f}s)")

    # 5. Auditoria de Tube Augmentation ('aug')
    T_aug = 0
    if "aug_timestep_idx" in tensors_np:
        T_aug = int(len(tensors_np["aug_timestep_idx"]))
        if T_aug > 0:
            if tensors_np["aug_scene_pos"].shape != (T_aug, expected_points, 3):
                issues.append(f"invalid_aug_scene_pos_shape ({tensors_np['aug_scene_pos'].shape})")
            if tensors_np["aug_pose_se3"].shape != (T_aug, 4, 4):
                issues.append(f"invalid_aug_pose_se3_shape ({tensors_np['aug_pose_se3'].shape})")
            else:
                issues.extend(_check_so3_batch(tensors_np["aug_pose_se3"][:, :3, :3], "aug.pose_se3"))
            if tensors_np["aug_action_se3"].shape != (T_aug, expected_horizon, 4, 4):
                issues.append(f"invalid_aug_action_se3_shape ({tensors_np['aug_action_se3'].shape})")
            else:
                issues.extend(_check_so3_batch(tensors_np["aug_action_se3"][:, :, :3, :3], "aug.action_se3"))

            dp_norms = np.linalg.norm(tensors_np["aug_delta_pos_m"], axis=1)
            max_dp_norm = float(np.max(dp_norms))
            if max_dp_norm > TUBE_AUG_SPHERE_RADIUS_M + 1e-4:
                issues.append(
                    f"aug.delta_pos_m:exceeds_euclidean_sphere_bound "
                    f"(max_norm={max_dp_norm:.4f}m > {TUBE_AUG_SPHERE_RADIUS_M:.2f}m)"
                )

    return {
        "path": str(path),
        "valid": len(issues) == 0,
        "T": T,
        "T_aug": T_aug,
        "horizon": int(action_se3.shape[1]) if action_se3.ndim >= 2 else 0,
        "horizon_duration_s": round(horizon_duration_s, 4) if horizon_duration_s is not None else None,
        "issues": issues,
    }


def audit_dataset(
    dataset_dir: Path,
    output_dir: Path,
    train_ratio: float = 0.70,
    val_ratio: float = 0.15,
    held_out_ratio: float = 0.15,
    seed: int = 42,
    min_accepted_episodes: int = DEFAULT_MIN_ACCEPTED_EPISODES,
    no_export: bool = False,
) -> Dict[str, Any]:
    """Ejecuta la auditoría completa en 3 capas del dataset y genera las particiones."""
    dataset_dir = Path(dataset_dir)
    output_dir = Path(output_dir)
    if not dataset_dir.is_dir():
        print(f"[ERROR] El directorio del dataset no existe: {dataset_dir}")
        sys.exit(1)

    task_dirs = sorted([d for d in dataset_dir.iterdir() if d.is_dir() and d.name.startswith("task_")])
    if not task_dirs:
        print(f"[WARN] No se encontraron carpetas 'task_*' en {dataset_dir}.")
        return {}

    print("=" * 72)
    print("  AUDITORÍA EN 3 CAPAS DE DATASET — INSTANT POLICY UAV")
    print(f"  Directorio: {dataset_dir}")
    print(f"  Tareas encontradas: {len(task_dirs)}  |  Min aceptados/tarea: {min_accepted_episodes}")
    print("=" * 72)

    tasks_by_signature: Dict[str, List[str]] = {}
    eligible_tasks_by_signature: Dict[str, List[str]] = {}
    task_metadata: Dict[str, Dict[str, Any]] = {}
    primitive_counts: Dict[str, int] = {}
    landmark_length_counts: Dict[int, int] = {}

    # Contadores Capa 1 (Vuelos crudos: intento original vs reintentos)
    total_episodes = 0
    total_accepted = 0
    total_rejected = 0
    total_timeout = 0

    primary_attempts_total = 0
    primary_attempts_accepted = 0
    retry_attempts_total = 0
    retry_attempts_accepted = 0
    total_episode_slots = 0
    accepted_episode_slots = 0

    rejection_reason_counts: Dict[str, int] = {}
    tasks_with_identical_waypoints: List[str] = []

    # Contadores Capa 2 (Tensores .pt / .npz)
    tensor_files_audited = 0
    tensor_files_valid = 0
    tensor_files_invalid = 0
    tensor_audit_details: List[Dict[str, Any]] = []
    tensor_issue_counts: Dict[str, int] = {}

    for t_dir in task_dirs:
        task_id = t_dir.name
        cfg_file = t_dir / "task_config.yaml"
        if not cfg_file.is_file():
            print(f"  [WARN] {task_id} no contiene task_config.yaml. Saltando.")
            continue

        with open(cfg_file, "r", encoding="utf-8") as f:
            cfg = yaml.safe_load(f)

        signature = get_task_signature(cfg)
        if signature not in tasks_by_signature:
            tasks_by_signature[signature] = []
        tasks_by_signature[signature].append(task_id)

        n_landmarks = len(cfg.get("landmarks", []))
        landmark_length_counts[n_landmarks] = landmark_length_counts.get(n_landmarks, 0) + 1

        for prim in cfg.get("primitives", []):
            ptype = prim.get("type", "unknown")
            primitive_counts[ptype] = primitive_counts.get(ptype, 0) + 1

        ep_dirs = sorted([d for d in t_dir.iterdir() if d.is_dir() and d.name.startswith("episode_")])
        ep_count = len(ep_dirs)
        ep_accepted = 0
        ep_waypoints_signatures: Set[str] = set()
        slots_status: Dict[str, Dict[str, bool]] = {}

        for ep in ep_dirs:
            total_episodes += 1
            meta_file = ep / "episode_meta.json"
            meta: Optional[Dict[str, Any]] = None
            is_accepted = False

            if meta_file.is_file():
                try:
                    with open(meta_file, "r", encoding="utf-8") as f:
                        meta = json.load(f)
                    is_accepted = _is_meta_accepted(meta)
                    if meta.get("status") == "timeout":
                        total_timeout += 1
                    if not is_accepted:
                        reasons = meta.get("rejection_reasons") or []
                        if not reasons and meta.get("status") not in ("success", "accepted"):
                            reasons = [str(meta.get("status", "unknown_failure"))]
                        for r in reasons:
                            # Normalizar prefijo del motivo de rechazo (sin sufijos entre parentesis)
                            r_key = str(r).split(" (")[0].strip()
                            rejection_reason_counts[r_key] = rejection_reason_counts.get(r_key, 0) + 1
                except Exception:
                    is_accepted = False
                    rejection_reason_counts["corrupt_episode_meta"] = (
                        rejection_reason_counts.get("corrupt_episode_meta", 0) + 1
                    )
            else:
                rejection_reason_counts["missing_episode_meta"] = (
                    rejection_reason_counts.get("missing_episode_meta", 0) + 1
                )

            slot_id, attempt_idx = _parse_episode_slot_and_attempt(ep.name, meta)
            if slot_id not in slots_status:
                slots_status[slot_id] = {"primary_accepted": False, "any_accepted": False}

            if attempt_idx == 0:
                primary_attempts_total += 1
                if is_accepted:
                    primary_attempts_accepted += 1
                    slots_status[slot_id]["primary_accepted"] = True
            else:
                retry_attempts_total += 1
                if is_accepted:
                    retry_attempts_accepted += 1

            if is_accepted:
                total_accepted += 1
                ep_accepted += 1
                slots_status[slot_id]["any_accepted"] = True
            else:
                total_rejected += 1

            # Verificar poses físicas de los waypoints
            wp_file = ep / "waypoints.yaml"
            if wp_file.is_file():
                try:
                    with open(wp_file, "r", encoding="utf-8") as f:
                        wp_data = yaml.safe_load(f)
                    poses_token = tuple(
                        (
                            lm.get("id"),
                            round(float(lm.get("x", 0)), 2),
                            round(float(lm.get("y", 0)), 2),
                            round(float(lm.get("z", 0)), 2),
                        )
                        for lm in wp_data.get("landmarks", [])
                    )
                    ep_waypoints_signatures.add(str(poses_token))
                except Exception:
                    pass

            # Capa 2: Auditar tensor heterodata.pt / heterodata.npz si existe en el episodio
            pt_candidate = ep / "heterodata.pt"
            npz_candidate = ep / "heterodata.npz"
            tensor_target = pt_candidate if pt_candidate.is_file() else (npz_candidate if npz_candidate.is_file() else None)
            if tensor_target is not None:
                tensor_files_audited += 1
                t_res = audit_heterodata_file(tensor_target)
                tensor_audit_details.append(t_res)
                if t_res["valid"]:
                    tensor_files_valid += 1
                else:
                    tensor_files_invalid += 1
                    for iss in t_res["issues"]:
                        iss_key = iss.split(" (")[0].strip()
                        tensor_issue_counts[iss_key] = tensor_issue_counts.get(iss_key, 0) + 1

        total_episode_slots += len(slots_status)
        accepted_episode_slots += sum(1 for s in slots_status.values() if s["any_accepted"])

        if ep_count > 1 and len(ep_waypoints_signatures) == 1:
            tasks_with_identical_waypoints.append(task_id)

        task_metadata[task_id] = {
            "signature": signature,
            "episodes_total": ep_count,
            "episodes_accepted": ep_accepted,
            "episodes_success": ep_accepted,
            "episode_slots": len(slots_status),
            "episode_slots_accepted": sum(1 for s in slots_status.values() if s["any_accepted"]),
            "eligible_for_split": bool(ep_accepted >= int(min_accepted_episodes)),
            "logic": cfg.get("logic", "A1_DIRECT"),
            "profile": cfg.get("profile", "B1_LINEAR"),
        }

        if ep_accepted >= int(min_accepted_episodes):
            if signature not in eligible_tasks_by_signature:
                eligible_tasks_by_signature[signature] = []
            eligible_tasks_by_signature[signature].append(task_id)

    unique_signatures_count = len(tasks_by_signature)
    duplicate_task_groups = {sig: ids for sig, ids in tasks_by_signature.items() if len(ids) > 1}
    excluded_tasks = sorted([
        tid for tid, tmeta in task_metadata.items() if not tmeta["eligible_for_split"]
    ])

    primary_acceptance_rate = (
        (primary_attempts_accepted / primary_attempts_total) if primary_attempts_total > 0 else 0.0
    )
    post_retry_acceptance_rate = (
        (accepted_episode_slots / total_episode_slots) if total_episode_slots > 0 else 0.0
    )

    # ----------------------------------------------------------------
    #  Reporte en terminal (3 Capas)
    # ----------------------------------------------------------------
    print(f"\n📊 CAPA 1 — INTEGRIDAD SEMÁNTICA Y VUELOS CRUDOS:")
    print(f"  Total de carpetas de tareas:     {len(task_metadata)}")
    print(f"  Firmas semánticas únicas:        {unique_signatures_count}")
    if duplicate_task_groups:
        print(f"  ⚠️ COLISIONES / DUPLICADOS DETECTADOS: {len(duplicate_task_groups)} firmas compartidas")
    else:
        print("  ✅ CERO DUPLICADOS: Todas las tareas tienen firmas semánticas estrictamente únicas.")

    print(f"  Total intentos grabados:         {total_episodes} (Aceptados={total_accepted}, Rechazados={total_rejected})")
    print(
        f"  Tasa aceptación intento original: "
        f"{primary_attempts_accepted}/{primary_attempts_total} ({primary_acceptance_rate * 100.0:.1f}%)"
    )
    print(
        f"  Tasa aceptación tras reintentos:  "
        f"{accepted_episode_slots}/{total_episode_slots} ({post_retry_acceptance_rate * 100.0:.1f}%)"
    )
    if rejection_reason_counts:
        print("  Desglose de motivos de rechazo:")
        for reason, r_cnt in sorted(rejection_reason_counts.items(), key=lambda x: (-x[1], x[0])):
            print(f"    * {reason}: {r_cnt}")

    print(f"\n🧮 CAPA 2 — AUDITORÍA DE TENSORES (.pt / .npz):")
    print(f"  Archivos auditados: {tensor_files_audited} (Válidos={tensor_files_valid}, Inválidos={tensor_files_invalid})")
    if tensor_issue_counts:
        print("  ⚠️ Problemas detectados en tensores:")
        for iss, cnt in sorted(tensor_issue_counts.items(), key=lambda x: (-x[1], x[0])):
            print(f"    * {iss}: {cnt}")
    elif tensor_files_audited > 0:
        print("  ✅ Conforme: Finitud, dimensiones, ortogonalidad SO(3), valid_mask y horizonte 0.8s verificados.")

    # ----------------------------------------------------------------
    #  Capa 3: Generación de Partición (Held-Out Split con filtro >= min_accepted_episodes)
    # ----------------------------------------------------------------
    eligible_signatures = sorted(list(eligible_tasks_by_signature.keys()))
    rng = random.Random(seed)
    rng.shuffle(eligible_signatures)

    n_sigs = len(eligible_signatures)
    if n_sigs == 0:
        n_train = n_val = n_held_out = 0
    else:
        n_train = max(1, int(round(n_sigs * train_ratio)))
        n_val = int(round(n_sigs * val_ratio))
        if n_train + n_val > n_sigs:
            n_val = max(0, n_sigs - n_train)
        n_held_out = n_sigs - n_train - n_val

        if n_held_out <= 0 and n_sigs >= 3:
            n_held_out = 1
            n_train -= 1

    train_sigs = set(eligible_signatures[:n_train])
    val_sigs = set(eligible_signatures[n_train:n_train + n_val])
    held_out_sigs = set(eligible_signatures[n_train + n_val:])

    # Validación formal de cero fuga
    assert len(train_sigs.intersection(held_out_sigs)) == 0, "Fuga detectada entre train y held_out!"
    assert len(train_sigs.intersection(val_sigs)) == 0, "Fuga detectada entre train y val!"
    assert len(val_sigs.intersection(held_out_sigs)) == 0, "Fuga detectada entre val y held_out!"

    train_tasks = sorted([t for sig in train_sigs for t in eligible_tasks_by_signature[sig]])
    val_tasks = sorted([t for sig in val_sigs for t in eligible_tasks_by_signature[sig]])
    held_out_tasks = sorted([t for sig in held_out_sigs for t in eligible_tasks_by_signature[sig]])

    print(f"\n📁 CAPA 3 — PARTICIÓN FORMAL DEL DATASET (min_accepted={min_accepted_episodes}, seed={seed}):")
    if excluded_tasks:
        print(
            f"  ⚠️ Tareas excluidas por tener < {min_accepted_episodes} episodios aceptados "
            f"({len(excluded_tasks)}): {excluded_tasks[:10]}"
        )
    print(f"  * Train    ({len(train_sigs)} firmas): {len(train_tasks)} tareas ({train_tasks[:5]})")
    print(f"  * Val      ({len(val_sigs)} firmas): {len(val_tasks)} tareas ({val_tasks[:5]})")
    print(f"  * Held-Out ({len(held_out_sigs)} firmas): {len(held_out_tasks)} tareas ({held_out_tasks[:5]})")

    layer1_summary = {
        "total_tasks": len(task_metadata),
        "unique_signatures": unique_signatures_count,
        "total_episodes": total_episodes,
        "total_accepted": total_accepted,
        "total_rejected": total_rejected,
        "total_timeout": total_timeout,
        "primary_attempts_total": primary_attempts_total,
        "primary_attempts_accepted": primary_attempts_accepted,
        "primary_acceptance_rate": round(primary_acceptance_rate, 4),
        "retry_attempts_total": retry_attempts_total,
        "retry_attempts_accepted": retry_attempts_accepted,
        "total_episode_slots": total_episode_slots,
        "accepted_episode_slots": accepted_episode_slots,
        "post_retry_acceptance_rate": round(post_retry_acceptance_rate, 4),
        "rejection_reason_counts": rejection_reason_counts,
        "tasks_with_identical_waypoints": tasks_with_identical_waypoints,
    }
    layer2_summary = {
        "tensor_files_audited": tensor_files_audited,
        "tensor_files_valid": tensor_files_valid,
        "tensor_files_invalid": tensor_files_invalid,
        "tensor_issue_counts": tensor_issue_counts,
    }
    layer3_summary = {
        "min_accepted_episodes": int(min_accepted_episodes),
        "eligible_tasks_count": len(train_tasks) + len(val_tasks) + len(held_out_tasks),
        "excluded_insufficient_episodes_tasks": excluded_tasks,
        "train_tasks_count": len(train_tasks),
        "val_tasks_count": len(val_tasks),
        "held_out_tasks_count": len(held_out_tasks),
        "split_seed": seed,
    }

    if not no_export:
        output_dir.mkdir(parents=True, exist_ok=True)
        with open(output_dir / "train_tasks.json", "w", encoding="utf-8") as f:
            json.dump({"tasks": train_tasks, "signatures": sorted(list(train_sigs))}, f, indent=2)

        with open(output_dir / "val_tasks.json", "w", encoding="utf-8") as f:
            json.dump({"tasks": val_tasks, "signatures": sorted(list(val_sigs))}, f, indent=2)

        with open(output_dir / "held_out_tasks.json", "w", encoding="utf-8") as f:
            json.dump({"tasks": held_out_tasks, "signatures": sorted(list(held_out_sigs))}, f, indent=2)

        summary = {
            "total_tasks": len(task_metadata),
            "unique_signatures": unique_signatures_count,
            "total_episodes": total_episodes,
            "total_success": total_accepted,
            "success_rate": (total_accepted / total_episodes) if total_episodes > 0 else 0.0,
            "primary_acceptance_rate": round(primary_acceptance_rate, 4),
            "post_retry_acceptance_rate": round(post_retry_acceptance_rate, 4),
            "rejection_reason_counts": rejection_reason_counts,
            "primitive_counts": primitive_counts,
            "landmark_length_counts": landmark_length_counts,
            "min_accepted_episodes": int(min_accepted_episodes),
            "excluded_insufficient_episodes_tasks": excluded_tasks,
            "train_tasks_count": len(train_tasks),
            "val_tasks_count": len(val_tasks),
            "held_out_tasks_count": len(held_out_tasks),
            "split_seed": seed,
            "layer1_raw_flights": layer1_summary,
            "layer2_tensors": layer2_summary,
            "layer3_splits": layer3_summary,
        }
        with open(output_dir / "dataset_summary.json", "w", encoding="utf-8") as f:
            json.dump(summary, f, indent=2)

        print(f"\n  💾 Particiones exportadas en: {output_dir}/")
        print(f"     - {output_dir / 'train_tasks.json'}")
        print(f"     - {output_dir / 'val_tasks.json'}")
        print(f"     - {output_dir / 'held_out_tasks.json'}")
        print(f"     - {output_dir / 'dataset_summary.json'}")

    print("=" * 72)
    return {
        "train_tasks": train_tasks,
        "val_tasks": val_tasks,
        "held_out_tasks": held_out_tasks,
        "excluded_tasks": excluded_tasks,
        "unique_signatures": unique_signatures_count,
        "layer1_raw_flights": layer1_summary,
        "layer2_tensors": layer2_summary,
        "layer3_splits": layer3_summary,
        "tensor_audit_details": tensor_audit_details,
    }


def main():
    parser = argparse.ArgumentParser(
        description="Audita datasets en 3 capas para Instant Policy y genera particiones Train/Val/Held-Out."
    )
    parser.add_argument(
        "--dataset-dir", type=Path, default=Path("datasets"),
        help="Directorio raíz del dataset con las carpetas task_XXXX"
    )
    parser.add_argument(
        "--output-dir", type=Path, default=None,
        help="Directorio donde exportar los archivos JSON de partición (default: <dataset_dir>/splits)"
    )
    parser.add_argument("--train-ratio", type=float, default=0.70, help="Proporción de train (default: 0.70)")
    parser.add_argument("--val-ratio", type=float, default=0.15, help="Proporción de validación (default: 0.15)")
    parser.add_argument("--held-out-ratio", type=float, default=0.15, help="Proporción de held-out (default: 0.15)")
    parser.add_argument(
        "--min-accepted-episodes",
        type=int,
        default=DEFAULT_MIN_ACCEPTED_EPISODES,
        help=f"Mínimo de episodios aceptados por tarea para incluirla en los splits (default: {DEFAULT_MIN_ACCEPTED_EPISODES})",
    )
    parser.add_argument("--seed", type=int, default=42, help="Semilla determinista para barajar las firmas")
    parser.add_argument("--no-export", action="store_true", help="Solo audita en terminal sin escribir archivos")
    args = parser.parse_args()

    if args.output_dir is None:
        args.output_dir = args.dataset_dir / "splits"

    audit_dataset(
        dataset_dir=args.dataset_dir,
        output_dir=args.output_dir,
        train_ratio=args.train_ratio,
        val_ratio=args.val_ratio,
        held_out_ratio=args.held_out_ratio,
        seed=args.seed,
        min_accepted_episodes=args.min_accepted_episodes,
        no_export=args.no_export,
    )


if __name__ == "__main__":
    main()
