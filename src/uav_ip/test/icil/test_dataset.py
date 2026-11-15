"""Pruebas unitarias de Etapa 2: splits JSON, keyframes, horizonte incremental y estadisticas."""

from pathlib import Path
import sys

import pytest

PKG_ROOT = Path(__file__).resolve().parent.parent.parent
REPO_ROOT = PKG_ROOT.parent.parent
if str(PKG_ROOT) not in sys.path:
    sys.path.insert(0, str(PKG_ROOT))

torch = pytest.importorskip("torch")
from torch.utils.data import DataLoader  # noqa: E402

from uav_ip.icil.dataset import (  # noqa: E402
    InContextDataset,
    collate_icil_batch,
    compute_action_statistics,
    load_split_tasks,
    select_keyframe_indices,
)


def test_splits_strict_and_disjoint(tmp_path: Path) -> None:
    """Verifica que los splits sean no vacios y disjuntos, que 'test' equivalga a 'held_out' y que falle si faltan."""
    for ds_name in ("kinematic", "sitl"):
        ds_root = REPO_ROOT / "datasets" / ds_name
        if not ds_root.is_dir():
            pytest.skip(f"No existe {ds_root}")

        train_tasks = set(load_split_tasks(ds_root, "train"))
        val_tasks = set(load_split_tasks(ds_root, "val"))
        held_out_tasks = set(load_split_tasks(ds_root, "held_out"))
        test_alias_tasks = set(load_split_tasks(ds_root, "test"))

        assert len(train_tasks) > 0
        assert len(val_tasks) > 0
        assert len(held_out_tasks) > 0
        assert held_out_tasks == test_alias_tasks
        assert train_tasks.isdisjoint(val_tasks)
        assert train_tasks.isdisjoint(held_out_tasks)
        assert val_tasks.isdisjoint(held_out_tasks)

    with pytest.raises(FileNotFoundError):
        load_split_tasks(tmp_path, "train")


def test_keyframe_selector_fixed_and_adaptive() -> None:
    """Valida seleccion fija y adaptativa de keyframes: formas, extremos, aterrizaje, orden temporal y mascara."""
    t_steps = torch.linspace(0.0, 2.0 * torch.pi, 120)
    pos = torch.stack([3.0 * torch.cos(t_steps), 3.0 * torch.sin(t_steps), 1.5 + 0.2 * t_steps], dim=-1)
    yaw = torch.atan2(torch.cos(t_steps), -torch.sin(t_steps)).unsqueeze(-1)
    flight_state = torch.ones((120, 1), dtype=torch.float32)
    flight_state[100:] = 0.0

    idx_10, mask_10 = select_keyframe_indices(pos, yaw, flight_state, num_keyframes=10)
    assert idx_10.shape == (10,)
    assert mask_10.shape == (10,)
    assert bool(mask_10.all().item())
    assert int(idx_10[0].item()) == 0
    assert int(idx_10[-1].item()) == 119
    assert 100 in idx_10.tolist()
    assert torch.all(idx_10[1:] >= idx_10[:-1])

    idx_adap, mask_adap = select_keyframe_indices(
        pos, yaw, flight_state, num_keyframes="adaptive", max_keyframes=16, target_arc_step_m=3.5
    )
    assert idx_adap.shape == (16,)
    assert mask_adap.shape == (16,)
    k_valid = int(mask_adap.sum().item())
    assert 4 <= k_valid <= 16
    assert bool(mask_adap[:k_valid].all().item())
    assert not bool(mask_adap[k_valid:].any().item())
    assert torch.all(idx_adap[1:] >= idx_adap[:-1])


def test_dataset_and_collate_batch_shapes() -> None:
    """Comprueba formas, dtypes, finitud y soporte de modo adaptativo/stride en InContextDataset y collate_icil_batch."""
    ds_root = REPO_ROOT / "datasets" / "sitl"
    if not ds_root.is_dir():
        pytest.skip(f"No existe {ds_root}")

    ds_fixed = InContextDataset(
        dataset_root=ds_root,
        split="train",
        num_demos=2,
        num_keyframes=10,
        horizon=8,
        action_stride=1,
        include_aug=True,
        max_tasks=2,
    )
    loader = DataLoader(ds_fixed, batch_size=4, shuffle=True, collate_fn=collate_icil_batch)
    batch = next(iter(loader))

    assert batch["live_cloud"].shape == (4, 2048, 3)
    assert batch["live_vel_flu"].shape == (4, 3)
    assert batch["live_flight_state"].shape == (4, 1)
    assert batch["demo_clouds"].shape == (4, 2, 10, 2048, 3)
    assert batch["demo_poses_se3"].shape == (4, 2, 10, 4, 4)
    assert batch["demo_vel_flu"].shape == (4, 2, 10, 3)
    assert batch["demo_flight_state"].shape == (4, 2, 10, 1)
    assert batch["demo_mask"].shape == (4, 2, 10)
    assert batch["demo_mask"].dtype == torch.bool
    assert bool(batch["demo_mask"].all().item())
    assert batch["gt_delta_se3"].shape == (4, 8, 4, 4)
    assert batch["gt_flight_state"].shape == (4, 8, 1)
    assert batch["action_valid_mask"].shape == (4, 8)
    assert batch["action_valid_mask"].dtype == torch.bool

    for name, tensor in batch.items():
        if tensor.is_floating_point():
            assert torch.isfinite(tensor).all(), f"NaN/Inf detectado en {name}"

    ds_grid = InContextDataset(
        dataset_root=ds_root,
        split="val",
        num_demos=3,
        num_keyframes="adaptive",
        max_keyframes=14,
        horizon=6,
        action_stride=2,
        include_aug=False,
        max_tasks=2,
    )
    sample = ds_grid[0]
    assert sample["demo_clouds"].shape == (3, 14, 2048, 3)
    assert sample["demo_mask"].shape == (3, 14)
    assert sample["gt_delta_se3"].shape == (6, 4, 4)
    assert sample["gt_flight_state"].shape == (6, 1)
    assert sample["action_valid_mask"].shape == (6,)


def test_action_statistics_kinematic_and_sitl() -> None:
    """Verifica que las estadisticas p99 y max de acciones en sitl y kinematic esten dentro de rangos fisicos validos."""
    for ds_name in ("sitl", "kinematic"):
        ds_root = REPO_ROOT / "datasets" / ds_name
        if not ds_root.is_dir():
            pytest.skip(f"No existe {ds_root}")

        ds = InContextDataset(
            dataset_root=ds_root,
            split="train",
            num_demos=2,
            num_keyframes=10,
            horizon=8,
            action_stride=1,
            include_aug=True,
            max_cached_episodes=16,
        )
        stats = compute_action_statistics(ds)
        print(f"\n[Action Stats - {ds_name} (train, stride=1, T=8)]")
        for k, v in stats.items():
            print(f"  {k}: {v:.6f}")

        assert stats["num_valid_steps"] > 0
        assert 0.01 < stats["trans_norm_p99_m"] <= stats["trans_norm_max_m"] < 2.0
        assert 0.001 < stats["yaw_abs_p99_rad"] <= stats["yaw_abs_max_rad"] <= torch.pi