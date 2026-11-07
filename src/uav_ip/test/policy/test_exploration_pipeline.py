"""
test_exploration_pipeline.py — Pruebas unitarias y de integracion del subpaquete `uav_ip.policy`
y del soporte `--rtf` en `world_generator.py`.
"""

import ast
from pathlib import Path
import sys
import xml.etree.ElementTree as ET

import numpy as np
import pytest
import yaml

PKG_ROOT = Path(__file__).resolve().parent.parent.parent
REPO_ROOT = PKG_ROOT.parent.parent
SCRIPTS_DIR = PKG_ROOT / "scripts"
if str(PKG_ROOT) not in sys.path:
    sys.path.insert(0, str(PKG_ROOT))
if str(SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_DIR))

torch = pytest.importorskip("torch")

from uav_ip.policy import (  # noqa: E402
    DEFAULT_SCALE_FACTOR,
    ActionBounds4DoF,
    InContextNavigationDataset,
    MiniInstantPolicyNet,
    OracleReplayPolicy,
    action_se3_to_world_waypoints_enu_m,
    align_keypoints_svd_horizontal_flu,
    apply_keypoint_flow_and_svd_4dof,
    build_context_pack_from_episodes,
    build_se3_horizontal_flu_numpy,
    build_se3_horizontal_flu_torch,
    collate_in_context_batch,
    compute_decoupled_keypoint_flow,
    cumulative_to_incremental_se3,
    get_uav_keypoints_body_numpy,
    get_uav_keypoints_body_torch,
    incremental_to_cumulative_se3,
    load_heterodata_episode,
    run_kinematic_closed_loop_rollout,
    se3_to_4dof_vector,
    transform_keypoints,
    vector_4dof_to_se3,
    verify_sample_contract,
)


def test_unidirectional_dependency_rule_core_never_imports_policy_or_torch():
    """
    Verifica la regla arquitectonica de AGENT.md:
      1. Ningun archivo del nucleo `src/uav_ip/uav_ip/*.py` (fuera de `policy/`) importa
         `policy` ni `uav_ip.policy` en ningun lugar.
      2. Ningun archivo del nucleo tiene `import torch` o `import torch_geometric` incondicional
         a nivel de modulo (`tree.body`), garantizando que los nodos ROS 2 del sistema arranquen
         sin tener PyTorch instalado.
    """
    core_dir = PKG_ROOT / "uav_ip"
    forbidden_top_level = {"torch", "torch_geometric", "policy"}

    for py_file in sorted(core_dir.glob("*.py")):
        tree = ast.parse(py_file.read_text(encoding="utf-8"), filename=str(py_file))
        # 1. Prohibicion total de importar `policy` en cualquier parte del nucleo
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                for alias in node.names:
                    parts = alias.name.split(".")
                    assert "policy" not in parts, (
                        f"El modulo del nucleo {py_file.name} importa '{alias.name}', "
                        f"violando la regla unidireccional hacia uav_ip/policy/."
                    )
            elif isinstance(node, ast.ImportFrom):
                parts = (node.module or "").split(".")
                assert "policy" not in parts, (
                    f"El modulo del nucleo {py_file.name} importa desde '{node.module}', "
                    f"violando la regla unidireccional hacia uav_ip/policy/."
                )

        # 2. Prohibicion de import incondicional a nivel superior de torch / torch_geometric
        for stmt in tree.body:
            if isinstance(stmt, ast.Import):
                for alias in stmt.names:
                    root_mod = alias.name.split(".")[0]
                    assert root_mod not in forbidden_top_level, (
                        f"El modulo del nucleo {py_file.name} importa '{alias.name}' a nivel superior."
                    )
            elif isinstance(stmt, ast.ImportFrom):
                root_mod = (stmt.module or "").split(".")[0]
                assert root_mod not in forbidden_top_level, (
                    f"El modulo del nucleo {py_file.name} importa desde '{stmt.module}' a nivel superior."
                )


def test_se3_cumulative_incremental_and_4dof_roundtrip():
    """Verifica exactitud numerica en conversion acumulada <-> incremental y normalizacion 4-DoF."""
    rng = np.random.default_rng(123)
    dpos = rng.uniform(-0.04, 0.04, size=(5, 8, 3)).astype(np.float32)
    dyaw = rng.uniform(-0.35, 0.35, size=(5, 8)).astype(np.float32)

    delta_np = build_se3_horizontal_flu_numpy(dpos, dyaw)
    cum_np = incremental_to_cumulative_se3(delta_np)
    rec_delta_np = cumulative_to_incremental_se3(cum_np)
    np.testing.assert_allclose(delta_np, rec_delta_np, atol=1e-5)

    # Version PyTorch
    delta_th = torch.from_numpy(delta_np)
    cum_th = incremental_to_cumulative_se3(delta_th)
    rec_delta_th = cumulative_to_incremental_se3(cum_th)
    assert torch.allclose(delta_th, rec_delta_th, atol=1e-5)

    # Normalizacion 4-DoF
    bounds = ActionBounds4DoF()
    vec4 = se3_to_4dof_vector(delta_th)
    normed = bounds.normalize(vec4, clip=True)
    assert float(normed.abs().max().item()) <= 1.0 + 1e-6
    unnormed = bounds.unnormalize(normed)
    rec_se3 = vector_4dof_to_se3(unnormed)
    assert torch.allclose(delta_th, rec_se3, atol=1e-5)


def test_decoupled_keypoint_flow_and_4dof_svd_roundtrip_and_gravity_alignment():
    """
    Verifica que el flujo cartesiano desacoplado [Δp_t, Δp_r] sobre los 6 keypoints
    reconstruye exactamente cualquier transformacion SE(2) x R ⊂ SE(3) en K=1 y K=4 pasos,
    y que ante ruido 3D arbitrario mantiene roll=0, pitch=0 exactamente.
    """
    rng = np.random.default_rng(77)
    dpos = rng.uniform(-0.05, 0.05, size=(10, 8, 3)).astype(np.float32)
    dyaw = rng.uniform(-0.50, 0.50, size=(10, 8)).astype(np.float32)
    target_se3 = build_se3_horizontal_flu_torch(
        torch.from_numpy(dpos), torch.from_numpy(dyaw)
    )
    kp = get_uav_keypoints_body_torch(scale_factor=DEFAULT_SCALE_FACTOR)

    # K = 1 paso desde identidad
    curr_1 = torch.eye(4, dtype=torch.float32).view(1, 1, 4, 4).expand_as(target_se3).clone()
    flow_1 = compute_decoupled_keypoint_flow(curr_1, target_se3, keypoints_body=kp)
    rec_1 = apply_keypoint_flow_and_svd_4dof(curr_1, flow_1, keypoints_body=kp, step_scale=1.0)
    assert torch.allclose(target_se3, rec_1, atol=1e-5)

    # K = 4 pasos DDIM desde identidad
    curr_4 = curr_1.clone()
    for step_k in range(4, 0, -1):
        fl_k = compute_decoupled_keypoint_flow(curr_4, target_se3, keypoints_body=kp)
        curr_4 = apply_keypoint_flow_and_svd_4dof(
            curr_4, fl_k, keypoints_body=kp, step_scale=1.0 / float(step_k)
        )
    assert torch.allclose(target_se3, curr_4, atol=1e-5)

    # Perturbacion ruidosa 3D en keypoints destino -> SVD 4-DoF debe mantener R[2,2]==1 y det(R)==+1
    kp_np = get_uav_keypoints_body_numpy()
    noisy_dst = transform_keypoints(target_se3.numpy(), keypoints_body=kp_np) + rng.normal(
        0.0, 0.005, size=(10, 8, 6, 3)
    ).astype(np.float32)
    T_aligned = align_keypoints_svd_horizontal_flu(kp_np, noisy_dst)
    np.testing.assert_allclose(T_aligned[..., 2, 0], 0.0, atol=1e-6)
    np.testing.assert_allclose(T_aligned[..., 2, 1], 0.0, atol=1e-6)
    np.testing.assert_allclose(T_aligned[..., 2, 2], 1.0, atol=1e-6)
    dets = np.linalg.det(T_aligned[..., :3, :3])
    np.testing.assert_allclose(dets, 1.0, atol=1e-5)


def test_in_context_dataset_and_verify_sample_contract():
    """Verifica la carga de muestras nominales y Tube Augmentation y sus contratos E/S."""
    kin_dir = REPO_ROOT / "datasets" / "kinematic"
    ds = InContextNavigationDataset(
        dataset_dir=kin_dir,
        task_ids=["task_0000", "task_0006"],
        num_context_demos=2,
        include_aug=True,
    )
    assert len(ds) > 100

    # Verificar muestra nominal y muestra aug
    nominal_sample = next(ds[i] for i in range(len(ds)) if ds.samples_index[i][2] == "nominal")
    aug_sample = next(ds[i] for i in range(len(ds)) if ds.samples_index[i][2] == "aug")

    res_nom = verify_sample_contract(nominal_sample, expected_num_demos=2)
    res_aug = verify_sample_contract(aug_sample, expected_num_demos=2)
    assert res_nom["valid"] is True
    assert res_aug["valid"] is True


def test_oracle_and_mini_instant_policy_net_contracts_and_closed_loop():
    """
    Verifica que:
      1. `OracleReplayPolicy` reconstruye la trayectoria en lazo cerrado aprobando `validate_episode_data()`.
      2. `MiniInstantPolicyNet` ejecuta `compute_loss` y `predict_actions` respetando todas las formas E/S.
    """
    kin_dir = REPO_ROOT / "datasets" / "kinematic"
    t_dir = kin_dir / "task_0006"
    ep_paths = sorted(t_dir.glob("episode_*/heterodata.pt"))
    wp_cfg = yaml.safe_load((ep_paths[0].parent / "waypoints.yaml").read_text(encoding="utf-8"))
    ep0_data = load_heterodata_episode(ep_paths[0])
    ref_pos_m = (ep0_data["agent"].pos_enu.numpy() / DEFAULT_SCALE_FACTOR).astype(np.float64)

    ctx_pack = build_context_pack_from_episodes(ep_paths[1:3], num_keyframes=10)
    oracle = OracleReplayPolicy(reference_episode=ep0_data)

    rollout = run_kinematic_closed_loop_rollout(
        policy=oracle,
        waypoints_cfg=wp_cfg,
        context_pack=ctx_pack,
        execution_horizon=4,
        num_ddim_steps=1,
        max_steps_10hz=len(ref_pos_m) + 5,
        initial_pos_enu_m=ref_pos_m[0],
        initial_yaw_enu_rad=float(ep0_data["agent"].yaw_enu[0, 0].item()),
        initial_vel_flu_mps=ep0_data["agent"].vel_flu[0].numpy(),
        reference_pos_enu_m=ref_pos_m,
    )
    assert rollout["validation_status"] == "accepted", f"Oracle rollout rejected: {rollout['rejection_reasons']}"
    assert rollout["ade_m"] < 0.05

    # Probar MiniInstantPolicyNet forward + backward + predict_actions
    ds = InContextNavigationDataset(kin_dir, task_ids=["task_0006"], num_context_demos=2)
    from torch.utils.data import DataLoader

    batch = next(iter(DataLoader(ds, batch_size=4, shuffle=False)))
    net = MiniInstantPolicyNet(d_model=32, num_heads=4, num_centroids=16, num_diffusion_steps=2)
    loss_dict = net.compute_loss(batch)
    assert torch.isfinite(loss_dict["loss"])
    loss_dict["loss"].backward()

    pred = net.predict_actions(ds[0], num_ddim_steps=2)
    assert pred.pred_action_se3.shape == (8, 4, 4)
    assert pred.pred_delta_se3.shape == (8, 4, 4)
    assert pred.pred_flight_state.shape == (8, 1)
    assert pred.pred_world_waypoints_enu_m.shape == (8, 4)
    assert pred.pred_keypoint_flow.shape == (8, 6, 6)
    assert np.all(np.isfinite(pred.pred_world_waypoints_enu_m))


def test_world_generator_rtf_scales_physics_update_rate():
    """Verifica que `modify_sdf(..., rtf=0.25)` ajusta `<real_time_factor>` y `<real_time_update_rate>`."""
    from world_generator import modify_sdf

    template_sdf = PKG_ROOT / "worlds" / "circuito_aros_01.sdf"
    hoops = [{"id": "landmark_1", "type": "square_gate", "x": 5.0, "y": 0.0, "z": 1.5, "yaw": 0.0}]
    landing = {"x": 10.0, "y": 0.0, "z": 0.05}

    tree = modify_sdf(template_sdf, "test_rtf_world", hoops, landing, rtf=0.25)
    world_elem = tree.getroot().find("world")
    physics_elem = world_elem.find("physics")
    assert physics_elem is not None
    assert float(physics_elem.find("max_step_size").text) == pytest.approx(0.004)
    assert float(physics_elem.find("real_time_factor").text) == pytest.approx(0.25)
    assert float(physics_elem.find("real_time_update_rate").text) == pytest.approx(62.5)


def test_variable_keyframes_fixed_and_adaptive_masking():
    """
    Verifica que `num_keyframes` soporta tanto enteros arbitrarios (ej. L=16) como modo
    `'adaptive'` proporcional a la longitud de arco, colacionando lotes multi-tarea con
    `context_keyframe_mask` sin alterar las dimensiones de los pesos de `MiniInstantPolicyNet`.
    """
    kin_dir = REPO_ROOT / "datasets" / "kinematic"
    ds_fixed16 = InContextNavigationDataset(
        dataset_dir=kin_dir,
        task_ids=["task_0006"],
        num_context_demos=2,
        num_keyframes=16,
    )
    res_16 = verify_sample_contract(ds_fixed16[0], expected_num_demos=2, expected_num_keyframes=16)
    assert res_16["valid"] is True, res_16["errors"]

    ds_adapt = InContextNavigationDataset(
        dataset_dir=kin_dir,
        task_ids=["task_0000", "task_0006", "task_0010"],
        num_context_demos=2,
        num_keyframes="adaptive",
        adaptive_arc_m=1.2,
        include_aug=True,
    )
    # Tomar una muestra por tarea y forzar colacion con longitudes L distintas
    per_task_samples = []
    seen = set()
    for idx, (tid, _ep, _kind, _step) in enumerate(ds_adapt.samples_index):
        if tid not in seen:
            s_i = ds_adapt[idx]
            chk = verify_sample_contract(s_i, expected_num_demos=2, expected_num_keyframes="adaptive")
            assert chk["valid"] is True, chk["errors"]
            per_task_samples.append(s_i)
            seen.add(tid)

    Ls = [int(s["context_scene_pos"].shape[1]) for s in per_task_samples]
    batch = collate_in_context_batch(per_task_samples)
    assert batch["context_scene_pos"].shape[:3] == (len(per_task_samples), 2, max(Ls))
    assert batch["context_keyframe_mask"].shape == (len(per_task_samples), 2, max(Ls))
    for b_idx, orig_L in enumerate(Ls):
        assert int(batch["context_keyframe_mask"][b_idx, 0].sum().item()) == orig_L
        if orig_L < max(Ls):
            assert not bool(batch["context_keyframe_mask"][b_idx, 0, -1].item())

    net = MiniInstantPolicyNet(d_model=32, num_heads=4, num_centroids=16, num_diffusion_steps=2)
    loss_dict = net.compute_loss(batch)
    assert torch.isfinite(loss_dict["loss"])
    loss_dict["loss"].backward()

    pred = net.predict_actions(per_task_samples[0], num_ddim_steps=2)
    assert pred.pred_action_se3.shape == (8, 4, 4)
    assert np.all(np.isfinite(pred.pred_world_waypoints_enu_m))


def test_variable_horizon_stride_and_execution_horizon_above_8():
    """
    Verifica la reconstruccion al vuelo de `horizon=12`, `action_stride=2`, `num_context_demos=3`
    y ejecucion en lazo cerrado con `execution_horizon=9` (> 8) sin regenerar `heterodata.pt`.
    """
    kin_dir = REPO_ROOT / "datasets" / "kinematic"
    ds = InContextNavigationDataset(
        dataset_dir=kin_dir,
        task_ids=["task_0006"],
        num_context_demos=3,
        num_keyframes=12,
        horizon=12,
        action_stride=2,
        include_aug=True,
    )
    nom_s = next(ds[i] for i in range(len(ds)) if ds.samples_index[i][2] == "nominal")
    aug_s = next(ds[i] for i in range(len(ds)) if ds.samples_index[i][2] == "aug")
    assert verify_sample_contract(nom_s, expected_num_demos=3, expected_num_keyframes=12, expected_horizon=12)["valid"]
    assert verify_sample_contract(aug_s, expected_num_demos=3, expected_num_keyframes=12, expected_horizon=12)["valid"]

    batch = collate_in_context_batch([nom_s, aug_s])
    net = MiniInstantPolicyNet(
        d_model=32,
        num_heads=4,
        num_centroids=16,
        horizon=12,
        action_stride=2,
        num_diffusion_steps=2,
    )
    loss_d = net.compute_loss(batch)
    assert torch.isfinite(loss_d["loss"])
    loss_d["loss"].backward()
    pred = net.predict_actions(nom_s, num_ddim_steps=2)
    assert pred.pred_action_se3.shape == (12, 4, 4)
    assert pred.pred_world_waypoints_enu_m.shape == (12, 4)

    # Verificar OracleReplayPolicy en lazo cerrado con horizon=12, action_stride=2 y execution_horizon=9 (> 8)
    t_dir = kin_dir / "task_0006"
    ep_paths = sorted(t_dir.glob("episode_*/heterodata.pt"))
    wp_cfg = yaml.safe_load((ep_paths[0].parent / "waypoints.yaml").read_text(encoding="utf-8"))
    ep0_data = load_heterodata_episode(ep_paths[0])
    ref_pos_m = (ep0_data["agent"].pos_enu.numpy() / DEFAULT_SCALE_FACTOR).astype(np.float64)
    ctx_pack = build_context_pack_from_episodes(ep_paths[1:4], num_keyframes=12, horizon=12, action_stride=2)

    oracle = OracleReplayPolicy(reference_episode=ep0_data, horizon=12, action_stride=2)
    ro = run_kinematic_closed_loop_rollout(
        policy=oracle,
        waypoints_cfg=wp_cfg,
        context_pack=ctx_pack,
        execution_horizon=9,
        num_ddim_steps=1,
        max_steps_10hz=len(ref_pos_m) + 20,
        action_stride=2,
        initial_pos_enu_m=ref_pos_m[0],
        initial_yaw_enu_rad=float(ep0_data["agent"].yaw_enu[0, 0].item()),
        initial_vel_flu_mps=ep0_data["agent"].vel_flu[0].numpy(),
        reference_pos_enu_m=ref_pos_m,
    )
    assert ro["validation_status"] == "accepted", f"Strided Oracle rollout rejected: {ro['rejection_reasons']}"
    assert ro["ade_m"] < 0.05

