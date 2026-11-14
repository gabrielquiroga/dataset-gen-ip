"""Pruebas unitarias de Etapa 1: geometria SE(3) 4-DoF, flujo desacoplado y SVD."""

from pathlib import Path
import sys

import pytest

PKG_ROOT = Path(__file__).resolve().parent.parent.parent
REPO_ROOT = PKG_ROOT.parent.parent
if str(PKG_ROOT) not in sys.path:
    sys.path.insert(0, str(PKG_ROOT))

torch = pytest.importorskip("torch")

from uav_ip.icil.geometry import (  # noqa: E402
    DEFAULT_ARM_LENGTH_M,
    DEFAULT_SCALE_FACTOR,
    DEFAULT_Z_HEIGHT_M,
    align_keypoints_svd_4dof,
    apply_flow_and_svd,
    compute_keypoint_flow,
    cumulative_to_incremental,
    get_uav_keypoints,
    incremental_to_cumulative,
    invert_se3,
    se3_to_vec4,
    transform_keypoints,
    vec4_to_se3,
    wrap_angle,
)


def test_canonical_keypoints_shape_and_values() -> None:
    kp = get_uav_keypoints()
    assert kp.shape == (6, 3)
    arm = DEFAULT_ARM_LENGTH_M * DEFAULT_SCALE_FACTOR
    z_up = DEFAULT_Z_HEIGHT_M * DEFAULT_SCALE_FACTOR
    assert torch.allclose(kp[0], torch.zeros(3))
    assert torch.isclose(kp[1, 0], torch.tensor(arm))
    assert torch.isclose(kp[2, 0], torch.tensor(-arm))
    assert torch.isclose(kp[3, 1], torch.tensor(arm))
    assert torch.isclose(kp[4, 1], torch.tensor(-arm))
    assert torch.isclose(kp[5, 2], torch.tensor(z_up))


def test_vec4_se3_and_inverse_roundtrip() -> None:
    torch.manual_seed(42)
    vec = torch.empty((16, 8, 4)).uniform_(-0.5, 0.5)
    vec[..., 3] = torch.empty((16, 8)).uniform_(-torch.pi, torch.pi)
    T = vec4_to_se3(vec)
    vec_rec = se3_to_vec4(T)

    assert (vec[..., :3] - vec_rec[..., :3]).abs().max().item() < 1e-6
    assert wrap_angle(vec[..., 3] - vec_rec[..., 3]).abs().max().item() < 1e-6
    assert (torch.matmul(T, invert_se3(T)) - torch.eye(4)).abs().max().item() < 1e-6


def test_cumulative_incremental_roundtrip() -> None:
    pt_file = REPO_ROOT / "datasets/kinematic/task_0000/episode_00/heterodata.pt"
    if pt_file.is_file():
        data = torch.load(pt_file, map_location="cpu", weights_only=False)
        action_se3 = data["agent"].action_se3.to(torch.float32)
    else:
        torch.manual_seed(7)
        action_se3 = vec4_to_se3(torch.randn(50, 8, 4) * 0.05)

    deltas = cumulative_to_incremental(action_se3)
    reconstructed = incremental_to_cumulative(deltas)
    err_cum_m = (reconstructed[..., :3, 3] - action_se3[..., :3, 3]).abs().max().item() / DEFAULT_SCALE_FACTOR
    assert err_cum_m < 1e-5


def test_decoupled_flow_and_svd_4dof() -> None:
    torch.manual_seed(123)
    kp = get_uav_keypoints()
    T_curr = vec4_to_se3(torch.randn(32, 8, 4) * torch.tensor([0.3, 0.3, 0.15, 2.5]))
    T_target = vec4_to_se3(torch.randn(32, 8, 4) * torch.tensor([0.3, 0.3, 0.15, 2.5]))

    flow = compute_keypoint_flow(T_curr, T_target, kp)
    assert flow.shape == (32, 8, 6, 6)
    T_svd = apply_flow_and_svd(T_curr, flow, kp, step_scale=1.0)

    v_orig = se3_to_vec4(T_target)
    v_svd = se3_to_vec4(T_svd)
    err_pos_m = (v_svd[..., :3] - v_orig[..., :3]).abs().max().item() / DEFAULT_SCALE_FACTOR
    err_yaw_rad = wrap_angle(v_svd[..., 3] - v_orig[..., 3]).abs().max().item()
    assert err_pos_m < 1e-5
    assert err_yaw_rad < 1e-5


def test_svd_4dof_se3_equivariance() -> None:
    torch.manual_seed(99)
    kp = get_uav_keypoints()
    T_0 = vec4_to_se3(torch.randn(16, 4) * torch.tensor([0.2, 0.2, 0.1, 2.0]))
    T_1 = vec4_to_se3(torch.randn(16, 4) * torch.tensor([0.2, 0.2, 0.1, 2.0]))
    p0 = transform_keypoints(T_0, kp)
    p1 = transform_keypoints(T_1, kp)

    T_corr = align_keypoints_svd_4dof(p0, p1)
    G = vec4_to_se3(torch.tensor([1.2, -0.7, 0.3, 2.1]))
    T_corr_G = align_keypoints_svd_4dof(transform_keypoints(G, p0), transform_keypoints(G, p1))
    expected = torch.matmul(G, torch.matmul(T_corr, invert_se3(G)))
    assert (T_corr_G - expected).abs().max().item() < 1e-5