"""
uav_ip.policy — Subpaquete aislado de red neuronal, matematicas SE(3)/SVD 4-DoF,
muestreador In-Context y evaluador en lazo cerrado para UAV Instant Policy.

Regla de dependencias:
  - `uav_ip.policy` puede importar del nucleo `uav_ip` (`synthetic_lidar`, `validation_geometry`, `validate_episode`).
  - El nucleo `uav_ip` nunca importa `uav_ip.policy` ni `torch` / `torch_geometric`.
  - Ejecutar siempre con el entorno virtual `~/venv_ip/bin/python3`.
"""

from .in_context_dataset import (
    InContextNavigationDataset,
    TENSOR_CONTRACT_SPEC,
    build_context_pack_from_episodes,
    collate_in_context_batch,
    extract_context_from_episode,
    load_heterodata_episode,
    parse_num_keyframes_arg,
    recompute_aug_horizon_tensors,
    recompute_nominal_horizon_tensors,
    select_geometric_keyframe_indices,
    verify_sample_contract,
)
from .kinematic_rollout_eval import run_kinematic_closed_loop_rollout
from .se3_diffusion_math import (
    DEFAULT_ARM_LENGTH_M,
    DEFAULT_SCALE_FACTOR,
    DEFAULT_Z_HEIGHT_M,
    ActionBounds4DoF,
    action_se3_to_world_waypoints_enu_m,
    align_keypoints_svd_full_se3_numpy,
    align_keypoints_svd_horizontal_flu,
    apply_keypoint_flow_and_svd_4dof,
    build_se3_horizontal_flu_numpy,
    build_se3_horizontal_flu_torch,
    compute_decoupled_keypoint_flow,
    cumulative_to_incremental_se3,
    get_uav_keypoints_body_numpy,
    get_uav_keypoints_body_torch,
    incremental_to_cumulative_se3,
    invert_rigid_se3_numpy,
    invert_rigid_se3_torch,
    se3_to_4dof_vector,
    transform_keypoints,
    vector_4dof_to_se3,
    wrap_angle_numpy,
    wrap_angle_torch,
)
from .simple_validation_net import (
    MiniInstantPolicyNet,
    MiniScenePatchEncoder,
    OracleReplayPolicy,
    PolicyPrediction,
    SinusoidalPosEmb3D,
    subsample_scene_centroids_torch,
)

__all__ = [
    "DEFAULT_SCALE_FACTOR",
    "DEFAULT_ARM_LENGTH_M",
    "DEFAULT_Z_HEIGHT_M",
    "ActionBounds4DoF",
    "get_uav_keypoints_body_numpy",
    "get_uav_keypoints_body_torch",
    "wrap_angle_numpy",
    "wrap_angle_torch",
    "build_se3_horizontal_flu_numpy",
    "build_se3_horizontal_flu_torch",
    "invert_rigid_se3_numpy",
    "invert_rigid_se3_torch",
    "cumulative_to_incremental_se3",
    "incremental_to_cumulative_se3",
    "se3_to_4dof_vector",
    "vector_4dof_to_se3",
    "transform_keypoints",
    "compute_decoupled_keypoint_flow",
    "apply_keypoint_flow_and_svd_4dof",
    "align_keypoints_svd_horizontal_flu",
    "align_keypoints_svd_full_se3_numpy",
    "action_se3_to_world_waypoints_enu_m",
    "TENSOR_CONTRACT_SPEC",
    "load_heterodata_episode",
    "parse_num_keyframes_arg",
    "select_geometric_keyframe_indices",
    "recompute_nominal_horizon_tensors",
    "recompute_aug_horizon_tensors",
    "extract_context_from_episode",
    "build_context_pack_from_episodes",
    "collate_in_context_batch",
    "verify_sample_contract",
    "InContextNavigationDataset",
    "PolicyPrediction",
    "subsample_scene_centroids_torch",
    "SinusoidalPosEmb3D",
    "MiniScenePatchEncoder",
    "MiniInstantPolicyNet",
    "OracleReplayPolicy",
    "run_kinematic_closed_loop_rollout",
]
