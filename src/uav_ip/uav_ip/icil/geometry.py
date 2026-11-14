"""
geometry.py — Operaciones geometricas SE(3) en marco Horizontal FLU (4-DoF),
transformacion de los 6 keypoints del UAV, flujo desacoplado [dp_t, dp_r] y SVD 4-DoF.
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional

import torch

DEFAULT_SCALE_FACTOR: float = 0.1
DEFAULT_ARM_LENGTH_M: float = 0.25
DEFAULT_Z_HEIGHT_M: float = 0.15


def get_uav_keypoints(
    scale_factor: float = DEFAULT_SCALE_FACTOR,
    arm_length: float = DEFAULT_ARM_LENGTH_M,
    z_height: float = DEFAULT_Z_HEIGHT_M,
    device: Optional[torch.device] = None,
    dtype: torch.dtype = torch.float32,
) -> torch.Tensor:
    """
    Retorna los 6 keypoints canonicos del UAV en marco local Horizontal FLU (escala s=0.1).
    Forma: (6, 3) -> [Centro, Proa +X, Popa -X, Babor +Y, Estribor -Y, Cenit +Z].
    """
    kp_m = torch.tensor(
        [
            [0.0, 0.0, 0.0],
            [arm_length, 0.0, 0.0],
            [-arm_length, 0.0, 0.0],
            [0.0, arm_length, 0.0],
            [0.0, -arm_length, 0.0],
            [0.0, 0.0, z_height],
        ],
        dtype=dtype,
        device=device,
    )
    return kp_m * float(scale_factor)


def wrap_angle(angle: torch.Tensor) -> torch.Tensor:
    """Envuelve angulos en radianes al intervalo [-pi, pi)."""
    return torch.atan2(torch.sin(angle), torch.cos(angle))


def vec4_to_se3(vec4: torch.Tensor) -> torch.Tensor:
    """
    Convierte vectores 4-DoF [dx, dy, dz, dyaw] de forma (..., 4)
    en matrices homogeneas SE(3) Horizontal FLU (roll=0, pitch=0) de forma (..., 4, 4).
    """
    dx, dy, dz, yaw = vec4.unbind(dim=-1)
    c = torch.cos(yaw)
    s = torch.sin(yaw)
    zeros = torch.zeros_like(c)
    ones = torch.ones_like(c)

    row0 = torch.stack([c, -s, zeros, dx], dim=-1)
    row1 = torch.stack([s, c, zeros, dy], dim=-1)
    row2 = torch.stack([zeros, zeros, ones, dz], dim=-1)
    row3 = torch.stack([zeros, zeros, zeros, ones], dim=-1)
    return torch.stack([row0, row1, row2, row3], dim=-2)


def se3_to_vec4(T: torch.Tensor) -> torch.Tensor:
    """
    Extrae el vector 4-DoF [dx, dy, dz, dyaw] de forma (..., 4)
    a partir de matrices homogeneas SE(3) Horizontal FLU de forma (..., 4, 4).
    """
    dpos = T[..., :3, 3]
    dyaw = torch.atan2(T[..., 1, 0], T[..., 0, 0]).unsqueeze(-1)
    return torch.cat([dpos, dyaw], dim=-1)


def invert_se3(T: torch.Tensor) -> torch.Tensor:
    """Inversa analitica exacta de transformaciones rigidas SE(3) de forma (..., 4, 4)."""
    R = T[..., :3, :3]
    t = T[..., :3, 3:4]
    R_T = R.transpose(-1, -2)
    t_inv = -(R_T @ t)

    top = torch.cat([R_T, t_inv], dim=-1)
    bot = torch.zeros(T.shape[:-2] + (1, 4), dtype=T.dtype, device=T.device)
    bot[..., 0, 3] = 1.0
    return torch.cat([top, bot], dim=-2)


def cumulative_to_incremental(action_se3: torch.Tensor) -> torch.Tensor:
    """
    Convierte poses relativas acumuladas T_{EA}^{(t, 1..H)} de forma (..., H, 4, 4)
    en saltos paso a paso dT_k = (T_{EA}^{k-1})^{-1} @ T_{EA}^k de forma (..., H, 4, 4).
    """
    H = action_se3.shape[-3]
    if H <= 1:
        return action_se3.clone()
    prev_inv = invert_se3(action_se3[..., :-1, :, :])
    deltas_rest = prev_inv @ action_se3[..., 1:, :, :]
    return torch.cat([action_se3[..., :1, :, :], deltas_rest], dim=-3)


def incremental_to_cumulative(delta_se3: torch.Tensor) -> torch.Tensor:
    """
    Reconstruye poses acumuladas T_{EA}^{(t, 1..H)} de forma (..., H, 4, 4)
    multiplicando los saltos paso a paso: T_{EA}^{(t, k)} = dT_1 @ dT_2 @ ... @ dT_k.
    """
    H = delta_se3.shape[-3]
    curr = delta_se3[..., 0, :, :]
    out = [curr]
    for k in range(1, H):
        curr = curr @ delta_se3[..., k, :, :]
        out.append(curr)
    return torch.stack(out, dim=-3)


def transform_keypoints(
    T: torch.Tensor,
    keypoints: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """
    Transforma los 6 keypoints (6, 3) usando matrices T de forma (..., 4, 4).
    Retorna coordenadas 3D de forma (..., 6, 3): P_i = R @ p_g^i + t.
    """
    if keypoints is None:
        keypoints = get_uav_keypoints(device=T.device, dtype=T.dtype)
    R = T[..., :3, :3]
    t = T[..., :3, 3].unsqueeze(-2)
    return torch.matmul(keypoints, R.transpose(-1, -2)) + t


def compute_keypoint_flow(
    T_curr: torch.Tensor,
    T_target: torch.Tensor,
    keypoints: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """
    Calcula el flujo cartesiano desacoplado [dp_t, dp_r] de forma (..., 6, 6)
    entre la pose actual T_curr (..., 4, 4) y la pose objetivo T_target (..., 4, 4)
    (Instant Policy, Sec. 3.3):
      - dp_t = t_target - t_curr              (traslacion del CoM, igual en los 6 puntos)
      - dp_r = (R_target - R_curr) @ p_g^i    (rotacion pura del brazo alrededor del CoM)
    """
    if keypoints is None:
        keypoints = get_uav_keypoints(device=T_curr.device, dtype=T_curr.dtype)
    n_kp = keypoints.shape[0]
    dt = (T_target[..., :3, 3] - T_curr[..., :3, 3]).unsqueeze(-2).expand(T_curr.shape[:-2] + (n_kp, 3))
    dR = T_target[..., :3, :3] - T_curr[..., :3, :3]
    dr = torch.matmul(keypoints, dR.transpose(-1, -2))
    return torch.cat([dt, dr], dim=-1)


def align_keypoints_svd_4dof(
    p_src: torch.Tensor,
    p_dst: torch.Tensor,
) -> torch.Tensor:
    """
    Alineamiento rigido proyectado al subgrupo 4-DoF Horizontal FLU (roll=0, pitch=0).
    Dados puntos fuente p_src (6, 3) o (..., 6, 3) y destino p_dst (..., 6, 3),
    calcula en forma cerrada la matriz T = [R_z(yaw), t; 0, 1] de forma (..., 4, 4).
    """
    X = p_src.to(dtype=torch.float64)
    Y = p_dst.to(dtype=torch.float64)
    if X.ndim == 2:
        X = X.expand_as(Y)

    x_mean = X.mean(dim=-2, keepdim=True)
    y_mean = Y.mean(dim=-2, keepdim=True)
    Xc = X - x_mean
    Yc = Y - y_mean

    H = torch.matmul(Xc.transpose(-1, -2), Yc)  # (..., 3, 3)
    yaw = torch.atan2(H[..., 0, 1] - H[..., 1, 0], H[..., 0, 0] + H[..., 1, 1])

    c = torch.cos(yaw)
    s = torch.sin(yaw)
    zeros = torch.zeros_like(c)
    ones = torch.ones_like(c)
    R = torch.stack(
        [
            torch.stack([c, -s, zeros], dim=-1),
            torch.stack([s, c, zeros], dim=-1),
            torch.stack([zeros, zeros, ones], dim=-1),
        ],
        dim=-2,
    )
    t = y_mean.squeeze(-2) - torch.matmul(R, x_mean.squeeze(-2).unsqueeze(-1)).squeeze(-1)
    vec4 = torch.cat([t, yaw.unsqueeze(-1)], dim=-1)
    return vec4_to_se3(vec4).to(dtype=p_dst.dtype)


def apply_flow_and_svd(
    T_curr: torch.Tensor,
    flow: torch.Tensor,
    keypoints: Optional[torch.Tensor] = None,
    step_scale: float = 1.0,
) -> torch.Tensor:
    """
    Aplica el flujo predicho [dp_t, dp_r] (..., 6, 6) sobre los keypoints de T_curr (..., 4, 4)
    y recupera la nueva matriz rigida SE(3) 4-DoF mediante align_keypoints_svd_4dof.
    """
    if keypoints is None:
        keypoints = get_uav_keypoints(device=T_curr.device, dtype=T_curr.dtype)
    p_curr = transform_keypoints(T_curr, keypoints)
    dp = flow[..., :3] + flow[..., 3:6]
    p_dst = p_curr + float(step_scale) * dp
    return align_keypoints_svd_4dof(keypoints, p_dst)
