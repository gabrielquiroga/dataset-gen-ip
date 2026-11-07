"""
se3_diffusion_math.py — Transformaciones SE(3), flujo cartesiano desacoplado sobre
6 keypoints y alineamiento SVD restringido a Horizontal FLU (4-DoF) para UAV Instant Policy.

Este modulo implementa el puente matematico entre:
  1. Las matrices homogeneas relativas acumuladas `action_se3` (T, 8, 4, 4) almacenadas
     en `heterodata.pt` (en marco egocentrico Horizontal FLU con factor de escala s=0.1).
  2. La formulacion incremental paso a paso `delta_se3` (δT_k = (T_{EA}^{k-1})^{-1} T_{EA}^k)
     y su normalizacion al hipercubo [-1, 1] para el proceso de difusion.
  3. El flujo cartesiano desacoplado [Δp_t, Δp_r] sobre los 6 keypoints canonicos del
     cuadricoptero x500 y la reconstruccion rigida mediante SVD proyectado al subgrupo
     planar estabilizado por gravedad SE(2) x R ⊂ SE(3) (roll = 0, pitch = 0).
  4. La proyeccion de las acciones predichas hacia waypoints globales en Gazebo ENU (metros)
     y PX4 NED para la ejecucion en lazo cerrado.
"""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Dict, Optional, Tuple, Union

import numpy as np
import torch

DEFAULT_SCALE_FACTOR: float = 0.1
DEFAULT_ARM_LENGTH_M: float = 0.25
DEFAULT_Z_HEIGHT_M: float = 0.15


def get_uav_keypoints_body_numpy(
    scale_factor: float = DEFAULT_SCALE_FACTOR,
    arm_length: float = DEFAULT_ARM_LENGTH_M,
    z_height: float = DEFAULT_Z_HEIGHT_M,
) -> np.ndarray:
    """
    Retorna los 6 nodos keypoint canonicos {p_g^i}_{i=1}^6 del UAV en el marco
    local estabilizado (Horizontal FLU) escalados por `scale_factor` (s = 0.1):
      0: Centro de masa (0, 0, 0)
      1: Proa     +X (+arm_length, 0, 0)
      2: Popa     -X (-arm_length, 0, 0)
      3: Babor    +Y (0, +arm_length, 0)
      4: Estribor -Y (0, -arm_length, 0)
      5: Cenit    +Z (0, 0, +z_height)
    """
    kp_m = np.array(
        [
            [0.0, 0.0, 0.0],
            [arm_length, 0.0, 0.0],
            [-arm_length, 0.0, 0.0],
            [0.0, arm_length, 0.0],
            [0.0, -arm_length, 0.0],
            [0.0, 0.0, z_height],
        ],
        dtype=np.float32,
    )
    return (kp_m * float(scale_factor)).astype(np.float32)


def get_uav_keypoints_body_torch(
    scale_factor: float = DEFAULT_SCALE_FACTOR,
    arm_length: float = DEFAULT_ARM_LENGTH_M,
    z_height: float = DEFAULT_Z_HEIGHT_M,
    device: Optional[torch.device] = None,
    dtype: torch.dtype = torch.float32,
) -> torch.Tensor:
    """Version PyTorch de `get_uav_keypoints_body_numpy` con forma (6, 3)."""
    kp_np = get_uav_keypoints_body_numpy(
        scale_factor=scale_factor,
        arm_length=arm_length,
        z_height=z_height,
    )
    return torch.as_tensor(kp_np, dtype=dtype, device=device)


def wrap_angle_numpy(angle: Union[float, np.ndarray]) -> np.ndarray:
    """Envuelve angulos en radianes al intervalo [-pi, pi)."""
    ang = np.asarray(angle, dtype=np.float64)
    wrapped = np.arctan2(np.sin(ang), np.cos(ang))
    wrapped = np.where(wrapped >= math.pi, wrapped - 2.0 * math.pi, wrapped)
    return wrapped.astype(np.float32 if np.asarray(angle).dtype == np.float32 else np.float64)


def wrap_angle_torch(angle: torch.Tensor) -> torch.Tensor:
    """Envuelve un tensor de angulos en radianes al intervalo [-pi, pi)."""
    return torch.atan2(torch.sin(angle), torch.cos(angle))


def build_se3_horizontal_flu_numpy(
    dx_dy_dz: np.ndarray,
    dyaw: Union[float, np.ndarray],
) -> np.ndarray:
    """
    Construye matrices homogeneas SE(3) estabilizadas por gravedad (roll=0, pitch=0)
    a partir de traslaciones (..., 3) y angulos yaw (...,) o (..., 1).
    Retorna un arreglo de forma (..., 4, 4) dtype float32.
    """
    pos = np.asarray(dx_dy_dz, dtype=np.float64)
    yaw = np.asarray(dyaw, dtype=np.float64)
    if yaw.ndim == pos.ndim and yaw.shape[-1] == 1:
        yaw = yaw[..., 0]
    batch_shape = pos.shape[:-1]
    c = np.cos(yaw)
    s = np.sin(yaw)

    T = np.zeros(batch_shape + (4, 4), dtype=np.float64)
    T[..., 0, 0] = c
    T[..., 0, 1] = -s
    T[..., 1, 0] = s
    T[..., 1, 1] = c
    T[..., 2, 2] = 1.0
    T[..., 3, 3] = 1.0
    T[..., :3, 3] = pos
    return T.astype(np.float32)


def build_se3_horizontal_flu_torch(
    dx_dy_dz: torch.Tensor,
    dyaw: torch.Tensor,
) -> torch.Tensor:
    """
    Construye tensores homogeneos SE(3) estabilizados por gravedad (roll=0, pitch=0)
    a partir de traslaciones (..., 3) y angulos yaw (...,) o (..., 1).
    Retorna un tensor de forma (..., 4, 4).
    """
    if dyaw.ndim == dx_dy_dz.ndim and dyaw.shape[-1] == 1:
        yaw = dyaw[..., 0]
    else:
        yaw = dyaw
    batch_shape = dx_dy_dz.shape[:-1]
    c = torch.cos(yaw)
    s = torch.sin(yaw)
    zeros = torch.zeros_like(c)
    ones = torch.ones_like(c)

    row0 = torch.stack([c, -s, zeros, dx_dy_dz[..., 0]], dim=-1)
    row1 = torch.stack([s, c, zeros, dx_dy_dz[..., 1]], dim=-1)
    row2 = torch.stack([zeros, zeros, ones, dx_dy_dz[..., 2]], dim=-1)
    row3 = torch.stack([zeros, zeros, zeros, ones], dim=-1)
    return torch.stack([row0, row1, row2, row3], dim=-2).reshape(batch_shape + (4, 4))


def invert_rigid_se3_numpy(T: np.ndarray) -> np.ndarray:
    """Inversa analitica exacta de una transformacion rigida SE(3) de forma (..., 4, 4)."""
    T64 = np.asarray(T, dtype=np.float64)
    R = T64[..., :3, :3]
    p = T64[..., :3, 3:4]
    R_T = np.swapaxes(R, -1, -2)
    p_inv = -(R_T @ p)[..., 0]

    T_inv = np.zeros_like(T64)
    T_inv[..., :3, :3] = R_T
    T_inv[..., :3, 3] = p_inv
    T_inv[..., 3, 3] = 1.0
    return T_inv.astype(T.dtype if T.dtype in (np.float32, np.float64) else np.float32)


def invert_rigid_se3_torch(T: torch.Tensor) -> torch.Tensor:
    """Inversa analitica exacta de un tensor de transformaciones rigidas SE(3) (..., 4, 4)."""
    R = T[..., :3, :3]
    p = T[..., :3, 3:4]
    R_T = R.transpose(-1, -2)
    p_inv = -(R_T @ p)

    top = torch.cat([R_T, p_inv], dim=-1)
    bot = torch.zeros(T.shape[:-2] + (1, 4), dtype=T.dtype, device=T.device)
    bot[..., 0, 3] = 1.0
    return torch.cat([top, bot], dim=-2)


def cumulative_to_incremental_se3(
    action_se3: Union[np.ndarray, torch.Tensor],
) -> Union[np.ndarray, torch.Tensor]:
    """
    Convierte un horizonte de transformaciones relativas acumuladas T_{EA}^{(t, 1..H)}
    de forma (..., H, 4, 4) en incrementos paso a paso δT_k de forma (..., H, 4, 4):
        δT_1 = T_{EA}^{(t, 1)}
        δT_k = (T_{EA}^{(t, k-1)})^{-1} @ T_{EA}^{(t, k)},   k = 2, ..., H

    Cada δT_k expresa el movimiento relativo entre el paso k-1 y el paso k (Δt = 0.1 s)
    en el marco local estabilizado del paso k-1, manteniendo todos los pasos en la misma
    escala estadistica.
    """
    if isinstance(action_se3, torch.Tensor):
        H = action_se3.shape[-3]
        if H <= 1:
            return action_se3.clone()
        prev_inv = invert_rigid_se3_torch(action_se3[..., :-1, :, :])
        deltas_rest = prev_inv @ action_se3[..., 1:, :, :]
        return torch.cat([action_se3[..., :1, :, :], deltas_rest], dim=-3)

    arr = np.asarray(action_se3, dtype=np.float64)
    H = arr.shape[-3]
    if H <= 1:
        return arr.astype(np.float32)
    prev_inv = invert_rigid_se3_numpy(arr[..., :-1, :, :]).astype(np.float64)
    deltas_rest = prev_inv @ arr[..., 1:, :, :]
    out = np.concatenate([arr[..., :1, :, :], deltas_rest], axis=-3)
    return out.astype(np.float32)


def incremental_to_cumulative_se3(
    delta_se3: Union[np.ndarray, torch.Tensor],
) -> Union[np.ndarray, torch.Tensor]:
    """
    Reconstruye el horizonte acumulado T_{EA}^{(t, 1..H)} de forma (..., H, 4, 4)
    a partir de los incrementos paso a paso δT_{1..H}:
        T_{EA}^{(t, k)} = δT_1 @ δT_2 @ ... @ δT_k
    """
    if isinstance(delta_se3, torch.Tensor):
        H = delta_se3.shape[-3]
        cum_list = []
        curr = delta_se3[..., 0, :, :]
        cum_list.append(curr)
        for k in range(1, H):
            curr = curr @ delta_se3[..., k, :, :]
            cum_list.append(curr)
        return torch.stack(cum_list, dim=-3)

    arr = np.asarray(delta_se3, dtype=np.float64)
    H = arr.shape[-3]
    out = np.zeros_like(arr)
    if H == 0:
        return out.astype(np.float32)
    out[..., 0, :, :] = arr[..., 0, :, :]
    for k in range(1, H):
        out[..., k, :, :] = out[..., k - 1, :, :] @ arr[..., k, :, :]
    return out.astype(np.float32)


def se3_to_4dof_vector(
    T: Union[np.ndarray, torch.Tensor],
) -> Union[np.ndarray, torch.Tensor]:
    """
    Extrae el vector 4-DoF [dx, dy, dz, dyaw] de forma (..., 4) a partir de
    matrices homogeneas SE(3) estabilizadas por gravedad de forma (..., 4, 4).
    """
    if isinstance(T, torch.Tensor):
        dpos = T[..., :3, 3]
        dyaw = torch.atan2(T[..., 1, 0], T[..., 0, 0]).unsqueeze(-1)
        return torch.cat([dpos, dyaw], dim=-1)

    arr = np.asarray(T, dtype=np.float64)
    dpos = arr[..., :3, 3]
    dyaw = np.arctan2(arr[..., 1, 0], arr[..., 0, 0])[..., None]
    return np.concatenate([dpos, dyaw], axis=-1).astype(np.float32)


def vector_4dof_to_se3(
    vec_4dof: Union[np.ndarray, torch.Tensor],
) -> Union[np.ndarray, torch.Tensor]:
    """
    Convierte vectores 4-DoF [dx, dy, dz, dyaw] de forma (..., 4) en matrices
    homogeneas SE(3) estabilizadas por gravedad de forma (..., 4, 4).
    """
    if isinstance(vec_4dof, torch.Tensor):
        return build_se3_horizontal_flu_torch(vec_4dof[..., :3], vec_4dof[..., 3])
    arr = np.asarray(vec_4dof, dtype=np.float64)
    return build_se3_horizontal_flu_numpy(arr[..., :3], arr[..., 3])


@dataclass(frozen=True)
class ActionBounds4DoF:
    """
    Cotas simetricas auxiliares para reescalar vectores 4-DoF [dx_s, dy_s, dz_s, dyaw_rad]
    al intervalo [-1, 1] durante las pruebas exploratorias Pre-Hito 4.

    NOTA: Los valores por defecto provienen de los percentiles observados en el corpus exploratorio
    actual (Δt = 0.1 s, escala s = 0.1) y NO constituyen cotas universales para la red final.
    Usar `clip=True` con cotas fijas puede truncar maniobras futuras con mayor velocidad angular
    o vertical; para el Hito 4 conviene revisar normalizacion por el schedule de ruido de difusion,
    estadisticos dinamicos del split de entrenamiento (sin clipping duro en inferencia) o limites
    fisicos del vehiculo.
    """

    max_dx_scaled: float = 0.075
    max_dy_scaled: float = 0.075
    max_dz_scaled: float = 0.050
    max_dyaw_rad: float = 0.650

    @classmethod
    def for_stride(cls, action_stride: int = 1) -> "ActionBounds4DoF":
        """Escala las cotas incrementales proporcionalmente al paso temporal `action_stride`."""
        s = max(1, int(action_stride))
        if s == 1:
            return cls()
        return cls(
            max_dx_scaled=0.075 * float(s),
            max_dy_scaled=0.075 * float(s),
            max_dz_scaled=0.050 * float(s),
            max_dyaw_rad=min(math.pi, 0.650 * float(s)),
        )

    def scale_vector_numpy(self) -> np.ndarray:
        return np.array(
            [self.max_dx_scaled, self.max_dy_scaled, self.max_dz_scaled, self.max_dyaw_rad],
            dtype=np.float32,
        )

    def scale_vector_torch(
        self,
        device: Optional[torch.device] = None,
        dtype: torch.dtype = torch.float32,
    ) -> torch.Tensor:
        return torch.tensor(
            [self.max_dx_scaled, self.max_dy_scaled, self.max_dz_scaled, self.max_dyaw_rad],
            dtype=dtype,
            device=device,
        )

    def normalize(
        self,
        vec_4dof: Union[np.ndarray, torch.Tensor],
        clip: bool = True,
    ) -> Union[np.ndarray, torch.Tensor]:
        if isinstance(vec_4dof, torch.Tensor):
            s = self.scale_vector_torch(device=vec_4dof.device, dtype=vec_4dof.dtype)
            normed = vec_4dof / s
            return torch.clamp(normed, -1.0, 1.0) if clip else normed
        s = self.scale_vector_numpy()
        normed = np.asarray(vec_4dof, dtype=np.float32) / s
        return np.clip(normed, -1.0, 1.0) if clip else normed

    def unnormalize(
        self,
        normed_4dof: Union[np.ndarray, torch.Tensor],
    ) -> Union[np.ndarray, torch.Tensor]:
        if isinstance(normed_4dof, torch.Tensor):
            s = self.scale_vector_torch(device=normed_4dof.device, dtype=normed_4dof.dtype)
            out = normed_4dof * s
            yaw = wrap_angle_torch(out[..., 3:4])
            return torch.cat([out[..., :3], yaw], dim=-1)
        s = self.scale_vector_numpy()
        out = np.asarray(normed_4dof, dtype=np.float32) * s
        out[..., 3] = wrap_angle_numpy(out[..., 3])
        return out


def transform_keypoints(
    T_se3: Union[np.ndarray, torch.Tensor],
    keypoints_body: Optional[Union[np.ndarray, torch.Tensor]] = None,
) -> Union[np.ndarray, torch.Tensor]:
    """
    Transforma los 6 keypoints canonicos `keypoints_body` (6, 3) mediante un lote de
    matrices homogeneas `T_se3` de forma (..., 4, 4).
    Retorna las coordenadas 3D de los keypoints con forma (..., 6, 3):
        P_out[..., i, :] = R @ p_g^i + t
    """
    if isinstance(T_se3, torch.Tensor):
        kp = (
            get_uav_keypoints_body_torch(device=T_se3.device, dtype=T_se3.dtype)
            if keypoints_body is None
            else torch.as_tensor(keypoints_body, device=T_se3.device, dtype=T_se3.dtype)
        )
        R = T_se3[..., :3, :3]  # (..., 3, 3)
        t = T_se3[..., :3, 3].unsqueeze(-2)  # (..., 1, 3)
        return torch.matmul(kp, R.transpose(-1, -2)) + t

    T_arr = np.asarray(T_se3, dtype=np.float64)
    kp_arr = (
        get_uav_keypoints_body_numpy().astype(np.float64)
        if keypoints_body is None
        else np.asarray(keypoints_body, dtype=np.float64)
    )
    R = T_arr[..., :3, :3]
    t = T_arr[..., :3, 3][..., None, :]
    out = np.matmul(kp_arr, np.swapaxes(R, -1, -2)) + t
    return out.astype(np.float32)


def compute_decoupled_keypoint_flow(
    T_curr: Union[np.ndarray, torch.Tensor],
    T_target: Union[np.ndarray, torch.Tensor],
    keypoints_body: Optional[Union[np.ndarray, torch.Tensor]] = None,
) -> Union[np.ndarray, torch.Tensor]:
    """
    Calcula el flujo cartesiano desacoplado [Δp_t, Δp_r] de forma (..., 6, 6)
    entre la hipotesis actual `T_curr` (..., 4, 4) y la pose objetivo `T_target` (..., 4, 4),
    siguiendo Vosylius & Johns (ICLR 2025, Sec. 3.2 y 3.3, Ec. 3-4):
      - Δp_t^i = t_* - t_curr           (traslacion pura del origen del marco, identica en los 6 nodos)
      - Δp_r^i = (R_* - R_curr) p_g^i   (rotacion pura del brazo de palanca alrededor del CoM)

    Propiedad fundamental:
      p_curr^i + Δp_t^i + Δp_r^i = (t_curr + R_curr p_g^i) + (t_* - t_curr) + (R_* - R_curr) p_g^i
                                 = t_* + R_* p_g^i = p_target^i
    """
    if isinstance(T_curr, torch.Tensor):
        T_tgt = torch.as_tensor(T_target, device=T_curr.device, dtype=T_curr.dtype)
        kp = (
            get_uav_keypoints_body_torch(device=T_curr.device, dtype=T_curr.dtype)
            if keypoints_body is None
            else torch.as_tensor(keypoints_body, device=T_curr.device, dtype=T_curr.dtype)
        )
        dt = (T_tgt[..., :3, 3] - T_curr[..., :3, 3]).unsqueeze(-2).expand(T_curr.shape[:-2] + (kp.shape[0], 3))
        dR = T_tgt[..., :3, :3] - T_curr[..., :3, :3]
        dr = torch.matmul(kp, dR.transpose(-1, -2))
        return torch.cat([dt, dr], dim=-1)

    T_c = np.asarray(T_curr, dtype=np.float64)
    T_t = np.asarray(T_target, dtype=np.float64)
    kp_arr = (
        get_uav_keypoints_body_numpy().astype(np.float64)
        if keypoints_body is None
        else np.asarray(keypoints_body, dtype=np.float64)
    )
    n_kp = kp_arr.shape[0]
    dt = np.broadcast_to((T_t[..., :3, 3] - T_c[..., :3, 3])[..., None, :], T_c.shape[:-2] + (n_kp, 3))
    dR = T_t[..., :3, :3] - T_c[..., :3, :3]
    dr = np.matmul(kp_arr, np.swapaxes(dR, -1, -2))
    return np.concatenate([dt, dr], axis=-1).astype(np.float32)


def apply_keypoint_flow_and_svd_4dof(
    T_curr: Union[np.ndarray, torch.Tensor],
    flow_6d: Union[np.ndarray, torch.Tensor],
    keypoints_body: Optional[Union[np.ndarray, torch.Tensor]] = None,
    step_scale: float = 1.0,
) -> Union[np.ndarray, torch.Tensor]:
    """
    Aplica un paso de flujo desacoplado `flow_6d` [Δp_t, Δp_r] (..., 6, 6) escalado por
    `step_scale` sobre los keypoints actuales `P(T_curr)` y recupera la nueva transformacion
    rigida SE(3) estabilizada por gravedad mediante SVD en 4-DoF (Ec. 4 y 5 del paper):
        p_dst^i = p_curr^i + step_scale * (Δp_t^i + Δp_r^i)
        T_next  = align_keypoints_svd_horizontal_flu(p_g, p_dst)
    """
    p_curr = transform_keypoints(T_curr, keypoints_body=keypoints_body)
    if isinstance(T_curr, torch.Tensor):
        f = torch.as_tensor(flow_6d, device=T_curr.device, dtype=T_curr.dtype)
        dp = f[..., :3] + f[..., 3:6]
        p_dst = p_curr + float(step_scale) * dp
        kp = (
            get_uav_keypoints_body_torch(device=T_curr.device, dtype=T_curr.dtype)
            if keypoints_body is None
            else torch.as_tensor(keypoints_body, device=T_curr.device, dtype=T_curr.dtype)
        )
        return align_keypoints_svd_horizontal_flu(kp, p_dst)

    f_arr = np.asarray(flow_6d, dtype=np.float64)
    dp = f_arr[..., :3] + f_arr[..., 3:6]
    p_dst = np.asarray(p_curr, dtype=np.float64) + float(step_scale) * dp
    kp_arr = (
        get_uav_keypoints_body_numpy()
        if keypoints_body is None
        else np.asarray(keypoints_body, dtype=np.float32)
    )
    return align_keypoints_svd_horizontal_flu(kp_arr, p_dst)


def align_keypoints_svd_horizontal_flu(
    p_src: Union[np.ndarray, torch.Tensor],
    p_dst: Union[np.ndarray, torch.Tensor],
) -> Union[np.ndarray, torch.Tensor]:
    """
    Alineamiento rigido de Arun/Kabsch proyectado analiticamente al subgrupo 4-DoF
    Horizontal FLU (SE(2) x R ⊂ SE(3), roll = 0, pitch = 0).

    Dados puntos fuente `p_src` (6, 3) o (..., 6, 3) y puntos destino `p_dst` (..., 6, 3),
    encuentra la matriz homogenea T = [R_z(ψ), t; 0, 1] que minimiza:
        sum_{i=1}^6 || (R_z(ψ) p_src^i + t) - p_dst^i ||_2^2

    Demostracion cerrada en SO(2) x R:
      Al restringir R = R_z(ψ), Tr(R H^T) = (H_00 + H_11) cos(ψ) + (H_01 - H_10) sin(ψ) + H_22,
      donde H = X_c^T Y_c. El maximo global sin degeneraciones ni reflexiones impropias (det=+1)
      se alcanza exactamente en:
          ψ* = atan2(H_01 - H_10, H_00 + H_11)
          t* = mean(p_dst) - R_z(ψ*) mean(p_src)
    """
    if isinstance(p_dst, torch.Tensor):
        X = torch.as_tensor(p_src, device=p_dst.device, dtype=torch.float64)
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
        T = build_se3_horizontal_flu_torch(t, yaw)
        return T.to(dtype=p_dst.dtype)

    Y = np.asarray(p_dst, dtype=np.float64)
    X = np.asarray(p_src, dtype=np.float64)
    if X.ndim == 2:
        X = np.broadcast_to(X, Y.shape)
    x_mean = np.mean(X, axis=-2, keepdims=True)
    y_mean = np.mean(Y, axis=-2, keepdims=True)
    Xc = X - x_mean
    Yc = Y - y_mean
    H = np.matmul(np.swapaxes(Xc, -1, -2), Yc)
    yaw = np.arctan2(H[..., 0, 1] - H[..., 1, 0], H[..., 0, 0] + H[..., 1, 1])

    T = build_se3_horizontal_flu_numpy(np.zeros(Y.shape[:-2] + (3,), dtype=np.float64), yaw).astype(np.float64)
    R = T[..., :3, :3]
    xm = x_mean[..., 0, :][..., None]
    ym = y_mean[..., 0, :]
    t = ym - (R @ xm)[..., 0]
    T[..., :3, 3] = t
    return T.astype(np.float32)


def align_keypoints_svd_full_se3_numpy(
    p_src: np.ndarray,
    p_dst: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Alineamiento SVD 6-DoF completo en SE(3) (Arun et al., 1987) para comparar en la
    exploracion el error inducido por roll/pitch espurios frente a la proyeccion 4-DoF.
    Retorna:
      - T_se3: (..., 4, 4)
      - rpy_rad: (..., 3) angulos [roll, pitch, yaw] extraidos de R.
    """
    Y = np.asarray(p_dst, dtype=np.float64)
    X = np.asarray(p_src, dtype=np.float64)
    if X.ndim == 2:
        X = np.broadcast_to(X, Y.shape)
    batch_shape = Y.shape[:-2]
    X_flat = X.reshape(-1, X.shape[-2], 3)
    Y_flat = Y.reshape(-1, Y.shape[-2], 3)

    N_b = X_flat.shape[0]
    T_out = np.zeros((N_b, 4, 4), dtype=np.float64)
    rpy_out = np.zeros((N_b, 3), dtype=np.float64)

    for b in range(N_b):
        xm = X_flat[b].mean(axis=0)
        ym = Y_flat[b].mean(axis=0)
        H = (X_flat[b] - xm).T @ (Y_flat[b] - ym)
        U, _, Vt = np.linalg.svd(H)
        R = Vt.T @ U.T
        if np.linalg.det(R) < 0.0:
            Vt[2, :] *= -1.0
            R = Vt.T @ U.T
        t = ym - R @ xm
        T_out[b, :3, :3] = R
        T_out[b, :3, 3] = t
        T_out[b, 3, 3] = 1.0

        sy = math.hypot(R[0, 0], R[1, 0])
        roll = math.atan2(R[2, 1], R[2, 2])
        pitch = math.atan2(-R[2, 0], sy)
        yaw = math.atan2(R[1, 0], R[0, 0])
        rpy_out[b] = [roll, pitch, yaw]

    return (
        T_out.reshape(batch_shape + (4, 4)).astype(np.float32),
        rpy_out.reshape(batch_shape + (3,)).astype(np.float32),
    )


def action_se3_to_world_waypoints_enu_m(
    live_pose_se3: Union[np.ndarray, torch.Tensor],
    pred_action_se3: Union[np.ndarray, torch.Tensor],
    scale_factor: float = DEFAULT_SCALE_FACTOR,
) -> np.ndarray:
    """
    Proyecta un horizonte de acciones relativas predichas `pred_action_se3` (H, 4, 4)
    o (B, H, 4, 4) desde la pose actual del agente `live_pose_se3` (4, 4) o (B, 4, 4)
    al marco global Gazebo ENU en metros reales y radianes:
        T_{WE}^{(t, k)} = T_{WE}^t @ T_{EA}^{(t, k)}
        X_enu_m, Y_enu_m, Z_enu_m = T_{WE}^{(t, k)}[:3, 3] / scale_factor
        yaw_enu_rad = atan2(T_{WE}^{(t, k)}[1, 0], T_{WE}^{(t, k)}[0, 0])

    Retorna un arreglo NumPy de forma (H, 4) o (B, H, 4) con columnas:
        [X_enu_m, Y_enu_m, Z_enu_m, yaw_enu_rad].
    """
    if isinstance(live_pose_se3, torch.Tensor):
        T_we = live_pose_se3.detach().cpu().numpy().astype(np.float64)
    else:
        T_we = np.asarray(live_pose_se3, dtype=np.float64)

    if isinstance(pred_action_se3, torch.Tensor):
        T_ea = pred_action_se3.detach().cpu().numpy().astype(np.float64)
    else:
        T_ea = np.asarray(pred_action_se3, dtype=np.float64)

    single_query = T_ea.ndim == 3
    if T_we.ndim == 2:
        T_we = T_we[None, :, :]
    if single_query:
        T_ea = T_ea[None, :, :, :]

    T_world = T_we[:, None, :, :] @ T_ea  # (B, H, 4, 4)
    pos_enu_m = T_world[..., :3, 3] / float(scale_factor)
    yaw_enu = np.arctan2(T_world[..., 1, 0], T_world[..., 0, 0])[..., None]
    waypoints = np.concatenate([pos_enu_m, yaw_enu], axis=-1).astype(np.float32)
    return waypoints[0] if single_query else waypoints
