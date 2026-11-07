"""
simple_validation_net.py — Red Simple Validadora (`MiniInstantPolicyNet`) y Politica Oraculo
(`OracleReplayPolicy`) para validar el contrato E/S y lazo cerrado de UAV Instant Policy.

Ambas politicas implementan exactamente el mismo contrato `.predict_actions(...)`:
  Entrada:
    - Contexto: `context_scene_pos` (N, L, 2048, 3), `context_pose_se3` (N, L, 4, 4),
                `context_vel_flu` (N, L, 3), `context_flight_state` (N, L, 1),
                `context_action_se3` (N, L, T, 4, 4) / `context_delta_se3` (N, L, T, 4, 4),
                `context_keyframe_mask` opcional (N, L) bool para `L` adaptativo.
    - Observacion actual (O_t):
                `live_scene_pos` (1, 2048, 3), `live_pose_se3` (1, 4, 4),
                `live_vel_flu` (1, 3), `live_flight_state` (1, 1)
  Salida (`PolicyPrediction`):
    - `pred_action_se3`:            (T, 4, 4) poses relativas acumuladas en Horizontal FLU (s=0.1)
    - `pred_delta_se3`:             (T, 4, 4) incrementos paso a paso δT_k en Horizontal FLU (s=0.1)
    - `pred_flight_state`:          (T, 1)    probabilidad de vuelo activo en [0, 1]
    - `pred_world_waypoints_enu_m`: (T, 4)    [X_enu_m, Y_enu_m, Z_enu_m, yaw_enu_rad] en metros reales
    - `pred_keypoint_flow`:         (T, 6, 6) flujo cartesiano desacoplado [Δp_t, Δp_r] en los 6 keypoints
"""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from .se3_diffusion_math import (
    DEFAULT_SCALE_FACTOR,
    ActionBounds4DoF,
    action_se3_to_world_waypoints_enu_m,
    apply_keypoint_flow_and_svd_4dof,
    compute_decoupled_keypoint_flow,
    cumulative_to_incremental_se3,
    get_uav_keypoints_body_torch,
    incremental_to_cumulative_se3,
    se3_to_4dof_vector,
    transform_keypoints,
    vector_4dof_to_se3,
)


@dataclass
class PolicyPrediction:
    """Estructura tipada de salida de cualquier politica de navegacion UAV Instant Policy."""

    pred_action_se3: torch.Tensor  # (T, 4, 4) float32 en Horizontal FLU (s=0.1)
    pred_delta_se3: torch.Tensor  # (T, 4, 4) float32 en Horizontal FLU (s=0.1)
    pred_flight_state: torch.Tensor  # (T, 1) float32 en [0, 1]
    pred_world_waypoints_enu_m: np.ndarray  # (T, 4) float32 [X_m, Y_m, Z_m, yaw_rad]
    pred_keypoint_flow: torch.Tensor  # (T, 6, 6) float32 [dp_t, dp_r]


def subsample_scene_centroids_torch(
    points: torch.Tensor,
    num_centroids: int = 16,
    num_neighbors: int = 16,
    ground_z_cutoff_scaled: float = -0.11,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Submuestrea `M = num_centroids` centroides representativos de una nube LiDAR (B, N, 3)
    y agrupa sus `K_nn = num_neighbors` vecinos locales mas cercanos para construir parches locales:
      1. En esta version exploratoria aplica un filtro heuristico `z_flu > ground_z_cutoff_scaled`
         antes de FPS. NOTA: Este corte fijo en Z es unicamente exploratorio y NO se recomienda como
         patron fijo para la red final, ya que a gran altitud (ej. `overfly`) puede descartar puntos
         bajos legitimos como el `landing_pad`.
      2. Extrae los `num_neighbors` vecinos mas cercanos a cada centroide y devuelve:
         - `centroids`: (B, M, 3)
         - `local_diffs`: (B, M, K_nn, 3) desplazamientos relativos (p_j - c_m)
    """
    B, N, _ = points.shape
    M = min(int(num_centroids), N)
    K_nn = min(int(num_neighbors), N)
    device = points.device
    dtype = points.dtype

    centroids = torch.zeros((B, M, 3), dtype=dtype, device=device)

    for b in range(B):
        pts = points[b]  # (N, 3)
        non_ground_mask = pts[:, 2] > ground_z_cutoff_scaled
        non_ground_idx = torch.nonzero(non_ground_mask, as_tuple=False).squeeze(-1)
        if non_ground_idx.numel() >= M:
            cand = pts[non_ground_idx]
        else:
            cand = pts
        C_len = cand.shape[0]

        # FPS determinista arrancando desde el punto de obstaculo mas cercano al dron
        dists_origin = torch.sum(cand * cand, dim=-1)
        first_idx = int(torch.argmin(dists_origin).item())
        selected = torch.zeros((M,), dtype=torch.long, device=device)
        selected[0] = first_idx
        min_dists = torch.sum((cand - cand[first_idx : first_idx + 1]) ** 2, dim=-1)

        for m in range(1, M):
            farthest = int(torch.argmax(min_dists).item())
            selected[m] = farthest
            d_new = torch.sum((cand - cand[farthest : farthest + 1]) ** 2, dim=-1)
            min_dists = torch.minimum(min_dists, d_new)

        centroids[b] = cand[selected]

    # Distancias (B, M, N) hacia todos los puntos para k-NN local
    dist_matrix = torch.cdist(centroids, points, p=2.0)  # (B, M, N)
    _, knn_idx = torch.topk(dist_matrix, k=K_nn, dim=-1, largest=False)  # (B, M, K_nn)

    batch_idx = torch.arange(B, device=device)[:, None, None].expand(B, M, K_nn)
    neighbors = points[batch_idx, knn_idx]  # (B, M, K_nn, 3)
    local_diffs = neighbors - centroids.unsqueeze(-2)
    return centroids, local_diffs


class SinusoidalPosEmb3D(nn.Module):
    """
    Codificacion posicional sinusoidal de Fourier para coordenadas/desplazamientos 3D
    (Vosylius & Johns, ICLR 2025, Sec. 3.3).
    """

    def __init__(self, num_freqs: int = 6, include_input: bool = True) -> None:
        super().__init__()
        self.num_freqs = int(num_freqs)
        self.include_input = bool(include_input)
        freqs = (2.0 ** torch.arange(self.num_freqs, dtype=torch.float32)) * math.pi
        self.register_buffer("freqs", freqs, persistent=False)

    @property
    def out_dim(self) -> int:
        base = 3 if self.include_input else 0
        return base + 3 * 2 * self.num_freqs

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (..., 3)
        scaled = x.unsqueeze(-1) * self.freqs  # (..., 3, F)
        sin_feat = torch.sin(scaled)
        cos_feat = torch.cos(scaled)
        fourier = torch.cat([sin_feat, cos_feat], dim=-1).flatten(start_dim=-2)  # (..., 6*F)
        if self.include_input:
            return torch.cat([x, fourier], dim=-1)
        return fourier


class MiniScenePatchEncoder(nn.Module):
    """
    Codificador geometrico ligero φ_e (inspirado en PointNet++ Set Abstraction):
    Comprime (B, 2048, 3) -> M centroides con coordenadas `c_s` (B, M, 3)
    y descriptores locales invariantes a permutacion `h_s` (B, M, d_model).

    Incluye cache geometrico determinista opcional para el paso libre de parametros
    (FPS + k-NN), evitando recalcular `torch.cdist` sobre la misma nube en epocas sucesivas
    mientras recalcula siempre el MLP diferenciable con gradientes completos.
    """

    def __init__(
        self,
        d_model: int = 64,
        num_centroids: int = 16,
        num_neighbors: int = 16,
        use_patch_cache: bool = True,
    ) -> None:
        super().__init__()
        self.d_model = int(d_model)
        self.num_centroids = int(num_centroids)
        self.num_neighbors = int(num_neighbors)
        self.use_patch_cache = bool(use_patch_cache)
        self._patch_cache: Dict[Tuple[Any, ...], Tuple[torch.Tensor, torch.Tensor]] = {}
        self.pos_emb = SinusoidalPosEmb3D(num_freqs=4, include_input=True)
        in_dim = self.pos_emb.out_dim * 2  # centroide + diferencia local

        self.patch_mlp = nn.Sequential(
            nn.Linear(in_dim, d_model),
            nn.LayerNorm(d_model),
            nn.GELU(),
            nn.Linear(d_model, d_model),
        )
        self.out_norm = nn.LayerNorm(d_model)

    def _get_patches(self, scene_pos: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        if not self.use_patch_cache:
            return subsample_scene_centroids_torch(
                scene_pos,
                num_centroids=self.num_centroids,
                num_neighbors=self.num_neighbors,
            )

        B = scene_pos.shape[0]
        device = scene_pos.device
        dtype = scene_pos.dtype
        M = min(self.num_centroids, scene_pos.shape[1])
        K_nn = min(self.num_neighbors, scene_pos.shape[1])

        centroids_list: List[Optional[torch.Tensor]] = [None] * B
        diffs_list: List[Optional[torch.Tensor]] = [None] * B
        missing_indices: List[int] = []
        missing_keys: List[Tuple[Any, ...]] = []

        with torch.no_grad():
            pts_cpu = scene_pos.detach().cpu()
            for b in range(B):
                pb = pts_cpu[b]
                s_all = float(pb.sum().item())
                s_x = float(pb[::32, 0].sum().item())
                s_y = float(pb[::32, 1].sum().item())
                s_z = float(pb[::16, 2].sum().item())
                key = (
                    self.num_centroids,
                    self.num_neighbors,
                    round(s_all, 4),
                    round(s_x, 5),
                    round(s_y, 5),
                    round(s_z, 5),
                )
                cached = self._patch_cache.get(key)
                if cached is not None:
                    centroids_list[b] = cached[0].to(device=device, dtype=dtype)
                    diffs_list[b] = cached[1].to(device=device, dtype=dtype)
                else:
                    missing_indices.append(b)
                    missing_keys.append(key)

            if missing_indices:
                sub_pts = scene_pos[missing_indices]
                c_new, d_new = subsample_scene_centroids_torch(
                    sub_pts,
                    num_centroids=self.num_centroids,
                    num_neighbors=self.num_neighbors,
                )
                for idx_local, b_orig in enumerate(missing_indices):
                    c_item = c_new[idx_local]
                    d_item = d_new[idx_local]
                    centroids_list[b_orig] = c_item
                    diffs_list[b_orig] = d_item
                    if len(self._patch_cache) < 8192:
                        self._patch_cache[missing_keys[idx_local]] = (
                            c_item.detach().cpu(),
                            d_item.detach().cpu(),
                        )

        centroids = torch.stack([c for c in centroids_list if c is not None], dim=0).reshape(B, M, 3)
        local_diffs = torch.stack([d for d in diffs_list if d is not None], dim=0).reshape(B, M, K_nn, 3)
        return centroids, local_diffs

    def forward(self, scene_pos: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Entrada:
          - `scene_pos`: (B, N_pts, 3) en Horizontal FLU (s=0.1)
        Salida:
          - `centroids`: (B, M, 3)
          - `scene_feat`: (B, M, d_model)
        """
        centroids, local_diffs = self._get_patches(scene_pos)
        c_emb = self.pos_emb(centroids).unsqueeze(-2).expand(-1, -1, local_diffs.shape[-2], -1)
        d_emb = self.pos_emb(local_diffs)
        patch_in = torch.cat([c_emb, d_emb], dim=-1)
        h = self.patch_mlp(patch_in)
        h_pool = torch.max(h, dim=-2).values  # Symmetric MaxPool sobre los K_nn vecinos
        return centroids, self.out_norm(h_pool)


class MiniInstantPolicyNet(nn.Module):
    """
    Red Simple Validadora (`MiniInstantPolicyNet`) creada exclusivamente para verificar en CPU
    el flujo de tensores de extremo a extremo (φ_e -> σ -> φ -> ψ + readout SVD 4-DoF) durante
    la exploracion Pre-Hito 4.

    IMPORTANTE — ALCANCE EXPLORATORIO (NO COPIAR COMO PATRON DE LA RED FINAL):
    Para converger en pocos segundos sobre pruebas minimas (1 episodio o 3 muestras), esta clase
    incluye varios atajos ad-hoc que NO deben trasladarse sin revision a la arquitectura del Hito 4:
      - `flow_scale` manual y `ActionBounds4DoF` con `clip=True` ajustados a este dataset.
      - Uso opcional de `pose_se3` global y kernel gaussiano longitudinal (`ctx_x - live_x`),
        lo cual rompe la invariancia espacial SE(3) entre episodios con obstaculos desplazados.
      - Separacion manual `Key`/`Value` y `live_modality_dropout` pensados para el test de 3 muestras.
      - Puente interpolado determinista desde la identidad `I_4` en lugar de un schedule DDPM/DDIM
        estandar en se(3), y caches en memoria por suma hash para acelerar el overfit repetitivo.
    """

    def __init__(
        self,
        d_model: int = 64,
        num_heads: int = 4,
        num_centroids: int = 16,
        num_neighbors: int = 16,
        horizon: int = 8,
        action_stride: int = 1,
        num_diffusion_steps: int = 4,
        scale_factor: float = DEFAULT_SCALE_FACTOR,
        live_modality_dropout: float = 0.0,
    ) -> None:
        super().__init__()
        self.d_model = int(d_model)
        self.num_heads = int(num_heads)
        self.num_centroids = int(num_centroids)
        self.horizon = max(1, int(horizon))
        self.action_stride = max(1, int(action_stride))
        self.num_diffusion_steps = int(num_diffusion_steps)
        self.scale_factor = float(scale_factor)
        self.live_modality_dropout = float(live_modality_dropout)
        self.bounds = ActionBounds4DoF.for_stride(self.action_stride)
        s_mult = float(self.action_stride)
        flow_scale = torch.tensor(
            [0.04 * s_mult, 0.04 * s_mult, 0.03 * s_mult, 0.012, 0.012, 0.012],
            dtype=torch.float32,
        )
        self.register_buffer("flow_scale", flow_scale, persistent=False)

        self.pos_emb_3d = SinusoidalPosEmb3D(num_freqs=4, include_input=True)
        self.scene_encoder = MiniScenePatchEncoder(
            d_model=d_model,
            num_centroids=num_centroids,
            num_neighbors=num_neighbors,
        )

        # Embeddings de identidad para los 6 keypoints del UAV, los T pasos del horizonte y progreso continuo s_l in [0, 1]
        self.keypoint_id_emb = nn.Embedding(6, d_model)
        self.horizon_step_emb = nn.Embedding(self.horizon, d_model)
        self.keyframe_progress_mlp = nn.Sequential(
            nn.Linear(16, d_model),
            nn.GELU(),
            nn.Linear(d_model, d_model),
        )

        # Etapa σ (Local Scene -> 6 Agent Keypoints + propiocepcion vel_flu, flight_state y progreso espacial Fourier)
        self.proprio_proj = nn.Sequential(
            nn.Linear(self.pos_emb_3d.out_dim * 2 + 6, d_model),
            nn.LayerNorm(d_model),
            nn.GELU(),
            nn.Linear(d_model, d_model),
        )
        self.sigma_rel_proj = nn.Linear(self.pos_emb_3d.out_dim + d_model, d_model)
        self.sigma_attn = nn.MultiheadAttention(d_model, num_heads=num_heads, batch_first=True)
        self.sigma_scene_gate = nn.Parameter(torch.tensor(0.15, dtype=torch.float32))
        self.sigma_norm = nn.LayerNorm(d_model)

        # Etapa φ (Contexto Demo [L keyframes + acciones demostradas] -> 6 Keypoints Actuales)
        # Separamos Claves (observacion σ) y Valores (acciones demostradas normalizadas)
        self.demo_action_encoder = nn.Sequential(
            nn.Linear(self.horizon * 5, d_model),
            nn.LayerNorm(d_model),
            nn.GELU(),
            nn.Linear(d_model, d_model),
        )
        self.phi_cross_attn = nn.MultiheadAttention(d_model, num_heads=num_heads, batch_first=True)
        self.phi_ffn = nn.Sequential(
            nn.Linear(d_model * 2, d_model),
            nn.LayerNorm(d_model),
            nn.GELU(),
            nn.Linear(d_model, d_model),
        )
        self.phi_norm = nn.LayerNorm(d_model)

        # Etapa ψ (6 Bottleneck Keypoints -> T x 6 Noisy Action Keypoints)
        self.diff_step_mlp = nn.Sequential(
            nn.Linear(16, d_model),
            nn.GELU(),
            nn.Linear(d_model, d_model),
        )
        self.action_kp_proj = nn.Sequential(
            nn.Linear(self.pos_emb_3d.out_dim * 2 + d_model, d_model),
            nn.LayerNorm(d_model),
            nn.GELU(),
        )
        self.psi_cross_attn = nn.MultiheadAttention(d_model, num_heads=num_heads, batch_first=True)
        self.psi_self_attn = nn.MultiheadAttention(d_model, num_heads=num_heads, batch_first=True)
        self.psi_norm = nn.LayerNorm(d_model)

        # Cabezal de prediccion por nodo keypoint futuro: [Δp_t_norm (3), Δp_r_norm (3), flight_state_logit (1)]
        self.flow_head = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.GELU(),
            nn.Linear(d_model, 7),
        )

    @staticmethod
    def _sinusoidal_scalar_features(val: torch.Tensor) -> torch.Tensor:
        """Codificacion sinusoidal escalar para v in [0, 1] -> (N, 16)."""
        half = 8
        v_col = val.reshape(-1, 1)
        freqs = torch.exp(
            -math.log(100.0) * torch.arange(half, dtype=torch.float32, device=val.device) / max(1, half - 1)
        )
        args = v_col * freqs
        return torch.cat([torch.sin(args), torch.cos(args)], dim=-1)

    def _encode_diffusion_step(self, k_step: torch.Tensor) -> torch.Tensor:
        """Codificacion sinusoidal escalar del paso de difusion k in [0, 1] -> (B, d_model)."""
        return self.diff_step_mlp(self._sinusoidal_scalar_features(k_step))

    def _encode_keyframe_progress(self, prog: torch.Tensor) -> torch.Tensor:
        """Codificacion continua de progreso relativo del keyframe s_l in [0, 1] -> (N, d_model) invariante a L."""
        return self.keyframe_progress_mlp(self._sinusoidal_scalar_features(prog))

    def _run_sigma_local(
        self,
        scene_pos: torch.Tensor,
        vel_flu: torch.Tensor,
        flight_state: torch.Tensor,
        keypoints_body: torch.Tensor,
        pose_se3: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Ejecuta φ_e + σ sobre un lote de observaciones locales:
          - `scene_pos`: (B_obs, 2048, 3)
          - `vel_flu`: (B_obs, 3)
          - `flight_state`: (B_obs, 1)
          - `keypoints_body`: (6, 3)
          - `pose_se3`: (B_obs, 4, 4) opcional para anclaje de fase longitudinal
        Retorna embeddings de los 6 keypoints del agente: (B_obs, 6, d_model).
        """
        B_obs = scene_pos.shape[0]
        device = scene_pos.device
        centroids, scene_feat = self.scene_encoder(scene_pos)  # (B_obs, M, 3), (B_obs, M, d)

        kp = keypoints_body.unsqueeze(0).expand(B_obs, 6, 3)  # (B_obs, 6, 3)
        kp_emb = self.pos_emb_3d(kp * 10.0)  # Escalar brazo de palanca 0.05 -> 0.5 para Fourier

        if pose_se3 is not None:
            p_xyz = pose_se3[:, :3, 3] * torch.tensor([0.35, 1.0, 1.0], device=device, dtype=scene_pos.dtype)
            p_emb = self.pos_emb_3d(p_xyz).unsqueeze(1).expand(B_obs, 6, -1)
            cos_y = pose_se3[:, 0, 0:1]
            sin_y = pose_se3[:, 1, 0:1]
            yaw_cs = torch.cat([cos_y, sin_y], dim=-1)
        else:
            p_emb = torch.zeros((B_obs, 6, self.pos_emb_3d.out_dim), device=device, dtype=scene_pos.dtype)
            yaw_cs = torch.zeros((B_obs, 2), device=device, dtype=scene_pos.dtype)

        proprio = torch.cat([vel_flu * 0.25, flight_state, yaw_cs], dim=-1).unsqueeze(1).expand(B_obs, 6, 6)
        kp_ids = self.keypoint_id_emb(torch.arange(6, device=device)).unsqueeze(0).expand(B_obs, 6, -1)
        q_agent = self.proprio_proj(torch.cat([kp_emb, p_emb, proprio], dim=-1)) + kp_ids  # (B_obs, 6, d)

        # Geometria relativa centroide - Centro de Masa (CoM = 0 en Horizontal FLU)
        c_pos_emb = self.pos_emb_3d(centroids)  # (B_obs, M, pos_dim)
        kv_scene = self.sigma_rel_proj(torch.cat([c_pos_emb, scene_feat], dim=-1))  # (B_obs, M, d)

        attn_out, _ = self.sigma_attn(query=q_agent, key=kv_scene, value=kv_scene)
        return self.sigma_norm(q_agent + self.sigma_scene_gate * attn_out)  # (B_obs, 6, d_model)

    def encode_context(
        self,
        context_scene_pos: torch.Tensor,
        context_vel_flu: torch.Tensor,
        context_flight_state: torch.Tensor,
        context_delta_4dof: torch.Tensor,
        context_target_flight_state: torch.Tensor,
        keypoints_body: torch.Tensor,
        context_pose_se3: Optional[torch.Tensor] = None,
        context_keyframe_mask: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Precalcula y cachea las representaciones de contexto (φ_e + σ + codificacion de acciones)
        para las `N_demos x L` observaciones demostradas (soportando `L` arbitrario o adaptativo).
        Acepta entrada con o sin dimension de batch:
          - `context_scene_pos`: (B, N_demos, L, 2048, 3) o (N_demos, L, 2048, 3)
          - `context_keyframe_mask`: opcional (B, N_demos, L) o (N_demos, L) bool
        Retorna:
          - `ctx_keys`:   (B, N_demos * L * 6, d_model) claves en el espacio de observacion σ
          - `ctx_values`: (B, N_demos * L * 6, d_model) valores con la accion demostrada codificada
        """
        if context_scene_pos.ndim == 4:
            context_scene_pos = context_scene_pos.unsqueeze(0)
            context_vel_flu = context_vel_flu.unsqueeze(0)
            context_flight_state = context_flight_state.unsqueeze(0)
            context_delta_4dof = context_delta_4dof.unsqueeze(0)
            context_target_flight_state = context_target_flight_state.unsqueeze(0)
            if context_pose_se3 is not None and context_pose_se3.ndim == 4:
                context_pose_se3 = context_pose_se3.unsqueeze(0)
            if context_keyframe_mask is not None and context_keyframe_mask.ndim == 2:
                context_keyframe_mask = context_keyframe_mask.unsqueeze(0)

        B, N_demos, L, N_pts, _ = context_scene_pos.shape
        device = context_scene_pos.device
        if int(context_delta_4dof.shape[-2]) != self.horizon:
            raise ValueError(
                f"MiniInstantPolicyNet configurada con horizon={self.horizon}, "
                f"pero context_delta_4dof tiene T={context_delta_4dof.shape[-2]}."
            )

        norm_d4 = self.bounds.normalize(context_delta_4dof, clip=True)
        demo_act_5d = torch.cat([norm_d4, context_target_flight_state], dim=-1)  # (B, N, L, T, 5)

        if B > 1:
            unique_b_idx: List[int] = []
            inv_map: List[int] = []
            seen_sigs: Dict[Tuple[float, float, float, float, int], int] = {}
            with torch.no_grad():
                s_sum = context_scene_pos.reshape(B, -1).sum(dim=-1).detach().cpu()
                a_sum = context_delta_4dof.reshape(B, -1).sum(dim=-1).detach().cpu()
                a_y = context_delta_4dof[:, 0, :, :, 1].reshape(B, -1).sum(dim=-1).detach().cpu()
                a_x = context_delta_4dof[:, 0, :, :, 0].reshape(B, -1).sum(dim=-1).detach().cpu()
                m_sum = (
                    context_keyframe_mask.reshape(B, -1).sum(dim=-1).detach().cpu()
                    if context_keyframe_mask is not None
                    else torch.full((B,), N_demos * L, dtype=torch.long)
                )
                for b in range(B):
                    sig = (
                        round(float(s_sum[b].item()), 3),
                        round(float(a_sum[b].item()), 5),
                        round(float(a_y[b].item()), 5),
                        round(float(a_x[b].item()), 5),
                        int(m_sum[b].item()),
                    )
                    if sig not in seen_sigs:
                        seen_sigs[sig] = len(unique_b_idx)
                        unique_b_idx.append(b)
                    inv_map.append(seen_sigs[sig])
            U = len(unique_b_idx)
        else:
            unique_b_idx = [0]
            inv_map = [0]
            U = 1

        U_flat = U * N_demos * L
        u_scene = context_scene_pos[unique_b_idx].reshape(U_flat, N_pts, 3)
        u_vel = context_vel_flu[unique_b_idx].reshape(U_flat, 3)
        u_fs = context_flight_state[unique_b_idx].reshape(U_flat, 1)
        u_act_flat = demo_act_5d[unique_b_idx].reshape(U_flat, self.horizon * 5)
        u_pose = (
            context_pose_se3[unique_b_idx].reshape(U_flat, 4, 4)
            if context_pose_se3 is not None
            else None
        )
        if context_keyframe_mask is not None:
            u_mask = context_keyframe_mask[unique_b_idx].to(device=device, dtype=torch.bool)  # (U, N_demos, L)
        else:
            u_mask = torch.ones((U, N_demos, L), dtype=torch.bool, device=device)

        # Garantizar que al menos el primer keyframe sea valido en cada secuencia para evitar NaN en softmax
        if not bool(torch.all(u_mask.any(dim=-1))):
            u_mask = u_mask.clone()
            u_mask[:, :, 0] = True

        sigma_ctx = self._run_sigma_local(
            scene_pos=u_scene,
            vel_flu=u_vel,
            flight_state=u_fs,
            keypoints_body=keypoints_body,
            pose_se3=u_pose,
        )  # (U * N_demos * L, 6, d_model)

        act_emb = self.demo_action_encoder(u_act_flat).unsqueeze(1).expand(U_flat, 6, self.d_model)
        l_valid_counts = u_mask.sum(dim=-1, keepdim=True).clamp(min=1).to(dtype=torch.float32)  # (U, N_demos, 1)
        l_grid = torch.arange(L, device=device, dtype=torch.float32).view(1, 1, L)
        prog_norm = (l_grid / (l_valid_counts - 1.0).clamp(min=1.0)).clamp(0.0, 1.0)  # (U, N_demos, L)
        l_emb = (
            self._encode_keyframe_progress(prog_norm.reshape(-1, 1))
            .view(U, N_demos, L, 1, self.d_model)
            .expand(U, N_demos, L, 6, self.d_model)
            .reshape(U_flat, 6, self.d_model)
        )

        u_keys = (sigma_ctx + 0.15 * l_emb).reshape(U, N_demos * L * 6, self.d_model)
        u_values = (act_emb + 0.25 * sigma_ctx).reshape(U, N_demos * L * 6, self.d_model)
        u_token_mask = u_mask.unsqueeze(-1).expand(U, N_demos, L, 6).reshape(U, N_demos * L * 6)

        # Codificar coordenadas longitudinales x_s de los keyframes si context_pose_se3 existe
        if u_pose is not None:
            u_x_kf = u_pose[:, 0, 3].reshape(U, N_demos * L, 1).expand(U, N_demos * L, 6).reshape(U, N_demos * L * 6, 1)
        else:
            u_x_kf = torch.linspace(0.0, 2.5, steps=max(2, L), device=device)[:L].view(1, 1, L, 1).expand(U, N_demos, L, 6).reshape(U, N_demos * L * 6, 1)

        if B > 1 and U < B:
            inv_tensor = torch.tensor(inv_map, dtype=torch.long, device=device)
            ctx_keys = u_keys[inv_tensor]
            ctx_values = u_values[inv_tensor]
            ctx_x_kf = u_x_kf[inv_tensor]
            ctx_token_mask = u_token_mask[inv_tensor]
        else:
            ctx_keys = u_keys
            ctx_values = u_values
            ctx_x_kf = u_x_kf
            ctx_token_mask = u_token_mask

        self._last_ctx_x_kf = ctx_x_kf
        self._last_ctx_token_mask = ctx_token_mask
        return ctx_keys, ctx_values

    def compute_bottleneck(
        self,
        live_scene_pos: torch.Tensor,
        live_vel_flu: torch.Tensor,
        live_flight_state: torch.Tensor,
        keypoints_body: torch.Tensor,
        ctx_tokens: torch.Tensor,
        ctx_values: Optional[torch.Tensor] = None,
        live_pose_se3: Optional[torch.Tensor] = None,
        ctx_token_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Ejecuta φ_e + σ sobre la observacion actual O_t y luego el propagador de contexto φ
        emparejando `query = sigma_live` contra `key = ctx_keys` para extraer `value = ctx_values`
        con ponderacion de localidad longitudinal sobre los keyframes demostrados (enmascarando
        slots de padding cuando `L` es adaptativo), produciendo `(B, 6, d_model)`.
        """
        if ctx_values is None:
            ctx_values = ctx_tokens
        B = live_scene_pos.shape[0]
        sigma_live = self._run_sigma_local(
            scene_pos=live_scene_pos,
            vel_flu=live_vel_flu,
            flight_state=live_flight_state,
            keypoints_body=keypoints_body,
            pose_se3=live_pose_se3,
        )  # (B, 6, d_model)

        if (
            ctx_token_mask is None
            and hasattr(self, "_last_ctx_token_mask")
            and self._last_ctx_token_mask is not None
            and self._last_ctx_token_mask.shape[0] in (1, B)
            and self._last_ctx_token_mask.shape[1] == ctx_tokens.shape[1]
        ):
            ctx_token_mask = self._last_ctx_token_mask

        key_padding_mask: Optional[torch.Tensor] = None
        kf_valid_mask: Optional[torch.Tensor] = None
        if ctx_token_mask is not None:
            tm = ctx_token_mask.to(device=sigma_live.device, dtype=torch.bool)
            if tm.shape[0] == 1 and B > 1:
                tm = tm.expand(B, -1)
            kf_valid_mask = tm.reshape(B, -1, 6)[:, :, 0]
            if not bool(torch.all(tm)):
                key_padding_mask = ~tm

        cross_ctx, _ = self.phi_cross_attn(
            query=sigma_live,
            key=ctx_tokens,
            value=ctx_values,
            key_padding_mask=key_padding_mask,
        )

        # Kernel de localidad de progreso longitudinal sobre los N_demos * L keyframes
        ctx_vals_kf = ctx_values.reshape(B, -1, 6, self.d_model)  # (B, K_kf, 6, d_model)
        if (
            live_pose_se3 is not None
            and hasattr(self, "_last_ctx_x_kf")
            and self._last_ctx_x_kf is not None
            and self._last_ctx_x_kf.shape[0] in (1, B)
            and self._last_ctx_x_kf.shape[1] == ctx_values.shape[1]
        ):
            ctx_x = self._last_ctx_x_kf.reshape(-1, ctx_vals_kf.shape[1], 6)[:, :, 0]  # (1 or B, K_kf)
            if ctx_x.shape[0] == 1 and B > 1:
                ctx_x = ctx_x.expand(B, -1)
            live_x = live_pose_se3[:, 0, 3:4]  # (B, 1)
            # Ventana gaussiana longitudinal nitida (sigma_x = 0.07 en s=0.1, equivalente a 0.70 m)
            dist_sq = (ctx_x - live_x) ** 2
            logits = -dist_sq / (2.0 * (0.07 ** 2))
            if kf_valid_mask is not None:
                logits = logits.masked_fill(~kf_valid_mask, -1e9)
            phase_weights = torch.softmax(logits, dim=-1)  # (B, K_kf)
            local_ctx_act = torch.sum(ctx_vals_kf * phase_weights[:, :, None, None], dim=1)  # (B, 6, d_model)
        else:
            if kf_valid_mask is not None:
                w = kf_valid_mask.to(dtype=ctx_vals_kf.dtype)
                denom = w.sum(dim=1, keepdim=True).clamp(min=1.0).unsqueeze(-1)
                local_ctx_act = (ctx_vals_kf * w[:, :, None, None]).sum(dim=1) / denom
            else:
                local_ctx_act = ctx_vals_kf.mean(dim=1)

        ctx_anchor = 0.4 * cross_ctx + 0.6 * local_ctx_act

        if self.live_modality_dropout >= 1.0:
            sigma_for_fuse = torch.zeros_like(sigma_live)
        elif self.training and self.live_modality_dropout > 0.0:
            keep = (
                torch.rand((sigma_live.shape[0], 1, 1), device=sigma_live.device) >= self.live_modality_dropout
            ).to(sigma_live.dtype)
            sigma_for_fuse = sigma_live * keep
        else:
            sigma_for_fuse = sigma_live

        fused = self.phi_ffn(torch.cat([sigma_for_fuse, ctx_anchor], dim=-1))
        # Anclar el camino residual en `ctx_anchor` garantiza estabilidad en lazo cerrado y condicionamiento in-context
        return self.phi_norm(ctx_anchor + fused)  # (B, 6, d_model)

    def decode_action_flow_step(
        self,
        bottleneck: torch.Tensor,
        curr_delta_se3: torch.Tensor,
        k_norm: torch.Tensor,
        keypoints_body: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Etapa ψ: Propaga informacion desde los 6 nodos del cuello de botella (`bottleneck`, [B, 6, d])
        hacia los `T x 6` nodos keypoint de las acciones futuras hipotesis `curr_delta_se3`
        (B, T, 4, 4) en el paso de difusion `k_norm` (B, 1).

        Retorna:
          - `pred_flow_6d`: (B, T, 6, 6) flujo cartesiano [Δp_t, Δp_r] para cada paso k=1..T
          - `pred_flight_logits`: (B, T, 1) logits del estado binario de vuelo s_{t+1..t+T}
        """
        B = bottleneck.shape[0]
        device = bottleneck.device

        # Posiciones relativas de los 6 keypoints respecto al cuerpo (escaladas a O(1) para Fourier)
        curr_cum_se3 = incremental_to_cumulative_se3(curr_delta_se3)  # (B, T, 4, 4)
        kp_body_exp = keypoints_body.view(1, 1, 6, 3)
        delta_scale = 0.04 * float(self.action_stride)
        cum_scale = 0.02 * float(self.horizon * self.action_stride)
        kp_delta_rel = (transform_keypoints(curr_delta_se3, keypoints_body=keypoints_body) - kp_body_exp) / delta_scale
        kp_cum_rel = (transform_keypoints(curr_cum_se3, keypoints_body=keypoints_body) - kp_body_exp) / cum_scale

        emb_delta = self.pos_emb_3d(kp_delta_rel)  # (B, T, 6, pos_dim)
        emb_cum = self.pos_emb_3d(kp_cum_rel)  # (B, T, 6, pos_dim)

        h_ids = self.horizon_step_emb(torch.arange(self.horizon, device=device))[None, :, None, :]  # (1, T, 1, d)
        kp_ids = self.keypoint_id_emb(torch.arange(6, device=device))[None, None, :, :]  # (1, 1, 6, d)
        k_emb = self._encode_diffusion_step(k_norm)[:, None, None, :]  # (B, 1, 1, d)

        # Inyeccion directa por keypoint i desde bottleneck[:, None, :, :] + codificacion de paso
        node_cond = bottleneck.unsqueeze(1) + h_ids + kp_ids + k_emb  # (B, T, 6, d)
        act_in = torch.cat([emb_delta, emb_cum, node_cond], dim=-1)
        act_tokens = self.action_kp_proj(act_in).reshape(B, self.horizon * 6, self.d_model)

        # Atencion cruzada desde el cuello de botella actual (6 nodos) hacia los T*6 nodos de accion
        cross_out, _ = self.psi_cross_attn(query=act_tokens, key=bottleneck, value=bottleneck)
        act_tokens = act_tokens + cross_out

        # Autoatencion entre los T*6 nodos de accion para coherencia espacial-temporal del horizonte
        self_out, _ = self.psi_self_attn(query=act_tokens, key=act_tokens, value=act_tokens)
        act_tokens = self.psi_norm(act_tokens + self_out)

        raw_out = self.flow_head(act_tokens).reshape(B, self.horizon, 6, 7)
        pred_flow_norm = raw_out[..., :6]  # (B, T, 6, 6) en unidades O(1)
        pred_flow_6d = pred_flow_norm * self.flow_scale.view(1, 1, 1, 6)
        pred_flight_logits = raw_out[..., 6].mean(dim=-1, keepdim=True)  # Promedio sobre los 6 keypoints -> (B, T, 1)
        return pred_flow_6d, pred_flight_logits

    def compute_loss(
        self,
        batch: Dict[str, Any],
        noise_std_pos: float = 0.020,
        noise_std_yaw: float = 0.15,
    ) -> Dict[str, torch.Tensor]:
        """
        Calcula la perdida de entrenamiento de flujo en SE(3)/4-DoF sobre los 6 keypoints
        mas la entropia cruzada binaria del estado de vuelo `flight_state`, ponderada por `valid_mask`.
        Entrena simultaneamente:
          1. Desde una hipotesis ruidosa `δT_noisy` (simulando cualquier etapa del proceso inverso DDIM).
          2. Desde la identidad `I_4` (garantizando convergencia exacta en K=1..4 pasos desde identidad).
        """
        device = next(self.parameters()).device
        ctx_scene = batch["context_scene_pos"].to(device)
        ctx_vel = batch["context_vel_flu"].to(device)
        ctx_fs = batch["context_flight_state"].to(device)
        ctx_d4 = batch["context_delta_4dof"].to(device)
        ctx_tfs = batch["context_target_flight_state"].to(device)
        ctx_pose = batch.get("context_pose_se3")
        if ctx_pose is not None:
            ctx_pose = ctx_pose.to(device)
        ctx_mask = batch.get("context_keyframe_mask")
        if ctx_mask is not None:
            ctx_mask = ctx_mask.to(device)

        live_scene = batch["live_scene_pos"].to(device)
        if live_scene.ndim == 4 and live_scene.shape[1] == 1:
            live_scene = live_scene.squeeze(1)
        live_vel = batch["live_vel_flu"].to(device)
        if live_vel.ndim == 3 and live_vel.shape[1] == 1:
            live_vel = live_vel.squeeze(1)
        live_fs = batch["live_flight_state"].to(device)
        if live_fs.ndim == 3 and live_fs.shape[1] == 1:
            live_fs = live_fs.squeeze(1)
        live_pose = batch.get("live_pose_se3")
        if live_pose is not None:
            live_pose = live_pose.to(device)
            if live_pose.ndim == 4 and live_pose.shape[1] == 1:
                live_pose = live_pose.squeeze(1)

        kp_body = batch["keypoints_body"].to(device)
        if kp_body.ndim == 3:
            kp_body = kp_body[0]

        target_delta_se3 = batch["target_delta_se3"].to(device)  # (B, T, 4, 4)
        target_delta_4dof = batch["target_delta_4dof"].to(device)  # (B, T, 4)
        target_fs = batch["target_flight_state"].to(device)  # (B, T, 1)
        valid_mask = batch["target_valid_mask"].to(device)  # (B, T)

        B = target_delta_se3.shape[0]
        ctx_keys, ctx_values = self.encode_context(
            context_scene_pos=ctx_scene,
            context_vel_flu=ctx_vel,
            context_flight_state=ctx_fs,
            context_delta_4dof=ctx_d4,
            context_target_flight_state=ctx_tfs,
            keypoints_body=kp_body,
            context_pose_se3=ctx_pose,
            context_keyframe_mask=ctx_mask,
        )
        bottleneck = self.compute_bottleneck(
            live_scene_pos=live_scene,
            live_vel_flu=live_vel,
            live_flight_state=live_fs,
            keypoints_body=kp_body,
            ctx_tokens=ctx_keys,
            ctx_values=ctx_values,
            live_pose_se3=live_pose,
        )

        # Muestrear paso del puente Rectified Flow / Brownian Bridge u in {1/4, 2/4, 3/4, 1.0} U Uniform(0, 1)
        u = torch.rand((B, 1), device=device)
        half_b = B // 2
        if half_b > 0:
            # Muestrear exactamente los nodos discretos usados por K in {1, 2, 4} (u in {0.25, 0.5, 0.75, 1.0})
            discrete_u = torch.tensor([1.0, 0.5, 0.75, 0.25], device=device, dtype=target_delta_4dof.dtype)
            u[:half_b, 0] = discrete_u[torch.arange(half_b, device=device) % 4]

        noise_4d = torch.randn_like(target_delta_4dof)
        pos_std = noise_std_pos * float(self.action_stride)
        noise_scale = torch.tensor(
            [pos_std, pos_std, pos_std, noise_std_yaw],
            device=device,
            dtype=target_delta_4dof.dtype,
        )
        # Puente interpolado entre Identidad (u=1 -> 0) y Target (u=0 -> target_delta_4dof)
        bridge_sigma = u[:, :, None] * (1.0 - u[:, :, None]) * 2.0
        noisy_4dof = (1.0 - u[:, :, None]) * target_delta_4dof + bridge_sigma * (noise_4d * noise_scale)
        curr_delta_se3 = vector_4dof_to_se3(noisy_4dof)

        gt_flow_6d = compute_decoupled_keypoint_flow(
            T_curr=curr_delta_se3,
            T_target=target_delta_se3,
            keypoints_body=kp_body,
        )  # (B, T, 6, 6)

        pred_flow_6d, pred_fs_logits = self.decode_action_flow_step(
            bottleneck=bottleneck,
            curr_delta_se3=curr_delta_se3,
            k_norm=u,
            keypoints_body=kp_body,
        )

        scale_view = self.flow_scale.view(1, 1, 1, 6)
        pred_norm = pred_flow_6d / scale_view
        gt_norm = gt_flow_6d / scale_view

        # Ponderar pasos validos (los pasos terminales de aterrizaje siguen teniendo peso 0.25 para mantener quietud)
        step_weights = torch.where(
            valid_mask,
            torch.ones_like(target_fs[..., 0]),
            0.25 * torch.ones_like(target_fs[..., 0]),
        )
        flow_mse_per_step = F.mse_loss(pred_norm, gt_norm, reduction="none").mean(dim=(-1, -2))
        loss_flow = (flow_mse_per_step * step_weights).mean()

        loss_fs = F.binary_cross_entropy_with_logits(pred_fs_logits, target_fs)
        total_loss = loss_flow * 10.0 + loss_fs
        return {
            "loss": total_loss,
            "loss_flow": loss_flow,
            "loss_flight_state": loss_fs,
        }

    @torch.no_grad()
    def predict_actions(
        self,
        sample: Dict[str, Any],
        num_ddim_steps: Optional[int] = None,
        cached_context: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
    ) -> PolicyPrediction:
        """
        Ejecuta la inferencia de dos niveles de Instant Policy para una observacion `sample`:
          1. Si `cached_context` no fue provisto, codifica las N_demos x L observaciones de contexto.
          2. Calcula el cuello de botella `bottleneck` (1, 6, d_model) UNA SOLA VEZ para O_t.
          3. Ejecuta el bucle interno DDIM de `K = num_ddim_steps` pasos sobre los `T x 6` nodos de
             accion futura, aplicando `apply_keypoint_flow_and_svd_4dof` en cada iteracion.
          4. Compone los incrementos `δT_{1..T}` en `T_{EA}^{(t, 1..T)}` y proyecta los waypoints
             al marco Gazebo ENU en metros reales.
        """
        self.eval()
        device = next(self.parameters()).device
        K = int(num_ddim_steps) if num_ddim_steps is not None else self.num_diffusion_steps
        K = max(1, K)

        kp_body = sample.get("keypoints_body")
        if kp_body is None:
            kp_body = get_uav_keypoints_body_torch(scale_factor=self.scale_factor, device=device)
        else:
            kp_body = torch.as_tensor(kp_body, dtype=torch.float32, device=device)
            if kp_body.ndim == 3:
                kp_body = kp_body[0]

        ctx_pose = sample.get("context_pose_se3")
        if ctx_pose is not None:
            ctx_pose = torch.as_tensor(ctx_pose, dtype=torch.float32, device=device)
        ctx_mask = sample.get("context_keyframe_mask")
        if ctx_mask is not None:
            ctx_mask = torch.as_tensor(ctx_mask, dtype=torch.bool, device=device)

        if cached_context is None:
            ctx_keys, ctx_values = self.encode_context(
                context_scene_pos=sample["context_scene_pos"].to(device),
                context_vel_flu=sample["context_vel_flu"].to(device),
                context_flight_state=sample["context_flight_state"].to(device),
                context_delta_4dof=sample["context_delta_4dof"].to(device),
                context_target_flight_state=sample["context_target_flight_state"].to(device),
                keypoints_body=kp_body,
                context_pose_se3=ctx_pose,
                context_keyframe_mask=ctx_mask,
            )
        else:
            ctx_keys, ctx_values = cached_context

        live_scene = sample["live_scene_pos"].to(device)
        if live_scene.ndim == 2:
            live_scene = live_scene.unsqueeze(0)
        live_vel = sample["live_vel_flu"].to(device)
        if live_vel.ndim == 1:
            live_vel = live_vel.unsqueeze(0)
        live_fs = sample["live_flight_state"].to(device)
        if live_fs.ndim == 1:
            live_fs = live_fs.unsqueeze(0)

        live_pose_raw = sample.get("live_pose_se3")
        live_pose_t = None
        if live_pose_raw is not None:
            live_pose_t = torch.as_tensor(live_pose_raw, dtype=torch.float32, device=device)
            if live_pose_t.ndim == 2:
                live_pose_t = live_pose_t.unsqueeze(0)
            elif live_pose_t.ndim == 4 and live_pose_t.shape[1] == 1:
                live_pose_t = live_pose_t.squeeze(1)

        bottleneck = self.compute_bottleneck(
            live_scene_pos=live_scene,
            live_vel_flu=live_vel,
            live_flight_state=live_fs,
            keypoints_body=kp_body,
            ctx_tokens=ctx_keys,
            ctx_values=ctx_values,
            live_pose_se3=live_pose_t,
        )

        # Inicializar hipotesis incremental en identidad I_4: (1, T, 4, 4)
        curr_delta_se3 = (
            torch.eye(4, dtype=torch.float32, device=device)
            .view(1, 1, 4, 4)
            .expand(1, self.horizon, 4, 4)
            .clone()
        )

        pred_flow_6d = torch.zeros((1, self.horizon, 6, 6), dtype=torch.float32, device=device)
        pred_fs_logits = torch.zeros((1, self.horizon, 1), dtype=torch.float32, device=device)

        # Bucle interno de refinamiento DDIM (K pasos sobre SE(2) x R ⊂ SE(3) via SVD)
        for step_i in range(K, 0, -1):
            u_val = float(step_i) / float(K)
            k_norm = torch.tensor([[u_val]], dtype=torch.float32, device=device)
            pred_flow_6d, pred_fs_logits = self.decode_action_flow_step(
                bottleneck=bottleneck,
                curr_delta_se3=curr_delta_se3,
                k_norm=k_norm,
                keypoints_body=kp_body,
            )
            # Paso de actualizacion amortiguado hacia la pose libre de ruido predicha en cada iteracion
            step_gain = 1.0 / float(step_i)
            curr_delta_se3 = apply_keypoint_flow_and_svd_4dof(
                T_curr=curr_delta_se3,
                flow_6d=pred_flow_6d,
                keypoints_body=kp_body,
                step_scale=step_gain,
            )

        pred_delta = curr_delta_se3[0]  # (T, 4, 4)
        pred_action = incremental_to_cumulative_se3(pred_delta)  # (T, 4, 4)
        pred_fs = torch.sigmoid(pred_fs_logits[0])  # (T, 1)

        live_pose = sample["live_pose_se3"]
        if isinstance(live_pose, torch.Tensor) and live_pose.ndim == 3:
            live_pose = live_pose[0]
        world_wp = action_se3_to_world_waypoints_enu_m(
            live_pose_se3=live_pose,
            pred_action_se3=pred_action,
            scale_factor=self.scale_factor,
        )

        return PolicyPrediction(
            pred_action_se3=pred_action.detach().cpu(),
            pred_delta_se3=pred_delta.detach().cpu(),
            pred_flight_state=pred_fs.detach().cpu(),
            pred_world_waypoints_enu_m=world_wp,
            pred_keypoint_flow=pred_flow_6d[0].detach().cpu(),
        )


class OracleReplayPolicy:
    """
    Politica Oraculo determinista para control de cordura del pipeline E/S y lazo cerrado:
    Dado un episodio de referencia `reference_episode`, para cualquier consulta `sample` en el
    instante `t_step`:
      1. Toma las etiquetas `delta_se3` reales del episodio (o calcula la accion relativa exacta
         desde `live_pose_se3` hacia los futuros `T` estados con paso `action_stride`).
      2. Las convierte en flujo cartesiano desacoplado `[Δp_t, Δp_r]` sobre los 6 keypoints.
      3. Las pasa por el estimador SVD 4-DoF (`apply_keypoint_flow_and_svd_4dof`) y la composicion
         incremental -> acumulada (`incremental_to_cumulative_se3`) exactamente igual que la red.
      4. Proyecta los waypoints al mundo ENU en metros reales.

    Permite verificar con 0% de error de aprendizaje que toda la matematica SE(3)/SVD, el
    simulador en lazo cerrado y `validate_episode_data()` aprueban al 100%.
    """

    def __init__(
        self,
        reference_episode: Optional[Any] = None,
        scale_factor: float = DEFAULT_SCALE_FACTOR,
        horizon: int = 8,
        action_stride: int = 1,
    ) -> None:
        self.reference_episode = reference_episode
        self.scale_factor = float(scale_factor)
        self.horizon = max(1, int(horizon))
        self.action_stride = max(1, int(action_stride))

    def predict_actions(
        self,
        sample: Dict[str, Any],
        step_cursor: Optional[int] = None,
    ) -> PolicyPrediction:
        kp_body = sample.get("keypoints_body")
        if kp_body is None:
            kp_body = get_uav_keypoints_body_torch(scale_factor=self.scale_factor)
        else:
            kp_body = torch.as_tensor(kp_body, dtype=torch.float32)
            if kp_body.ndim == 3:
                kp_body = kp_body[0]

        live_pose = sample["live_pose_se3"]
        if isinstance(live_pose, torch.Tensor):
            live_pose_t = live_pose[0] if live_pose.ndim == 3 else live_pose
        else:
            live_pose_t = torch.as_tensor(live_pose, dtype=torch.float32)
            if live_pose_t.ndim == 3:
                live_pose_t = live_pose_t[0]

        stride = max(1, int(sample.get("action_stride", self.action_stride)))
        if "target_delta_se3" in sample and step_cursor is None:
            gt_delta = torch.as_tensor(sample["target_delta_se3"], dtype=torch.float32)
            gt_fs = torch.as_tensor(sample["target_flight_state"], dtype=torch.float32)
            H = int(gt_delta.shape[0])
        elif self.reference_episode is not None:
            ep = self.reference_episode
            ref_poses = ep["agent"].pose_se3.to(dtype=torch.float32)  # (T_ep, 4, 4)
            ref_fs = ep["agent"].flight_state.to(dtype=torch.float32)  # (T_ep, 1)
            T_total = int(ref_poses.shape[0])
            if "context_delta_se3" in sample:
                H = int(sample["context_delta_se3"].shape[-3])
            else:
                H = self.horizon

            if step_cursor is None:
                # Buscar el indice temporal mas cercano a live_pose_t
                dists = torch.norm(ref_poses[:, :3, 3] - live_pose_t[:3, 3].unsqueeze(0), dim=-1)
                t0 = int(torch.argmin(dists).item())
            else:
                t0 = min(T_total - 1, max(0, int(step_cursor)))

            # Construir las H poses relativas exactas desde live_pose_t hacia ref_poses[t0 + k*stride]
            from .se3_diffusion_math import invert_rigid_se3_torch

            live_inv = invert_rigid_se3_torch(live_pose_t)
            cum_list = []
            fs_list = []
            for k in range(1, H + 1):
                tk = min(T_total - 1, t0 + k * stride)
                cum_list.append(live_inv @ ref_poses[tk])
                fs_list.append(ref_fs[tk])
            gt_cum = torch.stack(cum_list, dim=0)  # (H, 4, 4)
            gt_delta = cumulative_to_incremental_se3(gt_cum)  # (H, 4, 4)
            gt_fs = torch.stack(fs_list, dim=0)  # (H, 1)
        else:
            raise ValueError("OracleReplayPolicy requiere `target_delta_se3` en sample o `reference_episode`.")

        # Pasar por la misma tuberia de flujo de 6 keypoints + reconstruccion SVD 4-DoF desde I_4
        identity_seq = torch.eye(4, dtype=torch.float32).unsqueeze(0).expand(H, 4, 4).clone()
        flow_6d = compute_decoupled_keypoint_flow(
            T_curr=identity_seq,
            T_target=gt_delta,
            keypoints_body=kp_body,
        )
        reconstructed_delta = apply_keypoint_flow_and_svd_4dof(
            T_curr=identity_seq,
            flow_6d=flow_6d,
            keypoints_body=kp_body,
            step_scale=1.0,
        )
        reconstructed_action = incremental_to_cumulative_se3(reconstructed_delta)
        world_wp = action_se3_to_world_waypoints_enu_m(
            live_pose_se3=live_pose_t,
            pred_action_se3=reconstructed_action,
            scale_factor=self.scale_factor,
        )

        return PolicyPrediction(
            pred_action_se3=reconstructed_action,
            pred_delta_se3=reconstructed_delta,
            pred_flight_state=gt_fs,
            pred_world_waypoints_enu_m=world_wp,
            pred_keypoint_flow=flow_6d,
        )
