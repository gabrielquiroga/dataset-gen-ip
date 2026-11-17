# src/uav_ip/uav_ip/icil/scene_encoder.py
from __future__ import annotations

import math
from pathlib import Path
from typing import Dict, Optional, Tuple, Union

import torch
import torch.nn as nn


# ============================================================================
# 1. Codificación Posicional de Fourier 3D (con scale=0.1 interno)
# ============================================================================

class SinusoidalPosEmb3D(nn.Module):
    """
    Codificación posicional sinusoidal 3D para coordenadas o desplazamientos
    relativos en metros reales (s = 1.0).

    Aplica internamente el factor `scale` (por defecto 0.1) antes de proyectar
    en las bandas de frecuencia para evitar aliasing geométrico en el rango
    del sensor LiDAR (R <= 10.0 m).
    """

    def __init__(
        self,
        dim: int = 128,
        scale: float = 0.1,
        max_period: float = 10000.0,
    ) -> None:
        super().__init__()
        if dim < 6:
            raise ValueError(f"dim debe ser >= 6, recibido: {dim}")
        self.dim = int(dim)
        self.scale = float(scale)
        self.max_period = float(max_period)

        # Dividimos la dimensión equitativamente entre los 3 ejes (x, y, z)
        self.dim_per_axis = (self.dim // 6) * 2
        self.pad_dim = self.dim - 3 * self.dim_per_axis

        half_axis = self.dim_per_axis // 2
        exponent = -math.log(self.max_period) * torch.arange(
            half_axis, dtype=torch.float32
        ) / max(half_axis, 1)
        self.register_buffer("freqs", torch.exp(exponent), persistent=False)

    def forward(self, coords: torch.Tensor) -> torch.Tensor:
        """
        Args:
            coords: Tensor (..., 3) en metros.
        Returns:
            emb: Tensor (..., dim).
        """
        if coords.shape[-1] != 3:
            raise ValueError(f"Se esperaba última dimensión 3, recibido: {coords.shape}")

        # Escalado interno para evitar aliasing periódico
        scaled = coords * self.scale  # (..., 3)
        # (..., 3, 1) * (half_axis,) -> (..., 3, half_axis)
        angles = scaled.unsqueeze(-1) * self.freqs.to(dtype=coords.dtype, device=coords.device)
        emb_axes = torch.cat([torch.sin(angles), torch.cos(angles)], dim=-1)  # (..., 3, dim_per_axis)
        emb = emb_axes.flatten(start_dim=-2)  # (..., 3 * dim_per_axis)

        if self.pad_dim > 0:
            pad = torch.zeros(
                (*coords.shape[:-1], self.pad_dim),
                dtype=coords.dtype,
                device=coords.device,
            )
            emb = torch.cat([emb, pad], dim=-1)
        return emb


# ============================================================================
# 2. Utilidades Geométricas de Nube de Puntos (FPS y k-NN en PyTorch)
# ============================================================================

def farthest_point_sample(
    xyz: torch.Tensor,
    npoint: int,
    deterministic: bool = True,
) -> torch.Tensor:
    """
    Farthest Point Sampling (FPS) iterativo sobre lotes en PyTorch.

    Args:
        xyz: Nube de puntos de entrada (B, N, 3).
        npoint: Número de centroides a muestrear (M).
        deterministic: Si True, inicia siempre en el punto más alejado del
            baricentro para garantizar reproducibilidad en evaluación.
    Returns:
        centroids_idx: Índices seleccionados (B, npoint) en int64.
    """
    B, N, _ = xyz.shape
    device = xyz.device
    npoint = int(npoint)

    if N <= npoint:
        base_idx = torch.arange(N, device=device, dtype=torch.long).unsqueeze(0).expand(B, N)
        if N == npoint:
            return base_idx
        pad_idx = base_idx[:, :1].expand(B, npoint - N)
        return torch.cat([base_idx, pad_idx], dim=1)

    centroids = torch.zeros(B, npoint, dtype=torch.long, device=device)
    distance = torch.full((B, N), 1e10, dtype=xyz.dtype, device=device)

    if deterministic:
        mean_xyz = xyz.mean(dim=1, keepdim=True)
        farthest = ((xyz - mean_xyz) ** 2).sum(dim=-1).argmax(dim=-1)
    else:
        farthest = torch.randint(0, N, (B,), dtype=torch.long, device=device)

    batch_indices = torch.arange(B, dtype=torch.long, device=device)
    for i in range(npoint):
        centroids[:, i] = farthest
        centroid = xyz[batch_indices, farthest, :].unsqueeze(1)  # (B, 1, 3)
        dist = ((xyz - centroid) ** 2).sum(dim=-1)  # (B, N)
        distance = torch.minimum(distance, dist)
        farthest = distance.argmax(dim=-1)

    return centroids


def gather_points(points: torch.Tensor, idx: torch.Tensor) -> torch.Tensor:
    """
    Indexa puntos o features según índices por lote.

    Args:
        points: (B, N, C)
        idx: (B, M) o (B, M, K)
    Returns:
        gathered: (B, M, C) o (B, M, K, C)
    """
    B, N, C = points.shape
    if idx.dim() == 2:
        idx_exp = idx.unsqueeze(-1).expand(B, idx.shape[1], C)
        return torch.gather(points, 1, idx_exp)
    elif idx.dim() == 3:
        _, M, K = idx.shape
        idx_flat = idx.reshape(B, M * K).unsqueeze(-1).expand(B, M * K, C)
        out = torch.gather(points, 1, idx_flat)
        return out.reshape(B, M, K, C)
    raise ValueError(f"idx debe tener 2 o 3 dimensiones, recibido: {idx.shape}")


def knn_group(
    support_xyz: torch.Tensor,
    query_xyz: torch.Tensor,
    k: int,
    support_feats: Optional[torch.Tensor] = None,
) -> Tuple[torch.Tensor, Optional[torch.Tensor], torch.Tensor]:
    """
    Agrupa los k vecinos más cercanos de `support_xyz` alrededor de cada punto
    en `query_xyz`.

    Args:
        support_xyz: (B, N, 3)
        query_xyz: (B, M, 3)
        k: Número de vecinos a buscar.
        support_feats: (B, N, C) opcional.
    Returns:
        rel_xyz: (B, M, k, 3) desplazamientos relativos (support - query).
        grouped_feats: (B, M, k, C) o None.
        knn_idx: (B, M, k) índices de los vecinos más cercanos.
    """
    k_eff = min(int(k), support_xyz.shape[1])
    dists = torch.cdist(query_xyz, support_xyz, p=2.0)  # (B, M, N)
    _, knn_idx = torch.topk(dists, k=k_eff, dim=-1, largest=False, sorted=True)

    neighbor_xyz = gather_points(support_xyz, knn_idx)  # (B, M, k_eff, 3)
    rel_xyz = neighbor_xyz - query_xyz.unsqueeze(2)

    grouped_feats = None
    if support_feats is not None:
        grouped_feats = gather_points(support_feats, knn_idx)  # (B, M, k_eff, C)

    return rel_xyz, grouped_feats, knn_idx


# ============================================================================
# 3. Encoder Jerárquico de Escena \phi_e (Set Abstraction)
# ============================================================================

class SetAbstractionLayer(nn.Module):
    """
    Capa Set Abstraction estilo PointNet++ con codificación de Fourier 3D:
      1. Submuestrea `num_centroids` puntos mediante FPS.
      2. Agrupa los `k_neighbors` vecinos más cercanos.
      3. Codifica desplazamientos relativos con `SinusoidalPosEmb3D`.
      4. Aplica MLP por punto y max-pooling sobre la vecindad local.
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        num_centroids: int,
        k_neighbors: int = 16,
        pos_scale: float = 0.1,
    ) -> None:
        super().__init__()
        self.in_channels = int(in_channels)
        self.out_channels = int(out_channels)
        self.num_centroids = int(num_centroids)
        self.k_neighbors = int(k_neighbors)

        self.pos_emb = SinusoidalPosEmb3D(dim=out_channels, scale=pos_scale)
        mlp_in = out_channels + 3 + self.in_channels
        self.mlp = nn.Sequential(
            nn.Linear(mlp_in, out_channels),
            nn.LayerNorm(out_channels),
            nn.GELU(),
            nn.Linear(out_channels, out_channels),
            nn.LayerNorm(out_channels),
            nn.GELU(),
        )

    def forward(
        self,
        xyz: torch.Tensor,
        feats: Optional[torch.Tensor] = None,
        deterministic_fps: bool = True,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            xyz: (B, N, 3)
            feats: (B, N, C_in) o None
        Returns:
            centroids: (B, M, 3)
            new_feats: (B, M, out_channels)
        """
        fps_idx = farthest_point_sample(xyz, self.num_centroids, deterministic=deterministic_fps)
        centroids = gather_points(xyz, fps_idx)  # (B, M, 3)

        rel_xyz, grouped_feats, _ = knn_group(
            support_xyz=xyz,
            query_xyz=centroids,
            k=self.k_neighbors,
            support_feats=feats,
        )
        rel_emb = self.pos_emb(rel_xyz)  # (B, M, k, out_channels)

        if grouped_feats is not None:
            cat_in = torch.cat([rel_emb, rel_xyz, grouped_feats], dim=-1)
        else:
            cat_in = torch.cat([rel_emb, rel_xyz], dim=-1)

        point_feats = self.mlp(cat_in)  # (B, M, k, out_channels)
        pooled_feats = point_feats.max(dim=2).values  # (B, M, out_channels)
        return centroids, pooled_feats


class SceneEncoderPhi(nn.Module):
    """
    Encoder Geométrico Local de Escena \phi_e.
    Comprime una nube LiDAR egocéntrica (B, 2048, 3) en M centroides 3D y
    sus descriptores geométricos (B, M, d_model).
    """

    def __init__(
        self,
        num_centroids: int = 32,
        d_model: int = 128,
        num_mid_points: int = 128,
        k_stage1: int = 16,
        k_stage2: int = 16,
        pos_scale: float = 0.1,
    ) -> None:
        super().__init__()
        self.num_centroids = int(num_centroids)
        self.d_model = int(d_model)
        self.num_mid_points = int(num_mid_points)
        self.k_stage1 = int(k_stage1)
        self.k_stage2 = int(k_stage2)
        self.pos_scale = float(pos_scale)

        mid_dim = max(self.d_model // 2, 32)
        self.sa1 = SetAbstractionLayer(
            in_channels=0,
            out_channels=mid_dim,
            num_centroids=self.num_mid_points,
            k_neighbors=self.k_stage1,
            pos_scale=self.pos_scale,
        )
        self.sa2 = SetAbstractionLayer(
            in_channels=mid_dim,
            out_channels=self.d_model,
            num_centroids=self.num_centroids,
            k_neighbors=self.k_stage2,
            pos_scale=self.pos_scale,
        )
        self.centroid_pos_emb = SinusoidalPosEmb3D(dim=self.d_model, scale=self.pos_scale)
        self.out_proj = nn.Sequential(
            nn.Linear(self.d_model * 2, self.d_model),
            nn.LayerNorm(self.d_model),
            nn.GELU(),
            nn.Linear(self.d_model, self.d_model),
            nn.LayerNorm(self.d_model),
        )

    def forward(
        self,
        cloud: torch.Tensor,
        deterministic_fps: Optional[bool] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            cloud: Nube de puntos (B, N, 3) en metros (marco FLU).
            deterministic_fps: Si None, usa `not self.training`.
        Returns:
            centroids: (B, M, 3) coordenadas 3D en metros.
            features: (B, M, d_model) descriptores locales.
        """
        if deterministic_fps is None:
            deterministic_fps = not self.training

        mid_xyz, mid_feats = self.sa1(cloud, feats=None, deterministic_fps=deterministic_fps)
        centroids, local_feats = self.sa2(
            mid_xyz, feats=mid_feats, deterministic_fps=deterministic_fps
        )
        pos_feats = self.centroid_pos_emb(centroids)
        features = self.out_proj(torch.cat([local_feats, pos_feats], dim=-1))
        return centroids, features


# ============================================================================
# 4. Decoder Implícito de Ocupancia 3D \psi_e (Solo Pre-entrenamiento)
# ============================================================================

class OccupancyDecoderPsi(nn.Module):
    """
    Decoder de ocupancia implícita \psi_e para pre-entrenar \phi_e.
    Para cada punto query 3D q \in R^3, interpola/agrega los k centroides más
    cercanos ponderados por su desplazamiento relativo codificado en Fourier.
    """

    def __init__(
        self,
        d_model: int = 128,
        k_neighbors: int = 4,
        pos_scale: float = 0.1,
    ) -> None:
        super().__init__()
        self.d_model = int(d_model)
        self.k_neighbors = int(k_neighbors)
        self.pos_emb = SinusoidalPosEmb3D(dim=self.d_model, scale=pos_scale)

        in_dim = self.d_model * 2 + 4  # feat_j + pos_emb(c_j - q) + (c_j - q) + ||c_j - q||
        self.neighbor_mlp = nn.Sequential(
            nn.Linear(in_dim, self.d_model),
            nn.LayerNorm(self.d_model),
            nn.GELU(),
            nn.Linear(self.d_model, self.d_model),
            nn.GELU(),
        )
        self.attn_score = nn.Linear(self.d_model, 1)
        self.head = nn.Sequential(
            nn.Linear(self.d_model, self.d_model),
            nn.LayerNorm(self.d_model),
            nn.GELU(),
            nn.Linear(self.d_model, 1),
        )

    def forward(
        self,
        centroids: torch.Tensor,
        features: torch.Tensor,
        query_xyz: torch.Tensor,
    ) -> torch.Tensor:
        """
        Args:
            centroids: (B, M, 3)
            features: (B, M, d_model)
            query_xyz: (B, Q, 3) puntos de consulta en metros.
        Returns:
            logits: (B, Q, 1) logits de ocupancia (aplicar BCEWithLogitsLoss).
        """
        rel_xyz, grouped_feats, _ = knn_group(
            support_xyz=centroids,
            query_xyz=query_xyz,
            k=self.k_neighbors,
            support_feats=features,
        )
        dist = torch.linalg.norm(rel_xyz, dim=-1, keepdim=True)  # (B, Q, k, 1)
        rel_emb = self.pos_emb(rel_xyz)  # (B, Q, k, d_model)

        cat = torch.cat([grouped_feats, rel_emb, rel_xyz, dist], dim=-1)
        msg = self.neighbor_mlp(cat)  # (B, Q, k, d_model)
        weights = torch.softmax(self.attn_score(msg), dim=2)  # (B, Q, k, 1)
        agg = (weights * msg).sum(dim=2)  # (B, Q, d_model)
        return self.head(agg)


# ============================================================================
# 5. Generador de Escenas 3D Sintéticas para Pre-entrenamiento de \phi_e
# ============================================================================

def _sample_primitive_points(
    prim_type: int,
    n_pts: int,
    device: torch.device,
) -> torch.Tensor:
    """
    Muestrea puntos sobre la superficie canónica de una primitiva del dominio UAV
    en metros reales (s = 1.0):
      0: pole (cilindro vertical r=0.10m, h=3.0m)
      1: square_gate (marco de 4 barras 1.5m x 1.6m, espesor 0.10m)
      2: circular_gate (toroide R=0.75m, r=0.05m en plano Y-Z)
      3: landing_pad (plataforma 2.0m x 2.0m x 0.10m)
    """
    if prim_type == 0:  # Pole
        theta = torch.empty(n_pts, device=device).uniform_(0.0, 2.0 * math.pi)
        z = torch.empty(n_pts, device=device).uniform_(-1.5, 1.5)
        r = 0.10
        return torch.stack([r * torch.cos(theta), r * torch.sin(theta), z], dim=-1)

    if prim_type == 1:  # Square gate (4 barras en plano Y-Z)
        n_bar = n_pts // 4
        rem = n_pts - 3 * n_bar
        # Barra superior (z = +0.75)
        top = torch.stack([
            torch.empty(n_bar, device=device).uniform_(-0.05, 0.05),
            torch.empty(n_bar, device=device).uniform_(-0.75, 0.75),
            torch.empty(n_bar, device=device).uniform_(0.70, 0.80),
        ], dim=-1)
        # Barra inferior (z = -0.75)
        bot = torch.stack([
            torch.empty(n_bar, device=device).uniform_(-0.05, 0.05),
            torch.empty(n_bar, device=device).uniform_(-0.75, 0.75),
            torch.empty(n_bar, device=device).uniform_(-0.80, -0.70),
        ], dim=-1)
        # Pilar izquierdo (y = +0.70)
        left = torch.stack([
            torch.empty(n_bar, device=device).uniform_(-0.05, 0.05),
            torch.empty(n_bar, device=device).uniform_(0.65, 0.75),
            torch.empty(n_bar, device=device).uniform_(-0.70, 0.70),
        ], dim=-1)
        # Pilar derecho (y = -0.70)
        right = torch.stack([
            torch.empty(rem, device=device).uniform_(-0.05, 0.05),
            torch.empty(rem, device=device).uniform_(-0.75, -0.65),
            torch.empty(rem, device=device).uniform_(-0.70, 0.70),
        ], dim=-1)
        return torch.cat([top, bot, left, right], dim=0)

    if prim_type == 2:  # Circular gate (toroide en plano Y-Z)
        u = torch.empty(n_pts, device=device).uniform_(0.0, 2.0 * math.pi)
        v = torch.empty(n_pts, device=device).uniform_(0.0, 2.0 * math.pi)
        R, r = 0.75, 0.05
        x = r * torch.sin(v)
        y = (R + r * torch.cos(v)) * torch.cos(u)
        z = (R + r * torch.cos(v)) * torch.sin(u)
        return torch.stack([x, y, z], dim=-1)

    # Landing pad (2.0m x 2.0m x 0.10m)
    x = torch.empty(n_pts, device=device).uniform_(-1.0, 1.0)
    y = torch.empty(n_pts, device=device).uniform_(-1.0, 1.0)
    z = torch.empty(n_pts, device=device).uniform_(-0.05, 0.05)
    return torch.stack([x, y, z], dim=-1)


@torch.no_grad()
def sample_synthetic_occupancy_batch(
    batch_size: int = 16,
    num_cloud_points: int = 2048,
    num_queries: int = 1024,
    surface_threshold_m: float = 0.08,
    device: Optional[torch.device] = None,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Genera un lote de escenas 3D sintéticas en metros reales (s = 1.0) con
    múltiples obstáculos del dominio UAV y puntos query etiquetados por ocupancia.

    Returns:
        clouds: (B, num_cloud_points, 3) nube LiDAR de entrada.
        queries: (B, num_queries, 3) puntos 3D de consulta.
        labels: (B, num_queries, 1) etiquetas binarias {0.0, 1.0}.
    """
    if device is None:
        device = torch.device("cpu")

    dense_pts_per_obj = 1024
    num_objs = 4
    clouds_list = []
    queries_list = []
    labels_list = []

    for _ in range(batch_size):
        scene_chunks = []
        for obj_idx in range(num_objs):
            prim_type = int(torch.randint(0, 4, (1,), device=device).item())
            pts = _sample_primitive_points(prim_type, dense_pts_per_obj, device=device)

            yaw = float(torch.empty(1, device=device).uniform_(-math.pi, math.pi).item())
            c, s = math.cos(yaw), math.sin(yaw)
            rot = torch.tensor([[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]], device=device)

            tx = float(torch.empty(1, device=device).uniform_(-5.5, 5.5).item())
            ty = float(torch.empty(1, device=device).uniform_(-5.5, 5.5).item())
            tz = float(torch.empty(1, device=device).uniform_(-1.2, 1.8).item())
            trans = torch.tensor([tx, ty, tz], device=device)

            scene_chunks.append(pts @ rot.T + trans)

        dense_surface = torch.cat(scene_chunks, dim=0)  # (4096, 3)

        # Submuestreo aleatorio uniforme a 2048 puntos para la nube observada
        perm = torch.randperm(dense_surface.shape[0], device=device)[:num_cloud_points]
        cloud = dense_surface[perm]

        # Muestreo balanceado de queries:
        # - 50% cerca de la superficie (sigma = 0.025 m -> ocupados)
        # - 25% en cáscara intermedia (sigma = 0.25 m -> borde libre/ocupado)
        # - 25% uniforme en el volumen [-6, 6] x [-6, 6] x [-1.8, 2.5] m
        q_pos_n = num_queries // 2
        q_shell_n = num_queries // 4
        q_free_n = num_queries - q_pos_n - q_shell_n

        idx_pos = torch.randint(0, dense_surface.shape[0], (q_pos_n,), device=device)
        q_pos = dense_surface[idx_pos] + torch.randn(q_pos_n, 3, device=device) * 0.025

        idx_shell = torch.randint(0, dense_surface.shape[0], (q_shell_n,), device=device)
        q_shell = dense_surface[idx_shell] + torch.randn(q_shell_n, 3, device=device) * 0.25

        q_free = torch.stack([
            torch.empty(q_free_n, device=device).uniform_(-6.0, 6.0),
            torch.empty(q_free_n, device=device).uniform_(-6.0, 6.0),
            torch.empty(q_free_n, device=device).uniform_(-1.8, 2.5),
        ], dim=-1)

        queries = torch.cat([q_pos, q_shell, q_free], dim=0)  # (Q, 3)

        # Etiqueta exacta por distancia mínima a la superficie densa
        min_dists = torch.cdist(queries.unsqueeze(0), dense_surface.unsqueeze(0)).squeeze(0).min(dim=-1).values
        labels = (min_dists <= surface_threshold_m).to(torch.float32).unsqueeze(-1)  # (Q, 1)

        clouds_list.append(cloud)
        queries_list.append(queries)
        labels_list.append(labels)

    return (
        torch.stack(clouds_list, dim=0),
        torch.stack(queries_list, dim=0),
        torch.stack(labels_list, dim=0),
    )


def load_pretrained_scene_encoder(
    checkpoint_path: Union[str, Path],
    device: Optional[torch.device] = None,
    freeze: bool = True,
) -> Tuple[SceneEncoderPhi, Dict]:
    """
    Carga un `SceneEncoderPhi` desde un checkpoint guardado por
    `pretrain_scene_encoder.py` y opcionalmente congela sus pesos.
    """
    ckpt_path = Path(checkpoint_path)
    if not ckpt_path.is_file():
        raise FileNotFoundError(f"No se encontró el checkpoint de scene_encoder: {ckpt_path}")

    ckpt = torch.load(ckpt_path, map_location=device or "cpu", weights_only=False)
    encoder = SceneEncoderPhi(
        num_centroids=int(ckpt.get("num_centroids", 32)),
        d_model=int(ckpt.get("d_model", 128)),
        num_mid_points=int(ckpt.get("num_mid_points", 128)),
        k_stage1=int(ckpt.get("k_stage1", 16)),
        k_stage2=int(ckpt.get("k_stage2", 16)),
        pos_scale=float(ckpt.get("pos_scale", 0.1)),
    )
    encoder.load_state_dict(ckpt["model_state_dict"])
    if device is not None:
        encoder = encoder.to(device)
    if freeze:
        encoder.eval()
        for param in encoder.parameters():
            param.requires_grad = False
    return encoder, ckpt