# src/uav_ip/test/icil/test_scene_encoder.py
from pathlib import Path
import sys

import pytest

torch = pytest.importorskip("torch")

REPO_ROOT = Path(__file__).resolve().parents[4]
PKG_ROOT = REPO_ROOT / "src" / "uav_ip"
if str(PKG_ROOT) not in sys.path:
    sys.path.insert(0, str(PKG_ROOT))

from uav_ip.icil.scene_encoder import (
    OccupancyDecoderPsi,
    SceneEncoderPhi,
    SinusoidalPosEmb3D,
    farthest_point_sample,
    knn_group,
    load_pretrained_scene_encoder,
    sample_synthetic_occupancy_batch,
)


def test_sinusoidal_pos_emb_3d_shapes_and_scale() -> None:
    emb_fn = SinusoidalPosEmb3D(dim=128, scale=0.1)
    x = torch.randn(4, 32, 16, 3) * 5.0  # desplazamientos hasta ~10m en metros
    out = emb_fn(x)
    assert out.shape == (4, 32, 16, 128)
    assert torch.isfinite(out).all()

    # Verificar que dos desplazamientos separados por 2*pi metros NO colapsan gracias a scale=0.1
    p1 = torch.tensor([[1.0, 0.0, 0.0]])
    p2 = torch.tensor([[1.0 + 2.0 * torch.pi, 0.0, 0.0]])
    e1 = emb_fn(p1)
    e2 = emb_fn(p2)
    assert not torch.allclose(e1, e2, atol=1e-2)


def test_fps_and_knn_group() -> None:
    torch.manual_seed(42)
    xyz = torch.randn(2, 256, 3)
    fps_idx = farthest_point_sample(xyz, npoint=32, deterministic=True)
    assert fps_idx.shape == (2, 32)
    # Verificar determinismo
    fps_idx_2 = farthest_point_sample(xyz, npoint=32, deterministic=True)
    assert torch.equal(fps_idx, fps_idx_2)

    feats = torch.randn(2, 256, 16)
    query_xyz = xyz[:, :32, :]
    rel_xyz, grouped_feats, knn_idx = knn_group(xyz, query_xyz, k=8, support_feats=feats)
    assert rel_xyz.shape == (2, 32, 8, 3)
    assert grouped_feats is not None and grouped_feats.shape == (2, 32, 8, 16)
    assert knn_idx.shape == (2, 32, 8)
    # El vecino más cercano de cada query (que pertenece a xyz) debe tener distancia 0
    assert rel_xyz[:, :, 0, :].abs().max().item() < 1e-5


@pytest.mark.parametrize("num_centroids", [16, 32, 64])
def test_scene_encoder_phi_multi_m(num_centroids: int) -> None:
    torch.manual_seed(7)
    encoder = SceneEncoderPhi(num_centroids=num_centroids, d_model=128)
    encoder.eval()

    cloud = torch.randn(2, 2048, 3) * 3.0
    with torch.no_grad():
        centroids, feats = encoder(cloud)

    assert centroids.shape == (2, num_centroids, 3)
    assert feats.shape == (2, num_centroids, 128)
    assert torch.isfinite(centroids).all()
    assert torch.isfinite(feats).all()


def test_occupancy_decoder_psi_and_gradients() -> None:
    torch.manual_seed(13)
    encoder = SceneEncoderPhi(num_centroids=32, d_model=64, num_mid_points=64)
    decoder = OccupancyDecoderPsi(d_model=64, k_neighbors=4)

    clouds, queries, labels = sample_synthetic_occupancy_batch(
        batch_size=2, num_cloud_points=512, num_queries=128
    )
    assert clouds.shape == (2, 512, 3)
    assert queries.shape == (2, 128, 3)
    assert labels.shape == (2, 128, 1)

    centroids, feats = encoder(clouds)
    logits = decoder(centroids, feats, queries)
    assert logits.shape == (2, 128, 1)

    loss = torch.nn.functional.binary_cross_entropy_with_logits(logits, labels)
    loss.backward()

    enc_grad_norm = sum(
        p.grad.norm().item() for p in encoder.parameters() if p.grad is not None
    )
    dec_grad_norm = sum(
        p.grad.norm().item() for p in decoder.parameters() if p.grad is not None
    )
    assert enc_grad_norm > 0.0
    assert dec_grad_norm > 0.0


def test_pretrained_checkpoint_if_present(tmp_path: Path) -> None:
    # Prueba de guardado y carga congelada de SceneEncoderPhi
    encoder = SceneEncoderPhi(num_centroids=32, d_model=128)
    ckpt_file = tmp_path / "scene_encoder_test.pt"
    torch.save(
        {
            "model_state_dict": encoder.state_dict(),
            "num_centroids": 32,
            "d_model": 128,
            "num_mid_points": 128,
            "k_stage1": 16,
            "k_stage2": 16,
            "pos_scale": 0.1,
            "val_accuracy": 0.95,
        },
        ckpt_file,
    )
    loaded, meta = load_pretrained_scene_encoder(ckpt_file, freeze=True)
    assert meta["num_centroids"] == 32
    assert not loaded.training
    assert all(not p.requires_grad for p in loaded.parameters())

    # Si ya existe el checkpoint oficial pre-entrenado, verificar que supera 90% de accuracy
    real_ckpt = REPO_ROOT / "checkpoints" / "scene_encoder.pt"
    if real_ckpt.is_file():
        _, real_meta = load_pretrained_scene_encoder(real_ckpt, freeze=True)
        assert float(real_meta.get("val_accuracy", 0.0)) >= 0.90