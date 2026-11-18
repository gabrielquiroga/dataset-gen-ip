#!/usr/bin/env python3
# src/uav_ip/scripts/icil/pretrain_scene_encoder.py
"""
Pre-entrenamiento autosupervisado de SceneEncoderPhi (\phi_e) mediante
predicción de ocupancia 3D implícita con OccupancyDecoderPsi (\psi_e).

Uso:
    ~/venv_ip/bin/python3 src/uav_ip/scripts/icil/pretrain_scene_encoder.py \
        --epochs 25 --steps-per-epoch 40 --num-centroids 32 \
        --save-path checkpoints/scene_encoder.pt
"""
from __future__ import annotations

import argparse
from pathlib import Path
import sys
import time

import torch
import torch.nn as nn

REPO_ROOT = Path(__file__).resolve().parents[4]
PKG_ROOT = REPO_ROOT / "src" / "uav_ip"
if str(PKG_ROOT) not in sys.path:
    sys.path.insert(0, str(PKG_ROOT))

from uav_ip.icil.scene_encoder import (
    OccupancyDecoderPsi,
    SceneEncoderPhi,
    sample_synthetic_occupancy_batch,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Pre-entrenar SceneEncoderPhi por ocupancia 3D.")
    parser.add_argument("--epochs", type=int, default=25, help="Número de épocas.")
    parser.add_argument("--steps-per-epoch", type=int, default=40, help="Batches por época.")
    parser.add_argument("--val-batches", type=int, default=12, help="Batches de validación.")
    parser.add_argument("--batch-size", type=int, default=16, help="Tamaño de lote.")
    parser.add_argument("--num-centroids", type=int, default=32, help="Número de centroides M (16, 32, 64).")
    parser.add_argument("--d-model", type=int, default=128, help="Dimensión latente d_model.")
    parser.add_argument("--num-mid-points", type=int, default=128, help="Puntos en nivel 1 de Set Abstraction.")
    parser.add_argument("--k-stage1", type=int, default=16, help="Vecinos k-NN en nivel 1.")
    parser.add_argument("--k-stage2", type=int, default=16, help="Vecinos k-NN en nivel 2.")
    parser.add_argument("--pos-scale", type=float, default=0.1, help="Escala interna en SinusoidalPosEmb3D.")
    parser.add_argument("--num-queries", type=int, default=1024, help="Puntos query 3D por escena.")
    parser.add_argument("--lr", type=float, default=1e-3, help="Learning rate para AdamW.")
    parser.add_argument(
        "--save-path",
        type=str,
        default=str(REPO_ROOT / "checkpoints" / "scene_encoder.pt"),
        help="Ruta donde guardar el checkpoint pre-entrenado.",
    )
    parser.add_argument("--device", type=str, default="auto", help="'cpu', 'cuda' o 'auto'.")
    parser.add_argument("--seed", type=int, default=42, help="Semilla aleatoria.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    torch.manual_seed(args.seed)

    if args.device == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(args.device)

    encoder = SceneEncoderPhi(
        num_centroids=args.num_centroids,
        d_model=args.d_model,
        num_mid_points=args.num_mid_points,
        k_stage1=args.k_stage1,
        k_stage2=args.k_stage2,
        pos_scale=args.pos_scale,
    ).to(device)

    decoder = OccupancyDecoderPsi(
        d_model=args.d_model,
        k_neighbors=4,
        pos_scale=args.pos_scale,
    ).to(device)

    params = list(encoder.parameters()) + list(decoder.parameters())
    optimizer = torch.optim.AdamW(params, lr=args.lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=max(args.epochs * args.steps_per_epoch, 1)
    )
    criterion = nn.BCEWithLogitsLoss()

    # Pre-generamos un set fijo de validación para medir el progreso de forma consistente
    val_sets = [
        sample_synthetic_occupancy_batch(
            batch_size=args.batch_size,
            num_queries=args.num_queries,
            device=device,
        )
        for _ in range(args.val_batches)
    ]

    save_path = Path(args.save_path)
    save_path.parent.mkdir(parents=True, exist_ok=True)

    best_val_acc = 0.0
    best_val_loss = float("inf")

    print(
        f"[Pretrain SceneEncoderPhi] device={device} | M={args.num_centroids} | "
        f"d_model={args.d_model} | pos_scale={args.pos_scale}"
    )

    for epoch in range(1, args.epochs + 1):
        t0 = time.time()
        encoder.train()
        decoder.train()
        train_loss_sum = 0.0
        train_acc_sum = 0.0

        for _ in range(args.steps_per_epoch):
            clouds, queries, labels = sample_synthetic_occupancy_batch(
                batch_size=args.batch_size,
                num_queries=args.num_queries,
                device=device,
            )
            optimizer.zero_grad(set_to_none=True)
            centroids, feats = encoder(clouds, deterministic_fps=False)
            logits = decoder(centroids, feats, queries)
            loss = criterion(logits, labels)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(params, max_norm=1.0)
            optimizer.step()
            scheduler.step()

            with torch.no_grad():
                preds = (logits > 0.0).to(torch.float32)
                acc = (preds == labels).to(torch.float32).mean().item()
                train_loss_sum += float(loss.item())
                train_acc_sum += acc

        train_loss = train_loss_sum / args.steps_per_epoch
        train_acc = train_acc_sum / args.steps_per_epoch

        # Evaluación en validación
        encoder.eval()
        decoder.eval()
        val_loss_sum = 0.0
        val_acc_sum = 0.0
        with torch.no_grad():
            for v_clouds, v_queries, v_labels in val_sets:
                v_centroids, v_feats = encoder(v_clouds, deterministic_fps=True)
                v_logits = decoder(v_centroids, v_feats, v_queries)
                v_loss = criterion(v_logits, v_labels)
                v_preds = (v_logits > 0.0).to(torch.float32)
                v_acc = (v_preds == v_labels).to(torch.float32).mean().item()
                val_loss_sum += float(v_loss.item())
                val_acc_sum += v_acc

        val_loss = val_loss_sum / len(val_sets)
        val_acc = val_acc_sum / len(val_sets)
        dt = time.time() - t0

        if val_acc >= best_val_acc:
            best_val_acc = val_acc
            best_val_loss = val_loss
            torch.save(
                {
                    "model_state_dict": encoder.state_dict(),
                    "decoder_state_dict": decoder.state_dict(),
                    "num_centroids": args.num_centroids,
                    "d_model": args.d_model,
                    "num_mid_points": args.num_mid_points,
                    "k_stage1": args.k_stage1,
                    "k_stage2": args.k_stage2,
                    "pos_scale": args.pos_scale,
                    "val_accuracy": best_val_acc,
                    "val_loss": best_val_loss,
                    "epoch": epoch,
                },
                save_path,
            )

        print(
            f"Epoch {epoch:02d}/{args.epochs:02d} ({dt:.1f}s) | "
            f"train_loss={train_loss:.4f} train_acc={train_acc * 100:.2f}% | "
            f"val_loss={val_loss:.4f} val_acc={val_acc * 100:.2f}% | "
            f"best_val_acc={best_val_acc * 100:.2f}%"
        )

    print(f"\n[OK] Checkpoint guardado en: {save_path} (best_val_acc={best_val_acc * 100:.2f}%)")


if __name__ == "__main__":
    main()