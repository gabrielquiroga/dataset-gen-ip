#!/usr/bin/env python3
"""
audit_dataset.py — Auditor Universal de Datasets y Generador de Particiones Held-Out.

Inspecciona cualquier estructura de dataset (datasets/task_XXXX/episode_YY):
  1. Extrae la firma semántica canónica de cada tarea (vía task_generator.get_task_signature).
  2. Detecta y alerta sobre tareas duplicadas (mismas firmas semánticas).
  3. Audita la variabilidad espacial intra-tarea (verifica que las poses en waypoints.yaml varíen).
  4. Analiza tasas de éxito de vuelo (success, timeout, fallos) a partir de episode_meta.json.
  5. Desglosa la distribución de primitivas y landmarks utilizados.
  6. Genera particiones formales a nivel de firmas semánticas (train_tasks.json, val_tasks.json,
     held_out_tasks.json), garantizando cero fuga de datos (zero data leakage) en evaluación held-out.

Uso:
  python3 audit_dataset.py --dataset-dir datasets
  python3 audit_dataset.py --dataset-dir datasets --output-dir datasets/splits --seed 42
"""

import argparse
import json
import random
import sys
from pathlib import Path
from typing import Dict, Any, List, Set
import yaml

# Importar get_task_signature de task_generator
try:
    from task_generator import get_task_signature
except ImportError:
    script_dir = Path(__file__).resolve().parent
    sys.path.append(str(script_dir))
    from task_generator import get_task_signature


def audit_dataset(
    dataset_dir: Path,
    output_dir: Path,
    train_ratio: float = 0.70,
    val_ratio: float = 0.15,
    held_out_ratio: float = 0.15,
    seed: int = 42,
    no_export: bool = False,
) -> Dict[str, Any]:
    """Ejecuta la auditoría completa del dataset y genera las particiones."""
    if not dataset_dir.is_dir():
        print(f"[ERROR] El directorio del dataset no existe: {dataset_dir}")
        sys.exit(1)

    task_dirs = sorted([d for d in dataset_dir.iterdir() if d.is_dir() and d.name.startswith("task_")])
    if not task_dirs:
        print(f"[WARN] No se encontraron carpetas 'task_*' en {dataset_dir}.")
        return {}

    print("=" * 70)
    print("  AUDITORÍA DE DATASET — INSTANT POLICY UAV")
    print(f"  Directorio: {dataset_dir}")
    print(f"  Tareas encontradas: {len(task_dirs)}")
    print("=" * 70)

    tasks_by_signature: Dict[str, List[str]] = {}
    task_metadata: Dict[str, Dict[str, Any]] = {}
    primitive_counts: Dict[str, int] = {}
    landmark_length_counts: Dict[int, int] = {}
    total_episodes = 0
    total_success = 0
    total_timeout = 0
    total_failed = 0
    tasks_with_identical_waypoints: List[str] = []

    for t_dir in task_dirs:
        task_id = t_dir.name
        cfg_file = t_dir / "task_config.yaml"
        if not cfg_file.is_file():
            print(f"  [WARN] {task_id} no contiene task_config.yaml. Saltando.")
            continue

        with open(cfg_file, "r") as f:
            cfg = yaml.safe_load(f)

        signature = get_task_signature(cfg)
        if signature not in tasks_by_signature:
            tasks_by_signature[signature] = []
        tasks_by_signature[signature].append(task_id)

        # Contar cantidad de landmarks
        n_landmarks = len(cfg.get("landmarks", []))
        landmark_length_counts[n_landmarks] = landmark_length_counts.get(n_landmarks, 0) + 1

        # Contar primitivas
        for prim in cfg.get("primitives", []):
            ptype = prim.get("type", "unknown")
            primitive_counts[ptype] = primitive_counts.get(ptype, 0) + 1

        # Auditar episodios
        ep_dirs = sorted([d for d in t_dir.iterdir() if d.is_dir() and d.name.startswith("episode_")])
        ep_count = len(ep_dirs)
        ep_success = 0
        ep_waypoints_signatures: Set[str] = set()

        for ep in ep_dirs:
            total_episodes += 1
            meta_file = ep / "episode_meta.json"
            if meta_file.is_file():
                try:
                    with open(meta_file, "r") as f:
                        meta = json.load(f)
                    status = meta.get("status", "unknown")
                    if status == "success":
                        total_success += 1
                        ep_success += 1
                    elif status == "timeout":
                        total_timeout += 1
                    else:
                        total_failed += 1
                except Exception:
                    total_failed += 1
            else:
                total_failed += 1

            # Verificar poses físicas de los waypoints
            wp_file = ep / "waypoints.yaml"
            if wp_file.is_file():
                try:
                    with open(wp_file, "r") as f:
                        wp_data = yaml.safe_load(f)
                    # Serializar solo las posiciones x, y, z para comprobar variación
                    poses_token = tuple(
                        (lm.get("id"), round(float(lm.get("x", 0)), 2), round(float(lm.get("y", 0)), 2), round(float(lm.get("z", 0)), 2))
                        for lm in wp_data.get("landmarks", [])
                    )
                    ep_waypoints_signatures.add(str(poses_token))
                except Exception:
                    pass

        # Si hay más de un episodio pero todas las poses físicas fueron idénticas
        if ep_count > 1 and len(ep_waypoints_signatures) == 1:
            tasks_with_identical_waypoints.append(task_id)

        task_metadata[task_id] = {
            "signature": signature,
            "episodes_total": ep_count,
            "episodes_success": ep_success,
            "logic": cfg.get("logic", "A1_DIRECT"),
            "profile": cfg.get("profile", "B1_LINEAR"),
        }

    unique_signatures_count = len(tasks_by_signature)
    duplicate_task_groups = {sig: ids for sig, ids in tasks_by_signature.items() if len(ids) > 1}

    # ----------------------------------------------------------------
    #  Reporte en terminal
    # ----------------------------------------------------------------
    print(f"\n📊 RESUMEN DE INTEGRIDAD SEMÁNTICA:")
    print(f"  Total de carpetas de tareas: {len(task_metadata)}")
    print(f"  Firmas semánticas únicas:   {unique_signatures_count}")
    if duplicate_task_groups:
        print(f"  ⚠️ COLISIONES / DUPLICADOS DETECTADOS: {len(duplicate_task_groups)} firmas compartidas:")
        for sig, ids in list(duplicate_task_groups.items())[:5]:
            print(f"     * {sig[:60]}... -> en {ids}")
    else:
        print("  ✅ CERO DUPLICADOS: Todas las tareas tienen firmas semánticas estrictamente únicas.")

    print(f"\n📈 ESTADÍSTICAS DE VUELOS Y CALIDAD:")
    print(f"  Total episodios grabados:   {total_episodes}")
    if total_episodes > 0:
        rate = (total_success / total_episodes) * 100
        print(f"  Exitosos:                   {total_success} ({rate:.1f}%)")
        print(f"  Timeouts:                   {total_timeout}")
        print(f"  Fallidos:                   {total_failed}")

    print(f"\n🎯 DISTRIBUCIÓN DE PRIMITIVAS:")
    for ptype, count in sorted(primitive_counts.items()):
        print(f"  - {ptype:15s}: {count} apariciones")

    print(f"\n📏 DISTRIBUCIÓN DE CANTIDAD DE LANDMARKS:")
    for n_lm, count in sorted(landmark_length_counts.items()):
        print(f"  - Tareas con {n_lm} landmarks: {count}")

    print(f"\n🔍 VARIABILIDAD ESPACIAL INTRA-TAREA:")
    if tasks_with_identical_waypoints:
        print(f"  ⚠️ ATENCIÓN: {len(tasks_with_identical_waypoints)} tareas tienen waypoints idénticos entre episodios:")
        print(f"     Tareas afectadas: {tasks_with_identical_waypoints[:10]}...")
    else:
        print("  ✅ Conforme: Todos los episodios dentro de cada tarea exhiben variación espacial 3D.")

    # ----------------------------------------------------------------
    #  Generación de Partición (Held-Out Split sin fuga de datos)
    # ----------------------------------------------------------------
    # El split se realiza obligatoriamente sobre el conjunto de firmas semánticas únicas
    unique_signatures = sorted(list(tasks_by_signature.keys()))
    rng = random.Random(seed)
    rng.shuffle(unique_signatures)

    n_sigs = len(unique_signatures)
    n_train = max(1, int(round(n_sigs * train_ratio)))
    n_val = int(round(n_sigs * val_ratio))
    n_held_out = n_sigs - n_train - n_val

    if n_held_out <= 0 and n_sigs >= 3:
        n_held_out = 1
        n_train -= 1

    train_sigs = set(unique_signatures[:n_train])
    val_sigs = set(unique_signatures[n_train:n_train + n_val])
    held_out_sigs = set(unique_signatures[n_train + n_val:])

    # Validación formal de cero fuga
    assert len(train_sigs.intersection(held_out_sigs)) == 0, "Fuga detectada entre train y held_out!"

    train_tasks = [t for sig in train_sigs for t in tasks_by_signature[sig]]
    val_tasks = [t for sig in val_sigs for t in tasks_by_signature[sig]]
    held_out_tasks = [t for sig in held_out_sigs for t in tasks_by_signature[sig]]

    print(f"\n📁 PARTICIÓN FORMAL DEL DATASET (Semilla split={seed}):")
    print(f"  * Train    ({len(train_sigs)} firmas): {len(train_tasks)} tareas ({train_tasks[:5]}...)")
    print(f"  * Val      ({len(val_sigs)} firmas): {len(val_tasks)} tareas ({val_tasks[:5]}...)")
    print(f"  * Held-Out ({len(held_out_sigs)} firmas): {len(held_out_tasks)} tareas ({held_out_tasks[:5]}...)")

    if not no_export:
        output_dir.mkdir(parents=True, exist_ok=True)
        with open(output_dir / "train_tasks.json", "w") as f:
            json.dump({"tasks": sorted(train_tasks), "signatures": sorted(list(train_sigs))}, f, indent=2)

        with open(output_dir / "val_tasks.json", "w") as f:
            json.dump({"tasks": sorted(val_tasks), "signatures": sorted(list(val_sigs))}, f, indent=2)

        with open(output_dir / "held_out_tasks.json", "w") as f:
            json.dump({"tasks": sorted(held_out_tasks), "signatures": sorted(list(held_out_sigs))}, f, indent=2)

        summary = {
            "total_tasks": len(task_metadata),
            "unique_signatures": unique_signatures_count,
            "total_episodes": total_episodes,
            "total_success": total_success,
            "success_rate": (total_success / total_episodes) if total_episodes > 0 else 0.0,
            "primitive_counts": primitive_counts,
            "landmark_length_counts": landmark_length_counts,
            "train_tasks_count": len(train_tasks),
            "val_tasks_count": len(val_tasks),
            "held_out_tasks_count": len(held_out_tasks),
            "split_seed": seed,
        }
        with open(output_dir / "dataset_summary.json", "w") as f:
            json.dump(summary, f, indent=2)

        print(f"\n  💾 Particiones exportadas en: {output_dir}/")
        print(f"     - {output_dir / 'train_tasks.json'}")
        print(f"     - {output_dir / 'val_tasks.json'}")
        print(f"     - {output_dir / 'held_out_tasks.json'}")
        print(f"     - {output_dir / 'dataset_summary.json'}")

    print("=" * 70)
    return {
        "train_tasks": train_tasks,
        "val_tasks": val_tasks,
        "held_out_tasks": held_out_tasks,
        "unique_signatures": unique_signatures_count,
    }


def main():
    parser = argparse.ArgumentParser(
        description="Audita datasets para Instant Policy y genera particiones Train/Val/Held-Out."
    )
    parser.add_argument(
        "--dataset-dir", type=Path, default=Path("datasets"),
        help="Directorio raíz del dataset con las carpetas task_XXXX"
    )
    parser.add_argument(
        "--output-dir", type=Path, default=None,
        help="Directorio donde exportar los archivos JSON de partición (default: <dataset_dir>/splits)"
    )
    parser.add_argument("--train-ratio", type=float, default=0.70, help="Proporción de train (default: 0.70)")
    parser.add_argument("--val-ratio", type=float, default=0.15, help="Proporción de validación (default: 0.15)")
    parser.add_argument("--held-out-ratio", type=float, default=0.15, help="Proporción de held-out (default: 0.15)")
    parser.add_argument("--seed", type=int, default=42, help="Semilla determinista para barajar las firmas")
    parser.add_argument("--no-export", action="store_true", help="Solo audita en terminal sin escribir archivos")
    args = parser.parse_args()

    if args.output_dir is None:
        args.output_dir = args.dataset_dir / "splits"

    audit_dataset(
        dataset_dir=args.dataset_dir,
        output_dir=args.output_dir,
        train_ratio=args.train_ratio,
        val_ratio=args.val_ratio,
        held_out_ratio=args.held_out_ratio,
        seed=args.seed,
        no_export=args.no_export,
    )


if __name__ == "__main__":
    main()
