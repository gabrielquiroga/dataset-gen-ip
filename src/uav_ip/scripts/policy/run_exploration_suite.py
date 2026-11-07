#!/usr/bin/env python3
"""
run_exploration_suite.py — Thin wrapper CLI para ejecutar la Suite de Exploracion Pre-Hito 4
o probar combinaciones arbitrarias de hiperparametros (`L`, `T`, `action_stride`, `N_demos`,
`T_exec`, `K`, `M`, `d_model`) dentro del entorno virtual `~/venv_ip` sin regenerar `heterodata.pt`.

Ejemplos de uso:
  1) Ejecutar la suite canonica completa de 5 experimentos:
       ~/venv_ip/bin/python3 src/uav_ip/scripts/policy/run_exploration_suite.py \
         --kinematic-dir datasets/kinematic \
         --sitl-dir datasets/sitl \
         --output-report docs/EXPLORACION_RED_NEURONAL.md

  2) Probar rapidamente una combinacion personalizada de hiperparametros (ej. L adaptativo y T=12):
       ~/venv_ip/bin/python3 src/uav_ip/scripts/policy/run_exploration_suite.py \
         --custom-eval-only \
         --num-keyframes adaptive --adaptive-arc-m 1.5 \
         --horizon 12 --action-stride 1 \
         --execution-horizon 6 --num-context-demos 2 --num-ddim-steps 4
"""

import argparse
import json
from pathlib import Path
import sys

# Asegurar que `src/uav_ip` sea importable al invocar el script directamente
_PKG_ROOT = Path(__file__).resolve().parent.parent.parent
if str(_PKG_ROOT) not in sys.path:
    sys.path.insert(0, str(_PKG_ROOT))

try:
    import torch  # noqa: F401
except ImportError as exc:
    sys.stderr.write(
        "ERROR: `torch` no esta instalado en este interprete de Python.\n"
        "Ejecuta los scripts de `scripts/policy/` usando el entorno virtual de ML:\n"
        "    ~/venv_ip/bin/python3 src/uav_ip/scripts/policy/run_exploration_suite.py ...\n"
    )
    raise SystemExit(1) from exc

from uav_ip.policy.exploration_suite import (  # noqa: E402
    generate_markdown_exploration_report,
    run_all_explorations,
    run_custom_hyperparameter_probe,
)
from uav_ip.policy.in_context_dataset import parse_num_keyframes_arg  # noqa: E402


def main() -> None:
    repo_root = _PKG_ROOT.parent.parent
    parser = argparse.ArgumentParser(
        description="Ejecuta la suite de exploracion Pre-Hito 4 o evalua combinaciones variables "
        "de keyframes (L fijo o 'adaptive'), horizonte (T), stride, N_demos, T_exec y pasos DDIM (K).",
    )
    parser.add_argument(
        "--kinematic-dir",
        type=Path,
        default=repo_root / "datasets" / "kinematic",
        help="Directorio del dataset cinematico auditado (default: datasets/kinematic)",
    )
    parser.add_argument(
        "--sitl-dir",
        type=Path,
        default=repo_root / "datasets" / "sitl",
        help="Directorio del dataset PX4 SITL auditado (default: datasets/sitl)",
    )
    parser.add_argument(
        "--output-report",
        type=Path,
        default=repo_root / "docs" / "EXPLORACION_RED_NEURONAL.md",
        help="Ruta del reporte Markdown de salida (default: docs/EXPLORACION_RED_NEURONAL.md)",
    )
    parser.add_argument(
        "--output-json",
        type=Path,
        default=None,
        help="Ruta opcional para exportar en JSON el resultado de `--custom-eval-only`",
    )
    parser.add_argument(
        "--max-kinematic-tasks",
        type=int,
        default=40,
        help="Numero de tareas cinematicas a escanear en el Experimento 1 (default: 40 = 200 episodios)",
    )
    parser.add_argument(
        "--overfit-epochs",
        type=int,
        default=35,
        help="Epocas para el entrenamiento rapido de validacion (default: 35)",
    )
    parser.add_argument(
        "--num-keyframes",
        type=str,
        default="10",
        help="Cantidad L de keyframes por demo: entero positivo (ej. 10, 16, 24) o 'adaptive' (default: 10)",
    )
    parser.add_argument(
        "--adaptive-arc-m",
        type=float,
        default=1.5,
        help="Separacion media de arco en metros por keyframe cuando `--num-keyframes adaptive` (default: 1.5)",
    )
    parser.add_argument(
        "--horizon",
        type=int,
        default=8,
        help="Cantidad T de pasos futuros del horizonte de prediccion (ej. 6, 8, 12, 16; default: 8)",
    )
    parser.add_argument(
        "--action-stride",
        type=int,
        default=1,
        help="Separacion temporal en ticks de 10 Hz entre pasos del horizonte (1=0.1s, 2=0.2s; default: 1)",
    )
    parser.add_argument(
        "--num-context-demos",
        type=int,
        default=2,
        help="Cantidad N_demos de demostraciones de contexto por consulta (default: 2)",
    )
    parser.add_argument(
        "--execution-horizon",
        type=int,
        default=4,
        help="Pasos T_exec in [1, T] ejecutados antes de replanificar en lazo cerrado (default: 4)",
    )
    parser.add_argument(
        "--num-ddim-steps",
        type=int,
        default=4,
        help="Iteraciones internas K de denoising DDIM (default: 4)",
    )
    parser.add_argument(
        "--num-centroids",
        type=int,
        default=16,
        help="Numero M de centroides FPS en el codificador de escena φ_e (default: 16)",
    )
    parser.add_argument(
        "--d-model",
        type=int,
        default=64,
        help="Dimension oculta d_model de MiniInstantPolicyNet (default: 64)",
    )
    parser.add_argument(
        "--task-id",
        type=str,
        default="task_0006",
        help="Tarea objetivo para evaluacion en lazo cerrado en `--custom-eval-only` (default: task_0006)",
    )
    parser.add_argument(
        "--custom-eval-only",
        action="store_true",
        help="Ejecuta una evaluacion rapida (contratos + Oraculo + red + latencia) solo para la combinacion dada sin sobrescribir el reporte",
    )
    parser.add_argument(
        "--report-only",
        action="store_true",
        help="Regenera unicamente docs/EXPLORACION_RED_NEURONAL.md a partir de exploration_metrics.json",
    )
    args = parser.parse_args()
    parsed_kf = parse_num_keyframes_arg(args.num_keyframes)

    if args.report_only:
        metrics_path = args.output_report.parent / "assets" / "exploration" / "exploration_metrics.json"
        results = json.loads(metrics_path.read_text(encoding="utf-8"))
        generate_markdown_exploration_report(results, output_report_path=args.output_report)
        print(f"Reporte regenerado en: {args.output_report}", flush=True)
        return

    if args.custom_eval_only:
        probe_res = run_custom_hyperparameter_probe(
            kinematic_dir=args.kinematic_dir,
            sitl_dir=args.sitl_dir,
            task_id=args.task_id,
            num_keyframes=parsed_kf,
            adaptive_arc_m=args.adaptive_arc_m,
            horizon=args.horizon,
            action_stride=args.action_stride,
            num_context_demos=args.num_context_demos,
            execution_horizon=args.execution_horizon,
            num_ddim_steps=args.num_ddim_steps,
            num_centroids=args.num_centroids,
            d_model=args.d_model,
            overfit_epochs=args.overfit_epochs,
            output_json=args.output_json,
        )
        if args.output_json is not None:
            print(f"Resultados exportados en: {args.output_json}", flush=True)
        else:
            _ = probe_res
        return

    run_all_explorations(
        kinematic_dir=args.kinematic_dir,
        sitl_dir=args.sitl_dir,
        output_report=args.output_report,
        max_kinematic_tasks=args.max_kinematic_tasks,
        overfit_epochs=args.overfit_epochs,
        num_keyframes=parsed_kf,
        adaptive_arc_m=args.adaptive_arc_m,
        horizon=args.horizon,
        action_stride=args.action_stride,
        num_context_demos=args.num_context_demos,
        execution_horizon=args.execution_horizon,
        num_ddim_steps=args.num_ddim_steps,
        num_centroids=args.num_centroids,
        d_model=args.d_model,
    )


if __name__ == "__main__":
    main()
