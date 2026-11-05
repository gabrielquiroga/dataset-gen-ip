#!/usr/bin/env python3
"""
run_exploration_suite.py — Thin wrapper CLI para ejecutar la Suite de Exploracion Pre-Hito 4
de UAV Instant Policy dentro del entorno virtual `~/venv_ip`.

Uso:
    ~/venv_ip/bin/python3 src/uav_ip/scripts/policy/run_exploration_suite.py \
      --kinematic-dir datasets/kinematic \
      --sitl-dir datasets/sitl \
      --output-report docs/EXPLORACION_RED_NEURONAL.md
"""

import argparse
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

from uav_ip.policy.exploration_suite import run_all_explorations  # noqa: E402


def main() -> None:
    repo_root = _PKG_ROOT.parent.parent
    parser = argparse.ArgumentParser(
        description="Ejecuta los 5 experimentos de exploracion y validacion Pre-Hito 4 "
        "y genera docs/EXPLORACION_RED_NEURONAL.md.",
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
        "--max-kinematic-tasks",
        type=int,
        default=40,
        help="Numero de tareas cinematicas a escanear en el Experimento 1 (default: 40 = 200 episodios)",
    )
    parser.add_argument(
        "--overfit-epochs",
        type=int,
        default=35,
        help="Epocas para la prueba de Overfit del Experimento 3 (default: 35)",
    )
    args = parser.parse_args()

    run_all_explorations(
        kinematic_dir=args.kinematic_dir,
        sitl_dir=args.sitl_dir,
        output_report=args.output_report,
        max_kinematic_tasks=args.max_kinematic_tasks,
        overfit_epochs=args.overfit_epochs,
    )


if __name__ == "__main__":
    main()
