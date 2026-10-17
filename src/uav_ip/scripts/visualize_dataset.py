#!/usr/bin/env python3
"""
visualize_dataset.py — Thin Wrapper CLI / Re-export hacia uav_ip.visualize_dataset.

Mantiene una unica fuente de verdad (Single Source of Truth) en:
    src/uav_ip/uav_ip/visualize_dataset.py
permitiendo simultaneamente:
  1. Ejecucion directa por CLI: python3 src/uav_ip/scripts/visualize_dataset.py --help
  2. Importacion local desde scripts/: import visualize_dataset
  3. Importacion estandar de paquete ROS 2 / Python: from uav_ip.visualize_dataset import ...
"""

import sys
from pathlib import Path

_PKG_ROOT = Path(__file__).resolve().parent.parent
if str(_PKG_ROOT) not in sys.path:
    sys.path.insert(0, str(_PKG_ROOT))

from uav_ip.visualize_dataset import *  # noqa: F401,F403
from uav_ip.visualize_dataset import main

if __name__ == "__main__":
    main()
