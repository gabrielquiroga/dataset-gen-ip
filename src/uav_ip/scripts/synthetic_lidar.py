#!/usr/bin/env python3
"""
synthetic_lidar.py — Thin Wrapper CLI / Re-export hacia uav_ip.synthetic_lidar.

Mantiene una unica fuente de verdad (Single Source of Truth) en:
    src/uav_ip/uav_ip/synthetic_lidar.py
permitiendo simultaneamente:
  1. Ejecucion directa por CLI: python3 src/uav_ip/scripts/synthetic_lidar.py --test
  2. Importacion local desde scripts/: import synthetic_lidar
  3. Importacion estandar de paquete ROS 2 / Python: from uav_ip.synthetic_lidar import ...
"""

import sys
from pathlib import Path

_PKG_ROOT = Path(__file__).resolve().parent.parent
if str(_PKG_ROOT) not in sys.path:
    sys.path.insert(0, str(_PKG_ROOT))

from uav_ip.synthetic_lidar import *  # noqa: F401,F403
from uav_ip.synthetic_lidar import main

if __name__ == "__main__":
    main()
