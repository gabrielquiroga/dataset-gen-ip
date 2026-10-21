#!/usr/bin/env python3
"""
validate_episode.py — Thin Wrapper CLI / Re-export hacia uav_ip.validate_episode.

Mantiene una unica fuente de verdad (Single Source of Truth) en:
    src/uav_ip/uav_ip/validate_episode.py
permitiendo simultaneamente:
  1. Ejecucion directa por CLI: python3 src/uav_ip/scripts/validate_episode.py --dataset-dir datasets
  2. Importacion local desde scripts/: import validate_episode
  3. Importacion estandar de paquete ROS 2 / Python: from uav_ip.validate_episode import ...
"""

import sys
from pathlib import Path

_PKG_ROOT = Path(__file__).resolve().parent.parent
if str(_PKG_ROOT) not in sys.path:
    sys.path.insert(0, str(_PKG_ROOT))

from uav_ip.validate_episode import *  # noqa: F401,F403
from uav_ip.validate_episode import main

if __name__ == "__main__":
    main()
