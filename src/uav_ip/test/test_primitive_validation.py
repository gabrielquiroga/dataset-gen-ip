#!/usr/bin/env python3
"""
Pruebas modulares de validacion temporal progresiva de primitivas e intervalos
[start_index, completion_index] (Bloque 4: C9).
"""

import math
from typing import Any, Dict, List, Tuple

import numpy as np
import pytest

from uav_ip.synthetic_lidar import (
    ned_to_gazebo_enu as gazebo_enu_to_ned,
    yaw_ned_to_gazebo_enu as yaw_gazebo_enu_to_ned,
)
from uav_ip.validate_episode import validate_episode_data


def _samples_to_telemetry(
    pos_enu: np.ndarray,
    yaw_enu: np.ndarray,
    *,
    dt_s: float = 0.05,
    t0_s: float = 1_700_000_000.0,
) -> Dict[str, List[Dict[str, Any]]]:
    N = len(pos_enu)
    vel_enu = np.zeros_like(pos_enu)
    vel_enu[:-1] = np.diff(pos_enu, axis=0) / dt_s
    odom_msgs: List[Dict[str, Any]] = []
    land_msgs: List[Dict[str, Any]] = []

    for i in range(N):
        t_s = t0_s + i * dt_s
        ts_us = int(round(t_s * 1e6))
        ts_ns = int(round(t_s * 1e9))
        p_ned = gazebo_enu_to_ned(pos_enu[i])
        v_ned = gazebo_enu_to_ned(vel_enu[i])
        h_ned = float(yaw_gazebo_enu_to_ned(float(yaw_enu[i])))
        q = [math.cos(0.5 * h_ned), 0.0, 0.0, math.sin(0.5 * h_ned)]
        odom_msgs.append(
            {
                "timestamp": ts_ns,
                "timestamp_us": ts_us,
                "timestamp_sample_us": ts_us,
                "pose_frame": 1,
                "position": [float(p_ned[0]), float(p_ned[1]), float(p_ned[2])],
                "q": [float(c) for c in q],
                "velocity_frame": 1,
                "velocity": [float(v_ned[0]), float(v_ned[1]), float(v_ned[2])],
                "angular_velocity": [0.0, 0.0, 0.0],
                "reset_counter": 1,
                "quality": 100,
            }
        )
        if i % 4 == 0 or i == N - 1:
            is_landed = i >= N - 20
            land_msgs.append(
                {
                    "timestamp": ts_ns,
                    "timestamp_us": ts_us,
                    "freefall": False,
                    "ground_contact": is_landed,
                    "maybe_landed": is_landed,
                    "landed": is_landed,
                    "at_rest": is_landed,
                }
            )
    return {"odometry": odom_msgs, "land_detected": land_msgs}


def _append_smooth_landing(
    pts: List[List[float]],
    yaws: List[float],
    land_xy: Tuple[float, float],
) -> Tuple[np.ndarray, np.ndarray]:
    lx, ly = land_xy
    cur = np.asarray(pts[-1], dtype=np.float64)
    target_hover = np.array([lx, ly, 1.5], dtype=np.float64)
    dist = float(np.linalg.norm(target_hover - cur))
    n_transit = max(10, int(math.ceil(dist / 0.08)))
    for k in range(1, n_transit + 1):
        alpha = k / float(n_transit)
        p = (1.0 - alpha) * cur + alpha * target_hover
        pts.append([float(p[0]), float(p[1]), float(p[2])])
        yaws.append(0.0)
    # Descenso vertical y reposo durante 25 muestras (1.25 s)
    for k in range(1, 21):
        z = 1.5 - (1.5 - 0.05) * (k / 20.0)
        pts.append([lx, ly, z])
        yaws.append(0.0)
    for _ in range(25):
        pts.append([lx, ly, 0.05])
        yaws.append(0.0)
    return np.asarray(pts, dtype=np.float64), np.asarray(yaws, dtype=np.float64)


def test_nominal_progressive_primitives_accepted_with_ordered_intervals():
    """
    C9: Una trayectoria nominal que ejecuta `traverse` -> `orbit` en orden temporal estricto
    es aceptada y reporta `start_index <= event_index <= completion_index` con
    `start_index[1] >= completion_index[0]`.
    """
    waypoints_cfg = {
        "landmarks": [
            {"id": "gate_1", "type": "square_gate", "x": 3.0, "y": 0.0, "z": 1.5, "yaw": 0.0},
            {"id": "pole_2", "type": "pole", "x": 7.0, "y": 0.0, "z": 1.5, "yaw": 0.0},
        ],
        "primitives": [
            {"type": "traverse", "target": "gate_1"},
            {"type": "orbit", "target": "pole_2"},
        ],
        "landing": {"x": 11.0, "y": 0.0, "z": 0.05},
    }

    pts: List[List[float]] = []
    yaws: List[float] = []
    # Avance recto desde x=0 hasta x=5.5 atravesando gate_1 en x=3.0
    for x in np.linspace(0.0, 5.5, 70):
        pts.append([float(x), 0.0, 1.5])
        yaws.append(0.0)

    # Orbita antihoraria de 270 grados (de -pi a +pi/2) alrededor de pole_2 (7.0, 0.0) con R=1.5 m
    for ang in np.linspace(-math.pi, 0.5 * math.pi, 75):
        px = 7.0 + 1.5 * math.cos(ang)
        py = 0.0 + 1.5 * math.sin(ang)
        pts.append([float(px), float(py), 1.5])
        yaws.append(math.atan2(0.0 - py, 7.0 - px))

    pos_enu, yaw_enu = _append_smooth_landing(pts, yaws, (11.0, 0.0))
    tel = _samples_to_telemetry(pos_enu, yaw_enu)
    rep = validate_episode_data(tel, waypoints_cfg, pilot_exit_code=0)

    assert rep["validation_status"] == "accepted", f"Deberia aceptarse, reasons={rep['rejection_reasons']}"
    p0, p1 = rep["primitive_results"]
    assert p0["passed"] and p1["passed"]
    assert p0["start_index"] <= p0["event_index"] <= p0["completion_index"]
    assert p1["start_index"] <= p1["event_index"] <= p1["completion_index"]
    assert p1["start_index"] >= p0["completion_index"]


def test_inverted_primitive_execution_rejected():
    """
    C9: Si el dron ejecuta primero la segunda primitiva (`gate_2`) y luego retrocede para
    ejecutar la primera (`gate_1`), el validador rechaza el episodio con `primitive_order_violation`.
    """
    waypoints_cfg = {
        "landmarks": [
            {"id": "gate_1", "type": "square_gate", "x": 3.0, "y": 2.0, "z": 1.5, "yaw": 0.0},
            {"id": "gate_2", "type": "square_gate", "x": 3.0, "y": -2.0, "z": 1.5, "yaw": 0.0},
        ],
        "primitives": [
            {"type": "traverse", "target": "gate_1"},
            {"type": "traverse", "target": "gate_2"},
        ],
        "landing": {"x": 8.0, "y": 0.0, "z": 0.05},
    }

    pts: List[List[float]] = []
    yaws: List[float] = []
    # Primero cruza gate_2 (y = -2.0) en direccion +x
    for x in np.linspace(0.5, 4.5, 50):
        pts.append([float(x), -2.0, 1.5])
        yaws.append(0.0)
    # Rodea hacia atras hasta (0.5, +2.0) sin cruzar en +x
    for u in np.linspace(0.0, 1.0, 50):
        pts.append([4.5 - 4.0 * u, -2.0 + 4.0 * u, 1.5])
        yaws.append(0.0)
    # Ahora cruza gate_1 (y = +2.0) en direccion +x
    for x in np.linspace(0.5, 5.0, 50):
        pts.append([float(x), 2.0, 1.5])
        yaws.append(0.0)

    pos_enu, yaw_enu = _append_smooth_landing(pts, yaws, (8.0, 0.0))
    tel = _samples_to_telemetry(pos_enu, yaw_enu)
    rep = validate_episode_data(tel, waypoints_cfg, pilot_exit_code=0)

    assert rep["validation_status"] == "rejected"
    assert any(r.startswith("primitive_order_violation") for r in rep["rejection_reasons"])


def test_next_primitive_starting_before_orbit_completes_is_rejected():
    """
    C9: Si la primitiva 1 es `orbit` y la primitiva 2 (`traverse`) se cruza después del punto medio
    de la órbita (`event_index`) pero ANTES de que la órbita complete los `240°` (`completion_index`),
    el validador rechaza el episodio con `primitive_order_violation`.
    """
    # Colocar gate_2 sobre el último tramo del arco orbital (en angulo +45 deg, antes de alcanzar +60 deg = 240 deg de barrido)
    R = 1.5
    ang_gate = math.radians(45.0)  # Barrido desde -180 deg hasta +45 deg es 225 deg (< 240 deg)
    gx = 4.0 + R * math.cos(ang_gate)
    gy = 0.0 + R * math.sin(ang_gate)
    # Tangente antihoraria en ang_gate: (-sin, cos)
    gyaw = math.atan2(math.cos(ang_gate), -math.sin(ang_gate))

    waypoints_cfg = {
        "landmarks": [
            {"id": "pole_1", "type": "pole", "x": 4.0, "y": 0.0, "z": 1.5, "yaw": 0.0},
            {"id": "gate_2", "type": "square_gate", "x": gx, "y": gy, "z": 1.5, "yaw": gyaw},
        ],
        "primitives": [
            {"type": "orbit", "target": "pole_1"},
            {"type": "traverse", "target": "gate_2"},
        ],
        "landing": {"x": 9.0, "y": 0.0, "z": 0.05},
    }

    pts: List[List[float]] = []
    yaws: List[float] = []
    for x in np.linspace(0.0, 2.5, 35):
        pts.append([float(x), 0.0, 1.5])
        yaws.append(0.0)

    # Orbita completa de 270 grados (de -180 deg a +90 deg):
    # - event_index de la orbita queda en ~120 deg de barrido (angulo -60 deg)
    # - gate_2 se cruza en 225 deg de barrido (angulo +45 deg) -> DESPUES de event_index pero ANTES de completion_index (240 deg = +60 deg)!
    for ang in np.linspace(-math.pi, 0.5 * math.pi, 90):
        px = 4.0 + R * math.cos(ang)
        py = 0.0 + R * math.sin(ang)
        pts.append([float(px), float(py), 1.5])
        yaws.append(math.atan2(0.0 - py, 4.0 - px))

    pos_enu, yaw_enu = _append_smooth_landing(pts, yaws, (9.0, 0.0))
    tel = _samples_to_telemetry(pos_enu, yaw_enu)
    rep = validate_episode_data(tel, waypoints_cfg, pilot_exit_code=0)

    assert rep["validation_status"] == "rejected"
    assert any(r.startswith("primitive_order_violation") for r in rep["rejection_reasons"])


def test_repeated_crossing_does_not_hide_premature_out_of_order_execution():
    """
    C9: Si el dron cruza prematuramente `gate_2` antes de `gate_1`, luego cruza `gate_1` y finalmente
    vuelve a cruzar `gate_2` por segunda vez después de `gate_1`, el segundo cruce NO debe ocultar
    la violación de secuencia: el episodio debe ser rechazado con `primitive_order_violation`.
    """
    waypoints_cfg = {
        "landmarks": [
            {"id": "gate_1", "type": "square_gate", "x": 3.0, "y": 2.0, "z": 1.5, "yaw": 0.0},
            {"id": "gate_2", "type": "square_gate", "x": 3.0, "y": -2.0, "z": 1.5, "yaw": 0.0},
        ],
        "primitives": [
            {"type": "traverse", "target": "gate_1"},
            {"type": "traverse", "target": "gate_2"},
        ],
        "landing": {"x": 8.0, "y": 0.0, "z": 0.05},
    }

    pts: List[List[float]] = []
    yaws: List[float] = []
    # 1. Cruce prematuro de gate_2 (y = -2.0)
    for x in np.linspace(1.0, 4.5, 40):
        pts.append([float(x), -2.0, 1.5])
        yaws.append(0.0)
    # 2. Transicion hacia (1.0, +2.0)
    for u in np.linspace(0.0, 1.0, 45):
        pts.append([4.5 - 3.5 * u, -2.0 + 4.0 * u, 1.5])
        yaws.append(0.0)
    # 3. Cruce de gate_1 (y = +2.0)
    for x in np.linspace(1.0, 4.5, 40):
        pts.append([float(x), 2.0, 1.5])
        yaws.append(0.0)
    # 4. Transicion de vuelta a (1.0, -2.0)
    for u in np.linspace(0.0, 1.0, 45):
        pts.append([4.5 - 3.5 * u, 2.0 - 4.0 * u, 1.5])
        yaws.append(0.0)
    # 5. Segundo cruce de gate_2 (y = -2.0)
    for x in np.linspace(1.0, 5.0, 40):
        pts.append([float(x), -2.0, 1.5])
        yaws.append(0.0)

    pos_enu, yaw_enu = _append_smooth_landing(pts, yaws, (8.0, 0.0))
    tel = _samples_to_telemetry(pos_enu, yaw_enu)
    rep = validate_episode_data(tel, waypoints_cfg, pilot_exit_code=0)

    assert rep["validation_status"] == "rejected"
    assert any(r.startswith("primitive_order_violation") for r in rep["rejection_reasons"])


def test_orbit_with_outward_yaw_is_rejected():
    """
    C9: Una órbita con barrido antihorario completo (`>= 240°`) pero con el dron mirando hacia
    afuera del poste (`mean_centripetal_yaw_err_deg > 45°`) es rechazada.
    """
    waypoints_cfg = {
        "landmarks": [{"id": "pole_1", "type": "pole", "x": 4.0, "y": 0.0, "z": 1.5, "yaw": 0.0}],
        "primitives": [{"type": "orbit", "target": "pole_1"}],
        "landing": {"x": 8.0, "y": 0.0, "z": 0.05},
    }

    pts: List[List[float]] = []
    yaws: List[float] = []
    for x in np.linspace(0.0, 2.5, 35):
        pts.append([float(x), 0.0, 1.5])
        yaws.append(0.0)

    for ang in np.linspace(-math.pi, 0.5 * math.pi, 75):
        px = 4.0 + 1.5 * math.cos(ang)
        py = 0.0 + 1.5 * math.sin(ang)
        pts.append([float(px), float(py), 1.5])
        # Yaw mirando en sentido opuesto al centro del poste (+180 grados de error)
        outward_yaw = math.atan2(py - 0.0, px - 4.0)
        yaws.append(outward_yaw)

    pos_enu, yaw_enu = _append_smooth_landing(pts, yaws, (8.0, 0.0))
    tel = _samples_to_telemetry(pos_enu, yaw_enu)
    rep = validate_episode_data(tel, waypoints_cfg, pilot_exit_code=0)

    assert rep["validation_status"] == "rejected"
    assert any(r.startswith("orbit_pole_1_excessive_centripetal_yaw_error") for r in rep["rejection_reasons"])
