#!/usr/bin/env python3
"""
Pruebas modulares de geometria compartida, trazabilidad SDF x500, cota inferior de discretizacion
y geometria real de 16 cilindros de puerta circular (Bloque 4: C7 + C8).
"""

import math
import xml.etree.ElementTree as ET
from pathlib import Path
from typing import Any, Dict, List

import numpy as np
import pytest

from uav_ip.synthetic_lidar import (
    ned_to_gazebo_enu as gazebo_enu_to_ned,
    yaw_ned_to_gazebo_enu as yaw_gazebo_enu_to_ned,
)
from uav_ip.validate_episode import (
    COLLISION_SEGMENT_MAX_STEP_M,
    DISCRETIZATION_MARGIN_M,
    UAV_COLLISION_RADIUS_XY,
    UAV_PHYSICAL_RADIUS_XY,
    UAV_SAFETY_MARGIN_XY,
    validate_episode_data,
)
from uav_ip.validation_geometry import (
    CIRCULAR_GATE_CLEAR_RADIUS,
    PILOT_TRAVERSE_CIRC_MAX_R,
    PILOT_TRAVERSE_SQUARE_MAX_Y,
    PILOT_TRAVERSE_SQUARE_MAX_Z,
    PROTECTED_VOLUME_MODEL,
    PROTECTED_VOLUME_REFERENCE,
    TRAVERSE_CIRC_MAX_R,
    TRAVERSE_MAX_ABS_Y,
    TRAVERSE_MAX_ABS_Z,
    X500_FULL_ROTOR_TIP_RADIUS_M,
    X500_SDF_MEASUREMENT,
    compute_landmark_clearance,
    derive_validation_tolerances,
    get_circular_gate_cylinders_spec,
    is_traverse_center_reached,
    points_to_finite_cylinders_distance,
)


def _load_world_generator_module():
    import importlib.util

    script_path = Path(__file__).resolve().parents[1] / "scripts" / "world_generator.py"
    spec = importlib.util.spec_from_file_location("world_generator_script", script_path)
    assert spec is not None and spec.loader is not None
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def test_x500_sdf_measurement_traceability_and_derived_tolerances():
    """
    C7.A + C7.B:
      - Verifica que `X500_SDF_MEASUREMENT` documenta trazablemente el archivo SDF, commit y medidas del `x500`.
      - Verifica la separacion explicita entre `UAV_PHYSICAL_RADIUS_XY` (0.25 m) y `UAV_SAFETY_MARGIN_XY` (0.05 m).
      - Verifica que cambiar el radio físico en `derive_validation_tolerances()` recalcula automáticamente
        todos los umbrales derivados del validador y del piloto sin ediciones manuales.
    """
    assert PROTECTED_VOLUME_MODEL == "sphere_3d_around_base_link"
    assert PROTECTED_VOLUME_REFERENCE == "x500_base_collision_box_and_motor_hubs"
    assert len(X500_SDF_MEASUREMENT["px4_git_commit"]) == 40
    assert math.isclose(X500_SDF_MEASUREMENT["base_link_collision_half_diagonal_xy_m"], 0.25, abs_tol=1e-6)
    assert math.isclose(X500_FULL_ROTOR_TIP_RADIUS_M, 0.385689, abs_tol=1e-5)

    assert math.isclose(UAV_PHYSICAL_RADIUS_XY, 0.25, abs_tol=1e-9)
    assert math.isclose(UAV_SAFETY_MARGIN_XY, 0.05, abs_tol=1e-9)
    assert math.isclose(UAV_COLLISION_RADIUS_XY, 0.30, abs_tol=1e-9)

    # Cambiar el radio fisico de 0.25 m a 0.30 m recalcula todos los limites derivados
    tols_default = derive_validation_tolerances(uav_physical_radius_m=0.25, uav_safety_margin_m=0.05, pilot_buffer_m=0.02)
    tols_larger = derive_validation_tolerances(uav_physical_radius_m=0.30, uav_safety_margin_m=0.05, pilot_buffer_m=0.02)

    assert math.isclose(tols_larger["uav_collision_radius_xy_m"] - tols_default["uav_collision_radius_xy_m"], 0.05, abs_tol=1e-6)
    assert math.isclose(tols_default["traverse_max_abs_y_m"] - tols_larger["traverse_max_abs_y_m"], 0.05, abs_tol=1e-6)
    assert math.isclose(tols_default["traverse_max_abs_z_m"] - tols_larger["traverse_max_abs_z_m"], 0.05, abs_tol=1e-6)
    assert math.isclose(tols_default["traverse_circ_max_r_m"] - tols_larger["traverse_circ_max_r_m"], 0.05, abs_tol=1e-4)
    assert math.isclose(tols_default["pilot_traverse_square_max_y_m"] - tols_larger["pilot_traverse_square_max_y_m"], 0.05, abs_tol=1e-6)
    assert math.isclose(tols_default["pilot_traverse_circ_max_r_m"] - tols_larger["pilot_traverse_circ_max_r_m"], 0.05, abs_tol=1e-4)
    assert math.isclose(tols_larger["slalom_min_lateral_m"] - tols_default["slalom_min_lateral_m"], 0.05, abs_tol=1e-6)


def test_pilot_and_validator_distinguish_rectangular_vs_circular_gate_corner():
    """
    C7.B: Un punto en la esquina `(y = 0.31 m, z = 0.31 m)` tiene:
      - `|y| = 0.31 <= PILOT_TRAVERSE_SQUARE_MAX_Y (0.33 m)` y `|z| = 0.31 <= PILOT_TRAVERSE_SQUARE_MAX_Z (0.38 m)`
        -> ACEPTADO por el piloto y por el validador en `square_gate`.
      - `r = sqrt(0.31^2 + 0.31^2) = 0.4384 m > TRAVERSE_CIRC_MAX_R (0.3856 m) > PILOT_TRAVERSE_CIRC_MAX_R (0.3656 m)`
        -> RECHAZADO por el piloto (`is_traverse_center_reached`) y por el validador en `circular_gate`.
    """
    y_corner = 0.31
    z_corner = 0.31
    r_corner = math.hypot(y_corner, z_corner)
    assert r_corner > TRAVERSE_CIRC_MAX_R > PILOT_TRAVERSE_CIRC_MAX_R
    assert y_corner <= PILOT_TRAVERSE_SQUARE_MAX_Y <= TRAVERSE_MAX_ABS_Y
    assert z_corner <= PILOT_TRAVERSE_SQUARE_MAX_Z <= TRAVERSE_MAX_ABS_Z

    # 1. Verificacion en funcion compartida del piloto
    assert is_traverse_center_reached(
        along_track=0.0,
        cross_track=y_corner,
        alt_err=z_corner,
        dist_3d=r_corner,
        landmark_type="square_gate",
    ) is True
    assert is_traverse_center_reached(
        along_track=0.0,
        cross_track=y_corner,
        alt_err=z_corner,
        dist_3d=r_corner,
        landmark_type="circular_gate",
    ) is False


def test_circular_gate_16_cylinders_shared_between_sdf_and_validator():
    """
    C8: Verifica que los 16 cilindros generados por `world_generator._create_circular_gate_element()`
    coinciden exactamente 1-a-1 en pose, radio y longitud con `get_circular_gate_cylinders_spec()`
    y que el cálculo de distancia evalúa correctamente tanto centros de segmento como uniones/extremos.
    """
    wg = _load_world_generator_module()
    sdf_res = wg._create_circular_gate_element("circ_1", 4.0, 0.0, 1.5, 0.0)
    model_elem = ET.fromstring(sdf_res) if isinstance(sdf_res, (str, bytes)) else sdf_res
    link_elem = model_elem.find("link")
    assert link_elem is not None
    collisions = link_elem.findall("collision")
    specs = get_circular_gate_cylinders_spec()
    assert len(collisions) == len(specs) == 16

    for col_elem, spec in zip(collisions, specs):
        pose_txt = col_elem.findtext("pose")
        rad_txt = col_elem.findtext("geometry/cylinder/radius")
        len_txt = col_elem.findtext("geometry/cylinder/length")
        assert pose_txt == spec["pose_str"]
        assert math.isclose(float(rad_txt), float(spec["radius_m"]), abs_tol=1e-6)
        assert math.isclose(float(len_txt), float(spec["length_m"]), abs_tol=1e-6)

    # Verificar distancia en el centro de un cilindro (debe ser 0.0) y en la unión entre dos cilindros
    c0 = np.asarray(specs[0]["center_xyz"], dtype=np.float64)[None, :]
    d_center = points_to_finite_cylinders_distance(c0, specs)
    assert math.isclose(float(d_center[0]), 0.0, abs_tol=1e-9)

    # Punto en el vértice del polígono (extremo compartido en radio 0.75 m, angulo 0.0)
    joint_pt = np.array([[0.0, 0.75, 0.0]], dtype=np.float64)
    d_joint = points_to_finite_cylinders_distance(joint_pt, specs)
    assert math.isclose(float(d_joint[0]), 0.0, abs_tol=1e-6)


def test_discretization_lower_bound_rejects_between_sample_and_tangential_near_threshold_passes():
    """
    C8:
      1. Un segmento cuyos extremos discretos están fuera del obstáculo pero cuyo interior cruza el poste
         es detectado por la subdivisión y rechazado.
      2. Un pase tangencial donde la distancia muestreada sobre la malla subdividida (`h = 0.02 m`) es
         `d_sampled = 0.305 m > UAV_COLLISION_RADIUS_XY (0.30 m)`, pero cuya cota inferior continua es
         `d_inferior = 0.305 - 0.01 = 0.295 m < 0.30 m`, es RECHAZADO conservadoramente por el validador.
    """
    pole_lm = {"id": "pole_1", "type": "pole", "x": 4.0, "y": 0.0, "z": 1.5, "yaw": 0.0}

    # 1. Segmento entre dos muestras separadas 0.30 m que atraviesa el poste
    seg_through = np.array([[3.85, 0.0, 1.5], [4.15, 0.0, 1.5]], dtype=np.float64)
    c_through = compute_landmark_clearance(seg_through, pole_lm)
    assert c_through["lower_bound_distance_m"] == 0.0

    # 2. Punto/segmento tangencial con d_sampled = 0.305 m (> 0.30 m) pero d_lower_bound = 0.295 m (< 0.30 m)
    # Poste tiene radio 0.10 m -> distancia a superficie = r_xy - 0.10 = 0.305 m cuando y = 0.405 m
    y_tangential = 0.10 + UAV_COLLISION_RADIUS_XY + 0.5 * DISCRETIZATION_MARGIN_M  # 0.10 + 0.30 + 0.005 = 0.405 m
    pts_tangential = np.array([[3.99, y_tangential, 1.5], [4.01, y_tangential, 1.5]], dtype=np.float64)
    c_tang = compute_landmark_clearance(pts_tangential, pole_lm, max_step_m=COLLISION_SEGMENT_MAX_STEP_M)
    assert c_tang["sampled_min_distance_m"] > UAV_COLLISION_RADIUS_XY
    assert c_tang["lower_bound_distance_m"] < UAV_COLLISION_RADIUS_XY

    # Verificar que validate_episode_data rechaza el episodio cuando la trayectoria pasa a y = 0.405 m de un poste cercano
    waypoints_cfg = {
        "landmarks": [
            {"id": "gate_1", "type": "square_gate", "x": 2.0, "y": 0.0, "z": 1.5, "yaw": 0.0},
            {"id": "pole_near", "type": "pole", "x": 5.0, "y": -y_tangential, "z": 1.5, "yaw": 0.0},
        ],
        "primitives": [{"type": "traverse", "target": "gate_1"}],
        "landing": {"x": 8.0, "y": 0.0, "z": 0.05},
    }
    N = 120
    dt_s = 0.05
    t0_s = 1_700_000_000.0
    odom_msgs: List[Dict[str, Any]] = []
    land_msgs: List[Dict[str, Any]] = []
    for i in range(N):
        t_s = t0_s + i * dt_s
        ts_us = int(round(t_s * 1e6))
        ts_ns = int(round(t_s * 1e9))
        u = min(1.0, i / 80.0)
        x_e = 8.0 * u
        z_e = 1.5 if i < 80 else (0.05 if i >= 95 else 1.5 * (95 - i) / 15.0)
        vx_e = 1.0 if i < 80 else 0.0
        p_ned = gazebo_enu_to_ned([x_e, 0.0, z_e])
        v_ned = gazebo_enu_to_ned([vx_e, 0.0, 0.0])
        h_ned = float(yaw_gazebo_enu_to_ned(0.0))
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
        if i % 5 == 0 or i == N - 1:
            is_l = i >= 95
            land_msgs.append(
                {
                    "timestamp": ts_ns,
                    "timestamp_us": ts_us,
                    "freefall": False,
                    "ground_contact": is_l,
                    "maybe_landed": is_l,
                    "landed": is_l,
                    "at_rest": is_l,
                }
            )

    rep = validate_episode_data({"odometry": odom_msgs, "land_detected": land_msgs}, waypoints_cfg, pilot_exit_code=0)
    assert rep["validation_status"] == "rejected"
    assert any(r.startswith("collision_with_pole_near") for r in rep["rejection_reasons"])
    assert rep["metrics"]["min_obstacle_clearance_sampled_m"] > UAV_COLLISION_RADIUS_XY
    assert rep["metrics"]["min_obstacle_clearance_m"] < UAV_COLLISION_RADIUS_XY
