#!/usr/bin/env python3
"""
validation_geometry.py — Modulo Ligero de Geometria, Envolvente del UAV y Tolerancias Compartidas.

Centraliza sin dependencias de ROS 2 ni de lectura de bolsas MCAP:
  1. La medicion trazable del modelo `gz_x500` (`x500` + `x500_base`) en PX4-Autopilot.
  2. La definicion explicita del volumen protegido (esfera 3D alrededor de `base_link`) y la
     separacion entre radio fisico (`UAV_PHYSICAL_RADIUS_XY`) y margen de seguridad (`UAV_SAFETY_MARGIN_XY`).
  3. El calculo dinamico de todas las tolerancias derivadas (`derive_validation_tolerances`)
     compartidas entre el piloto (`waypoint_controller.py`), el generador de mundos
     (`world_generator.py`) y el validador (`validate_episode.py`).
  4. La geometria exacta de los 16 cilindros finitos del `circular_gate` compartida con el SDF.
  5. El calculo de distancia a obstaculos con subdivision de segmentos a paso `<= h` y cota
     inferior explicita de discretizacion sobre la polilinea:
       d_inferior = max(0.0, d_muestreada - h / 2.0).
"""

import math
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

# ============================================================================
# 1. Medicion Trazable del Modelo `gz_x500` en PX4-Autopilot
# ============================================================================

X500_SDF_MEASUREMENT: Dict[str, Any] = {
    "px4_commit": "cbc6a6f7abbd06fbb2ff0aed3e57e3e95f88232e",
    "px4_git_commit": "cbc6a6f7abbd06fbb2ff0aed3e57e3e95f88232e",
    "source_files": [
        "Tools/simulation/gz/models/x500/model.sdf",
        "Tools/simulation/gz/models/x500_base/model.sdf",
    ],
    "model_spawn_pose_z_m": 0.24,
    "base_link_collision_0_pose_xyz_m": (0.0, 0.0, 0.007),
    "base_link_collision_0_size_xyz_m": (0.35355339059327373, 0.35355339059327373, 0.05),
    # Semidiagonal exacta en XY de base_link_collision_0: 0.35355339059327373 / sqrt(2) = 0.25 m
    "base_link_collision_0_xy_half_diagonal_m": 0.25,
    "base_link_collision_half_diagonal_xy_m": 0.25,
    "skid_collision_poses_xyz_m": [
        (0.0, -0.132, -0.2195),
        (0.0, 0.132, -0.2195),
    ],
    "skid_collision_size_xyz_m": (0.25, 0.015, 0.015),
    "skid_bottom_offset_from_base_link_z_m": -0.227,
    "rotor_hub_poses_xyz_m": [
        (0.174, -0.174, 0.06),
        (-0.174, 0.174, 0.06),
        (0.174, 0.174, 0.06),
        (-0.174, -0.174, 0.06),
    ],
    "rotor_hub_xy_radius_m": round(math.hypot(0.174, 0.174), 6),  # 0.246073 m
    "rotor_hub_3d_radius_m": round(math.sqrt(0.174**2 + 0.174**2 + 0.06**2), 6),  # 0.253282 m
    "rotor_prop_collision_size_xyz_m": (
        0.2792307692307692,
        0.016923076923076923,
        0.0008461538461538462,
    ),
    "rotor_prop_radius_m": round(0.2792307692307692 / 2.0, 6),  # 0.139615 m
    "full_rotor_tip_xy_radius_m": round(
        math.hypot(0.174, 0.174) + 0.2792307692307692 / 2.0, 6
    ),  # 0.385689 m
}

# Definicion explicita del volumen protegido:
# Al comparar la distancia euclidea 3D desde el punto de referencia (`base_link`) hasta
# los solidos de cada obstaculo contra un unico radio escalar, el volumen protegido es una
# ESFERA 3D centrada en `base_link`.
PROTECTED_VOLUME_MODEL: str = "sphere_3d_around_base_link"
PROTECTED_VOLUME_REFERENCE: str = "x500_base_collision_box_and_motor_hubs"

X500_CHASSIS_AND_HUBS_RADIUS_M: float = 0.25  # Medido de base_link_collision_0 (0.35355339 / sqrt(2))
X500_FULL_ROTOR_TIP_RADIUS_M: float = float(X500_SDF_MEASUREMENT["full_rotor_tip_xy_radius_m"])

# ============================================================================
# 2. Dimensiones Nominales de Obstaculos (Compartidas con world_generator.py)
# ============================================================================

GATE_HALF_WIDTH: float = 0.70           # Pilares izquierdo/derecho centrados en y = ±0.70 m
GATE_HALF_HEIGHT: float = 0.75          # Vigas superior/inferior centradas en z = ±0.75 m
GATE_BAR_THICKNESS: float = 0.10        # Espesor de las 4 barras (semiespesor = 0.05 m)
SQUARE_GATE_CLEAR_HALF_Y: float = GATE_HALF_WIDTH - 0.5 * GATE_BAR_THICKNESS   # 0.65 m
SQUARE_GATE_CLEAR_HALF_Z: float = GATE_HALF_HEIGHT - 0.5 * GATE_BAR_THICKNESS  # 0.70 m

CIRCULAR_MAJOR_RADIUS: float = 0.75     # Radio mayor del anillo circular (m)
CIRCULAR_TUBE_RADIUS: float = 0.05      # Radio de cada cilindro del anillo (m)
CIRCULAR_NUM_SEGMENTS: int = 16         # Numero de cilindros en world_generator.py
CIRCULAR_LENGTH_OVERLAP_FACTOR: float = 1.05
CIRCULAR_POLY_SAGITTA: float = CIRCULAR_MAJOR_RADIUS * (
    1.0 - math.cos(math.pi / CIRCULAR_NUM_SEGMENTS)
)
CIRCULAR_GATE_CLEAR_RADIUS: float = (
    CIRCULAR_MAJOR_RADIUS - CIRCULAR_POLY_SAGITTA - CIRCULAR_TUBE_RADIUS
)  # ~0.6856 m

POLE_RADIUS: float = 0.10               # Radio del poste cilindrico vertical (m)
POLE_HEIGHT: float = 3.00               # Altura total del poste apoyado en z=0 (m)
LANDING_PAD_HALF_SIZE: float = 1.00     # Semilado de plataforma cuadrada 2.0m x 2.0m

# Paso espacial maximo para subdivision de segmentos de la polilinea (h) y margen (h/2)
COLLISION_SEGMENT_MAX_STEP_M: float = 0.02
DISCRETIZATION_MARGIN_M: float = 0.5 * COLLISION_SEGMENT_MAX_STEP_M  # 0.01 m

# ============================================================================
# 3. Envolvente por Defecto y Derivacion Dinamica de Tolerancias
# ============================================================================

UAV_PHYSICAL_RADIUS_XY: float = X500_CHASSIS_AND_HUBS_RADIUS_M  # 0.25 m (medido del SDF)
UAV_SAFETY_MARGIN_XY: float = 0.05                              # 0.05 m
PILOT_APERTURE_BUFFER_M: float = 0.02                           # Margen adicional del controlador respecto al validador
PILOT_BUFFER_M: float = PILOT_APERTURE_BUFFER_M


def derive_validation_tolerances(
    uav_physical_radius_m: float = UAV_PHYSICAL_RADIUS_XY,
    uav_safety_margin_m: float = UAV_SAFETY_MARGIN_XY,
    pilot_buffer_m: float = PILOT_APERTURE_BUFFER_M,
) -> Dict[str, float]:
    """
    Deriva todas las tolerancias geometricas del validador y del piloto a partir del
    radio fisico (`uav_physical_radius_m`) y del margen de seguridad (`uav_safety_margin_m`).
    Cambiar cualquiera de estos parametros actualiza automaticamente todas las cotas derivadas.
    """
    phys_r = float(uav_physical_radius_m)
    margin_r = float(uav_safety_margin_m)
    buf_r = float(pilot_buffer_m)
    if phys_r <= 0.0 or margin_r < 0.0:
        raise ValueError("uav_physical_radius_m debe ser > 0 y uav_safety_margin_m >= 0")

    col_r = round(phys_r + margin_r, 6)
    safe_pole_center = round(POLE_RADIUS + col_r, 6)

    traverse_max_y = round(max(0.0, SQUARE_GATE_CLEAR_HALF_Y - col_r), 4)
    traverse_max_z = round(max(0.0, SQUARE_GATE_CLEAR_HALF_Z - col_r), 4)
    traverse_circ_max_r = round(max(0.0, CIRCULAR_GATE_CLEAR_RADIUS - col_r), 4)

    pilot_sq_max_y = round(max(0.05, traverse_max_y - buf_r), 4)
    pilot_sq_max_z = round(max(0.05, traverse_max_z - buf_r), 4)
    pilot_circ_max_r = round(max(0.05, traverse_circ_max_r - buf_r), 4)

    slalom_min_lat = round(safe_pole_center + 0.07, 4)
    landing_max_xy = round(max(0.10, LANDING_PAD_HALF_SIZE - phys_r), 4)

    return {
        "uav_physical_radius_xy_m": phys_r,
        "uav_safety_margin_xy_m": margin_r,
        "uav_collision_radius_xy_m": col_r,
        "uav_safe_pole_center_dist_m": safe_pole_center,
        "traverse_max_abs_y_m": traverse_max_y,
        "traverse_max_abs_z_m": traverse_max_z,
        "traverse_circ_max_r_m": traverse_circ_max_r,
        "pilot_traverse_square_max_y_m": pilot_sq_max_y,
        "pilot_traverse_square_max_z_m": pilot_sq_max_z,
        "pilot_traverse_circ_max_r_m": pilot_circ_max_r,
        "slalom_min_lateral_m": slalom_min_lat,
        "landing_max_xy_err_m": landing_max_xy,
    }


_DEFAULT_TOLS = derive_validation_tolerances()

UAV_COLLISION_RADIUS_XY: float = _DEFAULT_TOLS["uav_collision_radius_xy_m"]          # 0.30 m
UAV_SAFE_POLE_CENTER_DIST: float = _DEFAULT_TOLS["uav_safe_pole_center_dist_m"]      # 0.40 m
TRAVERSE_MAX_ABS_Y: float = _DEFAULT_TOLS["traverse_max_abs_y_m"]                    # 0.35 m
TRAVERSE_MAX_ABS_Z: float = _DEFAULT_TOLS["traverse_max_abs_z_m"]                    # 0.40 m
TRAVERSE_CIRC_MAX_R: float = _DEFAULT_TOLS["traverse_circ_max_r_m"]                  # 0.3856 m
PILOT_TRAVERSE_SQUARE_MAX_Y: float = _DEFAULT_TOLS["pilot_traverse_square_max_y_m"]  # 0.33 m
PILOT_TRAVERSE_SQUARE_MAX_Z: float = _DEFAULT_TOLS["pilot_traverse_square_max_z_m"]  # 0.38 m
PILOT_TRAVERSE_CIRC_MAX_R: float = _DEFAULT_TOLS["pilot_traverse_circ_max_r_m"]      # 0.3656 m
SLALOM_MIN_LATERAL: float = _DEFAULT_TOLS["slalom_min_lateral_m"]                    # 0.47 m
SLALOM_MAX_LATERAL: float = 2.60
OVERFLY_MIN_ALT_M: float = POLE_HEIGHT + 0.30                                        # 3.30 m
OVERFLY_MAX_LAT_M: float = 0.85
ORBIT_MIN_SWEEP_DEG: float = 240.0
ORBIT_MAX_RADIUS_M: float = 3.20
ORBIT_MAX_MEAN_YAW_ERR_DEG: float = 45.0
LANDING_MAX_XY_ERR_M: float = _DEFAULT_TOLS["landing_max_xy_err_m"]                  # 0.75 m


def is_traverse_center_reached(
    along_track: float = 0.0,
    cross_track: float = 0.0,
    alt_err: float = 0.0,
    dist_3d: float = 0.0,
    landmark_type: str = "square_gate",
    tolerances: Optional[Dict[str, float]] = None,
    *,
    along_track_m: Optional[float] = None,
    cross_track_m: Optional[float] = None,
    alt_err_m: Optional[float] = None,
    dist_3d_m: Optional[float] = None,
    square_max_y: Optional[float] = None,
    square_max_z: Optional[float] = None,
    circ_max_r: Optional[float] = None,
) -> bool:
    """
    Evalua si el controlador experto puede dar por alcanzado el waypoint `center` de una
    primitiva `traverse`, distinguiendo explicitamente la ventana rectangular (`square_gate`)
    del limite radial (`circular_gate`).
    """
    tols = tolerances if tolerances is not None else _DEFAULT_TOLS
    at = float(along_track_m if along_track_m is not None else along_track)
    ct = float(cross_track_m if cross_track_m is not None else cross_track)
    ae = float(alt_err_m if alt_err_m is not None else alt_err)
    d3 = float(dist_3d_m if dist_3d_m is not None else dist_3d)

    longitudinal_ok = (d3 < 0.45) or (at >= -0.05)
    if not longitudinal_ok:
        return False

    lm_t = str(landmark_type).strip().lower()
    if "circ" in lm_t or "torus" in lm_t:
        radial_err = math.hypot(ct, ae)
        limit_r = float(circ_max_r if circ_max_r is not None else tols["pilot_traverse_circ_max_r_m"])
        return bool(radial_err <= limit_r)
    limit_y = float(square_max_y if square_max_y is not None else tols["pilot_traverse_square_max_y_m"])
    limit_z = float(square_max_z if square_max_z is not None else tols["pilot_traverse_square_max_z_m"])
    return bool(abs(ct) <= limit_y and abs(ae) <= limit_z)


# ============================================================================
# 4. Geometria Compartida de Obstaculos y Cilindros Reales de `circular_gate`
# ============================================================================

SQUARE_GATE_BARS_LOCAL: List[Tuple[Tuple[float, float, float], Tuple[float, float, float]]] = [
    ((0.0, 0.0, 0.75), (0.10, 1.50, 0.10)),
    ((0.0, 0.0, -0.75), (0.10, 1.50, 0.10)),
    ((0.0, 0.70, 0.0), (0.10, 0.10, 1.40)),
    ((0.0, -0.70, 0.0), (0.10, 0.10, 1.40)),
]


def get_circular_gate_cylinders_spec(
    major_radius: float = CIRCULAR_MAJOR_RADIUS,
    tube_radius: float = CIRCULAR_TUBE_RADIUS,
    num_segments: int = CIRCULAR_NUM_SEGMENTS,
    length_overlap_factor: float = CIRCULAR_LENGTH_OVERLAP_FACTOR,
) -> List[Dict[str, Any]]:
    """
    Construye la especificacion geometrica exacta de los `num_segments` cilindros finitos
    que conforman un `circular_gate` en el marco local del landmark (plano Y-Z).
    Esta funcion es compartida por `world_generator.py` (para generar el SDF) y por el
    validador (para evaluar la distancia exacta a la union de cilindros).
    """
    seg_length = round(
        2.0 * float(major_radius) * math.sin(math.pi / int(num_segments)) * float(length_overlap_factor),
        4,
    )
    cylinders: List[Dict[str, Any]] = []
    for k in range(int(num_segments)):
        phi = (2.0 * math.pi * k) / int(num_segments)
        yk = float(major_radius) * math.cos(phi)
        zk = float(major_radius) * math.sin(phi)
        # Al rotar el eje local +Z del cilindro alrededor de +X por roll=phi,
        # el vector unitario del eje del cilindro en el marco del landmark es (0, -sin(phi), cos(phi)).
        axis_u = (0.0, -math.sin(phi), math.cos(phi))
        cylinders.append(
            {
                "index": k,
                "phi_rad": phi,
                "center_xyz": (0.0, yk, zk),
                "axis_unit_xyz": axis_u,
                "radius_m": round(float(tube_radius), 4),
                "length_m": float(seg_length),
                "pose_str": f"0 {yk:.4f} {zk:.4f} {phi:.4f} 0 0",
            }
        )
    return cylinders


def world_to_landmark_local(
    pos_enu: np.ndarray,
    lm_x: float,
    lm_y: float,
    lm_z: float,
    lm_yaw: float,
) -> np.ndarray:
    """
    Transforma puntos `pos_enu` (N, 3) en Gazebo ENU al marco local FLU del landmark:
      +x_loc: direccion de avance / normal del landmark (cos(yaw), sin(yaw))
      +y_loc: flanco izquierdo del landmark (-sin(yaw), cos(yaw))
      +z_loc: vertical hacia arriba (z - lm_z)
    """
    dx = pos_enu[:, 0] - float(lm_x)
    dy = pos_enu[:, 1] - float(lm_y)
    dz = pos_enu[:, 2] - float(lm_z)
    c = math.cos(float(lm_yaw))
    s = math.sin(float(lm_yaw))
    x_loc = dx * c + dy * s
    y_loc = -dx * s + dy * c
    return np.column_stack([x_loc, y_loc, dz])


def points_to_aabb_distance(
    pts_local: np.ndarray,
    box_center: Tuple[float, float, float],
    box_size: Tuple[float, float, float],
) -> np.ndarray:
    """Calcula la distancia euclidea desde N puntos (N, 3) a una caja 3D alineada."""
    c = np.asarray(box_center, dtype=np.float64)
    half = 0.5 * np.asarray(box_size, dtype=np.float64)
    excess = np.maximum(0.0, np.abs(pts_local - c) - half)
    return np.linalg.norm(excess, axis=1)


def points_to_finite_cylinders_distance(
    pts_local: np.ndarray,
    cylinders_spec: List[Dict[str, Any]],
) -> np.ndarray:
    """
    Calcula para cada punto en `pts_local` (N, 3) la distancia euclidea exacta a la union
    de los cilindros rectos finitos definidos en `cylinders_spec`.
    """
    if len(pts_local) == 0 or len(cylinders_spec) == 0:
        return np.full(len(pts_local), float("inf"), dtype=np.float64)

    centers = np.array([c["center_xyz"] for c in cylinders_spec], dtype=np.float64)      # (K, 3)
    axes = np.array([c["axis_unit_xyz"] for c in cylinders_spec], dtype=np.float64)     # (K, 3)
    radii = np.array([c["radius_m"] for c in cylinders_spec], dtype=np.float64)         # (K,)
    half_lens = 0.5 * np.array([c["length_m"] for c in cylinders_spec], dtype=np.float64)  # (K,)

    # diff: (N, K, 3)
    diff = pts_local[:, None, :] - centers[None, :, :]
    # Proyeccion axial escalar s = diff . axis: (N, K)
    s_proj = np.sum(diff * axes[None, :, :], axis=2)
    d_axial = np.maximum(0.0, np.abs(s_proj) - half_lens[None, :])

    # Componente perpendicular al eje del cilindro: (N, K, 3)
    v_perp = diff - s_proj[:, :, None] * axes[None, :, :]
    r_perp = np.linalg.norm(v_perp, axis=2)
    d_radial = np.maximum(0.0, r_perp - radii[None, :])

    dists_per_cyl = np.hypot(d_axial, d_radial)  # (N, K)
    return np.min(dists_per_cyl, axis=1)


def subdivide_trajectory_segments(
    pos_enu: np.ndarray,
    max_step_m: float = COLLISION_SEGMENT_MAX_STEP_M,
) -> Tuple[np.ndarray, float]:
    """
    Subdivide linealmente todo segmento entre muestras consecutivas cuya longitud supere
    `max_step_m` (default: 0.02 m) y retorna `(pts_dense, effective_max_step_h_m)`.
    Para una polilinea de al menos 2 puntos con `max_step_m > 0`, el paso de discretizacion
    efectivo `h` sobre cada segmento subdividido queda acotado por `max_step_m`.
    """
    pts = np.asarray(pos_enu, dtype=np.float64)
    if len(pts) <= 1 or max_step_m <= 0.0:
        if len(pts) <= 1:
            return pts, 0.0
        raw_max = float(np.max(np.linalg.norm(np.diff(pts, axis=0), axis=1)))
        return pts, raw_max

    diffs = np.diff(pts, axis=0)
    seg_lens = np.linalg.norm(diffs, axis=1)
    if np.all(seg_lens <= max_step_m):
        actual_h = float(np.max(seg_lens)) if len(seg_lens) > 0 else 0.0
        return pts, max(actual_h, float(max_step_m)) if actual_h > 0.0 else 0.0

    chunks: List[np.ndarray] = []
    max_sub_step = 0.0
    for i in range(len(diffs)):
        L = float(seg_lens[i])
        if L <= max_step_m:
            chunks.append(pts[i : i + 1])
            max_sub_step = max(max_sub_step, L)
        else:
            n_sub = int(math.ceil(L / max_step_m))
            sub_h = L / float(n_sub)
            max_sub_step = max(max_sub_step, sub_h)
            alphas = np.linspace(0.0, 1.0, n_sub, endpoint=False, dtype=np.float64)[:, None]
            seg_pts = pts[i] + alphas * diffs[i]
            chunks.append(seg_pts)
    chunks.append(pts[-1:])
    # La cota solicitada en C8 usa explicitamente el paso de subdivision h = max_step_m
    return np.vstack(chunks), float(max_step_m)


def compute_landmark_clearance(
    pos_enu: np.ndarray,
    landmark: Dict[str, Any],
    max_step_m: float = COLLISION_SEGMENT_MAX_STEP_M,
) -> Dict[str, float]:
    """
    Evalua la distancia desde la polilinea `pos_enu` hasta la superficie solida de `landmark`,
    separando explicitamente:
      - `sampled_min_distance_m` (d_muestreada sobre los puntos subdivididos a paso <= h)
      - `discretization_step_h_m` (h)
      - `discretization_margin_m` (h / 2)
      - `lower_bound_distance_m` = max(0.0, d_muestreada - h / 2)
    """
    pts = np.asarray(pos_enu, dtype=np.float64)
    if len(pts) == 0:
        return {
            "sampled_min_distance_m": float("inf"),
            "discretization_step_h_m": 0.0,
            "discretization_margin_m": 0.0,
            "lower_bound_distance_m": float("inf"),
        }

    pts_dense, h_step = subdivide_trajectory_segments(pts, max_step_m=max_step_m)
    # Si hay al menos 2 muestras originales y max_step_m > 0, usar h = min(max_seg_len, max_step_m) o max_step_m
    if len(pts) >= 2 and max_step_m > 0.0:
        raw_seg_max = float(np.max(np.linalg.norm(np.diff(pts, axis=0), axis=1)))
        h_effective = float(max_step_m) if raw_seg_max > 0.0 else 0.0
    else:
        h_effective = float(h_step)

    lm_type = str(landmark.get("type", "gate")).lower()
    cx = float(landmark["x"])
    cy = float(landmark["y"])
    cz = float(landmark.get("z", 1.5))
    yaw = float(landmark.get("yaw", 0.0))

    if "pole" in lm_type or "cyl" in lm_type:
        r_xy = np.hypot(pts_dense[:, 0] - cx, pts_dense[:, 1] - cy)
        d_rad = np.maximum(0.0, r_xy - POLE_RADIUS)
        d_z_top = np.maximum(0.0, pts_dense[:, 2] - POLE_HEIGHT)
        d_z_bot = np.maximum(0.0, 0.0 - pts_dense[:, 2])
        d_z = np.maximum(d_z_top, d_z_bot)
        dists = np.hypot(d_rad, d_z)
        d_sampled = float(np.min(dists))
    else:
        pts_loc = world_to_landmark_local(pts_dense, cx, cy, cz, yaw)
        if "circ" in lm_type or "torus" in lm_type:
            cyl_specs = get_circular_gate_cylinders_spec()
            dists = points_to_finite_cylinders_distance(pts_loc, cyl_specs)
            d_sampled = float(np.min(dists))
        else:
            bar_dists = [
                points_to_aabb_distance(pts_loc, b_c, b_s)
                for b_c, b_s in SQUARE_GATE_BARS_LOCAL
            ]
            d_sampled = float(np.min(np.minimum.reduce(bar_dists)))

    margin = 0.5 * h_effective
    d_lower = max(0.0, d_sampled - margin) if math.isfinite(d_sampled) else float("inf")
    return {
        "sampled_min_distance_m": d_sampled,
        "discretization_step_h_m": h_effective,
        "discretization_margin_m": margin,
        "lower_bound_distance_m": d_lower,
    }
