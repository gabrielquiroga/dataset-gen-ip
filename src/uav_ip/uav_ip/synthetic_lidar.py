#!/usr/bin/env python3
"""
synthetic_lidar.py — Generador de Nubes de Puntos Sinteticas LiDAR 360°

Implementa la observacion privilegiada para UAV Instant Policy:
- Modela landmarks canonicos en alta resolucion:
    * square_gate (marco rectangular de 4 barras)
    * circular_gate (toroide / aro circular)
    * pole (cilindro vertical)
    * landing_pad (plataforma de aterrizaje prismatica o cilindrica)
- Transforma puntos canonicos al marco egocentrico del dron (FLU):
    P_dron = T_{world, dron}^{-1} * T_{world, landmark} * P_can
- Aplica filtro de alcance LiDAR 360° (R <= R_max = 10.0 m)
- Aplica factor de escala metrica s = 0.1 para evitar aliasing en embeddings sinusoidales
- Submuestrea con Farthest Point Sampling (FPS) a exactamente 2048 puntos
- Retorna tensores de forma (2048, 3)

Uso CLI:
    python3 synthetic_lidar.py --test
    python3 synthetic_lidar.py --waypoints /path/to/waypoints.yaml --drone-pose 5.0 0.0 1.5 0.0
"""

import argparse
import math
from pathlib import Path
from typing import Optional, Union, Sequence, Dict, Any, List

import numpy as np

# Intentar importar torch si esta disponible; si no, operar puramente con numpy
try:
    import torch
    HAS_TORCH = True
except ImportError:
    torch = None
    HAS_TORCH = False


# ============================================================================
# Transformaciones y Marcos de Coordenadas
# ============================================================================

def rot_z(yaw: float) -> np.ndarray:
    """Matriz de rotacion 3x3 para un angulo de yaw (alrededor de eje Z)."""
    c = math.cos(yaw)
    s = math.sin(yaw)
    return np.array([
        [c, -s, 0.0],
        [s,  c, 0.0],
        [0.0, 0.0, 1.0]
    ], dtype=np.float64)


def quaternion_to_rot_matrix(q: Sequence[float]) -> np.ndarray:
    """
    Convierte un cuaternion hamiltoniano [w, x, y, z] a matriz de rotacion 3x3.
    """
    w, x, y, z = float(q[0]), float(q[1]), float(q[2]), float(q[3])
    norm = math.sqrt(w * w + x * x + y * y + z * z)
    if norm < 1e-9:
        return np.eye(3, dtype=np.float64)
    w, x, y, z = w / norm, x / norm, y / norm, z / norm

    return np.array([
        [1.0 - 2.0 * (y * y + z * z), 2.0 * (x * y - w * z),       2.0 * (x * z + w * y)],
        [2.0 * (x * y + w * z),       1.0 - 2.0 * (x * x + z * z), 2.0 * (y * z - w * x)],
        [2.0 * (x * z - w * y),       2.0 * (y * z + w * x),       1.0 - 2.0 * (x * x + y * y)]
    ], dtype=np.float64)


def ned_to_gazebo_enu(pos_ned: Sequence[float]) -> np.ndarray:
    """
    Convierte traslacion desde PX4 NED (North-East-Down) a Gazebo ENU (East-North-Up).
    x_enu = y_ned (East)
    y_enu = x_ned (North)
    z_enu = -z_ned (Up)
    """
    return np.array([pos_ned[1], pos_ned[0], -pos_ned[2]], dtype=np.float64)


def yaw_ned_to_gazebo_enu(yaw_ned: float) -> float:
    """
    Convierte yaw en radianes desde PX4 NED (0=Norte, horario)
    a Gazebo ENU (0=Este, antihorario).
    yaw_enu = pi/2 - yaw_ned
    """
    return (math.pi / 2.0) - yaw_ned


def extract_yaw_from_rot_matrix(R: np.ndarray) -> float:
    """
    Extrae el angulo de yaw (rotacion alrededor de Z en ENU) a partir de una
    matriz de rotacion 3x3 FLU-a-ENU:
        yaw_enu = atan2(R[1, 0], R[0, 0])
    """
    return float(math.atan2(float(R[1, 0]), float(R[0, 0])))


def px4_odometry_to_gazebo_enu(
    pos_ned: Sequence[float],
    q_ned: Sequence[float],
    gravity_aligned: bool = False
) -> tuple[np.ndarray, np.ndarray]:
    """
    Convierte pose completa de PX4 (posicion NED + cuaternion FRD-a-NED)
    a Gazebo ENU (posicion ENU + matriz de rotacion FLU-a-ENU).

    Relacion:
        v_enu = M_world * R_ned * M_body * v_flu
    donde:
        M_world = [[0, 1, 0], [1, 0, 0], [0, 0, -1]]
        M_body  = diag([1, -1, -1])

    Si gravity_aligned=True, proyecta la rotacion al marco Horizontal FLU
    estabilizado por gravedad (conservando unicamente el yaw y anulando
    el acoplamiento de roll/pitch del chasis subactuado).
    """
    pos_enu = ned_to_gazebo_enu(pos_ned)
    R_ned = quaternion_to_rot_matrix(q_ned)

    M_world = np.array([
        [0.0, 1.0,  0.0],
        [1.0, 0.0,  0.0],
        [0.0, 0.0, -1.0]
    ], dtype=np.float64)
    M_body = np.diag([1.0, -1.0, -1.0])

    R_enu_flu = M_world @ R_ned @ M_body
    if gravity_aligned:
        yaw_enu = extract_yaw_from_rot_matrix(R_enu_flu)
        R_enu_flu = rot_z(yaw_enu)
    return pos_enu, R_enu_flu


# ============================================================================
# Farthest Point Sampling (FPS)
# ============================================================================

def fps_numpy(
    points: np.ndarray,
    n_samples: int = 2048,
    seed: Optional[int] = 42,
    empty_fallback_radius: float = 1.0
) -> np.ndarray:
    """
    Farthest Point Sampling optimizado en numpy con aceleracion vectorial BLAS.

    Garantiza una cobertura espacial homogenea y determinista (seed=42 por defecto).
    - Si N == 0: coloca los puntos dummy en un anillo sobre la frontera exterior
      del rango (R = empty_fallback_radius en el plano XY, z = 0), nunca en (0,0,0)
      para no falsificar obstaculos sobre el centro de masa del dron.
    - Si N < n_samples: rellena con repeticion aleatoria uniforme determinista.
    - Si N > 4096: realiza un presubmuestreo uniforme determinista a 4096 puntos
      para mantener latencia < 20 ms sin degradar la representacion geometrica.
    """
    points = np.asarray(points, dtype=np.float32)
    N = len(points)

    if N == 0:
        if empty_fallback_radius <= 0.0:
            return np.zeros((n_samples, 3), dtype=np.float32)
        angles = np.linspace(0.0, 2.0 * math.pi, n_samples, endpoint=False, dtype=np.float32)
        boundary_pts = np.column_stack([
            empty_fallback_radius * np.cos(angles),
            empty_fallback_radius * np.sin(angles),
            np.zeros(n_samples, dtype=np.float32)
        ])
        return boundary_pts.astype(np.float32)

    rng = np.random.RandomState(42 if seed is None else int(seed))

    if N <= n_samples:
        indices = np.arange(N)
        if N < n_samples:
            pad = rng.choice(N, n_samples - N, replace=True)
            indices = np.concatenate([indices, pad])
        return points[indices]

    # Pre-submuestreo determinista si la nube supera 4096 puntos
    if N > 4096:
        sub_idx = rng.choice(N, 4096, replace=False)
        working_points = points[sub_idx]
        N = 4096
    else:
        working_points = points

    selected = np.zeros(n_samples, dtype=np.int64)
    current = int(rng.randint(0, N))

    fps_c_fn = _get_fps_c_kernel()
    if fps_c_fn is not None and N <= 8192:
        import ctypes
        cols = np.ascontiguousarray(working_points.T, dtype=np.float32)
        fps_c_fn(
            cols[0].ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
            cols[1].ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
            cols[2].ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
            int(N),
            int(n_samples),
            int(current),
            selected.ctypes.data_as(ctypes.POINTER(ctypes.c_int64)),
        )
        return working_points[selected]

    distances = np.full(N, np.inf, dtype=np.float32)
    norm_sq = np.einsum("ij,ij->i", working_points, working_points)
    neg_two_wp = (-2.0 * working_points).astype(np.float32)
    dists = np.empty(N, dtype=np.float32)
    dot_fn = np.dot
    min_fn = np.minimum
    argmax_fn = distances.argmax

    for i in range(n_samples):
        selected[i] = current
        dot_fn(neg_two_wp, working_points[current], out=dists)
        dists += norm_sq
        dists += norm_sq[current]
        min_fn(distances, dists, out=distances)
        current = int(argmax_fn())

    return working_points[selected]


_FPS_C_KERNEL = False


def _get_fps_c_kernel():
    """Compila y carga dinamicamente (una sola vez) un kernel C SIMD para acelerar FPS ~7x."""
    global _FPS_C_KERNEL
    if _FPS_C_KERNEL is not False:
        return _FPS_C_KERNEL

    import ctypes
    import os
    import subprocess
    import tempfile

    so_path = Path(tempfile.gettempdir()) / f"uav_ip_fps_soa_py{os.getuid()}.so"
    c_code = """
#include <stdint.h>
void fps_soa(const float* restrict x, const float* restrict y, const float* restrict z,
             int N, int K, int start_idx, int64_t* restrict selected) {
    float dists[8192];
    for (int j = 0; j < N; ++j) dists[j] = 1e30f;
    int current = start_idx;
    for (int i = 0; i < K; ++i) {
        selected[i] = current;
        float px = x[current];
        float py = y[current];
        float pz = z[current];
        for (int j = 0; j < N; ++j) {
            float dx = x[j] - px;
            float dy = y[j] - py;
            float dz = z[j] - pz;
            float d = dx * dx + dy * dy + dz * dz;
            dists[j] = (d < dists[j]) ? d : dists[j];
        }
        int best_idx = 0;
        float best_d = dists[0];
        for (int j = 1; j < N; ++j) {
            if (dists[j] > best_d) {
                best_d = dists[j];
                best_idx = j;
            }
        }
        current = best_idx;
    }
}
"""
    try:
        if not so_path.is_file():
            with tempfile.NamedTemporaryFile(suffix=".c", mode="w", delete=False) as f_c:
                f_c.write(c_code)
                c_name = f_c.name
            tmp_so = so_path.with_suffix(f".{os.getpid()}.so")
            try:
                subprocess.run(
                    ["gcc", "-O3", "-march=native", "-ffast-math", "-shared", "-fPIC", c_name, "-o", str(tmp_so)],
                    check=True,
                    stdout=subprocess.DEVNULL,
                    stderr=subprocess.DEVNULL,
                    timeout=10,
                )
                os.replace(tmp_so, so_path)
            finally:
                if os.path.exists(c_name):
                    os.remove(c_name)
                if os.path.exists(tmp_so):
                    os.remove(tmp_so)

        lib = ctypes.CDLL(str(so_path))
        lib.fps_soa.argtypes = [
            ctypes.POINTER(ctypes.c_float),
            ctypes.POINTER(ctypes.c_float),
            ctypes.POINTER(ctypes.c_float),
            ctypes.c_int,
            ctypes.c_int,
            ctypes.c_int,
            ctypes.POINTER(ctypes.c_int64),
        ]
        _FPS_C_KERNEL = lib.fps_soa
    except Exception:
        _FPS_C_KERNEL = None

    return _FPS_C_KERNEL



# ============================================================================
# Generadores de Geometrias Canonicas
# ============================================================================

def sample_box_surface(
    center: Sequence[float],
    size: Sequence[float],
    n_points: int,
    rng: Optional[np.random.RandomState] = None
) -> np.ndarray:
    """
    Muestrea n_points homogeneamente sobre las 6 caras de una caja 3D.
    """
    if rng is None:
        rng = np.random.RandomState()

    cx, cy, cz = center
    dx, dy, dz = size
    hx, hy, hz = dx / 2.0, dy / 2.0, dz / 2.0

    # Areas de las caras
    area_x = dy * dz  # 2 caras en ±X
    area_y = dx * dz  # 2 caras en ±Y
    area_z = dx * dy  # 2 caras en ±Z
    total_area = 2.0 * (area_x + area_y + area_z)

    # Probabilidad proporcional a superficie
    p = np.array([area_x, area_x, area_y, area_y, area_z, area_z]) / total_area
    face_counts = rng.multinomial(n_points, p)

    pts = []
    # +X face
    if face_counts[0] > 0:
        pts.append(np.column_stack([
            np.full(face_counts[0], cx + hx),
            rng.uniform(cy - hy, cy + hy, face_counts[0]),
            rng.uniform(cz - hz, cz + hz, face_counts[0])
        ]))
    # -X face
    if face_counts[1] > 0:
        pts.append(np.column_stack([
            np.full(face_counts[1], cx - hx),
            rng.uniform(cy - hy, cy + hy, face_counts[1]),
            rng.uniform(cz - hz, cz + hz, face_counts[1])
        ]))
    # +Y face
    if face_counts[2] > 0:
        pts.append(np.column_stack([
            rng.uniform(cx - hx, cx + hx, face_counts[2]),
            np.full(face_counts[2], cy + hy),
            rng.uniform(cz - hz, cz + hz, face_counts[2])
        ]))
    # -Y face
    if face_counts[3] > 0:
        pts.append(np.column_stack([
            rng.uniform(cx - hx, cx + hx, face_counts[3]),
            np.full(face_counts[3], cy - hy),
            rng.uniform(cz - hz, cz + hz, face_counts[3])
        ]))
    # +Z face
    if face_counts[4] > 0:
        pts.append(np.column_stack([
            rng.uniform(cx - hx, cx + hx, face_counts[4]),
            rng.uniform(cy - hy, cy + hy, face_counts[4]),
            np.full(face_counts[4], cz + hz)
        ]))
    # -Z face
    if face_counts[5] > 0:
        pts.append(np.column_stack([
            rng.uniform(cx - hx, cx + hx, face_counts[5]),
            rng.uniform(cy - hy, cy + hy, face_counts[5]),
            np.full(face_counts[5], cz - hz)
        ]))

    return np.vstack(pts) if pts else np.zeros((0, 3), dtype=np.float32)


def generate_canonical_square_gate(
    n_points: int = 8000,
    width: float = 1.5,
    height: float = 1.6,
    thickness: float = 0.1,
    seed: Optional[int] = 42
) -> np.ndarray:
    """
    Genera nube canonica para una puerta cuadrada ('square_gate' / 'gate')
    centrada en (0, 0, 0), con la abertura normal al eje X (orientada para cruce en +X).
    Consiste en 4 vigas rectangulares identicas al modelo en circuito_aros_01.sdf:
      - Viga superior: [0, 0, 0.75], size [0.1, 1.5, 0.1]
      - Viga inferior: [0, 0, -0.75], size [0.1, 1.5, 0.1]
      - Pilar izquierdo: [0, 0.7, 0], size [0.1, 0.1, 1.4]
      - Pilar derecho: [0, -0.7, 0], size [0.1, 0.1, 1.4]
    """
    rng = np.random.RandomState(seed)
    pts_per_beam = n_points // 4

    b_top = sample_box_surface((0.0, 0.0, 0.75), (thickness, width, thickness), pts_per_beam, rng)
    b_bot = sample_box_surface((0.0, 0.0, -0.75), (thickness, width, thickness), pts_per_beam, rng)
    b_left = sample_box_surface((0.0, 0.70, 0.0), (thickness, thickness, height - 2 * thickness), pts_per_beam, rng)
    b_right = sample_box_surface((0.0, -0.70, 0.0), (thickness, thickness, height - 2 * thickness), pts_per_beam, rng)

    return np.vstack([b_top, b_bot, b_left, b_right]).astype(np.float32)


def generate_canonical_circular_gate(
    n_points: int = 8000,
    major_radius: float = 0.75,
    tube_radius: float = 0.05,
    seed: Optional[int] = 42
) -> np.ndarray:
    """
    Genera nube canonica para un aro circular / toroide centrado en (0, 0, 0)
    en el plano Y-Z, con eje de cruce a lo largo de X.
    Ecuaciones parametricas:
      x = r * sin(theta)
      y = (R + r * cos(theta)) * cos(phi)
      z = (R + r * cos(theta)) * sin(phi)
    """
    rng = np.random.RandomState(seed)
    theta = rng.uniform(0.0, 2.0 * math.pi, n_points)
    phi = rng.uniform(0.0, 2.0 * math.pi, n_points)

    x = tube_radius * np.sin(theta)
    r_ring = major_radius + tube_radius * np.cos(theta)
    y = r_ring * np.cos(phi)
    z = r_ring * np.sin(phi)

    return np.column_stack([x, y, z]).astype(np.float32)


def generate_canonical_pole(
    n_points: int = 8000,
    radius: float = 0.1,
    height: float = 3.0,
    seed: Optional[int] = 42
) -> np.ndarray:
    """
    Genera nube canonica para un poste cilindrico centrado en (0, 0, 0)
    a lo largo del eje Z (z de -height/2 a +height/2).
    """
    rng = np.random.RandomState(seed)
    # Area lateral vs tapas
    area_lat = 2.0 * math.pi * radius * height
    area_cap = 2.0 * math.pi * (radius ** 2)
    p_lat = area_lat / (area_lat + area_cap)

    n_lat = int(n_points * p_lat)
    n_caps = n_points - n_lat

    # Superficie lateral
    theta_lat = rng.uniform(0.0, 2.0 * math.pi, n_lat)
    z_lat = rng.uniform(-height / 2.0, height / 2.0, n_lat)
    x_lat = radius * np.cos(theta_lat)
    y_lat = radius * np.sin(theta_lat)
    pts_lat = np.column_stack([x_lat, y_lat, z_lat])

    # Tapas
    n_top = n_caps // 2
    n_bot = n_caps - n_top
    r_top = radius * np.sqrt(rng.uniform(0.0, 1.0, n_top))
    t_top = rng.uniform(0.0, 2.0 * math.pi, n_top)
    pts_top = np.column_stack([
        r_top * np.cos(t_top),
        r_top * np.sin(t_top),
        np.full(n_top, height / 2.0)
    ])

    r_bot = radius * np.sqrt(rng.uniform(0.0, 1.0, n_bot))
    t_bot = rng.uniform(0.0, 2.0 * math.pi, n_bot)
    pts_bot = np.column_stack([
        r_bot * np.cos(t_bot),
        r_bot * np.sin(t_bot),
        np.full(n_bot, -height / 2.0)
    ])

    return np.vstack([pts_lat, pts_top, pts_bot]).astype(np.float32)


def generate_canonical_landing_pad(
    n_points: int = 1000,
    size_x: float = 2.0,
    size_y: float = 2.0,
    height: float = 0.1,
    seed: Optional[int] = 42,
    include_bottom: bool = False
) -> np.ndarray:
    """
    Genera nube canonica para la plataforma de aterrizaje centrada en (0, 0, 0).
    Por defecto muestrea exclusivamente la cara superior (+Z) y los bordes laterales
    (excluyendo la cara inferior invisible apoyada en el suelo) con una cuota acotada
    (n_points=1000) para evitar eclipsar los landmarks principales en el submuestreo FPS.
    """
    rng = np.random.RandomState(42 if seed is None else int(seed))
    if include_bottom:
        return sample_box_surface((0.0, 0.0, 0.0), (size_x, size_y, height), n_points, rng).astype(np.float32)

    hx, hy, hz = size_x / 2.0, size_y / 2.0, height / 2.0
    area_x = size_y * height
    area_y = size_x * height
    area_top = size_x * size_y
    total_area = 2.0 * area_x + 2.0 * area_y + area_top

    p = np.array([area_x, area_x, area_y, area_y, area_top]) / total_area
    face_counts = rng.multinomial(n_points, p)

    pts = []
    # +X face
    if face_counts[0] > 0:
        pts.append(np.column_stack([
            np.full(face_counts[0], hx),
            rng.uniform(-hy, hy, face_counts[0]),
            rng.uniform(-hz, hz, face_counts[0])
        ]))
    # -X face
    if face_counts[1] > 0:
        pts.append(np.column_stack([
            np.full(face_counts[1], -hx),
            rng.uniform(-hy, hy, face_counts[1]),
            rng.uniform(-hz, hz, face_counts[1])
        ]))
    # +Y face
    if face_counts[2] > 0:
        pts.append(np.column_stack([
            rng.uniform(-hx, hx, face_counts[2]),
            np.full(face_counts[2], hy),
            rng.uniform(-hz, hz, face_counts[2])
        ]))
    # -Y face
    if face_counts[3] > 0:
        pts.append(np.column_stack([
            rng.uniform(-hx, hx, face_counts[3]),
            np.full(face_counts[3], -hy),
            rng.uniform(-hz, hz, face_counts[3])
        ]))
    # +Z top face
    if face_counts[4] > 0:
        pts.append(np.column_stack([
            rng.uniform(-hx, hx, face_counts[4]),
            rng.uniform(-hy, hy, face_counts[4]),
            np.full(face_counts[4], hz)
        ]))

    return np.vstack(pts).astype(np.float32) if pts else np.zeros((0, 3), dtype=np.float32)


# ============================================================================
# Generador Sintetico LiDAR 360° Principal
# ============================================================================

class SyntheticLidarGenerator:
    """
    Generador de nubes LiDAR 360° omnidireccionales sinteticas (R <= 10.0 m, 2048 puntos, escala s = 0.1).
    Por defecto opera en el marco egocentrico estabilizado por gravedad (Horizontal FLU,
    gravity_aligned=True) para eliminar la distorsion geometrica inducida por el cabeceo (pitch)
    y alabeo (roll) de cuadricopteros subactuados.
    """

    def __init__(
        self,
        r_max: float = 10.0,
        r_min: float = 0.1,
        scale_factor: float = 0.1,
        target_points: int = 2048,
        cache_models: bool = True,
        gravity_aligned: bool = True
    ):
        """
        Args:
            r_max: Rango maximo del LiDAR en metros (default: 10.0 m).
            r_min: Rango minimo del LiDAR en metros (default: 0.1 m).
            scale_factor: Factor de escala metrica s (default: 0.1 para coincidir con espacio sinusoidal).
            target_points: Cantidad exacta de puntos de salida (default: 2048).
            cache_models: Si True, precalcula y cachea los modelos canonicos de alta resolucion.
            gravity_aligned: Si True (default), proyecta al marco Horizontal FLU estabilizado por
                gravedad (rotando exclusivamente por yaw).
        """
        self.r_max = float(r_max)
        self.r_min = float(r_min)
        self.scale_factor = float(scale_factor)
        self.target_points = int(target_points)
        self.gravity_aligned = bool(gravity_aligned)

        self._canonical_cache: Dict[str, np.ndarray] = {}
        if cache_models:
            self._init_cache()

    def _init_cache(self) -> None:
        """Inicializa los modelos canonicos de alta resolucion."""
        self._canonical_cache["square_gate"] = generate_canonical_square_gate(8000)
        self._canonical_cache["gate"] = self._canonical_cache["square_gate"]
        self._canonical_cache["circular_gate"] = generate_canonical_circular_gate(8000)
        self._canonical_cache["torus"] = self._canonical_cache["circular_gate"]
        self._canonical_cache["pole"] = generate_canonical_pole(8000)
        self._canonical_cache["cylinder"] = self._canonical_cache["pole"]
        self._canonical_cache["landing_pad"] = generate_canonical_landing_pad(1000)
        self._canonical_cache["pad"] = self._canonical_cache["landing_pad"]

    def get_canonical_points(self, landmark_type: str) -> np.ndarray:
        """Retorna puntos canonicos de superficie para el tipo de landmark dado."""
        key = landmark_type.lower()
        if key in self._canonical_cache:
            return self._canonical_cache[key]

        # Si no esta en cache, generar segun tipo
        if "gate" in key or "aro" in key:
            if "circ" in key:
                pts = generate_canonical_circular_gate(8000)
            else:
                pts = generate_canonical_square_gate(8000)
        elif "pole" in key or "poste" in key or "cyl" in key:
            pts = generate_canonical_pole(8000)
        elif "pad" in key or "land" in key or "plat" in key:
            pts = generate_canonical_landing_pad(1000)
        else:
            # Fallback a caja generica
            pts = sample_box_surface((0.0, 0.0, 0.0), (1.0, 1.0, 1.0), 4000).astype(np.float32)

        self._canonical_cache[key] = pts
        return pts

    def generate_pointcloud(
        self,
        drone_pose: Union[Dict[str, Any], Sequence[float]],
        landmarks: List[Dict[str, Any]],
        landing: Optional[Dict[str, Any]] = None,
        as_tensor: bool = False,
        seed: Optional[int] = 42,
        gravity_aligned: Optional[bool] = None
    ) -> Union[np.ndarray, "torch.Tensor"]:
        """
        Genera la observacion LiDAR 360° sintetica para una pose dada del dron.

        Args:
            drone_pose: Pose del dron en coordenadas Gazebo ENU.
                Puede ser:
                  - dict: {'x': float, 'y': float, 'z': float, 'yaw': float}
                  - dict: {'x': float, 'y': float, 'z': float, 'q_ned': [w,x,y,z]}
                  - dict: {'x': float, 'y': float, 'z': float, 'R': np.ndarray(3,3)}
                  - lista/tupla: [x, y, z, yaw]
            landmarks: Lista de diccionarios con landmarks desde waypoints.yaml:
                [{'id': str, 'type': str, 'x': float, 'y': float, 'z': float, 'yaw': float}, ...]
            landing: Opcional dict con posicion de la plataforma de llegada:
                {'x': float, 'y': float, 'z': optional}
            as_tensor: Si True, retorna torch.Tensor (requiere torch instalado).
            seed: Semilla determinista para el submuestreo FPS (default: 42).
            gravity_aligned: Si se especifica, sobreescribe self.gravity_aligned.

        Returns:
            Arreglo o tensor de dimension (2048, 3) escalado por s = 0.1 en marco egocentrico FLU
            (o Horizontal FLU si gravity_aligned=True).
        """
        use_gravity_aligned = self.gravity_aligned if gravity_aligned is None else bool(gravity_aligned)

        # Extraer posicion y rotacion del dron en Gazebo ENU
        if isinstance(drone_pose, dict):
            p_drone = np.array([drone_pose["x"], drone_pose["y"], drone_pose["z"]], dtype=np.float64)
            if "yaw" in drone_pose and ("R" not in drone_pose and "q_ned" not in drone_pose and "q" not in drone_pose):
                R_w_drone = rot_z(float(drone_pose["yaw"]))
            elif "R" in drone_pose:
                R_w_drone = np.asarray(drone_pose["R"], dtype=np.float64)
            elif "q_ned" in drone_pose:
                _, R_w_drone = px4_odometry_to_gazebo_enu(
                    [0.0, 0.0, 0.0], drone_pose["q_ned"], gravity_aligned=use_gravity_aligned
                )
            elif "q" in drone_pose:
                R_w_drone = quaternion_to_rot_matrix(drone_pose["q"])
            else:
                R_w_drone = np.eye(3, dtype=np.float64)

            if use_gravity_aligned:
                yaw_enu = extract_yaw_from_rot_matrix(R_w_drone)
                R_w_drone = rot_z(yaw_enu)
        else:
            p_drone = np.array([drone_pose[0], drone_pose[1], drone_pose[2]], dtype=np.float64)
            yaw = float(drone_pose[3]) if len(drone_pose) > 3 else 0.0
            R_w_drone = rot_z(yaw)

        # Preparar lista completa de objetos (landmarks + landing pad)
        all_objects = []
        for lm in landmarks:
            lm_c = dict(lm)
            if "id" not in lm_c and "name" in lm_c:
                lm_c["id"] = lm_c["name"]
            if "type" not in lm_c:
                lm_c["type"] = "square_gate"
            # Soportar tanto 'x','y','z','yaw' planos como 'position' y 'rpy' de waypoints.yaml
            if "x" not in lm_c and "position" in lm_c:
                pos_seq = lm_c["position"]
                lm_c["x"] = float(pos_seq[0])
                lm_c["y"] = float(pos_seq[1])
                lm_c["z"] = float(pos_seq[2])
            if "yaw" not in lm_c and "rpy" in lm_c:
                lm_c["yaw"] = float(lm_c["rpy"][2])
            all_objects.append(lm_c)

        if landing is not None:
            landing_obj = {
                "id": "landing_pad",
                "type": "landing_pad",
                "x": float(landing["x"]),
                "y": float(landing["y"]),
                "z": float(landing.get("z", 0.05)),
                "yaw": float(landing.get("yaw", 0.0))
            }
            all_objects.append(landing_obj)

        accumulated_world_pts = []

        # Rango de pre-filtrado con margen de radio de objeto
        prefilter_r = self.r_max + 3.0

        for obj in all_objects:
            obj_type = str(obj.get("type", "square_gate")).lower()
            raw_z = obj.get("z", None)
            if raw_z is None or (float(raw_z) < 0.1 and ("pole" in obj_type or "poste" in obj_type or "cyl" in obj_type)):
                obj_z = 1.5
            else:
                obj_z = float(raw_z)

            obj_pos = np.array([float(obj["x"]), float(obj["y"]), obj_z], dtype=np.float64)
            dist_to_drone = np.linalg.norm(obj_pos - p_drone)

            # Pre-filtro espacial: si el centro del objeto esta a mas de (r_max + 3m), descartar
            if dist_to_drone > prefilter_r:
                continue

            # Obtener puntos canonicos
            pts_can = self.get_canonical_points(obj_type)
            obj_yaw = float(obj.get("yaw", 0.0))
            R_w_obj = rot_z(obj_yaw)

            # Transformar a Gazebo World ENU: P_world = P_can * R_{w,obj}^T + p_{obj}
            pts_world = pts_can @ R_w_obj.T + obj_pos
            accumulated_world_pts.append(pts_world)

        if not accumulated_world_pts:
            pts_drone_filtered = np.zeros((0, 3), dtype=np.float32)
        else:
            all_pts_world = np.vstack(accumulated_world_pts)

            # Transformar a marco local del dron (Horizontal FLU si gravity_aligned=True):
            # P_drone = (P_world - p_drone) * R_{w,drone}
            pts_drone = (all_pts_world - p_drone) @ R_w_drone

            # Filtro estricto de visibilidad por distancia (LiDAR 360° de radio r_max)
            distances = np.linalg.norm(pts_drone, axis=1)
            mask = (distances >= self.r_min) & (distances <= self.r_max)
            pts_drone_filtered = pts_drone[mask]

        # Aplicar factor de escala metrica s = 0.1
        pts_scaled = (pts_drone_filtered * self.scale_factor).astype(np.float32)

        # Submuestreo por Farthest Point Sampling a exactamente 2048 puntos.
        # Si N=0 (ningun objeto dentro de r_max), coloca los puntos en la frontera exterior
        # s * r_max (ej. 1.0), nunca en el origen (0,0,0).
        pts_sampled = fps_numpy(
            pts_scaled,
            n_samples=self.target_points,
            seed=seed,
            empty_fallback_radius=self.r_max * self.scale_factor,
        )

        if as_tensor:
            if not HAS_TORCH:
                raise RuntimeError("PyTorch no esta instalado en este entorno de Python.")
            return torch.from_numpy(pts_sampled)

        return pts_sampled


# ============================================================================
# Pruebas y CLI
# ============================================================================

def run_self_tests() -> bool:
    """Ejecuta una bateria completa de tests unitarios y de consistencia."""
    print("Iniciando bateria de tests para synthetic_lidar.py...")

    # 1. Test de modelos canonicos y exclusion de cara inferior en landing_pad
    g_sq = generate_canonical_square_gate(2000)
    assert g_sq.shape == (2000, 3), f"Forma incorrecta square_gate: {g_sq.shape}"
    g_circ = generate_canonical_circular_gate(2000)
    assert g_circ.shape == (2000, 3), f"Forma incorrecta circular_gate: {g_circ.shape}"
    p_pole = generate_canonical_pole(2000)
    assert p_pole.shape == (2000, 3), f"Forma incorrecta pole: {p_pole.shape}"
    pad = generate_canonical_landing_pad(1000, height=0.1)
    assert pad.shape == (1000, 3), f"Forma incorrecta landing_pad: {pad.shape}"
    assert np.sum(np.isclose(pad[:, 2], 0.05)) > 800, "La mayoria de puntos de landing_pad deben estar en la cara superior"
    print("  [OK] Generacion de modelos canonicos verificada.")

    # 2. Test de Farthest Point Sampling (FPS) y fallback en frontera exterior
    test_pts = np.random.uniform(-5, 5, size=(5000, 3)).astype(np.float32)
    sampled = fps_numpy(test_pts, 2048, seed=42)
    assert sampled.shape == (2048, 3), f"FPS retorno forma {sampled.shape}"
    # Verificar caso N < 2048
    small_pts = np.random.uniform(-1, 1, size=(50, 3)).astype(np.float32)
    sampled_pad = fps_numpy(small_pts, 2048, seed=42)
    assert sampled_pad.shape == (2048, 3), f"FPS padding fallo: {sampled_pad.shape}"
    # Verificar caso N = 0 -> anillo en frontera exterior (r = 1.0), nunca en (0,0,0)
    empty_pts = np.zeros((0, 3), dtype=np.float32)
    sampled_empty = fps_numpy(empty_pts, 2048, seed=42, empty_fallback_radius=1.0)
    assert sampled_empty.shape == (2048, 3)
    assert np.allclose(np.linalg.norm(sampled_empty, axis=1), 1.0, atol=1e-5)
    print("  [OK] Farthest Point Sampling (FPS) y casos de borde verificados.")

    # 3. Test de transformaciones de marco (NED/FRD -> ENU/FLU) para multiples rumbos
    pos_ned = [5.0, 10.0, -2.0]
    pos_enu = ned_to_gazebo_enu(pos_ned)
    assert np.allclose(pos_enu, [10.0, 5.0, 2.0]), f"Conversion NED->ENU erro: {pos_enu}"

    # Caso A: Drone mirando al Este (yaw_ned = pi/2 -> yaw_enu = 0)
    yaw_ned_east = math.pi / 2.0
    yaw_enu_east = yaw_ned_to_gazebo_enu(yaw_ned_east)
    assert abs(yaw_enu_east - 0.0) < 1e-6, f"Conversion yaw fallo: {yaw_enu_east}"
    q_ned_east = [math.cos(yaw_ned_east / 2.0), 0.0, 0.0, math.sin(yaw_ned_east / 2.0)]
    p_conv_east, R_conv_east = px4_odometry_to_gazebo_enu(pos_ned, q_ned_east)
    assert np.allclose(p_conv_east, pos_enu)
    assert np.allclose(R_conv_east, np.eye(3)), f"R_conv con yaw_enu=0 deberia ser Identidad: {R_conv_east}"

    # Caso B: Drone mirando al Norte (yaw_ned = 0 -> yaw_enu = pi/2)
    yaw_ned_north = 0.0
    yaw_enu_north = yaw_ned_to_gazebo_enu(yaw_ned_north)
    assert abs(yaw_enu_north - math.pi / 2.0) < 1e-6
    q_ned_north = [1.0, 0.0, 0.0, 0.0]
    _, R_conv_north = px4_odometry_to_gazebo_enu([0, 0, 0], q_ned_north)
    assert np.allclose(R_conv_north, rot_z(math.pi / 2.0)), f"R_conv al Norte debe ser rot_z(pi/2): {R_conv_north}"
    print("  [OK] Transformaciones matematicas de marcos verificadas en multiples cuadrantes.")

    # 4. Test de proyeccion egocentrica rigurosa, soporte R / q_ned e invarianza a pitch/roll
    gen = SyntheticLidarGenerator(r_max=10.0, scale_factor=0.1, target_points=2048, gravity_aligned=True)

    # Dron mirando al Norte hacia un aro colocado exactamente al Norte a 5.0m
    lm_north = [{"id": "gate_n", "type": "square_gate", "x": 0.0, "y": 5.0, "z": 1.5, "yaw": math.pi / 2.0}]
    drone_north_R = {"x": 0.0, "y": 0.0, "z": 1.5, "R": R_conv_north}
    cloud_north_R = gen.generate_pointcloud(drone_north_R, lm_north, seed=42)
    mean_north_R = np.mean(cloud_north_R, axis=0)
    assert mean_north_R[0] > 0.45, f"Aro al Norte con rumbo Norte debe tener X_flu > 0.45: {mean_north_R}"
    assert abs(mean_north_R[1]) < 0.05, f"Aro al Norte con rumbo Norte debe tener Y_flu ~ 0: {mean_north_R}"

    # Verificar que un cabeceo (pitch = 20 deg) en q_ned NO altera la nube con gravity_aligned=True
    pitch_rad = math.radians(20.0)
    q_ned_north_pitched = [math.cos(pitch_rad / 2.0), 0.0, math.sin(pitch_rad / 2.0), 0.0]
    drone_north_pitched = {"x": 0.0, "y": 0.0, "z": 1.5, "q_ned": q_ned_north_pitched}
    cloud_north_pitched = gen.generate_pointcloud(drone_north_pitched, lm_north, seed=42, gravity_aligned=True)
    assert np.allclose(cloud_north_R, cloud_north_pitched, atol=1e-6), "gravity_aligned=True debe ser invariante al pitch!"
    print("  [OK] Proyeccion egocentrica e invarianza al pitch/roll (Horizontal FLU) verificadas.")

    # 5. Test del Generador Sintetico completo y retrocompatibilidad con hoops
    dummy_landmarks = [
        {"name": "aro_1", "x": 5.0, "y": 0.0, "z": 1.5, "yaw": 0.0},
        {"id": "pole_1", "type": "pole", "x": 12.0, "y": 1.0, "z": 0.0, "yaw": 0.0},  # z=0 debe centrarse a 1.5m
        {"id": "aro_2", "type": "circular_gate", "x": 30.0, "y": 0.0, "z": 1.5, "yaw": 0.0},  # Fuera de rango
    ]
    landing = {"x": 15.0, "y": 0.0, "z": 0.05}

    drone_pose = {"x": 0.0, "y": 0.0, "z": 1.5, "yaw": 0.0}
    cloud = gen.generate_pointcloud(drone_pose, dummy_landmarks, landing=landing, seed=42)

    assert cloud.shape == (2048, 3), f"Nube sintetica debe ser (2048, 3), obtenida {cloud.shape}"
    max_dist = np.max(np.linalg.norm(cloud, axis=1))
    assert max_dist <= 1.0 + 1e-4, f"Puntos no deben exceder s * r_max = 1.0m. Obtenido {max_dist}"
    mean_x = np.mean(cloud[:, 0])
    assert mean_x > 0.0, f"Los landmarks al frente deben proyectarse con X_flu > 0, obtenido mean_x={mean_x}"
    print(f"  [OK] Generador completo y retrocompatibilidad con hoops validada. Radio maximo: {max_dist:.3f} m.")
    return True


def main():
    parser = argparse.ArgumentParser(description="Generador de Nube de Puntos Sintetica LiDAR 360°")
    parser.add_argument("--test", action="store_true", help="Ejecutar suite de tests unitarios")
    parser.add_argument("--waypoints", type=Path, help="Ruta al archivo waypoints.yaml")
    parser.add_argument(
        "--drone-pose",
        nargs=4,
        type=float,
        metavar=("X", "Y", "Z", "YAW"),
        default=[0.0, 0.0, 1.5, 0.0],
        help="Pose del dron en Gazebo ENU [X Y Z Yaw(rad)]"
    )
    parser.add_argument("--scale", type=float, default=0.1, help="Factor de escala metrica (default: 0.1)")
    parser.add_argument("--r-max", type=float, default=10.0, help="Alcance maximo en metros (default: 10.0)")
    parser.add_argument("--points", type=int, default=2048, help="Puntos objetivo FPS (default: 2048)")

    args = parser.parse_args()

    if args.test:
        success = run_self_tests()
        if success:
            print("\nTODOS LOS TESTS DE SYNTHETIC_LIDAR PASARON SATISFACTORIAMENTE.")
        return

    if args.waypoints:
        import yaml
        with open(args.waypoints, "r") as f:
            cfg = yaml.safe_load(f)

        raw_lms = cfg.get("landmarks") or cfg.get("hoops") or []
        landmarks = []
        for lm in raw_lms:
            lm_c = dict(lm)
            if "id" not in lm_c and "name" in lm_c:
                lm_c["id"] = lm_c["name"]
            if "type" not in lm_c:
                lm_c["type"] = "square_gate"
            landmarks.append(lm_c)
        landing = cfg.get("landing", None)

        gen = SyntheticLidarGenerator(
            r_max=args.r_max,
            scale_factor=args.scale,
            target_points=args.points
        )

        drone_pose = {
            "x": args.drone_pose[0],
            "y": args.drone_pose[1],
            "z": args.drone_pose[2],
            "yaw": args.drone_pose[3]
        }

        cloud = gen.generate_pointcloud(drone_pose, landmarks, landing)
        print(f"Nube generada exitosamente:")
        print(f"  Shape: {cloud.shape}")
        print(f"  Min bounds: {np.min(cloud, axis=0)}")
        print(f"  Max bounds: {np.max(cloud, axis=0)}")
        print(f"  Norm max: {np.max(np.linalg.norm(cloud, axis=1)):.3f}")
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
