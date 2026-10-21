#!/usr/bin/env python3
"""
world_generator.py — Generador de Mundos Aleatorios para Instant Policy.

Toma el SDF plantilla (circuito_aros_01.sdf), aplica variaciones aleatorias
factibles a las poses de los landmarks (puertas cuadradas, aros circulares,
postes cilíndricos), la pose aérea inicial y la plataforma de aterrizaje,
y exporta:
  1. Un archivo .sdf validado para Gazebo Harmonic.
  2. Un archivo waypoints.yaml con las poses para el piloto experto.

Uso:
    python3 world_generator.py --task-id task_0001
    python3 world_generator.py --task-id task_0001 --seed 42
    python3 world_generator.py --task-id task_0001 --output-dir /tmp/worlds
"""

import argparse
import math
import random
import shutil
import sys
import xml.etree.ElementTree as ET
from pathlib import Path
from typing import Optional

import yaml

# ============================================================================
# Geometria del aro (puerta cuadrada y circular)
# ============================================================================
# La puerta cuadrada se extiende ±0.75m en Z y ±0.7m en Y respecto a su centro.
# El aro circular tiene radio mayor 0.75m y radio tubular 0.05m.
GATE_HALF_HEIGHT: float = 0.75
GATE_HALF_WIDTH: float = 0.70
CIRCULAR_MAJOR_RADIUS: float = 0.75
CIRCULAR_TUBE_RADIUS: float = 0.05

# ============================================================================
# Rangos de randomizacion
# ============================================================================
# Primer landmark: posicion absoluta desde el origen (spawn del dron)
FIRST_GATE_X_MIN: float = 4.0
FIRST_GATE_X_MAX: float = 7.0

# Separacion entre landmarks consecutivos (en X)
MIN_GATE_SEP_X: float = 4.0
MAX_GATE_SEP_X: float = 7.0

# Rango lateral (Y) del centro de cada landmark
MIN_GATE_Y: float = -3.0
MAX_GATE_Y: float = 3.0

# Rango de altura (Z) del centro de cada aro
# Minimo: GATE_HALF_HEIGHT + margen, para que la barra inferior no toque el suelo
MIN_GATE_Z: float = 1.0   # barra inferior queda a Z = 1.0 - 0.75 = 0.25m
MAX_GATE_Z: float = 3.0

# Yaw del landmark (radianes): rotacion respecto al eje +X
MAX_GATE_YAW: float = 0.4

# Plataforma de aterrizaje: offset en X desde el ultimo landmark
LANDING_OFFSET_X_MIN: float = 4.0
LANDING_OFFSET_X_MAX: float = 6.0
LANDING_OFFSET_Y_MAX: float = 1.5   # ligero descentrado lateral

# Variacion acotada de la pose inicial en vuelo (tras despegar desde el origen EKF2 (0,0))
START_Y_MAX: float = 0.8
START_Z_MIN: float = 1.2
START_Z_MAX: float = 1.8
START_YAW_MAX: float = 0.20

# Maximo cambio lateral entre landmarks consecutivos
MAX_DELTA_Y_BETWEEN_GATES: float = 3.0

LANDING_MODEL_NAME = "plataforma_llegada"

# Ruta default de PX4 worlds
PX4_WORLDS_DIR = Path.home() / "PX4-Autopilot" / "Tools" / "simulation" / "gz" / "worlds"


# ============================================================================
# Constructores Modulares de Modelos SDF
# ============================================================================

def _create_gate_element(
    name: str, x: float, y: float, z: float, yaw: float
) -> ET.Element:
    """Crea un elemento <model> SDF para un aro o puerta cuadrada con 4 barras."""
    model = ET.Element("model", name=name)
    ET.SubElement(model, "static").text = "true"
    ET.SubElement(model, "pose").text = f"{x} {y} {z} 0 0 {yaw}"

    link = ET.SubElement(model, "link", name="link")

    bars = [
        # (nombre, pose, size, color)
        ("top", "0 0 0.75 0 0 0", "0.1 1.5 0.1", "0.8 0 0 1"),
        ("bot", "0 0 -0.75 0 0 0", "0.1 1.5 0.1", "0.8 0 0 1"),
        ("left", "0 0.7 0 0 0 0", "0.1 0.1 1.4", "0.8 0 0 1"),
        ("right", "0 -0.7 0 0 0 0", "0.1 0.1 1.4", "0.8 0 0 1"),
    ]
    for b_name, b_pose, b_size, b_color in bars:
        vis = ET.SubElement(link, "visual", name=f"{b_name}_v")
        ET.SubElement(vis, "pose").text = b_pose
        g = ET.SubElement(vis, "geometry")
        box = ET.SubElement(g, "box")
        ET.SubElement(box, "size").text = b_size
        mat = ET.SubElement(vis, "material")
        ET.SubElement(mat, "ambient").text = b_color
        ET.SubElement(mat, "diffuse").text = b_color

        col = ET.SubElement(link, "collision", name=f"{b_name}_c")
        ET.SubElement(col, "pose").text = b_pose
        gc = ET.SubElement(col, "geometry")
        box_c = ET.SubElement(gc, "box")
        ET.SubElement(box_c, "size").text = b_size

    return model


def _create_circular_gate_element(
    name: str,
    x: float,
    y: float,
    z: float,
    yaw: float,
    major_radius: float = CIRCULAR_MAJOR_RADIUS,
    tube_radius: float = CIRCULAR_TUBE_RADIUS,
    num_segments: int = 16,
) -> ET.Element:
    """Crea un elemento <model> SDF para un aro circular / toroide en el plano Y-Z.

    Se construye mediante un anillo poligonal de num_segments cilindros cuyas
    dimensiones coinciden exactamente con generate_canonical_circular_gate()
    en synthetic_lidar.py (major_radius=0.75m, tube_radius=0.05m).
    """
    model = ET.Element("model", name=name)
    ET.SubElement(model, "static").text = "true"
    ET.SubElement(model, "pose").text = f"{x} {y} {z} 0 0 {yaw}"

    link = ET.SubElement(model, "link", name="link")
    seg_length = 2.0 * major_radius * math.sin(math.pi / num_segments) * 1.05
    color = "0.85 0.15 0.15 1"

    for k in range(num_segments):
        phi = (2.0 * math.pi * k) / num_segments
        yk = major_radius * math.cos(phi)
        zk = major_radius * math.sin(phi)
        # Rotar el cilindro (eje local Z) alrededor de X por roll=phi
        # alinea su eje con la tangente (0, -sin(phi), cos(phi)) del toroide en Y-Z.
        seg_pose = f"0 {yk:.4f} {zk:.4f} {phi:.4f} 0 0"

        vis = ET.SubElement(link, "visual", name=f"seg_{k}_v")
        ET.SubElement(vis, "pose").text = seg_pose
        g = ET.SubElement(vis, "geometry")
        cyl = ET.SubElement(g, "cylinder")
        ET.SubElement(cyl, "radius").text = f"{tube_radius:.4f}"
        ET.SubElement(cyl, "length").text = f"{seg_length:.4f}"
        mat = ET.SubElement(vis, "material")
        ET.SubElement(mat, "ambient").text = color
        ET.SubElement(mat, "diffuse").text = color

        col = ET.SubElement(link, "collision", name=f"seg_{k}_c")
        ET.SubElement(col, "pose").text = seg_pose
        gc = ET.SubElement(col, "geometry")
        cyl_c = ET.SubElement(gc, "cylinder")
        ET.SubElement(cyl_c, "radius").text = f"{tube_radius:.4f}"
        ET.SubElement(cyl_c, "length").text = f"{seg_length:.4f}"

    return model


def _create_pole_element(
    name: str, x: float, y: float, yaw: float, radius: float = 0.1, length: float = 3.0
) -> ET.Element:
    """Crea un elemento <model> SDF para un poste cilíndrico vertical."""
    model = ET.Element("model", name=name)
    ET.SubElement(model, "static").text = "true"
    ET.SubElement(model, "pose").text = f"{x} {y} 0.0 0 0 {yaw}"

    link = ET.SubElement(model, "link", name="link")

    vis = ET.SubElement(link, "visual", name="pole_v")
    ET.SubElement(vis, "pose").text = f"0 0 {length / 2.0} 0 0 0"
    g = ET.SubElement(vis, "geometry")
    cyl = ET.SubElement(g, "cylinder")
    ET.SubElement(cyl, "radius").text = str(radius)
    ET.SubElement(cyl, "length").text = str(length)
    mat = ET.SubElement(vis, "material")
    ET.SubElement(mat, "ambient").text = "1 0.5 0 1"
    ET.SubElement(mat, "diffuse").text = "1 0.5 0 1"

    col = ET.SubElement(link, "collision", name="pole_c")
    ET.SubElement(col, "pose").text = f"0 0 {length / 2.0} 0 0 0"
    gc = ET.SubElement(col, "geometry")
    cyl_c = ET.SubElement(gc, "cylinder")
    ET.SubElement(cyl_c, "radius").text = str(radius)
    ET.SubElement(cyl_c, "length").text = str(length)

    return model


# ============================================================================
# Generacion de poses aleatorias con validacion
# ============================================================================

def generate_start_pose(seed: Optional[int] = None) -> dict:
    """Genera una pose inicial en vuelo (post-despegue) suavemente aleatorizada.

    Nota: El spawn físico en tierra permanece en (0, 0) para que el origen NED
    de PX4 EKF2 coincida exactamente con el origen ENU de Gazebo, mientras que
    el punto aéreo de inicio del circuito varía entre episodios.
    """
    if seed is None:
        return {"x": 0.0, "y": 0.0, "z": 1.5, "yaw": 0.0}
    rng = random.Random(seed ^ 0x5A5A5A)
    return {
        "x": 0.0,
        "y": round(rng.uniform(-START_Y_MAX, START_Y_MAX), 2),
        "z": round(rng.uniform(START_Z_MIN, START_Z_MAX), 2),
        "yaw": round(rng.uniform(-START_YAW_MAX, START_YAW_MAX), 3),
    }


# ============================================================================
# Verificacion de Factibilidad Geometrica de Transiciones entre Primitivas
# ============================================================================

APPROACH_OFFSET_M: float = 1.2
OVERFLY_Z_OFFSET_M: float = 2.5
SLALOM_LAT_OFFSET_M: float = 1.5
ORBIT_RADIUS_M: float = 1.5

MIN_TRANSITION_DX_M: float = 1.0
MIN_TRANSITION_DXY_M: float = 1.4
MAX_SLOPE_WITH_TRAVERSE: float = 1.15
MAX_SLOPE_OPEN_POLE: float = 1.60
MIN_LANDING_DX_M: float = 1.2


def compute_primitive_anchor_waypoints(
    landmark: dict, primitive: Optional[dict] = None
) -> dict:
    """Calcula los waypoints de entrada ('pre') y salida ('post') de una primitiva en ENU."""
    cx = float(landmark["x"])
    cy = float(landmark["y"])
    cz = float(landmark["z"])
    yaw = float(landmark.get("yaw", 0.0))
    fwd_x = math.cos(yaw)
    fwd_y = math.sin(yaw)
    right_x = math.sin(yaw)
    right_y = -math.cos(yaw)

    p_type = str(primitive.get("type", "traverse")) if primitive else "traverse"

    if p_type == "overfly":
        pre = (
            cx - APPROACH_OFFSET_M * fwd_x,
            cy - APPROACH_OFFSET_M * fwd_y,
            cz + OVERFLY_Z_OFFSET_M,
        )
        post = (
            cx + APPROACH_OFFSET_M * fwd_x,
            cy + APPROACH_OFFSET_M * fwd_y,
            cz + OVERFLY_Z_OFFSET_M,
        )
    elif p_type == "slalom":
        side = str(primitive.get("side", "right")).lower() if primitive else "right"
        sign = 1.0 if side == "right" else -1.0
        rx = right_x * sign * SLALOM_LAT_OFFSET_M
        ry = right_y * sign * SLALOM_LAT_OFFSET_M
        pre = (
            cx - APPROACH_OFFSET_M * fwd_x + rx,
            cy - APPROACH_OFFSET_M * fwd_y + ry,
            cz,
        )
        post = (
            cx + APPROACH_OFFSET_M * fwd_x + rx,
            cy + APPROACH_OFFSET_M * fwd_y + ry,
            cz,
        )
    elif p_type == "orbit":
        pre = (
            cx - ORBIT_RADIUS_M * fwd_x,
            cy - ORBIT_RADIUS_M * fwd_y,
            cz,
        )
        post = (
            cx + ORBIT_RADIUS_M * fwd_x - ORBIT_RADIUS_M * right_x,
            cy + ORBIT_RADIUS_M * fwd_y - ORBIT_RADIUS_M * right_y,
            cz,
        )
    else:
        pre = (
            cx - APPROACH_OFFSET_M * fwd_x,
            cy - APPROACH_OFFSET_M * fwd_y,
            cz,
        )
        post = (
            cx + APPROACH_OFFSET_M * fwd_x,
            cy + APPROACH_OFFSET_M * fwd_y,
            cz,
        )

    return {
        "target": str(landmark["id"]),
        "primitive": p_type,
        "pre": pre,
        "post": post,
    }


def check_circuit_transition_feasibility(
    hoops: list[dict],
    primitives: Optional[list[dict]] = None,
    landing: Optional[dict] = None,
    start_pose: Optional[dict] = None,
) -> tuple[bool, list[str]]:
    """Verifica que las transiciones entre primitivas consecutivas sean fisicamente ejecutables.

    Comprueba:
      1. Progresion longitudinal monotona entre `post_i` y `pre_{i+1}` (`dx >= 1.0m`).
      2. Distancia horizontal minima de maniobra (`dxy >= 1.4m`).
      3. Pendiente vertical maxima (`|dz| / dxy <= 1.15` si interviene un `traverse`,
         o `<= 1.60` entre maniobras abiertas sobre postes).
      4. Ausencia de interferencia entre puntos `pre`/`post` y otros landmarks vecinos.
      5. Margen longitudinal suficiente desde `post_N` hasta la plataforma de aterrizaje.

    Returns:
        `(feasible, reasons)` donde `reasons` es vacia si `feasible == True`.
    """
    reasons: list[str] = []
    if not hoops:
        return True, reasons

    hoops_map = {str(h["id"]): h for h in hoops}
    if primitives:
        prims_sorted = sorted(
            primitives,
            key=lambda p: float(hoops_map.get(str(p.get("target")), {}).get("x", 0.0)),
        )
    else:
        prims_sorted = [{"type": "traverse", "target": h["id"]} for h in hoops]

    anchors = []
    for prim in prims_sorted:
        tid = str(prim.get("target", ""))
        if tid not in hoops_map:
            continue
        anchors.append(compute_primitive_anchor_waypoints(hoops_map[tid], prim))

    if not anchors:
        return True, reasons

    # 1. Verificar salida desde start_pose hasta el primer punto pre
    sx = float((start_pose or {}).get("x", 0.0))
    sy = float((start_pose or {}).get("y", 0.0))
    pre0 = anchors[0]["pre"]
    dx0 = pre0[0] - sx
    dxy0 = math.hypot(pre0[0] - sx, pre0[1] - sy)
    if dx0 < 1.2 or dxy0 < 1.5:
        reasons.append(
            f"start_to_first_pre_too_close (dx={dx0:.2f}m, dxy={dxy0:.2f}m)"
        )

    # 2. Verificar cada transicion post_i -> pre_{i+1}
    for i in range(len(anchors) - 1):
        a_curr = anchors[i]
        a_next = anchors[i + 1]
        p_post = a_curr["post"]
        p_pre = a_next["pre"]

        dx = p_pre[0] - p_post[0]
        dy = p_pre[1] - p_post[1]
        dz = p_pre[2] - p_post[2]
        dxy = math.hypot(dx, dy)

        if dx < MIN_TRANSITION_DX_M:
            reasons.append(
                f"transition_{i}_{i+1}_ longitudinal_overlap (dx={dx:.2f}m < {MIN_TRANSITION_DX_M:.2f}m)"
            )
        if dxy < MIN_TRANSITION_DXY_M:
            reasons.append(
                f"transition_{i}_{i+1}_too_short_xy (dxy={dxy:.2f}m < {MIN_TRANSITION_DXY_M:.2f}m)"
            )

        slope = abs(dz) / max(1e-3, dxy)
        involves_traverse = (
            a_curr["primitive"] == "traverse" or a_next["primitive"] == "traverse"
        )
        max_slope = MAX_SLOPE_WITH_TRAVERSE if involves_traverse else MAX_SLOPE_OPEN_POLE
        if slope > max_slope:
            reasons.append(
                f"transition_{i}_{i+1}_excessive_slope (slope={slope:.2f} > {max_slope:.2f})"
            )

    # 3. Verificar que ningun waypoint pre/post invada el volumen de otro landmark distinto
    for anc in anchors:
        tid = anc["target"]
        for pt_name in ("pre", "post"):
            pt = anc[pt_name]
            for h in hoops:
                if str(h["id"]) == tid:
                    continue
                hx, hy, hz = float(h["x"]), float(h["y"]), float(h["z"])
                d_xy_other = math.hypot(pt[0] - hx, pt[1] - hy)
                if str(h.get("type", "gate")) == "pole":
                    if d_xy_other < 0.75 and pt[2] < 3.2:
                        reasons.append(
                            f"waypoint_{tid}_{pt_name}_collides_with_pole_{h['id']} (dxy={d_xy_other:.2f}m)"
                        )
                else:
                    if d_xy_other < 0.95 and abs(pt[2] - hz) < 1.05:
                        reasons.append(
                            f"waypoint_{tid}_{pt_name}_collides_with_gate_{h['id']} (dxy={d_xy_other:.2f}m)"
                        )

    # 4. Verificar margen longitudinal desde el ultimo post hasta la plataforma de aterrizaje
    if landing is not None and "x" in landing and "y" in landing:
        last_post = anchors[-1]["post"]
        lx = float(landing["x"])
        ly = float(landing["y"])
        dx_land = lx - last_post[0]
        dxy_land = math.hypot(lx - last_post[0], ly - last_post[1])
        if dx_land < MIN_LANDING_DX_M or dxy_land < 1.5:
            reasons.append(
                f"last_post_to_landing_too_close (dx={dx_land:.2f}m, dxy={dxy_land:.2f}m)"
            )

    return len(reasons) == 0, reasons


def generate_random_circuit(
    landmarks_info: list,
    seed: Optional[int] = None,
    max_attempts: int = 100,
    primitives: Optional[list] = None,
) -> tuple[list[dict], dict]:
    """Genera poses aleatorias para los landmarks y la plataforma de aterrizaje.

    Returns:
        (hoops, landing) donde:
          hoops = [{"id": str, "type": str, "x": float, "y": float, "z": float, "yaw": float}, ...]
          landing = {"x": float, "y": float, "z": float}

    Raises:
        RuntimeError: si no se logra un circuito factible en max_attempts intentos.
    """
    rng = random.Random(seed)

    for _ in range(max_attempts):
        hoops = []
        prev_x = 0.0   # el dron despega en X=0
        prev_y = 0.0   # el dron despega en Y=0
        feasible = True

        for i in range(len(landmarks_info)):
            # Posicion X: avance progresivo
            if i == 0:
                x = rng.uniform(FIRST_GATE_X_MIN, FIRST_GATE_X_MAX)
            else:
                sep = rng.uniform(MIN_GATE_SEP_X, MAX_GATE_SEP_X)
                x = prev_x + sep

            # Posicion Y: lateral, pero limitando el delta con el landmark anterior
            y = rng.uniform(MIN_GATE_Y, MAX_GATE_Y)
            if abs(y - prev_y) > MAX_DELTA_Y_BETWEEN_GATES:
                y = prev_y + rng.uniform(
                    -MAX_DELTA_Y_BETWEEN_GATES,
                    MAX_DELTA_Y_BETWEEN_GATES,
                )
                y = max(MIN_GATE_Y, min(MAX_GATE_Y, y))

            lm_type = str(landmarks_info[i].get("type", "gate"))

            # Posicion Z: altura
            if lm_type == "pole":
                z = 1.5   # Centro geometrico de un poste de 3.0m apoyado en el suelo
            else:
                z = rng.uniform(MIN_GATE_Z, MAX_GATE_Z)

            # Yaw: rotacion de la abertura
            yaw = rng.uniform(-MAX_GATE_YAW, MAX_GATE_YAW)

            # Validar que el borde inferior no toque el suelo (para gates cuadrados y circulares)
            half_h = (
                CIRCULAR_MAJOR_RADIUS + CIRCULAR_TUBE_RADIUS
                if "circ" in lm_type.lower() or lm_type == "torus"
                else GATE_HALF_HEIGHT
            )
            if lm_type != "pole" and z - half_h < 0.1:
                feasible = False
                break

            hoops.append({
                "id": landmarks_info[i]["id"],
                "type": lm_type,
                "x": round(x, 2),
                "y": round(y, 2),
                "z": round(z, 2),
                "yaw": round(yaw, 3),
            })

            prev_x = x
            prev_y = y

        if not feasible:
            continue

        # Plataforma de aterrizaje: adelante del ultimo landmark
        last_hoop = hoops[-1]
        land_x = last_hoop["x"] + rng.uniform(
            LANDING_OFFSET_X_MIN, LANDING_OFFSET_X_MAX
        )
        land_y = rng.uniform(-LANDING_OFFSET_Y_MAX, LANDING_OFFSET_Y_MAX)

        landing = {
            "x": round(land_x, 2),
            "y": round(land_y, 2),
            "z": 0.05,
        }

        trans_ok, _ = check_circuit_transition_feasibility(
            hoops, primitives=primitives, landing=landing
        )
        if not trans_ok:
            continue

        return hoops, landing

    raise RuntimeError(
        f"No se logro un circuito factible en {max_attempts} intentos"
    )


# ============================================================================
# Manipulacion del SDF
# ============================================================================

def modify_sdf(
    template_path: Path,
    world_name: str,
    hoops: list[dict],
    landing: dict,
    remove_takeoff_platform: bool = False,
) -> ET.ElementTree:
    """Parsea el SDF plantilla, limpia modelos previos e inyecta dinámicamente los landmarks."""
    tree = ET.parse(template_path)
    root = tree.getroot()

    world_elem = root.find("world")
    if world_elem is None:
        raise ValueError("No se encontro <world> en el SDF plantilla")
    world_elem.set("name", world_name)

    if remove_takeoff_platform:
        takeoff_model = _find_model(world_elem, "plataforma_salida")
        if takeoff_model is not None:
            world_elem.remove(takeoff_model)

    # 1. Eliminar cualquier modelo de landmark preexistente en la plantilla (aro_*, landmark_*)
    for model in list(world_elem.findall("model")):
        m_name = model.get("name", "")
        if m_name.startswith(("aro_", "landmark_")):
            world_elem.remove(model)

    # 2. Inyectar dinámicamente cada uno de los N landmarks según su tipo geométrico
    for hoop in hoops:
        h_type = str(hoop.get("type", "gate")).lower()
        if h_type == "pole":
            model_elem = _create_pole_element(
                hoop["id"], hoop["x"], hoop["y"], hoop["yaw"]
            )
        elif "circ" in h_type or h_type == "torus":
            model_elem = _create_circular_gate_element(
                hoop["id"], hoop["x"], hoop["y"], hoop["z"], hoop["yaw"]
            )
        else:
            model_elem = _create_gate_element(
                hoop["id"], hoop["x"], hoop["y"], hoop["z"], hoop["yaw"]
            )
        world_elem.append(model_elem)

    # 3. Modificar pose de la plataforma de aterrizaje (siempre adelante del último landmark)
    landing_model = _find_model(world_elem, LANDING_MODEL_NAME)
    if landing_model is None:
        raise ValueError(
            f"Modelo '{LANDING_MODEL_NAME}' no encontrado en la plantilla SDF"
        )
    landing_z = landing.get("z", 0.05)
    landing_pose_str = f"{landing['x']} {landing['y']} {landing_z} 0 0 0"
    _set_model_pose(landing_model, landing_pose_str)

    try:
        ET.indent(tree, space="  ")
    except AttributeError:
        pass

    return tree


def _find_model(world_elem: ET.Element, model_name: str) -> Optional[ET.Element]:
    """Busca un <model name='...'> dentro del <world>."""
    for model in world_elem.iter("model"):
        if model.get("name") == model_name:
            return model
    return None


def _set_model_pose(model: ET.Element, pose_str: str) -> None:
    """Modifica el <pose> directo del modelo (no el de los links internos)."""
    for child in model:
        if child.tag == "pose":
            child.text = pose_str
            return
    pose_elem = ET.SubElement(model, "pose")
    pose_elem.text = pose_str


# ============================================================================
# Exportacion
# ============================================================================

def export_waypoints_yaml(
    landmarks: list[dict],
    primitives: list[dict],
    landing: dict,
    output_path: Path,
    start_pose: Optional[dict] = None,
) -> None:
    """Exporta el archivo waypoints.yaml compatible con waypoint_controller.py."""
    landmarks_map = {lm["id"]: lm for lm in landmarks}
    sorted_primitives = sorted(
        primitives,
        key=lambda p: float(landmarks_map.get(p.get("target"), {}).get("x", 0.0))
    )
    data = {}
    if start_pose is not None:
        data["start_pose"] = start_pose
    data["landmarks"] = landmarks
    data["primitives"] = sorted_primitives
    data["landing"] = landing

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        f.write("# Auto-generated by world_generator.py\n")
        f.write("# Coordenadas en marco Gazebo (ENU): X Y Z Yaw(rad)\n\n")
        yaml.dump(data, f, default_flow_style=False, sort_keys=False)


def export_sdf(tree: ET.ElementTree, output_path: Path) -> None:
    """Escribe el SDF modificado con declaracion XML."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    tree.write(
        output_path,
        encoding="UTF-8",
        xml_declaration=True,
    )


def copy_to_px4(sdf_path: Path, px4_worlds_dir: Path = PX4_WORLDS_DIR) -> Path:
    """Copia el SDF generado al directorio de mundos de PX4."""
    if not px4_worlds_dir.is_dir():
        raise FileNotFoundError(
            f"Directorio de mundos PX4 no encontrado: {px4_worlds_dir}\n"
            f"Verifica que PX4-Autopilot este instalado en ~/PX4-Autopilot/"
        )
    dest = px4_worlds_dir / sdf_path.name
    shutil.copy2(sdf_path, dest)
    return dest


# ============================================================================
# CLI
# ============================================================================

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Genera un mundo SDF aleatorio + waypoints.yaml "
        "para el pipeline de Instant Policy.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=(
            "Ejemplo:\n"
            "  python3 world_generator.py --task-id task_0001\n"
            "  python3 world_generator.py --task-id task_0001 --seed 42\n"
            "  python3 world_generator.py --task-id task_0001 --no-copy\n"
        ),
    )
    parser.add_argument(
        "--task-id",
        required=True,
        help="Identificador unico de la tarea (ej: task_0001).",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=None,
        help="Semilla para el generador aleatorio (reproducibilidad).",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Directorio de salida. Default: generated_worlds/ junto al script.",
    )
    parser.add_argument(
        "--template",
        type=Path,
        default=None,
        help="Ruta al SDF plantilla. Default: worlds/circuito_aros_01.sdf",
    )
    parser.add_argument(
        "--no-copy",
        action="store_true",
        help="No copiar el SDF a PX4-Autopilot/Tools/simulation/gz/worlds/.",
    )
    parser.add_argument(
        "--no-takeoff-platform",
        action="store_true",
        help="Elimina la plataforma de salida (despegue) del mundo generado.",
    )

    args = parser.parse_args()

    script_dir = Path(__file__).resolve().parent
    pkg_dir = script_dir.parent  # src/uav_ip/

    if args.template is None:
        args.template = pkg_dir / "worlds" / "circuito_aros_01.sdf"

    if args.output_dir is None:
        args.output_dir = pkg_dir / "generated_worlds"

    if not args.template.is_file():
        print(f"ERROR: Plantilla SDF no encontrada: {args.template}")
        sys.exit(1)

    task_dir = args.output_dir / args.task_id
    config_path = task_dir / "task_config.yaml"

    cfg = None
    if config_path.exists():
        with open(config_path, "r") as f:
            cfg = yaml.safe_load(f)

    if cfg is None or "landmarks" not in cfg:
        print("ERROR: task_config.yaml con landmarks no encontrado. Ejecuta task_generator.py primero.")
        sys.exit(1)

    landmarks_info = cfg["landmarks"]
    primitives = cfg["primitives"]

    print(f"Generando circuito para '{args.task_id}' (seed={args.seed})...")
    hoops, landing = generate_random_circuit(
        landmarks_info, seed=args.seed, primitives=primitives
    )
    start_pose = generate_start_pose(seed=args.seed)

    print(
        f"  Pose inicial aerea: X={start_pose['x']:5.2f}  Y={start_pose['y']:5.2f}  "
        f"Z={start_pose['z']:5.2f}  Yaw={start_pose['yaw']:+.3f} rad"
    )
    print("  Landmarks generados:")
    for h in hoops:
        print(
            f"    {h['id']} ({h['type']}): X={h['x']:6.2f}  Y={h['y']:6.2f}  "
            f"Z={h['z']:6.2f}  Yaw={h['yaw']:+.3f} rad "
            f"({math.degrees(h['yaw']):+.1f}°)"
        )
    print(
        f"  Plataforma: X={landing['x']:6.2f}  Y={landing['y']:6.2f}  Z={landing['z']:4.2f}"
    )

    tree = modify_sdf(args.template, args.task_id, hoops, landing, args.no_takeoff_platform)

    sdf_path = task_dir / f"{args.task_id}.sdf"
    yaml_path = task_dir / "waypoints.yaml"

    export_sdf(tree, sdf_path)
    export_waypoints_yaml(hoops, primitives, landing, yaml_path, start_pose=start_pose)

    print(f"\n  SDF  -> {sdf_path}")
    print(f"  YAML -> {yaml_path}")

    if not args.no_copy:
        try:
            px4_dest = copy_to_px4(sdf_path)
            print(f"  PX4  -> {px4_dest}")
            spawn_z = "0.15" if args.no_takeoff_platform else "0.35"
            print(
                f"\nLanzar con: PX4_GZ_WORLD={args.task_id} "
                f"PX4_GZ_MODEL_POSE=\"0,0,{spawn_z},0,0,0\" "
                f"make px4_sitl gz_x500"
            )
        except FileNotFoundError as e:
            print(f"\n  WARN: {e}")
            print("  El SDF fue generado pero no copiado a PX4.")
    else:
        print("\n  (--no-copy: SDF no copiado a PX4)")

    print("\nDone.")


if __name__ == "__main__":
    main()
