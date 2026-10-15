#!/usr/bin/env python3
"""
world_generator.py — Generador de Mundos Aleatorios para Instant Policy.

Toma el SDF plantilla (circuito_aros_01.sdf), aplica variaciones aleatorias
factibles a las poses de los aros y la plataforma de aterrizaje, y exporta:
  1. Un archivo .sdf validado para Gazebo Harmonic.
  2. Un archivo waypoints.yaml con las poses para el piloto experto.

Uso:
    python3 world_generator.py --task-id task_0001
    python3 world_generator.py --task-id task_0001 --seed 42
    python3 world_generator.py --task-id task_0001 --output-dir /tmp/worlds

El script también copia el .sdf generado a PX4 para que pueda ser
lanzado con PX4_GZ_WORLD=<task_id>.

Reglas de Factibilidad:
    - Los aros se generan en orden creciente de X (el dron vuela hacia +X).
    - Separacion minima entre aros consecutivos: MIN_GATE_SEP_X metros.
    - Altura minima del centro del aro: MIN_GATE_Z (para que la barra inferior
      no penetre el suelo: centro - 0.75m > 0).
    - La plataforma de aterrizaje siempre se coloca adelante del ultimo aro.
    - Maximo desplazamiento lateral (Y) limitado para que la camara de
      profundidad mantenga visibilidad del siguiente aro.
    - Yaw del aro limitado a +-MAX_GATE_YAW para no crear angulos imposibles.
"""

import argparse
import copy
import math
import os
import random
import shutil
import sys
import xml.etree.ElementTree as ET
from pathlib import Path
from typing import Optional

import yaml

# ============================================================================
# Geometria del aro (puerta cuadrada)
# ============================================================================
# La puerta se extiende ±0.75m en Z y ±0.7m en Y respecto a su centro.
# Barras de 0.1m de grosor.
GATE_HALF_HEIGHT: float = 0.75
GATE_HALF_WIDTH: float = 0.70

# ============================================================================
# Rangos de randomizacion
# ============================================================================
# Primer aro: posicion absoluta desde el origen (spawn del dron)
FIRST_GATE_X_MIN: float = 4.0
FIRST_GATE_X_MAX: float = 7.0

# Separacion entre aros consecutivos (en X)
MIN_GATE_SEP_X: float = 4.0
MAX_GATE_SEP_X: float = 7.0

# Rango lateral (Y) del centro de cada aro
MIN_GATE_Y: float = -3.0
MAX_GATE_Y: float = 3.0

# Rango de altura (Z) del centro de cada aro
# Minimo: GATE_HALF_HEIGHT + margen, para que la barra inferior no toque el suelo
MIN_GATE_Z: float = 1.0   # barra inferior queda a Z = 1.0 - 0.75 = 0.25m
MAX_GATE_Z: float = 3.0

# Yaw del aro (radianes): rotacion de la abertura respecto al eje +X
# 0 = abertura mira al +X (el dron vuela recto)
# +-0.4 rad ≈ +-23 grados: suficiente variacion sin crear angulos imposibles
MAX_GATE_YAW: float = 0.4

# Plataforma de aterrizaje: offset en X desde el ultimo aro
LANDING_OFFSET_X_MIN: float = 4.0
LANDING_OFFSET_X_MAX: float = 6.0
LANDING_OFFSET_Y_MAX: float = 1.5   # ligero descentrado lateral

# Maximo cambio lateral entre aros consecutivos (evita trayectorias
# que salen del FOV de la camara de profundidad)
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

def generate_random_circuit(
    landmarks_info: list,
    seed: Optional[int] = None,
    max_attempts: int = 100,
) -> tuple[list[dict], dict]:
    """Genera poses aleatorias para los aros y la plataforma de aterrizaje.

    Returns:
        (hoops, landing) donde:
          hoops = [{"id": str, "type": str, "x": float, "y": float, "z": float, "yaw": float}, ...]
          landing = {"x": float, "y": float}

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

            # Posicion Y: lateral, pero limitando el delta con el aro anterior
            y = rng.uniform(MIN_GATE_Y, MAX_GATE_Y)
            if abs(y - prev_y) > MAX_DELTA_Y_BETWEEN_GATES:
                # Forzar dentro del rango aceptable
                y = prev_y + rng.uniform(
                    -MAX_DELTA_Y_BETWEEN_GATES,
                    MAX_DELTA_Y_BETWEEN_GATES,
                )
                y = max(MIN_GATE_Y, min(MAX_GATE_Y, y))

            # Posicion Z: altura
            if landmarks_info[i].get("type") == "pole":
                z = 1.5   # Centro geometrico de un poste de 3.0m apoyado en el suelo
            else:
                z = rng.uniform(MIN_GATE_Z, MAX_GATE_Z)

            # Yaw: rotacion de la abertura
            yaw = rng.uniform(-MAX_GATE_YAW, MAX_GATE_YAW)

            # Validar que la barra inferior no toque el suelo (solo para gates)
            if landmarks_info[i].get("type") != "pole" and z - GATE_HALF_HEIGHT < 0.1:
                feasible = False
                break

            hoops.append({
                "id": landmarks_info[i]["id"],
                "type": landmarks_info[i]["type"],
                "x": round(x, 2),
                "y": round(y, 2),
                "z": round(z, 2),
                "yaw": round(yaw, 3),
            })

            prev_x = x
            prev_y = y

        if not feasible:
            continue

        # Plataforma de aterrizaje: adelante del ultimo aro
        last_hoop = hoops[-1]
        land_x = last_hoop["x"] + rng.uniform(
            LANDING_OFFSET_X_MIN, LANDING_OFFSET_X_MAX
        )
        land_y = rng.uniform(-LANDING_OFFSET_Y_MAX, LANDING_OFFSET_Y_MAX)

        landing = {
            "x": round(land_x, 2),
            "y": round(land_y, 2),
        }

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
    """Parsea el SDF plantilla, limpia modelos previos e inyecta dinámicamente los landmarks.

    Args:
        template_path: Ruta al archivo .sdf plantilla.
        world_name: Nombre para el <world> tag (usado por PX4_GZ_WORLD).
        hoops: Lista de dicts con {id, type, x, y, z, yaw}.
        landing: Dict con {x, y} de la plataforma de aterrizaje.
        remove_takeoff_platform: Si es True, elimina el modelo 'plataforma_salida' del SDF.

    Returns:
        ElementTree con el SDF modificado.
    """
    tree = ET.parse(template_path)
    root = tree.getroot()

    # Encontrar el <world> tag y cambiar su nombre
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

    # 2. Inyectar dinámicamente cada uno de los N landmarks
    for hoop in hoops:
        if hoop.get("type") == "pole":
            model_elem = _create_pole_element(
                hoop["id"], hoop["x"], hoop["y"], hoop["yaw"]
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
    landing_pose_str = f"{landing['x']} {landing['y']} 0.05 0 0 0"
    _set_model_pose(landing_model, landing_pose_str)

    # Formatear indentación XML limpia
    try:
        ET.indent(tree, space="  ")
    except AttributeError:
        pass

    return tree


def _find_model(world_elem: ET.Element, model_name: str) -> Optional[ET.Element]:
    """Busca un <model name='...'>  dentro del <world>."""
    for model in world_elem.iter("model"):
        if model.get("name") == model_name:
            return model
    return None


def _set_model_pose(model: ET.Element, pose_str: str) -> None:
    """Modifica el <pose> directo del modelo (no el de los links internos)."""
    # El <pose> que queremos es hijo directo de <model>, no de <link>
    for child in model:
        if child.tag == "pose":
            child.text = pose_str
            return
    # Si no existe un <pose> directo, crearlo
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
) -> None:
    """Exporta el archivo waypoints.yaml compatible con expert_pilot.py."""
    landmarks_map = {lm["id"]: lm for lm in landmarks}
    sorted_primitives = sorted(
        primitives,
        key=lambda p: float(landmarks_map.get(p.get("target"), {}).get("x", 0.0))
    )
    data = {
        "landmarks": landmarks,
        "primitives": sorted_primitives,
        "landing": landing,
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        # Header comment
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
    """Copia el SDF generado al directorio de mundos de PX4.

    Returns:
        Ruta destino dentro de PX4.

    Raises:
        FileNotFoundError: si el directorio de mundos de PX4 no existe.
    """
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
        help="Identificador unico de la tarea (ej: task_0001). "
        "Se usa como nombre del world y de los archivos de salida.",
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
        help="Directorio de salida. Default: generated_worlds/ junto "
        "al directorio del script.",
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

    # Resolver rutas
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

    # Generar circuito aleatorio
    print(f"Generando circuito para '{args.task_id}' (seed={args.seed})...")
    hoops, landing = generate_random_circuit(landmarks_info, seed=args.seed)

    print("  Landmarks generados:")
    for h in hoops:
        print(
            f"    {h['id']} ({h['type']}): X={h['x']:6.2f}  Y={h['y']:6.2f}  "
            f"Z={h['z']:6.2f}  Yaw={h['yaw']:+.3f} rad "
            f"({math.degrees(h['yaw']):+.1f}°)"
        )
    print(
        f"  Plataforma: X={landing['x']:6.2f}  Y={landing['y']:6.2f}"
    )

    # Modificar SDF
    tree = modify_sdf(args.template, args.task_id, hoops, landing, args.no_takeoff_platform)

    # Exportar archivos
    sdf_path = task_dir / f"{args.task_id}.sdf"
    yaml_path = task_dir / "waypoints.yaml"

    export_sdf(tree, sdf_path)
    export_waypoints_yaml(hoops, primitives, landing, yaml_path)

    print(f"\n  SDF  -> {sdf_path}")
    print(f"  YAML -> {yaml_path}")

    # Copiar a PX4
    if not args.no_copy:
        try:
            px4_dest = copy_to_px4(sdf_path)
            print(f"  PX4  -> {px4_dest}")
            print(
                f"\nLanzar con: PX4_GZ_WORLD={args.task_id} "
                f"PX4_GZ_MODEL_POSE=\"0,0,0.2,0,0,0\" "
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
