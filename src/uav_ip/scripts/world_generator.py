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

# Numero de aros en el circuito
NUM_GATES: int = 3

# Nombres de los modelos en el SDF (deben coincidir con la plantilla)
GATE_MODEL_NAMES = ["aro_1", "aro_2", "aro_3"]
LANDING_MODEL_NAME = "plataforma_llegada"

# Ruta default de PX4 worlds
PX4_WORLDS_DIR = Path.home() / "PX4-Autopilot" / "Tools" / "simulation" / "gz" / "worlds"


# ============================================================================
# Generacion de poses aleatorias con validacion
# ============================================================================

def generate_random_circuit(
    seed: Optional[int] = None,
    max_attempts: int = 100,
) -> tuple[list[dict], dict]:
    """Genera poses aleatorias para los aros y la plataforma de aterrizaje.

    Returns:
        (hoops, landing) donde:
          hoops = [{"name": str, "x": float, "y": float, "z": float, "yaw": float}, ...]
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

        for i in range(NUM_GATES):
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
            z = rng.uniform(MIN_GATE_Z, MAX_GATE_Z)

            # Yaw: rotacion de la abertura
            yaw = rng.uniform(-MAX_GATE_YAW, MAX_GATE_YAW)

            # Validar que la barra inferior no toque el suelo
            if z - GATE_HALF_HEIGHT < 0.1:
                feasible = False
                break

            hoops.append({
                "name": GATE_MODEL_NAMES[i],
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
) -> ET.ElementTree:
    """Parsea el SDF plantilla y modifica las poses de los aros y la plataforma.

    Args:
        template_path: Ruta al archivo .sdf plantilla.
        world_name: Nombre para el <world> tag (usado por PX4_GZ_WORLD).
        hoops: Lista de dicts con {name, x, y, z, yaw}.
        landing: Dict con {x, y} de la plataforma de aterrizaje.

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

    # Modificar poses de los aros
    for hoop in hoops:
        model = _find_model(world_elem, hoop["name"])
        if model is None:
            raise ValueError(
                f"Modelo '{hoop['name']}' no encontrado en la plantilla SDF"
            )
        pose_str = f"{hoop['x']} {hoop['y']} {hoop['z']} 0 0 {hoop['yaw']}"
        _set_model_pose(model, pose_str)

    # Modificar pose de la plataforma de aterrizaje
    landing_model = _find_model(world_elem, LANDING_MODEL_NAME)
    if landing_model is None:
        raise ValueError(
            f"Modelo '{LANDING_MODEL_NAME}' no encontrado en la plantilla SDF"
        )
    landing_pose_str = f"{landing['x']} {landing['y']} 0.05 0 0 0"
    _set_model_pose(landing_model, landing_pose_str)

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
    hoops: list[dict],
    landing: dict,
    output_path: Path,
) -> None:
    """Exporta el archivo waypoints.yaml compatible con expert_pilot.py."""
    data = {
        "hoops": hoops,
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

    # Generar circuito aleatorio
    print(f"Generando circuito para '{args.task_id}' (seed={args.seed})...")
    hoops, landing = generate_random_circuit(seed=args.seed)

    print("  Aros generados:")
    for h in hoops:
        print(
            f"    {h['name']}: X={h['x']:6.2f}  Y={h['y']:6.2f}  "
            f"Z={h['z']:6.2f}  Yaw={h['yaw']:+.3f} rad "
            f"({math.degrees(h['yaw']):+.1f}°)"
        )
    print(
        f"  Plataforma: X={landing['x']:6.2f}  Y={landing['y']:6.2f}"
    )

    # Modificar SDF
    tree = modify_sdf(args.template, args.task_id, hoops, landing)

    # Exportar archivos
    task_dir = args.output_dir / args.task_id
    sdf_path = task_dir / f"{args.task_id}.sdf"
    yaml_path = task_dir / "waypoints.yaml"

    export_sdf(tree, sdf_path)
    export_waypoints_yaml(hoops, landing, yaml_path)

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
                f"make px4_sitl gz_x500_depth"
            )
        except FileNotFoundError as e:
            print(f"\n  WARN: {e}")
            print("  El SDF fue generado pero no copiado a PX4.")
    else:
        print("\n  (--no-copy: SDF no copiado a PX4)")

    print("\nDone.")


if __name__ == "__main__":
    main()
