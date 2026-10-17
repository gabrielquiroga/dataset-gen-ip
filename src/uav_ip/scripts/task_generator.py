#!/usr/bin/env python3
"""
task_generator.py - Generates pseudo-task definitions according to the Instant Policy taxonomy.

Supports:
  1. Extensible catalog of atomic actions [(landmark_type, primitive_type, params_dict)].
  2. Bilateral slalom primitives (side='left' and side='right') and optional circular_gate.
  3. Canonical semantic signature generation (get_task_signature) free of phantom profile splits.
  4. Dynamic combinatorial enumeration (itertools.product) for any number of landmarks.
  5. Deterministic sampling without replacement via --task-index and shuffled permutation.
"""
import argparse
import itertools
import random
from pathlib import Path
from typing import Dict, Any, List, Tuple, Union
import yaml

ActionItem = Union[Tuple[str, str], Tuple[str, str, Dict[str, Any]]]

# Catálogo extensible de acciones atómicas factibles en pasillo 1D (+X).
# Cada elemento es una tupla: (tipo_landmark, tipo_primitiva, parametros_extra).
DEFAULT_ACTION_CATALOG: List[Tuple[str, str, Dict[str, Any]]] = [
    ("gate", "traverse", {}),
    ("pole", "overfly", {}),
    ("pole", "slalom", {"side": "left"}),
    ("pole", "slalom", {"side": "right"}),
    ("pole", "orbit", {}),
]

CIRCULAR_GATE_ACTION: Tuple[str, str, Dict[str, Any]] = ("circular_gate", "traverse", {})


def _unpack_action_item(item: ActionItem) -> Tuple[str, str, Dict[str, Any]]:
    """Normaliza una tupla de acción de 2 o 3 elementos a (lm_type, prim_type, params)."""
    if len(item) == 2:
        return item[0], item[1], {}
    elif len(item) == 3:
        return item[0], item[1], dict(item[2] or {})
    raise ValueError(f"Formato de acción inválido: {item}")


def get_task_signature(config: Dict[str, Any]) -> str:
    """Calcula una firma semántica canónica unívoca para cualquier configuración de tarea.

    Agnóstica a la cantidad de landmarks, tipos de objetos y parámetros.
    Ejemplo:
      "landmark_1:gate/traverse;landmark_2:pole/orbit;landmark_3:pole/slalom(side=left)|logic:A1_DIRECT"
    """
    lm_map = {lm["id"]: lm["type"] for lm in config.get("landmarks", [])}
    tokens = []
    for p in config.get("primitives", []):
        lm_id = p.get("target")
        lm_type = lm_map.get(lm_id, "unknown")
        prim_type = p.get("type", "unknown")
        # Parámetros adicionales ordenados alfabéticamente si existen (ej. side=left)
        params = [f"{k}={v}" for k, v in sorted(p.items()) if k not in ("target", "type")]
        param_str = f"({','.join(params)})" if params else ""
        tokens.append(f"{lm_id}:{lm_type}/{prim_type}{param_str}")

    logic = config.get("logic", "A1_DIRECT")
    return ";".join(tokens) + f"|logic:{logic}"


def get_combinatorial_catalog(
    action_catalog: List[ActionItem],
    num_landmarks: int
) -> List[Tuple[ActionItem, ...]]:
    """Genera exhaustivamente todas las secuencias combinatorias para num_landmarks.

    Para K acciones y N landmarks, genera K^N secuencias.
    Ejemplo: 5 acciones ^ 3 landmarks = 125 combinaciones únicas.
    """
    return list(itertools.product(action_catalog, repeat=num_landmarks))


def get_balanced_multilength_catalog(
    action_catalog: List[ActionItem],
    min_landmarks: int,
    max_landmarks: int,
    shuffle_seed: int = 42,
) -> List[Tuple[ActionItem, ...]]:
    """Genera un catálogo combinado balanceado e intercalado para un rango de longitudes [min..max].

    Baraja deterministamente cada longitud por separado e intercala sus secuencias
    para que todas las longitudes tengan una representación uniforme y se garantice
    cero duplicados por construcción.
    """
    buckets: Dict[int, List[Tuple[ActionItem, ...]]] = {}
    rng = random.Random(shuffle_seed)
    for n in range(min_landmarks, max_landmarks + 1):
        seqs = list(itertools.product(action_catalog, repeat=n))
        rng.shuffle(seqs)
        buckets[n] = seqs

    interleaved = []
    max_bucket_len = max(len(s) for s in buckets.values())
    for i in range(max_bucket_len):
        for n in range(min_landmarks, max_landmarks + 1):
            if i < len(buckets[n]):
                interleaved.append(buckets[n][i])

    return interleaved


def generate_task_dict(
    task_id: str,
    seed: int,
    action_sequence: Tuple[ActionItem, ...],
    logic: str = "A1_DIRECT",
    profile: str = "B1_CONTINUOUS",
) -> Dict[str, Any]:
    """Construye el diccionario canónico de configuración de la tarea."""
    landmarks = []
    primitives = []
    for i, action_item in enumerate(action_sequence):
        lm_type, prim_type, extra_params = _unpack_action_item(action_item)
        lm_id = f"landmark_{i+1}"
        landmarks.append({"id": lm_id, "type": lm_type})
        prim_dict: Dict[str, Any] = {"type": prim_type, "target": lm_id}
        for k, v in sorted(extra_params.items()):
            prim_dict[k] = v
        primitives.append(prim_dict)

    config = {
        "task_id": task_id,
        "seed": seed,
        "num_gates": len(landmarks),
        "logic": logic,
        "profile": profile,
        "landmarks": landmarks,
        "primitives": primitives,
    }
    config["signature"] = get_task_signature(config)
    return config


def main():
    parser = argparse.ArgumentParser(
        description="Genera la configuración de una pseudo-tarea (task_config.yaml) sin reemplazo."
    )
    parser.add_argument("--task-id", required=True, help="Identificador de la tarea (ej. task_0001)")
    parser.add_argument("--seed", type=int, default=42, help="Semilla pseudoaleatoria de generación")
    parser.add_argument(
        "--task-index", type=int, default=None,
        help="Índice combinatorio [0..M-1] para muestreo estricto sin reemplazo"
    )
    parser.add_argument(
        "--num-landmarks", "--num-gates", dest="num_landmarks", type=int, default=None,
        help="Cantidad exacta fija de landmarks (anula min/max si se especifica)"
    )
    parser.add_argument(
        "--min-landmarks", type=int, default=2,
        help="Cantidad mínima de landmarks acotados (default: 2)"
    )
    parser.add_argument(
        "--max-landmarks", type=int, default=5,
        help="Cantidad máxima de landmarks acotados (default: 5)"
    )
    parser.add_argument(
        "--shuffle-seed", type=int, default=42,
        help="Semilla para barajar la permutación del catálogo deterministamente"
    )
    parser.add_argument(
        "--include-circular-gates", action="store_true",
        help="Incluye 'circular_gate' en el catálogo combinatorio además de puertas cuadradas"
    )
    parser.add_argument(
        "--profile", type=str, default="B1_CONTINUOUS",
        help="Perfil cinemático de control (default: B1_CONTINUOUS)"
    )
    script_dir = Path(__file__).resolve().parent
    pkg_dir = script_dir.parent
    parser.add_argument("--output-dir", type=Path, default=pkg_dir / "generated_worlds")
    args = parser.parse_args()

    if args.num_landmarks is not None:
        min_l = max_l = args.num_landmarks
    else:
        min_l = args.min_landmarks
        max_l = args.max_landmarks

    if min_l > max_l:
        raise ValueError(f"min_landmarks ({min_l}) no puede ser mayor que max_landmarks ({max_l})")

    active_catalog = list(DEFAULT_ACTION_CATALOG)
    if args.include_circular_gates:
        active_catalog.append(CIRCULAR_GATE_ACTION)

    # Generar espacio combinatorio balanceado e intercalado
    catalog = get_balanced_multilength_catalog(active_catalog, min_l, max_l, args.shuffle_seed)
    catalog_size = len(catalog)

    if args.task_index is not None:
        # Modo sin reemplazo indexado sobre la permutación
        chosen_idx = args.task_index % catalog_size
        action_sequence = catalog[chosen_idx]
    else:
        # Fallback pseudoaleatorio si no se pasa índice
        rng_choice = random.Random(args.seed)
        action_sequence = rng_choice.choice(catalog)

    # En pasillo 1D (+X) la lógica válida y factible de avance monótono es A1_DIRECT
    logic = "A1_DIRECT"
    profile = args.profile

    task_config = generate_task_dict(
        task_id=args.task_id,
        seed=args.seed,
        action_sequence=action_sequence,
        logic=logic,
        profile=profile,
    )
    task_config["catalog_size"] = catalog_size
    task_config["min_landmarks"] = min_l
    task_config["max_landmarks"] = max_l
    if args.task_index is not None:
        task_config["task_index"] = args.task_index

    task_dir = args.output_dir / args.task_id
    task_dir.mkdir(parents=True, exist_ok=True)

    config_path = task_dir / "task_config.yaml"
    with open(config_path, "w") as f:
        yaml.dump(task_config, f, sort_keys=False)

    print(f"Task configuration written to {config_path}")
    print(f"  Firma Semántica: {task_config['signature']}")
    print(f"  Catálogo: {catalog_size} combinaciones disponibles (rango=[{min_l}..{max_l}], actual={len(action_sequence)})")


if __name__ == "__main__":
    main()
