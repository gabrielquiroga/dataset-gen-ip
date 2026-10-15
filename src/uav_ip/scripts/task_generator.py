#!/usr/bin/env python3
"""
task_generator.py - Generates pseudo-task definitions according to the Instant Policy taxonomy.

Supports:
  1. Extensible catalog of atomic actions [(landmark_type, primitive_type)].
  2. Canonical semantic signature generation (get_task_signature).
  3. Dynamic combinatorial enumeration (itertools.product) for any number of landmarks.
  4. Deterministic sampling without replacement via --task-index and shuffled permutation.
"""
import argparse
import itertools
import random
from pathlib import Path
from typing import Dict, Any, List, Tuple
import yaml

# Catálogo extensible de acciones atómicas factibles en pasillo 1D (+X).
# Cada elemento es una tupla: (tipo_landmark, tipo_primitiva).
# Ampliable en el futuro con nuevos landmarks o primitivas sin alterar la arquitectura.
DEFAULT_ACTION_CATALOG: List[Tuple[str, str]] = [
    ("gate", "traverse"),
    ("pole", "overfly"),
    ("pole", "slalom"),
    ("pole", "orbit"),
]

def get_task_signature(config: Dict[str, Any]) -> str:
    """Calcula una firma semántica canónica unívoca para cualquier configuración de tarea.
    
    Agnóstica a la cantidad de landmarks, tipos de objetos y parámetros.
    Ejemplo:
      "landmark_1:gate/traverse;landmark_2:pole/orbit;landmark_3:pole/slalom|logic:A1_DIRECT|profile:B1_LINEAR"
    """
    lm_map = {lm["id"]: lm["type"] for lm in config.get("landmarks", [])}
    tokens = []
    for p in config.get("primitives", []):
        lm_id = p.get("target")
        lm_type = lm_map.get(lm_id, "unknown")
        prim_type = p.get("type", "unknown")
        # Parámetros adicionales ordenados alfabéticamente si existen
        params = [f"{k}={v}" for k, v in sorted(p.items()) if k not in ("target", "type")]
        param_str = f"({','.join(params)})" if params else ""
        tokens.append(f"{lm_id}:{lm_type}/{prim_type}{param_str}")
        
    logic = config.get("logic", "A1_DIRECT")
    profile = config.get("profile", "B1_LINEAR")
    return ";".join(tokens) + f"|logic:{logic}|profile:{profile}"


def get_combinatorial_catalog(
    action_catalog: List[Tuple[str, str]], 
    num_landmarks: int
) -> List[Tuple[Tuple[str, str], ...]]:
    """Genera exhaustivamente todas las secuencias combinatorias para num_landmarks.
    
    Para K acciones y N landmarks, genera K^N secuencias.
    Ejemplo: 4 acciones ^ 3 landmarks = 64 combinaciones únicas.
    """
    return list(itertools.product(action_catalog, repeat=num_landmarks))


def get_balanced_multilength_catalog(
    action_catalog: List[Tuple[str, str]],
    min_landmarks: int,
    max_landmarks: int,
    shuffle_seed: int = 42,
) -> List[Tuple[Tuple[str, str], ...]]:
    """Genera un catálogo combinado balanceado e intercalado para un rango de longitudes [min..max].
    
    Baraja deterministamente cada longitud por separado e intercala sus secuencias
    para que todas las longitudes tengan una representación uniforme y se garantice
    cero duplicados por construcción.
    """
    buckets: Dict[int, List[Tuple[Tuple[str, str], ...]]] = {}
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
    action_sequence: Tuple[Tuple[str, str], ...],
    logic: str = "A1_DIRECT",
    profile: str = "B1_LINEAR",
) -> Dict[str, Any]:
    """Construye el diccionario canónico de configuración de la tarea."""
    landmarks = []
    primitives = []
    for i, (lm_type, prim_type) in enumerate(action_sequence):
        lm_id = f"landmark_{i+1}"
        landmarks.append({"id": lm_id, "type": lm_type})
        primitives.append({"type": prim_type, "target": lm_id})
        
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

    # Generar espacio combinatorial balanceado e intercalado
    catalog = get_balanced_multilength_catalog(DEFAULT_ACTION_CATALOG, min_l, max_l, args.shuffle_seed)
    catalog_size = len(catalog)

    if args.task_index is not None:
        # Modo sin reemplazo indexado sobre la permutación
        chosen_idx = args.task_index % catalog_size
        action_sequence = catalog[chosen_idx]
    else:
        # Fallback pseudoaleatorio si no se pasa índice
        rng_choice = random.Random(args.seed)
        action_sequence = rng_choice.choice(catalog)

    # Lógica y perfil cinemático
    # En pasillo 1D (+X) la lógica válida y factible de avance monótono es A1_DIRECT
    logic = "A1_DIRECT"
    rng_profile = random.Random(args.seed)
    profile = rng_profile.choice(["B1_LINEAR", "B2_BSPLINE"])

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
