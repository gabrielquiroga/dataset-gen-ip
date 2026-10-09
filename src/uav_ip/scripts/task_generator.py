#!/usr/bin/env python3
"""
task_generator.py - Generates pseudo-task definitions according to the Instant Policy taxonomy.
"""
import argparse
import random
import yaml
from pathlib import Path

def main():
    parser = argparse.ArgumentParser(description="Generates task_config.yaml")
    parser.add_argument("--task-id", required=True)
    parser.add_argument("--seed", type=int, default=42)
    script_dir = Path(__file__).resolve().parent
    pkg_dir = script_dir.parent
    parser.add_argument("--output-dir", type=Path, default=pkg_dir / "generated_worlds")
    args = parser.parse_args()

    random.seed(args.seed)

    num_gates = 3
    logics = ["A1_DIRECT", "A2_REVERSE", "A3_ALTERNATING"]
    logic = random.choice(logics)
    
    profiles = ["B1_LINEAR", "B2_BSPLINE"]
    profile = random.choice(profiles)
    
    landmarks = []
    for i in range(num_gates):
        landmark_id = f"landmark_{i+1}"
        landmark_type = random.choice(["gate", "pole"])
        landmarks.append({"id": landmark_id, "type": landmark_type})
        
    ordered_landmarks = list(landmarks)
    if logic == "A2_REVERSE":
        ordered_landmarks = ordered_landmarks[::-1]
    elif logic == "A3_ALTERNATING" and num_gates >= 3:
        ordered_landmarks = [ordered_landmarks[0], ordered_landmarks[2], ordered_landmarks[1]]
        
    primitives = []
    for lm in ordered_landmarks:
        if lm["type"] == "gate":
            prim_type = "traverse"
        else:
            prim_type = random.choice(["overfly", "slalom", "orbit"])
        primitives.append({"type": prim_type, "target": lm["id"]})
    
    task_config = {
        "task_id": args.task_id,
        "seed": args.seed,
        "num_gates": num_gates,
        "logic": logic,
        "profile": profile,
        "landmarks": landmarks,
        "primitives": primitives,
    }

    task_dir = args.output_dir / args.task_id
    task_dir.mkdir(parents=True, exist_ok=True)
    
    config_path = task_dir / "task_config.yaml"
    with open(config_path, "w") as f:
        yaml.dump(task_config, f)
        
    print(f"Task configuration written to {config_path}")

if __name__ == "__main__":
    main()
