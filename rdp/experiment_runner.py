#!/usr/bin/env python3
"""
Catalyst RVC — Experiment Runner
STRICT experiment control. Every run is fully traceable.

Requirements:
  - exp_id: unique identifier (exp_001, exp_002, etc.)
  - dataset: MUST be one of: clean, natural, raw
  - config: MUST be one of: baseline, high_quality
  - epochs, batch_size: exact values

Storage structure (IMMUTABLE):
  experiments/
    exp_001/
      config.json           ← defines EXACTLY what ran
      dataset_manifest.txt  ← which audio files used
      model/
      samples/
      logs/
"""

import json
import os
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Optional

_here = str(Path(__file__).parent)
if _here not in sys.path:
    sys.path.insert(0, _here)

from src.utils import Logger, load_json, save_json


class ExperimentSpec:
    """Defines an experiment. Immutable once written."""

    VALID_DATASETS = {"clean", "natural", "raw"}
    VALID_CONFIGS = {"baseline", "high_quality"}

    def __init__(
        self,
        exp_id: str,
        dataset: str,
        config: str,
        epochs: int,
        batch_size: int,
    ):
        if not exp_id or not exp_id.startswith("exp_"):
            raise ValueError(f"exp_id must start with 'exp_', got: {exp_id}")
        if dataset not in self.VALID_DATASETS:
            raise ValueError(f"dataset must be one of {self.VALID_DATASETS}, got: {dataset}")
        if config not in self.VALID_CONFIGS:
            raise ValueError(f"config must be one of {self.VALID_CONFIGS}, got: {config}")
        if epochs < 1:
            raise ValueError(f"epochs must be >= 1, got: {epochs}")
        if batch_size < 1:
            raise ValueError(f"batch_size must be >= 1, got: {batch_size}")

        self.exp_id = exp_id
        self.dataset = dataset
        self.config = config
        self.epochs = epochs
        self.batch_size = batch_size

    def to_dict(self) -> dict:
        return {
            "exp_id": self.exp_id,
            "dataset": self.dataset,
            "config": self.config,
            "epochs": self.epochs,
            "batch_size": self.batch_size,
        }

    @staticmethod
    def from_dict(d: dict) -> "ExperimentSpec":
        return ExperimentSpec(
            exp_id=d["exp_id"],
            dataset=d["dataset"],
            config=d["config"],
            epochs=d["epochs"],
            batch_size=d["batch_size"],
        )


class ExperimentRunner:
    """Strict control over experiment execution."""

    def __init__(self, base_dir: str):
        self.base_dir = base_dir
        self.exp_dir = os.path.join(base_dir, "experiments")
        self.log = Logger()
        os.makedirs(self.exp_dir, exist_ok=True)

    def create_experiment(self, spec: ExperimentSpec) -> str:
        """Create experiment directory structure. Fail if exp_id already exists."""
        exp_path = os.path.join(self.exp_dir, spec.exp_id)

        if os.path.exists(exp_path):
            raise RuntimeError(
                f"Experiment {spec.exp_id} already exists at {exp_path}. "
                "Cannot redefine. Create exp_002, exp_003, etc instead."
            )

        os.makedirs(exp_path, exist_ok=True)
        for subdir in ["model", "samples", "logs"]:
            os.makedirs(os.path.join(exp_path, subdir), exist_ok=True)

        config_path = os.path.join(exp_path, "config.json")
        save_json(config_path, spec.to_dict())

        self.log.ok(f"Created experiment directory: {exp_path}")
        return exp_path

    def run_experiment(self, spec: ExperimentSpec) -> dict:
        """Run experiment. MUST have been created first."""
        exp_path = os.path.join(self.exp_dir, spec.exp_id)

        if not os.path.exists(exp_path):
            return {
                "success": False,
                "exp_id": spec.exp_id,
                "error": f"Experiment not created. Call create_experiment() first.",
            }

        try:
            result = subprocess.run(
                [sys.executable, os.path.join(_here, "main.py"), "--only", spec.exp_id],
                capture_output=True,
                text=True,
                timeout=3600,
            )

            log_path = os.path.join(exp_path, "logs", "run.log")
            with open(log_path, "w") as f:
                f.write(result.stdout)
                if result.stderr:
                    f.write("\n--- STDERR ---\n")
                    f.write(result.stderr)

            return {
                "success": result.returncode == 0,
                "exp_id": spec.exp_id,
                "exp_dir": exp_path,
                "log_file": log_path,
            }
        except subprocess.TimeoutExpired:
            return {
                "success": False,
                "exp_id": spec.exp_id,
                "error": "Timeout after 1 hour",
            }
        except Exception as e:
            return {
                "success": False,
                "exp_id": spec.exp_id,
                "error": str(e),
            }

    def get_experiment_info(self, exp_id: str) -> Optional[dict]:
        """Get full info on an experiment."""
        exp_path = os.path.join(self.exp_dir, exp_id)
        config_path = os.path.join(exp_path, "config.json")

        if not os.path.exists(config_path):
            return None

        spec_dict = load_json(config_path)
        model_path = os.path.join(exp_path, "model")
        model_exists = os.path.exists(model_path) and len(os.listdir(model_path)) > 0

        return {
            "exp_id": exp_id,
            "spec": spec_dict,
            "exp_dir": exp_path,
            "model_exists": model_exists,
            "created_at": os.path.getctime(exp_path),
        }

    def list_experiments(self) -> list[dict]:
        """List all experiments (order: oldest first)."""
        if not os.path.exists(self.exp_dir):
            return []

        exps = []
        for exp_id in sorted(os.listdir(self.exp_dir)):
            info = self.get_experiment_info(exp_id)
            if info:
                exps.append(info)

        return sorted(exps, key=lambda e: e["created_at"])


def main():
    """CLI for manual experiment control."""
    import argparse

    parser = argparse.ArgumentParser(description="Strict Experiment Runner")
    subparsers = parser.add_subparsers(dest="command")

    # Command: create EXP_ID DATASET CONFIG EPOCHS BATCH_SIZE
    create_parser = subparsers.add_parser("create")
    create_parser.add_argument("exp_id")
    create_parser.add_argument("dataset", choices=["clean", "natural", "raw"])
    create_parser.add_argument("config", choices=["baseline", "high_quality"])
    create_parser.add_argument("epochs", type=int)
    create_parser.add_argument("batch_size", type=int)

    # Command: run EXP_ID
    run_parser = subparsers.add_parser("run")
    run_parser.add_argument("exp_id")

    # Command: info EXP_ID
    info_parser = subparsers.add_parser("info")
    info_parser.add_argument("exp_id")

    # Command: list
    subparsers.add_parser("list")

    args = parser.parse_args()
    runner = ExperimentRunner("/kaggle/working/catalyst_rvc")

    if args.command == "create":
        try:
            spec = ExperimentSpec(
                exp_id=args.exp_id,
                dataset=args.dataset,
                config=args.config,
                epochs=args.epochs,
                batch_size=args.batch_size,
            )
            runner.create_experiment(spec)
            print(json.dumps({"success": True, "exp_id": args.exp_id}))
        except Exception as e:
            print(json.dumps({"success": False, "error": str(e)}))

    elif args.command == "run":
        info = runner.get_experiment_info(args.exp_id)
        if not info:
            print(json.dumps({"success": False, "error": f"Experiment {args.exp_id} not found"}))
            return

        spec = ExperimentSpec.from_dict(info["spec"])
        result = runner.run_experiment(spec)
        print(json.dumps(result))

    elif args.command == "info":
        info = runner.get_experiment_info(args.exp_id)
        if not info:
            print(json.dumps({"error": f"Experiment {args.exp_id} not found"}))
            return
        print(json.dumps(info, indent=2, default=str))

    elif args.command == "list":
        exps = runner.list_experiments()
        for exp in exps:
            print(f"{exp['exp_id']:12} {exp['spec']['dataset']:8} {exp['spec']['config']:12} epochs={exp['spec']['epochs']:3} batch={exp['spec']['batch_size']}")


if __name__ == "__main__":
    main()
