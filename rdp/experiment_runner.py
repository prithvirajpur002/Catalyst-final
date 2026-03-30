#!/usr/bin/env python3
"""
Catalyst RVC — Experiment Runner API
Simple wrapper around main.py for orchestrating controlled experiments.

This module provides:
  - Run a single experiment
  - List completed experiments
  - Compare experiment results
  - Get best model by composite score

No philosophy. Just what's needed.
"""

import json
import os
import subprocess
import sys
from pathlib import Path
from typing import Optional

_here = str(Path(__file__).parent)
if _here not in sys.path:
    sys.path.insert(0, _here)

from src.utils import Logger, load_json, save_json
from src.registry import get_best_model, print_registry


class ExperimentRunner:
    """Orchestrate experiments with a simple interface."""

    def __init__(self, base_dir: str):
        self.base_dir = base_dir
        self.log = Logger()

    def run_experiment(self, exp_id: str) -> dict:
        """Run a single experiment and return results."""
        try:
            result = subprocess.run(
                [sys.executable, os.path.join(_here, "main.py"), "--only", exp_id],
                capture_output=True,
                text=True,
                timeout=3600,  # 1 hour max
            )
            return {
                "success": result.returncode == 0,
                "exp_id": exp_id,
                "stdout": result.stdout[-500:] if result.stdout else "",
                "stderr": result.stderr[-500:] if result.stderr else "",
            }
        except subprocess.TimeoutExpired:
            return {
                "success": False,
                "exp_id": exp_id,
                "error": "Timeout after 1 hour",
            }
        except Exception as e:
            return {
                "success": False,
                "exp_id": exp_id,
                "error": str(e),
            }

    def list_experiments(self) -> list[dict]:
        """List all completed experiments with scores."""
        registry_path = os.path.join(self.base_dir, "models", "registry.json")
        if not os.path.exists(registry_path):
            return []

        try:
            registry = load_json(registry_path)
            return [
                {
                    "id": exp_id,
                    "model_path": exp.get("model_path"),
                    "scores": exp.get("scores", {}),
                    "dataset": exp.get("dataset_name"),
                }
                for exp_id, exp in registry.items()
            ]
        except Exception:
            return []

    def get_best_model_info(self) -> Optional[dict]:
        """Get info on the best performing model."""
        best = get_best_model(self.base_dir)
        if not best:
            return None
        return {
            "exp_id": best.get("experiment_id"),
            "model_path": best.get("model_path"),
            "index_path": best.get("index_path"),
            "scores": best.get("scores"),
        }

    def compare_two_experiments(self, exp_id_1: str, exp_id_2: str) -> dict:
        """Compare scores between two experiments."""
        experiments = self.list_experiments()
        exp1 = next((e for e in experiments if e["id"] == exp_id_1), None)
        exp2 = next((e for e in experiments if e["id"] == exp_id_2), None)

        if not exp1 or not exp2:
            return {"error": "One or both experiments not found"}

        def composite_score(scores: dict) -> float:
            return (
                scores.get("naturalness", 0) * 0.45 +
                scores.get("clarity", 0) * 0.35 +
                scores.get("identity", 0) * 0.20
            )

        s1 = composite_score(exp1.get("scores", {}))
        s2 = composite_score(exp2.get("scores", {}))

        return {
            "exp_id_1": exp_id_1,
            "score_1": round(s1, 3),
            "exp_id_2": exp_id_2,
            "score_2": round(s2, 3),
            "winner": exp_id_1 if s1 > s2 else exp_id_2,
            "margin": round(abs(s1 - s2), 3),
        }


def main():
    """CLI interface for experiment runner."""
    import argparse

    parser = argparse.ArgumentParser(description="Experiment Runner API")
    subparsers = parser.add_subparsers(dest="command")

    subparsers.add_argument("run", help="run EXP_ID")
    subparsers.add_argument("list", help="list all experiments")
    subparsers.add_argument("best", help="get best experiment")
    subparsers.add_argument("compare", help="compare EXP_1 EXP_2")

    args = parser.parse_args()
    runner = ExperimentRunner("/kaggle/working/catalyst_rvc")

    if args.command == "run":
        result = runner.run_experiment(sys.argv[2])
        print(json.dumps(result, indent=2))
    elif args.command == "list":
        exps = runner.list_experiments()
        print(json.dumps(exps, indent=2))
    elif args.command == "best":
        best = runner.get_best_model_info()
        print(json.dumps(best, indent=2))
    elif args.command == "compare":
        comp = runner.compare_two_experiments(sys.argv[2], sys.argv[3])
        print(json.dumps(comp, indent=2))


if __name__ == "__main__":
    main()
