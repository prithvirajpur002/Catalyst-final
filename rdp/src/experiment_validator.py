"""
Catalyst RVC — Experiment Validator
RULE: Each new experiment changes EXACTLY ONE variable from previous.

This prevents confusion. If results improve, you know WHY.
"""

from typing import Optional
from .utils import load_json


class ExperimentValidator:
    """Validate that only one variable changes between consecutive experiments."""

    def __init__(self, experiments_dir: str):
        self.exp_dir = experiments_dir

    def get_spec(self, exp_id: str) -> Optional[dict]:
        """Load experiment spec (config.json)."""
        import os
        config_path = os.path.join(self.exp_dir, exp_id, "config.json")
        try:
            return load_json(config_path)
        except:
            return None

    def count_changes(self, spec_1: dict, spec_2: dict) -> tuple[int, list[str]]:
        """Count how many fields differ between two specs."""
        changes = []
        for key in ["dataset", "config", "epochs", "batch_size"]:
            if spec_1.get(key) != spec_2.get(key):
                changes.append(key)
        return len(changes), changes

    def validate_single_change(self, prev_exp_id: str, new_exp_id: str) -> tuple[bool, str]:
        """
        Validate that new_exp differs from prev_exp in exactly ONE variable.
        Returns (is_valid, reason).
        """
        prev_spec = self.get_spec(prev_exp_id)
        new_spec = self.get_spec(new_exp_id)

        if not prev_spec:
            return False, f"Could not load spec for {prev_exp_id}"
        if not new_spec:
            return False, f"Could not load spec for {new_exp_id}"

        num_changes, changes = self.count_changes(prev_spec, new_spec)

        if num_changes == 0:
            return False, "No variables changed. Experiments are identical."
        elif num_changes == 1:
            var = changes[0]
            old_val = prev_spec.get(var)
            new_val = new_spec.get(var)
            return True, f"✓ Single change: {var} ({old_val} → {new_val})"
        else:
            return False, f"Too many changes ({num_changes}): {', '.join(changes)}. Change ONE variable only."

    def get_recommendation(self, prev_exp_id: str, winner_exp_id: str) -> str:
        """
        Given that winner_exp_id beat prev_exp_id, suggest next change.
        RULE: vary only what wasn't tested yet.
        """
        winner_spec = self.get_spec(winner_exp_id)
        if not winner_spec:
            return "Cannot get recommendation: spec not found"

        dataset = winner_spec.get("dataset")
        config = winner_spec.get("config")
        epochs = winner_spec.get("epochs")
        batch_size = winner_spec.get("batch_size")

        options = [
            f"Try {dataset.upper()} with different CONFIG: swap {config} ↔ high_quality",
            f"Try {dataset.upper()} with more EPOCHS: {epochs} → {epochs + 100}",
            f"Try {dataset.upper()} with different BATCH_SIZE: {batch_size} → {batch_size + 2}",
            f"Try different DATASET: {dataset} → natural/clean/raw",
        ]

        return "\n  ".join(options)
