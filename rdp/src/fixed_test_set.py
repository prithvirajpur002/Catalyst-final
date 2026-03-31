"""
Catalyst RVC — Fixed Test Set Manager
ALL experiments MUST evaluate on the SAME inputs.

Test inputs are IMMUTABLE once created.
This ensures fair comparison across all experiments.
"""

import os
from pathlib import Path
from typing import Optional
import soundfile as sf
import numpy as np
from .utils import Logger, save_json, load_json


class FixedTestSet:
    """Manage the canonical test set used by ALL experiments."""

    def __init__(self, test_dir: str):
        self.test_dir = test_dir
        self.manifest_path = os.path.join(test_dir, "manifest.json")
        self.log = Logger()
        os.makedirs(test_dir, exist_ok=True)

    def create_test_input(self, name: str, duration_s: float, sample_rate: int = 40000) -> str:
        """
        Create a standardized test audio file.
        SHOULD NOT BE CALLED BY USER — test set is predefined.
        """
        path = os.path.join(self.test_dir, f"{name}.wav")
        if os.path.exists(path):
            self.log.warn(f"Test input {name} already exists. Not overwriting.")
            return path

        # Generate simple sine wave at 440 Hz (neutral pitch)
        t = np.linspace(0, duration_s, int(duration_s * sample_rate), dtype=np.float32)
        y = 0.3 * np.sin(2 * np.pi * 440 * t).astype(np.float32)
        sf.write(path, y, sample_rate, subtype="PCM_16")

        self.log.ok(f"Created test input: {name} ({duration_s}s)")
        return path

    def get_manifest(self) -> dict:
        """Get test set manifest (file list + metadata)."""
        if os.path.exists(self.manifest_path):
            return load_json(self.manifest_path)
        return {"test_inputs": {}}

    def update_manifest(self, manifest: dict) -> None:
        """Save manifest. IMMUTABLE once written."""
        save_json(self.manifest_path, manifest)

    def ensure_test_set_exists(self) -> dict:
        """
        Create canonical test set if it doesn't exist.
        This is the ONLY valid test set for all experiments.
        """
        manifest = self.get_manifest()

        if manifest.get("test_inputs"):
            self.log.ok("Test set already exists. Using canonical set.")
            return manifest

        # Create the three standard test inputs
        test_files = {
            "neutral": 5.0,    # 5 seconds at 440 Hz
            "sustained": 8.0,  # 8 seconds (tests longer duration)
            "short": 3.0,      # 3 seconds (tests minimum viable)
        }

        for name, duration in test_files.items():
            self.create_test_input(name, duration)

        manifest = {
            "test_inputs": {
                "neutral": f"neutral.wav",
                "sustained": f"sustained.wav",
                "short": f"short.wav",
            },
            "sample_rate": 40000,
            "note": "CANONICAL TEST SET — DO NOT MODIFY. Used by all experiments.",
        }

        self.update_manifest(manifest)
        self.log.ok("Canonical test set created and locked.")
        return manifest

    def get_test_input_path(self, test_name: str) -> Optional[str]:
        """Get path to a test input by name."""
        manifest = self.get_manifest()
        test_inputs = manifest.get("test_inputs", {})

        if test_name not in test_inputs:
            self.log.warn(f"Test input '{test_name}' not in manifest")
            return None

        path = os.path.join(self.test_dir, test_inputs[test_name])
        if not os.path.exists(path):
            self.log.warn(f"Test input file missing: {path}")
            return None

        return path

    def list_test_inputs(self) -> list[str]:
        """List all available test inputs."""
        manifest = self.get_manifest()
        return list(manifest.get("test_inputs", {}).keys())
