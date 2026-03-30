# -*- coding: utf-8 -*-
"""
Catalyst RVC — src/utils.py
Shared helpers used across all modules.

Fixes applied:
  - Bug 16: Logger is now a context manager (__enter__/__exit__) — no more leaked file handles
  - Bug 10: compute_dataset_hash() now includes file content checksum sampling to
            detect same-name/same-size swaps (content changes)
  - New: DryRunError for --dry-run mode path validation
  - New: check_paths() helper for pre-flight validation
"""

from __future__ import annotations

import hashlib
import json
import os
import shutil
import time
from datetime import datetime
from pathlib import Path
from typing import Any


# ── Terminal colour helpers ───────────────────────────────────────────────────

class C:
    """ANSI colour codes. Gracefully degrades when stdout is not a TTY."""
    _tty = hasattr(os, "isatty") and os.isatty(1)
    GREEN  = "\033[92m" if _tty else ""
    YELLOW = "\033[93m" if _tty else ""
    RED    = "\033[91m" if _tty else ""
    CYAN   = "\033[96m" if _tty else ""
    DIM    = "\033[2m"  if _tty else ""
    BOLD   = "\033[1m"  if _tty else ""
    RESET  = "\033[0m"  if _tty else ""


class Logger:
    """
    Lightweight structured logger — writes to stdout and an optional log file.
    Supports context manager protocol so file handles are never leaked:

        with Logger(log_file="run.log") as log:
            log.info("starting...")

    Can also be used without 'with' — call log.close() manually in that case.
    """

    def __init__(self, log_file: str | None = None):
        self._f = open(log_file, "a", encoding="utf-8") if log_file else None

    # ── Context manager ───────────────────────────────────────────────────────

    def __enter__(self) -> "Logger":
        return self

    def __exit__(self, *_) -> None:
        self.close()

    # ── Internal ──────────────────────────────────────────────────────────────

    def _ts(self) -> str:
        return datetime.now().strftime("%H:%M:%S")

    def _write(self, line: str) -> None:
        print(line)
        if self._f:
            import re
            clean = re.sub(r"\033\[[0-9;]*m", "", line)
            self._f.write(clean + "\n")
            self._f.flush()

    # ── Public log methods ────────────────────────────────────────────────────

    def info(self, msg: str) -> None:
        self._write(f"{C.DIM}[{self._ts()}]{C.RESET} {msg}")

    def ok(self, msg: str) -> None:
        self._write(f"{C.DIM}[{self._ts()}]{C.RESET} {C.GREEN}✅ {msg}{C.RESET}")

    def warn(self, msg: str) -> None:
        self._write(f"{C.DIM}[{self._ts()}]{C.RESET} {C.YELLOW}⚠️  {msg}{C.RESET}")

    def error(self, msg: str) -> None:
        self._write(f"{C.DIM}[{self._ts()}]{C.RESET} {C.RED}❌ {msg}{C.RESET}")

    def section(self, title: str) -> None:
        bar = "─" * 58
        self._write(f"\n{C.CYAN}{C.BOLD}{bar}\n   {title}\n{bar}{C.RESET}")

    def close(self) -> None:
        if self._f:
            self._f.close()
            self._f = None


# ── JSON helpers ──────────────────────────────────────────────────────────────

def load_json(path: str) -> dict:
    """Load JSON file. Raises FileNotFoundError if missing."""
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def save_json(path: str, data: dict, indent: int = 2) -> None:
    """Save dict to JSON atomically (write-then-rename), creating parent dirs as needed."""
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    tmp = str(p) + ".tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=indent, default=str)
    os.replace(tmp, path)


# ── Dataset fingerprinting ────────────────────────────────────────────────────

def compute_dataset_hash(audio_dir: str) -> str:
    """
    Stable hash over all WAV files in audio_dir.

    Uses: filename + file size + first 4 KB of content (sampled checksum).
    The content sample catches the case where two files have the same name
    and size but different content (e.g., after silent re-recording).

    Fast: reads only the first 4 KB per file, not the full content.
    """
    wav_files = sorted(
        f for f in Path(audio_dir).glob("**/*.wav") if f.is_file()
    )
    h = hashlib.md5()
    for f in wav_files:
        h.update(f.name.encode())
        h.update(str(f.stat().st_size).encode())
        # Sample first 4 KB of content to detect same-size content swaps
        try:
            with open(f, "rb") as fh:
                h.update(fh.read(4096))
        except OSError:
            pass
    return h.hexdigest()[:16]


# ── Path validation (dry-run support) ────────────────────────────────────────

class DryRunError(RuntimeError):
    """Raised when --dry-run finds missing required paths."""


def check_paths(required: dict[str, str], log: Logger) -> list[str]:
    """
    Validate that all required file/directory paths exist.

    Args:
        required: {label: path} mapping — e.g. {"HuBERT": "/path/to/hubert.pt"}
        log:      Logger instance.

    Returns:
        List of missing paths (empty = all good).
    """
    missing = []
    for label, path in required.items():
        if path and os.path.exists(path):
            log.ok(f"  {label}: {path}")
        else:
            log.error(f"  {label}: MISSING — {path}")
            missing.append(path)
    return missing


# ── System resource checks ────────────────────────────────────────────────────

def get_free_disk_gb(path: str = "/") -> float:
    """Free disk space at path in GB."""
    usage = shutil.disk_usage(path)
    return usage.free / 1e9


def get_free_vram_gb() -> float:
    """Free VRAM on GPU 0 in GB. Returns 0.0 if no CUDA available."""
    try:
        import torch
        if not torch.cuda.is_available():
            return 0.0
        props = torch.cuda.get_device_properties(0)
        used  = torch.cuda.memory_allocated(0)
        return (props.total_memory - used) / 1e9
    except Exception:
        return 0.0


def get_gpu_info() -> dict:
    """Returns dict with gpu_name, vram_total_gb, vram_free_gb."""
    try:
        import torch
        if not torch.cuda.is_available():
            return {"gpu_name": "none", "vram_total_gb": 0.0, "vram_free_gb": 0.0}
        props = torch.cuda.get_device_properties(0)
        total = props.total_memory / 1e9
        free  = (props.total_memory - torch.cuda.memory_allocated(0)) / 1e9
        return {
            "gpu_name":      props.name,
            "vram_total_gb": round(total, 1),
            "vram_free_gb":  round(free,  1),
        }
    except Exception:
        return {"gpu_name": "unknown", "vram_total_gb": 0.0, "vram_free_gb": 0.0}


# ── Directory setup ───────────────────────────────────────────────────────────

def setup_dirs(base: str, experiment_id: str) -> dict:
    """
    Creates and returns a standardised directory layout for one experiment.

    Returns:
        {
            "base":     str,   # catalyst_rvc/
            "logs":     str,   # experiments/<exp_id>/logs/
            "eval":     str,   # experiments/<exp_id>/eval/
            "model":    str,   # models/<exp_id>/
            "cache":    str,   # cache/<dataset_hash>/  (shared across experiments)
        }
    """
    dirs = {
        "base":  base,
        "logs":  os.path.join(base, "experiments", experiment_id, "logs"),
        "eval":  os.path.join(base, "experiments", experiment_id, "eval"),
        "model": os.path.join(base, "models", experiment_id),
        "cache": os.path.join(base, "cache"),
    }
    for d in dirs.values():
        Path(d).mkdir(parents=True, exist_ok=True)
    return dirs


# ── Misc ──────────────────────────────────────────────────────────────────────

def now_iso() -> str:
    return datetime.now().isoformat(timespec="seconds")


def elapsed_str(start: float) -> str:
    secs = int(time.time() - start)
    m, s = divmod(secs, 60)
    h, m = divmod(m, 60)
    if h:
        return f"{h}h {m}m {s}s"
    if m:
        return f"{m}m {s}s"
    return f"{s}s"
