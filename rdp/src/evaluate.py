# -*- coding: utf-8 -*-
"""
Catalyst RVC — src/evaluate.py
Fixed test evaluation: runs inference on a consistent set of test clips
so experiment results are directly comparable.

FIXED_TESTS are always the same files — never randomly sampled.
This makes comparison between experiments (exp_001 vs exp_002) meaningful.

Fixes applied:
  - Bug 2:  sf.read() on PCM_16 files returns int16 range (-32768 to 32767),
            NOT float32 [-1, 1]. Added explicit normalization so RMS/peak
            calculations are correct. Previously all metrics were off by
            factor of 32768, making composite scores completely wrong.
  - Bug 6:  build_test_clips() no longer pretends files are "neutral",
            "emotional", "fast_speech" — they are named test_1/2/3 since
            alphabetical ordering has nothing to do with speech content.
            Also added warning that test clips should come from a HELD-OUT
            set, not from training data.
  - Bug 13: Inference failures now capture and log stderr so you know WHY
            inference failed (OOM, wrong model, wrong index format, etc.)
            instead of silently recording "success: False".
"""

from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path
from typing import Optional

import numpy as np
import soundfile as sf

from .utils import Logger, now_iso, save_json

# ── Fixed test set ────────────────────────────────────────────────────────────
# These files must exist in test_audio/ before evaluation.
# They are never modified — read-only inputs.
# Names are test_1/2/3 — do NOT label them neutral/emotional/fast unless you
# manually curated them for those properties.

FIXED_TESTS: dict[str, str] = {
    "test_1": "test_audio/test_1.wav",
    "test_2": "test_audio/test_2.wav",
    "test_3": "test_audio/test_3.wav",
}

# Default inference settings — conservative, prioritize naturalness
_DEFAULT_INFER = {
    "f0_up_key":      0,       # No pitch shift
    "index_rate":     0.75,    # 0.75 = good blend of identity + source prosody
    "filter_radius":  3,       # F0 curve smoothness (1=sharp, 7=smooth)
    "rms_mix_rate":   0.0,     # 0 = keep source energy envelope (more natural)
    "protect":        0.33,    # Consonant breath protection
    "f0_method":      "rmvpe",
}


def _find_cli(rvc_repo: str) -> str:
    candidates = [
        "tools/infer_cli.py",
        "infer_cli.py",
    ]
    for rel in candidates:
        p = os.path.join(rvc_repo, rel)
        if os.path.exists(p):
            return p
    raise RuntimeError(
        "infer_cli.py not found in RVC repo.\n"
        f"Searched: {candidates}\n"
        "Re-clone the repo in bootstrap."
    )


def _run_inference(
    cli: str,
    rvc_repo: str,
    model_path: str,
    index_path: Optional[str],
    input_path: str,
    output_path: str,
    settings: dict,
    log: Logger,
) -> tuple[bool, str]:
    """
    Run single-file RVC inference.

    Returns:
        (success: bool, error_message: str)

    Bug 13 fix: capture_output=True was swallowing all errors. Now stderr
    is captured and returned so the caller can log why inference failed.
    """
    if os.path.exists(output_path) and os.path.getsize(output_path) > 1000:
        return True, ""  # Already done

    has_index = index_path and os.path.exists(index_path)

    cmd = [
        sys.executable, cli,
        "--f0up_key",      str(settings["f0_up_key"]),
        "--input_path",    input_path,
        "--index_path",    index_path if has_index else "",
        "--f0method",      settings["f0_method"],
        "--opt_path",      output_path,
        "--model_name",    model_path,
        "--index_rate",    str(settings["index_rate"]),
        "--device",        "cuda:0",
        "--is_half",       "True",
        "--filter_radius", str(settings["filter_radius"]),
        "--resample_sr",   "0",
        "--rms_mix_rate",  str(settings["rms_mix_rate"]),
        "--protect",       str(settings["protect"]),
    ]

    r = subprocess.run(cmd, capture_output=True, text=True, cwd=rvc_repo)

    succeeded = (
        r.returncode == 0
        and os.path.exists(output_path)
        and os.path.getsize(output_path) > 500
    )

    if not succeeded:
        # Surface the actual error — could be OOM, missing model, bad index, etc.
        err = (r.stderr or r.stdout or "no output captured").strip()[-800:]
        return False, err

    return True, ""


def _audio_metrics(path: str) -> dict:
    """
    Compute lightweight audio quality metrics from the output WAV file.

    Bug 2 fix: soundfile.read() on PCM_16 files returns values in the range
    [-32768, 32767], not [-1.0, 1.0]. The code now normalizes to float32
    [-1, 1] before computing RMS/peak/crest so all metrics are in correct
    dBFS range (e.g. -20 dBFS, not +90 dBFS).

    These are HEURISTIC proxy metrics — not perceptual scores — but they
    correctly flag: silence, clipping, and unusual dynamic range.
    """
    try:
        y, sr = sf.read(path, always_2d=False)
        if y.ndim > 1:
            y = y[:, 0]

        # Normalize to float32 [-1, 1] regardless of source bit depth.
        # sf.read returns int16 range for PCM_16 files unless dtype is set.
        y = y.astype(np.float64)
        if np.max(np.abs(y)) > 2.0:
            # PCM_16: values in [-32768, 32767] → normalize
            y = y / 32768.0
        y = y.astype(np.float32)

        rms_db    = float(20.0 * np.log10(np.sqrt(np.mean(y ** 2)) + 1e-9))
        peak_db   = float(20.0 * np.log10(np.max(np.abs(y)) + 1e-9))
        crest_db  = peak_db - rms_db
        duration  = len(y) / sr

        # silence_ratio: fraction of samples below -46 dBFS (near-silent)
        silence_r = float(np.mean(np.abs(y) < 0.005))

        return {
            "rms_db":        round(rms_db,    2),
            "peak_db":       round(peak_db,   2),
            "crest_db":      round(crest_db,  2),
            "duration_s":    round(duration,  2),
            "silence_ratio": round(silence_r, 4),
        }
    except Exception as e:
        return {"error": str(e)}


# ── Public API ────────────────────────────────────────────────────────────────

def evaluate(
    model_path: str,
    index_path: str,
    eval_dir: str,
    rvc_repo: str,
    base_dir: str = ".",
    infer_settings: Optional[dict] = None,
    log: Optional[Logger] = None,
) -> dict:
    """
    Run inference on all FIXED_TESTS and collect quality metrics.

    Args:
        model_path:     Path to the stripped .pth model file.
        index_path:     Path to the .index file (may be empty string).
        eval_dir:       Directory to write output WAV files.
        rvc_repo:       Path to cloned RVC repo.
        base_dir:       Root directory where test_audio/ lives.
        infer_settings: Override default inference settings (optional).
        log:            Logger instance.

    Returns:
        Dict of {test_name: {output_path, metrics, success}} per test.
    """
    if log is None:
        log = Logger()

    settings = {**_DEFAULT_INFER, **(infer_settings or {})}
    os.makedirs(eval_dir, exist_ok=True)

    if not os.path.exists(model_path):
        raise RuntimeError(f"Model not found: {model_path}")

    cli     = _find_cli(rvc_repo)
    model_mb = os.path.getsize(model_path) / 1e6
    has_idx  = bool(index_path and os.path.exists(index_path))
    idx_mb   = os.path.getsize(index_path) / 1e6 if has_idx else 0.0

    log.section("EVALUATION")
    log.info(f"Model: {Path(model_path).name}  ({model_mb:.0f} MB)")
    log.info(f"Index: {Path(index_path).name if has_idx else '(none)'}  ({idx_mb:.0f} MB)")
    log.info(f"Settings: index_rate={settings['index_rate']}  protect={settings['protect']}")

    results: dict = {}
    any_success = False

    for test_name, rel_path in FIXED_TESTS.items():
        src = os.path.join(base_dir, rel_path)
        dst = os.path.join(eval_dir, f"{test_name}.wav")

        if not os.path.exists(src):
            log.warn(f"  {test_name}: source missing ({src}) — skipped")
            results[test_name] = {"success": False, "reason": "source_missing"}
            continue

        log.info(f"\n  ▶  {test_name}  ({os.path.basename(src)})")

        ok, err_msg = _run_inference(
            cli, rvc_repo, model_path,
            index_path if has_idx else None,
            src, dst, settings, log,
        )

        if not ok:
            # Bug 13 fix: log the actual error message from stderr
            log.warn(f"     Inference FAILED for {test_name}")
            if err_msg:
                log.warn(f"     Error: {err_msg}")
            results[test_name] = {
                "success":       False,
                "output_path":   dst,
                "error_message": err_msg,
            }
        else:
            metrics = _audio_metrics(dst)
            log.ok(f"     Output: {dst}")
            log.info(
                f"     RMS={metrics.get('rms_db','?')} dBFS  "
                f"Peak={metrics.get('peak_db','?')} dBFS  "
                f"Crest={metrics.get('crest_db','?')} dB  "
                f"Dur={metrics.get('duration_s','?')}s  "
                f"Silence={metrics.get('silence_ratio','?'):.1%}"
            )
            results[test_name] = {
                "success":     True,
                "output_path": dst,
                "metrics":     metrics,
            }
            any_success = True

    if not any_success:
        log.warn("No test outputs were produced. Check model + RVC inference setup.")
    else:
        n_ok = sum(1 for v in results.values() if v.get("success"))
        log.ok(f"Evaluation done — {n_ok}/{len(FIXED_TESTS)} tests passed")

    summary = {
        "model_path":     model_path,
        "index_path":     index_path,
        "infer_settings": settings,
        "results":        results,
        "evaluated_at":   now_iso(),
    }
    save_json(os.path.join(eval_dir, "eval_results.json"), summary)

    return results


def build_test_clips(
    test_audio_dir: str,
    input_wav_dir: str,
    log: Optional[Logger] = None,
) -> None:
    """
    Bootstrap helper: copies the first 3 WAV files from your dataset
    into test_audio/ as fixed test clips, if test_audio/ is empty.

    IMPORTANT — Bug 6 fix:
      Files are named test_1.wav, test_2.wav, test_3.wav. They are NOT
      labeled neutral/emotional/fast_speech because alphabetical ordering
      of dataset files has nothing to do with speech content.

    BEST PRACTICE:
      Ideally, your test clips should come from a DIFFERENT source than
      your training data so evaluation is not contaminated by in-distribution
      bias. If you only have one data source, this helper still works, but
      keep it in mind when interpreting scores.

    Call this ONCE before running experiments. After that, test_audio/
    is frozen — never regenerate from a different source.
    """
    if log is None:
        log = Logger()

    os.makedirs(test_audio_dir, exist_ok=True)
    existing = [f for f in os.listdir(test_audio_dir) if f.endswith(".wav")]
    if len(existing) >= 3:
        log.info(f"test_audio/ already has {len(existing)} clips — skipping.")
        return

    import shutil
    wav_files = sorted(Path(input_wav_dir).glob("*.wav"))
    if len(wav_files) < 3:
        log.warn(f"Need at least 3 WAVs in {input_wav_dir} to build test clips.")
        return

    names = ["test_1.wav", "test_2.wav", "test_3.wav"]
    for src, dst_name in zip(wav_files[:3], names):
        dst = os.path.join(test_audio_dir, dst_name)
        shutil.copy2(src, dst)
        log.ok(f"  {src.name} → {dst_name}")

    log.ok("Fixed test clips ready in test_audio/")
    log.info(
        "  NOTE: For unbiased evaluation, consider using audio from a "
        "different source than your training data."
    )
