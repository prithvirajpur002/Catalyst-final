# -*- coding: utf-8 -*-
"""
Catalyst RVC — main.py
Experiment runner entry point.

HOW TO USE
----------
1. Edit EXPERIMENTS below: define which dataset × config combinations to run.
2. Set PATHS to match your environment (Kaggle or local).
3. Run:  python main.py
   or in a notebook:  exec(open("main.py").read())

FLAGS
-----
  --dry-run          Validate all paths and configs, then exit. No training.
  --only exp_001     Run only the specified experiment (can repeat).
  --retry-failed     Re-run only experiments that previously failed.
  --force-extract    Re-extract features even if cache is valid.

Each experiment will:
  1. Preprocess the dataset (skipped if already done and source unchanged)
  2. Extract HuBERT + F0 features (skipped if cache valid)
  3. Train the RVC v2 model (resumes from checkpoint if interrupted)
  4. Build the FAISS index
  5. Run fixed-test evaluation (3 reference clips → quality scores)
  6. Register the model in models/registry.json

After all experiments, the runner prints a comparison table and identifies
the best model by composite score (naturalness-weighted).

Fixes applied:
  - Bug 5:  Late import of elapsed_str moved to module level with all other imports.
  - Bug 15: Dry-run mode added — validate all paths before wasting GPU time.
  - New:    --only flag to run a subset of experiments.
  - New:    --retry-failed flag to re-run only experiments that failed last time.
  - New:    FAISS index shape validation (warns if index vectors != expected dim).
"""

from __future__ import annotations

import os
import sys
import time
from pathlib import Path

# ── Paths — edit these to match your setup ────────────────────────────────────

# For Kaggle: /kaggle/working   |   For Colab: /content/drive/MyDrive/Catalyst
BASE_DIR    = "/kaggle/working/catalyst_rvc"

# Your cloned RVC repo
RVC_REPO    = "/kaggle/working/rvc"

# Pre-trained model assets (downloaded in bootstrap)
PRETRAIN_G  = f"{RVC_REPO}/assets/pretrained_v2/f0G40k.pth"
PRETRAIN_D  = f"{RVC_REPO}/assets/pretrained_v2/f0D40k.pth"
HUBERT_PATH = f"{RVC_REPO}/assets/hubert/hubert_base.pt"

# Input voice audio — WAV files from Kuku Studio or Catalyst Data Engine
INPUT_AUDIO = {
    "v1_clean":   "/kaggle/input/rvc-voice-v1/",
    "v2_natural": "/kaggle/input/rvc-voice-v2/",
}

# ── Experiment definitions ────────────────────────────────────────────────────
# Add or remove entries freely. Each entry is one training run.
#
# Fields:
#   id      : unique name (no spaces) — used for log dirs, model files
#   dataset : key into INPUT_AUDIO above
#   config  : filename in configs/ (without .json)
#   mode    : preprocessing mode — "clean" | "natural" | "raw"

EXPERIMENTS = [
    {"id": "exp_001", "dataset": "v1_clean",   "config": "baseline",      "mode": "clean"},
    {"id": "exp_002", "dataset": "v2_natural",  "config": "baseline",      "mode": "natural"},
    {"id": "exp_003", "dataset": "v1_clean",   "config": "high_quality",  "mode": "clean"},
]

# ── Module-level imports ───────────────────────────────────────────────────────
# Bug 5 fix: elapsed_str was imported at the very end of main(), making it
# easy to miss import errors until after a multi-hour training run completes.
# All src imports now happen here, at module level, before any work begins.

_here = str(Path(__file__).parent)
if _here not in sys.path:
    sys.path.insert(0, _here)

from src.utils import (
    Logger, DryRunError, check_paths, elapsed_str,
    get_free_disk_gb, get_gpu_info, load_json,
    now_iso, save_json, setup_dirs,
)
from src.preprocess      import preprocess_dataset
from src.feature_extract import extract_features
from src.train           import train
from src.evaluate        import evaluate, build_test_clips
from src.compare         import compare_experiments, save_comparison, score_experiment
from src.registry        import (
    register_model, get_best_model, print_registry,
    validate_registry, mark_champion,
)


# ── FAISS index builder ───────────────────────────────────────────────────────

def _build_index(logs_dir: str, model_output: str, exp_id: str, log: Logger) -> str:
    """
    Build FAISS index from HuBERT features.
    Returns path to .index file, or empty string if skipped.

    Production addition: validates that all feature arrays have consistent
    dimensions before building — catches corrupted .npy files early.
    """
    import numpy as np

    try:
        import faiss
    except ImportError:
        log.warn("faiss not installed — skipping index build.  (pip install faiss-gpu)")
        return ""

    feat_dir = Path(logs_dir) / "3_feature768"
    if not feat_dir.exists():
        feat_dir = Path(logs_dir) / "3_feature256"

    npy_files = sorted(feat_dir.glob("*.npy")) if feat_dir.exists() else []
    if not npy_files:
        log.warn("No feature files found — cannot build index.")
        return ""

    log.info(f"Building FAISS index from {len(npy_files)} feature files...")

    arrays: list[np.ndarray] = []
    skipped = 0
    ref_dim: int | None = None

    for f in npy_files:
        try:
            arr = np.load(str(f))
            if arr.ndim != 2 or arr.shape[1] == 0:
                skipped += 1
                continue
            # Validate shape consistency
            if ref_dim is None:
                ref_dim = arr.shape[1]
            if arr.shape[1] != ref_dim:
                log.warn(
                    f"Skipping {f.name}: dim={arr.shape[1]} != expected {ref_dim}. "
                    "Corrupted feature file."
                )
                skipped += 1
                continue
            arrays.append(arr)
        except Exception as e:
            log.warn(f"  Could not load {f.name}: {e}")
            skipped += 1

    if skipped:
        log.warn(f"  {skipped} feature files skipped (shape mismatch or corrupted)")

    if not arrays:
        log.warn("No valid feature arrays — skipping index.")
        return ""

    features = np.concatenate(arrays, axis=0).astype(np.float32)
    dim, n   = features.shape[1], features.shape[0]
    log.info(f"  Indexed: {n:,} vectors × {dim} dim")

    n_ivf       = max(4, min(int(n ** 0.5), n // 4))
    min_for_ivf = n_ivf * 39

    if n >= min_for_ivf:
        q     = faiss.IndexFlatL2(dim)
        index = faiss.IndexIVFFlat(q, dim, n_ivf)
        index.train(features)
        index.add(features)
        idx_type = "IVFFlat"
    else:
        index    = faiss.IndexFlatL2(dim)
        index.add(features)
        idx_type = "FlatL2"

    os.makedirs(model_output, exist_ok=True)
    idx_path = os.path.join(model_output, f"{exp_id}.index")
    faiss.write_index(index, idx_path)

    size_mb = os.path.getsize(idx_path) / 1e6

    # Sanity-query: verify the index can retrieve its own first vector
    try:
        _, I = index.search(features[:1], 1)
        if I[0][0] != 0:
            log.warn("Index sanity check failed — top-1 match is not self. Index may be corrupt.")
        else:
            log.ok(
                f"FAISS {idx_type} index: {index.ntotal:,} vectors → "
                f"{idx_path}  ({size_mb:.1f} MB) ✓ self-query passed"
            )
    except Exception:
        log.ok(
            f"FAISS {idx_type} index: {index.ntotal:,} vectors → "
            f"{idx_path}  ({size_mb:.1f} MB)"
        )

    return idx_path


# ── Dry-run validation ────────────────────────────────────────────────────────

def dry_run(log: Logger) -> None:
    """
    Validate all configured paths and experiment configs without running anything.

    Checks:
      - RVC repo exists
      - Pre-trained models exist
      - HuBERT model exists
      - All INPUT_AUDIO directories exist
      - All config files exist and are valid JSON
      - BASE_DIR is writable

    Raises DryRunError if any required path is missing.
    """
    log.section("DRY RUN — PATH VALIDATION")

    required_assets = {
        "RVC repo":          RVC_REPO,
        "Pretrained G":      PRETRAIN_G,
        "Pretrained D":      PRETRAIN_D,
        "HuBERT model":      HUBERT_PATH,
    }

    missing = check_paths(required_assets, log)

    for key, path in INPUT_AUDIO.items():
        if os.path.isdir(path):
            n_wavs = len(list(Path(path).glob("*.wav")))
            log.ok(f"  Dataset [{key}]: {path}  ({n_wavs} WAV files)")
        else:
            log.error(f"  Dataset [{key}]: MISSING — {path}")
            missing.append(path)

    log.info("\nChecking experiment configs...")
    for exp in EXPERIMENTS:
        config_path = os.path.join(_here, "configs", f"{exp['config']}.json")
        if os.path.exists(config_path):
            try:
                cfg = load_json(config_path)
                log.ok(
                    f"  [{exp['id']}] config={exp['config']}.json  "
                    f"epochs={cfg.get('epochs')}  "
                    f"batch={cfg.get('batch_size')}"
                )
            except Exception as e:
                log.error(f"  [{exp['id']}] config JSON parse error: {e}")
                missing.append(config_path)
        else:
            log.error(f"  [{exp['id']}] config NOT FOUND: {config_path}")
            missing.append(config_path)

        if exp["dataset"] not in INPUT_AUDIO:
            log.error(
                f"  [{exp['id']}] dataset key '{exp['dataset']}' not in INPUT_AUDIO dict"
            )
            missing.append(f"INPUT_AUDIO[{exp['dataset']}]")

    # Write test
    try:
        os.makedirs(BASE_DIR, exist_ok=True)
        test_file = os.path.join(BASE_DIR, ".write_test")
        with open(test_file, "w") as f:
            f.write("ok")
        os.remove(test_file)
        log.ok(f"  BASE_DIR writable: {BASE_DIR}")
    except OSError as e:
        log.error(f"  BASE_DIR not writable: {BASE_DIR}  ({e})")
        missing.append(BASE_DIR)

    if missing:
        raise DryRunError(
            f"\n{len(missing)} required item(s) are missing:\n"
            + "\n".join(f"  • {p}" for p in missing)
            + "\n\nFix the above before running the full pipeline."
        )

    log.ok(f"\nAll checks passed — {len(EXPERIMENTS)} experiments are ready to run.")
    log.info("Remove --dry-run to start training.")


# ── Status tracking ───────────────────────────────────────────────────────────

def _load_run_status(base_dir: str) -> dict:
    """Load previous run status (which experiments passed/failed)."""
    path = os.path.join(base_dir, "run_status.json")
    if os.path.exists(path):
        try:
            return load_json(path)
        except Exception:
            pass
    return {}


def _save_run_status(base_dir: str, status: dict) -> None:
    save_json(os.path.join(base_dir, "run_status.json"), status)


# ── Main runner ───────────────────────────────────────────────────────────────

def main(
    dry_run_mode:   bool = False,
    only_ids:       list[str] | None = None,
    retry_failed:   bool = False,
    force_extract:  bool = False,
) -> None:
    t_total = time.time()

    os.makedirs(BASE_DIR, exist_ok=True)
    log_file = os.path.join(BASE_DIR, "runner.log")

    with Logger(log_file=log_file) as log:

        # ── Dry-run mode ──────────────────────────────────────────────────────
        if dry_run_mode:
            try:
                dry_run(log)
            except DryRunError as e:
                log.error(str(e))
                sys.exit(1)
            return

        # ── Environment summary ───────────────────────────────────────────────
        gpu  = get_gpu_info()
        disk = get_free_disk_gb(BASE_DIR)

        log.section("CATALYST RVC — EXPERIMENT RUNNER")
        log.info(f"GPU      : {gpu['gpu_name']}  ({gpu['vram_total_gb']:.1f} GB)")
        log.info(f"Disk free: {disk:.1f} GB")
        log.info(f"Base dir : {BASE_DIR}")
        log.info(f"Experiments: {len(EXPERIMENTS)}\n")

        if gpu["vram_total_gb"] < 8.0:
            log.warn("VRAM < 8 GB — some experiments may OOM. Consider reducing batch_size.")

        if disk < 10.0:
            log.warn(
                f"Low disk space ({disk:.1f} GB). "
                "Models + logs need ~5 GB per experiment."
            )

        # ── Validate existing registry ────────────────────────────────────────
        broken = validate_registry(BASE_DIR, log=log)
        if broken:
            log.warn(f"Registry has broken paths for: {broken} — they will be re-run.")

        # ── Ensure test clips exist ───────────────────────────────────────────
        test_audio_dir = os.path.join(BASE_DIR, "test_audio")
        first_input    = list(INPUT_AUDIO.values())[0]
        if os.path.exists(first_input):
            build_test_clips(test_audio_dir, first_input, log=log)
        else:
            log.warn(
                f"Input audio not found: {first_input}\n"
                "  Test clips will not be generated until INPUT_AUDIO paths are accessible."
            )

        # ── Filter experiments ────────────────────────────────────────────────
        run_status  = _load_run_status(BASE_DIR)
        experiments = EXPERIMENTS

        if only_ids:
            experiments = [e for e in experiments if e["id"] in only_ids]
            log.info(f"--only: running {[e['id'] for e in experiments]}")

        if retry_failed:
            failed_ids  = {eid for eid, st in run_status.items() if st != "complete"}
            failed_ids |= set(broken)  # broken registry paths treated as failed
            experiments = [e for e in experiments if e["id"] in failed_ids]
            if experiments:
                log.info(f"--retry-failed: re-running {[e['id'] for e in experiments]}")
            else:
                log.ok("No failed experiments to retry.")
                return

        if not experiments:
            log.warn("No experiments to run after applying filters.")
            return

        # ── Run experiments ───────────────────────────────────────────────────
        RESULTS: dict[str, dict] = {}

        for exp in experiments:
            exp_id      = exp["id"]
            dataset_key = exp["dataset"]
            config_name = exp["config"]
            mode        = exp.get("mode", "natural")

            log.section(f"EXPERIMENT: {exp_id}")

            # Load config
            config_path = os.path.join(_here, "configs", f"{config_name}.json")
            if not os.path.exists(config_path):
                log.error(f"Config not found: {config_path} — skipping {exp_id}")
                run_status[exp_id] = "config_missing"
                _save_run_status(BASE_DIR, run_status)
                continue
            config = load_json(config_path)
            log.info(f"Config: {config_name}  ({config.get('epochs')} epochs)")

            # Input audio dir
            input_dir = INPUT_AUDIO.get(dataset_key, "")
            if not input_dir or not os.path.exists(input_dir):
                log.error(f"Input dir not found: {input_dir} — skipping {exp_id}")
                run_status[exp_id] = "input_missing"
                _save_run_status(BASE_DIR, run_status)
                continue

            # Setup directory layout
            dirs = setup_dirs(BASE_DIR, exp_id)

            # RVC-specific directories
            rvc_logs    = os.path.join(RVC_REPO, "logs", exp_id)
            dataset_dir = os.path.join(rvc_logs, "dataset")
            os.makedirs(rvc_logs,    exist_ok=True)
            os.makedirs(dataset_dir, exist_ok=True)

            try:
                # ── Step 1: Preprocess ────────────────────────────────────────
                meta_path = os.path.join(dataset_dir, "metadata.json")
                if not os.path.exists(meta_path):
                    preprocess_dataset(
                        input_dir=input_dir,
                        output_dir=dataset_dir,
                        mode=mode,
                        target_sr=config.get("sample_rate", 40000),
                        log=log,
                    )
                else:
                    log.ok("Dataset already preprocessed — skipping")

                # ── Step 2: Feature extraction ────────────────────────────────
                extract_features(
                    dataset_dir=dataset_dir,
                    cache_dir=dirs["cache"],   # Bug 4 fix: now actually used
                    logs_dir=rvc_logs,
                    hubert_path=HUBERT_PATH,
                    rvc_repo=RVC_REPO,
                    sample_rate=config.get("sample_rate", 40000),
                    force=force_extract,
                    log=log,
                )

                # ── Step 3: Train ─────────────────────────────────────────────
                model_pth = train(
                    experiment_id=exp_id,
                    dataset_dir=dataset_dir,
                    logs_dir=rvc_logs,
                    model_output=dirs["model"],
                    config=config,
                    rvc_repo=RVC_REPO,
                    pretrain_g=PRETRAIN_G,
                    pretrain_d=PRETRAIN_D,
                    resume=True,
                    log=log,
                )

                # ── Step 4: FAISS index ───────────────────────────────────────
                idx_path = _build_index(rvc_logs, dirs["model"], exp_id, log)

                # ── Step 5: Evaluate ──────────────────────────────────────────
                eval_results = evaluate(
                    model_path=model_pth,
                    index_path=idx_path,
                    eval_dir=dirs["eval"],
                    rvc_repo=RVC_REPO,
                    base_dir=BASE_DIR,
                    log=log,
                )

                scores = score_experiment(eval_results)
                RESULTS[exp_id] = scores

                # ── Step 6: Register ──────────────────────────────────────────
                register_model(
                    experiment_id=exp_id,
                    model_path=model_pth,
                    index_path=idx_path,
                    config=config,
                    scores=scores,
                    dataset_name=dataset_key,
                    base_dir=BASE_DIR,
                    log=log,
                )

                run_status[exp_id] = "complete"
                log.ok(
                    f"{exp_id} complete — "
                    f"naturalness={scores.get('naturalness',0):.3f}  "
                    f"composite={scores.get('naturalness',0)*0.45 + scores.get('clarity',0)*0.35:.3f}"
                )

            except Exception as e:
                run_status[exp_id] = "failed"
                log.error(f"Experiment {exp_id} FAILED: {e}")
                import traceback
                log.info(traceback.format_exc())

            finally:
                # Always persist status so --retry-failed works
                _save_run_status(BASE_DIR, run_status)

        # ── Compare and summarise ─────────────────────────────────────────────
        if RESULTS:
            report_path = os.path.join(BASE_DIR, "comparison_report.json")
            save_comparison(RESULTS, report_path, log=log)

            best_id = compare_experiments(RESULTS, log=log)
            log.section(f"DONE — Best model: {best_id}")

            best = get_best_model(BASE_DIR)
            if best:
                log.ok(f"Deploy model : {best['model_path']}")
                if best.get("index_path"):
                    log.ok(f"Deploy index : {best['index_path']}")
                if best.get("champion"):
                    log.ok(f"  (manually selected as champion)")
                else:
                    log.info(
                        "  Tip: after listening test, call mark_champion(experiment_id) "
                        "to lock in your preferred model."
                    )
        else:
            log.warn("No experiments completed successfully.")

        print_registry(BASE_DIR, log=log)

        # Report failed experiments
        failed = [eid for eid, st in run_status.items() if st != "complete"]
        if failed:
            log.warn(
                f"Failed experiments: {failed}\n"
                "  Re-run with --retry-failed to attempt them again."
            )

        log.info(f"\nTotal runtime: {elapsed_str(t_total)}")
        log.info(f"Full log: {log_file}")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Catalyst RVC — experiment runner",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
examples:
  python main.py                                # run all experiments
  python main.py --dry-run                      # validate paths only
  python main.py --only exp_001 --only exp_003  # run specific experiments
  python main.py --retry-failed                 # re-run failed experiments
  python main.py --force-extract                # re-extract features (bypass cache)
        """,
    )
    parser.add_argument(
        "--dry-run", action="store_true",
        help="Validate all paths and configs, then exit without training.",
    )
    parser.add_argument(
        "--only", action="append", dest="only_ids", metavar="EXP_ID",
        help="Run only this experiment ID. Can be specified multiple times.",
    )
    parser.add_argument(
        "--retry-failed", action="store_true",
        help="Re-run only experiments that failed in the previous run.",
    )
    parser.add_argument(
        "--force-extract", action="store_true",
        help="Re-extract features even if the cache is valid.",
    )

    args = parser.parse_args()
    main(
        dry_run_mode  = args.dry_run,
        only_ids      = args.only_ids,
        retry_failed  = args.retry_failed,
        force_extract = args.force_extract,
    )
