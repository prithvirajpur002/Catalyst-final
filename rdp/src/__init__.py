# -*- coding: utf-8 -*-
"""
Catalyst RVC — src package
"""

from .utils          import Logger, DryRunError, check_paths, elapsed_str, get_gpu_info
from .preprocess     import preprocess_dataset, MODES
from .feature_extract import extract_features
from .train          import train
from .evaluate       import evaluate, build_test_clips, FIXED_TESTS
from .compare        import compare_experiments, save_comparison, score_experiment
from .registry       import (
    register_model, get_best_model, list_models,
    print_registry, validate_registry, mark_champion, clear_champion,
)

__all__ = [
    "Logger", "DryRunError", "check_paths", "elapsed_str", "get_gpu_info",
    "preprocess_dataset", "MODES",
    "extract_features",
    "train",
    "evaluate", "build_test_clips", "FIXED_TESTS",
    "compare_experiments", "save_comparison", "score_experiment",
    "register_model", "get_best_model", "list_models",
    "print_registry", "validate_registry", "mark_champion", "clear_champion",
]
