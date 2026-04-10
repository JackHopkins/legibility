from pathlib import Path

WORKSPACE = Path("/workspace/10-4-2026")
REPO_DIR = WORKSPACE / "legibility"
CACHE_DIR = WORKSPACE / "cache"
COT_CACHE = CACHE_DIR / "cots"
INTERVENTION_CACHE = CACHE_DIR / "interventions"
RESULTS_DIR = WORKSPACE / "results"
FIGURES_DIR = WORKSPACE / "figures"

MODEL_NAME = "Qwen/Qwen3-4B"
CROSS_MODEL_NAME = "Qwen/Qwen3-8B"
NUM_LAYERS = 36
HIDDEN_SIZE = 2560
ZERO_AT_LAYERS = [0, 9, 18, 27, 35]

CONDITIONS = ["self_prefill"] + [f"zeroed_layer_{k}" for k in ZERO_AT_LAYERS]

MAX_COT_TOKENS = 1024
TEMPERATURE = 0.0  # greedy

# GSM8K
DATASET_NAME = "openai/gsm8k"
DATASET_CONFIG = "main"
DATASET_SPLIT = "train"

# Subset size for experiments
SUBSET_SIZE = 128