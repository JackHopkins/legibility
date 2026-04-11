from pathlib import Path

WORKSPACE = Path("/workspace/10-4-2026")
REPO_DIR = WORKSPACE / "legibility"
CACHE_DIR = WORKSPACE / "cache"
COT_CACHE = CACHE_DIR / "cots"
PARAPHRASE_CACHE = CACHE_DIR / "paraphrases"
PREFILL_CACHE = CACHE_DIR / "prefills"
INTERVENTION_CACHE = CACHE_DIR / "interventions"
RESULTS_DIR = WORKSPACE / "results"
FIGURES_DIR = WORKSPACE / "figures"

MODEL_NAME = "Qwen/Qwen3-4B"
PARAPHRASER_MODEL = "Qwen/Qwen3-8B"
CROSS_MODEL_NAME = "Qwen/Qwen3-8B"

# Model architecture (for intervention experiments)
NUM_LAYERS = 36
HIDDEN_SIZE = 2560
ZERO_AT_LAYERS = list(range(NUM_LAYERS))

# Paraphrase experiment conditions
CONDITIONS = [
    "no_cot",
    "normal",
    "self_prefill",
    "paraphrase_light",
    "paraphrase_heavy",
    "shuffled_steps",
    "corrupted_numbers",
]

# Intervention experiment conditions
INTERVENTION_CONDITIONS = ["self_prefill"] + [f"zeroed_layer_{k}" for k in ZERO_AT_LAYERS]

MAX_COT_TOKENS = 2048
MAX_ANSWER_TOKENS = 32
TEMPERATURE = 0.0  # greedy throughout

# GSM8K
DATASET_NAME = "openai/gsm8k"
DATASET_CONFIG = "main"
DATASET_SPLIT = "test"

# Subset size for experiments
SUBSET_SIZE = 256
