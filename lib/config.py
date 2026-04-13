from pathlib import Path

WORKSPACE = Path("/workspace/13-4-2026")
REPO_DIR = WORKSPACE / "legibility"
CACHE_DIR = WORKSPACE / "cache"
COT_CACHE = CACHE_DIR / "cots"
PARAPHRASE_CACHE = CACHE_DIR / "paraphrases"
PREFILL_CACHE = CACHE_DIR / "prefills"
RESULTS_DIR = WORKSPACE / "results"
FIGURES_DIR = WORKSPACE / "figures"

# --- Models ---
PRIMARY_MODEL = "Qwen/Qwen3-4B"         # COT generator + self-prefill reader
CROSS_MODEL = "google/gemma-3-4b-it"     # Cross-model reader (different family)
PARAPHRASER_MODEL = "Qwen/Qwen3-8B"     # Neutral paraphraser

# --- Conditions ---
# Which model reads the COT for each condition
CONDITIONS = {
    "no_cot":                {"reader": PRIMARY_MODEL,  "cot_transform": None},
    "normal":                {"reader": PRIMARY_MODEL,  "cot_transform": None},
    "self_prefill":          {"reader": PRIMARY_MODEL,  "cot_transform": "verbatim"},
    "cross_prefill":         {"reader": CROSS_MODEL,    "cot_transform": "verbatim"},
    "paraphrase_self":       {"reader": PRIMARY_MODEL,  "cot_transform": "paraphrase_light"},
    "paraphrase_cross":      {"reader": CROSS_MODEL,    "cot_transform": "paraphrase_light"},
    "heavy_paraphrase_self": {"reader": PRIMARY_MODEL,  "cot_transform": "paraphrase_heavy"},
    "heavy_paraphrase_cross":{"reader": CROSS_MODEL,    "cot_transform": "paraphrase_heavy"},
    "shuffled_steps":        {"reader": PRIMARY_MODEL,  "cot_transform": "shuffle"},
    "corrupted_numbers":     {"reader": PRIMARY_MODEL,  "cot_transform": "corrupt_numbers"},
}

# --- Extension models ---
SCALING_MODELS = [
    "Qwen/Qwen3-1.7B",
    "Qwen/Qwen3-4B",
    "Qwen/Qwen3-8B",
    "Qwen/Qwen3-14B",
]

PARAPHRASER_SWEEP_MODELS = [
    "google/gemma-3-4b-it",
    "meta-llama/Llama-3.2-3B-Instruct",
    "mistralai/Mistral-7B-Instruct-v0.3",
]

CROSS_READER_MODELS = [
    "google/gemma-3-4b-it",
    "meta-llama/Llama-3.2-3B-Instruct",
    "mistralai/Mistral-7B-Instruct-v0.3",
]

# --- Generation parameters ---
MAX_COT_TOKENS = 2048
MAX_ANSWER_TOKENS = 32
TEMPERATURE = 0.0  # Greedy throughout

# --- Dataset ---
DATASET_NAME = "openai/gsm8k"
DATASET_SPLIT = "test"

# --- Batch processing ---
CHUNK_SIZE = 64
