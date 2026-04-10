# COT Faithfulness via Residual Stream Interventions

## Research Question

How much of a model's post-COT accuracy is carried by token content (recoverable via attention to the KV cache) vs. non-token "subliminal" information accumulated in the final-position residual stream?

## Experimental Conditions

1. **normal**: Generate COT + answer end-to-end. Accuracy ceiling.
2. **self_prefill**: Take generated COT text, re-tokenize, fresh forward pass, sample answer. Sanity check — should match normal (any delta = noise floor).
3. **zeroed_layer_k**: Generate COT normally. On the prefill forward pass, at the last token position, replace the residual entering layer `k` with the raw token embedding (output of `embed_tokens`). Sample answer from resulting logits. Run for k in `[0, L//4, L//2, 3L//4, L-1]`.
4. **cross_model**: Prefill a different model with the same COT text. Existing baseline.

The key comparison is **normal vs. zeroed_layer_0**. If accuracy holds, the token sequence is sufficient and attention to prior KV entries recovers the reasoning. If it drops, subliminal information exists in the residual.

## Task

GSM8K (grade school math). Numeric answers, unambiguous correctness, COT is standard.

- Use the `main` split, first 500 problems for development, full set for final results.
- Answers are integers. Extract with regex after the `####` delimiter.

## Models

**Primary:** `Qwen/Qwen3-4B` (36 layers, hidden_size 2560, standard transformer with GQA)
**Cross-model:** `Qwen/Qwen3-8B` (cross-model prefill condition)

Both are standard dense transformers with grouped query attention — no hybrid attention, no MoE, no architectural confounds. Both fit on a single H200. Qwen3 supports thinking mode (`<think>...</think>`) which gives us native COT.

## Implementation

### Stack

- **nnsight** for residual stream interventions. Install: `pip install nnsight`.
- **transformers** for tokenization and model loading.
- **torch** for tensor ops.
- **datasets** for GSM8K loading.
- No vLLM. Throughput doesn't matter here; intervention access does.

### Environment

- **Local (laptop):** Notebooks and shared utilities are authored here.
- **Remote (H200 Jupyter server):** Notebooks are executed here. All generated data (cache, results, figures) lives on the remote at `/workspace/10-4-2026/`.
- **Shared utilities:** Live in `lib/` and are pushed to `github.com/JackHopkins/legibility`. Remote notebooks clone/pull this repo to access them.

### Project Structure

**Local repo (`JackHopkins/legibility`):**
```
legibility/
├── CLAUDE.md
├── .gitignore              # cache/, results/, figures/
├── lib/
│   ├── __init__.py
│   ├── config.py
│   ├── data.py
│   └── intervention.py
├── 01_setup.ipynb
├── 02_generate_cots.ipynb
├── 03_interventions.ipynb
└── 04_analysis.ipynb
```

**Remote working directory (`/workspace/10-4-2026/`):**
```
/workspace/10-4-2026/
├── legibility/             # Cloned repo (notebooks + lib)
├── cache/                  # Resumable intermediate state (NOT in git)
│   ├── cots/               # One JSON file per problem: {problem_id}.json
│   └── interventions/      # One JSON file per (condition, problem): {condition}_{problem_id}.json
├── results/                # Final aggregated outputs (NOT in git)
│   ├── normal.jsonl
│   ├── self_prefill.jsonl
│   ├── zeroed_layer_0.jsonl
│   └── ...
└── figures/                # Plots (NOT in git)
```

### Caching & Resumability

Notebook sessions on remote GPUs disconnect frequently. Every notebook must be fully resumable.

**Principle: one cache file per atomic unit of work.** Never write a single monolithic output file that must complete fully. Instead:

- `02_generate_cots.ipynb` writes one file per problem to `cache/cots/{problem_id}.json`. On resume, it globs the cache dir, computes the set of already-done IDs, and skips them.
- `03_interventions.ipynb` writes one file per (condition, problem) to `cache/interventions/{condition}_{problem_id}.json`. Same skip logic on resume.
- Aggregation into `results/*.jsonl` happens at the end of each notebook (or in `04_analysis.ipynb`) by reading all cache files.

**Cache file format** (both dirs):
```json
{
  "problem_id": 42,
  "question": "...",
  "gold_answer": 7,
  "condition": "zeroed_layer_0",
  "cot_text": "...",
  "predicted_answer": 7,
  "top1_token": "7",
  "top1_prob": 0.95,
  "gold_token_rank": 1,
  "logits_top10": [["7", 0.95], ["8", 0.02], ...],
  "error": null
}
```

If a problem errors (OOM, nnsight bug), write the cache file anyway with `"error": "<traceback>"` and a null answer. This prevents retrying broken examples in a loop. Fix the bug, delete the errored cache files, and re-run.

**Standard resume pattern for all notebooks:**
```python
import json
from pathlib import Path

CACHE_DIR = Path("/workspace/10-4-2026/cache/cots")
CACHE_DIR.mkdir(parents=True, exist_ok=True)

done_ids = {int(p.stem) for p in CACHE_DIR.glob("*.json")}
todo = [ex for ex in dataset if ex["problem_id"] not in done_ids]
print(f"Resuming: {len(done_ids)} done, {len(todo)} remaining")

for ex in tqdm(todo):
    result = run_one(ex)
    (CACHE_DIR / f"{ex['problem_id']}.json").write_text(json.dumps(result))
```

### Key Implementation Details

**lib/config.py — all constants live here:**
```python
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
ZERO_AT_LAYERS = [0, 9, 18, 27, 35]

CONDITIONS = ["self_prefill"] + [f"zeroed_layer_{k}" for k in ZERO_AT_LAYERS]

MAX_COT_TOKENS = 1024
TEMPERATURE = 0.0  # greedy

# GSM8K
DATASET_NAME = "openai/gsm8k"
DATASET_SPLIT = "main"
```

**Notebook preamble — first cell of EVERY notebook (01–04):**
```python
import subprocess, sys
from pathlib import Path

WORKSPACE = Path("/workspace/10-4-2026")
REPO_DIR = WORKSPACE / "legibility"

# Clone or pull shared utilities
if REPO_DIR.exists():
    subprocess.run(["git", "-C", str(REPO_DIR), "pull"], check=True)
else:
    WORKSPACE.mkdir(parents=True, exist_ok=True)
    subprocess.run(["git", "clone", "https://github.com/JackHopkins/legibility.git", str(REPO_DIR)], check=True)

sys.path.insert(0, str(REPO_DIR))
from lib.config import *

# Create data directories (these live outside the repo on the remote)
for d in [CACHE_DIR, COT_CACHE, INTERVENTION_CACHE, RESULTS_DIR, FIGURES_DIR]:
    d.mkdir(parents=True, exist_ok=True)
```

**Further notebook conventions:**
- Use `tqdm` for all loops so progress is visible if you need to estimate time before a disconnect.
- Flush results to disk inside the loop, never batch-write at the end.

**Setup (01_setup.ipynb):**
- Preamble cell (clone/pull repo, as above).
- Install: `pip install -q nnsight transformers datasets tqdm matplotlib seaborn`
- Download and cache the model: load it once with nnsight so weights are cached in HF home.
- Download GSM8K: `load_dataset("openai/gsm8k", "main")`.
- Verify GPU: print `torch.cuda.get_device_name()`, confirm H200 and sufficient memory.
- Smoke test: run a single forward pass with nnsight on Qwen3-4B, confirm residual shapes match expectations (`[1, seq_len, 2560]` for Qwen3-4B). If nnsight fails on Qwen3, test raw PyTorch hooks as fallback.

**COT generation (02_generate_cots.ipynb):**
- Use Qwen3's native thinking mode — it generates `<think>...</think>` before the answer by default.
- System prompt: "Solve step by step. After your reasoning, write 'The answer is: ' followed by the numeric answer."
- Use greedy decoding (temperature=0) for reproducibility.
- Extract the thinking content from `<think>` tags as the COT text.
- Save each problem's COT text and extracted answer to `results/cots.jsonl`.
- Fields: `{"problem_id", "question", "cot_text", "predicted_answer", "gold_answer"}`.

**Intervention forward pass (03_interventions.ipynb):**

The core intervention using nnsight for Qwen3:

```python
from nnsight import LanguageModel

model = LanguageModel("Qwen/Qwen3-4B", device_map="auto")

def zeroed_residual_logits(prefill_text: str, zero_at_layer: int):
    input_ids = model.tokenizer(prefill_text, return_tensors="pt").input_ids.to(model.device)

    with model.trace(input_ids):
        # Grab the raw embedding at the last position
        embedding = model.model.embed_tokens.output[0, -1, :].clone().save()

        # Replace residual entering target layer at last position
        model.model.layers[zero_at_layer].input[0][0][-1, :] = embedding

        # Grab output logits at last position
        logits = model.lm_head.output[0, -1, :].save()

    return logits.value
```

Verify this works by checking:
1. Without intervention, logits match a normal forward pass.
2. The shapes are correct — `embed_tokens.output` is `[batch, seq, hidden]`.
3. RoPE is applied inside attention, not to the embedding, so raw embedding replacement is clean.
4. Qwen3 uses `model.model.layers` and `model.model.embed_tokens` — same structure as Llama. Confirm in 01_setup.

**nnsight gotchas:**
- Proxy objects behave lazily. Always `.save()` anything you need outside the trace context.
- `.clone()` the embedding before using it as a replacement to avoid graph issues.
- `layer.input[0][0]` may vary by model architecture. Inspect with `model.model.layers[0].input` inside a trace to verify the shape and nesting.
- If `layer.input` doesn't work for assignment, try hooking via `layer.self_attn.input` or using `model.model.layers[k].output` of layer k-1 instead.
- nnsight may not yet support Qwen3 natively. If it fails, fall back to raw PyTorch `register_forward_pre_hook` on `model.model.layers[k]`. The hook receives the layer input tuple and can modify the residual in-place.

**Prefill string construction (Qwen3 chat template):**
```
<|im_start|>user
{question}<|im_end|>
<|im_start|>assistant
{cot_text} The answer is:
```
The model's next token after this should be the numeric answer. Extract the first integer from generated tokens.

**Self-prefill condition:**
Identical to zeroed but with no intervention. This is just a normal forward pass on the prefill string + sampling the next token. Compare to the normal end-to-end answer.

### Metrics

For each condition, compute:
- **Accuracy**: exact match of predicted integer vs. gold answer.
- **Top-1 confidence**: softmax probability of the top predicted token.
- **Gold token rank**: rank of the correct answer token in the logit distribution.
- **KL divergence**: between the logit distribution of each condition vs. normal.

### Analysis (04_analysis.ipynb)

1. **Accuracy table**: rows = conditions, columns = accuracy, 95% CI (bootstrap).
2. **Layer sweep plot**: x = layer index where zeroing occurs, y = accuracy. This is the key figure.
3. **Per-problem analysis**: for problems where zeroed_layer_0 fails but normal succeeds, inspect the COT. Are these problems where the COT is vague or skips steps?
4. **KL heatmap**: KL(zeroed_layer_k || normal) as a function of k.

### Expected Outcomes

- **Self-prefill ≈ normal**: confirms the two-pass setup introduces no artifact.
- **Zeroed at layer 0**: if accuracy is close to normal, subliminal info is minimal — the KV cache carries the reasoning. If accuracy drops significantly, there's load-bearing hidden state.
- **Gradual degradation across layers**: most likely. The interesting question is how early the drop starts.
- **If zeroed_layer_0 matches normal perfectly**: the experiment is less interesting but still publishable as a negative result — COTs in this model are "what you see is what you get."

## Pushing lib/ Changes

**Whenever a completed change is made to any file in `lib/`, push it immediately** so the remote notebooks pick up the latest code on their next preamble cell run.

Run the helper script from the repo root:
```bash
./push_lib.sh
```
This stages `lib/`, commits with a `HH:MM:DD-MM` timestamp message, and pushes to `origin main`.

If you are Claude Code: after editing any file in `lib/`, run `./push_lib.sh` before moving on to the next task.

## Workflow

1. Author notebooks and `lib/` locally.
2. Push to `github.com/JackHopkins/legibility` (use `./push_lib.sh` for lib changes).
3. Upload/open notebooks on the remote Jupyter server.
4. Execute — the preamble cell clones/pulls the latest `lib/` code automatically.
5. All data (cache, results, figures) stays on the remote at `/workspace/10-4-2026/`.

Execute notebooks in order. Each is fully resumable — if the kernel dies or the session disconnects, just re-run all cells. The cache skip logic picks up where it left off.

```
01_setup.ipynb          # Run once. Installs deps, downloads model, creates dirs.
02_generate_cots.ipynb  # Run/re-run until "0 remaining" is printed.
03_interventions.ipynb  # Run/re-run per condition. Set CONDITION variable at top.
04_analysis.ipynb       # Run after all conditions complete. Reads cache, produces figures.
```

**If a notebook is interrupted:** just restart kernel and run all cells. Do NOT clear cache dirs unless you want to redo work.

**If specific problems error:** check `cache/` for files with non-null `"error"` fields. Fix the issue, delete those specific cache files, re-run.

## Extensions

- **Partial zeroing**: instead of fully replacing with embedding, interpolate: `residual = alpha * residual + (1 - alpha) * embedding`. Sweep alpha from 0 to 1.
- **Subspace zeroing**: project out specific directions (e.g., top PCA components of residual - embedding) rather than full replacement. This identifies *which* subliminal info matters.
- **Attention knockout**: instead of zeroing the residual, mask attention at the last position so it can't attend to prior tokens. This tests the opposite direction — is the KV cache necessary, or is the residual alone sufficient?
- **Other models**: compare base vs. instruct, different sizes, reasoning-tuned models (which may rely more heavily on subliminal channels).
- **Other tasks**: MATH, ARC, HellaSwag — vary difficulty and reasoning type.