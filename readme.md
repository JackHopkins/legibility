# COT Legibility via Semantic Bottleneck

## Research Question

How much of a COT's value to the originating model comes from **semantic content** (the logical reasoning a human could follow) versus **token-level encoding** (the model exploiting its own specific token choices, phrasing, and formatting)?

## Method

A paraphrasing-based approach: an external model rewrites the COT, preserving logical steps but destroying token-level encoding. Prefilling the original model with the paraphrased COT forces it to work from semantic content alone.

### Conditions

| Condition | Description |
|---|---|
| `no_cot` | Direct answer, no reasoning (baseline) |
| `normal` | End-to-end COT + answer (ceiling) |
| `self_prefill` | Prefill with own COT, sample answer (noise floor) |
| `paraphrase_light` | Light reword by external model |
| `paraphrase_heavy` | Compressed to key steps by external model |
| `shuffled_steps` | COT steps randomly reordered |
| `corrupted_numbers` | Intermediate numbers replaced with random values |

### Legibility Score

```
L = (Acc(paraphrase) - Acc(no_cot)) / (Acc(self_prefill) - Acc(no_cot))
```

L = 1: all COT value is semantic. L = 0: all value is token-level encoding.

## Models

- **Primary**: Qwen/Qwen3-4B (COT generator + evaluator)
- **Paraphraser**: Qwen/Qwen3-8B (independent model for rewording)

## Task

GSM8K (grade school math) - full test set (1,319 problems).

## Notebooks

```
01_setup.ipynb              # Install deps, verify GPU, smoke test
02_generate_cots.ipynb      # Generate COTs + no_cot baseline
03_paraphrase.ipynb         # Generate all paraphrase variants
04_prefill_conditions.ipynb # Run all prefill conditions
05_analysis.ipynb           # Metrics, figures, legibility scores
```

## Running

1. Push lib/ changes: `./push_lib.sh`
2. Execute notebooks in order on remote H200 Jupyter server
3. Each notebook is fully resumable via per-problem caching
