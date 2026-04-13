# COT Legibility via Semantic Bottleneck

Measuring how much of a chain-of-thought's value comes from universally legible semantic content versus model-specific token encoding.

## Research Question

How much of a COT's value to the originating model comes from **universally legible semantic content** versus **model-specific token encoding**?

## Method

Pass COTs through two complementary **semantic bottlenecks**:

1. **Paraphrasing** — rewrites COT, preserving logic but destroying token-level patterns
2. **Cross-model prefill** — a different model family reads the COT, stripping model-specific representations

The 2x2 combination (original/paraphrased x same-model/cross-model) decomposes COT value into:
- Universally legible content (safety-relevant: what a monitor can rely on)
- Model-specific semantic content
- Token-level encoding

## Models

| Role | Model |
|---|---|
| Primary (COT generator) | Qwen/Qwen3-4B |
| Cross-model reader | google/gemma-3-4b-it |
| Paraphraser | Qwen/Qwen3-8B |

## Task

GSM8K test split (1,319 grade-school math problems).

## Key Metric

**Legibility score L** = (Acc(paraphrase_cross) - Acc(no_cot)) / (Acc(self_prefill) - Acc(no_cot))

- L = 1: all COT value is universally legible. Monitoring works.
- L = 0: all value is model-specific or encoded. Monitoring is unreliable.

## Notebooks

| Notebook | Purpose | GPU needed |
|---|---|---|
| 01_setup | Install deps, smoke-test models | Yes |
| 02_generate_cots | Generate COTs + no-COT baseline | Yes (Qwen3-4B) |
| 03_paraphrase | Light/heavy paraphrases + transforms | Yes (Qwen3-8B) |
| 04_prefill_conditions | Run all prefill conditions | Yes (Qwen3-4B, Gemma-3-4B) |
| 05_analysis | Compute metrics, generate figures | No |

## Conditions

| Condition | Reader | COT transform |
|---|---|---|
| no_cot | Primary | None |
| normal | Primary | None |
| self_prefill | Primary | Verbatim |
| cross_prefill | Cross | Verbatim |
| paraphrase_self | Primary | Light paraphrase |
| paraphrase_cross | Cross | Light paraphrase |
| heavy_paraphrase_self | Primary | Heavy paraphrase |
| heavy_paraphrase_cross | Cross | Heavy paraphrase |
| shuffled_steps | Primary | Steps shuffled |
| corrupted_numbers | Primary | Numbers randomized |
