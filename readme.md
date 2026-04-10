# COT Faithfulness via Residual Stream Interventions

## Research Question

How much of a model's post-COT accuracy is carried by token content (recoverable via attention to the KV cache) vs. non-token "subliminal" information accumulated in the final-position residual stream?

## Method

We generate chain-of-thought (COT) responses on GSM8K using Qwen3-4B, then run a second forward pass where we replace the residual stream at the last token position with the raw token embedding at various layers. If accuracy holds despite this intervention, the token text alone is sufficient -- the COT is faithful. If accuracy drops, hidden state carries load-bearing information beyond what the text encodes.

## Conditions

| Condition | Description |
|-----------|-------------|
| **normal** | Generate COT + answer end-to-end. Accuracy ceiling. |
| **self_prefill** | Re-tokenize COT, fresh forward pass. Sanity check (should match normal). |
| **zeroed_layer_k** | Replace residual at layer k with raw embedding. Tests subliminal info. |
| **cross_model** | Prefill a different model (Qwen3-8B) with the same COT text. |

## Project Structure

```
lib/
  config.py         # All constants
  data.py           # GSM8K loading, answer extraction, prefill construction
  intervention.py   # nnsight/hook-based residual interventions
01_setup.ipynb      # Install deps, download model, verify GPU, smoke test
02_generate_cots.ipynb  # Generate COT responses for all GSM8K problems
03_interventions.ipynb  # Run self_prefill and zeroed_layer_k conditions
04_analysis.ipynb       # Accuracy tables, layer sweep plot, KL analysis
```

## How to Run

1. Push this repo to `github.com/JackHopkins/legibility`
2. On the remote H200 Jupyter server, execute notebooks in order (01 through 04)
3. Each notebook is fully resumable -- if the session disconnects, re-run all cells
4. Results and figures are saved to `/workspace/10-4-2026/results/` and `figures/`

## Models

- **Primary:** Qwen/Qwen3-4B (36 layers, 2560 hidden dim)
- **Cross-model:** Qwen/Qwen3-8B

## Stack

- nnsight (residual stream interventions)
- transformers, torch, datasets
- matplotlib, seaborn (plotting)
