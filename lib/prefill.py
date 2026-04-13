"""Prefill + answer extraction via vLLM."""

import json
from pathlib import Path
from tqdm import tqdm

from lib.config import PREFILL_CACHE, CHUNK_SIZE, MAX_ANSWER_TOKENS, TEMPERATURE
from lib.data import extract_predicted_answer
from lib.prompts import build_prefill_prompt


def get_done_ids(condition: str) -> set:
    """Get problem IDs already cached for a given prefill condition."""
    done = set()
    for p in PREFILL_CACHE.glob(f"{condition}_*.json"):
        try:
            pid = int(p.stem.rsplit("_", 1)[-1])
            done.add(pid)
        except (ValueError, IndexError):
            pass
    return done


def run_prefill_condition(
    llm,
    condition: str,
    model_name: str,
    examples: list,
    cot_lookup: dict,
):
    """Run a single prefill condition over all examples.

    Args:
        llm: vLLM LLM instance.
        condition: Condition name (e.g. 'self_prefill', 'paraphrase_cross').
        model_name: Model name string (used to select chat template).
        examples: List of dicts with 'problem_id', 'question', 'gold_answer'.
        cot_lookup: Dict mapping problem_id -> COT text to prefill with.
    """
    from vllm import SamplingParams

    PREFILL_CACHE.mkdir(parents=True, exist_ok=True)
    done_ids = get_done_ids(condition)

    todo = [
        ex for ex in examples
        if ex["problem_id"] not in done_ids and ex["problem_id"] in cot_lookup
    ]
    print(f"[{condition}] Resuming: {len(done_ids)} done, {len(todo)} remaining")

    if not todo:
        return

    sampling = SamplingParams(temperature=TEMPERATURE, max_tokens=MAX_ANSWER_TOKENS)

    for i in tqdm(range(0, len(todo), CHUNK_SIZE), desc=condition):
        chunk = todo[i:i + CHUNK_SIZE]
        prompts = []
        for ex in chunk:
            cot_text = cot_lookup[ex["problem_id"]]
            prompt = build_prefill_prompt(ex["question"], cot_text, model_name)
            prompts.append(prompt)

        outputs = llm.generate(prompts, sampling)

        for ex, output in zip(chunk, outputs):
            generated = output.outputs[0].text.strip()
            predicted = extract_predicted_answer(generated)
            result = {
                "problem_id": ex["problem_id"],
                "condition": condition,
                "model_used": model_name,
                "prefill_text": cot_lookup[ex["problem_id"]][:200] + "...",
                "predicted_answer": predicted,
                "gold_answer": ex["gold_answer"],
                "generated_tokens": generated,
                "error": None,
            }
            cache_path = PREFILL_CACHE / f"{condition}_{ex['problem_id']}.json"
            cache_path.write_text(json.dumps(result))


def load_prefill_results(condition: str) -> list:
    """Load all cached results for a given condition."""
    results = []
    for p in sorted(PREFILL_CACHE.glob(f"{condition}_*.json")):
        results.append(json.loads(p.read_text()))
    return results
