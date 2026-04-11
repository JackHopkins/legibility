"""Prefill + answer extraction via vLLM."""

import json
from pathlib import Path

from lib.config import MAX_ANSWER_TOKENS, TEMPERATURE
from lib.data import extract_predicted_answer
from lib.prompts import build_prefill_string


def run_prefill_batch(llm, examples, condition, cache_dir, chunk_size=64):
    """Run prefill + short generation for a batch of examples.

    Args:
        llm: vLLM LLM instance (already loaded).
        examples: List of dicts with problem_id, question, gold_answer, cot_text.
        condition: Condition name string.
        cache_dir: Path to cache directory for this condition.
        chunk_size: Number of prompts per vLLM batch call.

    Returns:
        List of result dicts.
    """
    from vllm import SamplingParams
    from tqdm import tqdm

    cache_dir = Path(cache_dir)
    cache_dir.mkdir(parents=True, exist_ok=True)

    # Resume: find already-done problem IDs
    done_ids = set()
    for p in cache_dir.glob(f"{condition}_*.json"):
        try:
            pid = int(p.stem.split("_")[-1])
            done_ids.add(pid)
        except ValueError:
            pass

    todo = [ex for ex in examples if ex["problem_id"] not in done_ids]
    print(f"[{condition}] Resuming: {len(done_ids)} done, {len(todo)} remaining")

    if not todo:
        return []

    sampling = SamplingParams(temperature=TEMPERATURE, max_tokens=MAX_ANSWER_TOKENS)
    results = []

    for i in tqdm(range(0, len(todo), chunk_size), desc=condition):
        chunk = todo[i : i + chunk_size]
        prompts = [
            build_prefill_string(ex["question"], ex["cot_text"]) for ex in chunk
        ]
        outputs = llm.generate(prompts, sampling)

        for ex, output in zip(chunk, outputs):
            generated_text = output.outputs[0].text
            predicted = extract_predicted_answer(generated_text)

            result = {
                "problem_id": ex["problem_id"],
                "condition": condition,
                "question": ex["question"],
                "gold_answer": ex["gold_answer"],
                "cot_text": ex["cot_text"],
                "prefill_text": prompts[chunk.index(ex)] if len(chunk) <= chunk_size else "",
                "generated_tokens": generated_text,
                "predicted_answer": predicted,
                "error": None,
            }
            results.append(result)

            cache_path = cache_dir / f"{condition}_{ex['problem_id']}.json"
            cache_path.write_text(json.dumps(result))

    return results
