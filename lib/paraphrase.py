"""Paraphrasing logic using vLLM."""

import json
from pathlib import Path
from tqdm import tqdm

from lib.config import PARAPHRASE_CACHE, CHUNK_SIZE, MAX_COT_TOKENS, TEMPERATURE
from lib.prompts import build_paraphrase_messages


def get_done_ids(condition: str) -> set:
    """Get problem IDs already cached for a given paraphrase condition."""
    done = set()
    for p in PARAPHRASE_CACHE.glob(f"{condition}_*.json"):
        try:
            pid = int(p.stem.split("_", 1)[1])
            done.add(pid)
        except (ValueError, IndexError):
            pass
    return done


def paraphrase_cots(llm, tokenizer, cots: list, condition: str, heavy: bool = False):
    """Paraphrase a list of COTs using the loaded vLLM model.

    Args:
        llm: vLLM LLM instance (paraphraser model).
        tokenizer: The tokenizer for chat template formatting.
        cots: List of dicts with 'problem_id' and 'cot_text'.
        condition: Cache condition name (e.g. 'paraphrase_light').
        heavy: If True, use heavy paraphrase prompt.
    """
    from vllm import SamplingParams

    PARAPHRASE_CACHE.mkdir(parents=True, exist_ok=True)
    done_ids = get_done_ids(condition)
    todo = [c for c in cots if c["problem_id"] not in done_ids]
    print(f"[{condition}] Resuming: {len(done_ids)} done, {len(todo)} remaining")

    if not todo:
        return

    sampling = SamplingParams(temperature=TEMPERATURE, max_tokens=MAX_COT_TOKENS)

    for i in tqdm(range(0, len(todo), CHUNK_SIZE), desc=condition):
        chunk = todo[i:i + CHUNK_SIZE]
        messages_batch = [build_paraphrase_messages(c["cot_text"], heavy=heavy) for c in chunk]

        # Apply chat template
        prompts = []
        for msgs in messages_batch:
            prompt = tokenizer.apply_chat_template(
                msgs, tokenize=False, add_generation_prompt=True
            )
            prompts.append(prompt)

        outputs = llm.generate(prompts, sampling)

        for c, output in zip(chunk, outputs):
            paraphrased = output.outputs[0].text.strip()
            result = {
                "problem_id": c["problem_id"],
                "condition": condition,
                "original_cot": c["cot_text"],
                "paraphrased_cot": paraphrased,
            }
            cache_path = PARAPHRASE_CACHE / f"{condition}_{c['problem_id']}.json"
            cache_path.write_text(json.dumps(result))
