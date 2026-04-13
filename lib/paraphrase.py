"""Paraphrasing logic using vLLM."""

import json
import re
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


def strip_think_tags(text: str) -> str:
    """Remove <think>...</think> blocks from paraphraser output.

    Qwen3 models emit thinking blocks even when not requested.
    The content after </think> is the actual paraphrase.
    """
    # Remove everything inside <think>...</think>
    cleaned = re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL).strip()
    # If stripping removed everything, fall back to original minus the tags
    if not cleaned:
        cleaned = re.sub(r"</?think>", "", text).strip()
    return cleaned


def strip_answer_suffix(text: str) -> str:
    """Remove trailing 'The answer is: X' from paraphrased text.

    The paraphraser should compress reasoning, not state the answer.
    Leaving the answer in the text makes the prefill trivially correct.
    """
    # Remove trailing "The answer is: <number>" and anything after
    text = re.sub(r"\s*[Tt]he answer is:?\s*-?\d[\d,]*\.?\d*\s*\.?\s*$", "", text).strip()
    # Also remove trailing "Answer: <number>"
    text = re.sub(r"\s*\*?\*?[Aa]nswer:?\*?\*?\s*-?\d[\d,]*\.?\d*\s*\.?\s*$", "", text).strip()
    # Remove trailing "#### <number>"
    text = re.sub(r"\s*####\s*-?\d[\d,]*\s*$", "", text).strip()
    return text


def clean_paraphrase(text: str) -> str:
    """Apply all cleaning steps to paraphraser output."""
    text = strip_think_tags(text)
    text = strip_answer_suffix(text)
    return text


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

        # Apply chat template — explicitly disable thinking mode
        prompts = []
        for msgs in messages_batch:
            try:
                prompt = tokenizer.apply_chat_template(
                    msgs, tokenize=False, add_generation_prompt=True,
                    enable_thinking=False
                )
            except TypeError:
                # Tokenizer doesn't support enable_thinking kwarg
                prompt = tokenizer.apply_chat_template(
                    msgs, tokenize=False, add_generation_prompt=True
                )
            prompts.append(prompt)

        outputs = llm.generate(prompts, sampling)

        for c, output in zip(chunk, outputs):
            raw = output.outputs[0].text.strip()
            paraphrased = clean_paraphrase(raw)
            result = {
                "problem_id": c["problem_id"],
                "condition": condition,
                "original_cot": c["cot_text"],
                "paraphrased_cot": paraphrased,
            }
            cache_path = PARAPHRASE_CACHE / f"{condition}_{c['problem_id']}.json"
            cache_path.write_text(json.dumps(result))
