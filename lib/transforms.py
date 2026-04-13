"""COT transformations: shuffling and number corruption."""

import re
import random
import json
from pathlib import Path
from lib.config import PARAPHRASE_CACHE


def shuffle_steps(cot_text: str, seed: int) -> str:
    """Shuffle the reasoning steps in a COT.

    Splits on newlines (filtering blanks), shuffles with deterministic seed,
    and rejoins.
    """
    lines = [line.strip() for line in cot_text.split("\n") if line.strip()]
    if len(lines) <= 1:
        return cot_text

    rng = random.Random(seed)
    rng.shuffle(lines)
    return "\n".join(lines)


def corrupt_numbers(cot_text: str, seed: int) -> str:
    """Replace all intermediate numbers with random values.

    Preserves the final answer (last number in the text).
    Uses deterministic seed for reproducibility.
    """
    rng = random.Random(seed)

    # Find all numbers
    pattern = r'-?\d[\d,]*\.?\d*'
    matches = list(re.finditer(pattern, cot_text))

    if len(matches) <= 1:
        return cot_text

    # The last number is likely the final answer — preserve it
    result = []
    last_end = 0
    for match in matches[:-1]:
        result.append(cot_text[last_end:match.start()])
        original = match.group()
        # Generate a random replacement of similar magnitude
        try:
            orig_val = float(original.replace(",", ""))
            if orig_val == 0:
                replacement = str(rng.randint(1, 100))
            elif "." in original:
                replacement = f"{rng.uniform(0.1, abs(orig_val) * 2):.2f}"
            else:
                magnitude = max(1, abs(int(orig_val)))
                replacement = str(rng.randint(1, magnitude * 2))
        except ValueError:
            replacement = original
        result.append(replacement)
        last_end = match.end()

    result.append(cot_text[last_end:])
    return "".join(result)


def generate_transforms(cots: list):
    """Generate shuffled and corrupted COT variants, caching to disk.

    Args:
        cots: List of dicts with 'problem_id' and 'cot_text'.
    """
    PARAPHRASE_CACHE.mkdir(parents=True, exist_ok=True)

    for c in cots:
        pid = c["problem_id"]

        # Shuffled steps
        shuffle_path = PARAPHRASE_CACHE / f"shuffle_{pid}.json"
        if not shuffle_path.exists():
            shuffled = shuffle_steps(c["cot_text"], seed=pid)
            result = {
                "problem_id": pid,
                "condition": "shuffle",
                "original_cot": c["cot_text"],
                "paraphrased_cot": shuffled,
            }
            shuffle_path.write_text(json.dumps(result))

        # Corrupted numbers
        corrupt_path = PARAPHRASE_CACHE / f"corrupt_numbers_{pid}.json"
        if not corrupt_path.exists():
            corrupted = corrupt_numbers(c["cot_text"], seed=pid)
            result = {
                "problem_id": pid,
                "condition": "corrupt_numbers",
                "original_cot": c["cot_text"],
                "paraphrased_cot": corrupted,
            }
            corrupt_path.write_text(json.dumps(result))
