"""Paraphrasing and COT manipulation utilities."""

import re
import random


def shuffle_steps(cot_text: str, seed: int) -> str:
    """Randomly shuffle the steps of a COT.

    Splits on newlines (filtering blanks), shuffles with a fixed seed,
    and rejoins.
    """
    lines = [line.strip() for line in cot_text.split("\n") if line.strip()]
    if len(lines) <= 1:
        # Try splitting on sentence boundaries if no newlines
        lines = re.split(r"(?<=[.!?])\s+", cot_text.strip())

    rng = random.Random(seed)
    rng.shuffle(lines)
    return "\n".join(lines)


def corrupt_numbers(cot_text: str, final_answer: int, seed: int) -> str:
    """Replace all intermediate numbers with random values.

    Preserves the final answer if it appears. Numbers that match the
    final answer are left intact to avoid corrupting the target.
    """
    rng = random.Random(seed)

    def replace_number(match):
        num_str = match.group(0)
        num_val = int(num_str.replace(",", ""))
        # Don't corrupt the final answer
        if num_val == final_answer:
            return num_str
        # Generate a random replacement in a similar magnitude
        magnitude = max(1, abs(num_val))
        lower = max(1, magnitude // 10)
        upper = magnitude * 10
        replacement = rng.randint(lower, upper)
        if num_val < 0:
            replacement = -replacement
        return str(replacement)

    return re.sub(r"-?\d[\d,]*", replace_number, cot_text)
