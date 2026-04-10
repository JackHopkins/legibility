"""Data loading and preprocessing utilities for GSM8K."""

import re
from datasets import load_dataset
from lib.config import DATASET_NAME, DATASET_SPLIT


def load_gsm8k(split: str = DATASET_SPLIT):
    """Load GSM8K dataset and add problem_id field.

    Returns a list of dicts with keys:
        problem_id, question, answer_text, gold_answer
    """
    ds = load_dataset(DATASET_NAME, split)
    examples = []
    for i, row in enumerate(ds):
        gold = extract_answer(row["answer"])
        examples.append({
            "problem_id": i,
            "question": row["question"],
            "answer_text": row["answer"],
            "gold_answer": gold,
        })
    return examples


def extract_answer(answer_text: str) -> int | None:
    """Extract the numeric answer after #### delimiter in GSM8K format.

    Returns an integer or None if extraction fails.
    """
    match = re.search(r"####\s*(-?\d[\d,]*)", answer_text)
    if match:
        return int(match.group(1).replace(",", ""))
    return None


def extract_predicted_answer(text: str) -> int | None:
    """Extract predicted numeric answer from model output.

    Tries multiple patterns:
    1. 'The answer is: <number>'
    2. Last number in the text
    """
    # Try explicit pattern first
    match = re.search(r"[Tt]he answer is:?\s*(-?\d[\d,]*)", text)
    if match:
        return int(match.group(1).replace(",", ""))

    # Fall back to last number in text
    numbers = re.findall(r"-?\d[\d,]*", text)
    if numbers:
        return int(numbers[-1].replace(",", ""))

    return None


def build_prefill_string(question: str, cot_text: str) -> str:
    """Construct the prefill string using Qwen3 chat template.

    The model's next token after this string should be the numeric answer.

    Format:
        <|im_start|>user
        {question}<|im_end|>
        <|im_start|>assistant
        {cot_text} The answer is:
    """
    return (
        f"<|im_start|>user\n"
        f"{question}<|im_end|>\n"
        f"<|im_start|>assistant\n"
        f"{cot_text} The answer is:"
    )


def build_generation_messages(question: str) -> list[dict]:
    """Build chat messages for COT generation with Qwen3.

    Uses a system prompt that instructs step-by-step reasoning
    followed by a clear numeric answer.
    """
    return [
        {
            "role": "system",
            "content": (
                "Solve step by step. After your reasoning, "
                "write 'The answer is: ' followed by the numeric answer."
            ),
        },
        {
            "role": "user",
            "content": question,
        },
    ]