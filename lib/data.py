"""Data loading and preprocessing utilities for GSM8K."""

import re
from datasets import load_dataset
from lib.config import DATASET_NAME, DATASET_CONFIG, DATASET_SPLIT


def load_gsm8k(split: str = DATASET_SPLIT):
    """Load GSM8K dataset and add problem_id field.

    Returns a list of dicts with keys:
        problem_id, question, answer_text, gold_answer
    """
    ds = load_dataset(DATASET_NAME, DATASET_CONFIG, split=split)
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
    """Extract the numeric answer after #### delimiter in GSM8K format."""
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
    match = re.search(r"[Tt]he answer is:?\s*(-?\d[\d,]*)", text)
    if match:
        return int(match.group(1).replace(",", ""))

    numbers = re.findall(r"-?\d[\d,]*", text)
    if numbers:
        return int(numbers[-1].replace(",", ""))

    return None


def build_prefill_string(question: str, full_response: str) -> str:
    """Construct the prefill string using Qwen3 chat template.

    Includes the full model response (think tags + answer text) up to
    'The answer is: ', so the model's next token should be the number.
    """
    pattern = r"[Tt]he answer is:?\s*"
    matches = list(re.finditer(pattern, full_response))

    if matches:
        last_match = matches[-1]
        truncated = full_response[:last_match.end()]
    else:
        truncated = full_response.rstrip() + "\nThe answer is: "

    return (
        f"<|im_start|>user\n"
        f"{question}<|im_end|>\n"
        f"<|im_start|>assistant\n"
        f"{truncated}"
    )


def build_generation_messages(question: str) -> list[dict]:
    """Build chat messages for COT generation with Qwen3."""
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
