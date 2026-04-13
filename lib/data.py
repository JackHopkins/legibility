"""GSM8K data loading and answer extraction."""

import re
from datasets import load_dataset
from lib.config import DATASET_NAME, DATASET_SPLIT


def load_gsm8k():
    """Load GSM8K test split with problem IDs."""
    ds = load_dataset(DATASET_NAME, "main", split=DATASET_SPLIT)
    examples = []
    for i, row in enumerate(ds):
        gold = extract_gold_answer(row["answer"])
        examples.append({
            "problem_id": i,
            "question": row["question"],
            "answer_text": row["answer"],
            "gold_answer": gold,
        })
    return examples


def extract_gold_answer(answer_text: str) -> int | None:
    """Extract the numeric answer after #### from GSM8K format."""
    match = re.search(r"####\s*(-?\d[\d,]*)", answer_text)
    if match:
        return int(match.group(1).replace(",", ""))
    return None


def extract_predicted_answer(text: str) -> int | None:
    """Extract a predicted numeric answer from model output.

    Tries several patterns:
    1. "The answer is: <number>"
    2. "The answer is <number>"
    3. "#### <number>"
    4. Last number in the text
    """
    if not text:
        return None

    # Pattern 1: "The answer is: <number>" or "The answer is <number>"
    match = re.search(r"[Tt]he answer is:?\s*(-?\d[\d,]*)", text)
    if match:
        return int(match.group(1).replace(",", ""))

    # Pattern 2: "#### <number>"
    match = re.search(r"####\s*(-?\d[\d,]*)", text)
    if match:
        return int(match.group(1).replace(",", ""))

    # Pattern 3: last number in the text
    matches = re.findall(r"(-?\d[\d,]*)", text)
    if matches:
        return int(matches[-1].replace(",", ""))

    return None
