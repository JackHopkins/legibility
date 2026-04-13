"""Data loading and answer extraction for GSM8K and extension tasks."""

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


# --- Extension 7: Task diversity loaders ---

def load_svamp(split: str = "test", max_examples: int | None = None):
    """Load SVAMP dataset (simple arithmetic word problems).

    Returns list of dicts with: problem_id, question, gold_answer, answer_text, task.
    """
    ds = load_dataset("ChilleD/SVAMP", split=split)
    examples = []
    for i, row in enumerate(ds):
        if max_examples and i >= max_examples:
            break
        gold = int(float(row["Answer"]))
        examples.append({
            "problem_id": i,
            "question": row["question_concat"],
            "answer_text": row["Equation"],
            "gold_answer": gold,
            "task": "svamp",
        })
    return examples


def load_arc_challenge(split: str = "test", max_examples: int | None = None):
    """Load ARC-Challenge dataset.

    Returns list of dicts with: problem_id, question, gold_answer, choices.
    """
    ds = load_dataset("allenai/ai2_arc", "ARC-Challenge", split=split)
    examples = []
    for i, row in enumerate(ds):
        if max_examples and i >= max_examples:
            break
        choices = row["choices"]
        choice_text = "\n".join(
            f"{label}. {text}"
            for label, text in zip(choices["label"], choices["text"])
        )
        examples.append({
            "problem_id": i,
            "question": f"{row['question']}\n\n{choice_text}",
            "gold_answer": row["answerKey"],
            "answer_text": row["answerKey"],
            "task": "arc_challenge",
        })
    return examples


def load_hellaswag(split: str = "validation", max_examples: int | None = None):
    """Load HellaSwag dataset.

    Returns list of dicts with: problem_id, question, gold_answer, choices.
    """
    ds = load_dataset("Rowan/hellaswag", split=split)
    examples = []
    for i, row in enumerate(ds):
        if max_examples and i >= max_examples:
            break
        endings = row["endings"]
        choice_text = "\n".join(
            f"{chr(65+j)}. {e}" for j, e in enumerate(endings)
        )
        examples.append({
            "problem_id": i,
            "question": f"{row['ctx']}\n\n{choice_text}",
            "gold_answer": chr(65 + int(row["label"])),
            "answer_text": chr(65 + int(row["label"])),
            "task": "hellaswag",
        })
    return examples


def extract_predicted_answer_mc(text: str) -> str | None:
    """Extract a predicted multiple-choice answer (A/B/C/D/E) from model output."""
    if not text:
        return None
    # Look for "The answer is: X"
    match = re.search(r"[Tt]he answer is:?\s*([A-E])", text)
    if match:
        return match.group(1)
    # Look for standalone letter at start or end
    match = re.search(r"\b([A-E])\b", text)
    if match:
        return match.group(1)
    return None


def extract_predicted_answer_math(text: str) -> str | None:
    """Extract a predicted answer from MATH dataset model output.

    MATH answers can be LaTeX expressions, so we extract the boxed answer
    or the text after 'The answer is:'.
    """
    if not text:
        return None
    # Pattern 1: \boxed{...}
    match = re.search(r"\\boxed\{([^}]+)\}", text)
    if match:
        return match.group(1).strip()
    # Pattern 2: "The answer is: ..."
    match = re.search(r"[Tt]he answer is:?\s*(.+?)(?:\.|$)", text)
    if match:
        return match.group(1).strip()
    return text.strip().split("\n")[-1].strip() if text.strip() else None
