"""Prompt templates for COT generation, paraphrasing, and prefill."""

# --- COT Generation ---

COT_SYSTEM_PROMPT = (
    "Solve step by step. After your reasoning, write "
    "'The answer is: ' followed by the numeric answer."
)

NO_COT_SYSTEM_PROMPT = "Answer with just the number, no explanation."

# --- Paraphrasing ---

PARAPHRASE_LIGHT_TEMPLATE = """\
Rewrite the following mathematical reasoning in different words.
Preserve all logical steps, intermediate calculations, and the structure of the argument.
Change the phrasing, sentence structure, and word choices.
Do not add or remove any reasoning steps.
Do not change any numbers or calculations.
Output ONLY the rewritten reasoning inside <paraphrase> tags. No preamble.

Original reasoning:
{cot_text}

<paraphrase>"""

PARAPHRASE_HEAVY_TEMPLATE = """\
Extract only the key logical steps and intermediate results from the following
mathematical reasoning. Write them as a minimal, compressed sequence of calculations.
Remove all filler words, explanations, and narrative. Keep only the essential
mathematical operations and their results.
Output ONLY the compressed steps inside <paraphrase> tags. No preamble.

Original reasoning:
{cot_text}

<paraphrase>"""

# --- Prefill ---

PREFILL_TEMPLATE = """\
<|im_start|>user
{question}<|im_end|>
<|im_start|>assistant
{cot_text} The answer is:"""


def build_cot_messages(question: str) -> list[dict]:
    """Build chat messages for COT generation."""
    return [
        {"role": "system", "content": COT_SYSTEM_PROMPT},
        {"role": "user", "content": question},
    ]


def build_no_cot_messages(question: str) -> list[dict]:
    """Build chat messages for direct (no COT) answer."""
    return [
        {"role": "system", "content": NO_COT_SYSTEM_PROMPT},
        {"role": "user", "content": question},
    ]


def build_paraphrase_light_messages(cot_text: str) -> list[dict]:
    """Build chat messages for light paraphrasing."""
    return [
        {"role": "user", "content": PARAPHRASE_LIGHT_TEMPLATE.format(cot_text=cot_text)},
    ]


def build_paraphrase_heavy_messages(cot_text: str) -> list[dict]:
    """Build chat messages for heavy paraphrasing."""
    return [
        {"role": "user", "content": PARAPHRASE_HEAVY_TEMPLATE.format(cot_text=cot_text)},
    ]


def build_prefill_string(question: str, cot_text: str) -> str:
    """Build the prefill prompt string for answer extraction."""
    return PREFILL_TEMPLATE.format(question=question, cot_text=cot_text)
