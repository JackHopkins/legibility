"""All prompt templates — single source of truth."""

# --- COT generation (Qwen3 with thinking mode) ---
COT_SYSTEM = "Solve step by step. After your reasoning, write 'The answer is: ' followed by the numeric answer."

# --- No-COT generation ---
NO_COT_SYSTEM = "Answer with just the number, no explanation."

# --- Light paraphrase ---
LIGHT_PARAPHRASE_PROMPT = """Rewrite the following mathematical reasoning in different words.
Preserve all logical steps, intermediate calculations, and the structure of the argument.
Change the phrasing, sentence structure, and word choices.
Do not add or remove any reasoning steps.
Do not change any numbers or calculations.

Original reasoning:
{cot_text}

Rewritten reasoning:"""

# --- Heavy paraphrase ---
HEAVY_PARAPHRASE_PROMPT = """Extract only the key logical steps and intermediate results from the following
mathematical reasoning. Write them as a minimal, compressed sequence of calculations.
Remove all filler words, explanations, and narrative. Keep only the essential
mathematical operations and their results.

Original reasoning:
{cot_text}

Compressed steps:"""


def build_cot_messages(question: str):
    """Build chat messages for COT generation."""
    return [
        {"role": "system", "content": COT_SYSTEM},
        {"role": "user", "content": question},
    ]


def build_no_cot_messages(question: str):
    """Build chat messages for no-COT (direct answer) generation."""
    return [
        {"role": "system", "content": NO_COT_SYSTEM},
        {"role": "user", "content": question},
    ]


def build_paraphrase_messages(cot_text: str, heavy: bool = False):
    """Build chat messages for paraphrasing a COT."""
    template = HEAVY_PARAPHRASE_PROMPT if heavy else LIGHT_PARAPHRASE_PROMPT
    prompt = template.format(cot_text=cot_text)
    return [
        {"role": "user", "content": prompt},
    ]


def build_qwen_prefill_prompt(question: str, cot_text: str) -> str:
    """Build a prefill prompt for Qwen3 (manually formatted).

    Returns a string where the assistant has already 'written' the COT,
    and the model just needs to produce the final answer number.
    """
    return (
        f"<|im_start|>system\n{COT_SYSTEM}<|im_end|>\n"
        f"<|im_start|>user\n{question}<|im_end|>\n"
        f"<|im_start|>assistant\n{cot_text}\nThe answer is: "
    )


def build_gemma_prefill_prompt(question: str, cot_text: str) -> str:
    """Build a prefill prompt for Gemma 3 (manually formatted).

    Uses Gemma's chat template format.
    """
    return (
        f"<start_of_turn>user\n{question}<end_of_turn>\n"
        f"<start_of_turn>model\n{cot_text}\nThe answer is: "
    )


def build_prefill_prompt(question: str, cot_text: str, model_name: str) -> str:
    """Build a prefill prompt for the appropriate model."""
    if "gemma" in model_name.lower():
        return build_gemma_prefill_prompt(question, cot_text)
    else:
        return build_qwen_prefill_prompt(question, cot_text)
