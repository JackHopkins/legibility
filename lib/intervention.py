"""Residual stream intervention utilities using PyTorch hooks.

Uses raw transformers + register_forward_pre_hook for interventions.
This allows us to use model.generate() for multi-token decoding while
applying the intervention only on the prefill pass.
"""

import re
import torch
import torch.nn.functional as F
from contextlib import contextmanager

_model = None
_tokenizer = None

# Number of tokens to generate after the prefill
GENERATE_TOKENS = 10


def load_model(model_name: str, **kwargs):
    """Load a model for intervention experiments.

    Returns (model, tokenizer).
    """
    global _model, _tokenizer

    from transformers import AutoModelForCausalLM, AutoTokenizer
    _tokenizer = AutoTokenizer.from_pretrained(model_name)
    _model = AutoModelForCausalLM.from_pretrained(
        model_name, device_map="auto", torch_dtype=torch.bfloat16
    )
    _model.eval()
    print(f"Loaded {model_name} with transformers")
    return _model, _tokenizer


def get_device():
    """Get the device the model is on."""
    return next(_model.parameters()).device


def _tokenize(text: str):
    """Tokenize text and move to model device."""
    device = get_device()
    return _tokenizer(text, return_tensors="pt").input_ids.to(device)


@contextmanager
def _hook_zero_residual_once(zero_at_layer: int, input_ids: torch.Tensor):
    """Hook that replaces the residual at the last prefill position with the
    raw embedding. Fires only on the FIRST forward pass (the prefill), then
    auto-removes so subsequent generation steps are unaffected."""

    # Pre-compute the raw embedding at the last prefill position
    with torch.no_grad():
        embeddings = _model.model.embed_tokens(input_ids)
    embedding = embeddings[0, -1, :].clone()
    prefill_len = input_ids.shape[1]

    fired = [False]

    def hook_fn(module, args):
        if fired[0]:
            return args
        hidden_states = args[0]
        # Only intervene if this looks like the prefill pass
        # (sequence length matches the prefill input)
        if hidden_states.shape[1] == prefill_len:
            hidden_states = hidden_states.clone()
            hidden_states[0, -1, :] = embedding
            fired[0] = True
            return (hidden_states,) + args[1:]
        return args

    layer = _model.model.layers[zero_at_layer]
    handle = layer.register_forward_pre_hook(hook_fn)
    try:
        yield
    finally:
        handle.remove()


def generate_answer(text: str, zero_at_layer: int | None = None,
                    max_new_tokens: int = GENERATE_TOKENS):
    """Generate a short continuation after the prefill text.

    If zero_at_layer is not None, the residual at the last prefill position
    is replaced with the raw embedding at that layer during the prefill pass.

    Returns:
        dict with keys:
            generated_text: str - the generated continuation
            predicted_answer: int or None - extracted numeric answer
            first_token_logits: Tensor - logits at the first generated position
    """
    input_ids = _tokenize(text)

    with torch.no_grad():
        if zero_at_layer is not None:
            with _hook_zero_residual_once(zero_at_layer, input_ids):
                output_ids = _model.generate(
                    input_ids,
                    max_new_tokens=max_new_tokens,
                    do_sample=False,
                    pad_token_id=_tokenizer.pad_token_id or _tokenizer.eos_token_id,
                )
        else:
            output_ids = _model.generate(
                input_ids,
                max_new_tokens=max_new_tokens,
                do_sample=False,
                pad_token_id=_tokenizer.pad_token_id or _tokenizer.eos_token_id,
            )

    # Decode only the newly generated tokens
    new_ids = output_ids[0, input_ids.shape[1]:]
    generated_text = _tokenizer.decode(new_ids, skip_special_tokens=True).strip()

    # Extract numeric answer
    match = re.search(r"-?\d[\d,]*", generated_text)
    predicted_answer = int(match.group().replace(",", "")) if match else None

    # Also get logits at the first generated position for analysis
    # (single forward pass on the prefill, no generation)
    if zero_at_layer is not None:
        with _hook_zero_residual_once(zero_at_layer, input_ids):
            outputs = _model(input_ids)
    else:
        outputs = _model(input_ids)

    first_token_logits = outputs.logits[0, -1, :].float().cpu()

    return {
        "generated_text": generated_text,
        "predicted_answer": predicted_answer,
        "first_token_logits": first_token_logits,
    }


def forward_pass_logits(text: str, zero_at_layer: int | None = None):
    """Run a single forward pass, optionally with intervention.

    Returns logits at the last position (for backward compat / analysis).
    """
    input_ids = _tokenize(text)

    with torch.no_grad():
        if zero_at_layer is not None:
            with _hook_zero_residual_once(zero_at_layer, input_ids):
                outputs = _model(input_ids)
        else:
            outputs = _model(input_ids)

    return outputs.logits[0, -1, :].float().cpu()


def extract_logit_stats(logits: torch.Tensor, gold_answer: int | None, tokenizer=None):
    """Extract analysis statistics from a logit vector.

    Args:
        logits: Float tensor of shape [vocab_size].
        gold_answer: The correct integer answer (or None).
        tokenizer: Tokenizer instance. Uses global _tokenizer if None.

    Returns:
        dict with keys:
            top1_token: str
            top1_prob: float
            gold_token_rank: int or None (rank of first token of gold answer)
            logits_top10: list of [token_str, prob] pairs
    """
    if tokenizer is None:
        tokenizer = _tokenizer

    probs = F.softmax(logits, dim=-1)

    # Top-10 tokens
    top10_probs, top10_ids = torch.topk(probs, 10)
    top10 = []
    for prob, tid in zip(top10_probs.tolist(), top10_ids.tolist()):
        token_str = tokenizer.decode([tid]).strip()
        top10.append([token_str, round(prob, 6)])

    top1_token = top10[0][0]
    top1_prob = top10[0][1]

    # Gold token rank (rank of the FIRST token of the gold answer)
    gold_token_rank = None
    if gold_answer is not None:
        gold_str = str(gold_answer)
        gold_token_ids = tokenizer.encode(gold_str, add_special_tokens=False)
        if gold_token_ids:
            gold_tid = gold_token_ids[0]
            sorted_indices = torch.argsort(probs, descending=True)
            rank_positions = (sorted_indices == gold_tid).nonzero(as_tuple=True)[0]
            if len(rank_positions) > 0:
                gold_token_rank = rank_positions[0].item() + 1  # 1-indexed

    return {
        "top1_token": top1_token,
        "top1_prob": top1_prob,
        "gold_token_rank": gold_token_rank,
        "logits_top10": top10,
    }
