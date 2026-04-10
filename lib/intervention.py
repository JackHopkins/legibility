"""Residual stream intervention utilities using nnsight with PyTorch hook fallback."""

import torch
import torch.nn.functional as F
from contextlib import contextmanager

_model = None
_tokenizer = None
_use_nnsight = True


def load_model(model_name: str, use_nnsight: bool = True):
    """Load a model for intervention experiments.

    Tries nnsight first; falls back to raw transformers if nnsight fails.
    Returns (model, tokenizer).
    """
    global _model, _tokenizer, _use_nnsight
    _use_nnsight = use_nnsight

    if use_nnsight:
        try:
            from nnsight import LanguageModel
            _model = LanguageModel(model_name, device_map="auto", dispatch=True, dtype=torch.bfloat16)
            _tokenizer = _model.tokenizer
            print(f"Loaded {model_name} with nnsight")
            return _model, _tokenizer
        except Exception as e:
            print(f"nnsight failed: {e}")
            print("Falling back to raw transformers + hooks")
            _use_nnsight = False

    from transformers import AutoModelForCausalLM, AutoTokenizer
    _tokenizer = AutoTokenizer.from_pretrained(model_name)
    _model = AutoModelForCausalLM.from_pretrained(
        model_name, device_map="auto", torch_dtype=torch.bfloat16
    )
    _model.eval()
    print(f"Loaded {model_name} with transformers (hook mode)")
    return _model, _tokenizer


def get_device():
    """Get the device the model is on."""
    if _use_nnsight:
        return _model.device
    return next(_model.parameters()).device


def _tokenize(text: str):
    """Tokenize text and move to model device."""
    device = get_device()
    return _tokenizer(text, return_tensors="pt").input_ids.to(device)


# ---------------------------------------------------------------------------
# nnsight implementation
# ---------------------------------------------------------------------------

def _nnsight_forward(input_ids, zero_at_layer: int | None = None):
    """Run a forward pass with nnsight, optionally zeroing the residual at a layer.

    Returns logits tensor at the last position (on CPU).
    """
    with _model.trace(input_ids):
        if zero_at_layer is not None:
            # Grab raw embedding at last position
            embedding = _model.model.embed_tokens.output[0, -1, :].clone().save()
            # Replace residual entering target layer at last position
            _model.model.layers[zero_at_layer].input[0][0][-1, :] = embedding

        logits = _model.lm_head.output[0, -1, :].save()

    return logits.float().cpu()


# ---------------------------------------------------------------------------
# PyTorch hook fallback implementation
# ---------------------------------------------------------------------------

@contextmanager
def _hook_zero_residual(zero_at_layer: int, embedding_cache: dict):
    """Context manager that installs a forward pre-hook to zero the residual
    at a specific layer's last position, replacing it with the cached embedding."""

    def hook_fn(module, args):
        hidden_states = args[0]
        # Replace last position with cached embedding
        hidden_states[0, -1, :] = embedding_cache["embedding"]
        return (hidden_states,) + args[1:]

    layer = _model.model.layers[zero_at_layer]
    handle = layer.register_forward_pre_hook(hook_fn)
    try:
        yield
    finally:
        handle.remove()


def _hook_capture_embedding(input_ids):
    """Capture the raw token embedding at the last position."""
    with torch.no_grad():
        embeddings = _model.model.embed_tokens(input_ids)
    return embeddings[0, -1, :].clone()


def _hooks_forward(input_ids, zero_at_layer: int | None = None):
    """Run forward pass using PyTorch hooks for intervention.

    Returns logits tensor at the last position (on CPU).
    """
    with torch.no_grad():
        if zero_at_layer is not None:
            embedding = _hook_capture_embedding(input_ids)
            embedding_cache = {"embedding": embedding}
            with _hook_zero_residual(zero_at_layer, embedding_cache):
                outputs = _model(input_ids)
        else:
            outputs = _model(input_ids)

    return outputs.logits[0, -1, :].float().cpu()


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def forward_pass_logits(text: str, zero_at_layer: int | None = None):
    """Run a forward pass on text, optionally zeroing residual at a layer.

    Args:
        text: The full prefill string to run through the model.
        zero_at_layer: If not None, replace the residual entering this layer
                       at the last token position with the raw embedding.

    Returns:
        logits: Float tensor of shape [vocab_size] at the last position.
    """
    input_ids = _tokenize(text)

    if _use_nnsight:
        return _nnsight_forward(input_ids, zero_at_layer)
    else:
        return _hooks_forward(input_ids, zero_at_layer)


def zeroed_residual_logits(text: str, zero_at_layer: int):
    """Run forward pass with residual zeroing at the specified layer.

    Convenience wrapper around forward_pass_logits.
    """
    return forward_pass_logits(text, zero_at_layer=zero_at_layer)


def self_prefill_logits(text: str):
    """Run a clean forward pass on the prefill text (no intervention).

    This is the self_prefill condition: re-tokenize COT text, fresh forward
    pass, sample answer. Should match normal generation.
    """
    return forward_pass_logits(text, zero_at_layer=None)


def get_embedding_at_last_pos(text: str):
    """Get the raw token embedding at the last position of the input.

    Useful for debugging and verifying intervention mechanics.
    """
    input_ids = _tokenize(text)
    if _use_nnsight:
        with _model.trace(input_ids):
            emb = _model.model.embed_tokens.output[0, -1, :].save()
        return emb.float().cpu()
    else:
        return _hook_capture_embedding(input_ids).float().cpu()


def extract_logit_stats(logits: torch.Tensor, gold_answer: int | None, tokenizer=None):
    """Extract analysis statistics from a logit vector.

    Args:
        logits: Float tensor of shape [vocab_size].
        gold_answer: The correct integer answer (or None).
        tokenizer: Tokenizer instance. Uses global _tokenizer if None.

    Returns:
        dict with keys:
            predicted_answer: int or None
            top1_token: str
            top1_prob: float
            gold_token_rank: int or None
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

    # Try to parse predicted answer from top-1 token
    import re
    match = re.search(r"-?\d[\d,]*", top1_token)
    predicted_answer = int(match.group().replace(",", "")) if match else None

    # Gold token rank
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
        "predicted_answer": predicted_answer,
        "top1_token": top1_token,
        "top1_prob": top1_prob,
        "gold_token_rank": gold_token_rank,
        "logits_top10": top10,
    }