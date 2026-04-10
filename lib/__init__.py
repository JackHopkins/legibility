from lib.config import *
from lib.data import load_gsm8k, extract_answer, build_prefill_string
from lib.intervention import (
    load_model,
    get_embedding_at_last_pos,
    forward_pass_logits,
    zeroed_residual_logits,
    self_prefill_logits,
    extract_logit_stats,
)