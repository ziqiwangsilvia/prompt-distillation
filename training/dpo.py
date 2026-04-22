"""DPO (Direct Preference Optimization) loss for on-policy distillation.

Memory-efficient: computes ref log-probs without grad, then runs chosen and
rejected policy passes with separate backward calls so only one graph is held
at a time. Peak memory = 1 forward pass with grad.
"""
import torch
import torch.nn.functional as F


def dpo_loss(model, prompt_tokens, chosen_tokens, rejected_tokens, beta: float = 0.1):
    """Compute DPO loss via per-pass backward. Pass the unwrapped PeftModel.

    Calls backward internally to keep peak memory at one forward pass.
    Returns scalar loss value (float) for logging.
    """
    # Phase 1: all log-probs without grad
    with torch.no_grad():
        with model.disable_adapter():
            ref_chosen_lp = _sequence_logprobs(model, prompt_tokens, chosen_tokens).item()
            ref_rejected_lp = _sequence_logprobs(model, prompt_tokens, rejected_tokens).item()
        pi_chosen_lp = _sequence_logprobs(model, prompt_tokens, chosen_tokens).item()
        pi_rejected_lp = _sequence_logprobs(model, prompt_tokens, rejected_tokens).item()

    torch.cuda.empty_cache()

    # Phase 2: analytical gradient weight
    # loss = -log σ(β * margin)
    # d loss / d log π(y_w) = -β σ(-β * margin)
    # d loss / d log π(y_l) =  β σ(-β * margin)
    margin = beta * ((pi_chosen_lp - ref_chosen_lp) - (pi_rejected_lp - ref_rejected_lp))
    loss_val = -F.logsigmoid(torch.tensor(margin)).item()
    grad_weight = beta * torch.sigmoid(torch.tensor(-margin)).item()

    # Phase 3: backward through chosen (increase its probability)
    chosen_lp = _sequence_logprobs(model, prompt_tokens, chosen_tokens)
    (-grad_weight * chosen_lp).backward()
    del chosen_lp
    torch.cuda.empty_cache()

    # Phase 4: backward through rejected (decrease its probability)
    rejected_lp = _sequence_logprobs(model, prompt_tokens, rejected_tokens)
    (grad_weight * rejected_lp).backward()
    del rejected_lp
    torch.cuda.empty_cache()

    return loss_val


def _sequence_logprobs(model, prompt_tokens, response_tokens):
    """Compute total log-probability of response tokens given prompt."""
    input_ids = torch.cat([prompt_tokens, response_tokens], dim=1)
    prompt_len = prompt_tokens.size(1)
    logits = model(input_ids[:, :-1]).logits
    resp_logits = logits[:, prompt_len - 1:]
    resp_targets = input_ids[:, prompt_len:]
    log_probs = F.log_softmax(resp_logits, dim=-1)
    token_logps = log_probs.gather(2, resp_targets.unsqueeze(2)).squeeze(2)
    return token_logps.sum(dim=1)
