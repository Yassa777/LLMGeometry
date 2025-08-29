"""
Interventions: parent-vector edits at the final residual (last hidden state).

Provides a simple parent-steer utility and a smoke test helper that returns the
mean absolute change in logits to verify the hook is having an effect.
"""

from __future__ import annotations

from typing import Dict, List, Optional

import torch


@torch.no_grad()
def steer_parent_vector(
    model,
    tokenizer,
    prompts: List[str],
    parent_vector: torch.Tensor,
    *,
    magnitude: float = 1.0,
    device: Optional[str] = None,
) -> Dict[str, torch.Tensor]:
    """Apply a parent-vector intervention on the last token and return logits.

    Parameters
    ----------
    model : HF CausalLM
        Pretrained language model with output_hidden_states support.
    tokenizer : HF tokenizer
        Tokenizer for the model.
    prompts : list[str]
        Prompts to evaluate.
    parent_vector : torch.Tensor
        Parent direction in residual space [d]. Must be on the same dtype as model hidden states.
    magnitude : float
        Intervention magnitude α.
    device : str, optional
        Device override. If None, inferred from the model.
    """
    mdl_dev = next(model.parameters()).device
    dev = torch.device(device) if device is not None else mdl_dev

    tok = tokenizer(prompts, return_tensors="pt", padding=True, truncation=True).to(dev)
    # Ensure outputs contain hidden states
    outputs = model(**tok, output_hidden_states=True)
    baseline_logits = outputs.logits  # [B, T, V]
    hidden = outputs.hidden_states[-1].clone()  # [B, T, d]
    last_idx = tok["attention_mask"].sum(dim=1) - 1

    pv = parent_vector.to(hidden.dtype).to(dev)
    # Add α * ℓ_p to last token only
    hidden[torch.arange(hidden.size(0)), last_idx] += magnitude * pv

    # Project to logits via lm_head or output embeddings
    if hasattr(model, "lm_head") and isinstance(model.lm_head, torch.nn.Module):
        steered_logits = model.lm_head(hidden)
    else:
        head = model.get_output_embeddings()
        steered_logits = head(hidden)

    return {
        "baseline_logits": baseline_logits.detach().cpu(),
        "steered_logits": steered_logits.detach().cpu(),
        "logit_deltas": (steered_logits - baseline_logits).detach().cpu(),
    }


def steering_smoke_mean_abs_delta(
    model,
    tokenizer,
    prompts: List[str],
    parent_vector: torch.Tensor,
    *,
    magnitude: float = 1.0,
    device: Optional[str] = None,
) -> float:
    """Return mean |Δlogits| for a one-shot parent-vector intervention.

    Values > 0 indicate the hook is wired correctly and causes changes.
    """
    out = steer_parent_vector(model, tokenizer, prompts, parent_vector, magnitude=magnitude, device=device)
    mad = torch.mean(torch.abs(out["logit_deltas"]))
    return float(mad.item())

