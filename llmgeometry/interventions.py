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


@torch.no_grad()
def steer_at_layer(
    model,
    tokenizer,
    prompts: List[str],
    direction: torch.Tensor,
    *,
    magnitude: float = 0.5,
    layer_index: int = -1,
    device: Optional[str] = None,
):
    """Apply an edit at a specific transformer layer and return before/after logits.

    Notes:
      - Supports GPT2-like modules with attribute `transformer.h`.
      - Attempts to handle models exposing `model.layers` or `gpt_neox.layers` similarly.
      - Edits the last token only.
    """
    mdl_dev = next(model.parameters()).device
    dev = torch.device(device) if device is not None else mdl_dev

    tok = tokenizer(prompts, return_tensors="pt", padding=True, truncation=True).to(dev)

    # Baseline pass
    outputs0 = model(**tok)
    logits0 = outputs0.logits.detach().cpu()

    # Find layer module list
    modules = None
    if hasattr(model, "transformer") and hasattr(model.transformer, "h"):
        modules = list(model.transformer.h)
    elif hasattr(model, "model") and hasattr(model.model, "layers"):
        modules = list(model.model.layers)
    elif hasattr(model, "gpt_neox") and hasattr(model.gpt_neox, "layers"):
        modules = list(model.gpt_neox.layers)
    else:
        raise RuntimeError("Unsupported model architecture for steer_at_layer")

    n_layers = len(modules)
    idx = layer_index if layer_index >= 0 else (n_layers + layer_index)
    if idx < 0 or idx >= n_layers:
        raise IndexError(f"layer_index {layer_index} out of range for {n_layers} layers")
    target = modules[idx]

    last_idx = tok["attention_mask"].sum(dim=1) - 1
    vv = direction.to(dev, dtype=next(model.parameters()).dtype)
    mag = float(magnitude)

    def hook_fn(module, inputs, output):
        # output may be Tensor or tuple(Tensor, ...)
        if isinstance(output, tuple):
            h = output[0]
            rest = output[1:]
        else:
            h = output
            rest = ()
        hh = h.clone()
        hh[torch.arange(hh.size(0), device=hh.device), last_idx] = (
            hh[torch.arange(hh.size(0), device=hh.device), last_idx] + mag * vv
        )
        return (hh, *rest) if rest else hh

    handle = target.register_forward_hook(hook_fn)
    try:
        outputs1 = model(**tok)
    finally:
        handle.remove()
    logits1 = outputs1.logits.detach().cpu()
    return {"baseline_logits": logits0, "steered_logits": logits1}
