"""
Activation capture utilities for geometry experiments.

Provides simple HuggingFace-based capture of final residual (last hidden state)
for a list of prompts, returning last-token activations in float32 and utilities
to save/load hierarchical activations in HDF5.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import h5py
import torch
from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer


@dataclass
class ActivationConfig:
    model_name: str
    device: str = "cuda:0"
    dtype: torch.dtype = torch.bfloat16
    max_length: int = 128


class ActivationCapture:
    def __init__(self, config: ActivationConfig):
        self.config = config
        self.device = torch.device(config.device)

        hf_config = AutoConfig.from_pretrained(config.model_name)
        hf_config.output_hidden_states = True
        self.model = AutoModelForCausalLM.from_pretrained(
            config.model_name,
            config=hf_config,
            torch_dtype=config.dtype,
            low_cpu_mem_usage=True,
            device_map={"": config.device},
        )
        self.model.eval()
        self.tokenizer = AutoTokenizer.from_pretrained(config.model_name)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

    def _final_residual(self, outputs) -> torch.Tensor:
        if hasattr(outputs, "hidden_states") and outputs.hidden_states:
            return outputs.hidden_states[-1]
        if hasattr(outputs, "last_hidden_state"):
            return outputs.last_hidden_state
        raise RuntimeError("Could not retrieve final residual/hidden state from model outputs")

    def capture_last_token_activations(self, prompts: List[str]) -> torch.Tensor:
        """Return [B, d] last-token activations in float32 (CPU)."""
        inputs = self.tokenizer(
            prompts, return_tensors="pt", padding=True, truncation=True, max_length=self.config.max_length
        ).to(self.device)
        with torch.inference_mode(), torch.autocast(device_type=self.device.type, dtype=self.config.dtype):
            outputs = self.model(**inputs, output_hidden_states=True)
            acts = self._final_residual(outputs)  # [B, T, d]
        # Gather last non-pad token
        last_idx = inputs["attention_mask"].sum(dim=1) - 1
        idx = last_idx.view(-1, 1, 1).expand(-1, 1, acts.size(-1))
        last = acts.gather(dim=1, index=idx).squeeze(1)
        return last.float().cpu()

    def capture_pooled_activations(self, prompts: List[str]) -> torch.Tensor:
        """Return [B, d] pooled activations over sequence (mean over non-pad tokens)."""
        inputs = self.tokenizer(
            prompts, return_tensors="pt", padding=True, truncation=True, max_length=self.config.max_length
        ).to(self.device)
        with torch.inference_mode(), torch.autocast(device_type=self.device.type, dtype=self.config.dtype):
            outputs = self.model(**inputs, output_hidden_states=True)
            acts = self._final_residual(outputs)  # [B, T, d]
        mask = inputs["attention_mask"].to(acts.dtype)  # [B, T]
        mask3 = mask.unsqueeze(-1)  # [B, T, 1]
        summed = (acts * mask3).sum(dim=1)  # [B, d]
        counts = mask.sum(dim=1, keepdim=True).clamp_min(1.0)  # [B, 1]
        pooled = summed / counts
        return pooled.float().cpu()

    @staticmethod
    def save_hierarchical_activations(
        activations: Dict[str, Dict[str, torch.Tensor]], filepath: str
    ) -> None:
        """Save hierarchical activations {concept_id: {pos, neg}} to HDF5 in fp32."""
        with h5py.File(filepath, "w") as f:
            for cid, d in activations.items():
                grp = f.create_group(cid)
                for split in ("pos", "neg"):
                    if split in d and isinstance(d[split], torch.Tensor):
                        grp.create_dataset(split, data=d[split].to(torch.float32).cpu().numpy())

    @staticmethod
    def load_hierarchical_activations(filepath: str) -> Dict[str, Dict[str, torch.Tensor]]:
        out: Dict[str, Dict[str, torch.Tensor]] = {}
        with h5py.File(filepath, "r") as f:
            for cid in f.keys():
                grp = f[cid]
                out[cid] = {
                    k: torch.from_numpy(grp[k][...]) for k in grp.keys() if k in ("pos", "neg")
                }
        return out
