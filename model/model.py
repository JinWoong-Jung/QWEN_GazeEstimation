from __future__ import annotations

from typing import Any

import torch
import torch.nn as nn
import torch.nn.functional as F


class QwenTextGenerationModel(nn.Module):
    """Minimal wrapper: image+prompt -> Qwen(LoRA) -> text tokens/logits."""

    def __init__(
        self,
        qwen_model: nn.Module,
        object_embedding_dim: int = 512,
    ) -> None:
        super().__init__()
        self.qwen = qwen_model
        hidden_size = self.infer_hidden_size(qwen_model)
        self.object_embedding_dim = int(max(1, int(object_embedding_dim)))
        self.object_projector = nn.Linear(int(hidden_size), int(self.object_embedding_dim))

    @staticmethod
    def infer_hidden_size(qwen_model: nn.Module) -> int:
        cfg = getattr(qwen_model, "config", None)
        hs = getattr(cfg, "hidden_size", None)
        if hs is not None:
            return int(hs)
        emb = qwen_model.get_input_embeddings() if hasattr(qwen_model, "get_input_embeddings") else None
        emb_dim = getattr(emb, "embedding_dim", None)
        if emb_dim is not None:
            return int(emb_dim)
        raise RuntimeError("Could not infer hidden size for object projector initialization.")

    @staticmethod
    def pool_object_slot_hidden(
        last_hidden: torch.Tensor,
        object_slot_mask: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if (not torch.is_tensor(last_hidden)) or last_hidden.dim() != 3:
            raise ValueError("last_hidden must be a [B, L, H] tensor.")
        if (not torch.is_tensor(object_slot_mask)) or object_slot_mask.dim() != 2:
            raise ValueError("object_slot_mask must be a [B, L] tensor.")
        m = object_slot_mask.to(device=last_hidden.device, dtype=torch.bool)
        bsz, seqlen, hdim = int(last_hidden.shape[0]), int(last_hidden.shape[1]), int(last_hidden.shape[2])
        if int(m.shape[0]) != bsz or int(m.shape[1]) != seqlen:
            raise ValueError(
                f"object_slot_mask shape mismatch: got={tuple(m.shape)} expected=({bsz}, {seqlen})"
            )
        m_f = m.to(dtype=last_hidden.dtype).unsqueeze(-1)
        denom = m_f.sum(dim=1).clamp_min(1.0)
        pooled = (last_hidden * m_f).sum(dim=1) / denom
        valid = m.any(dim=1)
        return pooled, valid

    def forward(
        self,
        *,
        joint_inputs: dict[str, Any],
        labels: torch.Tensor | None = None,
        object_slot_mask: torch.Tensor | None = None,
        output_hidden_states: bool = False,
        use_cache: bool = False,
    ) -> dict[str, Any]:
        kwargs = dict(joint_inputs)
        kwargs["return_dict"] = True
        kwargs["use_cache"] = bool(use_cache)
        need_hidden = bool(output_hidden_states) or torch.is_tensor(object_slot_mask)
        kwargs["output_hidden_states"] = bool(need_hidden)
        if labels is not None:
            kwargs["labels"] = labels
        out = self.qwen(**kwargs)
        pred_object_emb = None
        object_slot_valid = None
        if torch.is_tensor(object_slot_mask):
            hs = getattr(out, "hidden_states", None)
            if hs is None or len(hs) <= 0:
                raise RuntimeError("Model forward did not return hidden_states for object slot projection.")
            last_hidden = hs[-1]
            pooled, object_slot_valid = self.pool_object_slot_hidden(
                last_hidden=last_hidden,
                object_slot_mask=object_slot_mask,
            )
            # Project object-slot hidden state to retrieval space, then L2-normalize.
            pred_object_emb = F.normalize(self.object_projector(pooled), p=2, dim=-1)
        return {
            "loss": getattr(out, "loss", None),
            "logits": getattr(out, "logits", None),
            "pred_object_emb": pred_object_emb,
            "object_slot_valid": object_slot_valid,
        }

    def generate(
        self,
        *,
        joint_inputs: dict[str, Any],
        max_new_tokens: int = 16,
        do_sample: bool = False,
        num_beams: int = 1,
        num_return_sequences: int = 1,
        **generate_kwargs: Any,
    ) -> torch.Tensor | Any:
        kwargs = dict(joint_inputs)
        return self.qwen.generate(
            **kwargs,
            max_new_tokens=max(1, int(max_new_tokens)),
            do_sample=bool(do_sample),
            num_beams=max(1, int(num_beams)),
            num_return_sequences=max(1, int(num_return_sequences)),
            use_cache=True,
            **generate_kwargs,
        )


# Backward-compatible alias for older imports.
QwenGazeIntegratedModel = QwenTextGenerationModel
