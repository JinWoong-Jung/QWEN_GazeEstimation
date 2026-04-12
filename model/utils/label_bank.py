from __future__ import annotations

"""Split-specific CLIP label embedding bank utilities.

Each split (train / val / test) may have a different label space.  This module
provides a single place to load, canonicalize, and query those banks so that
the rest of the codebase does not need to replicate the logic.

Usage
-----
bank = LabelBank.from_vocab_and_dir(
    vocab2id={"television": 0, "chair": 1, ...},
    label_embed_dir=Path("data/gazefollow/label-embeds"),
    embedding_dim=512,
)
top_ids = bank.topk(query_tensor, k=3)
label   = bank.id_to_label[top_ids[0]]
"""

import re
from dataclasses import dataclass, field
from pathlib import Path
import torch
import torch.nn.functional as F


def canonicalize(text: str) -> str:
    """Lowercase, strip, and collapse internal whitespace."""
    return re.sub(r"\s+", " ", str(text or "").strip().lower())


@dataclass
class LabelBank:
    """In-memory bank of label texts and their normalized CLIP embeddings.

    Attributes
    ----------
    label_texts:
        Ordered list of raw label strings; index == label id.
    label_to_id:
        Raw label text -> id mapping.
    canonical_to_id:
        Canonicalized label text -> id mapping (for fuzzy lookup).
    embedding_matrix:
        Float32 tensor [N, D], L2-normalized rows.
    multi_label_map:
        Optional per-sample GT id sets (used for multi-label top-k metrics).
    """

    label_texts: list[str]
    label_to_id: dict[str, int]
    canonical_to_id: dict[str, int]
    embedding_matrix: torch.Tensor  # [N, D], normalized
    multi_label_map: dict[int, list[int]] = field(default_factory=dict)

    # ------------------------------------------------------------------
    # Construction helpers
    # ------------------------------------------------------------------

    @classmethod
    def from_vocab_and_dir(
        cls,
        vocab2id: dict[str, int],
        label_embed_dir: Path | str | None,
        embedding_dim: int = 512,
        normalize: bool = True,
    ) -> "LabelBank":
        """Build a bank from a vocab2id dict and a label-embeds directory.

        The embed dir must contain either:
        - One file per label: ``{label_text}.pt`` (single [D] tensor)
        - A combined ``all.pt`` mapping label_text -> tensor

        Missing embeddings are zero-filled (the row will be masked at query
        time by the caller via ``embedding_matrix.norm > 0`` checks).
        """
        n = len(vocab2id)
        label_texts: list[str] = [""] * n
        label_to_id: dict[str, int] = {}
        canonical_to_id: dict[str, int] = {}

        for txt, idx in vocab2id.items():
            i = int(idx)
            label_texts[i] = str(txt)
            label_to_id[str(txt)] = i
            canonical_to_id[canonicalize(txt)] = i

        dim = max(1, int(embedding_dim))
        matrix = torch.zeros((n, dim), dtype=torch.float32)

        embed_dir = Path(label_embed_dir) if label_embed_dir is not None else None
        if embed_dir is not None and embed_dir.is_dir():
            combined = embed_dir / "all.pt"
            if combined.exists():
                try:
                    all_emb = torch.load(combined, map_location="cpu")
                    if isinstance(all_emb, dict):
                        for txt, emb in all_emb.items():
                            idx = label_to_id.get(str(txt), canonical_to_id.get(canonicalize(str(txt)), -1))
                            if idx < 0 or idx >= n:
                                continue
                            t = emb.float().flatten()
                            if t.numel() >= dim:
                                matrix[idx] = t[:dim]
                except Exception:
                    pass
            else:
                for txt, idx in label_to_id.items():
                    fname = embed_dir / f"{txt}.pt"
                    if not fname.exists():
                        fname = embed_dir / f"{canonicalize(txt)}.pt"
                    if fname.exists():
                        try:
                            t = torch.load(fname, map_location="cpu").float().flatten()
                            if t.numel() >= dim:
                                matrix[idx] = t[:dim]
                        except Exception:
                            pass

        if normalize:
            norms = matrix.norm(dim=1, keepdim=True).clamp_min(1e-8)
            valid = matrix.norm(dim=1) > 0
            matrix[valid] = matrix[valid] / norms[valid]

        return cls(
            label_texts=label_texts,
            label_to_id=label_to_id,
            canonical_to_id=canonical_to_id,
            embedding_matrix=matrix,
        )

    # ------------------------------------------------------------------
    # Lookup helpers
    # ------------------------------------------------------------------

    @property
    def id_to_label(self) -> list[str]:
        return self.label_texts

    def lookup_id(self, text: str) -> int:
        """Return label id for *text*, or -1 if not found."""
        raw_id = self.label_to_id.get(str(text), -1)
        if raw_id >= 0:
            return raw_id
        return self.canonical_to_id.get(canonicalize(text), -1)

    def topk(
        self,
        query: torch.Tensor,
        k: int = 1,
        temperature: float = 0.07,
        device: torch.device | None = None,
    ) -> list[int]:
        """Return top-k label ids by cosine similarity to *query*.

        Parameters
        ----------
        query:
            1-D float tensor [D].
        k:
            Number of results to return.
        temperature:
            Softmax temperature applied to similarity logits.
        device:
            Move bank to this device for computation; defaults to query device.
        """
        if not torch.is_tensor(query) or query.dim() != 1:
            return []
        dev = device if device is not None else query.device
        bank = self.embedding_matrix.to(device=dev, dtype=torch.float32)
        if int(bank.shape[0]) <= 0:
            return []
        q = F.normalize(query.unsqueeze(0).to(dev), p=2, dim=-1)
        b = F.normalize(bank, p=2, dim=-1)
        t = max(float(temperature), 1e-6)
        sim = (q @ b.t()) / t
        kk = min(max(1, int(k)), int(bank.shape[0]))
        return [int(x) for x in torch.topk(sim.squeeze(0), k=kk, largest=True, sorted=True).indices.tolist()]

    def topk_labels(
        self,
        query: torch.Tensor,
        k: int = 1,
        temperature: float = 0.07,
        device: torch.device | None = None,
    ) -> list[str]:
        """Like :meth:`topk` but returns label text strings."""
        ids = self.topk(query, k=k, temperature=temperature, device=device)
        return [self.label_texts[i] for i in ids if 0 <= i < len(self.label_texts)]

    def __len__(self) -> int:
        return len(self.label_texts)
