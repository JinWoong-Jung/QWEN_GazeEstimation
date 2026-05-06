from __future__ import annotations

from pathlib import Path
from typing import Any

import torch
from transformers import AutoModelForImageTextToText, AutoProcessor

from .common import resolve_path


ROOT = Path(__file__).resolve().parents[2]


def _download_model_to_local_dir(repo_id: str, local_dir: Path) -> str:
    try:
        from huggingface_hub import snapshot_download
    except ImportError as exc:
        raise ImportError(
            "huggingface_hub is required to download missing model paths. "
            "Install it or place the model files under the configured model_path."
        ) from exc

    local_dir = Path(local_dir)
    local_dir.parent.mkdir(parents=True, exist_ok=True)
    snapshot_download(repo_id=str(repo_id), local_dir=str(local_dir))
    return str(local_dir)


def resolve_model_source(path: str, *, default_namespace: str = "Qwen") -> str:
    raw = str(path).strip()
    if not raw:
        raise ValueError("Model path must not be empty.")

    local_path = resolve_path(raw, ROOT)
    if local_path.exists():
        return str(local_path)

    if raw.startswith("/"):
        model_name = Path(raw).name
        inferred_repo_id = f"{default_namespace}/{model_name}"
        print(
            f"[INFO] local model path not found: {raw}. "
            f"Downloading Hugging Face repo {inferred_repo_id} to {local_path}"
        )
        return _download_model_to_local_dir(inferred_repo_id, local_path)

    path_like_prefixes = ("./", "../", "model/", "models/", "checkpoints/", "data/")
    if raw.startswith(path_like_prefixes):
        model_name = Path(raw).name
        inferred_repo_id = f"{default_namespace}/{model_name}"
        print(
            f"[INFO] local model path not found: {local_path}. "
            f"Downloading Hugging Face repo {inferred_repo_id} to {local_path}"
        )
        return _download_model_to_local_dir(inferred_repo_id, local_path)

    if raw.count("/") == 1:
        model_name = raw.split("/", 1)[1]
        local_model_dir = ROOT / "model" / model_name
        if local_model_dir.exists():
            return str(local_model_dir)
        print(f"[INFO] downloading Hugging Face repo {raw} to {local_model_dir}")
        return _download_model_to_local_dir(raw, local_model_dir)

    model_name = Path(raw).name
    inferred_repo_id = f"{default_namespace}/{model_name}"
    local_model_dir = ROOT / "model" / model_name
    if local_model_dir.exists():
        return str(local_model_dir)
    print(
        f"[INFO] treating missing local model path as Hugging Face repo id "
        f"{inferred_repo_id}; downloading to {local_model_dir}"
    )
    return _download_model_to_local_dir(inferred_repo_id, local_model_dir)


def init_processor(
    *,
    model_path: str,
    checkpoint_dir: Path | None,
    min_pixels: int | None = None,
    max_pixels: int | None = None,
) -> Any:
    processor_path: str | Path = model_path
    if checkpoint_dir is not None and (checkpoint_dir / "processor").exists():
        processor_path = checkpoint_dir / "processor"
    kwargs: dict[str, Any] = {"trust_remote_code": True}
    if min_pixels is not None:
        kwargs["min_pixels"] = int(min_pixels)
    if max_pixels is not None:
        kwargs["max_pixels"] = int(max_pixels)
    return AutoProcessor.from_pretrained(str(processor_path), **kwargs)


def init_base_model(
    *,
    model_path: str,
    model_kwargs: dict[str, Any],
) -> Any:
    return AutoModelForImageTextToText.from_pretrained(str(model_path), **model_kwargs)


def enable_token_id_gradients(peft_model: Any, token_ids: list[int]) -> None:
    """Unfreeze exactly the embedding rows used by gaze special tokens.

    Safer than unfreezing [base_vocab_size:new_vocab_size]: in Qwen, the
    tokenizer length can be smaller than the model embedding matrix size, so
    newly registered special tokens may land in existing reserved rows below
    base_vocab_size. A range-based hook would silently leave those frozen.

    Uses index_select + index_copy instead of a range zero-out so only the
    exact gaze token rows receive gradient updates.
    """
    ids = sorted({int(x) for x in token_ids if int(x) >= 0})
    if not ids:
        raise ValueError("No gaze token ids provided to enable_token_id_gradients.")

    def _install_row_mask_hook(weight: torch.nn.Parameter, ids_: list[int]) -> int:
        n_rows = int(weight.shape[0])
        valid_ids = sorted({i for i in ids_ if 0 <= i < n_rows})
        if not valid_ids:
            return 0
        weight.requires_grad_(True)
        ids_cpu = torch.tensor(valid_ids, dtype=torch.long)

        def _mask_grad(grad: torch.Tensor) -> torch.Tensor:
            ids_dev = ids_cpu.to(device=grad.device)
            out = torch.zeros_like(grad)
            out.index_copy_(0, ids_dev, grad.index_select(0, ids_dev))
            return out

        weight.register_hook(_mask_grad)
        return len(valid_ids)

    input_emb = peft_model.get_input_embeddings()
    n_input = _install_row_mask_hook(input_emb.weight, ids)

    output_emb = (
        peft_model.get_output_embeddings()
        if hasattr(peft_model, "get_output_embeddings")
        else None
    )
    n_output = 0
    if output_emb is not None and hasattr(output_emb, "weight"):
        tied = (
            output_emb.weight is input_emb.weight
            or output_emb.weight.data_ptr() == input_emb.weight.data_ptr()
        )
        if not tied:
            n_output = _install_row_mask_hook(output_emb.weight, ids)

    print(f"[INFO] trainable gaze token rows installed: input={n_input}, output={n_output}")


def peft_config_has_trainable_tokens(peft_model: Any) -> bool:
    cfgs = getattr(peft_model, "peft_config", {}) or {}
    for cfg in cfgs.values():
        indices = getattr(cfg, "trainable_token_indices", None)
        if isinstance(indices, dict):
            if any(len(v) > 0 for v in indices.values()):
                return True
        elif indices:
            return True
    return False
