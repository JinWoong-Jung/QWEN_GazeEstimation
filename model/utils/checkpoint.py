from __future__ import annotations

import shutil
from pathlib import Path
from typing import Any

import torch



def resolve_qwen_model(model: Any) -> Any:
    if hasattr(model, "qwen"):
        return model.qwen
    raise AttributeError("Could not resolve qwen model from checkpoint target.")


def get_embedding_layers(model: Any) -> tuple[Any, Any | None]:
    qwen_model = resolve_qwen_model(model)
    input_emb = qwen_model.get_input_embeddings()
    output_emb = (
        qwen_model.get_output_embeddings()
        if hasattr(qwen_model, "get_output_embeddings")
        else None
    )
    return input_emb, output_emb


def get_effective_embedding_weight(embedding: Any) -> torch.Tensor:
    """Return adapter-effective weights when PEFT trainable-token rows are active."""
    if hasattr(embedding, "get_merged_weights"):
        active_adapters = getattr(embedding, "active_adapters", None)
        if active_adapters:
            if isinstance(active_adapters, str):
                active_adapters = [active_adapters]
            return embedding.get_merged_weights(list(active_adapters))
    return embedding.weight


def save_added_token_rows(
    ckpt_dir: Path,
    model: Any,
    base_vocab_size: int,
) -> None:
    input_emb, output_emb = get_embedding_layers(model)
    input_w = get_effective_embedding_weight(input_emb).detach().cpu()
    total_vocab = int(input_w.shape[0])
    base_vocab = int(base_vocab_size)

    if total_vocab <= base_vocab:
        return

    payload: dict[str, Any] = {
        "base_vocab_size": base_vocab,
        "new_vocab_size": total_vocab,
        "input_added_rows": input_w[base_vocab:].clone(),
        "has_output": False,
        "output_added_rows": None,
    }

    if output_emb is not None and hasattr(output_emb, "weight"):
        output_w = get_effective_embedding_weight(output_emb).detach().cpu()
        same_shape = tuple(output_w.shape) == tuple(input_w.shape)
        tied = same_shape and torch.equal(output_w, input_w)
        if not tied:
            payload["has_output"] = True
            payload["output_added_rows"] = output_w[base_vocab:].clone()

    torch.save(payload, ckpt_dir / "added_token_rows.pt")


def save_token_rows(
    ckpt_dir: Path,
    model: Any,
    token_ids: list[int],
) -> None:
    """Save embedding rows for exact gaze token IDs.

    Safer than save_added_token_rows: tokens may land below base_vocab_size
    in Qwen reserved rows, which the slice-based approach misses.
    """
    input_emb, output_emb = get_embedding_layers(model)
    input_w = get_effective_embedding_weight(input_emb).detach().cpu()
    n_rows = int(input_w.shape[0])

    ids = sorted({int(i) for i in token_ids if 0 <= int(i) < n_rows})
    if not ids:
        return

    ids_t = torch.tensor(ids, dtype=torch.long)
    payload: dict[str, Any] = {
        "token_ids": ids,
        "input_rows": input_w[ids_t].clone(),
        "has_output": False,
        "output_token_ids": None,
        "output_rows": None,
    }

    if output_emb is not None and hasattr(output_emb, "weight"):
        output_w = get_effective_embedding_weight(output_emb).detach().cpu()
        tied = (
            output_emb.weight is input_emb.weight
            or output_emb.weight.data_ptr() == input_emb.weight.data_ptr()
        )
        if not tied:
            out_n = int(output_w.shape[0])
            out_ids = [i for i in ids if i < out_n]
            if out_ids:
                out_ids_t = torch.tensor(out_ids, dtype=torch.long)
                payload["has_output"] = True
                payload["output_token_ids"] = out_ids
                payload["output_rows"] = output_w[out_ids_t].clone()

    torch.save(payload, ckpt_dir / "gaze_token_rows.pt")


def load_token_rows(
    ckpt_dir: Path,
    model: Any,
    device: torch.device,
) -> bool:
    """Load embedding rows from gaze_token_rows.pt (exact token-ID format).

    Returns True on success, False if the file does not exist.
    """
    path = ckpt_dir / "gaze_token_rows.pt"
    if not path.exists():
        return False

    payload = torch.load(path, map_location=device)

    input_emb, output_emb = get_embedding_layers(model)
    input_w = input_emb.weight.data

    ids_t = torch.tensor(payload["token_ids"], dtype=torch.long, device=input_w.device)
    rows = payload["input_rows"].to(device=input_w.device, dtype=input_w.dtype)
    input_w.index_copy_(0, ids_t, rows)

    if bool(payload.get("has_output", False)) and output_emb is not None and hasattr(output_emb, "weight"):
        output_w = output_emb.weight.data
        out_ids = payload.get("output_token_ids") or payload["token_ids"]
        out_ids_t = torch.tensor(out_ids, dtype=torch.long, device=output_w.device)
        out_rows = payload["output_rows"].to(device=output_w.device, dtype=output_w.dtype)
        output_w.index_copy_(0, out_ids_t, out_rows)

    return True


def load_added_token_rows(
    ckpt_dir: Path,
    model: Any,
    device: torch.device,
) -> bool:
    path = ckpt_dir / "added_token_rows.pt"
    if not path.exists():
        return False

    payload = torch.load(path, map_location=device)
    base_vocab = int(payload["base_vocab_size"])
    new_vocab = int(payload["new_vocab_size"])
    input_rows = payload["input_added_rows"]
    output_rows = payload.get("output_added_rows", None)
    has_output = bool(payload.get("has_output", False))

    input_emb, output_emb = get_embedding_layers(model)
    input_w = input_emb.weight.data
    current_vocab = int(input_w.shape[0])

    if current_vocab != new_vocab:
        raise RuntimeError(
            f"added_token_rows vocab mismatch: current_vocab={current_vocab}, saved_vocab={new_vocab}"
        )

    input_w[base_vocab:new_vocab].copy_(
        input_rows.to(device=input_w.device, dtype=input_w.dtype)
    )

    if has_output and output_emb is not None and hasattr(output_emb, "weight"):
        output_w = output_emb.weight.data
        output_w[base_vocab:new_vocab].copy_(
            output_rows.to(device=output_w.device, dtype=output_w.dtype)
        )

    return True


def save_checkpoint(
    ckpt_dir: Path,
    epoch: int,
    model: Any,
    processor: Any,
    optimizer: torch.optim.Optimizer,
    scheduler: Any,
    clear_dir: bool = False,
    base_vocab_size: int | None = None,
    token_ids_to_save: list[int] | None = None,
) -> None:
    if clear_dir and ckpt_dir.exists():
        for p in ckpt_dir.iterdir():
            if p.is_dir():
                shutil.rmtree(p, ignore_errors=True)
            else:
                try:
                    p.unlink()
                except Exception:
                    pass
    ckpt_dir.mkdir(parents=True, exist_ok=True)

    qwen_model = resolve_qwen_model(model)
    if hasattr(qwen_model, "save_pretrained"):
        qwen_model.save_pretrained(str(ckpt_dir / "lora_adapter"))
    if hasattr(processor, "save_pretrained"):
        processor.save_pretrained(str(ckpt_dir / "processor"))

    if base_vocab_size is not None:
        save_added_token_rows(
            ckpt_dir=ckpt_dir,
            model=model,
            base_vocab_size=int(base_vocab_size),
        )
    if token_ids_to_save is not None and len(token_ids_to_save) > 0:
        save_token_rows(
            ckpt_dir=ckpt_dir,
            model=model,
            token_ids=token_ids_to_save,
        )

    torch.save(
        {
            "epoch": int(epoch),
            "optimizer": optimizer.state_dict(),
            "scheduler": scheduler.state_dict() if scheduler is not None else None,
        },
        ckpt_dir / "trainer_state.pt",
    )


def load_checkpoint_for_eval(
    ckpt_dir: Path,
    model: Any,
    device: torch.device,
) -> bool:
    if not ckpt_dir.exists():
        return False

    adapter_loaded = False
    adapter_dir = ckpt_dir / "lora_adapter"
    if adapter_dir.exists():
        qwen_model = resolve_qwen_model(model)
        if hasattr(qwen_model, "load_adapter"):
            adapter_name = "best_eval"
            try:
                qwen_model.load_adapter(
                    str(adapter_dir),
                    adapter_name=adapter_name,
                    is_trainable=False,
                )
                if hasattr(qwen_model, "set_adapter"):
                    qwen_model.set_adapter(adapter_name)
                adapter_loaded = True
                print(f"[INFO] loaded LoRA adapter from: {adapter_dir}")
            except Exception as e:
                print(f"[WARN] failed to load LoRA adapter from {adapter_dir}: {e}")
    else:
        print(f"[WARN] no lora_adapter directory found in {ckpt_dir}")

    rows_loaded = load_token_rows(ckpt_dir=ckpt_dir, model=model, device=device)
    if rows_loaded:
        print(f"[INFO] restored gaze token rows from: {ckpt_dir / 'gaze_token_rows.pt'}")
    else:
        rows_loaded = load_added_token_rows(ckpt_dir=ckpt_dir, model=model, device=device)
        if rows_loaded:
            print(f"[INFO] restored added token rows from: {ckpt_dir / 'added_token_rows.pt'}")
        elif not (ckpt_dir / "added_token_rows.pt").exists() and not (ckpt_dir / "gaze_token_rows.pt").exists():
            print(f"[WARN] no token rows file found in {ckpt_dir}; token embeddings may be random")

    if not adapter_loaded:
        print(f"[WARN] checkpoint at {ckpt_dir} did not fully load (adapter missing or failed)")

    # Return True only if the adapter — the primary component — was loaded successfully.
    return adapter_loaded
