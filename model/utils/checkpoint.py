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


def save_added_token_rows(
    ckpt_dir: Path,
    model: Any,
    base_vocab_size: int,
) -> None:
    input_emb, output_emb = get_embedding_layers(model)
    input_w = input_emb.weight.detach().cpu()
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
        output_w = output_emb.weight.detach().cpu()
        same_shape = tuple(output_w.shape) == tuple(input_w.shape)
        tied = same_shape and torch.equal(output_w, input_w)
        if not tied:
            payload["has_output"] = True
            payload["output_added_rows"] = output_w[base_vocab:].clone()

    torch.save(payload, ckpt_dir / "added_token_rows.pt")


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

    rows_loaded = load_added_token_rows(
        ckpt_dir=ckpt_dir,
        model=model,
        device=device,
    )
    if rows_loaded:
        print(f"[INFO] restored added token rows from: {ckpt_dir / 'added_token_rows.pt'}")
    elif (ckpt_dir / "added_token_rows.pt").exists() is False:
        print(f"[WARN] added_token_rows.pt not found in {ckpt_dir}; token embeddings may be random")

    if not adapter_loaded:
        print(f"[WARN] checkpoint at {ckpt_dir} did not fully load (adapter missing or failed)")

    # Return True only if the adapter — the primary component — was loaded successfully.
    return adapter_loaded
