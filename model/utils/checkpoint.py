from __future__ import annotations

import shutil
from pathlib import Path
from typing import Any

import torch


def resolve_qwen_model(model: Any) -> Any:
    if hasattr(model, "qwen"):
        return model.qwen
    if hasattr(model, "backbone") and hasattr(model.backbone, "qwen"):
        return model.backbone.qwen
    raise AttributeError("Could not resolve qwen model from checkpoint target.")


def save_checkpoint(
    ckpt_dir: Path,
    epoch: int,
    model: Any,
    processor: Any,
    optimizer: torch.optim.Optimizer,
    scheduler: Any,
    clear_dir: bool = False,
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
    if hasattr(model, "object_projector"):
        try:
            torch.save(model.object_projector.state_dict(), ckpt_dir / "object_projector.pt")
        except Exception:
            pass

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
    loaded_any = False
    if not ckpt_dir.exists():
        return False

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
            except Exception:
                pass
            try:
                if hasattr(qwen_model, "set_adapter"):
                    qwen_model.set_adapter(adapter_name)
                loaded_any = True
            except Exception:
                pass

    trainer_state_path = ckpt_dir / "trainer_state.pt"
    if trainer_state_path.exists():
        _ = torch.load(trainer_state_path, map_location=device)
        loaded_any = True
    object_projector_path = ckpt_dir / "object_projector.pt"
    if object_projector_path.exists() and hasattr(model, "object_projector"):
        try:
            st = torch.load(object_projector_path, map_location=device)
            model.object_projector.load_state_dict(st, strict=False)
            loaded_any = True
        except Exception:
            pass
    return loaded_any
