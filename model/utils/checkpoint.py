from __future__ import annotations

import shutil
from pathlib import Path
from typing import Any

import torch


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

    qwen_model = model.backbone.qwen
    if hasattr(qwen_model, "save_pretrained"):
        qwen_model.save_pretrained(str(ckpt_dir / "lora_adapter"))
    if hasattr(processor, "save_pretrained"):
        processor.save_pretrained(str(ckpt_dir / "processor"))

    aux_state = {
        "summary": model.summary.state_dict(),
        "conditioner": model.conditioner.state_dict(),
        "localizer": model.localizer.state_dict(),
        "classifier": model.classifier.state_dict() if model.classifier is not None else None,
    }
    torch.save(aux_state, ckpt_dir / "heads.pt")
    torch.save(
        {
            "epoch": epoch,
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
    qwen_model = model.backbone.qwen
    if adapter_dir.exists() and hasattr(qwen_model, "load_adapter"):
        adapter_name = "best_eval"
        try:
            qwen_model.load_adapter(
                str(adapter_dir),
                adapter_name=adapter_name,
                is_trainable=False,
            )
        except Exception:
            # Adapter may already exist (e.g., repeated reload in same process); try to switch to it.
            pass
        try:
            if hasattr(qwen_model, "set_adapter"):
                qwen_model.set_adapter(adapter_name)
            loaded_any = True
        except Exception:
            pass

    heads_path = ckpt_dir / "heads.pt"
    if heads_path.exists():
        aux_state = torch.load(heads_path, map_location=device)
        if isinstance(aux_state, dict):
            if "summary" in aux_state:
                model.summary.load_state_dict(aux_state["summary"], strict=True)
                loaded_any = True
            if "conditioner" in aux_state:
                model.conditioner.load_state_dict(aux_state["conditioner"], strict=True)
                loaded_any = True
            if "localizer" in aux_state:
                model.localizer.load_state_dict(aux_state["localizer"], strict=True)
                loaded_any = True
            if model.classifier is not None and aux_state.get("classifier") is not None:
                try:
                    model.classifier.load_state_dict(aux_state["classifier"], strict=True)
                except Exception:
                    model.classifier.load_state_dict(aux_state["classifier"], strict=False)
                loaded_any = True
    return loaded_any
