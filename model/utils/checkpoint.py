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

