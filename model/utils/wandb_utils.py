from __future__ import annotations

import os
from pathlib import Path
from typing import Any


def train_progress_bucket(completed_updates: int, total_updates: int) -> int:
    total = max(1, int(total_updates))
    completed = max(0, int(completed_updates))
    if completed <= 0:
        return 0
    return min(100, (completed * 100) // total)


def parse_tags(raw: Any) -> list[str]:
    if isinstance(raw, (list, tuple)):
        return [str(x).strip() for x in raw if str(x).strip()]
    s = str(raw).strip()
    if not s:
        return []
    return [x.strip() for x in s.split(",") if x.strip()]


def init_wandb(args: Any, root: Path):
    if not bool(getattr(args, "wandb_enabled", False)):
        return None

    try:
        import wandb as _wandb
    except Exception:
        _wandb = None
    wandb = _wandb if (_wandb is not None and hasattr(_wandb, "init")) else None
    if wandb is None:
        print("[WARN] wandb is enabled in config, but package is not installed. W&B logging disabled.")
        return None

    os.environ.setdefault("WANDB_SILENT", "true")
    os.environ.setdefault("WANDB_CONSOLE", "off")
    wandb_dir = root / "wandb"
    wandb_dir.mkdir(parents=True, exist_ok=True)
    wb_kwargs: dict[str, Any] = {
        "project": str(getattr(args, "wandb_project", "gaze_mllm")),
        "config": vars(args),
        "dir": str(wandb_dir),
        "reinit": True,
    }
    if hasattr(wandb, "Settings"):
        try:
            wb_kwargs["settings"] = wandb.Settings(quiet=True, console="off")
        except Exception:
            pass
    if str(getattr(args, "wandb_entity", "")).strip():
        wb_kwargs["entity"] = str(args.wandb_entity).strip()
    if str(getattr(args, "wandb_run_name", "")).strip():
        wb_kwargs["name"] = str(args.wandb_run_name).strip()
    tags = parse_tags(getattr(args, "wandb_tags", []))
    if tags:
        wb_kwargs["tags"] = tags
    if str(getattr(args, "wandb_notes", "")).strip():
        wb_kwargs["notes"] = str(args.wandb_notes).strip()

    try:
        run = wandb.init(**wb_kwargs)
        print(f"[INFO] wandb enabled (project={args.wandb_project}, run={run.name}).")
        try:
            wandb.define_metric("train/loss", summary="min")
            wandb.define_metric("train/loss_point", summary="min")
            wandb.define_metric("train/loss_object", summary="min")
            wandb.define_metric("train/loss_format", summary="min")
            wandb.define_metric("val/dist", summary="min")
            wandb.define_metric("val/object_acc", summary="max")
            wandb.define_metric("val/format_valid", summary="max")
            wandb.define_metric("val/point_l2_valid_frac", summary="max")
            # RL metrics
            wandb.define_metric("rl/reward_mean", summary="max")
            wandb.define_metric("rl/reward_point_mean", summary="max")
            wandb.define_metric("rl/reward_object_mean", summary="max")
            wandb.define_metric("rl/reward_joint_mean", summary="max")
            wandb.define_metric("rl/invalid_format_rate", summary="min")
            wandb.define_metric("rl/extra_text_rate", summary="min")
            wandb.define_metric("rl/kl_mean", summary="last")
            wandb.define_metric("rl/policy_loss", summary="last")
        except Exception:
            pass
        return run
    except Exception as e:
        print(f"[WARN] failed to initialize wandb: {e}. W&B logging disabled.")
        return None


def finish_wandb(run: Any) -> None:
    if run is None:
        return
    try:
        run.finish()
    except Exception:
        pass


def test_log_payload(test_metrics: dict[str, float]) -> dict[str, float]:
    payload = {
        "test/FormatValid": float(test_metrics.get("FormatValid", 0.0)),
        "test/Avg_L2": float(test_metrics.get("Avg L2", 0.0)),
        "test/Min_L2": float(test_metrics.get("Min L2", 0.0)),
        "test/ObjectAcc": float(test_metrics.get("ObjectAcc", 0.0)),
        "test/MultiAcc@1": float(test_metrics.get("MultiAcc@1", 0.0)),
        "test/ExtraTextRate": float(test_metrics.get("ExtraTextRate", 0.0)),
        "test/num_samples": float(test_metrics.get("num_samples", 0.0)),
    }
    if "DistExpected" in test_metrics:
        payload["test/DistExpected"] = float(test_metrics.get("DistExpected", 0.0))
        payload["test/Avg_L2_Expected"] = float(test_metrics.get("Avg L2 Expected", test_metrics.get("DistExpected", 0.0)))
        payload["test/Min_L2_Expected"] = float(test_metrics.get("Min L2 Expected", test_metrics.get("DistExpected", 0.0)))
    return payload


def val_metric_log_payload(val_metrics: dict[str, float]) -> dict[str, float]:
    payload = {
        "val/dist": float(val_metrics.get("Dist", 0.0)),
        "val/object_acc": float(val_metrics.get("ObjectAcc", 0.0)),
        "val/format_valid": float(val_metrics.get("FormatValid", 0.0)),
        "val/extra_text_rate": float(val_metrics.get("ExtraTextRate", 0.0)),
        "val/point_l2_valid_frac": float(val_metrics.get("PointL2ValidFrac", 0.0)),
    }
    if "DistExpected" in val_metrics:
        payload["val/dist_expected"] = float(val_metrics.get("DistExpected", 0.0))
        payload["val/expected_l2_valid_frac"] = float(val_metrics.get("ExpectedL2ValidFrac", 0.0))
    return payload
