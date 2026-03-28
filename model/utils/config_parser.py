from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

import yaml


def _flatten_config(raw: dict[str, Any]) -> dict[str, Any]:
    flat: dict[str, Any] = {}

    def _walk_with_keys(obj: Any) -> None:
        if not isinstance(obj, dict):
            return
        for k, v in obj.items():
            key = str(k).strip()
            if not key:
                continue
            if isinstance(v, dict):
                _walk_with_keys(v)
            else:
                flat[key] = v

    _walk_with_keys(raw)
    return flat


def load_yaml_config(config_path: Path) -> dict[str, Any]:
    if not config_path.exists():
        raise FileNotFoundError(f"config file not found: {config_path}")
    with config_path.open("r", encoding="utf-8") as f:
        raw = yaml.safe_load(f) or {}
    if not isinstance(raw, dict):
        raise ValueError(f"config must be a YAML mapping object: {config_path}")
    return _flatten_config(raw)


def _default(defaults: dict[str, Any], key: str, fallback: Any) -> Any:
    return defaults.get(key, fallback)


def _parse_tags(raw: Any) -> list[str]:
    if isinstance(raw, (list, tuple)):
        return [str(x).strip() for x in raw if str(x).strip()]
    s = str(raw).strip()
    if not s:
        return []
    return [x.strip() for x in s.split(",") if x.strip()]


def build_parser(defaults: dict[str, Any] | None = None) -> argparse.ArgumentParser:
    d = defaults or {}
    p = argparse.ArgumentParser("Qwen gaze LoRA trainer")
    p.add_argument("--config", type=str, default=str(_default(d, "config", "config.yaml")))

    p.add_argument("--model_path", type=str, default=str(_default(d, "model_path", "model/Qwen3-VL-4B-Instruct")))
    p.add_argument("--image_root", type=str, default=str(_default(d, "image_root", "data/gazefollow_extended/train")))
    p.add_argument("--train_ann", type=str, default=str(_default(d, "train_ann", "data/gazefollow/train_annotations_new.txt")))
    p.add_argument("--val_ann", type=str, default=str(_default(d, "val_ann", "data/gazefollow/val_annotations_new.txt")))
    p.add_argument("--train_labels", type=str, default=str(_default(d, "train_labels", "data/gazefollow/gaze-labels-train.csv")))
    p.add_argument("--val_labels", type=str, default=str(_default(d, "val_labels", "data/gazefollow/gaze-labels-val.csv")))
    p.add_argument("--test_labels", type=str, default=str(_default(d, "test_labels", "data/gazefollow/gaze-labels-test.csv")))
    p.add_argument("--vocab2id", type=str, default=str(_default(d, "vocab2id", "data/gazefollow/vocab2id.json")))
    p.add_argument("--test_ann", type=str, default=str(_default(d, "test_ann", "data/gazefollow_extended/test_annotations_release.txt")))
    p.add_argument("--test_image_root", type=str, default=str(_default(d, "test_image_root", "data/gazefollow_extended/test2")))
    p.add_argument("--output_dir", type=str, default=str(_default(d, "output_dir", "checkpoints/gaze_lora")))
    p.add_argument("--checkpoint_dir", type=str, default=str(_default(d, "checkpoint_dir", "")))
    p.add_argument("--eval_only", dest="eval_only", action="store_true")
    p.add_argument("--no_eval_only", dest="eval_only", action="store_false")
    p.set_defaults(eval_only=bool(_default(d, "eval_only", False)))
    p.add_argument("--split_prefix", type=str, default=str(_default(d, "split_prefix", "train/")))
    p.add_argument("--strip_split_prefix", dest="strip_split_prefix", action="store_true")
    p.add_argument("--no_strip_split_prefix", dest="strip_split_prefix", action="store_false")
    p.set_defaults(strip_split_prefix=bool(_default(d, "strip_split_prefix", True)))
    p.add_argument("--max_train_samples", type=int, default=int(_default(d, "max_train_samples", 0)))
    p.add_argument("--max_val_samples", type=int, default=int(_default(d, "max_val_samples", 0)))
    p.add_argument("--max_test_samples", type=int, default=int(_default(d, "max_test_samples", 0)))
    p.add_argument("--batch_size", type=int, default=int(_default(d, "batch_size", 1)))
    p.add_argument("--test_batch_size", type=int, default=int(_default(d, "test_batch_size", _default(d, "batch_size", 1))))
    p.add_argument("--epochs", type=int, default=int(_default(d, "epochs", 1)))
    p.add_argument("--num_workers", type=int, default=int(_default(d, "num_workers", 2)))
    p.add_argument("--lr", type=float, default=float(_default(d, "lr", 1e-4)))
    p.add_argument("--weight_decay", type=float, default=float(_default(d, "weight_decay", 0.0)))
    p.add_argument("--warmup_ratio", type=float, default=float(_default(d, "warmup_ratio", 0.03)))
    p.add_argument("--grad_accum_steps", type=int, default=int(_default(d, "grad_accum_steps", 1)))
    p.add_argument("--max_grad_norm", type=float, default=float(_default(d, "max_grad_norm", 1.0)))
    p.add_argument("--seed", type=int, default=int(_default(d, "seed", 42)))
    p.add_argument("--device", type=str, default=str(_default(d, "device", "cuda")))
    p.add_argument("--dtype", type=str, default=str(_default(d, "dtype", "bfloat16")))
    p.add_argument("--gradient_checkpointing", dest="gradient_checkpointing", action="store_true")
    p.add_argument("--no_gradient_checkpointing", dest="gradient_checkpointing", action="store_false")
    p.set_defaults(gradient_checkpointing=bool(_default(d, "gradient_checkpointing", False)))
    p.add_argument("--show_tqdm", dest="show_tqdm", action="store_true")
    p.add_argument("--no_show_tqdm", dest="show_tqdm", action="store_false")
    p.set_defaults(show_tqdm=bool(_default(d, "show_tqdm", True)))
    p.add_argument("--run_test", dest="run_test", action="store_true")
    p.add_argument("--no_run_test", dest="run_test", action="store_false")
    p.set_defaults(run_test=bool(_default(d, "run_test", True)))
    p.add_argument("--test_split_prefix", type=str, default=str(_default(d, "test_split_prefix", "test2/")))
    p.add_argument("--test_strip_split_prefix", dest="test_strip_split_prefix", action="store_true")
    p.add_argument("--no_test_strip_split_prefix", dest="test_strip_split_prefix", action="store_false")
    p.set_defaults(test_strip_split_prefix=bool(_default(d, "test_strip_split_prefix", True)))
    p.add_argument("--test_bbox_round_decimals", type=int, default=int(_default(d, "test_bbox_round_decimals", 3)))
    p.add_argument("--acc_dist_threshold", type=float, default=float(_default(d, "acc_dist_threshold", 0.15)))

    p.add_argument("--scene_h", type=int, default=int(_default(d, "scene_h", 512)))
    p.add_argument("--scene_w", type=int, default=int(_default(d, "scene_w", 512)))
    p.add_argument("--head_h", type=int, default=int(_default(d, "head_h", 224)))
    p.add_argument("--head_w", type=int, default=int(_default(d, "head_w", 224)))
    p.add_argument("--heatmap_h", type=int, default=int(_default(d, "heatmap_h", 512)))
    p.add_argument("--heatmap_w", type=int, default=int(_default(d, "heatmap_w", 512)))
    p.add_argument("--heatmap_sigma", type=float, default=float(_default(d, "heatmap_sigma", 7.0)))

    p.add_argument("--head_tokens", type=int, default=int(_default(d, "head_tokens", 64)))
    p.add_argument("--text_tokens", type=int, default=int(_default(d, "text_tokens", 64)))
    p.add_argument("--max_text_length", type=int, default=int(_default(d, "max_text_length", 128)))
    p.add_argument("--conditioning_mode", type=str, default=str(_default(d, "conditioning_mode", "film")))
    p.add_argument("--pool_mode", type=str, default=str(_default(d, "pool_mode", "mean")))
    p.add_argument("--num_conditioning_heads", type=int, default=int(_default(d, "num_conditioning_heads", 8)))
    p.add_argument("--num_conditioning_layers", type=int, default=int(_default(d, "num_conditioning_layers", 1)))
    p.add_argument("--dropout", type=float, default=float(_default(d, "dropout", 0.1)))
    p.add_argument("--enable_recognition", dest="enable_recognition", action="store_true")
    p.add_argument("--no_enable_recognition", dest="enable_recognition", action="store_false")
    p.set_defaults(enable_recognition=bool(_default(d, "enable_recognition", True)))
    p.add_argument("--recognition_objective", type=str, default=str(_default(d, "recognition_objective", "infonce")))
    p.add_argument("--label_emb_dim", type=int, default=int(_default(d, "label_emb_dim", 512)))
    p.add_argument("--logit_scale_init", type=float, default=float(_default(d, "logit_scale_init", 0.07)))
    p.add_argument(
        "--label_embed_dir",
        type=str,
        default=str(_default(d, "label_embed_dir", "data/gazefollow/label-embeds")),
    )
    p.add_argument("--normalize_label_emb", dest="normalize_label_emb", action="store_true")
    p.add_argument("--no_normalize_label_emb", dest="normalize_label_emb", action="store_false")
    p.set_defaults(normalize_label_emb=bool(_default(d, "normalize_label_emb", True)))
    p.add_argument("--lambda_cls", type=float, default=float(_default(d, "lambda_cls", 1.0)))
    p.add_argument("--label_smoothing", type=float, default=float(_default(d, "label_smoothing", 0.0)))
    p.add_argument("--cls_ignore_index", type=int, default=int(_default(d, "cls_ignore_index", -100)))
    p.add_argument("--prompt_template", type=str, default=str(_default(d, "prompt_template", "")))
    p.add_argument("--prompt_text", type=str, default=str(_default(d, "prompt_text", "")))
    p.add_argument("--head_text", type=str, default=str(_default(d, "head_text", "Target subject head crop.")))

    p.add_argument("--lora_r", type=int, default=int(_default(d, "lora_r", 16)))
    p.add_argument("--lora_alpha", type=int, default=int(_default(d, "lora_alpha", 32)))
    p.add_argument("--lora_dropout", type=float, default=float(_default(d, "lora_dropout", 0.05)))
    p.add_argument("--lora_bias", type=str, default=str(_default(d, "lora_bias", "none")))
    lora_target_default = _default(d, "lora_target_modules", "q_proj,k_proj,v_proj,o_proj")
    if isinstance(lora_target_default, list):
        lora_target_default = ",".join(str(x) for x in lora_target_default)
    p.add_argument("--lora_target_modules", type=str, default=str(lora_target_default))
    p.add_argument("--attn_implementation", type=str, default=str(_default(d, "attn_implementation", "sdpa")))

    p.add_argument("--wandb_enabled", dest="wandb_enabled", action="store_true")
    p.add_argument("--no_wandb_enabled", dest="wandb_enabled", action="store_false")
    p.set_defaults(wandb_enabled=bool(_default(d, "wandb_enabled", _default(d, "enabled", False))))
    p.add_argument("--wandb_project", type=str, default=str(_default(d, "wandb_project", _default(d, "project", "gaze_mllm"))))
    p.add_argument("--wandb_entity", type=str, default=str(_default(d, "wandb_entity", _default(d, "entity", ""))))
    p.add_argument("--wandb_run_name", type=str, default=str(_default(d, "wandb_run_name", _default(d, "run_name", ""))))
    wandb_tags_default = _default(d, "wandb_tags", _default(d, "tags", []))
    if isinstance(wandb_tags_default, (list, tuple)):
        wandb_tags_default = ",".join(str(x) for x in wandb_tags_default)
    p.add_argument("--wandb_tags", type=str, default=str(wandb_tags_default))
    p.add_argument("--wandb_notes", type=str, default=str(_default(d, "wandb_notes", _default(d, "notes", ""))))
    p.add_argument("--wandb_log_every_steps", type=int, default=int(_default(d, "wandb_log_every_steps", 1)))
    return p

