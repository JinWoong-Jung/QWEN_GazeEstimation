from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

import yaml


def flatten_config(raw: dict[str, Any]) -> dict[str, Any]:
    flat: dict[str, Any] = {}

    def walk(obj: Any) -> None:
        if not isinstance(obj, dict):
            return
        for k, v in obj.items():
            key = str(k).strip()
            if not key:
                continue
            if isinstance(v, dict):
                walk(v)
            else:
                flat[key] = v

    walk(raw)
    return flat


def load_yaml_config(config_path: Path) -> dict[str, Any]:
    if not config_path.exists():
        raise FileNotFoundError(f"config file not found: {config_path}")
    with config_path.open("r", encoding="utf-8") as f:
        raw = yaml.safe_load(f) or {}
    if not isinstance(raw, dict):
        raise ValueError(f"config must be a YAML mapping object: {config_path}")
    return flatten_config(raw)


def default_value(defaults: dict[str, Any], key: str, fallback: Any) -> Any:
    return defaults.get(key, fallback)


def normalize_run_name(raw: Any) -> str:
    return str(raw or "").strip()


def output_dir_from_run_name(output_dir: str, run_name: str) -> str:
    raw_output_dir = str(output_dir or "").strip()
    name = normalize_run_name(run_name)
    if not raw_output_dir or not name:
        return raw_output_dir

    p = Path(raw_output_dir)
    if p.name == name:
        return str(p)

    parent = p.parent
    if str(parent) == ".":
        return str(p / name)
    return str(parent / name)


def build_parser(defaults: dict[str, Any] | None = None) -> argparse.ArgumentParser:
    d = defaults or {}
    p = argparse.ArgumentParser("Qwen text-generation LoRA trainer")
    p.add_argument("--config", type=str, default=str(default_value(d, "config", "config.yaml")))
    p.add_argument("--run_name", type=str, default=normalize_run_name(default_value(d, "run_name", default_value(d, "wandb_run_name", ""))))

    p.add_argument("--model_path", type=str, default=str(default_value(d, "model_path", "model/Qwen3-VL-4B-Instruct")))
    p.add_argument("--image_root", type=str, default=str(default_value(d, "image_root", "data/gazefollow_extended/train")))
    p.add_argument("--train_ann", type=str, default=str(default_value(d, "train_ann", "data/gazefollow/train_annotations_new.txt")))
    p.add_argument("--val_ann", type=str, default=str(default_value(d, "val_ann", "data/gazefollow/val_annotations_new.txt")))
    p.add_argument("--train_labels", type=str, default=str(default_value(d, "train_labels", "data/gazefollow/gaze-labels-train.csv")))
    p.add_argument("--val_labels", type=str, default=str(default_value(d, "val_labels", "data/gazefollow/gaze-labels-val.csv")))
    p.add_argument("--test_labels", type=str, default=str(default_value(d, "test_labels", "data/gazefollow/gaze-labels-test.csv")))
    p.add_argument("--labels_rgs", type=str, default=str(default_value(d, "labels_rgs", "data/gazefollow/gaze-labels-rgs.csv")))
    p.add_argument("--labels_ssa", type=str, default=str(default_value(d, "labels_ssa", "data/gazefollow/gaze-labels-ssa.csv")))
    p.add_argument("--vocab2id", type=str, default=str(default_value(d, "vocab2id", "data/gazefollow/vocab2id.json")))
    p.add_argument("--label_embed_dir", type=str, default=str(default_value(d, "label_embed_dir", "data/gazefollow/label-embeds")))
    p.add_argument("--test_ann", type=str, default=str(default_value(d, "test_ann", "data/gazefollow_extended/test_annotations_release.txt")))
    p.add_argument("--test_image_root", type=str, default=str(default_value(d, "test_image_root", "data/gazefollow_extended/test2")))
    p.add_argument("--output_dir", type=str, default=str(default_value(d, "output_dir", "checkpoints/qwen_text_only")))
    p.add_argument("--checkpoint_dir", type=str, default=str(default_value(d, "checkpoint_dir", "")))

    p.add_argument("--eval_only", dest="eval_only", action="store_true")
    p.add_argument("--no_eval_only", dest="eval_only", action="store_false")
    p.set_defaults(eval_only=bool(default_value(d, "eval_only", False)))
    p.add_argument("--test_only", dest="test_only", action="store_true")
    p.add_argument("--no_test_only", dest="test_only", action="store_false")
    p.set_defaults(test_only=bool(default_value(d, "test_only", False)))

    p.add_argument("--split_prefix", type=str, default=str(default_value(d, "split_prefix", "train/")))
    p.add_argument("--strip_split_prefix", dest="strip_split_prefix", action="store_true")
    p.add_argument("--no_strip_split_prefix", dest="strip_split_prefix", action="store_false")
    p.set_defaults(strip_split_prefix=bool(default_value(d, "strip_split_prefix", True)))
    p.add_argument("--batch_size", type=int, default=int(default_value(d, "batch_size", 1)))
    p.add_argument("--test_batch_size", type=int, default=int(default_value(d, "test_batch_size", default_value(d, "batch_size", 1))))
    p.add_argument("--epochs", type=int, default=int(default_value(d, "epochs", 1)))
    p.add_argument("--num_workers", type=int, default=int(default_value(d, "num_workers", 2)))
    p.add_argument("--lr", type=float, default=float(default_value(d, "lr", 1e-4)))
    p.add_argument("--weight_decay", type=float, default=float(default_value(d, "weight_decay", 0.0)))
    p.add_argument("--warmup_ratio", type=float, default=float(default_value(d, "warmup_ratio", 0.03)))
    p.add_argument("--grad_accum_steps", type=int, default=int(default_value(d, "grad_accum_steps", 1)))
    p.add_argument("--max_grad_norm", type=float, default=float(default_value(d, "max_grad_norm", 1.0)))
    p.add_argument("--seed", type=int, default=int(default_value(d, "seed", 42)))
    p.add_argument("--device", type=str, default=str(default_value(d, "device", "cuda")))
    p.add_argument("--dtype", type=str, default=str(default_value(d, "dtype", "bfloat16")))
    p.add_argument("--gradient_checkpointing", dest="gradient_checkpointing", action="store_true")
    p.add_argument("--no_gradient_checkpointing", dest="gradient_checkpointing", action="store_false")
    p.set_defaults(gradient_checkpointing=bool(default_value(d, "gradient_checkpointing", False)))
    p.add_argument("--show_tqdm", dest="show_tqdm", action="store_true")
    p.add_argument("--no_show_tqdm", dest="show_tqdm", action="store_false")
    p.set_defaults(show_tqdm=bool(default_value(d, "show_tqdm", True)))

    p.add_argument("--run_test", dest="run_test", action="store_true")
    p.add_argument("--no_run_test", dest="run_test", action="store_false")
    p.set_defaults(run_test=bool(default_value(d, "run_test", True)))
    p.add_argument("--run_val_metrics", dest="run_val_metrics", action="store_true")
    p.add_argument("--no_run_val_metrics", dest="run_val_metrics", action="store_false")
    p.set_defaults(run_val_metrics=bool(default_value(d, "run_val_metrics", True)))
    p.add_argument("--run_val_metrics_every_n_epochs", type=int, default=int(default_value(d, "run_val_metrics_every_n_epochs", 5)))
    p.add_argument("--checkpoint_monitor", type=str, default=str(default_value(d, "checkpoint_monitor", "val_loss")))
    p.add_argument("--checkpoint_monitor_mode", type=str, default=str(default_value(d, "checkpoint_monitor_mode", "auto")))

    p.add_argument("--test_split_prefix", type=str, default=str(default_value(d, "test_split_prefix", "test2/")))
    p.add_argument("--test_strip_split_prefix", dest="test_strip_split_prefix", action="store_true")
    p.add_argument("--no_test_strip_split_prefix", dest="test_strip_split_prefix", action="store_false")
    p.set_defaults(test_strip_split_prefix=bool(default_value(d, "test_strip_split_prefix", True)))
    p.add_argument("--test_bbox_round_decimals", type=int, default=int(default_value(d, "test_bbox_round_decimals", 3)))
    p.add_argument("--image_cache_size", type=int, default=int(default_value(d, "image_cache_size", 1000)))

    p.add_argument("--scene_h", type=int, default=int(default_value(d, "scene_h", 512)))
    p.add_argument("--scene_w", type=int, default=int(default_value(d, "scene_w", 512)))

    p.add_argument("--max_text_length", type=int, default=int(default_value(d, "max_text_length", 256)))
    p.add_argument("--generation_max_new_tokens", type=int, default=int(default_value(d, "generation_max_new_tokens", 24)))
    p.add_argument("--generation_num_beams", type=int, default=int(default_value(d, "generation_num_beams", 3)))
    p.add_argument("--repetition_penalty", type=float, default=float(default_value(d, "repetition_penalty", 1.0)))
    p.add_argument("--no_repeat_ngram_size", type=int, default=int(default_value(d, "no_repeat_ngram_size", 0)))
    p.add_argument("--preview_test_samples", type=int, default=int(default_value(d, "preview_test_samples", 0)))
    p.add_argument("--object_embedding_dim", type=int, default=int(default_value(d, "object_embedding_dim", 512)))
    p.add_argument("--test_retrieval_top_k", type=int, default=int(default_value(d, "test_retrieval_top_k", 3)))
    p.add_argument("--point_decimals", type=int, default=int(default_value(d, "point_decimals", 4)))
    p.add_argument("--loss_answer_weight", type=float, default=float(default_value(d, "loss_answer_weight", 1.0)))
    p.add_argument(
        "--object_temperature",
        type=float,
        default=float(default_value(d, "object_temperature", 0.07)),
    )
    p.add_argument(
        "--clip_model_path",
        type=str,
        default=str(default_value(d, "clip_model_path", "openai/clip-vit-base-patch32")),
    )

    p.add_argument("--prompt_template", type=str, default=str(default_value(d, "prompt_template", "")))
    p.add_argument("--prompt_text", type=str, default=str(default_value(d, "prompt_text", "")))
    p.add_argument("--visual_prompting", dest="visual_prompting", action="store_true")
    p.add_argument("--no_visual_prompting", dest="visual_prompting", action="store_false")
    p.set_defaults(visual_prompting=bool(default_value(d, "visual_prompting", False)))
    p.add_argument(
        "--answer_template",
        type=str,
        default=str(default_value(d, "answer_template", "Point: {point_x} {point_y}\nObject: {label_text}")),
    )
    p.add_argument("--fallback_target_text", type=str, default=str(default_value(d, "fallback_target_text", "unknown")))

    p.add_argument("--lora_r", type=int, default=int(default_value(d, "lora_r", 16)))
    p.add_argument("--lora_alpha", type=int, default=int(default_value(d, "lora_alpha", 32)))
    p.add_argument("--lora_dropout", type=float, default=float(default_value(d, "lora_dropout", 0.05)))
    p.add_argument("--lora_bias", type=str, default=str(default_value(d, "lora_bias", "none")))
    lora_target_default = default_value(d, "lora_target_modules", "q_proj,k_proj,v_proj,o_proj")
    if isinstance(lora_target_default, list):
        lora_target_default = ",".join(str(x) for x in lora_target_default)
    p.add_argument("--lora_target_modules", type=str, default=str(lora_target_default))
    p.add_argument("--attn_implementation", type=str, default=str(default_value(d, "attn_implementation", "sdpa")))

    p.add_argument("--wandb_enabled", dest="wandb_enabled", action="store_true")
    p.add_argument("--no_wandb_enabled", dest="wandb_enabled", action="store_false")
    p.set_defaults(wandb_enabled=bool(default_value(d, "wandb_enabled", default_value(d, "enabled", False))))
    p.add_argument("--wandb_project", type=str, default=str(default_value(d, "wandb_project", default_value(d, "project", "gaze_mllm"))))
    p.add_argument("--wandb_entity", type=str, default=str(default_value(d, "wandb_entity", default_value(d, "entity", ""))))
    p.add_argument("--wandb_run_name", type=str, default=str(default_value(d, "wandb_run_name", default_value(d, "run_name", ""))))
    wandb_tags_default = default_value(d, "wandb_tags", default_value(d, "tags", []))
    if isinstance(wandb_tags_default, (list, tuple)):
        wandb_tags_default = ",".join(str(x) for x in wandb_tags_default)
    p.add_argument("--wandb_tags", type=str, default=str(wandb_tags_default))
    p.add_argument("--wandb_notes", type=str, default=str(default_value(d, "wandb_notes", default_value(d, "notes", ""))))
    p.add_argument("--wandb_log_every_steps", type=int, default=int(default_value(d, "wandb_log_every_steps", 1)))
    return p
