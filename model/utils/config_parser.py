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
    p = argparse.ArgumentParser("Qwen structured-token gaze LoRA trainer")
    p.add_argument("--config", type=str, default=str(default_value(d, "config", "sft.yaml")))
    p.add_argument("--run_name", type=str, default=normalize_run_name(default_value(d, "run_name", default_value(d, "wandb_run_name", ""))))

    # --- paths ---
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
    p.add_argument("--test_ann", type=str, default=str(default_value(d, "test_ann", "data/gazefollow_extended/test_annotations_release.txt")))
    p.add_argument("--test_image_root", type=str, default=str(default_value(d, "test_image_root", "data/gazefollow_extended/test2")))
    p.add_argument("--output_dir", type=str, default=str(default_value(d, "output_dir", "checkpoints/qwen_text_only")))
    p.add_argument("--checkpoint_dir", type=str, default=str(default_value(d, "checkpoint_dir", "")))

    # --- eval mode flags ---
    p.add_argument("--eval_only", "--eval-only", dest="eval_only", action="store_true")
    p.add_argument("--no_eval_only", "--no-eval-only", dest="eval_only", action="store_false")
    p.set_defaults(eval_only=bool(default_value(d, "eval_only", False)))
    p.add_argument("--test_only", "--test-only", dest="test_only", action="store_true")
    p.add_argument("--no_test_only", "--no-test-only", dest="test_only", action="store_false")
    p.set_defaults(test_only=bool(default_value(d, "test_only", False)))

    # --- data ---
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

    # --- val/test control ---
    p.add_argument("--run_test", dest="run_test", action="store_true")
    p.add_argument("--no_run_test", dest="run_test", action="store_false")
    p.set_defaults(run_test=bool(default_value(d, "run_test", True)))
    p.add_argument("--run_val_metrics", dest="run_val_metrics", action="store_true")
    p.add_argument("--no_run_val_metrics", dest="run_val_metrics", action="store_false")
    p.set_defaults(run_val_metrics=bool(default_value(d, "run_val_metrics", True)))
    p.add_argument("--run_val_metrics_every_n_epochs", type=int, default=int(default_value(d, "run_val_metrics_every_n_epochs", 5)))
    p.add_argument("--checkpoint_monitor", type=str, default=str(default_value(d, "checkpoint_monitor", "val_dist")))
    p.add_argument("--checkpoint_monitor_mode", type=str, default=str(default_value(d, "checkpoint_monitor_mode", "auto")))

    # --- image/data ---
    p.add_argument("--test_split_prefix", type=str, default=str(default_value(d, "test_split_prefix", "test2/")))
    p.add_argument("--test_strip_split_prefix", dest="test_strip_split_prefix", action="store_true")
    p.add_argument("--no_test_strip_split_prefix", dest="test_strip_split_prefix", action="store_false")
    p.set_defaults(test_strip_split_prefix=bool(default_value(d, "test_strip_split_prefix", True)))
    p.add_argument("--test_bbox_round_decimals", type=int, default=int(default_value(d, "test_bbox_round_decimals", 3)))
    p.add_argument("--image_cache_size", type=int, default=int(default_value(d, "image_cache_size", 1000)))
    p.add_argument("--image_resize_mode", type=str, default=str(default_value(d, "image_resize_mode", "native")))
    p.add_argument("--scene_h", type=int, default=int(default_value(d, "scene_h", 512)))
    p.add_argument("--scene_w", type=int, default=int(default_value(d, "scene_w", 512)))
    p.add_argument("--min_pixels", type=int, default=int(default_value(d, "min_pixels", 12544)))
    p.add_argument("--max_pixels", type=int, default=int(default_value(d, "max_pixels", 2007040)))

    # --- model/generation ---
    p.add_argument("--coord_bins", type=int, default=int(default_value(d, "coord_bins", 1000)))
    p.add_argument("--max_text_length", type=int, default=int(default_value(d, "max_text_length", 256)))
    p.add_argument("--generation_max_new_tokens", type=int, default=int(default_value(d, "generation_max_new_tokens", 8)))
    p.add_argument("--generation_num_beams", type=int, default=int(default_value(d, "generation_num_beams", 1)))
    p.add_argument("--repetition_penalty", type=float, default=float(default_value(d, "repetition_penalty", 1.0)))
    p.add_argument("--no_repeat_ngram_size", type=int, default=int(default_value(d, "no_repeat_ngram_size", 0)))
    p.add_argument("--preview_test_samples", type=int, default=int(default_value(d, "preview_test_samples", 0)))
    p.add_argument("--preview_val_samples", type=int, default=int(default_value(d, "preview_val_samples", 0)))
    p.add_argument("--generation_stop_at_object_end", dest="generation_stop_at_object_end", action="store_true")
    p.add_argument("--no_generation_stop_at_object_end", dest="generation_stop_at_object_end", action="store_false")
    p.set_defaults(generation_stop_at_object_end=bool(default_value(d, "generation_stop_at_object_end", True)))

    # --- prompt ---
    p.add_argument("--prompt_template", type=str, default=str(default_value(d, "prompt_template", "")))
    p.add_argument("--prompt_text", type=str, default=str(default_value(d, "prompt_text", "")))
    p.add_argument("--visual_prompting", dest="visual_prompting", action="store_true")
    p.add_argument("--no_visual_prompting", dest="visual_prompting", action="store_false")
    p.set_defaults(visual_prompting=bool(default_value(d, "visual_prompting", False)))

    # --- structured token pipeline ---
    p.add_argument("--train_stage", type=str, default=str(default_value(d, "train_stage", "sft")))
    p.add_argument("--save_last_every_n_epochs", type=int, default=int(default_value(d, "save_last_every_n_epochs", 5)))
    p.add_argument("--filter_invalid_object_samples", dest="filter_invalid_object_samples", action="store_true")
    p.add_argument("--no_filter_invalid_object_samples", dest="filter_invalid_object_samples", action="store_false")
    p.set_defaults(filter_invalid_object_samples=bool(default_value(d, "filter_invalid_object_samples", True)))
    p.add_argument("--train_reasoning_dir", type=str, default=str(default_value(d, "train_reasoning_dir", "")))
    p.add_argument("--use_reasoning", dest="use_reasoning", action="store_true")
    p.add_argument("--no_use_reasoning", dest="use_reasoning", action="store_false")
    p.set_defaults(use_reasoning=bool(default_value(d, "use_reasoning", False)))

    # --- structured loss weights ---
    p.add_argument("--loss_point_weight", type=float, default=float(default_value(d, "loss_point_weight", 1.0)))
    p.add_argument("--loss_object_weight", type=float, default=float(default_value(d, "loss_object_weight", 1.0)))
    p.add_argument("--loss_format_weight", type=float, default=float(default_value(d, "loss_format_weight", 0.25)))
    p.add_argument("--loss_reasoning_weight", type=float, default=float(default_value(d, "loss_reasoning_weight", 0.3)))

    # --- RL (disabled by default) ---
    p.add_argument("--rl_enabled", dest="rl_enabled", action="store_true")
    p.add_argument("--no_rl_enabled", dest="rl_enabled", action="store_false")
    p.set_defaults(rl_enabled=bool(default_value(d, "rl_enabled", False)))
    p.add_argument("--rl_group_size", type=int, default=int(default_value(d, "rl_group_size", 4)))
    p.add_argument("--rl_clip_eps", type=float, default=float(default_value(d, "rl_clip_eps", 0.2)))
    p.add_argument("--rl_clip_ratio_low", type=float, default=float(default_value(d, "rl_clip_ratio_low", default_value(d, "rl_clip_eps", 0.2))))
    p.add_argument("--rl_clip_ratio_high", type=float, default=float(default_value(d, "rl_clip_ratio_high", default_value(d, "rl_clip_eps", 0.2))))
    p.add_argument("--rl_clip_ratio_dual", type=float, default=float(default_value(d, "rl_clip_ratio_dual", 3.0)))
    p.add_argument("--rl_kl_beta", type=float, default=float(default_value(d, "rl_kl_beta", 0.02)))
    p.add_argument("--rl_kl_type", type=str, default=str(default_value(d, "rl_kl_type", "fixed")))
    p.add_argument("--rl_kl_target", type=float, default=float(default_value(d, "rl_kl_target", 0.1)))
    p.add_argument("--rl_kl_horizon", type=float, default=float(default_value(d, "rl_kl_horizon", 10000.0)))
    p.add_argument("--rl_kl_penalty", type=str, default=str(default_value(d, "rl_kl_penalty", "low_var_kl")))
    p.add_argument("--rl_lr", type=float, default=float(default_value(d, "rl_lr", 1e-5)))
    p.add_argument("--rl_n_ppo_epochs", type=int, default=int(default_value(d, "rl_n_ppo_epochs", 1)))
    p.add_argument("--rl_logprob_micro_batch_size", type=int, default=int(default_value(d, "rl_logprob_micro_batch_size", -1)))
    p.add_argument("--reward_point_weight", type=float, default=float(default_value(d, "reward_point_weight", 1.0)))
    p.add_argument("--reward_object_weight", type=float, default=float(default_value(d, "reward_object_weight", 1.0)))
    p.add_argument("--reward_joint_bonus", type=float, default=float(default_value(d, "reward_joint_bonus", 0.25)))
    p.add_argument("--reward_extra_penalty", type=float, default=float(default_value(d, "reward_extra_penalty", 0.5)))
    p.add_argument("--reward_format_weight", type=float, default=float(default_value(d, "reward_format_weight", 0.5)))
    p.add_argument("--reward_point_beta", type=float, default=float(default_value(d, "reward_point_beta", 10.0)))
    p.add_argument("--rl_temperature", type=float, default=float(default_value(d, "rl_temperature", 0.7)))
    p.add_argument("--rl_top_p", type=float, default=float(default_value(d, "rl_top_p", 0.9)))

    # --- LoRA ---
    p.add_argument("--lora_r", type=int, default=int(default_value(d, "lora_r", 16)))
    p.add_argument("--lora_alpha", type=int, default=int(default_value(d, "lora_alpha", 32)))
    p.add_argument("--lora_dropout", type=float, default=float(default_value(d, "lora_dropout", 0.05)))
    p.add_argument("--lora_bias", type=str, default=str(default_value(d, "lora_bias", "none")))
    lora_target_default = default_value(d, "lora_target_modules", "q_proj,k_proj,v_proj,o_proj")
    if isinstance(lora_target_default, list):
        lora_target_default = ",".join(str(x) for x in lora_target_default)
    p.add_argument("--lora_target_modules", type=str, default=str(lora_target_default))
    p.add_argument("--attn_implementation", type=str, default=str(default_value(d, "attn_implementation", "sdpa")))

    # --- wandb ---
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
