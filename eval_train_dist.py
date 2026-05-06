#!/usr/bin/env python3
"""
Evaluate a trained SFT checkpoint on the GazeFollow train split.

Computes per-sample dist (min_l2) and object accuracy, saves all results to JSON,
and produces a dist histogram with Q1/Q3 statistics.

Usage:
    cd QWEN_GazeEstimation
    python eval_train_dist.py \\
        --config RL.yaml \\
        --checkpoint_dir checkpoints/baseline/best \\
        [--output_dir eval_train_dist_output] \\
        [--max_samples 10000] \\
        [--batch_size 4]
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Any

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from peft import PeftModel
from torch.utils.data import DataLoader
from tqdm.auto import tqdm
from transformers import AutoModelForImageTextToText, AutoProcessor

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

from model.datasets import GazeDataset
from model.model import QwenTextGenerationModel
from model.utils.checkpoint import load_added_token_rows, load_token_rows
from model.utils.common import to_device
from model.utils.config_parser import build_parser, load_yaml_config
from model.utils.data_utils import load_label_map, load_label_text_map, load_records, load_vocab2id
from model.utils.eval_utils import (
    decode_generated,
    l2_stats,
    make_gaze_obj_end_stopping_criteria,
)
from model.utils.gaze_tokens import parse_structured_output_text, register_gaze_special_tokens
from model.utils.processor_collate import QwenTestCollator


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def resolve_path(p: str) -> Path:
    path = Path(p)
    return path if path.is_absolute() else ROOT / path


def build_id2label(vocab2id: dict[str, int]) -> dict[int, str]:
    out: dict[int, str] = {}
    for text, idx in vocab2id.items():
        if int(idx) not in out:
            out[int(idx)] = str(text)
    return out


def parse_dtype(dtype: str) -> torch.dtype | str:
    v = str(dtype).strip().lower()
    if v in {"bf16", "bfloat16"}:
        return torch.bfloat16
    if v in {"fp16", "float16"}:
        return torch.float16
    if v in {"fp32", "float32"}:
        return torch.float32
    return "auto"


def to_autocast_dtype(dtype: torch.dtype | str) -> torch.dtype:
    if dtype == torch.float16:
        return torch.float16
    if dtype == torch.float32:
        return torch.float32
    return torch.bfloat16


# ---------------------------------------------------------------------------
# Inference loop — saves per-sample records
# ---------------------------------------------------------------------------

def run_inference(
    *,
    model: torch.nn.Module,
    loader: DataLoader,
    device: torch.device,
    amp_dtype: torch.dtype,
    processor: Any,
    id2label: dict[int, str],
    num_classes: int,
    coord_bins: int,
    max_new_tokens: int,
    num_beams: int,
    repetition_penalty: float,
    no_repeat_ngram_size: int,
    stop_at_object_end: bool,
    show_tqdm: bool,
) -> list[dict[str, Any]]:
    model.eval()
    results: list[dict[str, Any]] = []
    beam_k = max(1, int(num_beams))
    sample_index = 0

    with torch.no_grad():
        it = tqdm(loader, desc="Inference", dynamic_ncols=True, disable=not show_tqdm)
        for batch in it:
            joint = to_device(batch["joint_inputs"], device=device)

            target_texts: list[str] = [str(x) for x in batch.get("target_text", [])]
            target_valid = batch.get("target_text_valid", None)
            target_point_valid = batch.get("target_point_valid", None)
            target_object_valid = batch.get("target_object_valid", None)
            target_object_id = batch.get("target_object_id", None)
            target_point_bin = batch.get("target_point_bin", None)
            gt_points = batch.get("gt_points", None)
            image_rels: list[str] = [str(x) for x in batch.get("image_rel", [])]
            # GazeDataset doesn't carry target_label_text; fall back to id2label below
            target_label_texts: list[str] = [str(x) for x in batch.get("target_label_text", [])]

            bsz = len(target_texts)

            if target_valid is None:
                target_valid = torch.ones(bsz, dtype=torch.float32)
            target_valid = target_valid.to(dtype=torch.float32)
            if target_point_valid is None:
                target_point_valid = target_valid
            if target_object_valid is None:
                target_object_valid = target_valid
            target_point_valid = target_point_valid.to(dtype=torch.float32)
            target_object_valid = target_object_valid.to(dtype=torch.float32)

            prompt_len = int(joint["input_ids"].shape[1])
            stopping = make_gaze_obj_end_stopping_criteria(
                processor, prompt_len, stop_at_object_end=bool(stop_at_object_end)
            )

            with torch.autocast(
                device_type=device.type, dtype=amp_dtype, enabled=(device.type == "cuda")
            ):
                gen_kwargs: dict[str, Any] = dict(
                    max_new_tokens=int(max_new_tokens),
                    do_sample=False,
                    num_beams=beam_k,
                    repetition_penalty=max(float(repetition_penalty), 1.0),
                    no_repeat_ngram_size=max(0, int(no_repeat_ngram_size)),
                )
                if stopping is not None:
                    gen_kwargs["stopping_criteria"] = stopping
                generated_ids = model.generate(joint_inputs=joint, **gen_kwargs)

            generated_ids_cpu = generated_ids.detach().cpu()
            input_ids_cpu = joint["input_ids"].detach().cpu()
            attn_cpu = joint.get("attention_mask", None)
            if torch.is_tensor(attn_cpu):
                attn_cpu = attn_cpu.detach().cpu()

            preds = decode_generated(
                processor=processor,
                generated_ids=generated_ids_cpu,
                input_ids=input_ids_cpu,
                attention_mask=attn_cpu,
                num_return_sequences=1,
            )
            preds = preds[:bsz]

            for i in range(bsz):
                pred = preds[i] if i < len(preds) else ""
                parsed = parse_structured_output_text(
                    pred, int(num_classes), coord_bins=int(coord_bins)
                )

                # --- GT points ---
                serialized_gt: list[list[float]] = []
                stats: tuple[float, float] | None = None
                if isinstance(gt_points, list) and i < len(gt_points):
                    if parsed["point_xy"] is not None:
                        stats = l2_stats(parsed["point_xy"], gt_points[i])
                    if torch.is_tensor(gt_points[i]) and gt_points[i].numel() >= 2:
                        pts = gt_points[i].detach().cpu().to(dtype=torch.float32).view(-1, 2)
                        serialized_gt = [
                            [float(pts[j, 0].item()), float(pts[j, 1].item())]
                            for j in range(int(pts.shape[0]))
                        ]

                # --- GT object ---
                gt_obj_id = -1
                if torch.is_tensor(target_object_id) and i < int(target_object_id.shape[0]):
                    gt_obj_id = int(target_object_id[i].item())

                # Resolve label text: prefer batch field, fall back to id2label
                if i < len(target_label_texts) and target_label_texts[i]:
                    label_text = target_label_texts[i]
                else:
                    label_text = id2label.get(gt_obj_id, "") if gt_obj_id >= 0 else ""

                is_object_valid = (
                    i < int(target_object_valid.numel())
                    and float(target_object_valid[i].item()) > 0.0
                )

                # --- Object correctness ---
                object_correct: bool | None = None
                if is_object_valid and gt_obj_id >= 0 and parsed["object_id"] is not None:
                    object_correct = int(parsed["object_id"]) == gt_obj_id

                # --- GT point bins ---
                gt_pt_bins: list[int] = []
                if torch.is_tensor(target_point_bin) and i < int(target_point_bin.shape[0]):
                    gt_pt_bins = [
                        int(target_point_bin[i, 0].item()),
                        int(target_point_bin[i, 1].item()),
                    ]

                results.append({
                    "sample_index": sample_index,
                    "image_rel": image_rels[i] if i < len(image_rels) else "",
                    "target": {
                        "gt_points": serialized_gt,
                        "gt_point_bins": gt_pt_bins,
                        "object_id": gt_obj_id,
                        "label_text": label_text,
                        "target_text": target_texts[i] if i < len(target_texts) else "",
                        "object_valid": bool(is_object_valid),
                    },
                    "pred": {
                        "generated_text": pred,
                        "valid_format": bool(parsed["valid_format"]),
                        "has_extra_text": bool(parsed["has_extra_text"]),
                        "point_xy": (
                            list(parsed["point_xy"]) if parsed["point_xy"] is not None else None
                        ),
                        "point_bins": (
                            list(parsed["point_bins"]) if parsed["point_bins"] is not None else None
                        ),
                        "object_id": (
                            int(parsed["object_id"]) if parsed["object_id"] is not None else None
                        ),
                        "object_unknown": bool(parsed.get("object_unknown", False)),
                    },
                    "dist": {
                        "avg_l2": float(stats[0]) if stats is not None else None,
                        "min_l2": float(stats[1]) if stats is not None else None,
                    },
                    "object_correct": object_correct,
                })
                sample_index += 1

            # tqdm postfix: running mean of min_l2
            if show_tqdm:
                batch_min_l2 = [
                    r["dist"]["min_l2"]
                    for r in results[-bsz:]
                    if r["dist"]["min_l2"] is not None
                ]
                if batch_min_l2:
                    it.set_postfix(min_l2=f"{sum(batch_min_l2)/len(batch_min_l2):.4f}")

    return results


# ---------------------------------------------------------------------------
# Statistics
# ---------------------------------------------------------------------------

def compute_stats(results: list[dict[str, Any]]) -> dict[str, Any]:
    min_l2_vals = [
        r["dist"]["min_l2"] for r in results if r["dist"]["min_l2"] is not None
    ]
    avg_l2_vals = [
        r["dist"]["avg_l2"] for r in results if r["dist"]["avg_l2"] is not None
    ]
    obj_correct_list = [
        r["object_correct"] for r in results if r["object_correct"] is not None
    ]
    format_valid_count = sum(1 for r in results if r["pred"]["valid_format"])

    stats: dict[str, Any] = {
        "total_samples": len(results),
        "valid_point_samples": len(min_l2_vals),
        "valid_object_samples": len(obj_correct_list),
        "format_valid_count": format_valid_count,
        "format_valid_rate": float(format_valid_count / max(len(results), 1)),
        "object_acc": (
            float(sum(1 for x in obj_correct_list if x) / len(obj_correct_list))
            if obj_correct_list else None
        ),
    }

    if min_l2_vals:
        arr = np.array(min_l2_vals, dtype=np.float64)
        q1 = float(np.percentile(arr, 25))
        q3 = float(np.percentile(arr, 75))
        stats["min_l2"] = {
            "mean":   float(arr.mean()),
            "std":    float(arr.std()),
            "min":    float(arr.min()),
            "max":    float(arr.max()),
            "median": float(np.median(arr)),
            "q1":     q1,
            "q3":     q3,
            "iqr":    float(q3 - q1),
            # Distribution by quartile
            "below_q1_count":   int((arr < q1).sum()),
            "q1_to_q3_count":   int(((arr >= q1) & (arr <= q3)).sum()),
            "above_q3_count":   int((arr > q3).sum()),
            "below_q1_frac":    float((arr < q1).mean()),
            "q1_to_q3_frac":    float(((arr >= q1) & (arr <= q3)).mean()),
            "above_q3_frac":    float((arr > q3).mean()),
            # Fixed threshold counts (good / bad 경계 확인용)
            "below_0.05_count":  int((arr < 0.05).sum()),
            "below_0.10_count":  int((arr < 0.10).sum()),
            "below_0.15_count":  int((arr < 0.15).sum()),
            "below_0.20_count":  int((arr < 0.20).sum()),
            "below_0.05_frac":   float((arr < 0.05).mean()),
            "below_0.10_frac":   float((arr < 0.10).mean()),
            "below_0.15_frac":   float((arr < 0.15).mean()),
            "below_0.20_frac":   float((arr < 0.20).mean()),
        }

    if avg_l2_vals:
        arr_avg = np.array(avg_l2_vals, dtype=np.float64)
        stats["avg_l2"] = {
            "mean":   float(arr_avg.mean()),
            "median": float(np.median(arr_avg)),
            "q1":     float(np.percentile(arr_avg, 25)),
            "q3":     float(np.percentile(arr_avg, 75)),
        }

    return stats


# ---------------------------------------------------------------------------
# Histogram
# ---------------------------------------------------------------------------

def plot_histogram(
    results: list[dict[str, Any]],
    stats: dict[str, Any],
    out_path: Path,
) -> None:
    min_l2_vals = [
        r["dist"]["min_l2"] for r in results if r["dist"]["min_l2"] is not None
    ]
    if not min_l2_vals:
        print("[WARN] No valid min_l2 values — skipping histogram.")
        return

    arr = np.array(min_l2_vals, dtype=np.float64)
    l2s = stats.get("min_l2", {})
    q1         = float(l2s.get("q1", 0.0))
    q3         = float(l2s.get("q3", 1.0))
    mean_val   = float(l2s.get("mean", 0.0))
    median_val = float(l2s.get("median", 0.0))
    obj_acc    = stats.get("object_acc")

    fig, axes = plt.subplots(1, 2, figsize=(15, 5))

    def _draw_vlines(ax: plt.Axes) -> None:
        ax.axvline(q1, color="orange", linestyle="--", lw=1.6, label=f"Q1={q1:.4f}")
        ax.axvline(q3, color="red",    linestyle="--", lw=1.6, label=f"Q3={q3:.4f}")
        ax.axvline(mean_val,   color="green",  linestyle="-",  lw=1.5, label=f"Mean={mean_val:.4f}")
        ax.axvline(median_val, color="purple", linestyle="-.", lw=1.5, label=f"Median={median_val:.4f}")

    # --- Left: full-range histogram ---
    ax0 = axes[0]
    ax0.hist(arr, bins=60, color="steelblue", edgecolor="white", linewidth=0.4, alpha=0.85)
    _draw_vlines(ax0)
    ax0.axvspan(float(arr.min()), q1, alpha=0.07, color="orange")
    ax0.axvspan(q1, q3,           alpha=0.07, color="gray")
    ax0.axvspan(q3, float(arr.max()), alpha=0.07, color="red")
    ax0.set_xlabel("min_l2 (normalised [0,1])")
    ax0.set_ylabel("Count")
    ax0.set_title(f"Train dist distribution  (n={len(arr):,})")
    ax0.legend(fontsize=9)

    # --- Right: zoomed [0, 0.4] ---
    ax1 = axes[1]
    ax1.hist(arr, bins=80, range=(0.0, 0.4),
             color="steelblue", edgecolor="white", linewidth=0.4, alpha=0.85)
    _draw_vlines(ax1)
    thresholds = [(0.05, "cyan"), (0.10, "lime"), (0.15, "gold"), (0.20, "salmon")]
    for thr, col in thresholds:
        frac = float((arr < thr).mean())
        ax1.axvline(thr, color=col, linestyle=":", lw=1.2, alpha=0.9,
                    label=f"<{thr:.2f}: {frac:.1%}")
    ax1.set_xlabel("min_l2 (normalised [0,1])")
    ax1.set_ylabel("Count")
    ax1.set_title("Zoomed: [0, 0.4]")
    ax1.legend(fontsize=8)

    # --- Annotation box below both panels ---
    below_q1  = int(l2s.get("below_q1_count", 0))
    q1_q3     = int(l2s.get("q1_to_q3_count", 0))
    above_q3  = int(l2s.get("above_q3_count", 0))
    n = len(arr)
    obj_acc_str = f"{obj_acc:.4f}" if obj_acc is not None else "N/A"
    annot = (
        f"n={n:,}   format_valid={stats['format_valid_rate']:.1%}   obj_acc={obj_acc_str}\n"
        f"< Q1 ({q1:.4f}): {below_q1:,} ({below_q1/n:.1%})  │  "
        f"Q1–Q3: {q1_q3:,} ({q1_q3/n:.1%})  │  "
        f"> Q3 ({q3:.4f}): {above_q3:,} ({above_q3/n:.1%})"
    )
    fig.text(
        0.5, -0.04, annot,
        ha="center", va="top",
        fontsize=9.5, fontfamily="monospace",
        bbox=dict(boxstyle="round,pad=0.4", facecolor="lightyellow", alpha=0.85),
    )

    plt.tight_layout()
    plt.savefig(str(out_path), dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"[INFO] Histogram saved: {out_path}")


# ---------------------------------------------------------------------------
# Summary print
# ---------------------------------------------------------------------------

def print_summary(stats: dict[str, Any]) -> None:
    sep = "=" * 58
    print(f"\n{sep}")
    print("  TRAIN SPLIT DIST EVALUATION SUMMARY")
    print(sep)
    print(f"  Total samples        : {stats['total_samples']:,}")
    print(f"  Valid point samples  : {stats['valid_point_samples']:,}")
    print(f"  Format valid         : {stats['format_valid_count']:,}  "
          f"({stats['format_valid_rate']:.1%})")
    if stats.get("object_acc") is not None:
        print(f"  Object accuracy      : {stats['object_acc']:.4f}  "
              f"({stats['valid_object_samples']:,} samples)")

    l2s = stats.get("min_l2", {})
    if l2s:
        n = stats["valid_point_samples"]
        print(f"\n  ---- min_l2 statistics ----")
        print(f"  Mean   : {l2s['mean']:.4f}")
        print(f"  Median : {l2s['median']:.4f}")
        print(f"  Std    : {l2s['std']:.4f}")
        print(f"  Min    : {l2s['min']:.4f}   Max: {l2s['max']:.4f}")

        print(f"\n  ---- Quartile distribution ----")
        bq1 = int(l2s['below_q1_count'])
        q1q3 = int(l2s['q1_to_q3_count'])
        aq3  = int(l2s['above_q3_count'])
        print(f"  Q1 = {l2s['q1']:.4f}  │  < Q1  : {bq1:>7,}  ({l2s['below_q1_frac']:.1%})")
        print(f"  Q3 = {l2s['q3']:.4f}  │  Q1–Q3 : {q1q3:>7,}  ({l2s['q1_to_q3_frac']:.1%})")
        print(f"               │  > Q3  : {aq3:>7,}  ({l2s['above_q3_frac']:.1%})")

        print(f"\n  ---- Fixed threshold breakdown ----")
        for thr in [0.05, 0.10, 0.15, 0.20]:
            cnt  = int(l2s[f"below_{thr:.2f}_count"])
            frac = float(l2s[f"below_{thr:.2f}_frac"])
            print(f"  < {thr:.2f}  :  {cnt:>7,}  ({frac:.1%})")
    print(f"{sep}\n")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    cli = argparse.ArgumentParser(
        description="Evaluate SFT checkpoint on GazeFollow train split."
    )
    cli.add_argument("--config", type=str, default="RL.yaml",
                     help="YAML config path (same as used for training)")
    cli.add_argument("--checkpoint_dir", type=str, required=True,
                     help="Path to checkpoint directory (contains lora_adapter/)")
    cli.add_argument("--output_dir", type=str, default="eval_train_dist_output",
                     help="Output directory for JSON results and histogram PNG")
    cli.add_argument("--max_samples", type=int, default=None,
                     help="Limit number of train samples evaluated (default: all ~110k)")
    cli.add_argument("--batch_size", type=int, default=None,
                     help="Override test_batch_size from config")
    cli.add_argument("--num_workers", type=int, default=None,
                     help="Override num_workers from config")
    cli.add_argument("--device", type=str, default=None,
                     help="Override device (e.g. cuda, cuda:1, cpu)")
    cli.add_argument("--model_path", type=str, default=None,
                     help="Override model_path from config (e.g. model/Qwen3-VL-2B-Instruct)")
    cli.add_argument("--no_tqdm", action="store_true",
                     help="Suppress progress bar")
    cli_args = cli.parse_args()

    # ---- Load YAML config → argparse namespace ----
    config_path = resolve_path(cli_args.config)
    cfg = load_yaml_config(config_path)
    args = build_parser(defaults=cfg).parse_args(["--config", str(config_path)])

    # CLI overrides
    if cli_args.checkpoint_dir:
        args.checkpoint_dir = cli_args.checkpoint_dir
    if cli_args.model_path is not None:
        args.model_path = cli_args.model_path
    if cli_args.batch_size is not None:
        args.test_batch_size = cli_args.batch_size
    if cli_args.num_workers is not None:
        args.num_workers = cli_args.num_workers
    if cli_args.device is not None:
        args.device = cli_args.device

    show_tqdm = not cli_args.no_tqdm
    out_dir = resolve_path(cli_args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # ---- Device / dtype ----
    device = torch.device(
        str(args.device) if torch.cuda.is_available() or str(args.device) == "cpu"
        else "cpu"
    )
    load_dtype = parse_dtype(args.dtype)
    if device.type != "cuda" and load_dtype in {torch.bfloat16, torch.float16}:
        load_dtype = torch.float32
    amp_dtype = to_autocast_dtype(load_dtype)
    print(f"[INFO] device={device}  dtype={load_dtype}  amp={amp_dtype}")

    # ---- Checkpoint dir ----
    checkpoint_dir = resolve_path(str(args.checkpoint_dir).strip())
    if not checkpoint_dir.exists():
        raise RuntimeError(f"checkpoint_dir not found: {checkpoint_dir}")
    print(f"[INFO] checkpoint_dir: {checkpoint_dir}")

    # ---- Vocab ----
    vocab2id_path = resolve_path(args.vocab2id)
    vocab2id, vocab2id_lower = load_vocab2id(vocab2id_path)
    num_classes = len(vocab2id)
    id2label = build_id2label(vocab2id)
    print(f"[INFO] vocab loaded: num_classes={num_classes}")

    coord_bins = int(args.coord_bins)

    # ---- Scene size & pixel limits ----
    _resize_mode = str(getattr(args, "image_resize_mode", "native")).strip().lower()
    _fixed = _resize_mode == "fixed"
    scene_size: tuple[int, int] | None = (
        (int(args.scene_h), int(args.scene_w)) if _fixed else None
    )
    proc_min_pixels: int | None = None if _fixed else int(getattr(args, "min_pixels", 12544))
    proc_max_pixels: int | None = None if _fixed else int(getattr(args, "max_pixels", 2007040))

    # ---- Model source path ----
    model_path_raw = str(args.model_path).strip()
    local_model = resolve_path(model_path_raw)
    model_source = str(local_model) if local_model.exists() else model_path_raw

    # ---- Processor ----
    proc_src = (
        str(checkpoint_dir / "processor")
        if (checkpoint_dir / "processor").exists()
        else model_source
    )
    proc_kwargs: dict[str, Any] = {"trust_remote_code": True}
    if proc_min_pixels is not None:
        proc_kwargs["min_pixels"] = proc_min_pixels
    if proc_max_pixels is not None:
        proc_kwargs["max_pixels"] = proc_max_pixels
    processor = AutoProcessor.from_pretrained(proc_src, **proc_kwargs)
    print(f"[INFO] processor loaded from: {proc_src}")

    # ---- Register gaze special tokens ----
    register_gaze_special_tokens(
        tokenizer=processor.tokenizer,
        num_classes=num_classes,
        coord_bins=coord_bins,
    )
    new_vocab_size = len(processor.tokenizer)
    print(f"[INFO] tokenizer vocab_size after registration: {new_vocab_size}")

    # ---- Base model → resize → LoRA ----
    model_kwargs: dict[str, Any] = {
        "trust_remote_code": True,
        "attn_implementation": str(args.attn_implementation),
    }
    if load_dtype != "auto":
        model_kwargs["dtype"] = load_dtype

    print(f"[INFO] loading base model: {model_source}")
    base_qwen = AutoModelForImageTextToText.from_pretrained(model_source, **model_kwargs)
    base_qwen.resize_token_embeddings(new_vocab_size)
    print(f"[INFO] embeddings resized to {new_vocab_size}")

    adapter_dir = checkpoint_dir / "lora_adapter"
    if adapter_dir.exists():
        qwen_model = PeftModel.from_pretrained(
            base_qwen, model_id=str(adapter_dir), is_trainable=False,
        )
        print(f"[INFO] LoRA adapter loaded: {adapter_dir}")
    else:
        qwen_model = base_qwen
        print(f"[WARN] lora_adapter not found in {checkpoint_dir}; using base model.")

    model = QwenTextGenerationModel(qwen_model=qwen_model)

    # ---- Restore gaze token embeddings ----
    rows_ok = load_token_rows(ckpt_dir=checkpoint_dir, model=model, device=device)
    if rows_ok:
        print(f"[INFO] gaze_token_rows restored.")
    else:
        rows_ok = load_added_token_rows(ckpt_dir=checkpoint_dir, model=model, device=device)
        if rows_ok:
            print(f"[INFO] added_token_rows restored.")
        else:
            print(f"[WARN] no token rows file found; special token embeddings may be random.")

    model = model.to(device)
    model.eval()

    # ---- Load train data ----
    train_ann    = resolve_path(args.train_ann)
    image_root   = resolve_path(args.image_root)
    train_labels = resolve_path(args.train_labels)

    train_label_map, label_stats = load_label_map(
        train_labels,
        vocab2id=vocab2id,
        vocab2id_lower=vocab2id_lower,
        text_key="gaze_pseudo_label",
        use_embed_fallback=False,
    )
    train_label_text_map, _ = load_label_text_map(
        train_labels, text_key="gaze_pseudo_label",
    )
    print(
        f"[INFO] label_map: rows={label_stats['rows']} "
        f"mapped={label_stats['mapped']} "
        f"missing={label_stats['missing_text']}"
    )

    max_samples = int(cli_args.max_samples) if cli_args.max_samples is not None else 0
    train_records = load_records(
        annotation_file=train_ann,
        image_root=image_root,
        label_map=train_label_map,
        label_text_map=train_label_text_map,
        split_prefix=str(args.split_prefix),
        strip_split_prefix=bool(args.strip_split_prefix),
        max_samples=max_samples,
    )
    print(f"[INFO] train records loaded: {len(train_records):,}"
          + (f"  (limited to {max_samples:,})" if max_samples > 0 else ""))

    train_ds = GazeDataset(
        records=train_records,
        prompt_template=str(args.prompt_template),
        prompt_text=str(args.prompt_text or ""),
        apply_augmentation=False,
        id2label=id2label,
        vocab2id=vocab2id,
        vocab2id_lower=vocab2id_lower,
        num_classes=int(num_classes),
        visual_prompting=bool(args.visual_prompting),
        image_cache_size=max(0, int(getattr(args, "image_cache_size", 0))),
        filter_invalid_object_samples=False,
        coord_bins=coord_bins,
    )
    print(f"[INFO] GazeDataset size: {len(train_ds):,}")

    _nw = int(args.num_workers)
    train_loader = DataLoader(
        train_ds,
        batch_size=max(1, int(args.test_batch_size)),
        shuffle=False,
        num_workers=_nw,
        pin_memory=(device.type == "cuda"),
        collate_fn=QwenTestCollator(
            processor=processor,
            max_text_length=int(args.max_text_length),
            scene_size=scene_size,
        ),
        persistent_workers=(_nw > 0),
        prefetch_factor=(2 if _nw > 0 else None),
    )

    # ---- Run inference ----
    print(f"[INFO] starting inference on {len(train_ds):,} samples …")
    results = run_inference(
        model=model,
        loader=train_loader,
        device=device,
        amp_dtype=amp_dtype,
        processor=processor,
        id2label=id2label,
        num_classes=int(num_classes),
        coord_bins=coord_bins,
        max_new_tokens=int(args.generation_max_new_tokens),
        num_beams=int(getattr(args, "generation_num_beams", 1)),
        repetition_penalty=float(getattr(args, "repetition_penalty", 1.0)),
        no_repeat_ngram_size=int(getattr(args, "no_repeat_ngram_size", 0)),
        stop_at_object_end=bool(getattr(args, "generation_stop_at_object_end", True)),
        show_tqdm=show_tqdm,
    )

    # ---- Compute stats & print ----
    stats = compute_stats(results)
    print_summary(stats)

    # ---- Save JSON ----
    results_path = out_dir / "train_dist_results.json"
    results_path.write_text(
        json.dumps({"stats": stats, "samples": results}, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print(f"[INFO] results JSON saved: {results_path}  ({len(results):,} samples)")

    # Save stats-only for quick inspection
    stats_path = out_dir / "train_dist_stats.json"
    stats_path.write_text(
        json.dumps(stats, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(f"[INFO] stats JSON saved: {stats_path}")

    # ---- Histogram ----
    hist_path = out_dir / "train_dist_histogram.png"
    plot_histogram(results, stats, hist_path)


if __name__ == "__main__":
    main()
