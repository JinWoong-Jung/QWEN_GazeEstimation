#!/usr/bin/env python3
"""
RL_data_pipeline.py

eval_train_dist.py 결과(train_dist_results.json)를 읽어
min_l2 기준 3구간(< Q1 / Q1~Q3 / > Q3)으로 샘플을 나누고,
각 구간에서 지정한 비율로 랜덤 추출하여 RL 학습용 annotation 파일을 생성합니다.

Outputs:
  <output_dir>/rl_train_annotations.txt   -- 필터링된 annotation 파일 (trainer의 --train_ann로 사용)
  <output_dir>/rl_subset_info.json        -- 선택 메타데이터 및 통계
  <output_dir>/rl_dist_comparison.png     -- 전체 vs 선택 분포 비교 히스토그램

Usage:
    python RL_data_pipeline.py \\
        --eval_results eval_train_dist_output/train_dist_results.json \\
        --config RL.yaml \\
        --output_dir rl_data \\
        --n_total 5000 \\
        --ratio_easy 0.15 \\
        --ratio_mid  0.55 \\
        --ratio_hard 0.30 \\
        [--seed 42]
"""
from __future__ import annotations

import argparse
import csv
import json
import os
import random
import sys
from pathlib import Path
from typing import Any

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

from model.utils.config_parser import build_parser, load_yaml_config
from model.utils.data_utils import load_label_map, load_label_text_map, load_records, load_vocab2id


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


def load_annotation_rows(ann_path: Path) -> dict[tuple[str, int], list[str]]:
    """원본 annotation 파일을 읽어 (image_rel, sample_id) -> row 매핑을 반환."""
    mapping: dict[tuple[str, int], list[str]] = {}
    with ann_path.open("r", encoding="utf-8", newline="") as f:
        for row in csv.reader(f):
            if not row:
                continue
            try:
                image_rel = str(row[0]).strip()
                sample_id = int(float(row[1]))
            except Exception:
                continue
            key = (image_rel, sample_id)
            if key not in mapping:          # 중복 시 첫 번째만 유지
                mapping[key] = row
    return mapping


def write_annotation_file(rows: list[list[str]], out_path: Path) -> None:
    with out_path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f)
        for row in rows:
            writer.writerow(row)


# ---------------------------------------------------------------------------
# Bucket split
# ---------------------------------------------------------------------------

def split_buckets(
    samples: list[dict[str, Any]],
    q1: float,
    q3: float,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]]]:
    """min_l2 기준 4개 그룹으로 분리.

    Returns (easy, medium, hard, invalid)
    - easy   : min_l2 < Q1
    - medium : Q1 <= min_l2 <= Q3
    - hard   : min_l2 > Q3
    - invalid: min_l2 is None (format 실패 등)
    """
    easy, medium, hard, invalid = [], [], [], []
    for s in samples:
        d = s["dist"]["min_l2"]
        if d is None:
            invalid.append(s)
        elif d < q1:
            easy.append(s)
        elif d <= q3:
            medium.append(s)
        else:
            hard.append(s)
    return easy, medium, hard, invalid


# ---------------------------------------------------------------------------
# Sampling
# ---------------------------------------------------------------------------

def sample_buckets(
    easy: list[dict[str, Any]],
    medium: list[dict[str, Any]],
    hard: list[dict[str, Any]],
    n_total: int,
    ratio_easy: float,
    ratio_mid: float,
    ratio_hard: float,
    seed: int,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """각 버킷에서 지정 비율로 랜덤 추출. 부족한 경우 남은 할당량을 다른 버킷에서 보충."""
    rng = random.Random(seed)

    # 정규화 (합이 1이 아닐 수 있으므로)
    total_ratio = ratio_easy + ratio_mid + ratio_hard
    r_easy = ratio_easy / total_ratio
    r_mid  = ratio_mid  / total_ratio
    r_hard = ratio_hard / total_ratio

    target_easy   = round(n_total * r_easy)
    target_medium = round(n_total * r_mid)
    target_hard   = n_total - target_easy - target_medium   # 반올림 오차 흡수

    # 각 버킷에서 추출 (버킷이 부족하면 가능한 만큼만)
    picked_easy   = rng.sample(easy,   min(target_easy,   len(easy)))
    picked_medium = rng.sample(medium, min(target_medium, len(medium)))
    picked_hard   = rng.sample(hard,   min(target_hard,   len(hard)))

    # 부족분 계산
    shortage = (
        (target_easy   - len(picked_easy))
        + (target_medium - len(picked_medium))
        + (target_hard   - len(picked_hard))
    )

    # 부족분은 가장 큰 버킷(medium)에서 보충
    already_mid_indices = {id(s) for s in picked_medium}
    remaining_medium = [s for s in medium if id(s) not in already_mid_indices]
    supplement = rng.sample(remaining_medium, min(shortage, len(remaining_medium)))
    picked_medium = picked_medium + supplement

    selected = picked_easy + picked_medium + picked_hard
    rng.shuffle(selected)

    actual_shortage = n_total - len(selected)
    if actual_shortage > 0:
        print(f"[WARN] 전체 버킷 합이 n_total={n_total}보다 {actual_shortage}개 부족합니다.")

    sample_stat = {
        "target": {"easy": target_easy, "medium": target_medium, "hard": target_hard},
        "actual": {
            "easy":   len(picked_easy),
            "medium": len(picked_medium) - len(supplement),
            "hard":   len(picked_hard),
            "supplement_from_medium": len(supplement),
            "total":  len(selected),
        },
        "available": {"easy": len(easy), "medium": len(medium), "hard": len(hard)},
    }
    return selected, sample_stat


# ---------------------------------------------------------------------------
# Visualization
# ---------------------------------------------------------------------------

def plot_comparison(
    all_samples: list[dict[str, Any]],
    selected: list[dict[str, Any]],
    q1: float,
    q3: float,
    out_path: Path,
) -> None:
    all_l2 = [s["dist"]["min_l2"] for s in all_samples if s["dist"]["min_l2"] is not None]
    sel_l2 = [s["dist"]["min_l2"] for s in selected     if s["dist"]["min_l2"] is not None]
    if not all_l2 or not sel_l2:
        return

    arr_all = np.array(all_l2, dtype=np.float64)
    arr_sel = np.array(sel_l2, dtype=np.float64)

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # --- 왼쪽: 전체 분포 (with Q1/Q3 마킹) ---
    ax0 = axes[0]
    ax0.hist(arr_all, bins=60, color="steelblue", edgecolor="white", lw=0.4, alpha=0.8,
             label=f"전체 ({len(arr_all):,}개)")
    ax0.axvline(q1, color="orange", linestyle="--", lw=1.8, label=f"Q1={q1:.4f}")
    ax0.axvline(q3, color="red",    linestyle="--", lw=1.8, label=f"Q3={q3:.4f}")
    ax0.axvspan(float(arr_all.min()), q1, alpha=0.08, color="orange")
    ax0.axvspan(q1, q3,               alpha=0.08, color="gray")
    ax0.axvspan(q3, float(arr_all.max()), alpha=0.08, color="red")
    ax0.set_xlabel("min_l2")
    ax0.set_ylabel("Count")
    ax0.set_title("전체 train 분포")
    ax0.legend(fontsize=9)

    # --- 오른쪽: 선택 vs 전체 (정규화 밀도 비교) ---
    ax1 = axes[1]
    bins = np.linspace(0, min(float(arr_all.max()), 1.0), 61)
    ax1.hist(arr_all, bins=bins, density=True,
             color="steelblue", edgecolor="white", lw=0.3, alpha=0.5, label="전체 (density)")
    ax1.hist(arr_sel, bins=bins, density=True,
             color="tomato",    edgecolor="white", lw=0.3, alpha=0.6, label=f"선택 ({len(arr_sel):,}개, density)")
    ax1.axvline(q1, color="orange", linestyle="--", lw=1.5, label=f"Q1={q1:.4f}")
    ax1.axvline(q3, color="red",    linestyle="--", lw=1.5, label=f"Q3={q3:.4f}")
    ax1.set_xlabel("min_l2")
    ax1.set_ylabel("Density")
    ax1.set_title("전체 vs 선택 분포 비교")
    ax1.legend(fontsize=9)

    # 하단 요약
    sel_mean = float(arr_sel.mean())
    sel_med  = float(np.median(arr_sel))
    easy_frac  = float((arr_sel < q1).mean())
    mid_frac   = float(((arr_sel >= q1) & (arr_sel <= q3)).mean())
    hard_frac  = float((arr_sel > q3).mean())
    annot = (
        f"선택 n={len(arr_sel):,}  mean={sel_mean:.4f}  median={sel_med:.4f}\n"
        f"< Q1: {easy_frac:.1%}  │  Q1~Q3: {mid_frac:.1%}  │  > Q3: {hard_frac:.1%}"
    )
    fig.text(
        0.5, -0.03, annot, ha="center", va="top",
        fontsize=9.5, fontfamily="monospace",
        bbox=dict(boxstyle="round,pad=0.4", facecolor="lightyellow", alpha=0.85),
    )

    plt.tight_layout()
    plt.savefig(str(out_path), dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"[INFO] 비교 히스토그램 저장: {out_path}")


# ---------------------------------------------------------------------------
# Summary print
# ---------------------------------------------------------------------------

def print_summary(
    q1: float,
    q3: float,
    sample_stat: dict[str, Any],
    selected: list[dict[str, Any]],
    ann_written: int,
) -> None:
    sep = "=" * 60
    act = sample_stat["actual"]
    tgt = sample_stat["target"]
    avl = sample_stat["available"]

    sel_l2 = [s["dist"]["min_l2"] for s in selected if s["dist"]["min_l2"] is not None]
    sel_arr = np.array(sel_l2, dtype=np.float64) if sel_l2 else np.array([], dtype=np.float64)

    print(f"\n{sep}")
    print("  RL DATA PIPELINE 결과 요약")
    print(sep)
    print(f"  Q1 = {q1:.4f}   Q3 = {q3:.4f}")
    print()
    print(f"  {'버킷':<10} {'가용':>8} {'목표':>8} {'실제':>8}  비율")
    print(f"  {'-'*46}")
    total_act = act['total']
    print(f"  {'easy(<Q1)':<10} {avl['easy']:>8,} {tgt['easy']:>8,} {act['easy']:>8,}  "
          f"{act['easy']/max(total_act,1):.1%}")
    print(f"  {'mid(Q1~Q3)':<10} {avl['medium']:>8,} {tgt['medium']:>8,} "
          f"{act['medium']+act['supplement_from_medium']:>8,}  "
          f"{(act['medium']+act['supplement_from_medium'])/max(total_act,1):.1%}")
    print(f"  {'hard(>Q3)':<10} {avl['hard']:>8,} {tgt['hard']:>8,} {act['hard']:>8,}  "
          f"{act['hard']/max(total_act,1):.1%}")
    if act['supplement_from_medium'] > 0:
        print(f"  (medium에서 부족분 {act['supplement_from_medium']}개 보충)")
    print(f"  {'-'*46}")
    print(f"  {'합계':<10} {'':>8} {sum(tgt.values()):>8,} {total_act:>8,}")

    if sel_arr.size > 0:
        print(f"\n  ---- 선택 샘플 min_l2 통계 ----")
        print(f"  Mean   = {sel_arr.mean():.4f}")
        print(f"  Median = {float(np.median(sel_arr)):.4f}")
        print(f"  Std    = {sel_arr.std():.4f}")

    print(f"\n  annotation 파일 기록 완료: {ann_written:,}행")
    print(f"{sep}\n")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    cli = argparse.ArgumentParser(
        description="eval_train_dist 결과로 RL 학습용 annotation 파일 생성"
    )
    cli.add_argument("--eval_results", type=str,
                     default="eval_train_dist_output/train_dist_results.json",
                     help="eval_train_dist.py가 출력한 train_dist_results.json 경로")
    cli.add_argument("--config", type=str, default="RL.yaml",
                     help="학습에 사용한 YAML config 경로")
    cli.add_argument("--output_dir", type=str, default="rl_data",
                     help="출력 디렉토리")
    cli.add_argument("--n_total", type=int, default=5000,
                     help="총 선택 샘플 수 (default: 5000)")
    cli.add_argument("--ratio_easy", type=float, default=0.15,
                     help="< Q1 구간 비율 (default: 0.15)")
    cli.add_argument("--ratio_mid", type=float, default=0.55,
                     help="Q1~Q3 구간 비율 (default: 0.55)")
    cli.add_argument("--ratio_hard", type=float, default=0.30,
                     help="> Q3 구간 비율 (default: 0.30)")
    cli.add_argument("--seed", type=int, default=42,
                     help="랜덤 시드 (default: 42)")
    cli.add_argument("--exclude_format_invalid", action="store_true", default=True,
                     help="format 실패 샘플 제외 (default: True)")
    cli_args = cli.parse_args()

    out_dir = resolve_path(cli_args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # ---- eval 결과 로드 ----
    eval_path = resolve_path(cli_args.eval_results)
    if not eval_path.exists():
        raise FileNotFoundError(f"eval_results not found: {eval_path}")
    with eval_path.open("r", encoding="utf-8") as f:
        eval_data = json.load(f)

    all_samples: list[dict[str, Any]] = eval_data["samples"]
    stats_meta: dict[str, Any] = eval_data["stats"]

    q1 = float(stats_meta["min_l2"]["q1"])
    q3 = float(stats_meta["min_l2"]["q3"])
    print(f"[INFO] eval 결과 로드: {len(all_samples):,}개  Q1={q1:.4f}  Q3={q3:.4f}")

    # ---- 3구간 분리 ----
    easy, medium, hard, invalid = split_buckets(all_samples, q1, q3)
    print(
        f"[INFO] 버킷 분리 → "
        f"easy={len(easy):,}  mid={len(medium):,}  hard={len(hard):,}  invalid={len(invalid):,}"
    )

    # ---- 비율별 샘플링 ----
    selected, sample_stat = sample_buckets(
        easy=easy,
        medium=medium,
        hard=hard,
        n_total=int(cli_args.n_total),
        ratio_easy=float(cli_args.ratio_easy),
        ratio_mid=float(cli_args.ratio_mid),
        ratio_hard=float(cli_args.ratio_hard),
        seed=int(cli_args.seed),
    )
    print(f"[INFO] 선택 완료: {len(selected):,}개")

    # ---- 원본 annotation 파일 로드 ----
    config_path = resolve_path(cli_args.config)
    cfg = load_yaml_config(config_path)
    args = build_parser(defaults=cfg).parse_args(["--config", str(config_path)])

    vocab2id_path = resolve_path(args.vocab2id)
    vocab2id, vocab2id_lower = load_vocab2id(vocab2id_path)
    id2label = build_id2label(vocab2id)
    num_classes = len(vocab2id)

    train_ann    = resolve_path(args.train_ann)
    image_root   = resolve_path(args.image_root)
    train_labels = resolve_path(args.train_labels)

    print(f"[INFO] 원본 annotation 파일 로드: {train_ann}")
    ann_row_map = load_annotation_rows(train_ann)
    print(f"[INFO] annotation rows: {len(ann_row_map):,}")

    # ---- train records 로드 (sample_index → Record 매핑 확보) ----
    print("[INFO] train records 로드 중 (sample_index 매핑 용도) ...")
    train_label_map, _ = load_label_map(
        train_labels,
        vocab2id=vocab2id,
        vocab2id_lower=vocab2id_lower,
        text_key="gaze_pseudo_label",
        use_embed_fallback=False,
    )
    train_label_text_map, _ = load_label_text_map(
        train_labels, text_key="gaze_pseudo_label",
    )
    train_records = load_records(
        annotation_file=train_ann,
        image_root=image_root,
        label_map=train_label_map,
        label_text_map=train_label_text_map,
        split_prefix=str(args.split_prefix),
        strip_split_prefix=bool(args.strip_split_prefix),
    )
    print(f"[INFO] train records: {len(train_records):,}")

    # ---- 선택된 샘플 → annotation row 매핑 ----
    selected_rows: list[list[str]] = []
    missing_count = 0

    for s in selected:
        idx = int(s["sample_index"])
        if idx >= len(train_records):
            missing_count += 1
            continue
        rec = train_records[idx]
        # annotation 파일의 image_rel은 split_prefix 포함 여부에 따라 다름
        # load_records에서 strip_split_prefix=True면 rec.image_rel에 prefix 없음
        # ann_row_map의 key는 원본 annotation의 row[0] (prefix 포함)
        key_stripped = (rec.image_rel, rec.sample_id)
        key_prefixed = (f"{args.split_prefix}{rec.image_rel}", rec.sample_id)

        row = ann_row_map.get(key_stripped) or ann_row_map.get(key_prefixed)
        if row is not None:
            selected_rows.append(row)
        else:
            missing_count += 1

    if missing_count > 0:
        print(f"[WARN] annotation row 매핑 실패: {missing_count}개 (건너뜀)")

    # ---- annotation 파일 출력 ----
    ann_out_path = out_dir / "rl_train_annotations.txt"
    write_annotation_file(selected_rows, ann_out_path)
    print(f"[INFO] annotation 파일 저장: {ann_out_path}  ({len(selected_rows):,}행)")

    # ---- subset info JSON 저장 ----
    info = {
        "config": {
            "n_total": int(cli_args.n_total),
            "ratio_easy": float(cli_args.ratio_easy),
            "ratio_mid":  float(cli_args.ratio_mid),
            "ratio_hard": float(cli_args.ratio_hard),
            "seed": int(cli_args.seed),
        },
        "thresholds": {"q1": q1, "q3": q3},
        "bucket_sizes": {
            "easy":    len(easy),
            "medium":  len(medium),
            "hard":    len(hard),
            "invalid": len(invalid),
            "total_valid": len(easy) + len(medium) + len(hard),
        },
        "sample_stat": sample_stat,
        "annotation_rows_written": len(selected_rows),
        "selected_min_l2_stats": {
            "mean":   float(np.mean([s["dist"]["min_l2"] for s in selected
                                     if s["dist"]["min_l2"] is not None])),
            "median": float(np.median([s["dist"]["min_l2"] for s in selected
                                       if s["dist"]["min_l2"] is not None])),
            "std":    float(np.std([s["dist"]["min_l2"] for s in selected
                                    if s["dist"]["min_l2"] is not None])),
        } if any(s["dist"]["min_l2"] is not None for s in selected) else {},
        "selected_samples": [
            {
                "sample_index": s["sample_index"],
                "image_rel":    s["image_rel"],
                "min_l2":       s["dist"]["min_l2"],
                "bucket": (
                    "easy"   if (s["dist"]["min_l2"] is not None and s["dist"]["min_l2"] < q1)
                    else "medium" if (s["dist"]["min_l2"] is not None and s["dist"]["min_l2"] <= q3)
                    else "hard"
                ),
            }
            for s in selected
        ],
    }
    info_path = out_dir / "rl_subset_info.json"
    info_path.write_text(json.dumps(info, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"[INFO] subset info 저장: {info_path}")

    # ---- 비교 히스토그램 ----
    plot_comparison(
        all_samples=all_samples,
        selected=selected,
        q1=q1,
        q3=q3,
        out_path=out_dir / "rl_dist_comparison.png",
    )

    # ---- 요약 출력 ----
    print_summary(q1, q3, sample_stat, selected, len(selected_rows))

    # ---- 사용법 안내 ----
    print(f"[ RL 학습 시 아래 인자를 추가하세요 ]")
    print(f"  --train_ann {ann_out_path}")
    print()


if __name__ == "__main__":
    main()
