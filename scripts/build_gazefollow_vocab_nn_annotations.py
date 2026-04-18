#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import torch
import torch.nn.functional as F


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from model.utils.data_utils import (  # noqa: E402
    alias_label_text,
    annotation_candidates,
    canonicalize_label_text,
    clean_label,
)


DEFAULT_THRESHOLD = 0.55


@dataclass(frozen=True)
class LabelMatch:
    raw_label: str
    normalized_label: str
    has_embedding: bool
    strategy: str
    mapped_vocab_label: str
    mapped_vocab_id: int
    accepted: int
    cosine_sim: float
    top3_vocab_labels: list[str]
    top3_vocab_sims: list[float]


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description=(
            "Build Gazefollow raw-label -> vocab2id mappings using exact/alias/"
            "canonical matches first, then embedding nearest-neighbor fallback."
        )
    )
    p.add_argument(
        "--label-embed-dir",
        type=Path,
        default=REPO_ROOT / "data/gazefollow/label-embeds",
        help="Directory containing one embedding file per label, e.g. 'laptop-emb.pt'.",
    )
    p.add_argument(
        "--vocab2id",
        type=Path,
        default=REPO_ROOT / "data/gazefollow/vocab2id.json",
        help="Path to vocab2id.json used as the closed-set classification space.",
    )
    p.add_argument(
        "--train-csv",
        type=Path,
        default=REPO_ROOT / "data/gazefollow/gaze-labels-train.csv",
    )
    p.add_argument(
        "--val-csv",
        type=Path,
        default=REPO_ROOT / "data/gazefollow/gaze-labels-val.csv",
    )
    p.add_argument(
        "--test-csv",
        type=Path,
        default=REPO_ROOT / "data/gazefollow/gaze-labels-test.csv",
    )
    p.add_argument(
        "--output-dir",
        type=Path,
        default=REPO_ROOT / "data/gazefollow/nn_mapped",
        help="Output directory for the mapping table, summary, and remapped annotations.",
    )
    p.add_argument(
        "--threshold",
        type=float,
        default=DEFAULT_THRESHOLD,
        help="Minimum cosine similarity for accepting nearest-neighbor fallback.",
    )
    p.add_argument(
        "--include-all-embed-labels",
        action="store_true",
        help=(
            "Build mapping rows for every label embedding file, not just labels observed in "
            "the train/val/test CSVs."
        ),
    )
    return p.parse_args()


def load_vocab2id(path: Path) -> dict[str, int]:
    with path.open("r", encoding="utf-8") as f:
        vocab = json.load(f)
    return {str(k): int(v) for k, v in vocab.items()}


def load_label_embedding_bank(label_embed_dir: Path) -> dict[str, torch.Tensor]:
    bank: dict[str, torch.Tensor] = {}
    for p in sorted(label_embed_dir.glob("*-emb.pt")):
        label = p.name[:-7]
        vec = torch.load(p, map_location="cpu")
        if not isinstance(vec, torch.Tensor):
            raise TypeError(f"Expected Tensor in {p}, got {type(vec).__name__}.")
        if vec.ndim != 1:
            raise ValueError(f"Expected 1D embedding in {p}, got shape {tuple(vec.shape)}.")
        bank[label] = vec.float().contiguous()
    return bank


def infer_label_column(fieldnames: list[str]) -> str:
    for key in ("gaze_pseudo_label", "gaze_gt_label", "gaze_gt_labels", "annotation"):
        if key in fieldnames:
            return key
    raise ValueError(f"Could not infer label column from fields: {fieldnames}")


def read_observed_labels(csv_paths: Iterable[Path]) -> tuple[set[str], dict[str, dict[str, int]]]:
    observed: set[str] = set()
    per_split_stats: dict[str, dict[str, int]] = {}
    for csv_path in csv_paths:
        if not csv_path.exists():
            continue
        with csv_path.open("r", encoding="utf-8", newline="") as f:
            reader = csv.DictReader(f)
            if not reader.fieldnames:
                continue
            label_key = infer_label_column(list(reader.fieldnames))
            split_stats = {"rows": 0, "with_label": 0}
            for row in reader:
                split_stats["rows"] += 1
                txt = clean_label(str(row.get(label_key, "")))
                if txt:
                    split_stats["with_label"] += 1
                    observed.add(txt)
            per_split_stats[csv_path.name] = split_stats
    return observed, per_split_stats


def build_vocab_lookup(vocab2id: dict[str, int]) -> tuple[dict[str, int], dict[str, str]]:
    lower_to_id: dict[str, int] = {}
    lower_to_label: dict[str, str] = {}
    for label, idx in vocab2id.items():
        lower = label.lower()
        lower_to_id[lower] = int(idx)
        lower_to_label[lower] = label
    return lower_to_id, lower_to_label


def resolve_direct_vocab_match(
    raw_label: str,
    vocab2id: dict[str, int],
    vocab2id_lower: dict[str, int],
    lower_to_label: dict[str, str],
) -> tuple[str, str, int] | None:
    raw = clean_label(raw_label)
    if not raw:
        return None

    def _lookup(candidate: str) -> tuple[str, str, int] | None:
        txt = clean_label(candidate)
        if not txt:
            return None
        if txt in vocab2id:
            return txt, "direct", int(vocab2id[txt])
        lower = txt.lower()
        if lower in vocab2id_lower:
            return lower_to_label[lower], "direct_lower", int(vocab2id_lower[lower])
        return None

    ordered_candidates: list[tuple[str, str]] = []
    ordered_candidates.append(("exact", raw))

    aliased = clean_label(alias_label_text(raw))
    if aliased and aliased.lower() != raw.lower():
        ordered_candidates.append(("alias", aliased))

    canonical = clean_label(canonicalize_label_text(raw))
    if canonical and canonical.lower() not in {raw.lower(), aliased.lower() if aliased else ""}:
        ordered_candidates.append(("canonical", canonical))

    seen = {cand.lower() for _, cand in ordered_candidates if cand}
    for cand in annotation_candidates(raw):
        cln = clean_label(cand)
        if cln and cln.lower() not in seen:
            ordered_candidates.append(("candidate", cln))
            seen.add(cln.lower())

    for stage, cand in ordered_candidates:
        found = _lookup(cand)
        if found is not None:
            vocab_label, lookup_stage, vocab_id = found
            stage_name = stage if lookup_stage == "direct" else f"{stage}_lower"
            return vocab_label, stage_name, vocab_id
    return None


def build_vocab_embedding_bank(
    vocab2id: dict[str, int],
    embed_bank: dict[str, torch.Tensor],
) -> tuple[list[str], torch.Tensor]:
    vocab_labels = sorted(vocab2id, key=lambda x: vocab2id[x])
    missing = [label for label in vocab_labels if label not in embed_bank]
    if missing:
        raise RuntimeError(
            f"Missing embeddings for {len(missing)} vocab labels. Examples: {missing[:10]}"
        )
    mat = torch.stack([embed_bank[label] for label in vocab_labels], dim=0)
    mat = F.normalize(mat.float(), dim=1)
    return vocab_labels, mat


def build_label_matches(
    labels: Iterable[str],
    *,
    vocab2id: dict[str, int],
    vocab2id_lower: dict[str, int],
    lower_to_label: dict[str, str],
    embed_bank: dict[str, torch.Tensor],
    vocab_labels: list[str],
    vocab_mat: torch.Tensor,
    threshold: float,
) -> dict[str, LabelMatch]:
    out: dict[str, LabelMatch] = {}
    for raw_label in sorted(set(clean_label(x) for x in labels if clean_label(x))):
        direct = resolve_direct_vocab_match(raw_label, vocab2id, vocab2id_lower, lower_to_label)
        emb = embed_bank.get(raw_label)
        has_embedding = emb is not None

        if direct is not None:
            vocab_label, strategy, vocab_id = direct
            sim = 1.0 if raw_label == vocab_label else 0.0
            out[raw_label] = LabelMatch(
                raw_label=raw_label,
                normalized_label=clean_label(alias_label_text(raw_label)),
                has_embedding=has_embedding,
                strategy=strategy,
                mapped_vocab_label=vocab_label,
                mapped_vocab_id=int(vocab_id),
                accepted=1,
                cosine_sim=float(sim),
                top3_vocab_labels=[vocab_label],
                top3_vocab_sims=[float(sim)],
            )
            continue

        if emb is None:
            out[raw_label] = LabelMatch(
                raw_label=raw_label,
                normalized_label=clean_label(alias_label_text(raw_label)),
                has_embedding=False,
                strategy="missing_embedding",
                mapped_vocab_label="",
                mapped_vocab_id=-1,
                accepted=0,
                cosine_sim=float("nan"),
                top3_vocab_labels=[],
                top3_vocab_sims=[],
            )
            continue

        emb_n = F.normalize(emb.float().view(1, -1), dim=1)
        sims = torch.matmul(emb_n, vocab_mat.T).squeeze(0)
        topk = min(3, sims.numel())
        vals, inds = torch.topk(sims, k=topk, largest=True, sorted=True)
        top_labels = [vocab_labels[int(i)] for i in inds.tolist()]
        top_sims = [float(v) for v in vals.tolist()]
        best_label = top_labels[0]
        best_id = int(vocab2id[best_label])
        best_sim = float(top_sims[0])
        accepted = int(best_sim >= float(threshold))
        out[raw_label] = LabelMatch(
            raw_label=raw_label,
            normalized_label=clean_label(alias_label_text(raw_label)),
            has_embedding=True,
            strategy="nn_vocab",
            mapped_vocab_label=best_label if accepted else "",
            mapped_vocab_id=best_id if accepted else -1,
            accepted=accepted,
            cosine_sim=best_sim,
            top3_vocab_labels=top_labels,
            top3_vocab_sims=top_sims,
        )
    return out


def write_mapping_csv(path: Path, matches: dict[str, LabelMatch]) -> None:
    fieldnames = [
        "raw_label",
        "normalized_label",
        "has_embedding",
        "strategy",
        "mapped_vocab_label",
        "mapped_vocab_id",
        "accepted",
        "cosine_sim",
        "top3_vocab_labels",
        "top3_vocab_sims",
    ]
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for label in sorted(matches):
            m = matches[label]
            writer.writerow(
                {
                    "raw_label": m.raw_label,
                    "normalized_label": m.normalized_label,
                    "has_embedding": int(m.has_embedding),
                    "strategy": m.strategy,
                    "mapped_vocab_label": m.mapped_vocab_label,
                    "mapped_vocab_id": m.mapped_vocab_id,
                    "accepted": m.accepted,
                    "cosine_sim": "" if m.cosine_sim != m.cosine_sim else f"{m.cosine_sim:.6f}",
                    "top3_vocab_labels": " | ".join(m.top3_vocab_labels),
                    "top3_vocab_sims": " | ".join(f"{x:.6f}" for x in m.top3_vocab_sims),
                }
            )


def remap_annotation_csv(csv_path: Path, out_path: Path, matches: dict[str, LabelMatch]) -> dict[str, int]:
    stats = {
        "rows": 0,
        "with_label": 0,
        "accepted": 0,
        "rejected": 0,
        "missing_match": 0,
    }
    with csv_path.open("r", encoding="utf-8", newline="") as f_in, out_path.open(
        "w", encoding="utf-8", newline=""
    ) as f_out:
        reader = csv.DictReader(f_in)
        if not reader.fieldnames:
            raise ValueError(f"No header found in {csv_path}")
        label_key = infer_label_column(list(reader.fieldnames))
        extra_cols = [
            "mapped_vocab_label",
            "mapped_vocab_id",
            "mapped_similarity",
            "mapped_strategy",
            "mapped_accepted",
        ]
        writer = csv.DictWriter(f_out, fieldnames=list(reader.fieldnames) + extra_cols)
        writer.writeheader()

        for row in reader:
            stats["rows"] += 1
            raw_label = clean_label(str(row.get(label_key, "")))
            if raw_label:
                stats["with_label"] += 1
            match = matches.get(raw_label)
            if match is None:
                stats["missing_match"] += 1
                row.update(
                    {
                        "mapped_vocab_label": "",
                        "mapped_vocab_id": -1,
                        "mapped_similarity": "",
                        "mapped_strategy": "missing_label",
                        "mapped_accepted": 0,
                    }
                )
            else:
                if match.accepted:
                    stats["accepted"] += 1
                else:
                    stats["rejected"] += 1
                row.update(
                    {
                        "mapped_vocab_label": match.mapped_vocab_label,
                        "mapped_vocab_id": match.mapped_vocab_id,
                        "mapped_similarity": (
                            "" if match.cosine_sim != match.cosine_sim else f"{match.cosine_sim:.6f}"
                        ),
                        "mapped_strategy": match.strategy,
                        "mapped_accepted": match.accepted,
                    }
                )
            writer.writerow(row)
    return stats


def build_summary(
    matches: dict[str, LabelMatch],
    split_input_stats: dict[str, dict[str, int]],
    split_output_stats: dict[str, dict[str, int]],
) -> dict[str, object]:
    strategies: dict[str, int] = {}
    accepted = 0
    rejected = 0
    with_embedding = 0
    for m in matches.values():
        strategies[m.strategy] = strategies.get(m.strategy, 0) + 1
        accepted += int(m.accepted)
        rejected += int(not m.accepted)
        with_embedding += int(m.has_embedding)
    return {
        "num_labels": len(matches),
        "num_labels_with_embedding": with_embedding,
        "num_labels_accepted": accepted,
        "num_labels_rejected": rejected,
        "strategy_counts": strategies,
        "split_input_stats": split_input_stats,
        "split_output_stats": split_output_stats,
    }


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    vocab2id = load_vocab2id(args.vocab2id)
    vocab2id_lower, lower_to_label = build_vocab_lookup(vocab2id)
    embed_bank = load_label_embedding_bank(args.label_embed_dir)
    vocab_labels, vocab_mat = build_vocab_embedding_bank(vocab2id, embed_bank)

    csv_paths = [args.train_csv, args.val_csv, args.test_csv]
    observed_labels, split_input_stats = read_observed_labels(csv_paths)
    target_labels = set(embed_bank) if args.include_all_embed_labels else set(observed_labels)
    matches = build_label_matches(
        target_labels,
        vocab2id=vocab2id,
        vocab2id_lower=vocab2id_lower,
        lower_to_label=lower_to_label,
        embed_bank=embed_bank,
        vocab_labels=vocab_labels,
        vocab_mat=vocab_mat,
        threshold=float(args.threshold),
    )

    mapping_csv = args.output_dir / "label_to_vocab_nn.csv"
    write_mapping_csv(mapping_csv, matches)

    split_output_stats: dict[str, dict[str, int]] = {}
    for csv_path in csv_paths:
        out_name = csv_path.name.replace(".csv", ".nnmapped.csv")
        out_path = args.output_dir / out_name
        split_output_stats[csv_path.name] = remap_annotation_csv(csv_path, out_path, matches)

    summary = build_summary(matches, split_input_stats, split_output_stats)
    summary["threshold"] = float(args.threshold)
    summary["mapping_csv"] = str(mapping_csv)
    summary["output_dir"] = str(args.output_dir)

    summary_path = args.output_dir / "summary.json"
    with summary_path.open("w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, sort_keys=True)

    print(f"[INFO] wrote mapping table: {mapping_csv}")
    for csv_name, stats in split_output_stats.items():
        print(
            f"[INFO] {csv_name}: rows={stats['rows']} with_label={stats['with_label']} "
            f"accepted={stats['accepted']} rejected={stats['rejected']} "
            f"missing_match={stats['missing_match']}"
        )
    print(f"[INFO] wrote summary: {summary_path}")


if __name__ == "__main__":
    main()
