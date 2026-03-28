from __future__ import annotations

from typing import Any

import numpy as np
import torch
from torch.utils.data import DataLoader
from tqdm.auto import tqdm


def run_eval(
    model: torch.nn.Module,
    loader: DataLoader,
    device: torch.device,
    amp_dtype: torch.dtype,
    show_tqdm: bool = True,
    desc: str = "Eval",
) -> dict[str, float]:
    model.eval()
    sums: dict[str, float] = {"loss": 0.0, "l_hm": 0.0, "l_cls": 0.0, "dist": 0.0}
    cls_correct = 0
    cls_total = 0
    steps = 0

    with torch.no_grad():
        eval_iter = tqdm(
            loader,
            desc=desc,
            leave=False,
            dynamic_ncols=True,
            disable=not show_tqdm,
        )
        for batch in eval_iter:
            target_heatmap = batch["target_heatmap"].to(device)
            target_label = batch["target_label"].to(device)
            target_label_emb = batch["target_label_emb"].to(device)
            target_label_valid = batch["target_label_valid"].to(device)
            target_point = batch["target_point"].to(device)
            use_cls_id = bool(torch.any(target_label >= 0).item())
            use_cls_emb = bool(torch.any(target_label_valid > 0).item())
            backbone_kwargs = None
            if "joint_inputs" in batch:
                backbone_kwargs = {
                    "joint_inputs": batch["joint_inputs"],
                    "joint_bsz": int(batch.get("joint_bsz", len(batch.get("text_inputs", [])))),
                }

            with torch.autocast(
                device_type=device.type,
                dtype=amp_dtype,
                enabled=(device.type == "cuda"),
            ):
                out = model(
                    scene_image=batch.get("scene_images", None),
                    head_image=batch.get("head_images", None),
                    text_inputs=batch.get("text_inputs", None),
                    target_heatmap=target_heatmap,
                    target_label=target_label if use_cls_id else None,
                    target_label_emb=target_label_emb if use_cls_emb else None,
                    target_label_valid=target_label_valid if use_cls_emb else None,
                    use_softargmax=False,
                    compute_point_soft=False,
                    compute_point_hard=True,
                    backbone_kwargs=backbone_kwargs,
                )

            loss_dict = out.get("loss_dict", {})
            sums["loss"] += float(loss_dict.get("loss", out["loss"]).detach().item())
            sums["l_hm"] += float(loss_dict.get("l_hm", torch.tensor(0.0)).detach().item())
            if "l_cls" in loss_dict:
                sums["l_cls"] += float(loss_dict["l_cls"].detach().item())
            pred_point = out["point_hard"].detach().to(dtype=torch.float32)
            tgt_point = target_point.detach().to(dtype=torch.float32)
            dist = torch.linalg.norm(pred_point - tgt_point, dim=-1).mean()
            sums["dist"] += float(dist.item())

            if "logits" in out:
                valid = target_label >= 0
                if torch.any(valid):
                    pred = out["pred_label"][valid]
                    gt = target_label[valid]
                    cls_correct += int((pred == gt).sum().item())
                    cls_total += int(valid.sum().item())
            steps += 1
            if show_tqdm:
                eval_iter.set_postfix(loss=f"{(sums['loss'] / max(steps, 1)):.4f}")

    if steps == 0:
        return {"loss": 0.0, "l_hm": 0.0, "l_cls": 0.0, "dist": 0.0, "cls_acc": 0.0}
    out = {k: v / steps for k, v in sums.items()}
    out["cls_acc"] = (cls_correct / cls_total) if cls_total > 0 else 0.0
    return out


def _auc_from_heatmap(
    heatmap_2d: torch.Tensor,
    gt_points: torch.Tensor,
) -> float | None:
    h, w = int(heatmap_2d.shape[0]), int(heatmap_2d.shape[1])
    gt = np.zeros((h, w), dtype=np.uint8)
    for p in gt_points:
        x = float(p[0].item())
        y = float(p[1].item())
        ix = int(round(max(0.0, min(1.0, x)) * (w - 1)))
        iy = int(round(max(0.0, min(1.0, y)) * (h - 1)))
        gt[iy, ix] = 1

    labels = gt.reshape(-1)
    n_pos = int(labels.sum())
    n_neg = int(labels.shape[0] - n_pos)
    if n_pos <= 0 or n_neg <= 0:
        return None

    scores = heatmap_2d.detach().float().cpu().numpy().reshape(-1)
    order = np.argsort(scores)
    ranks = np.empty_like(order, dtype=np.float64)
    ranks[order] = np.arange(1, len(scores) + 1, dtype=np.float64)
    pos_ranks = float(ranks[labels == 1].sum())
    auc = (pos_ranks - (n_pos * (n_pos + 1) / 2.0)) / float(n_pos * n_neg)
    return float(max(0.0, min(1.0, auc)))


def _l2(a: tuple[float, float], b: tuple[float, float]) -> float:
    return ((a[0] - b[0]) ** 2 + (a[1] - b[1]) ** 2) ** 0.5


def _min_l2(pred: tuple[float, float], gt_points: list[tuple[float, float]]) -> float:
    return min(_l2(pred, gt) for gt in gt_points)


def _avg_l2(pred: tuple[float, float], gt_points: list[tuple[float, float]]) -> float:
    mx = sum(x for x, _ in gt_points) / len(gt_points)
    my = sum(y for _, y in gt_points) / len(gt_points)
    return _l2(pred, (mx, my))


def run_test_metrics(
    model: torch.nn.Module,
    loader: DataLoader,
    device: torch.device,
    amp_dtype: torch.dtype,
    show_tqdm: bool = True,
    desc: str = "Test",
    acc_dist_threshold: float = 0.15,
) -> dict[str, float]:
    _ = float(acc_dist_threshold)  # kept for CLI compatibility
    model.eval()
    aucs: list[float] = []
    avg_l2s: list[float] = []
    min_l2s: list[float] = []
    cls_acc1_correct = 0
    cls_acc3_correct = 0
    cls_total = 0
    multi_acc1_correct = 0
    multi_total = 0
    n = 0

    with torch.no_grad():
        test_iter = tqdm(
            loader,
            desc=desc,
            leave=False,
            dynamic_ncols=True,
            disable=not show_tqdm,
        )
        for batch in test_iter:
            backbone_kwargs = None
            if "joint_inputs" in batch:
                backbone_kwargs = {
                    "joint_inputs": batch["joint_inputs"],
                    "joint_bsz": int(batch.get("joint_bsz", len(batch.get("text_inputs", [])))),
                }
            with torch.autocast(
                device_type=device.type,
                dtype=amp_dtype,
                enabled=(device.type == "cuda"),
            ):
                out = model(
                    scene_image=batch.get("scene_images", None),
                    head_image=batch.get("head_images", None),
                    text_inputs=batch.get("text_inputs", None),
                    use_softargmax=False,
                    compute_point_soft=False,
                    compute_point_hard=True,
                    backbone_kwargs=backbone_kwargs,
                )

            heatmaps = out["heatmap"][:, 0, :, :].detach().cpu()
            points = out["point_hard"].detach().cpu()
            gt_points_batch: list[torch.Tensor] = batch["gt_points"]
            target_label = batch["target_label"].detach().cpu()
            target_label_ids = batch["target_label_ids"].detach().cpu()
            logits = out.get("logits", None)
            logits_cpu = logits.detach().float().cpu() if logits is not None else None

            for i in range(len(gt_points_batch)):
                gt_t = gt_points_batch[i]
                if gt_t.numel() == 0:
                    continue
                gt_list = [(float(p[0].item()), float(p[1].item())) for p in gt_t]
                pred = (float(points[i, 0].item()), float(points[i, 1].item()))
                hm = heatmaps[i]

                auc = _auc_from_heatmap(hm, gt_t)
                if auc is not None:
                    aucs.append(auc)
                avg_l2s.append(_avg_l2(pred, gt_list))
                min_l2s.append(_min_l2(pred, gt_list))

                if logits_cpu is not None:
                    gt_single = int(target_label[i].item())
                    logits_i = logits_cpu[i]
                    k3 = int(min(3, logits_i.numel()))
                    topk_idx = torch.topk(logits_i, k=max(1, k3), dim=-1).indices
                    pred_top1 = int(topk_idx[0].item())

                    if gt_single >= 0:
                        cls_total += 1
                        cls_acc1_correct += int(pred_top1 == gt_single)
                        cls_acc3_correct += int(bool((topk_idx == gt_single).any().item()))

                    gt_multi = target_label_ids[i]
                    valid_multi = gt_multi[gt_multi >= 0]
                    if valid_multi.numel() > 0:
                        multi_total += 1
                        multi_acc1_correct += int(bool((valid_multi == pred_top1).any().item()))
                n += 1

            if show_tqdm and n > 0:
                test_iter.set_postfix(
                    AUC=f"{(sum(aucs) / max(len(aucs), 1)):.4f}",
                    MinL2=f"{(sum(min_l2s) / max(len(min_l2s), 1)):.4f}",
                )

    if n == 0:
        return {
            "AUC": 0.0,
            "Avg L2": 0.0,
            "Min L2": 0.0,
            "Acc@1": 0.0,
            "Acc@3": 0.0,
            "multiAcc@1": 0.0,
            "num_samples": 0.0,
        }

    return {
        "AUC": float(sum(aucs) / max(len(aucs), 1)),
        "Avg L2": float(sum(avg_l2s) / len(avg_l2s)),
        "Min L2": float(sum(min_l2s) / len(min_l2s)),
        "Acc@1": float(cls_acc1_correct / cls_total) if cls_total > 0 else 0.0,
        "Acc@3": float(cls_acc3_correct / cls_total) if cls_total > 0 else 0.0,
        "multiAcc@1": float(multi_acc1_correct / multi_total) if multi_total > 0 else 0.0,
        "num_samples": float(n),
    }


def print_test_metrics_table(test_metrics: dict[str, float]) -> None:
    rows = [
        ("AUC", float(test_metrics.get("AUC", 0.0))),
        ("Avg L2", float(test_metrics.get("Avg L2", 0.0))),
        ("Min L2", float(test_metrics.get("Min L2", 0.0))),
        ("Acc@1", float(test_metrics.get("Acc@1", 0.0))),
        ("Acc@3", float(test_metrics.get("Acc@3", 0.0))),
        ("multiAcc@1", float(test_metrics.get("multiAcc@1", 0.0))),
    ]
    key_w = max(len(k) for k, _ in rows)
    val_w = 10
    line = "+" + "-" * (key_w + 2) + "+" + "-" * (val_w + 2) + "+"
    print("[TEST] metrics")
    print(line)
    print(f"| {'Metric'.ljust(key_w)} | {'Value'.rjust(val_w)} |")
    print(line)
    for k, v in rows:
        print(f"| {k.ljust(key_w)} | {v:>{val_w}.6f} |")
    print(line)
