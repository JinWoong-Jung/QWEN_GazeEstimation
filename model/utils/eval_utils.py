from __future__ import annotations

import argparse
import json
import math
import re
from pathlib import Path
from typing import Any

import torch
from torch.utils.data import DataLoader
from tqdm.auto import tqdm
from transformers import LogitsProcessor, LogitsProcessorList, StoppingCriteria, StoppingCriteriaList

from .common import to_device
from .gaze_tokens import parse_structured_output_text
from .special_tokens import (
    ANSWER_END,
    OBJECT_END_MARKER,
    OBJECT_START_MARKER,
    POINT_END_MARKER,
    POINT_START_MARKER,
    REASONING_END_MARKER,
    REASONING_START_MARKER,
    format_loc_token,
    format_obj_token,
)
from .loss_utils import compute_answer_loss


# ---------------------------------------------------------------------------
# Stopping criteria — stop when <gaze_obj_end> is generated
# ---------------------------------------------------------------------------

class _GazeObjEndStoppingCriteria(StoppingCriteria):
    """Stop generation once <gaze_obj_end> token appears in generated portion."""

    def __init__(self, prompt_len: int, obj_end_id: int) -> None:
        self.prompt_len = int(prompt_len)
        self.obj_end_id = int(obj_end_id)

    def __call__(
        self,
        input_ids: torch.LongTensor,
        scores: torch.FloatTensor,
        **kwargs: Any,
    ) -> bool:
        generated = input_ids[:, self.prompt_len :]
        return bool((generated == self.obj_end_id).any(dim=1).all().item())


class HybridConstrainedLogitsProcessor(LogitsProcessor):
    """Per-sequence state machine for hybrid constrained decoding.

    Phase 1: forces <|reasoning_start|>, then allows free generation until the
    model naturally emits <|reasoning_end|> or max_reasoning_tokens is reached.
    Phase 2: restricts each structured slot to its allowed token set.

    Designed for use with model.generate() so that KV-cache and batched GPU
    kernels are used — avoids the O(B * T^2) cost of the previous serial
    per-sample approach.
    """

    def __init__(
        self,
        prompt_len: int,
        reasoning_start_id: int,
        reasoning_end_id: int,
        point_start_id: int,
        point_end_id: int,
        object_start_id: int,
        object_end_id: int,
        loc_ids: list[int],
        obj_ids: list[int],
        target_order: str,
        max_reasoning_tokens: int,
    ) -> None:
        self.prompt_len = int(prompt_len)
        self.reasoning_start_id = int(reasoning_start_id)
        self.reasoning_end_id = int(reasoning_end_id)
        self.point_start_id = int(point_start_id)
        self.point_end_id = int(point_end_id)
        self.object_start_id = int(object_start_id)
        self.object_end_id = int(object_end_id)
        self.loc_ids_list = [int(x) for x in loc_ids]
        self.loc_ids_set = set(self.loc_ids_list)
        self.obj_ids_list = [int(x) for x in obj_ids]
        self.obj_ids_set = set(self.obj_ids_list)
        self.target_order = str(target_order)
        self.max_reasoning_tokens = int(max_reasoning_tokens)

    def _point_state(self, post: list[int], next_done_state: str) -> str:
        if not post or post[0] != self.point_start_id:
            return "FORCE_POINT_START"
        point_span = post[1:]
        n_loc = sum(1 for t in point_span if t in self.loc_ids_set)
        if n_loc == 0:
            return "POINT_X"
        if n_loc == 1:
            return "POINT_Y"
        if self.point_end_id not in point_span:
            return "FORCE_POINT_END"
        return next_done_state

    def _object_state(self, post: list[int], next_done_state: str) -> str:
        if not post or post[0] != self.object_start_id:
            return "FORCE_OBJECT_START"
        obj_span = post[1:]
        n_obj = sum(1 for t in obj_span if t in self.obj_ids_set)
        if n_obj == 0:
            return "OBJECT"
        if self.object_end_id not in obj_span:
            return "FORCE_OBJECT_END"
        return next_done_state

    def _get_state(self, gen: list[int]) -> str:
        """Derive decoding state purely from already-generated token ids."""
        if not gen or gen[0] != self.reasoning_start_id:
            return "FORCE_REASONING_START"

        rsn_end_idx = -1
        for k, t in enumerate(gen):
            if k > 0 and t == self.reasoning_end_id:
                rsn_end_idx = k
                break

        if rsn_end_idx == -1:
            n_reasoning = len(gen) - 1  # subtract <|reasoning_start|> itself
            if n_reasoning >= self.max_reasoning_tokens:
                return "FORCE_REASONING_END"
            return "REASONING"

        post = gen[rsn_end_idx + 1 :]
        if self.target_order == "reasoning_point_object":
            point_state = self._point_state(post, "AFTER_POINT")
            if point_state != "AFTER_POINT":
                return point_state
            try:
                point_end_pos = post.index(self.point_end_id)
            except ValueError:
                return "FORCE_POINT_END"
            return self._object_state(post[point_end_pos + 1 :], "DONE")
        else:  # reasoning_object_point
            object_state = self._object_state(post, "AFTER_OBJECT")
            if object_state != "AFTER_OBJECT":
                return object_state
            try:
                object_end_pos = post.index(self.object_end_id)
            except ValueError:
                return "FORCE_OBJECT_END"
            return self._point_state(post[object_end_pos + 1 :], "DONE")

    def __call__(
        self,
        input_ids: torch.LongTensor,
        scores: torch.FloatTensor,
    ) -> torch.FloatTensor:
        NEG_INF = float("-inf")
        _FORCE_MAP = {
            "FORCE_REASONING_START": self.reasoning_start_id,
            "FORCE_REASONING_END": self.reasoning_end_id,
            "FORCE_POINT_START": self.point_start_id,
            "FORCE_POINT_END": self.point_end_id,
            "FORCE_OBJECT_START": self.object_start_id,
            "FORCE_OBJECT_END": self.object_end_id,
        }
        for i in range(int(input_ids.shape[0])):
            gen = input_ids[i, self.prompt_len :].tolist()
            state = self._get_state(gen)

            if state in _FORCE_MAP:
                row = torch.full_like(scores[i], NEG_INF)
                row[_FORCE_MAP[state]] = 0.0
                scores[i] = row
            elif state in ("POINT_X", "POINT_Y"):
                row = torch.full_like(scores[i], NEG_INF)
                for lid in self.loc_ids_list:
                    row[lid] = scores[i, lid]
                scores[i] = row
            elif state == "OBJECT":
                row = torch.full_like(scores[i], NEG_INF)
                for oid in self.obj_ids_list:
                    row[oid] = scores[i, oid]
                scores[i] = row
            # REASONING, DONE: leave scores unchanged

        return scores


def make_gaze_obj_end_stopping_criteria(
    processor: Any,
    prompt_len: int,
    stop_at_object_end: bool = True,
) -> StoppingCriteriaList | None:
    if not stop_at_object_end:
        return None
    try:
        tok = getattr(processor, "tokenizer", None) or processor
        ids = tok.encode(ANSWER_END, add_special_tokens=False)
        if not ids:
            return None
        obj_end_id = int(ids[-1])
        criteria = _GazeObjEndStoppingCriteria(
            prompt_len=int(prompt_len),
            obj_end_id=obj_end_id,
        )
        return StoppingCriteriaList([criteria])
    except Exception:
        return None


# ---------------------------------------------------------------------------
# Decode helpers
# ---------------------------------------------------------------------------

def _single_token_id(tokenizer: Any, token: str) -> int:
    ids = tokenizer.encode(str(token), add_special_tokens=False)
    if len(ids) != 1:
        raise ValueError(
            f"Expected single-token special token, got {token!r} -> {ids}"
        )
    return int(ids[0])


def _loc_token_ids(tokenizer: Any, coord_bins: int) -> list[int]:
    width = max(3, len(str(max(0, int(coord_bins) - 1))))
    return [
        _single_token_id(tokenizer, format_loc_token(i, width))
        for i in range(int(coord_bins))
    ]


def _obj_token_ids(tokenizer: Any, num_classes: int) -> list[int]:
    width = max(3, len(str(max(0, int(num_classes) - 1))))
    return [
        _single_token_id(tokenizer, format_obj_token(i, width))
        for i in range(int(num_classes))
    ]


def _marker_id(tokenizer: Any, marker: str) -> int:
    return _single_token_id(tokenizer, marker)




def constrained_generate_hybrid(
    model: torch.nn.Module,
    joint: dict[str, Any],
    processor: Any,
    num_classes: int,
    coord_bins: int,
    amp_dtype: torch.dtype,
    target_order: str = "reasoning_point_object",
    temperature: float = 1.0,
    max_reasoning_tokens: int = 80,
) -> list[str]:
    """Batched hybrid constrained decoding via HybridConstrainedLogitsProcessor.

    All samples are decoded in a single model.generate() call; per-sample
    state is tracked inside HybridConstrainedLogitsProcessor.  KV-cache and
    batched GPU kernels are used automatically, avoiding the O(B*T^2) cost of
    the previous serial per-sample approach.
    """
    order = str(target_order or "reasoning_point_object").strip()
    if order not in {"reasoning_point_object", "reasoning_object_point"}:
        raise ValueError(
            f"constrained_generate_hybrid: unsupported target_order={order!r}. "
            "Supported: 'reasoning_point_object', 'reasoning_object_point'."
        )

    tokenizer = getattr(processor, "tokenizer", None) or processor
    prompt_len = int(joint["input_ids"].shape[1])
    device = joint["input_ids"].device

    loc_ids = _loc_token_ids(tokenizer, coord_bins=int(coord_bins))
    obj_ids = _obj_token_ids(tokenizer, num_classes=int(num_classes))
    reasoning_start_id = _marker_id(tokenizer, REASONING_START_MARKER)
    reasoning_end_id = _marker_id(tokenizer, REASONING_END_MARKER)
    point_start_id = _marker_id(tokenizer, POINT_START_MARKER)
    point_end_id = _marker_id(tokenizer, POINT_END_MARKER)
    object_start_id = _marker_id(tokenizer, OBJECT_START_MARKER)
    object_end_id = _marker_id(tokenizer, OBJECT_END_MARKER)

    lp = HybridConstrainedLogitsProcessor(
        prompt_len=prompt_len,
        reasoning_start_id=reasoning_start_id,
        reasoning_end_id=reasoning_end_id,
        point_start_id=point_start_id,
        point_end_id=point_end_id,
        object_start_id=object_start_id,
        object_end_id=object_end_id,
        loc_ids=loc_ids,
        obj_ids=obj_ids,
        target_order=order,
        max_reasoning_tokens=int(max_reasoning_tokens),
    )

    # Budget: reasoning start/end + free reasoning + point span(5) + object span(3).
    max_new = int(max_reasoning_tokens) + 10

    with torch.autocast(
        device_type=device.type, dtype=amp_dtype, enabled=(device.type == "cuda")
    ):
        generated_ids = model.generate(
            joint_inputs=joint,
            max_new_tokens=max_new,
            do_sample=False,
            logits_processor=LogitsProcessorList([lp]),
        )

    return decode_generated(
        processor=processor,
        generated_ids=generated_ids.detach().cpu(),
        input_ids=joint["input_ids"].detach().cpu(),
        attention_mask=joint.get("attention_mask", None),
        num_return_sequences=1,
    )


def _append_token_to_joint(
    joint: dict[str, Any], token_ids: torch.LongTensor
) -> dict[str, Any]:
    out = dict(joint)
    token_ids = token_ids.to(device=joint["input_ids"].device, dtype=torch.long)
    if token_ids.dim() == 1:
        token_ids = token_ids[:, None]
    out["input_ids"] = torch.cat([joint["input_ids"], token_ids], dim=1)
    if "attention_mask" in joint and torch.is_tensor(joint["attention_mask"]):
        new_mask = torch.ones_like(token_ids, dtype=joint["attention_mask"].dtype)
        out["attention_mask"] = torch.cat([joint["attention_mask"], new_mask], dim=1)
    return out


def _select_next_from_allowed(
    model: torch.nn.Module,
    joint: dict[str, Any],
    allowed_token_ids: list[int],
    amp_dtype: torch.dtype,
    temperature: float = 1.0,
    selection_strategy: str = "argmax",
) -> tuple[torch.LongTensor, torch.Tensor]:
    device = joint["input_ids"].device
    allowed = torch.tensor(allowed_token_ids, device=device, dtype=torch.long)
    with torch.autocast(
        device_type=device.type,
        dtype=amp_dtype,
        enabled=(device.type == "cuda"),
    ):
        out = model(joint_inputs=joint, use_cache=False)
        logits = out["logits"][:, -1, :]  # [B, vocab]
    allowed_logits = logits.index_select(dim=-1, index=allowed)  # [B, K]
    temp = max(float(temperature), 1e-6)
    allowed_probs = torch.softmax(allowed_logits / temp, dim=-1)
    strategy = str(selection_strategy or "argmax").strip().lower()
    if strategy in {"argmax", "max"}:
        selected_idx = torch.argmax(allowed_probs, dim=-1)  # [B]
    elif strategy in {"round_expectation", "expected_round", "expectation_round"}:
        bins = torch.arange(int(allowed_probs.shape[-1]), device=device, dtype=torch.float32)
        expected_idx = (allowed_probs.to(dtype=torch.float32) * bins).sum(dim=-1)
        selected_idx = expected_idx.round().to(dtype=torch.long).clamp_(0, int(allowed_probs.shape[-1]) - 1)
    else:
        raise ValueError(
            f"unsupported constrained loc selection_strategy={selection_strategy!r}; "
            "expected 'argmax' or 'round_expectation'"
        )
    selected_token_ids = allowed.index_select(dim=0, index=selected_idx)  # [B]
    return selected_token_ids, allowed_probs


def _expected_xy_from_loc_probs(
    x_probs: torch.Tensor,
    y_probs: torch.Tensor,
    coord_bins: int,
) -> list[tuple[float, float]]:
    if x_probs.dim() != 2 or y_probs.dim() != 2:
        raise ValueError(
            f"expected [B, C] loc probabilities, got x={tuple(x_probs.shape)} y={tuple(y_probs.shape)}"
        )
    if tuple(x_probs.shape) != tuple(y_probs.shape):
        raise ValueError(
            f"x/y loc probability shapes must match, got x={tuple(x_probs.shape)} y={tuple(y_probs.shape)}"
        )
    denom = float(max(1, int(coord_bins) - 1))
    bins = torch.arange(int(x_probs.shape[-1]), device=x_probs.device, dtype=torch.float32)
    x_exp = (x_probs.to(dtype=torch.float32) * bins).sum(dim=-1) / denom
    y_exp = (y_probs.to(dtype=torch.float32) * bins).sum(dim=-1) / denom
    xy = torch.stack([x_exp, y_exp], dim=-1).detach().cpu()
    return [(float(xy[i, 0].item()), float(xy[i, 1].item())) for i in range(int(xy.shape[0]))]


def constrained_generate_structured(
    model: torch.nn.Module,
    joint: dict[str, Any],
    processor: Any,
    num_classes: int,
    coord_bins: int,
    amp_dtype: torch.dtype,
    target_order: str = "point_object",
    temperature: float = 1.0,
    max_reasoning_tokens: int = 80,
    return_expected_xy: bool = False,
    loc_selection_strategy: str = "argmax",
) -> list[str] | tuple[list[str], list[tuple[float, float] | None]]:
    """Slot-level constrained decoding using Qwen LM logits directly.

    Supported target_order values:
      "point_object", "object_point"          — direct constrained decoding
      "reasoning_point_object",               — hybrid: free reasoning +
      "reasoning_object_point"                  constrained point/object slots
    """
    order = str(target_order or "point_object").strip()
    if order in {"reasoning_point_object", "reasoning_object_point"}:
        texts = constrained_generate_hybrid(
            model=model,
            joint=joint,
            processor=processor,
            num_classes=int(num_classes),
            coord_bins=int(coord_bins),
            amp_dtype=amp_dtype,
            target_order=order,
            temperature=float(temperature),
            max_reasoning_tokens=int(max_reasoning_tokens),
        )
        if bool(return_expected_xy):
            return texts, [None for _ in texts]
        return texts
    if order not in {"point_object", "object_point"}:
        raise ValueError(
            f"constrained_generate_structured: unsupported target_order={order!r}. "
            "Supported: 'point_object', 'object_point', "
            "'reasoning_point_object', 'reasoning_object_point'."
        )

    tokenizer = getattr(processor, "tokenizer", None) or processor
    device = joint["input_ids"].device
    bsz = int(joint["input_ids"].shape[0])

    loc_ids = _loc_token_ids(tokenizer, coord_bins=int(coord_bins))
    obj_ids = _obj_token_ids(tokenizer, num_classes=int(num_classes))

    point_start_id = _marker_id(tokenizer, POINT_START_MARKER)
    point_end_id = _marker_id(tokenizer, POINT_END_MARKER)
    object_start_id = _marker_id(tokenizer, OBJECT_START_MARKER)
    object_end_id = _marker_id(tokenizer, OBJECT_END_MARKER)

    point_start = torch.full((bsz,), point_start_id, device=device, dtype=torch.long)
    point_end = torch.full((bsz,), point_end_id, device=device, dtype=torch.long)
    object_start = torch.full((bsz,), object_start_id, device=device, dtype=torch.long)
    object_end = torch.full((bsz,), object_end_id, device=device, dtype=torch.long)

    cur = dict(joint)
    generated_steps: list[torch.LongTensor] = []
    x_probs: torch.Tensor | None = None
    y_probs: torch.Tensor | None = None

    if order == "point_object":
        # <|point_start|><loc_x><loc_y><|point_end|><|object_start|><obj_k><|object_end|>
        cur = _append_token_to_joint(cur, point_start)
        generated_steps.append(point_start)

        x_tok, x_probs = _select_next_from_allowed(
            model, cur, loc_ids,
            amp_dtype=amp_dtype,
            temperature=temperature,
            selection_strategy=loc_selection_strategy,
        )
        cur = _append_token_to_joint(cur, x_tok)
        generated_steps.append(x_tok)

        y_tok, y_probs = _select_next_from_allowed(
            model, cur, loc_ids,
            amp_dtype=amp_dtype,
            temperature=temperature,
            selection_strategy=loc_selection_strategy,
        )
        cur = _append_token_to_joint(cur, y_tok)
        generated_steps.append(y_tok)

        cur = _append_token_to_joint(cur, point_end)
        generated_steps.append(point_end)

        cur = _append_token_to_joint(cur, object_start)
        generated_steps.append(object_start)

        obj_tok, _ = _select_next_from_allowed(model, cur, obj_ids, amp_dtype=amp_dtype, temperature=temperature)
        generated_steps.append(obj_tok)
        generated_steps.append(object_end)

    else:
        # order == "object_point"
        # <|object_start|><obj_k><|object_end|><|point_start|><loc_x><loc_y><|point_end|>
        cur = _append_token_to_joint(cur, object_start)
        generated_steps.append(object_start)

        obj_tok, _ = _select_next_from_allowed(model, cur, obj_ids, amp_dtype=amp_dtype, temperature=temperature)
        cur = _append_token_to_joint(cur, obj_tok)
        generated_steps.append(obj_tok)

        cur = _append_token_to_joint(cur, object_end)
        generated_steps.append(object_end)

        cur = _append_token_to_joint(cur, point_start)
        generated_steps.append(point_start)

        x_tok, x_probs = _select_next_from_allowed(
            model, cur, loc_ids,
            amp_dtype=amp_dtype,
            temperature=temperature,
            selection_strategy=loc_selection_strategy,
        )
        cur = _append_token_to_joint(cur, x_tok)
        generated_steps.append(x_tok)

        y_tok, y_probs = _select_next_from_allowed(
            model, cur, loc_ids,
            amp_dtype=amp_dtype,
            temperature=temperature,
            selection_strategy=loc_selection_strategy,
        )
        generated_steps.append(y_tok)
        generated_steps.append(point_end)

    gen_ids = torch.stack(generated_steps, dim=1)  # [B, T]
    texts = tokenizer.batch_decode(gen_ids.detach().cpu(), skip_special_tokens=False)
    stripped = [str(t).strip() for t in texts]
    if bool(return_expected_xy):
        if x_probs is None or y_probs is None:
            return stripped, [None for _ in stripped]
        return stripped, _expected_xy_from_loc_probs(x_probs, y_probs, coord_bins=int(coord_bins))
    return stripped


def decode_generated(
    processor: Any,
    generated_ids: torch.Tensor,
    input_ids: torch.Tensor,
    attention_mask: torch.Tensor | None,
    num_return_sequences: int = 1,
) -> list[str]:
    out: list[str] = []
    tok = getattr(processor, "tokenizer", None)
    nrs = max(1, int(num_return_sequences))
    bsz = int(input_ids.shape[0]) if torch.is_tensor(input_ids) and input_ids.dim() >= 2 else 0
    for i in range(generated_ids.shape[0]):
        src_i = (i // nrs) if bsz > 0 else i
        src_i = min(max(0, int(src_i)), bsz - 1) if bsz > 0 else src_i
        start = int(input_ids.shape[1])
        new_tokens = generated_ids[i, start:]
        txt = (
            tok.decode(new_tokens, skip_special_tokens=False)
            if tok is not None
            else str(new_tokens.tolist())
        )
        # Keep parser-visible schema markers intact. The parser accepts <|im_end|>
        # as an optional trailing EOS, and gaze markers are part of the output schema.
        # Strip other Qwen chat markers (e.g. <|endoftext|>).
        txt = re.sub(
            r"<\|(?!(?:im_start|im_end|gaze_|point_|object_|reasoning_))[^>]+?\|>",
            "",
            str(txt),
        ).strip()
        out.append(str(txt))
    return out


def l2_stats(pred_xy: tuple[float, float], gt_points: torch.Tensor) -> tuple[float, float] | None:
    if (not torch.is_tensor(gt_points)) or gt_points.numel() < 2:
        return None
    if int(gt_points.numel()) % 2 != 0:
        return None
    pts = gt_points.to(dtype=torch.float32).view(-1, 2)
    if int(pts.shape[0]) <= 0:
        return None
    px, py = float(pred_xy[0]), float(pred_xy[1])
    dists = [
        math.sqrt((px - float(pts[j, 0].item())) ** 2 + (py - float(pts[j, 1].item())) ** 2)
        for j in range(int(pts.shape[0]))
    ]
    return float(sum(dists) / len(dists)), float(min(dists))


def valid_label_ids(label_ids: torch.Tensor | list[int] | tuple[int, ...] | None) -> list[int]:
    if label_ids is None:
        return []
    if torch.is_tensor(label_ids):
        vals = label_ids.detach().cpu().flatten().tolist()
    else:
        vals = list(label_ids)
    out: list[int] = []
    for v in vals:
        iv = int(v)
        if iv >= 0 and iv not in out:
            out.append(iv)
    return out


# ---------------------------------------------------------------------------
# Evaluation
# ---------------------------------------------------------------------------

def run_eval(
    model: torch.nn.Module,
    loader: DataLoader,
    device: torch.device,
    amp_dtype: torch.dtype,
    loss_weights: dict[str, float] | None = None,
    loc_token_ids: torch.Tensor | None = None,
    gaussian_point_sigma: float = 0.0,
    show_tqdm: bool = True,
    desc: str = "Eval",
) -> dict[str, float]:
    model.eval()
    if loss_weights is None:
        loss_weights = {"point": 1.0, "object": 1.0, "format": 0.25}

    loss_sum = 0.0
    pt_sum = 0.0
    obj_sum = 0.0
    fmt_sum = 0.0
    count = 0

    with torch.no_grad():
        it = tqdm(loader, desc=desc, leave=False, dynamic_ncols=True, disable=not show_tqdm)
        for batch in it:
            joint = to_device(batch["joint_inputs"], device=device)
            labels = batch["labels"].to(device)
            if torch.all(labels.eq(-100)):
                continue
            bsz = int(labels.shape[0])

            with torch.autocast(
                device_type=device.type, dtype=amp_dtype, enabled=(device.type == "cuda")
            ):
                out = model(joint_inputs=joint, use_cache=False)

            losses = compute_answer_loss(
                logits=out.get("logits", None),
                labels=labels,
                loss_mask_point=batch.get("loss_mask_point", None),
                loss_mask_object=batch.get("loss_mask_object", None),
                loss_mask_format=batch.get("loss_mask_format", None),
                loss_mask_reasoning=batch.get("loss_mask_reasoning", None),
                weight_point=float(loss_weights.get("point", 1.0)),
                weight_object=float(loss_weights.get("object", 1.0)),
                weight_format=float(loss_weights.get("format", 0.25)),
                weight_reasoning=float(loss_weights.get("reasoning", 0.3)),
                loc_token_ids=loc_token_ids.to(device) if loc_token_ids is not None else None,
                gaussian_sigma=float(gaussian_point_sigma),
            )
            loss_sum += float(losses["loss"].detach().item()) * float(bsz)
            pt_sum += float(losses["loss_point"].detach().item()) * float(bsz)
            obj_sum += float(losses["loss_object"].detach().item()) * float(bsz)
            fmt_sum += float(losses["loss_format"].detach().item()) * float(bsz)
            count += bsz
            if show_tqdm:
                it.set_postfix(loss=f"{(loss_sum / max(count, 1)):.4f}")

    if count <= 0:
        return {
            "loss": 0.0,
            "loss_point": 0.0,
            "loss_object": 0.0,
            "loss_format": 0.0,
        }
    d = float(count)
    return {
        "loss": float(loss_sum / d),
        "loss_point": float(pt_sum / d),
        "loss_object": float(obj_sum / d),
        "loss_format": float(fmt_sum / d),
    }


def run_test_metrics(
    model: torch.nn.Module,
    loader: DataLoader,
    device: torch.device,
    amp_dtype: torch.dtype,
    processor: Any,
    num_classes: int,
    coord_bins: int = 1000,
    show_tqdm: bool = True,
    desc: str = "Test",
    max_new_tokens: int = 8,
    num_beams: int = 1,
    repetition_penalty: float = 1.0,
    no_repeat_ngram_size: int = 0,
    stop_at_object_end: bool = True,
    constrained_decoding: bool = False,
    constrained_target_order: str = "point_object",
    constrained_temperature: float = 1.0,
    constrained_loc_decoding: str = "argmax",
    max_reasoning_tokens: int = 80,
    include_l2_breakdown: bool = True,
) -> dict[str, float]:
    model.eval()

    total = 0           # all samples
    valid_total = 0     # samples with target_text_valid > 0
    format_valid_total = 0  # samples with target_format_valid > 0 (denominator for FormatValid/ExtraTextRate)
    point_bin_den = 0   # samples with target_point_valid > 0 (denominator for PointBinExact)
    format_valid_count = 0
    extra_text_count = 0
    avg_l2_sum = 0.0
    min_l2_sum = 0.0
    l2_den = 0
    expected_avg_l2_sum = 0.0
    expected_min_l2_sum = 0.0
    expected_l2_den = 0
    point_bin_exact = 0
    object_acc = 0
    multi_acc_at_1 = 0
    obj_den = 0
    multi_obj_den = 0
    beam_k = max(1, int(num_beams))

    with torch.no_grad():
        it = tqdm(loader, desc=desc, leave=False, dynamic_ncols=True, disable=not show_tqdm)
        for batch in it:
            joint = to_device(batch["joint_inputs"], device=device)
            target_texts = [str(x) for x in batch.get("target_text", [])]
            target_valid = batch.get("target_text_valid", None)
            target_point_valid = batch.get("target_point_valid", None)
            target_object_valid = batch.get("target_object_valid", None)
            target_format_valid = batch.get("target_format_valid", None)
            target_point_bin = batch.get("target_point_bin", None)
            target_object_id = batch.get("target_object_id", None)
            target_label_ids = batch.get("target_label_ids", None)
            gt_points = batch.get("gt_points", None)
            if target_valid is None:
                target_valid = torch.ones((len(target_texts),), dtype=torch.float32)
            target_valid = target_valid.to(dtype=torch.float32)
            if target_point_valid is None:
                target_point_valid = target_valid
            if target_object_valid is None:
                target_object_valid = target_valid
            if target_format_valid is None:
                target_format_valid = target_valid
            target_point_valid = target_point_valid.to(dtype=torch.float32)
            target_object_valid = target_object_valid.to(dtype=torch.float32)
            target_format_valid = target_format_valid.to(dtype=torch.float32)
            expected_points: list[tuple[float, float] | None] | None = None

            if bool(constrained_decoding):
                with torch.no_grad():
                    constrained_out = constrained_generate_structured(
                        model=model,
                        joint=joint,
                        processor=processor,
                        num_classes=int(num_classes),
                        coord_bins=int(coord_bins),
                        amp_dtype=amp_dtype,
                        target_order=str(constrained_target_order),
                        temperature=float(constrained_temperature),
                        max_reasoning_tokens=int(max_reasoning_tokens),
                        return_expected_xy=True,
                        loc_selection_strategy=str(constrained_loc_decoding),
                    )
                    preds, expected_points = constrained_out
            else:
                prompt_len = int(joint["input_ids"].shape[1])
                stopping = make_gaze_obj_end_stopping_criteria(
                    processor, prompt_len, stop_at_object_end=bool(stop_at_object_end)
                )

                with torch.autocast(
                    device_type=device.type, dtype=amp_dtype, enabled=(device.type == "cuda")
                ):
                    generate_kwargs: dict[str, Any] = dict(
                        max_new_tokens=max_new_tokens,
                        do_sample=False,
                        num_beams=beam_k,
                        repetition_penalty=max(float(repetition_penalty), 1.0),
                        no_repeat_ngram_size=max(0, int(no_repeat_ngram_size)),
                    )
                    if stopping is not None:
                        generate_kwargs["stopping_criteria"] = stopping
                    generated_ids = model.generate(joint_inputs=joint, **generate_kwargs)

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
            bsz = len(target_texts)
            preds = preds[:bsz]
            if expected_points is not None:
                expected_points = expected_points[:bsz]

            for i in range(bsz):
                total += 1
                pred = preds[i] if i < len(preds) else ""
                parsed = parse_structured_output_text(pred, int(num_classes), coord_bins=int(coord_bins))
                is_text_valid = i < int(target_valid.numel()) and float(target_valid[i].item()) > 0.0
                is_point_valid = i < int(target_point_valid.numel()) and float(target_point_valid[i].item()) > 0.0
                is_object_valid = i < int(target_object_valid.numel()) and float(target_object_valid[i].item()) > 0.0
                is_format_valid_gt = i < int(target_format_valid.numel()) and float(target_format_valid[i].item()) > 0.0
                if is_text_valid:
                    valid_total += 1
                if is_format_valid_gt:
                    format_valid_total += 1
                if is_point_valid:
                    point_bin_den += 1

                # FormatValid / ExtraTextRate: denominator = format_valid_total
                if is_format_valid_gt:
                    if parsed["has_extra_text"]:
                        extra_text_count += 1
                    if parsed["valid_format"]:
                        format_valid_count += 1

                # point L2: valid-GT samples only (point GT is meaningful)
                if is_point_valid and parsed["point_xy"] is not None and isinstance(gt_points, list) and i < len(gt_points):
                    stats = l2_stats(parsed["point_xy"], gt_points[i])
                    if stats is not None:
                        avg_l2_sum += float(stats[0])
                        min_l2_sum += float(stats[1])
                        l2_den += 1
                if (
                    is_point_valid
                    and expected_points is not None
                    and i < len(expected_points)
                    and expected_points[i] is not None
                    and isinstance(gt_points, list)
                    and i < len(gt_points)
                ):
                    expected_stats = l2_stats(expected_points[i], gt_points[i])
                    if expected_stats is not None:
                        expected_avg_l2_sum += float(expected_stats[0])
                        expected_min_l2_sum += float(expected_stats[1])
                        expected_l2_den += 1

                # point bin exact match: denominator = point_bin_den (all is_point_valid samples,
                # format failures count as misses — consistent with ObjectAcc gating on obj_den)
                if (
                    is_point_valid
                    and parsed["point_bins"] is not None
                    and torch.is_tensor(target_point_bin)
                    and i < int(target_point_bin.shape[0])
                ):
                    gt_bx = int(target_point_bin[i, 0].item())
                    gt_by = int(target_point_bin[i, 1].item())
                    px, py = parsed["point_bins"]
                    if int(px) == gt_bx and int(py) == gt_by:
                        point_bin_exact += 1

                # object accuracy: valid-GT samples only
                if is_object_valid and torch.is_tensor(target_object_id) and i < int(target_object_id.shape[0]):
                    gt_obj = int(target_object_id[i].item())
                    obj_den += 1
                    if parsed["object_id"] is not None and int(parsed["object_id"]) == gt_obj:
                        object_acc += 1

                if is_object_valid and torch.is_tensor(target_label_ids) and i < int(target_label_ids.shape[0]):
                    gt_obj_ids = valid_label_ids(target_label_ids[i])
                    if gt_obj_ids:
                        multi_obj_den += 1
                        if parsed["object_id"] is not None and int(parsed["object_id"]) in gt_obj_ids:
                            multi_acc_at_1 += 1

            if show_tqdm and l2_den > 0:
                it.set_postfix(
                    l2=f"{(avg_l2_sum / max(l2_den, 1)):.4f}",
                    fmt=f"{(format_valid_count / max(format_valid_total, 1)):.3f}",
                )

    _L2_SENTINEL = math.sqrt(2.0)  # max possible L2 for normalized coords (0,0)→(1,1)
    if total <= 0:
        out = {
            "FormatValid": 0.0,
            "Dist": _L2_SENTINEL,
            "PointBinExact": 0.0,
            "ObjectAcc": 0.0,
            "MultiAcc@1": 0.0,
            "ExtraTextRate": 0.0,
            "PointL2ValidFrac": 0.0,
            "num_samples": 0.0,
            "num_valid_samples": 0.0,
        }
        if bool(include_l2_breakdown):
            out["Avg L2"] = _L2_SENTINEL
            out["Min L2"] = _L2_SENTINEL
        return out

    dist = float(avg_l2_sum / l2_den) if l2_den > 0 else _L2_SENTINEL
    min_l2 = float(min_l2_sum / l2_den) if l2_den > 0 else _L2_SENTINEL
    out = {
        "FormatValid": float(format_valid_count / max(format_valid_total, 1)),
        "Dist": dist,
        "PointBinExact": float(point_bin_exact / max(point_bin_den, 1)),
        "ObjectAcc": float(object_acc / max(obj_den, 1)),
        "MultiAcc@1": float(multi_acc_at_1 / max(multi_obj_den, 1)),
        "ExtraTextRate": float(extra_text_count / max(format_valid_total, 1)),
        "PointL2ValidFrac": float(l2_den / max(point_bin_den, 1)),
        "num_samples": float(total),
        "num_valid_samples": float(valid_total),
    }
    if bool(include_l2_breakdown):
        out["Avg L2"] = dist
        out["Min L2"] = min_l2
    if expected_l2_den > 0:
        expected_dist = float(expected_avg_l2_sum / expected_l2_den)
        expected_min_l2 = float(expected_min_l2_sum / expected_l2_den)
        out["DistExpected"] = expected_dist
        out["ExpectedL2ValidFrac"] = float(expected_l2_den / max(point_bin_den, 1))
        if bool(include_l2_breakdown):
            out["Avg L2 Expected"] = expected_dist
            out["Min L2 Expected"] = expected_min_l2
    return out


def collect_generation_samples(
    model: torch.nn.Module,
    loader: DataLoader,
    device: torch.device,
    amp_dtype: torch.dtype,
    processor: Any,
    num_classes: int,
    coord_bins: int = 1000,
    show_tqdm: bool = True,
    desc: str = "GenerationPreview",
    max_new_tokens: int = 8,
    num_beams: int = 1,
    repetition_penalty: float = 1.0,
    no_repeat_ngram_size: int = 0,
    max_samples: int = 8,
    stop_at_object_end: bool = True,
    constrained_decoding: bool = False,
    constrained_target_order: str = "point_object",
    constrained_temperature: float = 1.0,
    constrained_loc_decoding: str = "argmax",
    max_reasoning_tokens: int = 80,
) -> list[dict[str, Any]]:
    limit = max(0, int(max_samples))
    if limit <= 0:
        return []

    model.eval()
    previews: list[dict[str, Any]] = []
    beam_k = max(1, int(num_beams))

    with torch.no_grad():
        it = tqdm(loader, desc=desc, leave=False, dynamic_ncols=True, disable=not show_tqdm)
        for batch in it:
            joint = to_device(batch["joint_inputs"], device=device)
            target_texts = [str(x) for x in batch.get("target_text", [])]
            target_valid = batch.get("target_text_valid", None)
            target_label = batch.get("target_label", None)
            target_object_id = batch.get("target_object_id", None)
            target_point_bin = batch.get("target_point_bin", None)
            gt_points = batch.get("gt_points", None)
            prompt_texts = [str(x) for x in batch.get("text_input", [])]
            target_label_texts = [str(x) for x in batch.get("target_label_text", [])]
            image_rels = [str(x) for x in batch.get("image_rel", [])]
            if target_valid is None:
                target_valid = torch.ones((len(target_texts),), dtype=torch.float32)
            target_valid = target_valid.to(dtype=torch.float32)

            if bool(constrained_decoding):
                preds = constrained_generate_structured(
                    model=model,
                    joint=joint,
                    processor=processor,
                    num_classes=int(num_classes),
                    coord_bins=int(coord_bins),
                    amp_dtype=amp_dtype,
                    target_order=str(constrained_target_order),
                    temperature=float(constrained_temperature),
                    loc_selection_strategy=str(constrained_loc_decoding),
                    max_reasoning_tokens=int(max_reasoning_tokens),
                )
            else:
                prompt_len = int(joint["input_ids"].shape[1])
                stopping = make_gaze_obj_end_stopping_criteria(
                    processor, prompt_len, stop_at_object_end=bool(stop_at_object_end)
                )
                with torch.autocast(
                    device_type=device.type, dtype=amp_dtype, enabled=(device.type == "cuda")
                ):
                    generate_kwargs: dict[str, Any] = dict(
                        max_new_tokens=max_new_tokens,
                        do_sample=False,
                        num_beams=beam_k,
                        repetition_penalty=max(float(repetition_penalty), 1.0),
                        no_repeat_ngram_size=max(0, int(no_repeat_ngram_size)),
                    )
                    if stopping is not None:
                        generate_kwargs["stopping_criteria"] = stopping
                    generated_ids = model.generate(joint_inputs=joint, **generate_kwargs)

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
            bsz = len(target_texts)
            preds = preds[:bsz]

            for i in range(bsz):
                pred = preds[i] if i < len(preds) else ""
                parsed = parse_structured_output_text(pred, int(num_classes), coord_bins=int(coord_bins))

                serialized_gt_points: list[list[float]] = []
                stats = None
                if isinstance(gt_points, list) and i < len(gt_points):
                    if parsed["point_xy"] is not None:
                        stats = l2_stats(parsed["point_xy"], gt_points[i])
                    if torch.is_tensor(gt_points[i]) and gt_points[i].numel() >= 2:
                        pts = gt_points[i].detach().cpu().to(dtype=torch.float32).view(-1, 2)
                        serialized_gt_points = [
                            [float(pts[j, 0].item()), float(pts[j, 1].item())]
                            for j in range(int(pts.shape[0]))
                        ]

                gt_obj_id = -1
                if torch.is_tensor(target_object_id) and i < int(target_object_id.shape[0]):
                    gt_obj_id = int(target_object_id[i].item())
                gt_label_id = -1
                if torch.is_tensor(target_label) and i < int(target_label.shape[0]):
                    gt_label_id = int(target_label[i].item())
                gt_pt_bins: list[int] = []
                if torch.is_tensor(target_point_bin) and i < int(target_point_bin.shape[0]):
                    gt_pt_bins = [int(target_point_bin[i, 0].item()), int(target_point_bin[i, 1].item())]

                target_is_valid = (
                    bool(float(target_valid[i].item()) > 0.0)
                    if i < int(target_valid.numel())
                    else False
                )
                previews.append(
                    {
                        "sample_index": int(len(previews)),
                        "image_rel": image_rels[i] if i < len(image_rels) else "",
                        "prompt_text": prompt_texts[i] if i < len(prompt_texts) else "",
                        "target_text": target_texts[i] if i < len(target_texts) else "",
                        "target_text_valid": target_is_valid,
                        "generated_text": pred,
                        "parsed": {
                            "valid_format": bool(parsed["valid_format"]),
                            "has_extra_text": bool(parsed["has_extra_text"]),
                            "point_bins": list(parsed["point_bins"]) if parsed["point_bins"] else None,
                            "point_xy": list(parsed["point_xy"]) if parsed["point_xy"] else None,
                            "object_id": int(parsed["object_id"]) if parsed["object_id"] is not None else None,
                        },
                        "avg_l2": float(stats[0]) if stats is not None else None,
                        "min_l2": float(stats[1]) if stats is not None else None,
                        "gt_points": serialized_gt_points,
                        "gt_point_bins": gt_pt_bins,
                        "gt_object_id": gt_obj_id,
                        "gt_label_id": gt_label_id,
                        "target_label_text": (
                            target_label_texts[i] if i < len(target_label_texts) else ""
                        ),
                    }
                )
                if len(previews) >= limit:
                    return previews

    return previews


def print_generation_samples(samples: list[dict[str, Any]]) -> None:
    if not samples:
        print("[TEST] generation preview: no samples collected.")
        return
    print(f"[TEST] generation preview ({len(samples)} samples)")
    for sample in samples:
        idx = int(sample.get("sample_index", -1))
        image_rel = str(sample.get("image_rel", ""))
        parsed = sample.get("parsed", {})
        print(f"[SAMPLE {idx}] image={image_rel}")
        print(f"  target   : {sample.get('target_text', '')}")
        print(f"  generated: {sample.get('generated_text', '')}")
        print(f"  fmt_valid: {parsed.get('valid_format', False)}  "
              f"extra_text: {parsed.get('has_extra_text', False)}")
        print(f"  pred_bins: {parsed.get('point_bins', None)}  "
              f"pred_xy: {parsed.get('point_xy', None)}")
        print(f"  pred_obj : {parsed.get('object_id', None)}  "
              f"gt_obj: {sample.get('gt_object_id', -1)}")
        print(f"  gt_points: {sample.get('gt_points', [])}  "
              f"avg_l2: {sample.get('avg_l2', None)}  "
              f"min_l2: {sample.get('min_l2', None)}")


def print_test_metrics_table(test_metrics: dict[str, float]) -> None:
    rows = [
        ("FormatValid", float(test_metrics.get("FormatValid", 0.0))),
        ("Avg L2", float(test_metrics.get("Avg L2", test_metrics.get("Dist", 0.0)))),
        ("Min L2", float(test_metrics.get("Min L2", test_metrics.get("Dist", 0.0)))),
        ("PointBinExact", float(test_metrics.get("PointBinExact", 0.0))),
        ("ObjectAcc", float(test_metrics.get("ObjectAcc", 0.0))),
        ("MultiAcc@1", float(test_metrics.get("MultiAcc@1", 0.0))),
        ("ExtraTextRate", float(test_metrics.get("ExtraTextRate", 0.0))),
        ("num_samples", float(test_metrics.get("num_samples", 0.0))),
    ]
    if "DistExpected" in test_metrics:
        rows[3:3] = [
            ("Avg L2 Expected", float(test_metrics.get("Avg L2 Expected", test_metrics.get("DistExpected", 0.0)))),
            ("Min L2 Expected", float(test_metrics.get("Min L2 Expected", test_metrics.get("DistExpected", 0.0)))),
        ]
    key_w = max(len(k) for k, _ in rows)
    val_w = 12
    line = "+" + "-" * (key_w + 2) + "+" + "-" * (val_w + 2) + "+"
    print("[TEST] metrics")
    print(line)
    print(f"| {'Metric'.ljust(key_w)} | {'Value'.rjust(val_w)} |")
    print(line)
    for k, v in rows:
        if k.startswith("num_"):
            print(f"| {k.ljust(key_w)} | {v:>{val_w}.0f} |")
        else:
            print(f"| {k.ljust(key_w)} | {v:>{val_w}.6f} |")
    print(line)


def maybe_save_generation_preview(
    *,
    args: argparse.Namespace,
    out_dir: Path,
    model: torch.nn.Module,
    loader: DataLoader,
    device: torch.device,
    amp_dtype: torch.dtype,
    processor: Any,
    num_classes: int,
    coord_bins: int,
    preview_attr: str = "preview_test_samples",
    filename: str = "test_generation_preview.json",
    desc: str = "Test preview",
    log_prefix: str = "TEST",
) -> None:
    preview_n = max(0, int(getattr(args, preview_attr, 0)))
    if preview_n <= 0:
        return

    preview_samples = collect_generation_samples(
        model=model,
        loader=loader,
        device=device,
        amp_dtype=amp_dtype,
        processor=processor,
        num_classes=int(num_classes),
        coord_bins=int(coord_bins),
        show_tqdm=bool(args.show_tqdm),
        desc=desc,
        max_new_tokens=int(args.generation_max_new_tokens),
        num_beams=int(getattr(args, "generation_num_beams", 1)),
        repetition_penalty=float(getattr(args, "repetition_penalty", 1.0)),
        no_repeat_ngram_size=int(getattr(args, "no_repeat_ngram_size", 0)),
        max_samples=preview_n,
        stop_at_object_end=bool(getattr(args, "generation_stop_at_object_end", True)),
        constrained_decoding=bool(getattr(args, "constrained_decoding", False)),
        constrained_target_order=str(getattr(args, "constrained_target_order", "point_object")),
        constrained_temperature=float(getattr(args, "constrained_temperature", 1.0)),
        constrained_loc_decoding=str(getattr(args, "constrained_loc_decoding", "argmax")),
        max_reasoning_tokens=int(getattr(args, "max_reasoning_tokens", 80)),
    )
    preview_path = out_dir / filename
    preview_path.write_text(
        json.dumps(preview_samples, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print(f"[{log_prefix}] saved generation preview: {preview_path} (samples={len(preview_samples)})")
    print_generation_samples(preview_samples)
