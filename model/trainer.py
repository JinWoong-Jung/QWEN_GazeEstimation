from __future__ import annotations

import argparse
import json
import math
import os
import random
import time
from pathlib import Path
from typing import Any

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm.auto import tqdm
from transformers import (
    AutoModelForImageTextToText,
    AutoProcessor,
    get_cosine_schedule_with_warmup,
)

from peft import LoraConfig, PeftModel, TaskType, get_peft_model

from .utils.checkpoint import save_checkpoint
from .utils.config_parser import build_parser, load_yaml_config
from .utils.data_utils import (
    build_vocab_embedding_matrix,
    load_label_map,
    load_label_text_map,
    load_records,
    load_test_groups,
    load_test_label_map,
    load_vocab2id,
)
from .datasets import (
    GazeDataset,
    GazeTestDataset,
    collate_batch,
    collate_test_batch,
)
from .utils.eval_utils import (
    print_test_metrics_table,
    run_eval,
    run_test_metrics,
)
from .model import QwenGazeIntegratedModel
from .utils.wandb_utils import finish_wandb, init_wandb


ROOT = Path(__file__).resolve().parents[1]
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")


def resolve_path(path: str) -> Path:
    p = Path(path)
    return p if p.is_absolute() else ROOT / p


def set_seed(seed: int) -> None:
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)



class QwenBackboneAdapter(nn.Module):
    def __init__(
        self,
        qwen_model: nn.Module,
        processor: Any,
        scene_tokens: int | None,
        head_tokens: int,
        text_tokens: int,
        max_text_length: int = 128,
        head_text: str = "Target subject head crop.",
    ) -> None:
        super().__init__()
        self.qwen = qwen_model
        self.processor = processor
        self.scene_tokens = int(scene_tokens) if scene_tokens is not None else -1
        self.head_tokens = int(head_tokens)
        self.text_tokens = int(text_tokens)
        self.max_text_length = int(max_text_length)
        self.head_text = str(head_text)
        self.image_token_id = self._infer_image_token_id()
        self.spatial_merge_size = self._infer_spatial_merge_size()
        self.last_scene_grid_hw: tuple[int, int] | None = None
        self.last_head_grid_hw: tuple[int, int] | None = None
        self._scene_hint_warned = False

    def _infer_image_token_id(self) -> int:
        cands: list[int] = []
        for cfg in (
            getattr(self.qwen, "config", None),
            getattr(getattr(self.qwen, "base_model", None), "config", None),
            getattr(getattr(getattr(self.qwen, "base_model", None), "model", None), "config", None),
        ):
            if cfg is None:
                continue
            try:
                v = getattr(cfg, "image_token_id", None)
                if v is not None:
                    cands.append(int(v))
            except Exception:
                pass
        tok = getattr(self.processor, "tokenizer", None)
        if tok is not None and hasattr(tok, "convert_tokens_to_ids"):
            try:
                tid = int(tok.convert_tokens_to_ids("<|image_pad|>"))
                if tid >= 0:
                    cands.append(tid)
            except Exception:
                pass
        return cands[0] if cands else 151655

    def _infer_spatial_merge_size(self) -> int:
        cands: list[int] = []
        for cfg in (
            getattr(self.qwen, "config", None),
            getattr(getattr(self.qwen, "base_model", None), "config", None),
            getattr(getattr(getattr(self.qwen, "base_model", None), "model", None), "config", None),
        ):
            if cfg is None:
                continue
            try:
                vc = getattr(cfg, "vision_config", None)
                if vc is not None:
                    v = getattr(vc, "spatial_merge_size", None)
                    if v is not None:
                        cands.append(int(v))
            except Exception:
                pass
            if isinstance(cfg, dict):
                vc = cfg.get("vision_config", {})
                if isinstance(vc, dict):
                    v = vc.get("spatial_merge_size", None)
                    if v is not None:
                        try:
                            cands.append(int(v))
                        except Exception:
                            pass
        return cands[0] if cands else 1

    @staticmethod
    def _fallback_grid_hw(n_tokens: int) -> tuple[int, int]:
        s = int(round(float(n_tokens) ** 0.5))
        if s > 0 and s * s == int(n_tokens):
            return s, s
        raise RuntimeError(
            f"Failed to infer 2D scene grid from N={n_tokens}. "
            "Qwen image_grid_thw metadata is missing or inconsistent."
        )

    def _infer_grid_hw_from_inputs(
        self,
        image_grid_thw: torch.Tensor | None,
        token_lengths: list[int],
    ) -> tuple[int, int] | None:
        if image_grid_thw is None:
            return None
        if (not torch.is_tensor(image_grid_thw)) or image_grid_thw.dim() != 2:
            return None
        bsz = len(token_lengths)
        if image_grid_thw.shape[0] < bsz or image_grid_thw.shape[1] < 3:
            return None

        merge = max(1, int(self.spatial_merge_size))
        grids: list[tuple[int, int]] = []
        for b in range(bsz):
            t = int(image_grid_thw[b, 0].item())
            h = int(image_grid_thw[b, 1].item())
            w = int(image_grid_thw[b, 2].item())
            t = max(1, t)
            h_m = max(1, h // merge)
            w_m = max(1, w // merge)
            expected = t * h_m * w_m
            n = int(token_lengths[b])

            if expected != n:
                exp_no_merge = t * int(h) * int(w)
                if exp_no_merge == n:
                    h_m = int(h)
                    w_m = int(w)
                else:
                    # Final fallback if metadata does not align with extracted token length.
                    grids.append(self._fallback_grid_hw(n))
                    continue

            # Flatten temporal axis into height; keeps deterministic token order.
            gh = t * h_m
            gw = w_m
            if gh * gw != n:
                grids.append(self._fallback_grid_hw(n))
                continue
            grids.append((gh, gw))

        if not grids:
            return None
        uniq = set(grids)
        if len(uniq) > 1:
            raise RuntimeError(f"Vision grid differs across batch: {grids}")
        return grids[0]

    def _build_chat_text(self, text: str, with_image: bool) -> str:
        txt = str(text)
        if hasattr(self.processor, "apply_chat_template"):
            content: list[dict[str, str]] = []
            if with_image:
                content.append({"type": "image"})
            content.append({"type": "text", "text": txt})
            messages = [{"role": "user", "content": content}]
            try:
                return self.processor.apply_chat_template(
                    messages,
                    tokenize=False,
                    add_generation_prompt=False,
                )
            except TypeError:
                return self.processor.apply_chat_template(messages, tokenize=False)
        if with_image:
            return f"<|vision_start|><|image_pad|><|vision_end|>\n{txt}"
        return txt

    def _device(self) -> torch.device:
        return next(self.qwen.parameters()).device

    def _move_to_device(self, inputs: dict[str, Any]) -> dict[str, Any]:
        dev = self._device()
        out: dict[str, Any] = {}
        for k, v in inputs.items():
            out[k] = v.to(dev) if hasattr(v, "to") else v
        return out

    def _encode(
        self,
        texts: list[str],
        images: list[Any] | None,
        *,
        vision_only: bool = False,
    ) -> tuple[torch.Tensor, tuple[int, int] | None]:
        use_image = images is not None
        proc_texts = [self._build_chat_text(t, with_image=use_image) for t in texts]
        proc_kwargs: dict[str, Any] = {
            "text": proc_texts,
            "return_tensors": "pt",
            "padding": True,
        }
        if use_image:
            proc_kwargs["images"] = images
        else:
            proc_kwargs["truncation"] = True
            proc_kwargs["max_length"] = self.max_text_length
        inputs = self.processor(**proc_kwargs)
        inputs = self._move_to_device(inputs)
        out = self.qwen(
            **inputs,
            output_hidden_states=True,
            use_cache=False,
            return_dict=True,
        )
        if out.hidden_states is None or len(out.hidden_states) == 0:
            raise RuntimeError("Qwen backbone did not return hidden_states.")
        hidden = out.hidden_states[-1]  # [B, L, D]
        if not vision_only:
            return hidden, None
        if "input_ids" not in inputs:
            raise RuntimeError("input_ids are required to extract vision tokens.")
        input_ids = inputs["input_ids"]
        if input_ids.shape[:2] != hidden.shape[:2]:
            raise RuntimeError(
                f"input_ids/hidden shape mismatch: ids={tuple(input_ids.shape)} hidden={tuple(hidden.shape)}"
            )
        img_mask = input_ids.eq(int(self.image_token_id))
        vis_tokens: list[torch.Tensor] = []
        lengths: list[int] = []
        for b in range(hidden.shape[0]):
            tb = hidden[b][img_mask[b]]
            vis_tokens.append(tb)
            lengths.append(int(tb.shape[0]))
        if len(lengths) == 0 or min(lengths) <= 0:
            raise RuntimeError(
                "No vision tokens were extracted from hidden states. "
                "Please verify image placeholder tokens are present in text."
            )
        if min(lengths) != max(lengths):
            raise RuntimeError(f"Vision token counts differ across batch: {lengths}")
        grid_hw = self._infer_grid_hw_from_inputs(
            inputs.get("image_grid_thw", None),
            lengths,
        )
        return torch.stack(vis_tokens, dim=0), grid_hw

    @staticmethod
    def _fix_tokens(hidden: torch.Tensor, target_tokens: int) -> torch.Tensor:
        if target_tokens <= 0 or hidden.shape[1] == target_tokens:
            return hidden
        x = hidden.transpose(1, 2)
        x = F.adaptive_avg_pool1d(x, output_size=target_tokens)
        return x.transpose(1, 2)

    def forward(
        self,
        scene_image: Any,
        head_image: Any,
        text_inputs: Any,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        texts = [str(t) for t in text_inputs]
        scene_images = list(scene_image)
        head_images = list(head_image)
        if not (len(scene_images) == len(head_images) == len(texts)):
            raise ValueError("scene/head/text batch sizes must match.")

        head_texts = [self.head_text for _ in texts]
        h_s, scene_grid_hw = self._encode(texts=texts, images=scene_images, vision_only=True)
        h_h, head_grid_hw = self._encode(texts=head_texts, images=head_images, vision_only=True)
        h_t, _ = self._encode(texts=texts, images=None)

        if self.scene_tokens > 0 and h_s.shape[1] != self.scene_tokens and (not self._scene_hint_warned):
            print(
                "[WARN] scene token hint mismatch: "
                f"got={h_s.shape[1]} hint={self.scene_tokens}. "
                "Using dynamic scene grid from Qwen vision metadata."
            )
            self._scene_hint_warned = True

        if self.head_tokens > 0:
            h_h = self._fix_tokens(h_h, self.head_tokens)
        h_t = self._fix_tokens(h_t, self.text_tokens)

        if scene_grid_hw is None:
            scene_grid_hw = self._fallback_grid_hw(int(h_s.shape[1]))
        self.last_scene_grid_hw = scene_grid_hw
        self.last_head_grid_hw = head_grid_hw
        return h_s, h_h, h_t


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


def infer_hidden_dim(model: nn.Module, model_path: Path) -> int:
    values: list[int] = []

    def _push(v: Any) -> None:
        try:
            iv = int(v)
        except Exception:
            return
        if iv > 0:
            values.append(iv)

    def _from_cfg(cfg: Any) -> None:
        if cfg is None:
            return
        _push(getattr(cfg, "hidden_size", None))
        _push(getattr(getattr(cfg, "text_config", None), "hidden_size", None))
        if isinstance(cfg, dict):
            _push(cfg.get("hidden_size"))
            tc = cfg.get("text_config", {})
            if isinstance(tc, dict):
                _push(tc.get("hidden_size"))
        if hasattr(cfg, "to_dict"):
            try:
                d = cfg.to_dict()
            except Exception:
                d = {}
            if isinstance(d, dict):
                _push(d.get("hidden_size"))
                tc = d.get("text_config", {})
                if isinstance(tc, dict):
                    _push(tc.get("hidden_size"))

    _from_cfg(getattr(model, "config", None))
    _from_cfg(getattr(getattr(model, "base_model", None), "config", None))
    _from_cfg(getattr(getattr(getattr(model, "base_model", None), "model", None), "config", None))

    cfg_json = model_path / "config.json"
    if cfg_json.exists():
        try:
            raw = json.loads(cfg_json.read_text(encoding="utf-8"))
        except Exception:
            raw = {}
        _from_cfg(raw)

    return values[0] if values else 0


def infer_num_classes(
    train_label_map: dict[tuple[str, int], int],
    val_label_map: dict[tuple[str, int], int],
    vocab2id_path: Path | None,
) -> int:
    if vocab2id_path is not None and vocab2id_path.exists():
        obj = json.loads(vocab2id_path.read_text(encoding="utf-8"))
        if isinstance(obj, dict):
            return len(obj)
    mx = -1
    for v in train_label_map.values():
        mx = max(mx, int(v))
    for v in val_label_map.values():
        mx = max(mx, int(v))
    if mx >= 0:
        return mx + 1
    return 0






def main() -> None:
    config_parser = argparse.ArgumentParser(add_help=False)
    config_parser.add_argument("--config", type=str, default="config.yaml")
    cfg_args, _ = config_parser.parse_known_args()
    config_path = resolve_path(cfg_args.config)
    config_defaults = load_yaml_config(config_path)
    config_defaults["config"] = str(cfg_args.config)

    args = build_parser(defaults=config_defaults).parse_args()
    print(f"[INFO] loaded config: {resolve_path(args.config)}")
    set_seed(args.seed)

    out_dir = resolve_path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "train_args.json").write_text(
        json.dumps(vars(args), ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    wandb_run = init_wandb(args=args, root=ROOT)

    if args.device == "cuda" and (not torch.cuda.is_available()):
        raise RuntimeError("CUDA requested but no GPU is available.")
    device = torch.device(args.device)

    model_path = resolve_path(args.model_path)
    checkpoint_dir = resolve_path(args.checkpoint_dir) if str(args.checkpoint_dir).strip() else None
    train_ann = resolve_path(args.train_ann)
    val_ann = resolve_path(args.val_ann)
    test_ann = resolve_path(args.test_ann)
    image_root = resolve_path(args.image_root)
    test_image_root = resolve_path(args.test_image_root)
    train_labels = resolve_path(args.train_labels)
    val_labels = resolve_path(args.val_labels)
    test_labels = resolve_path(args.test_labels)
    vocab2id_path = resolve_path(args.vocab2id)
    label_embed_dir = resolve_path(args.label_embed_dir)

    recognition_enabled = bool(args.enable_recognition)
    recognition_objective = str(args.recognition_objective).strip().lower()
    use_infonce = recognition_enabled and (recognition_objective in {"infonce", "batch_local_infonce"})
    if use_infonce and (not label_embed_dir.exists()):
        print(f"[WARN] label_embed_dir does not exist: {label_embed_dir}")
    if recognition_enabled:
        vocab2id, vocab2id_lower = load_vocab2id(vocab2id_path)
        if vocab2id:
            print(f"[INFO] loaded vocab2id classes: {len(vocab2id)}")
        else:
            print("[WARN] vocab2id is missing/empty. classification labels will be set to ignore_index(-100).")

        print(
            "[INFO] label mapping rule: "
            "text label -> vocab2id id, and unmapped/missing text -> -100 (ignored in CE/acc)."
        )
        train_label_map, train_label_stats = load_label_map(
            train_labels,
            vocab2id=vocab2id,
            vocab2id_lower=vocab2id_lower,
            text_key="gaze_pseudo_label",
        )
        train_label_text_map, train_label_text_stats = load_label_text_map(
            train_labels,
            text_key="gaze_pseudo_label",
        )
        val_label_map, val_label_stats = load_label_map(
            val_labels,
            vocab2id=vocab2id,
            vocab2id_lower=vocab2id_lower,
            text_key="gaze_pseudo_label",
        )
        val_label_text_map, val_label_text_stats = load_label_text_map(
            val_labels,
            text_key="gaze_pseudo_label",
        )
        test_label_map, test_label_text_map, test_label_ids_map, test_label_stats = load_test_label_map(
            test_labels,
            vocab2id=vocab2id,
            vocab2id_lower=vocab2id_lower,
        )
        num_classes = infer_num_classes(train_label_map, val_label_map, vocab2id_path)
        if use_infonce and num_classes <= 0:
            print("[WARN] num_classes <= 0. recognition logits/metrics may be unavailable.")
        print(f"[INFO] inferred num_classes={num_classes}")
        print(
            "[INFO] train label map: "
            f"rows={train_label_stats['rows']} mapped={train_label_stats['mapped']} "
            f"missing_text={train_label_stats['missing_text']} unknown_text={train_label_stats['unknown_text']}"
        )
        print(
            "[INFO] val label map: "
            f"rows={val_label_stats['rows']} mapped={val_label_stats['mapped']} "
            f"missing_text={val_label_stats['missing_text']} unknown_text={val_label_stats['unknown_text']}"
        )
        print(
            "[INFO] test label map: "
            f"rows={test_label_stats['rows']} mapped={test_label_stats['mapped']} "
            f"missing_text={test_label_stats['missing_text']} unknown_text={test_label_stats['unknown_text']} "
            f"conflicts={test_label_stats['conflicts']}"
        )
        if use_infonce:
            print(
                "[INFO] recognition objective: InfoNCE (gaze_label_emb). "
                "train/val CE label_id coverage is informational only."
            )
            print(
                "[INFO] train label text coverage: "
                f"rows={train_label_text_stats['rows']} with_text={train_label_text_stats['with_text']} "
                f"missing_text={train_label_text_stats['missing_text']}"
            )
            print(
                "[INFO] val label text coverage: "
                f"rows={val_label_text_stats['rows']} with_text={val_label_text_stats['with_text']} "
                f"missing_text={val_label_text_stats['missing_text']}"
            )
        else:
            print("[INFO] recognition objective: CE (label_id)")
    else:
        vocab2id, vocab2id_lower = {}, {}
        train_label_map, val_label_map = {}, {}
        train_label_text_map, val_label_text_map = {}, {}
        test_label_map, test_label_text_map, test_label_ids_map = {}, {}, {}
        train_label_stats = {"rows": 0, "mapped": 0, "missing_text": 0, "unknown_text": 0}
        val_label_stats = {"rows": 0, "mapped": 0, "missing_text": 0, "unknown_text": 0}
        train_label_text_stats = {"rows": 0, "with_text": 0, "missing_text": 0}
        val_label_text_stats = {"rows": 0, "with_text": 0, "missing_text": 0}
        test_label_stats = {"rows": 0, "mapped": 0, "missing_text": 0, "unknown_text": 0, "conflicts": 0}
        num_classes = 0
        print("[INFO] recognition disabled: running localization-only pipeline.")

    train_records = load_records(
        annotation_file=train_ann,
        image_root=image_root,
        label_map=train_label_map,
        label_text_map=train_label_text_map,
        split_prefix=args.split_prefix,
        strip_split_prefix=bool(args.strip_split_prefix),
        max_samples=int(args.max_train_samples),
    )
    val_records = load_records(
        annotation_file=val_ann,
        image_root=image_root,
        label_map=val_label_map,
        label_text_map=val_label_text_map,
        split_prefix=args.split_prefix,
        strip_split_prefix=bool(args.strip_split_prefix),
        max_samples=int(args.max_val_samples),
    )
    if not train_records:
        raise RuntimeError("No train samples were loaded.")
    print(f"[INFO] train_records={len(train_records)} val_records={len(val_records)}")
    train_valid = sum(1 for r in train_records if int(r.label_id) >= 0)
    val_valid = sum(1 for r in val_records if int(r.label_id) >= 0)
    train_text_valid = sum(1 for r in train_records if str(r.label_text).strip())
    val_text_valid = sum(1 for r in val_records if str(r.label_text).strip())
    if recognition_enabled:
        if use_infonce:
            print(
                "[INFO] cls label coverage (text/emb): "
                f"train={train_text_valid}/{len(train_records)} ({(train_text_valid / max(1, len(train_records))):.3f}) "
                f"val={val_text_valid}/{len(val_records)} ({(val_text_valid / max(1, len(val_records))):.3f})"
            )
        else:
            print(
                "[INFO] cls label coverage (id): "
                f"train={train_valid}/{len(train_records)} ({(train_valid / max(1, len(train_records))):.3f}) "
                f"val={val_valid}/{len(val_records)} ({(val_valid / max(1, len(val_records))):.3f})"
            )
    else:
        print("[INFO] cls label coverage: recognition disabled (classification skipped).")

    train_ds = GazeDataset(
        records=train_records,
        heatmap_size=(args.heatmap_h, args.heatmap_w),
        heatmap_sigma=args.heatmap_sigma,
        prompt_template=args.prompt_template,
        prompt_text=args.prompt_text,
        apply_augmentation=True,
        recognition_objective=recognition_objective if recognition_enabled else "none",
        label_embed_dir=(label_embed_dir if recognition_enabled else None),
        label_emb_dim=int(args.label_emb_dim),
        normalize_label_emb=bool(args.normalize_label_emb),
    )
    val_ds = GazeDataset(
        records=val_records,
        heatmap_size=(args.heatmap_h, args.heatmap_w),
        heatmap_sigma=args.heatmap_sigma,
        prompt_template=args.prompt_template,
        prompt_text=args.prompt_text,
        recognition_objective=recognition_objective if recognition_enabled else "none",
        label_embed_dir=(label_embed_dir if recognition_enabled else None),
        label_emb_dim=int(args.label_emb_dim),
        normalize_label_emb=bool(args.normalize_label_emb),
    )
    train_loader = DataLoader(
        train_ds,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=(device.type == "cuda"),
        collate_fn=collate_batch,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=(device.type == "cuda"),
        collate_fn=collate_batch,
    )

    load_dtype = parse_dtype(args.dtype)
    if device.type != "cuda" and load_dtype in {torch.bfloat16, torch.float16}:
        print("[WARN] non-CUDA device detected; forcing model dtype to float32.")
        load_dtype = torch.float32
    model_kwargs: dict[str, Any] = {
        "trust_remote_code": True,
        "attn_implementation": args.attn_implementation,
    }
    if load_dtype != "auto":
        model_kwargs["dtype"] = load_dtype

    processor_path = model_path
    if checkpoint_dir is not None and (checkpoint_dir / "processor").exists():
        processor_path = checkpoint_dir / "processor"
    processor = AutoProcessor.from_pretrained(str(processor_path), trust_remote_code=True)
    base_qwen = AutoModelForImageTextToText.from_pretrained(str(model_path), **model_kwargs)

    if args.gradient_checkpointing and hasattr(base_qwen, "gradient_checkpointing_enable"):
        base_qwen.gradient_checkpointing_enable()
    if args.gradient_checkpointing and hasattr(base_qwen, "enable_input_require_grads"):
        base_qwen.enable_input_require_grads()

    adapter_dir = (checkpoint_dir / "lora_adapter") if checkpoint_dir is not None else None
    if adapter_dir is not None and adapter_dir.exists():
        qwen_lora = PeftModel.from_pretrained(
            base_qwen,
            model_id=str(adapter_dir),
            is_trainable=not bool(args.eval_only),
        )
        qwen_lora.to(device)
        print(f"[INFO] loaded LoRA adapter from: {adapter_dir}")
    else:
        target_modules = [x.strip() for x in str(args.lora_target_modules).split(",") if x.strip()]
        lora_cfg = LoraConfig(
            r=int(args.lora_r),
            lora_alpha=int(args.lora_alpha),
            lora_dropout=float(args.lora_dropout),
            bias=str(args.lora_bias),
            task_type=TaskType.CAUSAL_LM,
            target_modules=target_modules,
        )
        qwen_lora = get_peft_model(base_qwen, lora_cfg)
        qwen_lora.to(device)
        qwen_lora.print_trainable_parameters()

    hidden_dim = infer_hidden_dim(qwen_lora, model_path=model_path)
    print(f"[INFO] inferred hidden_dim={hidden_dim}")
    if hidden_dim <= 0:
        raise RuntimeError(
            "Failed to infer hidden_size from model config. "
            "Please verify model/config.json has `text_config.hidden_size`."
        )

    backbone = QwenBackboneAdapter(
        qwen_model=qwen_lora,
        processor=processor,
        scene_tokens=None,
        head_tokens=args.head_tokens,
        text_tokens=args.text_tokens,
        max_text_length=args.max_text_length,
        head_text=args.head_text,
    )
    model = QwenGazeIntegratedModel(
        backbone=backbone,
        hidden_dim=hidden_dim,
        scene_grid_size=None,
        num_classes=num_classes if num_classes > 0 else None,
        conditioning_mode=args.conditioning_mode,
        pool_mode=args.pool_mode,
        scene_input_size=(args.scene_h, args.scene_w),
        head_input_size=(args.head_h, args.head_w),
        heatmap_size=(args.heatmap_h, args.heatmap_w),
        num_conditioning_heads=args.num_conditioning_heads,
        num_conditioning_layers=args.num_conditioning_layers,
        dropout=args.dropout,
        recognition_objective=(recognition_objective if recognition_enabled else "none"),
        label_emb_dim=int(args.label_emb_dim),
        logit_scale_init=float(args.logit_scale_init),
        lambda_cls=args.lambda_cls,
        label_smoothing=args.label_smoothing,
        cls_ignore_index=args.cls_ignore_index,
    ).to(device)
    if recognition_enabled and (num_classes > 0):
        vocab_emb = build_vocab_embedding_matrix(
            vocab2id=vocab2id,
            label_embed_dir=label_embed_dir,
            label_emb_dim=int(args.label_emb_dim),
            normalize=bool(args.normalize_label_emb),
        )
        if vocab_emb is not None:
            model.set_vocab_embeddings(vocab_emb.to(device))
            print(
                "[INFO] loaded vocab embeddings: "
                f"shape={tuple(vocab_emb.shape)} from={label_embed_dir}"
            )
        else:
            print(
                "[WARN] failed to build vocab embedding matrix. "
                "test recognition logits from embedding-space may be unavailable."
            )
    if checkpoint_dir is not None and (checkpoint_dir / "heads.pt").exists():
        aux_state = torch.load(checkpoint_dir / "heads.pt", map_location=device)
        if isinstance(aux_state, dict):
            if "summary" in aux_state:
                model.summary.load_state_dict(aux_state["summary"], strict=True)
            if "conditioner" in aux_state:
                model.conditioner.load_state_dict(aux_state["conditioner"], strict=True)
            if "localizer" in aux_state:
                model.localizer.load_state_dict(aux_state["localizer"], strict=True)
            if model.classifier is not None and aux_state.get("classifier") is not None:
                try:
                    model.classifier.load_state_dict(aux_state["classifier"], strict=True)
                except Exception as e:
                    print(f"[WARN] classifier strict load failed: {e}; retrying strict=False")
                    model.classifier.load_state_dict(aux_state["classifier"], strict=False)
            print(f"[INFO] loaded heads from: {checkpoint_dir / 'heads.pt'}")

    trainable_params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(
        trainable_params,
        lr=args.lr,
        weight_decay=args.weight_decay,
    )
    updates_per_epoch = math.ceil(len(train_loader) / max(args.grad_accum_steps, 1))
    total_updates = max(1, updates_per_epoch * args.epochs)
    warmup_steps = int(total_updates * args.warmup_ratio)
    scheduler = get_cosine_schedule_with_warmup(
        optimizer=optimizer,
        num_warmup_steps=warmup_steps,
        num_training_steps=total_updates,
    )

    amp_dtype = to_autocast_dtype(load_dtype)
    best_dist = float("inf")
    global_step = 0
    start_time = time.time()
    effective_epochs = 0 if bool(args.eval_only) else int(args.epochs)
    if bool(args.eval_only):
        print("[INFO] eval_only=True; skipping training loop.")

    for epoch in range(1, effective_epochs + 1):
        model.train()
        sums: dict[str, float] = {"loss": 0.0, "l_hm": 0.0, "l_cls": 0.0, "dist": 0.0}
        cls_correct = 0
        cls_total = 0
        step_count = 0
        updates_done_in_epoch = 0

        optimizer.zero_grad(set_to_none=True)
        train_iter = tqdm(
            train_loader,
            desc=f"Train {epoch}/{args.epochs}",
            leave=False,
            dynamic_ncols=True,
            disable=not args.show_tqdm,
        )
        for step, batch in enumerate(train_iter, start=1):
            target_heatmap = batch["target_heatmap"].to(device)
            target_label = batch["target_label"].to(device)
            target_label_emb = batch["target_label_emb"].to(device)
            target_label_valid = batch["target_label_valid"].to(device)
            target_point = batch["target_point"].to(device)
            use_cls_id = bool(torch.any(target_label >= 0).item())
            use_cls_emb = bool(torch.any(target_label_valid > 0).item())
            batch_cls_acc: float | None = None

            with torch.autocast(
                device_type=device.type,
                dtype=amp_dtype,
                enabled=(device.type == "cuda"),
            ):
                out = model(
                    scene_image=batch["scene_images"],
                    head_image=batch["head_images"],
                    text_inputs=batch["text_inputs"],
                    target_heatmap=target_heatmap,
                    target_label=target_label if use_cls_id else None,
                    target_label_emb=target_label_emb if use_cls_emb else None,
                    target_label_valid=target_label_valid if use_cls_emb else None,
                    use_softargmax=True,
                )
                loss = out["loss"] / max(args.grad_accum_steps, 1)

            loss.backward()
            loss_dict = out.get("loss_dict", {})
            pred_point = out["point"].detach().to(dtype=torch.float32)
            tgt_point = target_point.detach().to(dtype=torch.float32)
            batch_dist = torch.linalg.norm(pred_point - tgt_point, dim=-1).mean()
            if "logits" in out:
                valid = target_label >= 0
                if torch.any(valid):
                    pred = out["pred_label"][valid]
                    gt = target_label[valid]
                    batch_cls_acc = float((pred == gt).float().mean().item())
                    cls_correct += int((pred == gt).sum().item())
                    cls_total += int(valid.sum().item())

            if step % max(args.grad_accum_steps, 1) == 0:
                grad_norm = torch.nn.utils.clip_grad_norm_(trainable_params, max_norm=args.max_grad_norm)
                optimizer.step()
                if hasattr(model, "logit_scale"):
                    with torch.no_grad():
                        model.logit_scale.data.clamp_(0.0, 4.6052)
                scheduler.step()
                optimizer.zero_grad(set_to_none=True)
                global_step += 1
                updates_done_in_epoch += 1
                if (
                    wandb_run is not None
                    and int(args.wandb_log_every_steps) > 0
                    and (global_step % int(args.wandb_log_every_steps) == 0)
                ):
                    grad_norm_value = float(grad_norm.detach().item()) if torch.is_tensor(grad_norm) else float(grad_norm)
                    epoch_progress = (float(epoch) - 1.0) + (
                        float(updates_done_in_epoch) / max(float(updates_per_epoch), 1.0)
                    )
                    step_log = {
                        "train/loss": float(loss_dict.get("loss", out["loss"]).detach().item()),
                        "train/hm": float(loss_dict.get("l_hm", torch.tensor(0.0)).detach().item()),
                        "train/cls": float(loss_dict.get("l_cls", torch.tensor(0.0)).detach().item()),
                        "train/dist": float(batch_dist.item()),
                        "train/learning_rate": float(optimizer.param_groups[0]["lr"]),
                        "train/grad_norm": grad_norm_value,
                        "train/global_step": float(global_step),
                        "train/epoch": epoch_progress,
                    }
                    if batch_cls_acc is not None:
                        step_log["train/acc"] = batch_cls_acc
                    wandb_run.log(step_log, step=global_step)

            sums["loss"] += float(loss_dict.get("loss", out["loss"]).detach().item())
            sums["l_hm"] += float(loss_dict.get("l_hm", torch.tensor(0.0)).detach().item())
            if "l_cls" in loss_dict:
                sums["l_cls"] += float(loss_dict["l_cls"].detach().item())
            sums["dist"] += float(batch_dist.item())

            step_count += 1
            if args.show_tqdm:
                train_iter.set_postfix(
                    loss=f"{(sums['loss'] / max(step_count, 1)):.4f}",
                    hm=f"{(sums['l_hm'] / max(step_count, 1)):.4f}",
                    dist=f"{(sums['dist'] / max(step_count, 1)):.4f}",
                )

        if step_count == 0:
            raise RuntimeError("No training batches were produced.")

        train_metrics = {k: v / step_count for k, v in sums.items()}
        train_metrics["cls_acc"] = (cls_correct / cls_total) if cls_total > 0 else 0.0

        val_metrics = (
            run_eval(
                model,
                val_loader,
                device,
                amp_dtype,
                show_tqdm=bool(args.show_tqdm),
                desc=f"Eval {epoch}/{args.epochs}",
            )
            if len(val_ds) > 0
            else {}
        )
        val_dist = float(val_metrics.get("dist", train_metrics["dist"]))

        print(
            f"[EPOCH {epoch}] "
            f"train_loss={train_metrics['loss']:.6f} "
            f"train_hm={train_metrics['l_hm']:.6f} "
            f"train_dist={train_metrics['dist']:.6f} "
            f"train_cls={train_metrics['l_cls']:.6f} "
            f"train_acc={train_metrics['cls_acc']:.4f}"
        )
        if val_metrics:
            print(
                f"[EPOCH {epoch}] "
                f"val_loss={val_metrics['loss']:.6f} "
                f"val_hm={val_metrics['l_hm']:.6f} "
                f"val_dist={val_metrics['dist']:.6f} "
                f"val_cls={val_metrics['l_cls']:.6f} "
                f"val_acc={val_metrics['cls_acc']:.4f}"
            )
        if wandb_run is not None:
            log_data = {
                "epoch/index": float(epoch),
                "epoch/global_step": float(global_step),
                "epoch/train_loss": float(train_metrics["loss"]),
                "epoch/train_hm": float(train_metrics["l_hm"]),
                "epoch/train_dist": float(train_metrics["dist"]),
                "epoch/train_cls": float(train_metrics["l_cls"]),
                "epoch/train_acc": float(train_metrics["cls_acc"]),
            }
            if val_metrics:
                log_data.update(
                    {
                        "val/epoch": float(epoch),
                        "val/loss": float(val_metrics["loss"]),
                        "val/hm": float(val_metrics["l_hm"]),
                        "val/dist": float(val_metrics["dist"]),
                        "metric/val/dist": float(val_metrics["dist"]),
                        "val/cls": float(val_metrics["l_cls"]),
                        "val/acc": float(val_metrics["cls_acc"]),
                    }
                )
            wandb_run.log(log_data, step=global_step)

        if val_dist < best_dist:
            best_dist = val_dist
            best_dir = out_dir / "best"
            save_checkpoint(
                best_dir,
                epoch,
                model,
                processor,
                optimizer,
                scheduler,
                clear_dir=True,
            )

    if bool(args.eval_only) and len(val_ds) > 0:
        val_metrics = run_eval(
            model,
            val_loader,
            device,
            amp_dtype,
            show_tqdm=bool(args.show_tqdm),
            desc="Eval (checkpoint)",
        )
        best_dist = float(val_metrics.get("dist", best_dist))
        print(
            "[EVAL] "
            f"val_loss={val_metrics['loss']:.6f} "
            f"val_hm={val_metrics['l_hm']:.6f} "
            f"val_dist={val_metrics['dist']:.6f} "
            f"val_cls={val_metrics['l_cls']:.6f} "
            f"val_acc={val_metrics['cls_acc']:.4f}"
        )
        if wandb_run is not None:
            wandb_run.log(
                {
                    "val/epoch": 0.0,
                    "val/loss": float(val_metrics["loss"]),
                    "val/hm": float(val_metrics["l_hm"]),
                    "val/dist": float(val_metrics["dist"]),
                    "metric/val/dist": float(val_metrics["dist"]),
                    "val/cls": float(val_metrics["l_cls"]),
                    "val/acc": float(val_metrics["cls_acc"]),
                },
                step=global_step,
            )

    elapsed = time.time() - start_time

    if args.run_test:
        test_groups = load_test_groups(
            annotation_file=test_ann,
            image_root=test_image_root,
            test_label_map=test_label_map,
            test_label_text_map=test_label_text_map,
            test_label_ids_map=test_label_ids_map,
            split_prefix=args.test_split_prefix,
            strip_split_prefix=bool(args.test_strip_split_prefix),
            bbox_round_decimals=int(args.test_bbox_round_decimals),
            max_groups=int(args.max_test_samples),
        )
        if not test_groups:
            print("[TEST] no valid test groups found.")
        else:
            print(f"[TEST] groups={len(test_groups)}")
            test_ds = GazeTestDataset(
                groups=test_groups,
                prompt_template=args.prompt_template,
                prompt_text=args.prompt_text,
            )
            test_loader = DataLoader(
                test_ds,
                batch_size=max(1, int(args.test_batch_size)),
                shuffle=False,
                num_workers=args.num_workers,
                pin_memory=(device.type == "cuda"),
                collate_fn=collate_test_batch,
            )
            test_metrics = run_test_metrics(
                model=model,
                loader=test_loader,
                device=device,
                amp_dtype=amp_dtype,
                show_tqdm=bool(args.show_tqdm),
                desc="Test",
                acc_dist_threshold=float(args.acc_dist_threshold),
            )
            print_test_metrics_table(test_metrics)
            if wandb_run is not None:
                wandb_run.log(
                    {
                        "test/epoch": float(effective_epochs),
                        "test/AUC": float(test_metrics["AUC"]),
                        "test/AvgL2": float(test_metrics["Avg L2"]),
                        "test/MinL2": float(test_metrics["Min L2"]),
                        "test/Acc@1": float(test_metrics["Acc@1"]),
                        "test/Acc@3": float(test_metrics["Acc@3"]),
                        "test/multiAcc@1": float(test_metrics["multiAcc@1"]),
                    },
                    step=global_step,
                )
            (out_dir / "test_metrics.json").write_text(
                json.dumps(test_metrics, ensure_ascii=False, indent=2),
                encoding="utf-8",
            )

    finish_wandb(wandb_run)
    print(f"[DONE] global_step={global_step} best_dist={best_dist:.6f} elapsed_sec={elapsed:.1f}")


if __name__ == "__main__":
    main()
