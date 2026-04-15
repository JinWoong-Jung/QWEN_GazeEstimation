from __future__ import annotations

import argparse
import json
import math
import os
import random
import time
from pathlib import Path
from typing import Any

import warnings

warnings.filterwarnings(
    "ignore",
    message=r"Flash Attention defaults to a non-deterministic algorithm.*",
    category=UserWarning,
    module=r"torch\.autograd\.graph",
)


import torch
from torch.utils.data import DataLoader
from tqdm.auto import tqdm
from transformers import (
    AutoModelForImageTextToText,
    AutoProcessor,
    get_cosine_schedule_with_warmup,
)

from peft import LoraConfig, PeftModel, TaskType, get_peft_model

from .datasets import GazeDataset, GazeTestDataset
from .model import QwenTextGenerationModel
from .utils.checkpoint import load_checkpoint_for_eval, save_checkpoint
from .utils.common import to_device
from .utils.config_parser import (
    build_parser,
    load_yaml_config,
    normalize_run_name,
    output_dir_from_run_name,
)
from .utils.data_utils import (
    build_vocab_embedding_matrix,
    load_label_map,
    load_label_text_map,
    load_records,
    load_test_groups,
    load_test_label_map,
    load_vocab2id,
)
from .utils.eval_utils import (
    CLIPTextEncoder,
    collect_generation_samples,
    print_generation_samples,
    print_test_metrics_table,
    run_eval,
    run_test_metrics,
)
from .utils.loss_utils import compute_answer_loss
from .utils.processor_collate import QwenTestCollator, QwenTrainCollator
from .utils.wandb_utils import finish_wandb, init_wandb, train_progress_bucket


ROOT = Path(__file__).resolve().parents[1]
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")


def resolve_path(path: str) -> Path:
    p = Path(path)
    return p if p.is_absolute() else ROOT / p


def set_seed(seed: int) -> None:
    os.environ["PYTHONHASHSEED"] = str(int(seed))
    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    if hasattr(torch.backends, "cudnn"):
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    if hasattr(torch, "use_deterministic_algorithms"):
        torch.use_deterministic_algorithms(True, warn_only=True)


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


def build_id2label(vocab2id: dict[str, int]) -> dict[int, str]:
    out: dict[int, str] = {}
    for label_text, idx in vocab2id.items():
        idx_i = int(idx)
        if idx_i not in out:
            out[idx_i] = str(label_text)
    return out


def count_valid_targets(records: list[Any]) -> int:
    n = 0
    for r in records:
        label_id = int(getattr(r, "label_id", -1))
        txt = str(getattr(r, "label_text", "") or "").strip()
        if (label_id >= 0) or bool(txt):
            n += 1
    return n


def env_flag(name: str) -> bool:
    v = str(os.environ.get(name, "")).strip().lower()
    return v in {"1", "true", "yes", "y", "on"}


def label_key(text: str) -> str:
    return " ".join(str(text or "").strip().lower().split())


def log_target_example(tag: str, dataset: Any) -> None:
    if not env_flag("QWEN_DEBUG_TARGET_EXAMPLE"):
        return
    try:
        n = len(dataset)
        if int(n) <= 0:
            print(f"[DEBUG] {tag} target example: dataset is empty.")
            return
        sample = dataset[0]
        tgt = str(sample.get("target_text", ""))
        print(f"[DEBUG] {tag} target example:\n{tgt}")
    except Exception as e:
        print(f"[DEBUG] {tag} target example unavailable: {e}")


def infer_num_classes(vocab2id: dict[str, int], vocab2id_path: Path) -> int:
    if (not isinstance(vocab2id, dict)) or (len(vocab2id) <= 0):
        raise RuntimeError(
            f"vocab2id is missing/empty: {vocab2id_path}. "
            "Object special tokens require a valid vocab2id mapping."
        )
    n = int(len(vocab2id))
    ids: list[int] = []
    for k, v in vocab2id.items():
        try:
            idx = int(v)
        except Exception as e:
            raise RuntimeError(
                f"vocab2id is malformed at key={k!r}, value={v!r}: {e}"
            ) from e
        ids.append(idx)
    expected = list(range(n))
    got = sorted(ids)
    if got != expected:
        first = got[:10]
        last = got[-10:] if len(got) > 10 else got
        raise RuntimeError(
            "vocab2id ids must be exactly contiguous 0..N-1. "
            f"path={vocab2id_path} N={n} got_min={min(got)} got_max={max(got)} "
            f"head={first} tail={last}"
        )
    return n


def init_processor(
    *,
    model_path: Path,
    checkpoint_dir: Path | None,
) -> Any:
    processor_path = model_path
    if checkpoint_dir is not None and (checkpoint_dir / "processor").exists():
        processor_path = checkpoint_dir / "processor"
    return AutoProcessor.from_pretrained(str(processor_path), trust_remote_code=True)


def init_base_model(
    *,
    model_path: Path,
    model_kwargs: dict[str, Any],
) -> Any:
    return AutoModelForImageTextToText.from_pretrained(str(model_path), **model_kwargs)


def test_log_payload(test_metrics: dict[str, float]) -> dict[str, float]:
    return {
        "test/Avg_L2": float(test_metrics.get("Avg L2", test_metrics.get("PointL2", 0.0))),
        "test/Min_L2": float(test_metrics.get("Min L2", 0.0)),
        "test/acc@1": float(test_metrics.get("acc@1", 0.0)),
        "test/acc@3": float(test_metrics.get("acc@3", 0.0)),
        "test/multiacc@1": float(test_metrics.get("multiacc@1", 0.0)),
        "test/ObjectParseFail": float(test_metrics.get("ObjectParseFail", 0.0)),
        "test/num_valid_targets": float(test_metrics.get("num_valid_targets", 0.0)),
    }


def val_metric_log_payload(val_metrics: dict[str, float]) -> dict[str, float]:
    return {
        "val/dist": float(val_metrics.get("Avg L2", val_metrics.get("PointL2", 0.0))),
        "val/acc@1": float(val_metrics.get("acc@1", 0.0)),
        "val/acc@3": float(val_metrics.get("acc@3", 0.0)),
        "val/multiacc@1": float(val_metrics.get("multiacc@1", 0.0)),
    }


def maybe_save_generation_preview(
    *,
    args: argparse.Namespace,
    out_dir: Path,
    model: torch.nn.Module,
    loader: DataLoader,
    device: torch.device,
    amp_dtype: torch.dtype,
    processor: Any,
    label_embedding_bank: torch.Tensor,
    retrieval_label_texts: list[str],
    retrieval_top_k: int,
    clip_encoder: CLIPTextEncoder | None,
    object_temperature: float,
    bank_canonical_ids: list[int] | None = None,
) -> None:
    preview_n = max(0, int(getattr(args, "preview_test_samples", 0)))
    if preview_n <= 0:
        return

    preview_samples = collect_generation_samples(
        model=model,
        loader=loader,
        device=device,
        amp_dtype=amp_dtype,
        processor=processor,
        label_embedding_bank=label_embedding_bank,
        retrieval_label_texts=retrieval_label_texts,
        retrieval_top_k=int(retrieval_top_k),
        clip_encoder=clip_encoder,
        object_temperature=float(object_temperature),
        show_tqdm=bool(args.show_tqdm),
        desc="Test preview",
        max_new_tokens=int(args.generation_max_new_tokens),
        num_beams=int(getattr(args, "generation_num_beams", 1)),
        repetition_penalty=float(getattr(args, "repetition_penalty", 1.0)),
        no_repeat_ngram_size=int(getattr(args, "no_repeat_ngram_size", 0)),
        max_samples=preview_n,
        bank_canonical_ids=bank_canonical_ids,
    )
    preview_path = out_dir / "test_generation_preview.json"
    preview_path.write_text(
        json.dumps(preview_samples, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print(f"[TEST] saved generation preview: {preview_path} (samples={len(preview_samples)})")
    print_generation_samples(preview_samples)


def infer_checkpoint_monitor_mode(monitor: str, mode: str) -> str:
    mode_norm = str(mode or "auto").strip().lower()
    if mode_norm in {"min", "max"}:
        return mode_norm
    if mode_norm != "auto":
        raise ValueError(f"checkpoint_monitor_mode must be one of auto|min|max, got: {mode}")
    monitor_norm = str(monitor or "").strip().lower()
    return "min" if ("loss" in monitor_norm or "dist" in monitor_norm or "l2" in monitor_norm) else "max"


def checkpoint_monitor_value(
    monitor: str,
    *,
    val_loss: float,
    val_gen_metrics: dict[str, float] | None,
) -> float | None:
    monitor_norm = str(monitor or "val_loss").strip().lower().replace("/", "_")
    monitor_norm = monitor_norm.replace("-", "_")
    if monitor_norm in {"val_loss", "loss"}:
        return float(val_loss)
    if val_gen_metrics is None:
        return None
    if monitor_norm in {"val_dist", "dist", "pointl2", "point_l2", "avg_l2", "avg l2"}:
        return float(val_gen_metrics.get("Avg L2", val_gen_metrics.get("PointL2", 0.0)))
    if monitor_norm in {"val_acc@1", "acc@1"}:
        return float(val_gen_metrics.get("acc@1", 0.0))
    if monitor_norm in {"val_acc@3", "acc@3"}:
        return float(val_gen_metrics.get("acc@3", 0.0))
    if monitor_norm in {"val_multiacc@1", "multiacc@1", "multi_acc@1"}:
        return float(val_gen_metrics.get("multiacc@1", 0.0))
    raise ValueError(
        "Unsupported checkpoint_monitor. Use one of: "
        "val_loss, val_dist, val_acc@1, val_acc@3, val_multiacc@1."
    )


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

    run_name = normalize_run_name(getattr(args, "run_name", ""))
    if run_name:
        args.output_dir = output_dir_from_run_name(args.output_dir, run_name)
        args.wandb_run_name = run_name
        print(f"[INFO] run_name={run_name}")

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
    labels_rgs = resolve_path(args.labels_rgs) if str(getattr(args, "labels_rgs", "")).strip() else None
    labels_ssa = resolve_path(args.labels_ssa) if str(getattr(args, "labels_ssa", "")).strip() else None
    vocab2id_path = resolve_path(args.vocab2id)

    vocab2id, vocab2id_lower = load_vocab2id(vocab2id_path)
    num_classes = infer_num_classes(vocab2id, vocab2id_path)
    id2label = build_id2label(vocab2id)
    print(f"[INFO] loaded vocab2id classes: {len(vocab2id)} (id_range=0..{max(num_classes - 1, 0)})")

    prompt_text_for_run = str(args.prompt_text or "")

    label_embed_dir = (
        resolve_path(getattr(args, "label_embed_dir"))
        if str(getattr(args, "label_embed_dir", "")).strip()
        else None
    )
    object_embedding_dim = int(getattr(args, "object_embedding_dim", 512))
    object_temperature = float(getattr(args, "object_temperature", 0.07))
    retrieval_top_k = max(1, int(getattr(args, "test_retrieval_top_k", 3)))
    clip_model_path = str(getattr(args, "clip_model_path", "openai/clip-vit-base-patch32")).strip()

    # Retrieval label embedding bank: row i = embedding for vocab label id i (vocab2id order).
    # topk_similarity indices on this bank are directly comparable to target_label ids.
    retrieval_label_texts: list[str] = [id2label.get(i, "") for i in range(num_classes)]
    retrieval_label_embedding_bank = build_vocab_embedding_matrix(
        vocab2id=vocab2id,
        label_embed_dir=label_embed_dir,
        label_emb_dim=int(object_embedding_dim),
        normalize=True,
    )
    if retrieval_label_embedding_bank is None:
        raise RuntimeError(
            "Failed to build retrieval label embedding bank. "
            f"label_embed_dir={label_embed_dir} object_embedding_dim={object_embedding_dim}"
        )
    _valid_rows = int(retrieval_label_embedding_bank.norm(dim=1).gt(0).sum().item())
    if _valid_rows <= 0:
        raise RuntimeError("Retrieval label embedding bank has no valid rows (all zero vectors).")
    print(
        "[INFO] retrieval label embedding bank: "
        f"vocab_size={num_classes} valid_rows={_valid_rows} "
        f"shape={tuple(retrieval_label_embedding_bank.shape)} topk={retrieval_top_k}"
    )

    if bool(getattr(args, "test_only", False)):
        print("[INFO] test_only=True; skipping train/val loading and training.")
        start_time = time.time()

        test_label_map, test_label_text_map, test_label_ids_map, test_label_stats = load_test_label_map(
            test_labels,
            vocab2id=vocab2id,
            vocab2id_lower=vocab2id_lower,
        )
        print(
            "[INFO] test label map: "
            f"rows={test_label_stats['rows']} mapped={test_label_stats['mapped']} "
            f"missing_text={test_label_stats['missing_text']} unknown_text={test_label_stats['unknown_text']} "
            f"conflicts={test_label_stats['conflicts']}"
        )

        load_dtype = parse_dtype(args.dtype)
        if device.type != "cuda" and load_dtype in {torch.bfloat16, torch.float16}:
            print("[WARN] non-CUDA device detected; forcing model dtype to float32.")
            load_dtype = torch.float32
        amp_dtype = to_autocast_dtype(load_dtype)

        model_kwargs: dict[str, Any] = {
            "trust_remote_code": True,
            "attn_implementation": args.attn_implementation,
        }
        if load_dtype != "auto":
            model_kwargs["dtype"] = load_dtype

        processor = init_processor(model_path=model_path, checkpoint_dir=checkpoint_dir)
        test_collator = QwenTestCollator(
            processor=processor,
            scene_size=(int(args.scene_h), int(args.scene_w)),
            max_text_length=int(args.max_text_length),
        )

        base_qwen = init_base_model(model_path=model_path, model_kwargs=model_kwargs)
        adapter_dir = (checkpoint_dir / "lora_adapter") if checkpoint_dir is not None else None
        if adapter_dir is not None and adapter_dir.exists():
            qwen_model = PeftModel.from_pretrained(
                base_qwen, model_id=str(adapter_dir), is_trainable=False,
            )
            print(f"[INFO] loaded LoRA adapter from: {adapter_dir}")
        else:
            qwen_model = base_qwen
            print("[INFO] adapter checkpoint not found; running zero-shot base model.")

        model = QwenTextGenerationModel(qwen_model=qwen_model).to(device)
        model.eval()

        clip_encoder = CLIPTextEncoder(model_path=clip_model_path, device=device)

        print(
            "[INFO] object evaluation bank: "
            f"global_vocab={num_classes} shape={tuple(retrieval_label_embedding_bank.shape)}"
        )

        test_groups = load_test_groups(
            annotation_file=test_ann,
            image_root=test_image_root,
            test_label_map=test_label_map,
            test_label_text_map=test_label_text_map,
            test_label_ids_map=test_label_ids_map,
            split_prefix=args.test_split_prefix,
            strip_split_prefix=bool(args.test_strip_split_prefix),
            bbox_round_decimals=int(args.test_bbox_round_decimals),
            )
        if not test_groups:
            print("[TEST] no valid test groups found.")
        else:
            print(f"[TEST] groups={len(test_groups)}")
            test_ds = GazeTestDataset(
                groups=test_groups,
                prompt_template=args.prompt_template,
                prompt_text=prompt_text_for_run,
                id2label=id2label,
                vocab2id=vocab2id,
                vocab2id_lower=vocab2id_lower,
                num_classes=int(num_classes),
                answer_template=args.answer_template,
                fallback_target_text=args.fallback_target_text,
                point_decimals=int(args.point_decimals),
                visual_prompting=bool(args.visual_prompting),
                image_cache_size=max(0, int(getattr(args, "image_cache_size", 0))),
            )
            log_target_example("test_only", test_ds)
            _tnw = int(args.num_workers)
            test_loader = DataLoader(
                test_ds,
                batch_size=max(1, int(args.test_batch_size)),
                shuffle=False,
                num_workers=_tnw,
                pin_memory=(device.type == "cuda"),
                collate_fn=test_collator,
                persistent_workers=(_tnw > 0),
                prefetch_factor=(2 if _tnw > 0 else None),
            )
            test_metrics = run_test_metrics(
                model=model,
                loader=test_loader,
                device=device,
                amp_dtype=amp_dtype,
                processor=processor,
                label_embedding_bank=retrieval_label_embedding_bank.to(device=device),
                retrieval_label_texts=retrieval_label_texts,
                retrieval_top_k=int(retrieval_top_k),
                clip_encoder=clip_encoder,
                object_temperature=object_temperature,
                show_tqdm=bool(args.show_tqdm),
                desc="Test",
                max_new_tokens=int(args.generation_max_new_tokens),
                num_beams=int(getattr(args, "generation_num_beams", 1)),
                repetition_penalty=float(getattr(args, "repetition_penalty", 1.0)),
                no_repeat_ngram_size=int(getattr(args, "no_repeat_ngram_size", 0)),
            )
            print_test_metrics_table(test_metrics)
            if wandb_run is not None:
                wandb_run.log(test_log_payload(test_metrics), step=0)
            (out_dir / "test_metrics.json").write_text(
                json.dumps(test_metrics, ensure_ascii=False, indent=2),
                encoding="utf-8",
            )
            maybe_save_generation_preview(
                args=args,
                out_dir=out_dir,
                model=model,
                loader=test_loader,
                device=device,
                amp_dtype=amp_dtype,
                processor=processor,
                label_embedding_bank=retrieval_label_embedding_bank.to(device=device),
                retrieval_label_texts=retrieval_label_texts,
                retrieval_top_k=int(retrieval_top_k),
                clip_encoder=clip_encoder,
                object_temperature=object_temperature,
            )

        elapsed = time.time() - start_time
        finish_wandb(wandb_run)
        print(f"[DONE] test_only=True elapsed_sec={elapsed:.1f}")
        return

    train_label_map, train_label_stats = load_label_map(
        train_labels,
        vocab2id=vocab2id,
        vocab2id_lower=vocab2id_lower,
        text_key="gaze_pseudo_label",
        use_embed_fallback=False,
    )
    val_label_map, val_label_stats = load_label_map(
        val_labels,
        vocab2id=vocab2id,
        vocab2id_lower=vocab2id_lower,
        text_key="gaze_pseudo_label",
        use_embed_fallback=False,
    )

    train_label_text_map, train_label_text_stats = load_label_text_map(
        train_labels, text_key="gaze_pseudo_label",
    )
    val_label_text_map, val_label_text_stats = load_label_text_map(
        val_labels, text_key="gaze_pseudo_label",
    )
    test_label_map, test_label_text_map, test_label_ids_map, test_label_stats = load_test_label_map(
        test_labels,
        vocab2id=vocab2id,
        vocab2id_lower=vocab2id_lower,
    )

    print(
        "[INFO] train label id coverage: "
        f"rows={train_label_stats['rows']} mapped={train_label_stats['mapped']} "
        f"missing_text={train_label_stats['missing_text']} unknown_text={train_label_stats['unknown_text']} "
        f"primary_mapped={train_label_stats.get('primary_mapped', 0)}"
    )
    print(
        "[INFO] val label id coverage: "
        f"rows={val_label_stats['rows']} mapped={val_label_stats['mapped']} "
        f"missing_text={val_label_stats['missing_text']} unknown_text={val_label_stats['unknown_text']}"
    )
    print(
        "[INFO] test label map: "
        f"rows={test_label_stats['rows']} mapped={test_label_stats['mapped']} "
        f"missing_text={test_label_stats['missing_text']} unknown_text={test_label_stats['unknown_text']} "
        f"conflicts={test_label_stats['conflicts']}"
    )
    print(
        "[INFO] object evaluation bank: "
        f"global_vocab={num_classes} shape={tuple(retrieval_label_embedding_bank.shape)}"
    )

    train_records = load_records(
        annotation_file=train_ann,
        image_root=image_root,
        label_map=train_label_map,
        label_text_map=train_label_text_map,
        split_prefix=args.split_prefix,
        strip_split_prefix=bool(args.strip_split_prefix),
    )
    val_records = load_records(
        annotation_file=val_ann,
        image_root=image_root,
        label_map=val_label_map,
        label_text_map=val_label_text_map,
        split_prefix=args.split_prefix,
        strip_split_prefix=bool(args.strip_split_prefix),
    )
    if not train_records:
        raise RuntimeError("No train samples were loaded.")

    print(
        f"[INFO] train_records={len(train_records)} val_records={len(val_records)} "
        f"train_valid_targets={count_valid_targets(train_records)} "
        f"val_valid_targets={count_valid_targets(val_records)}"
    )

    image_cache_size = max(0, int(getattr(args, "image_cache_size", 0)))
    train_ds = GazeDataset(
        records=train_records,
        prompt_template=args.prompt_template,
        prompt_text=prompt_text_for_run,
        apply_augmentation=True,
        id2label=id2label,
        vocab2id=vocab2id,
        vocab2id_lower=vocab2id_lower,
        num_classes=int(num_classes),
        answer_template=args.answer_template,
        fallback_target_text=args.fallback_target_text,
        point_decimals=int(args.point_decimals),
        visual_prompting=bool(args.visual_prompting),
        image_cache_size=image_cache_size,
    )
    val_ds = GazeDataset(
        records=val_records,
        prompt_template=args.prompt_template,
        prompt_text=prompt_text_for_run,
        apply_augmentation=False,
        id2label=id2label,
        vocab2id=vocab2id,
        vocab2id_lower=vocab2id_lower,
        num_classes=int(num_classes),
        answer_template=args.answer_template,
        fallback_target_text=args.fallback_target_text,
        point_decimals=int(args.point_decimals),
        visual_prompting=bool(args.visual_prompting),
        image_cache_size=image_cache_size,
    )
    log_target_example("train", train_ds)
    log_target_example("val", val_ds)

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

    processor = init_processor(model_path=model_path, checkpoint_dir=checkpoint_dir)

    train_collator = QwenTrainCollator(
        processor=processor,
        scene_size=(int(args.scene_h), int(args.scene_w)),
        max_text_length=int(args.max_text_length),
    )
    val_collator = QwenTrainCollator(
        processor=processor,
        scene_size=(int(args.scene_h), int(args.scene_w)),
        max_text_length=int(args.max_text_length),
    )
    test_collator = QwenTestCollator(
        processor=processor,
        scene_size=(int(args.scene_h), int(args.scene_w)),
        max_text_length=int(args.max_text_length),
    )

    _nw = int(args.num_workers)
    _persistent = _nw > 0
    _prefetch = 2 if _nw > 0 else None

    train_loader = DataLoader(
        train_ds,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=_nw,
        pin_memory=(device.type == "cuda"),
        collate_fn=train_collator,
        persistent_workers=_persistent,
        prefetch_factor=_prefetch,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=_nw,
        pin_memory=(device.type == "cuda"),
        collate_fn=val_collator,
        persistent_workers=_persistent,
        prefetch_factor=_prefetch,
    )
    val_metric_loader = None
    if bool(getattr(args, "run_val_metrics", True)):
        val_metric_loader = DataLoader(
            val_ds,
            batch_size=max(1, int(args.test_batch_size)),
            shuffle=False,
            num_workers=_nw,
            pin_memory=(device.type == "cuda"),
            collate_fn=test_collator,
            persistent_workers=_persistent,
            prefetch_factor=_prefetch,
        )

    base_qwen = init_base_model(model_path=model_path, model_kwargs=model_kwargs)
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
        qwen_lora.print_trainable_parameters()

    model = QwenTextGenerationModel(qwen_model=qwen_lora).to(device)

    # CLIP text encoder for object retrieval at eval/test time
    clip_encoder = CLIPTextEncoder(model_path=clip_model_path, device=device)

    trainable_params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(
        trainable_params,
        lr=args.lr,
        weight_decay=args.weight_decay,
    )

    accum_steps = max(int(args.grad_accum_steps), 1)
    num_train_batches = len(train_loader)
    updates_per_epoch = max(1, math.ceil(num_train_batches / accum_steps))
    total_updates = max(1, updates_per_epoch * max(int(args.epochs), 1))
    warmup_steps = int(total_updates * args.warmup_ratio)
    scheduler = get_cosine_schedule_with_warmup(
        optimizer=optimizer,
        num_warmup_steps=warmup_steps,
        num_training_steps=total_updates,
    )

    amp_dtype = to_autocast_dtype(load_dtype)
    loss_answer_weight = float(getattr(args, "loss_answer_weight", 1.0))
    run_val_metrics_every_n = max(1, int(getattr(args, "run_val_metrics_every_n_epochs", 5)))
    print(
        "[INFO] loss: answer_weight={loss_answer_weight:.4f} "
        f"(pure autoregressive NLL; CLIP retrieval for object at eval/test)"
    )
    checkpoint_monitor = str(getattr(args, "checkpoint_monitor", "val_loss")).strip() or "val_loss"
    checkpoint_monitor_mode = infer_checkpoint_monitor_mode(
        checkpoint_monitor,
        str(getattr(args, "checkpoint_monitor_mode", "auto")),
    )
    best_monitor_value = float("inf") if checkpoint_monitor_mode == "min" else -float("inf")
    best_val_loss = float("inf")
    print(f"[INFO] checkpoint monitor: {checkpoint_monitor} ({checkpoint_monitor_mode})")
    global_step = 0
    start_time = time.time()

    effective_epochs = 0 if bool(args.eval_only) else int(args.epochs)
    if bool(args.eval_only):
        print("[INFO] eval_only=True; skipping training loop.")

    for epoch in range(1, effective_epochs + 1):
        model.train()
        sum_loss = 0.0
        sample_count = 0
        step_count = 0
        updates_done_in_epoch = 0
        last_logged_train_bucket = 0
        skipped_all_ignore = 0

        optimizer.zero_grad(set_to_none=True)
        train_iter = tqdm(
            train_loader,
            desc=f"Train {epoch}/{args.epochs}",
            leave=False,
            dynamic_ncols=True,
            disable=not args.show_tqdm,
        )

        remainder_steps = num_train_batches % accum_steps
        last_window_start = (
            (num_train_batches - int(remainder_steps) + 1)
            if int(remainder_steps) > 0
            else (num_train_batches + 1)
        )

        for step, batch in enumerate(train_iter, start=1):
            labels = batch["labels"].to(device)
            if torch.all(labels.eq(-100)):
                skipped_all_ignore += 1
                continue

            joint_inputs = to_device(batch["joint_inputs"], device=device)
            bsz = int(labels.shape[0])
            is_last_batch = (step == num_train_batches)
            current_accum_steps = (
                int(remainder_steps)
                if (int(remainder_steps) > 0 and step >= int(last_window_start))
                else int(accum_steps)
            )

            with torch.autocast(
                device_type=device.type,
                dtype=amp_dtype,
                enabled=(device.type == "cuda"),
            ):
                out = model(joint_inputs=joint_inputs, use_cache=False)
                losses = compute_answer_loss(
                    logits=out.get("logits", None),
                    labels=labels,
                    loss_mask_answer=batch.get("loss_mask_answer", None),
                    weight_answer=loss_answer_weight,
                )
                raw_loss = losses["loss"]
                loss = raw_loss / float(max(current_accum_steps, 1))

            loss.backward()

            should_step = ((step % accum_steps) == 0) or is_last_batch
            if should_step:
                grad_norm = torch.nn.utils.clip_grad_norm_(trainable_params, max_norm=args.max_grad_norm)
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad(set_to_none=True)
                global_step += 1
                updates_done_in_epoch += 1

                if wandb_run is not None:
                    current_train_bucket = train_progress_bucket(
                        updates_done_in_epoch,
                        updates_per_epoch,
                    )
                    should_log_train = current_train_bucket > last_logged_train_bucket
                    if should_log_train:
                        last_logged_train_bucket = current_train_bucket
                        grad_norm_value = (
                            float(grad_norm.detach().item())
                            if torch.is_tensor(grad_norm)
                            else float(grad_norm)
                        )
                        epoch_progress = (float(epoch) - 1.0) + (
                            float(updates_done_in_epoch) / max(float(updates_per_epoch), 1.0)
                        )
                        wandb_run.log(
                            {
                                "train/loss": float(raw_loss.detach().item()),
                                "train/learning_rate": float(optimizer.param_groups[0]["lr"]),
                                "train/grad_norm": grad_norm_value,
                                "train/epoch": epoch_progress,
                            },
                            step=global_step,
                        )

            sum_loss += float(raw_loss.detach().item()) * float(bsz)
            sample_count += bsz
            step_count += 1
            if args.show_tqdm:
                train_iter.set_postfix(loss=f"{(sum_loss / max(sample_count, 1)):.4f}")

        if step_count == 0 or sample_count == 0:
            raise RuntimeError(
                "No effective training batches were produced. "
                "All batches may have empty target text (all labels ignored)."
            )

        train_loss = float(sum_loss / float(sample_count))
        if skipped_all_ignore > 0:
            print(f"[INFO] skipped all-ignore batches in train epoch: {skipped_all_ignore}")

        val_metrics = (
            run_eval(
                model,
                val_loader,
                device,
                amp_dtype,
                loss_answer_weight=loss_answer_weight,
                show_tqdm=bool(args.show_tqdm),
                desc=f"Eval {epoch}/{args.epochs}",
            )
            if len(val_ds) > 0
            else {"loss": train_loss}
        )
        val_loss = float(val_metrics.get("loss", train_loss))

        val_gen_metrics = None
        _run_val_gen = (
            val_metric_loader is not None
            and len(val_ds) > 0
            and (epoch % run_val_metrics_every_n == 0 or epoch == effective_epochs)
        )
        if _run_val_gen:
            val_gen_metrics = run_test_metrics(
                model=model,
                loader=val_metric_loader,
                device=device,
                amp_dtype=amp_dtype,
                processor=processor,
                label_embedding_bank=retrieval_label_embedding_bank.to(device=device),
                retrieval_label_texts=retrieval_label_texts,
                retrieval_top_k=int(retrieval_top_k),
                clip_encoder=clip_encoder,
                object_temperature=object_temperature,
                show_tqdm=bool(args.show_tqdm),
                desc=f"ValMetric {epoch}/{args.epochs}",
                max_new_tokens=int(args.generation_max_new_tokens),
                num_beams=int(getattr(args, "generation_num_beams", 1)),
                repetition_penalty=float(getattr(args, "repetition_penalty", 1.0)),
                no_repeat_ngram_size=int(getattr(args, "no_repeat_ngram_size", 0)),
            )

        epoch_msg = (
            f"[EPOCH {epoch}] train_loss={train_loss:.6f} val_loss={val_loss:.6f}"
        )
        if isinstance(val_gen_metrics, dict):
            epoch_msg += (
                f" val_dist={float(val_gen_metrics.get('PointL2', 0.0)):.6f}"
                f" val_acc1={float(val_gen_metrics.get('acc@1', 0.0)):.6f}"
                f" val_acc3={float(val_gen_metrics.get('acc@3', 0.0)):.6f}"
            )
        print(epoch_msg)

        if wandb_run is not None:
            payload = {
                "val/loss": float(val_loss),
            }
            if isinstance(val_gen_metrics, dict):
                payload.update(val_metric_log_payload(val_gen_metrics))
            wandb_run.log(payload, step=global_step)

        monitor_value = checkpoint_monitor_value(
            checkpoint_monitor,
            val_loss=val_loss,
            val_gen_metrics=val_gen_metrics,
        )
        monitor_improved = (
            monitor_value is not None
            and (
                (checkpoint_monitor_mode == "min" and monitor_value < best_monitor_value)
                or (checkpoint_monitor_mode == "max" and monitor_value > best_monitor_value)
            )
        )
        if monitor_value is None:
            print(f"[WARN] checkpoint monitor {checkpoint_monitor!r} is unavailable this epoch; skipping save.")
        elif monitor_improved:
            best_monitor_value = float(monitor_value)
            best_val_loss = val_loss
            best_dir = out_dir / "best"
            print(
                f"[INFO] new best checkpoint: {checkpoint_monitor}={best_monitor_value:.6f} "
                f"val_loss={val_loss:.6f}"
            )
            save_checkpoint(
                best_dir,
                epoch,
                model,
                processor,
                optimizer,
                scheduler,
                clear_dir=True,
            )

        # Always save the most recent epoch to last/
        save_checkpoint(
            out_dir / "last",
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
            loss_answer_weight=loss_answer_weight,
            show_tqdm=bool(args.show_tqdm),
            desc="Eval (checkpoint)",
        )
        best_val_loss = float(val_metrics.get("loss", best_val_loss))
        val_gen_metrics = None
        if val_metric_loader is not None:
            val_gen_metrics = run_test_metrics(
                model=model,
                loader=val_metric_loader,
                device=device,
                amp_dtype=amp_dtype,
                processor=processor,
                label_embedding_bank=retrieval_label_embedding_bank.to(device=device),
                retrieval_label_texts=retrieval_label_texts,
                retrieval_top_k=int(retrieval_top_k),
                clip_encoder=clip_encoder,
                object_temperature=object_temperature,
                show_tqdm=bool(args.show_tqdm),
                desc="Eval metrics (checkpoint)",
                max_new_tokens=int(args.generation_max_new_tokens),
                num_beams=int(getattr(args, "generation_num_beams", 1)),
                repetition_penalty=float(getattr(args, "repetition_penalty", 1.0)),
                no_repeat_ngram_size=int(getattr(args, "no_repeat_ngram_size", 0)),
            )
        print(f"[EVAL] val_loss={best_val_loss:.6f}")
        if wandb_run is not None:
            payload = {
                "val/loss": float(best_val_loss),
            }
            if isinstance(val_gen_metrics, dict):
                payload.update(val_metric_log_payload(val_gen_metrics))
            wandb_run.log(payload, step=global_step)

    elapsed = time.time() - start_time

    if args.run_test:
        best_dir = out_dir / "best"
        if best_dir.exists():
            loaded_best = load_checkpoint_for_eval(
                ckpt_dir=best_dir,
                model=model,
                device=device,
            )
            if loaded_best:
                print(f"[INFO] loaded best checkpoint for test: {best_dir}")
            else:
                print(f"[WARN] best checkpoint exists but could not be loaded fully: {best_dir}")
        else:
            print("[WARN] best checkpoint directory not found; testing current in-memory model.")

        test_groups = load_test_groups(
            annotation_file=test_ann,
            image_root=test_image_root,
            test_label_map=test_label_map,
            test_label_text_map=test_label_text_map,
            test_label_ids_map=test_label_ids_map,
            split_prefix=args.test_split_prefix,
            strip_split_prefix=bool(args.test_strip_split_prefix),
            bbox_round_decimals=int(args.test_bbox_round_decimals),
        )
        if not test_groups:
            print("[TEST] no valid test groups found.")
        else:
            print(f"[TEST] groups={len(test_groups)}")
            test_ds = GazeTestDataset(
                groups=test_groups,
                prompt_template=args.prompt_template,
                prompt_text=prompt_text_for_run,
                id2label=id2label,
                vocab2id=vocab2id,
                vocab2id_lower=vocab2id_lower,
                num_classes=int(num_classes),
                answer_template=args.answer_template,
                fallback_target_text=args.fallback_target_text,
                point_decimals=int(args.point_decimals),
                visual_prompting=bool(args.visual_prompting),
                image_cache_size=max(0, int(getattr(args, "image_cache_size", 0))),
            )
            log_target_example("test", test_ds)
            test_loader = DataLoader(
                test_ds,
                batch_size=max(1, int(args.test_batch_size)),
                shuffle=False,
                num_workers=_nw,
                pin_memory=(device.type == "cuda"),
                collate_fn=test_collator,
                persistent_workers=_persistent,
                prefetch_factor=_prefetch,
            )
            test_metrics = run_test_metrics(
                model=model,
                loader=test_loader,
                device=device,
                amp_dtype=amp_dtype,
                processor=processor,
                label_embedding_bank=retrieval_label_embedding_bank.to(device=device),
                retrieval_label_texts=retrieval_label_texts,
                retrieval_top_k=int(retrieval_top_k),
                clip_encoder=clip_encoder,
                object_temperature=object_temperature,
                show_tqdm=bool(args.show_tqdm),
                desc="Test",
                max_new_tokens=int(args.generation_max_new_tokens),
                num_beams=int(getattr(args, "generation_num_beams", 1)),
                repetition_penalty=float(getattr(args, "repetition_penalty", 1.0)),
                no_repeat_ngram_size=int(getattr(args, "no_repeat_ngram_size", 0)),
            )
            print_test_metrics_table(test_metrics)
            if wandb_run is not None:
                wandb_run.log(test_log_payload(test_metrics), step=global_step)
            (out_dir / "test_metrics.json").write_text(
                json.dumps(test_metrics, ensure_ascii=False, indent=2),
                encoding="utf-8",
            )
            maybe_save_generation_preview(
                args=args,
                out_dir=out_dir,
                model=model,
                loader=test_loader,
                device=device,
                amp_dtype=amp_dtype,
                processor=processor,
                label_embedding_bank=retrieval_label_embedding_bank.to(device=device),
                retrieval_label_texts=retrieval_label_texts,
                retrieval_top_k=int(retrieval_top_k),
                clip_encoder=clip_encoder,
                object_temperature=object_temperature,
            )

    finish_wandb(wandb_run)
    print(
        f"[DONE] global_step={global_step} best_val_loss={best_val_loss:.6f} "
        f"best_{checkpoint_monitor}={best_monitor_value:.6f} elapsed_sec={elapsed:.1f}"
    )


if __name__ == "__main__":
    main()
