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
from .utils.config_parser import build_parser, load_yaml_config
from .utils.data_utils import (
    build_vocab_embedding_matrix,
    load_label_map,
    load_label_text_map,
    load_records,
    load_test_vocab_texts,
    load_test_groups,
    load_test_label_map,
    load_vocab2id,
)
from .utils.eval_utils import print_test_metrics_table, run_eval, run_test_metrics
from .utils.loss_utils import compute_structured_losses
from .utils.object_tokens import (
    OBJ_SLOT,
    add_slot_token,
)
from .utils.processor_collate import QwenTestCollator, QwenTrainCollator
from .utils.wandb_utils import finish_wandb, init_wandb


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
        label_id = int(getattr(r, "label_id", -100))
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
            "vocab2id ids must be exactly contiguous 0..N-1 for object token migration. "
            f"path={vocab2id_path} N={n} got_min={min(got)} got_max={max(got)} "
            f"head={first} tail={last}"
        )
    return n


def ensure_slot_token(
    processor: Any,
) -> int:
    tok = getattr(processor, "tokenizer", None)
    if tok is None:
        raise RuntimeError("processor.tokenizer is missing; cannot register object slot token.")
    return int(add_slot_token(tok))


def resize_embeddings(model: Any, processor: Any) -> bool:
    tok = getattr(processor, "tokenizer", None)
    if tok is None:
        return False
    if not hasattr(model, "resize_token_embeddings"):
        return False
    target_vocab = int(len(tok))
    current_vocab = None
    try:
        emb = model.get_input_embeddings()
        current_vocab = int(getattr(emb, "num_embeddings", -1))
    except Exception:
        current_vocab = None
    if current_vocab is not None and current_vocab == target_vocab:
        return False
    try:
        model.resize_token_embeddings(target_vocab)
    except Exception as e:
        raise RuntimeError(
            f"Failed to resize model token embeddings to tokenizer size={target_vocab}: {e}"
        ) from e
    try:
        new_vocab = int(getattr(model.get_input_embeddings(), "num_embeddings", -1))
    except Exception as e:
        raise RuntimeError(f"Could not verify resized embedding size after resize_token_embeddings: {e}") from e
    if new_vocab != target_vocab:
        raise RuntimeError(
            f"Embedding resize mismatch: tokenizer_vocab={target_vocab} model_vocab={new_vocab}."
        )
    return True


def init_processor(
    *,
    model_path: Path,
    checkpoint_dir: Path | None,
) -> Any:
    processor_path = model_path
    if checkpoint_dir is not None and (checkpoint_dir / "processor").exists():
        processor_path = checkpoint_dir / "processor"
    processor = AutoProcessor.from_pretrained(str(processor_path), trust_remote_code=True)
    n_added_slot_token = ensure_slot_token(processor)
    print(
        "[INFO] object slot token: "
        f"token={OBJ_SLOT} added={int(n_added_slot_token)} source={processor_path}"
    )
    return processor


def init_base_model(
    *,
    model_path: Path,
    model_kwargs: dict[str, Any],
    processor: Any,
) -> Any:
    base_qwen = AutoModelForImageTextToText.from_pretrained(str(model_path), **model_kwargs)
    if resize_embeddings(base_qwen, processor):
        print(f"[INFO] resized model token embeddings to tokenizer size={len(processor.tokenizer)}")
    return base_qwen


def test_log_payload(test_metrics: dict[str, float], epoch: float) -> dict[str, float]:
    return {
        "test/epoch": float(epoch),
        "test/ExactMatch": float(test_metrics.get("ExactMatch", 0.0)),
        "test/Contains": float(test_metrics.get("Contains", 0.0)),
        "test/AvgL2": float(test_metrics.get("Avg L2", test_metrics.get("PointL2", 0.0))),
        "test/MinL2": float(test_metrics.get("Min L2", 0.0)),
        "test/PointL2": float(test_metrics.get("PointL2", 0.0)),
        "test/acc@1": float(test_metrics.get("acc@1", 0.0)),
        "test/acc@3": float(test_metrics.get("acc@3", 0.0)),
        "test/multiacc@1": float(test_metrics.get("multiacc@1", 0.0)),
        "test/ObjectTokenValidRate": float(test_metrics.get("ObjectTokenValidRate", 0.0)),
        "test/ObjectParseFailTop1Rate": float(test_metrics.get("ObjectParseFailTop1Rate", 0.0)),
        "test/ObjectParseFailBeamRate": float(test_metrics.get("ObjectParseFailBeamRate", 0.0)),
        "test/ObjectParseTop1FromTokenRate": float(test_metrics.get("ObjectParseTop1FromTokenRate", 0.0)),
        "test/ObjectParseBeamFromTokenRate": float(test_metrics.get("ObjectParseBeamFromTokenRate", 0.0)),
        "test/ObjectParseFailTop1Count": float(test_metrics.get("ObjectParseFailTop1Count", 0.0)),
        "test/ObjectParseFailBeamCount": float(test_metrics.get("ObjectParseFailBeamCount", 0.0)),
        "test/ObjectParseTop1FromTokenCount": float(test_metrics.get("ObjectParseTop1FromTokenCount", 0.0)),
        "test/ObjectParseBeamFromTokenCount": float(test_metrics.get("ObjectParseBeamFromTokenCount", 0.0)),
        "test/RetrievalQueryValidRate": float(test_metrics.get("RetrievalQueryValidRate", 0.0)),
        "test/RetrievalAcc@1": float(test_metrics.get("RetrievalAcc@1", 0.0)),
        "test/RetrievalAcc@3": float(test_metrics.get("RetrievalAcc@3", 0.0)),
        "test/RetrievalMultiAcc@1": float(test_metrics.get("RetrievalMultiAcc@1", 0.0)),
        "test/RetrievalDen": float(test_metrics.get("RetrievalDen", 0.0)),
        "test/num_samples": float(test_metrics.get("num_samples", 0.0)),
        "test/num_valid_targets": float(test_metrics.get("num_valid_targets", 0.0)),
    }


def val_metric_log_payload(val_metrics: dict[str, float], epoch: float) -> dict[str, float]:
    point_l2 = float(val_metrics.get("PointL2", val_metrics.get("Avg L2", 0.0)))
    acc1 = float(val_metrics.get("acc@1", 0.0))
    acc3 = float(val_metrics.get("acc@3", 0.0))
    multiacc1 = float(val_metrics.get("multiacc@1", 0.0))
    return {
        "val/epoch": float(epoch),
        "val/dist": point_l2,
        "val/acc@1": acc1,
        "val/acc@3": acc3,
        "val/multiacc@1": multiacc1,
        "metric/val/dist": point_l2,
        "metric/val/acc@1": acc1,
        "metric/val/acc@3": acc3,
        "metric/val/multiacc@1": multiacc1,
        "val/num_samples": float(val_metrics.get("num_samples", 0.0)),
        "val/num_valid_targets": float(val_metrics.get("num_valid_targets", 0.0)),
    }


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
    labels_rgs = resolve_path(args.labels_rgs) if str(getattr(args, "labels_rgs", "")).strip() else None
    labels_ssa = resolve_path(args.labels_ssa) if str(getattr(args, "labels_ssa", "")).strip() else None
    vocab2id_path = resolve_path(args.vocab2id)

    vocab2id, vocab2id_lower = load_vocab2id(vocab2id_path)
    num_classes = infer_num_classes(vocab2id, vocab2id_path)
    id2label = build_id2label(vocab2id)
    print(f"[INFO] loaded vocab2id classes: {len(vocab2id)} (id_range=0..{max(num_classes - 1, 0)})")

    prompt_text_for_run = str(args.prompt_text or "")
    # Pure-text mode: prompt_text in config.yaml already instructs the model to
    # output "Object: <object name>".  No automatic injection of <obj_emb> marker.

    label_embed_dir = (
        resolve_path(getattr(args, "label_embed_dir"))
        if str(getattr(args, "label_embed_dir", "")).strip()
        else None
    )
    object_embedding_dim = int(getattr(args, "object_embedding_dim", 512))
    object_temperature = float(getattr(args, "object_temperature", 0.07))
    test_retrieval_top_k = max(1, int(getattr(args, "test_retrieval_top_k", 3)))
    object_loss_mode = str(getattr(args, "object_loss_mode", "retrieval")).strip().lower()
    if object_loss_mode not in {"retrieval", "infonce", "full_bank_retrieval"}:
        raise RuntimeError(
            f"Unsupported object_loss_mode={object_loss_mode!r}. "
            "Use one of: retrieval, infonce, full_bank_retrieval."
        )

    train_label_embedding_bank = build_vocab_embedding_matrix(
        vocab2id=vocab2id,
        label_embed_dir=label_embed_dir,
        label_emb_dim=int(object_embedding_dim),
        normalize=True,
    )
    if train_label_embedding_bank is None:
        raise RuntimeError(
            "Failed to build training label embedding bank for object retrieval loss. "
            f"label_embed_dir={label_embed_dir} object_embedding_dim={object_embedding_dim}"
        )
    print(
        "[INFO] train label embedding bank: "
        f"shape={tuple(train_label_embedding_bank.shape)} "
        f"dim={int(object_embedding_dim)} mode={object_loss_mode}"
    )

    test_vocab_texts = load_test_vocab_texts(test_labels)
    test_vocab2id = {str(txt): int(i) for i, txt in enumerate(test_vocab_texts)}
    test_label_embedding_bank_raw = build_vocab_embedding_matrix(
        vocab2id=test_vocab2id,
        label_embed_dir=label_embed_dir,
        label_emb_dim=int(object_embedding_dim),
        normalize=True,
    )
    if test_label_embedding_bank_raw is None:
        raise RuntimeError(
            "Failed to build test label embedding bank for retrieval inference. "
            f"label_embed_dir={label_embed_dir} object_embedding_dim={object_embedding_dim}"
        )
    test_bank_norm = test_label_embedding_bank_raw.norm(dim=1)
    keep_rows = test_bank_norm.gt(0)
    if int(keep_rows.sum().item()) <= 0:
        raise RuntimeError("Test label embedding bank has no valid rows after filtering zero vectors.")
    keep_idx = [int(i) for i in torch.nonzero(keep_rows, as_tuple=False).flatten().tolist()]
    test_retrieval_texts = [str(test_vocab_texts[i]) for i in keep_idx]
    test_label_embedding_bank = test_label_embedding_bank_raw[keep_rows].contiguous()
    print(
        "[INFO] test label embedding bank: "
        f"raw_vocab={len(test_vocab_texts)} valid_rows={len(test_retrieval_texts)} "
        f"shape={tuple(test_label_embedding_bank.shape)} topk={test_retrieval_top_k}"
    )
    retrieval_query_text_to_label_id = {
        label_key(k): int(v) for k, v in vocab2id.items()
    }

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

        processor = init_processor(
            model_path=model_path,
            checkpoint_dir=checkpoint_dir,
        )

        print("[INFO] collator raw input passthrough: test=enabled.")
        test_collator = QwenTestCollator(
            processor=processor,
            scene_size=(int(args.scene_h), int(args.scene_w)),
            max_text_length=int(args.max_text_length),
            include_raw_inputs=True,
        )

        base_qwen = init_base_model(
            model_path=model_path,
            model_kwargs=model_kwargs,
            processor=processor,
        )
        adapter_dir = (checkpoint_dir / "lora_adapter") if checkpoint_dir is not None else None
        if adapter_dir is not None and adapter_dir.exists():
            qwen_model = PeftModel.from_pretrained(
                base_qwen,
                model_id=str(adapter_dir),
                is_trainable=False,
            )
            print(f"[INFO] loaded LoRA adapter from: {adapter_dir}")
        else:
            qwen_model = base_qwen
            print("[INFO] adapter checkpoint not found; running zero-shot base model.")

        model = QwenTextGenerationModel(
            qwen_model=qwen_model,
            object_embedding_dim=int(object_embedding_dim),
        ).to(device)
        train_label_embedding_bank_device = train_label_embedding_bank.to(device=device)
        test_label_embedding_bank_device = test_label_embedding_bank.to(device=device)
        if checkpoint_dir is not None:
            obj_proj_path = checkpoint_dir / "object_projector.pt"
            if obj_proj_path.exists():
                try:
                    st = torch.load(obj_proj_path, map_location=device)
                    model.object_projector.load_state_dict(st, strict=False)
                    print(f"[INFO] loaded object projector from: {obj_proj_path}")
                except Exception as e:
                    print(f"[WARN] failed to load object projector from {obj_proj_path}: {e}")
        model.eval()

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
        start_index = max(0, int(getattr(args, "test_start_index", 0)))
        if start_index > 0:
            test_groups = test_groups[start_index:]
        test_num_samples = int(getattr(args, "test_num_samples", 0))
        if test_num_samples > 0:
            test_groups = test_groups[: max(1, test_num_samples)]

        if not test_groups:
            print("[TEST] no valid test groups found.")
        else:
            print(
                f"[TEST] groups={len(test_groups)} "
                f"(start={start_index}, limit={(test_num_samples if test_num_samples > 0 else 'all')})"
            )
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
            )
            log_target_example("test_only", test_ds)
            test_loader = DataLoader(
                test_ds,
                batch_size=max(1, int(args.test_batch_size)),
                shuffle=False,
                num_workers=args.num_workers,
                pin_memory=(device.type == "cuda"),
                collate_fn=test_collator,
            )
            test_metrics = run_test_metrics(
                model=model,
                loader=test_loader,
                device=device,
                amp_dtype=amp_dtype,
                processor=processor,
                label_embedding_bank=test_label_embedding_bank_device,
                retrieval_label_texts=test_retrieval_texts,
                query_text_to_label_id=retrieval_query_text_to_label_id,
                query_id_to_label_text=id2label,
                retrieval_top_k=int(test_retrieval_top_k),
                object_temperature=object_temperature,
                show_tqdm=bool(args.show_tqdm),
                desc="Test",
                max_new_tokens=int(args.generation_max_new_tokens),
                num_beams=int(getattr(args, "generation_num_beams", 3)),
            )
            print_test_metrics_table(test_metrics)
            if wandb_run is not None:
                wandb_run.log(test_log_payload(test_metrics, epoch=0.0), step=0)
            (out_dir / "test_metrics.json").write_text(
                json.dumps(test_metrics, ensure_ascii=False, indent=2),
                encoding="utf-8",
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
        label_embed_dir=label_embed_dir,
        label_emb_dim=int(object_embedding_dim),
        use_embed_fallback=True,
        fallback_csvs=[labels_rgs, labels_ssa],
        fallback_text_key="annotation",
        split_name="train",
    )
    val_label_map, val_label_stats = load_label_map(
        val_labels,
        vocab2id=vocab2id,
        vocab2id_lower=vocab2id_lower,
        text_key="gaze_pseudo_label",
        label_embed_dir=label_embed_dir,
        label_emb_dim=int(object_embedding_dim),
        use_embed_fallback=True,
        fallback_csvs=[labels_rgs, labels_ssa],
        fallback_text_key="annotation",
        split_name="validation",
    )

    train_label_text_map, train_label_text_stats = load_label_text_map(
        train_labels,
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

    print(
        "[INFO] train label id coverage: "
        f"rows={train_label_stats['rows']} mapped={train_label_stats['mapped']} "
        f"missing_text={train_label_stats['missing_text']} unknown_text={train_label_stats['unknown_text']} "
        f"primary_mapped={train_label_stats.get('primary_mapped', 0)} "
        f"embed_fallback_mapped={train_label_stats.get('embed_fallback_mapped', 0)} "
        f"csv_fallback_mapped={train_label_stats.get('fallback_mapped', 0)}"
    )
    print(
        "[INFO] val label id coverage: "
        f"rows={val_label_stats['rows']} mapped={val_label_stats['mapped']} "
        f"missing_text={val_label_stats['missing_text']} unknown_text={val_label_stats['unknown_text']} "
        f"primary_mapped={val_label_stats.get('primary_mapped', 0)} "
        f"embed_fallback_mapped={val_label_stats.get('embed_fallback_mapped', 0)} "
        f"csv_fallback_mapped={val_label_stats.get('fallback_mapped', 0)}"
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
    print(
        "[INFO] test label map: "
        f"rows={test_label_stats['rows']} mapped={test_label_stats['mapped']} "
        f"missing_text={test_label_stats['missing_text']} unknown_text={test_label_stats['unknown_text']} "
        f"conflicts={test_label_stats['conflicts']}"
    )

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

    train_valid_targets = count_valid_targets(train_records)
    val_valid_targets = count_valid_targets(val_records)
    print(
        f"[INFO] train_records={len(train_records)} val_records={len(val_records)} "
        f"train_valid_targets={train_valid_targets} val_valid_targets={val_valid_targets}"
    )

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

    processor = init_processor(
        model_path=model_path,
        checkpoint_dir=checkpoint_dir,
    )

    # raw inputs (scene_images, text_inputs) are only needed at test time for
    # pred_embeddings_from_generated()'s second re-encode pass.
    # Passing them through train/val batches wastes CPU memory for no benefit.
    print("[INFO] collator raw input passthrough: train/val=disabled, test=enabled.")
    train_collator = QwenTrainCollator(
        processor=processor,
        scene_size=(int(args.scene_h), int(args.scene_w)),
        max_text_length=int(args.max_text_length),
        include_raw_inputs=False,
    )
    val_collator = QwenTrainCollator(
        processor=processor,
        scene_size=(int(args.scene_h), int(args.scene_w)),
        max_text_length=int(args.max_text_length),
        include_raw_inputs=False,
    )
    test_collator = QwenTestCollator(
        processor=processor,
        scene_size=(int(args.scene_h), int(args.scene_w)),
        max_text_length=int(args.max_text_length),
        include_raw_inputs=True,
    )

    train_loader = DataLoader(
        train_ds,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=(device.type == "cuda"),
        collate_fn=train_collator,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=(device.type == "cuda"),
        collate_fn=val_collator,
    )
    val_metric_loader = None
    if bool(getattr(args, "run_val_metrics", True)):
        val_metric_loader = DataLoader(
            val_ds,
            batch_size=max(1, int(args.test_batch_size)),
            shuffle=False,
            num_workers=args.num_workers,
            pin_memory=(device.type == "cuda"),
            collate_fn=test_collator,
        )

    base_qwen = init_base_model(
        model_path=model_path,
        model_kwargs=model_kwargs,
        processor=processor,
    )
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

    model = QwenTextGenerationModel(
        qwen_model=qwen_lora,
        object_embedding_dim=int(object_embedding_dim),
    ).to(device)
    train_label_embedding_bank_device = train_label_embedding_bank.to(device=device)
    test_label_embedding_bank_device = test_label_embedding_bank.to(device=device)

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
    loss_answer_weight = float(getattr(args, "loss_answer_weight", 0.1))
    loss_point_weight = float(getattr(args, "loss_point_weight", 1.0))
    loss_object_weight = float(getattr(args, "loss_object_weight", 1.5))
    loss_slot_weight = float(getattr(args, "loss_slot_weight", 0.0))
    loss_use_lm_fallback = bool(getattr(args, "loss_use_lm_fallback", False))
    print(
        "[INFO] structured loss weights: "
        f"answer={loss_answer_weight:.4f} "
        f"point={loss_point_weight:.4f} "
        f"object={loss_object_weight:.4f} "
        f"slot={loss_slot_weight:.4f} "
        f"object_mode={object_loss_mode} "
        f"object_temp={object_temperature:.4f} "
        f"lm_fallback={str(loss_use_lm_fallback).lower()}"
    )
    best_val_loss = float("inf")
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
            object_slot_mask = batch.get("loss_mask_object", None)
            if torch.is_tensor(object_slot_mask):
                object_slot_mask = object_slot_mask.to(device=device)
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
                # Pass labels only when lm_fallback is enabled; otherwise Qwen
                # computes a full-sequence LM loss that is never used, wasting
                # memory and compute every step.
                pass_labels = labels if loss_use_lm_fallback else None
                out = model(
                    joint_inputs=joint_inputs,
                    labels=pass_labels,
                    object_slot_mask=object_slot_mask,
                    use_cache=False,
                )
                lm_loss = out.get("loss", None)
                if loss_use_lm_fallback and lm_loss is None:
                    raise RuntimeError("Model forward must return loss when lm_fallback is enabled.")
                structured = compute_structured_losses(
                    logits=out.get("logits", None),
                    labels=labels,
                    loss_mask_answer=batch.get("loss_mask_answer", None),
                    loss_mask_point=batch.get("loss_mask_point", None),
                    loss_mask_object=batch.get("loss_mask_object", None),
                    pred_object_emb=out.get("pred_object_emb", None),
                    target_label=batch.get("target_label", None).to(device) if torch.is_tensor(batch.get("target_label", None)) else None,
                    target_object_valid=(
                        batch.get("target_object_valid", None).to(device)
                        if torch.is_tensor(batch.get("target_object_valid", None))
                        else None
                    ),
                    label_embedding_bank=train_label_embedding_bank_device,
                    object_loss_mode=object_loss_mode,
                    object_temperature=object_temperature,
                    weight_answer=loss_answer_weight,
                    weight_point=loss_point_weight,
                    weight_object=loss_object_weight,
                    weight_slot=loss_slot_weight,
                    fallback_loss=(lm_loss if loss_use_lm_fallback else None),
                )
                raw_loss = structured["loss"]
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

                if (
                    wandb_run is not None
                    and int(args.wandb_log_every_steps) > 0
                    and (global_step % int(args.wandb_log_every_steps) == 0)
                ):
                    grad_norm_value = float(grad_norm.detach().item()) if torch.is_tensor(grad_norm) else float(grad_norm)
                    epoch_progress = (float(epoch) - 1.0) + (
                        float(updates_done_in_epoch) / max(float(updates_per_epoch), 1.0)
                    )
                    wandb_run.log(
                        {
                            "train/loss/total": float(structured["loss_total"].detach().item()),
                            "train/loss/answer": float(structured["loss_answer"].detach().item()),
                            "train/loss/point": float(structured["loss_point"].detach().item()),
                            "train/loss/object": float(structured["loss_object"].detach().item()),
                            "train/loss/slot": float(structured["loss_slot"].detach().item()),
                            "train/loss/lm_fallback": float(lm_loss.detach().item()) if torch.is_tensor(lm_loss) else 0.0,
                            "train/loss": float(raw_loss.detach().item()),
                            "train/loss_answer": float(structured["loss_answer"].detach().item()),
                            "train/loss_lm_fallback": float(lm_loss.detach().item()) if torch.is_tensor(lm_loss) else 0.0,
                            "train/learning_rate": float(optimizer.param_groups[0]["lr"]),
                            "train/grad_norm": grad_norm_value,
                            "train/global_step": float(global_step),
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
                loss_point_weight=loss_point_weight,
                loss_object_weight=loss_object_weight,
                loss_slot_weight=loss_slot_weight,
                loss_use_lm_fallback=loss_use_lm_fallback,
                label_embedding_bank=train_label_embedding_bank_device,
                object_loss_mode=object_loss_mode,
                object_temperature=object_temperature,
                show_tqdm=bool(args.show_tqdm),
                desc=f"Eval {epoch}/{args.epochs}",
            )
            if len(val_ds) > 0
            else {"loss": train_loss}
        )
        val_loss = float(val_metrics.get("loss", train_loss))
        val_gen_metrics = None
        if val_metric_loader is not None and len(val_ds) > 0:
            val_gen_metrics = run_test_metrics(
                model=model,
                loader=val_metric_loader,
                device=device,
                amp_dtype=amp_dtype,
                processor=processor,
                label_embedding_bank=test_label_embedding_bank_device,
                retrieval_label_texts=test_retrieval_texts,
                query_text_to_label_id=retrieval_query_text_to_label_id,
                query_id_to_label_text=id2label,
                retrieval_top_k=int(test_retrieval_top_k),
                object_temperature=object_temperature,
                show_tqdm=bool(args.show_tqdm),
                desc=f"ValMetric {epoch}/{args.epochs}",
                max_new_tokens=int(args.generation_max_new_tokens),
                num_beams=int(getattr(args, "generation_num_beams", 3)),
            )

        epoch_msg = (
            f"[EPOCH {epoch}] "
            f"train_loss={train_loss:.6f} "
            f"val_loss={val_loss:.6f} "
            f"val_point={float(val_metrics.get('loss_point', 0.0)):.6f} "
            f"val_object={float(val_metrics.get('loss_object', 0.0)):.6f}"
        )
        if isinstance(val_gen_metrics, dict):
            epoch_msg += (
                f" val_dist={float(val_gen_metrics.get('PointL2', val_gen_metrics.get('Avg L2', 0.0))):.6f} "
                f"val_acc1={float(val_gen_metrics.get('acc@1', 0.0)):.6f} "
                f"val_acc3={float(val_gen_metrics.get('acc@3', 0.0)):.6f}"
            )
        print(epoch_msg)

        if wandb_run is not None:
            payload = {
                "epoch/index": float(epoch),
                "epoch/global_step": float(global_step),
                "epoch/train_loss": float(train_loss),
                "val/epoch": float(epoch),
                "val/loss": float(val_loss),
                "val/loss/total": float(val_metrics.get("loss_total", val_loss)),
                "val/loss/answer": float(val_metrics.get("loss_answer", 0.0)),
                "val/loss/point": float(val_metrics.get("loss_point", 0.0)),
                "val/loss/object": float(val_metrics.get("loss_object", 0.0)),
                "metric/val/loss": float(val_loss),
            }
            if isinstance(val_gen_metrics, dict):
                payload.update(val_metric_log_payload(val_gen_metrics, epoch=float(epoch)))
            wandb_run.log(payload, step=global_step)

        if val_loss < best_val_loss:
            best_val_loss = val_loss
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
            loss_answer_weight=loss_answer_weight,
            loss_point_weight=loss_point_weight,
            loss_object_weight=loss_object_weight,
            loss_slot_weight=loss_slot_weight,
            loss_use_lm_fallback=loss_use_lm_fallback,
            label_embedding_bank=train_label_embedding_bank_device,
            object_loss_mode=object_loss_mode,
            object_temperature=object_temperature,
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
                label_embedding_bank=test_label_embedding_bank_device,
                retrieval_label_texts=test_retrieval_texts,
                query_text_to_label_id=retrieval_query_text_to_label_id,
                query_id_to_label_text=id2label,
                retrieval_top_k=int(test_retrieval_top_k),
                object_temperature=object_temperature,
                show_tqdm=bool(args.show_tqdm),
                desc="Eval metrics (checkpoint)",
                max_new_tokens=int(args.generation_max_new_tokens),
                num_beams=int(getattr(args, "generation_num_beams", 3)),
            )
        print(
            "[EVAL] "
            f"val_loss={best_val_loss:.6f}"
        )
        if wandb_run is not None:
            payload = {
                "val/epoch": 0.0,
                "val/loss": float(best_val_loss),
                "val/loss/total": float(val_metrics.get("loss_total", best_val_loss)),
                "val/loss/answer": float(val_metrics.get("loss_answer", 0.0)),
                "val/loss/point": float(val_metrics.get("loss_point", 0.0)),
                "val/loss/object": float(val_metrics.get("loss_object", 0.0)),
                "metric/val/loss": float(best_val_loss),
            }
            if isinstance(val_gen_metrics, dict):
                payload.update(val_metric_log_payload(val_gen_metrics, epoch=0.0))
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
            max_groups=int(args.max_test_samples),
        )
        start_index = max(0, int(getattr(args, "test_start_index", 0)))
        if start_index > 0:
            test_groups = test_groups[start_index:]
        test_num_samples = int(getattr(args, "test_num_samples", 0))
        if test_num_samples > 0:
            test_groups = test_groups[: max(1, test_num_samples)]
        if not test_groups:
            print("[TEST] no valid test groups found.")
        else:
            print(
                f"[TEST] groups={len(test_groups)} "
                f"(start={start_index}, limit={(test_num_samples if test_num_samples > 0 else 'all')})"
            )
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
            )
            log_target_example("test", test_ds)
            test_loader = DataLoader(
                test_ds,
                batch_size=max(1, int(args.test_batch_size)),
                shuffle=False,
                num_workers=args.num_workers,
                pin_memory=(device.type == "cuda"),
                collate_fn=test_collator,
            )
            test_metrics = run_test_metrics(
                model=model,
                loader=test_loader,
                device=device,
                amp_dtype=amp_dtype,
                processor=processor,
                label_embedding_bank=test_label_embedding_bank_device,
                retrieval_label_texts=test_retrieval_texts,
                query_text_to_label_id=retrieval_query_text_to_label_id,
                query_id_to_label_text=id2label,
                retrieval_top_k=int(test_retrieval_top_k),
                object_temperature=object_temperature,
                show_tqdm=bool(args.show_tqdm),
                desc="Test",
                max_new_tokens=int(args.generation_max_new_tokens),
                num_beams=int(getattr(args, "generation_num_beams", 3)),
            )
            print_test_metrics_table(test_metrics)
            if wandb_run is not None:
                wandb_run.log(test_log_payload(test_metrics, epoch=float(effective_epochs)), step=global_step)
            (out_dir / "test_metrics.json").write_text(
                json.dumps(test_metrics, ensure_ascii=False, indent=2),
                encoding="utf-8",
            )

    finish_wandb(wandb_run)
    print(f"[DONE] global_step={global_step} best_val_loss={best_val_loss:.6f} elapsed_sec={elapsed:.1f}")


if __name__ == "__main__":
    main()
