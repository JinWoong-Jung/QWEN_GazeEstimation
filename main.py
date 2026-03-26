#!/usr/bin/env python3
from __future__ import annotations

import argparse
import fnmatch
import json
import os
import re
from pathlib import Path
from typing import Any

import requests
import torch
import yaml
from PIL import Image
from huggingface_hub import HfApi, hf_hub_url, snapshot_download
from transformers import AutoModelForImageTextToText, AutoProcessor

ROOT_DIR = Path(__file__).resolve().parent
DEFAULT_CONFIG = ROOT_DIR / "config.yaml"
MODEL_STORAGE_ROOT = ROOT_DIR / "model"

FLOAT_RE = re.compile(r"[-+]?(?:\d*\.\d+|\d+)(?:[eE][-+]?\d+)?")
XY_LINE_RE = re.compile(
    rf"^\s*({FLOAT_RE.pattern})\s*(?:,|\s)\s*({FLOAT_RE.pattern})\s*$"
)
XY_LABELED_RE = re.compile(
    rf"[xX]\s*[:=]\s*({FLOAT_RE.pattern})[^0-9+\-.eE]+[yY]\s*[:=]\s*({FLOAT_RE.pattern})"
)
XY_PAREN_RE = re.compile(
    rf"[\(\[]\s*({FLOAT_RE.pattern})\s*,\s*({FLOAT_RE.pattern})\s*[\)\]]"
)
HEAD_BBOX_RE = re.compile(
    r"Head bbox\s*\[xmin,ymin,xmax,ymax\]\s*\(normalized\):\s*\[([^\]]+)\]"
)


def load_yaml(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def resolve_path(base_dir: Path, raw_path: str) -> Path:
    p = Path(raw_path)
    if p.is_absolute():
        return p
    return (base_dir / p).resolve()


def ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def should_ignore(path: str, ignore_patterns: list[str]) -> bool:
    return any(fnmatch.fnmatch(path, pattern) for pattern in ignore_patterns)


def cleanup_partial_downloads(local_dir: Path) -> None:
    # Some object-storage mounts do not support append ("ab").
    # Remove partial artifacts so snapshot_download starts fresh.
    patterns = ("tmp_*", "*.incomplete", "*.lock")
    for pattern in patterns:
        for p in local_dir.glob(pattern):
            try:
                if p.is_dir():
                    for child in p.rglob("*"):
                        if child.is_file() or child.is_symlink():
                            child.unlink(missing_ok=True)
                    for d in sorted(
                        [x for x in p.rglob("*") if x.is_dir()],
                        key=lambda x: len(x.parts),
                        reverse=True,
                    ):
                        d.rmdir()
                    p.rmdir()
                else:
                    p.unlink(missing_ok=True)
            except Exception:
                # Best effort cleanup only.
                pass


def snapshot_download_no_append(
    repo_id: str,
    local_dir: Path,
    ignore_patterns: list[str],
) -> None:
    api = HfApi()
    repo_files = api.list_repo_files(repo_id=repo_id, repo_type="model")
    if not repo_files:
        raise RuntimeError(f"No files listed for repo: {repo_id}")

    kept = [f for f in repo_files if not should_ignore(f, ignore_patterns)]
    total = len(kept)
    for idx, rel_path in enumerate(kept, start=1):
        out_path = local_dir / rel_path
        if out_path.exists() and out_path.stat().st_size > 0:
            continue

        ensure_dir(out_path.parent)
        url = hf_hub_url(repo_id=repo_id, filename=rel_path, repo_type="model")
        tmp_path = out_path.with_suffix(out_path.suffix + ".tmpdl")
        print(f"[DL] ({idx}/{total}) {rel_path}")
        with requests.get(url, stream=True, timeout=120) as resp:
            resp.raise_for_status()
            with tmp_path.open("wb") as f:
                for chunk in resp.iter_content(chunk_size=8 * 1024 * 1024):
                    if chunk:
                        f.write(chunk)
        tmp_path.replace(out_path)


def model_name_from_repo(repo_id: str) -> str:
    return repo_id.split("/")[-1]


def select_device(device_cfg: str) -> str:
    if device_cfg != "auto":
        return device_cfg
    if torch.cuda.is_available():
        return "cuda"
    if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def select_torch_dtype(dtype_cfg: str) -> torch.dtype | str:
    if dtype_cfg == "auto":
        return "auto"
    mapping = {
        "float16": torch.float16,
        "bfloat16": torch.bfloat16,
        "float32": torch.float32,
    }
    if dtype_cfg not in mapping:
        raise ValueError(f"Unsupported torch_dtype: {dtype_cfg}")
    return mapping[dtype_cfg]


def prepare_model(
    model_cfg: dict[str, Any],
    config_dir: Path,
) -> tuple[Path, str, str, torch.dtype | str, bool]:
    repo_id = str(model_cfg.get("repo_id", "")).strip()
    local_model_dir_raw = str(model_cfg.get("local_model_dir", "")).strip()

    if local_model_dir_raw:
        local_dir = resolve_path(config_dir, local_model_dir_raw)
        model_name = str(model_cfg.get("model_name", "")).strip() or local_dir.name
        ensure_dir(local_dir)
    else:
        if not repo_id:
            raise ValueError("model.repo_id is required when model.local_model_dir is empty.")
        model_name = model_name_from_repo(repo_id)
        local_dir = MODEL_STORAGE_ROOT / model_name
        ensure_dir(local_dir)

    cache_dir = MODEL_STORAGE_ROOT / "cache"
    ensure_dir(cache_dir)

    os.environ["HF_HOME"] = str(cache_dir)
    os.environ["HUGGINGFACE_HUB_CACHE"] = str(cache_dir)
    os.environ["TRANSFORMERS_CACHE"] = str(cache_dir)
    os.environ.setdefault("HF_HUB_DISABLE_XET", "1")

    download_if_missing = bool(model_cfg.get("download_if_missing", True))
    has_model_files = (
        (any(local_dir.glob("*.json")) and any(local_dir.glob("*.safetensors")))
        or any(local_dir.glob("pytorch_model*.bin"))
    )

    if download_if_missing and not has_model_files:
        if not repo_id:
            raise ValueError(
                "model.download_if_missing=true requires model.repo_id when local model files are missing."
            )
        cleanup_partial_downloads(local_dir)
        ignore_patterns = ["*.msgpack", "*.h5", "*.ot", "*.tflite"]
        try:
            snapshot_download(
                repo_id=repo_id,
                local_dir=str(local_dir),
                max_workers=1,
                ignore_patterns=ignore_patterns,
            )
        except PermissionError as e:
            if "Operation not permitted" not in str(e):
                raise
            print(
                "[WARN] snapshot_download append-write failed on this filesystem; "
                "switching to no-append downloader."
            )
            snapshot_download_no_append(
                repo_id=repo_id,
                local_dir=local_dir,
                ignore_patterns=ignore_patterns,
            )

    device = select_device(str(model_cfg.get("device", "auto")))
    torch_dtype = select_torch_dtype(str(model_cfg.get("torch_dtype", "auto")))
    trust_remote_code = bool(model_cfg.get("trust_remote_code", True))
    return local_dir, model_name, device, torch_dtype, trust_remote_code


def build_processor_kwargs(model_cfg: dict[str, Any], trust_remote_code: bool) -> dict[str, Any]:
    kwargs = dict(model_cfg.get("processor_kwargs", {}) or {})
    kwargs.setdefault("trust_remote_code", trust_remote_code)
    return kwargs


def build_model_kwargs(
    model_cfg: dict[str, Any],
    trust_remote_code: bool,
    torch_dtype: torch.dtype | str,
) -> dict[str, Any]:
    kwargs = dict(model_cfg.get("model_kwargs", {}) or {})
    kwargs.setdefault("trust_remote_code", trust_remote_code)
    if torch_dtype != "auto":
        kwargs.setdefault("torch_dtype", torch_dtype)
    return kwargs


def build_generation_kwargs(generation_cfg: dict[str, Any]) -> dict[str, Any]:
    kwargs: dict[str, Any] = {}

    known_keys = (
        "max_new_tokens",
        "min_new_tokens",
        "do_sample",
        "temperature",
        "top_p",
        "top_k",
        "num_beams",
        "repetition_penalty",
        "length_penalty",
    )
    for key in known_keys:
        if key in generation_cfg:
            kwargs[key] = generation_cfg[key]

    extra_kwargs = dict(generation_cfg.get("extra_kwargs", {}) or {})
    kwargs.update(extra_kwargs)
    return kwargs


def list_batch_samples(input_cfg: dict[str, Any], config_dir: Path) -> list[tuple[Path, Path, str]]:
    image_dir = resolve_path(config_dir, str(input_cfg["image_dir"]))
    prompt_dir = resolve_path(config_dir, str(input_cfg["prompt_dir"]))
    image_exts = {
        ext.lower() for ext in input_cfg.get("image_extensions", [".jpg", ".jpeg", ".png"])
    }

    samples: list[tuple[Path, Path, str]] = []
    for image_path in sorted(image_dir.iterdir()):
        if not image_path.is_file() or image_path.suffix.lower() not in image_exts:
            continue
        stem = image_path.stem
        prompt_path = prompt_dir / f"{stem}.txt"
        if not prompt_path.exists():
            print(f"[SKIP] prompt not found: {prompt_path}")
            continue
        samples.append((image_path, prompt_path, stem))
    return samples


def list_single_sample(
    input_cfg: dict[str, Any],
    config_dir: Path,
) -> list[tuple[Path, Path | None, str]]:
    image_path = resolve_path(config_dir, str(input_cfg["image_path"]))
    prompt_path_raw = str(input_cfg.get("prompt_path", "")).strip()
    prompt_path = resolve_path(config_dir, prompt_path_raw) if prompt_path_raw else None
    return [(image_path, prompt_path, image_path.stem)]


def load_prompt(prompt_path: Path | None, prompt_text: str) -> str:
    if prompt_text.strip():
        return prompt_text
    if prompt_path is None:
        raise ValueError("No prompt provided. Set input.prompt_text or input.prompt_path.")
    return prompt_path.read_text(encoding="utf-8")


def enforce_numeric_output_prompt(prompt_text: str, prompt_cfg: dict[str, Any]) -> str:
    if not bool(prompt_cfg.get("append_numeric_instruction", True)):
        return prompt_text.rstrip()

    instruction = str(
        prompt_cfg.get(
            "numeric_instruction",
            (
                "Return only the final normalized gaze point as two numbers in one line.\n"
                "Format: x y\n"
                "No explanation. No labels. No extra text."
            ),
        )
    ).strip()
    if not instruction:
        return prompt_text.rstrip()
    return prompt_text.rstrip() + "\n\n" + instruction


def parse_head_bbox_from_prompt(prompt_text: str) -> tuple[float, float, float, float] | None:
    m = HEAD_BBOX_RE.search(prompt_text)
    if not m:
        return None
    try:
        vals = [float(x.strip()) for x in m.group(1).split(",")]
    except ValueError:
        return None
    if len(vals) != 4:
        return None

    x1, y1, x2, y2 = vals
    x1, x2 = sorted((max(0.0, min(1.0, x1)), max(0.0, min(1.0, x2))))
    y1, y2 = sorted((max(0.0, min(1.0, y1)), max(0.0, min(1.0, y2))))
    if x2 <= x1 or y2 <= y1:
        return None
    return x1, y1, x2, y2


def crop_head_from_bbox(
    image: Image.Image,
    bbox_norm: tuple[float, float, float, float],
) -> Image.Image | None:
    w, h = image.size
    if w <= 0 or h <= 0:
        return None

    x1n, y1n, x2n, y2n = bbox_norm
    x1 = int(round(x1n * w))
    y1 = int(round(y1n * h))
    x2 = int(round(x2n * w))
    y2 = int(round(y2n * h))

    x1 = max(0, min(x1, w))
    y1 = max(0, min(y1, h))
    x2 = max(0, min(x2, w))
    y2 = max(0, min(y2, h))

    if x2 <= x1:
        if x1 >= w:
            x1 = w - 1
        x2 = min(w, x1 + 1)
    if y2 <= y1:
        if y1 >= h:
            y1 = h - 1
        y2 = min(h, y1 + 1)
    if x2 <= x1 or y2 <= y1:
        return None

    return image.crop((x1, y1, x2, y2))


def is_normalized_point(point: tuple[float, float]) -> bool:
    return 0.0 <= point[0] <= 1.0 and 0.0 <= point[1] <= 1.0


def parse_prediction_point_from_text(text: str) -> tuple[tuple[float, float] | None, str]:
    lines = [line.strip() for line in text.splitlines() if line.strip()]

    strict_candidates: list[tuple[float, float]] = []
    for line in lines:
        m = XY_LINE_RE.match(line)
        if m:
            strict_candidates.append((float(m.group(1)), float(m.group(2))))
    if strict_candidates:
        for pt in reversed(strict_candidates):
            if is_normalized_point(pt):
                return pt, "strict_line"
        return None, "strict_line_out_of_range"

    labeled_matches = list(XY_LABELED_RE.finditer(text))
    if labeled_matches:
        for m in reversed(labeled_matches):
            pt = (float(m.group(1)), float(m.group(2)))
            if is_normalized_point(pt):
                return pt, "labeled_pair"
        return None, "labeled_pair_out_of_range"

    paren_matches = list(XY_PAREN_RE.finditer(text))
    if paren_matches:
        for m in reversed(paren_matches):
            pt = (float(m.group(1)), float(m.group(2)))
            if is_normalized_point(pt):
                return pt, "paren_pair"
        return None, "paren_pair_out_of_range"

    nums = [float(x) for x in FLOAT_RE.findall(text)]
    if len(nums) < 2:
        return None, "no_numbers"

    candidates = [(nums[i], nums[i + 1]) for i in range(len(nums) - 1)]
    in_range_candidates = [pt for pt in candidates if is_normalized_point(pt)]
    if in_range_candidates:
        return in_range_candidates[-1], "fallback_last_in_range_pair"
    return None, "fallback_no_in_range_pair"


def finalize_prediction(
    raw_prediction: str,
    output_cfg: dict[str, Any],
) -> tuple[str, tuple[float, float] | None, str]:
    parsed_point, parse_source = parse_prediction_point_from_text(raw_prediction)
    prefer_parsed = bool(output_cfg.get("prefer_parsed_output", True))
    decimals = int(output_cfg.get("parsed_output_decimals", 6))

    if prefer_parsed and parsed_point is not None:
        x, y = parsed_point
        return f"{x:.{decimals}f} {y:.{decimals}f}", parsed_point, parse_source
    return raw_prediction.strip(), parsed_point, parse_source


def validate_input_config(input_cfg: dict[str, Any], config_dir: Path) -> None:
    mode = str(input_cfg.get("mode", "")).strip()
    if mode == "batch":
        image_dir = resolve_path(config_dir, str(input_cfg.get("image_dir", "")))
        prompt_dir = resolve_path(config_dir, str(input_cfg.get("prompt_dir", "")))
        if not image_dir.exists():
            raise ValueError(f"batch mode: image_dir not found: {image_dir}")
        if not prompt_dir.exists():
            raise ValueError(f"batch mode: prompt_dir not found: {prompt_dir}")
        return

    if mode == "single":
        image_path_raw = str(input_cfg.get("image_path", "")).strip()
        prompt_path_raw = str(input_cfg.get("prompt_path", "")).strip()
        prompt_text = str(input_cfg.get("prompt_text", "")).strip()

        if not image_path_raw:
            raise ValueError("single mode: input.image_path is required.")
        image_path = resolve_path(config_dir, image_path_raw)
        if not image_path.exists():
            raise ValueError(f"single mode: image_path not found: {image_path}")

        if not prompt_text and not prompt_path_raw:
            raise ValueError("single mode: set either input.prompt_text or input.prompt_path.")
        if prompt_path_raw:
            prompt_path = resolve_path(config_dir, prompt_path_raw)
            if not prompt_path.exists():
                raise ValueError(f"single mode: prompt_path not found: {prompt_path}")
        return

    raise ValueError(f"Unsupported input.mode: {mode}")


def generate_one(
    processor: Any,
    model: Any,
    device: str,
    image_path: Path,
    prompt_text: str,
    input_cfg: dict[str, Any],
    prompt_cfg: dict[str, Any],
    generation_kwargs: dict[str, Any],
) -> str:
    use_image = bool(input_cfg.get("use_image", True))
    use_head_crop = use_image and bool(input_cfg.get("use_head_crop", True))
    strict_head_bbox = bool(input_cfg.get("strict_head_bbox", True))

    image = None
    head_crop_image = None
    if use_image:
        with Image.open(image_path) as img:
            image = img.convert("RGB")
        if use_head_crop:
            bbox_norm = parse_head_bbox_from_prompt(prompt_text)
            if bbox_norm is None:
                if strict_head_bbox:
                    raise RuntimeError(
                        f"Head bbox not found in prompt for sample: {image_path.stem}"
                    )
            else:
                head_crop_image = crop_head_from_bbox(image, bbox_norm)
                if head_crop_image is None and strict_head_bbox:
                    raise RuntimeError(
                        f"Failed to crop head image from bbox for sample: {image_path.stem}"
                    )

    prompt_text = enforce_numeric_output_prompt(prompt_text, prompt_cfg)
    head_crop_context_text = str(
        prompt_cfg.get(
            "head_crop_context_text",
            "The first image is the full scene and the second image is the cropped head region.",
        )
    ).strip()

    use_chat_template = bool(prompt_cfg.get("use_chat_template", True))
    add_generation_prompt = bool(prompt_cfg.get("add_generation_prompt", True))
    if use_chat_template and hasattr(processor, "apply_chat_template"):
        content: list[dict[str, Any]] = []
        if use_image:
            content.append({"type": "image", "image": image})
            if head_crop_image is not None:
                content.append({"type": "image", "image": head_crop_image})
                if head_crop_context_text:
                    content.append({"type": "text", "text": head_crop_context_text})
        content.append({"type": "text", "text": prompt_text})

        messages = [{"role": "user", "content": content}]
        prompt = processor.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=add_generation_prompt,
        )
    else:
        prompt = prompt_text

    processor_inputs: dict[str, Any] = {"text": [prompt], "return_tensors": "pt"}
    if use_image:
        image_inputs = [image]
        if head_crop_image is not None:
            image_inputs.append(head_crop_image)
        processor_inputs["images"] = image_inputs

    try:
        inputs = processor(**processor_inputs)
    except Exception as e:
        if use_image:
            raise RuntimeError(
                "Failed to build image+text inputs. "
                "This model/processor may not support vision inputs. "
                "Try a VLM repo_id (for example, a *-VL-* Instruct model), or set input.use_image=false."
            ) from e
        raise

    inputs = {k: v.to(device) if hasattr(v, "to") else v for k, v in inputs.items()}
    generated = model.generate(**inputs, **generation_kwargs)

    if "input_ids" in inputs and generated.shape[1] >= inputs["input_ids"].shape[1]:
        new_tokens = generated[:, inputs["input_ids"].shape[1] :]
    else:
        new_tokens = generated

    text = processor.batch_decode(
        new_tokens,
        skip_special_tokens=True,
        clean_up_tokenization_spaces=False,
    )[0]
    return text.strip()


def maybe_run_evaluation(
    cfg: dict[str, Any],
    config_dir: Path,
    pred_dir: Path,
    model_name: str,
) -> None:
    eval_cfg = dict(cfg.get("evaluation", {}) or {})
    if not bool(eval_cfg.get("run_after_generation", False)):
        return

    from evaluate import evaluate as evaluate_predictions

    gt_dir = resolve_path(config_dir, str(eval_cfg.get("gt_dir", "100_imgs_target")))
    if not gt_dir.exists():
        raise FileNotFoundError(f"GT directory not found for evaluation: {gt_dir}")

    result = evaluate_predictions(pred_dir=pred_dir, gt_dir=gt_dir)
    result["pred_dir"] = str(pred_dir)
    result["model_name"] = model_name

    print("[EVAL] done")
    print(f"[EVAL] num_obs={result['num_obs']}")
    print(f"[EVAL] avg_l2={result['avg_l2']:.6f}")
    print(f"[EVAL] min_l2={result['min_l2']:.6f}")
    print(f"[EVAL] dist_to_avg={result['dist_to_avg']:.6f}")

    save_json_raw = str(eval_cfg.get("save_json", "")).strip()
    if save_json_raw:
        save_json_path = resolve_path(config_dir, save_json_raw)
        save_json_path.parent.mkdir(parents=True, exist_ok=True)
        save_json_path.write_text(
            json.dumps(result, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        print(f"[EVAL] saved_json={save_json_path}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    args = parser.parse_args()

    config_path = args.config.resolve()
    config_dir = config_path.parent
    cfg = load_yaml(config_path)

    input_cfg = dict(cfg.get("input", {}) or {})
    prompt_cfg = dict(cfg.get("prompt", {}) or {})
    output_cfg = dict(cfg.get("output", {}) or {})
    generation_cfg = dict(cfg.get("generation", {}) or {})

    validate_input_config(input_cfg, config_dir)

    model_dir, model_name, device, torch_dtype, trust_remote_code = prepare_model(
        dict(cfg.get("model", {}) or {}),
        config_dir,
    )

    output_root = resolve_path(config_dir, str(output_cfg.get("root_dir", "100_imgs_output")))
    output_dir = output_root / model_name
    overwrite = bool(output_cfg.get("overwrite", True))
    ensure_dir(output_dir)

    print(f"[INFO] model_dir={model_dir}")
    print(f"[INFO] model_name={model_name}")
    print(f"[INFO] device={device}, torch_dtype={torch_dtype}")
    print(f"[INFO] output_dir={output_dir}")

    model_cfg = dict(cfg.get("model", {}) or {})
    processor_kwargs = build_processor_kwargs(model_cfg, trust_remote_code)
    model_kwargs = build_model_kwargs(model_cfg, trust_remote_code, torch_dtype)
    generation_kwargs = build_generation_kwargs(generation_cfg)

    processor = AutoProcessor.from_pretrained(str(model_dir), **processor_kwargs)
    try:
        model = AutoModelForImageTextToText.from_pretrained(str(model_dir), **model_kwargs)
    except Exception as e:
        raise RuntimeError(
            "Failed to load model with AutoModelForImageTextToText. "
            "For image+prompt inference, use a VLM checkpoint (for example, a *-VL-* model)."
        ) from e

    if "device_map" not in model_kwargs:
        model.to(device)
    model.eval()

    mode = str(input_cfg["mode"])
    if mode == "batch":
        samples = list_batch_samples(input_cfg, config_dir)
    elif mode == "single":
        samples = list_single_sample(input_cfg, config_dir)
    else:
        raise ValueError(f"Unsupported input.mode: {mode}")

    if not samples:
        print("[INFO] no samples to process.")
        return

    total = len(samples)
    print(f"[INFO] total_samples={total}")

    jsonl_path = output_dir / "results.jsonl"
    jsonl_mode = "w" if overwrite else "a"
    with jsonl_path.open(jsonl_mode, encoding="utf-8") as jf:
        for idx, (image_path, prompt_path, stem) in enumerate(samples, start=1):
            print(f"[RUN] ({idx}/{total}) {stem}")
            out_path = output_dir / f"{stem}.txt"
            if out_path.exists() and not overwrite:
                print(f"[SKIP] exists: {out_path}")
                continue

            prompt_text = load_prompt(prompt_path, str(input_cfg.get("prompt_text", "")))
            raw_pred = generate_one(
                processor=processor,
                model=model,
                device=device,
                image_path=image_path,
                prompt_text=prompt_text,
                input_cfg=input_cfg,
                prompt_cfg=prompt_cfg,
                generation_kwargs=generation_kwargs,
            )
            prediction_text, parsed_point, parse_source = finalize_prediction(
                raw_prediction=raw_pred,
                output_cfg=output_cfg,
            )

            out_path.write_text(prediction_text + "\n", encoding="utf-8")
            row = {
                "id": stem,
                "image_path": str(image_path),
                "prompt_path": str(prompt_path) if prompt_path else None,
                "used_head_crop": bool(input_cfg.get("use_head_crop", True)),
                "prediction_saved": prediction_text,
                "raw_prediction": raw_pred,
                "parsed_point": list(parsed_point) if parsed_point is not None else None,
                "parse_source": parse_source,
            }
            jf.write(json.dumps(row, ensure_ascii=False) + "\n")
            jf.flush()
            print(f"[DONE] ({idx}/{total}) {stem}")

    print(f"[INFO] saved results to: {output_dir}")
    maybe_run_evaluation(cfg=cfg, config_dir=config_dir, pred_dir=output_dir, model_name=model_name)


if __name__ == "__main__":
    main()
