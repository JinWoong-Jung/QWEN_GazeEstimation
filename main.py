#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Any

import torch
import yaml
from PIL import Image
from huggingface_hub import snapshot_download
from transformers import AutoModelForImageTextToText, AutoProcessor

ROOT_DIR = Path(__file__).resolve().parent
DEFAULT_CONFIG = ROOT_DIR / "config.yaml"
SUPPORTED_MODEL = "Qwen/Qwen3.5-9B"


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


def prepare_model(model_cfg: dict[str, Any], config_dir: Path) -> tuple[Path, str, torch.dtype | str, bool]:
    repo_id = model_cfg["repo_id"]
    if repo_id != SUPPORTED_MODEL:
        raise ValueError(f"Only {SUPPORTED_MODEL} is supported for now. Got: {repo_id}")

    model_name = model_name_from_repo(repo_id)
    local_dir = ROOT_DIR / "model" / model_name
    cache_dir = ROOT_DIR / "model" / "cache"
    ensure_dir(local_dir)
    ensure_dir(cache_dir)

    # Keep all HF/transformers caches under model/ as requested.
    os.environ["HF_HOME"] = str(cache_dir)
    os.environ["HUGGINGFACE_HUB_CACHE"] = str(cache_dir)
    os.environ["TRANSFORMERS_CACHE"] = str(cache_dir)

    download_if_missing = bool(model_cfg.get("download_if_missing", True))
    has_model_files = any(local_dir.glob("*.json")) and any(local_dir.glob("*.safetensors")) or any(local_dir.glob("pytorch_model*.bin"))
    if download_if_missing and not has_model_files:
        snapshot_download(
            repo_id=repo_id,
            local_dir=str(local_dir),
            local_dir_use_symlinks=False,
            ignore_patterns=["*.msgpack", "*.h5", "*.ot", "*.tflite"],
        )

    device = select_device(str(model_cfg.get("device", "auto")))
    torch_dtype = select_torch_dtype(str(model_cfg.get("torch_dtype", "auto")))
    trust_remote_code = bool(model_cfg.get("trust_remote_code", True))
    return local_dir, device, torch_dtype, trust_remote_code


def list_batch_samples(input_cfg: dict[str, Any], config_dir: Path) -> list[tuple[Path, Path, str]]:
    image_dir = resolve_path(config_dir, input_cfg["image_dir"])
    prompt_dir = resolve_path(config_dir, input_cfg["prompt_dir"])
    image_exts = {ext.lower() for ext in input_cfg.get("image_extensions", [".jpg", ".jpeg", ".png"])}

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


def list_single_sample(input_cfg: dict[str, Any], config_dir: Path) -> list[tuple[Path, Path | None, str]]:
    image_path = resolve_path(config_dir, input_cfg["image_path"])
    prompt_path_raw = str(input_cfg.get("prompt_path", "")).strip()
    prompt_path = resolve_path(config_dir, prompt_path_raw) if prompt_path_raw else None
    return [(image_path, prompt_path, image_path.stem)]


def load_prompt(prompt_path: Path | None, prompt_text: str) -> str:
    if prompt_text.strip():
        return prompt_text
    if prompt_path is None:
        raise ValueError("No prompt provided. Set input.prompt_text or input.prompt_path.")
    return prompt_path.read_text(encoding="utf-8")


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
            raise ValueError(
                "single mode: set either input.prompt_text or input.prompt_path."
            )
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
    gen_cfg: dict[str, Any],
) -> str:
    with Image.open(image_path) as img:
        image = img.convert("RGB")

    messages = [
        {
            "role": "user",
            "content": [
                {"type": "image", "image": image},
                {"type": "text", "text": prompt_text},
            ],
        }
    ]

    prompt = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    inputs = processor(text=[prompt], images=[image], return_tensors="pt")
    inputs = {k: v.to(device) for k, v in inputs.items()}

    generated = model.generate(
        **inputs,
        max_new_tokens=int(gen_cfg.get("max_new_tokens", 128)),
        do_sample=bool(gen_cfg.get("do_sample", False)),
        temperature=float(gen_cfg.get("temperature", 0.0)),
        top_p=float(gen_cfg.get("top_p", 1.0)),
    )
    new_tokens = generated[:, inputs["input_ids"].shape[1]:]
    text = processor.batch_decode(new_tokens, skip_special_tokens=True, clean_up_tokenization_spaces=False)[0]
    return text.strip()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    args = parser.parse_args()

    config_path = args.config.resolve()
    config_dir = config_path.parent
    cfg = load_yaml(config_path)
    validate_input_config(cfg["input"], config_dir)

    model_dir, device, torch_dtype, trust_remote_code = prepare_model(cfg["model"], config_dir)
    model_name = model_name_from_repo(cfg["model"]["repo_id"])
    output_dir = ROOT_DIR / "100_imgs_output" / model_name
    overwrite = bool(cfg["output"].get("overwrite", False))
    ensure_dir(output_dir)

    print(f"[INFO] model_dir={model_dir}")
    print(f"[INFO] device={device}, torch_dtype={torch_dtype}")

    processor = AutoProcessor.from_pretrained(str(model_dir), trust_remote_code=trust_remote_code)
    model = AutoModelForImageTextToText.from_pretrained(
        str(model_dir),
        trust_remote_code=trust_remote_code,
        torch_dtype=torch_dtype,
    )
    model.to(device)
    model.eval()

    input_cfg = cfg["input"]
    mode = input_cfg["mode"]
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
    with jsonl_path.open("a", encoding="utf-8") as jf:
        for idx, (image_path, prompt_path, stem) in enumerate(samples, start=1):
            print(f"[RUN] ({idx}/{total}) {stem}")
            out_path = output_dir / f"{stem}.txt"
            if out_path.exists() and not overwrite:
                print(f"[SKIP] exists: {out_path}")
                continue

            prompt_text = load_prompt(prompt_path, str(input_cfg.get("prompt_text", "")))
            pred = generate_one(
                processor=processor,
                model=model,
                device=device,
                image_path=image_path,
                prompt_text=prompt_text,
                gen_cfg=cfg["generation"],
            )

            # Save each sample result immediately.
            out_path.write_text(pred + "\n", encoding="utf-8")
            row = {
                "id": stem,
                "image_path": str(image_path),
                "prompt_path": str(prompt_path) if prompt_path else None,
                "prediction": pred,
            }
            jf.write(json.dumps(row, ensure_ascii=False) + "\n")
            jf.flush()
            print(f"[DONE] ({idx}/{total}) {stem}")

    print(f"[INFO] saved results to: {output_dir}")


if __name__ == "__main__":
    main()
