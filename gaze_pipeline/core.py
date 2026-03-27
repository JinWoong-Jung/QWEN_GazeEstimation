from __future__ import annotations

import fnmatch
import re
from pathlib import Path
from typing import Any

import requests
import torch
import yaml
from PIL import Image
from huggingface_hub import HfApi, hf_hub_url, snapshot_download

ROOT_DIR = Path(__file__).resolve().parents[1]
DEFAULT_CONFIG = ROOT_DIR / "config.yaml"
MODEL_STORAGE_ROOT = ROOT_DIR / "model"
CHECKPOINT_ROOT = ROOT_DIR / "checkpoints"

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

    os_env = {
        "HF_HOME": str(cache_dir),
        "HUGGINGFACE_HUB_CACHE": str(cache_dir),
        "TRANSFORMERS_CACHE": str(cache_dir),
    }
    for k, v in os_env.items():
        import os

        os.environ[k] = v
    import os

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
    if "torch_dtype" in kwargs and "dtype" not in kwargs:
        kwargs["dtype"] = kwargs.pop("torch_dtype")
    if torch_dtype != "auto":
        kwargs.setdefault("dtype", torch_dtype)
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


def quantize_point(x: float, y: float, bins: int = 1000) -> tuple[int, int]:
    if bins <= 1:
        raise ValueError("coord bins must be > 1")
    x = max(0.0, min(1.0, float(x)))
    y = max(0.0, min(1.0, float(y)))
    scale = bins - 1
    ix = int(round(x * scale))
    iy = int(round(y * scale))
    ix = max(0, min(ix, scale))
    iy = max(0, min(iy, scale))
    return ix, iy


def dequantize_point(ix: int, iy: int, bins: int = 1000) -> tuple[float, float]:
    if bins <= 1:
        raise ValueError("coord bins must be > 1")
    scale = bins - 1
    ix = max(0, min(int(ix), scale))
    iy = max(0, min(int(iy), scale))
    return ix / scale, iy / scale


def render_target_text(
    x: float,
    y: float,
    target_format: str = "xy_float",
    coord_bins: int = 1000,
    target_decimals: int = 6,
) -> str:
    fmt = str(target_format).strip().lower()
    if fmt == "xy_bins":
        ix, iy = quantize_point(x, y, bins=coord_bins)
        return f"{ix} {iy}"
    return f"{x:.{target_decimals}f} {y:.{target_decimals}f}"


def _is_integer_like(v: float, eps: float = 1e-6) -> bool:
    return abs(v - round(v)) <= eps


def _candidate_to_point(
    pt: tuple[float, float],
    coord_bins: int | None,
) -> tuple[tuple[float, float] | None, str]:
    if is_normalized_point(pt):
        return pt, "normalized"
    if coord_bins is None or coord_bins <= 1:
        return None, "not_normalized"

    x, y = pt
    if _is_integer_like(x) and _is_integer_like(y):
        ix = int(round(x))
        iy = int(round(y))
        if 0 <= ix < coord_bins and 0 <= iy < coord_bins:
            return dequantize_point(ix, iy, bins=coord_bins), "bins"
    return None, "not_bins"


def parse_prediction_point_from_text(
    text: str,
    coord_bins: int = 1000,
) -> tuple[tuple[float, float] | None, str]:
    lines = [line.strip() for line in text.splitlines() if line.strip()]

    strict_candidates: list[tuple[float, float]] = []
    for line in lines:
        m = XY_LINE_RE.match(line)
        if m:
            strict_candidates.append((float(m.group(1)), float(m.group(2))))
    if strict_candidates:
        for pt in reversed(strict_candidates):
            parsed, kind = _candidate_to_point(pt, coord_bins=coord_bins)
            if parsed is not None:
                if kind == "bins":
                    return parsed, "strict_line_bins"
                return parsed, "strict_line"
        return None, "strict_line_not_parseable"

    labeled_matches = list(XY_LABELED_RE.finditer(text))
    if labeled_matches:
        for m in reversed(labeled_matches):
            pt = (float(m.group(1)), float(m.group(2)))
            parsed, kind = _candidate_to_point(pt, coord_bins=coord_bins)
            if parsed is not None:
                if kind == "bins":
                    return parsed, "labeled_pair_bins"
                return parsed, "labeled_pair"
        return None, "labeled_pair_not_parseable"

    paren_matches = list(XY_PAREN_RE.finditer(text))
    if paren_matches:
        for m in reversed(paren_matches):
            pt = (float(m.group(1)), float(m.group(2)))
            parsed, kind = _candidate_to_point(pt, coord_bins=coord_bins)
            if parsed is not None:
                if kind == "bins":
                    return parsed, "paren_pair_bins"
                return parsed, "paren_pair"
        return None, "paren_pair_not_parseable"

    nums = [float(x) for x in FLOAT_RE.findall(text)]
    if len(nums) < 2:
        return None, "no_numbers"

    candidates = [(nums[i], nums[i + 1]) for i in range(len(nums) - 1)]
    for pt in reversed(candidates):
        parsed, kind = _candidate_to_point(pt, coord_bins=coord_bins)
        if parsed is not None:
            if kind == "bins":
                return parsed, "fallback_last_bins_pair"
            return parsed, "fallback_last_in_range_pair"
    return None, "fallback_no_parseable_pair"


def finalize_prediction(
    raw_prediction: str,
    output_cfg: dict[str, Any],
) -> tuple[str, tuple[float, float] | None, str]:
    coord_bins = int(output_cfg.get("coord_bins", 1000))
    parsed_point, parse_source = parse_prediction_point_from_text(
        raw_prediction,
        coord_bins=coord_bins,
    )
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
    bbox_norm: tuple[float, float, float, float] | None = None,
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
            bbox_to_use = bbox_norm
            if bbox_to_use is None:
                bbox_to_use = parse_head_bbox_from_prompt(prompt_text)
            if bbox_to_use is None:
                if strict_head_bbox:
                    raise RuntimeError(
                        f"Head bbox not found in prompt for sample: {image_path.stem}"
                    )
            else:
                head_crop_image = crop_head_from_bbox(image, bbox_to_use)
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


def sanitize_bbox_pixels(
    bbox: tuple[float, float, float, float],
    width: int,
    height: int,
) -> tuple[float, float, float, float] | None:
    xmin, ymin, xmax, ymax = bbox
    x1, x2 = sorted((xmin, xmax))
    y1, y2 = sorted((ymin, ymax))

    x1 = max(0.0, min(x1, float(width)))
    y1 = max(0.0, min(y1, float(height)))
    x2 = max(0.0, min(x2, float(width)))
    y2 = max(0.0, min(y2, float(height)))

    if x2 <= x1 or y2 <= y1:
        return None
    return x1, y1, x2, y2


def normalize_bbox_pixels(
    bbox: tuple[float, float, float, float],
    width: int,
    height: int,
) -> tuple[float, float, float, float]:
    xmin, ymin, xmax, ymax = bbox
    return xmin / width, ymin / height, xmax / width, ymax / height


def build_gazefollow_prompt(
    bbox_norm: tuple[float, float, float, float],
    prompt_cfg: dict[str, Any],
) -> str:
    xmin, ymin, xmax, ymax = bbox_norm
    template = str(prompt_cfg.get("train_prompt_template", "")).strip()
    if template:
        return template.format(xmin=xmin, ymin=ymin, xmax=xmax, ymax=ymax)
    return (
        "Given the highlighted subject, estimate where the subject is looking(normalized gaze point).\n"
        f"Head bbox [xmin,ymin,xmax,ymax] (normalized): [{xmin:.6f}, {ymin:.6f}, {xmax:.6f}, {ymax:.6f}]\n"
        "Important rules:\n"
        "1) Predict where the person is looking AT in the scene (target object/location),\n"
        "NOT the eye position, face center, or head center.\n"
        "2) Coordinates must be in FULL SCENE coordinates, not head-crop coordinates.\n"
        "3) Prefer a point on the likely attended object along the gaze direction.\n"
        "4) Avoid points inside the head bbox unless no other plausible target is visible."
    )


def parse_lora_target_modules(raw: Any) -> list[str]:
    if isinstance(raw, str):
        modules = [x.strip() for x in raw.split(",") if x.strip()]
        if modules:
            return modules
    if isinstance(raw, list):
        modules = [str(x).strip() for x in raw if str(x).strip()]
        if modules:
            return modules
    return [
        "q_proj",
        "k_proj",
        "v_proj",
        "o_proj",
        "gate_proj",
        "up_proj",
        "down_proj",
    ]
