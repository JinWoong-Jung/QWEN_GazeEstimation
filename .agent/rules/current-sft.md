---
description: Current active SFT behavior and config decisions.
alwaysApply: true
---

# Current SFT Behavior

The active Stage1 SFT path trains and evaluates the direct point-object schema:

```text
<|point_start|><loc_x><loc_y><|point_end|><|object_start|><obj_k><|object_end|>
```

## Active Decisions

- `sft.yaml` sets `train_stage: "sft"`.
- `output_format` is configurable in the active SFT trainer path. Supported values are `point_object` and `text_point_object`.
- Train uses `GazeDataset`; val uses `GazeDataset`; test uses grouped `GazeTestDataset`.
- Val/test do not run teacher-forced validation loss in the normal SFT loop; they use generation metrics.
- Current SFT config uses `constrained_decoding: true` and `constrained_loc_decoding: "argmax"`.
- `generation_max_new_tokens=8` because the direct schema is short.
- Visual prompting is enabled; the dataset draws the head box on the scene image when `visual_prompting: true`.
- `image_resize_mode: "fixed"` resizes PIL images to `scene_h x scene_w` before the Qwen processor. `"native"` skips this repo-level resize but the Qwen processor can still resize according to its pixel limits.
- DataLoader workers are persistent when `num_workers > 0`, so per-worker image caches persist across epochs.

## Active `sft.yaml` Values

- Base model: `model/Qwen3-VL-2B-Instruct`
- Image resize: fixed `512 x 512`
- Image cache size: `1000` per worker in the current config
- Batch size: `32`
- Grad accumulation: `8`
- Epochs: `20`
- Learning rate: `2e-5`
- Output format: `point_object`
- Coord bins: `128`
- LoRA rank/alpha: `32` / `64`
- LoRA target modules: `q_proj`, `k_proj`, `v_proj`, `o_proj`
- Train augmentation mode: `no_aug`
- Checkpoint monitor: `val_dist` with mode `min`
- `save_last_every_n_epochs: 3`

## Prompt

The active prompt asks the model to return only the configured point/object schema. Keep prompt/schema changes synchronized with `special_tokens.py`, `gaze_tokens.py`, collator masks, eval parsing, constrained decoding, and tests.

## Legacy Formats

Do not describe the active output as:

```text
Point: 0.4230 0.7112
Object: television
```

or:

```text
<|gaze_reasoning|><reasoning text><|gaze_point|><loc_x><loc_y><|gaze_object|><obj_k>
```

Those are legacy or inactive formats for the current SFT path.
