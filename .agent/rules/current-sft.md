---
description: Current active SFT behavior and config decisions.
alwaysApply: true
---

# Current SFT Behavior

The active SFT path always trains and evaluates the reasoning-point-object schema:

```text
<|gaze_reasoning|><reasoning text><|gaze_point|><loc_x><loc_y><|gaze_object|><obj_k>
```

## Active Decisions

- `target_order` is hardcoded as `reasoning_point_object` in the active trainer path.
- `force_reasoning_format=True` for train, val, test, and `test_only`.
- Train uses reasoning annotations when files exist.
- Val/test do not have GT reasoning annotations, but still generate the same schema.
- Val uses pure generation metrics only; it does not run teacher-forced loss.
- The old direct/reasoning multiview SFT sampler is removed from the active trainer path.
- `train/view_reasoning_frac` was removed because it only described the old multiview view mix.
- `constrained_decoding=false` because free-form reasoning is generated before point/object slots.
- `generation_stop_at_object_end=false`; allow natural EOS after the full schema.
- `generation_max_new_tokens=80`, based on sampled reasoning length plus schema overhead.

## Active `sft.yaml` Values

- Base model: `model/Qwen3-VL-2B-Instruct`
- Image resize: fixed `512 x 512`
- Batch size: `32`
- Grad accumulation: `4`
- Epochs: `20`
- Coord bins: `128`
- Reasoning cap: `max_reasoning_words=60`, `max_reasoning_chars=500`
- LoRA target modules: `q_proj`, `k_proj`, `v_proj`, `o_proj`
- Train reasoning dir: `/home/elicer/QWEN_GazeEstimation/data/bucket_data/data/gazefollow_reason/output/train`

## Prompt

The active prompt asks for one or two short reasoning sentences, then exactly one x-bin, one y-bin, and one object token. It explicitly mentions the loc/object token ranges. Keep prompt/schema changes synchronized with `gaze_tokens.py`, collator masks, eval parsing, and tests.

## Legacy Formats

Do not describe the active output as:

```text
Point: 0.4230 0.7112
Object: television
```

That is legacy only.
