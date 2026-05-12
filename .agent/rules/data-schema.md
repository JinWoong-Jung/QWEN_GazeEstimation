---
description: Dataset, prompt, reasoning, and special-token schema details.
---

# Data And Schema

## Data Flow

```text
annotation row
  -> Record
  -> GazeDataset / GazeTestDataset
  -> PIL scene image + prompt + target_text
  -> QwenTrainCollator / QwenTestCollator
  -> processor chat template + image tokens
  -> joint_inputs, labels, structured loss masks
  -> QwenTextGenerationModel
  -> logits
  -> compute_answer_loss()
```

At inference/eval:

```text
QwenTextGenerationModel.generate()
  -> decode_generated()
  -> parse_structured_output_text()
  -> point/object/format metrics
```

## Special Tokens

Implemented in `model/utils/special_tokens.py` and `model/utils/gaze_tokens.py`.

Active schema markers:

- `<|point_start|>`
- `<|point_end|>`
- `<|object_start|>`
- `<|object_end|>`

Coordinate tokens:

- Built by `build_gaze_special_tokens(num_classes, coord_bins)`.
- Current `coord_bins=128`.
- Valid active loc tokens are `<loc_000>` through `<loc_127>`.
- Coordinates are quantized by `quantize_coord(coord, bins)`.

Object tokens:

- `<obj_000>` through `<obj_{num_classes-1}>`, padded to at least 3 digits.
- Unknown object fallback: `<obj_unknown>`.

## Target Text

The active target text is direct point-object:

```text
<|point_start|><loc_x><loc_y><|point_end|><|object_start|><obj_k><|object_end|>
```

`output_format` currently supports:

- `point_object`: `<|point_start|><loc_x><loc_y><|point_end|><|object_start|><obj_k><|object_end|>`
- `text_point_object`: `Point:<loc_x><loc_y>\nObject:<obj_k>`

The old reasoning-first target order is not active.

Qwen chat templating wraps the assistant answer with chat markers. The collator searches for the raw target span, supervises point/object/format tokens, and additionally marks the trailing `<|im_end|>` token as format loss so the model learns to stop after the object span.

## Reasoning Text

Reasoning text is optional and currently relevant to SDFT teacher prompts, not Stage1 SFT targets.

`build_reasoning_index()` indexes reasoning files. `GazeDataset` can lazily load `reasoning_text` and `object_text`; the SDFT collator/trainer uses those fields to build teacher-side prompts when distillation is enabled.

## Datasets

- `GazeDataset`: active train/val dataset.
- `GazeTestDataset`: grouped test dataset with multiple GT points.
- `MultiViewGazeDataset`: legacy code still present, not used by the active trainer path.

Active train construction:

- always `GazeDataset`,
- `target_order="point_object"`,
- `apply_augmentation=True`.

Active val/test construction:

- same schema,
- no image augmentation.

## Collator

`model/utils/processor_collate.py`:

- Builds chat-template train/infer inputs.
- Uses `truncation=False` to avoid breaking Qwen VLM image token alignment.
- Builds `loss_mask_point`, `loss_mask_object`, and `loss_mask_format`.
- Emits rollout SDFT helper fields such as `scene_images`, `text_input`, `reasoning_texts`, `object_texts`, and `teacher_base_inputs`.
- When `distil_kl_weight > 0`, it can also build teacher inputs for teacher-forcing SDFT.
