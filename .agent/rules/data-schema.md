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

Implemented in `model/utils/gaze_tokens.py`.

Schema markers:

- `<|gaze_reasoning|>`
- `<|gaze_point|>`
- `<|gaze_object|>`

Coordinate tokens:

- Built by `build_gaze_special_tokens(num_classes, coord_bins)`.
- Current `coord_bins=128`.
- Valid active loc tokens are `<loc_000>` through `<loc_127>`.
- Coordinates are quantized by `quantize_coord(coord, bins)`.

Object tokens:

- `<obj_000>` through `<obj_{num_classes-1}>`, padded to at least 3 digits.
- Unknown object fallback: `<obj_unknown>`.

## Reasoning Text

Reasoning text is indexed by `build_reasoning_index()` and loaded lazily in `GazeDataset`.

Normalization:

- collapses whitespace,
- strips schema markers from reasoning content,
- truncates to `max_reasoning_words` and `max_reasoning_chars`,
- appends a trailing period if needed.

Current limits:

- `max_reasoning_words=60`
- `max_reasoning_chars=500`

Reasoning is capped in `GazeDataset` when reading train reasoning files. `build_structured_target_text()` only sanitizes formatting and does not apply a second length cap.

## Datasets

- `GazeDataset`: active train/val dataset.
- `GazeTestDataset`: grouped test dataset with multiple GT points.
- `MultiViewGazeDataset`: legacy code still present, not used by the active trainer path.

Active train construction:

- always `GazeDataset`,
- `force_reasoning_format=True`,
- `target_order="reasoning_point_object"`,
- `apply_augmentation=True`.

Active val/test construction:

- same schema,
- no image augmentation.

## Collator

`model/utils/processor_collate.py`:

- Builds chat-template train/infer inputs.
- Uses `truncation=False` to avoid breaking Qwen VLM image token alignment.
- Builds `loss_mask_reasoning`, `loss_mask_point`, `loss_mask_object`, and `loss_mask_format`.
- Still emits default `view_type`, but active training no longer logs view-based metrics.
