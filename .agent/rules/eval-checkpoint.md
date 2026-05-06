---
description: Evaluation, generation, parsing, and checkpoint behavior.
---

# Evaluation And Checkpoints

## Evaluation

`model/utils/eval_utils.py` handles generation and metrics.

Current eval/test generation:

- Generates free-form reasoning followed by point/object special tokens.
- Validation is pure generation-based; the active path does not run teacher-forced validation loss.
- Keep `preview_val_samples=0` when measuring validation speed; previews are a separate generation pass.
- `constrained_decoding=false`.
- `generation_max_new_tokens=80`.
- `generation_stop_at_object_end=false`.

Metrics are based on parsed structured output:

- validation point distance as `Dist`,
- test point distance as `Avg L2` and `Min L2`,
- object accuracy,
- format validity,
- point-bin exact accuracy.

`JointExact` is intentionally not measured. With quantized gaze bins it is too strict to be a useful model-selection signal.

When changing output schema, update parser tests and eval tests together.

## Checkpoints

Implemented in `model/utils/checkpoint.py`.

Saved content:

- LoRA adapter in `lora_adapter/`,
- processor in `processor/`,
- trainer state with epoch/optimizer/scheduler,
- `added_token_rows.pt` for added-token compatibility,
- `gaze_token_rows.pt` for exact gaze token row backup.

New trainable-token PEFT rows are saved through adapter save plus effective row backup.

Checkpoint layout:

- `last/`: most recent checkpoint.
- `best/`: checkpoint selected by `checkpoint_monitor`.

## Compatibility Notes

- Preserve old checkpoint compatibility unless the user explicitly accepts breaking it.
- Test both save and load paths when feasible.
- Gaze tokens may occupy reserved rows below the original base vocab size, so exact token row backups matter.
