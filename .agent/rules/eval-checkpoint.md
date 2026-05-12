---
description: Evaluation, generation, parsing, and checkpoint behavior.
---

# Evaluation And Checkpoints

## Evaluation

`model/utils/eval_utils.py` handles generation and metrics.

Current eval/test generation:

- Generates or constrains the direct point/object span-marker schema.
- Validation is pure generation-based; the active path does not run teacher-forced validation loss.
- Keep `preview_val_samples=0` when measuring validation speed; previews are a separate generation pass.
- Current `sft.yaml` uses `constrained_decoding=true`.
- `generation_max_new_tokens=8`.
- `constrained_loc_decoding` can be `argmax` or `round_expectation`.

Metrics are based on parsed structured output:

- validation point distance as `Dist`,
- test point distance as `Avg L2` and `Min L2`,
- object accuracy,
- format validity,
- point-bin exact accuracy.

`JointExact` is intentionally not measured. With quantized gaze bins it is too strict to be a useful model-selection signal.

When changing output schema, update parser tests and eval tests together.

Constrained decoding is natural for this repo's current closed-vocabulary point/object schemas. It supports both `point_object` and `text_point_object`. Free generation is stricter because it also measures format following; constrained generation is usually a cleaner point/object selection metric.

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

- `last/`: saved according to `save_last_every_n_epochs` and always on the final epoch.
- `best/`: checkpoint selected by `checkpoint_monitor`.

The trainer currently runs final test after training in the SFT/SDFT path. If adding a `run_test` option, keep the existing `test_only` path behavior intact.

## Compatibility Notes

- Preserve old checkpoint compatibility unless the user explicitly accepts breaking it.
- Test both save and load paths when feasible.
- Gaze tokens may occupy reserved rows below the original base vocab size, so exact token row backups matter.
