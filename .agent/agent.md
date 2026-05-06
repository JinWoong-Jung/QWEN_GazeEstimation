# QWEN_GazeEstimation Agent Guide

This is the entry document for all AI coding agents working in this repository.
Keep this file short. Put detailed instructions in `.agent/rules/`.

## Read Order

Start with these files:

1. `.agent/agent.md`
2. `.agent/rules/current-sft.md`
3. `.agent/rules/project-structure.md`
4. `.agent/rules/coding.md`
5. `.agent/rules/testing.md`

Load the more specific rule files when the task touches that area:

- `.agent/rules/data-schema.md` for datasets, prompts, target text, special tokens, and reasoning text.
- `.agent/rules/training-loss.md` for trainer behavior, PEFT, augmentation, and losses.
- `.agent/rules/eval-checkpoint.md` for generation metrics, parsing, and checkpoint save/load.

## Current Active State

The active SFT path fine-tunes Qwen3-VL with LoRA for gaze estimation from a scene image and head bounding box.

The model generates this structured special-token schema:

```text
<|gaze_reasoning|><reasoning text><|gaze_point|><loc_x><loc_y><|gaze_object|><obj_k>
```

Important current facts:

- The active target order is always `reasoning_point_object`.
- Train, validation, test, and `test_only` all use the same schema scaffold.
- Train uses reasoning annotations when available; val/test do not have GT reasoning text.
- The old direct/reasoning multiview SFT sampler is no longer active.
- `train/view_reasoning_frac` was removed as obsolete.
- Strong spatial augmentation is disabled; active train augmentation is weak photometric-only.
- New LoRA runs use `LoraConfig(trainable_token_indices=gaze_token_ids)` for gaze special token rows.
- Do not describe the active output as legacy pure text `Point: ... Object: ...`.

## Common Commands

```bash
python main.py --config sft.yaml
python main.py --config sft.yaml train.lr=5e-5 train.epochs=5
python -m pytest tests/
python -m py_compile model/trainer.py model/datasets.py model/utils/*.py
```

## Agent Rules

- Read existing code before editing.
- Prefer `rg` / `rg --files` for search.
- Keep edits scoped to the user request.
- Do not revert unrelated local changes.
- Update tests when behavior changes.
- When changing output schema, update tokens, masks, datasets, eval parsing, config, and tests together.
