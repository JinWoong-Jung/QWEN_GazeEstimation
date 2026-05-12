# QWEN_GazeEstimation Agent Guide

This is the entry document for AI coding agents working in this repository.
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
- `.agent/rules/training-loss.md` for trainer behavior, PEFT, augmentation, SFT/SDFT losses, and DataLoader efficiency.
- `.agent/rules/eval-checkpoint.md` for constrained generation metrics, parsing, and checkpoint save/load.

## Current Active State

The active Stage1 SFT path fine-tunes Qwen3-VL with LoRA for gaze estimation from a scene image and head bounding box.

The active target schema is the direct point-object span-marker format:

```text
<|point_start|><loc_x><loc_y><|point_end|><|object_start|><obj_k><|object_end|>
```

Important current facts:

- `train_stage` can be `sft`, `sdft`, or `rl`; Stage1 uses `sft.yaml` with `train_stage: "sft"`.
- The active SFT/SDFT output format is configured by `output_format`; supported values are `point_object` and `text_point_object`.
- Qwen chat templating wraps assistant answers with chat markers; loss masks supervise the answer span and the trailing `<|im_end|>` stop token.
- Validation/test use generation metrics, and current SFT config uses constrained decoding for point/object selection.
- New LoRA runs use `LoraConfig(trainable_token_indices=gaze_token_ids)` for gaze special token rows.
- Train DataLoaders use persistent workers; image caches are per worker and can cause CPU RAM OOM if sized too aggressively.
- `last/` checkpoints are saved according to `save_last_every_n_epochs`; `best/` is selected by `checkpoint_monitor`.
- Do not describe the active output as legacy pure text `Point: ... Object: ...` or as `<|gaze_reasoning|>...`.

## Common Commands

```bash
python main.py --config sft.yaml
python main.py --config sdft.yaml
python main.py --config RL.yaml
python -m pytest tests/
python -m py_compile model/trainer.py model/sft_trainer.py model/sdft_trainer.py model/rl_trainer.py model/utils/*.py
```

## Agent Rules

- Read existing code before editing.
- Prefer `rg` / `rg --files` for search.
- Keep edits scoped to the user request.
- Do not revert unrelated local changes.
- Update tests when behavior changes.
- When changing output schema, update tokens, masks, datasets, eval parsing, config, and tests together.
