---
description: Coding conventions and common pitfalls for all agents.
alwaysApply: true
---

# Coding Rules

## General

- Read existing implementation before editing.
- Prefer `rg` and `rg --files` for search.
- Keep changes scoped to the user request.
- Do not revert unrelated local changes.
- Use `apply_patch` for manual edits when possible.
- Preserve `from __future__ import annotations` in Python modules.
- Prefer explicit casts for config values: `int(...)`, `float(...)`, `bool(...)`.
- Keep code ASCII unless the file already uses non-ASCII or content requires it.
- Add comments only for non-obvious behavior.

## Schema Changes

When changing output schema, update all relevant surfaces:

1. `model/utils/gaze_tokens.py`
2. `model/utils/processor_collate.py`
3. `model/datasets.py`
4. `model/utils/eval_utils.py`
5. `sft.yaml`
6. `tests/`

## Common Pitfalls

- Do not reintroduce stale config keys just because older docs mention them.
- Do not describe the active output as `Point: x y\nObject: label`; that is legacy.
- Do not describe the active SFT output as `<|gaze_reasoning|>...`; current Stage1 SFT is direct point/object.
- Do not assume val/test have GT reasoning text.
- Do not add reasoning scaffold to the active SFT path unless the user explicitly requests a schema change.
- Do not use spatial augmentation unless coordinate, bbox, and visual-prompt consistency is preserved.
- Do not make all embedding rows trainable for new runs; use PEFT trainable token indices.
- Do not rely on `train/view_reasoning_frac`; it was removed as obsolete.
- Do not disable constrained decoding just because older docs mention reasoning; constrained decoding is valid for the active point/object schema.
- Do not truncate Qwen VLM processor inputs in a way that can break image token alignment.
- Do not increase `image_cache_size` without considering `num_workers * image_cache_size` and Slurm CPU memory limits.
