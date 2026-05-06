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
- Do not assume val/test have GT reasoning text.
- Do not make reasoning scaffold optional in the active SFT path.
- Do not use strong spatial augmentation with reasoning targets unless coordinate/reasoning consistency is redesigned.
- Do not make all embedding rows trainable for new runs; use PEFT trainable token indices.
- Do not rely on `train/view_reasoning_frac`; it was removed as obsolete.
- Do not enable constrained decoding around reasoning without redesigning free-form reasoning generation.
- Do not truncate Qwen VLM processor inputs in a way that can break image token alignment.
