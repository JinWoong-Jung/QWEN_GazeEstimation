---
description: Pytest commands and test maintenance rules.
globs: ["tests/**/*.py", "tests/*.py"]
---

# Testing Guide

Default command:

```bash
python -m pytest tests/
```

Useful focused commands:

```bash
python -m pytest tests/test_gaze_tokens.py tests/test_special_token_pipeline.py tests/test_constrained_decoding.py
python -m pytest tests/test_structured_loss.py
python -m pytest tests/test_full_generation.py
python -m pytest tests/test_run_name_config.py
python -m py_compile model/trainer.py model/sft_trainer.py model/sdft_trainer.py model/rl_trainer.py model/datasets.py model/utils/*.py
```

Testing notes:

- Unit tests should stay CPU-friendly.
- Prefer small tensors and lightweight mocks.
- When changing the structured schema, update parser, mask, generation, and config tests together.
- For loss tests, cover empty masks, invalid labels, and all-invalid batches.
- Full Qwen forward/backward is not expected in unit tests.
