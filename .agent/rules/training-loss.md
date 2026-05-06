---
description: Training loop, LoRA trainable tokens, augmentation, and losses.
---

# Training And Loss

## Trainer

`model/trainer.py` owns the SFT train/eval loop.

High-level order:

1. Load YAML and CLI overrides.
2. Seed RNGs.
3. Load processor and Qwen3-VL base model.
4. Register gaze special tokens and resize token embeddings.
5. Build train/val/test datasets.
6. Create LoRA model.
7. Use PEFT trainable token rows for gaze special tokens.
8. Train with structured CE losses.
9. Save `last/` and monitored `best/` checkpoints.
10. Run validation/test generation metrics when configured.

Per-epoch timing is printed as:

```text
[TIME train {epoch}] data_wait=... fwd_loss=... backward=... other_step=... steps=...
```

Use this before guessing about slow training.

## PEFT Trainable Tokens

New LoRA runs should use:

```python
LoraConfig(..., trainable_token_indices=gaze_token_ids)
```

This avoids making the full embedding matrix trainable. It is more efficient than setting `requires_grad=True` on all embedding rows and masking non-gaze gradients in a hook.

Compatibility:

- `peft_config_has_trainable_tokens()` checks loaded adapter metadata.
- Old checkpoints without trainable-token metadata fall back to `enable_token_id_gradients()`.
- Checkpoint row backups save effective PEFT trainable-token weights via `get_effective_embedding_weight()`.

Do not switch to a separate mixed `TrainableTokensConfig` adapter unless PEFT save/load behavior has been revalidated.

## Losses

Implemented in `model/utils/loss_utils.py`.

Structured masks:

- `loss_mask_reasoning`
- `loss_mask_point`
- `loss_mask_object`
- `loss_mask_format`

`compute_answer_loss()` dispatches to `compute_structured_loss()` when structured masks exist.

Current weighted terms:

- `loss_point_weight=3.0`
- `loss_object_weight=1.0`
- `loss_format_weight=0.2`
- `loss_reasoning_weight=0.1`
- `gaussian_point_sigma=7.0`
- `point_expectation_weight=0.1`
- `point_expectation_loss="l2"`

Point CE:

- If `gaussian_point_sigma > 0`, point tokens use Gaussian soft-label CE over loc bins.
- Otherwise point tokens use hard CE.
- Validation does not compute teacher-forced loss in the active path. It uses generation metrics only.

Expectation loss:

- Computes expected loc bin from the predicted distribution over loc tokens.
- Penalizes distance to GT bin using L1 or L2.
- It is auxiliary and lightweight compared with Qwen forward/backward, but still adds work on point-token logits.

## Augmentation

Implemented in `model/utils/data_utils.py`.

Active train augmentation is weak photometric-only:

- bbox sanitization,
- weak `color_jitter` with probability `0.5`,
- no horizontal flip,
- no random crop,
- no bbox expansion,
- no coordinate-changing spatial augmentation.

This preserves gaze coordinates and reasoning annotations.
