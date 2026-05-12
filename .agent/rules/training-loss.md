---
description: Training loop, LoRA trainable tokens, augmentation, and losses.
---

# Training And Loss

## Trainer

`model/trainer.py` owns setup and dispatch. Stage-specific epoch logic lives in:

- `model/sft_trainer.py` for pure SFT,
- `model/sdft_trainer.py` for SDFT teacher-forcing and rollout,
- `model/rl_trainer.py` for RL.

High-level order:

1. Load YAML and CLI overrides.
2. Seed RNGs.
3. Load processor and Qwen3-VL base model.
4. Register gaze special tokens and resize token embeddings.
5. Build train/val/test datasets.
6. Create LoRA model.
7. Use PEFT trainable token rows for gaze special tokens.
8. Dispatch to the selected training path based on `train_stage`.
9. Save `last/` according to `save_last_every_n_epochs` and monitored `best/` checkpoints.
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

- `loss_mask_point`
- `loss_mask_object`
- `loss_mask_format`

`compute_answer_loss()` uses the point/object/format masks for structured CE.

Current weighted terms:

- `loss_point_weight=3.0`
- `loss_object_weight=1.0`
- `loss_format_weight=1.0`
- `gaussian_point_sigma=3.0`

Point CE:

- If `gaussian_point_sigma > 0`, point tokens use Gaussian soft-label CE over loc bins.
- Otherwise point tokens use hard CE.
- Gaussian point CE normalizes over loc-token logits only, not the full vocabulary. This intentionally changes the absolute loss scale compared with older full-vocab normalization.
- Validation does not compute teacher-forced loss in the active path. It uses generation metrics only.

## SDFT

`sdft.yaml` uses `train_stage: "sdft"` and requires `checkpoint_dir` to point to a Stage1 SFT checkpoint.

Current SDFT rollout stabilization choices:

- `distil_kl_weight=0.5`
- `distil_temperature=2.0`
- `teacher_update="ema"`
- `teacher_ema_decay=0.999`
- `sdft_ce_weight=0.3`
- KL applies to point/object tokens, not format tokens (`kl_on_format=false`).

In rollout mode, if the valid batch has no reasoning or object text, teacher inputs are identical to student inputs. `train_step_sdft_rollout()` reuses the student processed inputs/masks for the teacher forward pass to avoid a duplicate Qwen image tokenization pass.

## Augmentation

Implemented in `model/utils/data_utils.py`.

Current SFT config uses `train_augmentation_mode_direct: "no_aug"`. Available modes include `full`, `no_crop`, `color_only`, `crop_only`, and `no_aug`.

Be careful with spatial augmentation: it must keep gaze coordinates, bbox coordinates, and visual prompting consistent.

## DataLoader And Cache

Train/val DataLoaders use persistent workers when `num_workers > 0`.

Image caches are per dataset instance and therefore per worker process. With `num_workers=8` and `image_cache_size=1000`, the effective upper bound is about 8000 cached PIL images. This improves repeated I/O but can exceed Slurm CPU RAM limits if `--mem` is too small.
