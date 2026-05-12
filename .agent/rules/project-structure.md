---
description: Repository layout, entrypoints, and module responsibilities.
alwaysApply: true
---

# Project Structure

## Summary

QWEN_GazeEstimation fine-tunes Qwen3-VL with LoRA for gaze estimation from a scene image plus a head bounding box. The model is an autoregressive text generator over gaze special tokens. There are no active auxiliary point/object heads.

## Entrypoints

- `main.py`: calls `model.trainer.main()`.
- `sft.yaml`: active SFT configuration.
- `sdft.yaml`: Stage2 self-distillation configuration.
- `RL.yaml`: separate RL path; do not assume SFT/SDFT changes automatically cover RL.
- `sft.sh`: Slurm launcher for `python main.py --config sft.yaml`.

## Layout

```text
QWEN_GazeEstimation/
├── main.py
├── sft.yaml
├── sdft.yaml
├── sft.sh
├── RL.yaml
├── RL_data_pipeline.py
├── eval_train_dist.py
├── model/
│   ├── model.py
│   ├── trainer.py
│   ├── sft_trainer.py
│   ├── sdft_trainer.py
│   ├── rl_trainer.py
│   ├── train_context.py
│   ├── datasets.py
│   ├── modules/preprocess.py
│   ├── Qwen3-VL-2B-Instruct/
│   ├── Qwen3-VL-4B-Instruct/
│   └── utils/
│       ├── checkpoint.py
│       ├── common.py
│       ├── config_parser.py
│       ├── data_utils.py
│       ├── eval_utils.py
│       ├── gaze_tokens.py
│       ├── label_bank.py
│       ├── loss_utils.py
│       ├── object_tokens.py
│       ├── processor_collate.py
│       ├── rl_utils.py
│       ├── special_tokens.py
│       └── wandb_utils.py
├── data/
├── scripts/
├── tests/
├── checkpoints/
├── outputs/
└── wandb/
```

Generated directories such as `checkpoints/`, `outputs/`, `wandb/`, `.pytest_cache/`, and `__pycache__/` should not be hand-edited unless the user explicitly asks.

## Module Responsibilities

- `model/model.py`: minimal `QwenTextGenerationModel` wrapper; returns logits and delegates generation.
- `model/trainer.py`: config loading, model setup, dataset setup, LoRA, train-stage dispatch, checkpointing, final eval/test orchestration.
- `model/sft_trainer.py`: pure SFT epoch loop.
- `model/sdft_trainer.py`: SDFT teacher-forcing and rollout epoch logic.
- `model/rl_trainer.py`: RL/GRPO-style training path.
- `model/train_context.py`: typed runtime context shared by training paths.
- `model/datasets.py`: train/val/test dataset classes and structured target construction.
- `model/utils/gaze_tokens.py`: structured target formatting, quantization, structured parser, legacy re-exports.
- `model/utils/special_tokens.py`: gaze special-token constants, loc/object token formatting, tokenizer registration.
- `model/utils/processor_collate.py`: chat template input building, labels, structured loss masks.
- `model/utils/loss_utils.py`: structured CE, Gaussian point CE over loc-token logits, KL distillation helpers.
- `model/utils/eval_utils.py`: constrained/free generation decode, structured parsing, validation/test metrics.
- `model/utils/data_utils.py`: annotation loading, label maps, reasoning index, augmentation.
- `model/utils/checkpoint.py`: LoRA/processor/trainer state and gaze token row save/load.
- `model/utils/config_parser.py`: YAML flattening and CLI override support.
- `model/utils/wandb_utils.py`: wandb setup and run metadata.
