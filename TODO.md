# TODO

## Medium Separation Plan: SFT / SDFT / RL

Goal: split the current monolithic `model/trainer.py` into stage-specific modules without changing training behavior. Keep this refactor incremental, testable, and easy to bisect.

Current first step already done:
- `model/sdft_trainer.py` contains `train_step_sdft_rollout(...)`.
- `model/trainer.py` still owns setup, dispatch, the SFT loop, teacher-forcing SDFT, RL, eval, logging, and checkpointing.

Target module layout:

```text
model/
  trainer.py              # orchestration: parse/setup/dispatch/eval-only/test-only
  sft_trainer.py          # pure SFT epoch loop
  sdft_trainer.py         # SDFT teacher-forcing + rollout training loop/helpers
  rl_trainer.py           # RL/GRPO training loop
  train_context.py        # shared dataclass/config objects for common state
  train_runtime.py        # optional shared optimizer/scheduler/checkpoint/eval helpers
```

Refactor rules:
- Do not change model outputs, loss formulas, masks, checkpoint names, or wandb metric names during structural moves.
- Move code first, then clean code. Avoid mixing behavior changes with file moves.
- Keep old public entrypoint behavior: `python main.py --config ...` should still go through `model.trainer.main()`.
- After every phase, run at least:

```bash
python -m py_compile model/trainer.py model/sft_trainer.py model/sdft_trainer.py model/rl_trainer.py model/utils/config_parser.py
pytest tests/test_structured_loss.py tests/test_constrained_decoding.py -q
```

If a phase does not yet create one of those files, omit it from `py_compile`.

---

## Phase 1: Stabilize Shared Training State

Create `model/train_context.py`.

Add small dataclasses, not a giant object:
- `TrainRuntime`: `device`, `amp_dtype`, `processor`, `coord_bins`, `num_classes`, `scene_size`
- `LossConfig`: point/object/format weights, gaussian sigma, loc/object token ids
- `EvalConfig`: checkpoint monitor, eval target order, generation max tokens, constrained decoding options
- `SdftConfig`: mode, CE weight, rollout generation settings, constrained rollout settings, KL scope, distil settings

Safety notes:
- Dataclasses should only collect already-computed values.
- Do not move dataset/model initialization in this phase.
- Keep construction in `trainer.py` so failures remain easy to compare.

Validation:
- `py_compile`
- existing focused tests
- inspect that `sdft.yaml` values still land in `args` and then in `SdftConfig`

---

## Phase 2: Extract Pure SFT Loop

Create `model/sft_trainer.py`.

Move only the non-SDFT branch from the current SFT/SDFT loop:
- GT labels all-ignore skip
- student forward
- `compute_answer_loss(...)`
- grad accumulation
- optimizer/scheduler/grad clipping
- wandb train metrics for SFT

Return a compact result object:
- `global_step`
- `best_monitor_value`
- epoch train loss
- timing counters if still needed

Keep in `trainer.py` for now:
- dataset/collator creation
- optimizer/scheduler creation if moving them would make the phase too large
- validation/test/checkpoint calls

Safety notes:
- The first extraction may accept many parameters. That is fine.
- Do not abstract optimizer stepping until behavior is locked down.
- SFT stage must not import SDFT helpers.

Validation:
- Run SFT smoke with `epochs=0` or `eval_only=True` if available.
- Run unit tests.
- Compare one short dry-run log before/after if a small fixture/config exists.

---

## Phase 3: Extract SDFT Training Loop

Expand `model/sdft_trainer.py` from step-helper ownership to loop ownership.

Move into `sdft_trainer.py`:
- rollout branch that calls `train_step_sdft_rollout(...)`
- teacher-forcing SDFT branch currently inside the SFT fallback path
- SDFT-specific wandb metrics:
  - `train/sdft_mode`
  - `train/sdft_kl_loss`
  - `train/rollout_format_valid`
  - `train/valid_kl_sample_frac`
  - `train/sdft_ce_loss`
  - existing `train/loss_kl`

Keep shared behavior identical:
- teacher-forcing SDFT remains `GT CE + distil_kl_weight * KL(teacher || student)`
- rollout SDFT remains generated-answer KL, optional GT CE via `sdft_ce_weight`
- constrained rollout remains controlled only by `rollout_constrained_decoding`

Safety notes:
- `sft_trainer.py` should no longer contain any `train_stage == "sdft"` checks after this phase.
- `trainer.py` should dispatch:
  - `train_stage == "sft"` -> `run_sft_training(...)`
  - `train_stage == "sdft"` -> `run_sdft_training(...)`
  - `train_stage == "rl"` -> `run_rl_training(...)`

Validation:
- `py_compile`
- focused tests
- verify `sdft_mode: teacher_forcing` and `sdft_mode: rollout` both reach distinct code paths
- verify KL direction remains `KL(teacher || student)`

---

## Phase 4: Extract RL Loop

Create `model/rl_trainer.py`.

Move `_run_rl_training(...)` out of `trainer.py` with minimal edits.

Keep dependencies explicit:
- pass model/path/checkpoint/runtime objects in
- avoid importing `main()` setup code back into `rl_trainer.py`
- keep RL reference model loading inside `rl_trainer.py` unless it becomes shared later

Safety notes:
- This phase is mostly a file move.
- Do not change GRPO reward, KL controller, rollout reuse, or logging names.
- If circular imports appear, move tiny shared helpers to `train_runtime.py`, not back into `trainer.py`.

Validation:
- `py_compile`
- existing RL utility tests
- optional `eval_only` or tiny dry run if available

---

## Phase 5: Reduce `trainer.py` to Orchestration

After SFT/SDFT/RL loops are extracted, `trainer.py` should keep only:
- config parsing
- path resolution
- processor/model/token setup
- dataset and dataloader construction
- checkpoint loading
- stage dispatch
- eval-only/test-only orchestration
- final wandb finish

Remove from `trainer.py`:
- optimizer step internals
- stage-specific rollout logic
- stage-specific loss composition
- stage-specific train logging payload construction

Safety notes:
- Keep evaluation functions in `eval_utils.py`; do not move them during this phase.
- Keep checkpoint helpers in `utils/checkpoint.py`.
- Do not rename config keys.

Validation:
- `rg -n "train_stage ==|sdft_mode|rl_" model/trainer.py`
  - `trainer.py` may contain dispatch/config reads, but not inner training-loop logic.
- `py_compile`
- focused tests

---

## Phase 6: Optional Cleanup After Separation

Only after the structural split is stable:
- Replace long parameter lists with dataclasses from `train_context.py`.
- Share optimizer/scheduler setup through `train_runtime.py` if SFT and SDFT still duplicate too much.
- Add small tests for config-to-dataclass mapping.
- Add one smoke test that imports all stage modules without initializing Qwen weights.

Do not do these before Phase 2-5 are green; they are polish, not separation blockers.
