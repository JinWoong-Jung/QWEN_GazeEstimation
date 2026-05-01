# TODO: Reasoning-based SFT refactor

## Goal

Current SFT is effectively `Point -> Object -> Reasoning`, so neither object nor reasoning can causally condition the point tokens. Refactor the SFT path so we can run clean ablations:

- `use_reasoning: false`: direct answer SFT, `Object -> Point`
- `use_reasoning: true`: reasoning-first SFT, preferably `Reasoning -> Object -> Point`
- optional compatibility mode: current post-hoc format, `Point -> Object -> Reasoning`

Primary target format for reasoning SFT:

```text
<think>
Reasoning: <grounded reasoning text>
</think>
Object: <obj_KKK>
Point: <loc_XXX><loc_YYY>
```

This makes both object and reasoning available before the final point generation step.

## Current Issues Confirmed

- `model/utils/gaze_tokens.py` currently builds `Point -> Object`, then appends `<think>Reasoning</think>` in `build_structured_target_text_with_reasoning()`.
- `model/utils/gaze_tokens.py` parser currently expects point-first output, with optional reasoning after object.
- `sft.yaml` asks the model to reason after returning point/object, and `generation_stop_at_object_end: true` stops generation before post-hoc reasoning matters at inference.
- `model/datasets.py` applies train augmentation before loading original-image reasoning, so hflip/crop can contradict left/right or spatial relation words in the reasoning text.
- `use_reasoning` currently mixes several meanings: load reasoning files, force empty reasoning scaffold, change prompt wording, and add reasoning loss. These should be separated enough to ablate cleanly.
- Even when `use_reasoning=false`, the new default should be `Object -> Point`, not `Point -> Object`, so object semantics can condition point generation.

## P0: Template and Config Refactor

- Add explicit config fields instead of overloading `use_reasoning`.

```yaml
train:
  use_reasoning: true

reasoning:
  target_order: "reasoning_object_point"  # object_point | point_object | point_object_reasoning | reasoning_object_point
  force_reasoning_format_train: true
  force_reasoning_format_eval: false
  parse_reasoning_object_line: true
  disable_spatial_aug_for_reasoning: true
  reasoning_safe_aug: "color_jitter_only"  # none | color_jitter_only | full
```

- Keep `train.use_reasoning` as the high-level true/false switch.
- Add `reasoning.target_order` to control teacher-forcing target format.
- Add separate prompt templates for direct and reasoning modes.
- Ensure `use_reasoning=false` produces no `<think>` scaffold and no reasoning loss tokens, while still using `Object -> Point`.
- Ensure eval does not force an empty reasoning block unless explicitly requested.

## P0: Target Text Format

- Update `build_structured_target_text()` so the direct baseline becomes object-first:

```text
Object: <obj_KKK>
Point: <loc_XXX><loc_YYY>
```

- Replace or extend `build_structured_target_text_with_reasoning()` with an order-aware builder, for example:

```python
build_structured_target_text(
    point_x,
    point_y,
    obj_id,
    num_classes,
    reasoning_text=None,
    target_order="object_point",
    force_reasoning_format=False,
)
```

- Support at least these orders:

```text
object_point
point_object
point_object_reasoning
reasoning_object_point
```

- Default direct training to `object_point`.
- Default reasoning training to `reasoning_object_point`.
- Keep `point_object` and `point_object_reasoning` only as backwards-compatible ablations.

## P0: Parser and Generation Stop Rule

- Make `parse_structured_output_text()` accept object-first, point-first, and reasoning-first formats.
- For `reasoning_object_point`, the parser must still extract:

```python
{
    "valid_format": bool,
    "point_bins": (x_bin, y_bin),
    "point_xy": (x, y),
    "object_id": int | None,
    "object_unknown": bool,
}
```

- Replace `generation_stop_at_object_end` with an order-aware stop policy.
- For `object_point`, stop after the final `Point` line.
- For `point_object`, stop after `Object`.
- For `point_object_reasoning`, either do not stop at object or treat it as post-hoc mode.
- For `reasoning_object_point`, stop after the final `Point` line, not after object.
- Add a small unit test proving `Object` before `Point` is parsed correctly.

## P0: Loss Masks and Teacher Forcing

- Update `build_structured_masks()` so masks are independent of output order.
- Point mask should cover exactly the two `<loc_*>` final answer tokens.
- Object mask should cover exactly one `<obj_*>` or `<obj_unknown>` token.
- Reasoning mask should cover only reasoning content inside `<think>...</think>`, not labels or tags.
- Format mask should cover labels/tags/newlines and chat EOS.
- Add tests for:

```text
Object -> Point
Point -> Object
Point -> Object -> Reasoning
Reasoning -> Object -> Point
```

- Verify masks remain disjoint and the union covers the full answer span.

## P0: Reasoning-safe Augmentation

- Do not apply horizontal flip or crop to samples whose reasoning text comes from the original image, unless the reasoning is transformed too.
- Recommended immediate implementation:

```text
if sample has reasoning and reasoning.disable_spatial_aug_for_reasoning:
    apply color_jitter only
else:
    apply full train augmentation
```

- Keep point-only samples on full augmentation.
- Log counts for:

```text
has_reasoning
reasoning_safe_aug_applied
full_aug_applied
```

- Later option: implement left/right text swapping for hflip, but do not rely on this first because natural language spatial phrases are messy.

## P1: Reasoning File Loader and Object Consistency

- Extend `load_reasoning_text()` or add `load_reasoning_record()` to parse both lines:

```text
Object: ...
Reasoning: ...
```

- Return a structured record:

```python
{
    "object_text": str | None,
    "reasoning_text": str | None,
}
```

- Compare reasoning `Object:` phrase against `gaze_pseudo_label` / `label_id`.
- Add alias/canonicalization for common mismatches like:

```text
computer monitor screen -> screen
smartphone -> phone
wine glass -> glass
```

- Log object consistency stats:

```text
reasoning_object_present
reasoning_object_vocab_hit
reasoning_object_matches_label
reasoning_object_mismatch
```

- Do not silently use a contradictory reasoning object as target until this is validated.

## P1: Reasoning Coverage and Debug Logs

- At dataset construction or first epoch start, log:

```text
reasoning_index_size
train_has_reasoning_count / train_len
val_has_reasoning_count / val_len
first 10 reasoning keys attempted
first 10 matched reasoning paths
```

- Keep `QWEN_DEBUG_TARGET_EXAMPLE=1`, but ensure it prints examples for both:

```text
use_reasoning=false
use_reasoning=true
```

- Add mask count logs:

```text
n_point_tokens
n_object_tokens
n_format_tokens
n_reasoning_tokens
```

## P1: Prompt Templates

- Direct prompt should not ask for reasoning:

```text
Return exactly:
Object: <obj_KKK>
Point: <loc_NNN><loc_MMM>
```

- Reasoning-first prompt should match the teacher-forcing target:

```text
First reason briefly about the gaze target, then choose the object, then output the final point.
Return exactly:
<think>
Reasoning: <your reasoning here>
</think>
Object: <obj_KKK>
Point: <loc_NNN><loc_MMM>
```

- Avoid a prompt/target mismatch where the prompt asks for `Point -> Object -> Reasoning` but target uses `Reasoning -> Object -> Point` or direct `Object -> Point`.

## P1: Ablations to Run

- A. Direct baseline: `use_reasoning=false`, `target_order=object_point`
- B. Current compatibility: `use_reasoning=true`, `target_order=point_object_reasoning`
- C. Proposed reasoning-first: `use_reasoning=true`, `target_order=reasoning_object_point`
- D. Reasoning-first with spatial augmentation disabled for reasoning samples
- E. Direct baseline with same augmentation policy as D, to isolate augmentation effects

Track:

```text
val/test AUC
val/test min_dist
val/test dist
object_acc
format_valid
reasoning coverage
reasoning token count
```

## P2: Spatial Loss Follow-up

- Token CE treats nearby and far-away bin mistakes equally.
- Add one of:

```text
Gaussian soft CE over x/y bins
2D heatmap auxiliary decoder
continuous point regression head from final hidden state
Stage-2 reward using exp(-beta * L2)
```

- This is independent of reasoning and should be tested as a point-performance upgrade.

## P2: Head Crop Multi-image Follow-up

- Add optional input mode:

```text
Image 1: full scene with red head box
Image 2: cropped/upsampled head region
```

- This may matter more than reasoning if full-scene resize loses head/eye cues.

## Acceptance Criteria

- `use_reasoning=false` trains and evaluates as `Object -> Point` without `<think>` tokens.
- `use_reasoning=true` can train with `Reasoning -> Object -> Point`.
- Parser validates generated `Object -> Point` output.
- Parser validates generated `Reasoning -> Object -> Point` output.
- Generation stop rule does not truncate before final point in reasoning-first mode.
- Reasoning samples no longer receive spatial augmentation that contradicts original reasoning text.
- Unit tests cover target text, parser, masks, and augmentation policy.
- Logs clearly show how many samples actually used reasoning.
