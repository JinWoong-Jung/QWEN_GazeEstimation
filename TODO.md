# TODO: Multi-view SFT for Reasoning as Train-time Auxiliary Supervision

## 0. Goal

Current reasoning SFT should move away from requiring reasoning at inference time. Use reasoning as an auxiliary train-time view while keeping validation and test on the short direct prediction path.

```text
Train View A: Direct View
Prompt: predict object and point only
Target: Object -> Point

Train View B: Reasoning View
Prompt: produce one short visual reasoning sentence, then object and point
Target: Reasoning -> Object -> Point

Val/Test:
Always Direct View only
Prompt/Target/Eval format: Object -> Point
```

Core principle:

```text
Use reasoning as train-time auxiliary supervision, not as the required inference-time output.
```

Because training time and compute are limited, do not plan broad ablations now. Implement one conservative default recipe and add enough logging to quickly detect whether reasoning is hurting direct gaze prediction.

Default policy:

```text
reasoning annotations used = all valid train reasoning files
direct train view = 100% of train records
reasoning train view = 100% of train records with valid reasoning
train sampling ratio = direct 80%, reasoning 20%
val/test reasoning usage = 0%
```

## 0.1 Immediate Low-risk Config Changes

Before the full multi-view refactor, apply the safe config changes that reduce the chance of reasoning CE hurting point/object learning:

```yaml
reasoning:
  force_reasoning_format_train: false
  force_reasoning_format_eval: false

eval:
  generation_max_new_tokens: 24
  generation_stop_at_object_end: false

loss:
  loss_reasoning_weight: 0.05
```

Rationale:

- `loss_reasoning_weight: 0.5` is too high for a `val/dist`-first objective.
- `generation_max_new_tokens: 128` is unnecessary for direct eval and encourages verbose output.
- `force_reasoning_format_train: true` can create empty reasoning scaffolds and misleading `n_reasoning_tokens`.
- Keep `generation_stop_at_object_end: false` because `Object -> Point` must not stop after object generation.

## 1. Dataset Structure

### TODO 1.1 Expand train records into direct and reasoning views

For each train record, create a direct view:

```python
{
    "base_record": record,
    "view_type": "direct",
    "prompt_type": "direct_object_point",
    "target_type": "object_point",
    "use_reasoning": False,
    "augmentation_policy": "full",
}
```

For each train record with a valid `Reasoning:` line, create a reasoning view:

```python
{
    "base_record": record,
    "view_type": "reasoning",
    "prompt_type": "reasoning_object_point",
    "target_type": "reasoning_object_point",
    "use_reasoning": True,
    "augmentation_policy": "safe",
}
```

Requirements:

- Direct view must exist for every train record.
- Reasoning view should exist for every train record with valid reasoning text.
- Do not create fake empty reasoning views in normal training.
- Return `view_type`, `prompt_type`, `target_type`, and `augmentation_policy` in each sample so the trainer can log view-specific metrics.

### TODO 1.2 Keep val/test direct-only

Val/test datasets must never create reasoning views.

```python
{
    "view_type": "direct",
    "prompt_type": "direct_object_point",
    "target_type": "object_point",
    "use_reasoning": False,
    "augmentation_policy": "none",
}
```

Requirements:

- Val/test prompt must not mention reasoning, explanation, or why.
- Val/test target format must be exactly `Object -> Point`.
- Val/test generation preview must use the direct prompt.

## 2. Sampling

### TODO 2.1 Add weighted sampling for train views

Expanded train dataset is approximately:

```text
direct N + reasoning N
```

Uniform sampling would produce a 1:1 ratio, which is too much reasoning for a `val/dist`-first objective. Add `WeightedRandomSampler` or an equivalent custom sampler.

Default config:

```yaml
train:
  use_multiview_sft: true
  direct_view_ratio: 0.8
  reasoning_view_ratio: 0.2
```

Recommended weighting:

```python
direct_weight = desired_direct_ratio / num_direct_views
reasoning_weight = desired_reasoning_ratio / num_reasoning_views

sampler = WeightedRandomSampler(
    weights=weights,
    num_samples=len(train_records),
    replacement=True,
)
```

Compute-constrained decision:

- Do not run 9:1, 8:2, 7:3 ablations now.
- Use fixed `8:2` as the first default.
- If early logs show direct format degradation or reasoning leakage into direct output, switch default to `9:1`.

## 3. Prompt Design

### TODO 3.1 Split direct and reasoning prompts

Do not use one prompt for both direct and reasoning samples. The prompt should clearly identify which task the model is doing.

Direct prompt:

```text
You are given an image with a marked person.
Predict the gaze target object and gaze point of the marked person.

Return only:
Object: <object_token>
Point: <x_token><y_token>
```

Reasoning prompt:

```text
You are given an image with a marked person.
First provide one short visual reasoning sentence about where the person is looking.
Then predict the gaze target object and gaze point.

Return exactly:
Reasoning: <one short sentence>
Object: <object_token>
Point: <x_token><y_token>
```

Requirements:

- Direct prompt must not include "reasoning", "explain", "why", or similar words.
- Reasoning prompt must request one short sentence.
- Do not encourage long chain-of-thought.
- Final answer block must always use `Object -> Point`.
- New multi-view targets must not use `<think>...</think>`.
- Keep legacy `<think>...</think>` parser support only for old checkpoints/debugging if needed.

### TODO 3.2 Explicitly migrate away from `<think>...</think>`

Current code still has `<think>...</think>` assumptions in target generation, parser, and reasoning masks. The multi-view design intentionally removes this wrapper.

Migration requirements:

- Target generation must output `Reasoning: ...` as a normal first line, not a `<think>` block.
- `build_structured_target_text()` and compatibility helpers must support non-`<think>` reasoning targets.
- `build_structured_masks()` must identify reasoning content after the `Reasoning:` label without relying on `<think>` / `</think>` boundaries.
- `parse_structured_output_text()` should support direct `Object -> Point` for metrics and `Reasoning -> Object -> Point` for preview/debug.
- Existing `<think>` parsing may remain only as legacy compatibility, not as the new training default.
- Unit tests must cover both new no-`<think>` multi-view targets and legacy `<think>` inputs if legacy support is kept.

## 4. Target Builders

### TODO 4.1 Direct target builder

Direct view target:

```text
Object: <obj_255>
Point: <loc_042><loc_087>
```

Example:

```python
def build_direct_target(object_token: str, x_token: str, y_token: str) -> str:
    return f"Object: {object_token}\nPoint: {x_token}{y_token}"
```

### TODO 4.2 Reasoning target builder

Reasoning view target:

```text
Reasoning: <short normalized reasoning>
Object: <obj_255>
Point: <loc_042><loc_087>
```

Example:

```python
def build_reasoning_target(
    reasoning_text: str,
    object_token: str,
    x_token: str,
    y_token: str,
) -> str:
    reasoning_text = normalize_reasoning_text(reasoning_text)
    return (
        f"Reasoning: {reasoning_text}\n"
        f"Object: {object_token}\n"
        f"Point: {x_token}{y_token}"
    )
```

### TODO 4.3 Normalize and limit reasoning length

Reasoning is auxiliary. Keep it short so it does not dominate token budget or compete with point/object learning.

Config:

```yaml
reasoning:
  max_reasoning_words: 30
  max_reasoning_chars: 220
```

Normalization:

```python
def normalize_reasoning_text(text: str) -> str:
    text = text.strip()
    text = text.replace("\n", " ")
    text = collapse_spaces(text)
    text = truncate_to_max_words(text, max_words=30)
    text = truncate_to_max_chars(text, max_chars=220)
    if text and not text.endswith("."):
        text += "."
    return text
```

Requirements:

- Missing reasoning should remove the reasoning view, not create an empty scaffold.
- Multi-line reasoning should become one line.
- Very long reasoning should be truncated.
- Directional words are acceptable only because reasoning view uses safe augmentation.

## 5. Object Label Handling

### TODO 5.1 Use canonical object token as supervised target

Do not use free-form `Object:` text from reasoning files as the supervised object target.

Reason:

- Reasoning object text is free-form.
- Current object vocabulary is a 346-class closed set.
- Reasoning object phrases can be fine-grained, e.g. `the dessert on the plate`, `the red-and-white circular print on the table`, or `the computer monitor screen`.

The supervised object token must come from the existing label pipeline:

```python
object_token = object_id_to_special_token(label_id)
```

### TODO 5.2 Use reasoning `Object:` line only for diagnostics

Reasoning file parser should read both:

```python
{
    "object_text": str | None,
    "reasoning_text": str | None,
}
```

Use `object_text` for:

- Debugging.
- Object label consistency checks.
- Bad reasoning sample filtering.
- Future canonicalization table construction.

Logging example:

```text
reasoning/object_text: the computer monitor screen
canonical_label: screen
object_token: <obj_255>
match: true
```

Do not silently replace canonical object labels with reasoning file object text.

## 6. Augmentation

### TODO 6.1 Direct view uses full augmentation

Direct view should keep strong augmentation:

```text
random crop
horizontal flip
color jitter
resize
```

Reason: direct view is the final val/test path and should retain spatial generalization.

### TODO 6.2 Reasoning view uses safe augmentation

Reasoning text is generated from the original image. Spatial transforms can make text wrong, especially left/right/up/down and relation phrases.

Reasoning view augmentation:

```text
resize
color jitter
no horizontal flip
no random crop, unless proven safe
```

Config:

```yaml
augmentation:
  direct:
    use_random_crop: true
    use_hflip: true
    use_color_jitter: true

  reasoning:
    use_random_crop: false
    use_hflip: false
    use_color_jitter: true
```

Requirements:

- Do not use one global augmentation path for both views.
- Log `augmentation_policy` for sampled batches.

## 7. Loss Masks and Weights

### TODO 7.1 Build view-aware token masks

Direct target:

```text
Object: <obj_255>
Point: <loc_042><loc_087>
```

Mask categories:

```text
Object:, Point:     -> format loss
<obj_255>           -> object loss
<loc_042><loc_087>  -> point loss
reasoning loss      -> none
```

Reasoning target:

```text
Reasoning: short sentence.
Object: <obj_255>
Point: <loc_042><loc_087>
```

Mask categories:

```text
Reasoning:          -> format loss
short sentence      -> reasoning loss
Object:, Point:     -> format loss
<obj_255>           -> object loss
<loc_042><loc_087>  -> point loss
```

Requirements:

- Reasoning mask must exclude whitespace-only tokens.
- Empty/missing reasoning must not produce `n_reasoning_tokens = batch_size`.
- Point mask must cover exactly two loc tokens.
- Object mask must cover exactly one obj token or `<obj_unknown>`.
- Add tests for direct view and reasoning view masks.

### TODO 7.2 Use conservative reasoning loss weight

Primary objective is `val/dist`, so point loss should dominate.

Recommended:

```yaml
loss:
  loss_point_weight: 3.0
  loss_object_weight: 1.0
  loss_format_weight: 0.2
  loss_reasoning_weight: 0.05
```

Avoid:

```yaml
loss_reasoning_weight: 0.5
```

Reasoning CE should not compete strongly with point/object learning.

## 8. Evaluation

### TODO 8.1 Val/Test metrics use direct prompt only

Eval generation must always expect:

```text
Object: <obj_k>
Point: <loc_x><loc_y>
```

Requirements:

- `eval_prompt_type = direct_object_point`
- `eval_target_type = object_point`
- No reasoning prompt for main val/test metric.
- No reasoning output required for main val/test metric.

### TODO 8.2 Reduce eval generation length

Direct output is short.

Recommended:

```yaml
eval:
  generation_max_new_tokens: 16
```

Fallback:

```yaml
eval:
  generation_max_new_tokens: 24
```

Do not keep `128` for main direct eval. Use long generation only for separate reasoning previews.

### TODO 8.3 Fix stop rule for Object -> Point

`Object -> Point` must not stop after object generation.

Recommended:

```yaml
eval:
  generation_stop_at_object_end: false
```

Do not require a new `generation_stop_at_point_end` in the first patch. Prefer the simpler path:

```text
generation_stop_at_object_end = false
generation_max_new_tokens = 24
stop on EOS or max_new_tokens
```

Implement `generation_stop_at_point_end` later only if direct generations frequently continue past the point line.

## 9. Parser

### TODO 9.1 Direct eval parser should be strict

Main val/test valid output:

```text
Object: <obj_255>
Point: <loc_042><loc_087>
```

Invalid in direct eval:

- Missing object.
- Missing point.
- Reasoning text mixed into direct output.
- Extra text after point.
- Point before object.

### TODO 9.2 Reasoning parser is for preview/debug only

Keep parser support for:

```text
Reasoning: ...
Object: <obj_255>
Point: <loc_042><loc_087>
```

Use it for:

- Debug preview.
- Unit tests.
- Optional small reasoning preview generation.

Do not use reasoning parser for main val/test metric.

## 10. Config

### TODO 10.1 Add multi-view SFT config

Recommended config:

```yaml
train:
  use_multiview_sft: true
  direct_view_ratio: 0.8
  reasoning_view_ratio: 0.2

reasoning:
  use_reasoning: true
  reasoning_view_enabled: true
  force_reasoning_format_train: false
  force_reasoning_format_eval: false
  max_reasoning_words: 30
  max_reasoning_chars: 220

target:
  direct_order: "object_point"
  reasoning_order: "reasoning_object_point"

prompt:
  train_direct_prompt_type: "direct_object_point"
  train_reasoning_prompt_type: "reasoning_object_point"
  eval_prompt_type: "direct_object_point"

loss:
  loss_point_weight: 3.0
  loss_object_weight: 1.0
  loss_format_weight: 0.2
  loss_reasoning_weight: 0.05

eval:
  generation_max_new_tokens: 24
  generation_stop_at_object_end: false
  preview_val_samples: 32
```

Requirements:

- `use_reasoning=false` must disable reasoning views and reasoning loss.
- `use_multiview_sft=false` should preserve a direct-only training path.
- Config parser must correctly read nested `train`, `reasoning`, `target`, `prompt`, `loss`, and `eval` keys.

## 11. Logging and Debugging

### TODO 11.1 View ratio logs

Log actual sampled view fractions:

```text
train/view_direct_frac
train/view_reasoning_frac
```

Expected:

```text
direct ~= 0.8
reasoning ~= 0.2
```

### TODO 11.2 View-specific loss logs

Log direct and reasoning losses separately:

```text
train/direct/loss_total
train/direct/loss_point
train/direct/loss_object
train/direct/loss_format

train/reasoning/loss_total
train/reasoning/loss_point
train/reasoning/loss_object
train/reasoning/loss_format
train/reasoning/loss_reasoning
```

### TODO 11.3 Eval metric logs

Always log:

```text
val/dist
val/object_acc
val/format_valid
val/point_l2_valid_frac
val/extra_text_rate
```

Interpretation:

- If `val/point_l2_valid_frac` is low, `val/dist` is not trustworthy.
- If `val/extra_text_rate` rises, direct prompt is leaking reasoning or verbose text.
- If `val/format_valid` drops, reasoning auxiliary training may be interfering with direct decoding.

### TODO 11.4 Preview files

Save direct prompt generation examples at eval:

```json
{
  "image_path": "...",
  "prompt_type": "direct_object_point",
  "generated_text": "Object: <obj_255>\nPoint: <loc_042><loc_087>",
  "parsed_object": 255,
  "parsed_point": [0.33, 0.68],
  "gt_point": [0.31, 0.70],
  "l2": 0.028,
  "format_valid": true
}
```

Also save a small reasoning preview separately, but never use it for the main metric.

## 12. Training Schedule

### TODO 12.1 Conservative fixed default schedule

Broad ablations are not feasible, and epoch-dependent sampler curriculum adds implementation complexity. Start with one fixed conservative schedule:

```text
direct:reasoning = 8:2
loss_reasoning_weight = 0.05
```

If direct format degrades, switch to:

```text
direct:reasoning = 9:1
loss_reasoning_weight = 0.03~0.05
```

Move curriculum to Priority 3. Do not implement epoch-dependent sampler updates in the first patch.

### TODO 12.2 No broad ablation for now

The analysis suggested A/B/C/D comparisons, but current compute constraints make them impractical.

Instead:

- Implement the single default multi-view recipe.
- Compare only against already-known historical direct/reasoning runs if available.
- Use view-specific logging and previews to detect failure quickly.

Success criteria:

```text
val/dist does not get worse than the current direct baseline trend
val/format_valid remains high
val/point_l2_valid_frac remains high
direct generation does not include reasoning text
reasoning view does not dominate token/loss budget
```

Failure criteria:

```text
val/dist rises
val/format_valid drops
val/point_l2_valid_frac drops
direct generation includes Reasoning:
train/n_reasoning_tokens is unexpectedly constant at batch_size
```

## 13. Implementation Priority

### Priority 1: Must do first

- Apply immediate config safety changes: `loss_reasoning_weight=0.05`, `generation_max_new_tokens=24`, `force_reasoning_format_train=false`.
- Remove `<think>` as the new training target format and migrate target/mask/parser code to `Reasoning: ...\nObject: ...\nPoint: ...`.
- Expand train dataset into direct/reasoning views.
- Add weighted sampler or equivalent direct/reasoning sampling control.
- Keep val/test direct-only.
- Split direct and reasoning prompts.
- Split direct and reasoning target builders.
- Ensure `Object -> Point` generation does not stop after object.
- Reduce main eval `generation_max_new_tokens` to 16-24.
- Add view-type metadata to batches.
- Add view ratio and view-specific loss logs.

### Priority 2: Strongly recommended

- Direct view full augmentation.
- Reasoning view safe augmentation.
- Reasoning max word/char normalization.
- Reasoning `Object:` line consistency diagnostics.
- Direct eval preview saving.
- `val/point_l2_valid_frac`, `val/format_valid`, `val/extra_text_rate` logging.
- Fix whitespace-only reasoning mask so empty reasoning does not count as one reasoning token per sample.

### Priority 3: Later improvements

- Curriculum for direct/reasoning ratio.
- Optional `generation_stop_at_point_end` if max-token stopping is insufficient.
- Bad reasoning sample filtering.
- Object phrase canonicalization table.
- Point soft-label CE or distance-aware auxiliary loss.
- Full scene + head crop multi-image input.

## 14. Expected Behavior

Train:

```text
Most sampled views:
Prompt: direct
Target: Object -> Point

Some sampled views:
Prompt: reasoning
Target: Reasoning -> Object -> Point
```

Default sampling:

```text
Direct view: 80%
Reasoning view: 20%
```

Val/Test:

```text
Prompt: direct
Generated:
Object: <obj_k>
Point: <loc_x><loc_y>
```

No reasoning is generated or required for the main val/test metric.

## 15. Claude Review Checklist

- Does `use_reasoning=false` produce direct-only samples with no reasoning scaffold?
- Does multi-view training create both views for valid reasoning train records?
- Are direct and reasoning prompts truly different?
- Does val/test always use direct prompt and strict `Object -> Point` parsing?
- Is the sampler actually producing the configured direct/reasoning ratio?
- Are reasoning views protected from hflip/crop?
- Are reasoning strings normalized and length-limited?
- Is canonical object token always used as supervised object target?
- Is reasoning file `Object:` used only for diagnostics?
- Are view-specific losses logged separately?
- Does `train/n_reasoning_tokens` reflect real reasoning text, not whitespace-only tokens?
- Is `generation_stop_at_object_end=false` for `Object -> Point`?
- Is main eval `generation_max_new_tokens` short enough for direct output?
