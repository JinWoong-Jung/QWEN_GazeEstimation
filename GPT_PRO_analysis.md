맞습니다. 자원이 제한적이면 **ablation을 늘리는 것보다 현재 best 설정에 “slot-constrained decoding”만 붙이는 게 가장 효율적**입니다.
목표는 모델 구조와 학습 loss는 유지하고, **validation/test generation 경로만 바꾸는 것**입니다.

아래는 그대로 Codex/Claude Code에 넘길 수 있는 수정 가이드라인입니다.

---

# 목표

현재 `run_test_metrics()`는 `model.generate()`로 전체 답변을 자유 생성한 뒤 text parser로 좌표를 읽습니다. 현재 eval 경로는 `model.generate()` → `decode_generated()` → `parse_structured_output_text()` 흐름입니다. ([GitHub][1])

이를 다음 방식으로 바꿉니다.

```text
Before:
prompt → model.generate() → 자유 text 생성 → parser

After:
prompt
→ 구조 token 강제 append
→ object slot에서는 <obj_*> 후보 중 softmax/argmax
→ point slot에서는 <loc_*> 후보 중 softmax/argmax
→ structured text 조립
→ 기존 parser로 평가
```

중요한 점은 **새 head를 추가하지 않는다**는 것입니다.

```text
사용하는 것:
- 기존 Qwen LM head logits
- 기존 <loc_*> special tokens
- 기존 <obj_*> special tokens
- 기존 parser/evaluator

추가하지 않는 것:
- coordinate regression head
- heatmap decoder
- 별도 localization module
```

---

# 1. 수정 파일

우선순위상 이 파일 하나만 수정하면 됩니다.

```text
model/utils/eval_utils.py
```

가능하면 `gaze_tokens.py`의 기존 helper를 import해서 token 문자열을 만들도록 합니다. `gaze_tokens.py`에는 `GAZE_POINT_MARKER`, `GAZE_OBJECT_MARKER`, `format_loc_token`, `format_obj_token`, `parse_structured_output_text` 등이 이미 존재합니다. 특히 parser는 object→point, point→object, reasoning 포함 schema를 모두 받도록 되어 있습니다. ([GitHub][2])

---

# 2. import 추가

`eval_utils.py` 상단 import를 아래처럼 확장합니다.

```python
from .gaze_tokens import (
    ANSWER_END,
    GAZE_OBJECT_MARKER,
    GAZE_POINT_MARKER,
    format_loc_token,
    format_obj_token,
    parse_structured_output_text,
)
```

현재는 `ANSWER_END`, `parse_structured_output_text`만 import하고 있을 가능성이 큽니다. ([GitHub][1])

---

# 3. token id helper 추가

`eval_utils.py`의 `Decode helpers` 근처에 아래 helper들을 추가합니다.

```python
def _single_token_id(tokenizer: Any, token: str) -> int:
    ids = tokenizer.encode(str(token), add_special_tokens=False)
    if len(ids) != 1:
        raise ValueError(
            f"Expected single-token special token, got {token!r} -> {ids}"
        )
    return int(ids[0])


def _loc_token_ids(tokenizer: Any, coord_bins: int) -> list[int]:
    width = max(3, len(str(max(0, int(coord_bins) - 1))))
    return [
        _single_token_id(tokenizer, format_loc_token(i, width))
        for i in range(int(coord_bins))
    ]


def _obj_token_ids(tokenizer: Any, num_classes: int) -> list[int]:
    width = max(3, len(str(max(0, int(num_classes) - 1))))
    return [
        _single_token_id(tokenizer, format_obj_token(i, width))
        for i in range(int(num_classes))
    ]


def _marker_id(tokenizer: Any, marker: str) -> int:
    return _single_token_id(tokenizer, marker)


def _append_token_to_joint(joint: dict[str, Any], token_ids: torch.LongTensor) -> dict[str, Any]:
    """
    Append one generated text token to joint_inputs.
    Keeps image-related tensors unchanged.
    """
    out = dict(joint)

    token_ids = token_ids.to(device=joint["input_ids"].device, dtype=torch.long)
    if token_ids.dim() == 1:
        token_ids = token_ids[:, None]

    out["input_ids"] = torch.cat([joint["input_ids"], token_ids], dim=1)

    if "attention_mask" in joint and torch.is_tensor(joint["attention_mask"]):
        new_mask = torch.ones_like(token_ids, dtype=joint["attention_mask"].dtype)
        out["attention_mask"] = torch.cat([joint["attention_mask"], new_mask], dim=1)

    return out
```

이 helper의 핵심은 `input_ids`와 `attention_mask`만 늘리고, image tensor는 그대로 유지하는 것입니다.

---

# 4. constrained next-token 선택 함수 추가

같은 파일에 아래 함수를 추가합니다.

```python
def _select_next_from_allowed(
    model: torch.nn.Module,
    joint: dict[str, Any],
    allowed_token_ids: list[int],
    amp_dtype: torch.dtype,
    temperature: float = 1.0,
) -> tuple[torch.LongTensor, torch.Tensor]:
    """
    Select next token from an allowed token set using Qwen LM logits.

    Returns:
        selected_token_ids: [B]
        allowed_probs: [B, K]
    """
    device = joint["input_ids"].device
    allowed = torch.tensor(allowed_token_ids, device=device, dtype=torch.long)

    with torch.autocast(
        device_type=device.type,
        dtype=amp_dtype,
        enabled=(device.type == "cuda"),
    ):
        out = model(joint_inputs=joint, use_cache=False)
        logits = out["logits"][:, -1, :]  # [B, vocab]

    allowed_logits = logits.index_select(dim=-1, index=allowed)  # [B, K]

    temp = max(float(temperature), 1e-6)
    allowed_probs = torch.softmax(allowed_logits / temp, dim=-1)

    selected_idx = torch.argmax(allowed_probs, dim=-1)  # [B]
    selected_token_ids = allowed.index_select(dim=0, index=selected_idx)  # [B]

    return selected_token_ids, allowed_probs
```

이 함수가 “softmax 방식”입니다.
즉, 전체 vocab이 아니라 허용된 token 후보에 대해서만 softmax를 계산합니다.

```text
point slot:
softmax(logits[<loc_000> ... <loc_127>])

object slot:
softmax(logits[<obj_000> ... <obj_345>])
```

출력 선택은 우선 `argmax`로 두는 것을 권장합니다. sampling은 localization metric을 흔들기 때문에 쓰지 않는 편이 낫습니다.

---

# 5. structured constrained generation 함수 추가

현재 target order가 `object_point` 또는 `point_object`일 수 있으므로 둘 다 지원하게 만듭니다. `gaze_tokens.py`의 default target order는 object→point 흐름을 지원하고, parser도 object→point와 point→object를 모두 처리합니다. ([GitHub][2])

```python
def constrained_generate_structured(
    model: torch.nn.Module,
    joint: dict[str, Any],
    processor: Any,
    num_classes: int,
    coord_bins: int,
    amp_dtype: torch.dtype,
    target_order: str = "object_point",
    temperature: float = 1.0,
) -> list[str]:
    """
    Generate structured gaze output with slot-level constrained decoding.

    No new head is used.
    The model still generates Qwen text tokens through LM logits.
    """
    tokenizer = getattr(processor, "tokenizer", None) or processor
    device = joint["input_ids"].device
    bsz = int(joint["input_ids"].shape[0])

    loc_ids = _loc_token_ids(tokenizer, coord_bins=int(coord_bins))
    obj_ids = _obj_token_ids(tokenizer, num_classes=int(num_classes))

    point_marker_id = _marker_id(tokenizer, GAZE_POINT_MARKER)
    object_marker_id = _marker_id(tokenizer, GAZE_OBJECT_MARKER)

    point_marker = torch.full(
        (bsz,), point_marker_id, device=device, dtype=torch.long
    )
    object_marker = torch.full(
        (bsz,), object_marker_id, device=device, dtype=torch.long
    )

    cur = dict(joint)
    generated_steps: list[torch.LongTensor] = []

    order = str(target_order or "object_point").strip()

    if order in {"point_object", "reasoning_point_object"}:
        # <|gaze_point|><loc_x><loc_y><|gaze_object|><obj_k>
        cur = _append_token_to_joint(cur, point_marker)
        generated_steps.append(point_marker)

        x_tok, _ = _select_next_from_allowed(
            model, cur, loc_ids, amp_dtype=amp_dtype, temperature=temperature
        )
        cur = _append_token_to_joint(cur, x_tok)
        generated_steps.append(x_tok)

        y_tok, _ = _select_next_from_allowed(
            model, cur, loc_ids, amp_dtype=amp_dtype, temperature=temperature
        )
        cur = _append_token_to_joint(cur, y_tok)
        generated_steps.append(y_tok)

        cur = _append_token_to_joint(cur, object_marker)
        generated_steps.append(object_marker)

        obj_tok, _ = _select_next_from_allowed(
            model, cur, obj_ids, amp_dtype=amp_dtype, temperature=temperature
        )
        generated_steps.append(obj_tok)

    else:
        # Default: <|gaze_object|><obj_k><|gaze_point|><loc_x><loc_y>
        cur = _append_token_to_joint(cur, object_marker)
        generated_steps.append(object_marker)

        obj_tok, _ = _select_next_from_allowed(
            model, cur, obj_ids, amp_dtype=amp_dtype, temperature=temperature
        )
        cur = _append_token_to_joint(cur, obj_tok)
        generated_steps.append(obj_tok)

        cur = _append_token_to_joint(cur, point_marker)
        generated_steps.append(point_marker)

        x_tok, _ = _select_next_from_allowed(
            model, cur, loc_ids, amp_dtype=amp_dtype, temperature=temperature
        )
        cur = _append_token_to_joint(cur, x_tok)
        generated_steps.append(x_tok)

        y_tok, _ = _select_next_from_allowed(
            model, cur, loc_ids, amp_dtype=amp_dtype, temperature=temperature
        )
        generated_steps.append(y_tok)

    gen_ids = torch.stack(generated_steps, dim=1)  # [B, T]
    texts = tokenizer.batch_decode(gen_ids.detach().cpu(), skip_special_tokens=False)
    return [str(t).strip() for t in texts]
```

이 구현은 구조 token을 강제로 붙입니다.
따라서 `FormatValid`는 거의 항상 1에 가까워져야 합니다. 실제 성능 개선 여부는 `Avg L2`, `Min L2`, `PointL2ValidFrac`로 봐야 합니다.

---

# 6. `run_test_metrics()`에 옵션 추가

함수 signature에 옵션을 추가합니다.

```python
def run_test_metrics(
    model: torch.nn.Module,
    loader: DataLoader,
    device: torch.device,
    amp_dtype: torch.dtype,
    processor: Any,
    num_classes: int,
    coord_bins: int = 1000,
    show_tqdm: bool = True,
    desc: str = "Test",
    max_new_tokens: int = 8,
    num_beams: int = 1,
    repetition_penalty: float = 1.0,
    no_repeat_ngram_size: int = 0,
    stop_at_object_end: bool = True,
    constrained_decoding: bool = False,
    constrained_target_order: str = "object_point",
    constrained_temperature: float = 1.0,
) -> dict[str, float]:
```

그리고 기존 `model.generate()` 부분을 아래처럼 분기합니다.

현재 `run_test_metrics()`는 내부에서 `generate_kwargs`를 만들고 `model.generate(joint_inputs=joint, **generate_kwargs)`를 호출합니다. ([GitHub][1])
그 부분을 다음 구조로 바꾸면 됩니다.

```python
if bool(constrained_decoding):
    preds = constrained_generate_structured(
        model=model,
        joint=joint,
        processor=processor,
        num_classes=int(num_classes),
        coord_bins=int(coord_bins),
        amp_dtype=amp_dtype,
        target_order=str(constrained_target_order),
        temperature=float(constrained_temperature),
    )
else:
    prompt_len = int(joint["input_ids"].shape[1])
    stopping = make_gaze_obj_end_stopping_criteria(
        processor,
        prompt_len,
        stop_at_object_end=bool(stop_at_object_end),
    )

    with torch.autocast(
        device_type=device.type,
        dtype=amp_dtype,
        enabled=(device.type == "cuda"),
    ):
        generate_kwargs: dict[str, Any] = dict(
            max_new_tokens=max_new_tokens,
            do_sample=False,
            num_beams=beam_k,
            repetition_penalty=max(float(repetition_penalty), 1.0),
            no_repeat_ngram_size=max(0, int(no_repeat_ngram_size)),
        )
        if stopping is not None:
            generate_kwargs["stopping_criteria"] = stopping

        generated_ids = model.generate(joint_inputs=joint, **generate_kwargs)

    generated_ids_cpu = generated_ids.detach().cpu()
    input_ids_cpu = joint["input_ids"].detach().cpu()
    attn_cpu = joint.get("attention_mask", None)
    if torch.is_tensor(attn_cpu):
        attn_cpu = attn_cpu.detach().cpu()

    preds = decode_generated(
        processor=processor,
        generated_ids=generated_ids_cpu,
        input_ids=input_ids_cpu,
        attention_mask=attn_cpu,
        num_return_sequences=1,
    )
```

그 아래의 metric 계산 코드는 그대로 둡니다.

```python
parsed = parse_structured_output_text(pred, int(num_classes), coord_bins=int(coord_bins))
```

기존 parser를 그대로 쓰므로 metric 계산부를 건드릴 필요가 없습니다.

---

# 7. `collect_generation_samples()`도 같은 방식으로 수정

`collect_generation_samples()`도 preview에서 기존 `model.generate()`를 사용합니다. `run_test_metrics()`만 바꾸면 validation metric은 바뀌지만 preview는 여전히 unconstrained 결과를 보여주게 됩니다. 혼동을 막으려면 여기도 같은 옵션을 추가하세요.

함수 signature에 추가:

```python
constrained_decoding: bool = False,
constrained_target_order: str = "object_point",
constrained_temperature: float = 1.0,
```

그리고 generation 부분을 동일하게 분기합니다.

```python
if bool(constrained_decoding):
    preds = constrained_generate_structured(
        model=model,
        joint=joint,
        processor=processor,
        num_classes=int(num_classes),
        coord_bins=int(coord_bins),
        amp_dtype=amp_dtype,
        target_order=str(constrained_target_order),
        temperature=float(constrained_temperature),
    )
else:
    # 기존 model.generate 경로 유지
```

---

# 8. config 옵션 추가

`config.yaml` 또는 `sft.yaml`의 eval/test 쪽에 아래 옵션을 추가합니다.

```yaml
eval:
  constrained_decoding: true
  constrained_target_order: object_point
  constrained_temperature: 1.0
```

현재 target text가 point-first라면:

```yaml
eval:
  constrained_decoding: true
  constrained_target_order: point_object
  constrained_temperature: 1.0
```

단, 최근 코드상 `build_structured_target_text()`의 기본값은 `object_point`입니다. `object_point`는 다음 형태입니다. ([GitHub][2])

```text
<|gaze_object|><obj_k><|gaze_point|><loc_x><loc_y>
```

학습 target과 eval constrained order가 반드시 같아야 합니다.

---

# 9. `main.py` 또는 호출부 연결

`run_test_metrics()`를 호출하는 부분에서 config 값을 넘겨야 합니다.

예시:

```python
metrics = run_test_metrics(
    model=model,
    loader=val_loader,
    device=device,
    amp_dtype=amp_dtype,
    processor=processor,
    num_classes=int(cfg["data"]["num_classes"]),
    coord_bins=int(cfg["model"]["coord_bins"]),
    max_new_tokens=int(cfg["eval"].get("generation_max_new_tokens", 8)),
    constrained_decoding=bool(cfg["eval"].get("constrained_decoding", False)),
    constrained_target_order=str(cfg["eval"].get("constrained_target_order", "object_point")),
    constrained_temperature=float(cfg["eval"].get("constrained_temperature", 1.0)),
)
```

preview 호출부도 동일하게 넘기세요.

---

# 10. 검증 체크리스트

수정 후 바로 확인해야 할 것은 아래입니다.

## 1차 확인: token이 single token인지

실행 초기에 아래 에러가 나면 special token 등록이 제대로 안 된 것입니다.

```text
Expected single-token special token
```

이 경우 tokenizer에 `<loc_*>`, `<obj_*>`, `<|gaze_point|>`, `<|gaze_object|>`가 추가되고 `resize_token_embeddings()`가 호출되는지 확인해야 합니다.

## 2차 확인: preview 출력

preview가 아래처럼 나와야 합니다.

```text
<|gaze_object|><obj_012><|gaze_point|><loc_057><loc_083>
```

또는 point-first 설정이면:

```text
<|gaze_point|><loc_057><loc_083><|gaze_object|><obj_012>
```

절대 아래처럼 나오면 안 됩니다.

```text
Point: ...
Object: ...
the person is looking at ...
```

## 3차 확인: metric 변화

기대되는 변화는 다음입니다.

```text
FormatValid: 상승 또는 1.0 근처
ExtraTextRate: 0.0 근처
PointL2ValidFrac: 1.0 근처
Avg L2: 실제 개선 여부 확인
ObjectAcc: object slot 제한으로 상승 가능
```

단, `Avg L2`가 크게 좋아지지 않을 수도 있습니다. 이 경우 원인은 decoding format이 아니라 model이 loc distribution 자체를 아직 잘 못 배운 것입니다. 그래도 constrained decoding을 적용하면 적어도 “format failure 때문에 L2가 손해 보는 문제”는 제거됩니다.

---

# 11. 가장 중요한 주의점

## `target_order`를 틀리면 성능이 망가집니다

학습 target이:

```text
<|gaze_object|><obj_k><|gaze_point|><loc_x><loc_y>
```

인데 eval을 point-first로 하면, 모델은 `<|gaze_point|>` 뒤에서 loc token을 잘 못 낼 수 있습니다. 반대로도 마찬가지입니다.

따라서 먼저 학습 target sample을 하나 출력해서 확인하세요.

```python
print(batch["target_text"][0])
```

그리고 config를 맞춥니다.

```yaml
constrained_target_order: object_point
```

또는:

```yaml
constrained_target_order: point_object
```

---

# 12. 자원 제한 상황에서의 최종 권장 적용안

다른 ablation 없이 바로 적용한다면 이렇게 하세요.

```yaml
eval:
  constrained_decoding: true
  constrained_target_order: object_point  # 반드시 현재 target_text와 맞출 것
  constrained_temperature: 1.0
```

그리고 학습 설정은 당장 건드리지 않습니다.

```text
reasoning 유지
gaussian CE 유지
object loss 유지
LoRA 설정 유지
```

수정 목적은 하나입니다.

```text
학습된 Qwen LM logits를 더 정확히 평가에 사용한다.
```

즉, 이번 수정은 성능을 “새로 학습해서” 올리는 방식이 아니라, **이미 학습한 token distribution을 더 엄격하게 읽어내는 evaluation/inference 개선**입니다.

[1]: https://raw.githubusercontent.com/JinWoong-Jung/QWEN_GazeEstimation/main/model/utils/eval_utils.py "raw.githubusercontent.com"
[2]: https://raw.githubusercontent.com/JinWoong-Jung/QWEN_GazeEstimation/main/model/utils/gaze_tokens.py "raw.githubusercontent.com"
