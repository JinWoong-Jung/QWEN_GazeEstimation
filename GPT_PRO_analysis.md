# 수정 TODO: Option 1 기반 Multi-View SFT 구조

## 0. 목표

현재 Stage-1 SFT를 다음 구조로 수정한다.

```text
For every train sample:
  View A: Direct View
    Prompt: object & point만 예측
    Target: Object → Point

  View B: Reasoning View
    Prompt: reasoning 후 object & point 예측
    Target: Reasoning → Object → Point

Val/Test:
  항상 Direct View만 사용
  Target/Eval format: Object → Point
```

핵심 목적은 다음이다.

1. **모든 train reasoning annotation을 보존한다.**
2. reasoning을 버리지 않고, 모든 sample에 대해 reasoning view를 만든다.
3. 다만 학습 sampling 비율은 `Direct > Reasoning`으로 둔다.
4. 최종 metric인 `val/dist`는 항상 direct prompt에서 평가한다.
5. reasoning은 inference-time requirement가 아니라 **train-time auxiliary task**로 쓴다.

---

# 1. Dataset 구조 수정

## TODO 1.1 Train sample을 두 개의 view로 확장

현재 train record 하나가 다음 정보를 가진다고 가정한다.

```text
image_path
sample_id
head_bbox
gaze_point
object_label_id / object_token
reasoning_text
```

이를 dataset construction 단계에서 다음 두 view로 확장한다.

```python
expanded_train_samples = []

for record in train_records:
    expanded_train_samples.append({
        "base_record": record,
        "view_type": "direct",
        "prompt_type": "direct_object_point",
        "target_type": "object_point",
        "use_reasoning": False,
        "augmentation_policy": "full",
    })

    expanded_train_samples.append({
        "base_record": record,
        "view_type": "reasoning",
        "prompt_type": "reasoning_object_point",
        "target_type": "reasoning_object_point",
        "use_reasoning": True,
        "augmentation_policy": "safe",
    })
```

### 중요

* train reasoning coverage가 100%이므로 모든 record에 reasoning view를 만든다.
* reasoning을 일부만 쓰는 것이 아니다.
* sampling 단계에서 direct view를 더 자주 뽑을 뿐이다.

---

## TODO 1.2 Val/Test dataset은 direct view만 생성

val/test에서는 reasoning view를 절대 만들지 않는다.

```python
val_samples = []

for record in val_records:
    val_samples.append({
        "base_record": record,
        "view_type": "direct",
        "prompt_type": "direct_object_point",
        "target_type": "object_point",
        "use_reasoning": False,
        "augmentation_policy": "none",
    })
```

test도 동일하다.

---

# 2. Sampling 비율 설정

## TODO 2.1 Weighted sampler 추가

expanded train dataset은 `direct N개 + reasoning N개 = 2N개`가 된다.
그 상태에서 uniform sampling을 하면 direct:reasoning이 1:1이 된다. 이는 권장하지 않는다.

`WeightedRandomSampler` 또는 custom sampler를 사용해서 direct view가 더 자주 뽑히게 한다.

추천 초기값:

```yaml
train:
  direct_view_ratio: 0.8
  reasoning_view_ratio: 0.2
```

즉 학습 step 기준으로:

```text
Direct Object→Point: 80%
Reasoning→Object→Point: 20%
```

### 구현 예시

```python
weights = []

for sample in expanded_train_samples:
    if sample["view_type"] == "direct":
        weights.append(direct_weight)
    elif sample["view_type"] == "reasoning":
        weights.append(reasoning_weight)
```

전체 view 수가 direct N, reasoning N으로 같다면 단순히 다음처럼 둘 수 있다.

```python
direct_weight = 0.8
reasoning_weight = 0.2
```

또는 더 엄밀히 normalize하려면:

```python
direct_weight = desired_direct_ratio / num_direct_views
reasoning_weight = desired_reasoning_ratio / num_reasoning_views
```

그리고:

```python
sampler = WeightedRandomSampler(
    weights=weights,
    num_samples=len(train_records),  # 또는 한 epoch당 원하는 step 수 기준
    replacement=True,
)
```

### 권장 실험값

```text
Ablation 1: direct:reasoning = 9:1
Ablation 2: direct:reasoning = 8:2
Ablation 3: direct:reasoning = 7:3
```

초기 default는 `8:2`.

---

# 3. Prompt 설계

## TODO 3.1 Direct prompt와 reasoning prompt를 명확히 분리

같은 prompt에서 어떤 경우는 reasoning을 내고, 어떤 경우는 reasoning을 안 내면 모델이 혼동한다.
따라서 prompt에 task identity를 명확히 넣는다.

### Direct prompt

```text
You are given an image with a marked person.
Predict the gaze target object and gaze point of the marked person.

Return only:
Object: <object_token>
Point: <x_token><y_token>
```

### Reasoning prompt

```text
You are given an image with a marked person.
First provide one short visual reasoning sentence about where the person is looking.
Then predict the gaze target object and gaze point.

Return exactly:
Reasoning: <one short sentence>
Object: <object_token>
Point: <x_token><y_token>
```

### 주의

* direct prompt에는 “reasoning”, “explain”, “why” 같은 단어를 넣지 않는다.
* reasoning prompt에는 “one short sentence”를 명시한다.
* reasoning을 장문으로 만들지 않는다.
* 최종 answer block은 항상 `Object → Point` 형식을 유지한다.

---

# 4. Target 생성 수정

## TODO 4.1 Direct target builder

direct view target은 다음만 포함한다.

```text
Object: <obj_255>
Point: <loc_042><loc_087>
```

예시 함수:

```python
def build_direct_target(object_token: str, x_token: str, y_token: str) -> str:
    return f"Object: {object_token}\nPoint: {x_token}{y_token}"
```

---

## TODO 4.2 Reasoning target builder

reasoning view target은 다음 순서를 사용한다.

```text
Reasoning: ...
Object: <obj_255>
Point: <loc_042><loc_087>
```

예시 함수:

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

---

## TODO 4.3 Reasoning 문장 길이 제한

reasoning은 가능한 짧게 제한한다.

추천:

```yaml
reasoning:
  max_reasoning_words: 30
  max_reasoning_chars: 220
```

예시 reasoning은 “head is angled toward the monitor”처럼 시선 근거를 설명하지만, 일부는 방향·공간 표현이 들어간다. 예를 들어 “upward and to the right”, “downward and outward” 같은 방향 표현이 포함되어 있다.  

따라서 긴 문장을 그대로 두기보다 normalize한다.

```python
def normalize_reasoning_text(text: str) -> str:
    text = text.strip()
    text = text.replace("\n", " ")
    text = collapse_spaces(text)
    text = truncate_to_max_words(text, max_words=30)
    if not text.endswith("."):
        text += "."
    return text
```

---

# 5. Object label 처리

## TODO 5.1 Object target은 반드시 canonical object token 사용

reasoning txt 안의 `Object:` line을 최종 object target으로 쓰지 않는다.

이유:

* reasoning object는 free-form phrase다.
* 현재 object vocabulary는 346개 closed-set class다. 예를 들어 `screen`, `cake`, `plate` 등이 canonical class로 정의되어 있다. 
* reasoning 예시의 object phrase는 “the dessert on the plate”, “the red-and-white circular print on the table”, “the computer monitor screen”처럼 fine-grained description이다.   

따라서 supervised object token은 기존 label pipeline에서 나온 `<obj_k>`를 사용한다.

```python
object_token = object_id_to_special_token(label_id)
```

---

## TODO 5.2 Reasoning file의 Object line은 consistency check에만 사용

reasoning file parser가 다음을 모두 읽게 한다.

```python
{
    "object_text": "...",
    "reasoning_text": "...",
}
```

하지만 `object_text`는 target token으로 직접 쓰지 않는다.

사용 목적:

1. debugging
2. object label consistency check
3. bad reasoning sample filtering
4. canonicalization table 구축 보조

로그 예시:

```text
reasoning/object_text: the computer monitor screen
canonical_label: screen
object_token: <obj_255>
match: true
```

---

# 6. Augmentation 정책 수정

## TODO 6.1 Direct view는 full augmentation 사용

direct view는 기존 baseline과 동일한 강한 spatial augmentation을 유지한다.

```text
Direct view augmentation:
  - random crop
  - horizontal flip
  - color jitter
  - resize
```

이 경로가 최종 val/test와 직접 연결되므로, spatial generalization을 최대한 유지한다.

---

## TODO 6.2 Reasoning view는 safe augmentation 사용

reasoning text는 원본 이미지 기준으로 생성되어 있다.
특히 left/right/up/down 같은 방향 표현이 들어갈 수 있으므로 hflip/crop과 충돌할 수 있다.

```text
Reasoning view augmentation:
  - resize
  - color jitter
  - no horizontal flip
  - no random crop, 또는 매우 제한적 crop
```

권장:

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

---

# 7. Loss mask 수정

## TODO 7.1 Token category mask가 view별로 정확히 생성되어야 함

각 target token에 대해 다음 mask를 만든다.

```text
format tokens
reasoning tokens
object tokens
point tokens
other ignored tokens
```

### Direct view

```text
Object: <obj_255>
Point: <loc_042><loc_087>
```

loss category:

```text
Object:, Point:     -> format loss
<obj_255>           -> object loss
<loc_042><loc_087>  -> point loss
```

reasoning loss는 없음.

---

### Reasoning view

```text
Reasoning: ...
Object: <obj_255>
Point: <loc_042><loc_087>
```

loss category:

```text
Reasoning:          -> format loss
reasoning sentence  -> reasoning loss
Object:, Point:     -> format loss
<obj_255>           -> object loss
<loc_042><loc_087>  -> point loss
```

---

## TODO 7.2 추천 loss weight

현재 목표가 `val/dist`이므로 point loss를 가장 강하게 둔다.

추천 초기값:

```yaml
loss:
  loss_point_weight: 3.0
  loss_object_weight: 1.0
  loss_format_weight: 0.2
  loss_reasoning_weight: 0.05
```

reasoning loss는 낮게 둔다.

금지에 가까운 설정:

```yaml
loss_reasoning_weight: 0.5
```

이 값은 현재 목적에서는 너무 크다. reasoning text generation이 point/object 학습과 경쟁할 가능성이 높다.

---

# 8. Evaluation 구조 수정

## TODO 8.1 Val/Test metric은 direct prompt만 사용

val/test에서 reasoning prompt를 사용하지 않는다.

```python
eval_prompt_type = "direct_object_point"
eval_target_type = "object_point"
```

metric 계산용 generation은 항상 다음 형식만 기대한다.

```text
Object: <obj_k>
Point: <loc_x><loc_y>
```

---

## TODO 8.2 Eval generation 길이 줄이기

direct output은 매우 짧다.

추천:

```yaml
eval:
  generation_max_new_tokens: 16
```

혹시 object token + point token + format token이 더 필요하면:

```yaml
eval:
  generation_max_new_tokens: 24
```

reasoning run에서 쓰던 `128`은 direct eval에는 불필요하다.

---

## TODO 8.3 Stop rule 확인

출력 순서가 `Object → Point`라면 `Object` 직후 generation을 멈추면 안 된다.

따라서 기존 config나 코드에 다음과 같은 설정이 있으면 수정한다.

```yaml
generation_stop_at_object_end: true
```

`Object → Point`에서는 point까지 생성해야 하므로 stop condition은 다음 중 하나여야 한다.

```text
1. Point line이 완성되면 stop
2. EOS token에서 stop
3. max_new_tokens=16/24로 제한
```

추천:

```yaml
eval:
  generation_stop_at_object_end: false
  generation_stop_at_point_end: true
```

만약 `generation_stop_at_point_end`가 없다면 새로 구현한다.

---

# 9. Parser 수정

## TODO 9.1 Direct output parser

다음 형식을 안정적으로 parse해야 한다.

```text
Object: <obj_255>
Point: <loc_042><loc_087>
```

필요한 값:

```python
parsed = {
    "object_id": 255,
    "point_xy": (x, y),
    "format_valid": True,
}
```

---

## TODO 9.2 Reasoning output parser는 train preview/debug용으로만 사용

val/test metric은 direct parser만 써도 된다.

다만 preview에서는 다음도 parse 가능하게 유지한다.

```text
Reasoning: ...
Object: <obj_255>
Point: <loc_042><loc_087>
```

---

## TODO 9.3 Parser는 line order를 명확히 제한

실험 안정성을 위해 val/test parser는 너무 관대하게 만들지 않는다.

direct eval에서는 다음만 valid로 본다.

```text
Object: ...
Point: ...
```

형식이 깨지면 `format_valid=False`.

---

# 10. Config 추가

## TODO 10.1 Multi-view SFT config 추가

예시:

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
  generation_max_new_tokens: 16
  generation_stop_at_object_end: false
  generation_stop_at_point_end: true
  preview_val_samples: 32
```

---

# 11. Logging 추가

## TODO 11.1 View sampling 비율 로그

매 epoch 또는 일정 step마다 실제 batch 내 view 비율을 기록한다.

```text
train/view_direct_frac
train/view_reasoning_frac
```

목표:

```text
direct ≈ 0.8
reasoning ≈ 0.2
```

---

## TODO 11.2 View별 loss 로그

전체 loss만 보면 reasoning이 point를 방해하는지 알 수 없다.

다음을 분리해서 기록한다.

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

---

## TODO 11.3 Eval metric 로그

기존 val/dist 외에 다음을 반드시 기록한다.

```text
val/dist
val/object_acc
val/format_valid
val/point_l2_valid_frac
val/extra_text_rate
```

특히 `val/point_l2_valid_frac`가 낮으면 val/dist가 좋아도 신뢰하기 어렵다.

---

## TODO 11.4 Preview 저장

매 eval마다 direct prompt generation 예시를 저장한다.

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

reasoning preview도 별도로 소량 저장한다. 단, metric 계산에는 쓰지 않는다.

---

# 12. Training schedule 권장

## TODO 12.1 기본 schedule

현재 문제는 reasoning이 val/dist를 방해하는 것이므로 보수적으로 시작한다.

```text
Epoch 1~2:
  direct:reasoning = 9:1
  loss_reasoning_weight = 0.03

Epoch 3 이후:
  direct:reasoning = 8:2
  loss_reasoning_weight = 0.05
```

처음부터 7:3으로 가지 않는다.

---

## TODO 12.2 비교 실험

최소 ablation:

| 실험 | Train 구성                   | Val/Test |
| -- | -------------------------- | -------- |
| A  | Direct only                | Direct   |
| B  | Direct 90% + Reasoning 10% | Direct   |
| C  | Direct 80% + Reasoning 20% | Direct   |
| D  | Direct 70% + Reasoning 30% | Direct   |

우선 B/C까지만 돌려도 된다.

성공 기준:

```text
B 또는 C의 val/dist가 A와 비슷하거나 더 낮아야 한다.
B/C의 val/format_valid와 val/point_l2_valid_frac가 A와 거의 같아야 한다.
```

실패 기준:

```text
val/dist 상승
val/format_valid 하락
val/point_l2_valid_frac 하락
direct generation에 reasoning text가 섞임
```

---

# 13. 코드 수정 우선순위

## Priority 1: 필수

```text
[ ] train dataset을 direct/reasoning 두 view로 확장
[ ] weighted sampler로 direct:reasoning 비율 제어
[ ] val/test dataset은 direct view만 생성
[ ] direct prompt와 reasoning prompt 분리
[ ] direct target과 reasoning target builder 분리
[ ] eval generation max_new_tokens를 16~24로 축소
[ ] Object→Point 출력에서 object 직후 stop하지 않도록 수정
[ ] view별 loss logging 추가
```

---

## Priority 2: 강력 권장

```text
[ ] direct view는 full augmentation 사용
[ ] reasoning view는 safe augmentation 사용
[ ] reasoning text max length 제한
[ ] reasoning Object line을 읽고 consistency check 로그 추가
[ ] preview_val_samples 저장
[ ] val/point_l2_valid_frac, val/format_valid, val/extra_text_rate 기록
```

---

## Priority 3: 후속 개선

```text
[ ] direct/reasoning ratio curriculum 적용
[ ] bad reasoning sample filtering
[ ] object phrase canonicalization table 구축
[ ] point soft-label CE 또는 distance-aware auxiliary loss 추가
[ ] full scene + head crop multi-image input 추가
```

---

# 14. Expected behavior

수정 후 기대되는 동작은 다음이다.

## Train

모델은 두 task를 모두 본다.

```text
Task A:
Prompt: direct
Target: Object → Point

Task B:
Prompt: reasoning
Target: Reasoning → Object → Point
```

하지만 sampling은 direct 중심이다.

```text
Direct view: 80%
Reasoning view: 20%
```

## Val/Test

모델은 항상 direct task로 평가된다.

```text
Prompt: direct
Generated:
Object: <obj_k>
Point: <loc_x><loc_y>
```

val/test에서는 reasoning을 생성하지 않는다.

---

# 15. 핵심 설계 원칙

이 수정의 핵심은 다음 한 문장이다.

```text
Use reasoning as train-time auxiliary supervision, not as the required inference-time output.
```

구체적으로는:

```text
reasoning coverage = 100%
reasoning view existence = 100%
reasoning sampling ratio = 10~30%
eval reasoning usage = 0%
```

즉 reasoning annotation은 전부 사용한다.
다만 `val/dist`가 최우선이므로, 최종 평가 경로와 같은 `Object → Point` direct path를 학습의 중심에 둔다.
