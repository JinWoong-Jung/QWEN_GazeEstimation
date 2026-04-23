# Qwen3-VL 기반 시선 추정/대상 인식 파이프라인 정리

## 1. 이번 구현에서 중요한 설계 포인트

- 핵심은 단순히 gaze estimation을 하는 것이 아니라,
  **입력과 출력을 완전히 구조화해서 Qwen3-VL이 일관된 형식으로 point와 object를 함께 생성하도록 만든 것**입니다.
- 전체 학습은 두 단계입니다.
  - Stage 1: 구조와 정답 형식을 안정적으로 학습하는 SFT
  - Stage 2: 실제 생성 품질을 직접 개선하는 RL

## 2. 입력 포맷

### 입력 구성

- 입력은 `장면 이미지 + head box 정보 + 지시 프롬프트`입니다.
- head box는 두 방식으로 반영됩니다.
  - 텍스트 프롬프트 안에 위치 정보를 넣음
  - 필요 시 이미지 위에 head box를 시각적으로 그려서 넣음

### 텍스트 프롬프트 형식

- 프롬프트에는 머리 박스를 정규화된 좌표 토큰으로 넣습니다.
- 예시:

```text
The head box [<loc_x1><loc_y1><loc_x2><loc_y2>] marks the person.
Predict the gaze point that this person is looking at, and the object located at that gaze point.
...
```

- 즉, 입력 단계부터 bbox를 자연어 설명이 아니라 **모델이 직접 다룰 수 있는 구조화 토큰**으로 제공합니다.

## 3. 출력 포맷

- 출력은 자유문장이 아니라 반드시 아래 두 줄 형식을 따르도록 설계했습니다.

```text
<|im_start|>Point: <loc_572><loc_563>
Object: <obj_059><|im_end|>
```

- `Point`
  - x, y를 각각 0~999 구간으로 양자화한 `<loc_###>` 토큰 2개로 표현
- `Object`
  - 시선 대상 클래스를 `<obj_###>` 토큰으로 표현
- 답변 래퍼는 Qwen의 기존 스페셜 토큰 `<|im_start|>`, `<|im_end|>`를 그대로 사용

### 이 포맷의 의미

- 회귀값을 직접 내는 대신 **좌표를 토큰화**해서 생성 문제로 바꿨습니다.
- point, object, format을 모두 토큰 단위로 관리할 수 있어
  학습, 파싱, 평가를 하나의 체계로 통일할 수 있습니다.

## 4. 전체 학습 파이프라인

1. 데이터에서 `scene image`, `head bbox`, `gaze point`, `object label`을 읽습니다.
2. bbox를 프롬프트 토큰으로 변환하고, 필요하면 이미지에도 시각적으로 표시합니다.
3. 정답은 `Point + Object` 구조화 텍스트로 만듭니다.
4. Qwen3-VL 입력 형식에 맞게 `이미지 + 사용자 프롬프트 + 정답 답변` 형태로 묶습니다.
5. Stage 1에서는 구조화 답변을 teacher forcing으로 학습합니다.
6. Stage 2에서는 Stage 1 체크포인트를 초기 policy로 두고, 샘플링 기반 RL로 실제 생성 품질을 미세 조정합니다.

## 5. Stage 1: SFT

### 목적

- 모델이 먼저 **형식을 안정적으로 지키면서**
  point와 object를 함께 생성하는 기본 능력을 갖추게 하는 단계입니다.

### 학습 방식

- 정답 답변 전체를 한 번에 학습하지 않고,
  답변 span을 세 부분으로 나눠 loss를 부여합니다.
  - point 토큰
  - object 토큰
  - format 토큰

### SFT Loss

```text
L_SFT = w_point * L_point + w_object * L_object + w_format * L_format
```

- 현재 기본 가중치:
  - `w_point = 1.0`
  - `w_object = 1.0`
  - `w_format = 0.25`

### 왜 이렇게 나눴는가

- `Point`와 `Object`는 실제 과업 성능에 직접 대응합니다.
- `Format`은 중요하지만 본질적인 과업 목표는 아니므로 더 작은 가중치를 줬습니다.
- 이렇게 하면 모델이 형식을 지키되, point/object 예측 성능에 더 집중하게 됩니다.

## 6. Stage 2: RL

### 목적

- SFT는 teacher forcing 기반이기 때문에,
  실제 generation 상황에서는 형식이 깨지거나 불필요한 텍스트가 붙을 수 있습니다.
- 그래서 Stage 2에서는
  **실제로 생성된 답변 자체를 평가해서 보상하는 방식**으로 후처리 학습을 구성했습니다.

### 기본 구조

- 초기 policy: Stage 1에서 학습한 best checkpoint
- reference model: 같은 SFT checkpoint를 동결한 모델
- 알고리즘: `GRPO`

### RL 한 step의 흐름

1. 한 입력에 대해 여러 개의 답변을 샘플링합니다.
2. 각 답변을 파싱해서 point, object, format이 유효한지 확인합니다.
3. 각 답변에 대해 reward를 계산합니다.
4. 같은 입력에서 나온 여러 답변 사이의 상대적 advantage를 계산합니다.
5. policy를 업데이트하되, reference model에서 너무 멀어지지 않도록 KL penalty를 둡니다.

## 7. RL Reward 설계

### 기본 원칙

- Stage 2는 “토큰을 맞추는 학습”이 아니라
  **최종 출력이 실제로 좋은 답변인지**를 직접 보상합니다.

### Reward 구성

1. 형식 검사
   - 형식이 틀리면 바로 `-1.0`
   - 즉, format은 hard gate 역할
2. point reward
   - GT point와의 최소 거리 기반
   - `r_point = exp(-beta * min_L2)`
3. object reward
   - 예측 object가 GT object와 일치하면 `1.0`, 아니면 `0.0`
4. joint bonus
   - point도 가깝고 object도 맞으면 추가 보상
5. extra text penalty
   - 정해진 형식 외 텍스트가 붙으면 패널티

### 최종 reward

```text
if invalid_format:
    r_total = -1.0
else:
    r_total =
        1.0 * r_point
      + 0.75 * r_object
      + 0.25 * r_joint
      - 0.5 * r_extra
```

### 의미

- point를 가장 중요한 목표로 두고,
- object를 두 번째 목표로 두며,
- 둘을 동시에 맞췄을 때 추가로 보상하고,
- 형식 붕괴와 불필요한 생성은 강하게 억제하는 구조입니다.

## 8. RL Objective

- RL은 PPO 계열의 `GRPO`를 사용했습니다.
- 입력 하나에 대해 여러 rollout을 만든 뒤, 그룹 내 상대 비교로 advantage를 계산합니다.

```text
A_i = (r_i - mean(group)) / (std(group) + eps)
```

- 그리고 policy는 clipped objective와 KL penalty를 함께 사용해 업데이트합니다.

```text
L = policy_loss + beta * KL(policy || ref)
```

- 현재 핵심 설정:
  - `group size = 4`
  - `clip eps = 0.2`
  - `KL beta = 0.02`
  - `temperature = 0.7`
  - `top-p = 0.9`

## 9. Stage 1과 Stage 2의 역할 분담

### Stage 1이 하는 일

- 출력 문법을 배우게 함
- `<loc_###>`, `<obj_###>` 토큰 사용법을 익히게 함
- point와 object를 함께 생성하는 기본 능력을 만듦

### Stage 2가 하는 일

- 실제 생성 결과를 더 task-oriented하게 미세 조정
- point 거리 개선
- object exactness 개선
- malformed output, extra text 억제

### 즉, 두 단계를 나눈 이유

- Stage 1 없이 바로 RL을 하면 형식 자체가 쉽게 무너질 수 있습니다.
- 그래서 먼저 SFT로 구조를 잡고,
  그 위에서 RL로 실제 생성 품질을 다듬는 2단계 구조를 채택했습니다.

## 10. 현재 구현 관점에서의 의미

- 이 파이프라인은 gaze task를 단순 좌표 회귀로 다루지 않고,
  **멀티모달 생성 문제로 재정의한 구현**입니다.
- 특히 중요한 점은
  - 입력 포맷을 구조화했고
  - 출력 포맷을 강하게 통제했으며
  - SFT와 RL의 역할을 분리했고
  - point / object / format을 서로 다른 신호로 학습시켰다는 점입니다.

## 11. 한 줄 정리

- 이번 구현은 **Qwen3-VL에 구조화된 입력/출력 포맷을 정의하고,
  Stage 1 SFT로 형식을 안정화한 뒤, Stage 2 RL로 실제 생성 품질을 개선하는 gaze 파이프라인**이라고 설명할 수 있습니다.
