# Stage 2 RL TODO

## Goal

Stage 1 SFT가 이미

- `Point: <loc_xxx><loc_yyy>`
- `Object: <obj_kkk>`
- format 안정화

까지 어느 정도 학습한 상태라고 가정하고,
Stage 2에서는 RL post-training으로 아래를 더 개선한다.

- point bin prediction을 실제 GT point에 더 가깝게 정렬
- object prediction의 exactness 향상
- malformed output / extra text / unstable generation 억제

즉 Rex-Omni식 `GRPO + geometry-aware reward`를 현재 gaze task에 맞게 옮긴다.

---

## Why RL Is Needed

SFT만으로는 다음 한계가 남는다.

1. point는 discrete token CE로 학습되므로, 실제 geometry quality와 token correctness 사이 gap이 남음
2. teacher forcing 기반이라 generation 시 이상 행동(extra text, malformed output)이 남을 수 있음
3. object / point / format을 jointly generation할 때, token-level CE만으로는 최종 task reward를 직접 최적화하지 못함

Stage 2 RL의 목적은:

- `token correctness -> task correctness`
- `teacher-forced behavior -> actual generation behavior`

로 초점을 옮기는 것이다.

---

## Stage 2 Policy Setup

### policy / ref model

- `policy_model`: Stage 1 best checkpoint에서 시작
- `ref_model`: 같은 Stage 1 checkpoint를 frozen reference로 유지

즉 RL은 SFT policy를 미세하게 다듬는 단계다.

### algorithm

- `GRPO`
- critic / value head 별도 추가 없이 group-relative advantage 사용

### initial hyperparameters

- `rl_group_size = 4`
- `rl_clip_eps = 0.2`
- `rl_kl_beta = 0.02`
- `rl_lr = 1e-5`

추천:

- Stage 1 lr가 `1e-4`라면 RL은 `1e-5` 또는 `2e-5`
- 너무 공격적으로 잡지 말 것

---

## Rollout Unit

입력 1개당 현재 policy가 `G=4`개의 답변을 샘플링한다.

각 rollout output 형식:

```text
<|im_start|>Point: <loc_xxx><loc_yyy>
Object: <obj_kkk><|im_end|>
```

### sampling setup

- `temperature = 0.7`
- `top_p = 0.9`
- `num_return_sequences = rl_group_size`
- `do_sample = True`

주의:

- RL에서는 greedy decoding이 아니라 샘플링이 필요함
- 그래야 group 내부에서 상대적으로 좋은 output과 나쁜 output이 나뉨

---

## Reward Design

현재 task에서는 box IoU reward 대신 `point distance reward`를 쓴다.

reward는 아래 4축으로 구성한다.

1. `format`
2. `point geometry`
3. `object correctness`
4. `behavior penalty`

### 1. Format reward

format이 틀리면 RL 전체가 불안정해지므로 hard gate로 둔다.

추천:

```text
if not valid_format:
    r_format = -1.0
else:
    r_format = 0.0
```

즉 invalid format이면 바로 강한 penalty.

이렇게 하면 RL이 Stage 1에서 어렵게 배운 format을 다시 무너뜨리는 것을 방지할 수 있다.

---

### 2. Point reward

현재 task는 point prediction이 핵심이므로 geometry reward의 중심은 point다.

추천 기본형:

```text
r_point = exp(-alpha * min_l2)
```

여기서:

- `min_l2`: GT point set과의 minimum normalized L2 distance
- `alpha = 10` 추천 시작점

설명:

- 예측 point가 GT에 가까울수록 `1`에 가까움
- 멀수록 빠르게 감소
- multi-point annotation에도 자연스럽게 대응 가능

single-point val/train sample에서는 사실상 GT 하나와의 거리다.

대안:

```text
r_point = 1 - clamp(min_l2 / sqrt(2), 0, 1)
```

하지만 초기 버전은 `exp(-alpha * min_l2)`가 더 부드럽고 튜닝이 쉽다.

---

### 3. Object reward

현재 프로젝트 목표가 gaze point + object recognition joint prediction이므로 object reward도 유지한다.

추천:

```text
r_obj = 1.0 if pred_obj in valid_gt_obj_ids else 0.0
```

설명:

- single-label sample에서는 exact match
- multi-label sample에서는 `MultiAcc@1` 기준 허용

즉 ambiguous / overlapping object의 경우에도 현재 test metric 철학과 일관된다.

---

### 4. Joint bonus

point와 object가 동시에 잘 맞는 출력을 더 선호하게 만든다.

추천:

```text
r_joint = 0.25 if (object_correct and min_l2 < 0.1) else 0.0
```

의도:

- point만 대충 맞고 object를 틀리는 경우
- object만 맞고 point가 멀리 가는 경우

보다, 둘 다 맞는 출력을 선호하도록 credit shaping.

---

### 5. Extra text / malformed penalty

현재 generation에서 중요한 failure mode는:

- extra text
- malformed structure
- wrapper mismatch

추천:

```text
r_extra = 0.5 if has_extra_text else 0.0
```

최종 reward에서는 subtraction:

```text
- r_extra
```

---

## Final Reward

초기 버전 추천:

```text
if not valid_format:
    r_total = -1.0
else:
    r_total = (
        1.0 * r_point
        + 0.75 * r_obj
        + 0.25 * r_joint
        - 0.5 * extra_text
    )
```

초기 weight 추천:

- point: `1.0`
- object: `0.75`
- joint bonus: `0.25`
- extra_text penalty: `0.5`

이유:

- point를 최우선 task로 유지
- object는 중요하지만 point보다 약간 낮게
- format은 hard gate로 이미 처리

---

## Advantage / Objective

GRPO의 standard group-relative advantage 사용:

```text
A_i = (r_i - mean(r_group)) / (std(r_group) + eps)
```

그리고 clipped policy objective:

```text
L_grpo = min(ratio * A, clip(ratio, 1-eps, 1+eps) * A) - beta * KL(pi || pi_ref)
```

여기서:

- `eps = rl_clip_eps = 0.2`
- `beta = rl_kl_beta = 0.02`

---

## Minimal Implementation Plan

### Phase 1. sampling-only rollout utility

추가 파일 후보:

- `model/utils/rl_utils.py`

필요 함수:

- `sample_group_responses(...)`
- `compute_point_reward(...)`
- `compute_object_reward(...)`
- `compute_total_reward(...)`
- `group_normalize_rewards(...)`

출력 구조 예시:

```python
[
    {
        "raw_text": "...",
        "parsed": {...},
        "reward_total": 0.73,
        "reward_point": 0.51,
        "reward_object": 1.0,
        "reward_joint": 0.25,
        "reward_extra": 0.0,
    },
    ...
]
```

목표:

- 아직 gradient update 없이 reward만 계산 가능하게 만들기
- rollout preview JSON으로 먼저 sanity check

### Phase 2. ref logprob / policy logprob 계산

필요:

- sampled sequence 전체에 대한 token logprob
- same sampled sequence에 대해 ref model logprob

현재 model wrapper가 generate는 지원하지만, sampled tokens의 per-token logprob를 바로 안 주므로
`forward(logits)`를 다시 돌려 logprob를 계산하는 helper가 필요하다.

필요 함수:

- `compute_sequence_logprobs(model, joint_inputs, sampled_ids, answer_mask)`

주의:

- RL 대상은 answer span tokens만
- prompt/image tokens는 objective에서 제외

### Phase 3. GRPO training loop

`trainer.py`에 새 branch 추가:

```python
if train_stage == "rl":
    run_rl_training(...)
```

현재 `NotImplementedError`를 이 분기로 교체.

RL loop 개요:

1. batch 로드
2. group rollout 생성
3. parsed output + reward 계산
4. reward 정규화 -> group advantage
5. current policy / ref policy logprob 계산
6. clipped objective + KL penalty 계산
7. backward / optimizer step

### Phase 4. periodic validation

RL epoch 중에도 아래를 계속 봐야 함:

- `val/dist`
- `val/object_acc`
- `val/joint_exact`
- `val/format_valid`
- `val/extra_text_rate`

RL reward가 좋아져도 validation generation이 무너지면 실패.

---

## RL Logging

wandb에 아래를 추가 추천:

- `rl/reward_mean`
- `rl/reward_std`
- `rl/reward_point_mean`
- `rl/reward_object_mean`
- `rl/reward_joint_mean`
- `rl/invalid_format_rate`
- `rl/extra_text_rate`
- `rl/kl_mean`
- `rl/adv_mean`
- `rl/adv_std`
- `rl/policy_loss`

그리고 preview 8~16개를 주기적으로 저장:

- raw output
- parsed result
- reward breakdown

---

## RL Entry Condition

아래 조건을 만족한 뒤 RL 진입 추천:

- `loss_format` 충분히 낮음
- `train/FormatValidRate` 높음
- preview 출력 대부분 parse 가능
- validation에서 `format_valid`가 안정화됨

실무적으로는:

- `val/format_valid >= 0.8`
- 혹은 preview 20개 중 대부분이 exact format

이후 RL 시작.

---

## Initial Safety Rules

초기 RL에서 반드시 지킬 것:

1. format invalid면 strong negative reward
2. KL beta를 너무 낮추지 말 것
3. RL lr는 SFT보다 충분히 낮출 것
4. object reward를 point보다 더 크게 두지 말 것
5. malformed output preview를 주기적으로 확인할 것

---

## Recommended First RL Config

초기 추천값:

```yaml
train:
  train_stage: rl
  lr: 1e-5

rl:
  rl_enabled: true
  rl_group_size: 4
  rl_clip_eps: 0.2
  rl_kl_beta: 0.02
  reward_point_weight: 1.0
  reward_object_weight: 0.75
  reward_format_weight: 1.0   # hard gate logic에서 사실상 invalid penalty로 사용
  reward_point_beta: 10.0
```

주의:

- `reward_point_beta`는 `exp(-beta * min_l2)`의 beta로 사용

---

## Smoke Tests Before Full RL

### 1. reward-only dry run

목표:

- checkpoint 로드
- rollout 8개
- reward 계산만
- update 없음

확인:

- malformed sample reward가 낮은지
- point/object가 맞는 sample reward가 높은지

### 2. one-batch RL update

목표:

- 1 batch만 업데이트
- NaN / inf / exploding KL 없는지 확인

### 3. short RL run

목표:

- few hundred updates
- `val/format_valid` 유지되는지
- `val/dist` 개선되는지

---

## Success Criteria

RL 성공 판단:

- `val/dist` 하락
- `val/object_acc` 또는 `test/MultiAcc@1` 유지/상승
- `val/format_valid` 유지
- `extra_text_rate` 감소

즉 point가 조금 좋아졌더라도 format이 다시 무너지면 실패로 본다.

---

## Notes

- 현재 task는 detection의 box reward가 아니라 point reward가 핵심이다.
- 따라서 Rex-Omni의 geometry-aware reward 아이디어는 유지하되,
  `IoU reward -> point distance reward`
  로 바꿔서 적용한다.
- object recognition을 함께 하므로 reward는 point-only가 아니라 joint task 관점에서 설계한다.
