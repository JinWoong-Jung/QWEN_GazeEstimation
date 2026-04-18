# 구현 결과 보고서

## 개요

Qwen3-VL 기반 시선 추정 모델에 **Stage 2 RL(강화학습) 파이프라인**을 구현했다.
기존 Stage 1 SFT 파이프라인은 그대로 유지하면서 `train_stage: "rl"` 설정으로 RL 학습을 독립 실행할 수 있도록 분기했다.

---

## 1. 출력 형식 변경 (SFT → RL 연동 기반)

### 변경 내용

| 항목 | 이전 | 이후 |
|------|------|------|
| 답변 래퍼 | `<gaze_point_start>` / `<gaze_obj_end>` (커스텀 신규 토큰) | `<\|im_start\|>` / `<\|im_end\|>` (Qwen 기존 스페셜 토큰 재사용) |
| 형식 | 단일 줄 | 2줄 (`Point:` / `Object:` 분리) |
| 신규 포맷 토큰 | 4개 추가 필요 | 0개 (기존 토큰 재사용) |

### 목표 출력 형식

```
<|im_start|>Point: <loc_572><loc_563>
Object: <obj_059><|im_end|>
```

- `<loc_NNN>`: x/y 각 1개, 0~999 범위의 정규화 좌표 빈 인덱스
- `<obj_KKK>`: 객체 클래스 ID 토큰 (클래스 수에 따라 자리수 결정)
- `<|im_start|>` / `<|im_end|>`은 신규 추가 없이 Qwen 어휘에 이미 존재

### 변경된 파일

**`model/utils/gaze_tokens.py`**

- `ANSWER_START = "<|im_start|>"`, `ANSWER_END = "<|im_end|>"` 로 교체
- `FORMAT_TOKENS = []` (포맷 전용 신규 토큰 없음)
- `_STRICT_RE`: 2줄 구조를 매칭하는 정규식
  ```python
  r"^<\|im_start\|>\s*Point:\s*(<loc_\d{3}>)(<loc_\d{3}>)\s*Object:\s*(<obj_\d+>|<obj_unknown>)\s*<\|im_end\|>$"
  ```
- `build_structured_target_text()`: 새 형식으로 타겟 텍스트 생성
- `parse_structured_output_text()`: 파싱 후 `{valid_format, point_xy, object_id, ...}` 반환

**`model/utils/eval_utils.py`**

- `decode_generated()` 내 특수토큰 제거 정규식 수정: `<|im_start|>` / `<|im_end|>`는 보존
  ```python
  # 이전: 모든 <|...|> 제거 → 파서가 항상 invalid 반환
  txt = re.sub(r"<\|[^>]+?\|>", "", txt)
  # 이후: 답변 래퍼는 보존
  txt = re.sub(r"<\|(?!im_start\||im_end\|)[^>]+?\|>", "", txt)
  ```
- `make_gaze_obj_end_stopping_criteria()`: `ANSWER_END` 토큰 ID로 정지 조건 생성
- `GAZE_OBJ_END` 임포트 → `ANSWER_END` 로 교체

**`config.yaml`**

- `generation_max_new_tokens`: `8` → `16` (새 형식 약 12 토큰 필요)

---

## 2. Stage 2 RL 파이프라인 (GRPO)

### 알고리즘 개요: GRPO (Group Relative Policy Optimization)

PPO 변형으로 별도 critic 없이 그룹 내 상대적 advantage를 사용한다.

```
loss = -E[ min(r·A, clip(r, 1-ε, 1+ε)·A) ] + β · E[log(π_θ / π_ref)]
```

- `r = exp((log π_θ - log π_old) / n_tokens)` : probability ratio
- `A` : 그룹 정규화 advantage
- `β · KL` : SFT 체크포인트로부터의 이탈 페널티

### 진입점 분기 (`model/trainer.py`)

```python
# main() 내부 — SFT 루프 진입 전에 완전히 분기
if train_stage == "rl" and not _inference_only:
    rl_global_step, rl_best_val_loss = _run_rl_training(...)
    if args.run_test:
        # 최종 테스트 평가
        ...
    finish_wandb(wandb_run)
    return  # SFT 루프 절대 진입하지 않음
```

SFT 루프는 `train_stage == "sft"`일 때만 실행된다.

---

### 새로 추가된 파일 / 함수

#### `model/utils/rl_utils.py` (신규)

| 함수 | 역할 |
|------|------|
| `compute_point_reward(min_l2, beta)` | `exp(-β · L2)`, L2→0일수록 1에 수렴 |
| `compute_object_reward(pred_id, gt_ids)` | 다중 레이블 exact match, 0 또는 1.0 |
| `compute_total_reward(parsed, gt_points, gt_obj_ids, ...)` | 포맷 하드 게이트 + 4개 성분 합산 |
| `group_normalize_advantages(rewards)` | GRPO 그룹 정규화: `(r - μ) / (σ + ε)` |
| `build_answer_span_mask(generated_ids, prompt_len, pad_id)` | 답변 토큰 위치 bool 마스크 |
| `compute_token_logprobs_sum(logits, input_ids, mask)` | Causal LM shift 후 답변 span log-prob 합산 |
| `compute_grpo_loss(new_lp, old_lp, ref_lp, n_tok, adv, ε, β)` | PPO clip + KL 페널티 합산 |

**보상 함수 설계**

```
포맷 무효  →  r_total = -1.0  (하드 게이트: SFT가 학습한 형식 구조 보호)

포맷 유효 시:
  r_point  = exp(-β · min_L2)          (β=10.0, 연속 보상)
  r_object = 1.0 if 정답 클래스 else 0  (다중 레이블 허용)
  r_joint  = 1.0 if r_object>0.5 AND min_L2<0.1 else 0  (보너스)
  r_extra  = 1.0 if 포맷 외 여분 텍스트 존재  (패널티 항)

  r_total = w_pt·r_pt + w_obj·r_obj + w_joint·r_joint - w_extra·r_extra
```

config.yaml 기본값: `w_pt=1.0, w_obj=0.75, w_joint=0.25, w_extra=0.5`

#### `model/utils/processor_collate.py` — `QwenRLCollator` (신규 클래스)

RL 학습에 필요한 배치를 구성한다. `QwenTrainCollator`와 달리 프롬프트만 토크나이즈하고 (답변 없음), 원본 PIL 이미지를 함께 반환하여 rollout 후 logprob 재계산 시 재사용할 수 있도록 한다.

반환 키:
- `joint_inputs`: 추론 모드 (프롬프트만, `add_generation_prompt=True`)
- `scene_images`: 리사이즈된 PIL 이미지 리스트 (logprob 패스 재처리용)
- `text_input`: 프롬프트 문자열 리스트
- `gt_points`, `target_label_ids`, `target_object_valid`: 보상 계산용 GT

#### `model/utils/wandb_utils.py` — RL 메트릭 추가

```
rl/reward_mean, rl/reward_point_mean, rl/reward_object_mean, rl/reward_joint_mean
rl/invalid_format_rate, rl/extra_text_rate, rl/kl_mean, rl/policy_loss
```

---

### `_run_rl_training()` 루프 상세 (`model/trainer.py:393`)

```
for epoch in range(rl_epochs):
    for batch in rl_loader:

        # 1. Rollout — G개 샘플 생성 (no_grad)
        generated_ids = policy_model.generate(..., num_return_sequences=G,
                                               do_sample=True, temperature, top_p)

        # 2. 디코딩 — B*G 출력 텍스트
        preds = decode_generated(processor, generated_ids, input_ids, ...)

        # 3. 보상 계산 — 각 rollout별 GT 대비 보상
        for k in range(B*G):
            parsed = parse_structured_output_text(preds[k], num_classes)
            rwd = compute_total_reward(parsed, gt_points[b], gt_ids[b], ...)

        # 4. Advantage 정규화 — 그룹(G)별 독립 정규화
        for b in range(B):
            adv[b*G:(b+1)*G] = group_normalize_advantages(rewards[b*G:(b+1)*G])

        # 5. Logprob 입력 구성 — B*G (scene, prompt, sampled_answer)를
        #    build_train_inputs()로 재처리 → joint_inputs with image features
        lp_joint, _, mask_pt, mask_obj, mask_fmt = build_train_inputs(
            processor, exp_scenes, exp_texts, preds, ...)

        # 6. Old logprobs — no_grad, 현재 policy로 계산 (ratio 분모)
        old_lp_sum, n_tok = compute_token_logprobs_sum(policy_model(...))

        # 7. Ref logprobs — no_grad, 동결 SFT 체크포인트 (KL 분모)
        ref_lp_sum, _ = compute_token_logprobs_sum(ref_model(...))

        # 8. New logprobs + GRPO loss — with grad
        new_lp_sum, _ = compute_token_logprobs_sum(policy_model(...))
        loss, stats = compute_grpo_loss(new_lp_sum, old_lp_sum, ref_lp_sum,
                                         n_tok, adv, ε, β)
        (loss / accum_steps).backward()

        # grad_accum_steps마다 optimizer.step() + scheduler.step()

    # epoch 종료 후: val loss + val metric 평가 + checkpoint 저장
```

#### Reference 모델 로딩

SFT 체크포인트에서 동결 복사본을 만들어 KL 기준점으로 사용한다.

```python
_ref_base = init_base_model(model_path, model_kwargs)
_ref_base.resize_token_embeddings(new_vocab_size)
_ref_qwen = PeftModel.from_pretrained(_ref_base, checkpoint_dir/"lora_adapter",
                                       is_trainable=False)
ref_model = QwenTextGenerationModel(_ref_qwen).to(device)
ref_model.eval()
for p in ref_model.parameters():
    p.requires_grad_(False)
```

`checkpoint_dir`가 없거나 어댑터가 없으면 베이스 모델을 ref로 사용하고 경고를 출력한다.

#### Logprob 마스크 전략

구조화 마스크(`mask_pt | mask_obj | mask_fmt`)를 우선 사용하고, 포맷이 invalid하여 구조화 토큰이 없는 샘플에 한해서만 `build_answer_mask()`의 전체 답변 span 마스크로 fallback한다.

---

## 3. config.yaml RL 관련 설정

```yaml
train:
  train_stage: "rl"   # "sft" 또는 "rl"

rl:
  rl_enabled: false         # 레거시 플래그 (현재는 train_stage로 제어)
  rl_group_size: 4          # G: 프롬프트당 rollout 샘플 수
  rl_clip_eps: 0.2          # PPO clip epsilon
  rl_kl_beta: 0.02          # KL penalty weight
  reward_point_weight: 1.0
  reward_object_weight: 0.75
  reward_joint_bonus: 0.25
  reward_extra_penalty: 0.5
  reward_point_beta: 10.0   # exp(-β·L2) 가파름 계수
  rl_temperature: 0.7       # rollout 샘플링 temperature
  rl_top_p: 0.9             # rollout top-p
```

---

## 4. 실행 방법

**Stage 1 SFT (기존과 동일)**
```bash
python main.py --config config.yaml
```

**Stage 2 RL (SFT 체크포인트 기반)**
```bash
python main.py --config config.yaml \
    --train_stage rl \
    --checkpoint_dir checkpoints/baseline/best
```

`checkpoint_dir`에 `lora_adapter/`가 있어야 정상적으로 SFT 가중치에서 RL이 시작된다.

> 파서는 argparse 기반이므로 Hydra 스타일(`train.train_stage=rl`)은 동작하지 않는다.

---

## 5. 검증

```
$ python -c "from model.trainer import main; print('OK')"
OK

$ python -m pytest tests/ -v
94 passed in 2.51s
```

기존 94개 SFT 테스트 전부 통과. RL 코드는 `train_stage == "rl"` 분기 안에만 존재하므로 SFT 파이프라인에 영향 없음.
