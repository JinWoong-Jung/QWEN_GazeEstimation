# Stage 2: RL (DPO) 계획

## 핵심 전제

- **GT reasoning text는 train set에만 존재**한다. val/test에는 없으므로 eval 메트릭은 point/object 기반만 의미 있음.
- **평가 포맷** (`output_format: "reasoning"`)에서 reasoning은 자유 생성이고 GT와 비교하지 않음.
- **하드웨어**: A100 80GB 단일 GPU.

---

## 구현 방향

현재 구현된 GRPO/PPO-style online RL 경로는 Stage 2 기본 학습 방식으로 사용하지 않는다.
Stage 2는 **offline DPO로 전환**하며, `train_stage: rl` 경로를 DPO로 대체하거나
별도 `train_stage: dpo` 경로로 분리한다.

DPO에서는 아래 GRPO 구성 요소를 **사용하지 않는다**:

| 제거 대상 | 위치 |
|---|---|
| Online rollout (generate × K per step) | `_run_rl_training` 내부 rollout loop |
| Group advantage normalization | `group_normalize_advantages()` |
| PPO clipping (asymmetric + dual-clip) | `compute_policy_loss_per_token()` |
| Old policy logprob 캐싱 | `infer_logprobs_chunked()` × 2 |
| Adaptive KL controller | `build_kl_controller()` |
| RL-specific hyperparams | `RL.yaml`: `rl_group_size`, `rl_clip_*`, `rl_kl_*` 등 |

대신 offline `(prompt, chosen, rejected)` pair dataset을 읽고, frozen SFT checkpoint를
reference model로 사용해 standard DPO loss를 계산한다.

---

## 왜 GRPO가 아닌 DPO인가

| | GRPO | DPO |
|---|---|---|
| reasoning 판단 | 불가 (outcome만 reward 가능) | 가능 (27B judge 오프라인 활용) |
| 데이터 | 온라인 (매 스텝 샘플링) | 오프라인 pair (1회 구축) |
| 27B 활용 방식 | 매 스텝 호출 → 비현실적 | 오프라인 스코어링 → 현실적 |
| 구현 복잡도 | 높음 (PPO 계열) | 낮음 (supervised loss) |

**핵심 이유**: reasoning 품질을 outcome-only reward로는 안정적으로 개선하기 어렵다.
Vision task에서 모델은 reasoning 없이 image feature만으로 correct point를 예측할 수 있어
reasoning이 outcome에 인과적으로 연결된다는 보장이 없다.
27B judge를 offline으로 써서 DPO pair를 만드는 방식이 더 설득력 있다.

---

## 기존 코드와의 관계 정리

| 파일 | 현재 역할 | DPO 전환 후 |
|---|---|---|
| `model/trainer.py` `_run_rl_training()` | GRPO online RL | DPO loss로 교체 또는 `_run_dpo_training()` 신규 작성 |
| `RL.yaml` | GRPO 하이퍼파라미터 | DPO 하이퍼파라미터로 교체 (`beta`, `dpo_lr` 등) |
| `RL_data_pipeline.py` | train distance 기반 annotation subset 생성용 | **DPO pair pipeline이 아님** — 별도 스크립트로 분리 |

### 신규 스크립트 (역할 분리)

```
scripts/dpo/
    sample_completions.py   # SFT → K completions 생성 및 저장
    score_outcomes.py       # GT 기반 outcome score 계산
    judge_reasoning.py      # 27B judge reasoning score 계산
    build_dpo_pairs.py      # score 조합 → pair 구성 + 필터링
    train_dpo.py            # DPO 학습 실행
```

---

## 전체 파이프라인

```
SFT checkpoint
    │
    ▼ sample_completions.py   (2B 모델, offline 1회)
    │
    ├─ score_outcomes.py      ← GT point / GT object 비교
    │
    └─ judge_reasoning.py     ← 27B judge (image + bbox + GT + generated reasoning)
         │
         ▼ build_dpo_pairs.py
         │   score 조합 → gap filtering → (chosen, rejected) dataset
         │
         ▼ train_dpo.py
         │
    Stage 2 model
```

---

## Phase 1: Completion 샘플링 (`sample_completions.py`)

### 설정
- 사용 데이터: train set 전체 (GT reasoning 유무 무관)
- `output_format: "reasoning"` 프롬프트 사용
- K = 8, temperature = 0.9 (**sweep 대상**: K ∈ {4, 8, 16}, temp ∈ {0.7, 0.9, 1.0})

### 저장 스키마 (per sample)

```json
{
  "sample_id": "...",
  "image_path": "...",
  "head_bbox": [x1, y1, x2, y2],
  "gt_points": [[gx1, gy1], [gx2, gy2], ...],
  "gt_object_ids": [12, 12, 15, ...],
  "completions": [
    {
      "full_text": "<|reasoning_start|>...<|reasoning_end|><|point_start|>...<|object_end|>",
      "reasoning_text": "...",
      "pred_point": [px, py],
      "pred_object_id": 12,
      "parse_valid": true,
      "format_valid": true,
      "reasoning_valid": true,
      "outcome_score": null,
      "reasoning_score": null,
      "final_score": null,
      "filter_reason": null
    },
    ...
  ]
}
```

> `gt_points`는 annotator별 복수, `gt_object_ids`도 list로 허용.
> `outcome_score` / `reasoning_score` / `final_score`는 이후 단계에서 채움.
> `filter_reason`은 quick_filter 또는 27B에서 reject된 이유를 기록 (디버깅용).

---

## Phase 2: Outcome 스코어링 (`score_outcomes.py`)

```python
def outcome_score(pred_point, pred_obj_id, gt_points, gt_object_ids, sigma=0.1):
    # sigma: sweep 대상 (0.05, 0.1, 0.15)
    l2 = min(euclidean(pred_point, gt) for gt in gt_points)
    r_point = math.exp(-l2 / sigma)

    r_obj = 1.0 if pred_obj_id in gt_object_ids else 0.0

    return 0.7 * r_point + 0.3 * r_obj
```

---

## Phase 3: Reasoning 스코어링 (`judge_reasoning.py`)

### 메모리: Qwen2.5-27B bf16 ≈ 54GB → A100 80GB에서 inference 가능

### 27B 호출 전 사전 필터 (비용 절감)

아래 조건 중 하나라도 해당하면 `reasoning_score = 0.0`으로 설정하고 27B 호출 생략:

```python
def quick_filter(reasoning_text, pred_point, head_bbox):
    head_cx = (head_bbox[0] + head_bbox[2]) / 2
    pred_left  = pred_point[0] < head_cx

    text = reasoning_text.lower()
    says_left   = "left"  in text
    says_right  = "right" in text

    # 방향 불일치 (대칭 조건 모두 처리)
    if pred_left  and says_right and not says_left:
        return 0.0, "direction_mismatch_right"
    if not pred_left and says_left and not says_right:
        return 0.0, "direction_mismatch_left"

    # 너무 짧은 reasoning
    if len(reasoning_text.split()) < 5:
        return 0.0, "too_short"

    # reasoning marker parse 실패
    if "<|reasoning_start|>" not in reasoning_text:
        return 0.0, "marker_parse_fail"

    # generic filler 감지 (확장 가능)
    generic_patterns = ["i cannot determine", "not visible", "unclear"]
    if any(p in text for p in generic_patterns):
        return 0.0, "generic_filler"

    return None, None  # None = 27B 판단 필요
```

### 27B 프롬프트

```
[System]
You are evaluating reasoning quality for gaze estimation.
Score 0.0–1.0. Respond with a single float only.

[User]
Image: {image}
Head bbox: ({x1},{y1}) to ({x2},{y2})

Generated reasoning:
"{reasoning_text}"

Ground truth:
- Gaze target object: "{gt_object_name}"
- Gaze point (normalized): ({gt_x:.3f}, {gt_y:.3f})

Scoring criteria:
1. Does it correctly identify the gaze direction toward the target? (weight 0.4)
2. Does it mention or imply the correct target object? (weight 0.4)
3. Is it specific to this image, not generic filler text? (weight 0.2)
```

### Judge Calibration (품질 검증)

DPO는 pair ranking 품질에 민감하므로 27B judge 신뢰성을 반드시 확인:

- [ ] 100개 spot-check: 사람이 직접 점수와 비교
- [ ] judge 재평가 분산 측정: 동일 샘플을 3회 judge → score 표준편차 기록
- [ ] score histogram 저장: 분포가 0/1 양극단에 쏠리는지 확인
- [ ] pair type별 개수 로그: outcome-driven / reasoning-driven / mixed 비율 기록

---

## Phase 4: Pair 구성 (`build_dpo_pairs.py`)

### 최종 score 조합

```python
# 가중치는 sweep 대상: (0.7, 0.3), (0.6, 0.4), (0.5, 0.5)
final_score = 0.6 * outcome_score + 0.4 * reasoning_score
```

### Pair 선택 전략

```python
for sample_id, data in dataset.items():
    completions = [c for c in data["completions"] if c["parse_valid"]]
    if len(completions) < 2:
        continue

    ranked = sorted(completions, key=lambda c: c["final_score"], reverse=True)
    chosen   = ranked[0]
    rejected = ranked[-1]
    gap      = chosen["final_score"] - rejected["final_score"]

    # gap threshold: sweep 대상 (0.2, 0.25, 0.3)
    if gap < 0.25:
        rejected["filter_reason"] = f"gap_too_small({gap:.3f})"
        continue

    save_dpo_pair(sample_id, chosen, rejected, gap=gap, pair_type=classify_pair(chosen, rejected))
```

### Pair 유형 분류

| 유형 | 조건 | 학습 효과 |
|---|---|---|
| **outcome-driven** | `\|r_outcome_c - r_outcome_r\| > 0.4` | 정확한 point/object 예측 강화 |
| **reasoning-driven** | outcome 비슷, `\|r_reasoning_c - r_reasoning_r\| > 0.4` | reasoning 품질 집중 학습 |
| **mixed** | 둘 다 차이 있음 | 일반적 품질 향상 |

---

## Phase 5: DPO 학습 (`train_dpo.py`)

### DPO Loss

```
L = -log σ( β * ( log π_θ(chosen|x) - log π_ref(chosen|x)
                - log π_θ(rejected|x) + log π_ref(rejected|x) ) )
```

- `π_ref`: SFT checkpoint (frozen)
- `x`: prompt (image + head bbox)

### Response Mask — 구현 주의사항

**반드시 reasoning 토큰까지 포함해야 한다.**

현재 GRPO의 response mask는 structured token 위주로만 구성되어 있어
reasoning content가 빠질 가능성이 있음:

```python
# 기존 GRPO (잘못된 방식)
response_mask = point_mask | object_mask | format_mask
# → reasoning 토큰 제외됨

# DPO에서 올바른 방식
response_mask = build_answer_mask(
    processor, joint_inputs, target_texts, target_valid
)
# → <|reasoning_start|>...<|reasoning_end|> + point + object 전체 포함
```

DPO loss는 `response_mask`가 True인 토큰에 대해서만 logprob을 계산하므로,
reasoning block이 빠지면 reasoning이 전혀 학습되지 않음.

### 하이퍼파라미터

```yaml
dpo:
  beta: 0.1           # sweep: 0.05, 0.1, 0.2
  lr: 5e-6            # SFT(1e-4)보다 낮게
  epochs: 1           # overfitting 주의, 최대 2
  batch_size: 8
  grad_accum_steps: 4
```

### 메모리 (A100 80GB)

| 구성 요소 | VRAM |
|---|---|
| Policy model (2B, LoRA trainable) | ~6GB |
| Reference model (2B, frozen) | ~6GB |
| Optimizer states (LoRA only) | ~8GB |
| 합계 | ~20GB (여유 충분) |

---

## 향후 확장: Iterative DPO

Offline DPO 검증 후, 성능이 포화되면:

```
현재 policy → K samples 재샘플링
→ 27B 재스코어링
→ 새 (chosen, rejected) pair 구성
→ 다음 DPO iteration
```

distribution drift 문제를 완화하고 성능을 점진적으로 개선 가능.

---

## Sweep 대상 정리

| 파라미터 | 초안값 | 탐색 범위 |
|---|---|---|
| K (completions per sample) | 8 | 4, 8, 16 |
| temperature | 0.9 | 0.7, 0.9, 1.0 |
| sigma (point reward decay) | 0.1 | 0.05, 0.1, 0.15 |
| outcome : reasoning 가중치 | 0.6 : 0.4 | (0.7:0.3), (0.6:0.4), (0.5:0.5) |
| gap threshold | 0.25 | 0.2, 0.25, 0.3 |
| DPO beta | 0.1 | 0.05, 0.1, 0.2 |

---

## 미결 사항

- [ ] `train_stage: rl` 경로를 DPO로 교체할지, `train_stage: dpo`를 신규 분기로 분리할지 결정
- [ ] 27B 모델 선택 (Qwen2.5-27B-Instruct vs 다른 후보)
- [ ] Judge calibration 결과에 따라 27B 프롬프트 조정
- [ ] reasoning-driven pair 비율 결정 (전체 pair 중 몇 % 포함할지)
- [ ] Iterative DPO 전환 시점 기준 (val_dist plateau 여부 등)
