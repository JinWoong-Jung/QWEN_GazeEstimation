# Stage 2 RL: Rex-Omni GRPO 방식 적용 계획

Rex-Omni(finetuning/verl/trainer/core_algos.py, engine/utils/qwen_grpo_module.py)의
GRPO 구현 방식을 현재 single-GPU 환경에 맞게 경량화하여 적용한다.

---

## 현재 구현 vs Rex-Omni 핵심 차이

| 항목 | 현재 | Rex-Omni | 우선순위 |
|------|------|----------|----------|
| Log-prob 단위 | 시퀀스 합산 [B] | **토큰별** [B, L] | ★★★ |
| PPO 클리핑 | 시퀀스 단위 | **토큰별 + Dual-clip** | ★★★ |
| KL 계산 | 단순 차분 | **low_var_kl (안정적)** | ★★★ |
| KL 계수 | 고정 | **AdaptiveKLController** | ★★☆ |
| Rollout 재사용 | 없음(매 step 새 rollout) | **n_ppo_epochs 만큼 재사용** | ★★★ |
| old_lp 처리 | new_lp.detach() | **rollout 직후 캐시, 재사용** | ★★★ |
| Response mask | structured tokens만 | **전체 생성 토큰** | ★★☆ |

---

## Task 목록

### [T1] rl_utils.py — 알고리즘 함수 교체 ✅ (완료)
- [x] `compute_token_logprobs_sum` → per-token 반환 버전 추가
- [ ] `compute_token_logprobs` 신규: [B, L-1] per-token log-prob 반환
- [ ] `compute_kl_per_token`: low_var_kl 모드 구현 (Rex-Omni `compute_kl`)
- [ ] `compute_policy_loss_per_token`: Rex-Omni `compute_policy_loss` 이식
  - asymmetric clip (clip_ratio_low / clip_ratio_high)
  - dual-clip (clip_ratio_dual) — adv < 0일 때 catastrophic update 방지
  - `response_mask` 기반 masked_mean
- [ ] `AdaptiveKLController` 클래스 추가 (fixed/adaptive 선택)
- [ ] `group_normalize_advantages` → index 텐서 기반으로 교체 (Rex-Omni 방식)

### [T2] trainer.py — 학습 루프 재구조화
현재: 매 batch마다 rollout → logprob → loss → backward (1회)
목표: rollout → cache → n_ppo_epochs만큼 mini-batch update

#### T2-1: Rollout 캐시 구조 도입
```
rollout_buffer = {
    "lp_joint_dev": ...,     # processor 출력 (B*G samples)
    "answer_mask": ...,      # response token mask
    "old_log_probs": ...,    # per-token [B*G, L] — rollout 직후 1회 계산
    "ref_log_probs": ...,    # per-token [B*G, L] — ref_model 1회 계산
    "advantages": ...,       # [B*G] — group normalize
    "n_tokens": ...,
}
```
- rollout 직후 old_lp를 **1회** 계산하고 캐시
- n_ppo_epochs(기본 1→2) 동안 재사용
- **old_lp가 이제 진짜 의미를 가짐** (파라미터 업데이트 후 new≠old)

#### T2-2: Per-token 학습 루프 전환
```python
# 기존 (sequence-level)
new_lp_sum, n_tok = compute_token_logprobs_sum(logits, input_ids, answer_mask)
old_lp_sum = new_lp_sum.detach()

# 신규 (token-level, Rex-Omni 방식)
new_log_probs = compute_token_logprobs(logits, input_ids)  # [B*G, L-1]
pg_loss, kl_loss = compute_policy_loss_per_token(
    old_log_probs=cached_old_lp,
    new_log_probs=new_log_probs,
    ref_log_probs=cached_ref_lp,
    advantages=adv_tensor,
    response_mask=response_mask,
    clip_ratio_low=clip_eps,
    clip_ratio_high=clip_eps,
    clip_ratio_dual=3.0,
    kl_beta=kl_ctrl.kl_coef,
)
```

#### T2-3: AdaptiveKLController 연동
```python
kl_ctrl = AdaptiveKLController(init_kl_coef=0.01, target_kl=0.1, horizon=10000)
# 매 optimizer step 후:
kl_ctrl.update(current_kl=mean_kl, n_steps=accum_steps)
```

### [T3] config_rl.yaml — 파라미터 추가
```yaml
rl:
  # 기존
  rl_clip_eps: 0.2
  rl_kl_beta: 0.01

  # 신규 (Rex-Omni 방식)
  rl_clip_ratio_low: 0.2        # PPO lower clip
  rl_clip_ratio_high: 0.2       # PPO upper clip (DAPO: 더 크게 설정 가능)
  rl_clip_ratio_dual: 3.0       # Dual-clip (adv<0 케이스 방지)
  rl_kl_type: "adaptive"        # "fixed" | "adaptive"
  rl_kl_target: 0.1             # adaptive KL target
  rl_kl_horizon: 10000          # adaptive KL horizon
  rl_n_ppo_epochs: 2            # rollout당 gradient update 횟수 (rollout 재사용)
  rl_kl_penalty: "low_var_kl"  # "kl" | "abs" | "mse" | "low_var_kl"
```

### [T4] 메모리 경량화 (단일 GPU 대응)
Rex-Omni는 8×GPU + DeepSpeed ZeRO-3 + vLLM 사용.
단일 GPU 환경에서:

- [ ] **ref_model CPU offload**: 추론 시에만 GPU 이동
  ```python
  ref_model = ref_model.cpu()
  # 사용 시:
  ref_model.to(device)
  with torch.no_grad(): out = ref_model(...)
  ref_model.cpu()
  torch.cuda.empty_cache()
  ```
- [ ] **gradient_checkpointing** policy_model에 적용 확인 (config에 이미 true)
- [ ] **rollout 시 torch.inference_mode()** 사용 (no_grad보다 약간 빠름)
- [ ] **lp_joint CPU 계산**: processor 출력 → CPU에서 마스크 계산 → GPU 이동 최소화

### [T5] Response mask 개선
현재: structured tokens (loc, obj, fmt)만 마스킹
목표: 전체 생성 토큰에 대해 per-token PPO 적용 (Rex-Omni 방식)

```python
# 전체 생성 구간 마스크 (prompt 이후 모든 토큰)
response_mask = build_answer_span_mask(
    generated_ids=lp_input_ids,
    prompt_len=prompt_len,
    pad_token_id=pad_token_id,
)  # [B*G, L]
```
- structured mask는 **SFT supervised loss** (보조 목표)로는 유지 가능
- RL PPO loss에는 response_mask 사용

---

## 구현 순서

```
T1 (rl_utils.py 알고리즘 교체)
  └─→ T3 (config 파라미터 추가)
       └─→ T2 (trainer.py 재구조화)
            ├─ T2-1 rollout buffer
            ├─ T2-2 per-token 루프
            └─ T2-3 adaptive KL
                └─→ T4 (메모리 최적화)
                     └─→ T5 (response mask)
```

---

## 완료 기준

- [x] T1: rl_utils.py에 per-token 함수군 추가, 기존 함수 호환 유지
- [x] T2: trainer.py 재구조화 (rollout buffer, n_ppo_epochs, adaptive KL)
- [x] T3: config_rl.yaml에 신규 파라미터 반영
- [x] T4: ref_model CPU offload 적용 (VRAM ~8GB 절감)
- [x] T5: response_mask 기반 per-token PPO loss 적용
- [ ] 실제 학습 실행 후 수렴 확인 (wandb 지표 모니터링)

---

## 예상 효과

| 항목 | Before | After |
|------|--------|-------|
| Forward pass / step | 2회 (ref + new) | 2회 유지 (rollout 재사용으로 실질 절감) |
| VRAM | policy+ref ~20GB | policy ~10GB + ref CPU offload |
| KL 안정성 | 고정 beta → 발산 가능 | Adaptive → 자동 조절 |
| PPO clipping | sequence-level | token-level + dual-clip |
| Rollout 효율 | 1 rollout = 1 update | 1 rollout = n_ppo_epochs updates |
