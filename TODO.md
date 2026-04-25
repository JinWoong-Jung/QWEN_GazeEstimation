# 성능 개선 TODO

`python main.py --config config.yaml` 에폭당 실행시간 단축을 위한 작업 목록.
Claude + Codex 교차검증 기반. 우선순위 순으로 정렬.

---

## P0 — 코드 수정 없이 즉시 적용 가능

### [ ] P0-1. config.yaml: val generation 평가 주기 변경
**파일**: `config.yaml:52-53`

```yaml
# 변경 전
run_val_metrics_every_n_epochs: 1
checkpoint_monitor: "val_dist"

# 변경 후
run_val_metrics_every_n_epochs: 5
checkpoint_monitor: "val_loss"
```

**이유**: 매 에폭 `run_test_metrics()`(autoregressive generation)를 실행 중. val 전체에 `max_new_tokens=16` generate로 teacher-forced forward 대비 최소 16배 오래 걸림. `checkpoint_monitor`를 `val_dist`에서 `val_loss`로 바꾸지 않으면 주기를 늘려도 best 체크포인트 저장이 스킵됨.

**예상 효과**: val generation이 에폭당 1회 → 5회당 1회로 줄어, 총 에폭 시간에서 generation 비중만큼 단축.

---

### [ ] P0-2. config.yaml: 속도 측정 중 test 평가 비활성화
**파일**: `config.yaml:50`

```yaml
# 변경 전
run_test: true

# 변경 후 (학습 속도 측정 시)
run_test: false
```

**이유**: 전체 학습이 끝난 후 test set generation이 자동 실행됨. 학습 속도 파악 중에는 불필요.

---

## P1 — GPU 병목 (코드 수정, 즉각적 효과)

### [ ] P1-1. set_seed(): cuDNN deterministic 모드 제거
**파일**: `model/trainer.py:137-141`

```python
# 제거 또는 환경 변수로 guard
def set_seed(seed: int) -> None:
    os.environ["PYTHONHASHSEED"] = str(int(seed))
-   os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
-   if hasattr(torch.backends, "cudnn"):
-       torch.backends.cudnn.deterministic = True
-       torch.backends.cudnn.benchmark = False
-   if hasattr(torch, "use_deterministic_algorithms"):
-       torch.use_deterministic_algorithms(True, warn_only=True)
```

재현성이 필요하면 `QWEN_DETERMINISTIC=1` 환경 변수로 guard:
```python
    if os.environ.get("QWEN_DETERMINISTIC", "").lower() in {"1", "true"}:
        os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        torch.use_deterministic_algorithms(True, warn_only=True)
    else:
        torch.backends.cudnn.benchmark = True
```

**이유**: `deterministic=True` + `use_deterministic_algorithms(True)` 조합이 GPU 처리량 20~50% 저하. 특히 scatter/interpolate ops의 결정적 구현은 현저히 느린 대안 사용.

---

### [ ] P1-2. masked_token_ce(): CE 3회 → 1회로 통합
**파일**: `model/utils/loss_utils.py:123-126`

```python
# 현재: [B*L, V] CE를 3번 계산
l_pt,  n_pt  = masked_token_ce(logits, labels, loss_mask_point)
l_obj, n_obj = masked_token_ce(logits, labels, loss_mask_object)
l_fmt, n_fmt = masked_token_ce(logits, labels, loss_mask_format)
```

point/object/format 마스크가 서로 배타적(disjoint)이므로:
```python
# 개선: valid union 위치에서만 CE 1번 계산 후 category별 분리
shift_logits = logits[:, :-1, :]
shift_labels = labels[:, 1:]

union_mask = loss_mask_point | loss_mask_object | loss_mask_format
valid_union = union_mask[:, 1:] & shift_labels.ne(-100)  # [B, L-1] bool

# valid token 위치에서만 CE materialization — [B*L, V] 전체 계산 회피
ce_valid = F.cross_entropy(
    shift_logits[valid_union],   # [N_valid, V]
    shift_labels[valid_union],   # [N_valid]
    reduction="none",
)  # [N_valid]

# 각 마스크 적용
def _masked_mean(ce, mask):
    m = mask[:, 1:] & valid_union   # valid_union 내에서 category 선택
    n = int(m[valid_union].sum().item())
    return (ce[m[valid_union]].mean() if n > 0 else zero), n

l_pt,  n_pt  = _masked_mean(ce_valid, loss_mask_point)
l_obj, n_obj = _masked_mean(ce_valid, loss_mask_object)
l_fmt, n_fmt = _masked_mean(ce_valid, loss_mask_format)
```

**이유**: Qwen-VL vocab 150k+ 기준 `[B*L, V]` CE materialization을 3번 → **valid token 수(`N_valid`)** 기준 1번으로 줄임. 3번 → 1번 감소에 더해 불필요한 padding/ignored 위치 연산까지 제거.

---

### [ ] P1-3. format_valid_rate argmax를 매 step → log step에서만 계산
**파일**: `model/utils/loss_utils.py:126`, `model/trainer.py:1662-1672`

```python
# 현재: compute_structured_loss()가 매 배치 masked_sample_exact_rate() 호출
fmt_valid_rate, n_fmt_samples = masked_sample_exact_rate(logits, labels, loss_mask_format)
# → 내부에서 preds = shift_logits.argmax(dim=-1) 실행 (매 배치 [B,L,V] argmax)
```

수정 방향:
- `compute_structured_loss()`에 `compute_format_rate: bool = False` 인자 추가
- trainer.py에서 wandb log step에 해당할 때만 `True` 전달
- 또는 `format_valid_rate`를 val 전용 메트릭으로 이동

**이유**: 학습 중 매 배치마다 `[B, L, V]` argmax 실행. wandb log 주기(20 step)와 무관하게 항상 발생. 진단 metric 계산을 위해 불필요한 GPU 연산 발생.

---

## P2 — CPU/Collator 병목 (코드 수정)

### [ ] P2-1. build_structured_masks(): target text 재토크나이즈 제거
**파일**: `model/utils/processor_collate.py:137-180`

```python
# 현재: build_structured_masks() 내부에서 샘플별 재토크나이즈
out = tokenizer(ans_txt, add_special_tokens=False, return_attention_mask=False)
ans_ids = out.get("input_ids", [])
```

`build_train_inputs()`에서 이미 `processor()`가 전체를 토크나이즈했으므로:
```python
# 개선: build_train_inputs()에서 target token ids를 미리 계산해서 전달
ans_ids_batch = tokenizer(
    target_texts,
    add_special_tokens=False,
    return_attention_mask=False,
    padding=False,
).input_ids  # list of lists, 1번의 배치 호출
# build_structured_masks()에 ans_ids_batch 인자로 전달
```

**이유**: batch_size=8 기준 배치당 tokenizer 호출 8번 → 1번으로 줄임. DataLoader worker 안에서 실행되므로 CPU 데이터 로딩 병목에 직접 영향.

---

### [ ] P2-2. convert_ids_to_tokens(): 배치 호출로 교체
**파일**: `model/utils/processor_collate.py:165`

```python
# 현재: 토큰 하나씩 N번 호출
tok_strs = [str(tokenizer.convert_ids_to_tokens(int(tid))) for tid in ans_ids]

# 수정: 리스트 한 번에 처리
raw_toks = tokenizer.convert_ids_to_tokens(ans_ids)
tok_strs = [str(t) for t in raw_toks]
```

**이유**: 1줄 변경으로 즉시 적용 가능. `build_structured_masks()` 안에서 배치당 `bsz × len(ans_ids)` 횟수만큼 반복되는 패턴.

---

### [ ] P2-3. StoppingCriteria: .tolist() 제거하고 텐서 연산으로 교체
**파일**: `model/utils/eval_utils.py:37-42`

```python
# 현재
def __call__(self, input_ids, scores, **kwargs):
    for row in input_ids:
        generated = row[self.prompt_len :].tolist()
        if self.obj_end_id not in generated:
            return False
    return True

# 수정
def __call__(self, input_ids, scores, **kwargs):
    generated = input_ids[:, self.prompt_len:]
    return bool((generated == self.obj_end_id).any(dim=1).all().item())
```

**이유**: generation의 각 decode step마다 호출. GPU 텐서 → Python list 변환 + 선형 탐색 → 텐서 비교 연산으로 교체.

---

## P3 — I/O 및 리소스 정리 (코드 수정)

### [ ] P3-1. load_checkpoint_for_eval(): 불필요한 trainer_state 로드 제거
**파일**: `model/utils/checkpoint.py:180-183`

```python
# 제거
trainer_state_path = ckpt_dir / "trainer_state.pt"
if trainer_state_path.exists():
    _ = torch.load(trainer_state_path, map_location=device)
```

**이유**: eval 경로에서 optimizer/scheduler state (수십 MB)를 로드하고 즉시 버림. 2줄 삭제로 불필요한 I/O 제거.

---

### [ ] P3-2. val_metric_loader: persistent_workers 비활성화 또는 워커 수 감소
**파일**: `model/trainer.py:1383-1394`

```python
# 현재: train(4) + val(4) + val_metric(4) = 12 persistent workers
val_metric_loader = DataLoader(
    val_ds,
    num_workers=_nw,           # 4
    persistent_workers=True,   # 상시 유지
    ...
)

# 수정: val_metric_loader는 드물게 사용되므로 persistent 불필요
val_metric_loader = DataLoader(
    val_ds,
    num_workers=min(_nw, 2),
    persistent_workers=False,
    ...
)
```

**이유**: P0-1에서 val_metric 주기를 늘리면 이 DataLoader는 거의 idle 상태로 12개 워커를 유지. OS 메모리 압박 발생 시 페이지 폴트로 전체 학습이 느려질 수 있음.

---

### [ ] P3-3. checkpoint 저장: last/ 저장 주기 제한
**파일**: `model/trainer.py:1828-1837`

```python
# 현재: 매 에폭 last/ 저장
save_checkpoint(out_dir / "last", epoch, model, processor, optimizer, scheduler,
                clear_dir=True, base_vocab_size=base_vocab_size)

# 수정: N 에폭마다 1회
save_last_every = int(getattr(args, "save_last_every_n_epochs", 5))
if (epoch % save_last_every == 0) or (epoch == effective_epochs):
    save_checkpoint(out_dir / "last", epoch, model, processor, optimizer, scheduler,
                    clear_dir=True, base_vocab_size=base_vocab_size)
```

config.yaml에 옵션 추가:
```yaml
train:
  save_last_every_n_epochs: 5
```

**이유**: LoRA adapter + processor 저장이 에폭마다 반복되는 I/O. 에폭마다 2회(best + last)에서 last는 주기 제한으로 줄임. (LoRA adapter 자체는 수십 MB 수준이나 매 에폭 반복이 문제)

---

## P-TIMING — 계측 항목 (수정 전후 숫자 비교용)

### [ ] TIM-1. 에폭별 구간 타이밍 로그 추가
**파일**: `model/trainer.py` — 학습 루프 내 주요 구간

수정 전/후 효과를 숫자로 비교하려면 구간별 시간을 에폭 로그에 찍어야 함:

```python
import time

# 학습 루프 내 계측 포인트
t0 = time.perf_counter()
# --- collate + DataLoader next() ---
batch = next(train_iter)
t_collate = time.perf_counter() - t0

t1 = time.perf_counter()
# --- forward + loss ---
outputs = model(...)
loss = compute_loss(...)
t_forward = time.perf_counter() - t1

t2 = time.perf_counter()
# --- backward ---
loss.backward()
t_backward = time.perf_counter() - t2

# 에폭 끝에서 합산 로그
wandb.log({
    "time/collate_per_step":   t_collate_sum / steps,
    "time/forward_per_step":   t_forward_sum / steps,
    "time/backward_per_step":  t_backward_sum / steps,
    "time/val_loss_epoch":     t_val_loss,
    "time/val_generation_epoch": t_val_gen,   # run_test_metrics() 실행 시만
    "time/checkpoint_save_epoch": t_ckpt,
})
```

**이유**: P0~P3 수정 후 "어느 구간이 얼마나 줄었는가"를 측정 없이 판단할 수 없음. 특히 collate 병목(P2)와 val generation 병목(P0-1)은 실측값이 있어야 추가 최적화 우선순위를 결정 가능.

---

## P4 — RL 전용 (SFT에는 무관)

### [ ] P4-1. _infer_logprobs_chunked(): prefix sum을 누적합 배열로 교체
**파일**: `model/trainer.py:448-449`

```python
# 현재: chunk마다 O(N) 재계산
for start in range(0, total, micro_bsz):
    end = min(start + micro_bsz, total)
    patch_start = sum(patches_per_sample[:start])
    patch_end   = sum(patches_per_sample[:end])

# 수정: 루프 전에 누적합 1회 계산
import itertools
cum = list(itertools.accumulate(patches_per_sample, initial=0))
for start in range(0, total, micro_bsz):
    end = min(start + micro_bsz, total)
    patch_start = cum[start]
    patch_end   = cum[end]
```

---

## 참고 — 검토 필요하지만 수정 판단 필요

### [ ] REF-1. 이미지 LRU 캐시 효과 측정
**파일**: `model/datasets.py:21-40`, `config.yaml:22`

`image_cache_size: 1000`이지만 num_workers=4 환경에서는 워커별 독립 캐시. 캐시 hit 시에도 bbox drawing + resize는 매번 반복됨. shuffle=True로 히트율이 낮을 가능성이 높음.

**검토 방향**: 히트율 계측 후 낮으면 `image_cache_size: 0`으로 끄거나, offline pre-processing 고려. 무조건 제거보다 측정 먼저.

---

### [ ] REF-2. max_text_length 설정의 실제 미적용 확인
**파일**: `model/utils/processor_collate.py:208-213`, `config.yaml:78`

`max_text_length: 256`이 config에 있으나 `processor()` 호출에는 `truncation=False`로 고정되어 있어 실제로 적용되지 않음. VLM 이미지 토큰 정렬 때문에 truncation을 끈 의도는 맞으나, config 옵션이 무효화된 상태. 문서화 또는 제거 필요.

---

## 완료 기준

| 작업 | 예상 효과 | 완료 |
|------|----------|------|
| TIM-1 구간 타이밍 로그 | 수정 전후 수치 비교 기반 마련 | [x] |
| P0-1 val 주기 변경 | 에폭 시간에서 val generation 비중 제거 | [x] |
| P1-1 deterministic 제거 | GPU 처리량 20~50% 회복 | [x] |
| P1-2 CE valid-union 1회 | [B*L,V] CE 3번 → valid token 위치 1번 | [x] |
| P1-3 argmax 조건부 | 매 배치 불필요한 [B,L,V] argmax 제거 | [x] |
| P2-1 재토크나이즈 제거 | 배치당 tokenizer 8회 → 1회 | [x] |
| P2-2 convert_ids 배치화 | 1줄 수정, 즉시 적용 | [x] |
| P2-3 StoppingCriteria 텐서화 | generation step당 GPU→CPU 변환 제거 | [x] |
| P3-1 trainer_state load 제거 | eval I/O 감소 | [x] |
| P3-2 val_metric_loader 워커 감소 | 상시 메모리 압박 감소 | [x] |
| P3-3 last/ 저장 주기 제한 | 에폭당 반복 I/O 감소 | [x] |
| P4-1 prefix sum 누적합 (RL) | RL 경로 minor 개선 | [x] |
