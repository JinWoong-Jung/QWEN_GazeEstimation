# SFT 수정 계획

## 배경

- 훈련 샘플: 108,955개, 전부 reasoning 파일 존재 확인 (`data/gazefollow_reason_train/`, 108,955개 1:1 커버리지)
- 네 가지 목표 변경 사항 (독립적이므로 순서대로 구현 가능)

---

## Task 1 — Full Dual-View Training (217,910 samples/epoch)

### 목표
- 현재: `WeightedRandomSampler(num_samples=108955)`로 매 에폭당 108,955개만 샘플링 (direct 80% + reasoning 20% 비율)
- 변경: 모든 뷰(direct 108,955 + reasoning 108,955 = **217,910**)를 매 에폭에 전부 학습

### 수정 파일

**`model/trainer.py`** (lines ~1542-1554)
- `WeightedRandomSampler` 생성 블록 제거
- `_train_sampler = None` 고정 (DataLoader가 `shuffle=True`로 직접 전체 217,910개를 섞음)
- `train_records` → `train_ds` 기준으로 스텝 수 재계산하는 부분 확인 (현재 `updates_per_epoch`가 `len(train_loader)` 기준이라면 자동 반영됨)

**`sft.yaml`**
- `paths.train_reasoning_dir`: `"data/gazefollow_reason_train"` 사용
- `reasoning.direct_view_ratio` / `reasoning.reasoning_view_ratio`: 삭제하거나 주석 처리 (코드에서 더 이상 사용 안 됨)

### 주의
- 에폭당 스텝 수가 2배 증가 → `warmup_ratio` 기반 warmup steps 자동 증가 (문제없음)
- `epochs: 20` → 실질적 학습량이 2배 → 필요 시 `epochs: 10`으로 줄이는 것 고려

---

## Task 2 — 출력 순서 변경: reasoning → point → object

### 목표
- 현재 포맷 (`reasoning_object_point`):
  ```
  Reasoning: <text>
  Object: <obj_KKK>
  Point: <loc_NNN><loc_MMM>
  ```
- 변경 포맷 (`reasoning_point_object`):
  ```
  Reasoning: <text>
  Point: <loc_NNN><loc_MMM>
  Object: <obj_KKK>
  ```
- 근거: primary metric이 point distance → point 예측을 autoregressive chain의 앞쪽에 배치 → object는 reasoning+point 양쪽 모두 conditioning

### 수정 파일

#### (A) `model/utils/gaze_tokens.py`

1. **`_STRICT_RE_REASONING_POINT_OBJ` 정규식 추가** (line ~45 근처에 삽입):
   ```python
   # reasoning_point_object: "Reasoning: ...\nPoint: <loc_X><loc_Y>\nObject: <obj_K>"
   # Groups: 1=loc_x, 2=loc_y, 3=obj
   _STRICT_RE_REASONING_POINT_OBJ = re.compile(
       r"^\s*Reasoning:.*?"
       r"\s*Point:\s*(<loc_\d+>)(<loc_\d+>)"
       r"\s*Object:\s*(<obj_\d+>|<obj_unknown>)"
       r"\s*(?:<\|im_end\|>)?\s*$",
       re.DOTALL,
   )
   ```

2. **`build_structured_target_text()` — 신규 케이스 추가** (line ~178 `if order == "reasoning_object_point":` 블록 다음에):
   ```python
   if order == "reasoning_point_object":
       reasoning_body = normalize_reasoning_text(str(reasoning_text or "").strip())
       if reasoning_body or bool(force_reasoning_format):
           content_line = f"{REASONING_PREFIX} {reasoning_body}" if reasoning_body else REASONING_PREFIX
           return f"{content_line}\n{point_str}\n{object_str}"
       return f"{point_str}\n{object_str}"
   ```
   - docstring에 `"reasoning_point_object"` 항목 추가

3. **`parse_structured_output_text()` — 우선순위 삽입** (line ~343 `_STRICT_RE_OBJ_FIRST.match` 다음, reasoning_first 전에):
   ```python
   # reasoning_point_object: "Reasoning: ...\nPoint: ...\nObject: ..."
   m = _STRICT_RE_REASONING_POINT_OBJ.match(s)
   if m is not None:
       return _extract_from_match(m, obj_group=3, x_group=1, y_group=2,
                                  num_classes=num_classes, coord_bins=coord_n)
   ```
   - 함수 docstring에 새 포맷 언급 추가

#### (B) `model/datasets.py`

- `MultiViewGazeDataset.__getitem__()` line 443:
  ```python
  # 기존:
  target_order = "reasoning_object_point"
  # 변경:
  target_order = "reasoning_point_object"
  ```

#### (C) `sft.yaml`

- `reasoning.target_order`: `"reasoning_object_point"` → `"reasoning_point_object"`
- `prompt.prompt_text` — Point/Object 순서 교체:
  ```yaml
  prompt_text: |
    The head box [{loc_x1}{loc_y1}{loc_x2}{loc_y2}] marks the person.
    Briefly reason about where this person is looking, then output the gaze point and object.
    Use exactly one x-bin and one y-bin token, each in the range <loc_{coord_min:03d}> to <loc_{coord_max:03d}>.
    Use exactly one object token in the range <obj_{obj_min:03d}> to <obj_{obj_max:03d}>.
    Return exactly:
    Reasoning: <your reasoning here>
    Point: <loc_NNN><loc_MMM>
    Object: <obj_KKK>
  ```
- `eval.generation_max_new_tokens`: `24` → `80`
  - 근거: reasoning(~30 words ≈ 40 tokens) + "Reasoning: \n" (4 tokens) + "Point: <loc_X><loc_Y>\n" (6 tokens) + "Object: <obj_K>" (3 tokens) ≈ 53 tokens 최소 필요 → 여유 있게 80

#### (D) `model/utils/processor_collate.py`

- `build_structured_masks()` 내 reasoning content offset 탐지 로직 (`elif rl_pos >= 0:` 블록, line ~217):
  - 현재 구현이 "Reasoning: 이후 첫 번째 줄바꿈까지" 를 reasoning 마스크로 지정 → 그 뒤 `<loc_*>` / `<obj_*>` 는 tok_str 패턴 매칭으로 처리
  - `reasoning_point_object` 순서에서도 동작 확인: Point가 Object 앞에 와도 `_LOC_RE` / `_OBJ_RE` 패턴 매칭 기반이라 **코드 수정 불필요** (이미 order-agnostic)
  - 단, `"reasoning_point_object"` 에서 direct view의 `target_order = "point_object"` 유지 확인 필요 (Point → Object 고정)

---

## Task 3 — Gaussian Soft-Label CE for Point Loss (σ=7)

### 목표
- 현재: `<loc_33>` vs `<loc_34>` 예측 → 표준 CE에서 동일하게 틀린 것으로 처리
- 변경: GT bin `b*` 기준으로 Gaussian 분포 soft label 생성 → 인접 bin에 gradient 부여
- σ=7 (128-bin 기준, 좌표 오차 ~7/127 ≈ 0.055 단위의 유연도)

### 구현 원리
```
soft_label[k] = exp(-0.5 * (k - b*)² / σ²),  k ∈ {0..C-1}
soft_label = soft_label / sum(soft_label)

L_pt = -Σ_k soft_label[k] * log_softmax(logits)[loc_token_ids[k]]
```
- `log_softmax`는 전체 vocab에 대해 계산, loc token 위치만 사용

### 수정 파일

#### (A) `model/utils/loss_utils.py`

1. **`gaussian_soft_label_ce()` 함수 추가** (파일 하단 또는 `masked_token_ce` 다음):
   ```python
   def gaussian_soft_label_ce(
       logits_at_pt: torch.Tensor,    # [N, V] — logits at N point-token positions
       gt_loc_ids: torch.Tensor,      # [N]    — GT loc token IDs in vocab
       loc_token_ids: torch.Tensor,   # [C]    — all loc vocab IDs, sorted by bin index
       sigma: float = 7.0,
   ) -> torch.Tensor:
       """Gaussian soft-label cross-entropy over loc token bins.

       Treats each loc token as a bin; nearby bins get partial credit.
       """
       C = int(loc_token_ids.shape[0])
       N = int(logits_at_pt.shape[0])
       if N <= 0 or C <= 0:
           return torch.zeros((), device=logits_at_pt.device, dtype=logits_at_pt.dtype)

       # GT bin index (0..C-1): position of gt_loc_ids within loc_token_ids
       loc_ids_dev = loc_token_ids.to(device=logits_at_pt.device)
       gt_bin = (gt_loc_ids.to(device=logits_at_pt.device).unsqueeze(1)
                 == loc_ids_dev.unsqueeze(0)).float().argmax(dim=1)  # [N]

       # Gaussian soft labels [N, C]
       k = torch.arange(C, device=logits_at_pt.device, dtype=logits_at_pt.dtype)
       diff = k.unsqueeze(0) - gt_bin.to(logits_at_pt.dtype).unsqueeze(1)  # [N, C]
       soft = torch.exp(-0.5 * diff ** 2 / (float(sigma) ** 2))
       soft = soft / soft.sum(dim=1, keepdim=True)                          # [N, C]

       # log_softmax over full vocab, slice to loc positions
       log_p = F.log_softmax(logits_at_pt, dim=-1)[:, loc_ids_dev]          # [N, C]
       return -(soft * log_p).sum(dim=-1).mean()
   ```

2. **`compute_structured_loss()` 수정** — パラメータ 추가 및 point CE 교체:
   ```python
   def compute_structured_loss(
       *,
       logits, labels,
       loss_mask_point, loss_mask_object, loss_mask_format,
       loss_mask_reasoning=None,
       weight_point=1.0, weight_object=1.0, weight_format=0.25, weight_reasoning=0.3,
       compute_format_rate=False,
       loc_token_ids: torch.Tensor | None = None,   # 추가
       gaussian_sigma: float = 0.0,                  # 추가
   ) -> dict[str, Any]:
   ```
   - `valid_union` CE 계산 후 `l_pt` 계산 부분 교체:
     ```python
     if loc_token_ids is not None and float(gaussian_sigma) > 0.0 and n_pt > 0:
         pt_positions = pt_valid[valid_union]  # bool mask within valid_union
         logits_at_pt = shift_logits[valid_union][pt_positions]   # [N_pt, V]
         gt_ids_at_pt = shift_labels[valid_union][pt_positions]   # [N_pt]
         l_pt = gaussian_soft_label_ce(logits_at_pt, gt_ids_at_pt, loc_token_ids, sigma=gaussian_sigma)
     else:
         l_pt = _cat_mean(pt_valid, n_pt)   # 기존 hard CE
     ```

3. **`compute_answer_loss()` 수정** — 파라미터 스루:
   ```python
   def compute_answer_loss(
       *, logits, labels,
       ...,  # 기존 파라미터
       loc_token_ids: torch.Tensor | None = None,
       gaussian_sigma: float = 0.0,
   ) -> dict[str, Any]:
   ```
   - `compute_structured_loss()` 호출 시 `loc_token_ids=loc_token_ids, gaussian_sigma=gaussian_sigma` 전달

#### (B) `model/trainer.py`

1. **`loc_token_ids` 텐서 구성** (`token_id_map` 직후, line ~1201 근처):
   ```python
   # loc tokens: <loc_000> ... <loc_127> in bin order
   _loc_token_id_list = []
   from .utils.gaze_tokens import format_loc_token, _loc_token_width
   _lw = _loc_token_width(coord_bins)
   for _b in range(coord_bins):
       _tok = format_loc_token(_b, _lw)
       _id = int(token_id_map.get(_tok, -1))
       if _id >= 0:
           _loc_token_id_list.append(_id)
   loc_token_ids_tensor: torch.Tensor | None = (
       torch.tensor(_loc_token_id_list, dtype=torch.long)
       if len(_loc_token_id_list) == coord_bins else None
   )
   ```

2. **`gaussian_point_sigma` config 읽기**:
   ```python
   gaussian_point_sigma = float(getattr(args, "gaussian_point_sigma", 0.0))
   ```

3. **SFT 학습 루프 내 `compute_answer_loss()` 호출 수정** (line ~1881):
   ```python
   losses = compute_answer_loss(
       ...
       loc_token_ids=loc_token_ids_tensor.to(device) if loc_token_ids_tensor is not None else None,
       gaussian_sigma=gaussian_point_sigma,
   )
   ```

#### (C) `sft.yaml`

```yaml
loss:
  loss_point_weight: 3.0
  loss_object_weight: 1.0
  loss_format_weight: 0.2
  loss_reasoning_weight: 0.05
  gaussian_point_sigma: 7.0    # 추가. 0.0이면 기존 hard CE 사용
```

---

## Task 4 — train/FormatValidRate를 view type별로 분리 로깅

### 문제
현재 `train/FormatValidRate`는 `compute_structured_loss` 내부에서 배치 전체를 대상으로 `masked_sample_exact_rate(logits, labels, loss_mask_format)`를 호출해 계산됨.

**Mixed batch(direct + reasoning 혼재)에서 두 종류의 포맷 토큰이 섞임:**
- **Direct view**: `"Point: \nObject: "` 구조 → 포맷 토큰 ~4-6개
- **Reasoning view**: `"Reasoning: ...\nPoint: \nObject: "` 구조 → 포맷 토큰 더 많음 (reasoning prefix + 줄바꿈 등)

`masked_sample_exact_rate`는 **모든 포맷 토큰이 완벽히 맞아야** 해당 샘플을 valid로 처리하는 per-sample exact match. 따라서:
- Direct 샘플: 토큰 수가 적으므로 구조적으로 유리 → 높은 rate
- Reasoning 샘플: 토큰 수가 많을수록 불리 → 낮은 rate
- **혼합 비율이 바뀌면 FormatValidRate가 의미 없이 흔들림** → 학습 진행 모니터링 불가

### 수정 파일

#### `model/trainer.py` — wandb 로깅 블록 수정 (line ~1927)

1. **import 추가** (파일 상단 line 63):
   ```python
   from .utils.loss_utils import compute_answer_loss, masked_sample_exact_rate
   ```

2. **`if should_step: if wandb_run is not None:` 블록 내**, `_view_types` 사용 부분(line ~1927) 확장:
   ```python
   _view_types = batch.get("view_type", [])
   _n_batch = max(len(_view_types), 1)
   _n_direct_batch = sum(1 for v in _view_types if v == "direct")
   _n_rsn_batch    = sum(1 for v in _view_types if v == "reasoning")

   # Per-view-type FormatValidRate (only at logging steps, no gradient needed)
   _fmt_rate_direct = 0.0
   _fmt_rate_rsn    = 0.0
   if _compute_fmt_rate and _view_types:
       _logits_det = out["logits"].detach()   # graph freed after backward, tensor still alive
       _fmt_mask   = batch.get("loss_mask_format")
       if torch.is_tensor(_fmt_mask):
           with torch.no_grad():
               _d_idx = [i for i, v in enumerate(_view_types) if v == "direct"]
               _r_idx = [i for i, v in enumerate(_view_types) if v == "reasoning"]
               if _d_idx:
                   _fmt_rate_direct, _ = masked_sample_exact_rate(
                       _logits_det[_d_idx], labels[_d_idx], _fmt_mask[_d_idx]
                   )
                   _fmt_rate_direct = float(_fmt_rate_direct.item())
               if _r_idx:
                   _fmt_rate_rsn, _ = masked_sample_exact_rate(
                       _logits_det[_r_idx], labels[_r_idx], _fmt_mask[_r_idx]
                   )
                   _fmt_rate_rsn = float(_fmt_rate_rsn.item())
   ```

3. **`wandb_run.log(...)` 딕셔너리에 항목 추가**:
   ```python
   "train/FormatValidRate_direct":    _fmt_rate_direct,
   "train/FormatValidRate_reasoning": _fmt_rate_rsn,
   ```
   - 기존 `"train/FormatValidRate"` 키는 유지 (combined, backward compat)

4. **`wandb_utils.py`** — `define_metric` 등록 추가 (line ~67 근처):
   ```python
   wandb.define_metric("train/FormatValidRate_direct",    summary="max")
   wandb.define_metric("train/FormatValidRate_reasoning", summary="max")
   ```

### 주의
- `out["logits"]`는 `loss.backward()` 이후에도 텐서 참조가 살아있어 접근 가능. 단, computation graph는 해제됨 → `torch.no_grad()` 필수
- `_fmt_mask`는 CPU에 남을 수 있으므로 view index는 Python list로 적용 (`_fmt_mask[_d_idx]`), GPU index 텐서로 CPU 텐서를 indexing하지 않음
- `_compute_fmt_rate`는 accum step + wandb 활성 시에만 True → 매 step마다 추가 연산 없음
- Direct 샘플만 있는 배치나 reasoning 샘플만 있는 배치에선 해당 rate가 0.0으로 로깅됨 (n=0 케이스) → wandb에서 필터링 시 주의

---

## 수정 순서 권장

1. **Task 2** (출력 순서) — 가장 단순, 효과 즉시 확인 가능
2. **Task 3** (Gaussian CE) — loss_utils 독립 변경, 테스트 작성 용이
3. **Task 4** (FormatValidRate 분리) — trainer 로깅 수정만, 학습 영향 없음
4. **Task 1** (전체 dual-view) — 학습 시간 증가, 나머지 완료 후 진행

## 테스트 체크리스트

- [x] `tests/test_gaze_tokens.py`: `reasoning_point_object` 포맷 생성/파싱 확인
- [x] `tests/test_structured_loss.py`: `gaussian_soft_label_ce` 단위 테스트 추가
- [x] `tests/test_reasoning_masks.py`: `build_structured_masks` 기본 reasoning 마스크 동작 확인
- [ ] `QWEN_DEBUG_MASKS=1` 환경변수로 학습 초반 마스크 분포 모니터링

## 미결 검토 사항

- `generation_stop_at_object_end: false` 유지 (현재 설정). reasoning_point_object에서 object가 마지막이므로 자연 EOS에 맡기면 충분
- Task 1 적용 후 `epochs: 20` → 실질 학습량 2배 증가. 현재 `sft.yaml`은 `epochs: 10`으로 조정됨
- Gaussian σ=7은 128-bin 기준으로 이웃 14개 bin(±7)에 유효한 gradient 부여 → 너무 soft해지지 않는지 초반 실험 필요 (loss_point_weight=3.0으로 이미 가중치 높음)
