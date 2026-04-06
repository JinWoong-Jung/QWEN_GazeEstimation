---
description: Always-load. 모든 모듈의 구현 로직 설명. 코드 수정/추가 시 반드시 참고.
alwaysApply: true
---

# Code Style & Implementation Logic

## 언어 및 스타일 규칙

- Python 3.10+, `from __future__ import annotations` 모든 파일 상단 필수
- 타입 힌트 필수: `torch.Tensor | None`, `dict[str, Any]`, `tuple[int, int]` 형태 사용
- 함수 반환값이 2개 이상이면 `tuple[...]` 명시
- 클래스 메서드는 `*` 이후 keyword-only 인자 권장 (ambiguity 방지)
- 전역 상수는 UPPER_CASE, 로컬 루프 변수는 단문자(`i`, `j`, `m`) 허용
- `float()`, `int()` 명시적 캐스팅: 외부 입력·config 값은 반드시 캐스팅 후 사용
- 배치 처리 시 경계 체크: `if i >= len(list): continue` 패턴으로 방어적 처리

---

## 모듈별 구현 로직

### `model/model.py` — QwenTextGenerationModel

**역할**: Qwen3-VL LoRA 모델 래퍼. 이미지+프롬프트 → 텍스트 토큰/로짓 + 객체 임베딩.

**핵심 구조**:
- `self.qwen`: LoRA 적용된 Qwen3-VL 기반 모델
- `self.object_projector`: `nn.Linear(hidden_size → 512)` — hidden state를 CLIP 공간으로 투영

**`forward()` 흐름**:
1. `joint_inputs`(dict)를 `**kwargs`로 풀어 `self.qwen`에 전달
2. `object_slot_mask`가 주어지면 `output_hidden_states=True`로 강제 설정
3. `pool_object_slot_hidden()` 호출 → 마스크 위치 hidden state 평균 pooling
4. `F.normalize(object_projector(pooled), p=2, dim=-1)` → L2 정규화된 객체 임베딩 반환
5. 반환: `{"loss", "logits", "pred_object_emb", "object_slot_valid"}`

**`pool_object_slot_hidden()`**:
- 입력: `last_hidden [B, L, H]`, `object_slot_mask [B, L]`
- mask가 True인 위치의 hidden state를 평균 → `pooled [B, H]`
- `valid [B]`: 마스크가 하나라도 True인 샘플 표시
- `clamp_min(1.0)`으로 division-by-zero 방지

**`generate()`**:
- `use_cache=True` 고정 (추론 속도)
- `do_sample=False`, `num_beams` 설정으로 빔 서치 사용

**레거시 호환**: `QwenGazeIntegratedModel = QwenTextGenerationModel` alias 유지.

---

### `model/datasets.py` — GazeDataset, GazeTestDataset

**역할**: 어노테이션 레코드 → (이미지, 프롬프트, 타겟 텍스트) 변환.

**`format_target_text()`**:
- 출력 포맷: `"Point: {x:.4f} {y:.4f}\nObject: {label_text}"`
- `label_text`가 없으면 `id2label` 딕셔너리로 역조회
- `vocab2id` / `vocab2id_lower`로 ID 유효성 검증 → `target_valid` (0.0 또는 1.0) 반환
- `num_classes=0`이면 `label_id >= 0`만으로 valid 판정

**`draw_head_bbox_prompt()`**:
- 빨간 박스(`outline=(255, 0, 0)`) 그리기
- `line_w = max(2, round(min(w,h) * 0.006))` — 이미지 크기 비례
- 좌표 범위 벗어나면 scene 그대로 반환

**`GazeDataset.__getitem__()`**:
- `target_text_valid`, `target_point_valid`는 항상 `1.0` (좌표 지도는 항상 활성)
- `target_object_valid`만 레이블 유효성에 따라 `0.0` 가능
- `gt_points`: shape `[1, 2]` (단일 GT)

**`GazeTestDataset.__getitem__()`**:
- 다중 GT points → 평균 좌표 계산 (`px = sum(x)/N, py = sum(y)/N`)
- `gt_points`: shape `[N, 2]` (다중 GT 그대로 보존)
- `target_text_valid = target_object_valid = target_valid` (레이블 품질 연동)

---

### `model/utils/processor_collate.py` — 배치 처리 및 마스킹

**역할**: 이미지+텍스트 배치를 processor로 토크나이즈하고, 손실 마스크 생성.

**`build_train_inputs()` 흐름**:
1. `chat_text()`로 채팅 템플릿 포매팅 (user + assistant 텍스트 포함)
2. `processor(truncation=False)` — VLM 이미지 토큰 정렬 파괴 방지 필수
3. `mask_padding_labels()` — 패딩 토큰을 `-100`으로 마스킹
4. `component_masks()` — answer/point/object 마스크 생성
5. `point_valid <= 0`이면 `point_mask` 비활성, `object_valid <= 0`이면 `object_mask` 비활성
6. `answer_mask`는 `point_valid`에만 연동 (object 무효여도 좌표 CE는 유지)

**`component_masks()` 알고리즘**:
1. 타겟 텍스트를 별도 토크나이즈 → `ans_ids`
2. `find_subseq(seq_ids, ans_ids, from_right=True)` — 전체 시퀀스에서 타겟 위치 탐색
3. `parse_target_spans()`으로 텍스트 내 x좌표/y좌표/객체 레이블의 character span 추출
4. offset mapping으로 character span → token index 변환
5. 각 토큰이 span에 겹치면 해당 마스크를 `True`로 설정

**`find_subseq()`**: 리스트 부분 수열 탐색, `from_right=True`로 마지막 등장 위치 사용.

**`parse_target_spans()`**:
- Point 좌표: `r"(?im)^\s*point\s*:\s*({num})\s*[,\s]+\s*({num})\s*$"` 정규식
- Object: `object_label_span()` 위임 (pure-text / legacy slot 모두 지원)

**`QwenTrainCollator.__call__()`**:
- `target_object_valid` 없으면 `target_text_valid`로 fallback
- `include_raw_inputs=True` 시 원본 이미지/텍스트 함께 반환 (디버깅용)

**`build_infer_inputs()`**: `add_generation_prompt=True`, `assistant_text=None` — 추론 전용.

---

### `model/utils/loss_utils.py` — 손실 함수

**역할**: 구조적 손실 계산. 마스크 기반으로 특정 토큰 구간만 지도.

**`masked_token_ce(logits, labels, mask)`**:
- Causal LM 정렬: `shift_logits = logits[:, :-1, :]`, `shift_labels = labels[:, 1:]`
- `valid = shift_mask & shift_labels.ne(-100)` — 패딩(-100) 제외
- `n_valid == 0`이면 `loss=0, n=0` 즉시 반환 (안전 처리)
- 반환: `(loss_scalar, n_valid_tokens)`

**`retrieval_ce_full_bank(pred_object_emb, target_label, target_object_valid, label_embedding_bank, temperature)`**:
- `pred_object_emb [B, 512]` vs `label_embedding_bank [V, 512]`
- `q @ k.T` → cosine similarity (둘 다 L2 정규화됨)
- temperature scaling 후 cross_entropy (InfoNCE 형태)
- `target_object_valid <= 0`인 샘플은 `valid` 마스크로 제외
- 반환: `(loss_scalar, n_valid)`

**`compute_structured_losses()`**:
- `object_loss_mode ∈ {"retrieval", "infonce", "full_bank_retrieval"}` → retrieval 손실
- 그 외 → `masked_token_ce` (토큰 CE)
- `weight_slot > 0.0`이면 추가로 slot 토큰 CE 계산
- 모든 valid 토큰 수 합이 0이면 `fallback_loss` 사용 (없으면 0)
- 반환 dict 키: `loss`, `loss_answer`, `loss_point`, `loss_object`, `loss_slot`, `n_*_tokens`, `used_fallback`

---

### `model/utils/data_utils.py` — 파일 I/O 및 레이블 맵

**역할**: 어노테이션 파일 파싱, vocab 로딩, CLIP 임베딩 행렬 구축.

**`load_records()`**:
- 탭 구분 txt 파일 파싱: `image_path, person_id, x_gaze, y_gaze, x1, y1, x2, y2`
- `split_prefix` 적용, 이미지 존재 여부 검증
- 반환: `list[Record]` — `Record(image_path, image_rel, gaze_x, gaze_y, bbox_px, label_id, label_text)`

**`load_label_map()`**:
- CSV `(path, id)` → `label_id` 매핑
- fallback: 임베딩 기반 퍼지 매칭 또는 대체 CSV
- 커버리지 통계 출력

**`build_vocab_embedding_matrix()`**:
- `label-embeds/{label_text}-emb.pt` 로드
- 반환: `[vocab_size, 512]` float tensor (없는 레이블은 zero 벡터)

---

### `model/utils/label_bank.py` — LabelBank

**역할**: 추론/평가 시 객체 임베딩 검색용 인메모리 뱅크.

**`LabelBank`**:
- `embeddings [V, 512]`: 전체 vocab 임베딩 행렬
- `labels [V]`: 레이블 텍스트 목록
- `canonical_map`: lowercased 레이블 → id (fuzzy lookup)
- `topk(query, k)`: `query @ embeddings.T` cosine similarity → top-k id 반환
- zero 벡터 임베딩은 검색 시 낮은 유사도로 자연스럽게 제외됨

---

### `model/utils/object_tokens.py` — 특수 토큰

**역할**: `<obj_emb>` 특수 토큰 및 객체 레이블 span 추출.

**`OBJ_SLOT = "<obj_emb>"`**: 레거시 slot 포맷용 특수 토큰.

**`object_label_span(text)`**:
- `"Object: television"` (pure-text) 또는 `"Object: <obj_emb>"` (legacy) 모두 지원
- 정규식으로 `Object:` 이후 레이블 텍스트의 character span `(start, end)` 반환

---

### `model/utils/checkpoint.py` — 체크포인트

**역할**: LoRA 어댑터 + object projector 저장/로드.

**`save_checkpoint()`**: LoRA adapter weights, processor, object projector state dict, trainer 상태를 함께 저장.

**`load_checkpoint_for_eval()`**: 평가 전용 로드. `model.eval()` 자동 호출.

---

### `model/utils/common.py` — 채팅 템플릿

**`chat_text(processor, user_text, assistant_text, with_image, add_generation_prompt)`**:
- Qwen3-VL 채팅 템플릿 포매팅
- `with_image=True`: 이미지 토큰 플레이스홀더 포함
- `add_generation_prompt=True`: 추론용 (assistant 답변 미포함)
- `add_generation_prompt=False`: 학습용 (assistant 답변 포함)

---

### `model/trainer.py` — 학습 파이프라인

**역할**: 전체 학습/평가 루프 관리.

**초기화 순서**:
1. YAML config 로드
2. wandb 초기화
3. Qwen3-VL-4B-Instruct 모델 + processor 로드
4. `<obj_emb>` 특수 토큰 추가, 임베딩 레이어 resize
5. LoRA 어댑터 초기화 (peft)
6. 어노테이션 CSV 로드, vocab2id, label_map 구축
7. CLIP 임베딩 뱅크 구축

**학습 루프**:
- Gradient accumulation: `loss.backward()` 후 `optimizer.step()`은 `grad_accum_steps` 마다
- `dtype=bfloat16` — autocast 사용
- val loss 기준 best checkpoint 저장
- 매 epoch 후 `run_test_metrics()` 로 전체 평가 지표 계산

---

## 공통 패턴 / 주의사항

### Tensor 안전 처리
```python
# 항상 is_tensor 체크 후 사용
if not torch.is_tensor(x) or x.dim() != 2:
    return zero_loss, 0
```

### device/dtype 전파
```python
bank = label_embedding_bank.to(device=pred_object_emb.device, dtype=pred_object_emb.dtype)
```

### 마스크 인덱싱
```python
# Bool mask → float 변환 후 평균 pooling
m_f = mask.to(dtype=hidden.dtype).unsqueeze(-1)
pooled = (hidden * m_f).sum(dim=1) / m_f.sum(dim=1).clamp_min(1.0)
```

### VLM Processor 사용 시
- `truncation=False` 고정 (이미지 토큰 정렬 파괴 방지)
- `return_offsets_mapping=True`로 character-to-token 매핑 활용

### 손실 가중치
- `loss_answer_weight=1.0` (항상 활성)
- `loss_point_weight=0.0` (현재 비활성, answer 손실이 좌표도 포함)
- `loss_object_weight=0.3` (retrieval 손실)
- `loss_slot_weight=0.0` (레거시, 사용 안 함)
