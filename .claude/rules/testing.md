---
description: 테스트 파일 작업 시에만 로드. pytest 기반 단위 테스트 규칙 및 기존 테스트 설명.
globs: ["tests/**/*.py", "tests/*.py"]
---

# Testing Guide

## 테스트 환경

- 프레임워크: `pytest`
- 테스트 경로: `tests/`
- 실행: `pytest tests/ -v`
- GPU 불필요: 모든 단위 테스트는 CPU에서 실행 가능하게 작성

## 기존 테스트 파일 설명

### `tests/test_full_generation.py`

순수 텍스트 생성 파이프라인의 핵심 유틸리티를 검증.

| 테스트 클래스 / 함수 | 검증 대상 | 핵심 케이스 |
|---------------------|-----------|-------------|
| `TestObjectLabelSpan` | `object_label_span()` | pure-text / legacy slot / 빈 값 / 대소문자 |
| `TestParseObjectText` | `parse_object_text()` | 정상 추출 / `<obj_emb>` → None / 노이즈 입력 |
| `TestLabelBank` | `LabelBank.topk()` | cosine similarity top-k, lookup, canonicalize |
| `TestFormatTargetText` | `format_target_text()` | 포맷 문자열 생성 및 valid 플래그 |
| `TestIdSpaceRetrieval` | `topk_similarity()` id-space 비교 | bank 인덱스 == label id, acc@1/acc@3 계산, invalid target 제외 |

---

### `tests/test_object_tokens.py`

`object_tokens.py`의 `object_label_span()` 함수 집중 테스트.

| 테스트 | 케이스 |
|--------|--------|
| pure-text 레이블 추출 | `"Object: chair"` |
| legacy slot 토큰 | `"Object: <obj_emb>"` |
| 빈 문자열 입력 | `None` 반환 확인 |
| 대소문자 변형 | `"object: Chair"` |

---

### `tests/test_retrieval_loss.py`

`retrieval_ce_full_bank()` 손실 함수 수치 검증.

| 테스트 | 케이스 |
|--------|--------|
| 정상 동작 | `pred_emb [2,2]`, `bank [3,2]`, `labels [0,1]` → 낮은 loss |
| `target_object_valid=0` | valid=0 샘플 제외 → `n_valid=0`, `loss=0` |

---

### `tests/test_special_token_pipeline.py`

`<obj_emb>` 특수 토큰 추가 및 `topk_similarity` 테스트.

| 테스트 | 케이스 |
|--------|--------|
| 토큰 추가 후 vocab 크기 확인 | `add_slot_token()` 후 vocab에 등록, 중복 추가 시 0 반환 |
| `topk_similarity` | bank[0] 에 가장 가까운 query → top-1 = 0 |

---

## 테스트 작성 규칙

### 파일 및 함수 네이밍
- 파일: `test_{module_or_feature}.py`
- 함수: `test_{기능}_{조건}()` 형태
  ```python
  def test_retrieval_ce_valid_samples():
  def test_retrieval_ce_all_invalid():
  def test_object_label_span_empty_string():
  ```

### 픽스처 패턴
```python
# 작은 mock tensor로 빠르게 테스트
pred_emb = F.normalize(torch.randn(2, 512), p=2, dim=-1)
bank = F.normalize(torch.randn(10, 512), p=2, dim=-1)
labels = torch.tensor([0, 1], dtype=torch.long)
```

### 경계 조건 반드시 테스트
- `None` 입력
- 빈 텐서 (`torch.zeros(0, 512)`)
- `valid` 플래그 모두 0인 경우
- `target_label = -1` (invalid) 샘플

### retrieval_ce_full_bank 테스트 패턴
```python
loss, n = retrieval_ce_full_bank(
    pred_object_emb=pred_emb,
    target_label=labels,
    target_object_valid=torch.ones(2),
    label_embedding_bank=bank,
    temperature=0.07,
)
assert n > 0
assert loss.item() >= 0.0
assert not torch.isnan(loss)
```

### id-space acc@1 테스트 패턴
```python
# bank[i] = label id i의 임베딩
bank = torch.zeros(num_classes, dim)
bank[target_id] = ...
topk_ids = topk_similarity(query, bank, k=3, temperature=0.07)
assert topk_ids[0] == target_id          # acc@1
assert target_id in topk_ids[:3]         # acc@3
```

## 테스트하지 않는 것

- GPU 전용 연산 (CI에서 GPU 불가)
- 모델 forward 전체 (단위 테스트 범위 초과)
- 파일 I/O가 필요한 데이터 로더 (실제 데이터 경로 의존)
- wandb 연동

## 주의 사항

- `processor`, `tokenizer` 모킹 시 `return_offsets_mapping` 지원 여부 확인 필요
- Causal LM shift 때문에 `L >= 2` 이상이어야 의미 있는 CE 계산 가능
- `find_subseq(from_right=True)` 사용 이유: 동일 토큰 시퀀스가 시스템 프롬프트에도 등장할 수 있어 마지막 위치를 정답으로 사용
- `retrieval_ce_full_bank`의 분모는 `target_object_valid > 0 AND 0 <= target_label < V` 동시 만족 샘플
