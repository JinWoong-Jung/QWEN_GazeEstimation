# TODO

## Phase 1: Constrained Decoding 구현 (`eval_utils.py`)

### 1.1 import 확장
- **파일**: `model/utils/eval_utils.py` (L13–16)
- 기존 import에 `GAZE_OBJECT_MARKER`, `GAZE_POINT_MARKER`, `format_loc_token`, `format_obj_token` 추가

```python
from .gaze_tokens import (
    ANSWER_END,
    GAZE_OBJECT_MARKER,
    GAZE_POINT_MARKER,
    format_loc_token,
    format_obj_token,
    parse_structured_output_text,
)
```

---

### 1.2 token ID helper 5종 추가
- **파일**: `model/utils/eval_utils.py` — `# Decode helpers` 섹션(L63) 바로 아래에 삽입
- `_single_token_id(tokenizer, token) → int` — single-token 아니면 ValueError
- `_loc_token_ids(tokenizer, coord_bins) → list[int]`
- `_obj_token_ids(tokenizer, num_classes) → list[int]`
- `_marker_id(tokenizer, marker) → int`
- `_append_token_to_joint(joint, token_ids) → dict` — `input_ids` + `attention_mask`만 확장, image tensor 불변

> 구현 코드는 `GPT_PRO_analysis.md` §3 참조.

---

### 1.3 `_select_next_from_allowed()` 추가
- **파일**: `model/utils/eval_utils.py`
- 허용 token 집합에 대해서만 softmax → argmax (sampling 없음)
- 반환: `(selected_token_ids [B], allowed_probs [B, K])`

> 구현 코드는 `GPT_PRO_analysis.md` §4 참조.

---

### 1.4 `constrained_generate_structured()` 추가
- **파일**: `model/utils/eval_utils.py`
- `target_order` 별 두 가지 경로 지원:
  - `object_point`: `<|gaze_object|><obj_k><|gaze_point|><loc_x><loc_y>`
  - `point_object` (현재 eval 기본값): `<|gaze_point|><loc_x><loc_y><|gaze_object|><obj_k>`
- 새 head 없음. Qwen LM logits만 사용.
- 반환: `list[str]` (batch_decode, skip_special_tokens=False)
- **reasoning body는 생성하지 않음**: val/test의 `_eval_target_order = "point_object"` (trainer.py L1445) 이므로 reasoning 포함 constrained 생성은 이번 범위 외. `reasoning_point_object` 등 미지원 order가 들어오면 **ValueError를 발생**시킨다. 조용히 remapping하면 config 실수가 숨겨지기 때문.

> 구현 코드는 `GPT_PRO_analysis.md` §5 참조.

---

### 1.5 `run_test_metrics()` signature 변경 및 분기 추가
- **파일**: `model/utils/eval_utils.py` (L193–208)
- 파라미터 3개 추가:

```python
constrained_decoding: bool = False,
constrained_target_order: str = "point_object",   # _eval_target_order 기본값과 일치
constrained_temperature: float = 1.0,
```

- 기존 `model.generate()` 블록을 아래 구조로 교체:

```python
if bool(constrained_decoding):
    preds = constrained_generate_structured(...)
else:
    # 기존 generate 경로 유지
```

- metric 계산부(L289 이하) 변경 없음. 기존 `parse_structured_output_text`를 그대로 사용.

> 구현 코드는 `GPT_PRO_analysis.md` §6 참조.

---

### 1.6 `collect_generation_samples()` signature 변경 및 분기 추가
- **파일**: `model/utils/eval_utils.py` (L403–419)
- `run_test_metrics()`와 동일한 3개 파라미터 추가
- generation 블록을 동일하게 분기

> 구현 코드는 `GPT_PRO_analysis.md` §7 참조.

---

## Phase 2: Config 및 Trainer 연결

### 2.0 `config_parser.py` argparse 옵션 추가 ← 없으면 yaml 값이 args에 안 들어옴
- **파일**: `model/utils/config_parser.py` — `# --- model/generation ---` 섹션(L140~) 끝에 추가

```python
p.add_argument("--constrained_decoding", dest="constrained_decoding", action="store_true")
p.add_argument("--no_constrained_decoding", dest="constrained_decoding", action="store_false")
p.set_defaults(constrained_decoding=bool(default_value(d, "constrained_decoding", False)))
p.add_argument("--constrained_target_order", type=str,
               default=str(default_value(d, "constrained_target_order", "point_object")))
p.add_argument("--constrained_temperature", type=float,
               default=float(default_value(d, "constrained_temperature", 1.0)))
```

---

### 2.1 `sft.yaml` eval 섹션에 옵션 추가
- **파일**: `sft.yaml` (eval: 섹션, L58 이하)
- `constrained_target_order`는 **학습 target_order가 아닌** val/test의 `_eval_target_order` 기준으로 설정
- trainer.py L1445: `_eval_target_order = "point_object"` 하드코딩 → 기본값은 `point_object`

```yaml
eval:
  constrained_decoding: false            # 검증 후 true로 전환
  constrained_target_order: point_object # _eval_target_order와 일치 (training target_order와 다를 수 있음)
  constrained_temperature: 1.0
```

> **주의**: `constrained_target_order`는 학습 target (`reasoning_point_object`)이 아니라 eval dataset의 target order(`point_object`)에 맞춰야 함. 두 값이 다른 것이 정상.

---

### 2.2 `trainer.py` 모든 call site 연결
- **파일**: `model/trainer.py`
- `run_test_metrics()` call site 전부 (approx. L988, L1005, L1314, L1764, L2001, L2118, L2197)에 추가:

```python
constrained_decoding=bool(getattr(args, "constrained_decoding", False)),
constrained_target_order=str(getattr(args, "constrained_target_order", "point_object")),
constrained_temperature=float(getattr(args, "constrained_temperature", 1.0)),
```

- `collect_generation_samples()` call site (approx. L389)도 동일하게 연결
- fallback 기본값은 `"point_object"` — `_eval_target_order`와 일치시킬 것

---

## Phase 3: 검증 체크리스트

### 3.1 single-token 검증
- 실행 초기 `ValueError: Expected single-token special token` 에러 → tokenizer에 `<loc_*>`, `<obj_*>`, `<|gaze_point|>`, `<|gaze_object|>` 등록 미완료 의미
- `register_gaze_special_tokens()` + `resize_token_embeddings()` 호출 경로 확인

### 3.2 preview 출력 확인
- constrained + `point_object` order 활성화 후 예상 출력:

```
<|gaze_point|><loc_057><loc_083><|gaze_object|><obj_012>
```

- reasoning body(`<|gaze_reasoning|>...`)는 constrained eval에서 생성되지 않음 — 의도된 동작
- 아래 형태가 나오면 unconstrained 경로가 여전히 동작 중인 것:

```
Point: ...
Object: ...
```

### 3.3 metric 변화 확인
- `FormatValid` → 1.0에 가까워야 함
- `ExtraTextRate` → 0.0에 가까워야 함
- `PointL2ValidFrac` → 1.0에 가까워야 함
- `Avg L2` / `ObjectAcc` — 실제 개선 여부 확인 (constrained decoding은 format 실패를 제거하지, loc distribution 자체를 바꾸지는 않음)

---

## Phase 4: 테스트 추가

### 4.1 `tests/test_constrained_decoding.py` 신규 작성
- `test_single_token_id_raises_on_multi_token` — multi-piece token으로 ValueError 확인
- `test_loc_token_ids_length` — `_loc_token_ids` 반환 길이 == `coord_bins`
- `test_obj_token_ids_length` — `_obj_token_ids` 반환 길이 == `num_classes`
- `test_append_token_to_joint_shape` — input_ids shape 검증, pixel_values 불변 확인
- `test_constrained_generate_structured_point_object` — mock model forward; 출력이 `_ST_POINT_OBJ_RE`에 match되는지 확인 (`point_object` = 현재 eval 기본 order)
- `test_constrained_generate_structured_object_point` — `object_point` order 동일 검증
- `test_point_object_eval_order_matches_trainer` — `constrained_target_order="point_object"` + mock tokenizer로 generate 후 `parse_structured_output_text`가 `valid_format=True` 반환하는지 확인 (eval schema 연동 통합 테스트)

---

## Phase 5: 레거시 / 데드코드 정리

> **Phase 1–4와 별도 커밋/PR로 분리할 것.** eval 동작 변경과 dead code 제거를 섞으면 회귀 원인 추적이 어려워짐.
> 각 항목은 **독립적**으로 적용 가능. 순서 무관.

---

### 5.1 `model/utils/object_tokens.py` 삭제
- 파일 자체가 "Legacy shim — superseded by gaze_tokens.py"라고 명시
- 현재 main 코드(`trainer.py`, `eval_utils.py`)에서 **단 한 곳도** import하지 않음
- 삭제 대상: `model/utils/object_tokens.py`
- 연관 테스트 정리:
  - `tests/test_object_tokens.py` — 전체 삭제 (dead code 테스트)
  - `tests/test_special_token_pipeline.py` — `add_slot_token` import 및 관련 테스트 제거 (나머지 테스트 유지)
  - `tests/test_full_generation.py` — `object_label_span` import(L24) 및 `TestObjectLabelSpan` 클래스(L39–72) 제거

---

### 5.2 `model/utils/label_bank.py` 삭제
- `LabelBank` / `canonicalize` / `topk` — `trainer.py`가 **전혀 import하지 않음**
- 현재 eval pipeline은 `parse_structured_output_text`로 `<obj_NNN>` 토큰을 직접 파싱; CLIP retrieval 없음
- 삭제 대상: `model/utils/label_bank.py`
- 연관 테스트 정리:
  - `tests/test_full_generation.py` — `LabelBank` import(L32), `TestLabelBank` 클래스(L217–259) 제거

---

### 5.3 `data_utils.py`에서 `build_vocab_embedding_matrix` 및 `build_split_bank` 제거
- `build_vocab_embedding_matrix` (L571~): CLIP 임베딩 행렬 구성 함수 — `trainer.py` import 없음
- `build_split_bank` (L609~): split별 임베딩 뱅크 구성 함수 — `trainer.py`, `eval_train_dist.py` 모두 import 없음
- 두 함수 및 docstring 제거
- 연관 테스트:
  - `tests/test_full_generation.py` L27의 `build_split_bank` import 및 `TestBuildSplitBank` 클래스(L108–139) 제거 — 이유: `build_split_bank`가 main pipeline에서 미사용 (5.2의 `build_vocab_embedding_matrix`와는 별개 함수이므로 이유를 정확히 명시)

---

### 5.4 `.claude/CLAUDE.md` Output Format 및 Evaluation Metrics 섹션 업데이트
- "Output Format" 섹션이 구 텍스트 포맷(`Point: 0.4230 0.7112 / Object: television`) 기준으로 기술되어 있음
- 현재 eval 포맷(기본): `<|gaze_point|><loc_057><loc_083><|gaze_object|><obj_012>`
- direct/object-first schema도 parser는 지원: `<|gaze_object|><obj_012><|gaze_point|><loc_057><loc_083>`
- "Evaluation Metrics" 섹션이 CLIP retrieval `acc@1 / acc@3 / multiacc@1`을 기술하나, 현재 `eval_utils.run_test_metrics()`는 이를 계산하지 않음
- 현재 실제 메트릭: `FormatValid`, `Avg L2`, `Min L2`, `PointBinExact`, `ObjectAcc`, `MultiAcc@1`, `JointExact`, `ExtraTextRate`, `PointL2ValidFrac`
- Data Flow 섹션의 CLIP retrieval 경로 설명 제거

---

## 정리 후 기대 상태

| 항목 | 전 | 후 |
|------|----|----|
| `object_tokens.py` | legacy shim (main에서 미사용) | 삭제 |
| `label_bank.py` | CLIP 검색 (main에서 미사용) | 삭제 |
| `build_vocab_embedding_matrix` | data_utils.py 내 dead function | 삭제 |
| `build_split_bank` | data_utils.py 내 dead function | 삭제 |
| `test_object_tokens.py` | dead code 테스트 | 삭제 |
| `TestObjectLabelSpan` in test_full_generation | dead code 테스트 | 제거 |
| `TestLabelBank` in test_full_generation | dead code 테스트 | 제거 |
| `TestBuildSplitBank` in test_full_generation | dead code 테스트 (build_split_bank 미사용) | 제거 |
| `config_parser.py` | constrained 옵션 argparse 미등록 | 3개 옵션 add_argument 추가 |
| `run_test_metrics()` | free generation only | constrained/unconstrained 분기 |
| `collect_generation_samples()` | free generation only | constrained/unconstrained 분기 |
| `sft.yaml` eval 섹션 | constrained 옵션 없음 | 3개 옵션 추가 (기본값 point_object) |
| CLAUDE.md | 구 포맷 기준 기술 | 현재 schema 기준으로 수정 |
