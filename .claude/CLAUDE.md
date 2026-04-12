# QWEN_GazeEstimation — Repository Overview

## Project Summary

VLM 기반 시선 추정 시스템. Qwen3-VL-4B-Instruct 모델을 LoRA fine-tuning하여 주어진 장면 이미지와 피험자 머리 바운딩박스로부터 **시선 좌표**와 **응시 객체 레이블**을 자연어 텍스트로 생성한다. 별도의 분류 헤드 없이 순수 텍스트 생성(full text generation) 방식으로 두 가지 태스크를 단일 모델에서 처리한다.

## Directory Structure

```
QWEN_GazeEstimation/
├── main.py                        # 학습 진입점 (trainer.run() 호출)
├── config.yaml                    # 전체 하이퍼파라미터 및 경로 설정
├── model/
│   ├── model.py                   # QwenTextGenerationModel (LoRA 래퍼, 순수 forward)
│   ├── trainer.py                 # 학습/평가 전 파이프라인
│   ├── datasets.py                # GazeDataset, GazeTestDataset, format_target_text
│   ├── modules/
│   │   └── preprocess.py          # resize_scene (이미지 전처리)
│   └── utils/
│       ├── processor_collate.py   # QwenTrainCollator, QwenTestCollator, build_train_inputs
│       ├── loss_utils.py          # masked_token_ce, retrieval_ce_full_bank, compute_answer_loss
│       ├── eval_utils.py          # CLIPTextEncoder, run_eval, run_test_metrics
│       ├── data_utils.py          # load_records, load_vocab2id, build_vocab_embedding_matrix
│       ├── label_bank.py          # LabelBank (CLIP 임베딩 기반 객체 검색)
│       ├── checkpoint.py          # save_checkpoint, load_checkpoint_for_eval
│       ├── config_parser.py       # YAML 설정 파싱
│       ├── wandb_utils.py         # wandb 로깅 유틸리티
│       └── common.py              # chat_text (채팅 템플릿 포매터)
├── data/
│   └── gazefollow/
│       ├── train_annotations_new.txt / val_annotations_new.txt
│       ├── gaze-labels-train.csv / gaze-labels-val.csv / gaze-labels-test.csv
│       ├── vocab2id.json          # 객체 클래스명 → ID
│       └── label-embeds/          # {label_text}-emb.pt (CLIP [512] 벡터)
└── tests/                         # pytest 기반 단위 테스트
```

## How to Run

```bash
# 학습
python main.py --config config.yaml

# 특정 설정 오버라이드
python main.py --config config.yaml train.lr=5e-5 train.epochs=5
```

## Key Configuration (`config.yaml`)

config.yaml은 8개 섹션으로 구성된다. `flatten_config()`로 평탄화하여 argparse에 주입.

| 섹션 | 주요 키 | 설명 |
|------|---------|------|
| `paths` | `model_path`, `output_dir` | 모델/출력 경로 |
| `data` | `scene_h`, `scene_w`, `image_cache_size`, `split_prefix` | 데이터 로딩 설정 |
| `train` | `batch_size`, `grad_accum_steps`, `epochs`, `lr` | 학습 하이퍼파라미터 |
| `eval` | `test_batch_size`, `generation_num_beams`, `retrieval_top_k`, `clip_model_path` | 평가/추론 설정 |
| `prompt` | `visual_prompting`, `point_decimals`, `answer_template` | 프롬프트 포맷 |
| `model` | `attn_implementation`, `lora_r`, `lora_alpha`, `lora_target_modules` | 모델 구조 |
| `loss` | `loss_answer_weight` | 손실 가중치 |
| `wandb` | `enabled`, `project`, `entity` | 실험 로깅 |

**샘플 개수 제한 없음**: 모든 실행은 전체 데이터셋 기준. `max_train_samples` 등 제거됨.

## Data Flow

```
Scene Image + Head BBox
      ↓  (optional: draw red bbox → visual prompting)
  GazeDataset / GazeTestDataset
      ↓  (resize to 512×512, format prompt text)
  QwenTrainCollator / QwenTestCollator
      ↓  (processor 통해 tokenize + image embedding)
  joint_inputs (input_ids, attention_mask, pixel_values, ...)
      ↓
  QwenTextGenerationModel.forward()
      ↓
  logits [B, L, vocab]
      ↓
  compute_answer_loss()
      └── loss_answer : 전체 답변 NLL (weight=1.0, 유일한 학습 목표)

[Test/Eval time only]
  model.generate() → generated_text
      ↓
  parse_object_text() → "television"   (None if "<obj_emb>" slot)
      ↓
  CLIPTextEncoder.encode(["television"]) → [512] embedding
      ↓
  cosine similarity vs retrieval_label_embedding_bank  → top-k label ids
      │  (bank[i] = embedding of vocab label id i; topk index == label id)
      ↓
  acc@1 / acc@3 / multiacc@1 compared against target_label (int id)
```

## Output Format

모델이 생성하는 텍스트는 항상 아래 포맷:

```
Point: 0.4230 0.7112
Object: television
```

- 좌표는 [0, 1] 정규화된 (x, y)
- `point_decimals: 4` (소수점 4자리)

## Model Architecture

- **Base**: Qwen3-VL-4B-Instruct (frozen, LoRA로 일부 학습)
- **LoRA 대상**: `q_proj`, `k_proj`, `v_proj`, `o_proj`
- **Auxiliary heads 없음**: object projector, point head 모두 없음

## Evaluation Metrics

| 메트릭 | 설명 |
|--------|------|
| `PointL2` | 예측 좌표 ↔ GT 좌표 L2 거리 (텍스트 파싱) |
| `ExactMatch` | 전체 생성 텍스트 정확 일치 (`target_text_valid > 0` 기준) |
| `acc@1 / acc@3` | CLIP retrieval top-k vs `target_label` (int id) 비교. 분모: `target_label >= 0` 샘플 |
| `multiacc@1` | CLIP retrieval top-1 vs `target_label_ids` (int set) 비교 |
| `ObjectParseFail` | "Object: <label>" 파싱 실패율 (`target_text_valid > 0` 기준) |

**id-space 평가**: acc@1/acc@3/multiacc@1은 텍스트 비교가 아닌 정수 label id 직접 비교. retrieval bank가 `vocab2id` 순서로 정렬되어 있으므로 `topk_similarity()`가 반환하는 인덱스 == label id.

## Checkpoint 구조

매 에폭마다 두 위치에 저장:
- `best/` — checkpoint_monitor 기준 최고 성능 모델 (개선 시에만 덮어씀)
- `last/` — 직전 에폭 모델 (매 에폭 덮어씀)

최종 테스트(`run_test=true`)는 항상 `best/`를 로드한 뒤 실행. `best/`가 없으면 in-memory 모델로 fallback.

## Dependencies

- `transformers` (Qwen3-VL 지원 버전, CLIP 포함)
- `peft` (LoRA)
- `torch` (bfloat16 학습)
- `Pillow` (이미지 처리)
- `wandb` (실험 로깅)
- CLIP 임베딩은 사전 계산되어 `label-embeds/` 에 `.pt`로 저장됨

## Important Design Decisions

1. **순수 텍스트 생성**: 분류 헤드 없이 좌표와 레이블을 자연어로 출력
2. **단일 손실(loss_answer)**: 전체 답변 토큰에 대한 teacher-forced NLL만 사용
3. **Test-time CLIP retrieval**: 생성된 "Object: <label>" 텍스트를 CLIP text encoder로 인코딩 → `retrieval_label_embedding_bank`와 cosine similarity → top-k label id 선택
4. **Retrieval bank = vocab2id 순서**: `bank[i]`는 label id `i`의 임베딩. `topk_similarity()` 반환값이 곧 label id이므로 `target_label` tensor와 직접 비교 가능
5. **VLM truncation 비활성화**: 이미지 토큰 정렬 파괴 방지를 위해 `truncation=False` 고정
6. **label-embeds와 CLIP 모델 일치 필수**: `clip_model_path`가 label-embeds 생성에 사용된 CLIP 모델과 동일해야 retrieval이 의미 있음
