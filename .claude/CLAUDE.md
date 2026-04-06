# QWEN_GazeEstimation — Repository Overview

## Project Summary

VLM 기반 시선 추정 시스템. Qwen3-VL-4B-Instruct 모델을 LoRA fine-tuning하여 주어진 장면 이미지와 피험자 머리 바운딩박스로부터 **시선 좌표**와 **응시 객체 레이블**을 자연어 텍스트로 생성한다. 별도의 분류 헤드 없이 순수 텍스트 생성(full text generation) 방식으로 두 가지 태스크를 단일 모델에서 처리한다.

## Directory Structure

```
QWEN_GazeEstimation/
├── main.py                        # 학습 진입점 (trainer.run() 호출)
├── config.yaml                    # 전체 하이퍼파라미터 및 경로 설정
├── model/
│   ├── model.py                   # QwenTextGenerationModel (LoRA 래퍼)
│   ├── trainer.py                 # 학습/평가 전 파이프라인
│   ├── datasets.py                # GazeDataset, GazeTestDataset, format_target_text
│   ├── modules/
│   │   └── preprocess.py          # resize_scene (이미지 전처리)
│   └── utils/
│       ├── processor_collate.py   # QwenTrainCollator, QwenTestCollator, build_train_inputs
│       ├── loss_utils.py          # compute_structured_losses, masked_token_ce, retrieval_ce_full_bank
│       ├── eval_utils.py          # run_eval, run_test_metrics
│       ├── data_utils.py          # load_records, load_vocab2id, build_vocab_embedding_matrix
│       ├── label_bank.py          # LabelBank (CLIP 임베딩 기반 객체 검색)
│       ├── object_tokens.py       # OBJ_SLOT 토큰, object_label_span
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

| 섹션 | 주요 키 | 설명 |
|------|---------|------|
| `paths` | `model_path` | Qwen3-VL-4B-Instruct 로컬 경로 |
| `train` | `batch_size`, `grad_accum_steps` | 실효 배치 = batch_size × grad_accum_steps |
| `loss` | `loss_answer_weight`, `loss_object_weight` | 각 손실 가중치 |
| `lora` | `lora_r`, `lora_alpha` | LoRA rank/scale (attention proj만 적용) |
| `prompt` | `visual_prompting` | 머리 바운딩박스를 이미지에 빨간 박스로 그림 |
| `text` | `generation_num_beams` | 추론 시 빔 서치 폭 |

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
  logits [B, L, vocab] + pred_object_emb [B, 512]
      ↓
  compute_structured_losses()
      ├── loss_answer : 전체 답변 NLL (weight=1.0)
      ├── loss_point  : 좌표 토큰 NLL  (weight=0.0, 현재 비활성)
      └── loss_object : retrieval CE   (weight=0.3)
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
- **Object Projector**: `nn.Linear(hidden_size → 512)` — hidden state를 CLIP 임베딩 공간으로 투영 후 L2 정규화

## Evaluation Metrics

| 메트릭 | 설명 |
|--------|------|
| `PointL2` | 예측 좌표 ↔ GT 좌표 L2 거리 |
| `ExactMatch` | 생성 객체 레이블 정확 일치 |
| `acc@1 / acc@3` | retrieval top-k 정확도 |
| `multiacc@1` | 다중 GT 레이블 중 하나와 일치 |

## Dependencies

- `transformers` (Qwen3-VL 지원 버전)
- `peft` (LoRA)
- `torch` (bfloat16 학습)
- `Pillow` (이미지 처리)
- `wandb` (실험 로깅)
- CLIP 임베딩은 사전 계산되어 `label-embeds/` 에 `.pt`로 저장됨

## Important Design Decisions

1. **순수 텍스트 생성**: 분류 헤드 없이 좌표와 레이블을 자연어로 출력
2. **구조적 손실 마스킹**: 답변 토큰 내 좌표/객체 구간을 별도 마스크로 분리해 각각 지도
3. **Retrieval 기반 객체 손실**: CLIP 임베딩 뱅크와 코사인 유사도로 객체 인식 (분류 softmax 아님)
4. **VLM truncation 비활성화**: 이미지 토큰 정렬 파괴 방지를 위해 `truncation=False` 고정
5. **target_object_valid 분리**: 좌표 지도는 항상 활성, 객체 손실은 레이블 품질에 따라 개별 비활성화 가능
