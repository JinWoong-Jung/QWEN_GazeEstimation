# QWEN_GazeEstimation

Qwen3-VL + LoRA 기반 gaze-point SFT 파이프라인입니다.

## 핵심 실행

1. 학습
```bash
python main.py
```
`config.yaml`의 `run.mode: train` 기준으로 동작합니다.

2. 최종 학습 아티팩트 평가 (권장)
```bash
python evaluate.py --mode model --artifact final_adapter --max-samples 0
```

3. 특정 체크포인트 평가
```bash
python evaluate.py \
  --mode model \
  --checkpoint-dir /home/elicer/QWEN_GazeEstimation/checkpoints/qwen3vl_gazefollow_lora/checkpoint-3750 \
  --max-samples 0
```

4. zero-shot(base model) 평가
```bash
python zeroshot_gazefollow_eval.py --max-samples 0
```

## 설정 가이드

- 단일 설정 파일: `config.yaml`
- 학습/평가 공통 핵심 섹션:
  - `model`
  - `data_paths`
  - `input`
  - `prompt`
  - `generation`
  - `train`

## 참고

- `train.test_eval.batch_size`로 test 평가 배치 크기를 조절할 수 있습니다.
- `train.test_eval.max_samples: 0`이면 전체 test를 평가합니다.
