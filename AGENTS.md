# QWEN_GazeEstimation — Agent / Codex Context

## High-Impact Files

코드 변경 시 파급 범위가 넓은 파일들:

- `model/trainer.py` — 전체 학습/평가 파이프라인 진입점
- `model/utils/data_utils.py` — datasets.py, trainer.py에서 공통 임포트
- `model/utils/eval_utils.py` — trainer.py에서 임포트 (평가 지표 계산)
- `model/utils/processor_collate.py` — trainer.py에서 임포트 (배치 처리)
- `model/utils/loss_utils.py` — trainer.py에서 임포트 (손실 계산)
- `model/model.py` — trainer.py, model/__init__.py에서 임포트

## Required Environment Variables

- `PYTHONHASHSEED` — `model/trainer.py` (재현성 보장)

## Key Conventions

- config.yaml은 `flatten_config()`로 평탄화 후 argparse에 주입 → 모든 섹션 키가 최상위로 올라옴
- Retrieval bank: `build_vocab_embedding_matrix(vocab2id=vocab2id)` 결과, `bank[i]` = label id `i`의 임베딩
- acc@1/acc@3/multiacc@1은 text 비교가 아닌 int label id 직접 비교
- 체크포인트: `best/` (최고 성능) + `last/` (최신 에폭) 두 곳에 저장

## Navigation

- 구현 로직 상세: `.claude/rules/code-style.md`
- 테스트 규칙: `.claude/rules/testing.md`
- 전체 개요: `.claude/CLAUDE.md`
