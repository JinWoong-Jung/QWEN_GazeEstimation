# TODO

## Teacher Reasoning Embedding Auxiliary Loss 방향성

- GPT-5.2가 train set에 대해 생성한 `Object:` / `Reasoning:` 텍스트를 추론 시 입력으로 쓰지 않고, 학습 중에만 teacher signal로 활용한다.
- 최종 Qwen 출력 형식은 계속 text-only로 유지한다.
  - 예: `Point: x y` + `Object: <object name>`
  - reasoning 문장을 inference output에 강제하지 않는다.
- teacher text는 sample별 reasoning embedding으로 변환해 둔다.
  - 우선은 `Object + Reasoning`을 하나의 teacher description으로 보고 embedding화하는 방향이 자연스럽다.
  - embedding encoder는 현재 label bank와 맞추기 쉬운 CLIP text encoder를 1차 후보로 둔다.
- Qwen은 이미지와 prompt를 보고 답변을 생성하기 직전의 내부 표현이 teacher reasoning embedding과 가까워지도록 auxiliary loss를 받는다.
  - 핵심은 “정답 문자열을 더 길게 외우게 하기”가 아니라, gaze 판단에 필요한 시각적/의미적 단서를 내부 표현에 이식하는 것이다.
- 전체 학습 objective는 기존 answer-token CE를 주 loss로 유지하고, teacher reasoning alignment는 작은 weight의 보조 loss로 둔다.
  - auxiliary weight는 처음에는 작게 시작해서 Point/Object 생성 성능을 해치지 않는지 확인한다.
- validation/test 및 실제 추론 경로는 기존처럼 reasoning 없이 동작해야 한다.
  - teacher embedding과 auxiliary head는 train-time regularizer로만 취급한다.
- 기대 효과:
  - 4B Qwen이 GPT-5.2의 gaze reasoning 단서(시선 방향, 머리 방향, 대상 후보, 상황 맥락)를 간접적으로 흡수한다.
  - text-only answer format은 유지하면서 recognition/localization 판단력을 보강한다.
- 주요 리스크:
  - auxiliary loss가 너무 강하면 answer CE 학습을 방해할 수 있다.
  - teacher reasoning이 길거나 noisy하면 embedding supervision도 흐려질 수 있다.
  - answer token hidden state를 사용하면 target leakage가 생길 수 있으므로, 답변 시작 직전 representation을 쓰는 방향을 우선 검토한다.
