다음 작업을 현재 워크스페이스에서 직접 수행하라. 분석만 하지 말고, 가능한 범위에서 실제 코드 수정, 설정 변경, 테스트, 간단한 검증까지 끝내라.

# 목표

`QWEN_GazeEstimation`을 `full text generation` 중심 구조로 리팩터링하라. localization과 object prediction 모두 최종 출력은 순수 텍스트여야 한다. 다만 object recognition은 train/val/test의 label space가 서로 다르므로, pure generation만으로 끝내지 말고 이미 저장되어 있는 CLIP text label embedding bank를 활용한 retrieval/reranking을 함께 설계하고 구현하라.

핵심 방향은 다음과 같다.

1. 출력 형식은 완전한 텍스트로 유지한다.
2. 메인 학습 objective는 assistant answer 전체 토큰에 대한 autoregressive NLL로 둔다.
3. object prediction은 split-specific label bank에 대한 retrieval 문제로도 다룬다.
4. `semgaze`가 사용하는 `predicted embedding -> split-specific vocab bank retrieval` 방식을 참고하되, `QWEN_GazeEstimation`의 generative pipeline에 맞게 자연스럽게 통합한다.

# 반드시 먼저 확인할 참고 코드

아래 파일들을 먼저 읽고 현재 동작을 정확히 파악하라.

- `/home/elicer/semgaze/semgaze/modeling/semgaze.py`
- `/home/elicer/semgaze/semgaze/datasets/gazefollow.py`
- `/home/elicer/semgaze/semgaze/losses.py`
- `/home/elicer/QWEN_GazeEstimation/model/model.py`
- `/home/elicer/QWEN_GazeEstimation/model/trainer.py`
- `/home/elicer/QWEN_GazeEstimation/model/datasets.py`
- `/home/elicer/QWEN_GazeEstimation/model/utils/processor_collate.py`
- `/home/elicer/QWEN_GazeEstimation/model/utils/loss_utils.py`
- `/home/elicer/QWEN_GazeEstimation/model/utils/eval_utils.py`
- `/home/elicer/QWEN_GazeEstimation/config.yaml`

# semgaze에서 반드시 이해해야 할 포인트

`semgaze`는 일반적인 classifier head로 label id를 직접 예측하지 않는다. 대신 다음 흐름을 쓴다.

1. 모델이 `predicted label embedding`을 생성한다.
2. `vocab2id.json`과 `label-embeds/*.pt`로부터 split의 label bank를 만든다.
3. 학습 시에는 predicted embedding을 GT label embedding 또는 GT label id와 비교하여 loss를 계산한다.
4. 추론 시에는 predicted embedding과 vocab embedding bank의 similarity로 logit을 만들고 top-k를 구한다.
5. test에서는 `gaze_label_ids`를 사용해 multi-label metric도 계산한다.

이 아이디어는 `QWEN_GazeEstimation`에 그대로 적용 가능하지만, 최종 출력은 embedding slot이 아니라 자연어 텍스트여야 한다.

# QWEN_GazeEstimation에서 바꾸고 싶은 최종 구조

최종 출력 예시는 다음과 같다.

```text
Point: 0.423 0.711
Object: television
```

필요하다면 out-of-frame을 아래처럼 확장할 수 있다.

```text
Status: out
Object: none
```

또는 in-frame일 때만

```text
Status: in
Point: 0.423 0.711
Object: television
```

단, 현재 turn에서는 최소한 `Point`와 `Object`가 pure text로 생성되도록 우선 구현하라. out-of-frame 문자열은 설계상 자연스럽다면 함께 넣어도 되지만, 무리하게 범위를 넓히진 말라.

# 구현 원칙

1. `full answer NLL`을 메인 loss로 사용하라.
2. 기존의 `Point token만 따로`, `Object slot만 따로` 같은 분해형 supervision은 주 objective에서 내려라.
3. object retrieval는 auxiliary objective 또는 inference reranking으로 유지하라.
4. output은 끝까지 pure text여야 하므로 `<obj_emb>` 같은 placeholder 의존도를 제거하거나 최소한 기본 경로에서 벗겨라.
5. label space mismatch를 고려하여 train/val/test 각각의 label bank를 분리해서 다룰 수 있게 하라.
6. 기존에 저장된 CLIP text label embeddings를 최대한 재사용하라.

# 구체 요구사항

## 1. 데이터 표현

`QWEN_GazeEstimation/model/datasets.py`와 관련 유틸을 수정해서 target text가 실제 label text를 포함하도록 바꿔라.

예:

```text
Point: 0.4230 0.7112
Object: television
```

현재처럼

```text
Object: <obj_emb>
```

를 기본 정답 형식으로 쓰지 마라.

다음도 함께 고려하라.

- train/val/test 각 split에서 사용 가능한 label text를 별도 bank로 만들 수 있게 데이터 경로와 로딩 구조를 정리하라.
- test에는 346개 label만 존재하므로, test split bank를 명시적으로 사용할 수 있어야 한다.
- multi-label 정답이 가능한 경우 semgaze의 `gaze_label_ids`처럼 후보 GT 집합을 유지할 수 있게 하라.

## 2. collator / span 처리

`processor_collate.py`를 수정해서 assistant answer 전체 span을 안정적으로 supervision할 수 있게 하라.

중요:

- main path는 answer 전체 토큰에 대한 NLL이어야 한다.
- object span과 point span을 여전히 찾을 수 있다면 auxiliary retrieval나 diagnostics에 활용하라.
- object text span의 hidden state를 pooling할 수 있게 span mask를 제공하라.
- 현재 구조상 `<obj_emb>` slot 하나만 찾는 로직이 있다면, 이를 `Object:` 뒤 실제 label text span을 처리하는 쪽으로 일반화하라.

## 3. 모델

`model/model.py`를 수정해서 object retrieval auxiliary를 pure-text 구조에 맞게 바꿔라.

권장 방향:

- `Object:` 텍스트 span의 hidden states를 pooling한다.
- 이 pooled hidden을 projector에 통과시켜 CLIP text label embedding space로 보낸다.
- projector output은 normalize한다.

즉, retrieval은 유지하되 `<obj_emb>` 특수 토큰 hidden이 아니라 `Object label text span hidden`을 사용하라.

단, transition 단계에서 backward compatibility가 필요하면 최소한의 호환 코드는 남겨도 된다. 하지만 기본 경로는 pure text여야 한다.

## 4. loss

`loss_utils.py`를 정리해서 loss를 다음처럼 재구성하라.

권장 기본식:

\[
L = L_{answer\_nll} + \lambda_{obj} L_{obj\_retrieval}
\]

여기서:

- `L_answer_nll`: assistant answer 전체에 대한 token-level autoregressive NLL
- `L_obj_retrieval`: pooled object-span hidden projected query와 split-specific CLIP label embedding bank 사이의 retrieval CE

주의:

- point 전용 CE, format 전용 CE, slot 전용 CE가 꼭 필요하지 않다면 과감히 제거하거나 secondary option으로 내리라.
- `loss_use_lm_fallback`처럼 우회 경로가 남더라도, 기본 학습 경로는 full answer NLL이 되게 하라.

## 5. split-specific retrieval bank

`semgaze`를 참고하여 split별 bank를 명시적으로 다뤄라.

필수 요구:

- train bank
- val bank
- test bank

각 bank는 다음을 포함해야 한다.

- ordered label text list
- label text -> id mapping
- normalized CLIP embedding matrix

가능하면 bank 관련 로직을 별도 utility로 분리하라.

## 6. inference

`eval_utils.py`와 `trainer.py`를 수정해서 pure generation + retrieval reranking 조합을 지원하라.

최소 요구 기능:

1. 모델이 `Point`와 `Object`를 텍스트로 생성한다.
2. 생성된 `Object:` 문자열을 파싱한다.
3. test split bank를 기준으로 object prediction을 보정할 수 있게 한다.

보정 방식은 아래 중 실용적인 쪽으로 구현하라.

- 생성된 object text를 CLIP text encoder로 임베딩해 bank retrieval
- object span hidden projector query를 bank retrieval
- 가능하다면 둘을 결합

권장 결합식 예:

\[
S(c) = \lambda_{lm} S_{LM}(c) + \lambda_{clip} \cos(z_{obj}, e_c)
\]

여기서:

- `c`: candidate label
- `S_LM(c)`: `Object: c`에 대한 sequence score 또는 candidate string score
- `z_obj`: object span query
- `e_c`: candidate CLIP text embedding

다만 구현 복잡도가 크다면 1차 버전에서는 아래 우선순위를 따르라.

1. object span hidden -> CLIP bank retrieval
2. 생성된 object text parsing
3. 가능하면 optional reranking

## 7. config 정리

`config.yaml`을 현재 구조에 맞게 정리하라.

예상 변경 포인트:

- `answer_template`
- `prompt_text`
- `object_embedding_dim`
- retrieval 관련 weight
- split-specific bank 경로
- generation 관련 옵션

설정명이 혼란스럽다면 정리해도 된다. 다만 기존 코드와의 연결은 명확히 유지하라.

## 8. 테스트와 검증

가능한 범위에서 최소한 아래를 수행하라.

- unit test 또는 regression test 추가
- object span mask가 실제 label text span을 잡는지 확인
- full answer NLL path가 동작하는지 확인
- retrieval bank shape와 label ordering이 split별로 일관적인지 확인
- generation 결과에서 `Point`/`Object` 파싱이 되는지 확인

기존 테스트가 있으면 갱신하고, 없으면 작은 단위 테스트라도 추가하라.

# 구현 시 주의사항

1. train/val/test label space가 다르다는 점을 절대 무시하지 마라.
2. object recognition을 단순 tokenizer-vocab generation 문제로 축소하지 마라.
3. classifier weight를 고정 클래스 수로 두는 방식으로 퇴행시키지 마라.
4. pure text output이라는 요구를 유지하라.
5. semgaze의 장점인 split-specific retrieval bank 아이디어를 적극 반영하라.

# 권장 구현 순서

1. 현재 코드베이스 읽고 현행 data flow 요약
2. dataset / prompt / answer_template를 pure text 기준으로 변경
3. collator에서 full answer supervision과 object span pooling mask 정리
4. model에서 object span projector query 구현
5. loss를 full answer NLL + object retrieval auxiliary로 재구성
6. split-specific bank 로딩 유틸 정리
7. eval/inference에 retrieval 보정 추가
8. 테스트 추가 및 실행
9. 변경 요약과 남은 리스크 정리

# 최종 산출물

작업이 끝나면 다음을 보고하라.

1. 무엇을 어떤 파일에서 바꿨는지
2. 왜 그렇게 바꿨는지
3. semgaze에서 어떤 아이디어를 가져왔는지
4. full generation 경로가 어떻게 동작하는지
5. retrieval bank가 split마다 어떻게 달라지는지
6. 실제로 돌린 테스트와 그 결과
7. 남아 있는 리스크나 후속 작업

# 추가 힌트

현재 `QWEN_GazeEstimation`에는 이미 retrieval 관련 뼈대가 일부 있다. 이를 버리지 말고 pure text 구조에 맞게 옮겨라.

특히 아래 방향을 우선 검토하라.

- `<obj_emb>` 특수토큰 기반 query 추출 -> `Object label text span` 기반 query 추출로 일반화
- object auxiliary loss는 유지
- main answer loss는 full NLL로 승격
- test bank 346 labels를 inference 때 명시적으로 활용

작업 중 불명확한 점이 생기면, 먼저 코드에서 확인 가능한 사실을 최대한 찾아서 진행하라. 쉽게 물어보며 멈추지 말고, 합리적인 가정을 하고 그 가정을 마지막에 명시하라.
