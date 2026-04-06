이 문서는 `/home/elicer/QWEN_GazeEstimation/AGENT_PROMPT_full_generation_retrieval.md` 와 함께 사용되는 상세 설계 명세다.

AI coding agent는 반드시 두 문서를 함께 읽고 작업하라.

- `AGENT_PROMPT_full_generation_retrieval.md`: 실행 지시, 작업 범위, 완료 조건
- `AGENT_DESIGN_SPEC_full_generation_retrieval.md`: 상세 설계 의도, 아키텍처 선택 이유, 권장 구현안, 주의점

이 문서는 특히 다음을 명확히 하기 위해 존재한다.

1. 왜 `full generation + retrieval auxiliary`가 필요한지
2. `semgaze`에서 무엇을 가져오고 무엇은 가져오지 않는지
3. `QWEN_GazeEstimation`의 어떤 부분을 어떻게 바꿔야 하는지
4. split마다 다른 label space를 어떻게 안전하게 다룰지
5. 구현 중 의사결정이 필요할 때 어떤 우선순위를 따라야 하는지

# 1. 문제 재정의

현재 목표는 gaze localization과 object recognition을 하나의 VLM answer generation 문제로 다루는 것이다.

하지만 object recognition은 일반적인 closed-set classification이 아니다.

- train label space와 val/test label space가 다르다.
- test에서는 특히 제한된 candidate label bank가 주어진다.
- 따라서 tokenizer vocabulary 위에서 자유 생성만 잘한다고 recognition 문제가 해결되지 않는다.
- 반대로 fixed classifier head를 두면 split-specific label space mismatch를 흡수하기 어렵다.

따라서 전체 문제는 아래처럼 재정의해야 한다.

## localization

`Point: x y`를 포함한 answer text 전체를 autoregressive generation으로 모델링한다.

## object recognition

최종 출력은 자연어 문자열 `Object: label_text` 이지만, 내부적으로는 split-specific candidate label bank에 대한 retrieval 문제로도 다룬다.

즉, object recognition은 다음 두 층을 가진다.

1. surface form generation
2. candidate-bank-aware retrieval / reranking

# 2. semgaze에서 가져와야 할 핵심 아이디어

`semgaze`는 여기서 중요한 선행 설계를 이미 가지고 있다.

## 가져와야 할 것

### 2.1 predicted embedding -> bank similarity

`semgaze`는 모델이 직접 class id를 내지 않는다. 대신 predicted semantic embedding을 만들고, 이것을 split의 label embedding bank와 비교한다.

참고 위치:

- `/home/elicer/semgaze/semgaze/modeling/semgaze.py`
- `/home/elicer/semgaze/semgaze/datasets/gazefollow.py`

### 2.2 split-specific bank usage

bank는 `vocab2id.json`과 `label-embeds/*.pt`로 구성되고, 해당 split의 label space에 맞는 ordered bank로 사용된다.

이 설계의 장점:

- classifier weight에 클래스 집합을 bake-in하지 않는다.
- train/val/test label space mismatch를 bank 교체만으로 대응할 수 있다.
- top-k retrieval, multi-label metric 계산이 쉽다.

### 2.3 multi-label test handling

`semgaze`는 single GT label뿐 아니라 test에서 가능한 여러 GT label id 집합도 별도로 유지한다.

이 아이디어는 `QWEN_GazeEstimation`에도 가져와야 한다. 특히 bank reranking 후 top-k 평가를 하려면 필요하다.

## 가져오지 말아야 할 것

### 2.4 heatmap-head 중심 구조

`semgaze`의 dense heatmap decoding은 이번 리팩터링의 중심이 아니다.

우리는 `QWEN_GazeEstimation`에서:

- VLM decoder
- prompt + image conditioning
- autoregressive answer generation

을 유지한다.

즉, `semgaze`에서 가져오는 것은 **recognition을 bank retrieval로 푸는 전략**이지, heatmap 모델 자체는 아니다.

# 3. 최종 목표 아키텍처

아래는 권장 최종 구조다.

## 입력

- scene image
- subject head bbox 또는 visual prompt
- textual instruction prompt

## 출력

권장 기본 형식:

```text
Point: 0.423 0.711
Object: television
```

확장 형식 후보:

```text
Status: in
Point: 0.423 0.711
Object: television
```

또는

```text
Status: out
Object: none
```

하지만 첫 번째 구현 단계에서는 `Point/Object`만 우선 안정적으로 가는 것이 더 중요하다.

## 내부 supervision

메인:

\[
L_{main} = L_{answer\_nll}
\]

보조:

\[
L_{aux} = \lambda_{obj} L_{obj\_retrieval}
\]

최종:

\[
L = L_{answer\_nll} + \lambda_{obj} L_{obj\_retrieval}
\]

이때 `L_obj_retrieval`은 Qwen hidden에서 추출한 object query와 CLIP text label embedding bank 사이의 retrieval CE다.

# 4. 왜 full answer NLL이 중심이어야 하는가

현재 구조는 부분적으로 이미 text generation을 사용하지만, loss가 `answer/point/object/slot` 같은 구조화된 분해에 기울어져 있다.

이번 설계의 중심은 이것을 뒤집는 것이다.

## 목표

정답 answer 전체를 canonical output으로 취급한다.

즉, 모델은 아래를 통째로 잘 생성해야 한다.

```text
Point: 0.4230 0.7112
Object: television
```

이렇게 해야 얻는 장점:

1. 논문 방향과 일치한다.
2. localization을 별도 회귀 head가 아니라 language-conditioned generation으로 통일할 수 있다.
3. object도 최종적으로는 자연어 text output으로 정리된다.
4. 향후 `Status`, `Reason`, `Candidate list` 등 확장도 쉬워진다.

## 무엇을 약화시킬 것인가

- point token만 따로 강하게 당기는 loss
- `<obj_emb>` 고정 토큰을 뱉게 만드는 slot loss
- 출력 포맷 일부만 supervising하는 설계

이런 것은 필요하면 남겨도 되지만, 기본 경로가 되면 안 된다.

# 5. object recognition을 pure generation만으로 끝내면 안 되는 이유

label space mismatch가 크기 때문이다.

예를 들어:

- train에는 수천 개 label
- test에는 346개 label

이 상황에서 pure free-form generation만 쓰면 다음 문제가 생긴다.

1. 생성 문자열이 canonical label과 미세하게 다를 수 있다.
2. 동의어, 복수형, 띄어쓰기 차이, 표현 흔들림이 발생한다.
3. test candidate bank가 주어져 있어도 모델은 이를 직접 반영하지 못한다.

따라서 object는 아래처럼 다뤄야 한다.

## 바깥 출력

`Object: television`

## 안쪽 판단

- object span hidden 기반 query
- candidate label bank
- 필요시 LM score와 결합한 reranking

즉, 출력은 text지만 판단은 retrieval-aware여야 한다.

# 6. 권장 object query 설계

현재 코드에는 `<obj_emb>` token 위치 hidden을 projector로 보내 retrieval하는 구조가 있다.

이번 리팩터링에서는 이를 아래 방식으로 일반화하는 것을 권장한다.

## 6.1 object text span pooling

assistant answer 안에서 `Object:` 뒤 label text span을 찾는다.

예:

```text
Point: 0.4230 0.7112
Object: television monitor
```

여기서 `television monitor`에 대응하는 token span을 잡는다.

## 6.2 pooled hidden -> projector -> CLIP space

그 span hidden states를 pooling하여 query를 만든다.

예시:

- mean pooling
- attention pooling
- first token pooling

1차 구현은 mean pooling이면 충분하다.

그 뒤 작은 projection head를 통해 CLIP text label embedding space에 정렬한다.

## 6.3 normalize

query와 bank embedding은 모두 L2 normalize한다.

## 6.4 retrieval CE

train 시 해당 split의 bank 위에서 GT label id를 맞추는 CE를 건다.

이 설계가 좋은 이유:

1. 최종 output은 pure text 유지
2. retrieval은 semgaze처럼 split-aware
3. 생성 문자열의 표면 흔들림과 내부 query를 분리할 수 있음

# 7. object query 대안과 우선순위

구현 중 선택지가 있다.

## 옵션 A. generated object text -> CLIP encoder -> bank retrieval

장점:

- 가장 pure text
- 구현이 직관적

단점:

- 생성 문자열 오타/흔들림에 취약
- 추론 때 CLIP text encoding 비용이 추가

## 옵션 B. object span hidden -> projector -> bank retrieval

장점:

- 현재 코드 자산 재사용 가능
- semgaze와 구조적으로 유사
- 생성 문자열의 surface noise에 덜 민감

단점:

- projector를 잘 학습시켜야 함

## 옵션 C. candidate sequence LM reranking

각 candidate `c`에 대해

\[
S_{LM}(c) = \sum_t \log p(c_t \mid image, prompt, prefix, c_{<t})
\]

를 계산해 rerank하는 방식이다.

장점:

- generative model의 언어적 선호를 직접 반영

단점:

- candidate 수가 크면 느릴 수 있음

## 권장 우선순위

1차 구현:

- 메인 generation
- 옵션 B retrieval auxiliary

2차 구현:

- 옵션 C reranking 추가

3차 구현:

- 옵션 A를 calibration 또는 fallback으로 추가

# 8. split-specific bank 설계 명세

retrieval bank는 반드시 split-aware여야 한다.

## 최소 자료구조

split마다 다음 정보를 갖는 구조체 또는 dict를 만들라.

- `label_texts`: ordered list[str]
- `label_to_id`: dict[str, int]
- `id_to_label`: list[str]
- `embedding_matrix`: Tensor [N, D], normalized

가능하면 아래도 포함하라.

- `normalized_label_keys`: canonicalized text mapping
- `multi_label_map`: sample별 가능한 GT label id 집합

## bank source

기본적으로 기존에 저장된 CLIP text label embedding 파일을 사용한다.

중요:

- train/val/test마다 bank가 다를 수 있도록 경로를 분리하라.
- 공통 `vocab2id.json`만 쓰는 구조로 고정하지 마라.
- test의 346 label bank가 명시적으로 로딩되도록 만들어라.

## canonicalization

가능하면 label text 비교 시 canonicalization을 추가하라.

예:

- lower
- strip
- multiple spaces collapse

이 canonical key는 parse된 generated object text를 bank label과 대조할 때 유용하다.

# 9. dataset 수준 설계

현재 `datasets.py`는 target text를 만들고 일부 validity flag를 실어준다.

이번 설계에서는 dataset sample이 아래를 제공하는 방향이 좋다.

## 권장 sample 필드

- `scene_image`
- `text_input`
- `target_text`
- `target_point`
- `target_label_text`
- `target_label`
- `target_label_ids`
- `target_object_valid`
- `target_point_valid`
- `target_text_valid`
- `target_label_emb` 또는 retrieval용 label metadata

## 중요한 설계 포인트

### 9.1 target_text는 실제 label text를 포함

예:

```text
Point: 0.4230 0.7112
Object: television
```

### 9.2 object validity와 point validity는 분리

object label이 없다고 point supervision이 사라지면 안 된다.

이 부분은 과거 구조에서 흔히 섞이기 쉬우므로 특히 주의하라.

### 9.3 test는 multi-label GT를 유지

`semgaze`처럼 test sample에 가능한 GT label id 집합을 유지하면 top-k metric 계산이 쉬워진다.

# 10. collator / span mask 설계

이 부분은 매우 중요하다.

## 목표

1. answer 전체 NLL이 잘 걸려야 한다.
2. object span mask를 auxiliary retrieval에 쓸 수 있어야 한다.
3. point parse와 object parse가 평가에 활용 가능해야 한다.

## 권장 mask 종류

- `answer_mask_full`
- `point_mask` (optional, diagnostics 용도)
- `object_text_mask`

중요:

- `answer_mask_full`이 main loss 대상이다.
- `object_text_mask`는 retrieval query 추출용이다.

## object span 파싱 규칙

기본적으로 `Object:` line의 content span을 잡는다.

예:

```text
Object: television monitor
```

이때 `television monitor`만 포함하고 `Object:` prefix 자체는 제외하는 편이 좋다.

다만 구현 단순성을 위해 prefix 포함 span으로 먼저 가도 괜찮다. 그 경우 retrieval 성능에 악영향이 있는지 확인하라.

# 11. loss 상세 명세

## 11.1 main answer NLL

teacher forcing으로 assistant answer 전체에 대해 causal LM CE를 건다.

이게 주 objective다.

## 11.2 object retrieval auxiliary

입력:

- predicted object query
- GT label id
- split-specific bank

출력:

- CE over candidate bank

형태:

\[
L_{obj} = CE(q E^\top / \tau, y)
\]

여기서:

- \(q\): normalized query
- \(E\): normalized bank embedding matrix
- \(y\): GT label id
- \(\tau\): temperature

## 11.3 weighting

초기 권장:

- `w_answer = 1.0`
- `w_object = 0.25 ~ 0.5`

object auxiliary가 main generation을 압도하지 않게 하라.

## 11.4 optional future extensions

필요하면 다음도 고려 가능하다.

- LM candidate reranking loss
- listwise ranking loss
- contrastive loss with hard negatives

하지만 1차 구현에서는 과하지 않게 가라.

# 12. inference 설계

권장 inference는 두 단계다.

## 12.1 generation

우선 모델이 full answer를 생성한다.

예:

```text
Point: 0.423 0.711
Object: tv
```

## 12.2 object resolution

생성된 object를 최종 canonical label로 정리한다.

가능한 방식:

### 방식 A

generated text parse -> canonical text exact match -> 실패 시 retrieval fallback

### 방식 B

object span hidden query -> test bank retrieval -> top-1 label 채택

### 방식 C

generated object text와 retrieval score를 결합하여 rerank

권장 1차 구현은 B다.  
그 이유:

- 현재 코드 자산을 재사용할 수 있음
- split-specific bank mismatch에 강함
- pure text output을 해치지 않음

즉, 최종 user-facing output은 text로 보여주되, 내부 final object prediction은 bank retrieval 결과로 canonicalize하는 방식이 좋다.

# 13. metric 설계

다음 지표를 유지하거나 추가하라.

## localization

- Point L2
- Min L2
- 가능하면 AUC

## object recognition

- top-1
- top-3
- multi-label top-1
- retrieval valid rate

## generation quality / parsing robustness

- point parse fail rate
- object parse fail rate
- canonical exact match rate

# 14. config 설계 가이드

설정은 아래처럼 분리하는 것이 좋다.

## prompt / answer

- prompt template
- answer template
- point decimals

## retrieval bank

- train label bank path
- val label bank path
- test label bank path
- embedding dim
- temperature

## loss

- answer NLL weight
- object retrieval weight

## inference

- generation max_new_tokens
- num_beams
- rerank enabled
- rerank weights

# 15. 구현 우선순위와 trade-off

작업 중 모든 것을 한 번에 하려 하지 말고 아래 우선순위를 따르라.

## 반드시 구현

1. pure text answer template
2. full answer NLL 메인화
3. object text span pooling
4. split-specific retrieval bank 로딩
5. object retrieval auxiliary
6. test-time object retrieval prediction

## 가능하면 구현

7. canonicalization
8. multi-label top-k metric
9. LM score와 retrieval 결합 reranking

## 후순위

10. out-of-frame textual reformulation 확장
11. constrained decoding
12. advanced listwise ranking

# 16. 구현 중 자주 틀리기 쉬운 부분

## 16.1 train bank와 test bank를 혼용하면 안 됨

train objective와 test inference에서 쓰는 bank는 분리되어야 한다.

## 16.2 object label invalid가 point supervision을 죽이면 안 됨

point와 object validity는 독립적으로 관리하라.

## 16.3 classifier head로 돌아가면 안 됨

고정 크기 linear classifier를 두는 방향은 이 문제에 맞지 않는다.

## 16.4 generated object string만 맹신하면 안 됨

surface mismatch가 있기 때문에 retrieval/canonicalization이 필요하다.

## 16.5 label ordering이 은근히 중요함

bank embedding row order와 label id order가 반드시 일치해야 한다.

# 17. 권장 코드 구조 변경안

아래는 예시 구조다. 꼭 똑같을 필요는 없지만 비슷한 분리가 좋다.

## 새 utility 후보

- `model/utils/label_bank.py`
  - split-specific bank 로딩
  - label text canonicalization
  - embedding matrix 생성

- `model/utils/object_span.py`
  - `Object:` text span 파싱
  - token span mask 생성 보조

## 수정 대상 핵심 파일

- `model/datasets.py`
- `model/utils/processor_collate.py`
- `model/model.py`
- `model/utils/loss_utils.py`
- `model/utils/eval_utils.py`
- `model/trainer.py`
- `config.yaml`

# 18. 최종적으로 기대하는 동작

훈련 시:

1. 모델은 image + prompt를 보고 full answer를 생성하도록 학습됨
2. answer 전체 token NLL이 메인 supervision
3. object text span hidden은 CLIP bank retrieval auxiliary로 추가 정렬됨

검증/테스트 시:

1. 모델이 `Point/Object`를 텍스트로 생성
2. object prediction은 split-specific bank 위에서 retrieval 또는 reranking으로 canonicalize
3. localization metric과 recognition metric을 함께 계산

# 19. 완료 후 보고 형식

작업이 끝나면 아래 형식으로 정리하라.

1. 현행 구조와 새 구조의 차이
2. semgaze에서 가져온 아이디어
3. pure text generation 흐름
4. retrieval bank 흐름
5. train/val/test split handling 방식
6. 테스트 결과
7. 아직 남은 리스크

# 20. 최종 의사결정 원칙

구현 중 갈림길이 있으면 아래 원칙을 따르라.

1. pure text output 유지가 최우선
2. full answer NLL이 main objective여야 함
3. object recognition은 split-specific retrieval-aware여야 함
4. semgaze의 bank retrieval 장점을 최대한 재사용
5. 현재 `QWEN_GazeEstimation` 코드 자산을 활용해 최소한의 파괴로 리팩터링

한 줄로 요약하면:

**이 작업은 `<obj_emb>` 기반 반구조적 출력에서, `full generated answer + split-specific CLIP retrieval-backed object prediction` 구조로 옮기는 리팩터링이다.**
