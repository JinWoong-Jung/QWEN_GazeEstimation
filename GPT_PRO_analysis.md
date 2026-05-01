## 결론

현재 성능 차이가 거의 없다면, 가장 큰 이유는 **reasoning이 실제 좌표 예측의 원인 경로에 들어가 있지 않기 때문**입니다. 공개 레포 기준으로 SFT target은 `Point → Object → Reasoning` 순서입니다. 즉 모델이 좌표 토큰을 생성할 때는 아직 reasoning 토큰을 생성하지 않았고, causal LM 구조상 뒤에 오는 reasoning은 앞의 point 예측을 조건화할 수 없습니다. `sft.yaml`도 `generation_stop_at_object_end: true`라서 평가/추론 시 reasoning 생성을 Object 직후 중단하도록 되어 있습니다. 이 구조라면 reasoning은 “좌표를 잘 찍기 위한 추론”이라기보다 “좌표/객체를 찍은 뒤 붙는 보조 설명”에 가깝습니다. ([GitHub][1])

따라서 지금의 Stage-1은 “reasoning-based SFT”라기보다는 **answer-first SFT + post-hoc rationale auxiliary loss**입니다. 이 자체가 나쁘지는 않지만, gaze point 성능 개선 효과는 제한적일 수밖에 없습니다.

---

## 1. 현재 파이프라인에서 허술해 보이는 지점

### 1.1 Reasoning 위치가 잘못되어 있다

현재 target 생성 코드는 기본 `Point/Object` 답변을 만든 뒤 reasoning block을 append합니다. 공개 코드상 `build_structured_target_text_with_reasoning()`은 `base = Point/Object`를 만든 뒤 `return f"{base}\n{reasoning_block}"` 형태입니다. ([GitHub][2])

이 경우 학습 loss는 reasoning 토큰에도 걸리지만, **좌표 토큰은 reasoning 이전에 생성**됩니다. 즉 reasoning loss가 모델 표현을 약간 regularize할 수는 있어도, “reasoning을 하고 그 결과로 point를 찍는” 동작은 학습되지 않습니다.

개선하려면 둘 중 하나를 선택해야 합니다.

| 목표                        | 추천 출력 구조                                         |
| ------------------------- | ------------------------------------------------ |
| 좌표 성능이 최우선                | reasoning을 빼고 point/object/heatmap에 직접 최적화       |
| reasoning을 실제 의사결정에 쓰고 싶음 | `Reasoning/Evidence → Object 후보 → Point` 순서로 바꾸기 |
| reasoning은 설명 가능성만 필요     | 현재처럼 뒤에 붙이되 성능 향상 기대는 낮게 잡기                      |

실제로 visual grounding 계열 MLLM 연구는 좌표/영역을 언어 생성 내부에 명시적으로 넣습니다. Shikra는 spatial coordinate input/output을 자연어 형태로 다루고, Kosmos-2는 text span과 bounding box location token을 연결하며, Ferret은 discrete coordinate와 continuous region feature를 함께 쓰는 region representation 및 hard negative 데이터를 사용합니다. ([arXiv][3])

---

### 1.2 Object도 Point 뒤에 있어 semantic cue로 작동하지 않는다

현재 출력은 `Point`가 먼저, `Object`가 뒤입니다. gaze target은 보통 “이 사람이 무엇을 보고 있는가”와 “그 물체가 어디 있는가”가 얽힌 문제인데, 지금 구조에서는 object token도 point 생성 이후에 나오므로 point 예측을 조건화하지 못합니다.

`Toward Semantic Gaze Target Detection`은 gaze following을 단순 2D point 예측에서 semantic label까지 함께 예측하는 문제로 확장합니다. 이 논문이 지적하듯, 순수 위치 예측만으로는 실제 응용에서 semantic target 정보를 충분히 제공하지 못합니다. ([NeurIPS Proceedings][4])

권장 구조는 다음 중 하나입니다.

```text
Reasoning: ...
Object: <obj_KKK>
Point: <loc_XXX><loc_YYY>
```

또는 더 강하게는:

```text
Evidence:
- Head direction: <loc_...>
- Candidate object: <obj_KKK>
Final:
Point: <loc_XXX><loc_YYY>
Object: <obj_KKK>
```

다만 이 구조를 쓰면 현재 parser도 바꿔야 합니다. 현재 strict parser는 기본적으로 `Point → Object` 형식을 기대합니다. ([GitHub][2])

---

### 1.3 Reasoning 파일 매칭 구조를 반드시 점검해야 한다

공개 코드 기준으로 reasoning index는 `reasoning_dir.rglob("*.txt")`를 돌며 key를 `"{txt.parent.name}/{txt.stem}"`로 만듭니다. Dataset 쪽에서는 `"{folder}/{stem}_{sample_id}"` 키로 reasoning을 찾습니다. 즉 실제 reasoning 파일이 다음 구조여야 안정적으로 매칭됩니다. ([GitHub][5])

```text
data/gazefollow_reason_train/
  00000006/
    00006029_108517.txt
  00000021/
    00021039_3325.txt
```

그런데 첨부 예시처럼 파일이 단순히 `[image_stem]_[id].txt` flat 구조로 들어가 있으면, 현재 코드에서는 key가 맞지 않아 reasoning이 거의 붙지 않을 가능성이 큽니다. 학습이 실제로 돌아가고 있다면 로컬 코드나 데이터 구조가 공개 레포와 다를 수 있으므로, 우선 아래 값을 로그로 찍어야 합니다.

```python
has_reasoning_count = sum(ds[i]["has_reasoning"] for i in range(len(ds)))
print(has_reasoning_count, len(ds), has_reasoning_count / len(ds))
```

추가로 `QWEN_DEBUG_TARGET_EXAMPLE=1`을 켜서 실제 target이 다음처럼 reasoning을 포함하는지 확인해야 합니다.

```text
Point: <loc_...><loc_...>
Object: <obj_...>
<reasoning block...>
```

---

### 1.4 학습 augmentation과 reasoning 문장이 충돌한다

Dataset은 먼저 train augmentation을 적용한 뒤, reasoning file에서 원본 이미지 기준 reasoning을 읽어 target에 넣습니다. 공개 코드상 train dataset은 `apply_augmentation=True`이고, augmentation에는 crop, horizontal flip, color jitter가 포함됩니다. ([GitHub][6])

이때 reasoning 예시는 “head angled upward and to the right”, “head angled downward and outward” 같은 방향 표현을 포함합니다. Horizontal flip이 걸리면 right/left가 뒤집히고, crop이 걸리면 “background”, “tabletop”, “beside the raft” 같은 문맥이 달라질 수 있습니다. 첨부 예시들도 이런 방향·관계 설명을 많이 사용합니다.  

따라서 reasoning SFT에서는 다음 중 하나를 권합니다.

1. reasoning이 있는 샘플에는 hflip/crop을 끄고, color jitter만 유지한다.
2. augmentation을 유지하되 reasoning을 방향 중립적으로 만든다.
3. 같은 샘플을 두 갈래로 둔다:
   `point-only augmented sample` + `reasoning unaugmented sample`.

지금처럼 원본 reasoning과 변형 이미지가 섞이면, reasoning loss가 오히려 noisy auxiliary task가 됩니다.

---

### 1.5 Reasoning 파일의 `Object:` 라인은 현재 사용되지 않는다

첨부 reasoning 파일은 `Object:`와 `Reasoning:` 두 줄 구조입니다. 예를 들어 “Object: the computer monitor screen / Reasoning: ...” 같은 형태입니다. 

하지만 공개 코드의 `load_reasoning_text()`는 파일에서 `Reasoning:`으로 시작하는 줄만 추출합니다. `Object:` 줄은 target object token 생성에 쓰이지 않습니다. ([GitHub][5])

이건 의도일 수도 있지만, 현재 상태에서는 다음 문제가 생깁니다.

* reasoning object phrase와 `gaze_pseudo_label`의 object token이 불일치해도 검증되지 않는다.
* “the dessert on the plate”와 vocab의 `cake`, “computer monitor screen”과 vocab의 `screen`처럼 semantic granularity가 다를 수 있다.  
* `vocab2id`는 346개 object class로 제한되어 있으므로, fine-grained object phrase를 그대로 살리기 어렵다. 

최소한 reasoning loader에서 object line도 읽고, `gaze_pseudo_label`과 semantic consistency check를 해야 합니다.

---

### 1.6 Point token CE는 거리 정보를 반영하지 못한다

현재 point는 x-bin token 1개, y-bin token 1개로 학습됩니다. `sft.yaml`에서는 `coord_bins: 128`이고, loss는 point/object/format/reasoning mask별 token CE로 계산됩니다. ([GitHub][1])

문제는 token CE가 “인접 bin 오차”와 “반대편 bin 오차”를 동일한 incorrect class로 취급한다는 점입니다. gaze target은 본질적으로 continuous spatial prediction입니다. 따라서 SFT에는 다음 중 하나를 붙이는 게 낫습니다.

* x/y bin에 Gaussian soft label CE 적용
* 2D heatmap decoder 추가 후 Gaussian heatmap MSE/KL loss 적용
* generated loc token 외에 hidden state에서 continuous `(x, y)` regression head를 추가
* Stage-2 RL에서 `exp(-β * L2)` reward를 사용해 거리 기반 보상 보완

Gaze-LLE도 gaze target estimation을 사람 appearance와 scene contents를 함께 reasoning해야 하는 문제로 보고, frozen DINOv2 scene feature와 person-specific positional prompt를 통해 gaze를 decode합니다. 즉 단순 token classification보다는 spatial decoder가 더 자연스러운 접근입니다. ([CVF Open Access][7])

---

## 2. Stage-1 SFT를 강화하는 구체적 방향

### 2.1 먼저 “reasoning 효과”가 있는지 검증하는 ablation을 짜야 한다

현재 run 하나만 보면 이유를 알 수 없습니다. 다음 5개를 같은 seed, 같은 train/val split, 같은 eval script로 돌려야 합니다.

| 실험                                               | 목적                          |
| ------------------------------------------------ | --------------------------- |
| A. no reasoning                                  | baseline                    |
| B. current post-hoc reasoning                    | 현재 방식의 auxiliary 효과         |
| C. reasoning-before-answer                       | reasoning이 point를 조건화하는지 확인 |
| D. current + no hflip/crop for reasoning samples | augmentation 충돌 제거          |
| E. dual-image input: scene + head crop           | fine gaze cue 부족 여부 확인      |

여기서 C가 B보다 좋아지면 reasoning 위치 문제가 핵심입니다. E가 가장 크게 좋아지면 reasoning보다 **head crop 해상도**가 병목입니다.

---

### 2.2 VLM 입력을 “전체 장면 1장”에서 “전체 장면 + head crop”으로 바꾸는 것이 우선순위가 높다

현재는 full scene에 red box visual prompt를 그려 넣는 구조입니다. 하지만 gaze target은 눈/얼굴 방향 cue와 scene cue를 동시에 요구합니다. Gaze-LLE의 문제 정의도 person appearance와 scene contents를 함께 고려해야 한다고 명시합니다. ([CVF Open Access][7])

Qwen-VL 계열은 multi-image input을 처리할 수 있으므로, 다음처럼 구성하는 것이 좋습니다.

```text
Image 1: full scene with red head bbox
Image 2: cropped head/face region, upsampled
Prompt:
The first image is the scene. The second image is the cropped head of the marked person.
Predict the gaze target point in the scene image.
```

이 변경은 reasoning보다 효과가 클 가능성이 큽니다. 현재 512 fixed resize에서 작은 head/eye cue가 손실되면, 모델은 결국 scene prior와 head box 위치 prior에 의존하게 됩니다.

---

### 2.3 Reasoning은 “자연어 설명”보다 “검증 가능한 grounded evidence”로 바꿔야 한다

현재 reasoning 예시는 대체로 다음 형태입니다.

```text
His eyes are angled downward ...
The person’s head is tilted downward ...
The line of sight aligns with ...
```

이건 사람이 보기에는 그럴듯하지만, 학습 관점에서는 검증 가능한 중간 supervision이 약합니다. ViGoRL은 시각 reasoning에서 각 reasoning step을 image coordinate에 anchor하는 방식이 중요하다고 보고, SFT로 grounded reasoning trace를 warm-start한 뒤 GRPO를 적용합니다. 또한 단순 RL은 ungrounded reasoning shortcut으로 빠질 수 있다고 분석합니다. ([arXiv][8])

gaze task에 맞춘 grounded reasoning format은 예를 들면 다음과 같습니다.

```text
Evidence:
Head center: <loc_XXX><loc_YYY>
Coarse gaze direction: down-right
Candidate region: <loc_AAA><loc_BBB>
Candidate object: <obj_KKK>
Point: <loc_XXX><loc_YYY>
Object: <obj_KKK>
```

이렇게 하면 reasoning token이 실제 point/object와 구조적으로 연결됩니다. 반대로 “eyes are angled downward” 같은 자유 텍스트만 넣으면 모델이 그 문장을 외우는 보조 과제에 그칠 수 있습니다.

---

### 2.4 Object vocabulary와 pseudo label 품질을 다시 봐야 한다

첨부 `vocab2id.json`은 346개 class이고, train label 파일은 `path, id, split, gaze_pseudo_label, label_id` 구조입니다.  

내가 첨부 파일 기준으로 간단히 계산해 보면, alias rule 적용 후에도 train pseudo label의 상당수가 346-class vocab에 직접 매핑되지 않습니다. 공개 코드에서는 invalid object일 때 unknown object token으로 target을 만들고 object loss를 끕니다. 이 경우 point는 학습되지만 semantic grounding 신호가 약해집니다.

권장 조치:

* `gaze_pseudo_label → vocab2id` coverage를 로그로 항상 기록한다.
* reasoning file의 `Object:` phrase도 읽어서 pseudo label과 비교한다.
* `paper`, `hut`, `roof`, `dessert` 같은 fine-grained phrase를 346-class로 접는 canonicalization table을 만든다.
* object token을 반드시 맞히는 task가 아니라면, object는 “auxiliary semantic hint”로 쓰고 최종 metric은 point 중심으로 둔다.

Semantic Gaze Target Detection 계열처럼 localization과 semantic label을 동시에 예측하려면, object label protocol 자체를 더 엄격히 해야 합니다. ([NeurIPS Proceedings][4])

---

### 2.5 Hard negative를 만들어야 한다

현재 supervised target은 정답 point/object만 있습니다. 하지만 gaze target은 “시선 방향 선상에 있는 distractor”, “가까운 saliency object”, “같은 object category의 다른 instance”를 구분해야 합니다.

Ferret은 refer-and-ground instruction tuning에서 hard negative를 대량 포함해 grounding robustness를 강화했습니다. ([arXiv][9])

gaze용 hard negative는 다음처럼 만들 수 있습니다.

```text
Positive: marked person looks at object A.
Negative candidates:
- object B on the same gaze ray but wrong depth
- salient object near the target
- same category object at a different location
- object inside head box / body region
```

학습 형태는 두 가지입니다.

```text
Candidate boxes:
A: <loc...> screen
B: <loc...> person
C: <loc...> table
Select gaze target object: <obj_A>
Point: <loc...>
```

또는 preference pair:

```text
Chosen: Point near GT
Rejected: Point on distractor
```

후자는 Stage-2 DPO/SimPO류로도 연결하기 쉽습니다.

---

## 3. Stage-2 RL 설계 제안

공개 레포에는 이미 초안 수준의 GRPO RL 경로가 들어 있습니다. `config_rl.yaml`은 `train_stage: "rl"`, `rl_group_size: 4`, `rl_lr: 1e-6`, point/object/joint reward, KL, rollout temperature 등을 정의하고 있고, trainer에는 frozen reference model을 두고 rollout을 생성한 뒤 reward와 KL로 policy loss를 계산하는 GRPO 스타일 코드가 있습니다. ([GitHub][10])

### 3.1 처음 RL은 reasoning 없이 하는 것이 낫다

첫 RL 목표는 명확해야 합니다.

```text
Output:
Point: <loc_X><loc_Y>
Object: <obj_K>
```

Reward:

```text
R = 1.0 * exp(-β * min_L2(pred_point, gt_points))
  + 0.5~0.75 * object_match
  + 0.1~0.25 * joint_bonus
  - format_penalty
  - KL_penalty
```

현재 config의 방향은 대체로 맞습니다. 다만 `reward_extra_penalty: 0.0`은 format drift가 생기면 다시 올리는 게 좋습니다. Reasoning을 넣지 않는 이유는 간단합니다. RL 초기에 reasoning까지 보상하려고 하면, 모델이 장황한 텍스트나 ungrounded rationale로 reward hacking을 할 수 있습니다. ViGoRL도 naive RL이 abstract/ungrounded reasoning으로 빠질 수 있다고 보고, grounded reasoning warm-start를 먼저 둡니다. ([arXiv][8])

---

### 3.2 Reasoning-RL을 하려면 “텍스트 reasoning”이 아니라 “grounded trace”를 보상해야 한다

Reasoning을 Stage-2에 넣고 싶다면, reward가 검증 가능한 항목이어야 합니다.

나쁜 보상:

```text
Reasoning이 길면 +0.1
Reasoning에 "head", "eyes", "object" 단어가 있으면 +0.1
```

이건 verbosity reward입니다.

좋은 보상:

```text
- intermediate candidate loc가 GT 근처이면 +
- candidate object token이 GT object와 맞으면 +
- final point가 candidate region 내부이면 +
- reasoning loc들이 head-to-target 방향과 크게 모순되지 않으면 +
```

추천 format:

```text
Thought: head direction suggests down-right.
Ground: <loc_A><loc_B>
Candidate: <obj_K>
Final Point: <loc_X><loc_Y>
Final Object: <obj_K>
```

이 구조는 ViGoRL식 “reasoning step + spatial grounding”에 가깝습니다. ([arXiv][8])

---

### 3.3 GRPO 말고 DPO/Best-of-N distillation도 고려할 만하다

A100 80GB 한 장이면 on-policy GRPO가 가능하긴 하지만, VLM RL은 rollout과 logprob 계산 때문에 비용이 큽니다. 더 가벼운 대안은 다음입니다.

1. SFT checkpoint에서 같은 prompt에 대해 N개 sample을 생성한다.
2. GT point와의 L2로 best/worst를 고른다.
3. `(chosen, rejected)` pair로 DPO/IPO/SimPO류 preference tuning을 한다.

장점:

* on-policy rollout loop보다 구현이 단순하다.
* reward 모델이 필요 없다.
* point L2 기반 preference가 명확하다.
* reasoning 없이도 point 성능 개선을 볼 수 있다.

단점:

* exploration이 SFT 모델의 sampling 분포 안에 갇힌다.
* GRPO처럼 online으로 policy가 바뀌며 탐색하는 효과는 적다.

현 단계에서는 **DPO/Best-of-N distillation → GRPO** 순서를 추천합니다.

---

## 4. A100 80GB에서 가능한가

### 4.1 2B LoRA SFT

가능합니다. 현재 설정은 Qwen3-VL-2B-Instruct, LoRA r=16, fixed 512 scene, batch 16, grad accumulation 8입니다. ([GitHub][1])

다만 reasoning을 길게 만들면 sequence length가 증가합니다. `max_text_length: 512`는 충분해 보이지만, 실제 processor가 vision tokens까지 포함하므로 truncation/length 로그를 찍는 게 좋습니다.

### 4.2 2B GRPO RL

가능할 가능성이 높습니다. 단, 안전 설정은 다음입니다.

```yaml
train:
  batch_size: 4        # 처음에는 8 말고 4
  grad_accum_steps: 8
  gradient_checkpointing: true

rl:
  rl_group_size: 4
  rl_logprob_micro_batch_size: 4
  rl_n_ppo_epochs: 1
  rl_lr: 1e-6
```

현재 `config_rl.yaml`에도 OOM 시 `rl_logprob_micro_batch_size`를 `rl_group_size(4)`로 줄이라는 주석이 있습니다. ([GitHub][10])

### 4.3 4B GRPO RL

A100 80GB에서 불가능하다고 보지는 않습니다. 하지만 2B보다 훨씬 빡빡합니다. 특히 policy model + reference model을 동시에 올리고, `B × G` rollout logprob를 계산하므로 memory bottleneck은 weight보다 activation과 vision token 쪽입니다. 4B를 바로 RL하지 말고, 2B에서 reward와 format이 안정된 뒤 4B로 옮기는 게 낫습니다.

### 4.4 Reasoning 포함 RL

처음부터 하지 않는 편이 좋습니다. `max_new_tokens`가 16에서 64~128로 늘면 rollout 저장, decode, logprob 계산 비용이 크게 늘고, reward도 불안정해집니다. Reasoning-RL은 grounded trace format과 verifier가 준비된 뒤에만 하세요.

---

## 5. 내가 추천하는 다음 실험 순서

### Step 0. 지금 run에서 즉시 확인

먼저 아래 4개를 확인해야 합니다.

```text
1. has_reasoning ratio
2. 실제 target_text 예시 20개
3. n_point_tokens / n_object_tokens / n_reasoning_tokens
4. val generation preview에서 reasoning이 실제로 생성되는지 여부
```

현재 `generation_stop_at_object_end: true`라면 val/test generation에서는 reasoning이 거의 보이지 않는 것이 정상입니다. ([GitHub][1])

---

### Step 1. Reasoning 매칭/augmentation 문제 수정

* reasoning file structure를 코드 key와 맞춘다.
* reasoning 샘플에는 hflip/crop을 끈다.
* reasoning object line을 읽어서 pseudo label과 consistency check한다.
* val에는 reasoning 없는 empty block을 강제로 넣지 않는다.
  현재처럼 `force_reasoning_format=True`인데 val reasoning이 없으면, 검증 loss가 빈 reasoning format을 학습하는 이상한 상태가 될 수 있습니다. ([GitHub][6])

---

### Step 2. Head crop multi-image input 추가

이걸 가장 먼저 해볼 가치가 있습니다.

```text
Image 1: full scene with bbox
Image 2: cropped head/face
Output: Point + Object
```

이 ablation이 baseline보다 크게 좋아지면, reasoning보다 visual cue resolution이 병목입니다.

---

### Step 3. Reasoning-before-answer ablation

현재 target:

```text
Point:
Object:
Reasoning:
```

대안:

```text
Reasoning:
Object:
Point:
```

또는:

```text
Evidence:
Candidate:
Point:
Object:
```

이때 parser와 stop rule도 같이 바꿔야 합니다. `generation_stop_at_object_end`는 final answer 뒤에서만 적용해야 합니다.

---

### Step 4. Soft spatial loss 추가

현재 token CE만으로는 인접 bin과 큰 오차를 구분하지 못합니다. 다음 중 하나를 붙이세요.

* Gaussian soft CE over x/y bins
* 2D heatmap auxiliary decoder
* hidden state 기반 continuous point regression
* direction/angular auxiliary loss

이건 reasoning 여부와 독립적으로 point 성능을 올릴 가능성이 큽니다.

---

### Step 5. Reasoning 없는 Stage-2 RL

SFT가 안정되면 다음으로 넘어갑니다.

```text
Prompt: direct point/object
Rollout: group_size 4
Reward: exp(-β*L2) + object + joint + format - KL
Output length: 16~24 tokens
```

현재 공개 config의 RL 방향은 이 목적에 가깝습니다. ([GitHub][10])

---

## 최종 판단

현재 설계에서 reasoning이 효과가 약한 것은 자연스러운 결과입니다. 핵심 문제는 세 가지입니다.

1. **Reasoning이 answer 뒤에 있어 point를 조건화하지 못한다.**
2. **Reasoning text가 augmentation과 충돌할 수 있다.**
3. **VLM 입력이 full scene 위주라 head/eye cue가 부족할 가능성이 크다.**

가장 실용적인 수정 순서는 다음입니다.

```text
1. reasoning 매칭/로그 검증
2. reasoning 샘플 augmentation 제한
3. full scene + head crop multi-image 입력
4. point soft/heatmap auxiliary loss
5. reasoning-before-answer ablation
6. reasoning 없는 GRPO 또는 Best-of-N DPO
7. grounded reasoning trace 기반 RL
```

지금의 SFT를 완전히 버릴 필요는 없습니다. 다만 현재 run은 “reasoning이 좌표 성능을 올리는 실험”이라기보다 “post-hoc explanation auxiliary loss가 도움이 되는지 보는 ablation”으로 해석하는 게 맞습니다.

[1]: https://raw.githubusercontent.com/JinWoong-Jung/QWEN_GazeEstimation/main/sft.yaml "raw.githubusercontent.com"
[2]: https://raw.githubusercontent.com/JinWoong-Jung/QWEN_GazeEstimation/main/model/utils/gaze_tokens.py "raw.githubusercontent.com"
[3]: https://arxiv.org/abs/2306.15195 "[2306.15195] Shikra: Unleashing Multimodal LLM's Referential Dialogue Magic"
[4]: https://proceedings.neurips.cc/paper_files/paper/2024/hash/dbeb7e621d4a554069a6a775da0f7273-Abstract-Conference.html "Toward Semantic Gaze Target Detection"
[5]: https://raw.githubusercontent.com/JinWoong-Jung/QWEN_GazeEstimation/main/model/utils/data_utils.py "raw.githubusercontent.com"
[6]: https://raw.githubusercontent.com/JinWoong-Jung/QWEN_GazeEstimation/main/model/datasets.py "raw.githubusercontent.com"
[7]: https://openaccess.thecvf.com/content/CVPR2025/html/Ryan_Gaze-LLE_Gaze_Target_Estimation_via_Large-Scale_Learned_Encoders_CVPR_2025_paper.html "CVPR 2025 Open Access Repository"
[8]: https://arxiv.org/html/2505.23678v1 "Grounded Reinforcement Learning for Visual Reasoning"
[9]: https://arxiv.org/abs/2310.07704 "[2310.07704] Ferret: Refer and Ground Anything Anywhere at Any Granularity"
[10]: https://raw.githubusercontent.com/JinWoong-Jung/QWEN_GazeEstimation/main/config_rl.yaml "raw.githubusercontent.com"
