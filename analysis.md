# Current SFT vs Legacy Baseline Analysis

비교 기준:

- 현재 repo: `/home/work/jinwoong/QWEN_GazeEstimation` working tree
- 과거 repo: `/home/work/jinwoong/legacy`
- 과거 기준 commit: `50eb81a44c2992e3efab8a1a15c62ab3576aea5f`
- 설정 매핑: 현재 `sft.yaml` vs 과거 `config.yaml`
- 제외: RL/SDFT 전용 변경, wandb run name, notes/tags 같은 로깅 명칭 변경

## 요약

현재 SFT는 과거 최고 성능 run과 상당히 다르다. 사소한 로그명 변경 수준이 아니라, 학습 target format, tokenizer special token set, validation/test decoding 방식, point loss 정의, loss weight, effective batch size, LoRA target module이 모두 바뀌었다. 이 중 가장 큰 차이는 `Point:\nObject:` 텍스트 포맷에서 `<|point_start|>...<|object_start|>...` span-marker 포맷으로 바뀐 점과, free generation 평가에서 constrained decoding 평가로 바뀐 점이다.

## 중대한 차이점

### 1. 출력 스키마와 prompt가 완전히 바뀜

과거 baseline은 target을 아래처럼 자연어 prefix 기반 2-line 포맷으로 학습했다.

```text
Point: <loc_NNN><loc_MMM>
Object: <obj_KKK>
```

근거:

- 과거 `config.yaml`: `prompt_text`가 `Point:` / `Object:` 두 줄 반환을 요구함.
- 과거 `model/utils/gaze_tokens.py`: `POINT_PREFIX = "Point:"`, `OBJECT_PREFIX = "Object:"`, regex도 이 포맷만 엄격히 파싱함.

현재 SFT는 direct target을 아래 span-marker 포맷으로 학습한다.

```text
<|point_start|><loc_NNN><loc_MMM><|point_end|><|object_start|><obj_KKK><|object_end|>
```

근거:

- 현재 `sft.yaml`: `prompt_text_direct`가 span-marker 포맷을 요구함.
- 현재 `model/utils/gaze_tokens.py`: `POINT_START_MARKER`, `OBJECT_START_MARKER` 등을 사용해 target text를 생성함.
- 현재 `model/utils/special_tokens.py`: marker token 6개가 gaze special token으로 추가됨.

영향:

- 과거와 현재는 같은 `<loc_*>`, `<obj_*>`를 쓰더라도 answer format 자체가 다르다.
- format mask에 들어가는 token 종류와 길이가 달라진다.
- 과거 checkpoint/metric과 현재 SFT metric을 직접 비교하기 어렵다.
- 현재는 marker token embedding도 새로 학습해야 하므로 tokenizer/embedding 학습 조건도 달라진다.

### 2. 평가/검증 decoding이 free generation에서 constrained decoding으로 바뀜

과거 baseline은 `generation_max_new_tokens: 16`으로 일반 autoregressive generation을 수행하고, 생성된 문자열을 `Point:` / `Object:` regex로 파싱했다.

현재 `sft.yaml`은 `constrained_decoding: true`, `generation_max_new_tokens: 8`이다. 현재 코드에서는 constrained mode일 때 모델이 자유롭게 전체 문자열을 생성하지 않고, 정해진 순서로 marker token과 loc/object token만 뽑는다.

근거:

- 현재 `sft.yaml`: `constrained_decoding: true`, `constrained_loc_decoding: "argmax"`.
- 현재 `model/utils/eval_utils.py`: `constrained_generate_structured()`가 `<|point_start|>`, loc x, loc y, `<|point_end|>`, `<|object_start|>`, object, `<|object_end|>` 순서로 강제 생성함.
- 현재 `run_test_metrics()`는 `constrained_decoding=True`면 일반 `model.generate()` 대신 constrained path를 사용함.

영향:

- 현재 validation/test의 `FormatValid`는 모델의 포맷 생성 능력보다 decoding constraint의 영향을 크게 받는다.
- 과거의 format-valid, extra-text 문제와 현재의 format-valid는 같은 의미가 아니다.
- point/object 선택 정확도도 "자유 생성 문자열 안에서 맞춘 결과"와 "허용 token 집합에서 고른 결과"라 비교 조건이 다르다.

### 3. Loss objective가 크게 바뀜

과거 baseline config:

- `loss_point_weight: 1.0`
- `loss_object_weight: 1.0`
- `loss_format_weight: 1.0`
- point loss는 hard CE.

현재 `sft.yaml`:

- `loss_point_weight: 3.0`
- `loss_object_weight: 1.0`
- `loss_format_weight: 0.5`
- `gaussian_point_sigma: 3.0`

현재 코드에는 loc token에 대해 Gaussian soft-label CE가 추가되어 있고, `gaussian_point_sigma > 0`이면 point token loss가 hard CE가 아니라 주변 bin에도 확률 질량을 주는 soft target CE로 바뀐다.

근거:

- 현재 `model/utils/loss_utils.py`: `gaussian_soft_label_ce()` 및 `gaussian_sigma` path.
- 현재 `model/trainer.py`: loc token id list를 만들고 `gaussian_point_sigma`를 `compute_answer_loss()`로 전달함.
- 과거 `model/utils/loss_utils.py`: point/object/format hard CE만 존재.

영향:

- 현재는 point 위치 학습을 3배 가중하고, format 학습은 절반으로 줄였다.
- 과거 최고 run은 format loss weight가 1.0이었고, `<|im_end|>`까지 format mask에 포함해 stop을 학습했다.
- 현재는 constrained decoding이 format을 보장해주는 대신, 학습 loss에서는 format 압력이 약해졌다.
- soft point CE는 L2 개선에 유리할 수 있지만 exact bin/object/format 균형은 과거와 달라진다.

### 4. Effective batch size와 optimizer update 수가 달라짐

과거 baseline:

- `batch_size: 16`
- `grad_accum_steps: 8`
- effective batch size: `128`

현재 SFT:

- `batch_size: 32`
- `grad_accum_steps: 8`
- effective batch size: `256`

학습 epoch, lr, warmup ratio는 같지만, batch size가 2배라 epoch당 optimizer update 수는 대략 절반이 된다. cosine schedule의 total update 수와 warmup step 수도 같이 달라진다.

영향:

- 같은 `epochs: 20`, `lr: 1e-4`라도 optimization trajectory가 다르다.
- 최고 성능이 특정 update granularity나 gradient noise에 의존했다면 재현성이 떨어질 수 있다.

### 5. LoRA target module 범위가 attention-only에서 attention+MLP로 확장됨

과거 baseline:

```yaml
lora_target_modules: ["q_proj", "k_proj", "v_proj", "o_proj"]
```

현재 SFT:

```yaml
lora_target_modules: ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]
```

영향:

- 현재는 MLP projection까지 LoRA가 들어가 trainable parameter와 적응 범위가 커진다.
- 표현력은 늘지만, 같은 lr/epoch/batch 조건에서 과거 attention-only LoRA와 학습 안정성/일반화가 달라질 수 있다.
- 과거 최고 run을 재현하려면 attention-only target으로 맞추는 것이 우선이다.

### 6. Attention implementation과 max text length 차이

과거 baseline:

- `attn_implementation: "sdpa"`
- `max_text_length: 256`

현재 SFT:

- `attn_implementation: "flash_attention_2"`
- `max_text_length: 512`

다만 `processor_collate.py`의 train/infer input 생성은 `truncation=False`로 processor를 호출하므로, 현재 코드상 `max_text_length`가 직접 truncation에 쓰이지 않는 경로가 많다. 따라서 `max_text_length` 자체보다 `attn_implementation` 변경이 더 실질적인 차이다.

영향:

- attention backend 차이는 수치/속도/메모리 측면에서 차이를 만들 수 있다.
- `max_text_length`는 현재 direct SFT에서는 영향이 제한적일 가능성이 있다.

## 상대적으로 덜 중요한 차이

- `image_cache_size: 1000 -> 0`: 성능/속도 차이는 있지만 학습 target이나 gradient 로직 자체를 바꾸지는 않는다.
- `min_pixels`, `max_pixels`: 둘 다 `image_resize_mode: fixed`이면 processor에는 `None`으로 들어가는 경로라 영향이 제한적이다.
- `run_name`, wandb metadata: 학습 로직 영향 없음.
- RL/SDFT 관련 parser/utility 추가: 현재 `train_stage: sft`, `sample_mode: direct_only`, distillation weight 기본값이면 대부분 비활성 경로다.

## 재현 우선순위

과거 최고 run을 재현하려면 아래부터 맞추는 것이 우선순위가 높다.

1. 현재 target schema를 과거 `Point:\nObject:` 포맷으로 되돌리거나, legacy repo에서 그대로 실험한다.
2. validation/test에서 constrained decoding을 끄고, 과거처럼 free generation + parser로 비교한다.
3. loss를 과거처럼 `point/object/format = 1/1/1`, hard CE로 맞춘다.
4. `batch_size: 16`, `grad_accum_steps: 8`로 effective batch size를 128로 맞춘다.
5. LoRA target을 `q_proj,k_proj,v_proj,o_proj`로 줄인다.
6. `attn_implementation: sdpa`로 맞춘다.

현재 설정은 과거 baseline의 단순 개선판이라기보다, "새 출력 스키마 + constrained decoding + point-biased soft loss" 실험에 가깝다. 따라서 과거 최고 성능을 넘기 어렵다면, 먼저 위 항목들을 하나씩 되돌려 ablation하는 것이 가장 안전하다.
