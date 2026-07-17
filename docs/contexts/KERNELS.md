# Transformers `kernels` 라이브러리 통합 정리

`transformers`에 Hugging Face Hub 기반 `kernels` 라이브러리가 통합되면서 생긴 구현 변경사항을 정리한다. 아울러 이 통합 메커니즘에 편입되지 않은 사례(FLA/Qwen3-Next)와의 구조적 차이도 함께 다룬다.

---

## 1. 배경 및 개념

`transformers`는 Hugging Face Hub에 올라간 커뮤니티 최적화 커널(주로 `kernels-community` 조직)을 `kernels` 라이브러리를 통해 모델에 바로 로드할 수 있도록 지원한다.

- 커널은 행렬 곱셈, 어텐션, 정규화 등 특정 연산을 타겟으로 더 빠르게 실행하도록 만들어짐
- 여러 연산을 하나의 커널로 융합(fuse)하면 GPU 메모리 접근 횟수가 줄어 메모리 대역폭 사용량과 연산당 실행 오버헤드가 감소
- 커널은 표준 PyTorch 연산을 대체하는 **드롭인(drop-in)** 방식으로 동작 — 모델 코드를 깨뜨리지 않고 forward 메서드를 최적화된 구현으로 교체

**문제의식**: 이런 저수준 커널은 CUDA/C++로 작성되어 타겟 시스템에서 컴파일이 필요하며, 각 커널 라이브러리마다 다른 빌드 시스템을 요구한다. 이를 `transformers`에 직접 의존성으로 추가하면 dependency bloat가 발생한다.

**해결책**: `kernels` 패키지는 사전 빌드된(pre-built) 바이너리를 Hub에서 다운로드하여 이 문제를 해결한다. 원하는 커널을 지정하면 `kernels`가 사용자의 Python/PyTorch/CUDA 버전에 맞는 바이너리를 찾아 첫 사용 시 다운로드한다.

**최소 버전**: `kernels >= 0.11.0`

---

## 2. API 정리

### 2.1 `use_kernels=True` — `from_pretrained()` 인자

해당 디바이스에서 사용 가능한 최적 커널을 Hub에서 불러와 지원되는 PyTorch 연산을 커널 구현으로 교체한다.

```python
from transformers import AutoModelForCausalLM

model = AutoModelForCausalLM.from_pretrained(
    "Qwen/Qwen3-0.6B",
    use_kernels=True,
    device_map="cuda"
)
```

디바이스별 기본 커널 매핑이 내장되어 있음(RMSNorm, MoE MLP, MLP/SwiGLU/GeGLU, Linear, 활성화 함수, RoPE, Causal LM loss, Deformable attention 등). CUDA/ROCm/XPU마다 기본 Hub 저장소가 지정되어 있고, 등록된 기본값이 없는 연산은 표준 PyTorch로 폴백된다.

- AMD GPU는 PyTorch상 디바이스 타입을 `cuda`로 보고하지만, `transformers`가 런타임에 ROCm을 감지해 AITER 빌드를 포함한 지원 연산을 AMD 커널로 자동 라우팅한다.

### 2.2 `KernelConfig` 클래스

어떤 커널을 어떤 레이어에 적용할지 세밀하게 제어하는 설정 클래스.

```python
from transformers import AutoModelForCausalLM, KernelConfig

kernel_config = KernelConfig(
    kernel_mapping={"RMSNorm": "kernels-community/rmsnorm"}
)
model = AutoModelForCausalLM.from_pretrained(
    "Qwen/Qwen3-0.6B",
    use_kernels=True,
    kernel_config=kernel_config,
)
```

주요 기능:

| 기능 | 설명 |
|---|---|
| 세부 항목 매핑 | 콜론(`:`)으로 저장소 안의 특정 커널 항목·레이어 매핑 가능. 예: `"kernels-community/liger_kernels:LigerRMSNorm"` |
| 디바이스별 구현체 지정 | cuda / rocm / metal / xpu 별로 다른 구현체 지정 가능 |
| 커널 비활성화 | 빈 문자열(`""`)을 매핑하면 해당 레이어의 커널 사용 비활성화 |
| 모듈 퓨전 | `(class_name, path_pattern)` 튜플 키로 인접 모듈들을 하나의 커널로 융합 (예: RMSNorm + MLP → fused 커널) |
| 로컬 커널 | `use_local_kernel=True`로 Hub 대신 로컬 파일 경로에서 커널 로드 |

### 2.3 어텐션 커널 — `attn_implementation`

`attn_implementation` 인자에 Hub 저장소 경로를 직접 지정해 FlashAttention 등 어텐션 커널을 로드한다.

```python
from transformers import AutoModelForCausalLM

model = AutoModelForCausalLM.from_pretrained(
    "Qwen/Qwen3-0.6B",
    attn_implementation="kernels-community/flash-attn2",
    device_map="cuda"
)
```

내부적으로 `ALL_ATTENTION_FUNCTIONS` / `AttentionInterface` 레지스트리를 통해 백엔드가 런타임에 선택된다. `sdpa`, `eager`, Hub 커널 경로가 모두 동일한 인터페이스 뒤에서 교체 가능한 백엔드로 추상화되어 있다.

**신뢰 정책**: `kernels-community` 저장소 외의 것은 `allow_all_kernels=True`를 추가로 요구한다(비-HF 모델의 `trust_remote_code=True`와 유사). 커널 로딩이 호스트에서 임의 코드 실행으로 이어질 수 있고 모든 저장소를 검증할 수 없기 때문이다.

```python
model = AutoModelForCausalLM.from_pretrained(
    "Qwen/Qwen3-0.6B",
    attn_implementation="random-repo/random-attention",
    allow_all_kernels=True,
    device_map="cuda"
)
```

**버전 고정**: `@v2.1.0`(특정 태그/브랜치) 또는 `@>=2.0,<3.0`(시맨틱 버저닝 제약) 형식 지원.

주요 어텐션 커널 예시:
- `kernels-community/flash-attn2`
- `kernels-community/vllm-flash-attn3` — attention sinks 지원(Hopper 아키텍처 한정), GPT-OSS에서 사용
- `kernels-community/paged-attention`

### 2.4 모드 인식(mode-awareness)

커널은 `model.training` 상태에 기반해 훈련/추론 모드에 자동으로 적응한다. `kernelize()` 함수로 `Mode.INFERENCE`, `Mode.TRAINING`, `Mode.TRAINING | Mode.TORCH_COMPILE` 등을 명시적으로 지정할 수도 있다.

### 2.5 전역 비활성화

환경 변수 `USE_HUB_KERNELS=0`(또는 `OFF`/`NO`)으로 커널 통합을 전역적으로 끌 수 있다.

### 2.6 통합 메커니즘의 핵심 데코레이터: `@use_kernel_forward_from_hub`

모델 코드 안에서 특정 레이어를 Hub 커널과 연결하는 실제 지점. 예시(GPT-OSS):

```python
@use_kernel_forward_from_hub("RMSNorm")
class GptOssRMSNorm(nn.Module):
    ...

@use_kernel_forward_from_hub("MegaBlocksMoeMLP")
class GptOssExperts(nn.Module):
    ...
```

데코레이터는 커뮤니티 기여 커널을 가리키는 포인터 역할만 한다. `RMSNorm`은 `kernels-community/liger_kernels`에서, `MegaBlocksMoeMLP`는 `kernels-community/megablocks`에서 오며, 디바이스(CUDA/ROCm)와 학습/추론 여부에 따라 적합한 커널이 자동으로 pull된다.

이 데코레이터가 붙은 레이어는 특정 모델 전용이 아니라 여러 모델에서 재사용된다(예: RMSNorm liger 커널은 다수 모델에서 공유).

**로그로 확인**:
```python
import logging
logging.basicConfig(level=logging.INFO)

model = AutoModelForCausalLM.from_pretrained(
    "openai/gpt-oss-20b",
    dtype="auto", device_map="auto",
    use_kernels=True,
)
```
```
INFO:root:Using layer `LigerRMSNorm` from repo `kernels-community/liger_kernels`
INFO:root:Using layer `MegaBlocksMoeMLP` from repo `kernels-community/megablocks`
```

---

## 3. MXFP4 양자화와의 결합

`transformers`는 MXFP4(4-bit floating, E2M1 + blockwise scaling) 양자화를 네이티브 지원하며, 이때도 Hub 커널 배포 체계를 그대로 활용한다.

- 32-element 블록과 스케일을 인식하는 GEMM/fused 연산이 필요 → `kernels-community/triton_kernels` 등에서 MXFP4-aware Triton 커널을 자동으로 pull
- MXFP4 경로는 `use_kernels=True`를 별도로 지정할 필요 없이 `transformers`에서 기본 동작(default)으로 설정되어 있음
- 요구사항: `accelerate`, `kernels`, `triton>=3.4` (PyTorch 2.8은 triton 3.4를 포함하므로 PyTorch 2.7 사용 시에만 별도 설치 필요), NVIDIA GPU compute capability ≥ 7.5
- 조건 미충족 시 `bfloat16` 경로로 자동 폴백(약 4배 메모리 사용)

캐시 확인:
```shell
hf cache scan
```
```
REPO ID                          REPO TYPE SIZE ON DISK
-------------------------------- --------- ------------
kernels-community/triton_kernels model           536.2K
```

---

## 4. Liger Kernel 통합 (학습 전용, `TrainingArguments`)

`kernels` Hub 메커니즘과는 별도로, 학습 파이프라인(`TrainingArguments`)에 `use_liger_kernel=True`를 설정하면 Liger Kernel이 적용된다.

```python
from transformers import TrainingArguments

args = TrainingArguments(
    use_liger_kernel=True,
    liger_kernel_config={
        "rope": True,
        "swiglu": True,
        "cross_entropy": True,
        "fused_linear_cross_entropy": True,
        "rms_norm": True,
    },
)
```

- RMSNorm, RoPE, SwiGLU, CrossEntropy, FusedLinearCrossEntropy 등을 단일 Triton 커널로 융합
- FlashAttention, FSDP, DeepSpeed와 호환
- 멀티 GPU 학습 처리량 향상 + 메모리 사용량 절감 → 더 큰 vocabulary, batch size, context length 가능

---

## 5. 통합 메커니즘 비교: Hub 커널(`kernels`) vs. FLA(선택적 의존성)

같은 "최적화 커널 사용"이라는 목표를 가지고 있지만, `transformers` 내에서 실제로 배선(wiring)되는 방식은 완전히 다른 두 트랙이 공존한다. FlashAttention/Liger/MXFP4는 전자, Qwen3-Next의 Gated DeltaNet(FLA)은 후자에 해당한다.

### 5.1 Hub 커널 방식 (FlashAttention, Liger, MXFP4 등)

- **통합 지점**: `AttentionInterface`/`ALL_ATTENTION_FUNCTIONS` 레지스트리, 또는 `@use_kernel_forward_from_hub` 데코레이터
- **백엔드 선택 시점**: 런타임 — `attn_implementation="kernels-community/flash-attn2"` 같은 문자열로 명시적 선택
- **배포 방식**: 사전 컴파일된 바이너리를 Hub에서 자동 다운로드(빌드 불필요)
- **사용자 스위치**: `use_kernels=True`, `KernelConfig`, `attn_implementation` 등 공식 API로 제어
- **폴백**: 등록된 기본 커널이 없으면 표준 PyTorch 연산으로 자동 폴백

### 5.2 선택적 의존성 방식 (FLA / Qwen3-Next Gated DeltaNet)

Qwen3-Next의 `modular_qwen3_next.py`(`Qwen3NextGatedDeltaNet`)를 보면 다음과 같은 별도 패턴이 쓰인다.

```python
if is_causal_conv1d_available():
    from causal_conv1d import causal_conv1d_fn, causal_conv1d_update
else:
    causal_conv1d_update, causal_conv1d_fn = None, None

if is_flash_linear_attention_available():
    from fla.modules import FusedRMSNormGated
    from fla.ops.gated_delta_rule import chunk_gated_delta_rule, fused_recurrent_gated_delta_rule
else:
    chunk_gated_delta_rule, fused_recurrent_gated_delta_rule = None, None
    FusedRMSNormGated = None

is_fast_path_available = all(
    (causal_conv1d_fn, causal_conv1d_update, chunk_gated_delta_rule, fused_recurrent_gated_delta_rule)
)
```

레이어 초기화 시 함수 포인터가 확정된다.

```python
self.causal_conv1d_fn = causal_conv1d_fn
self.causal_conv1d_update = causal_conv1d_update or torch_causal_conv1d_update
self.chunk_gated_delta_rule = chunk_gated_delta_rule or torch_chunk_gated_delta_rule
self.recurrent_gated_delta_rule = fused_recurrent_gated_delta_rule or torch_recurrent_gated_delta_rule

if not is_fast_path_available:
    logger.warning_once(
        "The fast path is not available because one of the required library is not installed. "
        "Falling back to torch implementation. ..."
    )
```

`fla`가 설치되어 있으면 그 Triton 커널을, 없으면 같은 파일에 순수 PyTorch로 직접 재구현된 `torch_chunk_gated_delta_rule` / `torch_recurrent_gated_delta_rule`을 사용한다. `forward()`는 어느 쪽이 바인딩되었는지 신경 쓰지 않고 `self.chunk_gated_delta_rule(...)`을 그대로 호출한다.

- **통합 지점**: 모듈 레벨 `if/else` import + 인스턴스 속성 바인딩 (레지스트리 없음)
- **백엔드 선택 시점**: 로드(import) 시점 — 패키지 설치 여부로 자동 결정, 런타임 문자열 스위치 없음
- **배포 방식**: `fla`, `causal_conv1d`는 일반 pip 패키지로 설치, Hub `kernels` 메커니즘을 타지 않음
- **사용자 스위치**: 없음 — `pip install fla`, `pip install causal-conv1d` 여부만이 유일한 제어 수단
- **폴백**: 동일 파일 내 순수 PyTorch 재구현(`torch_chunk_gated_delta_rule` 등)으로 자동 대체, 성능 저하 경고 로그 출력
- **부분 폴백 가능**: `causal_conv1d`와 `fla` 중 하나만 없어도 해당 부분만 개별적으로 대체됨

### 5.3 비교 표

| 항목 | Hub 커널(`kernels`) | FLA(선택적 의존성) |
|---|---|---|
| 통합 지점 | `AttentionInterface` 레지스트리 / `@use_kernel_forward_from_hub` | 모듈 레벨 `if/else` import |
| 선택 시점 | 런타임(문자열 인자) | 로드 시점(설치 여부) |
| 배포 | Hub에서 사전 컴파일 바이너리 자동 다운로드 | 일반 pip 패키지, 로컬 빌드/휠 필요 |
| 제어 API | `use_kernels`, `KernelConfig`, `attn_implementation` | 없음(설치 여부로만 결정) |
| 폴백 | 표준 PyTorch 연산 | 파일 내 하드코딩된 PyTorch 재구현 |
| 하드웨어 대응 | Hub가 시스템에 맞는 바이너리 자동 매칭 | 해당 아키텍처용 사전 빌드가 없으면 항상 폴백 (예: SM121 GPU에서 `fla`/`causal_conv1d` 빌드 부재 시 영구 폴백) |

---

## 6. 참고 링크

- https://huggingface.co/docs/transformers/kernels
- https://huggingface.co/docs/transformers/kernel_doc/loading_kernels
- https://huggingface.co/docs/transformers/en/main_classes/kernels
- https://huggingface.co/docs/transformers/en/attention_interface
- https://huggingface.co/blog/hello-hf-kernels
- https://huggingface.co/blog/faster-transformers
- https://huggingface.co/docs/transformers/model_doc/qwen3_5
- https://github.com/huggingface/transformers/blob/main/src/transformers/models/qwen3_next/modular_qwen3_next.py
- https://github.com/fla-org/flash-linear-attention
