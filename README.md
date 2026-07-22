<div align="center">

# Haan

**H**earing **A**lways, **A**cting **N**aturally

*한국어 Full-duplex 음성 대화 에이전트*

[![Python](https://img.shields.io/badge/Python-3.12+-3776AB?logo=python&logoColor=white)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-cu126-EE4C2C?logo=pytorch&logoColor=white)](https://pytorch.org/)
[![uv](https://img.shields.io/badge/uv-managed-DE5FE9?logo=uv&logoColor=white)](https://github.com/astral-sh/uv)
[![License](https://img.shields.io/badge/License-Apache_2.0-D22128.svg)](LICENSE)

</div>

---

**한국어 멀티턴 음성 대화 데이터셋 없이** 한국어 Full-duplex 대화 능력을 창발시키는 것을 목표로 하는 연구. Kyutai Moshi 아키텍처를 기반으로 백본을 Qwen3-8B로 교체해 "언어를 가르치는" 비용을 생략하고, 오디오 모달리티 접합과 turn-taking 구조 전이에 집중한다.

## 핵심 설계 (요약)

- **영어 멀티턴 KD는 Mimi semantic 토큰(level-0) logit에 한정** — turn-taking의 동시성 행동(맞장구·barge-in·오버랩) 전이가 목적. 공유 동결 Mimi 코덱 덕에 teacher(Moshi)–student(Qwen3) 출력공간이 동일.
- **음색은 KD가 아니라 in-context voice-prompt로 공급** (PersonaPlex Hybrid System Prompt 기법의 독립 구현; 가중치·데이터 미사용).
- **텍스트 채널은 Qwen3 IT 포맷(`<think>`/`<tool_call>`) 보존**, 태그 비발성 후처리로 agent 능력 유지.
- **acoustic prosody는 단계적 graft** (semantic-only 확립 후 turn-event 국소 KD, guard로 조기 종료).

## 셋업

### 1. 시스템 패키지 (apt)

Python 의존성만으로는 부족하다. `uv`가 관리하지 않는 시스템 라이브러리가 먼저 있어야 한다.

```bash
sudo apt-get update
sudo apt-get install -y ffmpeg sox
```

| 패키지 | 필요성 | 이유 |
|---|---|---|
| `ffmpeg` | **필수** | `torchcodec`이 FFmpeg 공유 라이브러리(`libav*`)를 런타임에 `dlopen`한다. 번들하지 않으므로 없으면 `import`부터 실패한다. 지원 범위는 FFmpeg **4~8** — Ubuntu 22.04의 기본 후보 4.4.2로 충분하다. |
| `sox` | 선택 | 데이터 준비 시 리샘플링·트리밍용 CLI. 현재 코드베이스가 직접 import하지는 않는다. |

> [!NOTE]
> `libsndfile`은 apt로 깔 필요가 없다. `soundfile` 휠이 자체 번들본을 싣고 다니며, 시스템에 구버전이 깔려 있어도 그쪽을 쓰지 않는다.

> [!IMPORTANT]
> FFmpeg은 `pyproject.toml`이 관리하지 않는 **시스템 전제조건**이다. 서버를 옮기면 `uv sync`만으로는 복구되지 않으니 이 단계를 반드시 다시 밟아야 한다.

### 2. Python 환경 (uv)

의존성은 [`uv`](https://github.com/astral-sh/uv)로 관리한다. `uv pip install`이 아니라 `uv add` / `uv sync`를 쓴다 — 그래야 `uv.lock`이 진실 원천으로 남는다.

```bash
uv sync --upgrade --extra personaplex --extra dev
```

| Extra | 내용 |
|---|---|
| *(기본)* | `torch`, `torchaudio`, `torchcodec`, `transformers`, `datasets`, `bitsandbytes`, `kernels`, `soundfile`, `scipy`, `matplotlib` |
| `personaplex` | `sentencepiece`, `protobuf`, `pyloudnorm` — 토크나이저 변환 및 voice-prompt 라우드니스 정규화 |
| `dev` | `ipykernel`, `jupyter-client`, `ipywidgets`, `ipywebrtc`, `nbconvert`, `nbformat` — 노트북 작업 |

`transformers`는 PyPI판이 아니라 Moshi 패치가 들어간 포크의 `feat/personaplex` 브랜치를 쓴다. `torch`·`torchaudio`·`torchcodec`은 PyPI 대신 PyTorch `cu126` 인덱스에 고정돼 있다 — `torchcodec`은 torch 의존성을 metadata에 **선언하지 않고** C++ ABI로만 붙기 때문에, 버전이 어긋나면 resolver가 잡아주지 못하고 import 시점에 터진다.

### 3. 설치 확인

```bash
uv run python -c "
import torch, torchcodec
from torchcodec.decoders import AudioDecoder
print('torch     ', torch.__version__)
print('torchcodec', torchcodec.__version__)
print('cuda      ', torch.cuda.is_available(), torch.cuda.device_count())
"
```

`OSError: libavutil.so.NN: cannot open shared object file`이 나오면 1단계의 `ffmpeg`이 빠진 것이다.

## 문서

| 문서 | 내용 |
|---|---|
| [`PROJECT.md`](PROJECT.md) | 개요·컨트리뷰션·관련연구·컴퓨팅 예산 |
| [`PERSONA.md`](PERSONA.md) | 캐릭터/페르소나 설정 |
| [`docs/contexts/ARCHITECTURE.md`](docs/contexts/ARCHITECTURE.md) | 백본 교체, 임베딩·Role Token, KD codebook routing, voice-prompt, Role 후속 로드맵 |
| [`docs/contexts/TRAINING_CURRICULUM.md`](docs/contexts/TRAINING_CURRICULUM.md) | Phase 0~5 커리큘럼, Joint loss, 커널 최적화 노트 |
| [`docs/contexts/DATA_STRATEGY.md`](docs/contexts/DATA_STRATEGY.md) | 싱글턴 데이터·합성 파이프라인·voice-prompt |
| [`docs/contexts/RISKS_AND_DIAGNOSTICS.md`](docs/contexts/RISKS_AND_DIAGNOSTICS.md) | 실패 메커니즘·진단 |

## License

[Apache License 2.0](LICENSE)
