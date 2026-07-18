# Haan (하안) — 한국어 Full-duplex 음성 대화 모델

> 모델명: **Haan**. 페르소나/캐릭터 설정은 별도 `PERSONA.md`에 기술됨 (본 문서 및 이하 기술 문서들은 아키텍처·학습·데이터 설계에 집중하며 페르소나 내용과는 독립적으로 관리).

## 1. 프로젝트 한 줄 요약

Kyutai의 Moshi 아키텍처를 기반으로, **한국어 멀티턴 음성 대화 데이터셋 없이** 한국어 Full-duplex 대화 능력을 창발시키는 것을 목표로 하는 연구. 백본을 Helium(Moshi 원본)에서 Qwen3 8B로 교체해 "한국어를 가르치는" 비용 자체를 생략하고, 대신 오디오 모달리티 접합과 turn-taking 구조 학습에 집중한다.

> **핵심 설계 결정 (요약, 상세는 각 문서 참조)**
> - **KD는 Mimi semantic 토큰(level-0) logit에 한정**한다. 영어 멀티턴 KD의 실질 목적은 콘텐츠·음향이 아니라 **turn-taking의 동시성 행동(맞장구·barge-in·오버랩)**의 전이이며, 동결 공유 Mimi 코덱 덕에 teacher(Moshi)–student(Qwen3)의 semantic 출력공간이 동일해 projection 없이 logit KD가 성립한다. hidden-state KD(이종 backbone)와 텍스트 logit KD(토크나이저 상이)는 폐기. (`ARCHITECTURE.md` §5.1, `RISKS_AND_DIAGNOSTICS.md` §7)
> - **화자/음색은 KD가 아니라 in-context voice-prompt(Hybrid System Prompt 기법의 독립 구현)로 공급**한다. PersonaPlex의 가중치·데이터는 라이선스 문제로 미사용하되, voice-prompt prefix 기법 자체는 일반적 in-context 조건화이므로 독립 구현한다.
> - **텍스트 채널은 Qwen3 IT 순정 포맷(`<think>`, `<tool_call>`)을 보존**하고 태그 구간을 비발성 후처리해 agent 능력을 유지한다. 부수적으로 `<think>`/`<tool_call>` 침묵 구간이 content-conditioned 응답 pause를 제공한다.
> - **acoustic(prosody) 전이는 단계적 graft**로 처리한다. semantic-only KD + voice-cloning으로 클로닝을 먼저 확립한 뒤, turn-event 국소 acoustic KD를 짧게 얹는다. (`TRAINING_CURRICULUM.md` Phase 3.5)
> - **user-stream 예측은 공유 Depth Transformer의 병렬 분리로 구현**한다. PersonaPlex의 16-step 순차(self→user) 대신, context를 MLP로 2채널 분리 + role embedding 주입 후 batch=2 병렬(8-step)로 예측 — user 예측 능력을 유지하면서 약 2배 빠르고, Moshi 본래 factorization에 더 충실하다. (`ARCHITECTURE.md` §5.4)

## 2. 배경 문제의식

- J-Moshi(일본어 Moshi 파생 모델)는 나고야대에서 실제 일본어 멀티턴 음성 대화 데이터로 파인튜닝해 만들어졌으나, 이런 멀티턴 대화 데이터셋은 수집 자체가 매우 어렵고 비용이 크다.
- Haan은 이 전제를 뒤집는다: 멀티턴 데이터셋 수집이라는 병목을 우회하고, **싱글턴(단일 교환) 데이터 + 구조적 전이**만으로 멀티턴 능력이 창발하는지를 검증하는 것이 연구의 핵심 질문이다.

## 3. 기대 컨트리뷰션 (3축)

### 3.1 한국어 Full-duplex 구현
Qwen3 8B 백본에 Mimi 기반 오디오 RVQ 토큰을 접합해, 실시간 양방향(사용자가 말하는 동안 동시에 듣고 응답할 수 있는) 한국어 음성 대화 모델을 구현.
- 음색은 in-context **voice-prompt**(Hybrid System Prompt 기법 독립 구현)로 공급하고, 텍스트 채널은 Qwen3 IT 포맷(`<think>`/`<tool_call>`)을 보존해 agent 능력을 유지한다.

### 3.2 영어로부터의 Multi-turn 전이
한국어 멀티턴 대화 데이터 없이, **영어 멀티턴 Knowledge Distillation(원본 Moshi를 teacher로 사용) + 한국어 싱글턴 데이터의 Joint Loss 학습**을 통해 한국어 멀티턴 대화 능력(턴테이킹 타이밍, 맞장구, 끼어들기 처리 등)이 창발하는지를 검증.
- **KD 대상은 Mimi semantic 토큰(level-0) logit으로 한정**한다. 이 KD가 실질적으로 전이하는 것은 콘텐츠(Qwen3가 이미 보유)나 음향(voice-prompt가 공급)이 아니라, **다른 방법으로는 스스로 만들기 어려운 turn-taking의 동시성 행동(맞장구·barge-in·오버랩)**이다 — teacher(Moshi)가 실제 대화로 이미 체득한 그 분포를 공유 코덱 출력공간에서 그대로 넘겨받는다.
- 핵심 가설: turn-taking은 언어 독립적인 **구조적/시간적 행동 패턴**이며, 이는 콘텐츠(의미)와 분리되어 전이 가능하다.
- 이 가설은 텍스트 도메인의 잘 알려진 cross-lingual zero-shot transfer(예: mBERT 기반 DST 전이)와는 성격이 다르다: 전이 대상이 "의미"가 아니라 오디오 채널의 "타이밍/행동"이며, 전이 경로가 단순 zero-shot 추론이 아니라 **서로 다른 두 목적함수(영어 멀티턴 KD + 한국어 싱글턴 TTS)의 그라디언트를 혼합하는 compositional generalization**이라는 점에서 novelty가 있다.

### 3.3 일본어로의 전이 (Mimi Semantic Token 효용성 리서치)
백본이 멀티링구얼(Qwen3)인 상태에서, Mimi의 시멘틱 RVQ 토큰을 매개로 **학습한 적 없는 언어(일본어)로의 말하기 능력 전이**가 가능한지 탐구.
- 완전 제로샷 전이가 안 되더라도, J-Moshi 대비(약 67,000시간의 일본어 데이터 사용) **현저히 적은 데이터로 일본어 발화가 가능해지는지**를 데이터 효율성 관점에서 리포트.
- 데이터 양 스윕(0h/1h/10h/100h)으로 성능 곡선을 그려 J-Moshi와 비교.

## 4. 관련 연구 비교

| | Moshi (원본) | J-Moshi | J-Moshi-ext | LLM-jp-Moshi-v1 | **Haan (본 연구)** |
|---|---|---|---|---|---|
| 백본 | Helium 7B (Kyutai 자체 학습, 영어 2.1T 토큰) | Helium 기반, 일본어 파인튜닝 | 좌동 | Helium 기반, 일본어 파인튜닝 | **Qwen3 8B** (사전학습된 멀티링구얼 LLM 재활용) |
| 목표 언어 | 영어 | 일본어 | 일본어 | 일본어 | 한국어 (+일본어 전이 실험) |
| 학습 데이터 | 실제+합성 대화 대량 | 실제 일본어 음성 대화(~67,000h, J-CHAT 등) | 실제 데이터 + 멀티스트림 TTS 합성 대화 추가 | J-CHAT 사전학습 + 자체 수집 LLM-jp-Zoom1로 파인튜닝 | **한국어 멀티턴 데이터 미사용.** 한국어 싱글턴(페어 구조, 합성 포함) + 영어 멀티턴 **semantic-only KD**(음색은 voice-prompt로 분리) |
| Mimi 코덱 처리 | 자체 학습 | **동결(freeze)** — 사전 실험에서 일본어 재구성 충분히 확인됨 | 동결 | 동결(추정) | 동결 유지 계획(단, 한국어 대상 round-trip 재구성 검증 별도 수행 필요) |
| 라이선스 | CC-BY 4.0 | CC BY-NC 4.0 (비상업) | CC BY-NC 4.0 | **Apache 2.0** (상업 이용 가능, 일본어 Full-duplex 최초) | (결정 예정) |
| 개발 주체 | Kyutai | 나고야대 東中 연구실 | 나고야대 東中 연구실 | NII/LLMC 주도, 와세다대·게이오대, 東中 교수 연속 참여 | (본 연구실) |
| 핵심 novelty | 최초의 실시간 full-duplex speech-to-speech LLM, split-RVQ Mimi, Inner Monologue | 일본어 최초 full-duplex 구현 | 합성 대화 데이터로 자연성 개선 | 라이선스 문제 해결 + 자체 데이터로 자연성/의미성 개선 | **멀티턴 데이터 없이 구조적 전이만으로 멀티턴 창발** + 백본 교체를 통한 언어 학습 비용 제거 + 시멘틱 토큰 기반 cross-lingual 전이 정량화 |

**J-Moshi 대비 Haan이 이미 검증된 사실 위에 놓이는 지점**: J-Moshi가 "Mimi를 동결해도 비영어권 언어(일본어) 음성을 충분히 잘 재구성할 수 있다"는 것을 사전 실험으로 확인했다는 사실은, Haan의 한국어 적용에서도 코덱 자체의 언어 편향 우려(WavLM이 영어 중심으로 학습되었다는 점에서 기인)를 완화하는 선례가 된다. 다만 이는 "코덱이 언어를 인코딩할 수 있는가(encoding)"의 문제이지 "처음 그 언어로 말하는 법을 배운 적 없는 멀티링구얼 백본이 발화를 생성할 수 있는가(generation)"와는 다른 질문이며, Haan의 컨트리뷰션은 정확히 후자에 집중되어야 J-Moshi/LLM-jp-Moshi와 차별화된다.

## 5. 컴퓨팅 예산

### 5.1 확보 자원
- **A100 80GB × 4장** (총 320GB)

### 5.2 FP8 관련 결론
A100(Ampere)은 FP8 텐서코어 가속을 지원하지 않음(Hopper 이상부터 지원). 따라서 행렬곱 자체를 FP8로 가속하는 것은 불가능. 대신 **8-bit optimizer**(bitsandbytes `PagedAdamW8bit` 등)로 옵티마이저 상태(momentum/variance) 메모리를 절감하는 방식이 사실상 동일한 효과를 제공하며 A100에서 실전 검증된 방법.

### 5.3 분산 전략: FSDP2 단일 스택 (파라미터 샤딩 불필요 판단)

**분산 프레임워크는 FSDP2(`torch.distributed.fsdp.fully_shard`, PyTorch 2.4+)로 단일화**한다. FSDP2는 별도 프레임워크 교체 없이 파라미터 그룹별 `reshard_after_forward` 플래그 하나로 ZeRO-2급/ZeRO-3급 동작을 전환할 수 있다. 두 모드 공통으로 **gradient·optimizer state는 항상 샤딩**되며, 차이는 오직 파라미터를 forward 후에도 샤딩할지 여부다.

| `reshard_after_forward` | 동등 개념 | 파라미터 처리 | 특성 |
|---|---|---|---|
| `False` (**기본 채택**) | ZeRO-2 (SHARD_GRAD_OP) | forward 후 unshard 유지 → all-gather 1회 절약 | 통신 ↓ / 메모리 ↑ · **처리량 우선** |
| `True` (폴백) | ZeRO-3 (FULL_SHARD) | forward 후 재샤딩 → backward 때 재-all-gather | 메모리 ↓ / 통신 ↑ |

**모델 크기 판단**: 9.15B 파라미터의 bf16 사본이 GPU당 약 16GB로 A100 80GB 한 장에 여유롭게 상주하므로, **파라미터 자체를 샤딩(ZeRO-3)할 필요가 없다**고 판단한다. 따라서 기본값은 `reshard_after_forward=False`(파라미터 복제 유지, grad/optim만 샤딩)이며, VRAM이 빠듯해질 때만 `True`로 폴백한다. (FSDP2에는 순수 ZeRO-1(옵티마이저만 샤딩) 모드가 없으며, 최소 단위가 grad+optim 샤딩인 ZeRO-2급이다.)

#### 메모리 계산 (GPU당, 4×A100 80GB, `reshard_after_forward=False` 기준)

| 항목 | fp32 master 유지 | bf16-only(8bit optim만) |
|---|---|---|
| bf16 파라미터 (복제, 미샤딩) | 16GB | 16GB |
| bf16 그라디언트 (샤딩, ÷4) | ~4GB | ~4GB |
| fp32 master weight (샤딩, ÷4) | ~8GB | (생략) |
| 8-bit optimizer states (샤딩, ÷4) | ~5GB | ~5GB |
| KD teacher(원본 Moshi 7B, bf16) — §5.3註 | +14GB | +14GB |
| **총합(대략)** | **~47GB** | **~39GB** |

註) KD teacher는 오프라인 로짓 덤프(teacher-forcing 구간)로 처리하면 GPU 상주가 불필요(§5 시간 단축 레버 1). on-policy KD 구간에서만 라이브 teacher(+14GB)가 필요하며, 이때도 파라미터 미샤딩 기준 GPU당 여유가 충분하다.

gradient checkpointing 병행 시 Phase 1~3 풀 파인튜닝(full FT)이 여유 있게 가능. **자원 제약을 이유로 LoRA로 타협할 필요는 없음** — LoRA/QLoRA 채택 여부는 순수하게 방법론적 근거(TRAINING_CURRICULUM.md 참조)로만 결정.

추천 스택: **FSDP2 `fully_shard`(`reshard_after_forward=False`)** + bf16 연산 + `PagedAdamW8bit` + gradient checkpointing + KD teacher는 `no_grad`/eval(또는 오프라인 덤프).

### 5.4 Phase별 예상 소요 시간 (근사치 — 실제로는 체크포인트 진단 결과에 따라 조기 종료/재설계 가능성 있음)

처리량 어림: 8B 백본 기준 프레임당 연산 ≈ 6×8e9 FLOPs, KD teacher 동반 시 +30% → 프레임당 ≈ 6.2×10¹⁰ FLOPs. A100×4 BF16 피크 1.25×10¹⁵ FLOPs/s, 현실적 MFU 35~40% 가정 → 유효 처리량 4.5~5×10¹⁴ FLOPs/s → 초당 약 7,000~8,000프레임(12.5Hz 기준 오디오-시간 환산 시 시간당 약 550~650h 상당 처리 가능).

| Phase | 목표 볼륨(추정) | 예상 기간 | 비고 |
|---|---|---|---|
| Phase 1 (영어 워밍업) | ~3,000~8,000h | 반나절~1일 | Qwen3의 기존 표현력 덕에 Moshi 원본보다 짧을 것으로 기대되나 검증 필요 |
| Phase 2 (조기 조인트) | ~8,000~15,000h | 1~2일 | Shortcut 형성 여부 체크포인트마다 모니터링 필수 |
| Phase 3 (본 학습) | ~15,000~30,000h | 1.5~3일 | 한국어 합성 페어 데이터 총량에 크게 좌우 |
| Phase 4 (일본어 스윕, LoRA/QLoRA) | 4개 조건(0/1/10/100h) | 조건당 수시간, 총 1일 이내 | 데이터량 작고 LoRA라 빠름 |
| Phase 5 (Ablation 전체) | Phase 1~3 규모 반복 | 7~12일 | 가장 시간 소요 큰 구간 |

**총합**: 본 트랙(Phase 1~4) 약 5~7일, Ablation 포함 시 2~3주. 여기에 디버깅/재시작/파이프라인 구축 버퍼로 1.5~2배를 추가 고려할 것.

**시간 단축 레버**:
1. Teacher-forcing 기반 KD 구간은 티처 출력을 오프라인으로 사전 캐싱 — 매 스텝 라이브 teacher forward 비용(+30%) 제거. 이것이 KD의 **기본 모드**이며 학습량 대부분을 여기서 처리한다. On-policy KD 구간(exposure bias 교정용, `RISKS_AND_DIAGNOSTICS.md` §7.3)에서만 라이브 teacher가 필요하므로, 이 구간을 학습 후반/주기적으로 **선택적으로** 켜 비용을 국소화한다.
2. Phase 5 ablation 중 초반 수렴 양상만으로 판단 가능한 축(예: 커리큘럼 타이밍)은 축소 규모 실험으로 대체 가능.

## 6. 문서 구조

- `PROJECT.md` (본 문서) — 개요, 컨트리뷰션, 관련 연구, 컴퓨팅 예산
- `ARCHITECTURE.md` — 백본 교체, 임베딩 구조, self/user 스트림 분리 문제와 공유 임베딩+Role Token 설계, KD codebook routing(semantic 중심), voice-prompt/IT 텍스트 채널, Mimi 코덱 구조, **Role 구분 후속 연구 로드맵**(구 `FUTURE.md` 통합)
- `TRAINING_CURRICULUM.md` — Phase 0~5 학습 커리큘럼, Joint loss 설계, Phase별 LoRA/QLoRA 판단, **커널 최적화 구현 노트**(구 `KERNELS.md` 통합)
- `DATA_STRATEGY.md` — 싱글턴 데이터 정의와 선택, 합성 파이프라인, voice-prompt/ko-ka interpolation, 양방향 학습 설계
- `RISKS_AND_DIAGNOSTICS.md` — 실패 메커니즘과 진단 방법
- `PERSONA.md` (별도 관리) — Haan의 캐릭터/페르소나 설정. 본 기술 문서들과 독립적으로 유지.
