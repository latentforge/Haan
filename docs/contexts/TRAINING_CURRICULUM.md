# Haan 학습 커리큘럼

## 1. 전체 설계 철학

Qwen3 백본은 오디오 토큰을 다뤄본 적이 없으므로 "말하는 법"부터 배워야 하지만, 텍스트 레벨의 한국어/일본어 이해·추론 능력은 이미 보유하고 있다. 따라서 커리큘럼의 핵심은 **언제, 어떤 순서로 (1) 오디오 모달리티 접합, (2) turn-taking 구조, (3) 한국어 음성 실현**을 노출시키느냐이다.

### 1.1 완전 순차(영어 완성 후 한국어) vs 완전 동시(step 0부터 조인트) 비교

- **완전 순차의 위험**: 영어만으로 먼저 멀티턴 능력을 정착시키면 turn-taking 회로가 critical learning period(Achille et al. 등에서 논의되는, 초기 학습 구간에 형성된 표현이 이후 학습으로 잘 뒤집히지 않는 현상) 동안 "영어 음향 토큰 분포에 조건부로만 작동"하도록 강하게 굳어질 위험이 있다. 이후 한국어를 붙이면 이미 고착된 회로를 "깨고 재배선"해야 해, 처음부터 같이 준 것보다 훨씬 어렵다. 즉 **늦게 넣을수록 전이 실패 위험이 커진다.**
- **완전 동시의 위험**: 학습 극초반에는 모델이 아직 오디오 토큰을 일관되게 생성하지 못하는 상태(임베딩만 코드북에서 초기화됐을 뿐, Qwen3 레이어들은 오디오 시퀀스를 처리한 적이 없음)이므로, 이 시점에 KD 신호와 TTS 신호가 둘 다 노이즈에 가까운 상태로 섞이면 그라디언트가 상쇄되거나 방향을 못 잡을 수 있다.

→ **결론: 짧은 워밍업 + 조기 조인트** 전략 채택 (완전 순차도, 완전 동시도 아님).

## 2. Phase 구성

### Phase 0. 준비 (학습 전)
- 백본: Qwen3 8B + 오디오 RVQ 토큰 임베딩, self/user 공유 임베딩 + Role Token 적용 (`ARCHITECTURE.md` §3). **임베딩 초기값은 Moshi의 user 쪽 테이블 `emb.8~15`에서 복사** — user 쪽만 다화자 분포에서 학습돼 고정 화자 편향이 없음 (`ARCHITECTURE.md` §5.4.2)
- Depth Transformer: **공유 1개 + 공유 프로젝션 + role embedding additive**, batch 2 병렬. **학습 q16(batch 2) / 라이브 추론 q8(batch 1) 모드 스위치** 구현 (`ARCHITECTURE.md` §5.4, §5.0.3)
- **Warm-start** (`ARCHITECTURE.md` §5.4.1): 백본 dim이 Helium·Qwen3 모두 **4096으로 일치**하고 Depth dim(1024)은 백본과 무관하므로 —
  - Depth 본체(per-index 파라미터 포함) / `linears.0~7` / `depformer_emb` → **Moshi에서 warm-start**(동결 Mimi 공유로 audio cardinality 2048 동일)
  - 오디오 입력 임베딩(공유 8개) → **Moshi `emb.8~15`(user 쪽)에서 복사**
  - $z_s$ 프로젝션(4096→1024) → Moshi 가중치로 **초기화하되 재학습**(두 4096 공간은 정렬돼 있지 않음)
  - 텍스트 임베딩/헤드 → 전이 불가(32000 vs 151936), Qwen3 자체 사용
- **Delay 설정** (Table 1 준용): acoustic delay 2(pre) → 1(이후), text delay ±0.6(pre) → 0(이후)
- **토크나이저 설정** (`ARCHITECTURE.md` §7.6): 예약 특수 토큰 슬롯에 **PAD/EPAD 신규 배정**(`<|im_end|>`/`<|im_start|>` 재사용 금지). **배치 `pad_token`과 스트림 PAD를 분리** — 배치 pad는 loss 완전 마스킹, 스트림 PAD는 예측 대상이되 다운웨이팅.
- **Instruction template 확정** (`ARCHITECTURE.md` §7): Zone A 정적 ChatML prefix(dense, loss 마스킹) → Zone B 오프닝 인사 voice prompt(정렬 전사 포함) → Zone C 대화(PAD/EPAD, 턴 마커 없음). `<|im_end|>`는 세션 종료에만.
- **텍스트 전용 사전 프로브** (`ARCHITECTURE.md` §7.7, 오디오 파이프라인 불필요): ① 닫히지 않는 assistant + PAD 스트림에서 지시 준수·품질 유지 여부(non-thinking 모드 기준) ② "발화 완결=PAD 전환" 재매핑 학습 속도
- **라이브 추론은 non-thinking 모드**로 운용(`/no_think` 계열 스위치 또는 `enable_thinking=False`) — 텍스트 채널 12.5 tok/s 제약 회피 (`ARCHITECTURE.md` §7.5)
- 텍스트 능력 anchor 데이터셋 확보: 순수 텍스트 loss를 전 구간에 소량 섞을 한국어/영어/일본어 텍스트 코퍼스 (catastrophic forgetting 방지용)
- 진단용 프로빙 세트 준비: turn-taking 관련 내부표현을 측정할 causal probe, 영어 held-out 멀티턴 평가셋, 한국어 오디오 프리픽스(학습엔 미사용, 프로빙 전용)
- Mimi 한국어 round-trip 재구성 테스트 (`ARCHITECTURE.md` §4.3)

### Phase 1. 영어 단독 워밍업 (전체 스텝의 약 5~10%)
- **목적**: "영어 멀티턴 마스터"가 아니라 오디오 토큰 시퀀스 생성이라는 새 모달리티에 대한 **bootstrap**.
- 데이터: 영어 멀티턴 KD only (원본 Moshi를 teacher로 사용)
- Loss: **semantic 토큰(level-0) logit KD** + 텍스트 anchor loss(소량). acoustic 코드북은 KD 대상에서 제외(`ARCHITECTURE.md` §5.1). 텍스트 채널은 Qwen3 IT 포맷 보존.
- 임베딩 처리: 코드북 임베딩은 낮은 lr 혹은 짧은 freeze 후 warmup (드리프트 방지)
- 종료 조건: "영어 대화 품질"이 아니라 "모달리티 안정성"(오디오 토큰이 붕괴하지 않고 그럴듯한 시퀀스를 생성하는지) 기준
- **체크포인트 A**: 한국어 오디오 프리픽스에 대해 turn-taking 관련 표현이 조금이라도 활성화되는지 프로빙 (사전 신호 확인용)

### Phase 2. 조기 조인트 진입 (critical period 구간)
- **목적**: turn-taking 회로가 본격 형성되는 시점에 두 언어 신호를 함께 노출시켜, 언어 종속적 shortcut이 아닌 언어 독립적 표현으로 유도.
- 데이터: 영어 멀티턴 semantic KD + 한국어 싱글턴 (데이터 상세는 `DATA_STRATEGY.md` 참조)
- **voice-cloning 학습 도입**: 한국어 싱글턴 학습 시 다양한 참조 음성을 in-context voice-prompt로 노출해 "프롬프트대로 음색 생성"을 심는다(voice-cloning CE). 고정 화자 편향 해소의 실질 레버. (`ARCHITECTURE.md` §5.2, `DATA_STRATEGY.md` §3)
- 비율: 한국어 데이터 비율을 **낮게 시작해 점진적으로 상승**(예: 10% → 30% → 50% 램프업)
- Loss 설계 주의사항:
  - 세 신호(semantic KD / 한국어 TTS / voice-cloning CE)의 loss scale/gradient norm 모니터링 — 한쪽이 지배하지 않도록 정규화
  - 침묵/비발화 프레임 클래스 불균형은 **semantic KD 내부의 프레임 가중치**로 처리(발화 전환 구간 가중치). 별도 보조 loss가 아님(self/user 구분은 Role Token, 발화/침묵은 semantic 토큰이 이미 담당하므로 중복 항 불필요) — `RISKS_AND_DIAGNOSTICS.md` §7.4
  - **토큰 종류별 loss 가중치** (PersonaPlex 준용, `ARCHITECTURE.md` §7.6): 스트림 **PAD 텍스트 토큰 ×0.3**, **non-semantic 오디오 토큰 ×0.02**를 기본값으로 두고 튜닝. 텍스트 채널의 PAD 편중(약 65%) 대응.
  - **시스템 프롬프트(Zone A) 구간은 loss 마스킹** — 모델이 프롬프트를 "생성하는 법"을 배우지 않고 조건으로만 쓰도록.
  - 텍스트 anchor loss 계속 유지
- 언어-태스크 상관관계 완화: 가능하다면 영어 데이터 일부를 싱글턴처럼 잘라 섞어, "언어"와 "턴 구조"가 완벽히 상관되지 않도록 배치 구성 (`RISKS_AND_DIAGNOSTICS.md` §1 참조)
- **체크포인트 B**: 영어 held-out 멀티턴 성능이 한국어 투입 이후 급격히 떨어지는지(간섭 여부) 추적. 한국어 프리픽스 turn-taking 프로빙 재측정 — Phase 1 대비 상승 여부가 핵심 신호

### Phase 3. 본 학습 (조인트 지속)
- 한국어 비율이 목표치에 도달한 뒤 안정적으로 유지하며 지속.
- 주기적으로 (A) 영어 멀티턴 유지력, (B) 한국어 창발 turn-taking 품질, (C) 순수 텍스트 능력(멀티링구얼 anchor) 세 가지를 동시 트래킹 — 셋 중 하나라도 무너지면 loss weight 재조정.
- **체크포인트 C**: 한국어 멀티턴 능력의 창발 여부 1차 판정 (메커니즘/타이밍 vs 내용/일관성 분리 평가, `RISKS_AND_DIAGNOSTICS.md` §2 참조)

### Phase 3.5. Acoustic prosody graft (선택적, guard로 조기 종료)
- **목적**: barge-in 시 pitch 상승 등 acoustic에 실린 paralinguistic 다이내믹스를 전이. semantic-only로는 안 넘어오는 부분.
- **전제 판단(measure-first)**: Phase 3의 Full-Duplex-Bench 결과에서 미세 타이밍/자연스러움 부족이 유의미할 때만 진입. 1차 turn-taking은 semantic+타이밍이 대부분 캐리하므로 불필요하면 생략.
- **방식** (`ARCHITECTURE.md` §5.3): 성숙한 클로닝 위에 turn-event 국소 acoustic KD를 짧게 graft.
  1. 페어드 voice-prompt (KD 타깃마다 대응 목소리 prompt)
  2. ko-ka interpolation 다화자화(중간 목소리 연속체로 KD 타깃 음색 다양화)
  3. 저weight/저LR + add (semantic KD·cloning CE 유지, swap 아님)
  4. barge-in/overlap 프레임에만 acoustic KD
  5. **guard로 early-stop**: turn-event 구간 화자 유사도로 클로닝 드리프트 감시, 임계 초과 또는 prosody 이득 plateau 시 정지
- **LoRA 여부**: Full FT 유지(본 트랙 통제 일관성). 짧은 구간이라 비용 작음.

### Phase 4. 일본어 전이 실험 (별도 트랙)
- Phase 3 완료 체크포인트("한국어 말하기 가능 + 멀티링구얼 유지")를 베이스로 분기.
- 일본어 데이터 양 스윕: 0h(완전 제로샷) → 1h → 10h → 100h, 각각 파인튜닝 후 성능 곡선 측정.
- J-Moshi(약 67,000시간)와의 데이터 효율 비교가 핵심.
- 본 트랙(한국어) 오염 방지를 위해 별도 브랜치로 분리 실행.

### Phase 5. Ablation & 최종 검증
- **커리큘럼 타이밍 자체를 실험 축으로**: (1) 완전 순차 (2) step-0 완전 조인트 (3) 워밍업+조기조인트, 세 조건 비교 — 이 비교 자체가 "언제 다른 언어 신호를 섞어야 cross-lingual transfer가 잘 일어나는가"라는 독립적 finding이 되며, critical period 가설의 실증적 검증이 됨.
- **★ semantic-distill on/off ablation (핵심 하중 축)**: 같은 Mimi에서 level-0를 WavLM-distilled semantic VQ vs 순수 acoustic VQ로 바꿔 **동일 파이프라인 전이 성공률**을 비교. "Mimi semantic 토큰이 전이의 원인"임을 통제된 델타로 증명 — 팀의 코덱 레벨 핵심 명제("Mimi가 다른 RVQ와 달리 semantic 토큰을 쓰는 이점"). 코덱 자체를 바꾸는 비교(예: Qwen3-Omni 계열)는 교란 변수가 많아 보조 arm. (선행: Qwen3-Omni 토크나이저에 semantic 채널이 실제로 없는지 스펙 확인)
- **KD-scope ablation**: semantic-only KD vs semantic+acoustic KD → 음색 유출 vs 타이밍/prosody 전이의 트레이드오프 정량화(Phase 3.5 graft 효과 포함).
- **voice-prompt / interpolation ablation**: voice-prompt 유무, ko-ka interpolation 유무가 클로닝·prosody graft에 주는 효과.
- **User-stream 예측 구조 ablation (PersonaPlex 순차 vs Haan 병렬)**: PersonaPlex식 16-step 순차 Depth vs 제안한 공유-Depth batch-2 병렬(split MLP + role emb, `ARCHITECTURE.md` §5.4)을 **품질 동등성 + latency**로 비교. 순차는 warmup 단계가 아니라 **대조군 baseline**으로 별도 학습. 지표는 특히 **user 예측 품질 + 오버랩/barge-in 자연스러움** 두 축(프레임 내 상관 손실이 실제로 유의미한지 판정). 필요 시 병렬에 경량 cross-attention 복원안 비교 포함.
  - **사전 파일럿(de-risk)**: 본 학습 컴퓨트 투입 전, 소규모로 순차 vs 병렬을 붙여 위 두 지표의 동등성만 선확인한 뒤 병렬로 커밋. (순차 풀학습 후 전환이 아니라 소규모 bake-off)
- 코드북 init ablation: **Moshi user 테이블(`emb.8~15`, 채택) vs self 테이블(`emb.0~7`) vs random init**. 채택안의 근거(user 쪽이 다화자 분포)가 실제로 클로닝·화자 일반화에서 이득으로 나타나는지 정량 확인 (`ARCHITECTURE.md` §5.4.2).
- KD 방식(teacher-forcing only vs on-policy 혼합) ablation.
- Self/user 임베딩 분리(원 Moshi) vs 공유+Role Token(제안) ablation. (semantic만 공유 vs semantic+acoustic 모두 공유 축 포함, `ARCHITECTURE.md` §3.6·§6.2)

**우선순위**: 리소스가 제한적이면 Phase 0~3(본 트랙)을 먼저 완주해 핵심 결과("한국어 멀티턴 창발")부터 확보하고, Phase 4·5는 서브 컨트리뷰션으로 뒤에 배치하는 것이 리스크 관리 측면에서 안전.

## 3. Phase별 LoRA / QLoRA 사용 여부 분석

### 3.1 공통 전제
오디오 RVQ 임베딩 테이블, RQ-Transformer(Depth Transformer) 출력 헤드, Role Token(학습형 additive 벡터 2개)은 Qwen3에 원래 없던 **완전히 새 파라미터**이므로, LoRA는 이들에 적용될 수 없다(LoRA는 기존 가중치 행렬에 저랭크 보정을 더하는 방식). 따라서 이들은 **어느 Phase든 항상 풀 파라미터로 학습**한다. 이하 논의는 "Qwen3 백본(어텐션/FFN)을 얼릴지 여부"에 대한 것이다.

### 3.2 Phase별 판단

| Phase | 권장 | 근거 |
|---|---|---|
| Phase 1 (워밍업) | **Full FT** | 백본이 오디오 모달리티를 처음 다루는 단계로, 32개 레이어 전반의 깊은 표현 재구성이 필요. LoRA의 저랭크 제약이 병목이 될 가능성. |
| Phase 2 (조기 조인트) | **Full FT (특히 중요)** | 저랭크 어댑터는 손실을 줄이는 가장 "값싼" 해를 찾는 경향이 강함. Turn-taking의 언어 독립적 일반 표현보다 "한국어=싱글턴, 영어=멀티턴" 같은 얕은 shortcut이 저랭크 입장에서 더 쉬운 해일 수 있음. 이 논문의 핵심 가설이 성립하려면 정확히 이 구간에서 LoRA를 피하는 것이 안전. |
| Phase 3 (본 학습) | **Full FT (Phase 1~2와 동일 방식 유지)** | 중간에 freeze+LoRA로 전환하면 그 자체가 교란 변수가 됨. |
| Phase 4 (일본어 스윕) | **LoRA/QLoRA 권장** | 핵심 회로가 이미 Phase 3에서 형성 완료된 상태 → "이미 있는 회로를 표면 데이터에 맞춰 얕게 조정"하는 정확히 LoRA가 잘하는 영역. 극소량 데이터(1h 등) 조건에서는 LoRA의 낮은 용량이 오히려 overfitting/파괴적 망각을 막는 자연스러운 정규화 역할. Base 체크포인트를 그대로 두고 조건별 어댑터만 교체 가능해 "본 트랙 오염 방지" 요구와 부합. QLoRA는 제한된 자원으로 여러 데이터 조건을 빠르게 실행하는 데 유리. |
| Phase 5 (커리큘럼 타이밍 ablation) | **Full FT** | 세 조건 모두 본 트랙과 동일 방식으로 통제해야 공정한 비교. |
| Phase 5 (코드북 init ablation) | **LoRA 여부와 무관** | 임베딩 테이블은 항상 풀 학습이므로 이 축의 결론은 LoRA 여부에 영향받지 않음. |
| Phase 5 (KD 방식 ablation) | **LoRA/QLoRA로 계산량 절감 가능** | On-policy KD는 학습 중 student 롤아웃 생성이 필요해 계산 비용이 이미 크게 증가 — 이 ablation만큼은 실용적으로 LoRA 사용 고려. |

### 3.3 자원 재검토 (A100 80GB×4 확보 이후)
FP8은 A100(Ampere)에서 텐서코어 가속이 불가능(Hopper 이상 지원)하지만, 8-bit optimizer(`PagedAdamW8bit`)로 동일한 메모리 절감 효과를 얻을 수 있다.

**분산 전략은 FSDP2(`fully_shard`)로 단일화**한다. FSDP2는 `reshard_after_forward` 플래그 하나로 ZeRO-2급(`False`)/ZeRO-3급(`True`)을 전환하며, 두 모드 모두 grad·optimizer state는 항상 샤딩한다. **9.15B 파라미터의 bf16 사본(GPU당 ~16GB)이 A100 80GB에 여유롭게 상주하므로 파라미터 자체를 샤딩할 필요는 없다는 판단** 하에, 기본값을 `reshard_after_forward=False`(파라미터 복제 유지, grad/optim만 샤딩)로 두고 VRAM이 빠듯할 때만 `True`로 폴백한다. 이 조합 + 8-bit optimizer 시 GPU당 약 40-47GB 수준(파라미터 미샤딩 포함)으로, **Phase 1~3의 Full FT가 자원 제약 없이 가능**하다 (상세 계산은 `PROJECT.md` §5.3 참조). 따라서 Phase 1~3의 Full FT 권장은 순수하게 방법론적 근거(shortcut 방지)로만 유지하면 되며, 자원 부족을 이유로 한 타협은 필요하지 않다. Phase 4의 LoRA/QLoRA 권장은 자원 문제가 아니라 저데이터 정규화·베이스 보존·빠른 조건 스와핑이라는 방법론적 이유이므로 자원 여유와 무관하게 유효하다.

## 4. Joint Loss 설계 시 주의사항 (Phase 2~3 공통)

- 영어 멀티턴 KD loss와 한국어 싱글턴 TTS loss의 scale/gradient norm이 크게 다르면 한쪽이 다른 쪽을 지배할 위험 → loss weighting, 혹은 PCGrad류 gradient surgery 기법 고려. 최소한 loss weighting ablation 리포트.
- 텍스트 anchor loss(멀티링구얼 능력 보존용)를 전 구간 소량 유지 — 이것이 무너지면 Phase 4(일본어 전이) 실험 자체가 무의미해짐.

## 5. 구현 노트: 커널 최적화 (구 `KERNELS.md` 통합)

Phase 1~3 Full FT를 4×A100에서 처리량/메모리 여유 있게 돌리기 위한 `transformers` 커널 통합 정리. 백본이 Qwen3라 아래 커널 경로가 직접 적용된다.

### 5.1 학습 파이프라인 — Liger Kernel (가장 직접적)
`TrainingArguments(use_liger_kernel=True, liger_kernel_config={...})`로 RMSNorm·RoPE·SwiGLU·CrossEntropy·FusedLinearCrossEntropy를 단일 Triton 커널로 융합. **FlashAttention·FSDP·DeepSpeed와 호환**되며 멀티GPU 처리량↑·메모리↓ → 더 큰 batch/context 가능. 본 프로젝트의 FSDP2 스택(`PROJECT.md` §5.3)과 병용 대상.

```python
args = TrainingArguments(
    use_liger_kernel=True,
    liger_kernel_config={"rope": True, "swiglu": True, "cross_entropy": True,
                          "fused_linear_cross_entropy": True, "rms_norm": True},
)
```

### 5.2 Hub 커널 로딩 — `use_kernels` / `KernelConfig` / `attn_implementation`
- `from_pretrained(..., use_kernels=True)`: 디바이스에 맞는 최적 커널을 Hub에서 자동 로드(RMSNorm, MoE MLP, SwiGLU, RoPE, Causal LM loss 등), 등록 기본값 없으면 표준 PyTorch 폴백. (`kernels>=0.11.0`)
- `attn_implementation="kernels-community/flash-attn2"` 등으로 어텐션 커널 직접 지정. `sdpa`/`eager`/Hub 커널이 동일 `AttentionInterface` 뒤에서 교체 가능.
- `KernelConfig`로 레이어별 매핑·디바이스별 구현·모듈 퓨전·로컬 커널 제어. 신뢰 정책: `kernels-community` 외 저장소는 `allow_all_kernels=True` 필요(임의 코드 실행 위험). 전역 off: `USE_HUB_KERNELS=0`.
- 통합 지점 데코레이터 `@use_kernel_forward_from_hub("RMSNorm")` — 모델 코드가 Hub 커널을 가리키는 포인터.

### 5.3 MXFP4 참고
`transformers`는 MXFP4(4-bit, E2M1+blockwise)를 네이티브 지원하며 Hub 커널(`kernels-community/triton_kernels`)을 자동 pull. 단 **A100(SM8.0)은 요구 compute capability·FP8/MXFP4 가속 대상이 아니므로**(cc≥7.5 GEMM 조건은 충족하나 실효 가속은 Hopper 이상) 본 프로젝트는 bf16 + 8-bit optimizer 경로가 기본(`PROJECT.md` §5.2).

### 5.4 통합 메커니즘 두 트랙 (커널을 직접 다룰 때 주의)
- **Hub 커널 방식**(FlashAttention/Liger/MXFP4): 런타임 문자열 인자로 선택, Hub 사전빌드 바이너리 자동 다운로드, 표준 PyTorch 폴백.
- **선택적 의존성 방식**(FLA / Qwen3-Next Gated DeltaNet): 모듈 레벨 `if/else` import + 설치 여부로 로드시점 결정, 레지스트리·런타임 스위치 없음, 파일 내 하드코딩 PyTorch 재구현으로 폴백. Qwen3 계열에서 Gated DeltaNet 변형(Qwen3-Next)을 쓸 경우 `fla`/`causal_conv1d` 설치 여부가 유일한 제어 수단이며, 해당 아키텍처용 사전빌드 부재 시 영구 폴백됨에 유의.

### 5.5 참고 링크
- https://huggingface.co/docs/transformers/kernels
- https://huggingface.co/docs/transformers/en/attention_interface
- https://huggingface.co/blog/hello-hf-kernels
- https://github.com/fla-org/flash-linear-attention
