# Haan 아키텍처 설계

## 1. 백본 교체: Helium → Qwen3 8B

### 1.1 동기
Moshi 원본은 Helium(Kyutai 자체 학습 7B, 영어 2.1T 토큰)을 백본으로 쓴다. J-Moshi/LLM-jp-Moshi 계열은 이 Helium을 일본어 데이터로 파인튜닝하는 방식을 택했는데, 이는 "모델에게 일본어 자체를 가르치는" 비용이 그대로 든다는 뜻이다.

Haan은 이 비용을 생략하기 위해 **Helium을 Qwen3 8B로 교체**한다. Qwen3는 이미 강력한 한국어/일본어/영어 이해 및 생성 능력을 텍스트 레벨에서 보유하고 있으므로, 남은 문제는 "언어를 가르치는 것"이 아니라 **"이미 아는 언어를 오디오 모달리티로 실현하는 법을 가르치는 것"**으로 축소된다.

### 1.2 확장 방식
- Qwen3 8B에 오디오 RVQ 토큰용 임베딩을 추가해 어휘를 확장.
- 임베딩 초기화는 Moshi의 코드북 RVQ 임베딩 값을 사용 (아래 §2에서 이 초기화가 self/user 스트림에 각각 어떻게 적용되어야 하는지 상세 논의).

## 2. Self/User 오디오 스트림 구조 (Moshi 공식 구현 분석 기반)

### 2.1 확인된 사실 (Moshi 공식 GitHub 레포, `moshi/moshi/models/lm.py` 및 `configs/moshi_7b_202409.json` 직접 확인)

```python
self.emb = nn.ModuleList(
    [EmbeddingFactory(self.card + 1, dim) for _ in range(n_q)]
)
...
for cb_index in range(self.num_audio_codebooks):
    audio_emb = self.emb[cb_index](input_sequence[:, cb_index + self.audio_offset])
    input_ = audio_emb if input_ is None else input_ + audio_emb
```

실제 배포 설정(`moshi_7b_202409.json`): `n_q=16, dep_q=8`.

- `self.emb`는 **16개의 완전히 분리된 `nn.Embedding`**이다.
  - 인덱스 0~7: Moshi 자신(self)의 오디오 코드북 (시멘틱 1 + 어쿠스틱 7)
  - 인덱스 8~15: 유저(user)의 오디오 코드북 (동일 레벨 구조)
- 물리적으로 동일한 Mimi 코덱에서 나온 같은 코드 ID라도, self와 user는 **서로 다른 파라미터 행렬**을 사용하며 학습 중 그라디언트가 독립적으로 흐른다.
- 모든 코드북(self 8개 + user 8개)의 임베딩은 **elementwise sum**으로 하나의 벡터로 합쳐진 뒤 Temporal Transformer에 입력된다. Concat이 아니라 sum이라는 점이 핵심.

### 2.2 왜 sum 구조에서는 임베딩을 공유하면 안 되는가 (설계 원리 추론)

Sum 기반 융합에서 만약 self/user가 임베딩 테이블을 공유한다면: "Moshi가 코드 42를 말했다"와 "유저가 코드 42를 말했다"가 **완전히 동일한 벡터**를 그 합산에 기여하게 된다. Concat이었다면 슬롯 위치로 최소한의 구분이 남지만, sum은 그 정보를 지운다. 결국 트랜스포머가 "이건 내가 방금 한 말인지, 상대가 방금 한 말인지"를 구별할 방법이 원천적으로 사라진다.

Moshi 아키텍처의 존재 이유가 "말하기와 듣기를 분리된 스트림으로 모델링해 turn-taking을 학습한다"는 것이므로, 임베딩을 공유하면 이 핵심 설계 목적이 무너진다. **원 Moshi의 완전 분리 설계는 sum 기반 융합 방식과 짝을 이루는 필수적 선택으로 보인다** (단, 이는 논문에 명시된 문장이 아니라 코드 구조로부터의 추론).

### 2.3 Haan 프로젝트에서의 문제: Cross-lingual/Cross-role transfer와의 충돌

원 Moshi 설계(완전 분리 임베딩)를 그대로 따를 경우:
- 한국어 싱글턴 TTS 방향(Haan이 "말하는" 것)만 학습하면, 매 타임스텝 유저 채널에는 실제 콘텐츠 없이 침묵/패딩 토큰만 들어간다.
- 유저 스트림 쪽 임베딩 행렬(인덱스 8~15)은 한국어 콘텐츠에 대한 그라디언트를 **한 번도 받지 못하고** 초기화값(Moshi 코드북 값)에 머무른다.
- 더 결정적으로, 트랜스포머 자체가 "합산 벡터의 유저 채널 성분에 침묵이 아닌 실제 한국어 콘텐츠가 들어있는 상황"을 학습 중 한 번도 마주치지 않는다.
- 즉 "말하기만 배우면 듣기가 따라온다"는 창발은 이 구조에서는 근거가 약하다.

## 3. 개선 설계: 공유 임베딩 + Role Token

> **이번 연구의 채택안은 Role Token(학습형 additive 벡터)이다.** 회전 축 기반의 Role-axis RoPE는 다중 화자 확장을 겨냥한 **별도 후속 연구(§6.3 논문 2)**로 분리했으며, 본 트랙(한국어 전이)에서는 채택하지 않는다. 본 절 말미(§3.4)에서 두 방식을 비교하고 Role Token을 택한 이유를 정리한다.

### 3.1 핵심 아이디어
오디오 임베딩 테이블은 self/user가 **공유**(=8권)하고, 대신 역할(role) 구분을 **학습형 role 토큰(additive 벡터)**으로 임베딩 합산 단계에서 주입한다.

이렇게 하면:
- 임베딩 테이블이 공유되므로, self 스트림(한국어 TTS 방향) 학습으로 갱신된 코드 표현이 동일한 코드가 user 스트림에 나타날 때도 즉시 재사용된다 — §2.3에서 지적한 "유저 채널이 전혀 학습되지 않는" 문제가 구조적으로 해소된다.
- Role 구분은 `RoleEmb[role_id]`(role_id ∈ {0=self, 1=user}) 를 오디오 임베딩에 더해 부여하므로, self-코드42와 user-코드42는 서로 다른 벡터로 도달해 트랜스포머가 구별할 수 있다.

### 3.2 왜 "그냥 공유"로는 부족한가 (중요한 함정)

self/user 두 스트림은 **매 프레임 동일한 글로벌 타임스텝 t**에서 나란히 진행된다(오버랩·끼어들기 모델링을 위해 필수). 만약 임베딩을 공유하면서 role 구분 신호를 전혀 주지 않으면, self-코드42와 user-코드42가 **완전히 동일한 벡터**로 합산에 기여해 트랜스포머가 "내가 한 말인지 상대가 한 말인지"를 구별할 수 없게 된다(§2.2와 동일한 붕괴). 따라서 role을 명시적으로 표시하는 **별도 신호**가 반드시 필요하다.

### 3.3 구체적 설계: Role Token (학습형 additive)

- 오디오 임베딩 테이블: self/user 공유, semantic + acoustic 전체 공유
- Role 구분: `RoleEmb[role_id]`(role_id ∈ {0=self, 1=user}) 를 각 프레임의 오디오 임베딩 합산 벡터에 **additive**로 더함 (self/user 각 1개, 총 2개의 학습 벡터)
- 시간축 위치 인코딩(Qwen3 기존 1D RoPE)은 **그대로 유지** — 이 설계는 백본의 포지셔널 체계를 건드리지 않는다.

Role은 애초에 시간과 무관한 **태그**이므로 시간축 RoPE와는 별개 메커니즘으로 두는 것이 자연스럽다. (초기 논의에서 role 구분을 시간축 RoPE 자체를 role별로 다르게 쓰는 방식으로 검토했으나, 이는 self/user 간 동기화(같은 timestep 관계)를 깨뜨릴 위험이 있어 기각했다. §6.2 참조.)

### 3.4 Role Token vs Role-axis RoPE — 왜 이번 연구는 Role Token인가

회전 축 기반(M-RoPE 스타일)으로 role을 고정 오프셋(self=0, user=Δ)으로 부여하는 대안도 있다. "임베딩 공유 유지 + role만 구분"이라는 **본 트랙의 목적에는 두 방식이 대체로 동등**하다. 그러나:

- **Role Token(채택)**: 구현이 간단하고 Qwen3 백본의 기존 RoPE 체계를 전혀 건드리지 않는다. 이진 self/user 구분에는 충분하다.
- **Role-axis RoPE(미채택, §6.3 논문 2)**: 결정론적 회전이라 role 구분이 학습으로 희석되지 않고, **임의 개수의 동시 화자(dyad→multi-party)로 구조적 외삽**이 가능하다는 강점이 있다. 다만 이 강점(다중 화자 확장)은 본 트랙(1:1 한국어 전이)의 목표가 아니며, 백본 RoPE 사후 개조라는 추가 검증 부담이 있다. 따라서 **다중 화자 확장을 정면으로 다루는 별도 논문**으로 분리한다.

두 방식의 정교한 비교(및 원 Moshi 완전 분리 설계와의 비교)는 §3.6 ablation에서 다룬다.

### 3.5 리스크 및 통합 시 주의점

1. **role 구분 신호의 강건성**: role token은 학습되는 파라미터라, 다른 loss 압력에 밀려 role 구분 신호가 약해지거나 다른 정보와 얽힐 이론적 위험이 있다. 학습 후 self/user role 벡터의 분리도(코사인 유사도)와, user 콘텐츠가 실제로 self와 구분되어 처리되는지를 프로빙으로 점검한다.
2. **동기화 보존**: self/user는 반드시 동일 타임스텝 t를 공유해야 오버랩/끼어들기 모델링이 유지된다. role token은 시간축과 독립적인 additive 태그이므로 이 동기화를 건드리지 않는다(이 점이 role token의 이점 중 하나).
3. **기존 acoustic delay 메커니즘과의 상호작용**: Moshi는 코드북 레벨 간 인위적 지연(delay)을 이미 사용 중이며(§5 참조). role token 추가가 이 메커니즘과 간섭하지 않는지 확인 필요.

### 3.6 컨트리뷰션으로서의 프레이밍
이는 단순 구현 디테일이 아니라 원 Moshi 설계(완전 분리 임베딩)와 다른 아키텍처적 선택이며, cross-lingual/cross-role transfer라는 연구 목표에 맞춰 재설계된 것이다. Ablation 후보: **(1) 원 Moshi 완전 분리(baseline) vs (2) 공유 임베딩 + Role Token(제안) vs (3) semantic만 공유 vs semantic+acoustic 모두 공유**. (참고로 공유+Role-axis RoPE 대안과의 비교는 §6.3 논문 2의 범위.) 특히 (1) vs (2)의 한국어 창발 성공률 차이가 본 트랙의 핵심 방법론적 기여다.

## 4. Mimi 코덱 구조와 언어 편향 고려사항

### 4.1 Split RVQ 구조
Mimi는 단일 계층적 RVQ가 아니라 **split-RVQ**를 사용한다: 첫 레벨은 WavLM(자기지도학습 SSL 모델)으로부터 시멘틱 정보를 증류받은 단순 VQ이고, 이와 병렬로 7단계 acoustic RVQ가 적용되어 둘의 출력을 합산한다. 이는 시멘틱 양자화기의 잔차에 음향 정보가 반드시 보존되어야 한다는 제약을 없애, "무슨 말을 했는가(semantic)"와 "어떻게 들리는가(acoustic)"를 설계상 명시적으로 분리한다.

### 4.2 언어 편향 우려와 해소 근거
WavLM은 사실상 영어 중심 코퍼스로 학습되었기 때문에, Mimi의 시멘틱 토큰이 영어 음소 대비(phonetic contrast) 구분에 최적화되어 있을 가능성이 이론적으로 존재한다. 특히 일본어/한국어는 영어와 음운론적으로 거리가 있다(모라 리듬, 어휘적 피치 악센트, 장단음 등).

**다만 J-Moshi의 사전 실험이 이 우려를 상당 부분 해소한다**: J-Moshi 팀은 Mimi가 일본어 음성을 충분히 잘 인코딩/재합성할 수 있음을 사전 확인 후 Mimi를 동결(freeze)하고 언어모델(백본)만 파인튜닝했다. 이는 코덱 자체의 언어 범용성이 최소한 재구성(reconstruction) 수준에서는 어느 정도 검증됐다는 뜻이다.

**단, 이는 "코덱이 언어를 표현할 수 있는가(encoding)"의 문제이지 "그 표현 위에서 LM이 처음부터 없던 언어를 새로 생성해낼 수 있는가(generation)"와는 별개의 질문**이다. Haan의 컨트리뷰션(§ PROJECT.md 3.3)은 후자에 집중되어야 한다.

### 4.3 권장 사전 진단 실험
본 학습 진입 전, LM을 배제한 **Mimi 단독 encode→decode 왕복 재구성 테스트**를 한국어 음성으로 수행할 것을 권장 (J-Moshi 방법론의 한국어 재현):
- 한국어 음성을 Mimi로 인코딩 후 디코딩 → ASR 재전사로 WER 측정 (음소 변별력 유지 확인)
- 피치 악센트/장단음 등 한국어 특유 음운 대비가 재구성 후에도 구분되는지 최소대립쌍 테스트

이를 통해 코덱 자체의 상한선(ceiling)을 먼저 확인하면, 이후 본 실험에서 관찰되는 성패를 "코덱 한계 대비 LM 전이 성능"으로 정확히 귀속시킬 수 있다.

### 4.4 평가 설계상 주의점
Acoustic RVQ(7레벨)는 시멘틱 레벨과 병렬 분리되어 있어, 시멘틱 콘텐츠가 틀려도 acoustic 레벨은 자연스럽게 들리는 음성 텍스처를 만들어낼 수 있다. 즉 **자연스럽게 들리지만 실제로는 의미가 틀리거나 알아들을 수 없는 발화("word salad")가 나올 위험**이 있다. 평가 시 MOS/naturalness만이 아니라 **반드시 ASR 기반 WER/CER로 실제 발화 내용의 정확성을 별도 측정**해야 한다.

## 5. RQ-Transformer / Depth Transformer 구조와 KD / voice-prompt 설계

### 5.0 Moshi 원 설계 (Moshi 논문 §3.4.1, Fig 3, 식 1~2 확인)
- **Temporal Transformer**(대형, Helium/Qwen3 백본): 과거 프레임 $(V_0,\dots,V_{s-1})$을 컨텍스트 벡터 $z_s$로 압축(식 1). 프레임당 하나의 벡터를 만들 뿐, 오디오 토큰을 직접 예측하지 않는다.
- **Depth Transformer**(소형): $l_{s,k}=\text{Tr}_{Depth}(z_s, V_{s,1},\dots,V_{s,k-1})$ (식 2)로 프레임 내 서브토큰을 순차 예측. $k=1$(텍스트, Inner Monologue)은 $z_s$에서 직접, $k\ge2$(오디오 코드북)은 Depth가 담당.
- 따라서 **semantic level-0을 포함한 모든 오디오 코드북은 Depth Transformer가 예측**한다(레포 lm.py 판독: depth 헤드 = self 스트림 `dep_q=8`개). "semantic은 backbone이 예측한다"는 이해는 틀림 — semantic/acoustic 구분은 축 B(모델)가 아니라 축 A(Mimi 코덱 §4.1)의 성질이다.
- 원 Moshi는 semantic-acoustic 간 안정적 생성을 위해 코드북 간 인위적 delay(acoustic delay)를 사용. Depth 출력 헤드는 self 스트림에만 존재 — user 코드북은 예측 대상이 아니라 입력 조건.

### 5.1 KD codebook routing — semantic 중심 (핵심 결정)
영어 멀티턴 KD의 목적은 **turn-taking의 동시성 행동(맞장구·barge-in·오버랩) 전이**이지 콘텐츠·음향 복제가 아니다. teacher(Moshi)와 student(Qwen3)가 **동결된 동일 Mimi 코덱**을 공유하므로:

- **KD는 Mimi semantic 토큰(level-0)의 logit에 건다.** 출력공간(코드 2048개)이 teacher·student 간 동일해 projection 없이 KL이 정의된다. 침묵/발화 여부도 semantic 스트림에 인코딩되므로 turn-taking 타이밍이 이 채널에 실린다.
- **acoustic 코드북(1~7)은 기본 KD에서 제외**한다 — teacher의 고정 화자(Moshi-ko/ka) 음색을 실어나르는 캐리어이기 때문. (단계적 graft는 §5.3)
- **hidden-state KD는 폐기**: Helium↔Qwen3는 이종 backbone이라 hidden space가 정렬돼 있지 않다. 공유 코덱이 출력공간을 이미 동일하게 만들어주므로 hidden을 억지로 맞출 이유가 없다. (`RISKS_AND_DIAGNOSTICS.md` §7.1)
- **텍스트 채널 logit KD도 폐기**: Helium≠Qwen3 토크나이저. 콘텐츠는 Qwen3 자체 능력 + inner monologue로 확보. (`RISKS_AND_DIAGNOSTICS.md` §7.2)

### 5.2 화자/음색 — in-context voice-prompt (PersonaPlex 기법의 독립 구현)
음색은 KD가 아니라 **참조 음성을 agent audio 채널에 프리픽스로 까는 in-context voice-prompt**로 공급한다. PersonaPlex의 Hybrid System Prompt(voice-prompt 구간 + role text 구간)가 정확히 이 방식이며 — 아키텍처를 건드리지 않고, learnable VQ·별도 speaker embedding·cross-attention 주입이 없다.

- **PersonaPlex의 가중치·데이터셋은 라이선스(NVIDIA Open Model License) 문제로 미사용**하되, voice-prompt prefix는 일반적 in-context 조건화(Moshi도 원래 가능)이므로 **독립 구현**한다.
- 실패했던 대안(참조 + learnable VQ 삽입 → 음색은 이동하나 content 붕괴)은 폐기. 원인은 미확정이나, PersonaPlex가 learnable VQ 없이 프리픽스만으로 달성함이 확인됨.
- 학습 트릭(PersonaPlex 준용): system prompt 구간 loss 마스킹, non-semantic 오디오 토큰 loss 다운웨이팅.

**텍스트 채널 IT 포맷 보존**: 텍스트 채널(inner monologue)은 Qwen3 IT 순정 포맷(`<think>`, `<tool_call>`)을 유지하고, 태그 구간은 **비발성 후처리**로 처리해 agent 능력을 보존한다. 타이밍 제어 토큰을 이 스트림에 섞지 않는다(고가치 IT 스트림과의 에러 커플링 방지). 부수 효과로 `<think>`/`<tool_call>` 침묵 구간이 **content-conditioned 응답 pause**를 제공한다(단, `<think>`가 과도히 길면 pause에 cap을 걸거나 filler를 오버레이).

### 5.3 acoustic prosody — 단계적 graft (Phase 3.5)
barge-in 시 pitch 상승 같은 paralinguistic 다이내믹스는 acoustic에 있으므로 semantic-only로는 전이되지 않는다. 이를 위해 **클로닝을 먼저 확립(semantic-only KD + voice-cloning CE)한 뒤, turn-event 국소 acoustic KD를 짧게 graft**한다. 순서를 이렇게 두는 이유는 성숙한 클로닝 prior가 timbre 유출에 저항하는 방파제가 되기 때문.

graft 규율:
1. **페어드 voice-prompt**: KD 타깃마다 그에 맞는 목소리 prompt를 페어링 → KD가 prompt 음색과 싸우지 않고 prosody 궤적만 지도.
2. **ko-ka interpolation 다화자화**: 검증된 ko+ka 가중합으로 중간 목소리 연속체를 만들어 KD 타깃 음색을 다양화 → "prosody는 음색 불변" 신호 강화, ko/ka에 묶는 지름길 억제. (한계: ko↔ka 1축 연속체이지 전체 voice-space 커버는 아님)
3. **저weight/저LR + add**: semantic KD·CE 항을 끄지 않고 acoustic KD만 얹음.
4. **turn-event 국소 적용**: barge-in/overlap 프레임에만 acoustic KD.
5. **guard로 early-stop**: turn-event 구간의 화자 유사도를 모니터링, prosody 이득 plateau거나 클로닝 드리프트가 임계 초과 시 정지. ("few step"을 고정 숫자가 아니라 드리프트 지표로 멈춤)

### 5.4 User-stream 예측 — 공유 Depth 병렬 분리 (PersonaPlex 순차 대비)

**배경 (세 설계 비교)**:
- **Moshi**: Depth는 self 코드북(dep_q=8)만 예측. user 스트림은 입력 조건일 뿐 예측 대상 아님.
- **PersonaPlex**: Depth의 step을 16으로 늘려 self 8 + user 8을 **한 자기회귀 롤아웃으로 순차 예측**(프레임 내 user가 self 뒤에 조건부). 사람도 상대 발화를 예측·모델링하듯 user 예측 능력을 갖추는 것이 목적.
- **Haan (채택)**: user 예측 능력의 **목적은 PersonaPlex와 동일**(유지되는 능력)하되, 순차의 부자연스러움·속도 저하를 피한다.

**채택 설계**: Depth Transformer는 **1개로 통합**(가중치 공유 = self 학습 처리 능력이 user에도 적용되는 정보 효율, §6.1 논리와 동일). Temporal의 context $z_s$를 **MLP로 2채널($z_{self}, z_{user}$)로 분리**하고, **role embedding을 Depth 입력에 명시 주입**한 뒤 **batch=2로 묶어 병렬 입력**. 각 스트림 8-step으로 코드북 예측.
- **속도**: 순차 16-step 대신 **8-step × batch 2**. user 예측을 추론에서도 유지한 채로 depth 자기회귀 길이가 절반 → 약 2배 빠름. (속도 이득은 user를 버려서가 아니라 시퀀스를 늘리는 대신 배치로 병렬화한 데서 나옴)
- 구현: "UpConv"가 아니라 채널 2배 라우팅 MLP(또는 2-헤드). Temporal은 불변, Depth 입력 경로만 변경.

**프레임 내 self↔user 상관 트레이드오프 (수용, 안전판 보유)**:
- 병렬은 $p(\text{self}_t\mid z_s)\cdot p(\text{user}_t\mid z_s)$로 프레임 내 순간 상관을 조건부 독립으로 둔다. 이는 오히려 **Moshi 본래 factorization(모든 프레임 토큰을 $z_s$에서 생성)에 충실**하며, PersonaPlex의 프레임 내 self→user 결합은 효용 미입증의 추가 가정이다.
- **스트리밍 인과성**: 추론 시 self_t는 동시 프레임 user_t를 아직 못 들은 상태에서 생성되므로 self_t가 simultaneous user_t에 의존하는 것은 인과적으로 불가능. 따라서 프레임 내 결합이 영향을 주는 곳은 **self 생성 품질(진짜 출력)이 아니라 user 예측 sharpness(aux 능력)뿐**.
- **결정**: "어쩔 수 없다"가 아니라 **"측정해 무시할 만함을 확인"**으로 굳힌다(소규모 순차 vs 병렬 bake-off, `TRAINING_CURRICULUM.md` §2 Phase 5). 만약 오버랩 자연스러움까지 유의미하게 떨어지면 **두 배치 스트림 간 경량 cross-attention을 Depth 내부에 추가**해 결합을 국소 복원(여전히 8-step, 순차 16-step 비용 없음). 즉 지금 병렬로 커밋해도 리스크가 갇히지 않는다.

## 6. Role 구분 메커니즘: 채택안과 후속 연구 로드맵 (구 `FUTURE.md` 통합)

본 절은 §2~3에서 다룬 self/user 오디오 임베딩 분리 재설계를, **본 트랙 채택안(Role Token)**과 그 **후속 연구 두 계열**로 정리한다. 두 후속 논문은 서로를 전제하지 않고 각자 완결된 기여를 갖되, 같은 아이디어 계열(role 구분을 임베딩 분리가 아닌 별도 신호로 옮긴다)에서 파생된다.

### 6.1 문제의 정확한 위치
Moshi(및 이를 초기화로 쓰는 PersonaPlex)의 한 타임스텝 Temporal Transformer 입력:

```
input_t = TextEmb(text_t) + Σ_k SelfAudioEmb_k(A_self[t,k]) + Σ_k UserAudioEmb_k(A_user[t,k])
```

`SelfAudioEmb_k`와 `UserAudioEmb_k`는 codebook k마다 role별로 완전 분리된 학습 파라미터다. **문제**: 한국어 학습이 self에 집중되면 `SelfAudioEmb`는 한국어에 맞게 조정되지만 `UserAudioEmb`는 노출이 적어 초기(영어 편향) 상태에 남아, self가 배운 한국어 인식 능력이 user 인식 경로로 전이되지 않는다.

- **레이어 구분(중요)**: 언어를 "배우는" 주체는 embedding table이 아니라 RQ-Transformer의 attention/FFN이다. 임베딩 공유가 기여하는 것은 "Transformer가 self 학습으로 얻은 처리 능력을 user 입력 벡터에도 적용 가능하게 하는 통로"이지, 임베딩 자체가 언어를 학습하는 게 아니다.
- **Temporal vs Depth 구분(중요)**: user 오디오 이해는 전적으로 Temporal Transformer의 역할이다(§5.0). Depth는 self 생성에만 관여하며 user 코드를 직접 입력받지 않는다. 따라서 **semantic embedding 공유**는 Temporal의 user 인식 능력에 직접·명확히 기여하지만, **acoustic embedding 공유**의 "생성 자연스러움이 user 쪽에도 전이된다"는 기대는 성립하지 않는다(Depth가 user 코드를 안 봄). 다만 Temporal이 user의 paralinguistic 뉘앙스를 이해하는 데는 제한적으로 기여할 수 있어, 최종적으로는 semantic/acoustic 모두 공유하되 acoustic 공유의 제한적 효과는 아래 논문 1 ablation에서 정직하게 다룬다.

### 6.2 논문 1: Role Token — 한국어 전이를 위한 임베딩 통합 (본 트랙 채택안)
- **컨트리뷰션**: Moshi의 self/user 오디오 임베딩 분리가 저자원 언어 전이를 저해함을 규명하고, 임베딩을 완전 공유하되 role 구분을 이진 role token(학습형 additive 벡터, self/user 각 1개)으로 대체. 단일 화자 발화(비멀티턴) 데이터만으로 이해(user 인식)와 생성(self 발화) 양쪽에 한국어 능력이 전이됨을 실증.
- **설계**: 오디오 임베딩 테이블 self/user 공유(semantic+acoustic 전체), role 구분은 `RoleEmb[role_id]`(0=self,1=user) additive, 시간축 RoPE는 그대로 유지. (§3.3과 동일)
- **검토했으나 기각/보류**: role 구분을 시간축 RoPE 자체를 role별로 다르게 쓰는 방식(동기화 붕괴 위험으로 기각); Dual PE(학습형) vs 3축 M-RoPE(회전축 고정)는 본 트랙 목적엔 대체로 동등하나 구현 단순성·백본 RoPE 불건드림 이유로 Role Token 채택.
- **선행 리스크**: Mimi 코드북이 한국어 음향을 담을 용량이 있는지는 임베딩 공유와 무관한 별도 문제 → §4.3 round-trip sanity check.
- **Ablation**: (1) 원 Moshi 완전 분리(baseline) (2) 공유+Role Token(제안) (3) 공유+Dual PE 대안 (4) semantic만 공유 vs semantic+acoustic 모두 공유.

### 6.3 논문 2: Role-axis RoPE — 다중 화자로의 구조적 외삽 (후속)
- **컨트리뷰션**: 논문 1의 "role을 별도 신호로 분리" 아이디어를, 학습형 additive가 아닌 **결정론적 회전 축**으로 구현하면 이진 self/user를 넘어 임의 개수 동시 화자를 자연스럽게 표현할 수 있음을 제안·실증. Moshi의 고정 2-스트림 제약을 해소하는 확장. 한국어 전이와 독립적으로 성립.
- **설계**: Temporal RoPE를 두 축으로 분해 — **Temporal 축**(기존과 동일, 모든 화자 슬롯 공통, 동일 프레임 인덱스로 동기화 보존) + **Role 축**(슬롯 인덱스별 고정 회전 오프셋: self=0, user1=Δ, user2=2Δ, …). 학습 시 self는 항상 슬롯 0 고정, 그 외는 "비-self"로 취급. Qwen2.5-Omni TMRoPE가 선례.
- **핵심 주장(왜 1:1 데이터만으로 다중 화자 외삽되나)**: 슬롯 구분이 학습 표상이 아니라 기하학적 고정 회전이므로 모델이 배울 건 "오프셋이 0인가 아닌가"라는 이진 판단뿐. RoPE attention score가 상대 회전각에 매끄럽게 변하므로 "0 근방=self, 멀수록 non-self"가 특정 값에 과적합되지 않고 일반화 — RoPE 길이 외삽과 동일 성질을 화자 슬롯 축에 적용. 학습형 role token은 본 슬롯 번호에 최적화돼 안 본 슬롯 일반화가 약하지만, 회전 기반은 오프셋 배수라 구조적으로 외삽에 유리.
- **엔지니어링 과제(설계 리스크 아님)**: 실시간 신규 화자 슬롯 동적 배정(diarization 연동), 슬롯 최대 개수 N과 오프셋 Δ의 회전 주기 충돌 방지, 텍스트(1D RoPE)와 오디오(role 축 추가) 간 회전 체계 일관성 검증(백본 사후 개조).

### 6.4 두 논문의 관계
| | 논문 1 (Role Token) | 논문 2 (Role-axis RoPE) |
|---|---|---|
| 핵심 문제 | 저자원 언어(한국어) 전이 | 화자 수(dyad→multi-party) 확장 |
| Role 구분 신호 | 학습형 additive 벡터(이진) | 결정론적 회전 오프셋(N-way) |
| 출발점 | PersonaPlex/Haan 초기화, 한국어 데이터 | 원 Moshi에서 독립 시작 가능 |
| 컨트리뷰션 성격 | Data efficiency | Architectural generality |
| 상호 의존성 | 없음 — 각자 완결, 같은 아이디어 계열의 후속 |
