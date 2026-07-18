# Haan
Haan: Hearing Always, Acting Naturally

**한국어 멀티턴 음성 대화 데이터셋 없이** 한국어 Full-duplex 대화 능력을 창발시키는 것을 목표로 하는 연구. Kyutai Moshi 아키텍처를 기반으로 백본을 Qwen3-8B로 교체해 "언어를 가르치는" 비용을 생략하고, 오디오 모달리티 접합과 turn-taking 구조 전이에 집중한다.

## 핵심 설계 (요약)
- **영어 멀티턴 KD는 Mimi semantic 토큰(level-0) logit에 한정** — turn-taking의 동시성 행동(맞장구·barge-in·오버랩) 전이가 목적. 공유 동결 Mimi 코덱 덕에 teacher(Moshi)–student(Qwen3) 출력공간이 동일.
- **음색은 KD가 아니라 in-context voice-prompt로 공급** (PersonaPlex Hybrid System Prompt 기법의 독립 구현; 가중치·데이터 미사용).
- **텍스트 채널은 Qwen3 IT 포맷(`<think>`/`<tool_call>`) 보존**, 태그 비발성 후처리로 agent 능력 유지.
- **acoustic prosody는 단계적 graft** (semantic-only 확립 후 turn-event 국소 KD, guard로 조기 종료).

## 문서
- [`PROJECT.md`](PROJECT.md) — 개요·컨트리뷰션·관련연구·컴퓨팅 예산
- [`PERSONA.md`](PERSONA.md) — 캐릭터/페르소나 설정
- [`docs/contexts/ARCHITECTURE.md`](docs/contexts/ARCHITECTURE.md) — 백본 교체, 임베딩·Role Token, KD codebook routing, voice-prompt, Role 후속 로드맵
- [`docs/contexts/TRAINING_CURRICULUM.md`](docs/contexts/TRAINING_CURRICULUM.md) — Phase 0~5 커리큘럼, Joint loss, 커널 최적화 노트
- [`docs/contexts/DATA_STRATEGY.md`](docs/contexts/DATA_STRATEGY.md) — 싱글턴 데이터·합성 파이프라인·voice-prompt
- [`docs/contexts/RISKS_AND_DIAGNOSTICS.md`](docs/contexts/RISKS_AND_DIAGNOSTICS.md) — 실패 메커니즘·진단
