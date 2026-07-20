# 상담사 AI 작업공간 (advisor-app)

`aiui.html`(상담사가 태블릿 가로 모드에서 보는 실시간 작업 화면)을
**React + Vite + Tailwind**로 컴포넌트화한 데모 앱입니다.

> 디자인 원칙(상담사UI_프롬프트.md §5.5): 구조·색·모션은 손대지 않고 **데이터만 실제로 교체**.
> `src/styles/advisor.css`는 원본 `aiui.html`의 `<style>`을 **그대로 이식**해 픽셀 동일성을 보장합니다.

## 실행

```bash
npm install
npm run dev      # http://localhost:5174
npm run build    # 프로덕션 번들 (dist/)
npm run preview  # 빌드 결과 미리보기
```

## 이 화면이 하는 일

3패널 + 하단 자막:

1. **현재 통화** — STT 전사가 실시간으로 줄 추가 (`CallPanel`).
2. **상담 작업대** — 고객 정보 · 본인확인 체크리스트(진행률) · 카드 후보 (`WorkPanel`).
3. **AI 판단 및 실행 흐름** — 발화 감지 → 감정 신호(불안 87% 카운트업) → 대응 판단(비채택
   후보 비교) → 추론 문장 → `verify_identity` 실행을 **라이브로 재생** (`AiFlowPanel`).

데모의 핵심 메시지: **본인확인(verify_identity) 완료 = 카드사 개인정보/실행 잠금 해제**.

## 실시간 연동 (Mock ↔ WebSocket 교체)

목업 타이머 대신 **이벤트소스 어댑터**로 구동합니다.

```
src/realtime/
  events.js                 # WebSocket 이벤트 계약 + 데모 재생 스크립트(원본 aiThinkingEvents 승격)
  MockEventSource.js        # 내장 Mock — 스크립트를 back-pressure 로 재생(데모 기본값)
  WebSocketEventSource.js   # 실제 백엔드(ws) 어댑터 — 동일 인터페이스
  createEventSource.js      # VITE_WS_URL 있으면 WebSocket, 없으면 Mock
src/hooks/useAdvisorFlow.js # 이벤트 → 화면 상태(타이핑/카운트업/도구 상태) 구동
```

실제 Mock 백엔드에 붙이려면 `.env`에 `VITE_WS_URL`을 지정하세요(`.env.example` 참고).
서버는 `events.js`의 계약(`transcript` / `ai_step` / `tool_call` / `verify_done` /
`caption` / `flow_end`)을 JSON으로 push 하면 됩니다.

## 이벤트 계약 요약

| 이벤트 | 페이로드 | 효과 |
|---|---|---|
| `transcript` | `{ speaker, text, partial }` | 현재 통화에 줄 추가 |
| `ai_step` | `{ step, reveal, typing[], count, toast }` | 오른쪽 흐름 노드 재생 |
| `tool_call` | `{ tool, state: ready\|running\|done }` | 도구 상태 전환 |
| `verify_done` | `{ check, unlocked }` | 본인확인 체크 자동 완료 → 잠금 해제 |
| `caption` | `{ text }` | 하단 자막 |
| `flow_end` | `{ toast }` | 흐름 종료 → 다시 재생 노출 |

## 태블릿 패키징

원본 전략과 동일하게 **Capacitor**로 이 웹 빌드를 안드로이드 태블릿 앱(가로 모드,
풀스크린)으로 감싸는 방식을 권장합니다. device-frame 목업은 제거되어 웹뷰 풀스크린으로
바로 사용 가능합니다.

## 데모 한계 (실서비스 전환 시)

- 음성엔진(STT/TTS)·자체 LLM 없음 → 어댑터 뒤 Mock으로 대체.
- 본인확인 성공을 그대로 잠금 해제로 연출 → 실제 인증·암호화·개인정보 동의 필요.
