# Haan 데모 — AI 카드상담 목업

카드사 고객이 **전화 통화**로 AI와 상담하고, 그 통화가 **상담사 콘솔**에 실시간 전사되며,
**AI의 판단·도구 실행 흐름**이 라이브로 스트리밍되는 데모입니다.
(요구사항: `../ClaudeCode_인수인계_요구사항.md`)

> **지금 단계 = 껍데기(목업) 우선.** 음성엔진(STT/TTS)·자체 LLM·카드사 시스템은 모두
> **어댑터 뒤의 Mock**으로 대체되어 있습니다. 실제 부품이 오면 어댑터만 교체하면 됩니다.

## 실행

두 프로세스로 구성된다: **백엔드(server/)** 와 **상담사 UI(ai_ui/advisor-app/)**.
둘은 `docs/contexts/SERVICE_ARCHITECTURE.md §6` 의 의미론 이벤트 계약으로 통신한다.

### 1) 백엔드 — 의존성 없음 (Node 18+ 표준 라이브러리만)

```bash
cd server
node server.js          # 포트 5173
```

- **고객 앱** → http://localhost:5173/customer
- (레거시 콘솔 http://localhost:5173/agent 은 **옛 계약 기반 바닐라 콘솔**로 참고용. 정식 상담사 화면은 advisor-app.)

### 2) 상담사 UI — React (정식 상담사 화면)

```bash
cd ai_ui/advisor-app
npm install
# 서버에 붙이기(실시간). 비우면 내장 MockEventSource 로 스크립트 재생만 됨.
echo "VITE_WS_URL=ws://localhost:5173" > .env.local
npm run dev             # 포트 5174
```

> **Windows PowerShell 주의**: 위 `echo ... > .env.local`을 PowerShell에서 그대로 실행하면
> 기본 인코딩이 UTF-16LE라 Vite가 `VITE_WS_URL`을 못 읽고 조용히 Mock으로 폴백합니다
> (에러 없이 그냥 안 붙음 — 원인 파악이 어려움). PowerShell에서는 반드시 UTF-8로 지정하세요:
> ```powershell
> "VITE_WS_URL=ws://localhost:5173" | Out-File -Encoding utf8 .env.local
> ```
> `.env.local`을 만든 뒤 값이 안 먹는 것 같으면 `file .env.local`(Git Bash) 등으로 인코딩부터 확인.

고객 앱(5173/customer)과 상담사 UI(5174)를 나란히 열어보자. 고객 앱은 접속하는 순간
자동으로 `call_start`를 보내 통화가 시작되고(별도의 "통화 시작" 버튼은 없음),
전사·판단·도구 실행이 상담사 UI 3패널에 실시간 스트리밍된다.

### 계약 테스트

```bash
cd server && npm test   # node --test (계약 준수·seq·잠금 권위 검증)
```

## 데모 흐름 (요구사항 §2)

1. 통화 시작 → MockSTT가 시나리오 대본(고객 발화)을 실시간 전사로 흘려보냄
2. 왼쪽 패널에 고객 발화가 한 줄씩 전사됨
3. MockLLM이 발화를 분석 → **본인확인**을 먼저 수행 (오른쪽 패널에 판단 노드 스트리밍)
4. `verify_identity` 성공 → **개인정보 잠금 해제** (고객 정보 카드가 블러 해제되며 열림)
5. `suspend_card`(카드 정지), `report_lost`(분실 신고) 도구가 대기→실행중→완료로 전이
6. 상담사 콘솔 하단 입력창으로 **수동 전사 주입**도 가능 (실제 STT 대체 테스트)

## 구조 (어댑터 패턴)

```
haan-demo/
├─ server.js                # HTTP 정적 서빙 + WebSocket 허브 + 세션 오케스트레이션
├─ lib/
│  ├─ ws.js                 # 무의존성 WebSocket 서버 (RFC 6455 최소 구현)
│  ├─ seed.js               # 가짜 고객/카드 데이터 (김민서 / 4821 / 카드 2매)
│  ├─ tools.js              # Mock 도구 3종 (verify_identity / suspend_card / report_lost)
│  ├─ scenario.js           # 데모 대본 (고객 발화 시퀀스)
│  └─ adapters/
│     ├─ stt.js             # STT 어댑터 인터페이스 + MockSTT (대본 재생)  ← 실제 STT 자리
│     └─ llm.js             # LLM 어댑터 인터페이스 + MockLLM (규칙 기반)  ← 실제 Haan 모델 자리
└─ public/
   ├─ agent.html            # 상담사 콘솔 UI (전사 + AI 판단·실행 흐름)
   └─ customer-bridge.js    # 고객 앱(user.html)에 주입되는 WebSocket 브리지
```

고객 앱 `user.html` 원본은 수정하지 않습니다. 서버가 서빙 시점에 브리지 스크립트만
`</body>` 앞에 주입합니다.

## 교체 지점 (실제 부품 연결 시)

| 부품 | 지금 (Mock) | 교체 방법 |
|---|---|---|
| STT | `lib/adapters/stt.js`의 `MockSTT` | `STTAdapter` 인터페이스 구현체로 교체, 오디오→`feed()` |
| LLM/판단 | `lib/adapters/llm.js`의 `MockLLM` | `plan({text, session})`가 동일한 action 스트림을 산출하도록 구현 |
| 카드사 API | `lib/tools.js`의 `ToolRunner` | 각 도구 메서드를 실 API 호출로 교체 |

## WebSocket 이벤트 계약

**클라이언트 → 서버**
- `{type:"hello", role:"agent"|"customer"}` — 접속 신고 (서버가 `snapshot` 회신)
- `{type:"call_start"}` / `{type:"call_end"}` / `{type:"reset"}`
- `{type:"transcript_in", text}` — 수동/실 STT 전사 주입

**서버 → 클라이언트 (broadcast)**
- `{type:"call_state", state:"active"|"ended"|"idle"}`
- `{type:"transcript", speaker:"customer"|"ai", text}`
- `{type:"ai_step", phase, label, detail, status:"active"|"done"}`
- `{type:"tool_call", tool, status:"running"|"done"|"error", request, response}`
- `{type:"identity", unlocked:true, profile}` — 본인확인 성공 = 개인정보 열림
- `{type:"snapshot", active, unlocked, profile, history}` — 재접속 동기화

## ⚠️ 보안 주의 (데모 컨셉)

데모 메시지는 "AI가 본인확인만 해주면 카드사 개인정보에 바로 접근되는 편리함"입니다.
그래서 별도 로그인 없이 **본인확인 성공 = 개인정보 잠금 해제**로 표현합니다.
**실서비스 전환 시**에는 실제 인증·암호화·개인정보 이용 동의가 반드시 필요합니다.
(코드 곳곳의 `⚠️ 실서비스 전환 시` 주석 참조)
