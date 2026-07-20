// ─────────────────────────────────────────────────────────────────────────────
// 실시간 이벤트 계약 (WebSocket) — 단일 출처(SSOT)
//
// docs/contexts/SERVICE_ARCHITECTURE.md §6 의 "관측 경로" 계약을 코드로 고정한 것.
// 서버(실서비스)와 MockEventSource(데모)가 **똑같은 이 계약**을 발행하고,
// useAdvisorFlow 가 이를 받아 화면 표현으로 변환한다.
//
// ── 설계 원칙: 계약은 "의미론"만 싣는다 ─────────────────────────────────────────
// 이벤트에는 emotionKicker 같은 DOM 노드 id 나 타이핑 순서(표현 정보)를 넣지 않는다.
// 모델(Haan 이든 캐스케이드든)은 phase/label/confidence 같은 "의미"만 낼 수 있을 뿐
// 화면 노드 id 를 알 수 없기 때문이다. 표현(어느 노드에 타이핑할지, 87% 카운트업,
// 분기선)은 useAdvisorFlow 의 PHASE_PRESENTATION 매핑이 전담한다.
//   → 서버가 confidence 를 주면 그 값으로 카운트업, 안 주면 프리셋 폴백.
//     (감정 수치의 출처는 SERVICE_ARCHITECTURE.md §6 미해결 항목 — 계약이 열어둔다.)
//
// ── 공통 봉투(envelope) ─────────────────────────────────────────────────────────
// 모든 이벤트는 런타임(서버/MockEventSource)이 아래 봉투를 씌워 전송한다.
//   { session_id, seq, ts, type, ...payload }
//   - seq: 오케스트레이터가 부여하는 단조 증가 번호. UI 는 seq 로 순서를 보장하고
//          갭이 감지되면 리플레이를 요청한다(재접속 복구).
//   - 아래 스크립트/스키마에는 payload 만 적는다. 봉투는 전송 계층이 주입.
//
// ── payload 스키마 ──────────────────────────────────────────────────────────────
//   { type: "flow_reset" }
//   { type: "transcript",  speaker: "고객"|"AI"|"상담", utterance_id, text,
//                          partial?, source?: "asr"|"inner_monologue" }
//   { type: "ai_step",     phase, label?, detail?, status?: "active"|"done",
//                          confidence?: 0..100, candidates?: string[], chosen?: string,
//                          caption?, toast? }
//   { type: "tool_call",   call_id, tool, state: "ready"|"running"|"done"|"error",
//                          request?, result?, error?, toast? }
//   { type: "verify_state",unlocked: bool, customer_id?, check?, profile?, caption? }
//   { type: "caption",     text }
//   { type: "call_state",  state: "active"|"ended", duration_ms? }
//   { type: "snapshot",    active, unlocked, profile?, history?, last_seq }
//   { type: "flow_end",    toast? }
//
// phase(의미) → workflow step(표현) 매핑은 useAdvisorFlow.PHASE_PRESENTATION 참조:
//   detect(발화 감지) · emotion/respond/reason(단서 추출) · verify(본인 확인) ·
//   result(결과 수신) · decide(응답 결정)
// 각 스크립트 항목의 `wait` 는 emit "전" 지연(ms) — 원본 aiui.html 재생 타이밍 보존.
// ─────────────────────────────────────────────────────────────────────────────

export const EVENT_TYPES = Object.freeze({
  FLOW_RESET: "flow_reset",
  TRANSCRIPT: "transcript",
  AI_STEP: "ai_step",
  TOOL_CALL: "tool_call",
  VERIFY_STATE: "verify_state",
  CAPTION: "caption",
  CALL_STATE: "call_state",
  SNAPSHOT: "snapshot",
  FLOW_END: "flow_end",
});

// AI 판단 흐름의 의미론적 단계. 서버/Mock 이 ai_step.phase 로 발행한다.
export const PHASES = Object.freeze({
  DETECT: "detect", // 발화 감지
  EMOTION: "emotion", // 감정/불안 신호 (confidence)
  RESPOND: "respond", // 대응 판단 (candidates/chosen)
  REASON: "reason", // 추론 문장
  VERIFY: "verify", // 본인확인 실행
  RESULT: "result", // 도구 결과 수신
  DECIDE: "decide", // 응답 결정
});

export const RESET_CAPTION = "고객 발화를 들으며 AI가 필요한 판단을 순서대로 구성합니다.";

// 원본 aiThinkingEvents 를 "의미론 계약"으로 승격한 데모 재생 스크립트.
// 표현 필드(reveal/typing/count 등)는 전부 제거됐고, 화면 표현은 useAdvisorFlow 가
// phase 로부터 복원한다. 실제 서버가 붙으면 WebSocketEventSource 가 동일 계약을 흘려보낸다.
export const advisorDemoScript = [
  { wait: 0, type: "flow_reset" },

  // 발화 감지 — 고객 발화가 현재 통화(transcript)와 AI 감지 노드에 함께 뜬다.
  {
    wait: 0,
    type: "transcript",
    speaker: "고객",
    utterance_id: "demo-u1",
    text: "카드를 잃어버려서 빨리 정지해야 한다고요. 너무 무서워요…",
    source: "asr",
  },
  {
    wait: 0,
    type: "ai_step",
    phase: "detect",
    label: "카드를 잃어버려서 빨리 정지해야 한다고요. 너무 무서워요…",
    toast: "발화 감지",
  },

  // 단서 추출 — 감정 신호(불안) + confidence 87 카운트업.
  {
    wait: 240,
    type: "ai_step",
    phase: "emotion",
    label: "불안 높음",
    confidence: 87,
    toast: "단서 추출",
  },

  // 대응 판단 — 비채택 후보(절차 안내/추가 질문) 비교 후 공감 대응 선택.
  {
    wait: 260,
    type: "ai_step",
    phase: "respond",
    label: "공감 대응",
    candidates: ["절차 안내", "추가 질문"],
    chosen: "공감 대응",
    toast: "응답 후보 비교",
  },

  // 추론 문장 생성.
  {
    wait: 260,
    type: "ai_step",
    phase: "reason",
    detail:
      "AI가 불안 신호를 기준으로 다른 응답 후보를 비교한 뒤, 지금은 공감 대응을 먼저 선택합니다.",
    toast: "추론 생성",
  },

  // 본인 확인 — verify_identity 실행 준비 → 실행 중.
  { wait: 280, type: "ai_step", phase: "verify", status: "active", toast: "실행 준비" },
  { wait: 0, type: "tool_call", call_id: "demo-c1", tool: "verify_identity", state: "ready" },
  {
    wait: 700,
    type: "tool_call",
    call_id: "demo-c1",
    tool: "verify_identity",
    state: "running",
    toast: "실행 중",
  },

  // 결과 수신 — 도구 완료 → 등록 휴대폰 끝자리 확인 자동 체크 → 개인정보 잠금 해제.
  { wait: 900, type: "ai_step", phase: "result", toast: "완료" },
  { wait: 0, type: "tool_call", call_id: "demo-c1", tool: "verify_identity", state: "done" },
  {
    wait: 0,
    type: "verify_state",
    unlocked: true,
    check: "phone",
    caption: "AI가 등록 휴대폰 끝자리 확인을 완료하고 결과 수신 단계로 넘어갑니다.",
  },

  // 응답 결정 — 흐름 종료 → 다시 재생 노출.
  { wait: 450, type: "ai_step", phase: "decide" },
  { wait: 0, type: "flow_end", toast: "응답 결정" },
];
