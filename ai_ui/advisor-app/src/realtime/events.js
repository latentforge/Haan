// ─────────────────────────────────────────────────────────────────────────────
// 실시간 이벤트 계약 (WebSocket)
// 상담사UI_프롬프트.md §3 / 인수인계 문서 §5 의 스키마를 그대로 승격한 것.
// 서버(또는 MockEventSource) → 화면(useAdvisorFlow) 으로 흘려보내는 이벤트 형태:
//
//   { type: "flow_reset" }
//   { type: "transcript",  speaker: "고객|AI|상담", text, partial }
//   { type: "ai_step",     step?, reveal?, revealGroup?, branches?,
//                          typing?: [{ id, text }], count?: { target, value },
//                          caption?, toast? }
//   { type: "tool_call",   tool: "verify_identity", state: "ready|running|done" }
//   { type: "verify_done", check: "phone", unlocked: true, caption? }
//   { type: "caption",     text }
//   { type: "flow_end",    toast? }
//
// step 0 발화 감지 → 1 단서 추출 → 2 본인 확인 → 3 결과 수신 → 4 응답 결정
// (원본 workflowSteps 와 1:1 매핑)
// 각 스크립트 항목의 `wait` 는 이 이벤트를 emit 하기 "전"의 지연(ms) — 원본 setTimeout 타이밍 보존.
// ─────────────────────────────────────────────────────────────────────────────

export const EVENT_TYPES = Object.freeze({
  FLOW_RESET: "flow_reset",
  TRANSCRIPT: "transcript",
  AI_STEP: "ai_step",
  TOOL_CALL: "tool_call",
  VERIFY_DONE: "verify_done",
  CAPTION: "caption",
  FLOW_END: "flow_end",
});

// 타이핑 대상 노드 id (원본 data-type-text 를 그대로 이식).
export const RESET_CAPTION = "고객 발화를 들으며 AI가 필요한 판단을 순서대로 구성합니다.";

// 원본 aiThinkingEvents 를 이벤트 계약 형태로 승격한 데모 재생 스크립트.
// 실제 서버가 붙으면 이 배열 대신 WebSocketEventSource 가 동일한 이벤트를 흘려보낸다.
export const advisorDemoScript = [
  { wait: 0, type: "flow_reset" },

  // 0. 발화 감지
  {
    wait: 0,
    type: "ai_step",
    step: 0,
    reveal: "voice",
    typing: [
      { id: "voiceTranscript", text: "“카드를 잃어버려서 빨리 정지해야 한다고요. 너무 무서워요…”" },
    ],
    toast: "발화 감지",
  },

  // 1. 단서 추출 (감정 신호 + 불안 87% 카운트업)
  {
    wait: 240,
    type: "ai_step",
    step: 1,
    reveal: "emotion",
    typing: [
      { id: "emotionKicker", text: "감정 신호 감지" },
      { id: "emotionLabel", text: "불안 높음" },
    ],
    count: { target: "anxietyScore", value: 87 },
    toast: "단서 추출",
  },

  // 응답 후보 비교 (분기선 + 비채택 후보: 절차 안내 / 추가 질문)
  {
    wait: 260,
    type: "ai_step",
    revealGroup: "response",
    branches: true,
    typing: [
      { id: "responseKicker", text: "대응 판단" },
      { id: "responseLabel", text: "공감 대응" },
    ],
    toast: "응답 후보 비교",
  },

  // 추론 문장 생성
  {
    wait: 260,
    type: "ai_step",
    reveal: "summary",
    typing: [
      {
        id: "judgmentSummary",
        text: "AI가 불안 신호를 기준으로 다른 응답 후보를 비교한 뒤, 지금은 공감 대응을 먼저 선택합니다.",
      },
    ],
    toast: "추론 생성",
  },

  // 2. 본인 확인 — verify_identity 실행 준비
  {
    wait: 280,
    type: "ai_step",
    step: 2,
    reveal: "execution",
    toast: "실행 준비",
  },
  { wait: 0, type: "tool_call", tool: "verify_identity", state: "ready" },

  // 실행 중
  { wait: 700, type: "tool_call", tool: "verify_identity", state: "running", toast: "실행 중" },

  // 3. 결과 수신 — 도구 완료 → 등록 휴대폰 끝자리 확인 자동 체크 → 개인정보 잠금 해제
  { wait: 900, type: "ai_step", step: 3, toast: "완료" },
  { wait: 0, type: "tool_call", tool: "verify_identity", state: "done" },
  {
    wait: 0,
    type: "verify_done",
    check: "phone",
    unlocked: true,
    caption: "AI가 등록 휴대폰 끝자리 확인을 완료하고 결과 수신 단계로 넘어갑니다.",
  },

  // 4. 응답 결정 — 흐름 종료 → 다시 재생 노출
  { wait: 450, type: "ai_step", step: 4 },
  { wait: 0, type: "flow_end", toast: "응답 결정" },
];
