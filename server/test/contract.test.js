"use strict";
/**
 * 계약 준수 테스트 (무의존성 — node:test / node:assert).
 *
 * 서버의 MockLLM + ToolRunner 를 실제로 구동해, 방출되는 모든 관측 이벤트가
 * SERVICE_ARCHITECTURE.md §6 / advisor-app 의 events.js 계약을 지키는지 검증한다.
 * UI(useAdvisorFlow.PHASE_PRESENTATION)가 이해할 수 있는 phase/state/speaker 만
 * 나오는지까지 확인해, "서버 붙이면 화면 절반이 빈다"는 계약 표류를 막는다.
 *
 * 실행:  node --test   (server/ 에서)
 */
const { test } = require("node:test");
const assert = require("node:assert/strict");

const { MockLLM } = require("../lib/adapters/llm");
const { ToolRunner } = require("../lib/tools");

// ── UI 계약이 인식하는 값들 (events.js / useAdvisorFlow 와 일치해야 함) ──────────
const KNOWN_PHASES = new Set([
  "detect", "emotion", "respond", "reason", "clue", "verify", "result", "decide",
]);
const KNOWN_TOOL_STATES = new Set(["ready", "running", "done", "error"]);
const KNOWN_SPEAKERS = new Set(["고객", "AI", "상담"]);

// ── 봉투를 씌우는 broadcast 스텁 (server.js 의 broadcast 와 동일 규약) ───────────
function makeBus() {
  const events = [];
  let seq = 0;
  const sessionId = "test-ses";
  const broadcast = (evt) => {
    const enveloped = { session_id: sessionId, seq: ++seq, ts: "2026-07-20T00:00:00.000Z", ...evt };
    events.push(enveloped);
    return enveloped;
  };
  return { events, broadcast };
}

// ── 이벤트별 스키마 검증 ────────────────────────────────────────────────────────
function assertEnvelope(evt) {
  assert.equal(typeof evt.session_id, "string", "session_id 필요");
  assert.equal(typeof evt.seq, "number", "seq(number) 필요");
  assert.equal(typeof evt.ts, "string", "ts 필요");
  assert.equal(typeof evt.type, "string", "type 필요");
}

function assertEventValid(evt) {
  assertEnvelope(evt);
  switch (evt.type) {
    case "transcript":
      assert.ok(KNOWN_SPEAKERS.has(evt.speaker), `speaker 계약 위반: ${evt.speaker}`);
      assert.equal(typeof evt.text, "string");
      break;
    case "ai_step":
      assert.ok(KNOWN_PHASES.has(evt.phase), `phase 계약 위반: ${evt.phase}`);
      break;
    case "tool_call":
      assert.ok(KNOWN_TOOL_STATES.has(evt.state), `tool_call.state 계약 위반: ${evt.state}`);
      assert.equal(typeof evt.tool, "string");
      assert.ok(evt.call_id, "call_id 필요");
      break;
    case "verify_state":
      assert.equal(typeof evt.unlocked, "boolean");
      break;
    case "call_state":
      assert.ok(["active", "ended", "idle"].includes(evt.state));
      break;
    default:
      assert.fail(`알 수 없는 이벤트 type: ${evt.type}`);
  }
}

// MockLLM.plan 이 낸 action 을 서버 runActions 와 동일 규약으로 이벤트화한다.
async function runActions(actions, bus, session) {
  const runner = new ToolRunner(bus.broadcast, session);
  for (const a of actions) {
    if (a.kind === "ai_step") {
      const base = { type: "ai_step", id: a.id, phase: a.phase, label: a.label, detail: a.detail, toast: a.label };
      bus.broadcast({ ...base, status: "active" });
      bus.broadcast({ ...base, status: "done" });
    } else if (a.kind === "say") {
      bus.broadcast({ type: "transcript", utterance_id: "u", speaker: "AI", text: a.text, source: "inner_monologue" });
    } else if (a.kind === "tool") {
      const fn = runner[a.tool];
      if (typeof fn === "function") await fn.call(runner, a.id, a.request);
    }
  }
}

test("전체 시나리오: 방출 이벤트가 모두 계약을 지킨다", async () => {
  const llm = new MockLLM();
  const bus = makeBus();
  const session = { unlocked: false, customer: null, pendingIntent: null, askedVerify: false };

  const scriptLines = [
    "여보세요, 제가 지갑을 잃어버려서 카드를 정지시켜야 할 것 같아요.",
    "네, 김민서이고요. 90년 3월 12일생, 전화 끝자리는 4821이에요.",
    "신한카드를 먼저 정지해 주세요.",
    "그리고 그 카드 분실 신고도 같이 접수해 주세요.",
    "네, 감사합니다. 그럼 부탁드릴게요.",
  ];

  for (const text of scriptLines) {
    bus.broadcast({ type: "transcript", utterance_id: "u", speaker: "고객", text, source: "asr" });
    const actions = await llm.plan({ text, session });
    await runActions(actions, bus, session);
    // verify 직후 후속(continueAfterVerify) 도 서버와 동일하게 재생
    if (actions.some((a) => a.kind === "tool" && a.tool === "verify_identity") && session.unlocked) {
      const follow = await llm.continueAfterVerify({ session });
      await runActions(follow, bus, session);
    }
  }

  assert.ok(bus.events.length > 0, "이벤트가 하나도 없음");
  for (const evt of bus.events) assertEventValid(evt);
});

test("seq 는 단조 증가한다", () => {
  const bus = makeBus();
  bus.broadcast({ type: "call_state", state: "active" });
  bus.broadcast({ type: "call_state", state: "ended" });
  assert.equal(bus.events[0].seq, 1);
  assert.equal(bus.events[1].seq, 2);
});

test("본인확인 전에는 개인정보 도구가 잠긴다(서버 권위)", async () => {
  const bus = makeBus();
  const session = { unlocked: false, customer: null };
  const runner = new ToolRunner(bus.broadcast, session);

  const res = await runner.suspend_card("c1", { customer_id: "MOCK-1001", card_id: "CARD-8842" });
  assert.equal(res.status, "denied", "미인증 상태에서 suspend_card 는 거부되어야 함");
  assert.equal(session.unlocked, false, "모델/도구가 임의로 잠금을 열면 안 됨");

  // 마지막 tool_call 이벤트가 error 상태로 방출됐는지
  const last = bus.events.filter((e) => e.type === "tool_call").pop();
  assert.equal(last.state, "error");
});

test("verify_identity 성공이 세션 잠금을 연다", async () => {
  const bus = makeBus();
  const session = { unlocked: false, customer: null };
  const runner = new ToolRunner(bus.broadcast, session);

  const res = await runner.verify_identity("c1", { name: "김민서", birth: "900312", phone_last4: "4821" });
  assert.equal(res.verified, true);
  assert.equal(session.unlocked, true, "verify 성공 시 서버 세션이 잠금 해제");

  const vs = bus.events.find((e) => e.type === "verify_state");
  assert.ok(vs, "verify_state 이벤트 방출 필요");
  assert.equal(vs.unlocked, true);
  assert.equal(vs.check, "phone");
});
