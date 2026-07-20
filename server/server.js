"use strict";
/**
 * Haan 데모 Mock 백엔드 — 무의존성 Node 서버.
 *
 * 역할:
 *   1) 정적 파일 서빙 (상담사 UI /agent, 고객 앱 /customer)
 *   2) 고객 앱(user.html)에 WebSocket 브리지 스크립트를 주입해 서빙
 *   3) WebSocket 허브 — 고객/상담사 클라이언트에 이벤트 브로드캐스트
 *   4) 세션 오케스트레이션 — MockSTT(전사) → MockLLM(판단) → ToolRunner(도구)
 *
 * 실행:  node server.js   (기본 포트 5173)
 *
 * ⚠️ 데모용: 인증/암호화/개인정보 동의 없음. 어댑터(STT/LLM/카드사 API)는 모두 Mock.
 *    실서비스 전환 시 lib/adapters/* 와 lib/tools.js 를 실제 구현으로 교체.
 */
const http = require("http");
const fs = require("fs");
const path = require("path");
const { attachWebSocket } = require("./lib/ws");
const { MockSTT } = require("./lib/adapters/stt");
const { MockLLM } = require("./lib/adapters/llm");
const { ToolRunner } = require("./lib/tools");
const scenario = require("./lib/scenario");

const PORT = process.env.PORT || 5173;
const PUBLIC = path.join(__dirname, "public");
// 고객 앱 원본. server/ 가 리포 루트로 승격되어, 고객 HTML 은 user_ui/ 아래에 있다.
const USER_HTML = path.join(__dirname, "..", "user_ui", "user.html");
const USER_ASSETS = path.join(__dirname, "..", "user_ui");

// ── WebSocket 클라이언트 관리 ────────────────────────────────
const clients = new Set();
// 공통 봉투(session_id/seq/ts)의 seq — 오케스트레이터가 부여하는 단조 증가 번호.
// 세션(통화)이 바뀌어도 seq 는 리셋하지 않는다(리플레이 기준의 단조성 보장).
let _seq = 0;
/**
 * 관측 경로 이벤트를 모든 클라이언트에 브로드캐스트한다.
 * SERVICE_ARCHITECTURE.md §6 계약대로 공통 봉투를 씌워 보내고 history 에 적재한다.
 * @returns {object} 봉투가 씌워진 이벤트 (호출부에서 필요 시 참조)
 */
function broadcast(evt) {
  const enveloped = { session_id: session.sessionId, seq: ++_seq, ts: nowIso(), ...evt };
  const msg = JSON.stringify(enveloped);
  for (const c of clients) {
    if (c.alive) c.send(msg);
  }
  recordHistory(enveloped);
  return enveloped;
}

// ── 세션 상태 (데모: 단일 통화) ──────────────────────────────
let session = freshSession();
function freshSession() {
  return {
    sessionId: `ses-${Date.now().toString(36)}`,
    active: false,
    unlocked: false, // 잠금 판정의 권위 — 모델이 아니라 이 서버 상태가 진실(§4)
    customer: null,
    pendingIntent: null,
    askedVerify: false,
    startedAt: null,
    history: [], // 늦게 접속한 상담사 UI에 리플레이할 이벤트 로그(봉투 포함)
  };
}
function recordHistory(evt) {
  if (!session) return;
  // 관측 이벤트만 보관 (스냅샷/재동기화 replay 용)
  if (["transcript", "ai_step", "tool_call", "verify_state", "call_state"].includes(evt.type)) {
    session.history.push(evt);
  }
}

const llm = new MockLLM();
let stt = null;
const sleep = (ms) => new Promise((r) => setTimeout(r, ms));

// ── 오케스트레이션: 전사 한 줄 → 판단 → 도구 ──────────────────
let processing = Promise.resolve(); // 발화 처리 직렬화
function enqueueUtterance(line) {
  processing = processing.then(() => handleUtterance(line)).catch((e) => {
    console.error("utterance error:", e);
  });
}

async function handleUtterance(line) {
  // 1) 고객 발화를 전사로 브로드캐스트 (계약: speaker 는 한글, utterance_id 로 partial 갱신 대상 식별)
  broadcast({
    type: "transcript",
    utterance_id: rid(),
    speaker: "고객",
    text: line.text,
    source: "asr", // 고객 발화는 스트리밍 ASR 산출물(§3.1). 실서비스에선 별도 ASR 사이드카.
  });
  await sleep(500); // AI가 "생각을 시작"하는 짧은 텀

  // 2) LLM 판단 → 행동 목록
  const actions = await llm.plan({ text: line.text, session });
  await runActions(actions);
}

async function runActions(actions) {
  const runner = new ToolRunner(broadcast, session);
  for (const a of actions) {
    if (a.kind === "ai_step") {
      // 노드 하나를 라이브로: active(진행중) → done(완료)
      // toast 에 label 을 실어, 원본 UI 의 감정/응답 노드가 없는 phase(clue 등)도 화면에 라벨이 뜨게 한다.
      const base = { type: "ai_step", id: a.id, phase: a.phase, label: a.label, detail: a.detail, toast: a.label };
      broadcast({ ...base, status: "active" });
      await sleep(750);
      broadcast({ ...base, status: "done" });
      await sleep(250);
    } else if (a.kind === "say") {
      broadcast({ type: "transcript", utterance_id: rid(), speaker: "AI", text: a.text, source: "inner_monologue" });
      await sleep(400);
    } else if (a.kind === "tool") {
      const wasUnlocked = session.unlocked;
      const fn = runner[a.tool];
      if (typeof fn === "function") {
        const res = await fn.call(runner, a.id, a.request);
        await sleep(300);
        // verify_identity 가 방금 성공해 잠금 해제됐다면 후속 안내를 이어서 스트리밍
        if (a.tool === "verify_identity" && !wasUnlocked && session.unlocked) {
          const follow = await llm.continueAfterVerify({ session });
          await runActions(follow);
        }
      }
    }
  }
}

// ── 통화 라이프사이클 ────────────────────────────────────────
function startCall() {
  if (session.active) return;
  session = freshSession();
  session.active = true;
  session.startedAt = Date.now();
  broadcast({ type: "call_state", state: "active" });
  // AI 인사
  broadcast({ type: "transcript", utterance_id: rid(), speaker: "AI", text: scenario.greeting, source: "inner_monologue" });

  // MockSTT 로 시나리오 대본 재생 (실제 STT면 이 자리에 오디오 스트림 연결)
  stt = new MockSTT(scenario.lines);
  stt.onTranscript((line) => enqueueUtterance(line));
  stt.start();
}

function endCall() {
  if (stt) { stt.stop(); stt = null; }
  const durMs = session.startedAt ? Date.now() - session.startedAt : 0;
  session.active = false;
  broadcast({ type: "call_state", state: "ended", duration_ms: durMs });
}

// ── WebSocket 메시지 처리 ────────────────────────────────────
function onClientMessage(conn, raw) {
  let msg;
  try { msg = JSON.parse(raw); } catch { return; }
  switch (msg.type) {
    case "hello": {
      conn.meta.role = msg.role || "observer";
      // 스냅샷: 현재 세션 요약 + 마지막 seq(리플레이 기준점).
      conn.send({
        type: "snapshot",
        session_id: session.sessionId,
        seq: _seq,
        ts: nowIso(),
        active: session.active,
        unlocked: session.unlocked,
        profile: session.customer ? require("./lib/seed").publicProfile(session.customer) : null,
        last_seq: _seq,
      });
      // 새로고침/재접속 시 과거 이력을 즉시 재생하지 않는다 — 화면은 백지에서 시작해
      // 이후 라이브 이벤트로만 채워진다. 같은 접속을 유지한 채의 seq 갭 복구는 resync(아래)로.
      break;
    }
    case "resync":
      // seq 갭을 감지한 클라이언트의 재동기화 요청.
      replayHistory(conn, Number(msg.last_seq) || 0);
      break;
    case "replay":
      // 서버 경로의 "다시 재생" = 데모 통화 재시작.
      startCall();
      break;
    case "call_start":
      startCall();
      break;
    case "call_end":
      endCall();
      break;
    case "transcript_in":
      // 수동 주입(상담사 UI 테스트 입력 또는 실제 STT 어댑터 피드)
      if (!session.active) startCall();
      enqueueUtterance({ speaker: "customer", text: String(msg.text || "").trim(), ts: Date.now() });
      break;
    case "reset":
      if (stt) { stt.stop(); stt = null; }
      session = freshSession();
      broadcast({ type: "call_state", state: "idle", at: nowIso() });
      break;
    default:
      break;
  }
}

// ── HTTP 정적 서빙 ───────────────────────────────────────────
const MIME = {
  ".html": "text/html; charset=utf-8",
  ".js": "text/javascript; charset=utf-8",
  ".css": "text/css; charset=utf-8",
  ".png": "image/png",
  ".svg": "image/svg+xml",
  ".json": "application/json; charset=utf-8",
};

const server = http.createServer((req, res) => {
  const url = req.url.split("?")[0];

  if (url === "/" || url === "/agent" || url === "/agent/") {
    return sendFile(res, path.join(PUBLIC, "agent.html"));
  }
  if (url === "/customer" || url === "/customer/") {
    return sendCustomerApp(res);
  }
  if (url === "/health") {
    res.writeHead(200, { "Content-Type": "application/json" });
    return res.end(JSON.stringify({ ok: true, active: session.active }));
  }

  // 정적 파일 (public/ 우선, 그다음 상위의 assets 등)
  const safe = path.normalize(url).replace(/^([/\\])+/, "");
  let filePath = path.join(PUBLIC, safe);
  if (!filePath.startsWith(PUBLIC)) return notFound(res);
  if (fs.existsSync(filePath) && fs.statSync(filePath).isFile()) {
    return sendFile(res, filePath);
  }
  // 고객 앱이 참조하는 assets/* 는 user_ui/ 에서 찾아본다
  const alt = path.join(USER_ASSETS, safe);
  if (alt.startsWith(USER_ASSETS) && fs.existsSync(alt) && fs.statSync(alt).isFile()) {
    return sendFile(res, alt);
  }
  return notFound(res);
});

/** 고객 앱(user.html)에 WebSocket 브리지 스크립트를 주입해 서빙. */
function sendCustomerApp(res) {
  fs.readFile(USER_HTML, "utf8", (err, html) => {
    if (err) return notFound(res);
    const inject = `\n<script src="/customer-bridge.js"></script>\n`;
    const out = html.includes("</body>")
      ? html.replace("</body>", inject + "</body>")
      : html + inject;
    res.writeHead(200, { "Content-Type": MIME[".html"] });
    res.end(out);
  });
}

function sendFile(res, filePath) {
  fs.readFile(filePath, (err, buf) => {
    if (err) return notFound(res);
    const ext = path.extname(filePath).toLowerCase();
    res.writeHead(200, { "Content-Type": MIME[ext] || "application/octet-stream" });
    res.end(buf);
  });
}
function notFound(res) {
  res.writeHead(404, { "Content-Type": "text/plain; charset=utf-8" });
  res.end("404 Not Found");
}

// ── WebSocket 부착 ───────────────────────────────────────────
const wss = attachWebSocket(server);
wss.on("connection", (conn) => {
  clients.add(conn);
  conn.on("message", (raw) => onClientMessage(conn, raw));
  conn.on("close", () => clients.delete(conn));
});

// ── util ─────────────────────────────────────────────────────
let _rid = 0;
function rid() { return `m-${Date.now().toString(36)}-${++_rid}`; }
function nowIso() { return new Date().toISOString(); }

/** history 에서 sinceSeq 이후 이벤트만 해당 conn 에 재전송(리플레이/재동기화 공통). */
function replayHistory(conn, sinceSeq) {
  for (const evt of session.history) {
    if (evt.seq > sinceSeq && conn.alive) conn.send(evt);
  }
}

server.listen(PORT, () => {
  console.log(`\n  Haan 데모 서버 실행 중`);
  console.log(`  ├─ 상담사 UI : http://localhost:${PORT}/agent`);
  console.log(`  └─ 고객 앱    : http://localhost:${PORT}/customer`);
  console.log(`\n  두 화면을 나란히 열고, 상담사 UI에서 "데모 통화 시작"을 누르세요.\n`);
});
