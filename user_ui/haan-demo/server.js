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
const USER_HTML = path.join(__dirname, "..", "user.html");

// ── WebSocket 클라이언트 관리 ────────────────────────────────
const clients = new Set();
function broadcast(evt) {
  const msg = JSON.stringify(evt);
  for (const c of clients) {
    if (c.alive) c.send(msg);
  }
  recordHistory(evt);
}

// ── 세션 상태 (데모: 단일 통화) ──────────────────────────────
let session = freshSession();
function freshSession() {
  return {
    active: false,
    unlocked: false,
    customer: null,
    pendingIntent: null,
    askedVerify: false,
    startedAt: null,
    history: [], // 늦게 접속한 상담사 UI에 리플레이할 이벤트 로그
  };
}
function recordHistory(evt) {
  if (!session) return;
  // call_reset 전까지의 이벤트만 보관 (스냅샷 replay 용)
  if (["transcript", "ai_step", "tool_call", "identity", "call_state"].includes(evt.type)) {
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
  // 1) 고객 발화를 전사로 브로드캐스트
  broadcast({ type: "transcript", id: rid(), speaker: "customer", text: line.text, at: nowIso() });
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
      broadcast({ type: "ai_step", id: a.id, phase: a.phase, label: a.label, detail: a.detail, status: "active", at: nowIso() });
      await sleep(750);
      broadcast({ type: "ai_step", id: a.id, phase: a.phase, label: a.label, detail: a.detail, status: "done", at: nowIso() });
      await sleep(250);
    } else if (a.kind === "say") {
      broadcast({ type: "transcript", id: rid(), speaker: "ai", text: a.text, at: nowIso() });
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
  broadcast({ type: "call_state", state: "active", at: nowIso() });
  // AI 인사
  broadcast({ type: "transcript", id: rid(), speaker: "ai", text: scenario.greeting, at: nowIso() });

  // MockSTT 로 시나리오 대본 재생 (실제 STT면 이 자리에 오디오 스트림 연결)
  stt = new MockSTT(scenario.lines);
  stt.onTranscript((line) => enqueueUtterance(line));
  stt.start();
}

function endCall() {
  if (stt) { stt.stop(); stt = null; }
  const durMs = session.startedAt ? Date.now() - session.startedAt : 0;
  session.active = false;
  broadcast({ type: "call_state", state: "ended", durationMs: durMs, at: nowIso() });
}

// ── WebSocket 메시지 처리 ────────────────────────────────────
function onClientMessage(conn, raw) {
  let msg;
  try { msg = JSON.parse(raw); } catch { return; }
  switch (msg.type) {
    case "hello":
      conn.meta.role = msg.role || "observer";
      // 스냅샷 리플레이 (통화 도중 접속한 상담사 UI 동기화)
      conn.send({ type: "snapshot", active: session.active, unlocked: session.unlocked,
        profile: session.customer ? require("./lib/seed").publicProfile(session.customer) : null,
        history: session.history });
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
  // 고객 앱이 참조하는 assets/* 는 상위 user_ui 에서 찾아본다
  const alt = path.join(__dirname, "..", safe);
  if (fs.existsSync(alt) && fs.statSync(alt).isFile()) {
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

server.listen(PORT, () => {
  console.log(`\n  Haan 데모 서버 실행 중`);
  console.log(`  ├─ 상담사 UI : http://localhost:${PORT}/agent`);
  console.log(`  └─ 고객 앱    : http://localhost:${PORT}/customer`);
  console.log(`\n  두 화면을 나란히 열고, 상담사 UI에서 "데모 통화 시작"을 누르세요.\n`);
});
