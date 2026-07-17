"use strict";
/**
 * 고객 앱(user.html) ↔ Mock 백엔드 WebSocket 브리지.
 *
 * user.html 의 디자인/로직은 건드리지 않고, 서버가 이 스크립트만 </body> 앞에 주입한다.
 * 역할:
 *   - 통화 시작/종료 버튼을 서버 이벤트와 동기화 (call_start / call_end)
 *   - 서버가 보내는 AI 처리 상태(ai_step / tool_call)를 하단 캡션 칩에 반영
 *   - AI 발화(transcript speaker=ai)를 상태 라벨에 잠깐 표시
 *
 * ⚠️ 실제 음성엔진이 붙으면: 마이크 오디오를 서버로 스트리밍(transcript_in 대체)하고,
 *    서버 STT 어댑터가 전사를 만든다. 지금은 서버의 MockSTT 시나리오가 전사를 생성한다.
 */
(function () {
  var proto = location.protocol === "https:" ? "wss" : "ws";
  var ws = null,
    retry = 0;

  var phone = document.getElementById("phone");
  var procText = document.getElementById("procText");
  var caption = document.getElementById("caption");
  var statusText = document.getElementById("statusText");

  var PHASE_LABEL = {
    detect: "의도 감지 중",
    clue: "단서 분석 중",
    verify: "본인확인 진행 중",
    tool: "요청 처리 중",
    answer: "응답 준비 중",
  };
  var TOOL_LABEL = {
    verify_identity: "본인확인 처리 중",
    suspend_card: "카드 정지 처리 중",
    report_lost: "분실 신고 접수 중",
  };

  function setProc(text) {
    if (!procText) return;
    if (caption) caption.classList.remove("waiting");
    procText.textContent = text;
  }
  function setWaiting() {
    if (caption) caption.classList.add("waiting");
    if (procText) procText.textContent = "요청을 기다리고 있어요";
  }

  function connect() {
    ws = new WebSocket(proto + "://" + location.host);
    ws.onopen = function () {
      retry = 0;
      ws.send(JSON.stringify({ type: "hello", role: "customer" }));
    };
    ws.onmessage = function (e) {
      var evt;
      try { evt = JSON.parse(e.data); } catch (_) { return; }
      handle(evt);
    };
    ws.onclose = function () {
      retry++;
      setTimeout(connect, Math.min(5000, 500 * retry));
    };
    ws.onerror = function () { try { ws.close(); } catch (_) {} };
  }

  function handle(evt) {
    switch (evt.type) {
      case "ai_step":
        setProc((PHASE_LABEL[evt.phase] || evt.label || "처리 중"));
        break;
      case "tool_call":
        if (evt.status === "running") setProc(TOOL_LABEL[evt.tool] || "요청 처리 중");
        else if (evt.status === "done") setProc("처리 완료");
        break;
      case "transcript":
        if (evt.speaker === "ai" && statusText) {
          // AI가 말하는 순간 상태 라벨을 잠깐 "말하는 중"으로
          statusText.textContent = "말하는 중…";
        }
        break;
      case "identity":
        if (evt.unlocked) setProc("본인확인 완료 · 정보 조회");
        break;
      case "call_state":
        if (evt.state === "ended") setWaiting();
        break;
      case "snapshot":
        // 재접속 시 별도 처리 불필요 (고객 화면은 현재 상태만 반영)
        break;
    }
  }

  // 고객 앱의 통화 종료 버튼을 서버에도 알림 (양방향 동기화)
  var endBtn = document.getElementById("endBtn");
  if (endBtn) {
    endBtn.addEventListener("click", function () {
      if (ws && ws.readyState === 1) ws.send(JSON.stringify({ type: "call_end" }));
    });
  }
  // "다시 통화하기" → 서버 통화 재시작
  var restartBtn = document.getElementById("restartBtn");
  if (restartBtn) {
    restartBtn.addEventListener("click", function () {
      if (ws && ws.readyState === 1) ws.send(JSON.stringify({ type: "call_start" }));
    });
  }

  connect();
})();
