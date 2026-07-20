// ─────────────────────────────────────────────────────────────────────────────
// WebSocketEventSource — 실제 Mock 백엔드(Node + ws)에 붙는 어댑터.
// MockEventSource 와 동일한 인터페이스(start/replay/stop)를 구현하므로
// createEventSource() 팩토리에서 무손실 교체된다.
//
// 서버는 events.js 의 의미론 계약(transcript / ai_step / tool_call / verify_state /
// caption / call_state / snapshot / flow_end)을 공통 봉투(session_id/seq/ts)에 담아
// JSON 으로 push 한다. 여기서는 봉투의 seq 로 순서를 추적하고, 갭이 감지되면
// 서버에 재동기화(resync)를 요청한다(재접속·유실 복구 — SERVICE_ARCHITECTURE.md §3.2).
// (서버가 push 주도 → back-pressure 불필요. onEvent 반환값은 무시.)
// ─────────────────────────────────────────────────────────────────────────────

export class WebSocketEventSource {
  constructor(url) {
    this.url = url;
    this.ws = null;
    this._onEvent = null;
    this._lastSeq = 0; // 마지막으로 수신·반영한 seq (관측 경로 순서 보장)
  }

  start(onEvent) {
    this._onEvent = onEvent;
    this._connect();
  }

  _connect() {
    this.ws = new WebSocket(this.url);
    this.ws.onopen = () => {
      // 접속/재접속 즉시 스냅샷 동기화 요청 — 통화 도중 붙은 상담사 UI 복구.
      this.ws.send(JSON.stringify({ type: "hello", role: "advisor", last_seq: this._lastSeq }));
    };
    this.ws.onmessage = (msg) => {
      if (!this._onEvent) return;
      let event;
      try {
        event = JSON.parse(msg.data);
      } catch (err) {
        // 계약 위반 메시지는 무시(데모 안정성). 실서비스에선 스키마 검증/로깅 필요.
        console.warn("[WebSocketEventSource] invalid event payload", err);
        return;
      }
      // seq 갭 감지 → 재동기화 요청. snapshot 은 리플레이의 기준점이라 갭 검사에서 제외.
      if (typeof event.seq === "number" && event.type !== "snapshot") {
        if (this._lastSeq > 0 && event.seq > this._lastSeq + 1) {
          this.ws.send(JSON.stringify({ type: "resync", last_seq: this._lastSeq }));
        }
        this._lastSeq = Math.max(this._lastSeq, event.seq);
      }
      this._onEvent(event);
    };
    this.ws.onclose = () => {
      // 데모용 단순 처리. 실서비스에선 지수 백오프 재연결 권장.
    };
  }

  // 서버 주도 스트림이라 클라이언트 replay 는 재요청 신호로 전송(서버가 스크립트 재생).
  replay() {
    if (this.ws && this.ws.readyState === WebSocket.OPEN) {
      this.ws.send(JSON.stringify({ type: "replay" }));
    }
  }

  stop() {
    this._onEvent = null;
    if (this.ws) {
      this.ws.onopen = null;
      this.ws.onmessage = null;
      this.ws.onclose = null;
      this.ws.close();
      this.ws = null;
    }
  }
}
