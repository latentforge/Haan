// ─────────────────────────────────────────────────────────────────────────────
// WebSocketEventSource — 실제 Mock 백엔드(Node + ws)에 붙는 어댑터.
// MockEventSource 와 동일한 인터페이스(start/replay/stop)를 구현하므로
// createEventSource() 팩토리에서 무손실 교체된다.
//
// 서버는 events.js 의 계약(transcript / ai_step / tool_call / verify_done /
// caption / flow_end)을 JSON 으로 push 한다. 여기서는 파싱해 onEvent 로 전달만.
// (서버가 push 주도 → back-pressure 불필요. onEvent 반환값은 무시.)
// ─────────────────────────────────────────────────────────────────────────────

export class WebSocketEventSource {
  constructor(url) {
    this.url = url;
    this.ws = null;
    this._onEvent = null;
  }

  start(onEvent) {
    this._onEvent = onEvent;
    this._connect();
  }

  _connect() {
    this.ws = new WebSocket(this.url);
    this.ws.onmessage = (msg) => {
      if (!this._onEvent) return;
      try {
        const event = JSON.parse(msg.data);
        this._onEvent(event);
      } catch (err) {
        // 계약 위반 메시지는 무시(데모 안정성). 실서비스에선 스키마 검증/로깅 필요.
        console.warn("[WebSocketEventSource] invalid event payload", err);
      }
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
      this.ws.onmessage = null;
      this.ws.onclose = null;
      this.ws.close();
      this.ws = null;
    }
  }
}
