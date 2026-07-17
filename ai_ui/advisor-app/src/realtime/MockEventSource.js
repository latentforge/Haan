import { advisorDemoScript } from "./events.js";

// ─────────────────────────────────────────────────────────────────────────────
// MockEventSource — 음성엔진/자체 LLM 없이 데모가 돌아가게 하는 "가짜 부품".
// 인수인계 문서 §3 어댑터 패턴: 실제 WebSocket 서버가 오면 WebSocketEventSource 로 교체.
//
// 특징: onEvent 가 반환하는 Promise 를 await 한다(back-pressure).
//   → 화면의 타이핑/카운트업 애니메이션이 끝난 뒤 다음 이벤트를 흘려보내므로
//     원본 aiui.html 의 "순차 재생" 타이밍이 그대로 유지된다.
// 실제 WebSocket 은 서버가 push 하므로 back-pressure 가 필요 없다(그쪽 구현 참고).
// ─────────────────────────────────────────────────────────────────────────────

const sleep = (ms) => new Promise((resolve) => setTimeout(resolve, ms));

export class MockEventSource {
  constructor(script = advisorDemoScript) {
    this.script = script;
    this._runId = 0;
    this._onEvent = null;
  }

  // onEvent: (event) => void | Promise<void>
  start(onEvent) {
    this._onEvent = onEvent;
    this._play();
  }

  // 다시 재생 — 현재 진행 중인 재생을 무효화하고 스크립트를 처음부터 흘려보낸다.
  replay() {
    this._play();
  }

  stop() {
    this._runId += 1; // 진행 중 루프 무효화
    this._onEvent = null;
  }

  async _play() {
    const runId = ++this._runId;
    for (const item of this.script) {
      if (runId !== this._runId) return; // stop()/replay() 로 취소됨
      const { wait = 0, ...event } = item;
      if (wait) await sleep(wait);
      if (runId !== this._runId) return;
      if (this._onEvent) await this._onEvent(event);
    }
  }
}
