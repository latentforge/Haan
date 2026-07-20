import { MockEventSource } from "./MockEventSource.js";
import { WebSocketEventSource } from "./WebSocketEventSource.js";

// 어댑터 선택 지점(자동차의 엔진 슬롯). VITE_WS_URL 이 있으면 실제 WebSocket,
// 없으면 내장 Mock 재생. 이 한 곳만 바꾸면 실서비스 백엔드로 전환된다.
export function createEventSource() {
  const wsUrl = import.meta.env.VITE_WS_URL;
  if (wsUrl) {
    return new WebSocketEventSource(wsUrl);
  }
  return new MockEventSource();
}
