"use strict";
/**
 * STT(음성→텍스트) 어댑터 인터페이스 + Mock 구현.
 *
 * 설계 원칙(요구사항 §3): 실제 음성 인식 엔진은 사용자(팀)가 자체 구현 중이다.
 * Claude Code 측은 "전사 텍스트가 한 줄씩 들어온다"고 가정하고 **입력 어댑터만** 둔다.
 * 나중에 실제 STT가 오면 이 인터페이스를 구현하는 클래스로 교체하면 된다.
 *
 *   interface STTAdapter {
 *     onTranscript(cb: (line:{speaker, text, ts}) => void): void
 *     feed(audioChunk): void      // 실제 엔진: 오디오 입력
 *     start(): void / stop(): void
 *   }
 *
 * MockSTT 는 오디오 대신 시나리오 대본을 타이머로 흘려보내 전사 스트림을 흉내낸다.
 */

class STTAdapter {
  onTranscript(cb) {
    this._cb = cb;
  }
  _emit(line) {
    if (this._cb) this._cb(line);
  }
  start() {}
  stop() {}
  /** 실제 STT 어댑터가 오디오/외부 전사를 주입하는 진입점 (수동 주입에도 사용). */
  feed(text, speaker = "customer") {
    this._emit({ speaker, text, ts: Date.now() });
  }
}

/**
 * MockSTT: 대본(customer 발화 배열)을 지정 딜레이로 순차 emit.
 * 실제 통화에서 고객이 말하는 것을 흉내낸다.
 */
class MockSTT extends STTAdapter {
  constructor(script) {
    super();
    this.script = script || [];
    this._timers = [];
    this._running = false;
  }

  start() {
    if (this._running) return;
    this._running = true;
    let elapsed = 0;
    for (const step of this.script) {
      elapsed += step.delay || 1500;
      const t = setTimeout(() => {
        if (!this._running) return;
        this._emit({ speaker: "customer", text: step.text, ts: Date.now() });
      }, elapsed);
      this._timers.push(t);
    }
  }

  stop() {
    this._running = false;
    this._timers.forEach(clearTimeout);
    this._timers = [];
  }
}

module.exports = { STTAdapter, MockSTT };
