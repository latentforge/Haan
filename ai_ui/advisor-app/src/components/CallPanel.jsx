import { useEffect, useRef } from "react";

// 현재 통화 패널: STT 전사가 실시간으로 줄 추가되는 왼쪽 패널.
// transcript 배열을 렌더하고, 맨 아래 speaking-line 으로 "말하는 중"을 표시.
export default function CallPanel({ transcript }) {
  const bodyRef = useRef(null);

  // 새 발화가 추가되면 최신 줄로 자동 스크롤.
  useEffect(() => {
    const el = bodyRef.current;
    if (el) el.scrollTo({ top: el.scrollHeight, behavior: "smooth" });
  }, [transcript]);

  return (
    <aside className="panel" data-od-id="call-panel">
      <div className="panel-header">
        <div>
          <h2 className="panel-title" data-od-id="call-panel-title">현재 통화</h2>
        </div>
        <span className="status-pill" data-od-id="call-status-pill">
          <span className="live-dot"></span>듣는 중
        </span>
      </div>
      <div className="panel-body" ref={bodyRef}>
        <div className="transcript" data-od-id="transcript-list">
          {transcript.map((line) => (
            <article className="line" key={line.id}>
              <span className="speaker">{line.speaker}</span>
              <p className="utterance">{line.text}</p>
            </article>
          ))}
          {/* 발화 중 인디케이터 — 실서비스에선 STT partial 스트림으로 토글 */}
          <article className="line speaking-line" data-od-id="transcript-customer-speaking">
            <span className="speaker">고객</span>
            <p className="utterance speaking-copy">
              <span className="speaking-dots" aria-hidden="true">
                <span></span>
                <span></span>
                <span></span>
              </span>
              <span>말하는 중</span>
            </p>
          </article>
        </div>
      </div>
    </aside>
  );
}
