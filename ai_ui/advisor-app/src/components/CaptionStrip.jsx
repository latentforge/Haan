// 하단 실시간 자막 스트립.
export default function CaptionStrip({ caption, toast }) {
  return (
    <footer className="caption-strip" data-od-id="live-caption-strip">
      <span className="caption-label">실시간 자막</span>
      <p className="caption-text" id="captionText">{caption}</p>
      <span className="toast" id="toast" data-od-id="toast-message">{toast}</span>
    </footer>
  );
}
