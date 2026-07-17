// 좌측 네비게이션 레일. 원본 aiui.html 의 .rail 마크업을 그대로 이식(정적).
export default function Rail() {
  return (
    <nav className="rail" data-od-id="workspace-rail" aria-label="상담사 메뉴">
      <div className="brand-mark" data-od-id="brand-mark">AI</div>
      <button className="rail-button is-active" type="button" data-od-id="nav-live-call" aria-label="실시간 통화">
        <svg viewBox="0 0 24 24" aria-hidden="true">
          <path d="M6.5 10.5c1.2 3.5 3.5 5.8 7 7l2.1-2.1c.3-.3.8-.4 1.2-.2 1.3.4 2.6.7 4 .7.7 0 1.2.5 1.2 1.2v3.3c0 .7-.5 1.2-1.2 1.2C10.4 21.6 2.4 13.6 2.4 3.2 2.4 2.5 2.9 2 3.6 2h3.3c.7 0 1.2.5 1.2 1.2 0 1.4.2 2.7.7 4 .1.4 0 .8-.3 1.2l-2 2.1Z" />
        </svg>
      </button>
      <button className="rail-button" type="button" data-od-id="nav-customer" aria-label="고객 정보">
        <svg viewBox="0 0 24 24" aria-hidden="true">
          <path d="M19 21a7 7 0 0 0-14 0" />
          <circle cx="12" cy="8" r="4" />
        </svg>
      </button>
      <button className="rail-button" type="button" data-od-id="nav-knowledge" aria-label="상담 지식">
        <svg viewBox="0 0 24 24" aria-hidden="true">
          <path d="M4 19.5A2.5 2.5 0 0 1 6.5 17H20" />
          <path d="M6.5 2H20v20H6.5A2.5 2.5 0 0 1 4 19.5v-15A2.5 2.5 0 0 1 6.5 2Z" />
        </svg>
      </button>
      <button className="rail-button" type="button" data-od-id="nav-history" aria-label="처리 이력">
        <svg viewBox="0 0 24 24" aria-hidden="true">
          <path d="M3 12a9 9 0 1 0 3-6.7" />
          <path d="M3 4v5h5" />
          <path d="M12 7v5l3 2" />
        </svg>
      </button>
    </nav>
  );
}
