// 상단바: 제목 + 보기 탭(실시간/확인/정리) + 통화 시계.
const TABS = [
  { view: "call", label: "실시간" },
  { view: "verify", label: "확인" },
  { view: "handoff", label: "정리" },
];

export default function TopBar({ activeTab, callClock, onTabChange }) {
  return (
    <header className="topbar" data-od-id="workspace-header">
      <div>
        <h1 data-od-id="workspace-title">AI 상담사 작업공간</h1>
      </div>
      <nav className="tabs" data-od-id="workspace-tabs" aria-label="작업 보기">
        {TABS.map((tab) => (
          <button
            key={tab.view}
            className="tab"
            type="button"
            aria-selected={activeTab === tab.view}
            data-view={tab.view}
            onClick={() => onTabChange(tab.view, tab.label)}
          >
            {tab.label}
          </button>
        ))}
      </nav>
      <div className="call-clock" data-od-id="call-clock">
        <span className="live-dot" aria-hidden="true"></span>
        <span id="callStatus">{callClock}</span>
      </div>
    </header>
  );
}
