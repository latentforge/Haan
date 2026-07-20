import { mockCustomer, mockCards } from "../data/mockData.js";

// 상담 작업대: 고객 정보 + 본인확인 체크리스트(진행률) + 카드 후보.
// 본인확인이 채워질수록 진행률이 오르고, 완료 시 개인정보/실행 잠금이 해제되는 연출.
export default function WorkPanel({ checks, onToggleCheck }) {
  const done = checks.filter((c) => c.checked).length;
  const total = checks.length;
  const percent = Math.round((done / total) * 100);
  const verifyLabel = done === total ? "확인 완료" : `${done}/${total} 확인`;

  return (
    <section className="panel" data-od-id="work-panel">
      <div className="panel-header">
        <div>
          <h2 className="panel-title" data-od-id="work-panel-title">상담 작업대</h2>
        </div>
        <span className="status-pill" id="verifyState" data-od-id="verify-state">{verifyLabel}</span>
      </div>
      <div className="panel-body">
        <article className="customer-card" data-od-id="customer-summary">
          <div className="customer-identity">
            <p className="eyebrow">고객 정보</p>
            <h2>{mockCustomer.name} 고객</h2>
            <span className="identity-phone-tag">휴대폰 끝자리 {mockCustomer.phoneLast4}</span>
          </div>
          <div className="customer-meta" data-od-id="customer-risk-summary">
            {mockCustomer.riskTags.map((tag) => (
              <span
                key={tag.label}
                className={tag.kind === "score" ? "tag risk-tag risk-score-pill" : "tag risk-tag"}
              >
                {tag.label}
              </span>
            ))}
          </div>
        </article>

        <section className="progress-card" data-od-id="identity-progress">
          <div className="progress-top">
            <p className="progress-title">본인 확인 진행</p>
            <span className="progress-count" id="progressCount">{done}/{total}</span>
          </div>
          <div className="progress-track" aria-hidden="true">
            <div className="progress-fill" id="progressFill" style={{ width: `${percent}%` }}></div>
          </div>
          <div className="checklist">
            {checks.map((check) => (
              <label className="check-row" key={check.id}>
                <input
                  className="check-input"
                  type="checkbox"
                  checked={check.checked}
                  onChange={() => onToggleCheck(check.id)}
                />
                <span className="check-label">{check.label}</span>
                <span className="check-state">{check.checked ? "완료" : "대기"}</span>
              </label>
            ))}
          </div>
        </section>

        <section data-od-id="card-candidates">
          <div className="progress-top">
            <p className="progress-title">카드 후보</p>
            <span className="progress-count">최근 사용순</span>
          </div>
          <div className="candidate-grid">
            {mockCards.map((card) => (
              <article className="candidate" key={card.id}>
                <div>
                  <strong>{card.name}</strong>
                  <span>{card.detail}</span>
                </div>
                <span className="confidence">{card.verdict}</span>
              </article>
            ))}
          </div>
        </section>
      </div>
    </section>
  );
}
