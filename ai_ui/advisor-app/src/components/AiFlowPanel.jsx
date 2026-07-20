import { useEffect, useRef } from "react";

// AI 판단 및 실행 흐름 — "정적 표시"가 아니라 "실시간 사고 재생 흐름"(프롬프트 §5.5).
// 발화 감지 → 감정 신호 → 대응 판단(비채택 후보 비교) → 추론 문장 → verify_identity 실행.

const WORKFLOW_STEPS = ["발화 감지", "단서 추출", "본인 확인", "결과 수신", "응답 결정"];

const TOOL_STATE_TEXT = { ready: "실행 준비", running: "실행 중", done: "완료", error: "실패" };

function stepClass(index, current) {
  if (index < current) return "done";
  if (index === current) return "current";
  return "pending";
}

// thinking-node/response-node 의 등장(reveal)을 is-visible 로 토글.
const vis = (base, on) => `${base}${on ? " is-visible" : ""}`;

export default function AiFlowPanel({
  workflowStep,
  reveal,
  branchesVisible,
  typed,
  anxietyScore,
  toolState,
  activeTool,
  flowEnded,
  onReplay,
}) {
  const bodyRef = useRef(null);

  // 노드가 하나씩 드러날 때마다 최신 위치로 자동 스크롤(원본 scrollWorkflowToLatest).
  useEffect(() => {
    const el = bodyRef.current;
    if (el) el.scrollTo({ top: el.scrollHeight, behavior: "smooth" });
  }, [reveal, branchesVisible, typed, flowEnded, toolState]);

  return (
    <aside className="panel" data-od-id="ai-assist-panel">
      <div className="panel-header">
        <div>
          <h2 className="panel-title" data-od-id="ai-assist-title">AI 판단 및 실행 흐름</h2>
        </div>
        <span className="status-pill flow-status" data-od-id="ai-ready-pill">
          <span className="flow-dot"></span>실시간 업데이트
        </span>
      </div>
      <div className="panel-body ai-flow-body workflow-panel" ref={bodyRef}>
        <nav className="workflow-steps" id="workflowSteps" data-od-id="ai-flow-timeline" aria-label="AI 처리 흐름">
          {WORKFLOW_STEPS.map((label, index) => (
            <span key={label} className={stepClass(index, workflowStep)} data-step={index}>
              {label}
            </span>
          ))}
        </nav>

        <section className="workflow-scene" id="aiWorkflowScene" data-od-id="animated-ai-workflow">
          <div className={vis("signal-card thinking-node", reveal.voice)} id="voiceNode" data-od-id="voice-detection-card">
            <span className="assist-label">고객 발화 감지</span>
            <p id="voiceTranscript">{typed.voiceTranscript}</p>
          </div>

          <div className={vis("judgment-flow thinking-node", reveal.emotion)} id="emotionNode" data-od-id="vertical-judgment-flow">
            <div className="flow-block" data-od-id="emotion-judgment-flow">
              <span className="flow-kicker" id="emotionKicker">{typed.emotionKicker}</span>
              <span className="flow-pill emotion">
                <span id="emotionLabel">{typed.emotionLabel}</span>
                <span className="emotion-score" id="anxietyScore">{anxietyScore}%</span>
              </span>
              <div
                className={vis("consideration-branches response-node", reveal.response || branchesVisible)}
                id="responseBranches"
                data-od-id="response-consideration-branches"
                aria-label="비채택 판단 후보"
              >
                <svg viewBox="0 0 240 44" preserveAspectRatio="none" aria-hidden="true">
                  <path className="branch-line selected" d="M28 0 C28 14 28 30 28 44" />
                  <path className="branch-line ghost" d="M28 7 C82 7 96 11 146 11" />
                  <path className="branch-line ghost secondary" d="M28 18 C72 24 84 33 128 35" />
                </svg>
                <span className="ghost-option primary">절차 안내</span>
                <span className="ghost-option secondary">추가 질문</span>
              </div>
              <span className={vis("flow-kicker response-kicker response-node", reveal.response)} id="responseKicker">
                {typed.responseKicker}
              </span>
              <span className={vis("flow-pill response response-node", reveal.response)} id="responsePill">
                <span id="responseLabel">{typed.responseLabel}</span>
              </span>
            </div>
          </div>

          <p
            className={vis("judgment-summary thinking-node", reveal.summary)}
            id="judgmentSummary"
            data-od-id="judgment-summary"
          >
            {typed.judgmentSummary}
          </p>
        </section>

        <section
          className={vis("next-execution-card thinking-node", reveal.execution)}
          id="nextExecutionCard"
          data-od-id="next-execution-card"
        >
          <span className="assist-label">다음 실행</span>
          <div className="next-execution-row">
            <div className="next-execution-meta">
              <span>Tool</span>
              <code>{activeTool}</code>
            </div>
            <span className={`execution-ready ${toolState}`} id="toolStatus">
              <span className="status-spinner" aria-hidden="true"></span>
              <span id="toolStatusText">{TOOL_STATE_TEXT[toolState]}</span>
            </span>
          </div>
          <button
            className={vis("replay-flow", flowEnded)}
            id="replayFlow"
            type="button"
            onClick={onReplay}
          >
            ▷ 다시 재생
          </button>
        </section>
      </div>
    </aside>
  );
}
