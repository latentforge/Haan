import { useAdvisorFlow } from "./hooks/useAdvisorFlow.js";
import Rail from "./components/Rail.jsx";
import TopBar from "./components/TopBar.jsx";
import CallPanel from "./components/CallPanel.jsx";
import WorkPanel from "./components/WorkPanel.jsx";
import AiFlowPanel from "./components/AiFlowPanel.jsx";
import CaptionStrip from "./components/CaptionStrip.jsx";

// 상담사 태블릿 앱 셸.
// device-frame(검은 태블릿 테두리)·안드로이드 상태바/네비 목업은 제거하고,
// tablet-screen 이하를 .advisor-screen 으로 풀스크린 렌더(프롬프트 §5/§6).
export default function App() {
  const flow = useAdvisorFlow();

  return (
    <main className="page" data-od-id="page-shell">
      <div className="advisor-screen" data-od-id="advisor-tablet-screen">
        <div className="app" data-od-id="advisor-workspace">
          <Rail />
          <section className="workspace" data-od-id="workspace-content">
            <TopBar activeTab={flow.activeTab} callClock={flow.callClock} onTabChange={flow.changeTab} />

            <section className="main-grid" data-od-id="advisor-grid">
              <CallPanel transcript={flow.transcript} />
              <WorkPanel checks={flow.checks} onToggleCheck={flow.toggleCheck} />
              <AiFlowPanel
                workflowStep={flow.workflowStep}
                reveal={flow.reveal}
                branchesVisible={flow.branchesVisible}
                typed={flow.typed}
                anxietyScore={flow.anxietyScore}
                toolState={flow.toolState}
                flowEnded={flow.flowEnded}
                onReplay={flow.replay}
              />
            </section>

            <CaptionStrip caption={flow.caption} toast={flow.toast} />
          </section>
        </div>
      </div>
    </main>
  );
}
