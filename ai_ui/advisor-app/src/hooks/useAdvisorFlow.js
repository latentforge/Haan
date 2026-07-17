import { useCallback, useEffect, useRef, useState } from "react";
import { createEventSource } from "../realtime/createEventSource.js";
import { RESET_CAPTION } from "../realtime/events.js";
import { initialChecks, initialTranscript } from "../data/mockData.js";

// ─────────────────────────────────────────────────────────────────────────────
// useAdvisorFlow — 이벤트소스(Mock/WebSocket)를 구독해 화면 상태를 구동한다.
// 원본 aiui.html 의 재생 함수(typeText / countUp / setToolStatus / setWorkflowStep /
// revealNode / completePhone / updateProgress)를 React 상태 갱신으로 이식한 것.
// UI 컴포넌트는 상태만 렌더하고, 애니메이션 타이밍은 여기서 관장한다.
// ─────────────────────────────────────────────────────────────────────────────

const TYPED_IDS = [
  "voiceTranscript",
  "emotionKicker",
  "emotionLabel",
  "responseKicker",
  "responseLabel",
  "judgmentSummary",
];

const emptyTyped = () => Object.fromEntries(TYPED_IDS.map((id) => [id, ""]));

const initialState = {
  transcript: initialTranscript,
  workflowStep: 0,
  reveal: { voice: false, emotion: false, response: false, summary: false, execution: false },
  branchesVisible: false,
  typed: emptyTyped(),
  anxietyScore: 0,
  toolState: "ready", // ready | running | done
  checks: initialChecks,
  caption: RESET_CAPTION,
  toast: "분석 준비",
  flowEnded: false,
};

export function useAdvisorFlow() {
  const [state, setState] = useState(initialState);
  const [elapsed, setElapsed] = useState(41); // 원본: 통화 중 00:41 부터 시작
  const [activeTab, setActiveTab] = useState("call");

  const sourceRef = useRef(null);
  const runIdRef = useRef(0); // 애니메이션 취소용 (원본 aiFlowRun)

  // 통화 시간 시계.
  useEffect(() => {
    const startedAt = Date.now() - 41000;
    const tick = () => setElapsed(Math.max(0, Math.floor((Date.now() - startedAt) / 1000)));
    tick();
    const timer = setInterval(tick, 1000);
    return () => clearInterval(timer);
  }, []);

  // ── 애니메이터 (runId 로 취소 가능) ──────────────────────────────────────────
  const typeText = useCallback((id, text, speed) => {
    return new Promise((resolve) => {
      const runId = runIdRef.current;
      let index = 0;
      const tick = () => {
        if (runId !== runIdRef.current) return resolve();
        const slice = text.slice(0, index);
        setState((s) => ({ ...s, typed: { ...s.typed, [id]: slice } }));
        index += 1;
        if (index <= text.length) setTimeout(tick, speed);
        else resolve();
      };
      tick();
    });
  }, []);

  const countUp = useCallback((value, duration = 620) => {
    return new Promise((resolve) => {
      const runId = runIdRef.current;
      const start = performance.now();
      const frame = (now) => {
        if (runId !== runIdRef.current) return resolve();
        const progress = Math.min(1, (now - start) / duration);
        const eased = 1 - Math.pow(1 - progress, 3);
        setState((s) => ({ ...s, anxietyScore: Math.round(value * eased) }));
        if (progress < 1) requestAnimationFrame(frame);
        else resolve();
      };
      requestAnimationFrame(frame);
    });
  }, []);

  // ── 이벤트 핸들러 (계약 events.js 를 상태로 반영) ────────────────────────────
  const handleEvent = useCallback(
    async (event) => {
      switch (event.type) {
        case "flow_reset": {
          runIdRef.current += 1; // 진행 중 애니메이션 취소
          setState((s) => ({
            ...s,
            workflowStep: 0,
            reveal: { voice: false, emotion: false, response: false, summary: false, execution: false },
            branchesVisible: false,
            typed: emptyTyped(),
            anxietyScore: 0,
            toolState: "ready",
            checks: s.checks.map((c) => (c.id === "phone" ? { ...c, checked: false } : c)),
            caption: RESET_CAPTION,
            toast: "분석 준비",
            flowEnded: false,
          }));
          return;
        }

        case "transcript": {
          const line = {
            id: `t-${Date.now()}-${Math.random().toString(36).slice(2, 7)}`,
            speaker: event.speaker,
            text: event.text,
            partial: !!event.partial,
          };
          setState((s) => ({ ...s, transcript: [...s.transcript, line] }));
          return;
        }

        case "ai_step": {
          if (typeof event.step === "number") {
            setState((s) => ({ ...s, workflowStep: event.step }));
          }
          if (event.reveal) {
            setState((s) => ({ ...s, reveal: { ...s.reveal, [event.reveal]: true } }));
          }
          if (event.revealGroup === "response") {
            setState((s) => ({ ...s, reveal: { ...s.reveal, response: true } }));
          }
          if (event.branches) {
            setState((s) => ({ ...s, branchesVisible: true }));
          }
          for (const node of event.typing || []) {
            // typeText 는 runId 취소 시 즉시 resolve 하므로 별도 가드 불필요.
            await typeText(node.id, node.text, node.id === "judgmentSummary" ? 16 : 20);
          }
          if (event.count) {
            await countUp(event.count.value);
          }
          if (event.caption) setState((s) => ({ ...s, caption: event.caption }));
          if (event.toast) setState((s) => ({ ...s, toast: event.toast }));
          return;
        }

        case "tool_call": {
          setState((s) => ({ ...s, toolState: event.state }));
          if (event.toast) setState((s) => ({ ...s, toast: event.toast }));
          return;
        }

        case "verify_done": {
          // ⚠️ 데모: verify_identity 성공 = 개인정보/실행 잠금 해제.
          //    실서비스 전환 시 실제 인증·암호화·개인정보 동의가 필요.
          setState((s) => ({
            ...s,
            checks: s.checks.map((c) => (c.id === event.check ? { ...c, checked: true } : c)),
          }));
          if (event.caption) setState((s) => ({ ...s, caption: event.caption }));
          return;
        }

        case "caption": {
          setState((s) => ({ ...s, caption: event.text }));
          return;
        }

        case "flow_end": {
          setState((s) => ({ ...s, flowEnded: true }));
          if (event.toast) setState((s) => ({ ...s, toast: event.toast }));
          return;
        }

        default:
          return;
      }
    },
    [typeText, countUp]
  );

  // ── 이벤트소스 구독 ─────────────────────────────────────────────────────────
  useEffect(() => {
    const source = createEventSource();
    sourceRef.current = source;
    source.start(handleEvent);
    return () => source.stop();
  }, [handleEvent]);

  // ── 액션 ───────────────────────────────────────────────────────────────────
  const replay = useCallback(() => {
    sourceRef.current?.replay();
  }, []);

  // 수동 체크박스 토글 (원본 updateProgress 의 caption/toast 규칙 이식).
  const toggleCheck = useCallback((id) => {
    setState((s) => {
      const checks = s.checks.map((c) => (c.id === id ? { ...c, checked: !c.checked } : c));
      const done = checks.filter((c) => c.checked).length;
      const total = checks.length;
      const complete = done === total;
      return {
        ...s,
        checks,
        caption: complete
          ? "본인 확인이 완료되어 상담사는 카드 정지 처리 안내로 넘어갈 수 있습니다."
          : "본인 확인 단계가 남아 있어 AI가 실행 전 체크리스트를 잠근 상태입니다.",
        toast: complete ? "확인 완료" : "확인 진행",
      };
    });
  }, []);

  const changeTab = useCallback((view, label) => {
    setActiveTab(view);
    setState((s) => ({
      ...s,
      caption: `${label} 보기로 전환했습니다. 상담 흐름은 유지되고 필요한 패널만 우선순위가 바뀝니다.`,
      toast: `${label} 보기`,
    }));
  }, []);

  const mm = String(Math.floor(elapsed / 60)).padStart(2, "0");
  const ss = String(elapsed % 60).padStart(2, "0");

  return {
    ...state,
    activeTab,
    callClock: `통화 중 ${mm}:${ss}`,
    replay,
    toggleCheck,
    changeTab,
  };
}
