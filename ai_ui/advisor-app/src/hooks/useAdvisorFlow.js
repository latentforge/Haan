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

// ── phase(의미) → 화면 표현 매핑 ─────────────────────────────────────────────
// events.js 는 phase/label/confidence 같은 "의미"만 싣는다. 어느 DOM 노드에 무엇을
// 타이핑할지, 87% 를 어디에 카운트업할지 같은 "표현"은 전적으로 이 테이블이 정한다.
// 모델(Haan/캐스케이드)은 이 테이블을 몰라도 phase 만 내면 화면이 복원된다.
//
//   step      : 강조할 workflow 단계(0~4). 생략 시 단계 이동 없음.
//   reveal    : is-visible 로 드러낼 노드 그룹.
//   branches  : 비채택 후보 분기선 노출.
//   nodes[]   : 타이핑 대상. { id, from } → event[from] 값을, { id, text } → 고정 프리셋.
//               from 값이 비어 있으면 text 로 폴백.
//   count     : event 의 이 필드(숫자)로 anxietyScore 카운트업. 없으면 생략.
const PHASE_PRESENTATION = {
  detect: { step: 0, reveal: "voice", nodes: [{ id: "voiceTranscript", from: "label" }] },
  emotion: {
    step: 1,
    reveal: "emotion",
    nodes: [
      { id: "emotionKicker", text: "감정 신호 감지" },
      { id: "emotionLabel", from: "label" },
    ],
    count: "confidence",
  },
  respond: {
    step: 1,
    reveal: "response",
    branches: true,
    nodes: [
      { id: "responseKicker", text: "대응 판단" },
      { id: "responseLabel", from: "label" },
    ],
  },
  // clue: 감정/응답 노드가 없는 일반 단서(서버 멀티턴의 "대상 카드 특정" 등).
  // 단서 추출 단계만 강조하고 라벨은 toast 로 표시한다(원본 UI 에 중립 슬롯이 없음).
  clue: { step: 1 },
  reason: { reveal: "summary", nodes: [{ id: "judgmentSummary", from: "detail" }] },
  verify: { step: 2, reveal: "execution" },
  result: { step: 3 },
  decide: { step: 4 },
};

const initialState = {
  transcript: initialTranscript,
  workflowStep: 0,
  reveal: { voice: false, emotion: false, response: false, summary: false, execution: false },
  branchesVisible: false,
  typed: emptyTyped(),
  anxietyScore: 0,
  toolState: "ready", // ready | running | done | error
  activeTool: "verify_identity", // tool_call 이 주는 실제 도구명으로 갱신(AiFlowPanel 표시용)
  unlocked: false, // 서버 권위 상태의 UI 미러 — verify_state 로만 바뀜
  checks: initialChecks,
  caption: RESET_CAPTION,
  toast: "분석 준비",
  flowEnded: false,
  callActive: false, // call_state 이벤트로만 갱신 — 통화 시작 전/종료 후엔 "말하는 중" 등 통화 중 UI 숨김
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
            activeTool: "verify_identity",
            unlocked: false,
            checks: s.checks.map((c) => (c.id === "phone" ? { ...c, checked: false } : c)),
            caption: RESET_CAPTION,
            toast: "분석 준비",
            flowEnded: false,
          }));
          return;
        }

        case "transcript": {
          // partial 갱신은 같은 utterance_id 의 미확정 줄을 덮어쓴다(계약: utterance_id).
          setState((s) => {
            const uid = event.utterance_id;
            const idx = uid ? s.transcript.findIndex((t) => t.utterance_id === uid && t.partial) : -1;
            const line = {
              id: idx >= 0 ? s.transcript[idx].id : `t-${Date.now()}-${Math.random().toString(36).slice(2, 7)}`,
              utterance_id: uid,
              speaker: event.speaker,
              text: event.text,
              partial: !!event.partial,
            };
            if (idx >= 0) {
              const next = s.transcript.slice();
              next[idx] = line;
              return { ...s, transcript: next };
            }
            return { ...s, transcript: [...s.transcript, line] };
          });
          return;
        }

        case "ai_step": {
          // 의미론(phase) → 표현. 알 수 없는 phase 는 조용히 무시(전방 호환).
          const preset = PHASE_PRESENTATION[event.phase];
          if (!preset) {
            if (event.caption) setState((s) => ({ ...s, caption: event.caption }));
            if (event.toast) setState((s) => ({ ...s, toast: event.toast }));
            return;
          }
          if (typeof preset.step === "number") {
            setState((s) => ({ ...s, workflowStep: preset.step }));
          }
          if (preset.reveal) {
            setState((s) => ({ ...s, reveal: { ...s.reveal, [preset.reveal]: true } }));
          }
          if (preset.branches) {
            setState((s) => ({ ...s, branchesVisible: true }));
          }
          for (const node of preset.nodes || []) {
            // from 값(서버 데이터)을 우선, 없으면 text(프리셋)로 폴백.
            const text = node.from ? event[node.from] ?? node.text ?? "" : node.text ?? "";
            if (!text) continue;
            // typeText 는 runId 취소 시 즉시 resolve 하므로 별도 가드 불필요.
            await typeText(node.id, text, node.id === "judgmentSummary" ? 16 : 20);
          }
          if (preset.count && typeof event[preset.count] === "number") {
            await countUp(event[preset.count]);
          }
          if (event.caption) setState((s) => ({ ...s, caption: event.caption }));
          if (event.toast) setState((s) => ({ ...s, toast: event.toast }));
          return;
        }

        case "tool_call": {
          setState((s) => ({
            ...s,
            toolState: event.state,
            activeTool: event.tool || s.activeTool,
          }));
          if (event.toast) setState((s) => ({ ...s, toast: event.toast }));
          return;
        }

        case "verify_state": {
          // ⚠️ 데모: verify_identity 성공 = 개인정보/실행 잠금 해제.
          //    잠금 판정의 권위는 서버(session.unlocked)에 있고, 여기서는 그 미러만 갱신한다.
          //    실서비스 전환 시 실제 인증·암호화·개인정보 동의가 필요.
          setState((s) => ({
            ...s,
            unlocked: !!event.unlocked,
            checks: event.check
              ? s.checks.map((c) => (c.id === event.check ? { ...c, checked: !!event.unlocked } : c))
              : s.checks,
          }));
          if (event.caption) setState((s) => ({ ...s, caption: event.caption }));
          return;
        }

        case "caption": {
          setState((s) => ({ ...s, caption: event.text }));
          return;
        }

        case "call_state": {
          // 통화 종료/초기화 신호 — 화면을 완전히 백지로 되돌려 다음 통화가
          // 이전 대화 위에 쌓이지 않게 한다(재시작 시 인사말부터 다시 시작).
          if (event.state === "ended" || event.state === "idle") {
            runIdRef.current += 1; // 진행 중 애니메이션 취소
            setState(initialState);
            return;
          }
          if (event.state === "active") {
            setState((s) => ({ ...s, callActive: true }));
          }
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
