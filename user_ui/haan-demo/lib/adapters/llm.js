"use strict";
/**
 * LLM(대화·판단) 어댑터 인터페이스 + Mock 구현.
 *
 * 설계 원칙(요구사항 §3): 자체 대화·판단 모델(Haan)은 아직 없다.
 * 지금은 규칙 기반 Mock 으로 대체하되, 실제 모델이 오면 아래 인터페이스만 구현해
 * 교체할 수 있도록 입출력 계약을 고정한다.
 *
 *   interface LLMAdapter {
 *     // 고객 발화 한 줄에 대해 "행동(action) 목록"을 반환한다.
 *     // 서버는 이 행동들을 순서대로 소비하며 상담사 UI에 스트리밍한다.
 *     async plan({ text, session }): Promise<Action[]>
 *   }
 *
 * Action 종류:
 *   { kind:"ai_step", phase, label, detail }   // 판단 노드 1개 (감지→단서→본인확인→도구)
 *   { kind:"say", text }                       // AI 음성 응답 (전사로도 표시)
 *   { kind:"tool", tool, request }             // 도구 실행 요청 (서버가 ToolRunner로 처리)
 *
 * 실제 Haan 모델은 오디오/텍스트를 받아 동일한 Action 스트림을 산출하면 그대로 붙는다.
 */

const seed = require("../seed");

let _seq = 0;
const nid = (p) => `${p}-${++_seq}`;

/** 발화에서 본인확인 3요소를 best-effort 로 추출. */
function parseIdentity(text) {
  const out = {};
  // 이름: 시드에 있는 알려진 고객명 우선 매칭 (데모)
  for (const c of seed.CUSTOMERS) {
    if (text.includes(c.name)) {
      out.name = c.name;
      break;
    }
  }
  // 생년월일: 6자리(YYMMDD) 또는 8자리(YYYYMMDD), 또는 "90년 3월 12일"류
  const ymd = text.match(/(\d{2,4})\s*[년.\-/]\s*(\d{1,2})\s*[월.\-/]\s*(\d{1,2})/);
  if (ymd) {
    const yy = ymd[1].length === 4 ? ymd[1].slice(2) : ymd[1].padStart(2, "0");
    out.birth = yy + ymd[2].padStart(2, "0") + ymd[3].padStart(2, "0");
  } else {
    const six = text.match(/\b(\d{6}|\d{8})\b/);
    if (six) out.birth = six[1].length === 8 ? six[1].slice(2) : six[1];
  }
  // 전화 끝자리 4: "끝자리/뒷자리 4821" 우선, 없으면 마지막 4자리 숫자
  const tail = text.match(/(?:끝자리|뒷자리|마지막)\D*(\d{4})/);
  if (tail) out.phone_last4 = tail[1];
  else {
    const four = text.match(/(\d{4})(?!.*\d)/); // 마지막 4자리 묶음
    if (four && !ymd) out.phone_last4 = four[1];
  }
  return out;
}

/** 발화에서 의도 분류. */
function detectIntent(text) {
  if (/(분실\s*신고|신고\s*접수|분실\s*접수)/.test(text)) return "report_lost";
  if (/(정지|막아|중지|멈춰|블록)/.test(text)) return "suspend";
  if (/(잃어|분실|잃어버|없어졌|도난|도둑)/.test(text)) return "lost";
  return null;
}

/** 발화에서 대상 카드 특정 (브랜드/끝자리 키워드 → 지시어 → 기본값). */
function pickCard(text, customer, session) {
  if (!customer) return null;
  // 1) 브랜드/상품명 명시
  const brand = customer.cards.find(
    (c) => text.includes(c.brand) || text.includes(c.product)
  );
  if (brand) return brand;
  // 2) 끝자리 명시
  const byLast4 = customer.cards.find((c) => text.includes(c.last4));
  if (byLast4) return byLast4;
  // 3) 지시어("그 카드/그거/방금/해당") → 직전에 다룬 카드
  if (session && session.lastCard && /(그\s*카드|그거|방금|해당|같은)/.test(text)) {
    const prev = customer.cards.find((c) => c.card_id === session.lastCard);
    if (prev) return prev;
  }
  // 4) 기본값: 활성 카드 우선, 없으면 첫 카드
  const active = customer.cards.filter((c) => c.status === "active");
  return active[0] || customer.cards[0] || null;
}

class MockLLM {
  /**
   * @param {{text:string, session:object}} input
   * @returns {Promise<Array>} action 목록
   */
  async plan({ text, session }) {
    const actions = [];
    const step = (phase, label, detail) =>
      actions.push({ kind: "ai_step", id: nid("step"), phase, label, detail });
    const say = (t) => actions.push({ kind: "say", text: t });
    const tool = (name, request) =>
      actions.push({ kind: "tool", id: nid("call"), tool: name, request });

    const intent = detectIntent(text);
    const ident = parseIdentity(text);
    const hasIdent = ident.name || ident.birth || ident.phone_last4;

    // ── 1) 아직 본인확인 전 ──────────────────────────────
    if (!session.unlocked) {
      // (a) 본인확인 정보가 들어왔다 → 검증 시도
      if (hasIdent && (session.pendingIntent || session.askedVerify)) {
        step("detect", "본인확인 정보 수신", `이름·생년월일·연락처 단서 감지`);
        step("verify", "본인확인 진행", "카드사 인증 서버에 3요소 대조 요청");
        tool("verify_identity", {
          name: ident.name,
          birth: ident.birth,
          phone_last4: ident.phone_last4,
        });
        // 성공 후 후속 처리는 서버가 verify 결과를 보고 재-plan (아래 continueAfterVerify)
        return actions;
      }

      // (b) 어떤 의도가 감지됐다 → 본인확인부터 요구
      if (intent) {
        const intentLabel =
          intent === "suspend" ? "카드 정지 요청" : "카드 분실 관련 문의";
        step("detect", intentLabel, `발화에서 '${intent}' 의도 감지`);
        step("clue", "개인정보 접근 필요", "카드/계정 조회가 필요한 요청으로 분류");
        step("verify", "본인확인 선행 필요", "미인증 상태 → 개인정보 접근 잠금");
        session.pendingIntent = intent;
        session.askedVerify = true;
        say(
          "네, 도와드리겠습니다. 먼저 본인확인이 필요해요. 성함과 생년월일, 전화번호 끝 4자리를 말씀해 주시겠어요?"
        );
        return actions;
      }

      // (c) 일반 발화
      step("detect", "발화 수신", "의도 분류 대기");
      session.askedVerify = true;
      say("안녕하세요, 무엇을 도와드릴까요? 본인확인부터 진행하겠습니다.");
      return actions;
    }

    // ── 2) 본인확인 완료 (개인정보 잠금 해제됨) ──────────────
    const customer = session.customer;
    const effIntent = intent || session.pendingIntent;

    if (effIntent === "suspend") {
      const card = pickCard(text, customer, session);
      step("detect", "카드 정지 요청", "본인확인 완료 상태에서 정지 실행 가능");
      step("clue", "대상 카드 특정", card ? `${card.product} (••${card.last4})` : "카드 미상");
      if (card) {
        session.lastCard = card.card_id;
        tool("suspend_card", { customer_id: customer.customer_id, card_id: card.card_id });
        session.pendingIntent = null;
      } else {
        say("정지할 카드를 특정하지 못했어요. 카드사나 끝 4자리를 알려주시겠어요?");
      }
      return actions;
    }

    if (effIntent === "lost" || effIntent === "report_lost") {
      const card = pickCard(text, customer, session);
      step("detect", "분실 신고 요청", "본인확인 완료 → 분실신고 접수 가능");
      step("clue", "대상 카드 특정", card ? `${card.product} (••${card.last4})` : "카드 미상");
      if (card) {
        session.lastCard = card.card_id;
        // 분실은 통상 정지 후 신고 — 데모에서는 신고 접수만 명시 실행
        tool("report_lost", {
          customer_id: customer.customer_id,
          card_id: card.card_id,
          memo: "고객 통화 중 분실 신고 접수",
        });
        session.pendingIntent = null;
      } else {
        say("분실 신고할 카드를 알려주시겠어요?");
      }
      return actions;
    }

    // 종료/감사 인사
    if (/(감사|고마워|괜찮|됐어|없어요|아니요|수고)/.test(text)) {
      say("네, 처리 완료되었습니다. 도움이 필요하시면 언제든 다시 말씀해 주세요. 감사합니다.");
      return actions;
    }

    // 기타: 개인정보 조회형 응답
    step("detect", "일반 조회 요청", "본인확인 완료 상태 — 개인정보 접근 허용");
    say("확인해 드리겠습니다. 어떤 카드에 대해 도와드릴까요?");
    return actions;
  }

  /**
   * verify_identity 성공 직후의 후속 발화 계획.
   * 서버가 검증 성공을 감지하면 호출한다. (본인확인 완료 → 개인정보 열림 안내)
   */
  async continueAfterVerify({ session }) {
    const actions = [];
    const step = (phase, label, detail) =>
      actions.push({ kind: "ai_step", id: nid("step"), phase, label, detail });
    const c = session.customer;
    step("verify", "본인확인 완료", `${c.name}님 확인됨 · 개인정보 접근 잠금 해제`);
    step("tool", "카드 정보 조회", `보유 카드 ${c.cards.length}건 로드`);
    actions.push({
      kind: "say",
      text: `${c.name}님, 본인확인이 완료되었습니다. 이제 카드 관련 요청을 바로 처리해 드릴 수 있어요.`,
    });
    return actions;
  }
}

module.exports = { MockLLM, parseIdentity, detectIntent };
