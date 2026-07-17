"use strict";
/**
 * Mock 도구(Tool) 3종 — 요구사항 §4 계약 구현.
 *
 * 각 도구는 실제 카드사 API 호출을 흉내내며, 인위적 지연(대기→실행중→완료)을 두어
 * 상담사 UI 오른쪽 패널에서 상태 전이가 라이브로 보이도록 한다.
 *
 * ⚠️ 실서비스 전환 시: 이 파일은 카드사 실 API 어댑터로 교체된다.
 * verify_identity 는 실제 인증(ARS/OTP/신분증)과 개인정보 동의 절차로 대체되어야 하며,
 * suspend_card / report_lost 는 실제 카드 원장 트랜잭션 + 감사 로그가 필요하다. (데모용 Mock)
 */
const seed = require("./seed");

const nowIso = () => new Date().toISOString();
const sleep = (ms) => new Promise((r) => setTimeout(r, ms));

/**
 * 도구 실행 컨텍스트.
 * @param {(evt:object)=>void} emit 상태 전이 이벤트를 브로드캐스트하는 콜백
 * @param {object} session 세션 상태(unlocked, customer 등)
 */
class ToolRunner {
  constructor(emit, session) {
    this.emit = emit;
    this.session = session;
  }

  /** 공통: tool_call 상태 전이 이벤트를 흘려보낸다. */
  _emitStatus(callId, tool, status, extra = {}) {
    this.emit({
      type: "tool_call",
      id: callId,
      tool,
      status, // running | done | error
      at: nowIso(),
      ...extra,
    });
  }

  /**
   * verify_identity (본인확인)
   * 요청 { name, birth, phone_last4 } → 응답 { verified, customer_id, unlocked }
   * verified=true 가 되면 이후 개인정보 조회/카드 도구가 열린다.
   */
  async verify_identity(callId, request) {
    this._emitStatus(callId, "verify_identity", "running", { request });
    await sleep(1100); // 인증 서버 왕복 흉내

    const customer = seed.findCustomer(request);
    if (!customer) {
      const response = { verified: false, reason: "일치하는 고객 정보 없음" };
      this._emitStatus(callId, "verify_identity", "error", { request, response });
      return response;
    }

    // 세션 잠금 해제 — 데모의 핵심: 본인확인 = 개인정보 접근 허용
    this.session.unlocked = true;
    this.session.customer = customer;

    const response = {
      verified: true,
      customer_id: customer.customer_id,
      unlocked: true,
    };
    this._emitStatus(callId, "verify_identity", "done", { request, response });

    // 개인정보 잠금 해제 이벤트(상담사 UI가 프로필/카드를 표시하도록)
    this.emit({
      type: "identity",
      unlocked: true,
      profile: seed.publicProfile(customer),
      at: nowIso(),
    });
    return response;
  }

  /**
   * suspend_card (카드정지)
   * 요청 { customer_id, card_id } → 응답 { status, card_id, at }
   */
  async suspend_card(callId, request) {
    this._emitStatus(callId, "suspend_card", "running", { request });
    await sleep(950);

    if (!this.session.unlocked) {
      const response = { status: "denied", reason: "본인확인 필요" };
      this._emitStatus(callId, "suspend_card", "error", { request, response });
      return response;
    }
    const customer = seed.getCustomer(request.customer_id) || this.session.customer;
    const card = customer && customer.cards.find((c) => c.card_id === request.card_id);
    if (!card) {
      const response = { status: "not_found", reason: "카드를 찾을 수 없음" };
      this._emitStatus(callId, "suspend_card", "error", { request, response });
      return response;
    }
    card.status = "suspended";
    const response = { status: "suspended", card_id: card.card_id, at: nowIso() };
    this._emitStatus(callId, "suspend_card", "done", { request, response });
    return response;
  }

  /**
   * report_lost (분실신고 접수)
   * 요청 { customer_id, card_id, memo } → 응답 { report_id, status }
   */
  async report_lost(callId, request) {
    this._emitStatus(callId, "report_lost", "running", { request });
    await sleep(1000);

    if (!this.session.unlocked) {
      const response = { status: "denied", reason: "본인확인 필요" };
      this._emitStatus(callId, "report_lost", "error", { request, response });
      return response;
    }
    const customer = seed.getCustomer(request.customer_id) || this.session.customer;
    const card = customer && customer.cards.find((c) => c.card_id === request.card_id);
    if (card) card.status = "lost";

    // 데모: 리포트 ID 고정 (요구사항 예시 MOCK-RPT-2001)
    const response = { report_id: "MOCK-RPT-2001", status: "received" };
    this._emitStatus(callId, "report_lost", "done", { request, response });
    return response;
  }
}

module.exports = { ToolRunner };
