"use strict";
/**
 * 가짜(Mock) 카드사 고객/카드 데이터.
 *
 * ⚠️ 실서비스 전환 시: 이 시드는 카드사 코어뱅킹/CRM API 조회 결과로 대체된다.
 * 개인정보(주소, 등급 등)는 본인확인(verify_identity) 성공 후에만 노출되어야 하며,
 * 실제로는 암호화 저장 + 접근 감사 로그 + 개인정보 이용 동의가 필요하다. (데모용 평문)
 *
 * 요구사항 §4 예시 준수: 김민서 / 전화 끝자리 4821 / 카드 2개.
 */

const CUSTOMERS = [
  {
    customer_id: "MOCK-1001",
    name: "김민서",
    birth: "900312", // YYMMDD (1990-03-12)
    phone_last4: "4821",
    // 아래는 본인확인 성공 시에만 열리는 민감 정보
    phone: "010-****-4821",
    address: "서울특별시 마포구 양화로 45, 8층",
    grade: "VIP",
    joined: "2016-08-21",
    cards: [
      {
        card_id: "CARD-8842",
        brand: "신한",
        product: "신한 Deep Dream 신용카드",
        type: "신용",
        last4: "8842",
        status: "active", // active | suspended | lost
      },
      {
        card_id: "CARD-2290",
        brand: "삼성",
        product: "삼성 taptap O 체크카드",
        type: "체크",
        last4: "2290",
        status: "active",
      },
    ],
  },
];

/** 본인확인 3요소로 고객을 찾는다. (데모: 정확 일치) */
function findCustomer({ name, birth, phone_last4 }) {
  return (
    CUSTOMERS.find(
      (c) =>
        (!name || c.name === name) &&
        (!birth || normalizeBirth(c.birth) === normalizeBirth(birth)) &&
        (!phone_last4 || c.phone_last4 === String(phone_last4))
    ) || null
  );
}

function getCustomer(customerId) {
  return CUSTOMERS.find((c) => c.customer_id === customerId) || null;
}

/** "1990-03-12", "900312", "90.03.12" 등을 YYMMDD 6자리로 정규화. */
function normalizeBirth(b) {
  if (!b) return "";
  const digits = String(b).replace(/\D/g, "");
  if (digits.length === 8) return digits.slice(2); // YYYYMMDD -> YYMMDD
  return digits;
}

/** 본인확인 성공 후 상담사/AI에게 공개할 안전한 프로필 뷰. */
function publicProfile(customer) {
  return {
    customer_id: customer.customer_id,
    name: customer.name,
    grade: customer.grade,
    phone: customer.phone,
    address: customer.address,
    joined: customer.joined,
    cards: customer.cards.map((c) => ({ ...c })),
  };
}

module.exports = {
  CUSTOMERS,
  findCustomer,
  getCustomer,
  normalizeBirth,
  publicProfile,
};
