/** @type {import('tailwindcss').Config} */
// Tailwind은 스택 요구사항으로 포함하지만, 디자인 원천은 advisor.css(원본 이식)입니다.
// 새 색/폰트를 창작하지 않기 위해(프롬프트 §5.5) 기존 oklch 토큰을 그대로 노출만 합니다.
export default {
  content: ["./index.html", "./src/**/*.{js,jsx}"],
  theme: {
    extend: {
      colors: {
        bg: "var(--bg)",
        surface: "var(--surface)",
        fg: "var(--fg)",
        muted: "var(--muted)",
        border: "var(--border)",
        accent: "var(--accent)",
        primary: "var(--primary)",
        success: "var(--success)",
        warn: "var(--warn)",
        danger: "var(--danger)",
      },
      fontFamily: {
        display: "var(--font-display)",
        body: "var(--font-body)",
        mono: "var(--font-mono)",
      },
    },
  },
  plugins: [],
};
