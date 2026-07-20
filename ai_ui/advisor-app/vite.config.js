import { defineConfig } from "vite";
import react from "@vitejs/plugin-react";

// 상담사 UI 데모. `npm run dev`로 로컬 실행.
// WebSocket 목업 서버 주소는 .env 의 VITE_WS_URL 로 주입 (없으면 내장 Mock 재생).
export default defineConfig({
  plugins: [react()],
  server: {
    host: true,
    port: 5174,
  },
});
