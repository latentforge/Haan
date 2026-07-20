import ReactDOM from "react-dom/client";
import App from "./App.jsx";

// 순서 중요: Tailwind base 먼저, 그 다음 원본 이식 디자인(advisor.css)이 이겨야 함.
import "./index.css";
import "./styles/advisor.css";

// StrictMode 미사용: dev 모드의 effect 이중 실행이 WebSocketEventSource를
// 두 번 연결시켜 히스토리 리플레이(인사말 등)가 중복 수신되는 문제가 있었음.
ReactDOM.createRoot(document.getElementById("root")).render(<App />);
