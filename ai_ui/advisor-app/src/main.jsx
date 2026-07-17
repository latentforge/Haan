import React from "react";
import ReactDOM from "react-dom/client";
import App from "./App.jsx";

// 순서 중요: Tailwind base 먼저, 그 다음 원본 이식 디자인(advisor.css)이 이겨야 함.
import "./index.css";
import "./styles/advisor.css";

ReactDOM.createRoot(document.getElementById("root")).render(
  <React.StrictMode>
    <App />
  </React.StrictMode>
);
