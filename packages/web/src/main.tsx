import { createRoot } from "react-dom/client";
import { App } from "./App.js";
import "./index.css";

const root = document.getElementById("root");
if (!root) {
  throw new Error("missing application root");
}

createRoot(root).render(<App />);
