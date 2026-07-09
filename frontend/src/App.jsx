import React from "react";
import LogInteractionScreen from "./components/LogInteractionScreen.jsx";

export default function App() {
  return (
    <div className="app-shell">
      <div className="topbar">
        <div className="brand">
          <span className="mark">Rx</span>
          HCP CRM
        </div>
        <div className="rep">Field Rep · Aisha Verma</div>
      </div>
      <LogInteractionScreen />
    </div>
  );
}
