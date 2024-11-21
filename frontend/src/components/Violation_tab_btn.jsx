// src/components/Violation_tab_btn.js

import React from 'react';

const Violation_tab_btn = ({ isActive, onClick, label }) => {
  return (
    <button
      className={`violation-tab-btn ${isActive ? 'active' : ''}`}
      onClick={onClick}
    >
      {label}
    </button>
  );
};

export default Violation_tab_btn;
