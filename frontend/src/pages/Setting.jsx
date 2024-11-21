import React, { useState, useEffect } from 'react';

const Setting = () => {
  const [settings, setSettings] = useState({
    setting1: false, // 자동 삭제
    setting2: false, // 로그 기록 저장
    setting4: false, // 화면 모드
    setting5: false, // 텍스트 크기 조정
  });

  const toggleSetting = (setting) => {
    setSettings({ ...settings, [setting]: !settings[setting] });
  };

  const handleLogout = () => {
    // 로그아웃 로직
    localStorage.removeItem('token'); // 예시: 토큰 제거
    window.location.href = '/login';  // 로그인 페이지로 리다이렉트
  };

  // 1. 자동 삭제 기능 (예제)
  useEffect(() => {
    if (settings.setting1) {
      const timer = setTimeout(() => {
        console.log("기록이 자동 삭제되었습니다.");
      }, 30 * 24 * 60 * 60 * 1000);

      return () => clearTimeout(timer);
    }
  }, [settings.setting1]);

  // 2. 로그 기록 저장 기능
  useEffect(() => {
    if (settings.setting2) {
      console.log("로그 기록이 저장됩니다.");
    }
  }, [settings.setting2]);

  // 3. 화면 모드 기능
  useEffect(() => {
    if (settings.setting4) {
      document.body.classList.add("dark-mode");
    } else {
      document.body.classList.remove("dark-mode");
    }
  }, [settings.setting4]);

  // 4. 텍스트 크기 조정 기능
  const textSizeClass = settings.setting5 ? "large-text" : "normal-text";

  return (
    <div className={`setting-page-container-unique ${textSizeClass}`}>
      <div className="setting-section-unique">
        <h3 className="setting-section-title-unique">기록 관리</h3>

        <div className="setting-item-unique">
          <div className="setting-text-container">
            <p className="setting-title">자동 삭제</p>
            <p className="setting-description">ex:1개월 후 자동 삭제</p>
          </div>
          <label className="toggle-switch-unique">
            <input
              type="checkbox"
              checked={settings.setting1}
              onChange={() => toggleSetting('setting1')}
            />
            <span className="slider-unique"></span>
          </label>
        </div>

        <hr className="divider-unique" />

        <div className="setting-item-unique">
          <div className="setting-text-container">
            <p className="setting-title">로그 기록 저장</p>
            <p className="setting-description">위반 기록 및 사용 로그를 저장할지 선택</p>
          </div>
          <label className="toggle-switch-unique">
            <input
              type="checkbox"
              checked={settings.setting2}
              onChange={() => toggleSetting('setting2')}
            />
            <span className="slider-unique"></span>
          </label>
        </div>
      </div>

      <hr className="section-divider-unique" />

      <div className="setting-section-unique">
        <h3 className="setting-section-title-unique">디스플레이 설정</h3>

        <div className="setting-item-unique">
          <div className="setting-text-container">
            <p className="setting-title">화면 모드</p>
            <p className="setting-description">다크모드 or 화이트모드</p>
          </div>
          <label className="toggle-switch-unique">
            <input
              type="checkbox"
              checked={settings.setting4}
              onChange={() => toggleSetting('setting4')}
            />
            <span className="slider-unique"></span>
          </label>
        </div>

        <hr className="divider-unique" />

        <div className="setting-item-unique">
          <div className="setting-text-container">
            <p className="setting-title">텍스트 크기 조정</p>
            <p className="setting-description">작게 or 크게</p>
          </div>
          <label className="toggle-switch-unique">
            <input
              type="checkbox"
              checked={settings.setting5}
              onChange={() => toggleSetting('setting5')}
            />
            <span className="slider-unique"></span>
          </label>
        </div>
      </div>
      <div className="logout-section">
        <button onClick={handleLogout} className="logout-button">로그아웃</button>
        
      </div>
    </div>
  );
};

export default Setting;
