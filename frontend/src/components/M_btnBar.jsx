import React from 'react';
import { useNavigate, useLocation } from 'react-router-dom';

const M_btnBar = ({ previewData }) => { // previewData prop 추가
  const navigate = useNavigate();
  const location = useLocation();

  const handleNavigation = (path) => {
    navigate(path);
  };

  const handleDownload = () => {
    if (previewData && previewData.length > 0) {
      navigate('/download', { state: { previewData } });
    } else {
      navigate('/download');
    }
  };

  // 현재 경로에 따라 `active` 클래스를 동적으로 적용
  return (
    <div className="btn_bar_container">
      <div className="btn_bar">
        {/* Home Button */}
        <div
          className={`btn_item ${location.pathname === '/M_calender' ? 'active' : ''}`}
          onClick={() => handleNavigation('/M_calender')}
          style={{ cursor: 'pointer' }}
        >
          <img src={`${process.env.PUBLIC_URL}/images/house.png`} alt="Home Icon" className="btn_icon" />
          <p>HOME</p>
        </div>
        {/* 등록차량 Button */}
        <div
          className={`btn_item ${location.pathname === '/register-car' ? 'active' : ''}`}
          onClick={() => handleNavigation('/register-car')}
          style={{ cursor: 'pointer' }}
        >
          <img src={`${process.env.PUBLIC_URL}/images/car.png`} alt="Car Icon" className="btn_icon" />
          <p>등록차량</p>
        </div>
        {/* 위반차량정보 Button */}
        <div
          className={`btn_item ${location.pathname === '/Violations' ? 'active' : ''}`}
          onClick={() => handleNavigation('/Violations')}
          style={{ cursor: 'pointer' }}
        >
          <img src={`${process.env.PUBLIC_URL}/images/violations.png`} alt="Violations Icon" className="btn_icon" />
          <p>위반정보</p>
        </div>
        {/* 다운로드 Button */}
        <div
          className={`btn_item ${location.pathname === '/download' ? 'active' : ''}`}
          onClick={handleDownload}
          style={{ cursor: 'pointer' }}
        >
          <img src={`${process.env.PUBLIC_URL}/images/download.png`} alt="Download Icon" className="btn_icon" />
          <p>다운로드</p>
        </div>
        {/* 설정 Button */}
        <div
          className={`btn_item ${location.pathname === '/setting' ? 'active' : ''}`}
          onClick={() => handleNavigation('/setting')}
          style={{ cursor: 'pointer' }}
        >
          <img src={`${process.env.PUBLIC_URL}/images/setting.png`} alt="Settings Icon" className="btn_icon" />
          <p>설정</p>
        </div>
      </div>
    </div>
  );
};

export default M_btnBar;
