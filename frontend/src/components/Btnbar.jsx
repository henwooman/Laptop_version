import React from 'react';
import { useNavigate, useLocation } from 'react-router-dom';

const Btnbar = () => {
  const navigate = useNavigate();
  const location = useLocation();

  const handleNavigation = (path) => {
    navigate(path);
  };

  return (
    <div className="bottom-bar">
      <div
        className={`btn_item ${location.pathname === '/userMain' ? 'active' : ''}`}
        onClick={() => handleNavigation('/userMain')}
        style={{ cursor: 'pointer' }}
      >
        <img src="/images/house.png" alt="사용자메인" className="btn_icon" />
        <p>HOME</p>
      </div>
      <div
        className={`btn_item ${location.pathname === '/parkingSearch' ? 'active' : ''}`}
        onClick={() => handleNavigation('/parkingSearch')}
        style={{ cursor: 'pointer' }}
      >
        <img src="/images/search.png" alt="주차장찾기" className="btn_icon" />
        <p>주차장 찾기</p>
      </div>
      <div
        className={`btn_item ${location.pathname === '/parkinglist' ? 'active' : ''}`}
        onClick={() => handleNavigation('/parkinglist')}
        style={{ cursor: 'pointer' }}
      >
        <img src="/images/star_.png" alt="즐겨찾기" className="btn_icon" />
        <p>즐겨찾기</p>
      </div>
      <div
        className={`btn_item ${location.pathname === '/setting' ? 'active' : ''}`}
        onClick={() => handleNavigation('/setting')}
        style={{ cursor: 'pointer' }}
      >
        <img src="/images/setting.png" alt="설정" className="btn_icon" />
        <p>설정</p>
      </div>
    </div>
  );
};

export default Btnbar;
