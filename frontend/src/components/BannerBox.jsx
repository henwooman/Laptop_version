import React from 'react';
import { useNavigate } from 'react-router-dom';

const BannerBox = ({ isAdminLoggedIn }) => {
  const navigate = useNavigate();

  const handleClick = () => {
    if (isAdminLoggedIn) {
      navigate('/M_calender'); // 관리자로 로그인된 경우 M_calender 페이지로 이동
    } else {
      navigate('/'); // 일반 사용자일 경우 메인 페이지로 이동
    }
  };

  return (
    <div className="BannerBox" onClick={handleClick} style={{ cursor: 'pointer' }}>
      <img src='/images/BN.png' alt='배너' className='banner_img' />
    </div>
  );
};

export default BannerBox;
