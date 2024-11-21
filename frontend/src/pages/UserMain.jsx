import React from 'react'
import { useNavigate } from 'react-router-dom';
import axios from 'axios';
import { useState, useEffect } from 'react';

export const UserMain = () => {
  const navigate = useNavigate();
  const [userName, setUserName] = useState('');  // 사용자 이름 상태
  const [carNumber, setCarNumber] = useState('');  // 차량 번호 상태

  // 사용자 정보 불러오기
  useEffect(() => {
    const fetchUserInfo = async () => {
      try {
        const response = await axios.get('http://localhost:4000/user/userinfo', {
          withCredentials: true, // 세션 쿠키 포함
        });
        if (response.status === 200) {
          setUserName(response.data.user_name);  // 사용자 이름 설정
          setCarNumber(response.data.car_number);  // 차량 번호 설정
        }
      } catch (error) {
        if (error.response && error.response.status === 401) {
          alert('로그인이 필요합니다.');
          navigate('/login'); // 로그인 페이지로 이동
        } else {
          console.error('사용자 정보를 가져오는 중 오류 발생:', error);
        }
      }
    };

    fetchUserInfo();
  }, [navigate]);



  const clean = () => {
    navigate('/editprofile');
  };

  const goToParkingSearch = () => {
    navigate('/parkingSearch');
  };

  const goToParkingList = () => {
    navigate('/parkinglist');
  };

  return (
    <div className='user_container'>
      <div className='user_info_box'>
        <div className='photo'>
          <img src='/images/avatar.png' alt='프사' />
        </div>
        <div className='user_info'>
          <p>이름 : <span style={{ fontSize: '18px', fontWeight: 'bold' }}>{userName}</span></p>
          <p>번호 : {carNumber}</p>
          <button className='members_edit_btn' onClick={clean}>회원정보수정</button>
        </div>
      </div>
      {/* 카드 컨테이너 추가 */}
      <div className='card_container'>
        <div className='parking_card' onClick={goToParkingSearch}>
          <img className='p_icon' alt='돋보기' src='/images/P.png' />
          <p className='card_text'>주차장 조회하기</p>
        </div>
        <div className='parking_card' onClick={goToParkingList}>
          <img className='parking_icon' alt='주차장' src='/images/parking.png' />
          <p className='card_text'>나의 주차장</p>
        </div>
      </div>
      {/* 카드 컨테이너 끝 */}
    </div>
  )
}
