import React, { useState } from 'react';
import { useNavigate, useLocation } from 'react-router-dom';
import axios from 'axios';

const Join = () => {
  const { state } = useLocation();
  const isUserMode = state?.mode === 'user'; // 사용자 모드인지 확인
  const navigate = useNavigate(); // useNavigate 훅 사용

  const [userData, setUserData] = useState({
    id: '',
    pw: '',
    name: '',
    phone: '',
    email: '',
    carNumber: '', // 차량 번호 또는 관리자 코드
  });

  const handleChange = (e) => {
    const { name, value } = e.target;
    setUserData({
      ...userData,
      [name]: value,
    });
  };

  const handleSubmit = async (e) => {
    e.preventDefault();
    try {
      const dataToSend = isUserMode ? userData : { ...userData, adminCode: userData.carNumber };
      const response = await axios.post('http://localhost:4000/user/join', dataToSend);
      console.log(response.data);
      navigate('/login'); // navigate 사용
    } catch (error) {
      console.error('Error during registration', error);
    }
  };

  return (
    <div className="join-wrapper">
      <form onSubmit={handleSubmit}>
        {/* 폼 요소들 */}
        <div className="form-group">
          <label>아이디</label>
          <input type="text" name="id" placeholder="아이디를 입력하세요" value={userData.id} onChange={handleChange} required />
        </div>

        <div className="form-group">
          <label>비밀번호</label>
          <input type="password" name="pw" placeholder="비밀번호를 입력하세요" value={userData.pw} onChange={handleChange} required />
        </div>

        <div className="form-group">
          <label>비밀번호 확인</label>
          <input type="password" placeholder="비밀번호를 다시 입력하세요" required />
        </div>

        <div className="form-group">
          <label>이름</label>
          <input type="text" name="name" placeholder="이름을 입력하세요" value={userData.name} onChange={handleChange} required />
        </div>

        <div className="form-group">
          <label>전화번호</label>
          <input type="tel" name="phone" placeholder="전화번호를 입력하세요" value={userData.phone} onChange={handleChange} required />
        </div>

        <div className="form-group">
          <label>이메일 주소</label>
          <input type="email" name="email" placeholder="이메일을 입력하세요" value={userData.email} onChange={handleChange} required />
        </div>

        {isUserMode ? (
          <div className="form-group">
            <label>차량 번호</label>
            <input type="text" name="carNumber" placeholder="차량 번호를 입력하세요" value={userData.carNumber} onChange={handleChange} />
          </div>
        ) : (
          <div className="form-group">
            <label>관리자 코드</label>
            <input type="text" name="carNumber" placeholder="관리자 인증키를 입력하세요" value={userData.carNumber} onChange={handleChange} />
          </div>
        )}

        <button type="submit" className="join-button">회원가입</button>
      </form>
    </div>
  );
};

export default Join;
