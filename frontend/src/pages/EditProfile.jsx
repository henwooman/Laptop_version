import React, { useState } from 'react';
import { useEffect } from 'react';
import { useNavigate } from 'react-router-dom';
import axios from 'axios';

const EditProfile = () => {
  const navigate = useNavigate();
  const [userInfo, setUserInfo] = useState({
    name: '',
    carNumber: '',
    phone: '',
    email: '',
  });
  const [password, setPassword] = useState('');
  const [confirmPassword, setConfirmPassword] = useState('');
  const [errorMessage, setErrorMessage] = useState('');
  const [updateMessage, setUpdateMessage] = useState('');


  useEffect(() => {
    // 사용자 정보 불러오기
    axios.get('http://localhost:4000/user/profile', { withCredentials: true })
      .then((response) => {
        if (response.status === 200) {
          setUserInfo({
            name: response.data.user_name,
            carNumber: response.data.car_number,
            phone: response.data.user_phone,
            email: response.data.user_email,
          });
        }
      })
      .catch((error) => {
        console.error('사용자 정보 불러오기 오류:', error);
      });
  }, []);

  const handlePasswordChange = (e) => {
    setPassword(e.target.value);
  };

  const handleConfirmPasswordChange = (e) => {
    setConfirmPassword(e.target.value);
    if (e.target.value !== password) {
      setErrorMessage('비밀번호가 일치하지 않습니다.');
    } else {
      setErrorMessage('');
    }
  };

  const handleSubmit = async (e) => {
    e.preventDefault();
    // 수정 처리 로직 추가
    if (errorMessage) {
      alert('비밀번호가 일치하지 않습니다.');
      return;
    }

    if (!password || !confirmPassword) {
      alert('비밀번호를 입력해주세요.');
      return;
    }

    try {
      // 서버로 수정 요청을 보냄
      const response = await axios.post(
        'http://localhost:4000/user/update',
        {
          currentPassword: password, // 사용자가 입력한 현재 비밀번호
          name: userInfo.name,
          carNumber: userInfo.carNumber,
          phone: userInfo.phone,
          email: userInfo.email,
        },
        { withCredentials: true }
      );

      if (response.status === 200) {
        setUpdateMessage('회원정보가 성공적으로 수정되었습니다!');
        setTimeout(() => {
          navigate('/userMain'); // 사용자 메인 페이지로 이동
        }, 1000); // 1초 후에 페이지 이동
      }
    } catch (error) {
      if (error.response && error.response.data.error) {
        setUpdateMessage(error.response.data.error);
      } else {
        setUpdateMessage('회원정보 수정 중 오류가 발생했습니다.');
      }
    }
  };

  return (
    <div className="edit-profile-container">
      <form onSubmit={handleSubmit}>
        <div className="edit-profile-label">이름</div>
        <input type="text" placeholder="이름을 입력해 주세요" className="edit-profile-input-large" value={userInfo.name} onChange={(e) => setUserInfo({ ...userInfo, name: e.target.value })} required />

        <div className="edit-profile-label">비밀번호
          <span className="edit-profile-error-text">20자 이내의 비밀번호를 입력해주세요.</span>
        </div>
        <div className="edit-profile-password-container">
          <input
            type="password"
            placeholder="비밀번호 입력(문자,숫자,특수문자 포함 8-20자)"
            className="edit-profile-input-large"
            value={password}
            onChange={handlePasswordChange}
            required
          />
        </div>

        <div className="edit-profile-label">비밀번호 확인

          <span className="edit-profile-error-text">{errorMessage}</span>
        </div>
        <input type="password" placeholder="비밀번호 재입력" className="edit-profile-input-large" required value={confirmPassword} onChange={handleConfirmPasswordChange} />

        <div className="edit-profile-label">차량번호</div>
        <input type="text" placeholder="차량 번호를 입력하세요." className="edit-profile-input-large" value={userInfo.carNumber} onChange={(e) => setUserInfo({ ...userInfo, carNumber: e.target.value })} required />

        <div className="edit-profile-label">전화번호</div>
        <input
          type="tel"
          placeholder="휴대폰 번호 입력(-제외 11자리 입력)"
          className="edit-profile-input-large"
          value={userInfo.phone}
          onChange={(e) => setUserInfo({ ...userInfo, phone: e.target.value })}
          required
        />

        <div className="edit-profile-label">이메일 주소</div>
        <div className="edit-profile-email-container">
          <input
            type="text"
            placeholder="이메일 주소"
            className="edit-profile-input-small"
            value={userInfo.email.split('@')[0]} // '@' 이전 부분
            onChange={(e) =>
              setUserInfo({ ...userInfo, email: `${e.target.value}@${userInfo.email.split('@')[1]}` })
            }
            required
          />
          <span className="edit-profile-email-icon">@</span>
          <input
            type="text"
            placeholder="gmail.com"
            className="edit-profile-input-small"
            value={userInfo.email.split('@')[1]} // '@' 이후 부분
            onChange={(e) =>
              setUserInfo({ ...userInfo, email: `${userInfo.email.split('@')[0]}@${e.target.value}` })
            }
            required
          />
        </div>

        <button type="submit" className="edit-profile-submit-button">수정하기</button>
      </form>
      {updateMessage && <p className="update-message">{updateMessage}</p>}
    </div>
  );
};

export default EditProfile;
