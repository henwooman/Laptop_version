import React, { useEffect, useState } from 'react';
import { useNavigate } from 'react-router-dom';
import axios from 'axios';

const U_login = ({ setIsAdmin }) => {
    const [isUser, setIsUser] = useState(true);  // 기본적으로 사용자 버튼이 활성화됨
    const [username, setUsername] = useState(''); // 아이디 입력 상태
    const [password, setPassword] = useState(''); // 비밀번호 입력 상태
    const navigate = useNavigate();
    const [loginData, setLoginData] = useState({
        id: '',
        pw: ''
    });
    const [isInputFilled, setIsInputFilled] = useState(false);

    const handleChange = (e) => {
        const { id, value } = e.target;
        setLoginData({
            ...loginData,
            [id]: value
        });
    };
    // 입력 필드 상태 확인
    useEffect(() => {
        setIsInputFilled(loginData.id !== '' && loginData.pw !== '');
    }, [loginData]);

    // 사용자와 관리자 탭 전환 핸들링
    const handleTabSwitch = (type) => {
        setIsUser(type === 'user');
    };

    // 로그인 클릭 시 처리
    const handleLoginClick = async () => {

        const endpoint = isUser
            ? 'http://localhost:4000/user/login'  // 사용자 로그인 엔드포인트
            : 'http://localhost:4000/user/admin-login'; // 관리자 로그인 엔드포인트

        try {
            const response = await axios.post(endpoint, {
                username: loginData.id,  // 수정: id를 username으로 매핑
                password: loginData.pw   // 수정: pw를 password로 매핑
            }, { withCredentials: true }); // 세션 쿠키를 포함하여 요청);
            if (response.status === 200) {
                alert('로그인 성공!');
                if (isUser) {
                    setIsAdmin(false); // 사용자로 설정
                    console.log("Setting isAdmin to false");

                    // 사용자 로그인 성공 시 로컬 스토리지에 userId 저장
                    sessionStorage.setItem("id", loginData.id);
                    console.log("Navigating to userMain page");
                    navigate('/userMain'); // 사용자 로그인 성공 시 이동 경로
                } else {
                    setIsAdmin(true); // 관리자 모드로 설정
                    console.log("Setting isAdmin to true");

                    sessionStorage.setItem("id", loginData.id);
                    console.log("Navigating to M_calender page");
                    navigate('/M_calender'); // 관리자 로그인 성공 시 이동 경로
                }
            }
        } catch (error) {
            if (error.response && error.response.status === 401) {
                alert('아이디 또는 비밀번호가 잘못되었습니다.');
            } else {
                alert('로그인 중 오류가 발생했습니다. 다시 시도해주세요.');
            }
        }
    };

    const handleRegisterClick = () => {
        navigate('/join', { state: { mode: isUser ? 'user' : 'admin' } });
    };

    useEffect(() => {
        window.sessionStorage.setItem("login", loginData);
        const a = sessionStorage.getItem('id')
        console.log("sessionStorage",a);
        
    },[loginData]);

    return (
        <div className="login_wrapper">
            <h2 className="greeting">반갑습니다!<br />서비스 이용을 위한 <br /> 본인 인증이 필요합니다.</h2>
            <div className="divider_line"></div>
            <div className="login_container">
                <div className="login_tabs">
                    <button
                        className={isUser ? 'active' : ''}
                        onClick={() => handleTabSwitch('user')}
                    >
                        사용자
                    </button>
                    <button
                        className={!isUser ? 'active' : ''}
                        onClick={() => handleTabSwitch('admin')}
                    >
                        관리자
                    </button>
                </div>
                <div className="login_form">
                    <input
                        type="text"
                        id="id"
                        value={loginData.id}
                        onChange={handleChange}
                        placeholder={isUser ? "아이디를 입력하세요." : "관리자 아이디를 입력하세요."}
                    />
                    <input
                        type="password"
                        id="pw"
                        value={loginData.pw}
                        onChange={handleChange}
                        placeholder={isUser ? "비밀번호를 입력하세요." : "관리자 비밀번호를 입력하세요."}
                    />
                    <button 
                            className={`login_button ${isInputFilled ? 'filled' : ''}`} 
                            onClick={handleLoginClick}
                            disabled={!isInputFilled} // 입력이 안 되면 클릭 불가
                        >로그인</button>

                    <div className="login_options">
                        <button className="find_credentials_button">아이디/비밀번호 찾기</button>
                        <button className="register_button" onClick={handleRegisterClick}>회원가입</button>
                    </div>
                </div>
            </div>
        </div>
    );
};

export default U_login;
