// Header.js
import React from 'react';
import { useNavigate } from 'react-router-dom';  //뒤로가기 버튼에 사용할 hook

const Header = ({ title }) => {
    console.log("Header title:", title); // title 값을 콘솔에 출력

    const navigate = useNavigate(); // 네비게이트 훅 사용

    const handleBackClick = () => {
        navigate(-1); // 이전 페이지로 이동
    };

    return (
        <div className="header_custom_container">
            <div className="header_custom_back_button" onClick={handleBackClick}>&#x2190;</div>
            <h1 className="header_custom_title">{title}</h1>
        </div>
    );
};

export default Header;
