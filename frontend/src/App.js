// App.js
import './App.css';
import Header from './components/Header';
import U_login from './pages/U_login';
import Join from './pages/Join';
import Download from './pages/Download';
import M_btnBar from './components/M_btnBar';
import Filtering from './pages/Filtering';
import Setting from './pages/Setting';
import M_calender from './pages/M_calender';
import Main from './pages/Main';
import ViolationsPage from './pages/ViolationsPage';
import M_detail from './pages/M_detail';
import { BrowserRouter as Router, Route, Routes, useLocation } from 'react-router-dom';
import { useState } from 'react';
import { Myparking_place } from './pages/Myparking_place';
import ParkingList from './pages/ParkingList';
import { ParkingSearch } from './pages/ParkingSearch';
import { UserMain } from './pages/UserMain';
import EditProfile from './pages/EditProfile';
import Btnbar from './components/Btnbar';
import Handicap_car from './pages/Handicap_car';

function AppContent() {
  const location = useLocation();
  const [violationsData, setViolationsData] = useState([]);
  const [detailData, setDetailData] = useState([]);
  const [isAdmin, setIsAdmin] = useState(false); // 기본값을 false로 설정

  // 현재 경로에 따라 페이지 이름 설정
  const getPageTitle = (pathname) => {
    console.log("Checking path:", pathname); // 현재 경로 확인용

    switch (pathname) {
      // 관리자모드
      case '/join': return '회원가입';
      case '/filtering': return '조회할 시간을 선택하세요.';
      case '/setting': return '설정'; 
      case '/download': return '다운로드'; 
      case '/M_calender': return '조회할 기간을 선택하세요.';
      case '/Violations': return '위반 정보';
      case '/detail': return '세부사항';
      case '/register-car': return '장애인등록차량 관리'; 
      // 사용자모드
      case '/userMain': return '메인페이지'; 
      case '/editprofile': return '회원정보수정'; 
      case '/parkingSearch': return '주차장찾기'; 
      case '/parkinglist': return '즐겨찾기'; 
      case '/Myparking_place': return '주차장 내부현황'; 
      default: return '페이지 제목 없음'; // 디폴트 제목 추가
    }
  };

  const pageTitle = getPageTitle(location.pathname);

  // 사용자 버튼바가 나타나야 하는 경로 확인
  const userPages = ['/myparking_place', '/parkinglist', '/parkingSearch', '/userMain'];
  const adminPages = ['/filtering', '/setting', '/download', '/M_calender', '/Violations', '/detail','/register-car'];

  return (
    <div className="mobile_frame">
      <img className="mobile_frame_img" src="/images/mobile.png" alt="iPhone Frame" />
      <div
        className="mobile_content"
        style={{ backgroundImage: `url(${process.env.PUBLIC_URL}/images/background.png)` }}
      >
        {/* Main, Login 페이지를 제외한 모든 페이지에서 Header 렌더링 */}
        {location.pathname !== '/' && location.pathname !== '/login' && <Header title={pageTitle} />}

        <Routes>
          {/* 사용자/ 관리자 */ }
          <Route path="/" element={<Main/>} />
          <Route path="/login" element={<U_login setIsAdmin={setIsAdmin} />} />
          <Route path="/join" element={<Join/>} />
          <Route path="/filtering" element={<Filtering/>} />
          <Route path="/setting" element={<Setting/>} />
          <Route path="/download" element={<Download/>} />
          <Route path="/m_calender" element={<M_calender/>} />
          <Route path="/violations" element={<ViolationsPage/>}/>
          <Route path="/detail" element={<M_detail/>} />
          <Route path='/register-car' element={<Handicap_car/>}/>
          {/* 사용자 */ }
          <Route path="/myparking_place" element={<Myparking_place/>} />
          <Route path="/parkinglist" element={<ParkingList/>} />
          <Route path="/parkingSearch" element={<ParkingSearch/>} />
          <Route path="/userMain" element={<UserMain/>} />
          <Route path="/editprofile" element={<EditProfile/>} />
        </Routes>

        {/* 사용자 버튼바 또는 관리자 버튼바 조건부 렌더링 */}
        {location.pathname !== '/' && location.pathname !== '/login' && location.pathname !== '/join' &&
          (isAdmin ? (
            adminPages.includes(location.pathname) ? <M_btnBar violationsData={violationsData} detailData={detailData} /> : null
          ) : (
            userPages.includes(location.pathname) ? <Btnbar/> : null
          ))
        }
      </div>
    </div>
  );
}

function App() {
  return (
    <Router>
      <AppContent />
    </Router>
  );
}

export default App;
