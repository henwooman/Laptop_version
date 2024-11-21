import React, { useState } from 'react';
import ParkingMa from '../components/ParkingMa';
import axios from 'axios';

export const ParkingSearch = () => {
  const [searchTerm, setSearchTerm] = useState('');
  const [coordinates, setCoordinates] = useState([]); // 위도, 경도 상태 추가

  const handleSearch = () => {
    const inputValue = document.getElementById('searchInput').value;
    setSearchTerm(inputValue);
  };

  const sendCoordinatesToServer = async () => {
    if (coordinates.length === 0) {
      alert('먼저 위도와 경도를 선택해주세요.');
      return;
    }
  
    try {
      for (let coord of coordinates) {
        // axios 요청에서 응답을 받으면 response 변수에 할당
        const response = await axios.post('http://localhost:4000/parkingSearch/parkinglist', {
          latitude: coord.latitude,
          longitude: coord.longitude
        });
  
        // 서버 응답을 콘솔에 출력하거나 사용
        console.log("서버 응답:", response.data);
      }
  
      alert('위도와 경도가 저장되었습니다!');
    } catch (error) {
      console.error('서버로 전송 실패:', error);
      alert('저장에 실패했습니다.');
    }
  };

  

  return (
    <div className="parking-container">
      <div className='parking_search'>
        <div className="address-input">
          <p>주소를 입력해주세요.</p>
        </div>
        <div className="searchbox">
          <div className="searchbox-container">
            <input
              id="searchInput"
              className="box"
              type="text"
              placeholder="목적지 또는 주소 검색"
            />
            <button className="searchbox-btn" onClick={handleSearch}>검색</button>
          </div>
        </div>
        <div className="example">
          <p>
            이렇게 검색해보세요!
            <br />· 도로명 + 건물번호 (예: 정자일로 95, 불정로 6)
            <br />· 동/읍/면/리 + 번지 (예: 정자동 178-4, 동면 만천리 1000)
          </p>
        </div>
      </div>
      
      {/* 지도 컴포넌트가 parking_search 아래에 위치 */}
      <div className="map-container">
        <ParkingMa searchTerm={searchTerm} setCoordinates={setCoordinates} />
      </div>

      {/* 저장 버튼 추가 */}
      {coordinates.length > 0 && (
        <div className="save-button-container">
          <button className="save-button" onClick={sendCoordinatesToServer}>저장</button>
        </div>
      )}
    </div>
  );
};

export default ParkingSearch;
