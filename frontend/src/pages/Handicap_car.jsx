import React, { useState } from 'react';
import axios from 'axios';

// 날짜 형식을 'YYYY-MM-DD'로 변환하는 함수
const formatDate = (isoDateString) => {
  if (!isoDateString) return '날짜 없음';
  const date = new Date(isoDateString);
  const year = date.getFullYear();
  const month = String(date.getMonth() + 1).padStart(2, '0');
  const day = String(date.getDate()).padStart(2, '0');
  return `${year}-${month}-${day}`; // 'YYYY-MM-DD' 형식으로 반환
};

const Handicap_car = () => {
  const [carNumber, setCarNumber] = useState(''); // 입력된 차량번호 상태
  const [carData, setCarData] = useState(null); // 조회된 차량 데이터 상태
  const [isLoading, setIsLoading] = useState(false); // 로딩 상태
  const [error, setError] = useState(''); // 오류 메시지 상태

  // 차량번호로 데이터 조회 함수
  const handleSearch = async () => {
    setIsLoading(true);
    setError('');
    try {
      const response = await axios.get('http://localhost:4000/user/search-by-car-number', {
        params: { carNumber },
      });
      console.log('API 응답 데이터:', response.data); // API 응답 확인


      if (response.data) {
        setCarData(response.data);
        console.log('핸디캡 상태 확인:', response.data.handicap); // handicap 값 확인
      } else {
        setCarData(null);
        alert('해당 차량에 대한 데이터가 없습니다.');
      }
    } catch (err) {
      setError('조회 중 오류가 발생했습니다.');
      console.error(err);
    } finally {
      setIsLoading(false);
    }
  };

  // 등록/해제 버튼 클릭 시 처리 함수
  const handleRegisterToggle = async () => {
    if (!carData) return;
    const newStatus = carData.handicap === 1 ? 0 : 1; // 현재 상태 반전

    try {
      await axios.post('http://localhost:4000/user/update-handicap', {
        carNumber: carData.car_number,
        handicapStatus: newStatus,
      });
      setCarData({ ...carData, handicap: newStatus }); // 상태 업데이트
      console.log('핸디캡 상태 업데이트:', newStatus); // 업데이트 후 상태 확인
    } catch (err) {
      console.error('업데이트 중 오류 발생:', err);
      setError('등록/해제 중 오류가 발생했습니다.');
    }
  };

  return (
    <div className="handicap_car_container">
      <div className="handicap_car_content">
        <p>반갑습니다. 관리자님!</p>
        <p>장애인 등록이 확인된 차량에는 <br /> 등록 버튼을 눌러주세요.</p>

        <div className="search_section">
          <input
            type="text"
            placeholder="조회할 차량 번호를 입력하세요."
            value={carNumber}
            onChange={(e) => setCarNumber(e.target.value)}
          />
          <button className="search_button" onClick={handleSearch}>
            조회하기
          </button>
        </div>

        {isLoading && <p>로딩 중...</p>}
        {error && <p style={{ color: 'red' }}>{error}</p>}

        {/* 데이터 출력 */}
        {carData && (
          <table className="car_list_table">
            <thead>
              <tr>
                <th>가입날짜</th>
                <th>차량번호</th>
                <th>이름</th>
                <th>연락처</th>
                <th>상태</th>
              </tr>
            </thead>
            <tbody>
              <tr>
                <td>{formatDate(carData.user_joined)}</td>
                <td>{carData.car_number}</td>
                <td>{carData.user_name}</td>
                <td>{carData.user_phone}</td>
                <td>
                  {carData.handicap !== null && carData.handicap !== undefined ? (
                    <button
                      onClick={handleRegisterToggle}
                      className={carData.handicap ? 'registered' : ''}
                    >
                      {carData.handicap ? '해제' : '등록'}
                    </button>
                  ) : (
                    <p>상태를 불러오는 중입니다...</p>
                  )}
                </td>
              </tr>
            </tbody>
          </table>
        )}
      </div>
    </div>
  );
};

export default Handicap_car;
