import React, { useState, useEffect } from 'react';
import { useNavigate } from 'react-router-dom';
import axios from 'axios';

const ViolationsList = ({ setData }) => {
  const navigate = useNavigate();
  const [violations, setViolations] = useState([]); // DB에서 가져온 데이터를 저장할 상태 
  const [loading, setLoading] = useState(true); 
  const [error, setError] = useState(''); // 에러를 표시

  useEffect(() => {
    const fetchViolations = async () => {
      try {
        const response = await axios.get('http://localhost:4000/violation/all', {
          withCredentials: true,
        });
        console.log("Fetched Data:", response.data); // 전체 데이터 확인

        // 필터링 조건을 정확하게 설정
        const validData = response.data.filter(
          violation => violation.upload_time !== null &&
                      violation.upload_time !== undefined &&
                      violation.upload_time !== 'default_value' &&
                      !isNaN(new Date(violation.upload_time).getTime())
        );
        console.log("Filtered Data:", validData); // 필터링된 데이터 확인

        setViolations(validData);
        setData(validData);
        setLoading(false);
      } catch (error) {
        console.error('서버 오류:', error);
        if (error.response) {
          // 서버가 응답을 반환했지만 상태 코드가 2xx가 아닌 경우
          console.error('응답 데이터:', error.response.data);
          setError(`데이터를 불러오는 중 오류가 발생했습니다: ${error.response.data.message || '알 수 없는 오류'}`);
        } else if (error.request) {
          // 요청이 만들어졌으나 응답을 받지 못한 경우
          console.error('요청이 이루어졌으나 응답을 받지 못했습니다:', error.request);
          setError('서버로부터 응답을 받지 못했습니다.');
        } else {
          // 오류를 발생시킨 요청을 설정하는 중에 문제가 발생한 경우
          console.error('오류 발생:', error.message);
          setError(`오류가 발생했습니다: ${error.message}`);
        }
        setLoading(false);
      }
    };

    fetchViolations();
  }, [setData]);  // 컴포넌트가 마운트될 때 한 번 실행

  const handleRowClick = (violation) => {
    navigate(`/detail`, { state: { violation } });
  };

  if (loading){
    return <p>로딩 중.....</p>;
  }

  if (error) {
    return <p style={{ color:'red' }}>{error}</p>;
  }

  // 날짜 형식을 YYYY-MM-DD로 변환하는 함수
  const formatDate = (uploadTime) => {
    if (!uploadTime || uploadTime === 'default_value') {
      return '날짜 없음';
    }
    const date = new Date(uploadTime);
    if (isNaN(date.getTime())) {
      return '유효하지 않은 날짜';
    }
    const year = date.getFullYear();
    const month = String(date.getMonth() + 1).padStart(2, '0');
    const day = String(date.getDate()).padStart(2, '0');
    return `${year}-${month}-${day}`;
  };

  // 시간을 HH:MM:SS 형식으로 변환하는 함수
  const formatTime = (uploadTime) => {
    if (!uploadTime || uploadTime === 'default_value') {
      return '시간 없음';
    }
    const date = new Date(uploadTime);
    if (isNaN(date.getTime())) {
      return '유효하지 않은 시간';
    }
    const hours = String(date.getHours()).padStart(2, '0');
    const minutes = String(date.getMinutes()).padStart(2, '0');
    const seconds = String(date.getSeconds()).padStart(2, '0');
    return `${hours}:${minutes}:${seconds}`;
  };

  return (
    <div className="violations-list-container">
      {/* 컬럼 제목 */}
      <div className="violation-header">
        <div className="violation-cell-header">날짜</div>
        <div className="violation-cell-header">주차시간</div>
        <div className="violation-cell-header">차량번호</div>
        <div className="violation-cell-header">장소</div>
        {/* 필요에 따라 추가 컬럼 */}
      </div>

      {/* 스크롤 가능한 리스트 컨테이너 */}
      <div className="violations-list-scroll">
        {/* 데이터 목록 또는 "위반 내역이 없습니다." 메시지 */}
        {violations.length > 0 ? (
          violations.map((violation, index) => (
            <div
              key={index}
              className="violation-row"
              onClick={() => handleRowClick(violation)}
              style={{ cursor: 'pointer' }}
            >
              <div className="violation-cell">{formatDate(violation.upload_time)}</div>
              <div className="violation-cell">{formatTime(violation.upload_time)}</div>
              <div className="violation-cell">{violation.violation_number}</div>
              <div className="violation-cell">{violation.violation_location || '정보 없음'}</div>
              {/* 필요에 따라 추가 데이터 표시 */}
            </div>
          ))
        ) : (
          <p style={{ textAlign: 'center', padding: '20px', color: 'gray' }}>
            위반 내역이 없습니다.
          </p>
        )}
      </div>
    </div>
  );
};

export default ViolationsList;
