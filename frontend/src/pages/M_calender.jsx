// M_calender 페이지 코드
import React, { useState } from 'react';
import Calendar from 'react-calendar';
import { useNavigate } from 'react-router-dom';
import 'react-calendar/dist/Calendar.css';
import { useEffect } from 'react';

const M_calender = () => {
  const [date, setDate] = useState(new Date()); // 기본 날짜
  const [startDate, setStartDate] = useState(null); // 시작 날짜
  const [endDate, setEndDate] = useState(null); // 종료 날짜
  const [carNumber, setCarNumber] = useState(null); // 차량 번호 저장
  const navigate = useNavigate();

  // 차량 번호 가져오는 useEffect 훅
  useEffect(() => {
    fetch('http://localhost:4000/user/carNumber')
      .then(response => {
        if (!response.ok) {
          throw new Error('Failed to fetch car number');
        }
        return response.json();
      })
      .then(data => {
        setCarNumber(data.carNumber);
        console.log("차량정보 조회되나 보자:", data.carNumber); 
      })
      .catch(error => {
        console.error('Error fetching car number:', error);
      });
  }, []);

  


  // 기간 조회 버튼 클릭 시 실행되는 함수
  const handleSearch = async () => {
    if (!startDate || !endDate) {
      alert("시작 기간과 종료 기간을 선택해 주세요.");
      return;
    }

    if (new Date(startDate) >= new Date(endDate)) {
      alert("시작 날짜는 종료 날짜보다 먼저여야 합니다.");
      return;
    }

    try {
      // API 호출 - 필터링 페이지로 이동하면서 파라미터 전달
      navigate(`/filtering?start=${startDate.toISOString().split("T")[0]}&end=${endDate.toISOString().split("T")[0]}&mode=period`);
    } catch (error) {
      console.error('API 호출 중 오류:', error);
      alert('API 호출 중 오류가 발생했습니다.');
    }
  };

  // 날짜에 하루를 더하는 함수
  const addOneDay = (date) => {
    const newDate = new Date(date);
    newDate.setDate(newDate.getDate() + 1); // 하루 추가
    return newDate;
  };

  // 날짜 클릭 시 필터링 페이지로 이동하는 함수
  const handleDayClick = (selectedDate) => {
    const adjustedDate = addOneDay(selectedDate);
    setDate(adjustedDate);
    navigate(`/filtering?date=${adjustedDate.toISOString().split("T")[0]}`);
  };

  return (
    <div className="m_calender-container-unique">
      <Calendar
        onChange={setDate}
        value={date}
        locale="en-US"
        onClickDay={handleDayClick} // 날짜 클릭 시 handleDayClick 함수 실행
        className="m_calender-unique"
      />

      <div className="m_calender-time-select-unique">
        <p className="m_calender-label-unique">기간 조회</p>
        <div className="m_calender-time-inputs-unique">
          <input
            type="date"
            value={startDate ? startDate.toISOString().split("T")[0] : ''}
            onChange={(e) => setStartDate(new Date(e.target.value))}
            className="m_calender-input-unique"
            placeholder="시작 기간"
          />
          <span className="m_calender-time-divider-unique">~</span>
          <input
            type="date"
            value={endDate ? endDate.toISOString().split("T")[0] : ''}
            onChange={(e) => setEndDate(new Date(e.target.value))}
            className="m_calender-input-unique"
            placeholder="종료 기간"
          />
        </div>
        <button className="m_calender-search-button-unique" onClick={handleSearch}>
          조회하기
        </button>
      </div>
    </div>
  );
};

export default M_calender;