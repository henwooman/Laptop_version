import React, { useState, useEffect } from 'react';
import { useNavigate, useLocation } from 'react-router-dom';
import DatePicker from 'react-datepicker';
import 'react-datepicker/dist/react-datepicker.css';

const Filtering = () => {
  const [selectedTime, setSelectedTime] = useState(''); // 조회시간 필터 (6시간, 12시간, 24시간)
  const [selectedOrder, setSelectedOrder] = useState(''); // 정렬 순서
  const [carNumber, setCarNumber] = useState(''); // 차량 번호 입력
  const [startDate, setStartDate] = useState(null); // 시작 시간
  const [endDate, setEndDate] = useState(null); // 종료 시간
  const [isTimeDisabled, setIsTimeDisabled] = useState(false); // 시간 선택 비활성화 여부

  const navigate = useNavigate();
  const location = useLocation();
  const queryParams = new URLSearchParams(location.search);
  const selectedDate = queryParams.get('date'); // 메인 페이지에서 선택한 날짜
  const mode = queryParams.get('mode'); // 기간 조회 모드 여부

  // 모드에 따라 시간 선택 비활성화
  useEffect(() => {
    if (mode === 'period') {
      // 기간 조회 모드일 때 시간 범위를 00:00~23:59로 고정
      setIsTimeDisabled(true);
      setStartDate(new Date(`${queryParams.get('start')}T00:00:00`));
      setEndDate(new Date(`${queryParams.get('end')}T23:59:59`));
    }
  }, [mode]);

  // 조회 시간 버튼 클릭 시 시간 범위 설정
  const handleTimeClick = (time) => {
    if (isTimeDisabled) return;

    setSelectedTime(time); // 선택한 조회시간 버튼 활성화
    setStartDate(null); // 슬라이드박스 시작 시간 초기화
    setEndDate(null); // 슬라이드박스 종료 시간 초기화
  };

  // 정렬 순서 버튼 클릭
  const handleOrderClick = (order) => setSelectedOrder(order);

  // 슬라이드박스에서 시작 시간 선택
  const handleStartDateChange = (time) => {
    setSelectedTime(''); // 조회 시간 버튼 초기화

    if (selectedDate) {
      const selectedStartDate = new Date(selectedDate); // 메인 페이지에서 선택한 날짜
      selectedStartDate.setHours(time.getHours());
      selectedStartDate.setMinutes(time.getMinutes());
      setStartDate(selectedStartDate); // 날짜와 시간 결합된 시작 시간 설정
    } else {
      setStartDate(time); // 선택한 시간만으로 시작 시간 설정
    }
  };

  // 슬라이드박스에서 종료 시간 선택
  const handleEndDateChange = (time) => {
    setSelectedTime(''); // 조회 시간 버튼 초기화

    if (selectedDate) {
      const selectedEndDate = new Date(selectedDate); // 메인 페이지에서 선택한 날짜
      selectedEndDate.setHours(time.getHours());
      selectedEndDate.setMinutes(time.getMinutes());
      setEndDate(selectedEndDate); // 날짜와 시간 결합된 종료 시간 설정
    } else {
      setEndDate(time); // 선택한 시간만으로 종료 시간 설정
    }
  };

  // 시간을 포맷하여 쿼리에 추가할 수 있도록 변환
  const formatDateTime = (date) => (date ? date.toISOString() : '');

  // 조회하기 버튼 클릭 시 쿼리 생성 및 상세 페이지로 이동
  // Filtering 컴포넌트에서 handleSearch 함수 수정
const handleSearch = () => {
  if (!carNumber) {
      alert("차량 번호를 입력해 주세요.");
      return;
  }
  if (!selectedTime && (!startDate || !endDate)) {
      alert("시간 범위를 설정하거나 시작 및 종료 시간을 모두 입력해 주세요.");
      return;
  }

  const query = new URLSearchParams();
  query.append("carNumber", carNumber);

  if (selectedTime) {
      query.append("selectedTime", selectedTime);
  } else {
      // 날짜 형식을 YYYY-MM-DD로 변환
      const formattedStartDate = startDate.toISOString().split("T")[0];
      const formattedEndDate = endDate.toISOString().split("T")[0];
      query.append("start", formattedStartDate);
      query.append("end", formattedEndDate);
  }

  if (selectedOrder) query.append("order", selectedOrder);
  if (selectedDate) query.append("date", selectedDate);

  // 최종 URL 확인 로그 추가
  console.log('Generated URL:', `/detail?${query.toString()}`);

  navigate(`/detail?${query.toString()}`);
};



  return (
    <div className="filtering-page-container">
      <div className="filtering-page-box">
        <div className="filtering-input-group">
          <label htmlFor="carNumber" className="filtering-label">차량번호</label>
          <input
            type="text"
            placeholder="차량 번호를 입력하세요."
            value={carNumber}
            onChange={(e) => setCarNumber(e.target.value)}
            className="filtering-input vehicle-number-input"
          />
        </div>

        <hr className="filtering_line" />

        <div className="filtering-time-section">
          <p className="filtering-section-title">조회시간</p>
          <div className="filtering-time-buttons">
            <button
              onClick={() => handleTimeClick('6시간')}
              className={selectedTime === '6시간' ? 'filtering-active' : ''}
              disabled={isTimeDisabled}
            >
              6시간
            </button>
            <button
              onClick={() => handleTimeClick('12시간')}
              className={selectedTime === '12시간' ? 'filtering-active' : ''}
              disabled={isTimeDisabled}
            >
              12시간
            </button>
            <button
              onClick={() => handleTimeClick('24시간')}
              className={selectedTime === '24시간' ? 'filtering-active' : ''}
              disabled={isTimeDisabled}
            >
              24시간
            </button>
          </div>

          <div className="filtering-order-buttons">
            <button
              onClick={() => handleOrderClick('최신순')}
              className={selectedOrder === '최신순' ? 'filtering-active' : ''}
              disabled={isTimeDisabled}
            >
              최신순
            </button>
            <button
              onClick={() => handleOrderClick('과거순')}
              className={selectedOrder === '과거순' ? 'filtering-active' : ''}
              disabled={isTimeDisabled}
            >
              과거순
            </button>
          </div>
        </div>

        <div className="filtering-time-range">
          <p className="filtering-section-title">시간 조회</p>
          <div className="filtering-time-inputs">
            <DatePicker
              selected={startDate || undefined} // 선택한 시작 시간이 없으면 undefined로 설정
              onChange={handleStartDateChange}
              showTimeSelect
              showTimeSelectOnly
              timeIntervals={15}
              timeCaption="시작 시간"
              dateFormat="h:mm aa"
              placeholderText="시작 시간"
              disabled={isTimeDisabled}
              className="filtering-input time-input" 
              timeClassName={() => "custom-time-list"}
            />
            <span className="filtering-time-divider">~</span>
            <DatePicker
              selected={endDate || undefined} // 선택한 종료 시간이 없으면 undefined로 설정
              onChange={handleEndDateChange}
              showTimeSelect
              showTimeSelectOnly
              timeIntervals={15}
              timeCaption="종료 시간"
              dateFormat="h:mm aa"
              placeholderText="종료 시간"
              disabled={isTimeDisabled}
              className="filtering-input time-input"
              timeClassName={() => "custom-time-list"}
            />
          </div>
        </div>

        <button className="filtering-search-button" onClick={handleSearch}>조회하기</button>
      </div>
    </div>
  );
};

export default Filtering;
