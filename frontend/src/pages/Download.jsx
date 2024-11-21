import React from 'react';
import { useLocation } from 'react-router-dom';
import { PDFDownloadLink } from '@react-pdf/renderer';
import PDF_form from '../components/PDF_form';


// 날짜 형식을 YYYY-MM-DD로 변환하는 함수
const formatDate = (uploadTime) => {
  if (!uploadTime || uploadTime === 'default_value') {
    return '날짜 없음';
  }
  const date = new Date(uploadTime);
  if (isNaN(date.getTime())) {
    return '유효하지 않은 날짜'; // 날짜 형식이 잘못된 경우
  }
  const year = date.getFullYear();
  const month = String(date.getMonth() + 1).padStart(2, '0');
  const day = String(date.getDate()).padStart(2, '0');
  return `${year}-${month}-${day}`; // YYYY-MM-DD 형식
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


const Download = () => {
  const location = useLocation();
  const previewData = location.state?.previewData || [];

  return (
    <div className="download_container">
      <div className="download_preview">
        {previewData.length > 0 ? (
          <table className="preview_table">
            <thead>
              <tr>
                {/* 적재물 데이터인지, 위반차량 데이터인지에 따라 컬럼 표시 */}
                {previewData[0].description ? (
                  <>
                    <th>ID</th>
                    <th>설명</th>
                    <th>탐지 시간</th>
                  </>
                ) : (
                  <>
                    <th>날짜</th>
                    <th>차량번호</th>
                    <th>장소</th>
                    <th>주차시간</th>
                  </>
                )}
              </tr>
            </thead>
            <tbody>
              {previewData.map((item, index) => (
                <tr key={index}>
                  {item.description ? (
                    // 적재물 데이터 렌더링
                    <>
                      <td>{item.id}</td>
                      <td>{item.description}</td>
                      <td>{new Date(item.detected_at).toLocaleString()}</td>
                    </>
                  ) : (
                    // 위반차량 데이터 렌더링
                    <>
                      <td>{formatDate(item.upload_time)}</td>
                      <td>{item.violation_number}</td>
                      <td>{item.violation_location || '정보 없음'}</td>
                      <td>{formatTime(item.upload_time)}</td>
                    </>
                  )}
                </tr>
              ))}
            </tbody>
          </table>
        ) : (
          <p>다운로드할 데이터가 없습니다.</p>
        )}
      </div>

      {previewData.length > 0 && (
        <div className="download_button_container">
          <PDFDownloadLink
            document={<PDF_form data={previewData} />}
            fileName="Violation_Report.pdf"
          >
            {({ loading }) =>
              loading ? 'PDF 생성 중...' : <button className="download_button">Download</button>
            }
          </PDFDownloadLink>
        </div>
      )}
    </div>
  );
};

export default Download;
