import React, { useEffect, useState } from 'react';
import { useLocation } from 'react-router-dom';
import axios from 'axios';
import { PDFDownloadLink } from '@react-pdf/renderer';
import PDF_form from '../components/PDF_form';

const M_detail = ({ setData }) => {
  const location = useLocation();
  const { violation } = location.state || {};

  const queryParams = new URLSearchParams(location.search);
  const date = queryParams.get('date');
  const startDate = queryParams.get('start');
  const endDate = queryParams.get('end');
  const carNumber = queryParams.get('carNumber');
  const selectedTime = queryParams.get('selectedTime');

  const [violations, setViolations] = useState(violation ? [violation] : []);
  const [loading, setLoading] = useState(!violation);
  const [videos, setVideos] = useState([]);
  const [selectedVideo, setSelectedVideo] = useState(null); // 선택된 동영상 상태
  const [selectedCarNumber, setSelectedCarNumber] = useState(carNumber); // 선택된 차량 번호

  let startTimeDisplay = startDate;
  let endTimeDisplay = endDate;

  // 동영상 다운로드 핸들러
  const handleVideoDownload = async () => {
    try {
      // `selectedVideo`의 URL 가져오기
      if (!selectedVideo || !selectedVideo.url) {
        alert('다운로드할 동영상이 없습니다.');
        return;
      }
  
      const videoUrl = selectedVideo.url;
      console.log('Downloading video from URL:', videoUrl);
  
      // Fetch 요청
      const response = await fetch(videoUrl, { mode: 'cors' }); // `mode: 'cors'`는 CORS 요청 허용
      if (!response.ok) {
        throw new Error(`Failed to fetch video: ${response.statusText}`);
      }
  
      // Blob으로 변환
      const blob = await response.blob();
      const blobUrl = URL.createObjectURL(blob);
  
      // Blob을 사용해 다운로드 트리거
      const link = document.createElement('a');
      link.href = blobUrl;
      link.download = selectedVideo.filename || 'video.mp4'; // 파일 이름
      document.body.appendChild(link);
      link.click();
      document.body.removeChild(link);
      URL.revokeObjectURL(blobUrl); // Blob URL 해제
    } catch (error) {
      console.error('Download error:', error);
      alert('동영상 다운로드 중 문제가 발생했습니다.');
    }
  };
  
  
  
  
  
  

  const formatDate = (isoDate) => {
    if (!isoDate) return '날짜 없음';
    const dateObj = new Date(isoDate);
    return isNaN(dateObj.getTime())
      ? '유효하지 않은 날짜'
      : `${String(dateObj.getFullYear()).slice(2)}-${String(dateObj.getMonth() + 1).padStart(2, '0')}-${String(dateObj.getDate()).padStart(2, '0')}`;
  };

  const formatTime = (isoDate) => {
    if (!isoDate) return '시간 없음';
    const dateObj = new Date(isoDate);
    return isNaN(dateObj.getTime())
      ? '유효하지 않은 시간'
      : `${String(dateObj.getHours()).padStart(2, '0')}:${String(dateObj.getMinutes()).padStart(2, '0')}`;
  };

  const formatDateTime = (date, showTimeOnly = false) => {
    if (!date) return '날짜 없음';
    const dateObj = new Date(date);
    return isNaN(dateObj.getTime())
      ? '유효하지 않은 날짜'
      : showTimeOnly
        ? `${String(dateObj.getHours()).padStart(2, '0')}:${String(dateObj.getMinutes()).padStart(2, '0')}`
        : `${dateObj.getFullYear()}-${String(dateObj.getMonth() + 1).padStart(2, '0')}-${String(dateObj.getDate()).padStart(2, '0')} ${String(dateObj.getHours()).padStart(2, '0')}:${String(dateObj.getMinutes()).padStart(2, '0')}`;
  };

  // 조회 시간 설정
  if (selectedTime) {
    const now = new Date();
    const selectedDateTime = new Date(date);
    selectedDateTime.setHours(now.getHours(), now.getMinutes(), 0, 0);

    const end = new Date(selectedDateTime);
    const start = new Date(selectedDateTime);

    if (selectedTime === '6시간') {
      start.setHours(selectedDateTime.getHours() - 6);
    } else if (selectedTime === '12시간') {
      start.setHours(selectedDateTime.getHours() - 12);
    } else if (selectedTime === '24시간') {
      start.setHours(selectedDateTime.getHours() - 24);
    }

    startTimeDisplay = formatDateTime(start);
    endTimeDisplay = formatDateTime(end);
  } else if (startDate && endDate) {
    startTimeDisplay = `${formatDateTime(startDate)}`;
    endTimeDisplay = `${formatDateTime(endDate)}`;
  }

  useEffect(() => {
    const fetchData = async () => {
      try {
        let response;
        if (startDate && endDate && carNumber) {
          response = await axios.get('http://localhost:4000/violation/filtering_dateRange', {
            params: {
              startDate: startDate.split('T')[0],
              endDate: endDate.split('T')[0],
              car_number: carNumber,
            },
          });
        } else if (date && carNumber) {
          response = await axios.get('http://localhost:4000/violation/filter_Violations', {
            params: {
              car_number: carNumber,
              startTime: startTimeDisplay,
              endTime: endTimeDisplay,
            },
          });
        } else {
          setLoading(false);
          return;
        }

        setViolations(response.data);
        if (setData) {
          setData(response.data);
        }

        // 첫 번째 차량 번호에 맞는 동영상 로드
        if (carNumber) {
          const videoResponse = await axios.get('http://localhost:4000/violation/videos', {
            params: { violationNumber: carNumber },
          });

          setVideos(videoResponse.data.data || []);
          setSelectedVideo(videoResponse.data.data?.[0] || null); // 첫 번째 동영상 선택
        }
      } catch (error) {
        console.error('데이터를 가져오는 중 오류 발생:', error);
      } finally {
        setLoading(false);
      }
    };

    fetchData();
  }, [carNumber, date, endDate, startDate, startTimeDisplay, endTimeDisplay, setData]);

  // 차량 선택 시 관련 동영상 로드 및 첫 번째 동영상 선택
  const handleCarSelection = async (violationNumber, dateTime) => {
    console.log(`Selected car number: ${violationNumber}`);
    setSelectedCarNumber(violationNumber);
  
    try {
      const videoResponse = await axios.get('http://localhost:4000/violation/videos', {
        params: { violationNumber, dateTime },
      });
  
      console.log('Fetched videos for car:', videoResponse.data);
  
      // videos와 selectedVideo 상태 업데이트
      setVideos(videoResponse.data.data || []);
      setSelectedVideo(videoResponse.data.data?.[0] || null); // 첫 번째 동영상 선택
      console.log('Updated selectedVideo:', videoResponse.data.data?.[0]);
    } catch (error) {
      console.error('동영상을 가져오는 중 오류 발생:', error);
      setVideos([]);
      setSelectedVideo(null);
    }
  };
  

  

  return (
    <div className="m_detail-container-unique">
      <div className="m_detail-filter-unique">
        <table className="filter-table">
          <tbody>
            <tr>
              <td>차량번호:</td>
              <td>{violation ? violation.violation_number : carNumber}</td>
            </tr>
            <tr>
              <td>선택된 날짜:</td>
              <td>{startDate && endDate ? `${formatDate(startDate)} ~ ${formatDate(endDate)}` : formatDate(date)}</td>
            </tr>
            <tr>
              <td>조회 기간:</td>
              <td>
                {`${startTimeDisplay} ~ ${endTimeDisplay}`}
                <br />
                {selectedTime ? ` (${selectedTime})` : ''}
              </td>
            </tr>
          </tbody>
        </table>
      </div>

      {/* 동영상 표시 영역 */}
      <div className="m_detail-photo-box-unique">
        {loading ? (
          <div className="m_detail-loading">동영상을 불러오는 중입니다...</div>
        ) : selectedVideo ? (
          <video controls src={selectedVideo.url} className="m_detail-video" />
        ) : (
          <div className="m_detail-placeholder">동영상을 선택하세요.</div>
        )}
      </div>

      {/* 위반 차량 리스트 */}
      <div className="m_detail-violation-unique">
        <table className="violation-table">
          <thead>
            <tr>
              <th>위반차량</th>
              <th>위반날짜</th>
              <th>위반시간</th>
              <th>위반장소</th>
            </tr>
          </thead>
          <tbody>
            {violations.length > 0 ? (
              violations.map((v, index) => (
                <tr key={index}>
                  <td
                    onClick={() => handleCarSelection(v.violation_number, v.upload_time)}
                    style={{ cursor: 'pointer', color: 'blue' }}
                  >
                    {v.violation_number}
                  </td>
                  <td>{formatDate(v.upload_time)}</td>
                  <td>{formatTime(v.upload_time)}</td>
                  <td>{v.violation_location || '장소 정보 없음'}</td>
                </tr>
              ))
            ) : (
              <tr>
                <td colSpan="4">위반 내역이 없습니다.</td>
              </tr>
            )}
          </tbody>
        </table>
      </div>

      {/* PDF 및 동영상 다운로드 버튼 */}
      <div className="m_detail-download-buttons">
        <PDFDownloadLink
          document={<PDF_form data={violations} />}
          fileName="Violation_Report.pdf"
        >
          {({ loading: pdfLoading }) =>
            pdfLoading ? (
              'PDF 생성 중...'
            ) : (
              <button className="m_detail-download-button">PDF 다운로드</button>
            )
          }
        </PDFDownloadLink>

        <button className="m_detail-download-button" onClick={handleVideoDownload}>
          동영상 다운로드
        </button>
      </div>
    </div>
  );
};

export default M_detail;
