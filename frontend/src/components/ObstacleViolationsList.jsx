import React, { useState, useEffect } from 'react';
import axios from 'axios';

const ObstacleViolationsList = ({ setData }) => {
  const [obstacles, setObstacles] = useState([]); // 적재물 데이터 상태
  const [loading, setLoading] = useState(true); // 로딩 상태
  const [error, setError] = useState(''); // 에러 상태

  useEffect(() => {
    const fetchObstacles = async () => {
      try {
        // API 호출로 적재물 위반 데이터를 가져옵니다.
        const response = await axios.get('http://localhost:4000/violation/obstacle_all', {
          withCredentials: true,
        });
        setObstacles(response.data);
        setData(response.data);
        setLoading(false);
      } catch (error) {
        console.error('서버 오류:', error);
        setError('데이터를 불러오는 중 오류가 발생했습니다.');
        setLoading(false);
      }
    };

    fetchObstacles();
  }, [setData]); // setData가 변하면 다시 fetch

  if (loading) {
    return <p>로딩 중...</p>;
  }

  if (error) {
    return <p style={{ color: 'red' }}>{error}</p>;
  }

  return (
    <div className="violations-list-container">
      {/* 컬럼 제목 */}
      <div className="violation-header">
         <div className="violation-cell-header id">ID</div>
         <div className="violation-cell-header description">설명</div>
         <div className="violation-cell-header detected-at">탐지 시간</div>
      </div>

      {/* 데이터 목록 또는 "적재물 내역이 없습니다." 메시지 */}
      {obstacles.length > 0 ? (
        obstacles.map((obstacle) => (
         <div key={obstacle.id} className="violation-row">
             <div className="violation-cell id">{obstacle.id}</div>
             <div className="violation-cell description">{obstacle.description}</div>
             <div className="violation-cell detected-at">
                 {new Date(obstacle.detected_at).toLocaleString()}
             </div>
         </div>
        ))
        ) : (
            <p style={{ textAlign: 'center', padding: '20px', color: 'gray' }}>
            적재물 내역이 없습니다.
            </p>
        )}
    </div>
  );
};


export default ObstacleViolationsList;
