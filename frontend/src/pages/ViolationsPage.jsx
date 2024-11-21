import React, { useState } from 'react';
import { useNavigate } from 'react-router-dom';
import ViolationsList from '../components/ViolationsList';
import ObstacleViolationsList from '../components/ObstacleViolationsList';
import Violation_tab_btn from '../components/Violation_tab_btn';

const ViolationsPage = () => {
  const navigate = useNavigate();
  const [activeTab, setActiveTab] = useState('violation');
  const [violationsData, setViolationsData] = useState([]);
  const [obstacleData, setObstacleData] = useState([]);

  const handleDownloadPageNavigation = () => {
    const dataToDownload = activeTab === 'violation' ? violationsData : obstacleData;
    navigate('/download', { state: { previewData: dataToDownload } });
  };

  return (
    <div className="violations-page-container">
      <div className="violations-page-tabs">
        <Violation_tab_btn
          isActive={activeTab === 'violation'}
          onClick={() => setActiveTab('violation')}
          label="위반차량"
        />
        <Violation_tab_btn
          isActive={activeTab === 'obstacle'}
          onClick={() => setActiveTab('obstacle')}
          label="적재물"
        />
      </div>

      <div className="violations-page-content">
        {activeTab === 'violation' ? (
          <ViolationsList setData={setViolationsData} />
        ) : (
          <ObstacleViolationsList setData={setObstacleData} />
        )}
      </div>

      <div className="violations-page-download-section">
        <div className="download-button-container">
          <button className="to_download" onClick={handleDownloadPageNavigation}>
            다운로드페이지로
          </button>
        </div>
      </div>
    </div>
  );
};

export default ViolationsPage;
