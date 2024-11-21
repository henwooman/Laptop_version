import { useEffect, useState } from 'react';
import { useLocation } from 'react-router-dom';
import Btnbar from '../components/Btnbar';
import axios from 'axios';

export const Myparking_place = () => {
    const location = useLocation();  // useLocation을 사용하여 URL 정보 가져오기
    const [parkingInfo, setParkingInfo] = useState({
        totalSpaces: 0,
        availableSpaces: 0,
        disabledSpaces: 0,
    }); // 주차장 정보 상태 관리
    const [coordinates, setCoordinates] = useState([]); // coordinates 상태 추가
    const [kakaoSdkLoaded, setKakaoSdkLoaded] = useState(false); // SDK 로드 여부 상태 추가

    // 쿼리 파라미터에서 위도(latitude), 경도(longitude), 주소(address) 값 가져오기
    const queryParams = new URLSearchParams(location.search);
    const latitude = queryParams.get('latitude');  // 위도 값 가져오기
    const longitude = queryParams.get('longitude');  // 경도 값 가져오기
    const address = queryParams.get('address');  // 주소 값 가져오기

    // 쿼리 파라미터 확인
    useEffect(() => {
        console.log("Latitude:", latitude, "Longitude:", longitude, "Address:", address);
    }, [latitude, longitude, address]);

    useEffect(()=>{
        if (latitude && longitude && !isNaN(latitude) && !isNaN(longitude)){
            initializeKakaoRoadview(parseFloat(latitude),parseFloat(longitude)); 
        }else{
            console.error('위도와 경도 정보가 유효하지 않습니다.');
        }
    },[latitude,longitude]); // 위도와 경도 값이 변경될 때 마다 로드뷰 업데이트

    // 랜덤 값을 생성하는 함수
    useEffect(() => {
        const generateRandomValue = () => Math.floor(Math.random() * 30) + 1;
        const generateRandomValue1 = () => Math.floor(Math.random() * 10) + 1;
        const generateRandomValue2 = () => Math.floor(Math.random() * 7) + 1;
        setParkingInfo({
            totalSpaces: generateRandomValue(),
            availableSpaces: generateRandomValue1(),
            disabledSpaces: generateRandomValue2(),
        });
    }, []);

    // 서버에서 DB의 경도와 위도 가져오기
    const fetchParkingData = async () => {
        try {
            const response = await axios.get('http://localhost:4000/parkingSearch/parkinglist3');
            setCoordinates(response.data.coordinates);
        } catch (error) {
            console.error('서버에서 데이터 가져오기 실패', error);
        }
    };

    useEffect(() => {
        fetchParkingData(); // 서버에서 데이터 가져오기
    }, []);

    const initializeKakaoRoadview = (latitude, longitude) => {
        // 카카오 SDK가 이미 로드된 경우 바로 로드뷰 초기화
        if (window.kakao) {
            console.log('Kakao SDK 이미 로드됨');
            setKakaoSdkLoaded(true); // SDK가 이미 로드되었으면 바로 상태 변경
        } else {
            // 카카오 SDK가 로드되지 않은 경우에만 스크립트 로드
            const script = document.createElement('script');
            
            // 여기에 자신의 카카오 REST API Key를 넣습니다.
            const kakaoAppKey = process.env.REACT_APP_PARKINGSERVER_API_KEY;  // 본인의 카카오 API Key로 변경
            
            // SDK를 로드할 때 API Key를 추가
            script.src = `https://developers.kakao.com/sdk/js/kakao.js?appkey=${kakaoAppKey}`;
            script.async = true; // 비동기 로드로 설정
            script.onload = () => {
                // SDK가 정상적으로 로드되었을 때
                if (window.kakao) {
                    console.log('Kakao SDK Loaded');
                    setKakaoSdkLoaded(true); // SDK 로드 완료 상태 변경
                } else {
                    console.error('Kakao SDK가 로드되지 않았습니다.');
                }
            };
    
            // SDK 로드 실패 시 에러 처리
            script.onerror = () => {
                console.error('카카오 SDK 로드 실패');
            };
    
            document.body.appendChild(script); // 스크립트 태그를 문서에 추가하여 로드 시도
        }
    };
    // 카카오 SDK가 로드되었을 때 로드뷰 초기화
    useEffect(() => {
        if (kakaoSdkLoaded && latitude && longitude) {
            const roadviewContainer = document.getElementById('roadviewContainer');
            const position = new window.kakao.maps.LatLng(parseFloat(latitude), parseFloat(longitude));

            // 로드뷰 객체 생성
            const roadviewClient = new window.kakao.maps.RoadviewClient();
            const roadview = new window.kakao.maps.Roadview(roadviewContainer);

            // 로드뷰 토글 함수
            roadviewClient.getNearestPanoId(position, 50, (panoId) => {
                if (panoId === null) {
                    roadviewContainer.style.display = 'none'; // 로드뷰를 넣은 컨테이너 숨기기
                } else {
                    roadviewContainer.style.display = 'block'; // 로드뷰를 넣은 컨테이너 보이게 하기
                    roadview.setPanoId(panoId, position); // 로드뷰에 해당 위치 파노 ID 설정
                    roadview.relayout(); // 로드뷰 컨테이너 재배열
                }
            });
        }
    }, [kakaoSdkLoaded, latitude, longitude]); // kakaoSdkLoaded가 true로 변경되었을 때만 로드뷰 초기화

    return (
        <div className="parkinglot-status-container">
            <div className="parkinglot-address-container">
                <div className="parkinglot-address">{address || '주차장 주소 나타내는 부분'}</div>
                <img className="parkinglot-star-icon" alt="스타 아이콘" src="images/stars.png" />
            </div>
            <div id="roadviewContainer" style={{ width: '100%', height: '180px', display: 'block', top: '0px' }}></div>  {/* 로드뷰 표시 */}
            <div className="parkinglot-info-button">주차장 정보</div>
            <div className="parkinglot-info-details">
                <div className="parkinglot-info-text">총 주차 구역수: <strong>{parkingInfo.totalSpaces}</strong></div>
                <div className="parkinglot-info-text">현재 주차 가능 수: <strong>{parkingInfo.availableSpaces}</strong></div>
                <div className="parkinglot-info-text">장애인전용주차 가능 수: <strong>{parkingInfo.disabledSpaces}</strong></div>
            </div>
            <div className="parkinglot-event-button">현재 이벤트</div>
            <div className="parkinglot-event-details">
                <div className="parkinglot-event-text">이중주차:<strong></strong> </div>
                <div className="parkinglot-event-text">통행방해:<strong></strong> </div>
            </div>
            <Btnbar />
        </div>
    );
};

export default Myparking_place;
