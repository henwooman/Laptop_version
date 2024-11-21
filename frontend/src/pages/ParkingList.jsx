import React, { useEffect, useState } from 'react';
import '../index.css';
import { useNavigate } from 'react-router-dom';
import ParkingList_Item from '../components/ParkingList_Item';
import axios from 'axios';

const ParkingList = () => {
  const [id, setId] = useState(); // 세션에서 ID를 가져와서 저장
  const [coordinates, setCoordinates] = useState([]); // 서버에서 받은 위도/경도 및 주소 데이터
  const navigate = useNavigate();

  // 세션 값 가져오는 코드
  useEffect(() => {
    setId(sessionStorage.getItem('id')); // 세션에서 ID 가져오기
    addresslist(); // 서버에서 즐겨찾기 목록을 받아오는 함수 호출
  }, []);

  // 서버에서 즐겨찾기 목록(위도, 경도)을 받아오는 함수
  const addresslist = async () => {
    try {
      const response = await axios.get('http://localhost:4000/parkingSearch/parkinglist2'); // 서버에서 위도/경도 데이터 요청
      if (response.data && response.data.coordinates) {
        // 서버에서 받은 위도/경도 데이터를 상태에 저장
        const coordinatesWithAddress = await Promise.all(response.data.coordinates.map(async (item) => {
          const address = await getAddress(item.latitude, item.longitude); // 주소 변환 API 호출
          return { ...item, address }; // 주소를 추가한 위도/경도 데이터 반환
        }));
        setCoordinates(coordinatesWithAddress); // 주소가 포함된 데이터를 상태에 저장
      } else {
        console.log('위도/경도 데이터가 없습니다.');
      }
    } catch (error) {
      console.log('서버에서 데이터 가져오기 실패:', error);
    }
  };

  // 위도/경도를 사용하여 주소를 얻는 함수 (카카오 API 호출)
  const getAddress = async (latitude, longitude) => {
    const kakaoApikey = process.env.REACT_APP_PARKINGSERVER_API_KEY;
    try {
      const url = `https://dapi.kakao.com/v2/local/geo/coord2address.json?x=${longitude}&y=${latitude}`;
      const response = await axios.get(url, {
        headers: {
          Authorization: `KakaoAK ${kakaoApikey}`,
        }
      });
  
      // API 응답 확인
      console.log("카카오 API 응답: ", response.data);
  
      // 카카오 API에서 반환하는 주소 정보
      const address = response.data.documents[0]?.address;
      if (address) {
        return address.address_name; // 주소 반환
      } else {
        console.log("주소 변환 실패: 주소 정보가 없습니다.");
        return '주소를 찾을 수 없습니다.';
      }
    } catch (error) {
      // 에러 응답 확인
      console.error("주소 변환 실패:", error.response ? error.response.data : error.message);
      return '주소를 찾을 수 없습니다.';
    }
  };

  // handleParkingClick 함수 수정
const handleParkingClick = (latitude, longitude, address) => {
  console.log(`클릭된 주소: ${address}, 위도: ${latitude}, 경도: ${longitude}`);
  // 쿼리 파라미터로 위도와 경도 전달
  navigate(`/Myparking_place?latitude=${latitude}&longitude=${longitude}`);
};

  return (
    <div className="parking_list_container">
      <div className="parking_list_header">
        <div className="parking_list_header_number">번호</div>
        <p className="parking_list_header_address">주소</p>
      </div>
      <hr />
      {coordinates.length === 0 ? (
        <p>데이터가 없습니다.</p> // 데이터가 없을 때 표시할 메시지
      ) : (
        coordinates.map((item, index) => (
          <ParkingList_Item
            key={index}
            number={index + 1} // 번호 설정
            address={item.address || `위도: ${item.latitude}, 경도: ${item.longitude}`} // 주소 표시, 없으면 위도/경도 표시
            onAddressClick={() => handleParkingClick(item.latitude, item.longitude, item.address)} // 클릭 시 이동
          />
        ))
      )}
    </div>
  );
};

export default ParkingList;
