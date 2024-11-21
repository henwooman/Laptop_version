import React, { useState, useEffect, useCallback } from 'react';
import { Map, MapMarker, MapTypeControl, ZoomControl } from "react-kakao-maps-sdk";
import useKakaoLoader from './useKakaoLoder';

const ParkingMa = ({ searchTerm, setCoordinates }) => {
  useKakaoLoader();

  const [markers, setMarkers] = useState([]);
  const [map, setMap] = useState(null);
  const [selectedPlace, setSelectedPlace] = useState(null);

  const displayPlaces = useCallback((places) => {
    if (!map) return;

    const bounds = new window.kakao.maps.LatLngBounds();
    const newMarkers = [];
    let newCoordinates = [];

    places.forEach((place) => {
      const position = new window.kakao.maps.LatLng(place.y, place.x);
      const marker = new window.kakao.maps.Marker({
        position: position,
        map: map
      });

      newCoordinates.push({
        latitude: place.y, // 위도
        longitude: place.x  // 경도
      });

      newMarkers.push({
        position,
        content: place.place_name,
        address: place.road_address || place.address_name
      });

      window.kakao.maps.event.addListener(marker, 'click', function () {
        setSelectedPlace({
          name: place.place_name,
          address: place.road_address || place.address_name,
          marker
        });
      });

      bounds.extend(position);
    });

    setMarkers(newMarkers);
    setCoordinates(newCoordinates); // 위도, 경도 상태 업데이트
    map.setBounds(bounds);
  }, [map, setCoordinates]);

  useEffect(() => {
    if (!map || !searchTerm) return;

    const ps = new window.kakao.maps.services.Places();
    ps.keywordSearch(searchTerm, (data, status) => {
      if (status === window.kakao.maps.services.Status.OK) {
        displayPlaces(data);
      } else {
        console.error('검색 실패', status);
      }
    });
  }, [map, searchTerm, displayPlaces]);

  return (
    <div>
      <Map
        id="map"
        center={{ lat: 35.146738, lng: 126.922251 }}
        // transform: "translateX(-15px) => 지도 내부의 버튼이 잘려서 왼쪽으로 이동시키는 css
        style={{ width: "380px", height: "300px", transform: "translateX(-15px)" }}
        level={3}
        onCreate={setMap}
      >
        <MapTypeControl position={"TOPRIGHT"} />
        <ZoomControl position={"RIGHT"} />
        {markers.map((marker, index) => (
          <MapMarker
            key={`marker-${index}`}
            position={marker.position}
            title={marker.content}
            onClick={() => {
              setSelectedPlace({
                name: marker.content,
                address: marker.address
              });
            }}
          />
        ))}
      </Map>
    </div>
  );
};

export default ParkingMa;
