import React, { useState, useEffect } from 'react';

const ParkingList_Item = ({ number, address, onAddressClick }) => {
    // 로컬 저장소에서 해당 아이템에 대한 즐겨찾기 상태를 가져옴
    const [isFavorite, setIsFavorite] = useState(() => {
        const savedFavorite = localStorage.getItem(`favorite_${number}`);
        return savedFavorite === 'true'; // true로 저장된 경우 true, 아니면 false
    });

    // 즐겨찾기 상태를 변경하는 함수
    const toggleFavorite = () => {
        const newFavoriteStatus = !isFavorite;
        setIsFavorite(newFavoriteStatus);

        // 로컬 저장소에 변경된 상태를 저장
        localStorage.setItem(`favorite_${number}`, newFavoriteStatus.toString());
    };

    return (
        <div className="parking_item_container">
            <span className="parking_item_number">{number}</span>
            <div className="parking_item_address" onClick={onAddressClick}>{address}</div>
            <img
                className={`parking_item_star_icon ${isFavorite ? 'favorite' : ''}`}
                alt="star"
                src="images/stars.png"
                onClick={toggleFavorite}
            />
        </div>
    );
}

export default ParkingList_Item;
