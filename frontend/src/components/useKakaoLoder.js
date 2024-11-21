import { useKakaoLoader as useKakaoLoaderOrigin } from "react-kakao-maps-sdk";
import { useEffect } from "react";

const useKakaoLoader = () => {
    const loaded = useKakaoLoaderOrigin({
        appkey: process.env.REACT_APP_PARKING_API_KEY,
        libraries: ["clusterer", "drawing", "services"],
    });

    useEffect(() => {
        console.log("Kakao Maps API loaded:", loaded);
    }, [loaded]);

    return loaded;
}

export default useKakaoLoader;
