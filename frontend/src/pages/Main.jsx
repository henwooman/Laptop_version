import React, { useState } from "react";
import { useNavigate } from "react-router-dom";
import axios from "axios";

// CameraModal 컴포넌트
const CameraModal = ({ onClose }) => {
    const videoRef = React.useRef(null);
    const mediaRecorderRef = React.useRef(null);
    const [recording, setRecording] = React.useState(false);
    const [capturedImage, setCapturedImage] = React.useState(null);
    const [message, setMessage] = React.useState(""); // 사용자 메시지 상태
    const [analysisResult, setAnalysisResult] = React.useState(null); // 분석 결과 상태
    const [assistCheckEnabled, setAssistCheckEnabled] = React.useState(false); // 보조기구 체크 활성화 상태
    const recordedChunks = React.useRef([]);

    // 카메라 활성화
    const startCamera = async () => {
        try {
            const stream = await navigator.mediaDevices.getUserMedia({ video: true });
            videoRef.current.srcObject = stream;
            setMessage("카메라가 활성화되었습니다.");
        } catch (error) {
            console.error("카메라 접근 실패:", error);
            setMessage("카메라 접근에 실패했습니다.");
        }
    };

    // 녹화 시작
    const startRecording = () => {
        setRecording(true);
        recordedChunks.current = [];
        const stream = videoRef.current.srcObject;
        mediaRecorderRef.current = new MediaRecorder(stream, {
            mimeType: "video/webm",
        });

        mediaRecorderRef.current.ondataavailable = (event) => {
            if (event.data.size > 0) recordedChunks.current.push(event.data);
        };

        mediaRecorderRef.current.start();
        setMessage("녹화가 시작되었습니다.");
    };

    // 녹화 종료 및 서버 전송
    const stopRecording = async () => {
        setRecording(false);
        mediaRecorderRef.current.stop();

        mediaRecorderRef.current.onstop = async () => {
            const blob = new Blob(recordedChunks.current, { type: "video/mp4" });
            const formData = new FormData();
            // formData.append("file", blob, "recorded_video.mp4");
            formData.append("video_filename", "recorded_video.mp4");

            try {
                const response = await axios.post("http://localhost:8000/upload_videos/", formData, {
                    headers: { "Content-Type": "multipart/form-data" },
                });

                if (response.status === 200) {
                    setMessage("영상이 서버로 전송 및 분석되었습니다.");
                    setAnalysisResult(response.data); // 서버 응답 결과 저장
                    console.log("서버 응답:", response.data);

                    if (response.data.status === "장애인 차량") {
                        setAssistCheckEnabled(true); // 보조기구 탐지를 활성화
                    }
                }
            } catch (error) {
                console.error("영상 전송 실패:", error);
                if (error.response) {
                    console.error("서버 응답:", error.response.data);
                    setMessage(`서버 오류: ${error.response.data.detail}`);
                } else {
                    setMessage("영상 전송에 실패했습니다.");
                }
            }
        };
    };

    // 캡처 및 서버 전송
    const captureImage = () => {
        const canvas = document.createElement("canvas");
        const video = videoRef.current;

        canvas.width = video.videoWidth;
        canvas.height = video.videoHeight;
        const ctx = canvas.getContext("2d");
        ctx.drawImage(video, 0, 0, canvas.width, canvas.height);

        const imageDataUrl = canvas.toDataURL("image/jpeg");
        setCapturedImage(imageDataUrl);

        const blob = dataURLToBlob(imageDataUrl);
        const formData = new FormData();
        formData.append("file", blob, "captured_image.jpg");
        formData.append("video_filename", "captured_image.jpg");

        axios
            .post("http://localhost:8000/upload_videos/", formData, {
                headers: { "Content-Type": "multipart/form-data" },
            })
            .then((response) => {
                setMessage("이미지가 서버로 전송 및 분석되었습니다.");
                setAnalysisResult(response.data); // 서버 응답 결과 저장
                console.log("서버 응답:", response.data);

                if (response.data.status === "장애인 차량") {
                    setAssistCheckEnabled(true); // 보조기구 탐지를 활성화
                }
            })
            .catch((error) => {
                console.error("이미지 전송 실패:", error);
                if (error.response) {
                    console.error("서버 응답:", error.response.data);
                    setMessage(`서버 오류: ${error.response.data.detail}`);
                } else {
                    setMessage("이미지 전송에 실패했습니다.");
                }
            });
    };

    // 보조기구 체크
    const assistCheck = () => {
        const canvas = document.createElement("canvas");
        const video = videoRef.current;

        canvas.width = video.videoWidth;
        canvas.height = video.videoHeight;
        const ctx = canvas.getContext("2d");
        ctx.drawImage(video, 0, 0, canvas.width, canvas.height);

        const imageDataUrl = canvas.toDataURL("image/jpeg");
        const blob = dataURLToBlob(imageDataUrl);
        const formData = new FormData();
        formData.append("file", blob, "assist_check.jpg");

        axios
            .post("http://localhost:8000/assist_check/", formData, {
                headers: { "Content-Type": "multipart/form-data" },
            })
            .then((response) => {
                setMessage("보조기구 탐지가 완료되었습니다.");
                console.log("보조기구 분석 결과:", response.data);
                setAnalysisResult(response.data);
            })
            .catch((error) => {
                console.error("보조기구 탐지 실패:", error);
                if (error.response) {
                    console.error("서버 응답:", error.response.data);
                    setMessage(`서버 오류: ${error.response.data.detail}`);
                } else {
                    setMessage("보조기구 탐지에 실패했습니다.");
                }
            });
    };

    // DataURL -> Blob 변환
    const dataURLToBlob = (dataURL) => {
        const parts = dataURL.split(",");
        const byteString = atob(parts[1]);
        const mimeString = parts[0].split(":")[1].split(";")[0];
        const arrayBuffer = new ArrayBuffer(byteString.length);
        const uint8Array = new Uint8Array(arrayBuffer);
        for (let i = 0; i < byteString.length; i++) {
            uint8Array[i] = byteString.charCodeAt(i);
        }
        return new Blob([uint8Array], { type: mimeString });
    };

    // 카메라 종료
    const stopCamera = () => {
        const stream = videoRef.current.srcObject;
        const tracks = stream.getTracks();
        tracks.forEach((track) => track.stop());
        onClose();
        setMessage("카메라가 종료되었습니다.");
    };

    return (
        <div className="camera-modal">
            <div className="camera-container">
                <video ref={videoRef} autoPlay playsInline style={{ width: "100%" }} />
                <div className="camera-controls">
                    {!recording && <button onClick={startCamera}>카메라 활성화</button>}
                    {!recording && <button onClick={startRecording}>녹화 시작</button>}
                    {recording && <button onClick={stopRecording}>녹화 중지</button>}
                    <button onClick={captureImage}>캡처</button>
                    <button onClick={stopCamera}>종료</button>
                    {assistCheckEnabled && <button onClick={assistCheck}>보조기구 탐지</button>}
                </div>
                {message && <p className="message">{message}</p>}
            </div>
            {capturedImage && <img src={capturedImage} alt="Captured" style={{ marginTop: "10px", width: "100%" }} />}
            {analysisResult && (
                <div className="analysis-result">
                    <h3>분석 결과</h3>
                    <p>상태: {analysisResult.status}</p>
                    {analysisResult.plate_number && <p>번호판: {analysisResult.plate_number}</p>}
                    {analysisResult.assist_status && <p>보조기구 상태: {analysisResult.assist_status}</p>}
                </div>
            )}
        </div>
    );
};

// Main 컴포넌트
const Main = () => {
    const navigate = useNavigate();
    const [showCamera, setShowCamera] = useState(false);

    const handleLogoClick = () => {
        navigate("/login");
    };

    const handleCameraButtonClick = () => {
        setShowCamera(true);
    };

    const handleCameraClose = () => {
        setShowCamera(false);
    };

    return (
        <div className="main_background">
            <img src="/images/start_IMG.png" alt="background_image" className="background_image" />
            <div className="centered_logo_container" onClick={handleLogoClick}>
                <img src="/images/logo.jpg" alt="centered_logo_image" className="centered_logo_image" />
            </div>
            <div onClick={handleCameraButtonClick} style={{ cursor: "pointer" }}>
                <img src="/images/button.png" alt="centered_button_image" className="centered_button_image" />
            </div>
            {showCamera && <CameraModal onClose={handleCameraClose} />}
        </div>
    );
};

export default Main;





// import React, { useState } from "react";
// import { useNavigate } from "react-router-dom";
// import axios from 'axios';

// // CameraModal 컴포넌트
// const CameraModal = ({ onClose }) => {
//     console.log("CameraModal rendered");
//     const videoRef = React.useRef(null);
//     const mediaRecorderRef = React.useRef(null);
//     const [recording, setRecording] = React.useState(false);
//     const [capturedImage, setCapturedImage] = React.useState(null);
//     const [message, setMessage] = React.useState(""); // 사용자 메시지 상태
//     const [analysisResult, setAnalysisResult] = React.useState(null); // 분석 결과 상태
//     const recordedChunks = React.useRef([]);

//     // 카메라 활성화
//     const startCamera = async () => {
//         console.log("Attempting to start camera...");
//         try {
//             console.log("Trying to access the camera...");
//             const stream = await navigator.mediaDevices.getUserMedia({ video: true });
//             videoRef.current.srcObject = stream;
//             setMessage("카메라가 활성화되었습니다.");
//             console.log("Camera access successful");
//         } catch (error) {
//             console.error("카메라 접근 실패:", error);
//             setMessage("카메라 접근에 실패했습니다.");
//         }
//     };

//     // 녹화 시작
//     const startRecording = () => {
//         setRecording(true);
//         recordedChunks.current = [];
//         const stream = videoRef.current.srcObject;
//         mediaRecorderRef.current = new MediaRecorder(stream, {
//             mimeType: "video/webm",
//         });

//         mediaRecorderRef.current.ondataavailable = (event) => {
//             if (event.data.size > 0) recordedChunks.current.push(event.data);
//         };

//         mediaRecorderRef.current.start();
//         setMessage("녹화가 시작되었습니다.");
//     };

//     // 녹화 종료 및 서버 전송
//     const stopRecording = async () => {
//         setRecording(false);
//         mediaRecorderRef.current.stop();

//         mediaRecorderRef.current.onstop = async () => {
//             const blob = new Blob(recordedChunks.current, { type: "video/mp4" });
//             const formData = new FormData();
//             formData.append("file", blob, "recorded_video.mp4");

//             try {   
//                 const response = await axios.post("http://localhost:8000/upload_videos", formData, {
//                     headers: { "Content-Type": "multipart/form-data" },
//                 });

//                 if (response.status === 200) {
//                     setMessage("영상이 서버로 전송 및 분석되었습니다.");
//                     setAnalysisResult(response.data); // 서버 응답 결과 저장
//                     console.log("서버 응답:", response.data);
//                 }
//             } catch (error) {
//                 console.error("영상 전송 실패:", error);
//                 setMessage("영상 전송에 실패했습니다.");
//             }
//         };
//     };

//     // 캡처 및 서버 전송
//     const captureImage = () => {
//         const canvas = document.createElement("canvas");
//         const video = videoRef.current;

//         canvas.width = video.videoWidth;
//         canvas.height = video.videoHeight;
//         const ctx = canvas.getContext("2d");
//         ctx.drawImage(video, 0, 0, canvas.width, canvas.height);

//         const imageDataUrl = canvas.toDataURL("image/jpeg");
//         setCapturedImage(imageDataUrl);

//         // 서버 전송
//         const blob = dataURLToBlob(imageDataUrl);
//         const formData = new FormData();
//         formData.append("file", blob, "captured_image.jpg");

//         axios
//             .post("http://localhost:8000/upload_videos", formData, {
//                 headers: { "Content-Type": "multipart/form-data" },
//             })
//             .then((response) => {
//                 setMessage("이미지가 서버로 전송 및 분석되었습니다.");
//                 setAnalysisResult(response.data); // 서버 응답 결과 저장
//                 console.log("서버 응답:", response.data);
//             })
//             .catch((error) => {
//                 console.error("이미지 전송 실패:", error);
//                 if (error.response) {
//                     console.error("서버 응답:", error.response.data);
//                     setMessage(`서버 오류: ${error.response.data.detail}`);
//                 } else {
//                     setMessage("이미지 전송에 실패했습니다.");
//                 }
//             });
            
//     };

//     // DataURL -> Blob 변환
//     const dataURLToBlob = (dataURL) => {
//         const parts = dataURL.split(",");
//         const byteString = atob(parts[1]);
//         const mimeString = parts[0].split(":")[1].split(";")[0];
//         const arrayBuffer = new ArrayBuffer(byteString.length);
//         const uint8Array = new Uint8Array(arrayBuffer);
//         for (let i = 0; i < byteString.length; i++) {
//             uint8Array[i] = byteString.charCodeAt(i);
//         }
//         return new Blob([uint8Array], { type: mimeString });
//     };

//     // 카메라 종료
//     const stopCamera = () => {
//         const stream = videoRef.current.srcObject;
//         const tracks = stream.getTracks();
//         tracks.forEach((track) => track.stop());
//         onClose();
//         setMessage("카메라가 종료되었습니다.");
//     };

//     return (
//         <div className="camera-modal">
//             <div className="camera-container">
//                 <video ref={videoRef} autoPlay playsInline style={{ width: "100%" }} onCanPlay={() => console.log("Video element ready")} />
//                 <div className="camera-controls">
//                     {!recording && <button onClick={startCamera}>카메라 활성화</button>}
//                     {!recording && <button onClick={startRecording}>녹화 시작</button>}
//                     {recording && <button onClick={stopRecording}>녹화 중지</button>}
//                     <button onClick={captureImage}>캡처</button>
//                     <button onClick={stopCamera}>종료</button>
//                 </div>
//                 {message && <p className="message">{message}</p>}
//             </div>

//             {capturedImage && <img src={capturedImage} alt="Captured" style={{ marginTop: "10px", width: "100%" }} />}

//             {/* 분석 결과 렌더링 */}
//             {analysisResult && (
//                 <div className="analysis-result">
//                     <h3>분석 결과</h3>
//                     <p>번호판: {analysisResult.plate_number}</p>
//                     <p>신뢰도: {analysisResult.confidence}</p>
//                     <img src={analysisResult.boxed_image_url} alt="분석된 결과" style={{ marginTop: "10px", width: "100%" }} />
//                 </div>
//             )}
//         </div>
//     );
// };

// // Main 컴포넌트
// const Main = () => {
//     const navigate = useNavigate();
//     const [showCamera, setShowCamera] = useState(false);

//     const handleLogoClick = () => {
//         console.log("로고 클릭");
//         navigate("/login");
//     };

//     const handleCameraButtonClick = () => {
//         console.log("Camera button clicked");
//         setShowCamera(true);
//     };

//     const handleCameraClose = () => {
//         setShowCamera(false);
//     };

//     return (
//         <div className="main_background">
//             <img src="/images/start_IMG.png" alt="background_image" className="background_image" />
//             <div className="centered_logo_container" onClick={handleLogoClick}>
//                 <img src="/images/logo.jpg" alt="centered_logo_image" className="centered_logo_image" />
//             </div>
//             <div onClick={handleCameraButtonClick} style={{ cursor: "pointer" }}>
//                 <img src="/images/button.png" alt="centered_button_image" className="centered_button_image" />
//             </div>
//             {showCamera && <CameraModal onClose={handleCameraClose} />}
//         </div>
//     );
// };

// export default Main;
