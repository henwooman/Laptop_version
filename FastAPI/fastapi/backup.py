from fastapi import FastAPI, UploadFile, File, HTTPException, Depends, Request
from fastapi.responses import JSONResponse, FileResponse
import shutil
from pathlib import Path
import torch
from PIL import Image
import logging
import cv2
import numpy as np
from paddleocr import PaddleOCR
from sqlalchemy import create_engine, Column, Integer, String, Boolean, DateTime
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker, Session
from datetime import datetime
import pathlib
import time
import re

# 로깅 설정
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('app.log'),
        logging.StreamHandler()
    ]
)

pathlib.PosixPath = pathlib.WindowsPath

# FastAPI 설정
app = FastAPI()

# CORS 설정
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)



# 디렉토리 설정
UPLOAD_DIR = Path("uploads")
UPLOAD_DIR.mkdir(exist_ok=True)

# 데이터베이스 설정
DB_CONFIG = {
    'user': 'Insa5_JSB_final_1',
    'password': 'aischool1',
    'host': 'project-db-stu3.smhrd.com',
    'database': 'Insa5_JSB_final_1',
    'port': '3307'
}

################ 데이터베이스 모델 ################
# DB 모델 정의
class User(Base):
    __tablename__ = 'USER'
    user_id = Column(Integer, primary_key=True, index=True)
    car_number = Column(String(20), unique=True, nullable=False)
    handicap = Column(Boolean, default=False)

class Violation(Base):
    __tablename__ = 'VIOLATION'
    violation_id = Column(Integer, primary_key=True, index=True)
    violation_number = Column(String(20))
    filename = Column(String(255), unique=True)
    url = Column(String(255))
    upload_time = Column(DateTime)

class ObstacleViolation(Base):
    __tablename__ = 'OBSTACLE_VIOLATION'
    id = Column(Integer, primary_key=True, index=True)
    description = Column(String(50))  # 적재물1, 적재물2 등을 저장
    detected_at = Column(DateTime, default=datetime.utcnow)
    video_url = Column(String(255)) 

# 모델 경로
plate_model_path = "C:/Users/USER/Desktop/laptop_LastProject/four-people/FastAPI/models/plate_ocr.pt"
assist_model_path = "C:/Users/USER/Desktop/laptop_LastProject/four-people/FastAPI/models/assist_device1.pt"
obstacle_model_path = "C:/Users/USER/Desktop/laptop_LastProject/four-people/FastAPI/models/parking_obj.pt"

# 모델 로드
try:
    plate_model = torch.hub.load('ultralytics/yolov5', 'custom', path=plate_model_path, force_reload=True)
    assist_model = torch.hub.load('ultralytics/yolov5', 'custom', path=assist_model_path, force_reload=True)
    obstacle_model = torch.hub.load('ultralytics/yolov5', 'custom', path=obstacle_model_path, force_reload=True)
    plate_model.eval()
    assist_model.eval()
    obstacle_model.eval()
except Exception as e:
    logging.error(f"모델 로드 실패: {e}")
    raise

# DB 연결 함수
def save_to_db(filename: str, violation_number: str = None):
    conn = mysql.connect(**DB_CONFIG)
    cursor = conn.cursor()
    try:
        upload_time = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        sql = 'INSERT INTO VIOLATION (filename, upload_time, violation_number) VALUES (%s, %s, %s)'
        cursor.execute(sql, (filename, upload_time, violation_number))
        conn.commit()
        logging.info(f"DB 저장 성공: {filename}")
    except mysql.Error as e:
        conn.rollback()
        logging.error(f"DB 저장 실패: {e}")
        raise HTTPException(status_code=500, detail="DB 저장 실패")
    finally:
        cursor.close()
        conn.close()

# 동영상에서 첫 프레임 추출 함수
def extract_frame_from_video(video_path: str) -> str:
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise ValueError("동영상을 열 수 없습니다.")
    ret, frame = cap.read()
    if not ret:
        raise ValueError("동영상에서 프레임을 읽을 수 없습니다.")
    frame_path = video_path.replace(".mp4", "_frame.jpg")
    cv2.imwrite(frame_path, frame)
    cap.release()
    return frame_path

# 바운딩박스 그리기
def draw_bounding_boxes(image_path: str, boxes: list):
    image = Image.open(image_path)
    draw = ImageDraw.Draw(image)
    for (x1, y1, x2, y2) in boxes:
        draw.rectangle([x1, y1, x2, y2], outline="red", width=3)
    output_path = image_path.replace(".jpg", "_with_boxes.jpg")
    image.save(output_path)
    return output_path

# 번호판 분석 함수
def analyze_plate_image(file_path: str):
    image = Image.open(file_path)
    results = plate_model(image)
    boxes = []
    plate_number = None
    confidence = 0.0
    for *box, conf, cls in results.xyxy[0]:
        x1, y1, x2, y2 = map(int, box)
        boxes.append((x1, y1, x2, y2))
        if conf > confidence:
            confidence = conf
            plate_number = "가상번호1234"
    boxed_image_path = draw_bounding_boxes(file_path, boxes)
    return plate_number, float(confidence), boxed_image_path

# 적재물 탐지 함수
def detect_obstacle(file_path: str) -> bool:
    image = Image.open(file_path)
    results = obstacle_model(image)
    detected = any(results.xyxy[0])  # 적재물 존재 여부
    return detected

@app.post("/upload_videos/")
async def upload_file(request: Request, file: UploadFile = File(...), db: Session = Depends(get_db)):
    try:
        print("\n=== 파일 업로드 및 번호판 분석 시작 ===")
        
        form_data = await request.form()
        video_filename = form_data.get("video_filename")
        
        if not video_filename:
            print("비디오 파일명이 누락되었습니다.")
            return JSONResponse(content={"error": "비디오 파일명이 누락되었습니다."}, status_code=400)

        # 파일 저장
        file_location = UPLOAD_DIR / file.filename
        with open(file_location, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)
        print(f"파일 저장 완료: {file_location}")

        # 1. 번호판 분석
        plate_number, confidence = analyze_plate_image(str(file_location))
        print(f"번호판 분석 결과: {plate_number}, 신뢰도: {confidence}")

        # violation_record 찾기
        violation_record = db.query(Violation).filter(
            Violation.filename == video_filename
        ).first()

        if not violation_record:
            return JSONResponse(content={"error": "비디오 파일 기록을 찾을 수 없습니다"}, status_code=404)
        
        if confidence < 0.5:
            print("번호판 인식 실패 - 적재물 탐지 시작")
            if detect_obstacle(str(file_location)):
                print("적재물 감지 완료")
                
                # 마지막 적재물 번호 조회
                last_obstacle = db.query(ObstacleViolation).order_by(
                    ObstacleViolation.id.desc()
                ).first()
                
                next_number = 1
                if last_obstacle:
                    # 마지막 description에서 숫자 추출
                    match = re.search(r'\d+', last_obstacle.description)
                    if match:
                        next_number = int(match.group()) + 1
                
                # **VIOLATION 테이블에서 URL 가져오기**
                violation_record = db.query(Violation).filter(
                    Violation.filename == video_filename
                ).first()

                if not violation_record or not violation_record.url:
                    print("VIOLATION 테이블에서 URL 정보를 찾을 수 없습니다.")
                    return JSONResponse(content={"error": "VIOLATION 테이블에 URL 정보 없음"}, status_code=404)

                # 새로운 적재물 위반 기록 생성
                new_obstacle = ObstacleViolation(
                    description=f"적재물{next_number}",
                    detected_at=datetime.now(),
                    video_url=violation_record.url  # VIOLATION 테이블의 URL 참조
                )
                
                try:
                    db.add(new_obstacle)
                    db.commit()
                    print(f"적재물 위반 기록 저장 완료: 적재물{next_number}")
                except Exception as db_error:
                    print(f"DB 저장 중 오류 발생: {db_error}")
                    db.rollback()
                
                return JSONResponse(content={
                    "status": "적재물 감지",
                    "type": "obstacle",
                    "description": f"적재물{next_number}",
                    "video_url": violation_record.url,
                    "detected_at": datetime.now().isoformat()
                })
            else:
                print("번호판 인식 실패 및 적재물 없음")
                return JSONResponse(content={"status": "번호판 인식 실패 및 적재물 없음"})

        # 2. DB 조회 - 차량 번호 확인
        logging.info(f"DB에서 차량 번호 조회 시작: {plate_number.strip()}")
        print(f"DB에서 차량 번호 조회 시작: {plate_number.strip()}")
        user_entry = db.query(User).filter(User.car_number == plate_number.strip()).first()
        
        if not user_entry:
            logging.info(f"미등록 차량으로 처리: {plate_number}")
            print(f"미등록 차량으로 처리: {plate_number}")
            violation_record.violation_number = plate_number
            db.commit()
            print(f"위반 차량 번호: {plate_number} DB 저장 완료")
            return JSONResponse(content={"status": "미등록 차량", "type": "illegal"})
        
        if not user_entry.handicap:
            logging.info(f"일반 등록 차량으로 처리: {plate_number}")
            print(f"일반 등록 차량으로 처리: {plate_number}")
            violation_record.violation_number = plate_number
            db.commit()
            print(f"위반 차량 번호: {plate_number} DB 저장 완료")
            return JSONResponse(content={"status": "일반 차량", "type": "illegal"})
        
        # 장애인 등록 차량인 경우
        logging.info(f"장애인 등록 차량 확인: {plate_number}")
        print(f"장애인 등록 차량 확인: {plate_number}")
        return JSONResponse(content={"status": "장애인 등록 차량", "action": "capture_assist", "plate_number": plate_number})

    except Exception as e:
        print(f"처리 중 에러 발생: {e}")
        raise HTTPException(status_code=500, detail=str(e))


# API 엔드포인트
@app.post("/assist_check/")
async def assist_check(request:Request, file: UploadFile = File(...), db: Session = Depends(get_db)):
    try:
        form_data = await request.form()
        video_filename = form_data.get("video_filename")
        plate_number = form_data.get("plate_number")

        if not video_filename or not plate_number:
            print("필수 정보 누락")
            return JSONResponse(
                content={"error": "비디오 파일명 또는 차량번호가 누락되었습니다"}, 
                status_code=400
            )

        print(f"Received file for assist_check: {file.filename}")
        print(f"Video filename: {video_filename}")
        print(f"Plate number: {plate_number}")
        print("\n=== 보조기구 확인 시작 ===")
        
        # 파일 저장
        file_location = UPLOAD_DIR / file.filename
        with open(file_location, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)
        print(f"파일 저장 완료: {file_location}")

        # 3. 보조기구 분석
        assist_result = analyze_assist_device(str(file_location))
        print(f"보조기구 분석 결과: {assist_result}")

        # violation_record 찾기
        violation_record = db.query(Violation).filter(
            Violation.filename == video_filename
        ).first()

        if not violation_record:
            print("비디오 파일 기록을 찾을 수 없습니다")
            return JSONResponse(content={"error": "비디오 파일 기록을 찾을 수 없습니다"}, status_code=404)

        if assist_result == "합법: 객체가 탐지되었습니다.":
            print("보조기구 확인 완료 - 합법 처리")
            violation_record.violation_number = None  # 합법이므로 위반 기록 제거
            db.commit()
            return JSONResponse(content={"status": "보조기구 확인", "type": "legal"})
        else:
            print("보조기구 없음 - 불법 처리")
            plate_number = form_data.get("plate_number")
            if plate_number:
                violation_record.violation_number = plate_number  # 불법이므로 위반 차량 번호 기록
                db.commit()
                print(f"위반 차량 번호 {plate_number} DB 저장 완료")
            return JSONResponse(content={"status": "보조기구 없음", "type": "illegal"})

    except Exception as e:
        print(f"처리 중 에러 발생: {e}")
        raise HTTPException(status_code=500, detail=str(e))