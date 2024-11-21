require('dotenv').config();  // 환경변수 로드
const express = require('express');
const router = express.Router();
const conn = require('../config/db');  // DB 연결
const axios = require('axios');

// 데이터 삭제 작업 실행(1분마다 불러온 리스트 삭제 함수)
setInterval(()=>{
  const deleteQuery = `
    DELETE m
    FROM maplist m
    JOIN (
      SELECT time
      FROM maplist
      ORDER BY time ASC
      LIMIT 1
    ) subquery
    ON m.time = subquery.time;
  `;
  conn.query(deleteQuery, (err, results) => {
    if (err) {
      console.error('데이터 삭제 실패:', err);
    } else {
      console.log('자동 데이터 삭제 성공:', results);
    }
  })
}, 600000);

// 주차장 찾기에서 얻은 위도, 경도 데이터를 DB에 저장하기
router.post('/parkinglist', (req, res) => {
  const { latitude, longitude } = req.body;

  // 받은 위도, 경도 데이터를 콘솔에 출력
  console.log('받은 데이터:', { latitude, longitude });

  const moment = require('moment-timezone');

  // 한국 시간으로 변환하여 저장
  const currentTime = moment.tz("Asia/Seoul").format('YYYY-MM-DD HH:mm:ss');

  // SQL 쿼리
  const query = 'INSERT INTO maplist (latitude, longitude, time) VALUES(?,?,?)';

  // DB에 데이터 넣기
  conn.query(query, [latitude, longitude,currentTime], (err, results) => {
      if (err) {
          console.error('DB 저장 실패:', err);
          return res.status(500).json({ error: 'DB 저장 실패' });
      }
      console.log('DB 저장 성공:', results);
      res.status(200).send({ message: '데이터 전송 성공!' });
  });
});

// kakao api 위도/ 경도 변환하기
router.get('/parkinglist2', async (req, res) => {
    console.log('Request received at /parkinglist2');
    try {
      const sql = 'SELECT latitude, longitude FROM maplist WHERE latitude IS NOT NULL AND longitude IS NOT NULL'; // DB에서 위도/경도 가져오기
  
      conn.query(sql, (err, results) => {
        if (err) {
          console.error('DB 조회 실패:', err);
          return res.status(500).json({ error: 'DB 조회 실패' });
        }
  
        if (results.length === 0) {
          return res.status(404).json({ message: 'DB에 데이터가 없습니다.' });
        }
  
        // 위도, 경도를 coordinates 배열로 변환
        const coordinates = results.map(item => ({
          latitude: item.latitude,
          longitude: item.longitude,
        }));
  
        console.log('Coordinates:', coordinates); // 서버에서 반환할 데이터 확인
        res.json({ coordinates });
      });
    } catch (error) {
      console.error('서버 오류:', error);
      res.status(500).json({ error: '서버 오류 발생' });
    }
  });

  // kakao api 로드뷰 설정 
  router.get('/parkinglist3', (req, res) => {
    const sql = 'SELECT latitude, longitude FROM maplist WHERE latitude IS NOT NULL AND longitude IS NOT NULL';
    
    // DB 쿼리 실행
    conn.query(sql, (err, results) => {
      if (err) {
        console.error('DB 조회 실패:', err);
        return res.status(500).json({ error: 'DB 조회 실패' });
      }
  
      if (results.length === 0) {
        return res.status(404).json({ message: 'DB에 데이터가 없습니다.' });
      }
  
      // 위도, 경도 데이터를 클라이언트로 반환
      const coordinates = results.map(item => ({
        latitude: item.latitude,
        longitude: item.longitude
      }));
  
      console.log('Coordinates:', coordinates); // 서버에서 반환할 데이터 확인
      res.json({ coordinates });
    });
  });
  
module.exports = router;
