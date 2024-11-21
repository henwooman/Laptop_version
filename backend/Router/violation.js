const express = require('express');
const router = express.Router();
const conn = require('../config/db');


// 날짜, 차량번호, 시간 범위를 기준으로 위반 데이터 필터링
router.get('/filter_Violations',(req,res)=>{
    console.log('GET /filter_Violations route hit'); // 로그 추가

    const { car_number, startTime, endTime } = req.query;
    console.log('Received parameters:', { car_number, startTime, endTime });

    // 파라미터 유효성 검사
    if (!car_number || !startTime || !endTime) {
        return res.status(400).json({ error: '차량 번호, 시작 시간, 종료 시간을 모두 제공해주세요.' });
    }

    // SQL 쿼리 작성: violation_date와 violation_time을 결합하여 DateTime 형식으로 비교
    const sql = `
        SELECT * 
        FROM VIOLATION 
        WHERE upload_time BETWEEN ? AND ? 
          AND violation_number = ? 
    `;

    // 쿼리 실행 전에 쿼리와 파라미터를 로그로 출력
    console.log('실행될 쿼리:', sql);
    console.log('쿼리 파라미터:', [startTime, endTime, car_number]);


    // 쿼리 실행
    conn.query(sql, [startTime, endTime, car_number], (err, rows) => {
        if (err) {
            console.error('위반데이터 fetching 에러 :', err);
            return res.status(500).json({ error: '위반데이터 fetching failed' });
        } else if (rows.length > 0) {
            return res.status(200).json(rows);
        } else {
            return res.status(200).json({ message: '데이터가 없습니다', data: [] });
        }
    });
});





// 기간으로 위반데이터 조회하기 
router.get('/filtering_dateRange', (req, res) => {
    console.log('GET /filtering_dateRange route hit');
    let { startDate, endDate, car_number } = req.query;

    // 파라미터 로그 추가
    console.log('Received parameters:', { startDate, endDate, car_number });

    if (!startDate || !endDate) {
        return res.status(400).json({ error: '시작 날짜와 끝 날짜를 모두 제공해주세요.' });
    }

    if (!car_number) {
        return res.status(400).json({ error: '차량 번호를 제공해주세요.' });
    }

    const startDateTime = `${startDate} 00:00:00`;
    const endDateTime = `${endDate} 23:59:59`;

    const sql = `
        SELECT * 
        FROM VIOLATION 
        WHERE upload_time BETWEEN ? AND ? 
          AND violation_number = ?
    `;
    const params = [startDateTime, endDateTime, car_number];

    console.log('실행될 쿼리:', sql);
    console.log('쿼리 파라미터:', params);

    conn.query(sql, params, (err, rows) => {
        if (err) {
            console.error('DB 조회 오류:', err);
            return res.status(500).json({ error: 'DB 조회 중 오류가 발생했습니다.' });
        } else if (rows.length > 0) {
            return res.status(200).json(rows);
        } else {
            return res.status(200).json({ message: '해당 기간과 차량 번호에 대한 데이터가 없습니다.', data: [] });
        }
    });
});


// 차량 위반데이터 전체 조회하기 
router.get('/all',(req,res)=>{
    //쿼리문 : violation_date를 기준으로 내림차순 정렬
    const sql = 'SELECT * FROM VIOLATION ORDER BY upload_time DESC';

    conn.query(sql,(err,rows)=>{
        if(err){
            console.error('DB조회 오류 : ',err);
            return res.status(500).json({ error : 'DB 조회 중 오류 발생'});            
        }
        //조회된 데이터 반환
        res.status(200).json(rows);
    });
});





// 특정 차량 번호(+날짜/시간)의 동영상 데이터 반환 API
router.get('/videos', (req, res) => {
    console.log('GET /videos route hit'); // 로그 추가

    const { violationNumber, dateTime } = req.query;

    // 파라미터 유효성 검사
    if (!violationNumber && !dateTime) {
        return res.status(400).json({ error: 'violationNumber(차량 번호) 또는 dateTime(날짜+시간)이 필요합니다.' });
    }

    // SQL 쿼리 작성
    const sql = `
        SELECT violation_id, violation_number, upload_time, url, filename
        FROM VIOLATION
        WHERE violation_number = ? AND upload_time = ?
        ORDER BY upload_time DESC
        LIMIT 1
    `;
    const params = [violationNumber, dateTime]; // 차량 번호와 날짜+시간 기준

    // 쿼리 실행 전에 로그 출력
    // console.log('실행될 쿼리:', sql);
    // console.log('쿼리 파라미터:', params);

    // 쿼리 실행
    conn.query(sql, params, (err, rows) => {
        if (err) {
            console.error('동영상 데이터 조회 오류:', err);
            return res.status(500).json({ error: '동영상 데이터를 가져오는 중 오류가 발생했습니다.' });
        }

        // 결과 반환
        return res.status(200).json({
            message: rows.length > 0
                ? '동영상 데이터를 성공적으로 가져왔습니다.'
                : '해당 조건에 맞는 동영상 데이터가 없습니다.',
            data: rows // 항상 `data` 필드로 응답
        });
    });
});




// 적재물 위반 데이터 전체조회하기 
router.get('/obstacle_all', (req, res) => {
    // 쿼리: 적재물 데이터를 날짜(detected_at) 기준으로 정렬, 같은 날짜일 경우 시간까지 고려
    const sql = 'SELECT * FROM OBSTACLE_VIOLATION ORDER BY DATE(detected_at) DESC, TIME(detected_at) DESC';

    conn.query(sql, (err, rows) => {
        if (err) {
            console.error('적재물 데이터 조회 오류:', err); // 구체적인 오류 메시지 출력
            return res.status(500).json({ error: '적재물 데이터 조회 중 오류가 발생했습니다.', details: err.message });
        }
        res.status(200).json(rows);
    });
});



module.exports = router;
