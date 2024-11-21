const express = require('express');
const router = express.Router();
const conn = require('../config/db');

// 사용자 회원가입
router.post('/join', (req, res) => {
    console.log('회원가입 요청');
    console.log('요청데이터 출력:', req.body);

    const { id, pw, name, phone, email, carNumber, adminCode } = req.body; // id와 pw 사용
    const username = id;
    const password = pw; // 여기에 추가

    const userQuery = `INSERT INTO USER (user_id, user_pw, user_email, car_number, user_joined, user_name, user_phone, handicap) VALUES (?, ?, ?, ?, ?, ?, ? ,?)`;
    const adminQuery = `INSERT INTO ADMIN (admin_id, admin_pw, admin_email, admin_joined ,admin_name, admin_phone, admin_auth_code) VALUES (?, ?, ?, ?, ?, ?, ?)`;


    const query = adminCode ? adminQuery : userQuery;

    const params = adminCode
        ? [id, password, email, new Date(), name, phone, adminCode]  // 관리자일 경우 adminCode를 포함
        : [id, password, email, carNumber, new Date(), name, phone, 1]; // 일반 사용자는 adminCode가 필요 없으므로 1로 설정


    conn.query(query, params, (err, results) => {
        if (err) {
            console.error('회원가입 에러:', err);
            res.status(500).json({ error: '서버 에러' });
        } else {
            res.status(201).json({ message: '회원가입 성공' });
        }
    });
});






// 사용자 로그인
router.post('/login', (req, res) => {

    console.log('서버에서 로그인 요청 수신:', req.body);

    const { username, password } = req.body; // 클라이언트에서 받은 데이터
    const query = `SELECT * FROM USER WHERE user_id = ?`;

    conn.query(query, [username], (err, results) => {
        if (err) {
            console.error('로그인 에러:', err);
            res.status(500).json({ error: '서버 에러' });
        } else if (results.length > 0) {
            const user = results[0];

            if (password === user.user_pw) {
                req.session.user_id = user.user_id;
                req.session.user_name = user.user_name;
                res.json({ message: '로그인 성공' });
            } else {
                res.status(401).json({ error: '아이디 또는 비밀번호가 일치하지 않습니다.' });
            }
        } else {
            res.status(401).json({ error: '아이디 또는 비밀번호가 일치하지 않습니다.' });
        }
    });
});





// 관리자 로그인
router.post('/admin-login', (req, res) => {
    console.log('관리자 로그인 요청 수신:', req.body);

    const { username, password } = req.body; // 클라이언트에서 받은 데이터
    const query = `SELECT * FROM ADMIN WHERE admin_id = ?`;

    conn.query(query, [username], (err, results) => {
        if (err) {
            console.error('관리자 로그인 에러:', err);
            res.status(500).json({ error: '서버 에러' });
        } else if (results.length > 0) {
            const admin = results[0];

            if (password === admin.admin_pw) {
                res.json({ message: '관리자 로그인 성공', admin });
            } else {
                res.status(401).json({ error: '아이디 또는 비밀번호가 일치하지 않습니다.' });
            }
        } else {
            res.status(401).json({ error: '아이디 또는 비밀번호가 일치하지 않습니다.' });
        }
    });
});




// 사용자 메인 화면에서 사용자 정보 띄우기 : 이름/차량번호
router.get('/userinfo', (req, res) => {
    if (!req.session.user_id) {
        return res.status(401).json({ error: '로그인이 필요합니다.' });
    }

    const query = 'SELECT user_name, car_number FROM USER WHERE user_id = ?';
    conn.query(query, [req.session.user_id], (error, results) => {
        if (error) {
            return res.status(500).json({ error: 'DB 오류' });
        }
        if (results.length > 0) {
            res.json(results[0]);
        } else {
            res.status(404).json({ error: '사용자를 찾을 수 없습니다.' });
        }
    });
});


// 회원정보 수정 페이지 컴포넌트 ( 비밀번호 제외 )
router.get('/profile', (req, res) => {
    if (!req.session.user_id) {
        return res.status(401).json({ error: '로그인이 필요합니다.' });
    }

    const query = 'SELECT user_name, user_phone, user_email, car_number FROM USER WHERE user_id = ?';
    conn.query(query, [req.session.user_id], (error, results) => {
        if (error) {
            return res.status(500).json({ error: 'DB오류' });
        }
        if (results.length > 0) {
            res.json(results[0]); //비밀번호를 제외한 사용자 정보를 반환
        } else {
            res.status(404).json({ error: '사용자를 찾을 수 없습니다.' });
        }
    });
});


// 사용자 정보 수정( 현재 비밀번호 확인 포함 )
router.post('/update', (req, res) => {
    if (!req.session.user_id) {
        return res.status(401).json({ error: '로그인이 필요합니다.' });
    }

    const { name, phone, email, carNumber, currentPassword, newPassword } = req.body;
    const userId = req.session.user_id;

    // 현재 비밀번호 확인
    const query = 'SELECT user_pw FROM USER WHERE user_id = ?';
    conn.query(query, [userId], (error, results) => {
        if (error) {
            console.error('DB 오류:', error);
            return res.status(500).json({ error: 'DB 오류' });
        }
        if (results.length === 0) {
            return res.status(404).json({ error: '사용자를 찾을 수 없습니다.' });
        }

        const storedPassword = results[0].user_pw;

        // 현재 비밀번호 확인
        if (currentPassword !== storedPassword) {
            return res.status(401).json({ error: '현재 비밀번호가 일치하지 않습니다.' });
        }

        const updateUser = (newPassword) => {
            const query = `UPDATE USER SET user_name = ?, user_phone = ?, user_email = ?, car_number = ?${newPassword ? ', user_pw = ?' : ''} WHERE user_id = ?`;
            const params = newPassword
                ? [name, phone, email, carNumber, newPassword, userId]
                : [name, phone, email, carNumber, userId];

            conn.query(query, params, (error, results) => {
                if (error) {
                    console.error('회원정보 수정 중 오류:', error);
                    return res.status(500).json({ error: 'DB 오류' });
                }
                res.json({ message: '회원정보가 수정되었습니다.' });
            });
        };

        updateUser(newPassword || null);
    });
});




// 사용자 차량 번호 조회 API
router.get('/carNumber', (req, res) => {
    // 세션에서 사용자 ID 가져오기
    const userId = req.session.user_id;

    // 로그인되지 않았을 경우
    if (!userId) {
        return res.status(401).json({ error: '로그인이 필요합니다.' });
    }

    // DB에서 사용자 ID로 차량 번호 조회
    const sql = 'SELECT car_number FROM USER WHERE user_id = ?';
    conn.query(sql, [userId], (err, result) => {
        if (err) {
            console.error('차량 번호 조회 중 오류:', err);
            return res.status(500).json({ error: '서버 에러' });
        }

        if (result.length > 0) {
            res.json({ carNumber: result[0].car_number }); // 차량 번호 반환
        } else {
            res.status(404).json({ error: '차량 번호를 찾을 수 없습니다.' });
        }
    });
});


// 차량 번호로 사용자 정보 조회=> 등록차량 관리페이지에 필요 
router.get('/search-by-car-number', (req, res) => {
    const { carNumber } = req.query;

    if (!carNumber) {
        return res.status(400).json({ error: '차량 번호를 입력해주세요.' });
    }

    // handicap 필드를 SELECT 절에 추가
    const sql = 'SELECT user_joined, car_number, user_name, user_phone, handicap FROM USER WHERE car_number = ?';
    conn.query(sql, [carNumber], (err, results) => {
        if (err) {
            console.error('차량 번호 조회 중 오류:', err);
            return res.status(500).json({ error: '서버 에러' });
        }

        if (results.length > 0) {
            res.status(200).json(results[0]);
        } else {
            res.status(404).json({ error: '해당 차량 번호로 사용자를 찾을 수 없습니다.' });
        }
    });
});


// 사용자테이블에서 handicap 칼럼 상태 업데이트
router.post('/update-handicap', (req, res) => {
    const { carNumber, handicapStatus } = req.body;

    if (!carNumber || handicapStatus === undefined) {
        return res.status(400).json({ error: '차량 번호와 handicap 상태를 제공해주세요.' });
    }

    const sql = 'UPDATE USER SET handicap = ? WHERE car_number = ?';
    conn.query(sql, [handicapStatus, carNumber], (err, results) => {
        if (err) {
            console.error('handicap 상태 업데이트 중 오류:', err);
            return res.status(500).json({ error: '서버 에러' });
        }

        if (results.affectedRows > 0) {
            res.status(200).json({ message: 'handicap 상태가 업데이트되었습니다.' });
        } else {
            res.status(404).json({ error: '해당 차량 번호로 사용자를 찾을 수 없습니다.' });
        }
    });
});





module.exports = router;
