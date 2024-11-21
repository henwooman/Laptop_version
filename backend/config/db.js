const mysql = require('mysql2');

const conn = mysql.createConnection({
    host: 'project-db-stu3.smhrd.com', // 또는 'project-db-stu3.smhrd.com' 선택
    user: 'Insa5_JSB_final_1',
    password: 'aischool1',
    port: '3307',
    database: 'Insa5_JSB_final_1'
});

conn.connect(err => {
    if (err) {
        console.error('DB 연결 오류:', err);
        return;
    }
    console.log('DB에 성공적으로 연결되었습니다.');
});

module.exports = conn;
