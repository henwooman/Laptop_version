const express = require('express');
const app = express();
const user = require('./Router/user');
const main = require('./Router/main');
const violation = require('./Router/violation');
const parkingad = require('./Router/parkingad')
const cors = require('cors');

// CORS 미들웨어를 먼저 설정
app.use(cors({
    origin: 'http://localhost:3000',
    credentials: true,  // 쿠키와 세션 정보를 포함시키도록 설정
}));

app.use(express.json()); // JSON 파싱 미들웨어 설정

// 세션 셋팅
const session = require("express-session");
const fileStore = require("session-file-store")(session);
app.use(session({
    httpOnly : true,  //http로 들어온 요청만 처리하겠다.
    resave : false,    // 세션을 항상 재 저장하겠냐? 
    secret : "secret",  // 암호화 할때 사용하는 키값
    store : new fileStore(),   // 세션을 저장하기 위한 저장소 셋팅
    saveUninitialized : false  //세션에 저장할 내용이 없더라도 저장 여부!
}))

// 라우터 설정
app.use('/user', user);
app.use('/', main);
app.use('/ParkingSearch', parkingad);
app.use('/violation',violation);




app.set('port', process.env.PORT || 4000)
app.listen(app.get('port'), ()=>{
    console.log(`Server is running on ${app.get('port')}`);
});



