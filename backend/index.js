const express = require("express");
const app = express();
const port = 4000;

// 기본 라우트 설정
app.get("/", (req, res) => {
  res.send("Hello from the backend server!");
});

// 서버 실행
app.listen(port, () => {
  console.log(`Backend server is running on http://localhost:${port}`);
});
