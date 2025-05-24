@echo off
REM start_api_server.bat - สคริปต์สำหรับเริ่มต้น API server บน Windows

REM กำหนด working directory เป็นโฟลเดอร์หลักของโปรเจค
cd /d "%~dp0\..\.."

REM ติดตั้งแพ็กเกจที่จำเป็น
echo กำลังติดตั้งแพ็กเกจที่จำเป็น...
pip install fastapi uvicorn

REM เริ่มต้นเซิร์ฟเวอร์
echo กำลังเริ่มต้น API server ที่ http://localhost:8000
uvicorn src.python.subject_prediction_api:app --host 0.0.0.0 --port 8000 --reload
