#!/bin/bash
# start_api_server.sh - สคริปต์สำหรับเริ่มต้น API server

# กำหนด working directory เป็นโฟลเดอร์หลักของโปรเจค
cd "$(dirname "$0")/../.."

# ติดตั้งแพ็กเกจที่จำเป็น
echo "กำลังติดตั้งแพ็กเกจที่จำเป็น..."
pip install fastapi uvicorn

# เริ่มต้นเซิร์ฟเวอร์
echo "กำลังเริ่มต้น API server ที่ http://localhost:8000"
uvicorn src.python.subject_prediction_api:app --host 0.0.0.0 --port 8000 --reload
