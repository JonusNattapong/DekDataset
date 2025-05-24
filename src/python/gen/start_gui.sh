#!/bin/bash
# start_gui.sh - สคริปต์สำหรับเริ่มต้น GUI application

# กำหนด working directory เป็นโฟลเดอร์หลักของโปรเจค
cd "$(dirname "$0")/../.."

# เริ่มต้น GUI
echo "กำลังเริ่มต้น GUI application..."
python src/python/subject_prediction_gui.py
