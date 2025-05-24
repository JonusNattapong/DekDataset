@echo off
REM train_qwen_model.bat - สคริปต์สำหรับการติดตั้งแพ็กเกจและเทรนโมเดล Qwen บน Windows

REM กำหนด working directory เป็นโฟลเดอร์หลักของโปรเจค
cd /d "%~dp0\..\.."

REM ติดตั้งแพ็กเกจที่จำเป็น
echo กำลังติดตั้งแพ็กเกจที่จำเป็น...
pip install -r src/python/requirements-train.txt

REM สร้างไดเรกทอรีสำหรับเก็บโมเดลและผลการประเมิน
mkdir data\output\thai-education-qwen-model 2>nul
mkdir data\output\evaluation-qwen-results 2>nul

REM เทรนโมเดล
echo กำลังเริ่มเทรนโมเดล Qwen...
python src/python/train_qwen_model.py

REM ทดสอบโมเดลหลังเทรน
echo กำลังทดสอบโมเดล...
python src/python/test_model.py
