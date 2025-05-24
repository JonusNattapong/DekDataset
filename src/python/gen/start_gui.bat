@echo off
REM start_gui.bat - สคริปต์สำหรับเริ่มต้น GUI application บน Windows

REM กำหนด working directory เป็นโฟลเดอร์หลักของโปรเจค
cd /d "%~dp0\..\.."

REM เริ่มต้น GUI
echo กำลังเริ่มต้น GUI application...
python src/python/subject_prediction_gui.py
