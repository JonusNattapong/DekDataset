@echo off
echo ===== DeepSeek Dataset Generator =====
echo ตัวช่วยสร้าง Dataset จาก Deepseek API

set /p API_KEY=กรุณาใส่ DeepSeek API Key: 
set /p QUESTIONS=จำนวนคำถามต่อการเรียก API (แนะนำ 10): 

if "%QUESTIONS%"=="" set QUESTIONS=10

echo.
echo กำลังสร้าง Dataset จาก Deepseek API...
python src/python/deepseek_data_generation.py --api_key %API_KEY% --questions_per_call %QUESTIONS% --output data/output/deepseek-dataset-raw.jsonl

echo.
echo กำลังเตรียม Dataset...
python src/python/prepare_deepseek_dataset.py --input data/output/deepseek-dataset-raw.jsonl --output_dir data/output/deepseek-dataset

echo.
echo เสร็จสิ้น! Dataset ถูกสร้างและเตรียมเรียบร้อยแล้ว
echo ไฟล์ dataset อยู่ที่ data/output/deepseek-dataset/
echo - train.jsonl: สำหรับเทรนโมเดล
echo - validation.jsonl: สำหรับตรวจสอบระหว่างเทรน
echo - test.jsonl: สำหรับทดสอบโมเดล
echo - analysis/: ผลการวิเคราะห์ข้อมูล

pause
