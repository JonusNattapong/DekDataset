@echo off
REM generate_and_augment_data.bat - สคริปต์สำหรับสร้างและเพิ่มข้อมูลสำหรับเทรนโมเดล

REM กำหนด working directory เป็นโฟลเดอร์หลักของโปรเจค
cd /d "%~dp0\..\.."

echo กำลังสร้างข้อคำถามที่หลากหลาย...
python src/python/generate_diverse_questions.py --output data/output/generated-diverse-questions.jsonl --questions_per_grade 10

echo กำลังรวมข้อมูลทั้งหมด...
python src/python/merge_primary_secondary.py

echo กำลังเพิ่มข้อมูลด้วย Data Augmentation...
python src/python/data_augmentation.py --input data/output/merged-primary-secondary.jsonl --output data/output/augmented-dataset.jsonl --augmentations 3 --generate

echo กระบวนการสร้างและเพิ่มข้อมูลเสร็จสมบูรณ์
echo ไฟล์ผลลัพธ์:
echo   - data/output/generated-diverse-questions.jsonl
echo   - data/output/merged-primary-secondary.jsonl
echo   - data/output/augmented-dataset.jsonl
