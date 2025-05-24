#!/bin/bash

# ตรวจสอบว่ามีการตั้งค่า API key หรือไม่
if [ -z "$DEEPSEEK_API_KEY" ]; then
    echo "กรุณาตั้งค่า DEEPSEEK_API_KEY ในตัวแปรแวดล้อมก่อน"
    echo "ตัวอย่าง: export DEEPSEEK_API_KEY=your_api_key_here"
    exit 1
fi

# สร้าง Dataset สำหรับการศึกษา
echo "กำลังสร้าง Dataset สำหรับระดับประถมศึกษา (ป.1-ป.6)..."
python src/python/generate_dataset.py primary_school_knowledge 50 --format jsonl

echo
echo "กำลังสร้าง Dataset สำหรับระดับมัธยมศึกษา (ม.1-ม.6)..."
python src/python/generate_dataset.py secondary_school_knowledge 50 --format jsonl

echo 
echo "กำลังสร้าง Dataset สำหรับข้อสอบเลื่อนชั้น..."
python src/python/generate_dataset.py school_grade_promotion_exam 50 --format jsonl

echo
echo "เสร็จสิ้น! ไฟล์ dataset ถูกบันทึกไว้ใน data/output/"
echo "คุณสามารถใช้ dataset เหล่านี้เพื่อเทรนโมเดลต่อไป"
