#!/bin/bash
# train_qwen_model.sh - สคริปต์สำหรับการติดตั้งแพ็กเกจและเทรนโมเดล Qwen

# กำหนด working directory เป็นโฟลเดอร์หลักของโปรเจค
cd "$(dirname "$0")/../.."

# ติดตั้งแพ็กเกจที่จำเป็น
echo "กำลังติดตั้งแพ็กเกจที่จำเป็น..."
pip install -r src/python/requirements-train.txt

# สร้างไดเรกทอรีสำหรับเก็บโมเดลและผลการประเมิน
mkdir -p data/output/thai-education-qwen-model
mkdir -p data/output/evaluation-qwen-results

# เทรนโมเดล
echo "กำลังเริ่มเทรนโมเดล Qwen..."
python src/python/train_qwen_model.py

# ทดสอบโมเดลหลังเทรน
echo "กำลังทดสอบโมเดล..."
python src/python/test_model.py
