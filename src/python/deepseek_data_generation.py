#!/usr/bin/env python
# coding: utf-8
# deepseek_data_generation.py - สร้าง Dataset โดยใช้ Deepseek API

import os
import json
import random
import argparse
import requests
import time
from tqdm import tqdm
import concurrent.futures

# ข้อมูลวิชาและระดับชั้น
SUBJECTS = [
    "ภาษาไทย", "คณิตศาสตร์", "วิทยาศาสตร์", "สังคมศึกษา", "ภาษาอังกฤษ",
    "ศิลปะ", "สุขศึกษาและพลศึกษา", "การงานอาชีพและเทคโนโลยี", "ดนตรี", 
    "เทคโนโลยี", "ภาษาจีน", "ประวัติศาสตร์", "ภูมิศาสตร์"
]

PRIMARY_GRADES = list(range(1, 7))  # ป.1-6
SECONDARY_GRADES = list(range(7, 13))  # ม.1-6
ALL_GRADES = PRIMARY_GRADES + SECONDARY_GRADES

# ตัวอย่าง prompt เพื่อสร้างข้อสอบ
PROMPT_TEMPLATE = """
โปรดสร้างข้อสอบวิชา{subject}สำหรับนักเรียนชั้น{grade_name} จำนวน {num_questions} ข้อ
แต่ละข้อให้มีตัวเลือก 4 ตัวเลือก (ก, ข, ค, ง) และระบุคำตอบที่ถูกต้อง

เงื่อนไข:
1. ข้อสอบต้องสอดคล้องกับหลักสูตรแกนกลางการศึกษาขั้นพื้นฐานของประเทศไทย
2. คำถามควรมีความชัดเจน ไม่กำกวม
3. ตัวเลือกต้องสมเหตุสมผล มีเพียงตัวเลือกเดียวที่ถูกต้อง
4. คำถามควรครอบคลุมเนื้อหาสำคัญของวิชานี้ในระดับชั้นดังกล่าว

โปรดให้ผลลัพธ์มาในรูปแบบ JSON ตามตัวอย่างนี้:
```json
[
  {{
    "subject": "{subject}",
    "grade": {grade},
    "question": "คำถาม...?",
    "choices": ["ตัวเลือก ก", "ตัวเลือก ข", "ตัวเลือก ค", "ตัวเลือก ง"],
    "answer": "ตัวเลือก ค",
    "difficulty": "กลาง",
    "tags": ["{subject}", "หัวข้อย่อย"]
  }},
  ...
]
```

โปรดสร้างข้อสอบที่หลากหลายและมีคุณภาพ ครอบคลุมเนื้อหาสำคัญของวิชา{subject}ระดับชั้น{grade_name}
"""

def call_deepseek_api(prompt, api_key, model="deepseek-chat", max_retries=3):
    """เรียกใช้ Deepseek API เพื่อสร้าง dataset"""
    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {api_key}"
    }
    
    url = "https://api.deepseek.com/v1/chat/completions"  # ต้องตรวจสอบ URL API ที่ถูกต้อง
    
    data = {
        "model": model,
        "messages": [{"role": "user", "content": prompt}],
        "temperature": 0.7,
        "max_tokens": 4096
    }
    
    for attempt in range(max_retries):
        try:
            response = requests.post(url, json=data, headers=headers)
            response.raise_for_status()
            return response.json()["choices"][0]["message"]["content"]
        except requests.exceptions.RequestException as e:
            if attempt == max_retries - 1:
                print(f"เกิดข้อผิดพลาด: {e}")
                return None
            print(f"เกิดข้อผิดพลาด: {e}, ลองอีกครั้งใน 5 วินาที...")
            time.sleep(5)

def extract_json_from_response(response_text):
    """แยกข้อมูล JSON จากข้อความตอบกลับ"""
    if not response_text:
        return []
    
    # ค้นหาเครื่องหมาย ``` ที่ครอบ JSON
    json_start = response_text.find("```json")
    if json_start != -1:
        json_start += 7  # ข้ามคำว่า ```json
    else:
        json_start = response_text.find("```")
        if json_start != -1:
            json_start += 3
        else:
            json_start = 0

    json_end = response_text.rfind("```")
    if json_end != -1 and json_end > json_start:
        json_text = response_text[json_start:json_end].strip()
    else:
        json_text = response_text[json_start:].strip()
    
    try:
        return json.loads(json_text)
    except json.JSONDecodeError as e:
        print(f"ไม่สามารถแปลง JSON ได้: {e}")
        print(f"ข้อความ: {json_text[:200]}...")
        return []

def generate_questions_for_subject_grade(args, subject, grade):
    """สร้างคำถามสำหรับวิชาและระดับชั้นที่กำหนด"""
    grade_name = f"ป.{grade}" if grade <= 6 else f"ม.{grade-6}"
    
    prompt = PROMPT_TEMPLATE.format(
        subject=subject,
        grade=grade,
        grade_name=grade_name,
        num_questions=args.questions_per_call
    )
    
    print(f"กำลังสร้างข้อสอบวิชา {subject} ระดับชั้น {grade_name}...")
    response = call_deepseek_api(prompt, args.api_key, model=args.model)
    questions = extract_json_from_response(response)
    
    # ตรวจสอบว่าข้อมูลถูกต้องหรือไม่
    valid_questions = []
    for q in questions:
        if isinstance(q, dict) and "subject" in q and "question" in q and "choices" in q and "answer" in q:
            # ตรวจสอบให้แน่ใจว่ามีการตั้งค่า subject และ grade ให้ถูกต้อง
            q["subject"] = subject
            q["grade"] = grade
            valid_questions.append(q)
    
    return valid_questions

def generate_diverse_dataset(args):
    """สร้าง dataset ที่หลากหลายสำหรับทุกวิชาและระดับชั้น"""
    all_questions = []
    
    if args.parallel:
        # สร้างแบบขนานโดยใช้ ThreadPoolExecutor
        tasks = []
        with concurrent.futures.ThreadPoolExecutor(max_workers=args.max_workers) as executor:
            for subject in SUBJECTS:
                for grade in ALL_GRADES:
                    # สร้างงานสำหรับแต่ละวิชาและระดับชั้น
                    task = executor.submit(generate_questions_for_subject_grade, args, subject, grade)
                    tasks.append(task)
            
            # รวบรวมผลลัพธ์
            for task in tqdm(concurrent.futures.as_completed(tasks), total=len(tasks), desc="กำลังสร้างข้อสอบ"):
                result = task.result()
                all_questions.extend(result)
    else:
        # สร้างแบบลำดับ
        for subject in SUBJECTS:
            for grade in tqdm(ALL_GRADES, desc=f"กำลังสร้างข้อสอบวิชา {subject}"):
                questions = generate_questions_for_subject_grade(args, subject, grade)
                all_questions.extend(questions)
                # หน่วงเวลาเล็กน้อยเพื่อหลีกเลี่ยงการเรียกใช้ API มากเกินไป
                time.sleep(args.delay)
    
    return all_questions

def save_dataset(questions, output_file):
    """บันทึก dataset ไปยังไฟล์ JSONL"""
    with open(output_file, "w", encoding="utf-8") as f:
        for q in questions:
            json.dump(q, f, ensure_ascii=False)
            f.write("\n")
    
    print(f"บันทึกข้อสอบทั้งหมด {len(questions)} ข้อไปยัง {output_file}")

def main():
    parser = argparse.ArgumentParser(description="สร้าง Dataset โดยใช้ Deepseek API")
    parser.add_argument("--api_key", required=True, help="API Key สำหรับ Deepseek")
    parser.add_argument("--output", default="data/output/deepseek-dataset.jsonl", help="ไฟล์ output สำหรับ dataset")
    parser.add_argument("--questions_per_call", type=int, default=10, help="จำนวนคำถามต่อการเรียก API หนึ่งครั้ง")
    parser.add_argument("--delay", type=int, default=2, help="เวลาหน่วง (วินาที) ระหว่างการเรียก API")
    parser.add_argument("--model", default="deepseek-chat", help="โมเดล Deepseek ที่ต้องการใช้")
    parser.add_argument("--parallel", action="store_true", help="สร้างข้อมูลแบบขนาน")
    parser.add_argument("--max_workers", type=int, default=4, help="จำนวน worker สูงสุดในการสร้างแบบขนาน")
    parser.add_argument("--subjects", nargs="+", choices=SUBJECTS, help="เลือกวิชาเฉพาะที่ต้องการสร้าง")
    parser.add_argument("--grades", nargs="+", type=int, choices=ALL_GRADES, help="เลือกระดับชั้นเฉพาะที่ต้องการสร้าง")
    args = parser.parse_args()
    
    # สร้างไดเรกทอรีสำหรับ output ถ้ายังไม่มี
    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    
    # สร้าง dataset
    questions = generate_diverse_dataset(args)
    
    # บันทึก dataset
    save_dataset(questions, args.output)
    
    print(f"เสร็จสิ้น! สร้างข้อสอบทั้งหมด {len(questions)} ข้อ")

if __name__ == "__main__":
    main()
