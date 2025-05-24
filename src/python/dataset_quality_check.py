#!/usr/bin/env python
# coding: utf-8
# dataset_quality_check.py - ตรวจสอบคุณภาพของ Dataset

import os
import json
import argparse
import pandas as pd
import matplotlib
matplotlib.use('Agg')  # ใช้ Agg backend สำหรับเซิร์ฟเวอร์ที่ไม่มี GUI
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm

def load_jsonl(file_path):
    """โหลดข้อมูลจากไฟล์ JSONL"""
    data = []
    with open(file_path, 'r', encoding='utf-8') as f:
        for line in f:
            if line.strip():
                try:
                    item = json.loads(line)
                    data.append(item)
                except json.JSONDecodeError:
                    continue
    return data

def check_missing_fields(data):
    """ตรวจสอบฟิลด์ที่หายไป"""
    expected_fields = ["subject", "grade", "question", "choices", "answer"]
    missing_fields = {field: 0 for field in expected_fields}
    
    for item in data:
        for field in expected_fields:
            if field not in item or item[field] is None or item[field] == "":
                missing_fields[field] += 1
    
    return missing_fields

def check_subject_grade_distribution(data):
    """ตรวจสอบการกระจายของวิชาและระดับชั้น"""
    subjects = {}
    grades = {}
    subject_grade = {}
    
    for item in data:
        subject = item.get("subject", "unknown")
        grade = item.get("grade", 0)
        
        # นับตามวิชา
        if subject not in subjects:
            subjects[subject] = 0
        subjects[subject] += 1
        
        # นับตามระดับชั้น
        if grade not in grades:
            grades[grade] = 0
        grades[grade] += 1
        
        # นับตามวิชาและระดับชั้น
        sg_key = f"{subject}_{grade}"
        if sg_key not in subject_grade:
            subject_grade[sg_key] = 0
        subject_grade[sg_key] += 1
    
    return subjects, grades, subject_grade

def check_answer_distribution(data):
    """ตรวจสอบการกระจายของคำตอบ"""
    answer_positions = {}
    
    for item in data:
        choices = item.get("choices", [])
        answer = item.get("answer", "")
        
        if choices and answer in choices:
            position = choices.index(answer)
            if position not in answer_positions:
                answer_positions[position] = 0
            answer_positions[position] += 1
    
    return answer_positions

def check_question_length(data):
    """ตรวจสอบความยาวของคำถาม"""
    question_lengths = []
    
    for item in data:
        question = item.get("question", "")
        question_lengths.append(len(question))
    
    return question_lengths

def check_duplicate_questions(data):
    """ตรวจสอบคำถามที่ซ้ำกัน"""
    questions = {}
    duplicates = 0
    
    for item in data:
        question = item.get("question", "")
        if question in questions:
            duplicates += 1
        else:
            questions[question] = 1
    
    return duplicates, len(questions)

def plot_distribution(data, title, output_file, is_bar=True, figsize=(10, 6)):
    """สร้างกราฟแสดงการกระจายข้อมูล"""
    plt.figure(figsize=figsize)
    
    if is_bar:
        # สร้าง bar plot
        if isinstance(data, dict):
            keys = sorted(data.keys())
            values = [data[k] for k in keys]
            plt.bar(keys, values)
        else:
            plt.bar(range(len(data)), data)
        
    else:
        # สร้าง histogram
        sns.histplot(data, kde=True)
    
    plt.title(title)
    plt.tight_layout()
    plt.savefig(output_file)
    plt.close()

def main():
    parser = argparse.ArgumentParser(description="ตรวจสอบคุณภาพของ Dataset")
    parser.add_argument("--input", required=True, help="ไฟล์ JSONL ที่ต้องการตรวจสอบ")
    parser.add_argument("--output-dir", default="data/output/quality_report", help="ไดเรกทอรีสำหรับบันทึกรายงาน")
    args = parser.parse_args()
    
    # ตรวจสอบและสร้างไดเรกทอรีสำหรับบันทึกรายงาน
    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)
    
    print(f"กำลังโหลดข้อมูลจาก {args.input}...")
    data = load_jsonl(args.input)
    print(f"โหลดข้อมูลเสร็จสิ้น จำนวน {len(data)} รายการ")
    
    # ตรวจสอบฟิลด์ที่หายไป
    print("กำลังตรวจสอบฟิลด์ที่หายไป...")
    missing_fields = check_missing_fields(data)
    
    # ตรวจสอบการกระจายของวิชาและระดับชั้น
    print("กำลังตรวจสอบการกระจายของวิชาและระดับชั้น...")
    subjects, grades, subject_grade = check_subject_grade_distribution(data)
    
    # ตรวจสอบการกระจายของคำตอบ
    print("กำลังตรวจสอบการกระจายของคำตอบ...")
    answer_positions = check_answer_distribution(data)
    
    # ตรวจสอบความยาวของคำถาม
    print("กำลังตรวจสอบความยาวของคำถาม...")
    question_lengths = check_question_length(data)
    
    # ตรวจสอบคำถามที่ซ้ำกัน
    print("กำลังตรวจสอบคำถามที่ซ้ำกัน...")
    duplicates, unique_questions = check_duplicate_questions(data)
    
    # สร้างกราฟและบันทึกผลลัพธ์
    print("กำลังสร้างกราฟและบันทึกผลลัพธ์...")
    
    # 1. กราฟแสดงจำนวนฟิลด์ที่หายไป
    plot_distribution(missing_fields, "จำนวนข้อมูลที่ขาดฟิลด์", os.path.join(args.output_dir, "missing_fields.png"))
    
    # 2. กราฟแสดงการกระจายของวิชา
    plot_distribution(subjects, "การกระจายตามวิชา", os.path.join(args.output_dir, "subject_distribution.png"))
    
    # 3. กราฟแสดงการกระจายของระดับชั้น
    plot_distribution(grades, "การกระจายตามระดับชั้น", os.path.join(args.output_dir, "grade_distribution.png"))
    
    # 4. กราฟแสดงการกระจายของตำแหน่งคำตอบ
    plot_distribution(answer_positions, "การกระจายของตำแหน่งคำตอบ", os.path.join(args.output_dir, "answer_position_distribution.png"))
    
    # 5. กราฟแสดงความยาวของคำถาม
    plot_distribution(question_lengths, "ความยาวของคำถาม", os.path.join(args.output_dir, "question_length_distribution.png"), is_bar=False)
    
    # 6. สร้าง heat map แสดงการกระจายตามวิชาและระดับชั้น
    plt.figure(figsize=(14, 10))
    
    # แปลง subject_grade เป็น DataFrame
    sg_data = []
    for key, count in subject_grade.items():
        subject, grade = key.split('_')
        sg_data.append({"subject": subject, "grade": grade, "count": count})
    
    df = pd.DataFrame(sg_data)
    df["grade"] = df["grade"].astype(int)
      # สร้าง pivot table
    pivot_table = df.pivot_table(index="subject", columns="grade", values="count", fill_value=0)
    
    # สร้าง heat map
    # ใช้ fmt=".0f" สำหรับแสดงค่า float เป็นจำนวนเต็มไม่มีทศนิยม
    sns.heatmap(pivot_table, annot=True, cmap="YlGnBu", fmt=".0f")
    plt.title("การกระจายตามวิชาและระดับชั้น")
    plt.tight_layout()
    plt.savefig(os.path.join(args.output_dir, "subject_grade_heatmap.png"))
    plt.close()
    
    # สร้างรายงานสรุป
    report = {
        "total_samples": len(data),
        "missing_fields": missing_fields,
        "subject_distribution": subjects,
        "grade_distribution": grades,
        "answer_position_distribution": {str(k): v for k, v in answer_positions.items()},
        "question_length_stats": {
            "min": min(question_lengths),
            "max": max(question_lengths),
            "avg": sum(question_lengths) / len(question_lengths) if question_lengths else 0
        },
        "duplicates": duplicates,
        "unique_questions": unique_questions
    }
    
    # บันทึกรายงานเป็นไฟล์ JSON
    with open(os.path.join(args.output_dir, "quality_report.json"), 'w', encoding='utf-8') as f:
        json.dump(report, f, ensure_ascii=False, indent=4)
    
    # แสดงสรุปรายงาน
    print("\n=== รายงานสรุปคุณภาพของ Dataset ===")
    print(f"จำนวนตัวอย่างทั้งหมด: {report['total_samples']}")
    
    print("\nฟิลด์ที่หายไป:")
    for field, count in report['missing_fields'].items():
        percentage = (count / report['total_samples']) * 100
        print(f"- {field}: {count} ({percentage:.2f}%)")
    
    print("\nการกระจายตามวิชา:")
    for subject, count in sorted(report['subject_distribution'].items(), key=lambda x: x[1], reverse=True)[:5]:
        percentage = (count / report['total_samples']) * 100
        print(f"- {subject}: {count} ({percentage:.2f}%)")
    
    print("\nการกระจายตามระดับชั้น:")
    for grade, count in sorted(report['grade_distribution'].items()):
        percentage = (count / report['total_samples']) * 100
        grade_name = f"ป.{grade}" if grade <= 6 else f"ม.{grade-6}"
        print(f"- {grade_name}: {count} ({percentage:.2f}%)")
    
    print("\nสถิติความยาวของคำถาม:")
    print(f"- ต่ำสุด: {report['question_length_stats']['min']} ตัวอักษร")
    print(f"- สูงสุด: {report['question_length_stats']['max']} ตัวอักษร")
    print(f"- เฉลี่ย: {report['question_length_stats']['avg']:.2f} ตัวอักษร")
    
    print(f"\nคำถามที่ซ้ำกัน: {report['duplicates']} รายการ ({(report['duplicates'] / report['total_samples']) * 100:.2f}%)")
    print(f"คำถามที่ไม่ซ้ำกัน: {report['unique_questions']} รายการ")
    
    print(f"\nบันทึกรายงานและกราฟไปยัง {args.output_dir}")
    print("รายงานเสร็จสมบูรณ์!")

if __name__ == "__main__":
    main()
