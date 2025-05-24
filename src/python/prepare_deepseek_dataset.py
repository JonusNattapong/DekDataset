#!/usr/bin/env python
# coding: utf-8
# prepare_deepseek_dataset.py - เตรียม Dataset จาก Deepseek API

import os
import json
import random
import argparse
from tqdm import tqdm
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from collections import Counter

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

def save_jsonl(data, file_path):
    """บันทึกข้อมูลเป็นไฟล์ JSONL"""
    with open(file_path, 'w', encoding='utf-8') as f:
        for item in data:
            json.dump(item, f, ensure_ascii=False)
            f.write('\n')
    print(f"บันทึกข้อมูล {len(data)} รายการไปยัง {file_path}")

def balance_dataset(data, max_per_group=50):
    """ปรับความสมดุลของข้อมูลตามวิชาและระดับชั้น"""
    print("กำลังปรับความสมดุลของข้อมูล...")
    
    # จัดกลุ่มตามวิชาและระดับชั้น
    groups = {}
    for item in data:
        subject = item.get('subject')
        grade = item.get('grade')
        if subject and grade:
            key = f"{subject}_{grade}"
            if key not in groups:
                groups[key] = []
            groups[key].append(item)
    
    # แสดงจำนวนข้อมูลในแต่ละกลุ่ม
    print("จำนวนข้อมูลในแต่ละกลุ่มก่อนปรับสมดุล:")
    for key, group in sorted(groups.items()):
        print(f"- {key}: {len(group)} รายการ")
    
    # ปรับจำนวนแต่ละกลุ่มให้เท่ากัน
    balanced_data = []
    for key, group in tqdm(groups.items(), desc="ปรับความสมดุลของกลุ่ม"):
        # หากมีข้อมูลมากเกินไป ให้สุ่มเลือก
        if len(group) > max_per_group:
            group = random.sample(group, max_per_group)
        balanced_data.extend(group)
    
    # สับเปลี่ยนข้อมูล
    random.shuffle(balanced_data)
    
    return balanced_data

def split_dataset(data, train_ratio=0.8, validation_ratio=0.1, test_ratio=0.1):
    """แบ่งข้อมูลเป็นส่วน train, validation และ test"""
    print("กำลังแบ่งข้อมูลเป็นส่วน train, validation และ test")
    
    # ตรวจสอบว่า ratio รวมกันเป็น 1.0
    assert abs((train_ratio + validation_ratio + test_ratio) - 1.0) < 1e-10, "อัตราส่วนต้องรวมกันเป็น 1.0"
    
    # สับเปลี่ยนข้อมูล
    random.shuffle(data)
    
    # หาจุดแบ่ง
    train_end = int(len(data) * train_ratio)
    val_end = train_end + int(len(data) * validation_ratio)
    
    # แบ่งข้อมูล
    train_data = data[:train_end]
    val_data = data[train_end:val_end]
    test_data = data[val_end:]
    
    print(f"แบ่งข้อมูลเสร็จสิ้น: train {len(train_data)} รายการ, validation {len(val_data)} รายการ, test {len(test_data)} รายการ")
    
    return train_data, val_data, test_data

def analyze_dataset(data, output_dir):
    """วิเคราะห์ข้อมูลและสร้างกราฟ"""
    print("กำลังวิเคราะห์ข้อมูล...")
    
    # สร้างไดเรกทอรีสำหรับบันทึกกราฟ
    os.makedirs(output_dir, exist_ok=True)
    
    # วิเคราะห์การกระจายตามวิชา
    subjects = [item.get('subject') for item in data if 'subject' in item]
    subject_counts = Counter(subjects)
    
    # วิเคราะห์การกระจายตามระดับชั้น
    grades = [item.get('grade') for item in data if 'grade' in item]
    grade_counts = Counter(grades)
    
    # วิเคราะห์การกระจายตามระดับความยาก
    difficulties = [item.get('difficulty') for item in data if 'difficulty' in item]
    difficulty_counts = Counter(difficulties)
    
    # วิเคราะห์ความยาวของคำถาม
    question_lengths = [len(item.get('question', '')) for item in data]
    
    # วิเคราะห์ส่วน missing data
    missing_data = {
        'subject': sum(1 for item in data if 'subject' not in item or not item['subject']),
        'grade': sum(1 for item in data if 'grade' not in item or not item['grade']),
        'question': sum(1 for item in data if 'question' not in item or not item['question']),
        'choices': sum(1 for item in data if 'choices' not in item or not item['choices']),
        'answer': sum(1 for item in data if 'answer' not in item or not item['answer'])
    }
    
    # สร้างกราฟการกระจายตามวิชา
    plt.figure(figsize=(12, 6))
    sns.barplot(x=list(subject_counts.keys()), y=list(subject_counts.values()))
    plt.title('การกระจายตามวิชา')
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'subject_distribution.png'))
    plt.close()
    
    # สร้างกราฟการกระจายตามระดับชั้น
    plt.figure(figsize=(10, 6))
    grade_labels = [f"ป.{grade}" if grade <= 6 else f"ม.{grade-6}" for grade in sorted(grade_counts.keys())]
    grade_values = [grade_counts[grade] for grade in sorted(grade_counts.keys())]
    sns.barplot(x=grade_labels, y=grade_values)
    plt.title('การกระจายตามระดับชั้น')
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'grade_distribution.png'))
    plt.close()
    
    # สร้างกราฟการกระจายตามระดับความยาก
    if difficulties:
        plt.figure(figsize=(8, 6))
        sns.barplot(x=list(difficulty_counts.keys()), y=list(difficulty_counts.values()))
        plt.title('การกระจายตามระดับความยาก')
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, 'difficulty_distribution.png'))
        plt.close()
    
    # สร้างกราฟการกระจายความยาวของคำถาม
    plt.figure(figsize=(10, 6))
    sns.histplot(question_lengths, kde=True)
    plt.title('การกระจายความยาวของคำถาม')
    plt.xlabel('จำนวนตัวอักษร')
    plt.ylabel('จำนวน')
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'question_length_distribution.png'))
    plt.close()
    
    # สร้างกราฟข้อมูลที่หายไป
    plt.figure(figsize=(8, 6))
    sns.barplot(x=list(missing_data.keys()), y=list(missing_data.values()))
    plt.title('จำนวนข้อมูลที่หายไป')
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'missing_data.png'))
    plt.close()
    
    # สร้าง heatmap แสดงการกระจายตามวิชาและระดับชั้น
    subject_grade_data = []
    for item in data:
        subject = item.get('subject')
        grade = item.get('grade')
        if subject and grade:
            subject_grade_data.append({'subject': subject, 'grade': grade})
    
    if subject_grade_data:
        df = pd.DataFrame(subject_grade_data)
        pivot_table = df.pivot_table(index='subject', columns='grade', values='grade', 
                                     aggfunc='count', fill_value=0)
        
        plt.figure(figsize=(14, 10))
        sns.heatmap(pivot_table, annot=True, cmap="YlGnBu", fmt=".0f")
        plt.title('การกระจายตามวิชาและระดับชั้น')
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, 'subject_grade_heatmap.png'))
        plt.close()
    
    # สร้างรายงานสถิติ
    stats = {
        'total_samples': len(data),
        'subject_distribution': {k: v for k, v in sorted(subject_counts.items(), key=lambda x: x[1], reverse=True)},
        'grade_distribution': {str(k): v for k, v in sorted(grade_counts.items())},
        'difficulty_distribution': {k: v for k, v in sorted(difficulty_counts.items())} if difficulties else {},
        'question_length': {
            'min': min(question_lengths) if question_lengths else 0,
            'max': max(question_lengths) if question_lengths else 0,
            'avg': sum(question_lengths) / len(question_lengths) if question_lengths else 0
        },
        'missing_data': missing_data
    }
    
    # บันทึกรายงานเป็นไฟล์ JSON
    with open(os.path.join(output_dir, 'dataset_stats.json'), 'w', encoding='utf-8') as f:
        json.dump(stats, f, ensure_ascii=False, indent=2)
    
    print(f"วิเคราะห์ข้อมูลเสร็จสิ้น บันทึกผลลัพธ์ไปยัง {output_dir}")
    
    return stats

def main():
    parser = argparse.ArgumentParser(description="เตรียม Dataset จาก Deepseek API")
    parser.add_argument("--input", required=True, help="ไฟล์ dataset ที่ได้จาก Deepseek API")
    parser.add_argument("--output_dir", default="data/output/deepseek-dataset", help="ไดเรกทอรีสำหรับบันทึกผลลัพธ์")
    parser.add_argument("--max_per_group", type=int, default=50, help="จำนวนตัวอย่างสูงสุดต่อกลุ่มวิชาและระดับชั้น")
    parser.add_argument("--train_ratio", type=float, default=0.8, help="สัดส่วนข้อมูลสำหรับ training")
    parser.add_argument("--validation_ratio", type=float, default=0.1, help="สัดส่วนข้อมูลสำหรับ validation")
    parser.add_argument("--test_ratio", type=float, default=0.1, help="สัดส่วนข้อมูลสำหรับ testing")
    parser.add_argument("--skip_balance", action="store_true", help="ข้ามขั้นตอนการปรับสมดุลข้อมูล")
    parser.add_argument("--skip_analysis", action="store_true", help="ข้ามขั้นตอนการวิเคราะห์ข้อมูล")
    args = parser.parse_args()
    
    # สร้างไดเรกทอรีสำหรับ output ถ้ายังไม่มี
    os.makedirs(args.output_dir, exist_ok=True)
    
    # โหลดข้อมูล
    print(f"กำลังโหลดข้อมูลจาก {args.input}...")
    data = load_jsonl(args.input)
    print(f"โหลดข้อมูลเสร็จสิ้น จำนวน {len(data)} รายการ")
    
    # ปรับความสมดุลของข้อมูล
    if not args.skip_balance:
        balanced_data = balance_dataset(data, args.max_per_group)
        balanced_file = os.path.join(args.output_dir, "balanced.jsonl")
        save_jsonl(balanced_data, balanced_file)
    else:
        balanced_data = data
        print("ข้ามขั้นตอนการปรับสมดุลข้อมูล")
    
    # แบ่งข้อมูล
    train_data, val_data, test_data = split_dataset(
        balanced_data, 
        args.train_ratio, 
        args.validation_ratio, 
        args.test_ratio
    )
    
    # บันทึกข้อมูลแยกส่วน
    train_file = os.path.join(args.output_dir, "train.jsonl")
    val_file = os.path.join(args.output_dir, "validation.jsonl")
    test_file = os.path.join(args.output_dir, "test.jsonl")
    
    save_jsonl(train_data, train_file)
    save_jsonl(val_data, val_file)
    save_jsonl(test_data, test_file)
    
    # วิเคราะห์ข้อมูล
    if not args.skip_analysis:
        analysis_dir = os.path.join(args.output_dir, "analysis")
        stats = analyze_dataset(balanced_data, analysis_dir)
        
        print("\n=== สรุปสถิติของ Dataset ===")
        print(f"จำนวนตัวอย่างทั้งหมด: {stats['total_samples']}")
        
        print("\nการกระจายตามวิชา (แสดง 5 วิชาแรก):")
        for subject, count in list(stats['subject_distribution'].items())[:5]:
            print(f"- {subject}: {count} รายการ ({count/stats['total_samples']*100:.1f}%)")
        
        print("\nการกระจายตามระดับชั้น:")
        for grade, count in stats['grade_distribution'].items():
            grade_int = int(grade)
            grade_name = f"ป.{grade_int}" if grade_int <= 6 else f"ม.{grade_int-6}"
            print(f"- {grade_name}: {count} รายการ ({count/stats['total_samples']*100:.1f}%)")
        
        print("\nความยาวของคำถาม:")
        print(f"- ต่ำสุด: {stats['question_length']['min']} ตัวอักษร")
        print(f"- สูงสุด: {stats['question_length']['max']} ตัวอักษร")
        print(f"- เฉลี่ย: {stats['question_length']['avg']:.2f} ตัวอักษร")
        
        if stats['missing_data']:
            missing_total = sum(stats['missing_data'].values())
            if missing_total > 0:
                print("\nข้อมูลที่หายไป:")
                for field, count in stats['missing_data'].items():
                    if count > 0:
                        print(f"- {field}: {count} รายการ ({count/stats['total_samples']*100:.1f}%)")
    
    print("\nเตรียม Dataset เสร็จสมบูรณ์!")
    print(f"ไฟล์สำหรับการเทรน: {train_file} ({len(train_data)} รายการ)")
    print(f"ไฟล์สำหรับการ validation: {val_file} ({len(val_data)} รายการ)")
    print(f"ไฟล์สำหรับการทดสอบ: {test_file} ({len(test_data)} รายการ)")

if __name__ == "__main__":
    main()
