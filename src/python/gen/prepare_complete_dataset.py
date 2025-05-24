#!/usr/bin/env python
# coding: utf-8
# prepare_complete_dataset.py - เตรียม Dataset ให้ครบทุกส่วน

import os
import json
import random
import glob
import argparse
import subprocess
from tqdm import tqdm

def run_command(command):
    """รันคำสั่ง command line"""
    print(f"กำลังรัน: {command}")
    result = subprocess.run(command, shell=True, text=True, capture_output=True, encoding="utf-8")
    if result.returncode != 0:
        print(f"เกิดข้อผิดพลาด: {result.stderr}")
    else:
        print(f"สำเร็จ: {result.stdout}")
    return result

def create_directory_if_not_exists(directory):
    """สร้างไดเรกทอรีหากยังไม่มี"""
    if not os.path.exists(directory):
        os.makedirs(directory)
        print(f"สร้างไดเรกทอรี {directory}")

def count_jsonl_entries(file_path):
    """นับจำนวนรายการในไฟล์ JSONL"""
    if not os.path.exists(file_path):
        return 0
    
    count = 0
    with open(file_path, 'r', encoding='utf-8') as f:
        for line in f:
            if line.strip():
                count += 1
    return count

def balance_subjects_and_grades(input_file, output_file, max_per_group=500):
    """ปรับความสมดุลของข้อมูลตามวิชาและระดับชั้น"""
    print(f"กำลังปรับความสมดุลของข้อมูลจาก {input_file} ไปยัง {output_file}")
    
    # อ่านข้อมูลทั้งหมด
    data = []
    with open(input_file, 'r', encoding='utf-8') as f:
        for line in f:
            if line.strip():
                try:
                    item = json.loads(line)
                    data.append(item)
                except json.JSONDecodeError:
                    continue
    
    print(f"อ่านข้อมูลทั้งหมด {len(data)} รายการ")
    
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
    
    # ปรับจำนวนแต่ละกลุ่มให้เท่ากัน
    balanced_data = []
    for key, group in tqdm(groups.items(), desc="กำลังปรับความสมดุลของกลุ่ม"):
        # หากมีข้อมูลมากเกินไป ให้สุ่มเลือก
        if len(group) > max_per_group:
            group = random.sample(group, max_per_group)
        balanced_data.extend(group)
    
    # สับเปลี่ยนข้อมูล
    random.shuffle(balanced_data)
    
    # บันทึกข้อมูล
    with open(output_file, 'w', encoding='utf-8') as f:
        for item in balanced_data:
            json.dump(item, f, ensure_ascii=False)
            f.write('\n')
    
    print(f"บันทึกข้อมูลที่ปรับสมดุลแล้ว {len(balanced_data)} รายการไปยัง {output_file}")
    return len(balanced_data)

def create_train_validation_test_split(input_file, output_dir, train_ratio=0.8, validation_ratio=0.1, test_ratio=0.1):
    """แบ่งข้อมูลเป็นส่วน train, validation และ test"""
    print(f"กำลังแบ่งข้อมูลจาก {input_file} เป็นส่วน train, validation และ test")
    
    # ตรวจสอบว่า ratio รวมกันเป็น 1.0
    assert abs((train_ratio + validation_ratio + test_ratio) - 1.0) < 1e-10, "อัตราส่วนต้องรวมกันเป็น 1.0"
    
    # อ่านข้อมูลทั้งหมด
    data = []
    with open(input_file, 'r', encoding='utf-8') as f:
        for line in f:
            if line.strip():
                try:
                    item = json.loads(line)
                    data.append(item)
                except json.JSONDecodeError:
                    continue
    
    # สับเปลี่ยนข้อมูล
    random.shuffle(data)
    
    # หาจุดแบ่ง
    train_end = int(len(data) * train_ratio)
    val_end = train_end + int(len(data) * validation_ratio)
    
    # แบ่งข้อมูล
    train_data = data[:train_end]
    val_data = data[train_end:val_end]
    test_data = data[val_end:]
    
    # สร้างไฟล์ output
    train_file = os.path.join(output_dir, 'train.jsonl')
    val_file = os.path.join(output_dir, 'validation.jsonl')
    test_file = os.path.join(output_dir, 'test.jsonl')
    
    # บันทึกไฟล์
    with open(train_file, 'w', encoding='utf-8') as f:
        for item in train_data:
            json.dump(item, f, ensure_ascii=False)
            f.write('\n')
    
    with open(val_file, 'w', encoding='utf-8') as f:
        for item in val_data:
            json.dump(item, f, ensure_ascii=False)
            f.write('\n')
    
    with open(test_file, 'w', encoding='utf-8') as f:
        for item in test_data:
            json.dump(item, f, ensure_ascii=False)
            f.write('\n')
    
    print(f"แบ่งข้อมูลเสร็จสิ้น: train {len(train_data)} รายการ, validation {len(val_data)} รายการ, test {len(test_data)} รายการ")
    return train_file, val_file, test_file

def create_dataset_stats(data_dir):
    """สร้างสถิติของ Dataset"""
    print(f"กำลังสร้างสถิติของ Dataset ใน {data_dir}")
    
    stats = {
        "files": {},
        "subjects": {},
        "grades": {},
        "total_samples": 0
    }
    
    # หาไฟล์ JSONL ทั้งหมด
    jsonl_files = glob.glob(os.path.join(data_dir, "*.jsonl"))
    
    for file_path in jsonl_files:
        file_name = os.path.basename(file_path)
        data_count = count_jsonl_entries(file_path)
        stats["files"][file_name] = data_count
        stats["total_samples"] += data_count
        
        # อ่านข้อมูลเพื่อวิเคราะห์
        with open(file_path, 'r', encoding='utf-8') as f:
            for line in f:
                if line.strip():
                    try:
                        item = json.loads(line)
                        
                        # นับตามวิชา
                        subject = item.get('subject')
                        if subject:
                            if subject not in stats["subjects"]:
                                stats["subjects"][subject] = 0
                            stats["subjects"][subject] += 1
                        
                        # นับตามระดับชั้น
                        grade = item.get('grade')
                        if grade:
                            grade_key = f"ป.{grade}" if grade <= 6 else f"ม.{grade-6}"
                            if grade_key not in stats["grades"]:
                                stats["grades"][grade_key] = 0
                            stats["grades"][grade_key] += 1
                    except json.JSONDecodeError:
                        continue
    
    # บันทึกสถิติเป็นไฟล์ JSON
    stats_file = os.path.join(data_dir, "dataset_statistics.json")
    with open(stats_file, 'w', encoding='utf-8') as f:
        json.dump(stats, f, ensure_ascii=False, indent=4)
    
    print(f"สร้างสถิติเสร็จสิ้น และบันทึกไปยัง {stats_file}")
    return stats

def main():
    parser = argparse.ArgumentParser(description="เตรียม Dataset ให้ครบทุกส่วน")
    parser.add_argument("--questions_per_grade", type=int, default=50, help="จำนวนคำถามต่อระดับชั้นต่อวิชาที่จะสร้าง (ค่าเริ่มต้น: 50)")
    parser.add_argument("--augmentations", type=int, default=3, help="จำนวน Data Augmentation ต่อตัวอย่าง (ค่าเริ่มต้น: 3)")
    parser.add_argument("--max_per_group", type=int, default=500, help="จำนวนตัวอย่างสูงสุดต่อกลุ่มวิชาและระดับชั้น (ค่าเริ่มต้น: 500)")
    args = parser.parse_args()
    # สร้างไดเรกทอรีสำหรับข้อมูลถ้ายังไม่มี
    dataset_dir = os.path.join("data", "output", "complete-dataset")
    create_directory_if_not_exists(dataset_dir)

    # 1. สร้างข้อคำถามที่หลากหลาย
    print("\n=== 1. กำลังสร้างข้อคำถามที่หลากหลาย ===")
    diverse_dataset_file = os.path.join(dataset_dir, "diverse-questions.jsonl")
    result = run_command(f"python src/python/generate_diverse_questions.py --output \"{diverse_dataset_file}\" --questions_per_grade {args.questions_per_grade}")
    if not os.path.exists(diverse_dataset_file):
        print(f"[ERROR] ไม่พบไฟล์ {diverse_dataset_file} กรุณาตรวจสอบการสร้างไฟล์!")
        return

    # 2. รวมข้อมูลทั้งหมดที่มีอยู่
    print("\n=== 2. กำลังรวมข้อมูลทั้งหมด ===")
    merged_output_file = os.path.join(dataset_dir, "merged-all.jsonl")
    temp_merge_script = os.path.join(dataset_dir, "temp_merge.py")
    with open(os.path.join("src", "python", "merge_primary_secondary.py"), 'r', encoding='utf-8') as f:
        merge_script = f.read()
    # แก้ไขเพื่อเพิ่มไฟล์ที่สร้างใหม่
    escaped_diverse_file = diverse_dataset_file.replace('\\', '\\\\')
    escaped_merged_file = merged_output_file.replace('\\', '\\\\')
    merge_script = merge_script.replace("'data/output/auto-dataset-primary-exam-sample.jsonl',", f"'data/output/auto-dataset-primary-exam-sample.jsonl',\n    '{escaped_diverse_file}',")
    merge_script = merge_script.replace("'data/output/complete-diverse-dataset.jsonl',", f"'{escaped_diverse_file}',")
    merge_script = merge_script.replace("'data/output/merged-primary-secondary.jsonl'", f"'{escaped_merged_file}'")
    with open(temp_merge_script, 'w', encoding='utf-8') as f:
        f.write(merge_script)
    result = run_command(f"python \"{temp_merge_script}\"")
    if not os.path.exists(merged_output_file):
        print(f"[ERROR] ไม่พบไฟล์ {merged_output_file} กรุณาตรวจสอบการรวมไฟล์!")
        return

    # 3. เพิ่มข้อมูลด้วย Data Augmentation
    print("\n=== 3. กำลังเพิ่มข้อมูลด้วย Data Augmentation ===")
    augmented_file = os.path.join(dataset_dir, "augmented-all.jsonl")
    result = run_command(f"python src/python/data_augmentation.py --input \"{merged_output_file}\" --output \"{augmented_file}\" --augmentations {args.augmentations} --generate")
    if not os.path.exists(augmented_file):
        print(f"[ERROR] ไม่พบไฟล์ {augmented_file} กรุณาตรวจสอบการเพิ่มข้อมูล!")
        return

    # 4. ปรับความสมดุลของข้อมูล
    print("\n=== 4. กำลังปรับความสมดุลของข้อมูล ===")
    balanced_file = os.path.join(dataset_dir, "balanced-all.jsonl")
    balance_subjects_and_grades(augmented_file, balanced_file, max_per_group=args.max_per_group)
    if not os.path.exists(balanced_file):
        print(f"[ERROR] ไม่พบไฟล์ {balanced_file} กรุณาตรวจสอบการปรับสมดุลข้อมูล!")
        return

    # 5. แบ่งข้อมูลเป็น train, validation และ test
    print("\n=== 5. กำลังแบ่งข้อมูลเป็น train, validation และ test ===")
    train_file, val_file, test_file = create_train_validation_test_split(balanced_file, dataset_dir)

    # 6. สร้างสถิติของ Dataset
    print("\n=== 6. กำลังสร้างสถิติของ Dataset ===")
    stats = create_dataset_stats(dataset_dir)

    # แสดงสรุปสถิติ
    print("\n=== สรุปสถิติของ Dataset ===")
    print(f"จำนวนตัวอย่างทั้งหมด: {stats['total_samples']}")
    print("จำนวนตัวอย่างตามวิชา:")
    for subject, count in sorted(stats["subjects"].items()):
        print(f"- {subject}: {count}")
    print("จำนวนตัวอย่างตามระดับชั้น:")
    for grade, count in sorted(stats["grades"].items()):
        print(f"- {grade}: {count}")

    print("\nเตรียม Dataset เสร็จสมบูรณ์! ไฟล์ทั้งหมดถูกบันทึกไว้ที่:", dataset_dir)
    print("ไฟล์สำหรับการเทรน:", train_file)
    print("ไฟล์สำหรับการ validation:", val_file)
    print("ไฟล์สำหรับการทดสอบ:", test_file)

if __name__ == "__main__":
    main()
