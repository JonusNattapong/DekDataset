#!/usr/bin/env python
# coding: utf-8
# data_augmentation.py - สคริปต์สำหรับเพิ่มข้อมูลด้วยเทคนิค data augmentation

import os
import json
import random
from tqdm import tqdm
import nltk
import re
import argparse

# ตรวจสอบและดาวน์โหลด NLTK resource ที่จำเป็น
try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    nltk.download('punkt')

def load_data(file_path):
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

def save_data(data, file_path):
    """บันทึกข้อมูลเป็นไฟล์ JSONL"""
    with open(file_path, 'w', encoding='utf-8') as f:
        for item in data:
            json.dump(item, f, ensure_ascii=False)
            f.write('\n')

def simple_tokenize(text):
    """ฟังก์ชั่นแบ่งคำแบบง่ายๆ ไม่ต้องใช้ NLTK"""
    # แบ่งคำด้วยช่องว่างและเครื่องหมายวรรคตอน
    return re.findall(r'\b\w+\b|[^\w\s]', text)

def augment_text(text, probability=0.3):
    """เพิ่มข้อมูลโดยการสับเปลี่ยนคำบางคำ (simple word swap)"""
    if not text:
        return text
    
    # แบ่งข้อความเป็นคำด้วยฟังก์ชั่นที่เขียนเอง
    words = simple_tokenize(text)
    if len(words) <= 3:  # ข้อความสั้นเกินไปไม่ต้องเปลี่ยน
        return text
    
    # สับเปลี่ยนคำบางคำ
    for i in range(len(words)):
        if random.random() < probability and len(words[i]) > 3:
            # สลับตำแหน่งตัวอักษรในคำ (เฉพาะคำที่ยาวพอ)
            chars = list(words[i])
            mid_chars = chars[1:-1]
            if len(mid_chars) > 1:
                random.shuffle(mid_chars)
                words[i] = chars[0] + ''.join(mid_chars) + chars[-1]
    
    return ' '.join(words)

def augment_choices(choices, probability=0.3):
    """เพิ่มข้อมูลโดยการสับเปลี่ยนตัวเลือก"""
    if not choices or not isinstance(choices, list):
        return choices
    
    # สลับตำแหน่งตัวเลือก
    if random.random() < probability:
        random.shuffle(choices)
    
    # เพิ่มข้อมูลในบางตัวเลือก
    augmented_choices = []
    for choice in choices:
        if random.random() < probability:
            augmented_choices.append(augment_text(choice, probability=0.2))
        else:
            augmented_choices.append(choice)
    
    return augmented_choices

def augment_questions(data, num_augmentations=3):
    """สร้างข้อมูลเพิ่มจากคำถามที่มีอยู่"""
    augmented_data = []
    
    for item in tqdm(data, desc="Augmenting questions"):
        # เพิ่มข้อมูลดั้งเดิม
        augmented_data.append(item)
        
        # สร้างข้อมูลเพิ่มเติม
        for i in range(num_augmentations):
            new_item = item.copy()
            
            # เพิ่มข้อมูลในคำถาม
            if "question" in new_item and new_item["question"]:
                new_item["question"] = augment_text(new_item["question"])
            
            # เพิ่มข้อมูลในตัวเลือก
            if "choices" in new_item and new_item["choices"]:
                original_choices = new_item["choices"]
                new_choices = augment_choices(original_choices)
                new_item["choices"] = new_choices
                
                # ปรับปรุงคำตอบหากตำแหน่งตัวเลือกเปลี่ยน
                if "answer" in new_item and original_choices and isinstance(original_choices, list):
                    original_answer = new_item["answer"]
                    if original_answer in original_choices:
                        answer_index = original_choices.index(original_answer)
                        new_item["answer"] = new_choices[answer_index]
            
            # ระบุว่าเป็นข้อมูลที่เพิ่มขึ้น
            new_item["is_augmented"] = True
            new_item["augmentation_method"] = "word_swap_and_shuffle"
            
            augmented_data.append(new_item)
    
    return augmented_data

def create_exam_questions(subjects, grades, num_questions=20):
    """สร้างข้อสอบใหม่โดยใช้ template"""
    exam_data = []
    
    # Template สำหรับข้อสอบวิชาต่างๆ
    templates = {
        "ภาษาไทย": [
            {"question": "คำในข้อใดมีความหมายตรงกับคำว่า {word}", "unit": "คำและความหมาย"},
            {"question": "ข้อใดเป็นคำ{word_type}", "unit": "ชนิดของคำ"},
            {"question": "ข้อความใดใช้ภาษาได้ถูกต้อง", "unit": "การใช้ภาษา"}
        ],
        "คณิตศาสตร์": [
            {"question": "ข้อใดเป็นคำตอบของ {equation}", "unit": "การคำนวณ"},
            {"question": "{number1} + {number2} เท่ากับเท่าไร", "unit": "การบวกลบเลข"},
            {"question": "ข้อใดเป็นผลลัพธ์ของ {number1} คูณด้วย {number2}", "unit": "การคูณ"}
        ],
        "วิทยาศาสตร์": [
            {"question": "ข้อใดเป็น{science_topic}", "unit": "ความรู้ทั่วไปเกี่ยวกับวิทยาศาสตร์"},
            {"question": "ข้อใดไม่ใช่{science_type}", "unit": "การจำแนกประเภท"},
            {"question": "{process} เกี่ยวข้องกับข้อใด", "unit": "กระบวนการทางวิทยาศาสตร์"}
        ],
        "สังคมศึกษา": [
            {"question": "ข้อใดเป็นเหตุการณ์ที่เกิดขึ้นในรัชสมัย{king}", "unit": "ประวัติศาสตร์"},
            {"question": "ข้อใดไม่ใช่ลักษณะของ{geography_feature}", "unit": "ภูมิศาสตร์"},
            {"question": "วัฒนธรรม{culture_type}มีลักษณะอย่างไร", "unit": "วัฒนธรรมและสังคม"}
        ]
    }
    
    # สร้างข้อสอบใหม่
    for subject in subjects:
        if subject not in templates:
            continue
            
        for grade in grades:
            for _ in range(num_questions):
                template = random.choice(templates[subject])
                
                question = template["question"]
                unit = template["unit"]
                
                # สุ่มตัวเลือก
                choices = ["ตัวเลือก 1", "ตัวเลือก 2", "ตัวเลือก 3", "ตัวเลือก 4"]
                answer = random.choice(choices)
                
                # สร้างข้อมูลใหม่
                new_item = {
                    "grade": grade,
                    "subject": subject,
                    "unit": unit,
                    "question": question,
                    "choices": choices,
                    "answer": answer,
                    "is_generated": True,
                    "generation_method": "template_based"
                }
                
                exam_data.append(new_item)
    
    return exam_data

def main():
    parser = argparse.ArgumentParser(description="เพิ่มข้อมูลด้วยเทคนิค data augmentation")
    parser.add_argument("--input", required=True, help="ไฟล์ JSONL ที่มีข้อมูลต้นฉบับ")
    parser.add_argument("--output", required=True, help="ไฟล์ JSONL ที่จะบันทึกข้อมูลที่เพิ่มแล้ว")
    parser.add_argument("--augmentations", type=int, default=3, help="จำนวนข้อมูลใหม่ต่อข้อมูลต้นฉบับ")
    parser.add_argument("--generate", action="store_true", help="สร้างข้อสอบใหม่ด้วย template")
    parser.add_argument("--num_generated", type=int, default=20, help="จำนวนข้อสอบที่จะสร้างต่อวิชาต่อระดับชั้น")
    args = parser.parse_args()
    
    # โหลดข้อมูล
    data = load_data(args.input)
    print(f"โหลดข้อมูล {len(data)} รายการจาก {args.input}")
    
    # เพิ่มข้อมูล
    augmented_data = augment_questions(data, num_augmentations=args.augmentations)
    print(f"สร้างข้อมูลเพิ่มเติมรวม {len(augmented_data)} รายการ")
    
    # สร้างข้อสอบใหม่ด้วย template ถ้ากำหนด
    if args.generate:
        subjects = ["ภาษาไทย", "คณิตศาสตร์", "วิทยาศาสตร์", "สังคมศึกษา"]
        primary_grades = list(range(1, 7))  # ป.1-6
        secondary_grades = list(range(7, 13))  # ม.1-6
        
        primary_data = create_exam_questions(subjects, primary_grades, args.num_generated)
        secondary_data = create_exam_questions(subjects, secondary_grades, args.num_generated)
        
        generated_data = primary_data + secondary_data
        print(f"สร้างข้อสอบใหม่ {len(generated_data)} รายการ")
        
        # รวมข้อมูล
        augmented_data.extend(generated_data)
    
    # บันทึกข้อมูล
    save_data(augmented_data, args.output)

if __name__ == "__main__":
    main()
