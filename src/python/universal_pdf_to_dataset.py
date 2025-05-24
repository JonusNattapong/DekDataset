import os
import json
import argparse
import pdfplumber
from pdf2image import convert_from_path
import pytesseract
from tqdm import tqdm
from PIL import Image
import re

def extract_text_from_pdf(pdf_path):
    """ดึงข้อความจาก PDF (text-based)"""
    texts = []
    with pdfplumber.open(pdf_path) as pdf:
        for page in pdf.pages:
            text = page.extract_text()
            if text:
                texts.append(text)
    return texts

def extract_text_from_ocr(pdf_path, dpi=300, lang='tha+eng'):
    """ดึงข้อความจาก PDF (image-based/OCR)"""
    images = convert_from_path(pdf_path, dpi=dpi)
    texts = []
    for img in tqdm(images, desc="OCR PDF images"):
        text = pytesseract.image_to_string(img, lang=lang)
        if text:
            texts.append(text)
    return texts

def is_text_pdf(pdf_path):
    """ตรวจสอบว่า PDF เป็น text-based หรือ image-based (scan)"""
    try:
        with pdfplumber.open(pdf_path) as pdf:
            for page in pdf.pages:
                text = page.extract_text()
                if text and len(text.strip()) > 10:
                    return True
    except Exception:
        pass
    return False

def group_questions(texts):
    """รวมกลุ่มข้อความที่เป็นโจทย์/ตัวเลือก/เฉลย เป็น 1 sample ต่อข้อ"""
    samples = []
    current = None
    for text in texts:
        lines = text.split('\n')
        for line in lines:
            line = line.strip()
            if not line:
                continue
            # เริ่มข้อใหม่
            if re.match(r'^(ข้อ|คำถาม) ?\d+', line):
                if current:
                    samples.append(current)
                current = {"raw": line, "choices": [], "answer": None}
            elif re.match(r'^[กขคง]\.', line):
                if current:
                    current.setdefault("choices", []).append(line)
            elif re.match(r'^(เฉลย|คำตอบ)[ :：]?', line):
                if current:
                    ans = re.sub(r'^(เฉลย|คำตอบ)[ :：]?', '', line)
                    current["answer"] = ans
            else:
                if current:
                    current["raw"] += ' ' + line
        if current:
            samples.append(current)
            current = None
    # post-process: แยก question/choices/answer
    result = []
    for s in samples:
        q = {}
        # แยก question ออกจาก raw
        m = re.match(r'^(ข้อ|คำถาม) ?\d+[ .:：)]*(.*)', s.get("raw", ""))
        q["question"] = m.group(2).strip() if m else s.get("raw", "")
        # แยกตัวเลือก
        q["choices"] = [re.sub(r'^[กขคง]\.', '', c).strip() for c in s.get("choices", [])]
        # เฉลย
        q["answer"] = s.get("answer")
        result.append(q)
    return result

def split_to_samples(texts, subject=None, grade=None):
    """แปลงข้อความเป็น dataset ตัวอย่าง (question, choices, answer, subject, grade)"""
    grouped = group_questions(texts)
    samples = []
    for g in grouped:
        sample = {
            "question": g.get("question"),
            "choices": g.get("choices") if g.get("choices") else None,
            "answer": g.get("answer"),
            "subject": subject or "ไม่ระบุ",
            "grade": grade or 0
        }
        # กรองเฉพาะที่มี question จริง
        if sample["question"] and len(sample["question"]) > 5:
            samples.append(sample)
    return samples

def main():
    parser = argparse.ArgumentParser(description="Universal PDF (text/OCR) to dataset JSONL")
    parser.add_argument("--pdf", required=True, help="ไฟล์ PDF ที่ต้องการแปลง")
    parser.add_argument("--output", required=True, help="ไฟล์ JSONL สำหรับบันทึก dataset")
    parser.add_argument("--subject", default=None, help="ชื่อวิชา (optional)")
    parser.add_argument("--grade", type=int, default=None, help="ระดับชั้น (optional)")
    parser.add_argument("--lang", default="tha+eng", help="ภาษา OCR (default: tha+eng)")
    args = parser.parse_args()

    print(f"ตรวจสอบประเภท PDF: {args.pdf}")
    if is_text_pdf(args.pdf):
        print("PDF นี้เป็น text-based, ดึงข้อความโดยตรง...")
        texts = extract_text_from_pdf(args.pdf)
    else:
        print("PDF นี้เป็น scan/image-based, ใช้ OCR ...")
        texts = extract_text_from_ocr(args.pdf, lang=args.lang)
    print(f"รวมข้อความ {len(texts)} หน้า")

    print("แปลงข้อความเป็น dataset ...")
    samples = split_to_samples(texts, subject=args.subject, grade=args.grade)
    print(f"สร้าง dataset ได้ {len(samples)} ตัวอย่าง")

    with open(args.output, "w", encoding="utf-8") as f:
        for s in samples:
            f.write(json.dumps(s, ensure_ascii=False) + "\n")
    print(f"บันทึกไฟล์ {args.output} เรียบร้อยแล้ว")

if __name__ == "__main__":
    main()
