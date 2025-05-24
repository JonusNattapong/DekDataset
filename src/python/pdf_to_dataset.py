import os
import json
import argparse
import pdfplumber
from tqdm import tqdm

def extract_text_from_pdf(pdf_path):
    """ดึงข้อความจากทุกหน้าของ PDF"""
    texts = []
    with pdfplumber.open(pdf_path) as pdf:
        for page in pdf.pages:
            text = page.extract_text()
            if text:
                texts.append(text)
    return texts

def split_to_samples(texts, subject=None, grade=None):
    """แปลงข้อความเป็น dataset ตัวอย่าง (content, question, subject, grade)"""
    samples = []
    for text in texts:
        # ตัวอย่าง logic: แยกแต่ละบรรทัดเป็น 1 sample ถ้ามีเครื่องหมาย ? หรือ ...
        for line in text.split('\n'):
            line = line.strip()
            if not line:
                continue
            # ถ้าเจอ ? หรือ "ข้อ" ให้ถือว่าเป็นคำถาม
            if '?' in line or line.startswith('ข้อ'):
                sample = {
                    "content": line,
                    "question": line,
                    "subject": subject or "ไม่ระบุ",
                    "grade": grade or 0
                }
                samples.append(sample)
            else:
                # อาจเป็นเนื้อหา/บทความ
                sample = {
                    "content": line,
                    "subject": subject or "ไม่ระบุ",
                    "grade": grade or 0
                }
                samples.append(sample)
    return samples

def main():
    parser = argparse.ArgumentParser(description="แปลง PDF เป็น dataset JSONL")
    parser.add_argument("--pdf", required=True, help="ไฟล์ PDF ที่ต้องการแปลง")
    parser.add_argument("--output", required=True, help="ไฟล์ JSONL สำหรับบันทึก dataset")
    parser.add_argument("--subject", default=None, help="ชื่อวิชา (optional)")
    parser.add_argument("--grade", type=int, default=None, help="ระดับชั้น (optional)")
    args = parser.parse_args()

    print(f"ดึงข้อความจาก {args.pdf} ...")
    texts = extract_text_from_pdf(args.pdf)
    print(f"พบ {len(texts)} หน้า")

    print("แปลงข้อความเป็น dataset ...")
    samples = split_to_samples(texts, subject=args.subject, grade=args.grade)
    print(f"สร้าง dataset ได้ {len(samples)} ตัวอย่าง")

    with open(args.output, "w", encoding="utf-8") as f:
        for s in samples:
            f.write(json.dumps(s, ensure_ascii=False) + "\n")
    print(f"บันทึกไฟล์ {args.output} เรียบร้อยแล้ว")

if __name__ == "__main__":
    main()
