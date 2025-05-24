import os
import json
import argparse
import pdfplumber
from pdf2image import convert_from_path
import pytesseract
from tqdm import tqdm
import requests
import time

def extract_text_from_pdf(pdf_path):
    texts = []
    with pdfplumber.open(pdf_path) as pdf:
        for page in pdf.pages:
            text = page.extract_text()
            if text:
                texts.append(text)
    return texts

def extract_text_from_ocr(pdf_path, dpi=300, lang='tha+eng'):
    images = convert_from_path(pdf_path, dpi=dpi)
    texts = []
    for img in tqdm(images, desc="OCR PDF images"):
        text = pytesseract.image_to_string(img, lang=lang)
        if text:
            texts.append(text)
    return texts

def is_text_pdf(pdf_path):
    try:
        with pdfplumber.open(pdf_path) as pdf:
            for page in pdf.pages:
                text = page.extract_text()
                if text and len(text.strip()) > 10:
                    return True
    except Exception:
        pass
    return False

def call_deepseek_api(text, api_key, schema, task_name, max_retries=3):
    url = "https://api.deepseek.com/chat/completions"
    prompt = (
        f"คุณคือ AI สำหรับสกัดข้อมูลข้อสอบ/โจทย์จากข้อความ PDF\n"
        f"Task: {task_name}\n"
        f"Schema: {schema}\n"
        f"โปรดแปลงข้อความต่อไปนี้เป็น JSON array ตาม schema ข้างต้น (ไม่ต้องอธิบายเพิ่ม ส่งเฉพาะ JSON array ที่ถูกต้อง):\n\n"
        f"---\n{text}\n---"
    )
    system_prompt = "You are a helpful AI dataset generator. Your task is to extract and structure exam/quiz data from text into valid JSON according to the provided schema. Only output valid JSON, nothing else."
    req_body = {
        "model": "deepseek-chat",
        "temperature": 0.1,
        "max_tokens": 4000,
        "response_format": {"type": "json_object"},
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": prompt}
        ]
    }
    headers = {"Authorization": f"Bearer {api_key}"}
    for attempt in range(max_retries):
        try:
            resp = requests.post(url, json=req_body, headers=headers, timeout=60)
            resp.raise_for_status()
            content = resp.json()["choices"][0]["message"]["content"]
            try:
                parsed = json.loads(content)
                if isinstance(parsed, list):
                    return parsed
                elif isinstance(parsed, dict) and "data" in parsed and isinstance(parsed["data"], list):
                    return parsed["data"]
                else:
                    return [parsed]
            except Exception as e:
                print(f"[DeepSeek JSON parse error] {e}")
                print(f"Content: {content[:200]}...")
        except Exception as e:
            print(f"[DeepSeek API error] {e}")
            time.sleep(5)
    return []

def main():
    parser = argparse.ArgumentParser(description="PDF (text/OCR) to dataset with DeepSeek API")
    parser.add_argument("--pdf", required=True, help="PDF file path")
    parser.add_argument("--output", required=True, help="Output JSONL file")
    parser.add_argument("--api_key", required=False, help="DeepSeek API Key (or set DEEPSEEK_API_KEY env)")
    parser.add_argument("--schema_file", required=True, help="Path to JSON schema file (fields only)")
    parser.add_argument("--task_name", default="primary_school_knowledge", help="Task name for prompt context")
    parser.add_argument("--lang", default="tha+eng", help="OCR language (default: tha+eng)")
    parser.add_argument("--group_pages", type=int, default=2, help="รวมข้อความทีละกี่หน้าในการส่ง DeepSeek (default: 2)")
    args = parser.parse_args()

    api_key = args.api_key or os.getenv("DEEPSEEK_API_KEY")
    if not api_key:
        print("[ERROR] กรุณาระบุ --api_key หรือ set DEEPSEEK_API_KEY ใน env")
        return

    with open(args.schema_file, encoding="utf-8") as f:
        schema = json.load(f)

    print(f"ตรวจสอบประเภท PDF: {args.pdf}")
    if is_text_pdf(args.pdf):
        print("PDF นี้เป็น text-based, ดึงข้อความโดยตรง...")
        texts = extract_text_from_pdf(args.pdf)
    else:
        print("PDF นี้เป็น scan/image-based, ใช้ OCR ...")
        texts = extract_text_from_ocr(args.pdf, lang=args.lang)
    print(f"รวมข้อความ {len(texts)} หน้า")

    # รวมข้อความทีละ group_pages หน้า
    groups = []
    for i in range(0, len(texts), args.group_pages):
        group = '\n'.join(texts[i:i+args.group_pages])
        groups.append(group)

    all_samples = []
    for idx, group_text in enumerate(tqdm(groups, desc="DeepSeek extraction")):
        samples = call_deepseek_api(group_text, api_key, schema, args.task_name)
        if samples:
            all_samples.extend(samples)
        time.sleep(2)  # ป้องกัน rate limit

    print(f"รวม dataset ได้ {len(all_samples)} ตัวอย่าง")
    with open(args.output, "w", encoding="utf-8") as f:
        for s in all_samples:
            f.write(json.dumps(s, ensure_ascii=False) + "\n")
    print(f"บันทึกไฟล์ {args.output} เรียบร้อยแล้ว")

if __name__ == "__main__":
    main()
