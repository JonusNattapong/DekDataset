from datasets import load_dataset
import os
import json
from tqdm import tqdm
import time
import requests
from dotenv import load_dotenv
import argparse
load_dotenv()

# --- CONFIG ---
DEEPSEEK_API_KEY = os.getenv("DEEPSEEK_API_KEY")
DEEPSEEK_API_URL = "https://api.deepseek.com/chat/completions"
if DEEPSEEK_API_KEY:
    print(f"[DEBUG] DEEPSEEK_API_KEY loaded: {DEEPSEEK_API_KEY[:6]}*** (length={len(DEEPSEEK_API_KEY)})")
else:
    print("[ERROR] DEEPSEEK_API_KEY is not set or empty!")

# --- Translation function ---
def translate_to_thai_deepseek(text):
    headers = {"Authorization": f"Bearer {DEEPSEEK_API_KEY}", "Content-Type": "application/json"}
    prompt = (
        "คุณคือผู้เชี่ยวชาญด้านการแปลภาษาอังกฤษเป็นภาษาไทยสำหรับข้อมูลทางการแพทย์ "
        "โปรดแปลข้อความต่อไปนี้เป็นภาษาไทยที่ถูกต้องตามบริบทและเหมาะสมกับการใช้งานใน dataset สำหรับ AI/ML "
        "ห้ามแปลทับศัพท์หรือใช้ภาษาอังกฤษปะปน และห้ามอธิบายเพิ่มใด ๆ ให้ตอบกลับเฉพาะข้อความที่แปลแล้วเท่านั้น\n\n"
        f"English: {text}\n"
        "Thai:"
    )
    req_body = {
        "model": "deepseek-chat",
        "temperature": 1.5,
        "messages": [
            {"role": "system", "content": "You are a professional English-to-Thai medical translator for AI datasets."},
            {"role": "user", "content": prompt}
        ]
    }
    for _ in range(3):
        resp = None
        try:
            print(f"[DEBUG] Translating: {text[:60]}...")
            resp = requests.post(DEEPSEEK_API_URL, headers=headers, json=req_body, timeout=30)
            print(f"[DEBUG] Status: {resp.status_code}, Response: {resp.text[:100]}")
            if resp.status_code == 429:
                time.sleep(5)
                continue
            resp.raise_for_status()
            data = resp.json()
            if "choices" not in data or not data["choices"]:
                print(f"[ERROR] Unexpected DeepSeek response: {data}")
                return "TRANSLATION_ERROR"
            result = data["choices"][0]["message"]["content"].strip()
            print(f"[DEBUG] Result: {result[:60]}")
            return result
        except Exception as e:
            print(f"[WARN] Deepseek error: {e}. Response: {getattr(resp, 'text', None)}. Retrying...")
            time.sleep(2)
    return "TRANSLATION_ERROR"

def main():
    parser = argparse.ArgumentParser(description="Translate medical-o1-reasoning-SFT to Thai.")
    parser.add_argument("--max_rows", type=int, default=10, help="Number of rows to translate (default: 10)")
    parser.add_argument("--output", type=str, default=None, help="Output file name (default: auto)")
    args = parser.parse_args()

    if not DEEPSEEK_API_KEY:
        print("[ERROR] DEEPSEEK_API_KEY is not set in environment. Please set it in your .env or shell.")
        return

    # Load dataset from HuggingFace
    ds = load_dataset("FreedomIntelligence/medical-o1-reasoning-SFT", "en")
    out_path = args.output or f"medical-o1-reasoning-SFT-thai-{args.max_rows}rows.jsonl"
    max_rows = args.max_rows
    with open(out_path, "w", encoding="utf-8") as fout:
        for i, row in enumerate(tqdm(ds["train"])):
            if i >= max_rows:
                break
            row_th = dict(row)
            for field, value in row.items():
                if field.endswith("_th"):
                    continue
                if isinstance(value, str) and value.strip():
                    print(f"[DEBUG] Translating field: {field}")
                    row_th[field+"_th"] = translate_to_thai_deepseek(value)
                    time.sleep(1.2)  # polite delay
            fout.write(json.dumps(row_th, ensure_ascii=False) + "\n")
    print(f"Done. Output: {out_path}")

if __name__ == "__main__":
    main()
