import os
import json
import requests
import argparse
from datetime import datetime
from typing import Dict, Any, List
from dotenv import load_dotenv
from task_definitions import get_task_definitions
from banner import print_ascii_banner
from colorama import Fore, Style
import time
import sys
import re

# ----------------- Environment Setup -----------------
load_dotenv()

# ----------------- Models -----------------
class DataEntry:
    def __init__(self, id: str, content: dict, metadata: dict = None):
        self.id = id
        self.content = content
        self.metadata = metadata or {"source": "DeepSeek-V3"}

    def to_dict(self):
        return {
            "id": self.id,
            "content": self.content,
            "metadata": self.metadata
        }

class GeneratedData:
    def __init__(self, task_name: str, format: str, data: List[DataEntry]):
        self.task_name = task_name
        self.format = format
        self.data = data

# ----------------- Task Definitions -----------------
def get_task_definitions() -> Dict[str, dict]:
    try:
        resp = requests.get("http://localhost:8000/tasks", timeout=2)
        resp.raise_for_status()
        return resp.json()
    except Exception:
        # fallback: local import
        from task_definitions import get_task_definitions as imported_get_task_definitions
        return imported_get_task_definitions()

# ----------------- Deepseek API Client -----------------
class DeepseekClient:
    def __init__(self, api_key: str):
        self.api_key = api_key
        self.api_url = "https://api.deepseek.com/chat/completions"

    def generate_dataset_with_prompt(self, task: dict, count: int) -> List[dict]:
        prompt = (
            f"คุณคือ AI สำหรับสร้าง dataset โจทย์: {task['name']}\n"
            f"รายละเอียด: {task['description']}\n"
            f"Schema: {task['schema']['fields']}\n"
            f"โปรดสร้างตัวอย่างข้อมูล {count} ตัวอย่างในรูปแบบ JSON array ตาม schema ข้างต้น "
            f"(ไม่ต้องอธิบายเพิ่ม, หลีกเลี่ยง placeholder, ใช้ภาษาไทยเป็นหลัก)"
        )
        req_body = {
            "model": "deepseek-chat",
            "temperature": 1.5,
            "messages": [
                {"role": "system", "content": "You are a helpful AI dataset generator."},
                {"role": "user", "content": prompt}
            ]
        }
        headers = {"Authorization": f"Bearer {self.api_key}"}
        resp = requests.post(self.api_url, json=req_body, headers=headers)
        resp.raise_for_status()
        resp_json = resp.json()
        content = resp_json["choices"][0]["message"]["content"]
        # Strip code block if present
        if content.strip().startswith("```json"):
            content = content.strip()[7:]
            if content.endswith("```"):
                content = content[:-3]
        content = content.strip()
        if not content:
            print("[ERROR] Deepseek output is empty!")
            return []
        # --- Robust JSON extraction ---
        try:
            # Find first JSON array in content
            match = re.search(r'\[.*?\]', content, re.DOTALL)
            if match:
                array_str = match.group(0)
                parsed = json.loads(array_str)
                if isinstance(parsed, list):
                    return parsed
            # fallback: try normal parse (may error)
            parsed = json.loads(content)
            if isinstance(parsed, list):
                return parsed
        except Exception as e:
            print("Error parsing Deepseek output:", e)
        return []

# ----------------- Main Logic -----------------
def main():
    print_ascii_banner()
    parser = argparse.ArgumentParser()
    parser.add_argument("task", help="Task name (e.g. sentiment_analysis)")
    parser.add_argument("count", type=int, help="Number of samples")
    parser.add_argument("--format", choices=["json", "jsonl"], default="jsonl")
    args = parser.parse_args()

    api_key = os.getenv("DEEPSEEK_API_KEY")
    if not api_key:
        raise RuntimeError("DEEPSEEK_API_KEY must be set in environment")

    tasks = get_task_definitions()
    if args.task not in tasks:
        raise ValueError(f"Task '{args.task}' not found. Available: {list(tasks.keys())}")
    task = tasks[args.task]

    client = DeepseekClient(api_key)
    print(f"\n{Fore.LIGHTCYAN_EX}Generating dataset for task: {args.task} ({args.count} samples)...{Style.RESET_ALL}")

    total = args.count
    # --- Batch Generation ---
    # ปรับ batch size อัตโนมัติ: ถ้า count <= 10 ให้ batch = count, ถ้า count <= 100 ให้ batch = 10, ถ้ามากกว่านั้น batch = 5
    if total <= 10:
        BATCH_SIZE = total
    elif total <= 100:
        BATCH_SIZE = 10
    else:
        BATCH_SIZE = 5
    all_entries = []
    num_batches = (total + BATCH_SIZE - 1) // BATCH_SIZE
    for batch_idx in range(num_batches):
        batch_start = batch_idx * BATCH_SIZE + 1
        batch_count = min(BATCH_SIZE, total - batch_idx * BATCH_SIZE)
        print(f"{Fore.YELLOW}Batch {batch_idx+1}/{num_batches} ({batch_count} samples)...{Style.RESET_ALL}")
        entries = client.generate_dataset_with_prompt(task, batch_count)
        if not entries:
            print(f"[ERROR] No data generated in batch {batch_idx+1}. Skipping this batch.")
            continue
        all_entries.extend(entries)
        # Progress bar per batch
        bar_length = 30
        percent = int(((batch_idx+1)/num_batches)*100)
        bar = f"{Fore.YELLOW}|{'█'*((batch_idx+1)*bar_length//num_batches)}{'.'*(bar_length-((batch_idx+1)*bar_length//num_batches))}|{percent:3d}%{Style.RESET_ALL}"
        sys.stdout.write(f"\r{bar}")
        sys.stdout.flush()
        time.sleep(0.1)
    print()  # Newline after bar

    if not all_entries:
        print(f"[ERROR] No data generated. Deepseek output was empty or invalid. File will not be saved.")
        return

    data_entries = [
        DataEntry(
            id=f"{args.task}-{i+1}",
            content=entry,
            metadata={"source": "DeepSeek-V3"}
        ).to_dict()
        for i, entry in enumerate(all_entries)
    ]

    # Export
    now = datetime.now().strftime("%Y%m%d-%H%M%S")
    filename = f"auto-dataset-{args.task}-{now}.{args.format}"
    output_dir = "data/output"
    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(output_dir, filename)
    if args.format == "jsonl":
        with open(output_path, "w", encoding="utf-8") as f:
            for entry in data_entries:
                f.write(json.dumps(entry, ensure_ascii=False) + "\n")
    else:
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(data_entries, f, ensure_ascii=False, indent=2)
    print(f"Dataset saved to {output_path}")

if __name__ == "__main__":
    main()
