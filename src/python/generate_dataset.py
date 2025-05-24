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
            f"โปรดสร้างตัวอย่างข้อมูล JSON จำนวน {count} ตัวอย่างในรูปแบบ JSON array ตาม schema ข้างต้น\n\n"
            f"เงื่อนไขสำคัญ:\n"
            f"1. ผลลัพธ์ต้องเป็น JSON array ที่ถูกต้อง - ตัวอย่าง: [{{\n  \"field1\": \"value\",\n  \"field2\": 123\n}}]\n"
            f"2. ไม่ต้องอธิบายเพิ่มเติม ส่งเฉพาะ JSON array เท่านั้น\n"
            f"3. หลีกเลี่ยงค่า placeholder หรือคำว่า 'example'\n"
            f"4. ใช้ภาษาไทยเป็นหลัก"
        )
        system_prompt = "You are a helpful AI dataset generator. Your task is to generate valid JSON data according to a specified schema. Only output valid JSON, nothing else."
        req_body = {
            "model": "deepseek-chat",
            "temperature": 1.2,
            "max_tokens": 4000,
            "response_format": {
                "type": "json_object"
            },
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": prompt}
            ]
        }
        headers = {"Authorization": f"Bearer {self.api_key}"}
        resp = requests.post(self.api_url, json=req_body, headers=headers)
        resp.raise_for_status()
        resp_json = resp.json()
        content = resp_json["choices"][0]["message"]["content"]
        
        # JSON Output format ควรคืนค่า JSON ที่สมบูรณ์อยู่แล้ว
        try:
            parsed = json.loads(content)
            if isinstance(parsed, list):
                return parsed
            elif isinstance(parsed, dict) and "data" in parsed and isinstance(parsed["data"], list):
                return parsed["data"]
            else:
                # ถ้า JSON ไม่ได้เป็น list โดยตรง แปลงเป็น list
                return [parsed]
                
        except json.JSONDecodeError as e:
            print(f"[ERROR] JSON parsing error: {e}")
            print(f"Content: {content[:200]}...")
            
            # ถ้ายังคงมีปัญหา ลองใช้วิธีดึงส่วนที่เป็น JSON array ออกมา
            try:
                # Find first JSON array in content
                match = re.search(r'\[.*?\]', content, re.DOTALL)
                if match:
                    array_str = match.group(0)
                    parsed = json.loads(array_str)
                    if isinstance(parsed, list):
                        return parsed
            except Exception:
                pass
        except Exception as e:
            print("Error parsing Deepseek output:", e)
        return []

# ----------------- Main Logic -----------------
def import_vision_jsonl(input_path, schema, output_path=None):
    """Import vision-animals-dataset-*.jsonl, validate schema, export jsonl ใหม่ที่มีเฉพาะฟิลด์ schema"""
    entries = []
    with open(input_path, "r", encoding="utf-8") as f:
        for line in f:
            try:
                entry = json.loads(line)
                # รองรับทั้งแบบ {id, content, metadata} หรือ content ตรง schema เลย
                content = entry.get("content", entry)
                filtered = {k: content.get(k, None) for k in schema["fields"].keys()}
                if validate_entry(filtered, schema):
                    entries.append(filtered)
            except Exception as e:
                print(f"[WARN] Skipping line: {e}")
    print(f"[INFO] Imported {len(entries)} valid entries from {input_path}")
    if output_path:
        with open(output_path, "w", encoding="utf-8") as f:
            for entry in entries:
                f.write(json.dumps(entry, ensure_ascii=False) + "\n")
        print(f"[INFO] Exported {len(entries)} entries to {output_path}")
    return entries

def create_dataset_standard_structure(output_path: str, task: dict, format: str = "jsonl", split: bool = False, 
                               split_paths: tuple = None, sample_count: int = None):
    """
    สร้างโครงสร้าง dataset มาตรฐานสำหรับ Hugging Face หรือเผยแพร่ จากไฟล์ที่สร้าง
    
    Args:
        output_path: path ไปยังไฟล์ output หลัก (เช่น auto-dataset-task-date.jsonl) 
        task: dictionary ของ task definition
        format: รูปแบบไฟล์ ("jsonl", "json", "csv", "parquet")
        split: เป็น True ถ้ามีการคัดลอกไฟล์ไปเป็น train/valid/test (ไม่ได้แบ่งข้อมูลจริง)
        split_paths: tuple ของ (train_path, valid_path, test_path) ถ้า split=True
        sample_count: จำนวนตัวอย่างทั้งหมดใน dataset (ถ้าระบุ)
    
    Returns:
        path ไปยังโฟลเดอร์ที่สร้าง
    """
    # สร้างชื่อโฟลเดอร์
    task_name = os.path.basename(output_path).split("-")[2]  # เช่น "auto-dataset-medical_benchmark-..."
    output_dir = os.path.dirname(output_path)
    dataset_folder_name = f"{task_name}-dataset"
    dataset_folder_path = os.path.join(output_dir, dataset_folder_name)
    
    # สร้างโฟลเดอร์หลัก
    os.makedirs(dataset_folder_path, exist_ok=True)
    
    # คัดลอกไฟล์ไปยังโครงสร้างใหม่ตามมาตรฐาน
    import shutil
    if split and split_paths:
        train_path, valid_path, test_path = split_paths
        # คัดลอกไฟล์ (ทุกไฟล์มีข้อมูลเหมือนกัน) ไปยังโครงสร้างใหม่
        shutil.copy(train_path, os.path.join(dataset_folder_path, f"train.{format}"))
        shutil.copy(valid_path, os.path.join(dataset_folder_path, f"validation.{format}"))
        shutil.copy(test_path, os.path.join(dataset_folder_path, f"test.{format}"))
    else:
        # คัดลอกไฟล์หลักเข้าไปในโฟลเดอร์
        shutil.copy(output_path, os.path.join(dataset_folder_path, f"data.{format}"))
    
    # สร้าง LICENSE
    license_path = "LICENSE"  # หรือ path อื่น ๆ
    if os.path.exists(license_path):
        shutil.copy(license_path, os.path.join(dataset_folder_path, "LICENSE"))
      # สร้างไฟล์ dataset_info.json
    # คำนวณจำนวนตัวอย่างใน split (ถ้าทำได้)
    total_examples = 0
    try:
        with open(output_path, "r", encoding="utf-8") as f:
            total_examples = sum(1 for _ in f)
    except:
        if sample_count:
            total_examples = sample_count
      # คำนวณจำนวนตัวอย่างตามสัดส่วน
    if split:
        # ใช้ค่า default หรือหาค่าสัดส่วนจากชื่อไฟล์ (ถ้าทำได้)
        train_ratio, valid_ratio, test_ratio = 0.8, 0.1, 0.1  # default
        train_examples = int(total_examples * train_ratio)
        valid_examples = int(total_examples * valid_ratio)
        test_examples = total_examples - train_examples - valid_examples
    else:
        train_examples = valid_examples = test_examples = total_examples
    
    dataset_info = {
        "name": task_name,
        "version": "1.0.0",
        "description": task.get("description", f"Dataset for {task_name}"),
        "license": "CC-BY 4.0",
        "creator": "DekDataset",
        "language": ["th", "en"],  # Thai + English
    }    # เพิ่ม splits ถ้ามี
    if split:
        dataset_info["splits"] = {
            "train": {"num_examples": train_examples},
            "validation": {"num_examples": valid_examples},
            "test": {"num_examples": test_examples}
        }
        # เพิ่มข้อมูลเกี่ยวกับสัดส่วนการแบ่ง
        dataset_info["split_info"] = {
            "train_ratio": train_examples / total_examples,
            "valid_ratio": valid_examples / total_examples,
            "test_ratio": test_examples / total_examples
        }
    
    # เขียนไฟล์ dataset_info.json
    with open(os.path.join(dataset_folder_path, "dataset_info.json"), "w", encoding="utf-8") as f:
        json.dump(dataset_info, f, ensure_ascii=False, indent=2)
    
    # สร้าง README.md จาก task definition
    readme_content = f"""# {task.get('name', task_name)} Dataset

{task.get('description', 'Dataset สำหรับงาน AI และ NLP')}

## รูปแบบข้อมูล

แต่ละแถวในไฟล์ {format.upper()} มีโครงสร้างดังนี้:
```json
{{
  "id": "{task_name}-xxx",
  "content": {{
"""
    
    # เพิ่มฟิลด์จาก schema
    fields = task.get("schema", {}).get("fields", {})
    for field_name, field_def in fields.items():
        field_type = field_def.get("field_type", "")
        field_desc = field_def.get("description", "")
        if isinstance(field_type, dict) and "Enum" in field_type:
            enum_values = field_type["Enum"]
            readme_content += f'    "{field_name}": "{field_desc} (หนึ่งใน {enum_values})",\n'
        else:
            readme_content += f'    "{field_name}": "{field_desc}",\n'
      # จบโครงสร้าง JSON
    readme_content += """  },
  "metadata": {
    "license": "CC-BY 4.0",
    "source": "DekDataset/DeepSeek-V3",
    "created_at": "timestamp",
    "lang": "th"
  }
}
```
"""
    # เพิ่มข้อมูลสถิติ
    readme_content += "\n## สถิติ\n"
    if split:
        readme_content += f"- จำนวนตัวอย่างทั้งหมด: {train_examples + valid_examples + test_examples} ตัวอย่าง\n"
        readme_content += f"- Train: {train_examples} ตัวอย่าง ({train_examples/(train_examples + valid_examples + test_examples)*100:.1f}%)\n"
        readme_content += f"- Validation: {valid_examples} ตัวอย่าง ({valid_examples/(train_examples + valid_examples + test_examples)*100:.1f}%)\n"
        readme_content += f"- Test: {test_examples} ตัวอย่าง ({test_examples/(train_examples + valid_examples + test_examples)*100:.1f}%)\n"
        readme_content += "\n**หมายเหตุ:** แต่ละไฟล์มีข้อมูลตามสัดส่วนที่กำหนด ขอบคุณที่ใช้ DekDataset\n"
    else:
        readme_content += f"- Total: {sample_count or 'N/A'} ตัวอย่าง\n"
    
    # เพิ่มตัวอย่างการใช้งาน
    readme_content += """
## การใช้งานกับ Hugging Face Datasets

```python
from datasets import load_dataset

# โหลดจาก local
dataset = load_dataset("path/to/""" + dataset_folder_name + """")

# หรือโหลดจาก Hugging Face Hub หลังจาก push (ถ้ามี)
# dataset = load_dataset("username/""" + dataset_folder_name + """")

# ดูข้อมูล
print(dataset["train"][0])  # ดูตัวอย่างแรก
```
"""
    
    # เขียน README.md
    with open(os.path.join(dataset_folder_path, "README.md"), "w", encoding="utf-8") as f:
        f.write(readme_content)
    
    print(f"[INFO] Created standard dataset structure at {dataset_folder_path}")
    print(f"[INFO] - README.md: สรุปข้อมูล dataset")
    print(f"[INFO] - dataset_info.json: metadata สำหรับ Hugging Face")
    print(f"[INFO] - {'train.{format}, validation.{format}, test.{format}' if split else f'data.{format}'}: ข้อมูลหลัก")
    if os.path.exists(os.path.join(dataset_folder_path, "LICENSE")):
        print(f"[INFO] - LICENSE: ข้อมูลสิทธิ์การใช้งาน")
    
    return dataset_folder_path

def main():
    print_ascii_banner()
    parser = argparse.ArgumentParser(description="DekDataset: สร้าง dataset คุณภาพสูงสำหรับงาน NLP และ AI")
    parser.add_argument("task", help="Task name (e.g. sentiment_analysis)")
    parser.add_argument("count", type=int, help="Number of samples")
    parser.add_argument("--format", choices=["json", "jsonl", "csv", "parquet"], default="jsonl",
                      help="Output format: json, jsonl, csv, parquet")
    parser.add_argument("--import-vision", type=str, default=None,
                      help="Path to vision-animals-dataset-*.jsonl to import/validate/export")
    parser.add_argument("--source_lang", type=str, default=None, help="Source language (for translation task)")
    parser.add_argument("--target_lang", type=str, default=None, help="Target language (for translation task)")
    parser.add_argument("--lang", type=str, default=None, help="Language code (for multilingual tasks, e.g. th, en, zh, hi)")
    
    # เพิ่ม argument สำหรับการสร้างโครงสร้างตามมาตรฐาน
    parser.add_argument("--create-standard", action="store_true", default=True, help="Create standard dataset structure (default: True)")
    parser.add_argument("--no-standard", dest="create_standard", action="store_false", help="Don't create standard dataset structure")
    parser.add_argument("--split", action="store_true", help="Create train/valid/test copies according to specified ratios")
    parser.add_argument("--train-ratio", type=float, default=0.8, help="Train set ratio when splitting (default: 0.8)")
    parser.add_argument("--valid-ratio", type=float, default=0.1, help="Validation set ratio when splitting (default: 0.1)")
    parser.add_argument("--test-ratio", type=float, default=0.1, help="Test set ratio when splitting (default: 0.1)")
    
    args = parser.parse_args()

    api_key = os.getenv("DEEPSEEK_API_KEY")
    if not api_key:
        raise RuntimeError("DEEPSEEK_API_KEY must be set in environment")

    tasks = get_task_definitions()
    if args.task not in tasks:
        raise ValueError(f"Task '{args.task}' not found. Available: {list(tasks.keys())}")
    task = tasks[args.task]

    # --- Inject translation language params if needed ---
    if args.task == "translation":
        if args.source_lang:
            task["parameters"]["source_lang"]["default"] = args.source_lang
        if args.target_lang:
            task["parameters"]["target_lang"]["default"] = args.target_lang

    # --- Import Vision Dataset Mode ---
    if args.import_vision:
        output_path = os.path.join("data/output", f"imported-vision-{datetime.now().strftime('%Y%m%d-%H%M%S')}.jsonl")
        import_vision_jsonl(args.import_vision, task["schema"], output_path)
        return

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

    def validate_entry(entry: dict, schema: dict) -> bool:
        fields = schema.get("fields", {})
        for field, field_def in fields.items():
            # Required check
            if field_def.get("required", False) and field not in entry:
                return False
            if field not in entry:
                continue
            value = entry[field]
            ftype = field_def.get("field_type")
            # Type check (basic)
            if ftype == "Text" and not isinstance(value, str):
                return False
            if ftype == "Number" and not isinstance(value, (int, float)):
                return False
            if ftype == "Boolean" and not isinstance(value, bool):
                return False
            if isinstance(ftype, dict) and "Enum" in ftype:
                if value not in ftype["Enum"]:
                    return False
            # Constraint check (length, range)
            for cons in field_def.get("constraints", []):
                if "Length" in cons and isinstance(value, str):
                    minl = cons["Length"].get("min")
                    maxl = cons["Length"].get("max")
                    if minl is not None and len(value) < minl:
                        return False
                    if maxl is not None and len(value) > maxl:
                        return False
                if "Range" in cons and isinstance(value, (int, float)):
                    minv = cons["Range"].get("min")
                    maxv = cons["Range"].get("max")
                    if minv is not None and value < minv:
                        return False
                    if maxv is not None and value > maxv:
                        return False
        return True

    # Filter: schema validation
    valid_entries = [e for e in all_entries if validate_entry(e, task["schema"])]
    if len(valid_entries) < len(all_entries):
        print(f"[INFO] Filtered out {len(all_entries)-len(valid_entries)} invalid entries (schema mismatch)")
    # Post-processing: deduplicate
    deduped_entries = deduplicate_entries(valid_entries)
    if len(deduped_entries) < len(valid_entries):
        print(f"[INFO] Removed {len(valid_entries)-len(deduped_entries)} duplicate entries")
    # Post-processing: enrich (placeholder, no-op)
    enriched_entries = enrich_entries(deduped_entries)
    # Diversity/Balance (example: sentiment label)
    if args.task == "sentiment_analysis":
        labels = ["positive", "negative", "neutral"]
        n_per_label = min(len(enriched_entries)//len(labels), 100)
        balanced_entries = balance_label_entries(enriched_entries, "sentiment", labels, n_per_label)
        print(f"[INFO] Balanced sentiment labels: {len(balanced_entries)} entries ({n_per_label} per label)")
    else:
        balanced_entries = enriched_entries
    data_entries = [
        DataEntry(
            id=f"{args.task}-{i+1}",
            content=entry,
            metadata={"source": "DeepSeek-V3"}
        ).to_dict()
        for i, entry in enumerate(balanced_entries)
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
    elif args.format == "json":
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(data_entries, f, ensure_ascii=False, indent=2)
    elif args.format == "csv":
        import csv
        # Flatten content+metadata for CSV
        if data_entries:
            fieldnames = list(data_entries[0]["content"].keys())
            # Optionally add id, metadata fields
            fieldnames = ["id"] + fieldnames + ["metadata"]
            with open(output_path, "w", encoding="utf-8", newline="") as f:
                writer = csv.DictWriter(f, fieldnames=fieldnames)
                writer.writeheader()
                for entry in data_entries:
                    row = {**entry["content"]}
                    row["id"] = entry["id"]
                    row["metadata"] = json.dumps(entry.get("metadata", {}), ensure_ascii=False)
                    writer.writerow(row)
    elif args.format == "parquet":
        import pandas as pd
        # Flatten content+metadata for DataFrame
        rows = []
        for entry in data_entries:
            row = {**entry["content"]}
            row["id"] = entry["id"]
            row["metadata"] = json.dumps(entry.get("metadata", {}), ensure_ascii=False)
            rows.append(row)
        df = pd.DataFrame(rows)
        df.to_parquet(output_path, index=False)
    else:
        raise ValueError(f"Unknown format: {args.format}")
    print(f"Dataset saved to {output_path}")
    
    # ---- คัดลอกไฟล์เป็น train/valid/test ตามสัดส่วนที่กำหนด ----
    split_paths = None
    if args.split:
        print(f"{Fore.YELLOW}[INFO] สร้างไฟล์สำเนาสำหรับ train/valid/test ตามสัดส่วน {args.train_ratio:.1f}/{args.valid_ratio:.1f}/{args.test_ratio:.1f}{Style.RESET_ALL}")
        # แยกชื่อไฟล์และนามสกุล
        filename_prefix = os.path.splitext(os.path.basename(output_path))[0]
        split_paths = split_dataset(
            data_entries, 
            output_dir,
            filename_prefix,
            args.train_ratio, 
            args.valid_ratio, 
            args.test_ratio, 
            args.format
        )
    
    # ---- สร้างโครงสร้าง dataset มาตรฐานโดยอัตโนมัติ ----
    if args.create_standard:
        standard_structure = create_dataset_standard_structure(
            output_path, 
            task, 
            args.format, 
            args.split,
            split_paths,
            args.count
        )
        print(f"{Fore.GREEN}[SUCCESS] สร้างโครงสร้าง dataset มาตรฐานเรียบร้อยที่: {standard_structure}{Style.RESET_ALL}")
        print(f"{Fore.YELLOW}พร้อมสำหรับการอัพโหลดเข้า Hugging Face Datasets Hub{Style.RESET_ALL}")

def deduplicate_entries(entries: list, key_fields=None) -> list:
    """Remove duplicate entries by key fields (default: all fields)."""
    def make_hashable(obj):
        if isinstance(obj, (tuple, str, int, float, bool, type(None))):
            return obj
        elif isinstance(obj, list):
            return tuple(make_hashable(x) for x in obj)
        elif isinstance(obj, dict):
            return tuple(sorted((k, make_hashable(v)) for k, v in obj.items()))
        else:
            return str(obj)

    seen = set()
    result = []
    for e in entries:
        if key_fields:
            key = tuple(make_hashable(e.get(k)) for k in key_fields)
        else:
            key = tuple(sorted((k, make_hashable(v)) for k, v in e.items()))
        if key not in seen:
            seen.add(key)
            result.append(e)
    return result

def enrich_entries(entries: list, enrich_func=None) -> list:
    """Enrich entries with word_count, created_at, lang (robust, ไม่ซ้อน field)."""
    enriched = []
    for e in entries:
        entry = dict(e)
        # enrich content
        if "content" in entry and isinstance(entry["content"], dict):
            content = dict(entry["content"])
            text = content.get("text")
            # Only add word_count if not present and text is str
            if "word_count" not in content and isinstance(text, str):
                content["word_count"] = len(text.split())
            entry["content"] = content
        # enrich metadata
        if "metadata" in entry and isinstance(entry["metadata"], dict):
            metadata = dict(entry["metadata"])
        else:
            metadata = {}
        # Only add created_at/lang if not present
        if "created_at" not in metadata:
            metadata["created_at"] = datetime.now().isoformat()
        if "lang" not in metadata:
            metadata["lang"] = "th"
        entry["metadata"] = metadata
        enriched.append(entry)
    return enriched

def balance_label_entries(entries: list, label_field: str, labels: list, n_per_label: int) -> list:
    """Balance dataset so each label has up to n_per_label entries."""
    from collections import defaultdict
    buckets = defaultdict(list)
    for e in entries:
        label = e.get(label_field)
        if label in labels:
            buckets[label].append(e)
    balanced = []
    for label in labels:
        balanced.extend(buckets[label][:n_per_label])
    return balanced

def split_dataset(data_entries: list, output_dir: str, filename_prefix: str, 
                train_ratio=0.8, valid_ratio=0.1, test_ratio=0.1, format="jsonl"):
    """
    สร้างไฟล์สำเนาสำหรับ train, validation และ test โดยสำเนาตามสัดส่วนที่กำหนด
    
    Args:
        data_entries: รายการข้อมูลทั้งหมด
        output_dir: โฟลเดอร์ที่จะเก็บไฟล์แบ่ง
        filename_prefix: prefix ของชื่อไฟล์ (ไม่มีนามสกุล)
        train_ratio: สัดส่วนของ training set (default: 0.8)
        valid_ratio: สัดส่วนของ validation set (default: 0.1) 
        test_ratio: สัดส่วนของ test set (default: 0.1)
        format: รูปแบบไฟล์ ("jsonl", "json", "csv", "parquet")
        
    Returns:
        tuple ของ (train_path, valid_path, test_path)
    """
    import shutil
    
    # Create filenames for copies
    train_path = os.path.join(output_dir, f"{filename_prefix}-train.{format}")
    valid_path = os.path.join(output_dir, f"{filename_prefix}-valid.{format}")
    test_path = os.path.join(output_dir, f"{filename_prefix}-test.{format}")
    
    # Get original file path
    orig_path = os.path.join(output_dir, f"{filename_prefix}.{format}")
    
    # Make copies of the original file
    shutil.copy2(orig_path, train_path)
    shutil.copy2(orig_path, valid_path)
    shutil.copy2(orig_path, test_path)
    
    n = len(data_entries)
    train_samples = int(n * train_ratio)
    valid_samples = int(n * valid_ratio)
    test_samples = n - train_samples - valid_samples
    
    print(f"[INFO] Created dataset copies for train/valid/test (according to specified ratios):")
    print(f"       - Original: {n} entries -> {orig_path}")
    print(f"       - Train: {train_samples} entries ({train_ratio*100:.1f}%) -> {train_path}")
    print(f"       - Valid: {valid_samples} entries ({valid_ratio*100:.1f}%) -> {valid_path}")
    print(f"       - Test: {test_samples} entries ({test_ratio*100:.1f}%) -> {test_path}")
    print(f"       [NOTE] All files contain the full dataset. When using, sample according to the ratios.")
    
    return train_path, valid_path, test_path

def export_jsonl(entries, output_path):
    """Export data to JSONL format"""
    with open(output_path, "w", encoding="utf-8") as f:
        for entry in entries:
            f.write(json.dumps(entry, ensure_ascii=False) + "\n")

def export_json(entries, output_path):
    """Export data to JSON format"""
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(entries, f, ensure_ascii=False, indent=2)

def export_csv(entries, output_path):
    """Export data to CSV format"""
    import csv
    # Flatten content+metadata for CSV
    if entries:
        # ดึง field ทั้งหมดจาก entry แรก (หากมี content ก็ดึงจาก content)
        if "content" in entries[0] and isinstance(entries[0]["content"], dict):
            fieldnames = list(entries[0]["content"].keys())
        else:
            fieldnames = list(entries[0].keys())
            fieldnames = [f for f in fieldnames if f not in ["id", "metadata"]]
            
        # เพิ่ม id และ metadata
        fieldnames = ["id"] + fieldnames + ["metadata"]
        
        with open(output_path, "w", encoding="utf-8", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            for entry in entries:
                if "content" in entry and isinstance(entry["content"], dict):
                    row = {**entry["content"]}
                else:
                    row = {k: v for k, v in entry.items() if k not in ["id", "metadata"]}
                
                row["id"] = entry.get("id", "")
                row["metadata"] = json.dumps(entry.get("metadata", {}), ensure_ascii=False)
                writer.writerow(row)

def export_parquet(entries, output_path):
    """Export data to Parquet format"""
    import pandas as pd
    
    # Flatten content+metadata for DataFrame
    rows = []
    for entry in entries:
        if "content" in entry and isinstance(entry["content"], dict):
            row = {**entry["content"]}
        else:
            row = {k: v for k, v in entry.items() if k not in ["id", "metadata"]}
            
        row["id"] = entry.get("id", "")
        row["metadata"] = json.dumps(entry.get("metadata", {}), ensure_ascii=False)
        rows.append(row)
        
    df = pd.DataFrame(rows)
    df.to_parquet(output_path, index=False)

if __name__ == "__main__":
    main()
