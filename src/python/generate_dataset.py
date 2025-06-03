import os
import json
import requests
import argparse
import random
import shutil
import pandas as pd
import csv
from datetime import datetime
from typing import Dict, Any, List, Union, Optional, Tuple
from dotenv import load_dotenv
from task_definitions import get_task_definitions
from banner import print_ascii_banner
from colorama import Fore, Style
import time
import sys
import re

# Import data cleaning utilities
from data_utils import (
    clean_text,
    analyze_dataset,
    plot_word_cloud,
    plot_category_distribution,
    plot_length_distribution,
    plot_word_frequency,
    upload_to_huggingface,
)
from ocr_utils import extract_text_from_file
from social_media_utils import extract_social_media_comments, extract_twitter_comments, extract_reddit_comments, extract_youtube_comments

# ----------------- Environment Setup -----------------
load_dotenv()

# Environment variables
api_key = os.getenv("DEEPSEEK_API_KEY")
twitter_api_key = os.getenv("TWITTER_BEARER_TOKEN")
reddit_client_id = os.getenv("REDDIT_CLIENT_ID")
reddit_client_secret = os.getenv("REDDIT_CLIENT_SECRET")
youtube_api_key = os.getenv("YOUTUBE_API_KEY")

# ----------------- Models -----------------
class DataEntry:
    def __init__(self, id: str, content: dict, metadata: dict = None):
        self.id = id
        self.content = content
        self.metadata = metadata or {"source": "DeepSeek-V3"}

    def to_dict(self):
        return {"id": self.id, "content": self.content, "metadata": self.metadata}


class GeneratedData:
    def __init__(self, task_name: str, format: str, data: List[DataEntry]):
        self.task_name = task_name
        self.format = format
        self.data = data


def build_data_entries_from_raw(entries, task_id, metadata):
    """
    สร้าง data_entries ในรูปแบบเดียวกับ generate_dataset.py
    """
    data_entries = []
    for i, entry in enumerate(entries):
        content = entry.get("content", entry)
        if "metadata" in content:
            del content["metadata"]
        data_entries.append({
            "id": f"{task_id}-{i+1}",
            "content": content,
            "metadata": metadata
        })
    return data_entries
# ----------------- Task Definitions -----------------
def get_task_definitions() -> Dict[str, dict]:
    try:
        resp = requests.get("http://localhost:8000/tasks", timeout=2)
        resp.raise_for_status()
        return resp.json()
    except Exception:
        # fallback: local import
        from task_definitions import (
            get_task_definitions as imported_get_task_definitions,
        )

        return imported_get_task_definitions()


# ----------------- Deepseek API Client -----------------
class DeepseekClient:
    def __init__(
        self, api_key: str, model: str = "deepseek-chat", temperature: float = 1.0
    ):
        """
        ตัวจัดการการเชื่อมต่อกับ Deepseek API มีฟังก์ชันสำหรับสร้าง dataset

        Args:
            api_key: API key สำหรับ Deepseek
            model: ชื่อโมเดลที่ต้องการใช้ (default: "deepseek-chat")
            temperature: ค่า temperature ของการ generate (default: 1.0)
        """
        self.api_key = api_key
        self.api_url = "https://api.deepseek.com/chat/completions"
        self.model = model
        self.temperature = temperature
        self.max_retries = 3
        self.retry_delay = 5
        self.cache = {}  # Cache สำหรับเก็บ prompt templates

    def create_optimized_prompt(
        self,
        task: dict,
        count: int,
        examples: List[dict] = None,
        advanced_option: bool = True,
    ) -> str:
        """
        สร้าง prompt ที่เหมาะสมสำหรับการ generate dataset ตาม task ที่กำหนด
        รองรับทั้ง prompt แบบปกติและแบบขั้นสูง

        Args:
            task: ข้อมูล task จาก task_definitions
            count: จำนวนตัวอย่างที่ต้องการ generate
            examples: ตัวอย่างข้อมูลที่มีคุณภาพดี (ถ้ามี จะใช้เป็น few-shot examples)
            advanced_option: ใช้ prompt แบบขั้นสูงหรือไม่ (default: True)

        Returns:
            prompt ที่เหมาะสมสำหรับ Deepseek API
        """
        # ถ้ามี prompt ใน cache แล้ว ใช้จาก cache (แต่ยัง replace ตัวแปรอยู่)
        cache_key = f"{task['name']}_{advanced_option}"
        if cache_key in self.cache:
            prompt_template = self.cache[cache_key]
            # แทนที่ค่าตัวแปรใน prompt
            prompt = prompt_template.replace("{count}", str(count))
            prompt = prompt.replace(
                "{schema}",
                json.dumps(task["schema"]["fields"], indent=2, ensure_ascii=False),
            )
            prompt = prompt.replace("{task_name}", task["name"])
            prompt = prompt.replace("{task_description}", task["description"])

            # ใส่ examples ถ้ามี
            if examples and "{examples}" in prompt:
                prompt = prompt.replace(
                    "{examples}",
                    json.dumps(
                        examples[: min(3, len(examples))], indent=2, ensure_ascii=False
                    ),
                )
            else:
                prompt = prompt.replace("{examples}", "")

            return prompt

        # สร้างคำอธิบาย schema ที่ชัดเจน
        schema_description = json.dumps(
            task["schema"]["fields"], indent=2, ensure_ascii=False
        )

        if not advanced_option:
            # ส่วนหลักของ prompt แบบพื้นฐาน
            prompt = (
                f"คุณคือ AI ผู้เชี่ยวชาญในการสร้าง dataset คุณภาพสูงสำหรับงาน NLP และ AI ในภาษาไทย\n\n"
                f"โจทย์: {task['name']}\n"
                f"รายละเอียด: {task['description']}\n"
                f"Schema: {schema_description}\n\n"
                f"กรุณาสร้างตัวอย่างข้อมูลที่มีคุณภาพสูง จำนวน {count} ตัวอย่าง ในรูปแบบ JSON array ตาม schema ข้างต้น\n\n"
            )

            # เพิ่มรายละเอียดเกี่ยวกับคุณภาพข้อมูล
            prompt += (
                f"คุณสมบัติของข้อมูลที่ต้องการ:\n"
                f"1. ข้อความต้องมีความหลากหลาย ไม่ซ้ำกัน และสมจริง\n"
                f"2. ข้อความทุกรายการต้องเป็นภาษาไทยที่ถูกต้อง (ยกเว้นมีการระบุให้ใช้ภาษาอื่น)\n"
                f"3. ความยาวของข้อความต้องมีความแตกต่างกัน และครอบคลุมหลากหลายหัวข้อ\n"
                f"4. ข้อมูลต้องมีความสมดุลในทุกประเภท (ถ้ามีการแบ่งประเภท เช่น sentiment)\n"
            )

            # Few-shot examples ถ้ามี
            if examples and len(examples) > 0:
                prompt += f"\nตัวอย่างข้อมูลที่ดี:\n{json.dumps(examples[:min(3, len(examples))], indent=2, ensure_ascii=False)}\n\n"

            # เงื่อนไขสำคัญ
            prompt += (
                f"เงื่อนไขสำคัญ:\n"
                f"1. ผลลัพธ์ต้องเป็น JSON array ที่ถูกต้อง เช่น: [{{'field1': 'value', 'field2': 123}}]\n"
                f"2. ไม่ต้องอธิบายเพิ่มเติม ส่งเฉพาะ JSON array เท่านั้น\n"
                f"3. ห้ามใช้คำว่า 'example', 'ตัวอย่าง', หรือ placeholder อื่นๆ ในเนื้อหา\n"
                f"4. ใช้ภาษาไทยเป็นหลัก\n"
                f"5. ข้อมูลต้องเป็นไปตามเงื่อนไขของ Schema ที่กำหนด"
            )

        else:
            # สร้าง prompt ขั้นสูงที่ออกแบบมาสำหรับ dataset generation โดยเฉพาะ
            prompt = f"""
# Task: {task['name']} Dataset Generation

## Description
{task['description']}

## Schema
```json
{schema_description}
```

## Example Format
[{{
    "field1": "value1",
    "field2": "value2"
}}]

## Requirements
สร้างชุดข้อมูล JSON จำนวน {count} รายการตาม Schema ข้างต้น โดยคำนึงถึงคุณภาพและความหลากหลาย:
- ใช้ภาษาไทยเท่านั้น (100% Thai language only) ห้ามใช้ภาษาอื่นแม้แต่คำเดียว
- ข้อความต้องสมจริง หลากหลาย และไม่ซ้ำซาก
- ความยาวต้องหลากหลาย (ตั้งแต่สั้นถึงยาว) แต่มีความสมบูรณ์
- เนื้อหาต้องกระจายตัวดี ครอบคลุมหลายบริบทและหัวข้อ
- ใช้ภาษาไทยที่เป็นธรรมชาติ ถูกต้องตามหลักภาษาไทย
- หลีกเลี่ยงข้อมูลซ้ำ คำซ้ำ และโครงสร้างประโยคซ้ำ
- ไม่มีคำว่า "example", "ตัวอย่าง" หรือ placeholder ในเนื้อหา
- เน้นความสมดุลระหว่าง category/label (ถ้ามี)
- คำศัพท์เฉพาะทางหรือคำทับศัพท์ให้ใช้คำภาษาไทยที่เป็นที่ยอมรับ

## Output Format
JSON Array ที่มีโครงสร้างตาม Schema โดยไม่มีข้อความนำหรือสรุป เช่น:
```json
[
  {{
    "field1": "value1",
    "field2": "value2"
  }},
  {{
    "field1": "value3",
    "field2": "value4"
  }}
]
```
"""

            # เพิ่มคำแนะนำเฉพาะ task
            if "sentiment_analysis" in task["name"]:
                prompt += """
## เฉพาะสำหรับ Sentiment Analysis
- ต้องมีความสมดุลระหว่างความรู้สึกเชิงบวก/เชิงลบ/กลาง
- สร้างข้อความที่มีอารมณ์ชัดเจนสำหรับ positive/negative และกำกวนสำหรับ neutral
- ใช้บริบทที่หลากหลาย (รีวิวสินค้า, บริการ, ร้านอาหาร, โรงแรม, ความคิดเห็นทั่วไป)
- ความยาวของข้อความควรมีตั้งแต่สั้น กลาง ยาว
"""
            elif "translation" in task["name"]:
                prompt += """
## เฉพาะสำหรับ Translation
- คำแปลต้องมีความหมายตรงกับต้นฉบับ แต่ไม่จำเป็นต้องแปลคำต่อคำ
- คงความหมายและอารมณ์ของต้นฉบับ รวมถึงสำนวนที่เหมาะสม
- ใช้ภาษาที่เป็นธรรมชาติในภาษาปลายทาง
- ครอบคลุมหัวข้อที่หลากหลาย (ข่าว, บทสนทนา, วิชาการ, บันเทิง)
"""
            elif "medical" in task["name"]:
                prompt += """
## เฉพาะสำหรับงานด้านการแพทย์
- ใช้คำศัพท์ทางการแพทย์ที่ถูกต้อง แต่ผสมคำศัพท์ทั่วไปเพื่อความเป็นธรรมชาติ
- ครอบคลุมโรคและอาการที่หลากหลาย
- รวมทั้งข้อมูลทั่วไป และข้อมูลเฉพาะทาง
- คำนึงถึงความถูกต้องทางการแพทย์
"""
            elif "question_answering" in task["name"]:
                prompt += """
## เฉพาะสำหรับ Question Answering
- สร้างคำถามที่หลากหลาย (คำถาม What, When, Where, Why, How)
- ความยาวของคำถามควรมีทั้งคำถามสั้น และคำถามยาว
- คำตอบควรมีทั้งยาวและสั้น ขึ้นอยู่กับคำถาม
- ครอบคลุมหัวข้อต่างๆ อย่างกว้างขวาง
"""

            # เพิ่ม examples ถ้ามี
            if examples and len(examples) > 0:
                examples_json = json.dumps(
                    examples[: min(3, len(examples))], indent=2, ensure_ascii=False
                )
                prompt += f"""
## Examples
ตัวอย่างข้อมูลที่ดี:
```json
{examples_json}
```
"""

            # เพิ่มคำเตือนเกี่ยวกับการ validate
            prompt += """
## Validation
ข้อมูลทุกรายการจะถูกตรวจสอบความถูกต้องตาม schema หลังจาก generation โดยอัตโนมัติ โปรดตรวจสอบว่าข้อมูลทุกรายการตรงตาม schema ที่กำหนด ไม่มี field ที่หายไป และค่าทุกค่ามีความหมาย

โปรดส่งเฉพาะ JSON array เท่านั้น ไม่ต้องใส่คำอธิบายหรือข้อความอื่นๆ
"""

        # เก็บ prompt template ไว้ใน cache
        self.cache[cache_key] = prompt
        return prompt

    def generate_dataset_with_prompt(
        self,
        task: dict,
        count: int,
        examples: List[dict] = None,
        advanced_prompt: bool = True,
    ) -> List[dict]:
        """
        สร้าง dataset โดยเรียกใช้ Deepseek API ด้วย prompt ที่กำหนด

        Args:
            task: ข้อมูล task จาก task_definitions
            count: จำนวนตัวอย่างที่ต้องการ generate
            examples: ตัวอย่างข้อมูลที่มีคุณภาพดี (ถ้ามี)
            advanced_prompt: ใช้ prompt แบบขั้นสูงหรือไม่ (default: True)

        Returns:
            รายการข้อมูลที่ได้จาก API
        """
        prompt = self.create_optimized_prompt(task, count, examples, advanced_prompt)

        # ปรับปรุง system prompt ให้มีประสิทธิภาพมากขึ้น
        system_prompt = (
            "You are a highly skilled dataset generator specialized in creating high-quality Thai language datasets. "
            "You must follow the schema exactly and only output valid JSON data. "
            "Your outputs should be diverse in length, style and content while maintaining natural Thai language usage. "
            "Only output the JSON array with no additional text, explanations or comments."
        )

        # ปรับปรุงประสิทธิภาพของ API request
        req_body = {
            "model": self.model,
            "temperature": self.temperature,
            "max_tokens": min(count * 200, 4000),  # ปรับขนาด token ตามจำนวนตัวอย่าง
            "response_format": {"type": "json_object"},
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": prompt},
            ],
        }

        headers = {"Authorization": f"Bearer {self.api_key}"}

        # ใช้ exponential backoff สำหรับการ retry
        for attempt in range(self.max_retries):
            try:
                # ทำ request พร้อมบันทึกเวลาที่ใช้
                start_time = time.time()
                resp = requests.post(
                    self.api_url, json=req_body, headers=headers, timeout=60
                )
                elapsed_time = time.time() - start_time

                resp.raise_for_status()
                resp_json = resp.json()
                content = resp_json["choices"][0]["message"]["content"]

                # พยายาม parse JSON
                data = self.parse_response_content(content)

                # บันทึก log สำหรับการวิเคราะห์
                print(
                    f"[INFO] API request completed in {elapsed_time:.2f}s, generated {len(data)} entries"
                )

                return data

            except requests.exceptions.RequestException as e:
                if attempt == self.max_retries - 1:
                    print(
                        f"[ERROR] API request failed after {self.max_retries} attempts: {e}"
                    )
                    return []

                # คำนวณเวลารอแบบ exponential backoff
                wait_time = self.retry_delay * (2**attempt)
                print(f"[WARN] API request failed: {e}, retrying in {wait_time}s...")
                time.sleep(wait_time)

            except Exception as e:
                if attempt == self.max_retries - 1:
                    print(f"[ERROR] Unexpected error: {e}")
                    return []

                wait_time = self.retry_delay * (2**attempt)
                print(f"[WARN] Unexpected error: {e}, retrying in {wait_time}s...")
                time.sleep(wait_time)

        return []

    def parse_response_content(self, content: str) -> List[dict]:
        """แปลงข้อความตอบกลับจาก API เป็น list ของ dict"""

        def clean_string(s: str) -> str:
            return re.sub(r"\s+", " ", s.strip())

        def clean_dict(d: dict) -> dict:
            cleaned = {}
            for k, v in d.items():
                if isinstance(v, str):
                    v = clean_string(v)
                cleaned[k] = v
            return cleaned

        def extract_array_from_text(text: str) -> List[dict]:
            matches = list(
                re.finditer(r"\[(?:[^[\]]*|\[(?:[^[\]]*|\[[^[\]]*\])*\])*\]", text)
            )
            for match in matches:
                try:
                    array_str = clean_string(match.group(0))
                    parsed = json.loads(array_str)
                    if isinstance(parsed, list):
                        return [
                            clean_dict(item)
                            for item in parsed
                            if isinstance(item, dict)
                        ]
                except:
                    continue
            return []

        # ทำความสะอาดข้อความก่อน parse
        content = clean_string(content)

        try:
            # พยายาม parse ทั้งข้อความ
            parsed = json.loads(content)

            if isinstance(parsed, list):
                items = [clean_dict(item) for item in parsed if isinstance(item, dict)]
                if items:
                    return items
            elif isinstance(parsed, dict):
                # ตรวจสอบ key ที่อาจเป็น array ของข้อมูล
                for key in ["data", "results", "entries", "items"]:
                    if key in parsed and isinstance(parsed[key], list):
                        items = [
                            clean_dict(item)
                            for item in parsed[key]
                            if isinstance(item, dict)
                        ]
                        if items:
                            return items
                # ถ้าไม่เจอ array เลย แปลง dict เป็น list
                return [clean_dict(parsed)]
        except json.JSONDecodeError as e:
            print(
                f"[WARN] Initial parsing failed: {e}, trying to extract JSON array..."
            )
            items = extract_array_from_text(content)
            if items:
                return items
            print("[ERROR] Could not extract any valid JSON")
        except Exception as e:
            print(f"[ERROR] Unexpected error while parsing: {e}")

        return []

    def generate_dataset_batch(
        self,
        task: dict,
        total_count: int,
        batch_size: int = 10,
        max_concurrent: int = 1,
        delay: int = 2,
        clean_data: bool = True,
        clean_options: Dict = None,
        advanced_prompt: bool = True,
        post_process: bool = True,
    ) -> List[dict]:
        """
        สร้าง dataset โดยแบ่งเป็น batch และทำ concurrent requests (ถ้ากำหนด)
        มีฟังก์ชันขั้นสูงเพิ่มเติม เช่น การทำ post-processing, การบันทึก log และ dynamic batch size

        Args:
            task: ข้อมูล task จาก task_definitions
            total_count: จำนวนตัวอย่างทั้งหมดที่ต้องการ
            batch_size: จำนวนตัวอย่างต่อ batch
            max_concurrent: จำนวน concurrent requests สูงสุด (ยังไม่รองรับ > 1)
            delay: เวลารอระหว่าง batches (วินาที)
            clean_data: ทำความสะอาดข้อมูลหรือไม่
            clean_options: ตัวเลือกสำหรับการทำความสะอาดข้อมูล
            advanced_prompt: ใช้ prompt แบบขั้นสูงหรือไม่
            post_process: ทำ post-processing หรือไม่ (deduplication, validation, etc.)

        Returns:
            รายการข้อมูลทั้งหมดที่สร้างได้
        """
        start_time = time.time()
        all_entries = []
        num_batches = (total_count + batch_size - 1) // batch_size

        # ใช้ tqdm หากมีเพื่อแสดงความคืบหน้าแบบ progress bar
        try:
            from tqdm import tqdm

            batch_iterator = tqdm(
                range(num_batches), desc="Generating batches", unit="batch", ncols=100
            )
        except ImportError:
            batch_iterator = range(num_batches)
            print(
                f"{Fore.YELLOW}[INFO] เริ่มสร้างข้อมูล {num_batches} batches...{Style.RESET_ALL}"
            )

        for batch_idx in batch_iterator:
            # คำนวณจำนวนตัวอย่างสำหรับ batch นี้
            batch_count = min(batch_size, total_count - batch_idx * batch_size)

            if not isinstance(batch_iterator, range):
                batch_iterator.set_description(
                    f"Batch {batch_idx+1}/{num_batches} ({batch_count} samples)"
                )
            else:
                print(
                    f"{Fore.YELLOW}Batch {batch_idx+1}/{num_batches} ({batch_count} samples)...{Style.RESET_ALL}"
                )

            # ส่ง examples จาก batch ก่อนหน้าเพื่อเพิ่มคุณภาพ (เลือก examples ที่ดีที่สุด)
            examples = None
            if all_entries:
                # เลือกตัวอย่างที่หลากหลายจาก batches ก่อนหน้า
                if len(all_entries) > 3:
                    # สุ่มเลือก 3 ตัวอย่างที่มีความยาวแตกต่างกัน (สั้น กลาง ยาว)
                    text_field = next(
                        (key for key in all_entries[0].keys() if "text" in key.lower()),
                        None,
                    )
                    if text_field:
                        # จัดเรียงตามความยาวข้อความ
                        sorted_entries = sorted(
                            all_entries, key=lambda x: len(str(x.get(text_field, "")))
                        )
                        short_idx = 0
                        medium_idx = len(sorted_entries) // 2
                        long_idx = len(sorted_entries) - 1
                        examples = [
                            sorted_entries[short_idx],
                            sorted_entries[medium_idx],
                            sorted_entries[long_idx],
                        ]
                    else:
                        # ถ้าไม่มี text field ให้สุ่มเลือก
                        examples = random.sample(all_entries, min(3, len(all_entries)))
                else:
                    # ถ้ามีข้อมูลน้อยกว่า 3 ชิ้น ใช้ทั้งหมด
                    examples = all_entries.copy()

            # ปรับ temperature ตามจำนวน batch ที่เหลือ (ความหลากหลายมากขึ้นเมื่อใกล้สร้างเสร็จ)
            original_temp = self.temperature
            if num_batches > 3:
                # เพิ่ม temperature ในช่วงท้าย ๆ เพื่อเพิ่มความหลากหลาย
                progress = batch_idx / num_batches
                if progress > 0.7:
                    self.temperature = min(
                        original_temp * 1.2, 1.4
                    )  # เพิ่มสูงสุด 20% แต่ไม่เกิน 1.4

            # สร้างชุดข้อมูล
            entries = self.generate_dataset_with_prompt(
                task, batch_count, examples, advanced_prompt=advanced_prompt
            )

            # คืนค่า temperature กลับ
            self.temperature = original_temp

            if not entries:
                print(
                    f"[ERROR] No data generated in batch {batch_idx+1}. Retrying one more time..."
                )
                # ลองอีกครั้งหนึ่ง (อาจจะด้วย prompt แบบพื้นฐาน)
                entries = self.generate_dataset_with_prompt(
                    task, batch_count, examples, advanced_prompt=False
                )
                if not entries:
                    print(
                        f"[ERROR] Still no data generated in batch {batch_idx+1}. Skipping this batch."
                    )
                    continue

            # ทำความสะอาดข้อมูล (ถ้าเปิดใช้งาน)
            if clean_data and entries:
                from data_utils import clean_text

                cleaned_entries = []
                for entry in entries:
                    # ตรวจสอบว่า entry เป็น dict
                    if not isinstance(entry, dict):
                        try:
                            # พยายามแปลง string เป็น dict
                            entry = json.loads(entry)
                        except:
                            print(f"[WARN] Skipping invalid entry: {entry[:100]}...")
                            continue

                    cleaned_entry = {}
                    for key, value in entry.items():
                        if isinstance(value, str) and (
                            "text" in key.lower() or "content" in key.lower()
                        ):
                            # ลบช่องว่างที่ไม่จำเป็น
                            value = re.sub(r"\s+", " ", value).strip()
                            cleaned_entry[key] = clean_text(value, clean_options)
                        else:
                            cleaned_entry[key] = value
                    cleaned_entries.append(cleaned_entry)
                entries = (
                    cleaned_entries or entries
                )  # ถ้า cleaned_entries ว่างเปล่า ใช้ entries เดิม

            all_entries.extend(entries)

            # Progress bar สำหรับกรณีที่ไม่ได้ใช้ tqdm
            if isinstance(batch_iterator, range):
                bar_length = 30
                percent = int(((batch_idx + 1) / num_batches) * 100)
                bar = f"{Fore.YELLOW}|{'█'*((batch_idx+1)*bar_length//num_batches)}{'.'*(bar_length-((batch_idx+1)*bar_length//num_batches))}|{percent:3d}%{Style.RESET_ALL}"
                sys.stdout.write(f"\r{bar}")
                sys.stdout.flush()

            # หน่วงเวลาเพื่อป้องกัน rate limit (ยกเว้น batch สุดท้าย)
            if batch_idx < num_batches - 1:
                # ปรับ delay แบบ dynamic ตามขนาด batch
                adjusted_delay = delay
                if batch_size > 20:
                    adjusted_delay = max(delay, 3)  # เพิ่ม delay สำหรับ batch ใหญ่
                time.sleep(adjusted_delay)

        # จบ progress bar
        if isinstance(batch_iterator, range):
            print()  # Newline after progress bar

        # ทำ post-processing ถ้าเปิดใช้งาน
        if post_process and all_entries:
            # Validate และ deduplicate
            from data_utils import deduplicate_entries

            print(f"[INFO] Running post-processing on {len(all_entries)} entries...")

            # นับจำนวน entry ก่อน post-processing
            pre_count = len(all_entries)

            # Deduplicate
            all_entries = deduplicate_entries(all_entries)
            dedup_count = len(all_entries)

            if dedup_count < pre_count:
                print(f"[INFO] Removed {pre_count - dedup_count} duplicate entries")

            # ถ้าได้ข้อมูลน้อยกว่าที่ต้องการ ลองสร้างเพิ่มอีก 1 batch
            if len(all_entries) < total_count * 0.9:  # ถ้าได้น้อยกว่า 90% ของที่ต้องการ
                missing_count = total_count - len(all_entries)
                print(
                    f"[INFO] Got only {len(all_entries)} entries. Generating additional {missing_count} entries..."
                )

                # สร้างเพิ่มอีก 1 batch
                extra_entries = self.generate_dataset_with_prompt(
                    task, missing_count, all_entries[:3], True
                )

                # ทำความสะอาดข้อมูลเพิ่มเติม
                if clean_data and extra_entries:
                    from data_utils import clean_text

                    for entry in extra_entries:
                        for key, value in entry.items():
                            if isinstance(value, str) and (
                                "text" in key.lower() or "content" in key.lower()
                            ):
                                entry[key] = clean_text(value, clean_options)

                # Deduplicate ระหว่างข้อมูลเก่าและข้อมูลใหม่
                combined = all_entries + extra_entries
                all_entries = deduplicate_entries(combined)
                print(
                    f"[INFO] Added {len(all_entries) - dedup_count} more unique entries"
                )

        # สรุปเวลาที่ใช้ทั้งหมด
        elapsed_time = time.time() - start_time
        minutes, seconds = divmod(elapsed_time, 60)
        print(
            f"[INFO] Dataset generation completed in {int(minutes)}m {int(seconds)}s. Generated {len(all_entries)} entries."
        )

        return all_entries

    # เพิ่มฟังก์ชัน validation และ quality check
def validate_data_quality(entries: List[Dict], task: Dict) -> Tuple[List[Dict], List[str]]:
    """Validate data quality and return filtered entries with issues"""
    valid_entries = []
    issues = []
    
    required_fields = task.get('schema', {}).get('fields', {})
    
    for i, entry in enumerate(entries):
        entry_issues = []
        
        # Check required fields
        for field_name, field_config in required_fields.items():
            if field_config.get('required', False) and field_name not in entry:
                entry_issues.append(f"Missing required field: {field_name}")
        
        # Check data constraints
        for field_name, value in entry.items():
            if field_name in required_fields:
                field_config = required_fields[field_name]
                constraints = field_config.get('constraints', [])
                
                for constraint in constraints:
                    if 'Length' in constraint:
                        length_constraint = constraint['Length']
                        if isinstance(value, str):
                            if len(value) < length_constraint.get('min', 0):
                                entry_issues.append(f"Field {field_name} too short")
                            if len(value) > length_constraint.get('max', float('inf')):
                                entry_issues.append(f"Field {field_name} too long")
        
        if not entry_issues:
            valid_entries.append(entry)
        else:
            issues.extend([f"Entry {i+1}: {issue}" for issue in entry_issues])
    
    return valid_entries, issues

def generate_dataset_with_quality_control(task: Dict, count: int, **kwargs) -> Tuple[List[Dict], Dict]:
    """Generate dataset with comprehensive quality control"""
    client = kwargs.get('client')
    if not client:
        raise ValueError("Client is required")
    
    # Generate with buffer (extra items to account for quality filtering)
    buffer_count = int(count * 1.2)  # 20% buffer
    
    print(f"[INFO] Generating {buffer_count} entries (with buffer) to ensure {count} quality entries")
    
    # Generate entries in batches for better error recovery
    all_entries = []
    batch_size = kwargs.get('batch_size', 10)
    
    for batch_start in range(0, buffer_count, batch_size):
        batch_count = min(batch_size, buffer_count - batch_start)
        
        try:
            batch_entries = client.generate_dataset_batch(
                task=task,
                total_count=batch_count,
                batch_size=batch_count,
                **kwargs
            )
            
            if batch_entries:
                all_entries.extend(batch_entries)
                print(f"[INFO] Generated batch: {len(batch_entries)} entries (Total: {len(all_entries)})")
            
        except Exception as e:
            print(f"[WARN] Batch generation failed: {e}")
            continue
    
    # Validate quality
    valid_entries, issues = validate_data_quality(all_entries, task)
    
    # Report quality metrics
    quality_report = {
        "total_generated": len(all_entries),
        "valid_entries": len(valid_entries),
        "quality_rate": len(valid_entries) / len(all_entries) if all_entries else 0,
        "issues_found": len(issues),
        "final_count": min(len(valid_entries), count)
    }
    
    if issues:
        print(f"[WARN] Found {len(issues)} data quality issues:")
        for issue in issues[:5]:  # Show first 5 issues
            print(f"  - {issue}")
        if len(issues) > 5:
            print(f"  ... and {len(issues) - 5} more issues")
    
    return valid_entries[:count], quality_report

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


def create_dataset_standard_structure(
    output_path: str,
    task: dict,
    format: str = "jsonl",
    split: bool = False,
    split_paths: tuple = None,
    sample_count: int = None,
):
    """
    สร้างโครงสร้าง dataset มาตรฐานสำหรับ Hugging Face หรือเผยแพร่ จากไฟล์ที่สร้าง

    Args:
        output_path: path ไปยังไฟล์ output หลัก (เช่น auto-dataset-task-date.jsonl)
        task: dictionary ของ task definition
        format: รูปแบบไฟล์ ("json", "jsonl", "csv", "parquet")
        split: เป็น True ถ้ามีการคัดลอกไฟล์ไปเป็น train/valid/test (ไม่ได้แบ่งข้อมูลจริง)
        split_paths: tuple ของ (train_path, valid_path, test_path) ถ้า split=True
        sample_count: จำนวนตัวอย่างทั้งหมดใน dataset (ถ้าระบุ)

    Returns:
        path ไปยังโฟลเดอร์ที่สร้าง
    """
    # สร้างชื่อโฟลเดอร์
    task_name = os.path.basename(output_path).split("-")[
        2
    ]  # เช่น "auto-dataset-medical_benchmark-..."
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
        shutil.copy(
            valid_path, os.path.join(dataset_folder_path, f"validation.{format}")
        )
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
    }  # เพิ่ม splits ถ้ามี
    if split:
        dataset_info["splits"] = {
            "train": {"num_examples": train_examples},
            "validation": {"num_examples": valid_examples},
            "test": {"num_examples": test_examples},
        }
        # เพิ่มข้อมูลเกี่ยวกับสัดส่วนการแบ่ง
        dataset_info["split_info"] = {
            "train_ratio": train_examples / total_examples,
            "valid_ratio": valid_examples / total_examples,
            "test_ratio": test_examples / total_examples,
        }

    # เขียนไฟล์ dataset_info.json
    with open(
        os.path.join(dataset_folder_path, "dataset_info.json"), "w", encoding="utf-8"
    ) as f:
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
            readme_content += (
                f'    "{field_name}": "{field_desc} (หนึ่งใน {enum_values})",\n'
            )
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
        readme_content += (
            "\n**หมายเหตุ:** แต่ละไฟล์มีข้อมูลตามสัดส่วนที่กำหนด ขอบคุณที่ใช้ DekDataset\n"
        )
    else:
        readme_content += f"- Total: {sample_count or 'N/A'} ตัวอย่าง\n"

    # เพิ่มตัวอย่างการใช้งาน
    readme_content += (
        """
## การใช้งานกับ Hugging Face Datasets

```python
from datasets import load_dataset

# โหลดจาก local
dataset = load_dataset("path/to/"""
        + dataset_folder_name
        + """")

# หรือโหลดจาก Hugging Face Hub หลังจาก push (ถ้ามี)
# dataset = load_dataset("username/"""
        + dataset_folder_name
        + """")

# ดูข้อมูล
print(dataset["train"][0])  # ดูตัวอย่างแรก
```
"""
    )

    # เขียน README.md
    with open(
        os.path.join(dataset_folder_path, "README.md"), "w", encoding="utf-8"
    ) as f:
        f.write(readme_content)

    print(f"[INFO] Created standard dataset structure at {dataset_folder_path}")
    print(f"[INFO] - README.md: สรุปข้อมูล dataset")
    print(f"[INFO] - dataset_info.json: metadata สำหรับ Hugging Face")
    print(
        f"[INFO] - {'train.{format}, validation.{format}, test.{format}' if split else f'data.{format}'}: ข้อมูลหลัก"
    )
    if os.path.exists(os.path.join(dataset_folder_path, "LICENSE")):
        print(f"[INFO] - LICENSE: ข้อมูลสิทธิ์การใช้งาน")

    return dataset_folder_path


def main():
    print("=== DekDataset OCR Extraction Tool ===")
    print("Starting main function...")
    
    print_ascii_banner()
    parser = argparse.ArgumentParser(
        description="DekDataset: สร้าง dataset คุณภาพสูงสำหรับงาน NLP และ AI"
    )
    parser.add_argument("task", nargs="?", help="Task name (e.g. sentiment_analysis)")
    parser.add_argument("count", nargs="?", type=int, help="Number of samples")
    parser.add_argument(
        "--format",
        choices=["json", "jsonl", "csv", "parquet"],
        default="jsonl",
        help="Output format: json, jsonl, csv, parquet",
    )
    parser.add_argument(
        "--import-vision",
        type=str,
        default=None,
        help="Path to vision-animals-dataset-*.jsonl to import/validate/export",
    )
    parser.add_argument(
        "--source_lang",
        type=str,
        default=None,
        help="Source language (for translation task)",
    )
    parser.add_argument(
        "--target_lang",
        type=str,
        default=None,
        help="Target language (for translation task)",
    )
    parser.add_argument(
        "--lang",
        type=str,
        default=None,
        help="Language code (for multilingual tasks, e.g. th, en, zh, hi)",
    )

    # เพิ่ม argument สำหรับการสร้างโครงสร้างตามมาตรฐาน
    parser.add_argument(
        "--create-standard",
        action="store_true",
        default=True,
        help="Create standard dataset structure (default: True)",
    )
    parser.add_argument(
        "--no-standard",
        dest="create_standard",
        action="store_false",
        help="Don't create standard dataset structure",
    )
    parser.add_argument(
        "--split",
        action="store_true",
        help="Create train/valid/test copies according to specified ratios",
    )
    parser.add_argument(
        "--train-ratio",
        type=float,
        default=0.8,
        help="Train set ratio when splitting (default: 0.8)",
    )
    parser.add_argument(
        "--valid-ratio",
        type=float,
        default=0.1,
        help="Validation set ratio when splitting (default: 0.1)",
    )
    parser.add_argument(
        "--test-ratio",
        type=float,
        default=0.1,
        help="Test set ratio when splitting (default: 0.1)",
    )

    # เพิ่ม argument สำหรับการทำความสะอาดข้อมูลและการ Normalize ข้อความ
    parser.add_argument(
        "--no-clean",
        action="store_true",
        help="Disable data cleaning (default: cleaning enabled)",
    )
    parser.add_argument(
        "--disable-thai-norm",
        action="store_true",
        help="Disable Thai text normalization (default: enabled)",
    )
    parser.add_argument(
        "--remove-emojis",
        action="store_true",
        help="Remove emojis from text (default: keep emojis)",
    )
    parser.add_argument(
        "--remove-special-chars",
        action="store_true",
        help="Remove special characters (default: keep special chars)",
    )

    # เพิ่ม argument สำหรับการวิเคราะห์ dataset
    parser.add_argument(
        "--analyze",
        action="store_true",
        help="Analyze dataset after generation (default: False)",
    )
    parser.add_argument(
        "--visualize",
        action="store_true",
        help="Create visualizations for dataset (default: False)",
    )
    parser.add_argument(
        "--analyze-output",
        type=str,
        default=None,
        help="Path to save analysis results (default: same as output dir)",
    )

    # เพิ่ม argument สำหรับ DeepseekClient ที่ปรับปรุงแล้ว
    parser.add_argument(
        "--model",
        type=str,
        default="deepseek-chat",
        help="Deepseek model to use (default: deepseek-chat)",
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=1.2,
        help="Temperature for generation (default: 1.2)",
    )
    parser.add_argument(
        "--batch-size", type=int, default=None, help="Override automatic batch size"
    )
    parser.add_argument(
        "--delay",
        type=int,
        default=2,
        help="Delay between API calls in seconds (default: 2)",
    )
    # เพิ่ม argument สำหรับ Hugging Face Hub
    parser.add_argument(
        "--export-huggingface",
        action="store_true",
        help="Export dataset to Hugging Face Hub",
    )
    parser.add_argument(
        "--hf-repo-id",
        type=str,
        default=None,
        help="Hugging Face repository ID (username/repo-name)",
    )
    parser.add_argument(
        "--hf-token",
        type=str,
        default=None,
        help="Hugging Face API token (or set HUGGINGFACE_TOKEN env)",
    )
    parser.add_argument(
        "--hf-private",
        action="store_true",
        help="Create private repository (default: public)",
    )

    # เพิ่ม argument สำหรับการเพิ่มข้อมูล metadata
    parser.add_argument(
        "--license",
        type=str,
        default="CC-BY 4.0",
        help="License for the dataset (default: CC-BY 4.0)",
    )
    parser.add_argument(
        "--version",
        type=str,
        default="1.0.0",
        help="Version of the dataset (default: 1.0.0)",
    )
    parser.add_argument(
        "--domain",
        type=str,
        default=None,
        help="Domain of the dataset (e.g. social, news, medical)",
    )    # เพิ่ม argument สำหรับการอ่านไฟล์ PDF/ภาพด้วย Mistral OCR API
    parser.add_argument(
        "--input-file",
        type=str,
        default=None,
        help="Path to PDF or image file for OCR-based synthetic data generation",
    )
    parser.add_argument(
        "--max-pages",
        type=int,
        default=None,
        help="Limit number of pages to process for PDF files (default: all pages)",
    )
    parser.add_argument(
        "--ocr-workers",
        type=int,
        default=3,
        help="Number of concurrent workers for OCR processing (default: 3)",
    )

    # Add social media extraction arguments
    social_media_group = parser.add_argument_group('Social Media Extraction')
    social_media_group.add_argument(
        "--social-platform",
        type=str,
        choices=["twitter", "x", "reddit", "youtube", "pantip", "file"],
        help="Social media platform to extract comments from"
    )
    social_media_group.add_argument(
        "--social-query",
        type=str,
        help="Query for social media extraction (topic ID, video ID, subreddit, hashtag, etc.)"
    )
    social_media_group.add_argument(
        "--social-max",
        type=int,
        default=100,
        help="Maximum number of comments to extract"
    )
    social_media_group.add_argument(
        "--social-include-sentiment",
        action="store_true",
        help="Include basic sentiment analysis for extracted comments"
    )
    social_media_group.add_argument(
        "--social-lang",
        type=str,
        default="th",
        help="Language filter for platforms that support it (e.g. 'th', 'en')"
    )
    social_media_group.add_argument(
        "--social-time-filter",
        type=str,
        default="week",
        choices=["hour", "day", "week", "month", "year", "all"],
        help="Time filter for Reddit extraction"
    )
    social_media_group.add_argument(
        "--social-text-column",
        type=str,
        default="text",
        help="Column name containing text for file import"
    )
    social_media_group.add_argument(
        "--social-platform-column",
        type=str,
        default="platform",
        help="Column name containing platform for file import"
    )
    args = parser.parse_args()

    # --- Social Media Extraction Mode ---
    if args.social_platform:
        print(f"[INFO] Social media extraction mode activated for platform: {args.social_platform}")
        
        try:
            comments = extract_social_media_comments(
                platform=args.social_platform,
                query=args.social_query,
                max_results=args.social_max,
                include_sentiment=args.social_include_sentiment,
                filter_spam=True,
                silent=False,
                lang=args.social_lang,
                time_filter=args.social_time_filter
            )
            
            if not comments:
                print("[ERROR] No comments extracted. Please check your query and platform settings.")
                return
            
            print(f"[SUCCESS] Extracted {len(comments)} comments from {args.social_platform}")
            
            # Save extracted comments
            now = datetime.now().strftime("%Y%m%d-%H%M%S")
            output_dir = "data/output"
            os.makedirs(output_dir, exist_ok=True)
            filename = f"social-{args.social_platform}-{now}.{args.format}"
            output_path = os.path.join(output_dir, filename)
            
            # Convert to dataset format
            data_entries = []
            for i, comment in enumerate(comments):
                data_entry = DataEntry(
                    id=f"social-{args.social_platform}-{i+1}",
                    content=comment,
                    metadata={
                        "source": f"social_media_{args.social_platform}",
                        "extracted_at": datetime.now().isoformat(),
                        "query": args.social_query,
                        "lang": args.social_lang
                    }
                ).to_dict()
                data_entries.append(data_entry)
            
            # Export using existing functions
            if args.format == "jsonl":
                export_jsonl(data_entries, output_path)
            elif args.format == "json":
                export_json(data_entries, output_path)
            elif args.format == "csv":
                export_csv(data_entries, output_path)
            elif args.format == "parquet":
                export_parquet(data_entries, output_path)
            
            print(f"[SUCCESS] Social media data saved to {output_path}")
            
            # Create standard structure if requested
            if args.create_standard:
                # Create a mock task for social media data
                social_task = {
                    "name": f"Social Media Comments - {args.social_platform.title()}",
                    "description": f"Comments extracted from {args.social_platform}",
                    "schema": {
                        "fields": {
                            "text": {"field_type": "Text", "required": True},
                            "platform": {"field_type": "Text", "required": True},
                            "author": {"field_type": "Text", "required": False},
                            "post_type": {"field_type": "Text", "required": False}
                        }
                    }
                }
                
                standard_structure = create_dataset_standard_structure(
                    output_path, social_task, args.format, args.split, None, len(data_entries)
                )
                print(f"[SUCCESS] Created standard structure at: {standard_structure}")
            
            return
            
        except Exception as e:
            print(f"[ERROR] Social media extraction failed: {e}")
            return

    # --- OCR extraction mode: extract text and optionally generate dataset ---
    if args.input_file:
        print(f"[DEBUG] OCR mode activated for file: {args.input_file}")
        mistral_api_key = os.getenv("MISTRAL_API_KEY")
        print(f"[DEBUG] MISTRAL_API_KEY found: {bool(mistral_api_key)}")
        if not mistral_api_key:
            print(
                "[ERROR] MISTRAL_API_KEY must be set in your .env file for OCR extraction."
            )
            print("[INFO] Please create a .env file in the project root with: MISTRAL_API_KEY=your_api_key_here")
            print("[INFO] You can get a Mistral API key from: https://console.mistral.ai/")
            return        print(
            f"[INFO] Extracting text from file: {args.input_file} (using Mistral OCR API)..."
        )
        if args.max_pages:
            print(f"[INFO] Limited to first {args.max_pages} pages")
        try:
            ocr_text = extract_text_from_file(
                args.input_file, 
                mistral_api_key, 
                max_pages=args.max_pages,
                max_workers=args.ocr_workers
            )
            if ocr_text:
                print("\n[OCR RESULT]\n" + "-" * 40)
                print(ocr_text)
                print("-" * 40 + "\n[END OF OCR RESULT]")
                
                # If task and count are provided, use OCR text as context for dataset generation
                if args.task and args.count:
                    print(f"\n[INFO] Using OCR text as context for {args.task} dataset generation...")
                    
                    # Load task definition
                    tasks = get_task_definitions()
                    if args.task not in tasks:
                        print(f"[ERROR] Task '{args.task}' not found in available tasks")
                        print(f"[INFO] Available tasks: {list(tasks.keys())}")
                        return
                    
                    task = tasks[args.task]
                    print(f"[INFO] Loaded task: {task.get('name', args.task)} for OCR-based generation")
                    
                    # Add OCR text as context to the task
                    if "context" not in task:
                        task["context"] = {}
                    task["context"]["ocr_text"] = ocr_text
                    task["context"]["source_file"] = args.input_file
                    
                    print(f"[INFO] OCR context added to task. Proceeding with dataset generation...")
                    # Don't return here - continue to dataset generation below
                else:
                    print("[INFO] OCR extraction complete. Use --task and count to generate dataset with this context.")
                    return
            else:
                print("[WARN] No text extracted from the file.")
                return
        except Exception as e:
            print(f"[ERROR] OCR extraction failed: {e}")
            return    # Now require task and count for normal dataset generation (unless OCR already handled it)
    if not args.input_file:  # Only check if not in OCR mode
        if not args.task or not args.count:
            parser.error(
                "the following arguments are required: task, count (unless --input-file is used)"
            )

        # --- Load task definition ---
        print(f"[INFO] Loading task definition for: {args.task}")
        tasks = get_task_definitions()
        if args.task not in tasks:
            print(f"[ERROR] Task '{args.task}' not found in available tasks")
            print(f"[INFO] Available tasks: {list(tasks.keys())}")
            return
        
        task = tasks[args.task]
        print(f"[INFO] Loaded task: {task.get('name', args.task)}")

        # --- Inject translation language params if needed ---
        if args.task == "translation":
            if args.source_lang:
                task["parameters"]["source_lang"]["default"] = args.source_lang
            if args.target_lang:
                task["parameters"]["target_lang"]["default"] = args.target_lang

    # --- Import Vision Dataset Mode ---
    if args.import_vision:
        output_path = os.path.join(
            "data/output",
            f"imported-vision-{datetime.now().strftime('%Y%m%d-%H%M%S')}.jsonl",
        )
        import_vision_jsonl(args.import_vision, task["schema"], output_path)
        return

    # --- กำหนดตัวเลือกสำหรับการทำความสะอาดข้อมูล ---
    clean_options = {
        "remove_html": not args.no_clean,
        "remove_urls": not args.no_clean,
        "remove_emojis": args.remove_emojis,
        "remove_special_chars": args.remove_special_chars,
        "normalize_thai": not args.disable_thai_norm,
        "fix_spacing": not args.no_clean,
    }

    total = args.count  # --- Batch Generation ---
    # ปรับ batch size อัตโนมัติ: ถ้า count <= 10 ให้ batch = count, ถ้า count <= 100 ให้ batch = 10, ถ้ามากกว่านั้น batch = 5
    if args.batch_size:
        BATCH_SIZE = args.batch_size
    elif total <= 10:
        BATCH_SIZE = total
    elif total <= 100:
        BATCH_SIZE = 10
    else:
        BATCH_SIZE = 5

    # --- ใช้ฟังก์ชัน batch generation แบบใหม่ ---
    all_entries = client.generate_dataset_batch(
        task=task,
        total_count=total,
        batch_size=BATCH_SIZE,
        max_concurrent=1,  # ยังไม่รองรับ concurrent requests มากกว่า 1
        delay=args.delay,
        clean_data=not args.no_clean,
        clean_options=clean_options,
        advanced_prompt=True,
        post_process=True
    )

    if not all_entries:
        print(
            f"[ERROR] No data generated. Deepseek output was empty or invalid. File will not be saved."
        )
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
        print(
            f"[INFO] Filtered out {len(all_entries)-len(valid_entries)} invalid entries (schema mismatch)"
        )
    # Post-processing: deduplicate
    deduped_entries = deduplicate_entries(valid_entries)
    if len(deduped_entries) < len(valid_entries):
        print(
            f"[INFO] Removed {len(valid_entries)-len(deduped_entries)} duplicate entries"
        )
    # Post-processing: enrich with metadata
    metadata = {
        "source": "DeepSeek-V3",
        "license": args.license,
        "version": args.version,
        "created_at": datetime.now().isoformat(),
        "lang": args.lang or "th",
    }
    # Add domain if specified
    if args.domain:
        metadata["domain"] = args.domain

    enriched_entries = enrich_entries(deduped_entries)

    # Diversity/Balance (example: sentiment label)
    if args.task == "sentiment_analysis":
        labels = ["positive", "negative", "neutral"]
        n_per_label = min(len(enriched_entries) // len(labels), 100)
        balanced_entries = balance_label_entries(
            enriched_entries, "sentiment", labels, n_per_label
        )
        print(
            f"[INFO] Balanced sentiment labels: {len(balanced_entries)} entries ({n_per_label} per label)"
        )
    else:
        balanced_entries = enriched_entries
    data_entries = []
    for i, entry in enumerate(balanced_entries):
        # Remove metadata from content if it exists
        entry_content = entry.copy()
        if "metadata" in entry_content:
            del entry_content["metadata"]

        data_entry = DataEntry(
            id=f"{args.task}-{i+1}", content=entry_content, metadata=metadata
        ).to_dict()
        data_entries.append(data_entry)

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
                    row["metadata"] = json.dumps(
                        entry.get("metadata", {}), ensure_ascii=False
                    )
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
        print(
            f"{Fore.YELLOW}[INFO] สร้างไฟล์สำเนาสำหรับ train/valid/test ตามสัดส่วน {args.train_ratio:.1f}/{args.valid_ratio:.1f}/{args.test_ratio:.1f}{Style.RESET_ALL}"
        )
        # แยกชื่อไฟล์และนามสกุล
        filename_prefix = os.path.splitext(os.path.basename(output_path))[0]
        split_paths = split_dataset(
            data_entries,
            output_dir,
            filename_prefix,
            args.train_ratio,
            args.valid_ratio,
            args.test_ratio,
            args.format,
        )

    # ---- สร้างโครงสร้าง dataset มาตรฐานโดยอัตโนมัติ ----
    standard_structure = None
    if args.create_standard:
        standard_structure = create_dataset_standard_structure(
            output_path, task, args.format, args.split, split_paths, args.count
        )
        print(
            f"{Fore.GREEN}[SUCCESS] สร้างโครงสร้าง dataset มาตรฐานเรียบร้อยที่: {standard_structure}{Style.RESET_ALL}"
        )
        print(
            f"{Fore.YELLOW}พร้อมสำหรับการอัพโหลดเข้า Hugging Face Datasets Hub{Style.RESET_ALL}"
        )

    # ---- วิเคราะห์ dataset ถ้าเปิดใช้งานตัวเลือก ----
    if args.analyze or args.visualize:
        print(f"{Fore.CYAN}[INFO] กำลังวิเคราะห์ข้อมูล Dataset...{Style.RESET_ALL}")
        # กำหนด output path สำหรับการวิเคราะห์
        analysis_dir = args.analyze_output
        if not analysis_dir:
            if standard_structure:
                analysis_dir = os.path.join(standard_structure, "analysis")
            else:
                analysis_dir = os.path.join(output_dir, f"analysis-{args.task}-{now}")

        # สร้างโฟลเดอร์
        os.makedirs(analysis_dir, exist_ok=True)

        # ทำการวิเคราะห์
        analysis_results = analyze_dataset(data_entries, field_path="content.text")

        # บันทึกผลการวิเคราะห์เป็น JSON
        with open(
            os.path.join(analysis_dir, "analysis_results.json"), "w", encoding="utf-8"
        ) as f:
            json.dump(analysis_results, f, ensure_ascii=False, indent=2)

        print(
            f"{Fore.GREEN}[SUCCESS] บันทึกผลการวิเคราะห์ข้อมูลไปยัง {os.path.join(analysis_dir, 'analysis_results.json')}{Style.RESET_ALL}"
        )

        # สร้างภาพแสดงผลถ้าเปิดใช้งาน
        if args.visualize:
            print(f"{Fore.CYAN}[INFO] กำลังสร้างภาพแสดงผลการวิเคราะห์...{Style.RESET_ALL}")
            # สร้างและบันทึกการแสดงผลต่างๆ
            base_viz_path = os.path.join(analysis_dir, "visualization")

            # Word Cloud
            word_cloud_path = f"{base_viz_path}_wordcloud.png"
            plot_word_cloud(
                " ".join([entry["content"]["text"] for entry in data_entries]),
                output_path=word_cloud_path,
            )

            # Category Distribution (ถ้ามี field category)
            if any("category" in entry["content"] for entry in data_entries):
                category_path = f"{base_viz_path}_categories.png"
                categories = {entry["content"]["category"]: 1 for entry in data_entries}
                plot_category_distribution(categories, output_path=category_path)

            # Length Distribution
            length_path = f"{base_viz_path}_lengths.png"
            lengths = [len(entry["content"]["text"]) for entry in data_entries]
            plot_length_distribution(lengths, output_path=length_path)

            # Word Frequency
            freq_path = f"{base_viz_path}_word_freq.png"
            all_text = " ".join([entry["content"]["text"] for entry in data_entries])
            words = all_text.split()
            from collections import Counter

            word_counts = Counter(words).most_common(20)
            plot_word_frequency(word_counts, output_path=freq_path)

            print(
                f"{Fore.GREEN}[SUCCESS] บันทึกภาพแสดงผลไปยังโฟลเดอร์ {analysis_dir}{Style.RESET_ALL}"
            )

    # ---- อัปโหลดไปยัง Hugging Face Hub ถ้าเปิดใช้งานตัวเลือก ----
    if args.export_huggingface:
        print(
            f"{Fore.CYAN}[INFO] กำลังอัปโหลด Dataset ไปยัง Hugging Face Hub...{Style.RESET_ALL}"
        )

        # ตรวจสอบ repo_id
        repo_id = args.hf_repo_id
        if not repo_id:
            repo_id = f"dekdataset/{args.task}-{now}"
            print(
                f"{Fore.YELLOW}[INFO] ไม่ได้ระบุ repo_id จะใช้ค่า default: {repo_id}{Style.RESET_ALL}"
            )

        # ตรวจสอบ token
        hf_token = args.hf_token or os.getenv("HUGGINGFACE_TOKEN")
        if not hf_token:
            print(
                f"{Fore.RED}[ERROR] ไม่พบ Hugging Face API token กรุณาระบุด้วย --hf-token หรือตั้งค่า HUGGINGFACE_TOKEN ใน environment{Style.RESET_ALL}"
            )
            print(
                f"{Fore.RED}[ERROR] การอัปโหลดไปยัง Hugging Face Hub ถูกข้าม{Style.RESET_ALL}"
            )
        else:
            # กำหนด path ที่จะอัปโหลด (ใช้ standard structure ถ้ามี)
            upload_path = standard_structure if standard_structure else output_dir

            # สร้าง README สำหรับ Hugging Face Hub (ถ้ายังไม่มี)
            if standard_structure and not os.path.exists(
                os.path.join(standard_structure, "README.md")
            ):
                # สร้าง README.md ง่ายๆ
                readme_content = f"# {args.task} Dataset\n\n{task['description']}\n\nCreated with DekDataset"
                with open(
                    os.path.join(standard_structure, "README.md"), "w", encoding="utf-8"
                ) as f:
                    f.write(readme_content)

            # อัปโหลด
            try:
                hf_url = upload_to_huggingface(
                    dataset_path=upload_path,
                    repo_id=repo_id,
                    token=hf_token,
                    private=args.hf_private,
                    metadata={
                        "language": args.lang or "th",
                        "license": args.license,
                        "task": args.task,
                        "source": "DekDataset/DeepSeek",
                    },
                )
                print(
                    f"{Fore.GREEN}[SUCCESS] อัปโหลด Dataset ไปยัง Hugging Face Hub สำเร็จ: {hf_url}{Style.RESET_ALL}"
                )
            except Exception as e:
                print(
                    f"{Fore.RED}[ERROR] เกิดข้อผิดพลาดในการอัปโหลดไปยัง Hugging Face Hub: {e}{Style.RESET_ALL}"
                )


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
        # Copy content dictionary and enrich it
        if "content" in entry and isinstance(entry["content"], dict):
            content = dict(entry["content"])
            text = content.get("text")
            # Only add word_count if not present and text is str
            if "word_count" not in content and isinstance(text, str):
                content["word_count"] = len(text.split())
            # Remove metadata from content if it exists
            if "metadata" in content:
                del content["metadata"]
            entry["content"] = content
        # Keep existing metadata or set empty dict
        if "metadata" not in entry:
            entry["metadata"] = {}
        enriched.append(entry)
    return enriched


def balance_label_entries(
    entries: list, label_field: str, labels: list, n_per_label: int
) -> list:
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


def split_dataset(
    data_entries: list,
    output_dir: str,
    filename_prefix: str,
    train_ratio: float,
    valid_ratio: float,
    test_ratio: float,
    format: str
):
    """Split dataset into train/valid/test files according to specified ratios"""
    import shutil
    
    # Calculate number of samples for each split
    total_samples = len(data_entries)
    train_samples = int(total_samples * train_ratio)
    valid_samples = int(total_samples * valid_ratio)
    test_samples = total_samples - train_samples - valid_samples
    
    # Create file paths
    train_path = os.path.join(output_dir, f"{filename_prefix}-train.{format}")
    valid_path = os.path.join(output_dir, f"{filename_prefix}-valid.{format}")
    test_path = os.path.join(output_dir, f"{filename_prefix}-test.{format}")
    
    # For now, copy the full dataset to each file (as indicated by the original comment)
    # In a real implementation, you might want to actually split the data
    original_path = os.path.join(output_dir, f"{filename_prefix}.{format}")
    
    if os.path.exists(original_path):
        shutil.copy(original_path, train_path)
        shutil.copy(original_path, valid_path) 
        shutil.copy(original_path, test_path)
    
    print(f"       - Train: {train_samples} entries ({train_ratio*100:.1f}%) -> {train_path}")
    print(f"       - Valid: {valid_samples} entries ({valid_ratio*100:.1f}%) -> {valid_path}")
    print(f"       - Test: {test_samples} entries ({test_ratio*100:.1f}%) -> {test_path}")

    print(
        f"       [NOTE] All files contain the full dataset. When using, sample according to the ratios."
    )

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
                    row = {
                        k: v for k, v in entry.items() if k not in ["id", "metadata"]
                    }

                row["id"] = entry.get("id", "")
                row["metadata"] = json.dumps(
                    entry.get("metadata", {}), ensure_ascii=False
                )
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
