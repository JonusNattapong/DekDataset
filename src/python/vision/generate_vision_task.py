import os
import requests
import json
import shutil
from PIL import Image
from tqdm import tqdm
from datetime import datetime
from transformers import BlipProcessor, BlipForConditionalGeneration
from dotenv import load_dotenv
import torch
import time
import sys
from task_definitions import get_task_definitions
from huggingface_hub import login
import random

load_dotenv()

# --- ใช้ HUGGING_FACE_TOKEN ---
HUGGING_FACE_TOKEN = os.getenv("HUGGING_FACE_TOKEN")
if HUGGING_FACE_TOKEN:
    login(token=HUGGING_FACE_TOKEN)

# --- CONFIG ---
PEXELS_API_KEY = os.getenv("PEXELS_API_KEY")
PIXABAY_API_KEY = os.getenv("PIXABAY_API_KEY")
BLIP_MODEL = "Salesforce/blip-image-captioning-base"
PER_CLASS = None  # จะกำหนดอัตโนมัติใน main

# --- Download Functions ---
def download_pexels(animal_en, output_dir, count=PER_CLASS, prefix=""):
    headers = {"Authorization": PEXELS_API_KEY}
    url = f"https://api.pexels.com/v1/search?query={animal_en}&per_page={count}"
    resp = requests.get(url, headers=headers)
    data = resp.json()
    os.makedirs(output_dir, exist_ok=True)
    paths = []
    photos = data.get("photos", [])
    for i, photo in enumerate(tqdm(photos, desc=f"[Pexels] {animal_en}", position=0, leave=False), 1):
        img_url = photo["src"]["original"]
        fname = f"{prefix}_{animal_en}_pexels_{i}.jpg"
        fpath = os.path.join(output_dir, fname)
        img_data = requests.get(img_url).content
        with open(fpath, "wb") as f:
            f.write(img_data)
        tqdm.write(f"[Pexels] {animal_en} | Q{i}/{len(photos)}: {fname}")
        paths.append(fpath)
    return paths

def download_pixabay(animal_en, output_dir, count=PER_CLASS, prefix=""):
    url = f"https://pixabay.com/api/?key={PIXABAY_API_KEY}&q={animal_en}&image_type=photo&per_page={count}"
    resp = requests.get(url)
    try:
        data = resp.json()
    except Exception as e:
        print(f"[ERROR] Pixabay API response for '{animal_en}': {resp.status_code} {resp.text}")
        return []
    if resp.status_code != 200 or "hits" not in data:
        print(f"[ERROR] Pixabay API error for '{animal_en}': {data}")
        return []
    os.makedirs(output_dir, exist_ok=True)
    paths = []
    hits = data.get("hits", [])
    for i, hit in enumerate(tqdm(hits, desc=f"[Pixabay] {animal_en}", position=1, leave=False), 1):
        img_url = hit.get("largeImageURL")
        if not img_url:
            continue
        fname = f"{prefix}_{animal_en}_pixabay_{i}.jpg"
        fpath = os.path.join(output_dir, fname)
        try:
            img_data = requests.get(img_url).content
            with open(fpath, "wb") as f:
                f.write(img_data)
            tqdm.write(f"[Pixabay] {animal_en} | Q{i}/{len(hits)}: {fname}")
            paths.append(fpath)
            time.sleep(1)
        except Exception as e:
            print(f"[ERROR] Downloading image {img_url}: {e}")
    return paths

def download_images_balanced(animal_en, output_dir, total_count):
    """พยายามดึงภาพจากทั้งสอง API ให้ครบ total_count ถ้า API ไหน error หรือได้ไม่ครบจะไปดึงอีก API ต่อ"""
    half = max(1, total_count // 2)
    pexels_count = total_count - half
    pixabay_count = half
    # ลองดึงจาก Pexels ก่อน
    pexels_paths = download_pexels(animal_en, output_dir, count=pexels_count, prefix="pexels")
    if len(pexels_paths) < pexels_count:
        # ดึงที่เหลือจาก Pixabay
        pixabay_count += pexels_count - len(pexels_paths)
    pixabay_paths = download_pixabay(animal_en, output_dir, count=pixabay_count, prefix="pixabay")
    if len(pexels_paths) + len(pixabay_paths) < total_count:
        # ถ้า Pixabay ได้ไม่ครบ quota ให้กลับไปดึงจาก Pexels อีก
        more_needed = total_count - (len(pexels_paths) + len(pixabay_paths))
        if more_needed > 0:
            more_pexels = download_pexels(animal_en, output_dir, count=more_needed, prefix="pexels2")
            pexels_paths += more_pexels
    all_paths = pexels_paths + pixabay_paths
    return all_paths[:total_count]

# --- BLIP & Translation ---
processor = BlipProcessor.from_pretrained(BLIP_MODEL)
blip_model = BlipForConditionalGeneration.from_pretrained(BLIP_MODEL)

def translate_to_thai_deepseek(text):
    """
    ใช้ Deepseek API (https://api.deepseek.com/chat/completions) แปลอังกฤษเป็นไทยแบบแม่นยำสำหรับ dataset
    """
    api_url = os.getenv("DEEPSEEK_API_URL", "https://api.deepseek.com/chat/completions")
    api_key = os.getenv("DEEPSEEK_API_KEY")
    headers = {"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"}
    prompt = (
        "คุณคือผู้เชี่ยวชาญด้านการแปลภาษาอังกฤษเป็นภาษาไทย "
        "โปรดแปลข้อความต่อไปนี้เป็นภาษาไทยที่เป็นธรรมชาติ ถูกต้องตามบริบท และเหมาะสมกับการใช้งานใน dataset สำหรับ AI/ML "
        "ห้ามแปลทับศัพท์หรือใช้ภาษาอังกฤษปะปน และห้ามอธิบายเพิ่มใด ๆ ให้ตอบกลับเฉพาะข้อความที่แปลแล้วเท่านั้น\n\n"
        f"English: {text}\n"
        "Thai:"
    )
    req_body = {
        "model": "deepseek-chat",
        "temperature": 0.2,
        "messages": [
            {"role": "system", "content": "You are a professional English-to-Thai translator for AI datasets."},
            {"role": "user", "content": prompt}
        ]
    }
    try:
        resp = requests.post(api_url, headers=headers, json=req_body, timeout=20)
        resp.raise_for_status()
        data = resp.json()
        # Deepseek API: content อยู่ใน data["choices"][0]["message"]["content"]
        return data["choices"][0]["message"]["content"].strip()
    except Exception as e:
        print(f"[ERROR] Deepseek translate: {e}")
        return ""

def get_vision_labels_from_task(task_name="vision_animals"):
    tasks = get_task_definitions()
    if task_name not in tasks:
        print(f"[ERROR] Task '{task_name}' not found in tasks.json", file=sys.stderr)
        sys.exit(1)
    task = tasks[task_name]
    # สมมติ schema มี field 'label' หรือ enum/array ของ label
    # ตัวอย่าง: {'parameters': {'labels': {'enum_values': [...]}}}
    labels = []
    if "parameters" in task:
        for param in task["parameters"].values():
            if param.get("param_type") == "enum" and "enum_values" in param:
                labels = param["enum_values"]
    return labels

def get_vision_labels_from_api(task_name="vision_animals", api_url="http://localhost:8000/tasks"):
    try:
        resp = requests.get(f"{api_url}/{task_name}", timeout=10)
        resp.raise_for_status()
        task = resp.json()
        labels = []
        if "parameters" in task:
            for param in task["parameters"].values():
                if param.get("param_type") == "enum" and "enum_values" in param:
                    labels = param["enum_values"]
        return labels
    except Exception as e:
        print(f"[ERROR] Fetching task from API: {e}")
        return []

def get_auto_per_class(total_images, num_classes):
    # กำหนดจำนวนภาพต่อ class อัตโนมัติ (เช่น 10,000 ภาพรวม)
    return max(1, total_images // num_classes)

# --- Main Workflow ---
def main():
    now = datetime.now().strftime("%Y%m%d-%H%M%S")
    dataset_name = f"vision-animals-dataset-{now}"
    outdir = os.path.join("data/output", dataset_name)
    img_outdir = os.path.join(outdir, "images")
    os.makedirs(img_outdir, exist_ok=True)
    output_path = os.path.join(outdir, f"{dataset_name}.jsonl")
    entries = []
    # --- ดึง labels จาก task API ---
    labels = get_vision_labels_from_api("vision_animals")
    # --- กำหนด PER_CLASS อัตโนมัติ ---
    total_images = int(os.getenv("VISION_TOTAL_IMAGES", "20"))  # default 20
    global PER_CLASS
    PER_CLASS = get_auto_per_class(total_images, len(labels))
    print(f"[INFO] PER_CLASS set to {PER_CLASS} (total_images={total_images}, num_classes={len(labels)})")
    img_counter = 1
    alphabet = 'ABCDEFGHIJKLMNOPQRSTUVWXYZ'
    all_entries = []
    for label_idx, label in enumerate(labels):
        animal_en = label["en"]
        animal_th = label["th"]
        # กำหนด output_dir ให้ถูกต้องก่อนเรียก download_images_balanced
        output_dir = os.path.join("photo/images", animal_en)
        os.makedirs(output_dir, exist_ok=True)
        all_paths = download_images_balanced(animal_en, output_dir, total_count=PER_CLASS)
        for img_path in tqdm(all_paths, desc=f"{animal_en}"):
            try:
                raw_image = Image.open(img_path).convert('RGB')
                inputs = processor(raw_image, return_tensors="pt")
                with torch.no_grad():
                    out = blip_model.generate(**inputs, max_new_tokens=30)
                # รอให้ BLIP สร้าง caption เสร็จจริง ๆ ก่อน (auto wait ตาม resource)
                import time as _time
                _time.sleep(min(2, max(0.5, len(all_paths)/100)))  # auto wait: 0.5-2s ตามจำนวนภาพ
                caption = processor.decode(out[0], skip_special_tokens=True)
                # เปิดการแปลด้วย Deepseek API (endpoint จริง)
                if not any(ord(c) > 127 for c in caption):
                    caption = translate_to_thai_deepseek(caption)
            except Exception as e:
                print(f"[ERROR] {img_path}: {e}")
                caption = ""
            # เพิ่มอักษรนำหน้ารหัสภาพ (A0001, B0001, ...)
            prefix = alphabet[label_idx % len(alphabet)]
            img_filename = f"{prefix}{img_counter:04d}.jpg"
            new_img_path = os.path.join("images", img_filename)
            shutil.copy2(img_path, os.path.join(img_outdir, img_filename))
            entry = {
                "id": f"{prefix}{img_counter:04d}",
                "content": {
                    "image_path": new_img_path.replace("\\", "/"),
                    "label_en": animal_en,
                    "label_th": animal_th,
                    "caption": caption
                },
                "metadata": {
                    "source": "pexels" if "pexels" in os.path.basename(img_path) else "pixabay",
                    "caption_model": "blip-base"
                }
            }
            all_entries.append(entry)
            img_counter += 1
    # --- สุ่มเติมจนครบ total_images ---
    if len(all_entries) < total_images:
        print(f"[INFO] Randomly sampling {total_images-len(all_entries)} more images to fill up to {total_images}")
        extra_entries = random.choices(all_entries, k=total_images-len(all_entries))
        all_entries.extend(extra_entries)
    # --- Assign รหัสไฟล์/ID ใหม่แบบเรียงลำดับ (A0001, A0002, ... B0001, ...) ---
    assigned_entries = []
    img_id_map = {}  # เก็บ mapping เดิม->ใหม่
    alphabet = 'ABCDEFGHIJKLMNOPQRSTUVWXYZ'
    per_label_counter = {i: 1 for i in range(len(labels))}
    for idx, entry in enumerate(all_entries[:total_images]):
        # หา label index
        label_en = entry['content']['label_en']
        label_idx = next((i for i, l in enumerate(labels) if l['en'] == label_en), 0)
        prefix = alphabet[label_idx % len(alphabet)]
        img_num = per_label_counter[label_idx]
        new_id = f"{prefix}{img_num:04d}"
        new_img_filename = f"{new_id}.jpg"
        new_img_path = os.path.join("images", new_img_filename)
        # เปลี่ยนชื่อไฟล์จริง (ถ้ายังไม่มีไฟล์นี้)
        old_img_full = os.path.join(img_outdir, os.path.basename(entry['content']['image_path']))
        new_img_full = os.path.join(img_outdir, new_img_filename)
        if not os.path.exists(new_img_full):
            try:
                shutil.move(old_img_full, new_img_full)
            except Exception as e:
                print(f"[WARN] Cannot move {old_img_full} -> {new_img_full}: {e}")
        # อัปเดต entry
        entry['id'] = new_id
        entry['content']['image_path'] = new_img_path.replace("\\", "/")
        assigned_entries.append(entry)
        per_label_counter[label_idx] += 1
    # Export jsonl
    with open(output_path, "w", encoding="utf-8") as f:
        for entry in assigned_entries:
            f.write(json.dumps(entry, ensure_ascii=False) + "\n")
    print(f"\nExported {len(assigned_entries)} entries to {output_path}")
    print(f"All images renamed and copied to {img_outdir}")

if __name__ == "__main__":
    main()
