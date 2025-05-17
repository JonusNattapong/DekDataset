import os
import json
from PIL import Image
from transformers import BlipProcessor, BlipForConditionalGeneration
from transformers import MarianMTModel, MarianTokenizer
import torch
from tqdm import tqdm
import shutil
from datetime import datetime

pexels_dir = "pexels_images"
pixabay_dir = "pixabay_images"
# --- สร้างโฟลเดอร์ output vision dataset ---
dataset_name = f"vision-cat-dataset-{datetime.now().strftime('%Y%m%d')}"
outdir = os.path.join("data/output", dataset_name)
img_outdir = os.path.join(outdir, "images")
os.makedirs(img_outdir, exist_ok=True)
output_path = os.path.join(outdir, f"{dataset_name}.jsonl")

# โหลด BLIP model
processor = BlipProcessor.from_pretrained("Salesforce/blip-image-captioning-base")
model = BlipForConditionalGeneration.from_pretrained("Salesforce/blip-image-captioning-base")

# โหลดโมเดลแปลอังกฤษ→ไทย (Helsinki-NLP/opus-mt-en-th)
mt_model_name = "Helsinki-NLP/opus-mt-en-th"
mt_tokenizer = MarianTokenizer.from_pretrained(mt_model_name)
mt_model = MarianMTModel.from_pretrained(mt_model_name)

def translate_to_thai(text):
    batch = mt_tokenizer([text], return_tensors="pt", padding=True)
    gen = mt_model.generate(**batch)
    return mt_tokenizer.decode(gen[0], skip_special_tokens=True)

entries = []

# รวม path ภาพจากทั้งสองแหล่ง
pexels_imgs = sorted([os.path.join(pexels_dir, f) for f in os.listdir(pexels_dir) if f.lower().endswith('.jpg')])
pixabay_imgs = sorted([os.path.join(pixabay_dir, f) for f in os.listdir(pixabay_dir) if f.lower().endswith('.jpg')])
all_imgs = [(img, "pexels") for img in pexels_imgs] + [(img, "pixabay") for img in pixabay_imgs]

for img_path, source in tqdm(all_imgs, desc="captioning"):
    try:
        raw_image = Image.open(img_path).convert('RGB')
        # ใช้ prompt ภาษาไทย (แต่ BLIP อาจตอบอังกฤษ)
        inputs = processor(raw_image, return_tensors="pt")
        prompt = "อธิบายภาพนี้เป็นภาษาไทยแบบสั้น ๆ"
        with torch.no_grad():
            out = model.generate(**inputs, max_new_tokens=30)
        caption = processor.decode(out[0], skip_special_tokens=True)
        # ถ้า BLIP ตอบอังกฤษ ให้แปลเป็นไทย
        if not any(ord(c) > 127 for c in caption):
            caption = translate_to_thai(caption)
    except Exception as e:
        print(f"[ERROR] {img_path}: {e}")
        caption = ""
    # คัดลอกภาพไปยัง images/ และอัปเดต path
    img_filename = os.path.basename(img_path)
    new_img_path = os.path.join("images", img_filename)
    shutil.copy2(img_path, os.path.join(img_outdir, img_filename))
    entry = {
        "id": os.path.splitext(img_filename)[0],
        "content": {
            "image_path": new_img_path.replace("\\", "/"),
            "label": "cat",
            "caption": caption
        },
        "metadata": {
            "source": source,
            "caption_model": "blip-base"
        }
    }
    entries.append(entry)

os.makedirs(os.path.dirname(output_path), exist_ok=True)
with open(output_path, "w", encoding="utf-8") as f:
    for entry in entries:
        f.write(json.dumps(entry, ensure_ascii=False) + "\n")

print(f"Exported {len(entries)} entries to {output_path}")
print(f"All images copied to {img_outdir}")
