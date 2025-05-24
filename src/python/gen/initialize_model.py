import os
import json
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch

# โหลดโมเดลที่มีอยู่ในฐานข้อมูล HuggingFace
model_name = "Qwen/Qwen3-0.6B"
tokenizer = AutoTokenizer.from_pretrained(model_name)

# กำหนด paths
output_dir = "data/output/thai-education-model"
os.makedirs(output_dir, exist_ok=True)

# กำหนด labels (วิชา)
subjects = [
    "ภาษาไทย", "คณิตศาสตร์", "วิทยาศาสตร์", "สังคมศึกษา", 
    "ศิลปะ", "สุขศึกษาและพลศึกษา", "การงานอาชีพและเทคโนโลยี", "ภาษาอังกฤษ"
]

# สร้าง id2label และ label2id
id2label = {i: label for i, label in enumerate(subjects)}
label2id = {label: i for i, label in enumerate(subjects)}

# โหลดโมเดลพร้อมกำหนดจำนวน labels
model = AutoModelForSequenceClassification.from_pretrained(
    model_name,
    num_labels=len(subjects),
    id2label=id2label,
    label2id=label2id,
)

# บันทึกโมเดลและ tokenizer
model.save_pretrained(output_dir)
tokenizer.save_pretrained(output_dir)

# ตรวจสอบว่า config.json มีข้อมูล id2label และ label2id
config_path = os.path.join(output_dir, "config.json")
with open(config_path, "r", encoding="utf-8") as f:
    config = json.load(f)

print(f"Saved config with {len(config.get('id2label', {}))} labels")
print(f"id2label: {config.get('id2label', {})}")

# ตรวจสอบไฟล์ที่บันทึก
print("\nSaved files:")
for file in os.listdir(output_dir):
    print(f"- {file}")

# ทดสอบโมเดล
test_texts = [
    "ไข่ -> ตัวหนอน -> ดักแด้ -> ผีเสื้อ",  # วิทยาศาสตร์
    "การบวกเลขสองหลัก 25 + 13 = 38",  # คณิตศาสตร์
    "พระเจ้าอยู่หัวรัชกาลที่ 9 เป็นที่รักของคนไทย",  # สังคมศึกษา
    "การอ่านพยัญชนะไทย ก ไก่ ข ไข่",  # ภาษาไทย
    "I can speak English very well",  # ภาษาอังกฤษ
    "วาดรูประบายสีและทัศนศิลป์",  # ศิลปะ
    "การออกกำลังกายทำให้ร่างกายแข็งแรง",  # สุขศึกษาและพลศึกษา
    "การเขียนโปรแกรมคอมพิวเตอร์",  # การงานอาชีพและเทคโนโลยี
]

print("\nTest predictions:")
for text in test_texts:
    inputs = tokenizer(text, return_tensors="pt", truncation=True, max_length=128)
    with torch.no_grad():
        outputs = model(**inputs)
    
    probs = torch.nn.functional.softmax(outputs.logits, dim=1)[0]
    prediction_id = torch.argmax(probs).item()
    prediction = id2label[prediction_id]
    confidence = probs[prediction_id].item()
    
    print(f"Text: {text}")
    print(f"Prediction: {prediction} (confidence: {confidence:.2%})")
    print()

print(f"Model initialized and saved to {output_dir}")
print("Note: This is NOT a trained model. It needs to be trained on your dataset.")
