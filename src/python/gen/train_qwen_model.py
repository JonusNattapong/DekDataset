import os
import json
import random
from transformers import AutoTokenizer, AutoModelForSequenceClassification, Trainer, TrainingArguments
from datasets import Dataset
import pandas as pd
import torch
import numpy as np
from sklearn.metrics import accuracy_score, f1_score

# --- CONFIG ---
MODEL_NAME = "Qwen/Qwen3-0.6B"  # โมเดล Qwen ขนาด 0.6B
OUTPUT_DIR = "data/output/thai-education-qwen-model"
EVAL_DIR = "data/output/evaluation-qwen-results"
os.makedirs(OUTPUT_DIR, exist_ok=True)
os.makedirs(EVAL_DIR, exist_ok=True)

# กำหนดวิชา (labels)
subjects = [
    "ภาษาไทย", "คณิตศาสตร์", "วิทยาศาสตร์", "สังคมศึกษา", 
    "ศิลปะ", "สุขศึกษาและพลศึกษา", "การงานอาชีพและเทคโนโลยี", "ภาษาอังกฤษ"
]

# สร้าง id2label และ label2id
id2label = {i: label for i, label in enumerate(subjects)}
label2id = {label: i for i, label in enumerate(subjects)}

print(f"เริ่มโหลดโมเดล {MODEL_NAME}...")

# --- โหลด Tokenizer ---
try:
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
except Exception as e:
    print(f"เกิดข้อผิดพลาดในการโหลด tokenizer: {e}")
    print("กำลังลอง fallback ไปใช้โมเดล wangchanberta...")
    MODEL_NAME = "airesearch/wangchanberta-base-att-spm-uncased"
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

# --- โหลด Dataset ---
print("กำลังโหลดชุดข้อมูล...")
# 1. โหลดข้อมูลตัวอย่าง (ป.1-ป.6)
primary_file = "data/output/auto-dataset-primary-exam-sample.jsonl"
with open(primary_file, encoding="utf-8") as f:
    primary_data = [json.loads(line) for line in f if line.strip()]

# 2. โหลดข้อมูลตัวอย่าง (ม.1-ม.6)
secondary_file = "data/output/auto-dataset-secondary-exam-sample.jsonl"
with open(secondary_file, encoding="utf-8") as f:
    secondary_data = [json.loads(line) for line in f if line.strip()]

# 3. รวมข้อมูล
all_data = primary_data + secondary_data
print(f"โหลดข้อมูลสำเร็จ: {len(primary_data)} ตัวอย่างระดับประถม + {len(secondary_data)} ตัวอย่างระดับมัธยม = {len(all_data)} ตัวอย่างทั้งหมด")

# --- สร้างชุดข้อมูลสำหรับเทรน ---
print("กำลังเตรียมข้อมูลสำหรับเทรน...")
train_examples = []
for item in all_data:
    subject = item.get("subject")
    if not subject or subject not in subjects:
        continue

    # ดึงข้อความจากหลายฟิลด์มาช่วยในการเรียนรู้
    text_parts = []
    if "content" in item and item["content"]:
        text_parts.append(item["content"])
    if "question" in item and item["question"]:
        text_parts.append(item["question"])
    if "choices" in item and item["choices"] and isinstance(item["choices"], list):
        text_parts.append(" ".join(item["choices"]))
    
    text = " ".join(text_parts)
    if not text:
        continue
    
    # เพิ่มตัวอย่าง
    train_examples.append({
        "text": text,
        "label": label2id[subject],
        "subject": subject
    })

# แบ่ง train/validation
random.seed(42)
random.shuffle(train_examples)
split = int(0.8 * len(train_examples))
train_data = train_examples[:split]
eval_data = train_examples[split:]

print(f"เตรียมข้อมูลสำเร็จ: ชุด Train: {len(train_data)} ตัวอย่าง, ชุด Eval: {len(eval_data)} ตัวอย่าง")

# --- สร้าง HuggingFace Dataset ---
train_dataset = Dataset.from_pandas(pd.DataFrame(train_data))
eval_dataset = Dataset.from_pandas(pd.DataFrame(eval_data))

# --- Preprocessing Function ---
def preprocess_function(examples):
    return tokenizer(
        examples["text"], 
        truncation=True, 
        padding="max_length",
        max_length=256  # เพิ่มความยาว sequence สำหรับ Qwen
    )

# ใช้ map เพื่อ tokenize ทั้งชุดข้อมูล
print("กำลัง tokenize ข้อมูล...")
train_dataset = train_dataset.map(preprocess_function, batched=True)
eval_dataset = eval_dataset.map(preprocess_function, batched=True)

# --- โหลด Model ---
print(f"กำลังโหลดโมเดล {MODEL_NAME} สำหรับการจำแนกข้อความ...")
try:
    model = AutoModelForSequenceClassification.from_pretrained(
        MODEL_NAME, 
        num_labels=len(subjects),
        id2label=id2label,
        label2id=label2id
    )
    # ตรวจสอบว่า model มี pad token ตรงกับ tokenizer หรือไม่
    if hasattr(model.config, 'pad_token_id') and model.config.pad_token_id is None:
        model.config.pad_token_id = tokenizer.pad_token_id
except Exception as e:
    print(f"เกิดข้อผิดพลาดในการโหลดโมเดล: {e}")
    print("กำลังลอง fallback ไปใช้โมเดล wangchanberta...")
    MODEL_NAME = "airesearch/wangchanberta-base-att-spm-uncased"
    model = AutoModelForSequenceClassification.from_pretrained(
        MODEL_NAME,
        num_labels=len(subjects),
        id2label=id2label,
        label2id=label2id
    )

# --- ฟังก์ชันประเมินผล ---
def compute_metrics(eval_pred):
    logits, labels = eval_pred
    predictions = np.argmax(logits, axis=-1)
    return {
        "accuracy": accuracy_score(labels, predictions),
        "f1_macro": f1_score(labels, predictions, average="macro"),
        "f1_weighted": f1_score(labels, predictions, average="weighted")
    }

# --- กำหนดค่า Training Arguments ---
print("กำลังเตรียมการเทรน...")
training_args = TrainingArguments(
    output_dir=OUTPUT_DIR,
    learning_rate=5e-5,
    per_device_train_batch_size=4,  # ลดขนาด batch ลงสำหรับ Qwen ที่ใหญ่กว่า
    per_device_eval_batch_size=4,
    num_train_epochs=5,
    weight_decay=0.01,
    eval_steps=10,
    save_steps=10,
    load_best_model_at_end=True,
    push_to_hub=False,
    logging_dir=f"{OUTPUT_DIR}/logs",
    logging_steps=5,
    report_to=["tensorboard"]
)

# --- เริ่มเทรน ---
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=eval_dataset,
    tokenizer=tokenizer,
    compute_metrics=compute_metrics,
)

print("เริ่มการเทรนโมเดล...")
trainer.train()

# --- บันทึกโมเดล ---
print(f"บันทึกโมเดลไปที่ {OUTPUT_DIR}")
trainer.save_model(OUTPUT_DIR)
tokenizer.save_pretrained(OUTPUT_DIR)

# --- ประเมินผล ---
print("กำลังประเมินผลโมเดล...")
eval_results = trainer.evaluate()
print(f"ผลการประเมิน: {eval_results}")

# บันทึกผลการประเมิน
with open(os.path.join(EVAL_DIR, "qwen_eval_results.json"), "w", encoding="utf-8") as f:
    json.dump(eval_results, f, ensure_ascii=False, indent=2)

print("การเทรนและประเมินผลเสร็จสิ้น!")

# --- ทดสอบโมเดลกับตัวอย่างที่หลากหลาย ---
print("\nทดสอบโมเดลกับตัวอย่างที่หลากหลาย:")
example_texts = [
    "ไข่ -> ตัวหนอน -> ดักแด้ -> ผีเสื้อ",  # วิทยาศาสตร์
    "การบวกเลขสองหลัก 25 + 13 = 38",  # คณิตศาสตร์
    "พระเจ้าอยู่หัวรัชกาลที่ 9 เป็นที่รักของคนไทย", # สังคมศึกษา
    "การอ่านพยัญชนะไทย ก ไก่ ข ไข่",  # ภาษาไทย
    "I can speak English very well", # ภาษาอังกฤษ
    "วาดรูประบายสีและทัศนศิลป์", # ศิลปะ
    "การออกกำลังกายทำให้ร่างกายแข็งแรง", # สุขศึกษาและพลศึกษา
    "การเขียนโปรแกรมคอมพิวเตอร์" # การงานอาชีพและเทคโนโลยี
]

for text in example_texts:
    inputs = tokenizer(text, return_tensors="pt", truncation=True)
    with torch.no_grad():
        outputs = model(**inputs)
    
    logits = outputs.logits
    probabilities = torch.nn.functional.softmax(logits, dim=-1)[0]
    predicted_class_id = torch.argmax(probabilities).item()
    predicted_class = id2label[predicted_class_id]
    confidence = probabilities[predicted_class_id].item()
    
    print(f"ข้อความ: {text}")
    print(f"ทำนาย: {predicted_class} (ความมั่นใจ: {confidence:.2%})")
    print("---")
