import json
import random
from transformers import AutoTokenizer, AutoModelForSequenceClassification, Trainer, TrainingArguments
from datasets import Dataset
import pandas as pd

# --- CONFIG ---
MODEL_NAME = "airesearch/wangchanberta-base-att-spm-uncased"  # โมเดลภาษาไทย
DATA_PATH = "data/output/merged-primary-secondary.jsonl"
OUTPUT_DIR = "data/output/model-primary-secondary"
LABEL_FIELD = "subject"  # หรือเปลี่ยนเป็น field อื่นที่ต้องการเทรน

# --- Load Data ---
with open(DATA_PATH, encoding="utf-8") as f:
    data = [json.loads(line) for line in f if line.strip()]

# --- Preprocess ---
labels = sorted(list(set(item[LABEL_FIELD] for item in data if LABEL_FIELD in item)))
label2id = {l: i for i, l in enumerate(labels)}
for item in data:
    item["label"] = label2id.get(item.get(LABEL_FIELD), 0)
    item["text"] = item.get("question") or item.get("content") or ""

# --- Convert to HuggingFace Dataset ---
df = pd.DataFrame(data)
dataset = Dataset.from_pandas(df)

# --- Tokenizer ---
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
def preprocess(example):
    return tokenizer(example["text"], truncation=True, padding="max_length", max_length=128)
dataset = dataset.map(preprocess)

# --- Train/Test Split ---
dataset = dataset.train_test_split(test_size=0.1, seed=42)

# --- Model ---
model = AutoModelForSequenceClassification.from_pretrained(MODEL_NAME, num_labels=len(labels))

# --- Training ---
training_args = TrainingArguments(
    output_dir=OUTPUT_DIR,
    # evaluation_strategy="epoch",  # ลบออกจาก param หรือแก้ตามเวอร์ชัน
    # save_strategy="epoch",  # ลบออกจาก param หรือแก้ตามเวอร์ชัน
    eval_steps=100,  # evaluate every 100 steps
    save_steps=100,  # save model every 100 steps
    learning_rate=2e-5,
    per_device_train_batch_size=8,
    per_device_eval_batch_size=8,
    num_train_epochs=3,
    weight_decay=0.01,
    logging_dir=f"{OUTPUT_DIR}/logs",
    logging_steps=10,
)

# กำหนด id2label และ label2id ชัดเจน
labels = sorted(list(set(item[LABEL_FIELD] for item in data if LABEL_FIELD in item)))
label2id = {label: i for i, label in enumerate(labels)}
id2label = {i: label for i, label in enumerate(labels)}

print(f"Labels: {labels}")
print(f"label2id: {label2id}")
print(f"id2label: {id2label}")

# ปรับ config ของโมเดล
model.config.id2label = id2label
model.config.label2id = label2id

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=dataset["train"],
    eval_dataset=dataset["test"],
    tokenizer=tokenizer,
)

trainer.train()

# บันทึกโมเดลและ tokenizer แบบเต็มรูปแบบ
trainer.save_model(OUTPUT_DIR)  # บันทึกโมเดลและ config
tokenizer.save_pretrained(OUTPUT_DIR)  # บันทึก tokenizer

# ตรวจสอบการบันทึก
import os
print("ไฟล์ที่บันทึก:")
for file in os.listdir(OUTPUT_DIR):
    print(f"- {file}")

print("Training complete! Model saved at:", OUTPUT_DIR)
