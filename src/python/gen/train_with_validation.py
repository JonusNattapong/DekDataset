import os
import json
import random
from transformers import AutoTokenizer, AutoModelForSequenceClassification, Trainer, TrainingArguments, DataCollatorWithPadding
from datasets import Dataset, load_dataset
import pandas as pd
import torch
import numpy as np
from sklearn.metrics import accuracy_score, f1_score

# --- CONFIG ---
MODEL_NAME = "airesearch/wangchanberta-base-att-spm-uncased"
OUTPUT_DIR = "data/output/thai-education-model"
EVAL_DIR = "data/output/evaluation-results"
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

# --- โหลด Dataset ---
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
print(f"Loaded {len(primary_data)} primary examples + {len(secondary_data)} secondary examples = {len(all_data)} total")

# --- สร้างชุดข้อมูลสำหรับเทรน ---
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

print(f"Train: {len(train_data)} examples, Eval: {len(eval_data)} examples")

# --- สร้าง HuggingFace Dataset ---
train_dataset = Dataset.from_pandas(pd.DataFrame(train_data))
eval_dataset = Dataset.from_pandas(pd.DataFrame(eval_data))

# --- Tokenizer & Preprocessing ---
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

def preprocess_function(examples):
    return tokenizer(
        examples["text"], 
        truncation=True, 
        padding="max_length",
        max_length=128
    )

# ใช้ map เพื่อ tokenize ทั้งชุดข้อมูล
train_dataset = train_dataset.map(preprocess_function, batched=True)
eval_dataset = eval_dataset.map(preprocess_function, batched=True)

# --- Model ---
model = AutoModelForSequenceClassification.from_pretrained(
    MODEL_NAME, 
    num_labels=len(subjects),
    id2label=id2label,
    label2id=label2id
)

# --- ประเมินผล ---
def compute_metrics(eval_pred):
    logits, labels = eval_pred
    predictions = np.argmax(logits, axis=-1)
    return {
        "accuracy": accuracy_score(labels, predictions),
        "f1_macro": f1_score(labels, predictions, average="macro")
    }

# --- Training ---
training_args = TrainingArguments(
    output_dir=OUTPUT_DIR,
    learning_rate=3e-5,
    per_device_train_batch_size=8,
    per_device_eval_batch_size=8,
    num_train_epochs=5,
    weight_decay=0.01,
    # evaluation_strategy="epoch",  # รุ่นเก่าไม่รองรับ
    # save_strategy="epoch",  # รุ่นเก่าไม่รองรับ
    eval_steps=100,  # evaluate every 100 steps
    save_steps=100,  # save model every 100 steps
    # load_best_model_at_end=True,  # รุ่นเก่าไม่รองรับ
    push_to_hub=False,
    logging_dir=f"{OUTPUT_DIR}/logs",
    logging_steps=10,
    # report_to=["tensorboard"],  # รุ่นเก่าไม่รองรับ
)

data_collator = DataCollatorWithPadding(tokenizer=tokenizer)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=eval_dataset,
    tokenizer=tokenizer,
    data_collator=data_collator,
    compute_metrics=compute_metrics,
)

# --- Train ---
print("Starting training...")
trainer.train()

# --- Save Model ---
print(f"Saving model to {OUTPUT_DIR}")
trainer.save_model(OUTPUT_DIR)
tokenizer.save_pretrained(OUTPUT_DIR)

# --- Evaluate ---
print("Evaluating model...")
eval_results = trainer.evaluate()
print(f"Evaluation results: {eval_results}")

# Save evaluation results
with open(os.path.join(EVAL_DIR, "eval_results.json"), "w") as f:
    json.dump(eval_results, f)

print("Training and evaluation completed!")

# Test model on a few examples
print("\nTesting model on a few examples:")
example_texts = [
    "ไข่ -> ตัวหนอน -> ดักแด้ -> ผีเสื้อ",  # วิทยาศาสตร์
    "การบวกเลขสองหลัก 25 + 13 = 38",  # คณิตศาสตร์
    "พระเจ้าอยู่หัวรัชกาลที่ 9 เป็นที่รักของคนไทย", # สังคมศึกษา
    "การอ่านพยัญชนะไทย ก ไก่ ข ไข่",  # ภาษาไทย
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
    
    print(f"Text: {text}")
    print(f"Predicted: {predicted_class} (confidence: {confidence:.2%})")
    print("---")
