import os
import json
import random
from transformers import AutoTokenizer, AutoModelForSequenceClassification, Trainer, TrainingArguments
from transformers import DataCollatorWithPadding, EvalPrediction
from datasets import Dataset
import pandas as pd
import numpy as np
from sklearn.metrics import accuracy_score, f1_score
import torch
import matplotlib.pyplot as plt
import time
from datetime import datetime

# --- CONFIG ---
MODEL_NAME = "airesearch/wangchanberta-base-att-spm-uncased"
OUTPUT_DIR = "data/output/thai-education-model-improved"
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

# --- โหลดข้อมูล ---
def load_data():
    print("กำลังโหลดข้อมูล...")
    
    # 1. โหลดข้อมูลตัวอย่าง (ป.1-ป.6)
    primary_file = "data/output/auto-dataset-primary-exam-sample.jsonl"
    with open(primary_file, encoding="utf-8") as f:
        primary_data = [json.loads(line) for line in f if line.strip()]
    print(f"โหลดข้อมูลระดับประถมศึกษา: {len(primary_data)} ตัวอย่าง")
    
    # 2. โหลดข้อมูลตัวอย่าง (ม.1-ม.6)
    secondary_file = "data/output/auto-dataset-secondary-exam-sample.jsonl"
    with open(secondary_file, encoding="utf-8") as f:
        secondary_data = [json.loads(line) for line in f if line.strip()]
    print(f"โหลดข้อมูลระดับมัธยมศึกษา: {len(secondary_data)} ตัวอย่าง")
    
    # 3. โหลดข้อมูลเพิ่มเติมจาก augmentation
    augmented_data = []
    augmented_file = "data/output/augmented_samples.jsonl"
    if os.path.exists(augmented_file):
        with open(augmented_file, encoding="utf-8") as f:
            augmented_data = [json.loads(line) for line in f if line.strip()]
        print(f"โหลดข้อมูลเพิ่มเติม: {len(augmented_data)} ตัวอย่าง")
    
    # 4. รวมข้อมูลทั้งหมด
    all_data = primary_data + secondary_data + augmented_data
    print(f"รวมข้อมูลทั้งหมด: {len(all_data)} ตัวอย่าง")
    
    return all_data

# --- เตรียมข้อมูล ---
def prepare_data(all_data):
    print("กำลังเตรียมข้อมูล...")
    
    # แปลงรูปแบบข้อมูลให้เหมาะกับการเทรน
    train_examples = []
    subject_count = {subject: 0 for subject in subjects}
    
    for item in all_data:
        subject = item.get("subject")
        if not subject or subject not in subjects:
            continue
        
        # ดึงข้อความจากหลายฟิลด์เพื่อเพิ่มข้อมูล
        if "text" in item and item["text"]:
            text = item["text"]
        else:
            text_parts = []
            if "content" in item and item["content"]:
                text_parts.append(item["content"])
            if "question" in item and item["question"]:
                text_parts.append(item["question"])
            if "choices" in item and item["choices"]:
                choices_text = " ".join([f"{i+1}. {c}" for i, c in enumerate(item["choices"])])
                text_parts.append(choices_text)
            
            text = " ".join(text_parts)
        
        if not text.strip():
            continue
        
        # เพิ่มตัวอย่าง
        train_examples.append({
            "text": text,
            "label": label2id[subject],
            "subject": subject
        })
        
        # นับจำนวนตัวอย่างในแต่ละวิชา
        subject_count[subject] += 1
    
    # แสดงสถิติจำนวนตัวอย่างในแต่ละวิชา
    print("\nจำนวนตัวอย่างในแต่ละวิชา:")
    for subject, count in subject_count.items():
        print(f"  {subject}: {count} ตัวอย่าง")
    
    # แบ่งข้อมูลเป็น train/validation (80%/20%)
    random.seed(42)
    random.shuffle(train_examples)
    split = int(0.8 * len(train_examples))
    train_data = train_examples[:split]
    eval_data = train_examples[split:]
    
    print(f"\nแบ่งข้อมูล: Train {len(train_data)} ตัวอย่าง, Validation {len(eval_data)} ตัวอย่าง")
    
    # สร้าง HuggingFace Dataset
    train_dataset = Dataset.from_pandas(pd.DataFrame(train_data))
    eval_dataset = Dataset.from_pandas(pd.DataFrame(eval_data))
    
    return train_dataset, eval_dataset

# --- เทรนโมเดล ---
def train_model(train_dataset, eval_dataset):
    print("\nกำลังเตรียมเทรนโมเดล...")
    
    # 1. โหลดโมเดลและ tokenizer
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    model = AutoModelForSequenceClassification.from_pretrained(
        MODEL_NAME,
        num_labels=len(subjects),
        id2label=id2label,
        label2id=label2id
    )
    
    # 2. เตรียมข้อมูลด้วย tokenizer
    def preprocess_function(examples):
        return tokenizer(
            examples["text"],
            truncation=True,
            padding="max_length",
            max_length=256  # เพิ่มความยาวให้รองรับข้อความที่ซับซ้อนขึ้น
        )
    
    print("  กำลัง tokenize ข้อมูล...")
    train_dataset = train_dataset.map(preprocess_function, batched=True)
    eval_dataset = eval_dataset.map(preprocess_function, batched=True)
    
    # 3. สร้าง compute_metrics function
    def compute_metrics(eval_pred):
        logits, labels = eval_pred
        predictions = np.argmax(logits, axis=-1)
        return {
            "accuracy": accuracy_score(labels, predictions),
            "f1_macro": f1_score(labels, predictions, average="macro")
        }
    
    # 4. กำหนด training arguments
    training_args = TrainingArguments(
        output_dir=OUTPUT_DIR,
        learning_rate=2e-5,
        per_device_train_batch_size=16,
        per_device_eval_batch_size=16,
        num_train_epochs=5,
        weight_decay=0.01,
        evaluation_strategy="epoch",
        save_strategy="epoch",
        load_best_model_at_end=True,
        metric_for_best_model="f1_macro",
        push_to_hub=False,
        logging_dir=f"{OUTPUT_DIR}/logs",
        logging_steps=10,
        report_to=["tensorboard"],
    )
    
    # 5. สร้าง data collator
    data_collator = DataCollatorWithPadding(tokenizer=tokenizer)
    
    # 6. สร้าง trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        tokenizer=tokenizer,
        data_collator=data_collator,
        compute_metrics=compute_metrics,
    )
    
    # 7. เทรนโมเดล
    print("\nเริ่มเทรนโมเดล...")
    start_time = time.time()
    
    train_result = trainer.train()
    
    end_time = time.time()
    training_time = end_time - start_time
    print(f"\nเทรนโมเดลเสร็จสมบูรณ์ในเวลา {training_time:.2f} วินาที")
    
    # 8. บันทึกโมเดล
    print(f"บันทึกโมเดลไปยัง {OUTPUT_DIR}")
    trainer.save_model(OUTPUT_DIR)
    tokenizer.save_pretrained(OUTPUT_DIR)
    
    # 9. ประเมินโมเดล
    print("\nประเมินโมเดล...")
    eval_results = trainer.evaluate()
    print(f"ผลการประเมิน: {eval_results}")
    
    # 10. บันทึกผลการประเมิน
    eval_output_dir = os.path.join(EVAL_DIR, datetime.now().strftime("%Y%m%d_%H%M%S"))
    os.makedirs(eval_output_dir, exist_ok=True)
    
    with open(os.path.join(eval_output_dir, "eval_results.json"), "w") as f:
        json.dump(eval_results, f)
    
    # 11. สร้างกราฟแสดงผลการเทรน
    metrics = trainer.state.log_history
    
    # แยก training loss และ validation metrics
    train_loss = []
    val_metrics = []
    
    for item in metrics:
        if 'loss' in item and 'epoch' in item:
            train_loss.append((item['epoch'], item['loss']))
        elif 'eval_loss' in item and 'epoch' in item:
            val_metrics.append((
                item['epoch'], 
                item['eval_loss'], 
                item.get('eval_accuracy', 0), 
                item.get('eval_f1_macro', 0)
            ))
    
    # สร้างกราฟ
    if train_loss and val_metrics:
        plt.figure(figsize=(12, 6))
        
        # Plot training loss
        epochs, loss = zip(*train_loss)
        plt.plot(epochs, loss, 'b-', label='Training Loss')
        
        # Plot validation metrics
        if val_metrics:
            val_epochs, val_loss, val_acc, val_f1 = zip(*val_metrics)
            plt.plot(val_epochs, val_loss, 'r-', label='Validation Loss')
            plt.plot(val_epochs, val_acc, 'g-', label='Validation Accuracy')
            plt.plot(val_epochs, val_f1, 'y-', label='Validation F1 Macro')
        
        plt.xlabel('Epochs')
        plt.ylabel('Value')
        plt.title('Training Metrics')
        plt.legend()
        plt.grid(True)
        plt.savefig(os.path.join(eval_output_dir, 'training_metrics.png'))
    
    # 12. ทดสอบโมเดลกับตัวอย่างง่ายๆ
    print("\nทดสอบโมเดลกับตัวอย่างง่ายๆ:")
    test_examples = [
        "วิธีการคำนวณพื้นที่ของสามเหลี่ยม",  # คณิตศาสตร์
        "กฎของเมนเดลในการถ่ายทอดลักษณะทางพันธุกรรม",  # วิทยาศาสตร์
        "การใช้คำราชาศัพท์ให้ถูกต้องตามกาลเทศะ",  # ภาษาไทย
        "หลักปรัชญาเศรษฐกิจพอเพียงของในหลวงรัชกาลที่ 9",  # สังคมศึกษา
        "Present Continuous Tense ใช้กับเหตุการณ์ที่กำลังดำเนินอยู่",  # ภาษาอังกฤษ
    ]
    
    for text in test_examples:
        inputs = tokenizer(text, return_tensors="pt", truncation=True)
        with torch.no_grad():
            outputs = model(**inputs)
        
        logits = outputs.logits
        probs = torch.nn.functional.softmax(logits, dim=-1)[0]
        prediction = subjects[torch.argmax(probs).item()]
        confidence = probs[torch.argmax(probs)].item()
        
        print(f"ข้อความ: {text}")
        print(f"ทำนาย: {prediction} (ความมั่นใจ: {confidence:.2%})\n")
    
    return model, tokenizer, eval_results

# --- เริ่มต้นทำงาน ---
if __name__ == "__main__":
    print("=== เริ่มต้นกระบวนการเทรนโมเดล Thai Education Subject Classifier ===")
    
    # 1. โหลดข้อมูล
    all_data = load_data()
    
    # 2. เตรียมข้อมูล
    train_dataset, eval_dataset = prepare_data(all_data)
    
    # 3. เทรนโมเดล
    model, tokenizer, eval_results = train_model(train_dataset, eval_dataset)
    
    print("\n=== เทรนโมเดลเสร็จสมบูรณ์ ===")
    print(f"โมเดลถูกบันทึกไว้ที่: {OUTPUT_DIR}")
    print(f"ความแม่นยำ (Accuracy): {eval_results.get('eval_accuracy', 0):.4f}")
    print(f"F1 Score (macro): {eval_results.get('eval_f1_macro', 0):.4f}")
    print("\nเพื่อใช้งานโมเดลนี้ ให้ทำตามขั้นตอนต่อไปนี้:")
    print("1. รันสคริปต์ comprehensive_test.py เพื่อประเมินโมเดลอย่างละเอียด")
    print("2. รันสคริปต์ subject_classifier_api.py เพื่อเริ่มต้น API สำหรับใช้งานโมเดล")
    print("3. เข้าใช้งาน API ที่ http://localhost:8000/ หลังจากรันสคริปต์ API")
