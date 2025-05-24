import os
import json
import numpy as np
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from datasets import Dataset
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score
import torch
from transformers import Trainer, TrainingArguments, DataCollatorWithPadding
import optuna
from optuna.pruners import MedianPruner
from optuna.samplers import TPESampler

# --- CONFIG ---
MODEL_NAME = "Qwen/Qwen3-0.6B"
MODEL_DIR = "data/output/thai-education-model"
OUTPUT_DIR = "data/output/thai-education-model-tuned"
EVAL_DIR = "data/output/evaluation-results"
STUDY_NAME = "thai-education-hyperparameter-optimization"
N_TRIALS = 10  # จำนวนการทดลอง
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
def load_data():
    # 1. โหลดข้อมูลตัวอย่าง (ป.1-ป.6)
    primary_file = "data/output/auto-dataset-primary-exam-sample.jsonl"
    with open(primary_file, encoding="utf-8") as f:
        primary_data = [json.loads(line) for line in f if line.strip()]

    # 2. โหลดข้อมูลตัวอย่าง (ม.1-ม.6)
    secondary_file = "data/output/auto-dataset-secondary-exam-sample.jsonl"
    with open(secondary_file, encoding="utf-8") as f:
        secondary_data = [json.loads(line) for line in f if line.strip()]

    # 3. โหลดข้อมูลเพิ่มเติม (ถ้ามี)
    augmented_file = "data/output/augmented_samples.jsonl"
    augmented_data = []
    if os.path.exists(augmented_file):
        with open(augmented_file, encoding="utf-8") as f:
            augmented_data = [json.loads(line) for line in f if line.strip()]
        print(f"Loaded {len(augmented_data)} augmented examples")

    # 4. รวมข้อมูล
    all_data = primary_data + secondary_data + augmented_data
    print(f"Loaded {len(primary_data)} primary examples + {len(secondary_data)} secondary examples + {len(augmented_data)} augmented examples = {len(all_data)} total")

    return all_data

# --- สร้างชุดข้อมูลสำหรับเทรน ---
def prepare_data(all_data):
    train_examples = []
    for item in all_data:
        subject = item.get("subject")
        if not subject or subject not in subjects:
            continue

        # ดึงข้อความจาก text หรือรวม content + question
        if "text" in item and item["text"]:
            text = item["text"]
        else:
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

    # แบ่ง train/validation/test (60%/20%/20%)
    train_val_data, test_data = train_test_split(train_examples, test_size=0.2, random_state=42)
    train_data, eval_data = train_test_split(train_val_data, test_size=0.25, random_state=42)  # 0.25 x 0.8 = 0.2

    print(f"Train: {len(train_data)} examples, Validation: {len(eval_data)} examples, Test: {len(test_data)} examples")

    # สร้าง HuggingFace Dataset
    train_dataset = Dataset.from_pandas(pd.DataFrame(train_data))
    eval_dataset = Dataset.from_pandas(pd.DataFrame(eval_data))
    test_dataset = Dataset.from_pandas(pd.DataFrame(test_data))

    return train_dataset, eval_dataset, test_dataset

def compute_metrics(eval_pred):
    logits, labels = eval_pred
    predictions = np.argmax(logits, axis=-1)
    return {
        "accuracy": accuracy_score(labels, predictions),
        "f1_macro": f1_score(labels, predictions, average="macro")
    }

def model_init(trial=None):
    model = AutoModelForSequenceClassification.from_pretrained(
        MODEL_NAME, 
        num_labels=len(subjects),
        id2label=id2label,
        label2id=label2id
    )
    return model

def objective(trial):
    # กำหนดช่วง hyperparameters ที่จะลองปรับ
    learning_rate = trial.suggest_float("learning_rate", 1e-5, 5e-5, log=True)
    batch_size = trial.suggest_int("batch_size", 8, 32, step=8)
    weight_decay = trial.suggest_float("weight_decay", 0.01, 0.1)
    num_epochs = trial.suggest_int("num_epochs", 3, 10)
    warmup_ratio = trial.suggest_float("warmup_ratio", 0.0, 0.2)

    # แปลงเป็น TrainingArguments
    training_args = TrainingArguments(
        output_dir=f"{OUTPUT_DIR}/trial-{trial.number}",
        learning_rate=learning_rate,
        per_device_train_batch_size=batch_size,
        per_device_eval_batch_size=batch_size,
        num_train_epochs=num_epochs,
        weight_decay=weight_decay,
        warmup_ratio=warmup_ratio,
        eval_steps=100,
        save_steps=100,
        logging_steps=50,
        evaluation_strategy="steps",
        load_best_model_at_end=True,
        metric_for_best_model="f1_macro",
        greater_is_better=True,
        push_to_hub=False,
    )

    # สร้าง Trainer
    trainer = Trainer(
        model_init=model_init,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        tokenizer=tokenizer,
        data_collator=data_collator,
        compute_metrics=compute_metrics,
    )

    # เทรนโมเดล
    trainer.train()

    # ประเมินบน validation set
    eval_results = trainer.evaluate()
    
    # บันทึกผลลัพธ์ของแต่ละ trial
    with open(os.path.join(EVAL_DIR, f"trial_{trial.number}_results.json"), "w") as f:
        json.dump(eval_results, f)
    
    # คืนค่า f1_macro สำหรับการปรับ hyperparameter
    return eval_results["eval_f1_macro"]

# --- ฟังก์ชันหลัก ---
if __name__ == "__main__":
    # 1. โหลดข้อมูล
    all_data = load_data()
    
    # 2. เตรียมข้อมูล
    train_dataset, eval_dataset, test_dataset = prepare_data(all_data)
    
    # 3. โหลด tokenizer
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    
    # 4. Preprocess dataset
    def preprocess_function(examples):
        return tokenizer(
            examples["text"], 
            truncation=True, 
            padding="max_length",
            max_length=256  # ปรับให้ยาวขึ้นเผื่อตัวอย่างที่มีความซับซ้อน
        )

    train_dataset = train_dataset.map(preprocess_function, batched=True)
    eval_dataset = eval_dataset.map(preprocess_function, batched=True)
    test_dataset = test_dataset.map(preprocess_function, batched=True)
    
    # 5. สร้าง data collator
    data_collator = DataCollatorWithPadding(tokenizer=tokenizer)
    
    # 6. เริ่มกระบวนการปรับ hyperparameters ด้วย Optuna
    print("Starting hyperparameter tuning...")
    study = optuna.create_study(
        study_name=STUDY_NAME,
        direction="maximize",
        sampler=TPESampler(),
        pruner=MedianPruner()
    )
    study.optimize(objective, n_trials=N_TRIALS)
    
    # 7. แสดงผลลัพธ์ที่ดีที่สุด
    best_trial = study.best_trial
    print(f"Best trial (#{best_trial.number}): {best_trial.value}")
    print("Best hyperparameters:")
    for key, value in best_trial.params.items():
        print(f"  {key}: {value}")
    
    # 8. เทรนโมเดลสุดท้ายด้วยค่า hyperparameters ที่ดีที่สุด
    print("\nTraining final model with best hyperparameters...")
    training_args = TrainingArguments(
        output_dir=OUTPUT_DIR,
        learning_rate=best_trial.params["learning_rate"],
        per_device_train_batch_size=best_trial.params["batch_size"],
        per_device_eval_batch_size=best_trial.params["batch_size"],
        num_train_epochs=best_trial.params["num_epochs"],
        weight_decay=best_trial.params["weight_decay"],
        warmup_ratio=best_trial.params["warmup_ratio"],
        evaluation_strategy="steps",
        eval_steps=100,
        save_steps=100,
        load_best_model_at_end=True,
        metric_for_best_model="f1_macro",
        greater_is_better=True,
        push_to_hub=False,
    )
    
    # สร้าง model สุดท้าย
    model = model_init()
    
    # สร้าง trainer สุดท้าย
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        tokenizer=tokenizer,
        data_collator=data_collator,
        compute_metrics=compute_metrics,
    )
    
    # เทรนโมเดลสุดท้าย
    trainer.train()
    
    # บันทึกโมเดลสุดท้าย
    trainer.save_model(OUTPUT_DIR)
    tokenizer.save_pretrained(OUTPUT_DIR)
    
    # ประเมินผลบน test set
    test_results = trainer.evaluate(test_dataset)
    print(f"Final test results: {test_results}")
    
    # บันทึกผลการทดสอบ
    with open(os.path.join(EVAL_DIR, "final_test_results.json"), "w") as f:
        json.dump(test_results, f)
    
    print(f"Hyperparameter tuning complete! Best model saved to {OUTPUT_DIR}")
    print(f"Final test accuracy: {test_results.get('eval_accuracy', 0):.4f}")
    print(f"Final test F1 score: {test_results.get('eval_f1_macro', 0):.4f}")
