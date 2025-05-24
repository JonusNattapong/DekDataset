from fastapi import FastAPI, HTTPException, Request, Form, Depends
from fastapi.templating import Jinja2Templates
from fastapi.staticfiles import StaticFiles
from fastapi.responses import HTMLResponse, JSONResponse
from pydantic import BaseModel
from typing import List, Dict, Any, Optional
import torch
import os
import json
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import numpy as np
import uvicorn
from datetime import datetime

# --- CONFIG ---
MODEL_DIR = "data/output/thai-education-model"  # หรือใช้ "data/output/thai-education-model-tuned" ถ้ามีโมเดลที่ปรับแล้ว
HISTORY_DIR = "data/output/prediction-history"
os.makedirs(HISTORY_DIR, exist_ok=True)

app = FastAPI(title="Thai Education Subject Classifier API", 
              description="API สำหรับทำนายวิชาจากข้อความภาษาไทย",
              version="1.0.0")

# สร้างโฟลเดอร์สำหรับไฟล์ static
os.makedirs("static", exist_ok=True)
os.makedirs("templates", exist_ok=True)

# สร้างไฟล์ CSS
css_content = """
body {
    font-family: 'Sarabun', 'Noto Sans Thai', sans-serif;
    margin: 0;
    padding: 0;
    background-color: #f5f5f5;
    color: #333;
}
.container {
    max-width: 1000px;
    margin: 0 auto;
    padding: 20px;
}
.header {
    background-color: #0066cc;
    color: white;
    padding: 20px;
    text-align: center;
    box-shadow: 0 2px 4px rgba(0,0,0,0.1);
}
.card {
    background-color: white;
    border-radius: 8px;
    padding: 20px;
    margin-bottom: 20px;
    box-shadow: 0 2px 4px rgba(0,0,0,0.1);
}
.form-group {
    margin-bottom: 15px;
}
textarea {
    width: 100%;
    padding: 10px;
    border: 1px solid #ddd;
    border-radius: 4px;
    font-family: 'Sarabun', 'Noto Sans Thai', sans-serif;
    font-size: 16px;
    resize: vertical;
    min-height: 100px;
}
button {
    background-color: #0066cc;
    color: white;
    border: none;
    padding: 10px 20px;
    border-radius: 4px;
    cursor: pointer;
    font-size: 16px;
}
button:hover {
    background-color: #0055aa;
}
.result {
    border-left: 5px solid #0066cc;
    padding-left: 15px;
    margin-top: 20px;
}
.confidence-bar {
    height: 20px;
    background-color: #e9ecef;
    border-radius: 4px;
    margin-top: 5px;
    margin-bottom: 15px;
    overflow: hidden;
}
.confidence-bar-fill {
    height: 100%;
    background-color: #0066cc;
    border-radius: 4px;
}
.subject-list {
    display: flex;
    flex-wrap: wrap;
    gap: 10px;
    margin-top: 20px;
}
.subject-tag {
    background-color: #e9ecef;
    padding: 5px 10px;
    border-radius: 20px;
    font-size: 14px;
}
.footer {
    text-align: center;
    padding: 20px;
    color: #666;
    font-size: 14px;
}
.history-item {
    padding: 10px;
    border-bottom: 1px solid #eee;
}
.history-header {
    display: flex;
    justify-content: space-between;
    font-size: 0.9em;
    color: #666;
}
"""

# บันทึกไฟล์ CSS
with open("static/style.css", "w", encoding="utf-8") as f:
    f.write(css_content)

# สร้างไฟล์ HTML สำหรับหน้าหลัก
index_html = """
<!DOCTYPE html>
<html lang="th">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>ระบบทำนายวิชาจากข้อความ</title>
    <link rel="stylesheet" href="https://fonts.googleapis.com/css2?family=Sarabun:wght@400;700&display=swap">
    <link rel="stylesheet" href="/static/style.css">
</head>
<body>
    <div class="header">
        <h1>ระบบทำนายวิชาจากข้อความ</h1>
        <p>สำหรับการศึกษาระดับประถมศึกษาและมัธยมศึกษาในประเทศไทย</p>
    </div>
    <div class="container">
        <div class="card">
            <h2>ทำนายวิชา</h2>
            <form id="prediction-form" method="post" action="/predict">
                <div class="form-group">
                    <label for="text">กรอกข้อความ (เช่น คำถาม เนื้อหาบทเรียน หรือโจทย์):</label>
                    <textarea id="text" name="text" required>{{ text }}</textarea>
                </div>
                <button type="submit">ทำนาย</button>
            </form>
        </div>
        
        {% if prediction %}
        <div class="card">
            <h2>ผลการทำนาย</h2>
            <div class="result">
                <h3>วิชา: {{ prediction }}</h3>
                <p>ความมั่นใจ: {{ confidence }}%</p>
                <div class="confidence-bar">
                    <div class="confidence-bar-fill" style="width: {{ confidence }}%"></div>
                </div>
                
                <h4>ความน่าจะเป็นในแต่ละวิชา</h4>
                <ul>
                {% for subject, prob in all_probabilities %}
                    <li>{{ subject }}: {{ prob }}%</li>
                {% endfor %}
                </ul>
            </div>
        </div>
        {% endif %}

        <div class="card">
            <h2>ประวัติการทำนายล่าสุด</h2>
            {% if history %}
                {% for item in history %}
                <div class="history-item">
                    <div class="history-header">
                        <span>เวลา: {{ item.timestamp }}</span>
                        <span>วิชา: <strong>{{ item.prediction }}</strong> ({{ item.confidence }}%)</span>
                    </div>
                    <p>{{ item.text|truncate(100) }}</p>
                </div>
                {% endfor %}
            {% else %}
                <p>ยังไม่มีประวัติการทำนาย</p>
            {% endif %}
        </div>
    </div>
    
    <div class="footer">
        <p>ระบบทำนายวิชาจากข้อความภาษาไทย ใช้โมเดล WangchanBERTa</p>
        <p>© 2023 DekDataset Project</p>
    </div>
</body>
</html>
"""

# บันทึกไฟล์ HTML
with open("templates/index.html", "w", encoding="utf-8") as f:
    f.write(index_html)

# ตั้งค่า static files และ templates
app.mount("/static", StaticFiles(directory="static"), name="static")
templates = Jinja2Templates(directory="templates")

# --- โหลดโมเดล ---
print(f"กำลังโหลดโมเดลจาก {MODEL_DIR}...")
try:
    tokenizer = AutoTokenizer.from_pretrained(MODEL_DIR)
    model = AutoModelForSequenceClassification.from_pretrained(MODEL_DIR)
    subjects = list(model.config.id2label.values())
    print(f"โหลดโมเดลสำเร็จ มีวิชาทั้งหมด {len(subjects)} วิชา: {subjects}")
except Exception as e:
    print(f"ไม่สามารถโหลดโมเดลได้: {e}")
    # ลองโหลดจาก WangchanBERTa แทน
    print("กำลังโหลดโมเดลจาก WangchanBERTa แทน...")
    tokenizer = AutoTokenizer.from_pretrained("airesearch/wangchanberta-base-att-spm-uncased")
    model = AutoModelForSequenceClassification.from_pretrained(
        "airesearch/wangchanberta-base-att-spm-uncased", 
        num_labels=8
    )
    subjects = [
        "ภาษาไทย", "คณิตศาสตร์", "วิทยาศาสตร์", "สังคมศึกษา", 
        "ศิลปะ", "สุขศึกษาและพลศึกษา", "การงานอาชีพและเทคโนโลยี", "ภาษาอังกฤษ"
    ]
    print(f"โหลดโมเดล WangchanBERTa สำเร็จ พร้อมวิชาพื้นฐาน: {subjects}")

# ประวัติการทำนาย
prediction_history = []
max_history_items = 10

# --- Pydantic Models ---
class PredictionRequest(BaseModel):
    text: str

class PredictionResponse(BaseModel):
    subject: str
    confidence: float
    all_probabilities: Dict[str, float]
    text: str

# --- ฟังก์ชันทำนาย ---
def predict_subject(text):
    """ทำนายวิชาจากข้อความ"""
    inputs = tokenizer(text, return_tensors="pt", truncation=True, max_length=256)
    with torch.no_grad():
        outputs = model(**inputs)
    
    logits = outputs.logits
    probabilities = torch.nn.functional.softmax(logits, dim=-1)[0]
    prediction_idx = torch.argmax(probabilities).item()
    prediction = subjects[prediction_idx]
    confidence = probabilities[prediction_idx].item()
    
    # สร้างความน่าจะเป็นของทุกวิชา
    all_probs = {subjects[i]: prob.item() for i, prob in enumerate(probabilities)}
    
    return prediction, confidence, all_probs

# --- บันทึกประวัติ ---
def save_prediction_history():
    with open(os.path.join(HISTORY_DIR, "prediction_history.json"), "w", encoding="utf-8") as f:
        json.dump(prediction_history, f, ensure_ascii=False, indent=2)

# --- โหลดประวัติ ---
def load_prediction_history():
    global prediction_history
    history_file = os.path.join(HISTORY_DIR, "prediction_history.json")
    if os.path.exists(history_file):
        with open(history_file, encoding="utf-8") as f:
            prediction_history = json.load(f)

# โหลดประวัติเมื่อเริ่มต้น
load_prediction_history()

# --- API Routes ---
@app.get("/", response_class=HTMLResponse)
async def read_root(request: Request):
    """หน้าหลัก"""
    return templates.TemplateResponse("index.html", {
        "request": request, 
        "prediction": None,
        "text": "",
        "history": prediction_history[:10]
    })

@app.post("/predict", response_class=HTMLResponse)
async def predict_form(request: Request, text: str = Form(...)):
    """รับข้อมูลจากฟอร์มและแสดงผลลัพธ์"""
    if not text or text.strip() == "":
        raise HTTPException(status_code=400, detail="กรุณากรอกข้อความ")
    
    prediction, confidence, all_probs = predict_subject(text)
    
    # เพิ่มลงในประวัติ
    prediction_entry = {
        "text": text,
        "prediction": prediction,
        "confidence": f"{confidence*100:.2f}",
        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    }
    prediction_history.insert(0, prediction_entry)
    if len(prediction_history) > max_history_items:
        prediction_history.pop()
    save_prediction_history()
    
    # เตรียมข้อมูลสำหรับแสดงผล
    all_probabilities = [(subject, f"{prob*100:.2f}") for subject, prob in 
                         sorted(all_probs.items(), key=lambda x: x[1], reverse=True)]
    
    return templates.TemplateResponse("index.html", {
        "request": request,
        "text": text,
        "prediction": prediction,
        "confidence": f"{confidence*100:.2f}",
        "all_probabilities": all_probabilities,
        "history": prediction_history[:10]
    })

@app.post("/api/predict", response_model=PredictionResponse)
async def predict_api(request: PredictionRequest):
    """API สำหรับทำนายวิชา"""
    text = request.text
    if not text or text.strip() == "":
        raise HTTPException(status_code=400, detail="กรุณากรอกข้อความ")
    
    prediction, confidence, all_probs = predict_subject(text)
    
    # เพิ่มลงในประวัติ
    prediction_entry = {
        "text": text,
        "prediction": prediction,
        "confidence": f"{confidence*100:.2f}",
        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    }
    prediction_history.insert(0, prediction_entry)
    if len(prediction_history) > max_history_items:
        prediction_history.pop()
    save_prediction_history()
    
    return PredictionResponse(
        subject=prediction,
        confidence=confidence,
        all_probabilities=all_probs,
        text=text
    )

@app.get("/api/subjects", response_model=List[str])
async def get_subjects():
    """API สำหรับดึงรายการวิชาทั้งหมด"""
    return subjects

@app.get("/api/history", response_model=List[Dict[str, Any]])
async def get_history():
    """API สำหรับดึงประวัติการทำนาย"""
    return prediction_history

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
