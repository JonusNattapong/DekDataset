#!/usr/bin/env python
# coding: utf-8
# subject_prediction_api.py - API สำหรับทำนายวิชาจากข้อความโดยใช้โมเดล Qwen หรือ WangchanBERTa ที่เทรนแล้ว

import os
import json
import torch
from fastapi import FastAPI, Request, HTTPException
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import uvicorn

# ตั้งค่าต่างๆ
MODEL_PATHS = {
    "qwen": "data/output/thai-education-qwen-model",
    "wangchanberta": "data/output/thai-education-model"
}

DEFAULT_MODEL = "qwen"  # หรือ "wangchanberta"

# กำหนดวิชาที่สามารถทำนายได้
SUBJECTS = [
    "ภาษาไทย", "คณิตศาสตร์", "วิทยาศาสตร์", "สังคมศึกษา", 
    "ศิลปะ", "สุขศึกษาและพลศึกษา", "การงานอาชีพและเทคโนโลยี", "ภาษาอังกฤษ",
    "ดนตรี", "เทคโนโลยี", "ภาษาจีน", "ประวัติศาสตร์", "ภูมิศาสตร์"
]

# ฟังก์ชันสำหรับโหลดโมเดล
def load_model(model_name):
    """โหลดโมเดลตามชื่อที่ระบุ"""
    if model_name not in MODEL_PATHS:
        raise ValueError(f"ไม่พบโมเดล {model_name} (ที่รองรับ: {list(MODEL_PATHS.keys())})")
    
    model_path = MODEL_PATHS[model_name]
    
    # ตรวจสอบว่ามีโฟลเดอร์โมเดลหรือไม่
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"ไม่พบโฟลเดอร์โมเดล {model_path}")
    
    try:
        # โหลด tokenizer
        if model_name == "qwen":
            tokenizer = AutoTokenizer.from_pretrained(model_path)
            if tokenizer.pad_token is None:
                tokenizer.pad_token = tokenizer.eos_token
        else:
            tokenizer = AutoTokenizer.from_pretrained(model_path)
            
        # โหลดโมเดล
        model = AutoModelForSequenceClassification.from_pretrained(model_path)
        
        # ตรวจสอบ labels
        id2label = model.config.id2label if hasattr(model.config, "id2label") else {i: SUBJECTS[i] for i in range(len(SUBJECTS))}
        
        return {
            "tokenizer": tokenizer, 
            "model": model,
            "id2label": id2label
        }
    except Exception as e:
        raise RuntimeError(f"เกิดข้อผิดพลาดในการโหลดโมเดล {model_name}: {str(e)}")

# สร้าง FastAPI app
app = FastAPI(
    title="DekDataset Thai Subject Classifier API",
    description="API สำหรับจำแนกข้อความภาษาไทยตามกลุ่มสาระการเรียนรู้",
    version="1.0.0",
)

# เพิ่ม CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # สำหรับการพัฒนา ในการใช้งานจริงควรระบุ origins ที่อนุญาตเท่านั้น
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# กำหนด Model สำหรับ request และ response
class PredictionRequest(BaseModel):
    text: str
    model_name: str = DEFAULT_MODEL

class PredictionResponse(BaseModel):
    subject: str
    confidence: float
    all_predictions: dict

# โหลดโมเดลตอนเริ่มต้น
model_cache = {}

@app.on_event("startup")
async def startup_event():
    """โหลดโมเดลเมื่อเริ่มต้น API"""
    try:
        for model_name in MODEL_PATHS:
            print(f"กำลังโหลดโมเดล {model_name}...")
            try:
                model_cache[model_name] = load_model(model_name)
                print(f"โหลดโมเดล {model_name} สำเร็จ")
            except Exception as e:
                print(f"ไม่สามารถโหลดโมเดล {model_name} ได้: {e}")
    except Exception as e:
        print(f"เกิดข้อผิดพลาดในการโหลดโมเดล: {e}")

@app.post("/predict", response_model=PredictionResponse)
async def predict(request: PredictionRequest):
    """ทำนายวิชาจากข้อความภาษาไทย"""
    if not request.text:
        raise HTTPException(status_code=400, detail="กรุณาระบุข้อความที่ต้องการทำนาย")
    
    if request.model_name not in model_cache:
        # ลองโหลดโมเดลอีกครั้ง
        try:
            model_cache[request.model_name] = load_model(request.model_name)
        except Exception as e:
            raise HTTPException(status_code=400, detail=f"ไม่พบโมเดล {request.model_name} หรือไม่สามารถโหลดได้: {str(e)}")
    
    # ดึงโมเดลจาก cache
    model_data = model_cache[request.model_name]
    tokenizer = model_data["tokenizer"]
    model = model_data["model"]
    id2label = model_data["id2label"]
    
    # เตรียม input สำหรับโมเดล
    inputs = tokenizer(request.text, return_tensors="pt", truncation=True, padding=True)
    
    # ทำนาย
    try:
        with torch.no_grad():
            outputs = model(**inputs)
        
        # แปลงเป็นความน่าจะเป็น
        probabilities = torch.nn.functional.softmax(outputs.logits, dim=-1)[0]
        predicted_class_id = torch.argmax(probabilities).item()
        
        # สร้าง response
        all_predictions = {id2label[i]: float(probabilities[i]) for i in range(len(probabilities))}
        
        return {
            "subject": id2label[predicted_class_id],
            "confidence": float(probabilities[predicted_class_id]),
            "all_predictions": all_predictions
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"เกิดข้อผิดพลาดในการทำนาย: {str(e)}")

@app.get("/subjects")
async def get_subjects():
    """แสดงรายการวิชาที่โมเดลสามารถทำนายได้"""
    return {"subjects": SUBJECTS}

@app.get("/models")
async def get_models():
    """แสดงรายการโมเดลที่รองรับ"""
    return {
        "available_models": list(MODEL_PATHS.keys()),
        "default_model": DEFAULT_MODEL,
        "loaded_models": list(model_cache.keys())
    }

@app.get("/")
async def root():
    """หน้าหลักของ API"""
    return {
        "message": "DekDataset Thai Subject Classifier API",
        "docs": "/docs",
        "endpoints": [
            {"method": "POST", "path": "/predict", "description": "ทำนายวิชาจากข้อความ"},
            {"method": "GET", "path": "/subjects", "description": "แสดงรายการวิชาที่โมเดลสามารถทำนายได้"},
            {"method": "GET", "path": "/models", "description": "แสดงรายการโมเดลที่รองรับ"}
        ]
    }

if __name__ == "__main__":
    uvicorn.run("subject_prediction_api:app", host="0.0.0.0", port=8000, reload=True)
