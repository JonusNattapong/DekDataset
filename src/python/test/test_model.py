from transformers import AutoTokenizer, AutoModelForSequenceClassification, CamembertConfig
import torch
import os
import json

# โหลดโมเดลที่เทรนเสร็จแล้ว
model_path = os.path.join('data', 'output', 'model-primary-secondary')
tokenizer = AutoTokenizer.from_pretrained('airesearch/wangchanberta-base-att-spm-uncased')

# ตรวจสอบว่ามีไฟล์ config.json หรือไม่
config_path = os.path.join(model_path, 'config.json')
if not os.path.exists(config_path):
    print(f"ไม่พบไฟล์ config.json ใน {config_path}")
    print("กำลังสร้าง config ใหม่...")
    # สร้าง config ใหม่
    config = CamembertConfig.from_pretrained('airesearch/wangchanberta-base-att-spm-uncased')
    config.num_labels = 8  # ปรับตามจำนวนวิชา
    config.id2label = {
        0: "ภาษาไทย", 
        1: "คณิตศาสตร์", 
        2: "วิทยาศาสตร์", 
        3: "สังคมศึกษา", 
        4: "ศิลปะ",
        5: "สุขศึกษาและพลศึกษา", 
        6: "การงานอาชีพและเทคโนโลยี", 
        7: "ภาษาอังกฤษ"
    }
    config.label2id = {v: k for k, v in config.id2label.items()}
    
    # บันทึก config
    with open(config_path, 'w', encoding='utf-8') as f:
        json.dump(config.to_dict(), f, ensure_ascii=False, indent=2)
    
    # โหลดโมเดลด้วย config ใหม่
    try:
        model = AutoModelForSequenceClassification.from_pretrained(model_path, config=config)
    except Exception as e:
        print(f"ไม่สามารถโหลดโมเดลได้: {e}")
        print("ลองใช้วิธีอื่น...")
        # ลอง load แบบอื่น
        model = AutoModelForSequenceClassification.from_pretrained(
            'airesearch/wangchanberta-base-att-spm-uncased',
            num_labels=8
        )
        # ลองโหลด state_dict
        state_dict_path = os.path.join(model_path, 'pytorch_model.bin')
        if os.path.exists(state_dict_path):
            model.load_state_dict(torch.load(state_dict_path, map_location='cpu'))
        else:
            print(f"ไม่พบไฟล์ pytorch_model.bin ใน {state_dict_path}")
else:
    # ถ้ามี config.json อยู่แล้ว ก็โหลดโมเดลตามปกติ
    try:
        model = AutoModelForSequenceClassification.from_pretrained(model_path)
    except Exception as e:
        print(f"ไม่สามารถโหลดโมเดลได้: {e}")
        print("ลองใช้วิธีอื่น...")
        config = CamembertConfig.from_json_file(config_path)
        model = AutoModelForSequenceClassification.from_pretrained(
            'airesearch/wangchanberta-base-att-spm-uncased', 
            config=config
        )

# รายการป้ายกำกับ (labels)
labels = list(model.config.id2label.values()) if hasattr(model.config, 'id2label') else [
    "ภาษาไทย", "คณิตศาสตร์", "วิทยาศาสตร์", "สังคมศึกษา", 
    "ศิลปะ", "สุขศึกษาและพลศึกษา", "การงานอาชีพและเทคโนโลยี", "ภาษาอังกฤษ"
]
print(f"รายการป้ายกำกับ (labels): {labels}")

# สร้างฟังก์ชันทำนาย
def predict(text):
    inputs = tokenizer(text, return_tensors='pt', truncation=True)
    with torch.no_grad():
        outputs = model(**inputs)
    probs = torch.nn.functional.softmax(outputs.logits, dim=1)[0]
    prediction = labels[torch.argmax(probs).item()]
    confidence = probs[torch.argmax(probs)].item()
    return prediction, confidence

# ทดสอบกับข้อความใหม่
examples = [
    'ไข่ -> ตัวหนอน -> ดักแด้ -> ผีเสื้อ',
    'ฝนตกหนักท้องฟ้าร้องคำรามฟ้าผ่าลงมา',
    'x + 5 = 12',
    'พระเจ้าอยู่หัวรัชกาลที่ 9',
    'I can speak English very well',
    'วาดรูประบายสี',
    'การออกกำลังกายทำให้ร่างกายแข็งแรง',
    'การเขียนโปรแกรมคอมพิวเตอร์'
]

for text in examples:
    pred, conf = predict(text)
    print(f'ข้อความ: {text}')
    print(f'ทำนาย: {pred} (ความมั่นใจ: {conf:.2%})\n')
