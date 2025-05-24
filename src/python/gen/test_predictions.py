from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch
import os

# โหลดโมเดลที่สร้างไว้
model_path = "data/output/thai-education-model"
tokenizer = AutoTokenizer.from_pretrained(model_path)
model = AutoModelForSequenceClassification.from_pretrained(model_path)

# สร้างฟังก์ชันทำนาย
def predict_subject(text):
    # ใช้ tokenizer
    inputs = tokenizer(text, return_tensors="pt", truncation=True, max_length=128)
    
    # ใช้โมเดลทำนาย
    with torch.no_grad():
        outputs = model(**inputs)
    
    # แปลงผลลัพธ์
    probs = torch.nn.functional.softmax(outputs.logits, dim=1)[0]
    prediction_id = torch.argmax(probs).item()
    prediction = model.config.id2label[str(prediction_id)]
    confidence = probs[prediction_id].item()
    
    return prediction, confidence

# รายการข้อความทดสอบ - เพิ่มเติมได้
test_texts = [
    # วิทยาศาสตร์
    "ไข่ -> ตัวหนอน -> ดักแด้ -> ผีเสื้อ",
    "ออกซิเจนเป็นสิ่งจำเป็นสำหรับการหายใจ",
    "โลกหมุนรอบตัวเองและโคจรรอบดวงอาทิตย์",
    
    # คณิตศาสตร์
    "การบวกเลขสองหลัก 25 + 13 = 38",
    "x + 5 = 12 หาค่า x",
    "เส้นขนานไม่มีวันตัดกัน",
    
    # สังคมศึกษา
    "พระเจ้าอยู่หัวรัชกาลที่ 9 เป็นที่รักของคนไทย",
    "กรุงเทพมหานครเป็นเมืองหลวงของประเทศไทย",
    "ประเทศในอาเซียนมีทั้งหมด 10 ประเทศ",
    
    # ภาษาไทย
    "การอ่านพยัญชนะไทย ก ไก่ ข ไข่",
    "วรรณคดีไทยเรื่องรามเกียรติ์",
    "การใช้คำราชาศัพท์ให้ถูกต้องตามกาละเทศะ",
    
    # ภาษาอังกฤษ
    "I can speak English very well",
    "The cat is on the table",
    "She went to the market yesterday",
    
    # ศิลปะ
    "วาดรูประบายสีและทัศนศิลป์",
    "การวาดภาพทิวทัศน์และภาพเหมือน",
    "ศิลปะแบบนามธรรมและศิลปะแบบเหมือนจริง",
    
    # สุขศึกษาและพลศึกษา
    "การออกกำลังกายทำให้ร่างกายแข็งแรง",
    "การรักษาสุขภาพและโภชนาการที่ดี",
    "กีฬาประเภททีมช่วยฝึกการทำงานร่วมกัน",
    
    # การงานอาชีพและเทคโนโลยี
    "การเขียนโปรแกรมคอมพิวเตอร์",
    "การทำอาหารไทยและขนมไทย",
    "การใช้เครื่องมือช่างและการซ่อมแซมอุปกรณ์"
]

# ทดสอบ
print("=" * 50)
print("ผลการทำนายวิชาจากข้อความ")
print("=" * 50)
correct = 0
total = 0

# วิชาที่คาดหวัง (ตามกลุ่มในรายการข้อความ)
expected_subjects = (
    ["วิทยาศาสตร์"] * 3 + 
    ["คณิตศาสตร์"] * 3 + 
    ["สังคมศึกษา"] * 3 + 
    ["ภาษาไทย"] * 3 + 
    ["ภาษาอังกฤษ"] * 3 + 
    ["ศิลปะ"] * 3 + 
    ["สุขศึกษาและพลศึกษา"] * 3 + 
    ["การงานอาชีพและเทคโนโลยี"] * 3
)

# ทดสอบแต่ละข้อความ
for i, text in enumerate(test_texts):
    prediction, confidence = predict_subject(text)
    expected = expected_subjects[i]
    is_correct = prediction == expected
    if is_correct:
        correct += 1
    total += 1
    
    print(f"ข้อความ: {text}")
    print(f"ทำนาย: {prediction} (ความมั่นใจ: {confidence:.2%})")
    print(f"คาดหวัง: {expected}")
    print(f"ผลลัพธ์: {'✓' if is_correct else '✗'}")
    print("-" * 50)

# สรุปผล
print("\nสรุปผล:")
print(f"ความแม่นยำ: {correct}/{total} คิดเป็น {correct/total:.2%}")
print("=" * 50)
