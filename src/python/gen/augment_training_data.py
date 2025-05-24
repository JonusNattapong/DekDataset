import json
import random
import nltk
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch

# โหลด tokenizer และ model ที่เทรนไว้แล้ว
model_dir = "data/output/thai-education-model"
tokenizer = AutoTokenizer.from_pretrained(model_dir)
model = AutoModelForSequenceClassification.from_pretrained(model_dir)

# กำหนดวิชา (subjects)
subjects = [
    "ภาษาไทย", "คณิตศาสตร์", "วิทยาศาสตร์", "สังคมศึกษา", 
    "ศิลปะ", "สุขศึกษาและพลศึกษา", "การงานอาชีพและเทคโนโลยี", "ภาษาอังกฤษ"
]

# ตัวอย่างข้อมูลต้นแบบในแต่ละวิชา
subject_examples = {
    "ภาษาไทย": [
        "การอ่านพยัญชนะไทย ก ไก่ ข ไข่",
        "คำที่มีตัวสะกดในมาตราแม่ กง ได้แก่ จง ปลง",
        "อักษรสูงได้แก่ ข ฃ ฉ ฐ ถ ผ ฝ ศ ษ ส ห",
        "อักษรต่ำได้แก่ ค ฅ ฆ ช ซ ฌ ฑ ฒ ณ",
        "คำนามคือคำที่ใช้เรียกชื่อคน สัตว์ สิ่งของและสถานที่",
        "การอ่านจับใจความสำคัญ"
    ],
    "คณิตศาสตร์": [
        "การบวกเลขสองหลัก 25 + 13 = 38",
        "สูตรพีทาโกรัสคือ a² + b² = c²",
        "แก้สมการ x + 5 = 12",
        "พื้นที่วงกลม = πr²",
        "ถ้า 2x + 3 = 11 แล้ว x = 4",
        "ค่า sin 30° = 0.5"
    ],
    "วิทยาศาสตร์": [
        "ไข่ -> ตัวหนอน -> ดักแด้ -> ผีเสื้อ",
        "DNA เป็นสารพันธุกรรมที่พบในนิวเคลียสของเซลล์",
        "พลังงานไม่สามารถสร้างขึ้นใหม่หรือทำให้สูญหายได้",
        "น้ำระเหยกลายเป็นไอเมื่อได้รับความร้อน",
        "ไมโทคอนเดรียในเซลล์มีหน้าที่สร้างพลังงาน",
        "แรงโน้มถ่วงของโลกเท่ากับ 9.8 m/s²"
    ],
    "สังคมศึกษา": [
        "พระเจ้าอยู่หัวรัชกาลที่ 9 เป็นที่รักของคนไทย",
        "วันสำคัญของหมู่บ้านคือวันสงกรานต์",
        "การปกครองระบอบประชาธิปไตยอันมีพระมหากษัตริย์ทรงเป็นประมุข",
        "วันที่ 5 ธันวาคม เป็นวันเฉลิมพระชนมพรรษาของในหลวงรัชกาลที่ 9",
        "รายได้เป็นปัจจัยที่มีผลต่ออุปสงค์",
        "ASEAN ก่อตั้งขึ้นเมื่อวันที่ 8 สิงหาคม 2510"
    ],
    "ศิลปะ": [
        "วาดรูประบายสีและทัศนศิลป์",
        "ศิลปะสมัยอยุธยา",
        "ทฤษฎีสีและการผสมสี สีขั้นที่ 1 คือ แดง เหลือง น้ำเงิน",
        "การวาดภาพทิวทัศน์",
        "จุด เส้น รูปร่าง รูปทรงในงานศิลปะ",
        "การร้องเพลงที่ถูกต้อง"
    ],
    "สุขศึกษาและพลศึกษา": [
        "การออกกำลังกายทำให้ร่างกายแข็งแรง",
        "กติกากีฬาฟุตบอล",
        "การปฐมพยาบาลเบื้องต้น",
        "อาหารห้าหมู่ที่จำเป็นต่อร่างกาย",
        "การเล่นกีฬาช่วยลดความเสี่ยงของโรคหัวใจ",
        "การเติบโตและพัฒนาการของวัยรุ่น"
    ],
    "การงานอาชีพและเทคโนโลยี": [
        "การเขียนโปรแกรมคอมพิวเตอร์",
        "การใช้โปรแกรม Microsoft Office",
        "การปลูกพืชผักสวนครัว",
        "การซ่อมแซมเสื้อผ้า",
        "การประกอบอาหาร",
        "การดูแลรักษาอุปกรณ์อิเล็กทรอนิกส์"
    ],
    "ภาษาอังกฤษ": [
        "I can speak English very well",
        "What is the English word for 'พ่อ'?",
        "Present Perfect Tense: I have eaten",
        "Past Simple Tense: I ate",
        "Modal verbs: can, could, may, might, shall, should",
        "Comparative: bigger, better, more beautiful"
    ]
}

# สร้างข้อมูลเพิ่มด้วยการผสมและแปรผันตัวอย่าง
def generate_more_samples():
    augmented_data = []
    for subject, examples in subject_examples.items():
        # สร้างตัวอย่างเพิ่มจากตัวอย่างที่มีอยู่
        for _ in range(10):  # สร้างเพิ่ม 10 ตัวอย่างต่อวิชา
            # เลือกตัวอย่างแบบสุ่ม 1-3 ตัวอย่าง
            selected = random.sample(examples, random.randint(1, min(3, len(examples))))
            # รวมตัวอย่างเป็นข้อความเดียว
            text = " ".join(selected)
            augmented_data.append({
                "text": text,
                "subject": subject
            })
            
    return augmented_data

# สร้างข้อมูลตัวอย่างเพิ่ม
augmented_samples = generate_more_samples()
print(f"สร้างตัวอย่างเพิ่มได้ {len(augmented_samples)} ตัวอย่าง")

# บันทึกข้อมูลตัวอย่างเพิ่ม
output_file = "data/output/augmented_samples.jsonl"
with open(output_file, "w", encoding="utf-8") as f:
    for sample in augmented_samples:
        f.write(json.dumps(sample, ensure_ascii=False) + "\n")

print(f"บันทึกตัวอย่างลงในไฟล์ {output_file} เรียบร้อยแล้ว")
print("ตัวอย่าง 5 รายการแรก:")
for i, sample in enumerate(augmented_samples[:5]):
    print(f"{i+1}. {sample['subject']}: {sample['text'][:50]}...")
