# การใช้โมเดล Qwen/Qwen3-0.6B ในโปรเจค DekDataset

## เกี่ยวกับโมเดล Qwen/Qwen3-0.6B

Qwen/Qwen3-0.6B เป็นโมเดลภาษาขนาดเล็กที่พัฒนาโดย Alibaba Cloud มีความสามารถในการเข้าใจและประมวลผลภาษาธรรมชาติได้หลากหลายภาษา รวมถึงภาษาไทย โดยในโปรเจคนี้เราได้นำมาปรับแต่ง (fine-tune) เพื่อจำแนกข้อความภาษาไทยตามกลุ่มสาระการเรียนรู้

## ข้อดีของโมเดล Qwen/Qwen3-0.6B

1. **ขนาดเล็ก**: ขนาด 0.6B พารามิเตอร์ ทำให้ใช้ทรัพยากรน้อยกว่าโมเดลขนาดใหญ่
2. **รองรับหลายภาษา**: รองรับภาษาไทยและภาษาอื่นๆ เช่น ภาษาอังกฤษ ภาษาจีน
3. **ประสิทธิภาพดี**: ให้ผลลัพธ์ที่ดีแม้จะมีขนาดเล็ก

## การเตรียมพร้อมก่อนการเทรน

### ความต้องการของระบบ

1. **ฮาร์ดแวร์**:
   - CPU: อย่างน้อย 4 cores
   - RAM: อย่างน้อย 8GB
   - GPU (แนะนำ): NVIDIA GPU ที่มีหน่วยความจำอย่างน้อย 4GB
   - พื้นที่ดิสก์: อย่างน้อย 5GB สำหรับโมเดลและข้อมูล

2. **ซอฟต์แวร์**:
   - Python 3.8+
   - PyTorch 1.10+
   - Transformers 4.25+
   - accelerate
   - datasets
   - scikit-learn

### การติดตั้ง Dependencies

```bash
pip install -r src/python/requirements-train.txt
```

## การเทรนโมเดล

### การเทรนด้วย Default Parameters

การเทรนโมเดลด้วยค่า default ทำได้ง่าย:

```bash
python src/python/train_qwen_model.py
```

### การปรับแต่งพารามิเตอร์

หากต้องการปรับแต่งพารามิเตอร์ต่างๆ สามารถแก้ไขไฟล์ `train_qwen_model.py` ในส่วนของ `TrainingArguments`:

```python
training_args = TrainingArguments(
    output_dir=OUTPUT_DIR,
    learning_rate=5e-5,           # อัตราการเรียนรู้ (ปรับได้ระหว่าง 1e-5 ถึง 1e-4)
    per_device_train_batch_size=4, # ขนาด batch (ปรับตามหน่วยความจำ GPU)
    per_device_eval_batch_size=4,
    num_train_epochs=5,           # จำนวนรอบการเทรน (ปรับได้ระหว่าง 3 ถึง 10)
    weight_decay=0.01,            # ค่า weight decay สำหรับป้องกัน overfitting
    eval_steps=10,                # ความถี่ในการประเมินผล
    save_steps=10,                # ความถี่ในการบันทึกโมเดล
    load_best_model_at_end=True,  # โหลดโมเดลที่ดีที่สุดในตอนจบ
    push_to_hub=False,            # ไม่ส่งโมเดลขึ้น HuggingFace Hub
    logging_dir=f"{OUTPUT_DIR}/logs",
    logging_steps=5,              # ความถี่ในการบันทึกล็อก
    report_to=["tensorboard"]     # รายงานผลไปยัง TensorBoard
)
```

## การตรวจสอบผลการเทรน

### บันทึกการเทรน

ระหว่างการเทรน ระบบจะบันทึกค่าต่างๆ ใน TensorBoard ซึ่งสามารถดูได้โดย:

```bash
tensorboard --logdir=data/output/thai-education-qwen-model/logs
```

### การประเมินผลหลังเทรน

หลังจากเทรนเสร็จ ระบบจะบันทึกผลการประเมินใน:
- `data/output/evaluation-qwen-results/qwen_eval_results.json`

ซึ่งประกอบด้วยค่า:
1. **accuracy**: ความแม่นยำในการทำนายวิชา
2. **f1_macro**: ค่าเฉลี่ย F1 score ของทุกวิชาโดยให้น้ำหนักเท่ากัน
3. **f1_weighted**: ค่าเฉลี่ย F1 score โดยให้น้ำหนักตามจำนวนตัวอย่างในแต่ละวิชา

## การใช้งานโมเดลหลังเทรน

### การโหลดโมเดลสำหรับการใช้งาน

```python
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch

# โหลดโมเดลและ tokenizer
model_path = "data/output/thai-education-qwen-model"
tokenizer = AutoTokenizer.from_pretrained(model_path)
model = AutoModelForSequenceClassification.from_pretrained(model_path)

# ฟังก์ชันสำหรับทำนาย
def predict_subject(text):
    inputs = tokenizer(text, return_tensors="pt", truncation=True)
    with torch.no_grad():
        outputs = model(**inputs)
    
    probs = torch.nn.functional.softmax(outputs.logits, dim=1)[0]
    predicted_class_id = torch.argmax(probs).item()
    
    # แปลง class id เป็นชื่อวิชา
    subjects = model.config.id2label
    predicted_subject = subjects[predicted_class_id]
    confidence = probs[predicted_class_id].item()
    
    return predicted_subject, confidence
```

### API สำหรับการทำนาย

เราได้พัฒนา API สำหรับการทำนายวิชาจากข้อความ ซึ่งสามารถเริ่มต้นได้โดย:

```bash
# สำหรับ Linux/Mac
bash src/python/start_api_server.sh

# สำหรับ Windows
.\src\python\start_api_server.bat
```

### GUI สำหรับการทำนาย

สำหรับผู้ใช้ทั่วไปที่ไม่คุ้นเคยกับการเขียนโค้ด เราได้พัฒนา GUI อย่างง่ายสำหรับการทำนายวิชาจากข้อความ:

```bash
# สำหรับ Linux/Mac
bash src/python/start_gui.sh

# สำหรับ Windows
.\src\python\start_gui.bat
```

## เทคนิคการเพิ่มประสิทธิภาพโมเดล

1. **เพิ่มข้อมูลเทรน**:
   - ใช้สคริปต์ `data_augmentation.py` เพื่อเพิ่มข้อมูลด้วยเทคนิคต่างๆ
   - รวบรวมตัวอย่างข้อความจากหนังสือเรียนหรือข้อสอบเพิ่มเติม

2. **Hyperparameter Tuning**:
   - ปรับค่า learning rate (ลองระหว่าง 1e-5 ถึง 1e-4)
   - ปรับค่า batch size ตามความเหมาะสมของฮาร์ดแวร์
   - ปรับจำนวนรอบการเทรน (epochs)
   - ลองใช้เทคนิค learning rate scheduler

3. **การปรับปรุงคุณภาพข้อมูล**:
   - ตรวจสอบและแก้ไขข้อมูลที่อาจมีความไม่ถูกต้อง
   - เพิ่มความหลากหลายของตัวอย่างในแต่ละวิชา
   - ปรับสมดุลของจำนวนตัวอย่างในแต่ละวิชา

## ตัวอย่างการตรวจสอบโมเดล

```python
# ตัวอย่างข้อความสำหรับทดสอบ
example_texts = [
    "ไข่ -> ตัวหนอน -> ดักแด้ -> ผีเสื้อ",  # วิทยาศาสตร์
    "การบวกเลขสองหลัก 25 + 13 = 38",  # คณิตศาสตร์
    "พระเจ้าอยู่หัวรัชกาลที่ 9 เป็นที่รักของคนไทย", # สังคมศึกษา
    "การอ่านพยัญชนะไทย ก ไก่ ข ไข่",  # ภาษาไทย
    "I can speak English very well", # ภาษาอังกฤษ
]

# ทำนายและแสดงผล
for text in example_texts:
    subject, confidence = predict_subject(text)
    print(f"ข้อความ: {text}")
    print(f"ทำนาย: {subject} (ความมั่นใจ: {confidence:.2%})")
    print("---")
```
