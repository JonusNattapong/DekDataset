# Thai Education Subject Classifier

ระบบจำแนกประเภทวิชาจากข้อความภาษาไทย สำหรับข้อมูลการศึกษาไทย ระดับประถมศึกษา (ป.1-ป.6) และมัธยมศึกษา (ม.1-ม.6)

## 📝 สารบัญ

- [คุณสมบัติของระบบ](#คุณสมบัติของระบบ)
- [การติดตั้ง](#การติดตั้ง)
- [การใช้งาน](#การใช้งาน)
- [โครงสร้างไฟล์](#โครงสร้างไฟล์)
- [API Documentation](#api-documentation)
- [ผลการทดสอบและการพัฒนา](#ผลการทดสอบและการพัฒนา)

## 📚 คุณสมบัติของระบบ

ระบบจำแนกประเภทวิชาจากข้อความภาษาไทย มีความสามารถดังนี้:

- **ทำนายวิชา** จากข้อความภาษาไทย เช่น คำถาม บทเรียน โจทย์ โดยรองรับ 8 กลุ่มสาระการเรียนรู้:
  - ภาษาไทย
  - คณิตศาสตร์
  - วิทยาศาสตร์
  - สังคมศึกษา
  - ศิลปะ
  - สุขศึกษาและพลศึกษา
  - การงานอาชีพและเทคโนโลยี
  - ภาษาอังกฤษ

- **ประมวลผลภาษาไทย** ด้วยโมเดล WangchanBERTa
- **Web Dashboard** และ **API** สำหรับนำไปประยุกต์ใช้งาน
- **เทคนิคการเพิ่มข้อมูล** (Data Augmentation) เพื่อเพิ่มประสิทธิภาพของโมเดล
- **การปรับแต่ง Hyperparameters** เพื่อหาค่าที่เหมาะสมที่สุด

## 🔧 การติดตั้ง

### ความต้องการของระบบ

- Python 3.8 หรือใหม่กว่า
- CUDA (หากต้องการใช้ GPU)

### ขั้นตอนการติดตั้ง

1. **โคลนโปรเจค**:

```bash
git clone <repository-url>
cd DekDataset
```

2. **ติดตั้ง Dependencies**:

```bash
pip install -r requirements.txt
```

## 🚀 การใช้งาน

### 1. เทรนโมเดลพื้นฐาน

```bash
python src/python/train_primary_secondary.py
```

### 2. เพิ่มข้อมูลสำหรับเทรนด้วย Data Augmentation

```bash
python src/python/augment_training_data.py
```

### 3. ปรับแต่ง Hyperparameters

```bash
python src/python/hyperparameter_tuning.py
```

### 4. เทรนโมเดลใหม่ด้วยข้อมูลที่เพิ่มขึ้น

```bash
python src/python/retrain_improved_model.py
```

### 5. ทดสอบโมเดลอย่างละเอียด

```bash
python src/python/comprehensive_test.py
```

### 6. เริ่มต้น Web API และ Dashboard

```bash
python src/python/subject_classifier_api.py
```

เมื่อ API ทำงานแล้ว สามารถใช้งาน Dashboard ได้ที่: http://localhost:8000/

## 📁 โครงสร้างไฟล์

```
DekDataset/
├── data/
│   └── output/
│       ├── auto-dataset-primary-exam-sample.jsonl    # ข้อมูลตัวอย่างระดับประถม
│       ├── auto-dataset-secondary-exam-sample.jsonl  # ข้อมูลตัวอย่างระดับมัธยม
│       ├── augmented_samples.jsonl                   # ข้อมูลที่เพิ่มขึ้นจาก augmentation
│       ├── thai-education-model/                     # โมเดลที่เทรนแล้ว
│       ├── thai-education-model-tuned/               # โมเดลที่ปรับแต่งแล้ว
│       ├── thai-education-model-improved/            # โมเดลที่เทรนด้วยข้อมูลที่เพิ่มขึ้น
│       └── evaluation-results/                       # ผลการประเมินโมเดล
├── src/
│   └── python/
│       ├── train_primary_secondary.py                # สคริปต์เทรนโมเดลพื้นฐาน
│       ├── initialize_model.py                       # สคริปต์สร้างโมเดลพื้นฐาน
│       ├── train_with_validation.py                  # สคริปต์เทรนพร้อม validation
│       ├── test_model.py                             # สคริปต์ทดสอบโมเดล
│       ├── augment_training_data.py                  # สคริปต์สร้างข้อมูลเพิ่ม
│       ├── hyperparameter_tuning.py                  # สคริปต์ปรับแต่ง hyperparameters
│       ├── comprehensive_test.py                     # สคริปต์ทดสอบโมเดลอย่างละเอียด
│       ├── retrain_improved_model.py                 # สคริปต์เทรนโมเดลด้วยข้อมูลที่เพิ่มขึ้น
│       └── subject_classifier_api.py                 # สคริปต์ API และ Dashboard
├── requirements.txt                                  # Dependencies ของโปรเจค
└── README.md                                         # ไฟล์นี้
```

## 📖 API Documentation

### 1. ทำนายวิชา

```
POST /api/predict
```

**Request Body**:
```json
{
  "text": "ข้อความที่ต้องการทำนาย"
}
```

**Response**:
```json
{
  "subject": "คณิตศาสตร์",
  "confidence": 0.85,
  "all_probabilities": {
    "ภาษาไทย": 0.05,
    "คณิตศาสตร์": 0.85,
    "วิทยาศาสตร์": 0.04,
    ...
  },
  "text": "ข้อความที่ต้องการทำนาย"
}
```

### 2. ดึงรายชื่อวิชา

```
GET /api/subjects
```

### 3. ดึงประวัติการทำนาย

```
GET /api/history
```

## 📊 ผลการทดสอบและการพัฒนา

### ผลการทดสอบ

โมเดลได้รับการทดสอบและให้ผลลัพธ์ดังนี้:

- **ความแม่นยำ (Accuracy)**: 
  - โมเดลพื้นฐาน: ~33-40% 
  - โมเดลที่ปรับปรุง: ~75-85% (ขึ้นอยู่กับข้อมูลเทรน)
- **F1 Score (Macro)**:
  - โมเดลพื้นฐาน: ~0.3-0.4
  - โมเดลที่ปรับปรุง: ~0.7-0.8

### แนวทางการพัฒนาต่อ

1. **เพิ่มข้อมูลเทรน**: รวบรวมตัวอย่างเพิ่มเติมในแต่ละวิชาเพื่อปรับปรุงความแม่นยำ
2. **ปรับปรุงโมเดล**: ทดลองใช้โมเดลอื่นๆ เช่น XLM-RoBERTa หรือโมเดลภาษาไทยรุ่นใหม่
3. **รองรับการจำแนกย่อย**: เพิ่มความสามารถในการแยกหัวข้อย่อยในแต่ละวิชา
4. **เพิ่มคุณสมบัติของ API**: พัฒนาฟีเจอร์อื่นๆ เช่น การแนะนำสื่อการสอนที่เกี่ยวข้อง

---

พัฒนาเพื่อโครงการ DekDataset
