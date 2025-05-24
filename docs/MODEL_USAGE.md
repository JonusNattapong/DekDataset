# DekDataset - ระบบจัดการข้อมูลการศึกษาไทย

ระบบ DekDataset สำหรับการศึกษาไทย (ป.1-ม.6) ให้บริการเกี่ยวกับการสร้าง/ประเมิน dataset สำหรับวิชาต่างๆ ตามหลักสูตรแกนกลางการศึกษาขั้นพื้นฐาน พ.ศ. 2551 รวมถึงโมเดล AI ที่สามารถจำแนกข้อความตามกลุ่มสาระวิชา

## สิ่งที่ได้รับการพัฒนา

1. เพิ่ม task schema ใน tasks.json สำหรับ:
   - primary_school_knowledge (ป.1-ป.6)
   - secondary_school_knowledge (ม.1-ม.6)
   - school_grade_promotion_exam (สอบเลื่อนชั้น)

2. เพิ่มรายวิชาใหม่ในทุก schema:
   - ดนตรี
   - เทคโนโลยี
   - ภาษาจีน
   - ประวัติศาสตร์
   - ภูมิศาสตร์

3. เพิ่ม field เพิ่มเติมในทุก schema:
   - difficulty (ระดับความยาก: ง่าย, กลาง, ยาก)
   - source (แหล่งที่มาของข้อมูล)
   - year (ปีของข้อมูล/ข้อสอบ)
   - tags (ป้ายกำกับเพิ่มเติม)
   - explanation (คำอธิบายเพิ่มเติม)

4. ระบบประมวลผลข้อมูลและโมเดล AI:
   - สคริปต์รวมข้อมูล ป.1-ม.6 (`merge_primary_secondary.py`)
   - สคริปต์เทรนโมเดลจัดหมวดหมู่ข้อความตามวิชา (`train_primary_secondary.py`)
   - สคริปต์ทดสอบโมเดล (`test_model.py`)
   - สคริปต์ตั้งค่า labels และ config ของโมเดล (`initialize_model.py`)
   - สคริปต์เทรนพร้อมประเมินผล (`train_with_validation.py`)
   - สคริปต์เทรนด้วย Qwen/Qwen3-0.6B (`train_qwen_model.py`)
   - สคริปต์เพิ่มข้อมูลด้วย Data Augmentation (`data_augmentation.py`)

5. เครื่องมือใช้งานโมเดล:
   - API สำหรับการทำนายวิชา (`subject_prediction_api.py`)
   - GUI สำหรับการทำนายวิชา (`subject_prediction_gui.py`)

## วิธีการใช้งาน

### การติดตั้ง

1. Clone repository:
```bash
git clone https://github.com/yourusername/DekDataset.git
cd DekDataset
```

2. ติดตั้งแพ็กเกจที่จำเป็นสำหรับการเทรน:
```bash
pip install -r src/python/requirements-train.txt
```

### การเทรนโมเดล

1. เทรนโมเดล WangchanBERTa:
```bash
python src/python/train_primary_secondary.py
```

2. เทรนโมเดล Qwen/Qwen3-0.6B:
```bash
python src/python/train_qwen_model.py
```

หรือใช้สคริปต์อัตโนมัติ:
- สำหรับ Linux/Mac: `bash src/python/train_qwen_model.sh`
- สำหรับ Windows: `.\src\python\train_qwen_model.bat`

### การทดสอบโมเดล

```bash
python src/python/test_model.py
```

### การเพิ่มข้อมูลด้วย Data Augmentation

```bash
python src/python/data_augmentation.py --input data/output/merged-primary-secondary.jsonl --output data/output/augmented-dataset.jsonl --augmentations 3 --generate
```

พารามิเตอร์:
- `--input`: ไฟล์ต้นฉบับ
- `--output`: ไฟล์ผลลัพธ์
- `--augmentations`: จำนวนการเพิ่มข้อมูลต่อข้อความต้นฉบับ (ค่าเริ่มต้น: 3)
- `--generate`: เปิดใช้การสร้างข้อมูลใหม่ด้วย template
- `--num_generated`: จำนวนข้อมูลใหม่ที่จะสร้างต่อวิชาต่อระดับชั้น (ค่าเริ่มต้น: 20)

### การใช้งาน API

1. เริ่มต้น API server:
```bash
# สำหรับ Linux/Mac
bash src/python/start_api_server.sh

# สำหรับ Windows
.\src\python\start_api_server.bat
```

2. เข้าถึง API ได้ที่ http://localhost:8000 (API docs: http://localhost:8000/docs)

#### ตัวอย่างการใช้งาน API

```python
import requests

# ทำนายวิชาจากข้อความ
response = requests.post("http://localhost:8000/predict", json={
    "text": "การบวกเลขสองหลัก 25 + 13 = 38",
    "model_name": "qwen"  # หรือ "wangchanberta"
})

# แสดงผลลัพธ์
print(response.json())
```

### การใช้งาน GUI

เริ่มต้น GUI application:

```bash
# สำหรับ Linux/Mac
bash src/python/start_gui.sh

# สำหรับ Windows
.\src\python\start_gui.bat
```

## โครงสร้างไฟล์

```
/src/python/
  ├── tasks.json                    # โครงสร้าง schema สำหรับข้อมูล
  ├── merge_primary_secondary.py    # สคริปต์รวมข้อมูล ป.1-ม.6
  ├── train_primary_secondary.py    # สคริปต์เทรนโมเดล WangchanBERTa
  ├── train_qwen_model.py           # สคริปต์เทรนโมเดล Qwen
  ├── test_model.py                 # สคริปต์ทดสอบโมเดล
  ├── initialize_model.py           # สคริปต์ตั้งค่าโมเดล
  ├── train_with_validation.py      # สคริปต์เทรนพร้อมประเมินผล
  ├── data_augmentation.py          # สคริปต์เพิ่มข้อมูล
  ├── subject_prediction_api.py     # API สำหรับทำนายวิชา
  ├── subject_prediction_gui.py     # GUI สำหรับทำนายวิชา
  ├── requirements-train.txt        # แพ็กเกจสำหรับการเทรน
  ├── train_qwen_model.sh/.bat      # สคริปต์เทรนอัตโนมัติ
  ├── start_api_server.sh/.bat      # สคริปต์เริ่ม API server
  └── start_gui.sh/.bat             # สคริปต์เริ่ม GUI
```

## ข้อแนะนำในการพัฒนาต่อ

1. เพิ่มข้อมูลเทรนให้มากขึ้นเพื่อปรับปรุงประสิทธิภาพโมเดล
2. พัฒนาเทคนิค data augmentation เพิ่มเติม
3. เพิ่มภาษาอื่น ๆ นอกจากภาษาไทยและภาษาอังกฤษ
4. พัฒนา API ให้รองรับการเพิ่มข้อมูลใหม่
5. เพิ่มความสามารถในการตอบคำถามเกี่ยวกับเนื้อหาในแต่ละวิชา
