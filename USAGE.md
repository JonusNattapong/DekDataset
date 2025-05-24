# DekDataset - Thai Education Model ขั้นตอนการใช้งาน

ไฟล์นี้อธิบายขั้นตอนการใช้งานโมเดลสำหรับจำแนกข้อความตามกลุ่มสาระการเรียนรู้

## 1. การเตรียมข้อมูล

### การสร้างข้อมูล

```bash
# Windows
.\src\python\generate_and_augment_data.bat

# Linux/Mac
bash src/python/generate_and_augment_data.sh
```

สคริปต์นี้จะดำเนินการ:
1. สร้างข้อคำถามที่หลากหลายด้วย `generate_diverse_questions.py`
2. รวมข้อมูลที่มีอยู่ด้วย `merge_primary_secondary.py`
3. เพิ่มข้อมูลด้วย Data Augmentation ด้วย `data_augmentation.py`

ผลลัพธ์ที่ได้:
- `data/output/generated-diverse-questions.jsonl` - ข้อคำถามที่สร้างใหม่
- `data/output/merged-primary-secondary.jsonl` - ข้อมูลที่รวมแล้ว
- `data/output/augmented-dataset.jsonl` - ข้อมูลที่ผ่านการเพิ่มเติมแล้ว

## 2. การเทรนโมเดล

### เทรนโมเดล Qwen/Qwen3-0.6B

```bash
# Windows
.\src\python\train_qwen_model.bat

# Linux/Mac
bash src/python/train_qwen_model.sh
```

โมเดลที่เทรนเสร็จแล้วจะถูกบันทึกที่: `data/output/thai-education-qwen-model/`

### เทรนโมเดล WangchanBERTa

```bash
python src/python/train_primary_secondary.py
```

โมเดลที่เทรนเสร็จแล้วจะถูกบันทึกที่: `data/output/thai-education-model/`

## 3. การทดสอบโมเดล

```bash
python src/python/test_model.py
```

## 4. การใช้งานโมเดล

### API สำหรับการทำนาย

```bash
# Windows
.\src\python\start_api_server.bat

# Linux/Mac
bash src/python/start_api_server.sh
```

API จะทำงานที่ http://localhost:8000 (เอกสาร API: http://localhost:8000/docs)

### GUI สำหรับการทำนาย

```bash
# Windows
.\src\python\start_gui.bat

# Linux/Mac
bash src/python/start_gui.sh
```

## 5. เอกสารเพิ่มเติม

- หากต้องการข้อมูลเพิ่มเติมเกี่ยวกับโมเดล Qwen/Qwen3-0.6B โปรดดูที่ [docs/QWEN_MODEL.md](docs/QWEN_MODEL.md)
- หากต้องการข้อมูลเพิ่มเติมเกี่ยวกับการใช้งานโมเดล โปรดดูที่ [docs/MODEL_USAGE.md](docs/MODEL_USAGE.md)
