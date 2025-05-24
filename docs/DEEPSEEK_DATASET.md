# การใช้ Deepseek API สร้าง Dataset สำหรับการศึกษาไทย

คู่มือนี้อธิบายวิธีการสร้าง Dataset คุณภาพสูงสำหรับการศึกษาไทย (ป.1-ม.6) โดยใช้ Deepseek API

## สิ่งที่ต้องเตรียม

1. **Deepseek API Key** - สมัครและรับ API Key ได้ที่ [platform.deepseek.com](https://platform.deepseek.com)
2. **Python 3.8+** - พร้อมแพ็คเกจที่จำเป็น (รัน `pip install -r requirements.txt`)

## ขั้นตอนการสร้าง Dataset

### วิธีที่ 1: ใช้สคริปต์อัตโนมัติ

วิธีนี้ง่ายที่สุด โดยสคริปต์จะดำเนินการทุกขั้นตอนให้อัตโนมัติ:

#### สำหรับ Windows
```
src\python\generate_deepseek_dataset.bat
```

#### สำหรับ Linux/macOS
```
bash src/python/generate_deepseek_dataset.sh
```

### วิธีที่ 2: รันคำสั่งแยกทีละขั้นตอน

หากต้องการควบคุมแต่ละขั้นตอนได้มากขึ้น:

#### 1. สร้าง Dataset จาก Deepseek API

```
python src/python/deepseek_data_generation.py --api_key YOUR_API_KEY --questions_per_call 10 --output data/output/deepseek-dataset-raw.jsonl
```

พารามิเตอร์ที่สำคัญ:
- `--api_key` - API Key ของ Deepseek
- `--questions_per_call` - จำนวนคำถามต่อการเรียก API หนึ่งครั้ง
- `--parallel` - เรียก API แบบขนาน (เร็วขึ้นแต่ใช้ API มากขึ้น)
- `--max_workers` - จำนวน worker สูงสุดในการรันแบบขนาน
- `--model` - โมเดล Deepseek ที่ต้องการใช้
- `--output` - ไฟล์ output สำหรับบันทึกข้อมูลดิบ

เรียก API เฉพาะบางวิชาหรือบางระดับชั้น:
```
python src/python/deepseek_data_generation.py --api_key YOUR_API_KEY --subjects "คณิตศาสตร์" "วิทยาศาสตร์" --grades 1 2 3
```

#### 2. เตรียมและแบ่ง Dataset

```
python src/python/prepare_deepseek_dataset.py --input data/output/deepseek-dataset-raw.jsonl --output_dir data/output/deepseek-dataset
```

พารามิเตอร์ที่สำคัญ:
- `--input` - ไฟล์ข้อมูลดิบจาก Deepseek API
- `--output_dir` - ไดเรกทอรีสำหรับไฟล์ output
- `--max_per_group` - จำนวนตัวอย่างสูงสุดต่อกลุ่มวิชา-ระดับชั้น (balance)
- `--train_ratio` - สัดส่วนข้อมูลสำหรับ training (default: 0.8)
- `--validation_ratio` - สัดส่วนข้อมูลสำหรับ validation (default: 0.1)
- `--test_ratio` - สัดส่วนข้อมูลสำหรับ testing (default: 0.1)
- `--skip_balance` - ข้ามขั้นตอนการทำ balance
- `--skip_analysis` - ข้ามขั้นตอนการวิเคราะห์ข้อมูล

## โครงสร้างข้อมูลที่สร้าง

Dataset ที่สร้างจะอยู่ในรูปแบบ JSONL โดยแต่ละรายการมีโครงสร้างดังนี้:

```json
{
  "subject": "คณิตศาสตร์",
  "grade": 1,
  "question": "5 + 3 เท่ากับเท่าไร?",
  "choices": ["7", "8", "9", "10"],
  "answer": "8",
  "difficulty": "ง่าย",
  "tags": ["คณิตศาสตร์", "การบวก", "ป.1"]
}
```

ไฟล์ที่สร้างขึ้น:
- `train.jsonl` - ข้อมูลสำหรับเทรนโมเดล (80%)
- `validation.jsonl` - ข้อมูลสำหรับตรวจสอบระหว่างการเทรน (10%)
- `test.jsonl` - ข้อมูลสำหรับทดสอบโมเดล (10%)
- `analysis/` - โฟลเดอร์บันทึกผลการวิเคราะห์ข้อมูล

## การใช้งานกับโมเดล Qwen

หลังจากได้ Dataset แล้ว สามารถใช้สคริปต์ train_qwen_model.py เพื่อเทรนโมเดล:

```
python src/python/train_qwen_model.py --train_file data/output/deepseek-dataset/train.jsonl --validation_file data/output/deepseek-dataset/validation.jsonl --output_dir models/qwen-thai-education --num_train_epochs 3
```

## เคล็ดลับ

1. **ประหยัด API Credit** - เริ่มด้วย `--questions_per_call` ค่าน้อยๆ (5-10) เพื่อทดสอบ
2. **แก้ไขปัญหาข้อมูลไม่สมดุล** - ใช้ `--max_per_group` เพื่อจำกัดจำนวนข้อมูลในแต่ละกลุ่ม
3. **คุณภาพโมเดล** - โมเดล Deepseek ขนาดใหญ่จะให้ข้อมูลที่มีคุณภาพดีกว่า
4. **ใช้ batch เมื่อต้องการข้อมูลจำนวนมาก** - การรันแบบขนานช่วยประหยัดเวลา
