# DekDataset - Dataset Generation Tools

## 🚀 New Features

### Data Cleaning & Text Normalization

สามารถทำความสะอาดและ normalize ข้อความภาษาไทยได้อัตโนมัติ:

```bash
# ใช้การทำความสะอาดข้อความแบบ default (เปิดใช้งานโดย default)
python src/python/generate_dataset.py sentiment_analysis 100

# ปิดการทำความสะอาดข้อความ
python src/python/generate_dataset.py sentiment_analysis 100 --no-clean

# ปิดการ normalize ภาษาไทย
python src/python/generate_dataset.py sentiment_analysis 100 --disable-thai-norm

# ลบ emoji และอักขระพิเศษ
python src/python/generate_dataset.py sentiment_analysis 100 --remove-emojis --remove-special-chars
```

[ดูรายละเอียดเพิ่มเติมเกี่ยวกับ Data Cleaning](./DATA_CLEANING.md)

DekDataset มีฟังก์ชันใหม่สำหรับทำความสะอาดและ normalize ข้อมูลอัตโนมัติ:
- ลบแท็ก HTML
- ลบ emoji
- Normalize ช่องว่าง
- ปรับปรุงสระภาษาไทย (เช่น เเ → แ)
- ตรวจสอบภาษาอัตโนมัติ
- ปรับแต่งเฉพาะทางการแพทย์ (สำหรับ medical_benchmark)

### การแบ่ง Train/Valid/Test

สามารถแบ่ง dataset เป็น train/validation/test ได้อัตโนมัติ:
```bash
python src/python/generate_dataset.py medical_benchmark 500 --split --train-ratio 0.8 --valid-ratio 0.1 --test-ratio 0.1
```

### การกำจัดความซ้ำซ้อนข้าม batch

ปรับปรุงการตรวจหาข้อมูลซ้ำซ้อนให้ทำงานทั้งภายใน batch และข้าม batch:
- ลดความซ้ำซ้อนในชุดข้อมูล
- สามารถระบุฟิลด์ที่สำคัญสำหรับตรวจสอบความซ้ำ

### เพิ่ม Custom Metadata

เพิ่ม metadata เช่น license, version, domain ในข้อมูล:
```bash
python src/python/generate_dataset.py sentiment_analysis 100 --license "MIT" --version "2.0.0" --domain "social-media"
```

หรือใช้ JSON string สำหรับ metadata ซับซ้อน:
```bash
python src/python/generate_dataset.py summarization 50 --metadata '{"source": "news", "tags": ["Thai", "politics"]}'
```

### เพิ่มข้อมูลต่อจากไฟล์เดิม

สามารถเพิ่มข้อมูลต่อจากไฟล์ที่มีอยู่แล้ว:

```bash
python src/python/generate_dataset.py medical_benchmark 100 --append data/output/auto-dataset-medical_benchmark-20250524-120000.jsonl
```

### วิเคราะห์และสร้างภาพแสดงผลข้อมูล

สามารถวิเคราะห์คุณภาพของชุดข้อมูล และสร้างภาพแสดงผลได้:
```bash
# วิเคราะห์ข้อมูล
python src/python/generate_dataset.py sentiment_analysis 100 --analyze

# วิเคราะห์และสร้างภาพแสดงผล
python src/python/generate_dataset.py sentiment_analysis 100 --analyze --visualize

# กำหนดตำแหน่งผลการวิเคราะห์
python src/python/generate_dataset.py sentiment_analysis 100 --analyze --analyze-output data/analysis/my-dataset
```
[ดูรายละเอียดเพิ่มเติมเกี่ยวกับ Dataset Analysis](./DATASET_ANALYSIS.md)

### อัปโหลดไปยัง Hugging Face Hub

สามารถอัปโหลดชุดข้อมูลไปยัง Hugging Face Hub ได้โดยตรง:
```bash
# อัปโหลดไปยัง Hugging Face Hub
python src/python/generate_dataset.py sentiment_analysis 100 --export-huggingface --hf-repo-id username/my-dataset

# อัปโหลดเป็น private repository
python src/python/generate_dataset.py sentiment_analysis 100 --export-huggingface --hf-repo-id username/my-dataset --hf-private
```
[ดูรายละเอียดเพิ่มเติมเกี่ยวกับ Hugging Face Hub Integration](./HUGGINGFACE_HUB.md)

## 👨‍💻 Unit Tests

DekDataset มี unit tests สำหรับฟังก์ชันสำคัญ เช่น clean_text, deduplicate_entries:
```bash
python src/python/test_dataset_utils.py
```

## 📚 Command Line Arguments ทั้งหมด

```
usage: generate_dataset.py [-h] [--format {json,jsonl,csv,parquet}] [--import-vision IMPORT_VISION] [--source_lang SOURCE_LANG]
                          [--target_lang TARGET_LANG] [--lang LANG] [--no-clean] [--disable-thai-norm] [--license LICENSE]
                          [--version VERSION] [--domain DOMAIN] [--split] [--train-ratio TRAIN_RATIO] [--valid-ratio VALID_RATIO]
                          [--test-ratio TEST_RATIO] [--append APPEND] [--metadata METADATA]
                          task count

DekDataset: Generate high-quality dataset for NLP and other tasks

positional arguments:
  task                  Task name (e.g. sentiment_analysis, medical_benchmark)
  count                 Number of samples to generate

options:
  -h, --help            show this help message and exit
  --format {json,jsonl,csv,parquet}
                        Output format: json, jsonl, csv, parquet
  --import-vision IMPORT_VISION
                        Path to vision-animals-dataset-*.jsonl to import/validate/export
  --source_lang SOURCE_LANG
                        Source language (for translation task)
  --target_lang TARGET_LANG
                        Target language (for translation task)
  --lang LANG           Language code (for multilingual tasks, e.g. th, en, zh, hi)
  --no-clean            Skip data cleaning/normalization
  --disable-thai-norm   Disable Thai text normalization
  --license LICENSE     License for the dataset metadata
  --version VERSION     Version for the dataset metadata
  --domain DOMAIN       Domain for the dataset metadata (e.g. medical, education, finance)
  --split               Split dataset into train/valid/test
  --train-ratio TRAIN_RATIO
                        Train set ratio when splitting (default: 0.8)
  --valid-ratio VALID_RATIO
                        Validation set ratio when splitting (default: 0.1)
  --test-ratio TEST_RATIO
                        Test set ratio when splitting (default: 0.1)
  --append APPEND       Path to existing dataset file to append to (deduplicate across both)
  --metadata METADATA   Additional metadata as JSON string, e.g. '{"source": "custom", "tags": ["medical"]}'
```
