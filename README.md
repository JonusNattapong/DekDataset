# DekDataset

![Rust](https://img.shields.io/badge/Rust-%23dea584?style=flat-square&logo=rust&logoColor=black)
![Python](https://img.shields.io/badge/Python-%233776AB?style=flat-square&logo=python&logoColor=white)
![DeepSeek API](https://img.shields.io/badge/DeepSeek-API-blueviolet?style=flat-square)
![License: MIT](https://img.shields.io/badge/License-MIT-green?style=flat-square)

---

## ✨ Overview

DekDataset คือเครื่องมือโอเพ่นซอร์สสำหรับสร้างชุดข้อมูล (dataset) ภาษาไทยสำหรับงาน AI/ML (NLP, Classification, QA, NER ฯลฯ) แบบอัตโนมัติ รองรับทั้ง Rust และ Python

- **Unified Schema:** Rust & Python ใช้ API เดียวกัน (FastAPI)
- **Export:** JSONL, Parquet, Arrow, CSV
- **Metadata:** ทุก output มี `{ "source": "zombit" }`
- **Beautiful CLI:** Banner, progress bar, สีสัน
- **Batch Mode:** สร้างหลาย task/format ได้ในคำสั่งเดียว

---

## 🚀 Quick Start (Windows/Bash)

### 1. Clone & Install

```bash
# Clone repo
cd DekDataset
# Rust dependencies
cargo build --release
# Python dependencies
pip install -r requirements.txt
```

### 2. Set up DeepSeek API

- สร้างไฟล์ `.env` แล้วใส่

```env
DEEPSEEK_API_KEY=your_api_key
```

### 3. Run Task Definitions API (Python)

```bash
python src/python/task_definitions_api.py
```

### 4. Generate Dataset (Rust)

```bash
cargo run -- summarization,ner 10 --parquet
```

- รองรับหลาย task พร้อมกัน (คั่นด้วย ,)
- สามารถเลือก output format: `--parquet`, `--arrow`, `--both` หรือไม่ใส่ (default: jsonl)

### 5. Generate Dataset (Python)

```bash
python src/python/generate_dataset.py summarization 10 --format jsonl
```

- รองรับ format: `json`, `jsonl`
- สามารถใช้ task อื่น ๆ ได้ เช่น `sentiment_analysis`, `translation`, `ner`, `text_classification`, `question_answer`

### 6. Export to Parquet/Arrow (optional)

```bash
python data/output/export_parquet_arrow.py data/output/auto-dataset-summarization-YYYYMMDD-HHMMSS.jsonl parquet
python data/output/export_parquet_arrow.py data/output/auto-dataset-summarization-YYYYMMDD-HHMMSS.jsonl arrow
```

- สามารถแปลงไฟล์ json/jsonl/csv เป็น parquet หรือ arrow ได้ทันที

---

## 🛠️ Features

- **Unified Task Schema:** Rust & Python fetch จาก API เดียวกัน
- **Batch & Flexible Output:** สร้างหลาย task, เลือก format ได้
- **Progress Bar & Banner:** CLI สวยงาม
- **Robust Export:** รองรับ field ซ้อน, metadata, empty struct
- **Metadata:** ทุก entry มี `{ "source": "zombit" }`
- **Extensible:** เพิ่ม task ใหม่ได้ง่ายใน `tasks.json`/API

---

## 📁 Project Structure

```text
DekDataset/
├── src/
│   ├── main.rs, models.rs, api_client.rs, generator.rs, banner.rs
│   └── python/
│       ├── generate_dataset.py, banner.py, task_definitions.py, task_definitions_api.py
├── data/output/           # All generated datasets & exports
├── docs/                  # Documentation, task.md
├── README.md
```

---

## 👤 Credits

- Developer: zombit | JonusNattapong
- GitHub: [zombitx64](https://github.com/zombitx64)
- Contact: [zombitx64@gmail.com](mailto:zombitx64@gmail.com)

## 📝 License

MIT

---

## 💡 Tips (ภาษาไทย)

- ตั้งค่า `DEEPSEEK_API_KEY` ก่อนใช้งาน
- ต้องรัน API server ก่อน Rust/Python จะ fetch task ได้
- ทุก output มี metadata สำหรับตรวจสอบแหล่งที่มา
- ดูตัวอย่าง schema เพิ่มเติมใน `docs/` หรือ README
- ใช้ Bash หรือ Command Prompt ได้ (แต่ path ต้องถูกต้อง)

---

## ⚡️ Batch Generation (Auto)

ตั้งแต่เวอร์ชัน 2025.05 ขึ้นไป DekDataset รองรับการแบ่ง batch อัตโนมัติทั้ง Python และ Rust:

- ระบบจะเลือก batch size ที่เหมาะสมให้เอง เช่น
  - ถ้าขอ ≤10 ตัวอย่าง: ส่งทีเดียว
  - ถ้าขอ 11-100 ตัวอย่าง: ส่งทีละ 10
  - ถ้ามากกว่า 100 ตัวอย่าง: ส่งทีละ 5
- วนส่งหลายรอบจนครบจำนวนที่ต้องการ (เช่น 200 ตัวอย่างจะวน 20-40 รอบ)
- เหมาะกับ DeepSeek API ที่มี soft limit ต่อ 1 คำขอ (เช่น ตอบกลับสูงสุด ~5-10 ตัวอย่างต่อครั้ง)
- ได้ผลลัพธ์ครบตามที่สั่งโดยอัตโนมัติ ไม่ต้องแก้ไขโค้ดเอง

**ข้อควรระวัง:**
- หาก DeepSeek API เปลี่ยน limit หรือเกิด timeout อาจได้ข้อมูลน้อยกว่าที่ขอ (ระบบจะเตือน)
- สามารถปรับ batch size เองในโค้ดได้หากต้องการ

---

> สร้าง AI ภาษาไทยได้ง่าย ๆ ด้วย DekDataset! 🇹🇭✨
