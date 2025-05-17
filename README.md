# DekDataset 🇹🇭

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

## 🚀 Quick Start

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
cargo run -- summarization,ner 10
```

### 5. Generate Dataset (Python)

```bash
python src/python/generate_dataset.py summarization 10 --format jsonl
```

### 6. Export to Parquet/Arrow (optional)

```bash
python data/output/export_parquet_arrow.py data/output/auto-dataset-summarization-YYYYMMDD-HHMMSS.jsonl parquet
```

---

## 🛠️ Features

- **Unified Task Schema:** Rust & Python fetch จาก API เดียวกัน
- **Batch & Flexible Output:** สร้างหลาย task, เลือก format ได้
- **Progress Bar & Banner:** CLI สวยงาม
- **Robust Export:** รองรับ field ซ้อน, metadata, empty struct
- **Metadata:** ทุก entry มี `{ "source": "DEEPSEEK-V3" }`
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

## 👤 Credits Developer

<div align="center">
  <table>
    <tr>
      <td align="center"><a href="https://github.com/JonusNattapong"><img src="https://github.com/JonusNattapong.png" width="100px;" alt="JonusNattapong"/><br /><sub><b>JonusNattapong</b></sub></a></td>
      <td align="center"><a href="https://github.com/zombitx64"><img src="https://github.com/zombitx64.png" width="100px;" alt="zombitx64"/><br /><sub><b>zombitx64</b></sub></a></td>
    </tr>
  </table>
</div>


## 📝 License

MIT

---

## 💡 Tips (ภาษาไทย)

- ตั้งค่า `DEEPSEEK_API_KEY` ก่อนใช้งาน
- ต้องรัน API server ก่อน Rust/Python จะ fetch task ได้
- ทุก output มี metadata สำหรับตรวจสอบแหล่งที่มา
- ดูตัวอย่าง schema เพิ่มเติมใน `docs/` หรือ README

---

> สร้าง AI ภาษาไทยได้ง่าย ๆ ด้วย DekDataset! 🇹🇭✨
