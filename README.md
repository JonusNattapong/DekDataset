# DekDataset

![Rust](https://img.shields.io/badge/Rust-%23dea584?style=flat-square&logo=rust&logoColor=black)
![Python](https://img.shields.io/badge/Python-%233776AB?style=flat-square&logo=python&logoColor=white)
![DeepSeek API](https://img.shields.io/badge/DeepSeek-API-blueviolet?style=flat-square)
![License: MIT](https://img.shields.io/badge/License-MIT-green?style=flat-square)

---

## 🏗️ Overview (รายละเอียดภาพรวมและหลักการทำงาน)

DekDataset คือระบบโอเพ่นซอร์สสำหรับสร้างชุดข้อมูล (dataset) ภาษาไทยและสากล สำหรับงาน AI/ML ทั้งด้าน NLP (Natural Language Processing), Computer Vision (Image Classification, OCR), และงาน Data-centric อื่น ๆ โดยเน้นความง่ายในการใช้งาน ความยืดหยุ่น และความสามารถในการขยายต่อยอด รองรับทั้งสายงานวิจัยและอุตสาหกรรม

### ที่มาและเป้าหมาย

- ปัญหาหลักของวงการ AI/NLP/Computer Vision ภาษาไทย คือขาดชุดข้อมูลคุณภาพสูงที่มีความหลากหลายและ metadata ครบถ้วน
- DekDataset ถูกออกแบบมาเพื่อให้ทุกคนสามารถสร้าง dataset ที่มี schema มาตรฐาน, metadata, และรองรับการใช้งานกับเครื่องมือสมัยใหม่ (เช่น HuggingFace, PyArrow, Parquet, Pandas) ได้ทันที
- รองรับการสร้าง dataset ทั้งแบบ text, image, OCR, multi-modal, และสามารถขยาย schema ได้เอง
- เน้นความ robust, reproducible, และสามารถ integrate กับ pipeline อื่น ๆ ได้ง่าย

### หลักการทำงานและภาพรวมระบบ

1. **Unified Task Schema**
   - ทุก task (เช่น summarization, sentiment_analysis, vision_animals, ocr_thai) จะมี schema กลางที่นิยามใน `tasks.json` หรือ API (FastAPI)
   - Schema กำหนด field, type, enum, constraints, ตัวอย่าง, parameter ฯลฯ
   - Rust และ Python pipeline จะ fetch schema เดียวกัน ทำให้ output สอดคล้องกันเสมอ

2. **Dataset Generation Pipeline**
   - ผู้ใช้เลือก task และจำนวนตัวอย่างที่ต้องการ (ผ่าน CLI หรือ script)
   - ระบบจะสร้าง prompt สำหรับ LLM (DeepSeek, OpenAI, ฯลฯ) เพื่อ generate ข้อมูลตาม schema
   - รองรับ batch mode (แบ่งรอบ, ข้าม batch ที่ error, สุ่มเติม quota)
   - Validate, deduplicate, enrich, balance label, และ export เป็น jsonl, parquet, arrow, csv
   - ทุก entry มี metadata (source, created_at, lang) เพื่อความโปร่งใสและ reproducibility

3. **Vision & OCR Dataset**
   - รองรับการสร้าง dataset ภาพ (image classification, OCR, captioning) โดยดึง label/class อัตโนมัติจาก API/schema
   - ดึงภาพจากหลายแหล่ง (Bing, DuckDuckGo, Pexels, Pixabay, AI generate)
   - สร้าง caption อัตโนมัติ (BLIP/AI), แปล caption (DeepSeek API)
   - สุ่มเติม quota, robust ต่อ error, assign id/filename, export jsonl + images/ พร้อม metadata
   - สำหรับ OCR Thai: มีระบบ scraping ป้าย, ฉลาก, เอกสาร, พร้อม metadata

4. **Web Scraping & Multi-source Image Download**
   - ฟังก์ชัน scraping ภาพจาก search engine/API หลายแหล่ง (Bing, DuckDuckGo, Pexels, Pixabay)
   - รวม url, shuffle, remove duplicates, ดาวน์โหลดจนกว่าจะครบ quota
   - สร้างโฟลเดอร์ output ตาม timestamp, เก็บภาพใน images/, metadata ใน scraped_metadata.jsonl

5. **Extensibility & Integration**
   - เพิ่ม/แก้ไข task/schema ได้ง่ายใน `tasks.json` หรือ API แล้ว pipeline จะรองรับอัตโนมัติ
   - สามารถ merge vision dataset, text dataset, OCR dataset ได้ง่าย (schema compatible)
   - ใช้ .env สำหรับ API Key (DeepSeek, Pexels, Pixabay, HuggingFace)
   - Output พร้อมใช้งานกับ HuggingFace, PyArrow, Pandas, Parquet, ML pipeline

6. **Error Handling & Robustness**
   - ทุกฟังก์ชันมี try/except, log error, retry, fallback
   - Batch mode: ข้าม batch ที่ error, ไม่หยุดทั้ง pipeline
   - Validate schema ก่อน export, enrich metadata อัตโนมัติ
   - Web scraping: shuffle url, remove duplicates, quota per source

### สรุป

DekDataset คือเครื่องมือสร้าง dataset ภาษาไทย/สากลที่ครบวงจร รองรับทั้งสาย NLP, Vision, OCR, และงาน data-centric อื่น ๆ เหมาะสำหรับนักวิจัย นักพัฒนา และองค์กรที่ต้องการสร้างหรือขยายชุดข้อมูล AI/ML อย่างมีมาตรฐานและยืดหยุ่นสูง

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

## 🖼️ Scrape OCR Thai Images (Bing, DuckDuckGo, Pexels, Pixabay)

สามารถดึงภาพ OCR ภาษาไทย (เช่น ป้าย, ฉลาก, เอกสาร) จากหลายแหล่งพร้อมกัน พร้อม metadata:

```bash
python src/python/web_scrape_images.py --query "ป้ายประกาศราชการ" "ฉลากสินค้าไทย" --num_images 10
```

- ภาพจะถูกเก็บใน scraped_images/scrape-ocr-YYYYMMDD-HHMMSS/images/
- มีไฟล์ scraped_metadata.jsonl (image_path, query, source) สำหรับแต่ละภาพ
- รองรับ Bing, DuckDuckGo (scraping), Pexels, Pixabay (API Key ต้องตั้งใน .env)

ตัวอย่าง .env:

```env
PEXELS_API_KEY=your_pexels_api_key
PIXABAY_API_KEY=your_pixabay_api_key
```

---

## 📚 Technical Details (รายละเอียดเชิงเทคนิค)

### 1. System Architecture

- **Rust Core:** สำหรับ batch dataset generation, export, schema validation, Parquet/Arrow, CLI
- **Python Modules:** สำหรับ flexible pipeline, web scraping, vision dataset, API integration, caption, translation
- **Task API:** FastAPI (Python) ให้บริการ task schema/definition (src/python/task_definitions_api.py)
- **Unified Schema:** ทุกโมดูลใช้ schema กลางจาก tasks.json หรือ API

### 2. Dataset Generation Pipeline

- **Input:** เลือก task (เช่น summarization, sentiment_analysis, vision_animals) และจำนวนตัวอย่าง
- **Process:**
  - ดึง schema/parameter จาก API
  - สร้าง prompt สำหรับ LLM/DeepSeek
  - Batch generate (แบ่งรอบ, robust ต่อ error)
  - Validate, deduplicate, enrich, balance label
  - Export เป็น jsonl, parquet, arrow, csv
- **Output:**
  - โฟลเดอร์ data/output/auto-dataset-<task>-<timestamp>.<ext>
  - ทุก entry มี metadata (source, created_at, lang)

### 3. Vision Dataset & Image Scraping

- **generate_vision_task.py:**
  - ดึง label/class อัตโนมัติจาก API
  - ดึงภาพจากหลายแหล่ง (Pexels, Pixabay, Bing, DuckDuckGo, AI generate)
  - สร้าง caption อัตโนมัติ (BLIP/AI)
  - แปล caption (DeepSeek API)
  - สุ่มเติม quota, robust ต่อ error, assign id/filename
  - Export jsonl + images/ (พร้อม metadata)
- **web_scrape_images.py:**
  - รวมภาพจาก Bing, DuckDuckGo (scraping), Pexels, Pixabay (API)
  - สร้างโฟลเดอร์ output ตาม timestamp, เก็บภาพใน images/, metadata ใน scraped_metadata.jsonl
  - Metadata: image_path, query, source

### 4. Task Schema & Customization

- **tasks.json:** กำหนด schema, parameter, enum, constraints สำหรับแต่ละ task (NLP, Vision, OCR ฯลฯ)
- **เพิ่ม/แก้ไข task:** แก้ tasks.json หรือ API แล้ว pipeline จะรองรับอัตโนมัติ
- **รองรับ custom field, enum, constraints, example, parameter**

### 5. Error Handling & Robustness

- ทุกฟังก์ชันมี try/except, log error, retry, fallback
- Batch mode: ข้าม batch ที่ error, ไม่หยุดทั้ง pipeline
- Validate schema ก่อน export, enrich metadata อัตโนมัติ
- Web scraping: shuffle url, remove duplicates, quota per source

### 6. Integration & Best Practices

- สามารถ merge vision dataset, text dataset, OCR dataset ได้ง่าย (schema compatible)
- ใช้ .env สำหรับ API Key (DeepSeek, Pexels, Pixabay, HuggingFace)
- แนะนำให้รัน task_definitions_api.py ก่อน เพื่อให้ Rust/Python fetch schema ได้
- ใช้ CLI/Script ได้ทั้ง Windows, Linux, Bash, Command Prompt
- Output พร้อมใช้งานกับ HuggingFace, PyArrow, Pandas, Parquet, ML pipeline

### 7. Example Output Structure

```
scraped_images/
  scrape-ocr-20250518-104729/
    images/
      ป้ายประกาศราชการ_bing_1.jpg
      ป้ายประกาศราชการ_pexels_2.jpg
      ...
    scraped_metadata.jsonl
  ...
data/output/
  auto-dataset-sentiment_analysis-20250517-115115.jsonl
  auto-dataset-sentiment_analysis-20250517-115115.parquet
  ...
photo/images/...
```

### 8. Limitations & Notes

- Web scraping อาจถูก block หรือ quota จำกัด (แนะนำใช้หลายแหล่ง)
- DuckDuckGo/Bing อาจเปลี่ยน HTML/vqd token บ่อย ต้องอัปเดต regex
- ภาพจาก AI generate ควรตรวจสอบคุณภาพก่อนใช้งานจริง
- หากต้องการเพิ่มแหล่งภาพ/โมเดลใหม่ สามารถเพิ่มฟังก์ชันใน pipeline ได้ทันที

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

> สร้าง AI ภาษาไทยได้ง่าย ๆ ด้วย DekDataset! 🇹🇭✨