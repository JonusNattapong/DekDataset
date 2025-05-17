# DekDataset: Thai AI/ML Dataset Generator (Rust + Python)

![Rust](https://img.shields.io/badge/Rust-%23dea584?style=flat-square&logo=rust&logoColor=black)
![Python](https://img.shields.io/badge/Python-%233776AB?style=flat-square&logo=python&logoColor=white)
![DeepSeek API](https://img.shields.io/badge/DeepSeek-API-blueviolet?style=flat-square)
![License: MIT](https://img.shields.io/badge/License-MIT-green?style=flat-square)

---

## ✨ Overview

DekDataset is a modern, robust toolkit for generating diverse, high-quality datasets for Thai AI/ML tasks (NLP, classification, QA, NER, etc.).

- **Languages:** Rust & Python
- **API:** Unified FastAPI task schema
- **Export:** JSONL, Parquet, Arrow, CSV
- **Metadata:** All outputs include `{ "source": "zombit" }`
- **Beautiful CLI:** Banner, progress bar, and color

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

- Copy `.env.example` to `.env` and set your `DEEPSEEK_API_KEY`

### 3. Run Task Definitions API (Python)

```bash
cd src/python
python -m uvicorn task_definitions_api:app --reload --host 0.0.0.0 --port 8000
```

### 4. Generate Dataset (Rust)

```bash
cargo run -- sentiment_analysis,text_classification 10
```

### 5. Generate Dataset (Python)

```bash
python src/python/generate_dataset.py sentiment_analysis 10 --format jsonl
```

### 6. Export to Parquet/Arrow

```bash
python data/output/export_parquet_arrow.py data/output/auto-dataset-sentiment_analysis-YYYYMMDD-HHMMSS.jsonl parquet
```

---

## 🛠️ Features

- **Unified Task Schema:** Rust & Python fetch from the same API
- **Batch & Flexible Output:** Generate multiple tasks, choose output format
- **Progress Bar & Banner:** Beautiful CLI experience
- **Robust Export:** Handles empty struct fields, nested metadata
- **Metadata:** All data entries include `{ "source": "zombit" }`
- **Extensible:** Add new tasks easily in `task_definitions.py`/API

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

## 💡 Tips

- Set `DEEPSEEK_API_KEY` before use
- API server must be running for Rust/Python to fetch tasks
- All output includes metadata for provenance
- See `docs/task.md` for task schema details
