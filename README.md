# DekDataset: Thai AI/ML Dataset Generator (Rust + Python)

## Overview

DekDataset is a robust, modern toolkit for generating diverse, high-quality datasets for AI/ML tasks (NLP, classification, QA, NER, etc.) in Thai (with support for technical/English terms). It supports both Rust and Python, and can fetch task definitions from a shared API for maximum flexibility and maintainability.

- **Supports:** Sentiment Analysis, Text Classification, QA, NER, Summarization, Translation, and more
- **API:** FastAPI server for task definitions (Python)
- **Batch Generation:** Generate multiple tasks in one run
- **Export:** JSONL, JSON, Parquet, Arrow, CSV
- **Metadata:** Every entry includes `{"source": "zombit"}`
- **Beautiful CLI Banner & Progress Bar**
- **Thai-centric, realistic, no placeholder/test**

## Quick Start

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

### 6. Export to Parquet/Arrow (auto or manual)

```bash
python data/output/export_parquet_arrow.py data/output/auto-dataset-sentiment_analysis-YYYYMMDD-HHMMSS.jsonl parquet
```

## Features

- **Unified Task Schema:** Rust & Python fetch from the same API
- **Batch & Flexible Output:** Generate multiple tasks, choose output format
- **Progress Bar & Banner:** Beautiful CLI experience
- **Robust Export:** Handles empty struct fields, nested metadata
- **Metadata:** All data entries include `{"source": "zombit"}`
- **Extensible:** Add new tasks easily in `task_definitions.py`/API

## Project Structure

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

## Credits

- Developer: zombit | JonusNattapong
- GitHub: [https://github.com/zombitx64](https://github.com/zombitx64)
- Contact: [zombitx64@gmail.com](mailto:zombitx64@gmail.com)

## License

MIT

---

**Tips:**

- Set `DEEPSEEK_API_KEY` before use
- API server must be running for Rust/Python to fetch tasks
- All output includes metadata for provenance
- See `docs/task.md` for task schema details