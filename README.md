# 🔥 DekDataset

<div align="center">

![Python](https://img.shields.io/badge/Python-3.10+-3776AB?style=for-the-badge&logo=python&logoColor=white)
![DeepSeek API](https://img.shields.io/badge/DeepSeek-API-FF6B6B?style=for-the-badge&logo=openai&logoColor=white)
![Mistral OCR](https://img.shields.io/badge/Mistral-OCR-4ECDC4?style=for-the-badge&logo=mistral&logoColor=white)
![License: MIT](https://img.shields.io/badge/License-MIT-2ECC71?style=for-the-badge)
![Status](https://img.shields.io/badge/Status-Production_Ready-27AE60?style=for-the-badge)

</div>

---

<div align="center">
  <h2>🎯 Advanced AI Dataset Generator for Thai & Multilingual Applications</h2>
  <p><i>Professional-grade synthetic data generation with OCR, NLP, Vision & Multi-modal capabilities</i></p>
  
  <b>⚡ Fast • 🎯 Accurate • 🔧 Extensible • 📊 Production-Ready</b>
</div>

---

## 🌟 Overview

**DekDataset** is a comprehensive open-source framework for generating high-quality AI/ML datasets in Thai and multiple languages. Designed for both research and enterprise applications, it seamlessly integrates OCR document processing, NLP tasks, computer vision, and multi-modal data generation into a unified, scalable platform.

### 🎯 Key Differentiators

- 🔍 **Advanced OCR Integration**: Extract text from PDFs, images, and documents using Mistral OCR API
- 🧠 **AI-Powered Generation**: Leverage DeepSeek LLM for context-aware synthetic data creation  
- 🌏 **Thai Language Optimized**: Native support for Thai NLP tasks and cultural context
- 🏭 **Enterprise-Ready**: Robust error handling, batch processing, and production deployment features

---

## 📑 Table of Contents

- [🌟 Overview](#-overview)
- [✨ Features](#-features)  
- [⚡ Quick Start](#-quick-start)
- [🔍 OCR Document Processing](#-ocr-document-processing)
- [🛠️ How It Works](#️-how-it-works)
- [📁 Project Structure](#-project-structure)
- [🖼️ Example Use Cases](#️-example-use-cases)
- [🧑‍💻 Technical Details](#-technical-details)
- [🙏 Credits & License](#-credits--license)

---

## ✨ Features

### 🔧 Core Capabilities

- **🎯 Unified Task Schema**: Support for NLP, Vision, OCR, and Multi-modal tasks with centralized schema management
- **🤖 Automatic Prompting**: Generate optimized prompts for LLMs (DeepSeek, OpenAI, etc.) automatically
- **⚡ Batch Generation**: Advanced batch processing with error recovery, quota management, and intelligent retry mechanisms
- **📊 Data Validation & Metadata**: Complete validation, deduplication, enrichment, label balancing, and metadata export
- **💾 Flexible Output**: Export to JSONL, Parquet, Arrow, CSV formats compatible with HuggingFace, PyArrow, and Pandas

### 🔍 Advanced OCR & Document Processing

- **📄 PDF Processing**: Extract text from multi-page PDF documents using Poppler integration
- **🖼️ Image OCR**: Process JPG, PNG, JPEG images with high-accuracy text recognition
- **🌐 URL Support**: Direct processing of remote documents and images via URLs
- **🎯 Context-Aware Generation**: Use extracted OCR text as context for LLM-based dataset creation

### 🌐 Web Integration & Media

- **🔍 Web Scraping**: Download images from Bing, DuckDuckGo, Pexels, Pixabay with metadata
- **🎨 AI Image Generation**: Integrated support for AI-generated images and captions
- **📱 Multi-modal**: Combine text, images, and metadata for comprehensive datasets

### 🛡️ Enterprise Features

- **🔒 Robust & Reproducible**: Comprehensive error handling, logging, retry mechanisms, and fallback strategies
- **📈 Scalable Architecture**: Support for large-scale batch processing and distributed generation
- **🔧 Extensible Design**: Easy task/schema customization via `tasks.json` or REST API
- **📋 Production Monitoring**: Built-in monitoring, logging, and performance tracking

---

## 📦 Installation

1. Install Python dependencies:
```bash
pip install -r requirements.txt

# Install additional required packages
pip install playwright bs4 nest-asyncio crawl4ai pandas
```

2. Install Playwright browsers:
```bash
playwright install
```

3. Install Poppler for PDF processing:
```bash
# Windows
# Download and extract poppler-windows.zip from the repository
unzip poppler-windows.zip -d ./poppler-local

# Linux
sudo apt-get update && sudo apt-get install -y poppler-utils

# Mac
brew install poppler
```

4. Verify OCR Installation:
```bash
# Test OCR setup
python src/python/ocr_utils.py --test

# Test PDF processing
python src/python/ocr_utils.py --test-pdf sample.pdf
```

## ⚡ Quick Start Examples

### Social Media Comment Extraction

Extract comments from various social media platforms:

```bash
# Extract Twitter/X comments
python src/python/get_comments.py x https://x.com/username/status/123456 --max_results 100

# Extract Facebook comments
python src/python/get_comments.py facebook https://facebook.com/post/123456 --max_results 50

# Extract YouTube comments
python src/python/get_comments.py youtube https://youtube.com/watch?v=videoId --max_results 200
```

### Save & Export Options

Save extracted comments in different formats:

```bash
# Save as JSONL (default)
python src/python/get_comments.py x https://x.com/post/123 --output comments.jsonl

# Save as CSV
python src/python/get_comments.py x https://x.com/post/123 --format csv --output comments.csv

# Save with sentiment analysis
python src/python/get_comments.py x https://x.com/post/123 --sentiment
```

### Advanced Features

```bash
# Extract with spam filtering
python src/python/get_comments.py x https://x.com/post/123 --filter-spam

# Extract with both sentiment and spam filtering
python src/python/get_comments.py x https://x.com/post/123 --sentiment --filter-spam

# Extract from local file
python src/python/get_comments.py file path/to/comments.jsonl
```

## 🔍 Troubleshooting

### Common Issues

1. Playwright Installation
```bash
# If you see browser-related errors, try:
playwright install
playwright install chromium
```

2. SSL Certificate Errors
```bash
# Set environment variable to ignore SSL
set PYTHONHTTPSVERIFY=0  # Windows
export PYTHONHTTPSVERIFY=0  # Linux/Mac
```

3. Rate Limiting
```bash
# Use delay between requests
python src/python/get_comments.py x https://x.com/post/123 --delay 5
```

### Platform-Specific Notes

#### Twitter/X
- Uses Playwright for extraction with fallback to Crawl4AI
- Handles both old twitter.com and new x.com URLs
- Supports thread extraction

#### Facebook
- Requires public posts
- May need longer delays between requests
- Supports post and comment extraction

#### YouTube
- Requires yt-dlp package
- Supports both video IDs and full URLs
- Can extract comments and replies

## 🌟 Usage

### 1. Installation & Setup

```bash
# Clone repository
git clone https://github.com/yourusername/DekDataset.git
cd DekDataset

# Install dependencies
pip install -r requirements.txt

# Install additional required packages
pip install playwright bs4 nest-asyncio crawl4ai pandas

# Install Playwright browsers
playwright install
```

### 2. Configure API Keys

Create a `.env` file in the project root:

```env
# Required: DeepSeek API for LLM generation
DEEPSEEK_API_KEY=your_deepseek_api_key

# Required: Mistral API for OCR processing
MISTRAL_API_KEY=your_mistral_api_key

# Optional: Image download services
PEXELS_API_KEY=your_pexels_api_key
PIXABAY_API_KEY=your_pixabay_api_key
```

### 3. Basic Dataset Generation

```bash
# Generate NLP dataset
python src/python/generate_dataset.py sentiment_analysis 100 --format jsonl

# Generate dataset from PDF document
python src/python/generate_dataset.py primary_school_knowledge 50 --input-file document.pdf

# Generate with custom settings
python src/python/generate_dataset.py medical_text_summarization 25 --delay 2 --format parquet

# Generate with DeepSeek API
python src/python/generate_dataset.py --model deepseek-chat --temperature 0.7 --task custom_task 100

# Process large PDF documents
python src/python/ocr_utils.py --input large_document.pdf --batch-size 10 --output extracted_text.txt

# Generate dataset with specific language
python src/python/generate_dataset.py sentiment_analysis 100 --language th --format jsonl

# Export and merge datasets
python src/python/data_utils.py merge dataset1.jsonl dataset2.jsonl --output merged.jsonl
python src/python/data_utils.py convert input.jsonl --format parquet --output final.parquet
```

### 4. Advanced Usage

```bash
# Start task definitions API server
python src/python/task_definitions_api.py

# DeepSeek API Advanced Usage
# Generate with specific model parameters
python src/python/generate_dataset.py custom_task 100 \
  --model deepseek-chat \
  --temperature 0.7 \
  --top-p 0.9 \
  --frequency-penalty 0.3

# Batch generation with error recovery
python src/python/generate_dataset.py sentiment_analysis 1000 \
  --batch-size 50 \
  --retry-attempts 3 \
  --delay 2

# Dataset Management
# Convert between formats
python src/python/data_utils.py convert dataset.jsonl \
  --format parquet \
  --compression snappy \
  --output processed.parquet

# Filter and clean dataset
python src/python/data_utils.py clean dataset.jsonl \
  --remove-duplicates \
  --min-length 10 \
  --language th \
  --output cleaned.jsonl

# Merge multiple datasets
python src/python/data_utils.py merge \
  dataset1.jsonl dataset2.jsonl dataset3.jsonl \
  --output merged.jsonl \
  --format parquet

# Split dataset
python src/python/data_utils.py split dataset.jsonl \
  --train 0.8 \
  --val 0.1 \
  --test 0.1

# Dataset Analysis
python src/python/data_utils.py analyze dataset.jsonl \
  --show-stats \
  --plot-distributions \
  --export-report report.pdf
```

---

## 🔍 OCR Document Processing

### Supported Formats

- **📄 PDF Documents**: Multi-page PDFs with automatic page-by-page processing
- **🖼️ Images**: JPG, PNG, JPEG files with high-accuracy text extraction
- **🌐 Remote Files**: Direct URL processing for online documents and images

### OCR Workflow Examples

#### Extract Text Only

```bash
# Extract from local PDF
python src/python/generate_dataset.py --input-file document.pdf

# Extract from image
python src/python/generate_dataset.py --input-file image.png

# Extract from URL
python src/python/generate_dataset.py --input-file https://example.com/document.pdf
```

#### Generate Dataset from Documents

```bash
# Create sentiment analysis dataset from PDF
python src/python/generate_dataset.py sentiment_analysis 20 --input-file textbook.pdf

# Generate medical dataset from research paper
python src/python/generate_dataset.py medical_text_summarization 15 --input-file research_paper.pdf

# Create Q&A dataset from educational content
python src/python/generate_dataset.py primary_school_knowledge 30 --input-file educational_material.png
```

### OCR Technical Features

- **🔧 Automatic Poppler Integration**: Built-in PDF processing without system dependencies
- **🎯 High-Accuracy Text Extraction**: Mistral OCR API with advanced text recognition
- **📊 Context-Aware Processing**: Extracted text feeds directly into LLM generation pipeline
- **⚡ Batch Document Processing**: Handle multiple documents and large files efficiently

---

## 🛠️ How It Works

### Generation Pipeline

1. **📋 Task Selection**: Choose from predefined tasks or create custom schemas
2. **🔍 Document Processing**: (Optional) Extract text from PDFs/images using OCR
3. **🤖 Prompt Generation**: Automatically generate optimized prompts for LLM
4. **⚡ Batch Generation**: Create synthetic data in intelligent batches with error recovery
5. **✅ Validation & Processing**: Validate, deduplicate, enrich, and balance generated data
6. **💾 Export & Metadata**: Export to multiple formats with comprehensive metadata

### Architecture Overview

```mermaid
graph TD
    A[Input Documents/Tasks] --> B[OCR Processing]
    B --> C[Context Extraction]
    C --> D[LLM Prompt Generation]
    D --> E[Batch Generation]
    E --> F[Data Validation]
    F --> G[Export & Metadata]
    G --> H[Multiple Output Formats]
```

---

## 📁 Project Structure

```text
DekDataset/
├── 📁 src/
│   ├── 🦀 main.rs, models.rs, api_client.rs, generator.rs, banner.rs
│   └── 🐍 python/
│       ├── generate_dataset.py      # Main generation script
│       ├── ocr_utils.py            # OCR processing module
│       ├── task_definitions.py     # Task schema management
│       └── task_definitions_api.py # REST API server
├── 📁 data/
│   ├── output/                     # Generated datasets
│   └── pdf/                        # Sample PDF documents
├── 📁 poppler-local/               # Portable PDF processing
├── 📁 docs/                        # Documentation
├── 📄 requirements.txt             # Python dependencies
└── 📄 README.md                    # This file
```

---

## 🖼️ Example Use Cases

### 1. Medical AI Dataset Creation

```bash
# Generate comprehensive medical benchmark dataset
python src/python/generate_dataset.py medical_benchmark 1000 --format jsonl --delay 1

# Create medical Q&A from research papers
python src/python/generate_dataset.py medical_text_summarization 200 --input-file medical_research.pdf
```

**Output**: High-quality medical datasets for training AI models, medical Q&A systems, or clinical decision support tools.

### 2. Educational Content Generation

```bash
# Thai primary school knowledge base
python src/python/generate_dataset.py primary_school_knowledge 500 --input-file thai_textbook.pdf

# Generate educational Q&A pairs
python src/python/generate_dataset.py qa_generation 300 --input-file educational_content.pdf
```

**Output**: Educational datasets for Thai language learning applications, automated tutoring systems, and knowledge assessment tools.

### 3. Thai NLP Model Training

```bash
# Sentiment analysis dataset in Thai
python src/python/generate_dataset.py sentiment_analysis 2000 --format parquet

# Text classification for Thai content
python src/python/generate_dataset.py text_classification 1500 --delay 2
```

**Output**: Large-scale Thai language datasets for training sentiment analysis, text classification, and other NLP models.

### 4. Document Processing Automation

```bash
# Process multiple PDF documents
python src/python/generate_dataset.py document_summarization 100 --input-file legal_documents.pdf

# Extract and analyze business reports
python src/python/generate_dataset.py business_analysis 75 --input-file quarterly_report.pdf
```

**Output**: Automated document analysis and summarization datasets for enterprise applications.

---

## 🧑‍💻 Technical Details

### Technology Stack

- **🐍 Python 3.10+**: Core processing and API integration
- **🦀 Rust Components**: High-performance data processing modules
- **🤖 DeepSeek API**: Advanced LLM for synthetic data generation
- **🔍 Mistral OCR**: State-of-the-art optical character recognition
- **📄 Poppler**: Robust PDF processing and conversion
- **⚡ FastAPI**: REST API for task management and automation

### Key Components

- **🔧 OCR Engine**: Mistral API integration with local Poppler support
- **📊 Data Pipeline**: Robust batch processing with error recovery
- **🎯 Task Management**: Flexible schema system with custom task support
- **💾 Export System**: Multi-format output with metadata preservation
- **🛡️ Error Handling**: Comprehensive logging and retry mechanisms

### Performance & Scalability

- **⚡ Batch Processing**: Intelligent batching for optimal API usage
- **🔄 Error Recovery**: Automatic retry with exponential backoff
- **📈 Scalable Architecture**: Support for distributed processing
- **📊 Monitoring**: Built-in performance tracking and logging

### Integration Capabilities

- **🤗 HuggingFace**: Direct compatibility with Datasets and Transformers
- **📊 PyArrow/Pandas**: Native support for data analysis workflows
- **☁️ Cloud Deployment**: Docker and cloud-ready architecture
- **🔌 API Integration**: RESTful API for external system integration

---

## 🙏 Credits & License

### Acknowledgments

- **🏢 DeepSeek**: Advanced LLM capabilities and API integration
- **🔍 Mistral AI**: High-quality OCR processing and text extraction
- **🤗 HuggingFace**: Ecosystem integration and dataset standards
- **📊 Apache Arrow**: High-performance data processing and storage
- **🌐 Open Source Community**: Various libraries and tools that make this project possible

### License & Usage

- **📄 License**: MIT License - see [LICENSE](LICENSE) for details
- **💼 Commercial Use**: Permitted under MIT license terms
- **🔧 Contributions**: Welcome via pull requests and issues
- **📞 Support**: Community support through GitHub issues

### Project Information

- **👨‍💻 Developed by**: ZOMBIT Team
- **🌐 Repository**: [github.com/JonusNattapong/DekDataset](https://github.com/JonusNattapong/DekDataset)
- **📧 Contact**: zombitx64@gmail.com
- **🎯 Version**: 2025.05 - Production Ready
- **🏷️ Tags**: AI, ML, Dataset, Thai NLP, OCR, Synthetic Data, DeepSeek, Python

---

<div align="center">

**⭐ Star this repository if you find it useful!**

[![GitHub stars](https://img.shields.io/github/stars/JonusNattapong/DekDataset?style=social)](https://github.com/JonusNattapong/DekDataset/stargazers)
[![GitHub forks](https://img.shields.io/github/forks/JonusNattapong/DekDataset?style=social)](https://github.com/JonusNattapong/DekDataset/network/members)
[![GitHub issues](https://img.shields.io/github/issues/JonusNattapong/DekDataset)](https://github.com/JonusNattapong/DekDataset/issues)

</div>
