# 🚀 DekDataset - AI Dataset Generator

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.104+-green.svg)](https://fastapi.tiangolo.com/)

**ZOMBIT: DekDataset** is a powerful Thai AI/ML dataset generator that creates high-quality, diverse datasets for machine learning projects. Built with modern web technologies and supporting both cloud and local AI models.

## ✨ Features

### 🎯 Core Capabilities

- **Multi-Model Support**: DeepSeek API and Ollama local models
- **Web Interface**: Modern, responsive dashboard with real-time preview
- **Interactive Preview**: Table view with expandable content and CSV export
- **Quality Control**: Advanced filtering and duplicate detection
- **Multiple Export Formats**: JSON Lines, CSV, and ZIP
- **Task Management**: Create, edit, and manage custom generation tasks
- **Real-time Generation**: Live progress tracking with cancellation support

### 🔧 Technical Features

- **Schema Validation**: Ensure consistent data structure
- **Batch Processing**: Efficient generation for large datasets
- **Caching System**: Store and reuse generated datasets
- **Error Handling**: Robust timeout and retry mechanisms
- **Document Processing**: PDF OCR and analysis (optional)
- **API Integration**: RESTful API for programmatic access

## 🚀 Quick Start

### Prerequisites

- Python 3.8 or higher
- Node.js (for web interface)
- Git

### Installation

1. **Clone the Repository**

```bash
git clone https://github.com/zombitx64/DekDataset.git
cd DekDataset
```

2. **Install Dependencies**

```bash
pip install -r requirements.txt
```

3. **Environment Setup**

```bash
# Create .env file
cp .env.example .env

# Edit .env with your API keys
DEEPSEEK_API_KEY=your_deepseek_api_key_here
MISTRAL_API_KEY=your_mistral_api_key_here  # Optional for document processing
```

4. **Start the Web Server**

```bash
cd src/web
python app.py
```

5. **Access the Interface**
Open your browser and navigate to `http://localhost:8000`

## 🎮 Usage

### Web Interface

#### 1. **Task Management**

- Create custom tasks with schema definitions
- Import existing task configurations
- Edit task parameters and validation rules

#### 2. **Model Selection**

- **DeepSeek Models**: `deepseek-chat`, `deepseek-reasoner`
- **Ollama Models**: Any locally installed model (e.g., `llama2`, `qwen`)

#### 3. **Dataset Generation**

- Set the number of entries to generate
- Configure quality control settings
- Monitor generation progress in real-time
- Preview results in interactive table

#### 4. **Data Export**

- **JSON Lines**: Machine learning ready format
- **CSV**: Spreadsheet compatible
- **ZIP**: Complete dataset package

### API Usage

#### Generate Dataset

```python
import requests

response = requests.post('http://localhost:8000/api/generate', json={
    'task_id': 'sentiment-analysis',
    'count': 100,
    'model': 'deepseek-chat'
})

dataset = response.json()
```

#### Test Generation

```python
# Test with small sample
response = requests.post('http://localhost:8000/api/test-generation', json={
    'task_id': 'sentiment-analysis',
    'model': 'ollama:llama2'
})
```

### Command Line Interface

```bash
# Generate dataset using Python CLI
cd src/python
python generate_dataset.py --task sentiment-analysis --count 1000 --output dataset.jsonl
```

## 📊 Preview Table Features

### Interactive Data Visualization

- **Sortable Columns**: Click headers to sort data
- **Expandable Content**: Click cells to view full content
- **Entry Details**: Modal view for complete entry inspection
- **Copy Functions**: Copy individual entries or full JSON
- **Export Options**: Direct CSV export from table
- **View Modes**: Compact and full view toggles

### Quality Indicators

- **Entry Count**: Total generated entries
- **Quality Score**: Automated quality assessment
- **Source Information**: Model and generation metadata
- **Generation Time**: Performance metrics

## 🛠️ Configuration

### Task Schema Example

```json
{
  "id": "sentiment-analysis",
  "name": "Thai Sentiment Analysis",
  "description": "Generate Thai text with sentiment labels",
  "schema": {
    "fields": {
      "text": "string",
      "sentiment": "string",
      "confidence": "number"
    }
  },
  "validation_rules": {
    "text_min_length": 10,
    "sentiment_values": ["positive", "negative", "neutral"]
  }
}
```

### Quality Control Settings

```json
{
  "min_length": 10,
  "max_length": 1000,
  "similarity_threshold": 0.8,
  "enable_duplicate_detection": true,
  "required_fields": ["text", "label"]
}
```

## 🔧 Advanced Features

### Ollama Integration

1. **Install Ollama**

```bash
# Visit https://ollama.ai for installation
curl -fsSL https://ollama.ai/install.sh | sh
```

2. **Pull Models**

```bash
ollama pull llama2
ollama pull qwen:7b
```

3. **Use in DekDataset**

- Models automatically detected in web interface
- Select "Ollama" provider and choose your model

### Document Processing (Optional)

Install additional dependencies for PDF/OCR support:

```bash
pip install pdfplumber pytesseract pdf2image mistralai
```

### API Documentation

Access interactive API docs at `http://localhost:8000/docs`

## 📁 Project Structure

```
DekDataset/
├── src/
│   ├── web/                 # Web application
│   │   ├── app.py          # FastAPI main application
│   │   ├── static/         # CSS, JS, assets
│   │   ├── templates/      # HTML templates
│   │   └── document_understanding.py
│   └── python/             # Core Python modules
│       ├── generate_dataset.py
│       ├── banner.py
│       └── task_manager.py
├── cache/                  # Generated dataset cache
├── config/                 # Configuration files
├── tasks.json             # Task definitions
├── requirements.txt       # Python dependencies
└── README.md
```

## 🎯 Use Cases

### Machine Learning

- **Text Classification**: Sentiment analysis, topic classification
- **Question Answering**: FAQ datasets, knowledge bases
- **Named Entity Recognition**: Entity extraction training data
- **Translation**: Parallel text generation

### Research & Development

- **Prototype Testing**: Quick dataset generation for experiments
- **Benchmark Creation**: Standardized evaluation datasets
- **Data Augmentation**: Expand existing datasets
- **Model Training**: Custom training data generation

### Business Applications

- **Chatbot Training**: Conversation datasets
- **Content Generation**: Marketing copy, product descriptions
- **Data Migration**: Format conversion and cleaning
- **Quality Assurance**: Test data generation

## 🌟 Key Advantages

### 🚀 **Performance**

- Batch processing for large datasets
- Parallel generation with multiple models
- Intelligent caching and reuse
- Optimized memory usage

### 🎨 **User Experience**

- Intuitive web interface
- Real-time progress tracking
- Interactive data preview
- One-click export options

### 🔧 **Flexibility**

- Custom schema definitions
- Multiple model support
- Configurable quality controls
- Extensible architecture

### 🛡️ **Reliability**

- Robust error handling
- Timeout protection
- Quality validation
- Data integrity checks

## 🤝 Contributing

We welcome contributions! Please see our [Contributing Guidelines](CONTRIBUTING.md) for details.

### Development Setup

```bash
# Clone repository
git clone https://github.com/zombitx64/DekDataset.git
cd DekDataset

# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install development dependencies
pip install -r requirements.txt
pip install -r requirements-dev.txt

# Run tests
python -m pytest tests/
```

## 📝 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

- **DeepSeek AI** for providing powerful language models
- **Ollama** for local model inference capabilities
- **FastAPI** for the excellent web framework
- **Bootstrap** for responsive UI components

## 📞 Contact & Support

- **Author**: JonusNattapong | zombit
- **Email**: <zombitx64@gmail.com>
- **GitHub**: [@zombitx64](https://github.com/zombitx64) | [@JonusNattapong](https://github.com/JonusNattapong)
- **Issues**: [GitHub Issues](https://github.com/zombitx64/DekDataset/issues)

## 📈 Roadmap

### Upcoming Features

- [ ] **Batch Task Processing**: Run multiple tasks simultaneously
- [ ] **Custom Model Integration**: Support for more local models
- [ ] **Data Validation Tools**: Advanced quality checking
- [ ] **Export Templates**: Pre-configured export formats
- [ ] **Collaboration Features**: Multi-user task sharing
- [ ] **Performance Analytics**: Generation speed and quality metrics

### Future Enhancements

- [ ] **Cloud Deployment**: Docker and Kubernetes support
- [ ] **Database Integration**: PostgreSQL/MongoDB backends
- [ ] **Workflow Automation**: Scheduled generation tasks
- [ ] **Plugin System**: Custom generation modules
- [ ] **Multi-language Support**: Beyond Thai language

---

**Made with ❤️ by ZOMBIT | DekDataset Team**

*Empowering AI/ML development with high-quality Thai datasets*
