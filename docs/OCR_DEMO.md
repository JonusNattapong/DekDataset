# OCR Integration Demo for DekDataset

The OCR functionality has been successfully integrated into DekDataset. Here's how to use it:

## 🚀 Quick Start

### 1. Extract text from a PDF or image and generate a dataset:

```bash
# Generate dataset using OCR-extracted text as context
python generate_dataset.py sentiment_analysis 50 --input-file "path/to/document.pdf"

# With remote URL (e.g., arXiv paper)
python generate_dataset.py text_classification 100 --input-file "https://arxiv.org/pdf/1706.03762.pdf"

# Generate medical Q&A dataset from medical textbook PDF
python generate_dataset.py medical_qa 30 --input-file "medical_textbook.pdf" --domain medical
```

### 2. Use OCR utility directly:

```python
from ocr_utils import extract_text_from_file
import os

# Load API key
api_key = os.getenv('MISTRAL_API_KEY')

# Extract from local PDF
text = extract_text_from_file('document.pdf', api_key)

# Extract from remote URL
text = extract_text_from_file('https://example.com/document.pdf', api_key)

# Extract from image
text = extract_text_from_file('screenshot.png', api_key)
```

## 📁 Supported File Types

- **PDFs**: Local and remote PDF files
- **Images**: JPG, PNG formats
- **URLs**: Direct links to documents (arXiv, research papers, etc.)

## 🔧 How It Works

1. **URL Detection**: Automatically detects if input is a URL or local file
2. **PDF Processing**: Converts PDF pages to images using pdf2image
3. **OCR Extraction**: Uses Mistral OCR API to extract text
4. **Context Integration**: Extracted text becomes context for LLM dataset generation

## 🎯 Use Cases

### Academic Research
```bash
# Generate Q&A dataset from research paper
python generate_dataset.py academic_qa 25 --input-file "https://arxiv.org/pdf/2301.00000.pdf"
```

### Medical Documentation
```bash
# Create medical text classification dataset
python generate_dataset.py medical_classification 40 --input-file "medical_report.pdf" --domain medical
```

### Educational Content
```bash
# Generate educational dataset from textbook
python generate_dataset.py educational_qa 60 --input-file "textbook_chapter.pdf" --lang th
```

### Business Documents
```bash
# Create sentiment analysis dataset from business reports
python generate_dataset.py sentiment_analysis 35 --input-file "business_report.pdf"
```

## 🔑 Environment Setup

Make sure your `.env` file contains:
```
MISTRAL_API_KEY=your_actual_mistral_api_key_here
DEEPSEEK_API_KEY=your_deepseek_api_key_here
```

## 📊 Output

The OCR integration works seamlessly with all existing DekDataset features:

- **Multiple formats**: JSON, JSONL, CSV, Parquet
- **Data splitting**: Train/validation/test splits
- **Thai language support**: Full Unicode and normalization
- **Hugging Face export**: Direct upload to HF Hub
- **Analysis tools**: Built-in dataset analysis and visualization

## 🐛 Troubleshooting

### PDF Processing Issues
If you get poppler errors:
```bash
# Windows (recommended)
conda install -c conda-forge poppler

# Or download from: https://github.com/oschwartz10612/poppler-windows/releases
```

### API Rate Limits
Add delays between API calls:
```bash
python generate_dataset.py task 100 --input-file "doc.pdf" --delay 3
```

### Large PDF Files
For multi-page PDFs, the system processes each page separately and combines the results.

## 🎉 Example Output

```json
{
  "id": "ocr_001",
  "text": "Based on the extracted content about machine learning algorithms...",
  "label": "positive",
  "source": "OCR-extracted from research_paper.pdf",
  "metadata": {
    "generated_at": "2025-05-25T10:30:00Z",
    "model": "deepseek-chat",
    "ocr_source": "research_paper.pdf",
    "task": "sentiment_analysis"
  }
}
```

## 🔄 Integration Status

✅ **Complete**: OCR module separated into `ocr_utils.py`  
✅ **Complete**: CLI integration with `--input-file` parameter  
✅ **Complete**: Support for both local files and remote URLs  
✅ **Complete**: Error handling and debug logging  
✅ **Complete**: Environment configuration  
✅ **Ready**: Full integration with dataset generation pipeline  

The OCR functionality is now ready for production use in DekDataset!
