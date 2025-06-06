"""
Document Understanding Module for DekDataset
Provides OCR, PDF processing, and document analysis capabilities
"""

import os
import io
import sys
import base64
import logging
import tempfile
import requests
import re
import json
from pathlib import Path
from typing import List, Dict, Any, Optional, Union
from dataclasses import dataclass, field
from datetime import datetime

# PDF and image processing
try:
    import pdfplumber
    import pytesseract
    from PIL import Image
    from pdf2image import convert_from_path
    PDF_AVAILABLE = True
except ImportError as e:
    logging.warning(f"PDF processing libraries not available: {e}")
    PDF_AVAILABLE = False

# Mistral AI for advanced OCR
try:
    from mistralai import Mistral
    MISTRAL_AVAILABLE = True
except ImportError:
    logging.warning("Mistral AI not available - advanced OCR features disabled")
    MISTRAL_AVAILABLE = False

logger = logging.getLogger(__name__)

@dataclass
class BBoxImageAnnotation:
    """Bounding box annotation for image elements"""
    x: float
    y: float
    width: float
    height: float
    label: str
    confidence: float = 0.0

@dataclass
class DocumentAnnotation:
    """Document analysis annotation"""
    text: str
    page_number: int = 1
    bbox: Optional[BBoxImageAnnotation] = None
    confidence: float = 0.0
    metadata: Dict[str, Any] = field(default_factory=dict)

    def __post_init__(self):
        if self.metadata is None:
            self.metadata = {}

class DocumentUnderstanding:
    """Advanced document understanding with OCR and analysis"""
    
    def __init__(self, mistral_api_key: Optional[str] = None, ocr_lang: str = 'tha+eng'):
        self.mistral_api_key = mistral_api_key or os.getenv('MISTRAL_API_KEY')
        self.ocr_lang = ocr_lang
        self.mistral_client = None
        
        if self.mistral_api_key and MISTRAL_AVAILABLE:
            try:
                self.mistral_client = Mistral(api_key=self.mistral_api_key)
            except Exception as e:
                logger.warning(f"Failed to initialize Mistral client: {e}")
        
        # Configure tesseract for Windows if available
        if os.name == "nt" and PDF_AVAILABLE:
            tesseract_path = r"C:\Program Files\Tesseract-OCR\tesseract.exe"
            if os.path.exists(tesseract_path):
                pytesseract.pytesseract.tesseract_cmd = tesseract_path

    def process_document(self, file_path: Union[str, Path]) -> List[DocumentAnnotation]:
        """Process any supported document type"""
        file_path = Path(file_path)
        
        if str(file_path).startswith(('http://', 'https://')):
            return self.process_url(str(file_path))
        
        if not file_path.exists():
            raise FileNotFoundError(f"File not found: {file_path}")
        
        # Determine processing method based on file extension
        extension = file_path.suffix.lower()
        
        if extension == '.pdf':
            return self.extract_text(file_path)
        elif extension in ['.jpg', '.jpeg', '.png', '.gif', '.bmp', '.tiff']:
            return self.extract_text(file_path)
        else:
            raise ValueError(f"Unsupported file type: {extension}")

    def extract_text(self, file_path: Union[str, Path], ocr_lang: Optional[str] = None) -> List[DocumentAnnotation]:
        """Unified entry: auto-detect file type and extract text with best method"""
        file_path = Path(file_path)
        ext = file_path.suffix.lower()
        lang = ocr_lang or self.ocr_lang
        
        if ext == '.pdf':
            return self.ocr_pdf(file_path, lang)
        elif ext in ['.jpg', '.jpeg', '.png', '.bmp', '.tiff']:
            return self.ocr_image(file_path, lang)
        else:
            raise ValueError(f"Unsupported file type: {ext}")

    def ocr_pdf(self, pdf_path: Union[str, Path], lang: str = 'tha+eng') -> List[DocumentAnnotation]:
        """Convert PDF pages to images, then OCR each page"""
        pdf_path = Path(pdf_path)
        if not pdf_path.exists():
            raise FileNotFoundError(f"PDF file not found: {pdf_path}")
        
        try:
            # First try text extraction if it's a text-based PDF
            if self.is_text_pdf(pdf_path):
                return self.extract_text_from_pdf(pdf_path)
            
            # If no text or image-based PDF, use OCR
            images = convert_from_path(str(pdf_path))
        except Exception as e:
            logger.error(f"Error converting PDF to images: {e}")
            return []
        
        annotations = []
        for i, img in enumerate(images, 1):
            try:
                text = self._extract_with_mistral_image(img) if self.mistral_client else self._extract_with_tesseract_image(img, lang)
                if text:
                    annotation = DocumentAnnotation(
                        text=text,
                        page_number=i,
                        confidence=0.8,
                        metadata={'extraction_method': 'mistral' if self.mistral_client else 'tesseract'}
                    )
                    annotations.append(annotation)
            except Exception as e:
                logger.error(f"Error extracting text from page {i}: {e}")
        
        return annotations

    def ocr_image(self, image_path: Union[str, Path], lang: str = 'tha+eng') -> List[DocumentAnnotation]:
        """OCR for a single image file"""
        image_path = Path(image_path)
        if not image_path.exists():
            raise FileNotFoundError(f"Image file not found: {image_path}")
        
        try:
            with Image.open(image_path) as img:
                text = self._extract_with_mistral_image(img) if self.mistral_client else self._extract_with_tesseract_image(img, lang)
                if text:
                    return [DocumentAnnotation(
                        text=text,
                        page_number=1,
                        confidence=0.8,
                        metadata={'extraction_method': 'mistral' if self.mistral_client else 'tesseract'}
                    )]
        except Exception as e:
            logger.error(f"Error extracting text from image: {e}")
        
        return []

    def is_text_pdf(self, pdf_path: Union[str, Path]) -> bool:
        """Return True if PDF is text-based, else False (scan/image-based)"""
        try:
            with pdfplumber.open(pdf_path) as pdf:
                for page in pdf.pages[:3]:  # Check first 3 pages
                    text = page.extract_text()
                    if text and text.strip():
                        return True
        except Exception as e:
            logger.error(f"Error checking PDF type: {e}")
        return False

    def extract_text_from_pdf(self, pdf_path: Union[str, Path]) -> List[DocumentAnnotation]:
        """Extract text from PDF using pdfplumber (for text-based PDFs)"""
        pdf_path = Path(pdf_path)
        if not pdf_path.exists():
            raise FileNotFoundError(f"PDF file not found: {pdf_path}")
        
        annotations = []
        try:
            with pdfplumber.open(pdf_path) as pdf:
                for i, page in enumerate(pdf.pages, 1):
                    text = page.extract_text()
                    if text and text.strip():
                        annotation = DocumentAnnotation(
                            text=text,
                            page_number=i,
                            confidence=0.9,
                            metadata={'extraction_method': 'pdfplumber'}
                        )
                        annotations.append(annotation)
        except Exception as e:
            logger.error(f"Error extracting text from PDF: {e}")
            raise
        
        return annotations

    def _extract_with_mistral_image(self, pil_image) -> str:
        """Extract text from PIL image using Mistral Vision API"""
        if not self.mistral_client:
            return ""
        
        try:
            buffered = io.BytesIO()
            pil_image.save(buffered, format="JPEG")
            image_base64 = base64.b64encode(buffered.getvalue()).decode('utf-8')
            
            messages = [
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": "Extract all text from this image. Return only the extracted text, no additional commentary."},
                        {"type": "image_url", "image_url": f"data:image/jpeg;base64,{image_base64}"}
                    ]
                }
            ]
            
            response = self.mistral_client.chat.complete(
                model="pixtral-12b-2409",
                messages=messages
            )
            
            return response.choices[0].message.content if response.choices else ""
        except Exception as e:
            logger.error(f"Mistral extraction failed: {e}")
            return ""

    def _extract_with_tesseract_image(self, pil_image, lang='tha+eng') -> str:
        """Extract text using Tesseract OCR"""
        try:
            return pytesseract.image_to_string(pil_image, lang=lang)
        except Exception as e:
            logger.error(f"Tesseract extraction failed: {e}")
            return ""

    def analyze_document_structure(self, annotations: List[DocumentAnnotation]) -> Dict[str, Any]:
        """Analyze document structure and provide insights"""
        if not annotations:
            return {'error': 'No annotations provided'}
        
        total_text = ' '.join(ann.text for ann in annotations)
        
        analysis = {
            'total_pages': len(set(ann.page_number for ann in annotations)),
            'total_characters': len(total_text),
            'total_words': len(total_text.split()),
            'average_confidence': sum(ann.confidence for ann in annotations) / len(annotations),
            'pages_with_content': len(annotations),
            'extraction_methods': list(set(ann.metadata.get('extraction_method', 'unknown') for ann in annotations)),
            'languages_detected': self._detect_languages(total_text),
            'content_summary': {
                'has_thai_content': self._has_thai_content(total_text),
                'has_english_content': self._has_english_content(total_text),
                'estimated_reading_time': max(1, len(total_text.split()) // 200)
            }
        }
        
        return analysis

    def get_combined_text(self, annotations: List[DocumentAnnotation]) -> str:
        """Get all text combined from annotations"""
        if not annotations:
            return ""
        
        sorted_annotations = sorted(annotations, key=lambda x: x.page_number)
        
        combined_text = ""
        current_page = 0
        
        for ann in sorted_annotations:
            if ann.page_number != current_page:
                if current_page > 0:
                    combined_text += "\n\n--- Page Break ---\n\n"
                current_page = ann.page_number
            combined_text += ann.text + "\n"
        
        return combined_text.strip()

    def export_annotations(self, annotations: List[DocumentAnnotation], format: str = 'json') -> Union[str, Dict]:
        """Export annotations in various formats"""
        if format.lower() == 'json':
            return {
                'annotations': [
                    {
                        'text': ann.text,
                        'page_number': ann.page_number,
                        'confidence': ann.confidence,
                        'metadata': ann.metadata
                    } for ann in annotations
                ],
                'analysis': self.analyze_document_structure(annotations)
            }
        elif format.lower() == 'text':
            return self.get_combined_text(annotations)
        else:
            raise ValueError(f"Unsupported export format: {format}")

    def to_dataset_format(self, annotations: List[DocumentAnnotation], dataset_type: str = 'text_generation') -> List[Dict[str, Any]]:
        """Convert extracted text to dataset-ready format"""
        dataset_entries = []
        
        if dataset_type == 'text_generation':
            dataset_entries = self._to_text_generation_format(annotations)
        elif dataset_type == 'question_answer':
            dataset_entries = self._to_qa_format(annotations)
        elif dataset_type == 'classification':
            dataset_entries = self._to_classification_format(annotations)
        elif dataset_type == 'translation':
            dataset_entries = self._to_translation_format(annotations)
        elif dataset_type == 'summarization':
            dataset_entries = self._to_summarization_format(annotations)
        else:
            raise ValueError(f"Unsupported dataset type: {dataset_type}")
        
        return dataset_entries

    def _to_text_generation_format(self, annotations: List[DocumentAnnotation]) -> List[Dict[str, Any]]:
        """Convert to text generation dataset format"""
        entries = []
        combined_text = self.get_combined_text(annotations)
        
        paragraphs = [p.strip() for p in combined_text.split('\n\n') if p.strip()]
        
        for i, paragraph in enumerate(paragraphs):
            if len(paragraph) > 50:
                entries.append({
                    'input': f"สร้างข้อความตามหัวข้อ: {paragraph[:100]}...",
                    'output': paragraph,
                    'source': 'pdf_extraction'
                })
        
        return entries

    def _to_qa_format(self, annotations: List[DocumentAnnotation]) -> List[Dict[str, Any]]:
        """Convert to question-answer dataset format"""
        entries = []
        combined_text = self.get_combined_text(annotations)
        
        sections = [s.strip() for s in combined_text.split('\n\n') if s.strip()]
        
        for i, section in enumerate(sections):
            if len(section) > 100:
                entries.append({
                    'question': f"บอกเกี่ยวกับ {section[:50]}... ?",
                    'answer': section,
                    'source': 'pdf_extraction'
                })
        
        return entries

    def _to_classification_format(self, annotations: List[DocumentAnnotation]) -> List[Dict[str, Any]]:
        """Convert to text classification dataset format"""
        entries = []
        combined_text = self.get_combined_text(annotations)
        
        doc_type = self._detect_document_type(combined_text)
        languages = self._detect_languages(combined_text)
        
        paragraphs = [p.strip() for p in combined_text.split('\n\n') if p.strip()]
        
        for i, paragraph in enumerate(paragraphs):
            if len(paragraph) > 30:
                entries.append({
                    'text': paragraph,
                    'label': doc_type,
                    'language': languages[0] if languages else 'unknown',
                    'source': 'pdf_extraction'
                })
        
        return entries

    def _to_translation_format(self, annotations: List[DocumentAnnotation]) -> List[Dict[str, Any]]:
        """Convert to translation dataset format"""
        entries = []
        combined_text = self.get_combined_text(annotations)
        
        has_thai = self._has_thai_content(combined_text)
        has_english = self._has_english_content(combined_text)
        
        if has_thai and has_english:
            sentences = combined_text.split('.')
            for sentence in sentences:
                if len(sentence.strip()) > 20:
                    entries.append({
                        'source_text': sentence.strip(),
                        'target_text': f"แปล: {sentence.strip()}",
                        'source_lang': 'auto',
                        'target_lang': 'thai'
                    })
        
        return entries

    def _to_summarization_format(self, annotations: List[DocumentAnnotation]) -> List[Dict[str, Any]]:
        """Convert to text summarization dataset format"""
        entries = []
        combined_text = self.get_combined_text(annotations)
        
        chunks = self._split_text_into_chunks(combined_text, max_chunk_size=1000)
        
        for i, chunk in enumerate(chunks):
            if len(chunk) > 200:
                summary = chunk[:200] + "..."
                entries.append({
                    'document': chunk,
                    'summary': summary,
                    'source': 'pdf_extraction'
                })
        
        return entries

    def _detect_languages(self, text: str) -> List[str]:
        """Detect languages in text"""
        languages = []
        
        if self._has_thai_content(text):
            languages.append('thai')
        
        if self._has_english_content(text):
            languages.append('english')
        
        return languages if languages else ['unknown']

    def _has_thai_content(self, text: str) -> bool:
        """Check if text contains Thai characters"""
        thai_range = range(0x0E00, 0x0E7F)
        return any(ord(char) in thai_range for char in text)

    def _has_english_content(self, text: str) -> bool:
        """Check if text contains English characters"""
        english_chars = sum(1 for char in text if char.isalpha() and ord(char) < 128)
        return english_chars > len(text) * 0.1

    def _detect_document_type(self, text: str) -> str:
        """Detect document type based on content"""
        text_lower = text.lower()
        
        if any(keyword in text_lower for keyword in ['abstract', 'บทคัดย่อ', 'reference', 'bibliography', 'งานวิจัย']):
            return 'academic'
        elif any(keyword in text_lower for keyword in ['สัญญา', 'ข้อกำหนด', 'กฎหมาย', 'มาตรา', 'contract']):
            return 'legal'
        elif any(keyword in text_lower for keyword in ['ข่าว', 'รายงาน', 'news', 'report', 'breaking']):
            return 'news'
        elif any(keyword in text_lower for keyword in ['api', 'function', 'algorithm', 'technical', 'specification']):
            return 'technical'
        elif any(keyword in text_lower for keyword in ['บทเรียน', 'การศึกษา', 'lesson', 'chapter', 'course']):
            return 'educational'
        else:
            return 'general'

    def _split_text_into_chunks(self, text: str, max_chunk_size: int = 1000) -> List[str]:
        """Split text into chunks for processing"""
        chunks = []
        sentences = text.split('.')
        current_chunk = ""
        
        for sentence in sentences:
            if len(current_chunk) + len(sentence) > max_chunk_size:
                if current_chunk:
                    chunks.append(current_chunk.strip())
                current_chunk = sentence
            else:
                current_chunk += sentence + "."
        
        if current_chunk:
            chunks.append(current_chunk.strip())
        
        return chunks

    def process_url(self, url: str) -> List[DocumentAnnotation]:
        """Process document from URL"""
        try:
            response = requests.get(url, timeout=30)
            response.raise_for_status()
            
            with tempfile.NamedTemporaryFile(delete=False, suffix='.pdf') as tmp:
                tmp.write(response.content)
                tmp_path = Path(tmp.name)
            
            try:
                return self.process_document(tmp_path)
            finally:
                tmp_path.unlink(missing_ok=True)
        except Exception as e:
            logger.error(f"Error processing URL: {e}")
            return []

# Utility functions for backward compatibility
def extract_text_from_pdf(pdf_path: Union[str, Path]) -> str:
    """Simple PDF text extraction function"""
    doc_processor = DocumentUnderstanding()
    annotations = doc_processor.extract_text_from_pdf(pdf_path)
    return doc_processor.get_combined_text(annotations)

def extract_text_from_image(image_path: Union[str, Path]) -> str:
    """Simple image text extraction function"""
    doc_processor = DocumentUnderstanding()
    annotations = doc_processor.ocr_image(image_path)
    return doc_processor.get_combined_text(annotations)

def process_document_file(file_path: Union[str, Path]) -> str:
    """Simple document processing function"""
    doc_processor = DocumentUnderstanding()
    annotations = doc_processor.process_document(file_path)
    return doc_processor.get_combined_text(annotations)

# Example usage and testing
if __name__ == "__main__":
    # Test the document understanding system
    doc_processor = DocumentUnderstanding()
    
    print("🔍 DekDataset Document Understanding System")
    print("=" * 50)
    
    # Test with sample files if available
    test_files = [
        'sample.pdf',
        'test_image.png',
        'document.jpg'
    ]
    
    for test_file in test_files:
        if Path(test_file).exists():
            print(f"\nTesting with {test_file}...")
            try:
                annotations = doc_processor.process_document(test_file)
                print(f"Extracted {len(annotations)} annotations")
            except Exception as e:
                print(f"Error processing {test_file}: {e}")
    
    print("\n🎉 Document Understanding system ready!")
