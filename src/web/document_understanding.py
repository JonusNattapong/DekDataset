"""
Document Understanding Module for DekDataset
Provides OCR, PDF processing, and document analysis capabilities
"""

import os
import logging
import tempfile
import base64
from pathlib import Path
from typing import List, Dict, Any, Optional, Union
from dataclasses import dataclass
from datetime import datetime

# PDF and image processing
try:
    import pdfplumber
    import requests
    from PIL import Image
    import pytesseract
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
    metadata: Dict[str, Any] = None

    def __post_init__(self):
        if self.metadata is None:
            self.metadata = {}

class DocumentUnderstanding:
    """Advanced document understanding with OCR and analysis"""
    
    def __init__(self, mistral_api_key: Optional[str] = None):
        """Initialize document understanding with optional Mistral API"""
        self.mistral_api_key = mistral_api_key or os.getenv('MISTRAL_API_KEY')
        self.mistral_client = None
        
        if self.mistral_api_key and MISTRAL_AVAILABLE:
            try:
                self.mistral_client = Mistral(api_key=self.mistral_api_key)
                logger.info("Mistral AI client initialized successfully")
            except Exception as e:
                logger.warning(f"Failed to initialize Mistral client: {e}")
        
        # Configure tesseract for Windows if available
        if os.name == "nt":
            tesseract_paths = [
                r"C:\Program Files\Tesseract-OCR\tesseract.exe",
                r"C:\Program Files (x86)\Tesseract-OCR\tesseract.exe",
                r".\tesseract\tesseract.exe"
            ]
            for path in tesseract_paths:
                if os.path.exists(path):
                    pytesseract.pytesseract.tesseract_cmd = path
                    break

    def extract_text_from_pdf(self, pdf_path: Union[str, Path]) -> List[DocumentAnnotation]:
        """Extract text from PDF with page-by-page processing"""
        if not PDF_AVAILABLE:
            raise ImportError("PDF processing libraries not available. Install: pip install pdfplumber pdf2image")
        
        annotations = []
        pdf_path = Path(pdf_path)
        
        if not pdf_path.exists():
            raise FileNotFoundError(f"PDF file not found: {pdf_path}")
        
        try:
            with pdfplumber.open(pdf_path) as pdf:
                logger.info(f"Processing PDF with {len(pdf.pages)} pages")
                
                for page_num, page in enumerate(pdf.pages, 1):
                    try:
                        # Extract text using pdfplumber
                        text = page.extract_text()
                        
                        if text and text.strip():
                            annotation = DocumentAnnotation(
                                text=text.strip(),
                                page_number=page_num,
                                confidence=0.9,  # High confidence for PDF text extraction
                                metadata={
                                    'extraction_method': 'pdfplumber',
                                    'page_dimensions': {
                                        'width': page.width,
                                        'height': page.height
                                    },
                                    'source_file': str(pdf_path),
                                    'extracted_at': datetime.now().isoformat()
                                }
                            )
                            annotations.append(annotation)
                            logger.debug(f"Extracted text from page {page_num}: {len(text)} characters")
                        else:
                            logger.warning(f"No text found on page {page_num}")
                            
                    except Exception as e:
                        logger.error(f"Error processing page {page_num}: {e}")
                        continue
                        
        except Exception as e:
            logger.error(f"Error opening PDF file: {e}")
            raise
        
        logger.info(f"Successfully extracted text from {len(annotations)} pages")
        return annotations

    def extract_text_from_image(self, image_path: Union[str, Path]) -> List[DocumentAnnotation]:
        """Extract text from image using OCR"""
        if not PDF_AVAILABLE:
            raise ImportError("Image processing libraries not available")
        
        image_path = Path(image_path)
        
        if not image_path.exists():
            raise FileNotFoundError(f"Image file not found: {image_path}")
        
        annotations = []
        
        try:
            # Use Mistral AI for advanced OCR if available
            if self.mistral_client:
                text = self._extract_with_mistral(image_path)
                confidence = 0.95
                method = 'mistral_ai'
            else:
                # Fallback to Tesseract
                text = self._extract_with_tesseract(image_path)
                confidence = 0.8
                method = 'tesseract'
            
            if text and text.strip():
                annotation = DocumentAnnotation(
                    text=text.strip(),
                    page_number=1,
                    confidence=confidence,
                    metadata={
                        'extraction_method': method,
                        'source_file': str(image_path),
                        'file_size': image_path.stat().st_size,
                        'extracted_at': datetime.now().isoformat()
                    }
                )
                annotations.append(annotation)
                logger.info(f"Extracted text from image: {len(text)} characters")
            else:
                logger.warning("No text found in image")
                
        except Exception as e:
            logger.error(f"Error processing image: {e}")
            raise
        
        return annotations

    def _extract_with_mistral(self, image_path: Path) -> str:
        """Extract text using Mistral AI Vision"""
        try:
            # Read and encode image
            with open(image_path, 'rb') as f:
                image_data = f.read()
            
            image_base64 = base64.b64encode(image_data).decode('utf-8')
            
            # Prepare Mistral API request
            messages = [
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "text",
                            "text": "Extract all text from this image. Return only the extracted text, no additional commentary."
                        },
                        {
                            "type": "image_url",
                            "image_url": f"data:image/jpeg;base64,{image_base64}"
                        }
                    ]
                }
            ]
            
            # Call Mistral API
            response = self.mistral_client.chat.complete(
                model="pixtral-12b-2409",
                messages=messages,
                max_tokens=1000
            )
            
            return response.choices[0].message.content.strip()
            
        except Exception as e:
            logger.error(f"Mistral OCR failed: {e}")
            # Fallback to Tesseract
            return self._extract_with_tesseract(image_path)

    def _extract_with_tesseract(self, image_path: Path) -> str:
        """Extract text using Tesseract OCR"""
        try:
            image = Image.open(image_path)
            
            # Optimize image for OCR
            if image.mode != 'RGB':
                image = image.convert('RGB')
            
            # Extract text with Thai language support
            text = pytesseract.image_to_string(
                image, 
                lang='tha+eng',  # Thai and English
                config='--oem 3 --psm 6'
            )
            
            return text.strip()
            
        except Exception as e:
            logger.error(f"Tesseract OCR failed: {e}")
            raise

    def process_url(self, url: str) -> List[DocumentAnnotation]:
        """Process document from URL"""
        try:
            # Download file to temporary location
            response = requests.get(url, timeout=30)
            response.raise_for_status()
            
            # Determine file type from content-type or URL
            content_type = response.headers.get('content-type', '').lower()
            
            with tempfile.NamedTemporaryFile(delete=False) as tmp_file:
                tmp_file.write(response.content)
                tmp_path = Path(tmp_file.name)
            
            try:
                if 'pdf' in content_type or url.lower().endswith('.pdf'):
                    return self.extract_text_from_pdf(tmp_path)
                elif any(img_type in content_type for img_type in ['image/', 'jpeg', 'png', 'jpg']):
                    return self.extract_text_from_image(tmp_path)
                else:
                    # Try to detect from URL extension
                    if url.lower().endswith(('.jpg', '.jpeg', '.png', '.gif', '.bmp')):
                        return self.extract_text_from_image(tmp_path)
                    elif url.lower().endswith('.pdf'):
                        return self.extract_text_from_pdf(tmp_path)
                    else:
                        raise ValueError(f"Unsupported file type: {content_type}")
                        
            finally:
                # Clean up temporary file
                tmp_path.unlink(missing_ok=True)
                
        except Exception as e:
            logger.error(f"Error processing URL {url}: {e}")
            raise

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
            return self.extract_text_from_pdf(file_path)
        elif extension in ['.jpg', '.jpeg', '.png', '.gif', '.bmp', '.tiff']:
            return self.extract_text_from_image(file_path)
        else:
            raise ValueError(f"Unsupported file type: {extension}")

    def analyze_document_structure(self, annotations: List[DocumentAnnotation]) -> Dict[str, Any]:
        """Analyze document structure and provide insights"""
        if not annotations:
            return {'error': 'No content to analyze'}
        
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
                'estimated_reading_time': max(1, len(total_text.split()) // 200)  # minutes
            }
        }
        
        return analysis

    def _detect_languages(self, text: str) -> List[str]:
        """Detect languages in text"""
        languages = []
        
        if self._has_thai_content(text):
            languages.append('thai')
        
        if self._has_english_content(text):
            languages.append('english')
        
        # Add more language detection as needed
        return languages if languages else ['unknown']

    def _has_thai_content(self, text: str) -> bool:
        """Check if text contains Thai characters"""
        thai_range = range(0x0E00, 0x0E7F)
        return any(ord(char) in thai_range for char in text)

    def _has_english_content(self, text: str) -> bool:
        """Check if text contains English characters"""
        english_chars = sum(1 for char in text if char.isalpha() and ord(char) < 128)
        return english_chars > len(text) * 0.1  # At least 10% English chars

    def get_combined_text(self, annotations: List[DocumentAnnotation]) -> str:
        """Get all text combined from annotations"""
        if not annotations:
            return ""
        
        # Sort by page number
        sorted_annotations = sorted(annotations, key=lambda x: x.page_number)
        
        # Combine text with page separators
        combined_text = ""
        current_page = 0
        
        for ann in sorted_annotations:
            if ann.page_number != current_page:
                if combined_text:
                    combined_text += "\n\n--- Page Break ---\n\n"
                current_page = ann.page_number
            
            combined_text += ann.text
            if not ann.text.endswith('\n'):
                combined_text += '\n'
        
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
                        'metadata': ann.metadata,
                        'bbox': ann.bbox.__dict__ if ann.bbox else None
                    }
                    for ann in annotations
                ],
                'analysis': self.analyze_document_structure(annotations),
                'exported_at': datetime.now().isoformat()
            }
        elif format.lower() == 'text':
            return self.get_combined_text(annotations)
        else:
            raise ValueError(f"Unsupported export format: {format}")

# Utility functions for backward compatibility
def extract_text_from_pdf(pdf_path: Union[str, Path]) -> str:
    """Simple PDF text extraction function"""
    doc_processor = DocumentUnderstanding()
    annotations = doc_processor.extract_text_from_pdf(pdf_path)
    return doc_processor.get_combined_text(annotations)

def extract_text_from_image(image_path: Union[str, Path]) -> str:
    """Simple image text extraction function"""
    doc_processor = DocumentUnderstanding()
    annotations = doc_processor.extract_text_from_image(image_path)
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
            try:
                print(f"\n📄 Processing: {test_file}")
                annotations = doc_processor.process_document(test_file)
                analysis = doc_processor.analyze_document_structure(annotations)
                
                print(f"✅ Success: {analysis['total_pages']} pages, {analysis['total_words']} words")
                print(f"📊 Languages: {', '.join(analysis['languages_detected'])}")
                print(f"🎯 Avg Confidence: {analysis['average_confidence']:.2f}")
                
            except Exception as e:
                print(f"❌ Error processing {test_file}: {e}")
    
    print("\n🎉 Document Understanding system ready!")
