from typing import Dict, Any
import easyocr

def extract_text_with_easyocr(image, langs=['th', 'en']):
    """Extract text from image using EasyOCR"""
    try:
        import easyocr
        import numpy as np
        from PIL import Image
        
        # Create reader instance
        reader = easyocr.Reader(langs)
        
        # Convert PIL Image to numpy array if needed
        if isinstance(image, Image.Image):
            image_np = np.array(image)
        else:
            image_np = image
            
        # Perform OCR
        results = reader.readtext(image_np)
        
        # Extract and join text
        extracted_text = "\n".join([result[1] for result in results])
        return extracted_text
        
    except Exception as e:
        print(f"[ERROR] EasyOCR processing failed: {e}")
        return ""

import os
import concurrent.futures
import time
import requests
import numpy as np
from PIL import Image
import io

# Add new imports
try:
    import fitz  # PyMuPDF
    PYMUPDF_AVAILABLE = True
except ImportError:
    PYMUPDF_AVAILABLE = False

def extract_text_with_pymupdf_easyocr(pdf_path: str, confidence_threshold: float = 0.5) -> str:
    """
    Extract text from PDF using PyMuPDF for rendering and EasyOCR for text detection
    
    Args:
        pdf_path: Path to PDF file
        confidence_threshold: Minimum confidence score for OCR results (0-1)
        
    Returns:
        Extracted text with page markers
    """
    if not PYMUPDF_AVAILABLE:
        raise ImportError("PyMuPDF (fitz) not installed. Install with: pip install PyMuPDF")

    try:
        doc = fitz.open(pdf_path)
        reader = easyocr.Reader(['th', 'en'])
        full_text = ""
        
        print(f"[INFO] Processing PDF with {doc.page_count} pages")
        
        for page_num in range(doc.page_count):
            try:
                page = doc[page_num]
                print(f"[INFO] Processing page {page_num + 1}/{doc.page_count}")
                
                # Render page at higher resolution
                mat = fitz.Matrix(3, 3)  # 3x scale
                pix = page.get_pixmap(matrix=mat)
                img_data = pix.tobytes("png")
                
                # Perform OCR
                results = reader.readtext(img_data)
                
                # Extract text with confidence filtering
                page_text = ""
                for (bbox, text, confidence) in results:
                    if confidence > confidence_threshold:
                        page_text += text + " "
                
                full_text += f"\n--- Page {page_num + 1} ---\n{page_text.strip()}\n"
                
            except Exception as e:
                print(f"[ERROR] Failed to process page {page_num + 1}: {e}")
                full_text += f"\n--- Page {page_num + 1} ---\n[ERROR] Page processing failed\n"
        
        doc.close()
        return full_text.strip()
        
    except Exception as e:
        print(f"[ERROR] PDF processing failed: {e}")
        return ""

def extract_educational_content(pdf_path: str, lang: str = 'th') -> Dict[str, Any]:
    """
    Extract educational content from PDF with enhanced Thai language support
    
    Args:
        pdf_path: Path to PDF file
        lang: Primary language ('th' or 'en')
    """
    if not PYMUPDF_AVAILABLE:
        raise ImportError("PyMuPDF (fitz) not installed. Install with: pip install PyMuPDF")

    try:
        doc = fitz.open(pdf_path)
        reader = easyocr.Reader(['th', 'en'])
        content = {
            'title': '',
            'sections': [],
            'topics': [],
            'examples': [],
            'exercises': []
        }
        
        current_section = ""
        current_text = ""
        
        for page_num in range(doc.page_count):
            try:
                page = doc[page_num]
                print(f"[INFO] Processing page {page_num + 1}/{doc.page_count}")
                
                # Get high-resolution render
                mat = fitz.Matrix(3, 3)
                pix = page.get_pixmap(matrix=mat)
                img_data = pix.tobytes("png")
                
                # Perform OCR with layout analysis
                results = reader.readtext(
                    img_data,
                    detail=1,
                    paragraph=True
                )
                
                # Process each text block
                for block in results:
                    text = block[1].strip()
                    if not text:
                        continue
                        
                    # Detect section headers
                    if len(text) < 100 and any(x in text for x in ['บทที่', 'หัวข้อ', 'เรื่อง']):
                        if current_section and current_text:
                            content['sections'].append({
                                'title': current_section,
                                'content': current_text.strip()
                            })
                        current_section = text
                        current_text = ""
                        continue
                    
                    # Detect examples
                    if any(x in text for x in ['ตัวอย่าง', 'เช่น', 'ยกตัวอย่าง']):
                        content['examples'].append(text)
                        continue
                        
                    # Detect exercises
                    if any(x in text for x in ['แบบฝึกหัด', 'คำถาม', 'จงตอบ']):
                        content['exercises'].append(text)
                        continue
                        
                    # Add to current section
                    current_text += text + "\n"
                    
                    # Extract key topics
                    if len(text) < 200:
                        for line in text.split('\n'):
                            if any(x in line for x in ['•', '-', '๑', '๒', '๓']):
                                content['topics'].append(line.strip())
                
            except Exception as e:
                print(f"[WARN] Error processing page {page_num + 1}: {e}")
                
        # Add final section
        if current_section and current_text:
            content['sections'].append({
                'title': current_section,
                'content': current_text.strip()
            })
            
        doc.close()
        return content
        
    except Exception as e:
        print(f"[ERROR] Failed to process educational PDF: {e}")
        return None