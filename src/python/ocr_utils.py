from typing import Dict, Any
import os
import concurrent.futures
import time
import requests
import numpy as np
from PIL import Image
import io
import mimetypes
from concurrent.futures import ThreadPoolExecutor
from typing import List, Optional

# Attempt to import easyocr and set a flag
try:
    import easyocr
    EASYOCR_AVAILABLE = True
except ModuleNotFoundError:
    easyocr = None  # Define easyocr as None if not available
    EASYOCR_AVAILABLE = False

# Placeholder for the EasyOCR reader instance
ocr_reader_instance = None

def get_easyocr_reader():
    """Initializes and returns the EasyOCR reader instance. Raises ImportError if not available."""
    global ocr_reader_instance
    if not EASYOCR_AVAILABLE:
        raise ImportError(
            "EasyOCR library is not installed, but is required for local OCR. "
            "Please install it by running: pip install easyocr"
        )
    if ocr_reader_instance is None:
        # Initialize EasyOCR reader (e.g., for English and Thai)
        # You might want to make languages configurable
        print("[INFO] Initializing EasyOCR reader (this might take a moment)...")
        ocr_reader_instance = easyocr.Reader(['en', 'th'], gpu=False) # Set gpu=True if a CUDA-enabled GPU is available
        print("[INFO] EasyOCR reader initialized.")
    return ocr_reader_instance

def extract_text_with_easyocr(image, langs=['th', 'en']):
    """Extract text from image using EasyOCR"""
    try:
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

def extract_text_from_image_local(image_path: str) -> str:
    """
    Extracts text from a single image file using EasyOCR (local processing).
    """
    if not os.path.exists(image_path):
        return f"Error: Image file not found at {image_path}"
    
    try:
        reader = get_easyocr_reader()
        result = reader.readtext(image_path)
        text = "\n".join([item[1] for item in result])
        return text
    except ImportError as e: # Catch the specific ImportError from get_easyocr_reader
        print(f"[WARN] Cannot perform local OCR: {e}")
        return "Error: EasyOCR not available for local image processing."
    except Exception as e:
        return f"Error during local OCR processing for {image_path}: {str(e)}"

def extract_text_from_image_mistral(image_path: str, mistral_api_key: str) -> str:
    """Extracts text from an image using Mistral API."""
    if not mistral_api_key:
        return "Error: Mistral API key not provided."
    if not os.path.exists(image_path):
        return f"Error: Image file not found at {image_path}"

    url = "https://api.mistral.ai/v1/embed" # This URL is for embeddings, OCR might be different or not public
    # Placeholder: Actual Mistral OCR API endpoint and request structure would be needed.
    # This is a conceptual example. Mistral's public API for direct OCR from image path might not exist
    # or might require uploading the image data.
    # For now, let's assume it's a placeholder and would need correct implementation.
    # If Mistral doesn't have a direct OCR API like this, this function would need to be adapted
    # or rely on a different service if 'mistral_api_key' is meant for a general Mistral model
    # that can perform OCR given image bytes.

    # This is a common pattern for file uploads with requests:
    try:
        with open(image_path, "rb") as f:
            files = {"file": (os.path.basename(image_path), f, mimetypes.guess_type(image_path)[0])}
            headers = {"Authorization": f"Bearer {mistral_api_key}"}
            
            # Replace with actual Mistral OCR API endpoint and payload structure
            # The following is a generic example and WILL NOT WORK with Mistral's embedding endpoint
            # response = requests.post( "ACTUAL_MISTRAL_OCR_ENDPOINT" , headers=headers, files=files, timeout=60)
            # response.raise_for_status()
            # return response.json().get("extracted_text", "Error: Could not extract text.")
            print(f"[WARN] Mistral OCR from image path is not fully implemented. Using placeholder/local fallback concept.")
            # Fallback or error if Mistral OCR part is not implemented
            return "Error: Mistral image OCR not implemented. Try local OCR if available."

    except requests.exceptions.RequestException as e:
        return f"Error connecting to Mistral API for image OCR: {str(e)}"
    except Exception as e:
        return f"An unexpected error occurred during Mistral image OCR: {str(e)}"


def extract_text_from_pdf_mistral(file_path: str, mistral_api_key: str, max_pages: Optional[int] = None, max_workers: int = 1) -> str:
    """
    Extracts text from a PDF file using Mistral API (conceptual - assumes page-by-page image conversion and OCR).
    This function would typically involve converting PDF pages to images first, then OCRing each image.
    Mistral's API might offer a direct PDF processing endpoint, or this would be a multi-step process.
    """
    if not mistral_api_key:
        return "Error: Mistral API key not provided."
    if not os.path.exists(file_path):
        return f"Error: PDF file not found at {file_path}"

    # Placeholder for PDF to image conversion and then calling extract_text_from_image_mistral for each page.
    # This requires a PDF library like PyMuPDF (fitz) or pdf2image.
    # Example using pdf2image (would need 'pip install pdf2image poppler-utils'):
    try:
        from pdf2image import convert_from_path
    except ModuleNotFoundError:
        return ("Error: pdf2image library not found. "
                "Please install it ('pip install pdf2image') and Poppler to process PDFs for Mistral OCR.")

    all_text = []
    try:
        print(f"[INFO] Converting PDF '{os.path.basename(file_path)}' to images...")
        images = convert_from_path(file_path, first_page=1, last_page=max_pages) # poppler_path can be specified if not in PATH
        
        page_count_to_process = len(images)
        print(f"[INFO] Converted {page_count_to_process} pages to images. Starting OCR...")

        # This part would ideally use Mistral's image OCR for each page image.
        # For now, it's a conceptual placeholder.
        # If extract_text_from_image_mistral were fully functional:
        # with ThreadPoolExecutor(max_workers=max_workers) as executor:
        #     futures = []
        #     for i, image in enumerate(images):
        #         temp_image_path = f"temp_page_{i}.png"
        #         image.save(temp_image_path, "PNG")
        #         futures.append(executor.submit(extract_text_from_image_mistral, temp_image_path, mistral_api_key))
            
        #     for i, future in enumerate(futures):
        #         page_text = future.result()
        #         all_text.append(f"--- Page {i+1} ---\n{page_text}")
        #         os.remove(f"temp_page_{i}.png") # Clean up temp image

        # Simplified: Using local OCR as a stand-in if Mistral image OCR is conceptual
        if EASYOCR_AVAILABLE:
            print("[INFO] Using local EasyOCR as a stand-in for PDF page processing due to conceptual Mistral image OCR.")
            reader = get_easyocr_reader()
            for i, image in enumerate(images):
                # EasyOCR works with image paths or numpy arrays.
                # To avoid saving/reloading, convert PIL Image to numpy array.
                import numpy as np
                img_np = np.array(image)
                result = reader.readtext(img_np)
                page_text = "\n".join([item[1] for item in result])
                all_text.append(f"--- Page {i+1} ---\n{page_text}")
                print(f"[INFO] Processed page {i+1}/{page_count_to_process} using local OCR.")
        else:
            return "Error: Mistral PDF OCR (via image conversion) is conceptual, and local OCR (EasyOCR) is not available as a fallback."


    except Exception as e:
        return f"Error processing PDF {file_path}: {str(e)}"
    
    return "\n\n".join(all_text)


def extract_text_from_file(
    file_path: str,
    mistral_api_key: Optional[str] = None,
    max_pages: Optional[int] = None, # For PDFs
    max_workers: int = 1 # For concurrent page processing in PDFs
) -> str:
    """
    Extracts text from a given file (PDF or Image).
    Uses Mistral API if key is provided and applicable, otherwise may attempt local OCR for images.
    """
    if not os.path.exists(file_path):
        return "Error: File not found."

    file_type = mimetypes.guess_type(file_path)[0]

    if file_type == "application/pdf":
        if mistral_api_key:
            return extract_text_from_pdf_mistral(file_path, mistral_api_key, max_pages, max_workers)
        else:
            return "Error: Mistral API key required for PDF processing."
    elif file_type and file_type.startswith("image/"):
        if mistral_api_key:
            # Prefer Mistral if API key is available and image OCR is implemented
            text = extract_text_from_image_mistral(file_path, mistral_api_key)
            # Basic check if Mistral call was conceptual/failed
            if "Error: Mistral image OCR not implemented" in text or "Error connecting to Mistral API" in text:
                print("[INFO] Mistral image OCR failed or not implemented, trying local OCR as fallback.")
                return extract_text_from_image_local(file_path)
            return text
        else:
            # Fallback to local OCR if Mistral API key is not available
            print("[INFO] Mistral API key not provided for image OCR, attempting local OCR with EasyOCR.")
            return extract_text_from_image_local(file_path)
    else:
        return f"Error: Unsupported file type: {file_type}. Please provide a PDF or image file."

# Example usage (optional, for testing this module directly)
if __name__ == '__main__':
    # Create a dummy image file for testing if you don't have one
    # from PIL import Image, ImageDraw, ImageFont
    # try:
    #     img = Image.new('RGB', (400, 100), color = (255, 255, 255))
    #     d = ImageDraw.Draw(img)
    #     font = ImageFont.truetype("arial.ttf", 20) # Ensure you have arial.ttf or change font
    #     d.text((10,10), "Hello EasyOCR Test", fill=(0,0,0), font=font)
    #     dummy_image_path = "dummy_test_image.png"
    #     img.save(dummy_image_path)
        
    #     print(f"--- Testing Local Image OCR (EasyOCR) for {dummy_image_path} ---")
    #     local_text = extract_text_from_image_local(dummy_image_path)
    #     print(f"Local OCR Result: {local_text}")
    #     os.remove(dummy_image_path)

    # except ImportError:
    #     print("Pillow or a valid font not found, skipping dummy image creation for test.")
    # except Exception as e:
    #     print(f"Error during dummy image test: {e}")

    print("\n--- Testing with non-existent file ---")
    non_existent_text = extract_text_from_file("non_existent_file.png")
    print(f"Result for non-existent file: {non_existent_text}")
    
    # To test Mistral parts, you'd need a valid API key and potentially a running service/mock
    # print("\n--- Testing Mistral Image OCR (conceptual) ---")
    # MISTRAL_API_KEY_TEST = os.getenv("MISTRAL_API_KEY_FOR_TESTING") # Set this env var for testing
    # if MISTRAL_API_KEY_TEST:
    #     # Create another dummy image
    #     # ...
    #     # mistral_text = extract_text_from_image_mistral(dummy_image_path, MISTRAL_API_KEY_TEST)
    #     # print(f"Mistral OCR Result: {mistral_text}")
    #     pass
    # else:
    #     print("MISTRAL_API_KEY_FOR_TESTING not set, skipping Mistral OCR test.")
    pass