def extract_text_from_file(file_path_or_url, MISTRAL_API_KEY, max_pages=None, max_workers=3):
    """
    Extract text from PDF/image file or document_url using Mistral OCR API.
    Supports: PDF, JPG, PNG, and remote URLs (e.g. arXiv PDF).
    
    Args:
        file_path_or_url: Path to file or URL
        MISTRAL_API_KEY: API key for Mistral
        max_pages: Limit number of pages to process (None = all pages)
        max_workers: Number of concurrent workers for processing (default: 3)
    """
    import requests
    import os
    import concurrent.futures
    import time
    
    try:
        from pdf2image import convert_from_path
        from PIL import Image
    except ImportError as e:
        raise ImportError(f"Required dependencies missing: {e}. Please install with: pip install pdf2image pillow")    # If input is a URL (http/https), use document_url mode
    if file_path_or_url.startswith("http://") or file_path_or_url.startswith("https://"):
        print(f"[DEBUG] Processing URL: {file_path_or_url}")
        payload = {
            "model": "mistral-ocr-2505",
            "id": "doc-ocr-001",
            "document": {
                "document_url": file_path_or_url,
                "document_name": os.path.basename(file_path_or_url),
                "type": "document_url",
            },
            "pages": list(range(max_pages)) if max_pages else [0],  # Use max_pages limit
            "include_image_base64": True,
            "image_limit": 0,
            "image_min_size": 0,        }
        print(f"[DEBUG] Sending request to Mistral OCR API...")
        response = requests.post(
            "https://api.mistral.ai/v1/ocr",
            headers={
                "Authorization": f"Bearer {MISTRAL_API_KEY}",
                "Content-Type": "application/json",
            },
            json=payload,
        )
        print(f"[DEBUG] Response status: {response.status_code}")
        if response.status_code == 200:
            ocr_result = response.json()
            text_result = ocr_result.get("text", "")
            print(f"[DEBUG] Extracted text length: {len(text_result)} characters")
            return text_result
        else:
            print(f"[OCR] Failed for URL: {response.status_code} - {response.text}")
            return ""
    else:
        # Local file (PDF/image)
        print(f"[DEBUG] Processing local file: {file_path_or_url}")
        images = []
        
        if file_path_or_url.lower().endswith(".pdf"):
            try:
                # Set poppler path for Windows
                poppler_path = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))), "poppler-local", "Library", "bin")
                print(f"[DEBUG] Using poppler path: {poppler_path}")
                
                if os.path.exists(poppler_path):
                    images = convert_from_path(file_path_or_url, poppler_path=poppler_path)
                else:
                    # Fallback to system poppler
                    images = convert_from_path(file_path_or_url)
                
                # Limit pages if specified
                if max_pages and len(images) > max_pages:
                    images = images[:max_pages]
                    print(f"[DEBUG] Limited to first {max_pages} pages")
                    
                print(f"[DEBUG] Converted PDF to {len(images)} images")
            except Exception as e:
                if "poppler" in str(e).lower():
                    raise Exception(
                        "Poppler is required for PDF processing. "
                        "Install it from: https://github.com/oschwartz10612/poppler-windows/releases "
                        "or use: conda install -c conda-forge poppler"
                    )
                else:
                    raise e
        else:
            images = [Image.open(file_path_or_url)]
            print(f"[DEBUG] Loaded single image file")

        def process_single_image(img_data):
            """Process a single image with OCR"""
            idx, img = img_data
            try:
                print(f"[DEBUG] Processing image {idx+1}/{len(images)}")
                temp_img_path = f"temp_ocr_{idx}_{time.time()}.png"
                img.save(temp_img_path, format='PNG')
                  # Convert image to base64 for Vision API
                import base64
                with open(temp_img_path, "rb") as f:
                    img_base64 = base64.b64encode(f.read()).decode()
                
                # Use Vision API for OCR instead of non-existent OCR API
                payload = {
                    "model": "pixtral-12b-2409",
                    "messages": [
                        {
                            "role": "user",
                            "content": [
                                {
                                    "type": "text",
                                    "text": "Extract all Thai and English text from this image. Return only the text content, no explanations."
                                },
                                {
                                    "type": "image_url",
                                    "image_url": f"data:image/png;base64,{img_base64}"
                                }
                            ]
                        }
                    ],
                    "max_tokens": 2000,
                    "temperature": 0
                }
                
                response = requests.post(
                    "https://api.mistral.ai/v1/chat/completions",
                    headers={
                        "Authorization": f"Bearer {MISTRAL_API_KEY}",
                        "Content-Type": "application/json"
                    },
                    json=payload,
                        timeout=30  # Add timeout
                    )
                  # Clean up temp file
                if os.path.exists(temp_img_path):
                    os.remove(temp_img_path)
                
                if response.status_code == 200:
                    ocr_result = response.json()
                    page_text = ocr_result["choices"][0]["message"]["content"]
                    print(f"[DEBUG] Page {idx+1} extracted {len(page_text)} characters")
                    return page_text
                else:
                    print(f"[OCR] Failed for page {idx+1}: {response.status_code} - {response.text}")
                    return ""
            except Exception as e:
                print(f"[OCR] Error processing page {idx+1}: {e}")
                # Clean up temp file on error
                temp_img_path = f"temp_ocr_{idx}_{time.time()}.png"
                if os.path.exists(temp_img_path):
                    os.remove(temp_img_path)
                return ""

        # Process images concurrently
        all_text = []
        if len(images) <= 3 or max_workers == 1:
            # Sequential processing for small number of images
            for idx, img in enumerate(images):
                page_text = process_single_image((idx, img))
                all_text.append(page_text)
        else:
            # Concurrent processing for many images
            print(f"[DEBUG] Using {max_workers} workers for concurrent processing")
            with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
                # Submit all tasks
                future_to_idx = {}
                for idx, img in enumerate(images):
                    future = executor.submit(process_single_image, (idx, img))
                    future_to_idx[future] = idx
                
                # Collect results in order
                results = {}
                for future in concurrent.futures.as_completed(future_to_idx):
                    idx = future_to_idx[future]
                    try:
                        page_text = future.result()
                        results[idx] = page_text
                    except Exception as e:
                        print(f"[OCR] Error in concurrent processing for page {idx+1}: {e}")
                        results[idx] = ""
                
                # Sort results by page index
                for idx in sorted(results.keys()):
                    all_text.append(results[idx])

        total_chars = sum(len(text) for text in all_text)
        print(f"[INFO] Total extracted text: {total_chars} characters from {len(images)} pages")
        return "\n".join(all_text)
