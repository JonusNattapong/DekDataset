import os
import sys
import json
import time
import shutil
import hashlib
import tempfile
import traceback
import zipfile
import csv
import requests
import logging
from pathlib import Path as PathLib
from datetime import datetime
from fastapi import BackgroundTasks, FastAPI, UploadFile, File, Form, HTTPException, Path, Request, Depends, Body
from fastapi.responses import HTMLResponse, JSONResponse, Response
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from pydantic import BaseModel, Field
from typing import List, Dict, Tuple, Any, Optional
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from dotenv import load_dotenv

# Add the parent directory to sys.path for imports
sys.path.append(os.path.join(os.path.dirname(__file__), "..", "python"))
from generate_dataset import build_data_entries_from_raw, export_jsonl
from banner import print_ascii_banner  # Add banner import
from openai import OpenAI

# PDF and OCR dependencies
try:
    import pdfplumber
    import pytesseract
    from PIL import Image
    from pdf2image import convert_from_path
    OCR_AVAILABLE = True
    # Set tesseract_cmd for Windows if not already set
    if os.name == "nt":
        tesseract_path = r"C:\\Program Files\\Tesseract-OCR\\tesseract.exe"
        if os.path.exists(tesseract_path):
            pytesseract.pytesseract.tesseract_cmd = tesseract_path
except ImportError:
    OCR_AVAILABLE = False
    logger = logging.getLogger(__name__)
    logger.warning("OCR dependencies not available. Install with: pip install pdfplumber pytesseract pdf2image Pillow")

# Vector Storage for PDF RAG
class PDFVectorStore:
    """Simple vector store for PDF content using TF-IDF"""
    
    def __init__(self):
        self.documents = {}  # {doc_id: {'text': str, 'metadata': dict}}
        self.vectorizer = None
        self.vectors = None
        self.doc_ids = []
        
    def add_document(self, doc_id: str, text: str, metadata: dict = None):
        """Add document to vector store"""
        self.documents[doc_id] = {
            'text': text,
            'metadata': metadata or {}
        }
        self._rebuild_index()
        
    def _rebuild_index(self):
        """Rebuild TF-IDF index"""
        if not self.documents:
            return
            
        texts = []
        doc_ids = []
        
        for doc_id, doc_data in self.documents.items():
            texts.append(doc_data['text'])
            doc_ids.append(doc_id)
            
        self.vectorizer = TfidfVectorizer(
            max_features=5000,
            stop_words='english',
            ngram_range=(1, 2)
        )
        
        self.vectors = self.vectorizer.fit_transform(texts)
        self.doc_ids = doc_ids
        
    def search(self, query: str, top_k: int = 5) -> List[Tuple[str, float, dict]]:
        """Search for similar documents"""
        if not self.vectorizer or not self.vectors:
            return []
            
        try:
            query_vector = self.vectorizer.transform([query])
            similarities = cosine_similarity(query_vector, self.vectors).flatten()
            
            # Get top k results
            top_indices = similarities.argsort()[-top_k:][::-1]
            
            results = []
            for idx in top_indices:
                doc_id = self.doc_ids[idx]
                score = similarities[idx]
                if score > 0.1:  # Minimum similarity threshold
                    results.append((
                        doc_id,
                        float(score),
                        self.documents[doc_id]
                    ))
                    
            return results
            
        except Exception as e:
            logger.error(f"Error searching vector store: {e}")
            return []
            
    def get_context(self, query: str, max_length: int = 2000) -> str:
        """Get relevant context for RAG"""
        results = self.search(query, top_k=3)
        
        if not results:
            return ""
            
        context_parts = []
        current_length = 0
        
        for doc_id, score, doc_data in results:
            text = doc_data['text']
            metadata = doc_data.get('metadata', {})
            
            # Add source info
            source_info = f"[Source: {metadata.get('filename', doc_id)}]"
            
            if current_length + len(text) + len(source_info) < max_length:
                context_parts.append(f"{source_info}\n{text}")
                current_length += len(text) + len(source_info)
            else:
                # Truncate last piece to fit
                remaining = max_length - current_length - len(source_info)
                if remaining > 100:  # Only add if meaningful
                    context_parts.append(f"{source_info}\n{text[:remaining]}...")
                break
                
        return "\n\n---\n\n".join(context_parts)

# Global PDF vector store
pdf_vector_store = PDFVectorStore()

def add_pdf_to_rag(pdf_content: str, filename: str, metadata: dict = None):
    """Add PDF content to RAG system"""
    try:
        # Create unique document ID
        doc_id = hashlib.md5(f"{filename}_{pdf_content[:100]}".encode()).hexdigest()
        
        # Split long content into chunks
        max_chunk_size = 1000
        chunks = []
        
        if len(pdf_content) > max_chunk_size:
            # Split by paragraphs first
            paragraphs = pdf_content.split('\n\n')
            current_chunk = ""
            
            for para in paragraphs:
                if len(current_chunk) + len(para) > max_chunk_size:
                    if current_chunk:
                        chunks.append(current_chunk.strip())
                    current_chunk = para
                else:
                    current_chunk += "\n\n" + para if current_chunk else para
                    
            if current_chunk:
                chunks.append(current_chunk.strip())
        else:
            chunks = [pdf_content]
            
        # Add each chunk as separate document
        for i, chunk in enumerate(chunks):
            chunk_id = f"{doc_id}_chunk_{i}"
            chunk_metadata = {
                'filename': filename,
                'chunk_index': i,
                'total_chunks': len(chunks),
                **(metadata or {})
            }
            pdf_vector_store.add_document(chunk_id, chunk, chunk_metadata)
            
        logger.info(f"Added PDF '{filename}' to RAG with {len(chunks)} chunks")
        return True
        
    except Exception as e:
        logger.error(f"Error adding PDF to RAG: {e}")
        return False

def get_rag_context(query: str) -> str:
    """Get RAG context for dataset generation"""
    return pdf_vector_store.get_context(query, max_length=1500)

def save_dataset_to_cache_sync(task_id, cache_data):
    """Save cache_data to cache/{task_id}.json file"""
    try:
        cache_dir = os.path.join(os.path.dirname(__file__), "..", "..", "cache")
        os.makedirs(cache_dir, exist_ok=True)
        cache_path = os.path.join(cache_dir, f"{task_id}.json")
        
        with open(cache_path, 'w', encoding='utf-8') as f:
            json.dump(cache_data, f, indent=2, ensure_ascii=False)
        
        logger.info(f"Dataset cached successfully: {cache_path}")
        return True
        
    except Exception as e:
        logger.error(f"Error saving dataset to cache: {e}")
        return False
# --- Ollama Integration ---
OLLAMA_API_URL = os.getenv("OLLAMA_API_URL", "http://localhost:11434")

def call_ollama_api(prompt, model_name):
    """
    เรียก Ollama local API เพื่อ generate ข้อความ
    """
    url = f"{OLLAMA_API_URL}/api/generate"
    payload = {
        "model": model_name,
        "prompt": prompt,
        "stream": False
    }
    resp = requests.post(url, json=payload, timeout=120)
    resp.raise_for_status()
    return resp.json()["response"]

from fastapi import FastAPI, UploadFile, File, Form, HTTPException
from typing import Optional

# PDF upload endpoint will be defined later after main app initialization

# Experiment tracking configuration loading removed

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Print the banner when the app starts
print_ascii_banner()
logger.info("🚀 DekDataset Web Server Starting...")

# Experiment tracking removed - focusing on core dataset generation functionality

# --- Pydantic Models ---
class TaskBase(BaseModel):
    id: str
    type: Optional[str] = 'custom'
    description: Optional[str] = None
    name: Optional[str] = None
    prompt_template: Optional[str] = None
    system_prompt: Optional[str] = None
    user_template: Optional[str] = None
    validation_rules: Optional[Dict[str, Any]] = {}
    data_schema: Optional[Dict[str, Any]] = Field(default={}, alias="schema")
    examples: Optional[List[Dict[str, Any]]] = []
    parameters: Optional[Dict[str, Any]] = {}
    format: Optional[str] = None

class TaskCreate(BaseModel):
    id: str
    type: Optional[str] = 'custom'
    description: Optional[str] = None
    system_prompt: Optional[str] = None
    user_template: Optional[str] = None

class TaskResponse(BaseModel):
    id: str
    type: Optional[str] = 'custom'
    description: Optional[str] = None
    name: Optional[str] = None
    prompt_template: Optional[str] = None
    system_prompt: Optional[str] = None
    user_template: Optional[str] = None
    validation_rules: Optional[Dict[str, Any]] = {}
    data_schema: Optional[Dict[str, Any]] = Field(default={}, alias="schema")
    examples: Optional[List[Dict[str, Any]]] = []
    parameters: Optional[Dict[str, Any]] = {}
    format: Optional[str] = None
    created_at: Optional[datetime] = None

    class Config:
        extra = 'allow'  # Allow extra fields that aren't defined in the model

class TaskListResponse(BaseModel):
    tasks: List[TaskResponse]
    count: int
    timestamp: datetime

class GenerateRequest(BaseModel):
    task_id: str
    count: int = Field(10, gt=0, le=1000)
    model: str = Field('deepseek-chat', description="DeepSeek model to use (deepseek-chat or deepseek-reasoner)")

class Entry(BaseModel):
    id: Optional[str] = None
    input: Optional[str] = None
    output: Optional[str] = None
    raw_data: Optional[Dict[str, Any]] = None 

    class Config:
        extra = 'allow'

class DatasetEntry(BaseModel):
    """Proper dataset entry format dataset"""
    id: str
    content: Dict[str, Any]
    metadata: Dict[str, Any]

class QualityReport(BaseModel):
    generated_entries: Optional[int] = None
    quality_score: Optional[float] = None
    duplicates_removed: Optional[int] = None
    average_length: Optional[float] = None
    details: Optional[Dict[str, Any]] = None

    class Config:
        extra = 'allow'

class GenerateResponse(BaseModel):
    entries: List[DatasetEntry]
    quality_report: QualityReport
    generated_at: datetime
    task_id: str
    count: int

class TestGenerationRequest(BaseModel):
    task_id: str
    model: str = Field('deepseek-chat', description="DeepSeek model to use (deepseek-chat or deepseek-reasoner)")

class TestGenerationResponse(BaseModel):
    test_entries: List[DatasetEntry]
    quality_report: QualityReport
    status: str
    task_id: str

class QualityConfig(BaseModel):
    min_length: int = Field(10, ge=0)
    max_length: int = Field(1000, gt=0)
    required_fields: Optional[List[str]] = []
    custom_validators: Optional[List[str]] = []
    similarity_threshold: float = Field(0.8, ge=0, le=1)

class QualityConfigResponse(BaseModel):
    config: QualityConfig

class MessageResponse(BaseModel):
    message: str
    task: Optional[TaskResponse] = None
    config: Optional[QualityConfig] = None

class CachedDatasetInfo(BaseModel):
    task_id: str
    generated_at: Optional[datetime] = None
    entry_count: Optional[int] = None
    file_size: Optional[int] = None

class CachedDatasetListResponse(BaseModel):
    cached_datasets: List[CachedDatasetInfo]

class StatusResponse(BaseModel):
    status: str  # "healthy", "degraded", "error"
    deepseek_api_configured: bool  # Fixed field name to match frontend
    deepseek_client_available: bool
    tasks_loaded: int
    timestamp: datetime
    version: str
    error: Optional[str] = None

class HealthResponse(BaseModel):
    status: str
    timestamp: datetime

# Add missing API Key models
class ApiKeyRequest(BaseModel):
    api_key: str = Field(..., description="DeepSeek API key starting with 'sk-'")

class ApiKeyResponse(BaseModel):
    message: str
    valid: bool

class ApiKeyTestRequest(BaseModel):
    api_key: str = Field(..., description="DeepSeek API key to test")

class ApiKeyTestResponse(BaseModel):
    valid: bool
    message: str
    model_accessible: Optional[str] = None

# --- Full DeepSeek Client Implementation (from generate_dataset.py) ---
class DeepSeekClient:
    def __init__(self, api_key: str, model: str = "deepseek-chat", temperature: float = 1.0, base_url: str = "https://api.deepseek.com"):
        """
        Full DeepSeek client for high-quality dataset generation
        """
        self.api_key = api_key
        self.api_url = f"{base_url}/chat/completions"
        self.model = model
        self.temperature = temperature
        self.max_retries = 3
        self.retry_delay = 5
        self.cache = {}
        self.base_url = base_url

    def get_openai_client(self):
        return OpenAI(api_key=self.api_key, base_url=self.base_url)

    def create_optimized_prompt(self, task: dict, count: int, examples=None, advanced_option=True) -> str:
        """Create optimized prompt for dataset generation"""
        cache_key = f"{task.get('name', task.get('id', 'unknown'))}_{advanced_option}"
        
        if cache_key in self.cache:
            prompt_template = self.cache[cache_key]
            prompt = prompt_template.replace("{count}", str(count))
            return prompt

        schema_description = json.dumps(
            task.get("schema", {}).get("fields", {}), indent=2, ensure_ascii=False
        )
        task_name = task.get('name', task.get('id', 'unknown'))
        task_description = task.get('description', 'Unknown task')

        if advanced_option:
            prompt = f"""
# Task: {task_name} Dataset Generation

## Description
{task_description}

## Schema
```json
{schema_description}
```

## Requirements
สร้างชุดข้อมูล JSON จำนวน {count} รายการตาม Schema ข้างต้น โดยคำนึงถึงคุณภาพและความหลากหลาย:

## Requirements
สร้างชุดข้อมูล JSON จำนวน {count} รายการตาม Schema ข้างต้น โดยคำนึงถึงคุณภาพและความหลากหลาย:
- ใช้ภาษาไทยเท่านั้น (100% Thai language only) ห้ามใช้ภาษาอื่นแม้แต่คำเดียว
- ข้อความต้องสมจริง หลากหลาย และไม่ซ้ำซาก
- ความยาวต้องหลากหลาย (ตั้งแต่สั้นถึงยาว) แต่มีความสมบูรณ์
- เนื้อหาต้องกระจายตัวดี ครอบคลุมหลายบริบทและหัวข้อ
- ใช้ภาษาไทยที่เป็นธรรมชาติ ถูกต้องตามหลักภาษาไทย
- หลีกเลี่ยงข้อมูลซ้ำ คำซ้ำ และโครงสร้างประโยคซ้ำ
- ไม่มีคำว่า "example", "ตัวอย่าง" หรือ placeholder ในเนื้อหา
- เน้นความสมดุลระหว่าง category/label (ถ้ามี)

## Output Format
JSON Array ที่มีโครงสร้างตาม Schema โดยไม่มีข้อความนำหรือสรุป เช่น:
```json
[
  {{
    "field1": "value1",
    "field2": "value2"
  }},
  {{
    "field1": "value3",
    "field2": "value4"
  }}
]
```

โปรดส่งเฉพาะ JSON array เท่านั้น ไม่ต้องใส่คำอธิบายหรือข้อความอื่นๆ
"""
        else:
            prompt = f"""
คุณคือ AI ผู้เชี่ยวชาญในการสร้าง dataset คุณภาพสูงสำหรับงาน NLP และ AI ในภาษาไทย

โจทย์: {task_name}
รายละเอียด: {task_description}
Schema: {schema_description}

กรุณาสร้างตัวอย่างข้อมูลที่มีคุณภาพสูง จำนวน {count} ตัวอย่าง ในรูปแบบ JSON array ตาม schema ข้างต้น

เงื่อนไขสำคัญ:
1. ผลลัพธ์ต้องเป็น JSON array ที่ถูกต้อง
2. ไม่ต้องอธิบายเพิ่มเติม ส่งเฉพาะ JSON array เท่านั้น
3. ห้ามใช้คำว่า 'example', 'ตัวอย่าง', หรือ placeholder อื่นๆ ในเนื้อหา
4. ใช้ภาษาไทยเป็นหลัก
5. ข้อมูลต้องเป็นไปตามเงื่อนไขของ Schema ที่กำหนด
"""

        self.cache[cache_key] = prompt
        return prompt

    def generate_dataset_with_prompt(self, task: dict, count: int, examples=None, advanced_prompt=True) -> list:
        """Generate dataset using optimized prompt"""
        prompt = self.create_optimized_prompt(task, count, examples, advanced_prompt)
        
        system_prompt = (
            "You are a highly skilled dataset generator specialized in creating high-quality Thai language datasets. "
            "You must follow the schema exactly and only output valid JSON data. "
            "Your outputs should be diverse in length, style and content while maintaining natural Thai language usage. "
            "Only output the JSON array with no additional text, explanations or comments."
        )

        schema_fields = list(task.get("schema", {}).get("fields", {}).keys())

        # Use OpenAI client for deepseek-reasoner and deepseek-chat
        if self.model in ["deepseek-reasoner", "deepseek-chat"]:
            client = OpenAI(api_key=self.api_key, base_url="https://api.deepseek.com")
            messages = [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": prompt},
            ]
            for attempt in range(self.max_retries):
                try:
                    response = client.chat.completions.create(
                        model=self.model,
                        messages=messages,
                    )
                    # Try reasoning_content first (for deepseek-reasoner), fallback to content
                    content = ""
                    try:
                        content = response.choices[0].message.reasoning_content
                    except Exception:
                        content = response.choices[0].message.content
                    parsed_entries = self.parse_response_content(content)
                    return [
                        {field: entry.get(field) for field in schema_fields}
                        for entry in parsed_entries if isinstance(entry, dict)
                    ]
                except Exception as e:
                    if attempt == self.max_retries - 1:
                        logger.error(f"API request failed after {self.max_retries} attempts: {e}")
                        return []
                    time.sleep(self.retry_delay * (2**attempt))
            return []
        # Fallback: legacy requests for other models
        req_body = {
            "model": self.model,
            "temperature": self.temperature,
            "max_tokens": min(count * 200, 4000),
            "response_format": {"type": "json_object"},
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": prompt},
            ],
        }
        headers = {"Authorization": f"Bearer {self.api_key}"}
        
        for attempt in range(self.max_retries):
            try:
                resp = requests.post(self.api_url, json=req_body, headers=headers, timeout=60)
                resp.raise_for_status()
                resp_json = resp.json()
                content = resp_json["choices"][0]["message"]["content"]
                parsed_entries = self.parse_response_content(content)
                return [
                    {field: entry.get(field) for field in schema_fields}
                    for entry in parsed_entries if isinstance(entry, dict)
                ]
            except Exception as e:
                if attempt == self.max_retries - 1:
                    logger.error(f"API request failed after {self.max_retries} attempts: {e}")
                    return []
                time.sleep(self.retry_delay * (2**attempt))
        return []

    def parse_response_content(self, content: str) -> list:
        """Parse API response content to extract JSON data"""
        def clean_dict(d: dict) -> dict:
            cleaned = {}
            for k, v in d.items():
                if isinstance(v, str):
                    v = re.sub(r"\s+", " ", v.strip())
                cleaned[k] = v
            return cleaned

        try:
            parsed = json.loads(content)
            if isinstance(parsed, list):
                return [clean_dict(item) for item in parsed if isinstance(item, dict)]
            elif isinstance(parsed, dict):
                for key in ["data", "results", "entries", "items"]:
                    if key in parsed and isinstance(parsed[key], list):
                        return [clean_dict(item) for item in parsed[key] if isinstance(item, dict)]
                return [clean_dict(parsed)]
        except json.JSONDecodeError:
            # Try to extract JSON array from text
            matches = list(re.finditer(r"\[(?:[^[\]]*|\[(?:[^[\]]*|\[[^[\]]*\])*\])*\]", content))
            for match in matches:
                try:
                    array_str = match.group(0)
                    parsed = json.loads(array_str)
                    if isinstance(parsed, list):
                        return [clean_dict(item) for item in parsed if isinstance(item, dict)]
                except:
                    continue
        return []

    def generate_dataset_batch(self, task: dict, total_count: int, batch_size: int = 10, **kwargs) -> list:
        """Generate dataset in batches for better quality"""
        all_entries = []
        for batch_start in range(0, total_count, batch_size):
            batch_count = min(batch_size, total_count - batch_start)
            entries = self.generate_dataset_with_prompt(task, batch_count)
            all_entries.extend(entries)
            if batch_start + batch_size < total_count:
                time.sleep(2)  # Rate limiting
        return all_entries

# --- Ollama Client Implementation ---
class OllamaClient:
    def __init__(self, model: str = "qwen3:1.7b", temperature: float = 1.0, base_url: str = "http://localhost:11434"):
        """
        Ollama client for local model dataset generation
        """
        self.model = model
        self.api_url = f"{base_url}/api/generate"
        self.temperature = temperature
        self.max_retries = 3
        self.retry_delay = 2
        self.cache = {}

    def create_optimized_prompt(self, task: dict, count: int, examples=None, advanced_option=True) -> str:
        """Create optimized prompt for dataset generation"""
        cache_key = f"{task.get('name', task.get('id', 'unknown'))}_{advanced_option}"
        
        if cache_key in self.cache:
            prompt_template = self.cache[cache_key]
            prompt = prompt_template.replace("{count}", str(count))
            return prompt

        schema_description = json.dumps(
            task.get("schema", {}).get("fields", {}), indent=2, ensure_ascii=False
        )

        if advanced_option:
            prompt = f"""# Task: {task.get('name', 'Unknown')} Dataset Generation

## Description
{task.get('description', 'No description available')}

## Schema
```json
{schema_description}
```

## Requirements
สร้างชุดข้อมูล JSON จำนวน {count} รายการตาม Schema ข้างต้น โดยคำนึงถึงคุณภาพและความหลากหลาย:
- ใช้ภาษาไทยเท่านั้น (100% Thai language only) ห้ามใช้ภาษาอื่นแม้แต่คำเดียว
- ข้อความต้องสมจริง หลากหลาย และไม่ซ้ำซาก
- ข้อมูลต้องมีคุณภาพสูงและเหมาะสมสำหรับการเรียนรู้ของ AI
- แต่ละรายการต้องแตกต่างกันอย่างชัดเจน
- ห้ามมีข้อมูลที่ซ้ำกันหรือคล้ายกันเกินไป
- ตอบด้วย JSON array เท่านั้น ห้ามมีข้อความอธิบายเพิ่มเติม

กรุณาสร้างข้อมูลตาม Schema และส่งคืนเป็น JSON array ที่มี {count} รายการ:"""

            if examples:
                examples_json = json.dumps(
                    examples[:min(3, len(examples))], indent=2, ensure_ascii=False
                )
                prompt += f"""

## Examples
ตัวอย่างข้อมูลที่ดี:
```json
{examples_json}
```"""

            prompt += """

## Output Format
ส่งคืนเฉพาะ JSON array เท่านั้น:
[
  {
    "field1": "value1",
    "field2": "value2"
  },
  ...
]"""
        else:
            prompt = f"""สร้างชุดข้อมูล {count} รายการสำหรับ {task.get('name', 'dataset')} ตาม schema: {schema_description}
ใช้ภาษาไทยเท่านั้น ส่งคืนเป็น JSON array"""

        self.cache[cache_key] = prompt
        return prompt

    def generate_dataset_with_prompt(self, task: dict, count: int, examples=None, advanced_prompt=True) -> list:
        """Generate dataset using optimized prompt"""
        prompt = self.create_optimized_prompt(task, count, examples, advanced_prompt)
        
        req_body = {
            "model": self.model,
            "prompt": prompt,
            "stream": False,
            "options": {
                "temperature": self.temperature,
                "num_predict": min(count * 150, 3000),
                "top_p": 0.9
            }
        }
        
        for attempt in range(self.max_retries):
            try:
                # Increase timeout to 180 seconds for larger models
                resp = requests.post(self.api_url, json=req_body, timeout=180)
                resp.raise_for_status()
                resp_json = resp.json()
                content = resp_json.get("response", "")
                
                # Parse JSON response
                schema_fields = list(task.get("schema", {}).get("fields", {}).keys())
                parsed_entries = self.parse_response_content(content)
                
                # Map only schema fields for each entry
                return [
                    {field: entry.get(field) for field in schema_fields}
                    for entry in parsed_entries if isinstance(entry, dict)
                ]
            except requests.exceptions.ReadTimeout as e:
                logger.error(f"Ollama API timeout (attempt {attempt + 1}): {e}")
                if attempt == self.max_retries - 1:
                    logger.error(f"Ollama API request failed after {self.max_retries} attempts due to timeout")
                    # Return empty list instead of raising to prevent complete failure
                    return []
                # Increase delay for timeout retries
                time.sleep(self.retry_delay * (2**(attempt + 1)))
            except Exception as e:
                logger.error(f"Ollama API request failed (attempt {attempt + 1}): {e}")
                if attempt == self.max_retries - 1:
                    logger.error(f"Ollama API request failed after {self.max_retries} attempts: {e}")
                    return []
                time.sleep(self.retry_delay * (2**attempt))
        return []

    def parse_response_content(self, content: str) -> list:
        """Parse API response content to extract JSON data"""
        try:
            # Try to find JSON array in the response
            json_start = content.find('[')
            json_end = content.rfind(']') + 1
            
            if json_start != -1 and json_end != -1:
                json_str = content[json_start:json_end]
                return json.loads(json_str)
            
            # Try to parse as direct JSON
            if content.strip().startswith('['):
                return json.loads(content.strip())
            
            # Try to extract multiple JSON objects
            lines = content.strip().split('\n')
            entries = []
            for line in lines:
                line = line.strip()
                if line.startswith('{') and line.endswith('}'):
                    try:
                        entries.append(json.loads(line))
                    except:
                        continue
            
            return entries if entries else []
            
        except json.JSONDecodeError as e:
            logger.error(f"JSON decode error: {e}")
            return []
        except Exception as e:
            logger.error(f"Content parsing error: {e}")
            return []

    def generate_dataset_batch(self, task: dict, total_count: int, batch_size: int = 5, **kwargs) -> list:
        """Generate dataset in batches for better quality with improved error handling"""
        all_entries = []
        batches = (total_count + batch_size - 1) // batch_size
        
        for i in range(batches):
            current_batch_size = min(batch_size, total_count - len(all_entries))
            if current_batch_size <= 0:
                break
                
            logger.info(f"Generating batch {i+1}/{batches} ({current_batch_size} entries)")
            
            try:
                batch_entries = self.generate_dataset_with_prompt(task, current_batch_size)
                
                if batch_entries:
                    all_entries.extend(batch_entries)
                    logger.info(f"Successfully generated {len(batch_entries)} entries in batch {i+1}")
                else:
                    logger.warning(f"No entries generated in batch {i+1}")
                    
            except Exception as e:
                logger.error(f"Error in batch {i+1}: {e}")
                # Continue with next batch instead of failing completely
                continue
            
            # Add delay between batches
            if i < batches - 1:
                time.sleep(1)
        
        logger.info(f"Total entries generated: {len(all_entries)} out of requested {total_count}")
        return all_entries[:total_count]
# --- Simple Task Manager ---
class SimpleTaskManager:
    def __init__(self, tasks_file: Path):
        self.tasks_file = tasks_file
        self._tasks = {}
        self.load_tasks()
    
    def load_tasks(self):
        """Load tasks from JSON file"""
        try:
            if self.tasks_file.exists():
                with open(self.tasks_file, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                    if isinstance(data, dict):
                        self._tasks = data
                    else:
                        self._tasks = {}
                logger.info(f"Loaded {len(self._tasks)} tasks from {self.tasks_file}")
            else:
                self._tasks = {}
                logger.info("No tasks file found, starting with empty tasks")
        except Exception as e:
            logger.error(f"Error loading tasks: {e}")
            self._tasks = {}
    
    def save_tasks(self):
        """Save tasks to JSON file"""
        try:
            with open(self.tasks_file, 'w', encoding='utf-8') as f:
                json.dump(self._tasks, f, indent=2, ensure_ascii=False)
            logger.info(f"Saved {len(self._tasks)} tasks")
        except Exception as e:
            logger.error(f"Error saving tasks: {e}")
    
    def list_tasks(self) -> List[Dict]:
        """Return list of all tasks"""
        task_list = []
        for task_id, task_data in self._tasks.items():
            task_with_id = {**task_data, 'id': task_id}
            task_list.append(task_with_id)
        return task_list
    
    def get_task(self, task_id: str) -> Optional[Dict]:
        """Get a specific task by ID"""
        task = self._tasks.get(task_id)
        if task:
            return {**task, 'id': task_id}
        return None
    
    def add_custom_task(self, task_data: Dict) -> bool:
        """Add a new custom task"""
        try:
            task_id = task_data.get('id')
            if not task_id:
                return False
            
            if task_id in self._tasks:
                return False  # Task already exists
            
            # Remove 'id' from task_data before storing
            task_data_copy = task_data.copy()
            task_data_copy.pop('id', None)
            self._tasks[task_id] = task_data_copy
            self.save_tasks()
            return True
        except Exception as e:
            logger.error(f"Error adding task: {e}")
            return False
    
    def remove_task(self, task_id: str) -> bool:
        """Remove a task by ID"""
        try:
            if task_id in self._tasks:
                del self._tasks[task_id]
                self.save_tasks()
                return True
            return False
        except Exception as e:
            logger.error(f"Error removing task: {e}")
            return False
    
    def update_task(self, task_id: str, task_data: Dict) -> bool:
        """Update an existing task"""
        try:
            if task_id not in self._tasks:
                return False
            
            # Remove 'id' from task_data if present
            task_data_copy = task_data.copy()
            task_data_copy.pop('id', None)
            
            # Update the task data
            self._tasks[task_id].update(task_data_copy)
            self.save_tasks()
            return True
        except Exception as e:
            logger.error(f"Error updating task: {e}")
            return False

# --- Simple Dataset Generation ---
# DocumentUnderstanding import will be handled within the endpoint

def real_generate_dataset(task: Dict, count: int, client: DeepSeekClient, use_rag: bool = True) -> tuple:
    """
    Real dataset generation using full DeepSeek client with proper output format
    Returns data in {id, content, metadata} format like translation dataset
    Enhanced with RAG context from uploaded PDFs
    """
    logger.info(f"[real_generate_dataset] Generating {count} entries for task: {task.get('id', 'unknown')}")
    
    try:
        # Check if RAG context is available and should be used
        rag_context = ""
        if use_rag:
            # Get RAG context based on task description and name
            task_query = f"{task.get('name', '')} {task.get('description', '')}"
            rag_context = get_rag_context(task_query)
            
            if rag_context:
                logger.info(f"[RAG] Found relevant context from uploaded PDFs (length: {len(rag_context)} chars)")
                # Temporarily modify the task to include RAG context
                original_description = task.get('description', '')
                enhanced_description = f"{original_description}\n\n## Reference Materials (from uploaded PDFs):\n{rag_context}\n\nโปรดใช้ข้อมูลอ้างอิงจาก PDF ข้างต้นเป็นแนวทางในการสร้างข้อมูล แต่ให้สร้างเนื้อหาใหม่ที่หลากหลายและไม่ซ้ำกับต้นฉบับ"
                task = dict(task)  # Create a copy
                task['description'] = enhanced_description
            else:
                logger.info("[RAG] No relevant context found from uploaded PDFs")
        
        # Use the full DeepSeek client for generation with enhanced task
        all_entries = client.generate_dataset_batch(
            task=task,
            total_count=count,
            batch_size=min(10, count),
            clean_data=True,
            advanced_prompt=True,
            post_process=True
        )
        
        # Convert to proper format with id, content, metadata
        formatted_entries = []
        task_name = task.get('name', task.get('id', 'unknown'))
        
        schema_fields = list(task.get("schema", {}).get("fields", {}).keys())
        for i, entry in enumerate(all_entries[:count], 1):
            # Fallback: if entry is nested (e.g. {"raw_data": {...}}), extract from raw_data
            if isinstance(entry, dict) and "raw_data" in entry and isinstance(entry["raw_data"], dict):
                entry = entry["raw_data"]
            content = {field: entry.get(field) for field in schema_fields}
            formatted_entry = {
                "id": f"{task_name}-{i}",
                "content": content,
                "metadata": {
                    "source": (
                        "DeepSeek-V3-0324" if client.model == "deepseek-chat"
                        else "DeepSeek-R1-0528" if client.model == "deepseek-reasoner"
                        else getattr(client, "provider", getattr(client, "model", "DeepSeek-V3"))
                    ),
                    "license": "CC-BY 4.0",
                    "version": "1.0.0",
                    "created_at": datetime.now().isoformat(),
                    "task": task_name,
                    "model": client.model,
                    "provider": getattr(client, "provider", "deepseek")
                }
            }
            formatted_entries.append(formatted_entry)
        
        # Create comprehensive quality report
        quality_report = {
            'generated_entries': len(formatted_entries),
            'quality_score': 0.95 if formatted_entries else 0.0, # High quality with real generation
            'duplicates_removed': max(0, len(all_entries) - len(formatted_entries)),
            'average_length': sum(len(str(entry['content'])) for entry in formatted_entries) / len(formatted_entries) if formatted_entries else 0,
            'details': {
                'total_generated': len(all_entries),
                'valid_entries': len(formatted_entries),
                'quality_rate': len(formatted_entries) / max(1, len(all_entries)),
                'issues_found': max(0, len(all_entries) - len(formatted_entries)),
                'final_count': len(formatted_entries),
                'generation_method': 'real_deepseek_api'
            }
        }
        
        logger.info(f"[real_generate_dataset] Successfully generated {len(formatted_entries)} entries")
        return formatted_entries, quality_report
        
    except Exception as e:
        logger.error(f"[real_generate_dataset] Error in dataset generation: {e}")
        # Do not create fallback entries - let the error propagate
        raise e

# --- AppConfig and Initialization ---
class AppConfig:
    def __init__(self):
        self.script_dir = PathLib(__file__).parent.absolute()
        self.project_root = self.script_dir.parent.parent
        self.python_dir = self.script_dir.parent / 'python'
        self.cache_dir = self.project_root / 'cache'
        self.config_dir = self.project_root / 'config'
        self.static_dir = self.script_dir / 'static'
        self.templates_dir = self.script_dir / 'templates'
        
        # Ensure directories exist
        self.cache_dir.mkdir(exist_ok=True)
        self.config_dir.mkdir(exist_ok=True)
        self.static_dir.mkdir(exist_ok=True)
        self.python_dir.mkdir(exist_ok=True)
        
        # File paths
        self.tasks_json_file = self.project_root / 'tasks.json'
        self.quality_config_file = self.config_dir / 'quality_config.json'

def initialize_app_config():
    """Initialize the application configuration"""
    config = AppConfig()
    
    if not config.tasks_json_file.exists():
        logger.info(f"Creating tasks.json at {config.tasks_json_file}")
        try:
            with open(config.tasks_json_file, 'w', encoding='utf-8') as f:
                json.dump({}, f, indent=2)
        except Exception as e:
            logger.error(f"Could not create tasks.json: {e}")
    
    if str(config.python_dir) not in sys.path:
        sys.path.insert(0, str(config.python_dir))
    
    return config

app_config = initialize_app_config()

# Initialize task manager
task_manager = SimpleTaskManager(app_config.tasks_json_file)

# --- FastAPI App Instance ---
app = FastAPI(title="DekDataset API", version="1.0.0")

# Mount static files
app.mount("/static", StaticFiles(directory=app_config.static_dir), name="static")

# Setup templates
templates = Jinja2Templates(directory=app_config.templates_dir)

# --- Global client instance and Dependencies ---
_deepseek_client: Optional[DeepSeekClient] = None
_ollama_client: Optional[OllamaClient] = None
_models_cache: Optional[List[str]] = None
_models_cache_time: Optional[float] = None
MODELS_CACHE_DURATION = 300  # 5 นาที

def get_deepseek_api_key():
    api_key = os.getenv('DEEPSEEK_API_KEY')
    if not api_key:
        logger.warning("DEEPSEEK_API_KEY environment variable not set.")
    return api_key

def get_deepseek_client_dependency(api_key: str = Depends(get_deepseek_api_key)) -> Optional[DeepSeekClient]:
    global _deepseek_client
    if not api_key:
        return None
    if _deepseek_client is None:
        _deepseek_client = DeepSeekClient(api_key=api_key)
    return _deepseek_client

def get_ollama_client_dependency() -> Optional[OllamaClient]:
    global _ollama_client
    if _ollama_client is None:
        try:
            # Test if Ollama is available
            response = requests.get("http://localhost:11434/api/tags", timeout=5)
            if response.status_code == 200:
                _ollama_client = OllamaClient()
            else:
                logger.warning("Ollama server not available")
                return None
        except Exception as e:
            logger.warning(f"Ollama not available: {e}")
            return None
    return _ollama_client

def get_available_ollama_models() -> List[str]:
    """Get list of available Ollama models with caching"""
    global _models_cache, _models_cache_time
    import time
    
    # ตรวจสอบ cache
    current_time = time.time()
    if _models_cache is not None and _models_cache_time is not None:
        if current_time - _models_cache_time < MODELS_CACHE_DURATION:
            return _models_cache
    
    try:
        response = requests.get("http://localhost:11434/api/tags", timeout=2)  # ลด timeout เป็น 2 วินาที
        if response.status_code == 200:
            data = response.json()
            models = [model["name"] for model in data.get("models", [])]
            # อัปเดต cache
            _models_cache = models
            _models_cache_time = current_time
            return models
        return []
    except Exception as e:
        logger.warning(f"Failed to get Ollama models: {e}")
        # คืนค่า cache เก่าถ้ามี
        return _models_cache or []

def get_generation_client(model: str, api_key: str = None):
    """Get appropriate client based on model selection"""
    if model.startswith('ollama:'):
        # Extract model name after 'ollama:'
        ollama_model = model[7:]  # Remove 'ollama:' prefix
        return OllamaClient(model=ollama_model)
    else:
        # Use DeepSeek for other models
        if not api_key:
            return None
        return DeepSeekClient(api_key=api_key, model=model)

def get_task_manager() -> SimpleTaskManager:
    """Get the global task manager instance"""
    return task_manager
def load_quality_config_sync():
    """Load quality control configuration"""
    try:
        if app_config.quality_config_file.exists():
            with open(app_config.quality_config_file, 'r') as f:
                return json.load(f)
        else:
            return {
                'min_length': 10,
                'max_length': 1000,
                'required_fields': [],
                'custom_validators': [],
                'similarity_threshold': 0.8
            }
    except Exception as e:
        logger.error(f"Error loading quality config: {e}")
        return {
            'min_length': 10,
            'max_length': 1000,
            'required_fields': [],
            'custom_validators': [],
            'similarity_threshold': 0.8
        }

# Experiment tracking functionality removed

# Experiment tracking functionality removed

# Experiment tracking functionality removed

# --- FastAPI Routes ---

@app.get("/", response_class=HTMLResponse, summary="Main Dashboard", tags=["UI"])
async def index(request: Request):
    tasks_data = []
    try:
        if app_config.tasks_json_file.exists():
            with open(app_config.tasks_json_file, 'r', encoding='utf-8') as f:
                loaded_tasks = json.load(f)
                
                if isinstance(loaded_tasks, dict):
                    for task_id, task_details in loaded_tasks.items():
                        if isinstance(task_details, dict):
                            task_details['id'] = task_details.get('id', task_id)
                            tasks_data.append(task_details)
                elif isinstance(loaded_tasks, list):
                    for task_item in loaded_tasks:
                        if isinstance(task_item, dict) and 'id' in task_item:
                            tasks_data.append(task_item)
    except Exception as e:
        logger.error(f"Error loading tasks for dropdown: {e}")

    # Ensure every task has a 'name' for display
    processed_tasks_data = []
    for task in tasks_data:
        if isinstance(task, dict):
            if 'name' not in task or not task['name']:
                task['name'] = task.get('id', 'Unnamed Task')
            if 'id' in task:
                processed_tasks_data.append(task)
    
    logger.info(f"Tasks being passed to template ({len(processed_tasks_data)} tasks)")
    return templates.TemplateResponse("index.html", {"request": request, "tasks": processed_tasks_data})

# Add the missing API endpoint that the frontend is looking for
@app.get("/api/app-config/tasks.json", summary="Get Tasks JSON", tags=["Tasks API"])
async def get_tasks_json():
    """Get tasks configuration as JSON for frontend."""
    try:
        if app_config.tasks_json_file.exists():
            with open(app_config.tasks_json_file, 'r', encoding='utf-8') as f:
                tasks_data = json.load(f)
                return JSONResponse(content=tasks_data)
        else:
            return JSONResponse(content={})
    except Exception as e:
        logger.error(f"Error loading tasks JSON: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/api/tasks", response_model=TaskListResponse, summary="Get All Tasks", tags=["Tasks API"])
async def get_tasks_api(tm = Depends(get_task_manager)):
    """Get all tasks from the task manager."""
    try:
        tasks_list = tm.list_tasks()
        processed_tasks = []
        
        for task_data in tasks_list:
            # Ensure we have the required 'id' field
            if 'id' not in task_data and 'task_id' in task_data:
                task_data['id'] = task_data.pop('task_id')
            
            # Ensure we have a name field for display
            if 'name' not in task_data or not task_data['name']:
                task_data['name'] = task_data.get('id', 'Unnamed Task')
            
            # Handle missing description
            if 'description' not in task_data:
                task_data['description'] = task_data.get('name', 'No description available')
            
            # Create TaskResponse with validation error handling
            try:
                processed_tasks.append(TaskResponse(**task_data))
            except Exception as e:
                logger.warning(f"Skipping invalid task {task_data.get('id', 'unknown')}: {e}")
                continue

        return TaskListResponse(
            tasks=processed_tasks,
            count=len(processed_tasks),
            timestamp=datetime.now()
        )
    except Exception as e:
        logger.error(f"Error getting tasks: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/api/tasks", response_model=MessageResponse, status_code=201, summary="Create New Task", tags=["Tasks API"])
async def create_task_api(task_data: TaskCreate, tm = Depends(get_task_manager)):
    """Create a new custom task."""
    try:
        task_dict = task_data.model_dump()
        task_dict['created_at'] = datetime.now().isoformat()
        
        # Ensure name field exists
        if 'name' not in task_dict or not task_dict['name']:
            task_dict['name'] = task_dict.get('id', 'Unnamed Task')

        success = tm.add_custom_task(task_dict)
        if not success:
            if tm.get_task(task_data.id):
                raise HTTPException(status_code=409, detail="Task ID already exists")
            raise HTTPException(status_code=500, detail="Failed to create task")
        
        created_task_info = tm.get_task(task_data.id)
        if not created_task_info:
            raise HTTPException(status_code=500, detail="Task created but could not be retrieved")

        logger.info(f"Created task: {task_data.id}")
        return MessageResponse(
            message="Task created successfully",
            task=TaskResponse(**created_task_info)
        )
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error creating task: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/api/tasks/{task_id}", response_model=TaskResponse, summary="Get Specific Task", tags=["Tasks API"])
async def get_task_api(task_id: str = Path(..., title="The ID of the task to get"), tm = Depends(get_task_manager)):
    """Get a specific task by its ID."""
    try:
        task = tm.get_task(task_id)
        if not task:
            raise HTTPException(status_code=404, detail="Task not found")
        
        # Ensure required fields
        if 'id' not in task and 'task_id' in task:
            task['id'] = task.pop('task_id')
        elif 'id' not in task and task_id:
            task['id'] = task_id
            
        if 'name' not in task or not task['name']:
            task['name'] = task.get('id', 'Unnamed Task')
            
        if 'description' not in task:
            task['description'] = task.get('name', 'No description available')

        return TaskResponse(**task)
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error getting task {task_id}: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.delete("/api/tasks/{task_id}", response_model=MessageResponse, summary="Delete Task", tags=["Tasks API"])
async def delete_task_api(task_id: str = Path(..., title="The ID of the task to delete"), tm = Depends(get_task_manager)):
    """Delete a task by its ID."""
    try:
        if not tm.get_task(task_id):
            raise HTTPException(status_code=404, detail="Task not found")
            
        success = tm.remove_task(task_id)
        if not success:
            raise HTTPException(status_code=404, detail="Task not found or could not be removed")
        
        logger.info(f"Deleted task: {task_id}")
        return MessageResponse(message="Task deleted successfully")
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error deleting task {task_id}: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.put("/api/tasks/{task_id}", response_model=MessageResponse, summary="Update Task", tags=["Tasks API"])
async def update_task_api(task_id: str = Path(..., title="The ID of the task to update"),
                         task_data: Dict[str, Any] = Body(...), 
                         tm = Depends(get_task_manager)):
    """Update an existing task by its ID."""
    try:
        if not tm.get_task(task_id):
            raise HTTPException(status_code=404, detail="Task not found")
        
        # Remove 'id' from task_data if present to avoid conflicts
        task_data_copy = task_data.copy()
        task_data_copy.pop('id', None)
        
        success = tm.update_task(task_id, task_data_copy)
        if not success:
            raise HTTPException(status_code=400, detail="Failed to update task")
        
        logger.info(f"Updated task: {task_id}")
        return MessageResponse(message="Task updated successfully")
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error updating task {task_id}: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/api/tasks/{task_id}/config", summary="Get Task Configuration", tags=["Tasks API"])
async def get_task_config_api(task_id: str = Path(..., title="The ID of the task to get config"),
                             tm = Depends(get_task_manager)):
    """Get the complete configuration for a specific task."""
    try:
        task = tm.get_task(task_id)
        if not task:
            raise HTTPException(status_code=404, detail="Task not found")
        
        logger.info(f"Retrieved configuration for task: {task_id}")
        return JSONResponse(content=task)
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error getting task configuration {task_id}: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/api/tasks/{task_id}/config", response_model=MessageResponse, summary="Update Task Configuration", tags=["Tasks API"])
async def update_task_config_api(task_id: str = Path(..., title="The ID of the task to update config"),
                                config_data: Dict[str, Any] = Body(...),
                                tm = Depends(get_task_manager)):
    """Update the complete configuration for a specific task."""
    try:
        if not tm.get_task(task_id):
            raise HTTPException(status_code=404, detail="Task not found")
        
        # Remove 'id' from config_data if present to avoid conflicts
        config_data_copy = config_data.copy()
        config_data_copy.pop('id', None)
        
        success = tm.update_task(task_id, config_data_copy)
        if not success:
            raise HTTPException(status_code=400, detail="Failed to update task configuration")
        
        logger.info(f"Updated configuration for task: {task_id}")
        return MessageResponse(message="Task configuration updated successfully")
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error updating task configuration {task_id}: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/api/generate", response_model=GenerateResponse, summary="Generate Dataset", tags=["Dataset Generation API"])
async def generate_dataset_api(
    payload: GenerateRequest,
    tm = Depends(get_task_manager),
    api_key: str = Depends(get_deepseek_api_key)
):
    """Generate dataset based on model selection (DeepSeek or Ollama)."""
    try:
        task = tm.get_task(payload.task_id)
        if not task:
            raise HTTPException(status_code=404, detail="Task not found")

        logger.info(f"Generating {payload.count} entries for task {payload.task_id} using model {payload.model}")
        if 'id' not in task:
            task['id'] = payload.task_id

        # Get appropriate client based on model selection
        client = get_generation_client(payload.model, api_key)
        if client is None:
            if payload.model.startswith('ollama:'):
                raise HTTPException(status_code=503, detail="Ollama server is not available. Please ensure Ollama is running on localhost:11434")
            else:
                raise HTTPException(status_code=503, detail="DeepSeek client is not available. Check API key configuration.")

        # Set model on client (for DeepSeek clients)
        if hasattr(client, 'model'):
            if payload.model.startswith('ollama:'):
                # For Ollama, the model is already set in the constructor
                pass
            else:
                client.model = payload.model
        
        # Use real generation function with proper format
        generation_start_time = datetime.now()
        
        # Use appropriate generation method based on client type
        if isinstance(client, OllamaClient):
            # Use Ollama client with better error handling
            try:
                all_entries = client.generate_dataset_batch(
                    task=task,
                    total_count=payload.count,
                    batch_size=min(3, payload.count)  # Even smaller batches for Ollama to reduce timeout risk
                )
            except requests.exceptions.ReadTimeout:
                # Handle timeout gracefully
                logger.warning("Ollama generation timed out")
                raise HTTPException(
                    status_code=408, 
                    detail="Generation timed out. The model might be taking too long to respond. Try reducing the number of entries or check if the model is available."
                )
            except Exception as e:
                logger.error(f"Ollama generation failed: {e}")
                # If Ollama completely fails, provide a helpful error message
                raise HTTPException(
                    status_code=503, 
                    detail=f"Ollama generation failed: {str(e)}. Please check if Ollama is running and the model '{client.model}' is available."
                )
            
            # Convert to proper format with id, content, metadata
            formatted_entries = []
            task_name = task.get('name', task.get('id', 'unknown'))
            schema_fields = list(task.get("schema", {}).get("fields", {}).keys())
            
            for i, entry in enumerate(all_entries[:payload.count], 1):
                if isinstance(entry, dict):
                    content = {field: entry.get(field) for field in schema_fields}
                    formatted_entry = {
                        "id": f"{task_name}-{i}",
                        "content": content,
                        "metadata": {
                            "source": f"Ollama-{client.model}",
                            "model": client.model,
                            "provider": "ollama",
                            "generated_at": datetime.now().isoformat(),
                            "task_id": payload.task_id,
                            "generation_method": "ollama_batch"
                        }
                    }
                    formatted_entries.append(formatted_entry)
            
            # If no entries were generated, provide fallback
            if not formatted_entries:
                logger.warning("No entries generated by Ollama, creating minimal fallback")
                formatted_entries = [{
                    'id': f"{task_name}-fallback-1",
                    'content': {'error': 'Ollama generation failed - timeout or model issue'},
                    'metadata': {
                        'source': getattr(client, "provider", f"Ollama-{client.model}"),
                        'model': client.model,
                        'provider': getattr(client, "provider", "ollama"),
                        'generated_at': datetime.now().isoformat(),
                        'task_id': payload.task_id,
                        'generation_method': 'ollama_fallback',
                        'error': 'Generation timeout or failure'
                    }
                }]
            
            entries_raw = formatted_entries
            quality_report_raw = {
                'generated_entries': len(entries_raw), 
                'quality_score': 0.8 if len([e for e in entries_raw if 'error' not in e.get('content', {})]) > 0 else 0.0,
                'duplicates_removed': 0,
                'average_length': sum(len(str(entry['content'])) for entry in entries_raw) / len(entries_raw) if entries_raw else 0,
                'details': {
                    'total_generated': len(entries_raw),
                    'valid_entries': len([e for e in entries_raw if 'error' not in e.get('content', {})]),
                    'issues_found': len([e for e in entries_raw if 'error' in e.get('content', {})]),                    'final_count': len(entries_raw),
                    'generation_method': 'ollama_batch'
                }
            }
        else:
            # Use DeepSeek client with timeout handling and RAG support
            try:
                entries_raw, quality_report_raw = real_generate_dataset(
                    task=task,
                    count=payload.count,
                    client=client,
                    use_rag=True
                )
            except requests.exceptions.ReadTimeout:
                logger.warning("DeepSeek generation timed out")
                raise HTTPException(
                    status_code=408, 
                    detail="Generation timed out. The API might be experiencing high load. Please try again with fewer entries."
                )

        # Convert to DatasetEntry format (entries_raw should already be in proper format)
        processed_entries = []
        for entry in entries_raw:
            if isinstance(entry, dict) and 'id' in entry and 'content' in entry and 'metadata' in entry:
                # Already in proper format
                processed_entries.append(DatasetEntry(**entry))
            else:
                # Fallback: create proper format from raw entry
                processed_entries.append(DatasetEntry(
                    id=entry.get('id', f"entry-{len(processed_entries)+1}"),
                    content=entry.get('content', entry),
                    metadata=entry.get('metadata', {
                        "source": client.model if hasattr(client, 'model') else "Unknown", 
                        "generated_at": datetime.now().isoformat(),
                        "task_id": payload.task_id
                    })
                ))
        
        processed_quality_report = QualityReport(**quality_report_raw) if isinstance(quality_report_raw, dict) else QualityReport()

        result_data = {
            'entries': processed_entries,
            'quality_report': processed_quality_report,
            'generated_at': datetime.now(),
            'task_id': payload.task_id,
            'count': len(processed_entries)
        }

        # Save dataset to cache with proper format
        cache_data = result_data.copy()
        cache_data['entries'] = [entry.model_dump() for entry in processed_entries]  # Save the proper format to cache
        cache_data['quality_report'] = processed_quality_report.model_dump()
        cache_data['generated_at'] = cache_data['generated_at'].isoformat()
        save_dataset_to_cache_sync(payload.task_id, cache_data)
        
        logger.info(f"Generated {len(processed_entries)} entries for task {payload.task_id} with real generation")
        return GenerateResponse(**result_data)
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error generating dataset: {e}")
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=f"Dataset generation failed: {str(e)}")

@app.post("/api/test-generation", response_model=TestGenerationResponse, summary="Test Dataset Generation", tags=["Dataset Generation API"])
async def test_generation_api(
    payload: TestGenerationRequest,
    tm = Depends(get_task_manager),
    api_key: str = Depends(get_deepseek_api_key)
):
    """Test dataset generation with a small sample using selected model (DeepSeek or Ollama)."""
    try:
        task = tm.get_task(payload.task_id)
        if not task:
            raise HTTPException(status_code=404, detail="Task not found")

        logger.info(f"Testing generation for task {payload.task_id} using model {payload.model}")
        if 'id' not in task:
            task['id'] = payload.task_id

        # Get appropriate client based on model selection
        client = get_generation_client(payload.model, api_key)
        if client is None:
            if payload.model.startswith('ollama:'):
                raise HTTPException(status_code=503, detail="Ollama server is not available. Please ensure Ollama is running on localhost:11434")
            else:
                raise HTTPException(status_code=503, detail="DeepSeek client is not available. Check API key configuration.")

        # Set model on client (for DeepSeek clients)
        if hasattr(client, 'model') and not payload.model.startswith('ollama:'):
            client.model = payload.model
        
        # Generate small test sample with appropriate client
        generation_start_time = datetime.now()
        
        if isinstance(client, OllamaClient):
            # Use Ollama client for test generation with shorter timeout
            try:
                test_entries = client.generate_dataset_with_prompt(
                    task=task,
                    count=3
                )
            except requests.exceptions.ReadTimeout:
                raise HTTPException(
                    status_code=408, 
                    detail="Test generation timed out. The model might be taking too long to respond."
                )
            
            # Convert to proper format
            formatted_entries = []
            task_name = task.get('name', task.get('id', 'unknown'))
            schema_fields = list(task.get("schema", {}).get("fields", {}).keys())
            
            for i, entry in enumerate(test_entries[:3], 1):
                if isinstance(entry, dict):
                    content = {field: entry.get(field) for field in schema_fields}
                    formatted_entry = {
                        "id": f"test-{task_name}-{i}",
                        "content": content,
                        "metadata": {
                            "source": f"Ollama-{client.model}",
                            "generated_at": datetime.now().isoformat(),
                            "task_id": payload.task_id,
                            "test": True,
                            "generation_method": "ollama"
                        }
                    }
                    formatted_entries.append(formatted_entry)
            
            entries_raw = formatted_entries
            quality_report_raw = {
                'generated_entries': len(entries_raw), 
                'quality_score': 0.8,
                'duplicates_removed': 0,
                'average_length': sum(len(str(entry['content'])) for entry in entries_raw) / len(entries_raw) if entries_raw else 0,
                'details': {
                    'total_generated': len(entries_raw),
                    'valid_entries': len(entries_raw),
                    'error_rate': 0.0,
                    'generation_time': (datetime.now() - generation_start_time).total_seconds(),
                    'test_mode': True
                }
            }
        else:
            # Use DeepSeek client with timeout handling
            try:
                entries_raw, quality_report_raw = real_generate_dataset(
                    task=task,
                    count=3,
                    client=client
                )
            except requests.exceptions.ReadTimeout:
                raise HTTPException(
                    status_code=408, 
                    detail="Test generation timed out. Please try again or check your connection."
                )
        
        # Convert to DatasetEntry format
        processed_entries = []
        for entry in entries_raw:
            if isinstance(entry, dict) and 'id' in entry and 'content' in entry and 'metadata' in entry:
                # Already in proper format
                processed_entries.append(DatasetEntry(**entry))
            else:
                # Fallback: create proper format from raw entry
                processed_entries.append(DatasetEntry(
                    id=entry.get('id', f"test-entry-{len(processed_entries)+1}"),
                    content=entry.get('content', entry),
                    metadata=entry.get('metadata', {
                        "source": client.model if hasattr(client, 'model') else "Unknown", 
                        "generated_at": datetime.now().isoformat(),
                        "task_id": payload.task_id,
                        "test": True
                    })
                ))
        
        processed_quality_report = QualityReport(**quality_report_raw) if isinstance(quality_report_raw, dict) else QualityReport()
        
        return TestGenerationResponse(
            test_entries=processed_entries,
            quality_report=processed_quality_report,
            status='success',
            task_id=payload.task_id
        )
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error in test generation: {e}")
        raise HTTPException(status_code=500, detail=f"Test generation failed: {str(e)}")

@app.get("/api/quality-config", response_model=QualityConfigResponse, summary="Get Quality Config", tags=["Quality Control API"])
async def get_quality_config_api():
    """Get current quality control configuration."""
    try:
        config_dict = load_quality_config_sync()
        return QualityConfigResponse(config=QualityConfig(**config_dict))
    except Exception as e:
        logger.error(f"Error getting quality config: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/api/quality-config", response_model=MessageResponse, summary="Update Quality Config", tags=["Quality Control API"])
async def update_quality_config_api(config_update: QualityConfig):
    """Update quality control configuration."""
    try:
        if config_update.min_length >= config_update.max_length:
            raise HTTPException(status_code=400, detail="min_length must be less than max_length")

        with open(app_config.quality_config_file, 'w') as f:
            json.dump(config_update.model_dump(), f, indent=2)
        
        logger.info("Quality config updated")
        return MessageResponse(
            message="Quality config updated successfully",
            config=config_update
        )
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error updating quality config: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/api/status", response_model=StatusResponse, summary="API Status Check", tags=["System API"])
async def get_api_status():
    """Get API status and health information."""
    try:
        # Check if we can create a DeepSeek client
        api_key = get_deepseek_api_key()
        client_available = api_key is not None
        
        # Check if tasks are loaded
        tm = get_task_manager()
        tasks_count = len(tm.list_tasks())
        
        status = {
            "status": "healthy" if client_available else "degraded",
            "deepseek_api_configured": client_available,  # For frontend compatibility
            "deepseek_client_available": client_available,
            "tasks_loaded": tasks_count,
            "timestamp": datetime.now(),
            "version": "1.0.0"
        }
        
        return StatusResponse(**status)
    except Exception as e:
        logger.error(f"Error checking API status: {e}")
        return StatusResponse(
            status="error",
            deepseek_api_configured=False,
            deepseek_client_available=False,
            tasks_loaded=0,
            timestamp=datetime.now(),
            version="1.0.0",
            error=str(e)
        )

@app.post("/api/config/api-key", response_model=ApiKeyResponse, summary="Save API Key", tags=["Configuration API"])
async def save_api_key(payload: ApiKeyRequest):
    """Save and validate API key configuration."""
    try:
        api_key = payload.api_key.strip()
        
        if not api_key:
            raise HTTPException(status_code=400, detail="API key cannot be empty")
        
        if not api_key.startswith('sk-'):
            raise HTTPException(status_code=400, detail="Invalid API key format. API key should start with 'sk-'")
        
        # Test the API key by making a simple request
        try:
            test_client = DeepSeekClient(api_key=api_key)
            openai_client = test_client.get_openai_client()
            
            # Make a simple test request
            response = openai_client.chat.completions.create(
                model="deepseek-chat",
                messages=[{"role": "user", "content": "Hello"}],
                max_tokens=10,
                timeout=10
            )
            
            if response and response.choices:
                # API key is valid - store it in environment (session-based)
                os.environ['DEEPSEEK_API_KEY'] = api_key
                
                # Update global client
                global _deepseek_client
                _deepseek_client = test_client
                
                logger.info("API key saved and validated successfully")
                return ApiKeyResponse(
                    message="API key saved and validated successfully",
                    valid=True
                )
            else:
                raise HTTPException(status_code=400, detail="Invalid API key - no response from DeepSeek API")
                
        except Exception as e:
            logger.error(f"API key validation failed: {e}")
            raise HTTPException(status_code=400, detail=f"Invalid API key: {str(e)}")
            
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error saving API key: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to save API key: {str(e)}")

@app.post("/api/test-api-key", response_model=ApiKeyTestResponse, summary="Test API Key", tags=["Configuration API"])
async def test_api_key(payload: ApiKeyTestRequest):
    """Test API key validity without saving."""
    try:
        api_key = payload.api_key.strip()
        
        if not api_key:
            return ApiKeyTestResponse(
                valid=False,
                message="API key cannot be empty"
            )
        
        if not api_key.startswith('sk-'):
            return ApiKeyTestResponse(
                valid=False,
                message="Invalid API key format. API key should start with 'sk-'"
            )
        
        # Test the API key
        try:
            test_client = DeepSeekClient(api_key=api_key)
            openai_client = test_client.get_openai_client()
            
            # Make a simple test request with timeout
            response = openai_client.chat.completions.create(
                model="deepseek-chat",
                messages=[{"role": "user", "content": "Test"}],
                max_tokens=5,
                timeout=10
            )
            
            if response and response.choices:
                return ApiKeyTestResponse(
                    valid=True,
                    message="API key is valid and working",
                    model_accessible="deepseek-chat"
                )
            else:
                return ApiKeyTestResponse(
                    valid=False,
                    message="No response from DeepSeek API"
                )
                
        except Exception as e:
            logger.warning(f"API key test failed: {e}")
            return ApiKeyTestResponse(
                valid=False,
                message=f"API key validation failed: {str(e)}"
            )
            
    except Exception as e:
        logger.error(f"Error testing API key: {e}")
        return ApiKeyTestResponse(
            valid=False,
            message=f"Error testing API key: {str(e)}"
        )

@app.post("/api/pdf-upload")
async def upload_pdf(
    file: UploadFile = File(...), 
    mistral_api_key: Optional[str] = Form(None),
    dataset_type: Optional[str] = Form('text_generation'),
    create_dataset: Optional[bool] = Form(False)
):
    """Handle PDF upload, process with OCR/PDF extraction, and optionally create dataset-ready format."""
    from pathlib import Path
    
    # Validate file type
    if not file.filename:
        raise HTTPException(status_code=400, detail="No file provided")
    
    file_ext = PathLib(file.filename).suffix.lower()
    if file_ext not in ['.pdf', '.png', '.jpg', '.jpeg', '.tiff', '.bmp']:
        raise HTTPException(status_code=400, detail="Unsupported file type. Please upload PDF or image files.")
    
    # Validate dataset type
    valid_dataset_types = ['text_generation', 'question_answer', 'classification', 'translation', 'summarization']
    if dataset_type not in valid_dataset_types:
        raise HTTPException(status_code=400, detail=f"Invalid dataset type. Must be one of: {valid_dataset_types}")
    
    # Save uploaded file to a temporary location
    try:
        with tempfile.NamedTemporaryFile(delete=False, suffix=file_ext) as tmp:
            shutil.copyfileobj(file.file, tmp)
            tmp_path = PathLib(tmp.name)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to save uploaded file: {e}")

    try:
        # Import DocumentUnderstanding here to avoid module loading issues
        try:
            from document_understanding import DocumentUnderstanding
        except ImportError:
            # Try relative import if absolute fails
            try:
                from .document_understanding import DocumentUnderstanding
            except ImportError:
                # Try importing from the src directory
                current_dir = os.path.dirname(os.path.abspath(__file__))
                src_dir = os.path.join(current_dir, 'src')
                sys.path.insert(0, src_dir)
                from document_understanding import DocumentUnderstanding
        
        # Process the file using DocumentUnderstanding
        doc_processor = DocumentUnderstanding(mistral_api_key=mistral_api_key)
        annotations = doc_processor.process_document(tmp_path)
        analysis = doc_processor.analyze_document_structure(annotations)
        combined_text = doc_processor.get_combined_text(annotations)
        export_json = doc_processor.export_annotations(annotations, format='json')
        
        # Add PDF content to RAG system
        pdf_rag_metadata = {
            "filename": file.filename,
            "upload_time": datetime.now().isoformat(),
            "file_type": file_ext,
            "extraction_method": "mistral_ai" if mistral_api_key else "tesseract_ocr"
        }
        
        rag_success = add_pdf_to_rag(combined_text, file.filename, pdf_rag_metadata)
        
        # Base response
        response_data = {
            "status": "success",
            "filename": file.filename,
            "annotations": export_json.get("annotations", []),
            "analysis": export_json.get("analysis", {}),
            "combined_text": combined_text,
            "message": "PDF processed successfully.",
            "rag_indexed": rag_success
        }
        
        # If dataset creation is requested, generate dataset formats
        if create_dataset and annotations:
            try:
                # Create dataset entries in the specified format
                dataset_entries = doc_processor.to_dataset_format(annotations, dataset_type)
                
                # Generate task ID for PDF-based dataset
                task_id = f"pdf_{dataset_type}_{int(time.time())}"
                
                # Convert to proper DatasetEntry format
                formatted_entries = []
                for i, entry in enumerate(dataset_entries, 1):
                    formatted_entry = {
                        "id": f"{task_id}-{i}",
                        "content": entry,
                        "metadata": {
                            "source": "pdf_document",
                            "filename": file.filename,
                            "dataset_type": dataset_type,
                            "created_at": datetime.now().isoformat(),
                            "extraction_method": "mistral_ai" if mistral_api_key else "tesseract_ocr"
                        }
                    }
                    formatted_entries.append(formatted_entry)
                
                # Cache the dataset
                cache_data = {
                    'entries': formatted_entries,
                    'quality_report': {
                        'generated_entries': len(formatted_entries),
                        'quality_score': 0.9,
                        'duplicates_removed': 0,
                        'average_length': sum(len(str(entry['content'])) for entry in formatted_entries) / len(formatted_entries) if formatted_entries else 0,
                        'details': {
                            'total_generated': len(formatted_entries),
                            'valid_entries': len(formatted_entries),
                            'source_type': 'pdf_document',
                            'dataset_type': dataset_type,
                            'extraction_confidence': sum(ann.confidence for ann in annotations) / len(annotations) if annotations else 0.8
                        }
                    },
                    'generated_at': datetime.now().isoformat(),
                    'task_id': task_id,
                    'count': len(formatted_entries)
                }
                
                # Save to cache
                save_dataset_to_cache_sync(task_id, cache_data)
                
                # Add dataset info to response
                response_data.update({
                    "dataset_created": True,
                    "dataset_type": dataset_type,
                    "task_id": task_id,
                    "dataset_entries": len(formatted_entries),
                    "dataset_preview": formatted_entries[:3],
                    "download_ready": True
                })
                
                logger.info(f"Created {dataset_type} dataset from PDF: {file.filename} with {len(formatted_entries)} entries")
                
            except Exception as dataset_error:
                logger.error(f"Error creating dataset from PDF: {dataset_error}")
                response_data.update({
                    "dataset_created": False,
                    "dataset_error": str(dataset_error)
                })
        else:
            response_data["dataset_created"] = False
        
        return response_data
        
    except Exception as e:
        logger.error(f"Error processing document: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to process document: {e}")
    finally:
        # Clean up temporary file
        try:
            tmp_path.unlink(missing_ok=True)
        except Exception as cleanup_error:
            logger.warning(f"Failed to cleanup temporary file: {cleanup_error}")

# RAG Management API
@app.get("/api/rag/status", summary="Get RAG System Status", tags=["RAG API"])
async def get_rag_status():
    """Get current status of RAG system."""
    try:
        doc_count = len(pdf_vector_store.documents)
        has_index = pdf_vector_store.vectorizer is not None
        
        return {
            "status": "active" if doc_count > 0 else "empty",
            "document_count": doc_count,
            "indexed": has_index,
            "documents": [
                {
                    "id": doc_id,
                    "filename": doc_data.get('metadata', {}).get('filename', 'Unknown'),
                    "chunks": doc_data.get('metadata', {}).get('total_chunks', 1),
                    "upload_time": doc_data.get('metadata', {}).get('upload_time', 'Unknown')
                }
                for doc_id, doc_data in pdf_vector_store.documents.items()
            ]
        }
    except Exception as e:
        logger.error(f"Error getting RAG status: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to get RAG status: {str(e)}")

@app.post("/api/rag/search", summary="Search RAG Documents", tags=["RAG API"])
async def search_rag_documents(query: str = Body(..., embed=True)):
    """Search for relevant documents in RAG system."""
    try:
        if not query or not query.strip():
            raise HTTPException(status_code=400, detail="Query cannot be empty")
            
        results = pdf_vector_store.search(query.strip(), top_k=5)
        
        formatted_results = []
        for doc_id, score, doc_data in results:
            formatted_results.append({
                "document_id": doc_id,
                "score": score,
                "filename": doc_data.get('metadata', {}).get('filename', 'Unknown'),
                "chunk_index": doc_data.get('metadata', {}).get('chunk_index', 0),
                "text_preview": doc_data['text'][:200] + "..." if len(doc_data['text']) > 200 else doc_data['text'],
                "metadata": doc_data.get('metadata', {})
            })
            
        return {
            "query": query,
            "results": formatted_results,
            "result_count": len(formatted_results)
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error searching RAG documents: {e}")
        raise HTTPException(status_code=500, detail=f"Search failed: {str(e)}")

@app.delete("/api/rag/clear", summary="Clear RAG Documents", tags=["RAG API"])
async def clear_rag_documents():
    """Clear all documents from RAG system."""
    try:
        global pdf_vector_store
        doc_count = len(pdf_vector_store.documents)
        pdf_vector_store = PDFVectorStore()  # Reset the vector store
        
        logger.info(f"Cleared {doc_count} documents from RAG system")
        return {
            "message": f"Successfully cleared {doc_count} documents from RAG system",
            "cleared_count": doc_count
        }
        
    except Exception as e:
        logger.error(f"Error clearing RAG documents: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to clear RAG documents: {str(e)}")

@app.get("/api/models/all", summary="Get All Available Models", tags=["System API"])
async def get_all_models():
    """Get all available models from both DeepSeek and Ollama."""
    try:
        models_data = {
            "deepseek": {
                "available": True,
                "models": [
                    {
                        "id": "deepseek-chat",
                        "name": "DeepSeek Chat",
                        "description": "Advanced reasoning model for high-quality conversations"
                    },
                    {
                        "id": "deepseek-reasoner", 
                        "name": "DeepSeek Reasoner",
                        "description": "Reasoning model with step-by-step thinking process"
                    }
                ]
            },
            "ollama": {
                "available": False,
                "models": []
            }
        }
        
        # Check if DeepSeek API is configured
        api_key = get_deepseek_api_key()
        models_data["deepseek"]["available"] = api_key is not None
        
        # Check Ollama availability and get models
        try:
            ollama_models = get_available_ollama_models()
            if ollama_models:
                models_data["ollama"]["available"] = True
                models_data["ollama"]["models"] = [
                    {
                        "id": f"ollama:{model}",
                        "name": model,
                        "description": f"Local Ollama model: {model}"
                    }
                    for model in ollama_models
                ]
            else:
                models_data["ollama"]["available"] = False
                
        except Exception as ollama_error:
            logger.warning(f"Error checking Ollama models: {ollama_error}")
            models_data["ollama"]["available"] = False
        
        return JSONResponse(content=models_data)
        
    except Exception as e:
        logger.error(f"Error getting models: {e}")
        # Return basic model info even if there's an error
        return JSONResponse(content={
            "deepseek": {
                "available": False,
                "models": [
                    {
                        "id": "deepseek-chat",
                        "name": "DeepSeek Chat", 
                        "description": "Advanced reasoning model (API key required)"
                    }
                ]
            },
            "ollama": {
                "available": False,
                "models": []
            }
        })


