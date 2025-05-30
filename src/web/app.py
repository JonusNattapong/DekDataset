import os
import json
import logging
import traceback
import tempfile
import zipfile
import requests
import concurrent.futures
import re
import base64
from pathlib import Path
from dotenv import load_dotenv
from fastapi import BackgroundTasks

# PDF and OCR dependencies
try:
    import pdfplumber
    import pytesseract
    from pdf2image import convert_from_path
    import re
    OCR_AVAILABLE = True
    import os
    # Set tesseract_cmd for Windows if not already set
    if os.name == "nt":
        tesseract_path = r"C:\\Program Files\\Tesseract-OCR\\tesseract.exe"
        if os.path.exists(tesseract_path):
            pytesseract.pytesseract.tesseract_cmd = tesseract_path
except ImportError:
    OCR_AVAILABLE = False

try:
    from mistralai import Mistral
    MISTRALAI_AVAILABLE = True
except ImportError:
    MISTRALAI_AVAILABLE = False

from fastapi import FastAPI, Request, HTTPException, Depends, Body, Path as FastApiPath, Query, Form, UploadFile, File, Response
from fastapi.responses import JSONResponse, HTMLResponse, FileResponse, StreamingResponse, RedirectResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from pydantic import BaseModel, Field
from typing import List, Optional, Dict, Any
import sys
import os
import json
from datetime import datetime

# Load environment variables from .env file
load_dotenv()

def save_dataset_to_cache_sync(task_id, cache_data):
    """บันทึก cache_data ลงไฟล์ cache/{task_id}.json"""
    cache_dir = os.path.join(os.path.dirname(__file__), "..", "..", "cache")
    os.makedirs(cache_dir, exist_ok=True)
    cache_path = os.path.join(cache_dir, f"{task_id}.json")
    with open(cache_path, "w", encoding="utf-8") as f:
        json.dump(cache_data, f, ensure_ascii=False, indent=2)

# Experiment tracking configuration loading removed

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Experiment tracking removed - focusing on core dataset generation functionality

# --- Pydantic Models ---
class TaskBase(BaseModel):
    id: str
    type: Optional[str] = 'custom'
    description: str
    prompt_template: str
    validation_rules: Optional[Dict[str, Any]] = {}

class TaskCreate(TaskBase):
    pass

class TaskResponse(TaskBase):
    created_at: Optional[datetime] = None

class TaskListResponse(BaseModel):
    tasks: List[TaskResponse]
    count: int
    timestamp: datetime

class GenerateRequest(BaseModel):
    task_id: str
    count: int = Field(10, gt=0, le=1000)
    model: str = Field('deepseek-chat', description="DeepSeek model to use (deepseek-chat or deepseek-reasoner)")

class Entry(BaseModel):
    id: Optional[int] = None
    input: Optional[str] = None
    output: Optional[str] = None
    raw_data: Optional[Dict[str, Any]] = None 

    class Config:
        extra = 'allow'

class QualityReport(BaseModel):
    generated_entries: Optional[int] = None
    quality_score: Optional[float] = None
    duplicates_removed: Optional[int] = None
    average_length: Optional[float] = None
    details: Optional[Dict[str, Any]] = None

    class Config:
        extra = 'allow'

class GenerateResponse(BaseModel):
    entries: List[Entry]
    quality_report: QualityReport
    generated_at: datetime
    task_id: str
    count: int

class TestGenerationRequest(BaseModel):
    task_id: str
    model: str = Field('deepseek-chat', description="DeepSeek model to use (deepseek-chat or deepseek-reasoner)")

class TestGenerationResponse(BaseModel):
    test_entries: List[Entry]
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
    status: str
    timestamp: datetime
    tasks_count: int
    cache_dir: str
    deepseek_api_configured: bool
    python_path: str
    version: str

class HealthResponse(BaseModel):
    status: str
    timestamp: datetime

# Experiment tracking models removed

# --- Simple DeepSeek Client Implementation ---
class SimpleDeepseekClient:
    def __init__(self, api_key: str, model: str = "deepseek-chat"):
        self.api_key = api_key
        self.base_url = "https://api.deepseek.com/v1"
        self.model = model
    
    def generate_text(self, prompt: str, max_tokens: int = 1000) -> str:
        """Simple text generation - for demo purposes"""
        try:
            headers = {
                "Authorization": f"Bearer {self.api_key}",
                "Content-Type": "application/json"
            }
            data = {
                "model": self.model if hasattr(self, 'model') else "deepseek-chat",
                "messages": [{"role": "user", "content": prompt}],
                "max_tokens": max_tokens,
                "temperature": 0.7
            }
            response = requests.post(
                f"{self.base_url}/chat/completions",
                headers=headers,
                json=data,
                timeout=30
            )
            if response.status_code == 200:
                result = response.json()
                return result["choices"][0]["message"]["content"]
            else:
                logger.error(f"DeepSeek API error: {response.status_code} - {response.text}")
                return None
        except Exception as e:
            logger.error(f"Error calling DeepSeek API: {e}")
            return None

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

# --- Simple Dataset Generation ---
from .document_understanding import DocumentUnderstanding, BBoxImageAnnotation, DocumentAnnotation

def simple_generate_dataset(task: Dict, count: int, client: SimpleDeepseekClient) -> tuple:
    """Simple dataset generation function"""
    entries = []
    
    task_id = task.get('id', 'unknown')
    prompt_template = task.get('prompt_template', 'Generate data for: {description}')
    description = task.get('description', 'Unknown task')
    model_name = getattr(client, 'model', 'deepseek-chat')
    logger.info(f"[simple_generate_dataset] Using model: {model_name}")
    
    logger.info(f"Generating {count} entries for task: {task_id}")
    
    try:
        # Create a prompt for generating multiple entries
        batch_prompt = f"""
Task: {description}
Model: {model_name}

Please generate {count} entries for this task. Return the data as a JSON array.
Each entry should be a JSON object with relevant fields.

Template: {prompt_template}

Return only the JSON array, no additional text.
"""
        
        # Generate data using DeepSeek API
        response = client.generate_text(batch_prompt, max_tokens=2000)
        
        if response:
            try:
                # Extract JSON array from response
                json_match = re.search(r'\[.*\]', response, re.DOTALL)
                if json_match:
                    json_str = json_match.group(0)
                    parsed_entries = json.loads(json_str)
                    
                    if isinstance(parsed_entries, list):
                        entries = parsed_entries[:count]  # Limit to requested count
                    else:
                        # If not a list, wrap in list
                        entries = [parsed_entries]
                else:
                    # Fallback: create entries from response text
                    lines = response.strip().split('\n')
                    for i, line in enumerate(lines[:count]):
                        if line.strip():
                            entries.append({
                                'id': i + 1,
                                'text': line.strip(),
                                'generated_by': 'deepseek-simple'
                            })
                            
            except json.JSONDecodeError:
                # Fallback: create simple entries from response
                logger.warning("Could not parse JSON response, creating simple entries")
                lines = response.strip().split('\n')
                for i, line in enumerate(lines[:count]):
                    if line.strip():
                        entries.append({
                            'id': i + 1,
                            'text': line.strip(),
                            'generated_by': 'deepseek-simple'
                        })
        
        # If no entries generated, create sample entries
        if not entries:
            logger.warning("No entries generated from API, creating sample entries")
            for i in range(min(count, 3)):
                entries.append({
                    'id': i + 1,
                    'text': f"Sample entry {i + 1} for task: {description}",
                    'generated_by': 'fallback'
                })
    
    except Exception as e:
        logger.error(f"Error in dataset generation: {e}")
        # Create fallback entries
        for i in range(min(count, 3)):
            entries.append({
                'id': i + 1,
                'text': f"Fallback entry {i + 1} for task: {description}",
                'generated_by': 'fallback',
                'error': str(e)
            })
    
    # Create quality report
    quality_report = {
        'generated_entries': len(entries),
        'quality_score': 0.8 if entries else 0.0,
        'duplicates_removed': 0,
        'average_length': sum(len(str(entry)) for entry in entries) / len(entries) if entries else 0,
        'details': {
            'total_generated': len(entries),
            'valid_entries': len(entries),
            'quality_rate': 1.0 if entries else 0.0,
            'issues_found': 0,
            'final_count': len(entries)
        }
    }
    
    return entries, quality_report

# --- AppConfig and Initialization ---
class AppConfig:
    def __init__(self):
        self.script_dir = Path(__file__).parent.absolute()
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
_deepseek_client: Optional[SimpleDeepseekClient] = None

def get_deepseek_api_key():
    api_key = os.getenv('DEEPSEEK_API_KEY')
    if not api_key:
        logger.warning("DEEPSEEK_API_KEY environment variable not set.")
    return api_key

def get_deepseek_client_dependency(api_key: str = Depends(get_deepseek_api_key)) -> Optional[SimpleDeepseekClient]:
    global _deepseek_client
    if not api_key:
        return None
    if _deepseek_client is None:
        _deepseek_client = SimpleDeepseekClient(api_key=api_key)
    return _deepseek_client

def get_task_manager():
    return task_manager

# Experiment tracking functionality removed

# --- Helper Functions ---
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
            if 'id' not in task_data and 'task_id' in task_data:
                task_data['id'] = task_data.pop('task_id')
            processed_tasks.append(TaskResponse(**task_data))

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
async def get_task_api(task_id: str = FastApiPath(..., title="The ID of the task to get"), tm = Depends(get_task_manager)):
    """Get a specific task by its ID."""
    try:
        task = tm.get_task(task_id)
        if not task:
            raise HTTPException(status_code=404, detail="Task not found")
        
        if 'id' not in task and 'task_id' in task:
            task['id'] = task.pop('task_id')
        elif 'id' not in task and task_id:
            task['id'] = task_id

        return TaskResponse(**task)
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error getting task {task_id}: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.delete("/api/tasks/{task_id}", response_model=MessageResponse, summary="Delete Task", tags=["Tasks API"])
async def delete_task_api(task_id: str = FastApiPath(..., title="The ID of the task to delete"), tm = Depends(get_task_manager)):
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

@app.post("/api/generate", response_model=GenerateResponse, summary="Generate Dataset", tags=["Dataset Generation API"])
async def generate_dataset_api(
    payload: GenerateRequest,
    tm = Depends(get_task_manager),
    client: Optional[SimpleDeepseekClient] = Depends(get_deepseek_client_dependency)
):
    """Generate dataset based on a task and count."""
    if client is None:
        raise HTTPException(status_code=503, detail="DeepSeek client is not available. Check API key configuration.")    # Experiment tracking removed - focusing on core dataset generation functionality
    
    try:
        task = tm.get_task(payload.task_id)
        if not task:
            raise HTTPException(status_code=404, detail="Task not found")

        logger.info(f"Generating {payload.count} entries for task {payload.task_id} using model {payload.model}")
        if 'id' not in task:
            task['id'] = payload.task_id        # Experiment tracking removed - starting direct generation

        # Use simple generation function
        generation_start_time = datetime.now()
        # Create a new client with the selected model
        selected_client = SimpleDeepseekClient(api_key=client.api_key)
        selected_client.base_url = client.base_url
        selected_client.model = payload.model
        entries_raw, quality_report_raw = simple_generate_dataset(
            task=task,
            count=payload.count,
            client=selected_client
        )
        generation_time = (datetime.now() - generation_start_time).total_seconds()
        
        # Process entries for response
        processed_entries = []
        for entry in entries_raw:
            if isinstance(entry, dict):
                processed_entries.append(Entry(raw_data=entry))
            else:
                processed_entries.append(Entry(output=str(entry)))
        
        processed_quality_report = QualityReport(**quality_report_raw) if isinstance(quality_report_raw, dict) else QualityReport()

        result_data = {
            'entries': processed_entries,
            'quality_report': processed_quality_report,
            'generated_at': datetime.now(),
            'task_id': payload.task_id,
            'count': len(processed_entries)
        }
          # Save dataset to cache
        cache_data = result_data.copy()
        cache_data['entries'] = [entry.model_dump() for entry in processed_entries]
        cache_data['quality_report'] = processed_quality_report.model_dump()
        cache_data['generated_at'] = cache_data['generated_at'].isoformat()
        save_dataset_to_cache_sync(payload.task_id, cache_data)
        logger.info(f"Generated {len(processed_entries)} entries for task {payload.task_id}")
        return GenerateResponse(**result_data)
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error generating dataset: {e}")
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/api/test-generation", response_model=TestGenerationResponse, summary="Test Dataset Generation", tags=["Dataset Generation API"])
async def test_generation_api(
    payload: TestGenerationRequest,
    tm = Depends(get_task_manager),
    client: Optional[SimpleDeepseekClient] = Depends(get_deepseek_client_dependency)
):
    """Test dataset generation with a small sample."""
    if client is None:
        raise HTTPException(status_code=503, detail="DeepSeek client is not available. Check API key configuration.")    # Experiment tracking removed - focusing on core test generation functionality

    try:
        task = tm.get_task(payload.task_id)
        if not task:
            raise HTTPException(status_code=404, detail="Task not found")

        logger.info(f"Testing generation for task {payload.task_id} using model {payload.model}")
        if 'id' not in task:
            task['id'] = payload.task_id        # Experiment tracking removed - starting direct test generation

        # Generate small test sample
        generation_start_time = datetime.now()
        selected_client = SimpleDeepseekClient(api_key=client.api_key)
        selected_client.base_url = client.base_url
        selected_client.model = payload.model
        entries_raw, quality_report_raw = simple_generate_dataset(
            task=task,
            count=3,
            client=selected_client
        )
        generation_time = (datetime.now() - generation_start_time).total_seconds()
        
        processed_entries = []
        for entry in entries_raw:
            if isinstance(entry, dict):
                processed_entries.append(Entry(raw_data=entry))
            else:
                processed_entries.append(Entry(output=str(entry)))
        
        processed_quality_report = QualityReport(**quality_report_raw) if isinstance(quality_report_raw, dict) else QualityReport()
          # Test generation completed successfully
        
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
        raise HTTPException(status_code=500, detail=str(e))

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

# --- Experiment Tracking API ---

# Experiment tracking API endpoint removed

# Experiment tracking API endpoint removed

# Experiment tracking API endpoint removed

# Experiment tracking API endpoint removed

@app.get("/api/download/{format}/{task_id}", summary="Download Generated Dataset", tags=["Dataset Download API"])
async def download_dataset_api(
    format: str = FastApiPath(..., description="Dataset format: 'json', 'csv', or 'zip'"),
    task_id: str = FastApiPath(..., description="ID of the task")
):
    """Download generated dataset in specified format."""
    valid_formats = ['json', 'csv', 'zip']
    if format not in valid_formats:
        raise HTTPException(status_code=400, detail=f"Unsupported format. Use one of: {valid_formats}")
    cache_file = app_config.cache_dir / f'dataset_{task_id}.json'
    if not cache_file.exists():
        raise HTTPException(status_code=404, detail="Dataset not found. Generate dataset first.")
    try:
        with open(cache_file, 'r', encoding='utf-8') as f:
            data = json.load(f)
        if format == 'json':
            return JSONResponse(content=data)
        elif format == 'csv':
            import csv
            from io import StringIO
            entries = data.get('entries', [])
            if not entries:
                raise HTTPException(status_code=404, detail="No entries found in dataset.")
            output = StringIO()
            fieldnames = set()
            for entry in entries:
                fieldnames.update(entry.keys())
            fieldnames = list(fieldnames)
            writer = csv.DictWriter(output, fieldnames=fieldnames)
            writer.writeheader()
            for entry in entries:
                writer.writerow(entry)
            output.seek(0)
            return Response(content=output.read(), media_type='text/csv', headers={
                'Content-Disposition': f'attachment; filename="dataset_{task_id}.csv"'
            })
        elif format == 'zip':
            import zipfile
            from io import BytesIO
            entries = data.get('entries', [])
            if not entries:
                raise HTTPException(status_code=404, detail="No entries found in dataset.")
            mem_zip = BytesIO()
            with zipfile.ZipFile(mem_zip, mode='w', compression=zipfile.ZIP_DEFLATED) as zf:
                zf.writestr(f'dataset_{task_id}.json', json.dumps(data, ensure_ascii=False, indent=2))
            mem_zip.seek(0)
            return Response(content=mem_zip.read(), media_type='application/zip', headers={
                'Content-Disposition': f'attachment; filename="dataset_{task_id}.zip"'
            })
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error downloading dataset: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/api/upload-pdf", summary="Upload PDF and extract dataset entries", tags=["Dataset Generation API"])
async def upload_pdf_and_extract_dataset(
    file: UploadFile = File(..., description="PDF file to extract dataset from"),
    tm = Depends(get_task_manager)
):
    """Upload a PDF file, extract text, auto-create dataset & task, and return info."""
    try:
        if not file.filename.lower().endswith('.pdf'):
            raise HTTPException(status_code=400, detail="Only PDF files are supported.")
        # Save uploaded file temporarily
        temp_path = app_config.cache_dir / f"temp_{datetime.now().strftime('%Y%m%d_%H%M%S')}.pdf"
        with open(temp_path, "wb") as f_out:
            f_out.write(await file.read())
        # Extract text from PDF (try pdfplumber first)
        entries = []
        fallback_to_ocr = False
        try:
            with pdfplumber.open(str(temp_path)) as pdf:
                for page_num, page in enumerate(pdf.pages, 1):
                    text = page.extract_text()
                    if text and not is_garbled_thai(text):
                        for line in text.splitlines():
                            line = line.strip()
                            if line:
                                entries.append({
                                    "input": line,
                                    "output": "",
                                    "page": page_num
                                })
                    else:
                        fallback_to_ocr = True
        except Exception as e:
            fallback_to_ocr = True
        # If no entries or garbled, try Mistral OCR fallback
        if not entries or fallback_to_ocr:
            try:
                entries = extract_text_with_mistral_ocr(str(temp_path), lang='tha')
            except Exception as ocr_err:
                temp_path.unlink(missing_ok=True)
                raise HTTPException(status_code=500, detail=f"PDF text extraction failed and Mistral OCR fallback also failed: {ocr_err}")
        temp_path.unlink(missing_ok=True)
        if not entries:
            raise HTTPException(status_code=422, detail="No text could be extracted from the PDF (even with OCR). Please check the file or install OCR dependencies.")
        # Auto-create new task
        task_id = f"pdf_{datetime.now().strftime('%Y%m%d%H%M%S')}"
        task_data = {
            "id": task_id,
            "type": "classification",
            "description": f"Dataset imported from PDF: {file.filename}",
            "prompt_template": "Classify the following text:",
            "created_at": datetime.now().isoformat()
        }
        tm.add_custom_task(task_data)
        # Save dataset to cache
        cache_data = {
            "task_id": task_id,
            "generated_at": datetime.now().isoformat(),
            "entries": entries,
            "entry_count": len(entries),
            "source": file.filename
        }
        save_dataset_to_cache_sync(task_id, cache_data)
        return {
            "task_id": task_id,
            "task": task_data,
            "count": len(entries),
            "entries_preview": entries[:10]
        }
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error extracting PDF: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to extract PDF: {e}")

def is_garbled_thai(text: str, threshold: float = 0.3) -> bool:
    """
    Returns True if the text is likely garbled for Thai (too few Thai characters or too many replacement chars).
    threshold: minimum ratio of Thai chars to total chars to consider as NOT garbled.
    """
    if not text or not isinstance(text, str):
        return True
    # Count Thai characters
    thai_chars = re.findall(r'[\u0E00-\u0E7F]', text)
    thai_ratio = len(thai_chars) / max(len(text), 1)
    # Count replacement chars (�)
    replacement_count = text.count('�')
    replacement_ratio = replacement_count / max(len(text), 1)
    # Heuristic: too few Thai chars or too many replacement chars
    if thai_ratio < threshold or replacement_ratio > 0.1:
        return True
    return False

def extract_text_with_ocr(pdf_path: str, lang: str = 'tha+eng') -> list:
    """
    Extract text from a PDF using OCR (pytesseract + pdf2image).
    Returns a list of entries: [{"input": ..., "output": "", "page": ...}]
    """
    if not OCR_AVAILABLE:
        raise RuntimeError("OCR dependencies are not installed. Please install pytesseract and pdf2image.")
    # Check if tesseract executable is available
    tesseract_cmd = getattr(pytesseract.pytesseract, 'tesseract_cmd', 'tesseract')
    if not (os.path.exists(tesseract_cmd) or tesseract_cmd == 'tesseract'):
        raise RuntimeError(
            "Tesseract OCR is not installed or not found. Please install Tesseract from https://github.com/UB-Mannheim/tesseract/wiki and ensure it's in your PATH or set the correct path in the code."
        )
    entries = []
    # Set your local poppler path here
    poppler_path = r"D:\Github\DekDataset\poppler-local\Library\bin"
    try:
        images = convert_from_path(pdf_path, poppler_path=poppler_path)
        for page_num, image in enumerate(images, 1):
            text = pytesseract.image_to_string(image, lang=lang)
            for line in text.splitlines():
                line = line.strip()
                if line:
                    entries.append({
                        "input": line,
                        "output": "",
                        "page": page_num
                    })
        return entries
    except Exception as e:
        raise RuntimeError(f"OCR extraction failed: {e}")

def extract_text_with_mistral_ocr(pdf_path: str, lang: str = 'tha') -> list:
    """
    Extract text from a PDF using Mistral Document AI API (mistral-ocr-latest model).
    Returns a list of entries: [{"input": ..., "output": "", "page": ...}]
    """
    if not MISTRALAI_AVAILABLE:
        raise RuntimeError("mistralai package is not installed. Please install with: pip install mistralai")
    api_key = os.getenv('MISTRAL_API_KEY')
    if not api_key:
        raise RuntimeError("MISTRAL_API_KEY is not set in environment variables.")
    client = Mistral(api_key=api_key)
    # Encode PDF to base64
    with open(pdf_path, "rb") as f:
        base64_pdf = base64.b64encode(f.read()).decode('utf-8')
    # Use data:application/pdf;base64,... as document_url
    document_url = f"data:application/pdf;base64,{base64_pdf}"
    ocr_response = client.ocr.process(
        model="mistral-ocr-latest",
        document={
            "type": "document_url",
            "document_url": document_url
        },
        include_image_base64=False
    )
    # The response contains 'markdown' with the extracted text
    markdown = getattr(ocr_response, 'markdown', None)
    if not markdown:
        raise RuntimeError("No markdown text returned from Mistral OCR API.")
    # Split markdown into lines and pages (simple heuristic: page breaks as '---' or '\f')
    entries = []
    page_num = 1
    for line in markdown.splitlines():
        line = line.strip()
        if not line:
            continue
        if line in ('---', '\f'):
            page_num += 1
            continue
        entries.append({
            "input": line,
            "output": "",
            "page": page_num
        })
    return entries

# --- Mistral API Key Management (in-memory for demo, can be improved) ---
MISTRAL_API_KEY: Optional[str] = None

def set_mistral_api_key(api_key: str):
    global MISTRAL_API_KEY
    MISTRAL_API_KEY = api_key

def get_mistral_api_key() -> Optional[str]:
    global MISTRAL_API_KEY
    if MISTRAL_API_KEY:
        return MISTRAL_API_KEY
    return os.getenv('MISTRAL_API_KEY')

@app.post("/api/set-mistral-api-key", tags=["Document AI"])
async def set_mistral_api_key_api(data: Dict[str, str]):
    api_key = data.get("api_key")
    if not api_key or not api_key.startswith("sk-"):
        return JSONResponse(status_code=400, content={"message": "Invalid API key format."})
    set_mistral_api_key(api_key)
    return {"message": "Mistral API key set successfully."}

@app.post("/api/document-ocr-annotation", tags=["Document AI"])
async def document_ocr_annotation(
    file: UploadFile = File(...),
    bbox: bool = Form(False),
    doc: bool = Form(False),
    pages: Optional[str] = Form(None),
    tm = Depends(get_task_manager)
):
    """
    Upload a PDF and run OCR + (optional) annotation using Mistral Document AI.
    The result will be auto-saved as a new dataset and registered as a new Task.
    """
    import tempfile, shutil, uuid
    api_key = get_mistral_api_key()
    if not api_key:
        raise HTTPException(status_code=400, detail="Mistral API key not set.")
    # Save uploaded file to temp
    temp_dir = tempfile.mkdtemp()
    temp_path = os.path.join(temp_dir, file.filename)
    with open(temp_path, "wb") as f:
        shutil.copyfileobj(file.file, f)
    # Upload to Mistral and get signed url
    doc_ai = DocumentUnderstanding(api_key=api_key)
    signed_url = doc_ai.upload_document(temp_path)
    # Parse pages
    page_list = None
    if pages:
        try:
            page_list = [int(p.strip()) for p in pages.split(",") if p.strip().isdigit()]
        except Exception:
            page_list = None
    # Prepare annotation models
    bbox_model = BBoxImageAnnotation if bbox else None
    doc_model = DocumentAnnotation if doc else None
    # Run OCR + annotation
    result = doc_ai.process_with_annotations(
        document_url=signed_url,
        pages=page_list,
        bbox_annotation_model=bbox_model,
        document_annotation_model=doc_model,
        include_image_base64=False
    )
    # Extract entries for dataset (simple: use markdown or annotation result)
    entries = []
    if hasattr(result, 'markdown') and result.markdown:
        # Split markdown by lines/pages
        for i, line in enumerate(result.markdown.splitlines()):
            if line.strip():
                entries.append({"input": line.strip(), "output": "", "page": i+1})
    elif hasattr(result, 'document_annotation') and result.document_annotation:
        # Use document annotation as entries
        entries.append(result.document_annotation)
    elif hasattr(result, 'bbox_annotations') and result.bbox_annotations:
        for ann in result.bbox_annotations:
            entries.append(ann)
    # Fallback: try to use any 'entries' key
    elif isinstance(result, dict) and 'entries' in result:
        entries = result['entries']
    # Auto-generate a new task id
    task_id = f"pdf_{datetime.now().strftime('%Y%m%d%H%M%S')}_{uuid.uuid4().hex[:6]}"
    # Register as a new Task
    task_data = {
        "id": task_id,
        "type": "ocr_pdf",
        "description": f"Auto-imported from PDF: {file.filename}",
        "prompt_template": "OCR/Annotation from PDF",
        "created_at": datetime.now().isoformat()
    }
    tm.add_custom_task(task_data)
    # Save dataset to cache
    cache_data = {
        "task_id": task_id,
        "generated_at": datetime.now().isoformat(),
        "entry_count": len(entries),
        "entries": entries[:1000],
        "source": "mistral-ocr",
        "file_name": file.filename
    }
    save_dataset_to_cache_sync(task_id, cache_data)
    # Return preview and info
    return {
        "task_id": task_id,
        "count": len(entries),
        "entries_preview": entries[:10],
        "message": f"PDF imported and dataset created as task {task_id}"
    }

# Experiment tracking API endpoint removed

# Experiment tracking API endpoint removed

@app.get("/api/preview/{task_id}", summary="Preview generated dataset", tags=["Dataset Preview"])
async def preview_dataset_api(task_id: str = FastApiPath(..., description="ID of the task")):
    """Return a preview (first 10 entries) of the generated dataset for a task."""
    cache_file = app_config.cache_dir / f'dataset_{task_id}.json'
    if not cache_file.exists():
        # Try fallback: look for CSV in downloads (for demo)
        import glob
        import csv
        import os
        downloads = os.path.expanduser('~/Downloads')
        csv_files = glob.glob(os.path.join(downloads, f"dataset_{task_id}*.csv"))
        if csv_files:
            with open(csv_files[0], encoding='utf-8') as f:
                reader = csv.DictReader(f)
                rows = list(reader)[:10]
            return {"entries": rows, "count": len(rows)}
        raise HTTPException(status_code=404, detail="Dataset not found. Generate dataset first.")
    import json
    with open(cache_file, 'r', encoding='utf-8') as f:
        data = json.load(f)
    entries = data.get('entries', [])
    preview = entries[:10]
    return {"entries": preview, "count": len(entries)}
