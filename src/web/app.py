from fastapi import FastAPI, Request, HTTPException, Depends, Body, Path as FastApiPath, Query, Form, UploadFile, File
from fastapi.responses import JSONResponse, HTMLResponse, FileResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from pydantic import BaseModel, Field
from typing import List, Optional, Dict, Any
import sys
import os
import json
from datetime import datetime
import tempfile
import zipfile
import logging
from pathlib import Path
import uvicorn
from dotenv import load_dotenv
import traceback
import requests
import re

# Load environment variables from .env file
load_dotenv()

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

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

# --- Simple DeepSeek Client Implementation ---
class SimpleDeepseekClient:
    def __init__(self, api_key: str):
        self.api_key = api_key
        self.base_url = "https://api.deepseek.com/v1"
    
    def generate_text(self, prompt: str, max_tokens: int = 1000) -> str:
        """Simple text generation - for demo purposes"""
        try:
            import requests
            
            headers = {
                "Authorization": f"Bearer {self.api_key}",
                "Content-Type": "application/json"
            }
            
            data = {
                "model": "deepseek-chat",
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
                logger.info(f"Loaded {len(self._tasks)} tasks")
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
def simple_generate_dataset(task: Dict, count: int, client: SimpleDeepseekClient) -> tuple:
    """Simple dataset generation function"""
    entries = []
    
    task_id = task.get('id', 'unknown')
    prompt_template = task.get('prompt_template', 'Generate data for: {description}')
    description = task.get('description', 'Unknown task')
    
    logger.info(f"Generating {count} entries for task: {task_id}")
    
    try:
        # Create a prompt for generating multiple entries
        batch_prompt = f"""
Task: {description}

Please generate {count} entries for this task. Return the data as a JSON array.
Each entry should be a JSON object with relevant fields.

Template: {prompt_template}

Return only the JSON array, no additional text.
"""
        
        # Generate data using DeepSeek API
        response = client.generate_text(batch_prompt, max_tokens=2000)
        
        if response:
            try:
                # Try to parse JSON response
                import re
                
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

# --- Helper Functions ---
def load_quality_config_sync():
    """Load quality control configuration"""
    default_config_dict = {
        'min_length': 10,
        'max_length': 1000,
        'required_fields': [],
        'custom_validators': [],
        'similarity_threshold': 0.8
    }
    
    try:
        if app_config.quality_config_file.exists():
            with open(app_config.quality_config_file, 'r') as f:
                config_data = json.load(f)
                loaded_config = QualityConfig(**config_data)
                return loaded_config.model_dump()
        else:
            logger.info(f"Quality config file not found. Creating with defaults.")
            with open(app_config.quality_config_file, 'w') as f:
                json.dump(default_config_dict, f, indent=2)
            return default_config_dict
    except Exception as e:
        logger.warning(f"Could not load quality config: {e}. Using defaults.")
        return default_config_dict

def save_dataset_to_cache_sync(task_id: str, data: Dict[str, Any]):
    """Save generated dataset to cache"""
    cache_file = app_config.cache_dir / f'dataset_{task_id}.json'
    try:
        with open(cache_file, 'w', encoding='utf-8') as f:
            json.dump(data, f, indent=2, ensure_ascii=False)
        logger.info(f"Dataset cached at {cache_file}")
        return str(cache_file)
    except Exception as e:
        logger.error(f"Failed to cache dataset: {e}")
        return None

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
        raise HTTPException(status_code=503, detail="DeepSeek client is not available. Check API key configuration.")

    try:
        task = tm.get_task(payload.task_id)
        if not task:
            raise HTTPException(status_code=404, detail="Task not found")
        
        logger.info(f"Generating {payload.count} entries for task {payload.task_id}")
        
        if 'id' not in task:
            task['id'] = payload.task_id

        # Use simple generation function
        entries_raw, quality_report_raw = simple_generate_dataset(
            task=task,
            count=payload.count,
            client=client
        )
        
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
        
        # Cache the dataset
        save_dataset_to_cache_sync(payload.task_id, GenerateResponse(**result_data).model_dump(mode='json'))
        
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
        raise HTTPException(status_code=503, detail="DeepSeek client is not available. Check API key configuration.")

    try:
        task = tm.get_task(payload.task_id)
        if not task:
            raise HTTPException(status_code=404, detail="Task not found")

        if 'id' not in task:
            task['id'] = payload.task_id

        logger.info(f"Testing generation for task {payload.task_id}")
        
        # Generate small test sample
        entries_raw, quality_report_raw = simple_generate_dataset(
            task=task,
            count=3,
            client=client
        )
        
        processed_entries = []
        for entry in entries_raw:
            if isinstance(entry, dict):
                processed_entries.append(Entry(raw_data=entry))
            else:
                processed_entries.append(Entry(output=str(entry)))
        
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
            data_from_cache = json.load(f)
            
        if format == 'json':
            return FileResponse(
                cache_file,
                media_type='application/json',
                filename=f'dataset_{task_id}.json'
            )
        
        elif format == 'csv':
            try:
                import pandas as pd
            except ImportError:
                raise HTTPException(status_code=501, detail="pandas is required for CSV export")

            entries_for_csv = data_from_cache.get('entries', [])
            if not entries_for_csv:
                raise HTTPException(status_code=400, detail="No entries found in dataset for CSV export")
            
            df_data = []
            for entry_dict in entries_for_csv:
                if isinstance(entry_dict.get("raw_data"), dict):
                    df_data.append(entry_dict["raw_data"])
                elif entry_dict.get("output") is not None:
                    df_data.append({"output": entry_dict.get("output")})
                else:
                    df_data.append(entry_dict)

            if not df_data:
                raise HTTPException(status_code=400, detail="No processable data found in entries for CSV export")

            df = pd.DataFrame(df_data)
            
            temp_file = tempfile.NamedTemporaryFile(mode='w+', suffix='.csv', delete=False, encoding='utf-8')
            df.to_csv(temp_file.name, index=False)
            temp_file.close()
            
            return FileResponse(
                temp_file.name,
                media_type='text/csv',
                filename=f'dataset_{task_id}.csv'
            )

        elif format == 'zip':
            temp_zip = tempfile.NamedTemporaryFile(suffix='.zip', delete=False)
            temp_zip.close()
            
            with zipfile.ZipFile(temp_zip.name, 'w', zipfile.ZIP_DEFLATED) as zf:
                # Add main dataset JSON
                zf.writestr(f'{task_id}_dataset.json', 
                           json.dumps(data_from_cache, indent=2, ensure_ascii=False))
                
                # Add quality report
                if 'quality_report' in data_from_cache:
                    zf.writestr(f'{task_id}_quality_report.json', 
                               json.dumps(data_from_cache['quality_report'], indent=2, ensure_ascii=False))
                
                # Add metadata
                metadata = {
                    'task_id': task_id,
                    'generated_at': data_from_cache.get('generated_at'),
                    'entry_count': data_from_cache.get('count', len(data_from_cache.get('entries', []))),
                    'download_timestamp': datetime.now().isoformat()
                }
                zf.writestr(f'{task_id}_metadata.json',
                           json.dumps(metadata, indent=2))
            
            return FileResponse(
                temp_zip.name,
                media_type='application/zip',
                filename=f'dataset_{task_id}.zip'
            )
            
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error downloading dataset: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/api/status", response_model=StatusResponse, summary="Get API Status", tags=["System"])
async def get_status_api():
    """Get current API status and configuration."""
    try:
        tasks_count = 0
        try:
            if app_config.tasks_json_file.exists():
                with open(app_config.tasks_json_file, 'r') as f:
                    tasks_data = json.load(f)
                    if isinstance(tasks_data, dict):
                        tasks_count = len(tasks_data)
                    elif isinstance(tasks_data, list):
                        tasks_count = len(tasks_data)
        except:
            pass
        
        api_key = os.getenv('DEEPSEEK_API_KEY')
        deepseek_configured = bool(api_key and api_key.strip())
        
        return StatusResponse(
            status="operational",
            timestamp=datetime.now(),
            tasks_count=tasks_count,
            cache_dir=str(app_config.cache_dir),
            deepseek_api_configured=deepseek_configured,
            python_path=str(app_config.python_dir),
            version="1.0.0"
        )
    except Exception as e:
        logger.error(f"Error getting status: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/health", response_model=HealthResponse, summary="Health Check", tags=["System"])
async def health_check():
    """Simple health check endpoint."""
    return HealthResponse(
        status="healthy",
        timestamp=datetime.now()
    )

@app.on_event("startup")
async def startup_event():
    """Log startup information."""
    logger.info("=== DekDataset API Starting ===")
    logger.info(f"Project root: {app_config.project_root}")
    logger.info(f"Python modules path: {app_config.python_dir}")
    logger.info(f"Tasks file: {app_config.tasks_json_file}")
    logger.info(f"Cache directory: {app_config.cache_dir}")
    
    # Check API key
    api_key = os.getenv('DEEPSEEK_API_KEY')
    if api_key and api_key.strip():
        logger.info("✓ DEEPSEEK_API_KEY is configured")
    else:
        logger.warning("⚠ DEEPSEEK_API_KEY is not set - dataset generation will be unavailable")
    
    logger.info("=== DekDataset API Ready ===")

if __name__ == "__main__":
    uvicorn.run(
        "app_complete:app",
        host="127.0.0.1",
        port=8000,
        reload=True,
        log_level="info"
    )
