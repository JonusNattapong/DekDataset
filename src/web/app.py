import os
import json

from fastapi import FastAPI, Request, HTTPException, Depends, Body, Path as FastApiPath, Query, Form, UploadFile, File
from fastapi.responses import JSONResponse, HTMLResponse, FileResponse, StreamingResponse
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
from io import StringIO
import pandas as pd
import csv

# Load environment variables from .env file
load_dotenv()

def save_dataset_to_cache_sync(task_id, cache_data):
    """บันทึก cache_data ลงไฟล์ cache/{task_id}.json"""
    cache_dir = os.path.join(os.path.dirname(__file__), "..", "..", "cache")
    os.makedirs(cache_dir, exist_ok=True)
    cache_path = os.path.join(cache_dir, f"{task_id}.json")
    with open(cache_path, "w", encoding="utf-8") as f:
        json.dump(cache_data, f, ensure_ascii=False, indent=2)

# Load experiment tracking configuration
if os.path.exists('.env.tracking'):
    load_dotenv('.env.tracking')

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Try to import experiment tracking (optional)
try:
    # Add the python directory to path for experiment tracking module
    import sys
    from pathlib import Path
    current_dir = Path(__file__).parent
    python_dir = current_dir.parent / 'python'
    if str(python_dir) not in sys.path:
        sys.path.insert(0, str(python_dir))
    
    from experiment_tracking import ExperimentTracker
    EXPERIMENT_TRACKING_AVAILABLE = True
    logger.info("✓ Experiment tracking module loaded successfully")
except ImportError as e:
    EXPERIMENT_TRACKING_AVAILABLE = False
    logger.warning(f"Experiment tracking module not available: {e}. Install MLflow/wandb packages for experiment tracking.")
    
    # Define a dummy class for when experiment tracking is not available
    class ExperimentTracker:
        def __init__(self, *args, **kwargs):
            pass
        def start_experiment(self, *args, **kwargs):
            pass
        def end_experiment(self, *args, **kwargs):
            pass
        def log_param(self, *args, **kwargs):
            pass
        def log_metric(self, *args, **kwargs):
            pass
        def log_dataset_info(self, *args, **kwargs):
            pass
        def log_quality_metrics(self, *args, **kwargs):
            pass
        def log_cost_metrics(self, *args, **kwargs):
            pass
        def __enter__(self):
            return self
        def __exit__(self, *args):
            pass
            pass
        def log_parameters(self, *args, **kwargs):
            pass
        def log_metrics(self, *args, **kwargs):
            pass
        def log_artifact(self, *args, **kwargs):
            pass
        def log_dataset_info(self, *args, **kwargs):
            pass
        def log_quality_metrics(self, *args, **kwargs):
            pass
        def log_cost_tracking(self, *args, **kwargs):
            pass

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

class ExperimentConfig(BaseModel):
    enable_mlflow: bool = False
    mlflow_uri: str = "http://localhost:5000"
    enable_wandb: bool = False
    wandb_project: str = ""
    wandb_entity: str = ""
    auto_log_datasets: bool = True
    auto_log_quality: bool = True
    auto_log_costs: bool = True

class ExperimentConfigResponse(BaseModel):
    config: ExperimentConfig

class TestMLflowRequest(BaseModel):
    uri: str

class TestWandBRequest(BaseModel):
    project: str
    entity: Optional[str] = None

class ExperimentTestResponse(BaseModel):
    status: str
    message: str
    details: Optional[Dict[str, Any]] = None

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

def create_experiment_tracker():
    """Create an experiment tracker based on current configuration."""
    if not EXPERIMENT_TRACKING_AVAILABLE:
        return None
    try:
        config = load_experiment_config_sync()
        if not config.get('enable_mlflow') and not config.get('enable_wandb'):
            return None
        tracker = ExperimentTracker(
            mlflow_config={
                'enabled': config.get('enable_mlflow', False),
                'tracking_uri': config.get('mlflow_uri', 'sqlite:///mlruns/dekdataset.db'),
                'experiment_name': 'DekDataset'
            },
            wandb_config={
                'enabled': config.get('enable_wandb', False),
                'project': config.get('wandb_project', 'dekdataset'),
                'entity': config.get('wandb_entity', None),
                'run_name': f"DekDataset_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
            }
        )
        return tracker
    except Exception as e:
        logger.error(f"Failed to create experiment tracker: {e}")
        return None

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

def load_experiment_config_sync():
    """Load experiment tracking configuration"""
    # Default configuration - safe defaults that don't require external services
    default_config = {
        'enable_mlflow': False,
        'mlflow_uri': 'sqlite:///mlruns/dekdataset.db',  # Use local SQLite by default
        'enable_wandb': False,
        'wandb_project': 'dekdataset',
        'wandb_entity': '',
        'auto_log_datasets': True,
        'auto_log_quality': True,
        'auto_log_costs': True
    }
    
    try:
        experiment_config_file = app_config.config_dir / 'experiment_config.json'
        if experiment_config_file.exists():
            with open(experiment_config_file, 'r') as f:
                config = json.load(f)
                # Merge with defaults to ensure all keys are present
                for key, value in default_config.items():
                    if key not in config:
                        config[key] = value
                return config
        else:
            # Check environment variables for initial configuration
            env_config = default_config.copy()
            if os.getenv('DEKDATASET_USE_MLFLOW', '').lower() == 'true':
                env_config['enable_mlflow'] = True
            if os.getenv('MLFLOW_TRACKING_URI'):
                env_config['mlflow_uri'] = os.getenv('MLFLOW_TRACKING_URI')
            if os.getenv('DEKDATASET_USE_WANDB', '').lower() == 'true':
                env_config['enable_wandb'] = True
            if os.getenv('WANDB_PROJECT'):
                env_config['wandb_project'] = os.getenv('WANDB_PROJECT')
            if os.getenv('WANDB_ENTITY'):
                env_config['wandb_entity'] = os.getenv('WANDB_ENTITY')
            
            return env_config
    except Exception as e:
        logger.error(f"Error loading experiment config: {e}")
        return default_config

def save_experiment_config_sync(config_data: Dict[str, Any]):
    """Save experiment tracking configuration"""
    try:
        experiment_config_file = app_config.config_dir / 'experiment_config.json'
        with open(experiment_config_file, 'w') as f:
            json.dump(config_data, f, indent=2)
        return True
    except Exception as e:
        logger.error(f"Error saving experiment config: {e}")
        return False

def get_experiment_tracker(task_id: str, experiment_name: Optional[str] = None) -> Optional[ExperimentTracker]:
    """Get an experiment tracker instance based on current configuration"""
    if not EXPERIMENT_TRACKING_AVAILABLE:
        return None
    
    try:
        config = load_experiment_config_sync()
        if not config.get('enable_mlflow') and not config.get('enable_wandb'):
            return None
        
        # Create experiment tracker
        tracker = ExperimentTracker(
            mlflow_config={
                'enabled': config.get('enable_mlflow', False),
                'tracking_uri': config.get('mlflow_uri', 'http://localhost:5000'),
                'experiment_name': experiment_name or f"DekDataset_{task_id}"
            },
            wandb_config={
                'enabled': config.get('enable_wandb', False),
                'project': config.get('wandb_project', 'dekdataset'),
                'entity': config.get('wandb_entity', None),
                'run_name': f"{task_id}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
            }
        )
        
        return tracker
    except Exception as e:
        logger.error(f"Error creating experiment tracker: {e}")
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
        raise HTTPException(status_code=503, detail="DeepSeek client is not available. Check API key configuration.")

    # Initialize experiment tracker
    tracker = create_experiment_tracker()
    
    try:
        task = tm.get_task(payload.task_id)
        if not task:
            raise HTTPException(status_code=404, detail="Task not found")

        logger.info(f"Generating {payload.count} entries for task {payload.task_id} using model {payload.model}")
        if 'id' not in task:
            task['id'] = payload.task_id

        # Start experiment tracking if available
        if tracker:
            run_name = f"generate_{payload.task_id}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
            tracker.start_experiment(run_name=run_name)
            tracker.log_parameters({
                'task_id': payload.task_id,
                'task_type': task.get('type', 'custom'),
                'task_description': task.get('description', ''),
                'count': payload.count,
                'model': payload.model,
                'timestamp': datetime.now().isoformat()
            })

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
        
        # Log experiment metrics and results
        if tracker:
            # Log generation metrics
            tracker.log_metrics({
                'generation_time_seconds': generation_time,
                'entries_generated': len(processed_entries),
                'success_rate': 1.0,
                'avg_generation_time_per_entry': generation_time / max(len(processed_entries), 1)
            })
            
            # Log quality metrics if available
            if processed_quality_report:
                quality_metrics = {}
                if processed_quality_report.quality_score is not None:
                    quality_metrics['quality_score'] = processed_quality_report.quality_score
                if processed_quality_report.duplicates_removed is not None:
                    quality_metrics['duplicates_removed'] = processed_quality_report.duplicates_removed
                if processed_quality_report.average_length is not None:
                    quality_metrics['average_length'] = processed_quality_report.average_length
                
                if quality_metrics:
                    tracker.log_quality_metrics(quality_metrics)
            
            # Save dataset to cache and log as artifact
            cache_data = result_data.copy()
            cache_data['entries'] = [entry.model_dump() for entry in processed_entries]
            cache_data['quality_report'] = processed_quality_report.model_dump()
            cache_data['generated_at'] = cache_data['generated_at'].isoformat()
            
            save_dataset_to_cache_sync(payload.task_id, cache_data)
              # End experiment tracking
            tracker.end_experiment()
        else:
            # Save to cache even without tracking
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
        raise HTTPException(status_code=503, detail="DeepSeek client is not available. Check API key configuration.")

    # Initialize experiment tracker for test
    tracker = create_experiment_tracker()

    try:
        task = tm.get_task(payload.task_id)
        if not task:
            raise HTTPException(status_code=404, detail="Task not found")

        logger.info(f"Testing generation for task {payload.task_id} using model {payload.model}")
        if 'id' not in task:
            task['id'] = payload.task_id

        # Start experiment tracking for test
        if tracker:
            run_name = f"test_{payload.task_id}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
            tracker.start_experiment(run_name=run_name)
            tracker.log_parameters({
                'task_id': payload.task_id,
                'task_type': task.get('type', 'custom'),
                'test_mode': True,
                'count': 3,
                'model': payload.model
            })

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
        
        # Log test metrics
        if tracker:
            tracker.log_metrics({
                'test_generation_time_seconds': generation_time,
                'test_entries_generated': len(processed_entries)
            })
            
            if processed_quality_report and processed_quality_report.quality_score is not None:
                tracker.log_metrics({'test_quality_score': processed_quality_report.quality_score})
            
            tracker.end_experiment()
        
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

@app.get("/api/experiment-config", response_model=ExperimentConfigResponse, summary="Get Experiment Config", tags=["Experiment Tracking API"])
async def get_experiment_config_api():
    """Get current experiment tracking configuration."""
    try:
        config_dict = load_experiment_config_sync()
        return ExperimentConfigResponse(config=ExperimentConfig(**config_dict))
    except Exception as e:
        logger.error(f"Error getting experiment config: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/api/experiment-config", response_model=MessageResponse, summary="Update Experiment Config", tags=["Experiment Tracking API"])
async def update_experiment_config_api(config_update: ExperimentConfig):
    """Update experiment tracking configuration."""
    try:
        success = save_experiment_config_sync(config_update.model_dump())
        if not success:
            raise HTTPException(status_code=500, detail="Failed to save experiment config")
        
        logger.info("Experiment config updated")
        return MessageResponse(
            message="Experiment tracking configuration saved successfully"
        )
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error updating experiment config: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/api/test-mlflow", response_model=ExperimentTestResponse, summary="Test MLflow Configuration", tags=["Experiment Tracking API"])
async def test_mlflow_api(request: TestMLflowRequest):
    """Test MLflow configuration and connectivity."""
    try:
        # Try to import MLflow
        try:
            import mlflow
            from mlflow.tracking import MlflowClient
        except ImportError:
            return ExperimentTestResponse(
                status="error",
                message="MLflow package not installed. Install with: pip install mlflow"
            )
          # Test connection to MLflow server with timeout
        try:
            # Set tracking URI
            mlflow.set_tracking_uri(request.uri)
            
            # For remote servers, test connectivity first
            if request.uri.startswith(('http://', 'https://')):
                logger.info(f"Testing remote MLflow server connection to {request.uri}")
                import requests
                test_url = f"{request.uri.rstrip('/')}/health"
                try:
                    # Test with short timeout
                    response = requests.get(test_url, timeout=10)
                    if response.status_code != 200:
                        return ExperimentTestResponse(
                            status="warning",
                            message=f"MLflow server at {request.uri} is reachable but health check returned status {response.status_code}",
                            details={"uri": request.uri, "status_code": response.status_code}
                        )
                except requests.exceptions.Timeout:
                    return ExperimentTestResponse(
                        status="error",
                        message=f"Connection to MLflow server at {request.uri} timed out",
                        details={"uri": request.uri, "error": "Connection timeout"}
                    )
                except requests.exceptions.ConnectionError:
                    return ExperimentTestResponse(
                        status="error",
                        message=f"Cannot connect to MLflow server at {request.uri}. Make sure the server is running and accessible.",
                        details={"uri": request.uri, "error": "Connection refused"}
                    )
                except requests.exceptions.RequestException as e:
                    return ExperimentTestResponse(
                        status="error",
                        message=f"Error connecting to MLflow server: {str(e)}",
                        details={"uri": request.uri, "error": str(e)}
                    )
            
            # Test MLflow client operations
            client = MlflowClient(tracking_uri=request.uri)
            
            # Try to list experiments with timeout handling
            try:
                import asyncio
                import concurrent.futures
                
                def test_experiments():
                    return client.search_experiments(max_results=1)
                
                # Run with timeout
                with concurrent.futures.ThreadPoolExecutor() as executor:
                    future = executor.submit(test_experiments)
                    try:
                        experiments = future.result(timeout=15)  # 15 second timeout
                        
                        return ExperimentTestResponse(
                            status="success",
                            message=f"Successfully connected to MLflow at {request.uri}",
                            details={
                                "uri": request.uri,
                                "experiments_found": len(experiments),
                                "mlflow_version": mlflow.__version__
                            }
                        )
                    except concurrent.futures.TimeoutError:
                        return ExperimentTestResponse(
                            status="error",
                            message=f"MLflow operations timed out for {request.uri}",
                            details={"uri": request.uri, "error": "Operation timeout"}
                        )
                        
            except Exception as client_error:
                return ExperimentTestResponse(
                    status="error",
                    message=f"MLflow client error: {str(client_error)}",
                    details={"uri": request.uri, "error": str(client_error)}
                )
            
        except Exception as mlflow_error:
            return ExperimentTestResponse(
                status="error",
                message=f"Failed to connect to MLflow: {str(mlflow_error)}",
                details={"uri": request.uri, "error": str(mlflow_error)}
            )
            
    except Exception as e:
        logger.error(f"Error testing MLflow: {e}")
        return ExperimentTestResponse(
            status="error",
            message=f"Error testing MLflow configuration: {str(e)}"
        )

@app.post("/api/test-wandb", response_model=ExperimentTestResponse, summary="Test Weights & Biases Configuration", tags=["Experiment Tracking API"])
async def test_wandb_api(request: TestWandBRequest):
    """Test Weights & Biases configuration and connectivity."""
    try:
        # Try to import wandb
        try:
            import wandb
        except ImportError:
            return ExperimentTestResponse(
                status="error",
                message="Weights & Biases package not installed. Install with: pip install wandb"
            )
        
        # Check if user is logged in
        try:
            # Check if API key is configured
            api_key = wandb.api.api_key
            if not api_key:
                return ExperimentTestResponse(
                    status="error",
                    message="W&B API key not found. Please run 'wandb login' or set WANDB_API_KEY environment variable"
                )
            
            # Try to access the API
            api = wandb.Api()
            
            # Test project access if entity is provided
            if request.entity:
                try:
                    # This will test if we can access the entity
                    entity_obj = api.user(request.entity)
                    return ExperimentTestResponse(
                        status="success",
                        message=f"Successfully connected to W&B. Ready to use project '{request.project}' under entity '{request.entity}'",
                        details={
                            "project": request.project,
                            "entity": request.entity,
                            "wandb_version": wandb.__version__
                        }
                    )
                except Exception as entity_error:
                    return ExperimentTestResponse(
                        status="warning",
                        message=f"Connected to W&B but couldn't access entity '{request.entity}'. The entity might not exist or you might not have access.",
                        details={
                            "project": request.project,
                            "entity": request.entity,
                            "wandb_version": wandb.__version__,
                            "entity_error": str(entity_error)
                        }
                    )
            else:
                return ExperimentTestResponse(
                    status="success",
                    message=f"Successfully connected to W&B. Ready to use project '{request.project}'",
                    details={
                        "project": request.project,
                        "wandb_version": wandb.__version__
                    }
                )
                
        except Exception as wandb_error:
            return ExperimentTestResponse(
                status="error",
                message=f"Failed to connect to W&B: {str(wandb_error)}",
                details={"error": str(wandb_error)}
            )
            
    except Exception as e:
        logger.error(f"Error testing W&B: {e}")
        return ExperimentTestResponse(
            status="error",
            message=f"Error testing W&B configuration: {str(e)}"
        )

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
                    df_data.append(entryDict)

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

@app.get("/api/download/csv/{task_id}")
async def download_csv(task_id: str):
    dataset_path = app_config.cache_dir / f"{task_id}.json"
    if not dataset_path.exists():
        raise HTTPException(status_code=404, detail="Dataset not found")
    with open(dataset_path, "r", encoding="utf-8") as f:
        entries = json.load(f)
    # Flatten content+metadata for CSV
    rows = []
    for entry in entries:
        row = {"id": entry.get("id", "")}
        if "content" in entry and isinstance(entry["content"], dict):
            row.update(entry["content"])
        row["metadata"] = json.dumps(entry.get("metadata", {}), ensure_ascii=False)
        rows.append(row)
    if not rows:
        raise HTTPException(status_code=404, detail="No data to export")
    fieldnames = list(rows[0].keys())
    csv_buffer = StringIO()
    writer = csv.DictWriter(csv_buffer, fieldnames=fieldnames)
    writer.writeheader()
    for row in rows:
        writer.writerow(row)
    csv_buffer.seek(0)
    return StreamingResponse(csv_buffer, media_type="text/csv", headers={
        "Content-Disposition": f"attachment; filename={task_id}.csv"
    })

@app.get("/api/download/json/{task_id}")
async def download_json(task_id: str):
    dataset_path = app_config.cache_dir / f"{task_id}.json"
    if not dataset_path.exists():
        raise HTTPException(status_code=404, detail="Dataset not found")
    return FileResponse(str(dataset_path), media_type="application/json", filename=f"{task_id}.json")

@app.get("/api/preview/{task_id}", response_class=HTMLResponse)
async def preview_dataset(task_id: str, limit: int = 20):
    dataset_path = app_config.cache_dir / f"{task_id}.json"
    if not dataset_path.exists():
        return HTMLResponse("<div class='alert alert-warning'>Dataset not found</div>")
    with open(dataset_path, "r", encoding="utf-8") as f:
        entries = json.load(f)
        # รองรับ dict ที่มี key "entries"
        if isinstance(entries, dict) and "entries" in entries:
            entries = entries["entries"]
        if not isinstance(entries, list):
            entries = list(entries.values())
        # Flatten for table
        rows = []
        for entry in entries[:limit]:
            if not isinstance(entry, dict):
                continue
            row = {}
            # Always include id if present
            row["id"] = entry.get("id", "")
            # Flatten content, raw_data, input, output
            if "content" in entry and isinstance(entry["content"], dict):
                row.update(entry["content"])
            if "raw_data" in entry and isinstance(entry["raw_data"], dict):
                row.update(entry["raw_data"])
            # If input/output are not None and not dict, add as columns
            if entry.get("input") is not None and not isinstance(entry.get("input"), dict):
                row["input"] = entry["input"]
            if entry.get("output") is not None and not isinstance(entry.get("output"), dict):
                row["output"] = entry["output"]
            # Optionally, include metadata as JSON string
            if "metadata" in entry:
                row["metadata"] = json.dumps(entry.get("metadata", {}), ensure_ascii=False)
            rows.append(row)
    if not rows:
        return HTMLResponse("<div class='alert alert-warning'>No data to preview</div>")
    # Collect all fieldnames from all rows for a complete header
    fieldnames = set()
    for row in rows:
        fieldnames.update(row.keys())
    fieldnames = list(fieldnames)
    # Generate HTML table
    table_html = "<div style='overflow-x:auto'><table class='table table-striped' style='width:100%;border-collapse:collapse;'>"
    table_html += "<thead><tr>" + "".join(f"<th style='background:#f8f9fa;border:1px solid #dee2e6;padding:8px'>{fn}</th>" for fn in fieldnames) + "</tr></thead>"
    table_html += "<tbody>"
    for row in rows:
        table_html += "<tr>" + "".join(f"<td style='border:1px solid #dee2e6;padding:8px'>{str(row.get(fn, ''))}</td>" for fn in fieldnames) + "</tr>"
    table_html += "</tbody></table></div>"
    return HTMLResponse(table_html)

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
        "app:app",
        host="127.0.0.1",
        port=8000,
        reload=True,
        log_level="info"
    )
