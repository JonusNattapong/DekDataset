"""
Task Manager for DekDataset
Manages different types of dataset generation tasks
"""

import json
import os
from typing import Dict, List, Any, Tuple, Optional
from dataclasses import dataclass, asdict
from datetime import datetime

@dataclass
class TaskTemplate:
    """Template for creating new tasks"""
    name: str
    description: str
    format: str
    schema: Dict[str, Any]
    examples: List[Dict[str, Any]]
    parameters: Dict[str, Any]
    difficulty: str = "medium"
    domain: str = "general"
    language: str = "th"
    created_at: Optional[str] = None
    version: str = "1.0"

class TaskManager:
    """Advanced task management system"""
    
    def __init__(self, tasks_file: str = "tasks.json"):
        self.tasks_file = tasks_file
        self.tasks = {}
        self.load_tasks()
    
    def load_tasks(self):
        """Load tasks from JSON file"""
        try:
            if os.path.exists(self.tasks_file):
                with open(self.tasks_file, 'r', encoding='utf-8') as f:
                    self.tasks = json.load(f)
                print(f"[INFO] Loaded {len(self.tasks)} tasks from {self.tasks_file}")
            else:
                print(f"[WARN] Tasks file {self.tasks_file} not found")
        except Exception as e:
            print(f"[ERROR] Failed to load tasks: {e}")
    
    def save_tasks(self):
        """Save tasks to JSON file"""
        try:
            with open(self.tasks_file, 'w', encoding='utf-8') as f:
                json.dump(self.tasks, f, ensure_ascii=False, indent=2)
            print(f"[INFO] Saved {len(self.tasks)} tasks to {self.tasks_file}")
        except Exception as e:
            print(f"[ERROR] Failed to save tasks: {e}")
    
    def create_task(self, task_id: str, template: TaskTemplate) -> bool:
        """Create a new task from template"""
        try:
            if task_id in self.tasks:
                print(f"[WARN] Task {task_id} already exists")
                return False
            
            template.created_at = datetime.now().isoformat()
            self.tasks[task_id] = asdict(template)
            self.save_tasks()
            
            print(f"[INFO] Created new task: {task_id}")
            return True
            
        except Exception as e:
            print(f"[ERROR] Failed to create task {task_id}: {e}")
            return False
    
    def get_task(self, task_id: str) -> Optional[Dict[str, Any]]:
        """Get task by ID"""
        return self.tasks.get(task_id)
    
    def list_tasks(self, domain: Optional[str] = None, difficulty: Optional[str] = None) -> List[str]:
        """List tasks with optional filtering"""
        filtered_tasks = []
        
        for task_id, task_data in self.tasks.items():
            if domain and task_data.get('domain') != domain:
                continue
            if difficulty and task_data.get('difficulty') != difficulty:
                continue
            filtered_tasks.append(task_id)
        
        return filtered_tasks
    
    def update_task(self, task_id: str, updates: Dict[str, Any]) -> bool:
        """Update existing task"""
        try:
            if task_id not in self.tasks:
                print(f"[ERROR] Task {task_id} not found")
                return False
            
            self.tasks[task_id].update(updates)
            self.tasks[task_id]['updated_at'] = datetime.now().isoformat()
            self.save_tasks()
            
            print(f"[INFO] Updated task: {task_id}")
            return True
            
        except Exception as e:
            print(f"[ERROR] Failed to update task {task_id}: {e}")
            return False
    
    def validate_task(self, task_data: Dict[str, Any]) -> Tuple[bool, List[str]]:
        """Validate task structure"""
        errors = []
        
        # Required fields
        required_fields = ['name', 'description', 'format', 'schema', 'examples']
        for field in required_fields:
            if field not in task_data:
                errors.append(f"Missing required field: {field}")
        
        # Schema validation
        if 'schema' in task_data:
            schema = task_data['schema']
            if 'fields' not in schema:
                errors.append("Schema must contain 'fields'")
        
        # Examples validation
        if 'examples' in task_data and task_data['examples']:
            examples = task_data['examples']
            if not isinstance(examples, list):
                errors.append("Examples must be a list")
            elif len(examples) == 0:
                errors.append("At least one example is required")
        
        return len(errors) == 0, errors

# Global task manager instance
task_manager = TaskManager()

def create_custom_task():
    """Interactive task creation"""
    print("\n=== Create Custom Task ===")
    
    task_id = input("Enter task ID: ").strip()
    if not task_id:
        print("[ERROR] Task ID is required")
        return
    
    name = input("Enter task name: ").strip()
    description = input("Enter task description: ").strip()
    format_type = input("Enter format (json/jsonl/csv): ").strip().lower()
    domain = input("Enter domain (optional): ").strip() or "general"
    difficulty = input("Enter difficulty (easy/medium/hard): ").strip().lower() or "medium"
    
    if not all([name, description, format_type]):
        print("[ERROR] Name, description, and format are required")
        return
    
    # Basic schema template
    schema = {
        "fields": {
            "input": {
                "field_type": "Text",
                "required": True,
                "description": "Input text"
            },
            "output": {
                "field_type": "Text", 
                "required": True,
                "description": "Expected output"
            }
        }
    }
    
    # Basic example
    examples = [
        {
            "input": "Example input",
            "output": "Example output"
        }
    ]
    
    template = TaskTemplate(
        name=name,
        description=description,
        format=format_type,
        schema=schema,
        examples=examples,
        parameters={},
        difficulty=difficulty,
        domain=domain
    )
    
    if task_manager.create_task(task_id, template):
        print(f"[SUCCESS] Created task '{task_id}'")
        print("You can now edit the task file to add more examples and customize the schema")
    else:
        print("[ERROR] Failed to create task")
