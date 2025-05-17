# task_definitions.py
# รวม task definition ตาม schema Rust (ตัวอย่าง 6 task หลัก)

from typing import Any, Dict, List, Optional, Union
import json
from pathlib import Path

def get_task_definitions() -> Dict[str, Any]:
    # All task definitions as Python dicts, not Rust code
    json_path = Path(__file__).parent / "tasks.json"
    with open(json_path, encoding="utf-8") as f:
        return json.load(f)
