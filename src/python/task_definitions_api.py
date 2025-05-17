# FastAPI server for serving task definitions to both Rust and Python clients
from fastapi import FastAPI
from fastapi.responses import JSONResponse
from task_definitions import get_task_definitions
from fastapi.middleware.cors import CORSMiddleware

app = FastAPI(title="DekDataset Task Definitions API")

# Allow CORS for local dev
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.get("/tasks")
def list_tasks():
    """Return all available task definitions as JSON."""
    tasks = get_task_definitions()
    return JSONResponse(content=tasks)

@app.get("/tasks/{task_name}")
def get_task(task_name: str):
    tasks = get_task_definitions()
    if task_name in tasks:
        return JSONResponse(content=tasks[task_name])
    return JSONResponse(content={"error": "Task not found"}, status_code=404)

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("task_definitions_api:app", host="0.0.0.0", port=8000, reload=True)
