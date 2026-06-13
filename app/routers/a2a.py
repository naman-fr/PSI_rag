import uuid
import logging
from datetime import datetime, timezone
from typing import Dict, Any, List, Optional
from fastapi import APIRouter, HTTPException, BackgroundTasks
from pydantic import BaseModel, Field

from app.main import get_orchestrator

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/a2a", tags=["a2a"])

# In-memory storage for tasks
tasks_db: Dict[str, Dict[str, Any]] = {}

# --- Pydantic Models ---
class TaskInput(BaseModel):
    question: str = Field(..., description="The logistics question to ask the RAG pipeline")
    username: str = Field(default="a2a_client", description="The requesting agent's username")

class CreateTaskRequest(BaseModel):
    skillId: str = Field(..., description="The ID of the skill to execute")
    input: TaskInput = Field(..., description="The inputs for the task")

class Artifact(BaseModel):
    name: str
    mimeType: str
    content: str

class TaskResponse(BaseModel):
    taskId: str
    status: str
    skillId: str
    input: TaskInput
    output: Optional[str] = None
    artifacts: List[Artifact] = []
    createdAt: str

# --- Endpoints ---

@router.get("/agent-card")
async def get_agent_card():
    """Returns the Agent Card describing capabilities and skills."""
    return {
        "name": "PSI-Logistics-RAG-Agent",
        "description": "An agent that provides verified, guardrailed carrier SLA advice, tariff definitions, and delay exception guidelines.",
        "version": "1.0.0",
        "url": "/a2a",
        "capabilities": {
            "streaming": False
        },
        "skills": [
            {
                "id": "logistics_sla_advisor",
                "name": "Logistics SLA Advisor",
                "description": "Exposes carrier agreement SLA terms, delay thresholds, compensations, and customs tariff reference queries.",
                "tags": ["logistics", "sla", "customs", "tariffs", "delays"]
            }
        ]
    }

@router.post("/tasks", response_model=TaskResponse)
async def create_task(req: CreateTaskRequest):
    """Creates a new stateful task for delegation."""
    if req.skillId != "logistics_sla_advisor":
        raise HTTPException(status_code=400, detail=f"Unsupported skill ID: {req.skillId}")

    task_id = f"task-{uuid.uuid4()}"
    task = {
        "taskId": task_id,
        "status": "created",
        "skillId": req.skillId,
        "input": req.input.model_dump() if hasattr(req.input, "model_dump") else req.input.dict(),
        "output": None,
        "artifacts": [],
        "createdAt": datetime.now(timezone.utc).isoformat()
    }
    tasks_db[task_id] = task
    return task

@router.get("/tasks/{task_id}", response_model=TaskResponse)
async def get_task(task_id: str):
    """Retrieves the status and results of a task."""
    task = tasks_db.get(task_id)
    if not task:
        raise HTTPException(status_code=404, detail=f"Task not found: {task_id}")
    return task

async def _execute_task_in_background(task_id: str):
    task = tasks_db.get(task_id)
    if not task:
        return
    
    orchestrator = get_orchestrator()
    if not orchestrator:
        task["status"] = "failed"
        task["output"] = "RAG orchestrator not initialized on server."
        return
        
    try:
        task["status"] = "running"
        question = task["input"]["question"]
        username = task["input"]["username"]
        
        response = await orchestrator.process_query(
            question=question,
            username=username
        )
        answer = response.get("answer", "No answer generated.")
        
        task["output"] = answer
        task["artifacts"] = [
            {
                "name": "grounded_response",
                "mimeType": "text/plain",
                "content": answer
            }
        ]
        task["status"] = "completed"
    except Exception as e:
        logger.exception("A2A task execution failed")
        task["status"] = "failed"
        task["output"] = f"Execution error: {str(e)}"

@router.put("/tasks/{task_id}/execute", response_model=TaskResponse)
async def execute_task(task_id: str, background_tasks: BackgroundTasks):
    """Triggers the execution of a created task."""
    task = tasks_db.get(task_id)
    if not task:
        raise HTTPException(status_code=404, detail=f"Task not found: {task_id}")
        
    if task["status"] not in ["created", "failed"]:
        raise HTTPException(status_code=400, detail=f"Task cannot be executed in state: {task['status']}")
        
    # Queue execution
    background_tasks.add_task(_execute_task_in_background, task_id)
    
    # Return running or updated task state
    task["status"] = "running"
    return task
