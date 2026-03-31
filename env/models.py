from pydantic import BaseModel
from typing import List, Dict, Any

class State(BaseModel):
    dataset: List[Dict]
    current_step: int
    max_steps: int

class Action(BaseModel):
    type: str
    params: Dict[str, Any]

class Observation(BaseModel):
    preview: List[Dict]
    message: str
