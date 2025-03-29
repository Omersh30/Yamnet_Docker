from pydantic import BaseModel
from typing import Dict, List, Optional, Any
from datetime import datetime

class AnalysisResponse(BaseModel):
    status: str
    results: Dict[str, Any]
    timestamp: datetime

class HealthCheck(BaseModel):
    status: str

class ErrorResponse(BaseModel):
    detail: str 