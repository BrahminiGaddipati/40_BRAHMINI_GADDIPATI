from pydantic import BaseModel
from typing import List

class PatientResponse(BaseModel):
    question: str
    explanation: str
    warnings: List[str]
    recommended_action: str
    disclaimer: str
