
class UserQuery(BaseModel):
    user_id: str
    question: str

class UserProfile(BaseModel):
    user_id: str
    goals: List[str]
    interactions: List[Dict] = []

class MemoryData(BaseModel):
    context: List[str] = []

class LLMResponse(BaseModel):
    followup: bool
    response: str
    update_db: bool
