from typing import Optional
from models import UserProfile, MemoryData

user_db = {}
memory_db = {}

async def get_user_profile(user_id: str) -> Optional[UserProfile]:
    return user_db.get(user_id)

async def update_user_profile(user_profile: UserProfile):
    user_db[user_profile.user_id] = user_profile

async def get_memory(user_id: str) -> MemoryData:
    return memory_db.get(user_id, MemoryData(messages=[]))

async def update_memory(user_id: str, memory_data: MemoryData):
    memory_db[user_id] = memory_data
