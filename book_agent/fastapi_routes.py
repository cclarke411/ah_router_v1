
@app.post("/query")
async def query_processor(request: UserQuery):
    user_profile = await get_user_profile(request.user_id)
    if not user_profile:
        raise HTTPException(status_code=404, detail="User not found")
    
    memory_data = await get_memory(request.user_id)
    llm_response = await query_llm(request.question, memory_data.context)
    
    if llm_response.followup:
        new_context = memory_data.context + [request.question, llm_response.response]
        await update_memory(request.user_id, MemoryData(context=new_context))
    
    if llm_response.update_db:
        user_profile.interactions.append({"question": request.question, "response": llm_response.response, "timestamp": datetime.utcnow().isoformat()})
        await update_user_profile(user_profile)
    
    return {
        "followup": llm_response.followup,
        "response": llm_response.response
    }

@app.get("/user/{user_id}")
async def get_user(user_id: str):
    user_profile = await get_user_profile(user_id)
    if user_profile:
        return user_profile
    raise HTTPException(status_code=404, detail="User not found")
