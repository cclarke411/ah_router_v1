
async def query_llm(question: str, context: List[str]) -> LLMResponse:
    context_str = " ".join(context)
    return client.chat.completions.create(
        model="gpt-3.5-turbo",
        messages=[
            {"role": "system", "content": context_str},
            {"role": "user", "content": question}
        ],
        response_model=LLMResponse,
    ) # type: ignore
