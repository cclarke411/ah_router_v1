from typing import List
import openai
from pydantic import BaseModel
import instructor
from flask import jsonify
from dotenv import load_dotenv
import os

load_dotenv()
api_key = os.getenv('OPENAI_API_KEY')

class Message(BaseModel):
    role: str
    content: str

class Choice(BaseModel):
    message: Message

class FollowupResponse(BaseModel):
    choices: List[Choice]

class LLMResponse(BaseModel):
    followup: bool
    response: str
    update_db: bool

client = instructor.from_openai(openai.OpenAI(api_key=api_key))

def generate_prompt(context: List[str], user_question: str) -> str:
    # Join the context list into a single string with line breaks
    context_str = "\n".join(context)
    user_question = f"{user_question}"

    # Define the prompt template
    prompt_template = """
    You are an intelligent assistant. Your task is to determine if a new user question is a follow-up to previous questions or a new, standalone question. Use the context of the conversation to make your decision. Here is the conversation context and the new user question:

    Context:
    {context_str}

    New User Question:
    "{user_question}"

    Instructions:
    Based on the context of the conversation and the new user question, determine if the new question is a follow-up to the previous questions or a new, standalone question. Respond with "Follow-up" if it is related to the previous context or "Standalone" if it is a new, independent question. Additionally, provide a structured JSON response with the following fields:
    - "followup": A boolean indicating if the question is a follow-up.
    - "response": A restatement of the user question.
    - "update_db": A boolean indicating whether the database should be updated with this new interaction.

    Output:
    json
    {
    "followup": <true_or_false>,
    "response": "<response_text>",
    "update_db": <true_or_false>
    }"""
    return prompt_template

def query_llm(question: str, context: List[str]) -> LLMResponse:
    # Use the generate_prompt function to create the prompt
    prompt = generate_prompt(context, question)
    response = client.chat.completions.create(
        model="gpt-4o",
        messages=[
            {"role": "system", "content": prompt},
            {"role": "user", "content": question}
        ],
        response_model=LLMResponse,  # Changed to FollowupResponse for parsing the choices
    )  # type: ignore

    # print(response.model_dump_json())  # Debugging output

    return response.model_dump_json()
