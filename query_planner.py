from models import AddKnowledge, Action, UserQueryData, Category
import openai
import instructor
import pandas as pd
from pydantic import BaseModel
from openai import OpenAI
from typing import List
from cache import instructor_cache
from dotenv import load_dotenv
import os

api_key = os.getenv("OPENAI_API_KEY")
# Define the response model
class ClassifyAndSplitResponse(BaseModel):
    book_related: str
    user_related: str

# Initialize the instructor client with OpenAI
client = instructor.from_openai(openai.OpenAI(api_key=api_key))

# Load the CSV file
csv_path = "/Users/clydeclarke/Documents/AH_Code_Architecture/data/ah_index.csv"
index_data = pd.read_csv(csv_path)
index_data.columns = index_data.columns.str.strip()

# Extract concepts and words related to "Atomic Habits"
book_concepts = index_data['concept'].tolist()
book_words = index_data['word'].tolist()
book_keywords = book_words + book_concepts

def classify_and_split_query(input_text: str, context: List[str]) -> ClassifyAndSplitResponse:
    context = f"Atomic Habits concepts and words: {', '.join(book_concepts)}"
    response = client.chat.completions.create(
        model="gpt-4o",
        messages=[
            {
                "role": "system",
                "content": "You are a helpful assistant that classifies parts of a user query as either book-related or user-related. If the query references goals or habits, it is likely book-related. If the query is more general, it is likely user-related. Do not include the user_id in the response",
            },
            {
                "role": "user",
                "content": f"Context: {context}\n\nQuery: {input_text}\n\nPlease split the query into book-related and user-related parts. Choose the concept that most closely aligns with the query pick the first one you find."
            }
        ],
        response_model=ClassifyAndSplitResponse,
    )
    return response

def generate_add_knowledge(input_text: str, context: List[str]) -> dict:
    classified_parts = classify_and_split_query(input_text, context)
    
    responses = {}

    # Handle user-related part
    if classified_parts.user_related:
        user_response = client.chat.completions.create(
            model="gpt-4o",
            messages=[
                            {
                    "role": "system",
                    "content": f"Context: {' '.join(context)}\nBased on the following user query, provide a response that can be converted into an AddKnowledge model: {classified_parts.user_related}"
                }
            ],
            response_model=AddKnowledge,
        )
        user_data = user_response.model_dump()
        user_data['category'] = user_data['category'].value
        user_data['action'] = user_data['action'].value
        responses["user"] = user_data

    # Handle book-related part
    if classified_parts.book_related:
        book_add_knowledge = AddKnowledge(
            user_id="1",  # You may need to adapt this part to match your user ID logic
            key="UserQuestion",
            knowledge=classified_parts.book_related,
            category=Category.Topic_Interest,
            action=Action.Retrieve
        )
        book_data = book_add_knowledge.model_dump()
        book_data['category'] = book_data['category'].value
        book_data['action'] = book_data['action'].value
        responses["book"] = book_data  # Ensure it is serializable

    # print("Responses:", responses)
    return responses

def parse_query(query: str, context: List[str]) -> dict:
    return generate_add_knowledge(query, context)

def process_add_knowledge(add_knowledge: AddKnowledge) -> UserQueryData:
    return UserQueryData(
        user_id=add_knowledge.user_id,
        key=add_knowledge.key,
        value=add_knowledge.knowledge if add_knowledge.knowledge else "",
        action=add_knowledge.action.value,
        category=add_knowledge.category.value
    ).model_dump()