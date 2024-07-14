from raptor import RetrievalAugmentation, RetrievalAugmentationConfig
from raptor.QAModels import GPT4QAModel
import instructor
import openai
from pydantic import BaseModel
from typing import List
from dotenv import load_dotenv
import os

load_dotenv()

api_key = os.getenv("OPENAI_API_KEY")

SAVE_PATH = "/Users/clydeclarke/Documents/server-side-example-python-flask/demo/atomic_habits"

# Initialize the RA class
RAC = RetrievalAugmentationConfig(qa_model=GPT4QAModel())
RA = RetrievalAugmentation(tree=SAVE_PATH, config=RAC)

class UserResponse(BaseModel):
    user_response:str

client = instructor.from_openai(openai.OpenAI(api_key=api_key))

def get_retrieval_book(question: str,context: str) -> str:
    return RA.answer_question(question=question,context=context)


def get_retrieval_user(question: str, context: List[str]):
    # Use the generate_prompt function to create the prompt
    user_name = context.split("[")[0]
    statement = question
    # print("**********THIS IS THE USERNAME LLM***********", user_name)
    response = client.chat.completions.create(
        model="gpt-4o",
        messages=[
            {"role": "system",
             "content": f"Respond with an appropriate response with the user_name and statement. Give an acknowledgement of the information provided by {user_name} who said '{statement}'. For example, you can say 'thank you for sharing that, {user_name}'. Add a follow-up to the question."
            },
            {"role": "user", "content": statement}
        ],
        response_model=UserResponse,  # Changed to FollowupResponse for parsing the choices
    )  # type: ignore

    print(response.model_dump_json())  # Debugging output

    return response