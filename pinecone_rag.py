import pinecone
import openai
import os
import tiktoken
import time
from datetime import datetime, date
from typing import List, Literal
from pydantic import BaseModel, Field
import instructor
from pinecone import Pinecone
from dotenv import load_dotenv
from openai import OpenAI
from groq import Groq

load_dotenv()
# Initialize the Pinecone client
pc_key = os.getenv("PINECONE_API_KEY")

pc = Pinecone(api_key=pc_key)
user_index = pc.Index("user-data-openai-embedding")
book_index = pc.Index("ah-test")

# Initialize the OpenAI client
openai.api_key = os.getenv("OPENAI_API_KEY")
client_openai = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
# client_groq = Groq(api_key=os.getenv('GROQ_API_KEY'))
client_groq = Groq(api_key=os.environ.get("GROQ_API_KEY"))
client = instructor.from_groq(client_groq, mode=instructor.Mode.TOOLS)

client_openai_instructor = instructor.from_openai(client_openai)  # Apply the patch to the OpenAI client

# Classification model
class ClassificationResponse(BaseModel):
    label: Literal["ATOMIC_HABITS", "PERSONAL","WEB"] = Field(
        ...,
        description="The predicted class label.",
    )

def classify(data: str, keywords: List[str]) -> ClassificationResponse:
    """Perform single-label classification on the input text."""
    return client_openai_instructor.chat.completions.create(
        model="gpt-3.5-turbo",
        # model = 'mixtral-8x7b-32768',
        response_model=ClassificationResponse,
        messages=[
            {
                'role': 'system',
                'content': f'Use these keywords to determine the appropriate classification if any of them match the data then the classification should be ATOMIC_HABITS: {" ".join(keywords)}'
            },
            {
                "role": "user",
                "content": f"Classify the following text: {data}",
            },
        ],
    )

def get_embedding(text, model="text-embedding-ada-002"):
    response = client_openai.embeddings.create(input=[text], model=model)
    return response.data[0].embedding


def query_pinecone_user(query_string,index, top_k=10, namespace="",filter={"user_id": 1}):
    xc = get_embedding(query_string)
    result = index.query(vector=xc, top_k=top_k, include_metadata=True, namespace=namespace,filter=filter)
    return result

def query_pinecone_book(query_string,index, top_k=10, namespace=""):
    xc = get_embedding(query_string)
    result = index.query(vector=xc, top_k=top_k, include_metadata=True, namespace=namespace)
    return result

def construct_prompt(data, query):
    prompt = f"Answer the question based on the context below, and if the question can't be answered based on the context, please provide a thoughtful response or indicate that you do not have enough information."

def get_context_string(contexts: List[str]) -> str:
    return " ".join(contexts)

async def manage_conversation_tokens(conversation: List[str], call_id: str) -> List[str]:
    def num_tokens_from_messages(messages, tkmodel="cl100k_base"):
        encoding = tiktoken.get_encoding(tkmodel)
        num_tokens = 0
        for message in messages:
            num_tokens += 4  # every message follows <im_start>{role/name}\\n{content}<im_end>\\n
            for key, value in message.items():
                num_tokens += len(encoding.encode(value))
                if key == "name":  # if there's a name, the role is omitted
                    num_tokens += -1  # role is always required and always 1 token
        return num_tokens + 2  # every reply is primed with <im_start>assistant

    conv_history_tokens = num_tokens_from_messages(conversation)
    print("tokens: " + str(conv_history_tokens))
    token_limit = 32000
    max_response_tokens = 300
    while (conv_history_tokens + max_response_tokens >= token_limit):
        del conversation[1]
        conv_history_tokens = num_tokens_from_messages(conversation)
    
    # Summarize if necessary
    if conv_history_tokens + max_response_tokens >= token_limit:
        summarization = summarize_conversation(conversation)
        summary_embedding = get_embedding(summarization)
        idv = "id" + str(time.time())
        user_index.upsert(vectors=[{"id": idv, 
                                    "values": summary_embedding, 
                                    "metadata": {'text': summarization, 'user_id': call_id}}],
                          namespace='user-data-openai-embedding')
        return [summarization]

    return conversation

# Summarization function
def summarize_conversation(context: List[str]) -> str:
    now = datetime.now()
    current_time = now.strftime("%H:%M:%S")
    today = date.today()
    date_string = str(today)
    
    completion =client.chat.completions.create(
        # engine="gpt-3.5-turbo",
        model="mixtral-8x7b-32768",
        prompt=f"Summarize the following conversation that occurred at {current_time} on {date_string}:\\n{context}",
        temperature=0.3,
        max_tokens=300,
        top_p=0.9,
        presence_penalty=0
    )
    summarization = completion.choices[0].text.strip()
    return summarization
