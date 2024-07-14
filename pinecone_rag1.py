import os
import time
import tiktoken
from datetime import datetime, date
from openai import OpenAI
from pinecone import Pinecone
from dotenv import load_dotenv
from pydantic import BaseModel, Field
from typing import Literal, List
import instructor
import pandas as pd

csv_path = "/Users/clydeclarke/Documents/AH_Code_Architecture/data/ah_index.csv"
index_data = pd.read_csv(csv_path)
index_data.columns = index_data.columns.str.strip()

# Extract concepts and words related to "Atomic Habits"
atomic_habits_keywords = index_data['concept'].tolist()
atomic_habits_keywords= [atomic_habits_keywords.rstrip() for atomic_habits_keywords in atomic_habits_keywords]

# Load environment variables
load_dotenv()

# Retrieve API keys
openai_api_token = os.environ.get("OPENAI_API_KEY")
pc_key = os.environ.get("PINECONE_API_KEY")

# Print API keys for verification
print(openai_api_token)
print(pc_key)

# Initialize Pinecone and OpenAI clients
pc = Pinecone(api_key=pc_key)
client_openai = OpenAI(api_key=openai_api_token)
client = instructor.from_openai(client_openai)  # Apply the patch to the OpenAI client

# Classification model
class ClassificationResponse(BaseModel):
    label: Literal["ATOMIC_HABITS", "PERSONAL"] = Field(
        ...,
        description="The predicted class label.",
    )

def classify(data: str, keywords: List[str]) -> ClassificationResponse:
    """Perform single-label classification on the input text."""
    return client.chat.completions.create(
        model="gpt-3.5-turbo",
        response_model=ClassificationResponse,
        messages=[
            {
                'role': 'system',
                'content': f'Use these keywords to determine the appropriate classification if any of them match the data then the classification should be ATOMIC_HABITS{" ".join(keywords)}'
            },
            {
                "role": "user",
                "content": f"Classify the following text: {data}",
            },
        ],
    )

# Main function to prompt user to initiate conversation
def wait_for_input():
    while True:
        user_input = input("You: ").strip()
        if user_input.lower() in ["start", "hello", "hi"]:
            conversation_time()
        elif user_input.lower() == "quit":
            print("Goodbye!")
            break

# Conversation function to handle the conversation flow
def conversation_time():
    # Setup initial conversation context
    now = datetime.now()
    current_time = now.strftime("%H:%M:%S")
    today = date.today()
    date_string = str(today)
    system_message = {"role": "system", "content": f"It is {current_time} on {date_string} and you are a friendly robot that enjoys being helpful to your human friend."}
    user_greeting = {"role": "user", "content": "Hey bot."}
    bot_response = {"role": "assistant", "content": "What's up?"}
    conversation = [system_message, user_greeting, bot_response]
    
    user_id = 2
    atomic_habits_keywords = ["habit", "Atomic Habits", "James Clear", "self-improvement", "routine", "productivity"]
    index = pc.Index("user-data-openai-embedding")  # User data index
    book_index = pc.Index("ah-test")  # Book data index
    
    while True:
        # User input section
        user_input = input("You: ").strip()
        if user_input.lower() == "quit":
            print("Goodbye!")
            break

        # Classify the input to determine context
        classification_result = classify(user_input, atomic_habits_keywords)
        classification_label = classification_result.label

        # Embedding the prompt using OpenAI
        response = client_openai.embeddings.create(
            input=[user_input],
            model="text-embedding-ada-002"  # Replace with the appropriate OpenAI model
        )
        xq = response.data[0].embedding

        # Vector context retrieval based on classification
        contexts = []
        if classification_label == "PERSONAL":
            namespace = 'user-data-openai-embedding'
            res = index.query(vector=xq, 
                              top_k=1, 
                              filter={"user_id": user_id},
                              include_metadata=True,
                              namespace=namespace)
            contexts = contexts + [x['metadata']['text'] for x in res['matches']]
            print(f"Response: {res}")
        elif classification_label == "ATOMIC_HABITS":
            namespace = 'ah-test'
            book_res = book_index.query(vector=xq,
                                        top_k=1,
                                        include_metadata=True,
                                        include_values=True,
                                        namespace=namespace)
            print(f"Response: {book_res}")
            contexts = contexts + [x['metadata']['text'] for x in book_res['matches']]
        
        print(f"Retrieved {len(contexts)} contexts")
        time.sleep(0.5)

        # Merge context with user query and update conversation history
        prompt_end = "The following may or may not be relevant information from past conversations. If it is not relevant to this conversation, ignore it:\n\n"
        prompt = user_input + "\n\n" + prompt_end + "\n\n---\n\n".join(contexts[:1])
        conversation.append({"role": "user", "content": prompt})

        # Count tokens in the conversation and adjust if necessary
        def num_tokens_from_messages(messages, tkmodel="cl100k_base"):
            encoding = tiktoken.get_encoding(tkmodel)
            num_tokens = 0
            for message in messages:
                num_tokens += 4  # every message follows <im_start>{role/name}\n{content}<im_end>\n
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

        # Pass conversation + context + prompt to OpenAI LLM
        completion = client_openai.chat.completions.create(
            model="gpt-4o",  # Replace with the appropriate OpenAI model
            messages=conversation,
            max_tokens=300,
            presence_penalty=0,
            temperature=0.5,
            top_p=0.9,
        )

        # Append result to conversation
        conversation.append({"role": "assistant", "content": completion.choices[0].message.content})
        response_text = completion.choices[0].message.content + "\n"
        print(f"Bot: {response_text}")

        if "goodbye" in user_input.lower():
            # Give the user a choice of what the bot will remember
            print("Bot: Do you want me to remember this conversation?")
            user_input_2 = input("You: ").strip()

            # If no, say goodbye and wait for next conversation
            if user_input_2.lower() == "no":
                print("Bot: Ok, I'll talk to you later then. Goodbye.")
                wait_for_input()
            else:
                # Remove system prompt
                del conversation[0]

                # Send conversation to OpenAI for summarization
                completion = client_openai.chat.completions.create(
                    model="gpt-4o",
                    messages=[
                        {"role": "system", "content": f"Briefly list the key points to remember about the user from this conversation that occurred at {current_time} on {date_string}: "},
                        {"role": "user", "content": str(conversation)},
                    ],
                    max_tokens=300,
                    temperature=0.3,
                    presence_penalty=0,
                    top_p=0.9,
                )
                summarization = completion.choices[0].message.content

                # Embed the summary using OpenAI
                response = client_openai.embeddings.create(
                    input=[summarization],
                    model="text-embedding-ada-002"  # Replace with the appropriate OpenAI model
                )
                summary_embedding = response.data[0].embedding

                # Upsert the embedding
                idv = "id" + str(time.time())
                index.upsert(vectors=[{"id": idv, 
                                       "values": summary_embedding, 
                                       "metadata": { 'text': summarization, 
                                                     'user_id': user_id}}],
                             namespace = namespace)
                time.sleep(1)
                index.describe_index_stats()

                # Return to wait for next conversation
                wait_for_input()

if __name__ == "__main__":
    wait_for_input()
