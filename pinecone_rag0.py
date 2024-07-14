import os
import time
import tiktoken
import requests
from datetime import datetime
from datetime import date
from octoai.client import OctoAI
from pinecone import Pinecone
from dotenv import load_dotenv
from openai import OpenAI

load_dotenv()
#keys and such
octoai_api_token = os.environ.get("OCTOAI_API_KEY")
openai_api_token = os.environ.get("OPENAI_API_KEY")
pc_key = os.environ.get("PINECONE_API_KEY")


print(octoai_api_token)
print(openai_api_token)
print(pc_key)

pc = Pinecone(api_key=pc_key)
client = OctoAI(api_key=octoai_api_token)

def wait_for_input():
    """Prompt user to initiate conversation."""
    while True:
        user_input = input("You: ").strip()
        if user_input.lower() in ["start", "hello", "hi"]:
            conversation_time()
        elif user_input.lower() == "quit":
            print("Goodbye!")
            break

def conversation_time():
    # Conversation setup that adds time/date info to prompt
    now = datetime.now()
    current_time = now.strftime("%H:%M:%S")
    today = date.today()
    date_string = str(today)
    system_message = {"role": "system", "content": f"It is {current_time} on {date_string} and you are a friendly robot that enjoys being helpful to your human friend."}
    user_greeting = {"role": "user", "content": "Hey bot."}
    bot_response = {"role": "assistant", "content": "What's up?"}
    conversation = [system_message, user_greeting, bot_response]
    
    user_id = 3
    while True:
        # User input section
        # print(conversation)
        user_input = input("You: ").strip()
        if user_input.lower() == "quit":
            print("Goodbye!")
            break

        user_index = pc.Index("user-data-openai-embedding")  # replace with your pinecone index name
        book_index = pc.Index("ah-test")

        # Embedding the prompt
        payload = {
            "input": user_input,
            "model": "thenlper/gte-large"
        }
        headers = {
            "Authorization": f"Bearer {octoai_api_token}",
            "Content-Type": "application/json",
        }
        res = requests.post("https://text.octoai.run/v1/embeddings", headers=headers, json=payload)
        print(res.json())
        xq = res.json()['data'][0]['embedding']

        # Vector context retrieval
        contexts = []
        user_res = user_index.query(vector=xq, 
                          top_k=1, 
                          filter={"user_id": user_id},
                          include_metadata=True)
        contexts = contexts + [x['metadata']['text'] for x in user_res['matches']]

        book_res = book_index.query(vector=xq,  
                                    top_k=1, 
                                    include_metadata=True)
        
        book_context = contexts + [x['metadata']['text'] for x in book_res['matches']]

        
        print(f"Retrieved {len(contexts)} contexts")
        time.sleep(0.5)

        # Merge context with prompt and merge into conversation
        prompt_end = "The following may or may not be relevant information from past conversations. If it is not relevant to this conversation, ignore it:\n\n"
        prompt = user_input + "\n\n" + prompt_end + "\n\n---\n\n".join(contexts[:1])
        conversation.append({"role": "user", "content": prompt})

        # Count tokens from conversation, move context window if max tokens exceeded
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

        # Pass conversation + context + prompt to LLM
        completion = client.text_gen.create_chat_completion(
            model="meta-llama-3-8b-instruct",  # The deployment name you chose when you deployed the ChatGPT or GPT-4 model.
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

                # Send conversation to LLM for summarization
                completion = client.text_gen.create_chat_completion(
                    messages=[
                        {"role": "system", "content": f"Briefly list the key points to remember about the user from this conversation that occurred at {current_time} on {date_string}: "},
                        {"role": "user", "content": str(conversation)},
                    ],
                    model="meta-llama-3-8b-instruct",
                    max_tokens=300,
                    temperature=0.3,
                    presence_penalty=0,
                    top_p=0.9,
                )
                summarization = completion.choices[0].message.content

                # Start embedding
                idv = "id" + str(time.time())
                payload = {
                    "input": summarization,
                    "model": "thenlper/gte-large"
                }
                headers = {
                    "Authorization": f"Bearer {octoai_api_token}",
                    "Content-Type": "application/json",
                }
                res = requests.post("https://text.octoai.run/v1/embeddings", headers=headers, json=payload)

                # Upsert the embedding
                user_index.upsert(vectors=[{"id": idv, "values": res.json()['data'][0]['embedding'], "metadata": {'text': summarization,'user_id': user_id}}])
                time.sleep(1)
                user_index.describe_index_stats()

                # Return to wait for next conversation
                wait_for_input()

if __name__ == "__main__":
    wait_for_input()