
from fastapi import FastAPI, HTTPException, Depends
from pydantic import BaseModel, Field
from typing import Optional, List, Dict
from openai import OpenAI
import instructor
from datetime import datetime
import openai

app = FastAPI()
openai.api_key = 'your-api-key'
client = instructor.from_openai(OpenAI())
