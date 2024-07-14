from fastapi import FastAPI, HTTPException, Depends, Request
from sqlalchemy.orm import Session
from pydantic import BaseModel
from models import AddKnowledge, UserQueryData, Action, Category
from database import create_user_data, update_user_data, delete_user_data, retrieve_user_data
from user_data import SessionLocal
from query_planner import parse_query, process_add_knowledge
from retrieval import get_retrieval_answer
import logging

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)
app = FastAPI()

# Dependency to get DB session
def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()

class UserQuery(BaseModel):
    query: str

app = FastAPI()

# Dependency to get DB session
def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()

class UserQuery(BaseModel):
    query: str

@app.exception_handler(Exception)
async def custom_exception_handler(request: Request, exc: Exception):
    logger.error(f"Exception occurred: {exc}")
    return HTTPException(status_code=500, detail=str(exc))

@app.post("/process_user_data/")
async def process_user_data(user_query: UserQuery, db: Session = Depends(get_db)):
    try:
        # Parse the user query to generate AddKnowledge models
        add_knowledge_parts = parse_query(user_query.query)
        responses = {}

        # Handle user-related information
        if "user" in add_knowledge_parts:
            user_data_dict = add_knowledge_parts["user"]
            user_data_dict['action'] = Action(user_data_dict['action']).value  # Ensure action is a string
            user_data_dict['category'] = Category(user_data_dict['category']).value  # Ensure category is a string
            user_add_knowledge = AddKnowledge(**user_data_dict)
            user_data_dict = process_add_knowledge(user_add_knowledge)  # Get the dictionary representation
            logger.info(f"Processed user data dict: {user_data_dict}")
            # Ensure all required fields are present
            if all(key in user_data_dict for key in ("user_id", "key", "value", "action")):
                user_data = UserQueryData(**user_data_dict)  # Convert back to UserQueryData object
                if user_add_knowledge.action == Action.Create:
                    create_user_data(db, user_data)
                elif user_add_knowledge.action == Action.Update:
                    update_user_data(db, user_data)
                elif user_add_knowledge.action == Action.Delete:
                    delete_user_data(db, user_data)
                elif user_add_knowledge.action == Action.Retrieve:
                    data = retrieve_user_data(db, user_data)
                    responses["user"] = data if data else "No data found"
            else:
                missing_keys = [key for key in ("user_id", "key", "value", "action") if key not in user_data_dict]
                logger.error(f"User data dictionary is missing required fields: {missing_keys}")
                raise HTTPException(status_code=400, detail=f"User data dictionary is missing required fields: {missing_keys}")
        
        # Handle book-related information
        if "book" in add_knowledge_parts:
            book_data_dict = add_knowledge_parts["book"]
            book_data_dict['action'] = Action(book_data_dict['action']).value  # Ensure action is a string
            book_data_dict['category'] = Category(book_data_dict['category']).value  # Ensure category is a string
            book_add_knowledge = AddKnowledge(**book_data_dict)
            answer = get_retrieval_answer(book_add_knowledge.knowledge)
            print(answer)
            responses["book"] = str(answer)

        logger.info(f"Responses: {responses}")
        return {"status": "success", "responses": responses}
    except Exception as e:
        logger.error(f"Error processing user data: {str(e)}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Internal Server Error: {str(e)}")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="127.0.0.1", port=8000, reload=True)