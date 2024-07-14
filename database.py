from sqlalchemy.orm import Session
from models import UserQueryData
from user_data import UserData  # Ensure you import the SQLAlchemy model
from models import Action, Category
from user_models import UserQueryData

def process_user_data(session: Session, user_data_dict: dict, user_add_knowledge, responses: dict):
    if user_data_dict and all(key in user_data_dict for key in ("user_id", "key", "value", "action", "category")):
        user_data = UserQueryData(**user_data_dict)  # Convert back to UserQueryData object
        if user_add_knowledge.action == Action.Create:
            session.add(user_data)
        elif user_add_knowledge.action == Action.Update:
            existing_data = session.query(UserQueryData).filter_by(user_id=user_data.user_id, key=user_data.key).first()
            if existing_data:
                existing_data.value = user_data.value
        elif user_add_knowledge.action == Action.Delete:
            session.query(UserQueryData).filter_by(user_id=user_data.user_id, key=user_data.key).delete()
        elif user_add_knowledge.action == Action.Retrieve:
            data = session.query(UserQueryData).filter_by(user_id=user_data.user_id, key=user_data.key).all()
            responses["user"] = [d.value for d in data] if data else "No data found"
        return user_data
    else:
        return None
    
def create_user_data(db: Session, user_data: UserQueryData):
    db_user_data = UserData(UserID=user_data.user_id, Key=user_data.key, Value=user_data.value)
    db.add(db_user_data)
    db.commit()
    db.refresh(db_user_data)
    return db_user_data

def update_user_data(db: Session, user_data: UserQueryData):
    db_user_data = db.query(UserData).filter(UserData.UserID == user_data.user_id, UserData.Key == user_data.key).first()
    if db_user_data:
        db_user_data.Value = user_data.value
        db.commit()
        db.refresh(db_user_data)
    return db_user_data

def delete_user_data(db: Session, user_data: UserQueryData):
    db_user_data = db.query(UserData).filter(UserData.UserID == user_data.user_id, UserData.Key == user_data.key).first()
    if db_user_data:
        db.delete(db_user_data)
        db.commit()
    return db_user_data

def retrieve_user_data(db: Session, user_data: UserQueryData):
    return db.query(UserData).filter(UserData.UserID == user_data.user_id, UserData.Key == user_data.key).first()

def search_user_data_by_user_id(session: Session, user_id: str):
    return session.query(UserQueryData).filter_by(user_id=user_id).all()