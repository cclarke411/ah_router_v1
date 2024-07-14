from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from user_models import Base

DATABASE_URL = "sqlite:///user_db.db"  # Update the path if needed

engine = create_engine(DATABASE_URL)
Session = sessionmaker(bind=engine)

def init_db():
    Base.metadata.create_all(engine)
    print("Database initialized.")

if __name__ == "__main__":
    init_db()
