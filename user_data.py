from sqlalchemy import create_engine, Column, Integer, String, DateTime, ForeignKey, Text, func
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker, relationship
from datetime import datetime

Base = declarative_base()

class User(Base):
    __tablename__ = 'users'
    UserID = Column(Integer, primary_key=True, autoincrement=True)
    Username = Column(String, unique=True, nullable=False)
    Email = Column(String, unique=True, nullable=False)
    Password = Column(String, nullable=False)
    SignupDate = Column(DateTime, default=datetime.utcnow)
    LastLogin = Column(DateTime)

class UserProfile(Base):
    __tablename__ = 'userprofiles'
    UserProfileID = Column(Integer, primary_key=True, autoincrement=True)
    UserID = Column(Integer, ForeignKey('users.UserID'), nullable=False)
    FirstName = Column(String)
    LastName = Column(String)
    BirthDate = Column(DateTime)
    GoalDescription = Column(Text)
    user = relationship("User", back_populates="profile")

User.profile = relationship("UserProfile", back_populates="user", uselist=False)

class Book(Base):
    __tablename__ = 'books'
    BookID = Column(Integer, primary_key=True, autoincrement=True)
    Title = Column(String, nullable=False)
    Author = Column(String, nullable=False)
    Summary = Column(Text)
    PublicationYear = Column(Integer)

class UserBookInteraction(Base):
    __tablename__ = 'userbookinteractions'
    InteractionID = Column(Integer, primary_key=True, autoincrement=True)
    UserID = Column(Integer, ForeignKey('users.UserID'), nullable=False)
    BookID = Column(Integer, ForeignKey('books.BookID'), nullable=False)
    LastAccessed = Column(DateTime, default=datetime.utcnow)
    Duration = Column(Integer)
    Notes = Column(Text)

class Goal(Base):
    __tablename__ = 'goals'
    GoalID = Column(Integer, primary_key=True, autoincrement=True)
    UserID = Column(Integer, ForeignKey('users.UserID'), nullable=False)
    Description = Column(Text, nullable=False)
    CreationDate = Column(DateTime, default=datetime.utcnow)
    DueDate = Column(DateTime)
    Status = Column(String)

class UserAction(Base):
    __tablename__ = 'useractions'
    ActionID = Column(Integer, primary_key=True, autoincrement=True)
    UserID = Column(Integer, ForeignKey('users.UserID'), nullable=False)
    ActionType = Column(String, nullable=False)
    Timestamp = Column(DateTime, default=datetime.utcnow)

class UserData(Base):
    __tablename__ = 'userdata'
    DataID = Column(Integer, primary_key=True, autoincrement=True)
    UserID = Column(Integer, ForeignKey('users.UserID'), nullable=False)
    Key = Column(String, nullable=False)
    Value = Column(Text, nullable=False)
    Timestamp = Column(DateTime, default=func.now())

class UserSession(Base):
    __tablename__ = 'usersessions'
    SessionID = Column(Integer, primary_key=True, autoincrement=True)
    UserID = Column(Integer, ForeignKey('users.UserID'), nullable=False)
    StartTime = Column(DateTime, default=datetime.utcnow)
    EndTime = Column(DateTime)
    DeviceType = Column(String)

class PersonalityTest(Base):
    __tablename__ = 'personalitytests'
    TestID = Column(Integer, primary_key=True, autoincrement=True)
    Name = Column(String, nullable=False)
    Description = Column(Text)

class UserPersonalityResult(Base):
    __tablename__ = 'userpersonalityresults'
    ResultID = Column(Integer, primary_key=True, autoincrement=True)
    UserID = Column(Integer, ForeignKey('users.UserID'), nullable=False)
    TestID = Column(Integer, ForeignKey('personalitytests.TestID'), nullable=False)
    Result = Column(String)

# Create engine and session
DATABASE_URL = "sqlite:///./test.db"  # Update to your database URL
engine = create_engine(DATABASE_URL)
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)
Base.metadata.create_all(bind=engine)