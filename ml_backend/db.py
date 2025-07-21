from sqlalchemy import create_engine, Column, String
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker
import os

# Database path setup
DB_PATH = os.getenv("DB_PATH", "keys.db")
DB_URL = f"sqlite:///{DB_PATH}"

# Create the engine and base
engine = create_engine(DB_URL, echo=False, connect_args={"check_same_thread": False})
Base = declarative_base()

# Define the APIKey model
class APIKey(Base):
    __tablename__ = 'api_keys'
    username = Column(String, primary_key=True)
    api_key = Column(String, unique=True)

# Session factory
Session = sessionmaker(bind=engine)

# Initialize the database
def init_db():
    Base.metadata.create_all(engine)

