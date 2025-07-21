from sqlalchemy import create_engine, Column, String
from sqlalchemy.orm import declarative_base, sessionmaker
import os

DB_PATH = os.getenv("DB_PATH", "sqlite:///keys.db")

# SQLAlchemy setup
engine = create_engine(DB_PATH, connect_args={"check_same_thread": False})
SessionLocal = sessionmaker(bind=engine)
Base = declarative_base()

# Model
class APIKey(Base):
    __tablename__ = "api_keys"
    key = Column(String, primary_key=True, unique=True)

# Initialize DB
def init_db():
    Base.metadata.create_all(bind=engine)

# Utility functions
def is_valid_key(key):
    session = SessionLocal()
    exists = session.query(APIKey).filter(APIKey.key == key).first() is not None
    session.close()
    return exists

def add_api_key(key):
    session = SessionLocal()
    if not session.query(APIKey).filter(APIKey.key == key).first():
        new_key = APIKey(key=key)
        session.add(new_key)
        session.commit()
    session.close()

