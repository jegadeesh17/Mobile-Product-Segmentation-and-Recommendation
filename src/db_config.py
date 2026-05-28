import os
from sqlalchemy import create_engine
from dotenv import load_dotenv

# Load environment variables from .env
load_dotenv()

def get_engine():
    db_user = os.getenv("DB_USER", "postgres")
    db_password = os.getenv("DB_PASSWORD")
    db_host = os.getenv("DB_HOST", "localhost")
    db_port = os.getenv("DB_PORT", "5432")
    db_name = os.getenv("DB_NAME", "product_segmentation")
    
    db_url = f"postgresql://{db_user}:{db_password}@{db_host}:{db_port}/{db_name}"
    return create_engine(db_url)
