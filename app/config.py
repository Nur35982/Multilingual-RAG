import os
from dotenv import load_dotenv

load_dotenv()

class Settings:
    OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY")
    EMBEDDING_MODEL = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
    VECTOR_STORE_PATH = "data/faiss_index"
    PDF_PATH = "data/HSC26-Bangla1st-Paper.pdf"
    CHUNK_SIZE = 500
    CHUNK_OVERLAP = 50

settings = Settings()