from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import Optional
from .rag_service import RAGService
from .pdf_processor import PDFProcessor
from .vector_store import VectorStoreManager
from .config import settings
import os

app = FastAPI(title="Bangla RAG System")

# CORS configuration
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

class Query(BaseModel):
    text: str
    language: Optional[str] = "bn"

# Initialize services at startup
@app.on_event("startup")
async def startup_event():
    # Check if vector store exists
    if not os.path.exists(settings.VECTOR_STORE_PATH):
        print("First-time setup: Processing PDF and creating vector store...")
        processor = PDFProcessor()
        vector_manager = VectorStoreManager()
        
        # Process PDF (one-time operation)
        sentences = processor.process_pdf(settings.PDF_PATH)
        chunks = vector_manager.create_chunks(sentences)
        
        # Create and save vector store (persistent)
        vector_manager.create_vector_store(chunks)
        print("Vector store created successfully.")
    else:
        print("Loading existing vector store...")
    
    # Initialize RAG service (will load existing vector store)
    app.state.rag_service = RAGService()

@app.post("/query")
async def handle_query(query: Query):
    try:
        result = app.state.rag_service.generate_answer(query.text)
        return {
            "answer": result["answer"],
            "context": result["context"],
            "sources": result["sources"]
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/clear_memory")
async def clear_memory():
    """Clear conversation memory"""
    app.state.rag_service.memory.clear()
    return {"status": "memory cleared"}

@app.get("/health")
async def health_check():
    return {"status": "healthy"}