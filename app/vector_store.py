from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from typing import List
import os
from .config import settings

class VectorStoreManager:
    def __init__(self):
        self.embeddings = HuggingFaceEmbeddings(
            model_name=settings.EMBEDDING_MODEL
        )
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=300,  # Smaller chunks for Bengali
            chunk_overlap=50,
            separators=["\n\n", "\n", "।", "?", "!", "।", "\\.", "\\?", "\\!", " "],
            keep_separator=True
        )

    def create_chunks(self, texts: List[str]) -> List[str]:
        """Create chunks from text with awareness of Bengali sentence structure"""
        chunks = []
        for text in texts:
            chunks.extend(self.text_splitter.split_text(text))
        return chunks

    def create_vector_store(self, chunks: List[str]) -> FAISS:
        """Create and save FAISS vector store with metadata"""
        metadatas = [{"source": f"chunk-{i}"} for i in range(len(chunks))]
        vector_store = FAISS.from_texts(chunks, self.embeddings, metadatas=metadatas)
        vector_store.save_local(settings.VECTOR_STORE_PATH)
        return vector_store

    def load_vector_store(self) -> FAISS:
        """Load existing FAISS vector store"""
        if not os.path.exists(settings.VECTOR_STORE_PATH):
            raise FileNotFoundError("Vector store not found. Please create it first.")
        return FAISS.load_local(
            settings.VECTOR_STORE_PATH, 
            self.embeddings,
            allow_dangerous_deserialization=True
        )