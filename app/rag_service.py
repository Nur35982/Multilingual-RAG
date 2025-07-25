from langchain.chains import RetrievalQA
from typing import Dict, Any
from .vector_store import VectorStoreManager
from .memory_manager import EnhancedConversationMemory
from .config import settings
import httpx
import json

class OpenRouterMistral:
    def __init__(self, api_key: str):
        self.api_key = api_key
        self.base_url = "https://openrouter.ai/api/v1"
        
    def __call__(self, prompt: str, history: list = None) -> str:
        """Call the OpenRouter API with conversation history"""
        messages = []
        
        if history:
            messages.extend(history)
        
        messages.append({"role": "user", "content": prompt})
        
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
            #"HTTP-Referer": "https://github.com/your-repo",  
            "X-Title": "Bangla RAG System"                    
        }
        
        payload = {
            "model": "mistralai/mistral-7b-instruct",
            "messages": messages,
            "temperature": 0.7
        }
        
        with httpx.Client() as client:
            response = client.post(
                f"{self.base_url}/chat/completions",
                headers=headers,
                json=payload,
                timeout=30
            )
            response.raise_for_status()
            return response.json()["choices"][0]["message"]["content"]

class RAGService:
    def __init__(self):
        self.vector_store = VectorStoreManager()
        self.memory = EnhancedConversationMemory()
        self.llm = OpenRouterMistral(api_key=settings.OPENROUTER_API_KEY)
        
        # This will load the existing vector store, not create a new one
        self.db = self.vector_store.load_vector_store()

    def generate_answer(self, question: str) -> Dict[str, Any]:
        """Generate answer with context from vector store and conversation memory"""
        # Get relevant chunks from vector store (long-term memory)
        relevant_docs = self.db.similarity_search(question, k=3)
        context = "\n\n".join([doc.page_content for doc in relevant_docs])
        
        # Get conversation history (short-term memory)
        history = self.memory.get_recent_history()
        
        # Create enhanced prompt
        prompt = f"""
        প্রশ্নের উত্তর দাও নিচের প্রাসঙ্গিক তথ্য ব্যবহার করে। 
        যদি উত্তর জানা না থাকে, বলুন "আমি জানি না"।
        
        প্রাসঙ্গিক তথ্য:
        {context}
        
        প্রশ্ন: {question}
        উত্তর:
        """
        
        # Generate answer
        answer = self.llm(prompt, history)
        
        # Save to memory
        self.memory.save_context(
            {"question": question},
            {"answer": answer}
        )
        
        return {
            "answer": answer,
            "context": context,
            "sources": [doc.metadata for doc in relevant_docs]
        }