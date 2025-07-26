# Bengali-English RAG System

[![Python Version](https://img.shields.io/badge/python-3.8%2B-blue)](https://www.python.org/)
[![FastAPI](https://img.shields.io/badge/Framework-FastAPI-green)](https://fastapi.tiangolo.com/)

A Retrieval-Augmented Generation system for answering questions from Bengali textbook content, supporting both English and Bengali queries.

## Table of Contents
- [Setup Guide](#setup-guide)
- [Used Tools/Libraries](#used-toolslibraries)
- [Sample Queries](#sample-queries)
- [API Documentation](#api-documentation)
- [Technical Q&A](#technical-qa)

## 🛠️ Setup Guide

### Prerequisites
- Python 3.8+
- Git
- PDF file (HSC26 Bangla 1st Paper)

### Installation
```bash
# Clone repository
git clone https://github.com/Nur35982/Multilingual-RAG.git
cd bangla-rag

# Create virtual environment
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Place your PDF in data/ directory
cp /path/to/HSC26_Bangla_1st_paper.pdf data/
```

### Running the System
```bash
uvicorn app.main:app --reload
```

## 📚 Used Tools/Libraries

| Component | Technology | Purpose |
|----------|------------|---------|
| PDF Processing | PyMuPDF  | Bengali text extraction |
| Text Cleaning | Custom Regex | Diacritic normalization |
| Embeddings | paraphrase-multilingual-MiniLM-L12-v2 | Multilingual vectorization |
| Vector Store | FAISS | Efficient similarity search |
| LLM | Mistral-7B (via OpenRouter) | Answer generation |
| Framework | FastAPI | REST API endpoints |

## 💬 Sample Queries

### Bengali Example
**Query:**
```json
{"text": "অনুপমের ভাষায় সুপুরুষ কাকে বলা হয়েছে?", "language": "bn"}
```

**Response:**
```json
{
  "answer": "শুম্ভুনাথ",
  "context": "অনুপম তার রচনায় শুম্ভুনাথকে আদর্শ সুপুরুষ হিসেবে বর্ণনা করেছেন...",
  "sources": [{"page": 42}]
}
```

### English Example
**Query:**
```json
{"text": "What was Kalyanis age during marriage?", "language": "en"}
```

**Response:**
```json
{
  "answer": "15 years",
  "context": "The text mentions Kalyanis age during marriage as 15 years...",
  "sources": [{"page": 87}]
}
```

## 📡 API Documentation

### Endpoints
| Endpoint | Method | Description | Request Body |
|----------|--------|-------------|--------------|
| /query | POST | Submit questions | {"text":str, "language":str} |
| /clear_memory | GET | Reset conversation | - |
| /health | GET | Service status | - |

Access interactive docs at: http://localhost:8000/docs




## ❓ Technical Q&A

### 1. Text Extraction Method
- **Library**: PyMuPDF 
- **Why Chosen**: Superior Bengali character preservation compared to alternatives
- **Challenges Faced**:
  - Diacritic marks misalignment (e.g., "িা" → "িয়া")
  - Column formatting issues in textbook layout
  - Required custom Unicode normalization

### 2. Chunking Strategy
- **Approach**: 400-character sliding windows with 100-character overlap
- **Effectiveness**:
  - Preserves complete Bengali sentences
  - Overlap maintains context across chunks
  - Optimal for semantic search in morphologically rich languages

### 3. Embedding Model
- **Model**: paraphrase-multilingual-MiniLM-L12-v2
- **Selection Criteria**:
  - Explicit Bengali language support
  - Balanced performance/accuracy tradeoff
  - 512 token context window fits our chunks
- **Semantic Capture**:
  - Creates similar vectors for:
    - "সুপুরুষ" ↔ "handsome man"
    - "ভাগ্য দেবতা" ↔ "fortune's god"

### 4. Similarity Comparison
- **Method**: Cosine similarity in FAISS index
- **Why Chosen**:
  - Directional similarity works better than Euclidean for text
  - Optimized for our chunk size (400 chars)
- **Storage**: Local FAISS files with metadata

### 5. Query Handling
- **Vague Query Mitigation**:
  - Automatic Bengali synonym expansion
  - Confidence thresholding (rejects <0.7 similarity)
  - Follow-up question suggestions
- **Missing Context Behavior**:
  - Returns "Could you clarify..." for vague queries
  - Provides 3 most relevant document sections
  - Scores context relevance (0-1) in debug mode

### 6. Result Improvement
**Current Limitations**:
- Sometimes misses paraphrased queries

**Improvement Pathways**:
1. **Chunking**:
   - Incorporate sentence boundaries
   - Dynamic chunk sizes (200-600 chars)
2. **Embeddings**:
   - paraphrase-multilingual-mpnet-base-v2
   - Fine-tuning on Bengali textbook corpus
3. **Retrieval**:
   - Hybrid BM25 + semantic search
   - Query expansion with Bengali WordNet
4. **PDF Quality**:
   - OCR for scanned pages
   - Manual correction of extraction errors
