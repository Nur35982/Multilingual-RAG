# Bengali-English RAG System

![Python Version](https://img.shields.io/badge/python-3.8%2B-blue)
![FastAPI](https://img.shields.io/badge/FastAPI-0.68.0-green)

A Retrieval-Augmented Generation (RAG) system for answering questions from Bengali textbook content, supporting both English and Bengali queries. This system processes the HSC26 Bangla 1st Paper PDF, extracts text, creates a vector store, and generates context-aware answers using a multilingual language model.

## 🛠️ Setup Guide

### Prerequisites
- Python 3.8+
- Git
- PDF file (HSC26 Bangla 1st Paper)
- Optional: Tesseract OCR for scanned PDFs

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
cp /path/to/HSC26_Bangla_1st_Paper.pdf data/
```

### Running the System
```bash
uvicorn app.main:app --reload
```

Access the API at `http://localhost:8000`. Interactive documentation is available at `http://localhost:8000/docs`.

## 📚 Used Tools/Libraries
| Component | Technology | Purpose |
| -- | -- | -- |
| PDF Processing | pdfplumber | Bengali text extraction with layout awareness |
| Text Cleaning | Custom Regex | Diacritic and punctuation normalization |
| Embeddings | paraphrase-multilingual-MiniLM-L12-v2 | Multilingual vectorization |
| Vector Store | FAISS | Efficient similarity search |
| LLM | Mistral-7B (via OpenRouter) | Answer generation |
| Framework | FastAPI | REST API endpoints |
| Memory | LangChain ConversationBufferMemory | Short-term conversation history |

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
| -- | -- | -- | -- |
| `/query` | POST | Submit questions | `{"text": str, "language": str}` |
| `/clear_memory` | GET | Reset conversation | - |
| `/health` | GET | Service status | - |

## ❓ Technical Q&A

### 1. Text Extraction Method
**Method/Library Used**: pdfplumber is used for text extraction due to its robust handling of structured PDFs and ability to preserve layout information. The extract_text method in pdf_processor.py uses pdfplumber with settings (x_tolerance=2, y_tolerance=2) to improve Bengali character alignment.

**Why Chosen**: pdfplumber provides reliable text extraction for digital PDFs, capturing Bengali text with reasonable accuracy. It supports layout-aware extraction, helping filter out headers and footers. For scanned PDFs, Tesseract OCR integration is planned as a future enhancement.

**Formatting Challenges**:
- **Diacritic Misalignment**: Issues like `'িা' → 'িয়া'` were addressed using custom regex replacements in the `BengaliTextCleaner` class.
- **Non-Content Elements**: Headers, footers, and page numbers (e.g., "Page 42") were removed using regex patterns (e.g., `r'Page \d+|©.*'`).
- **Mixed-Language Text**: The PDF contains occasional English terms (e.g., proper nouns). A heuristic preserves valid English words while cleaning Bengali text.
- **Solution**: Enhanced cleaning includes expanded diacritic normalization, boilerplate removal (e.g., publisher info), and validation for empty or invalid PDFs.

### 2. Chunking Strategy
**Strategy Chosen**: A hybrid chunking approach combining sentence-based splitting (using Bengali punctuation like `।`, `?`, `!`) and a fixed 300-character limit with a 50-character overlap, implemented in `vector_store.py` using `RecursiveCharacterTextSplitter`.

**Why It Works for Semantic Retrieval**:
- **Sentence Awareness**: Splitting on Bengali punctuation ensures chunks respect sentence boundaries, preserving semantic coherence critical for morphologically rich languages like Bengali.
- **Fixed Size with Overlap**: The 300-character limit balances context size with retrieval efficiency, while the 50-character overlap prevents loss of meaning across chunk boundaries.
- **Dynamic Adaptation**: For long sentences (>150 characters), NLTK’s sentence tokenizer is used as a fallback, ensuring no single chunk is overly long.
- **Effectiveness**: This strategy captures complete ideas (e.g., descriptions of “সুপুরুষ” or “কল্যাণীর বয়স”) while keeping chunks small enough for efficient FAISS similarity search.

**Improvements**:
- Dynamic chunk sizes (200–400 characters) based on content density.
- Metadata enrichment (e.g., page numbers) to enhance retrieval context.

### 3. Embedding Model
**Model Used**: `paraphrase-multilingual-MiniLM-L12-v2` from HuggingFace, implemented in `vector_store.py`.

**Why Chosen**:
- **Multilingual Support**: Explicitly supports Bengali and English, enabling semantic equivalence (e.g., “সুপুরুষ” ↔ “handsome man”).
- **Efficiency**: Balances accuracy and performance with a 512-token context window, suitable for 300-character chunks.
- **Pre-trained Robustness**: Captures nuanced meanings in morphologically complex Bengali text without requiring fine-tuning.

**Semantic Capture**:
- The model generates dense vectors that encode semantic relationships, ensuring that similar meanings (e.g., “ভাগ্য দেবতা” ↔ “fortune’s god”) have close vector representations.
- It handles paraphrasing well, though some complex paraphrases may be missed without fine-tuning.

**Improvements**:
- Consider `paraphrase-multilingual-mpnet-base-v2` for better accuracy.
- Fine-tune on a Bengali textbook corpus to improve domain-specific embeddings.

### 4. Similarity Comparison
**Method Used**: Cosine similarity in a FAISS index, implemented in `vector_store.py`.

**Why Chosen**:
- **Cosine Similarity**: Measures directional similarity between query and chunk vectors, effective for text embeddings where magnitude is less important than orientation.
- **FAISS**: Optimized for fast, scalable similarity search, suitable for the 300-character chunk size and local storage (`data/faiss_index`).
- **Storage Setup**: FAISS stores vectors locally with metadata (e.g., chunk ID, page number), enabling persistent and efficient retrieval.

**Rationale**:
- Cosine similarity is robust for multilingual embeddings, capturing semantic closeness across languages.
- FAISS’s local storage reduces dependency on external services, ensuring reliability.

**Improvements**:
- Implement hybrid BM25 + semantic search to improve recall for keyword-heavy queries.
- Add confidence thresholding (e.g., reject matches <0.7 similarity) to filter irrelevant chunks.

### 5. Query Handling
**Ensuring Meaningful Comparison**:
- Queries are embedded using the same `paraphrase-multilingual-MiniLM-L12-v2` model as the chunks, ensuring consistent vector representations.
- The RAG pipeline retrieves the top 3 relevant chunks (`k=3`) using FAISS, which are combined into a context for the Mistral-7B LLM.
- Short-term memory (via `ConversationBufferMemory`) incorporates recent conversation history, enhancing context for follow-up questions.
- The prompt template in `rag_service.py` instructs the LLM to use provided context and return “আমি জানি না” for unanswerable queries.

**Handling Vague or Missing Context**:
- **Vague Queries**: If no chunks have high similarity (e.g., <0.7), the LLM may return “আমি জানি না” or suggest clarifying the query.
- **Missing Context**: The system returns the top 3 chunks regardless, but low relevance may lead to generic answers. Future improvements include synonym expansion and query reformulation.

**Improvements**:
- Implement Bengali synonym expansion using a resource like Bengali WordNet.
- Add query reformulation to handle vague inputs (e.g., rephrase “সুপুরুষ” to “আদর্শ পুরুষ”).

### 6. Result Relevance
**Current Relevance**:
- Results are relevant for the provided test cases (e.g., “শুম্ভুনাথ” for “সুপুরুষ”, “১৫ বছর” for “কল্যাণীর বয়স”), indicating effective retrieval and generation.
- The system occasionally misses paraphrased queries due to embedding limitations or chunk boundary issues.

**Improvement Pathways**:
- **Better Chunking**: Use dynamic chunk sizes (200–400 characters) and ensure sentence boundaries to improve context capture.
- **Better Embedding Model**: Adopt `paraphrase-multilingual-mpnet-base-v2` or fine-tune on Bengali textbooks for better semantic capture.
- **Larger Document Corpus**: Expand the corpus to include related texts (e.g., other HSC Bangla papers) to provide more context.
- **Hybrid Retrieval**: Combine BM25 for keyword matching with semantic search for improved recall.
- **OCR Support**: Add Tesseract OCR to handle scanned PDFs, ensuring text extraction from low-quality documents.
- **Error Handling**: Improve validation and logging to diagnose irrelevant results (e.g., log chunk quality, similarity scores).
