# 📄 Company Policy & Document Q&A with Citations (RAG System)

A multilingual Retrieval-Augmented Generation (RAG) system built using **FastAPI**, **LangChain**, **ChromaDB**, **Groq LLM**, and **Streamlit**.

The system allows users to ask questions over a collection of documents and receive grounded answers with citations. It also supports document contradiction detection and multilingual queries.

---

# Features

- Document ingestion pipeline
- PDF parsing using PyMuPDF
- Recursive document chunking
- Sentence Transformer embeddings
- ChromaDB vector database
- Retrieval-Augmented Generation (RAG)
- Source citations with page and chunk references
- Document contradiction detection
- Multilingual query support (English, Hindi, Marathi)
- FastAPI backend
- Streamlit frontend
- Groq LLM integration

---

# Tech Stack

| Component | Technology |
|------------|------------|
| Backend | FastAPI |
| Frontend | Streamlit |
| Framework | LangChain |
| LLM | Groq (Llama 3.3 70B Versatile) |
| Embeddings | sentence-transformers/all-MiniLM-L6-v2 |
| Vector Database | ChromaDB |
| PDF Loader | PyMuPDF |
| Translation | deep-translator |
| Language Detection | langdetect |

---

# Project Structure

```
potens-intern-aiml-sankalp-kale/

│
├── app/
│   ├── api/
│   ├── core/
│   ├── prompts/
│   ├── services/
│   └── main.py
│
├── chroma_db/
├── documents/
├── scripts/
├── ui/
│
├── requirements.txt
├── README.md
└── .env.example
```

---

# Setup

## 1 Clone Repository

```bash
git clone <repository-url>

cd potens-intern-aiml-sankalp-kale
```

---

## 2 Create Virtual Environment

```bash
python -m venv .venv
```

Activate

Windows

```bash
.venv\Scripts\activate
```

Linux / Mac

```bash
source .venv/bin/activate
```

---

## 3 Install Dependencies

```bash
pip install -r requirements.txt
```

---

## 4 Create Environment File

Create

```
.env
```

Example

```env
GROQ_API_KEY=YOUR_GROQ_API_KEY

MODEL_NAME=llama-3.3-70b-versatile

EMBEDDING_MODEL=sentence-transformers/all-MiniLM-L6-v2

CHROMA_DB_PATH=./chroma_db

DOCUMENTS_PATH=./documents
```

---

# Build Vector Database

Run

```bash
python -m scripts.ingest
```

This will

- Load PDFs
- Split into chunks
- Generate embeddings
- Store vectors inside ChromaDB

---

# Run FastAPI

```bash
uvicorn app.main:app --reload
```

API Docs

```
http://127.0.0.1:8000/docs
```

---

# Run Streamlit

```bash
streamlit run ui/streamlit.py
```

---

# Docker

Build and run both services with Docker Compose:

```bash
docker compose up --build
```

Then open:

- FastAPI: http://127.0.0.1:8000/docs
- Streamlit: http://127.0.0.1:8501

If you only need the backend container:

```bash
docker compose up --build backend
```

---

# API Endpoints

## Health

```
GET /health
```

Returns backend status and indexed document count.

---

## Ask

```
POST /ask
```

Example

```json
{
    "question":"What is the leave policy?"
}
```

Returns

- Answer
- Confidence
- Citations

---

## Contradict

```
POST /contradict
```

Example

```json
{
    "doc1":"leave_policy",
    "doc2":"employee_handbook",
    "topic":"Annual Leave"
}
```

Returns

- Conflict status
- Reason
- Evidence

---

# Chunking Strategy

This project uses **RecursiveCharacterTextSplitter** from LangChain.

Configuration

- Chunk Size: **1000 characters**
- Chunk Overlap: **200 characters**

### Why this strategy?

- Preserves semantic meaning across chunk boundaries.
- Prevents important information from being split abruptly.
- Maintains enough context for accurate retrieval.
- Overlap helps avoid losing information spanning adjacent chunks.

Each chunk stores metadata including:

- Source PDF
- Page Number
- Chunk ID
- Document ID

These metadata are later used for citation generation.

---

# Citation Strategy

Every generated answer includes:

- Source document
- Page number
- Chunk ID
- Similarity score
- Supporting snippet

This ensures answers remain grounded in the uploaded documents.

---

# Multilingual Support

The system supports:

- English
- Hindi
- Marathi

Workflow:

1. Detect query language.
2. Translate to English (if required).
3. Perform retrieval.
4. Generate answer.
5. Translate answer back to the original language.

---

# Hallucination Prevention

To reduce hallucinations:

- Retrieval is performed before generation.
- Only retrieved context is provided to the LLM.
- Prompt instructs the model to answer **only** from the supplied context.
- If relevant information is unavailable, the system explicitly responds:

> "I couldn't find enough information in the provided documents."

---

# Future Improvements

- Metadata filtering for advanced retrieval
- Hybrid Search (BM25 + Dense Retrieval)
- Cross-Encoder Re-ranking
- User Authentication
- Docker Deployment
- Automatic document upload
- Conversation Memory
- Streaming Responses
- Production logging and monitoring

---

# Approach Summary (Approx. 150 Words)

This project implements a Retrieval-Augmented Generation (RAG) pipeline for answering questions over PDF documents with grounded citations. Documents are parsed using PyMuPDF, split into overlapping semantic chunks using LangChain's RecursiveCharacterTextSplitter, embedded using the sentence-transformers all-MiniLM-L6-v2 model, and indexed in ChromaDB. At query time, the system retrieves the most relevant chunks before sending them to Groq's Llama 3.3 model, ensuring responses are generated only from retrieved context. The application includes citation generation, multilingual support through translation, and a contradiction detection module that compares relevant sections of two documents. FastAPI provides REST endpoints while Streamlit offers an interactive interface. The design emphasizes modularity, transparency, and minimizing hallucinations through retrieval-first generation and explicit fallback responses when supporting evidence is unavailable.

---

# AI Use Log

| Tool | Purpose |
|------|---------|
| ChatGPT | System architecture, RAG pipeline design, FastAPI implementation, Streamlit UI development, debugging, prompt engineering, deployment guidance, README preparation, documentation, and code review. |
| Claude | Used for reviewing implementation ideas, validating prompt design, and comparing alternative architectural approaches. |
| GitHub Copilot | Assisted with boilerplate code completion, repetitive code generation, import suggestions, and minor syntax improvements during development. |

---

# Author

**Sankalp Kale**

AI/ML Engineering Internship Assignment

Potens IT Services and Consultancy Pvt. Ltd.

2026