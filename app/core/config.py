from pathlib import Path
import os

from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# ==========================
# Base Directory
# ==========================
BASE_DIR = Path(__file__).resolve().parents[2]

# ==========================
# Project Paths
# ==========================
DOCUMENTS_PATH = BASE_DIR / os.getenv(
    "DOCUMENTS_PATH",
    "documents"
)

CHROMA_DB_PATH = BASE_DIR / os.getenv(
    "CHROMA_DB_PATH",
    "chroma_db"
)

# ==========================
# Groq Configuration
# ==========================
GROQ_API_KEY = os.getenv("GROQ_API_KEY")

if not GROQ_API_KEY:
    raise ValueError(
        "GROQ_API_KEY environment variable is not set."
    )

MODEL_NAME = os.getenv(
    "MODEL_NAME",
    "llama-3.3-70b-versatile"
)

# ==========================
# Embedding Model
# ==========================
EMBEDDING_MODEL = os.getenv(
    "EMBEDDING_MODEL",
    "sentence-transformers/all-MiniLM-L6-v2"
)

# ==========================
# Chunking Strategy
# ==========================
CHUNK_SIZE = 700
CHUNK_OVERLAP = 150

# ==========================
# Retrieval
# ==========================
TOP_K = 4
