from pydantic import BaseModel, Field
from typing import List


# =====================================================
# Citation
# =====================================================

class Citation(BaseModel):
    source: str
    page: int
    chunk_id: str
    score: float
    snippet: str


# =====================================================
# Ask Endpoint
# =====================================================

class AskRequest(BaseModel):
    question: str = Field(
        ...,
        min_length=1,
        description="Question to ask the document collection."
    )


class AskResponse(BaseModel):
    answer: str
    confidence: float
    citations: List[Citation]


# =====================================================
# Contradiction Endpoint
# =====================================================

class ContradictRequest(BaseModel):
    doc1: str = Field(
        ...,
        description="First document ID"
    )

    doc2: str = Field(
        ...,
        description="Second document ID"
    )

    topic: str = Field(
        ...,
        description="Topic to compare"
    )


class ContradictResponse(BaseModel):
    conflict: bool
    reason: str
    evidence: str


# =====================================================
# Health Check
# =====================================================

class HealthResponse(BaseModel):
    status: str
    documents_indexed: int