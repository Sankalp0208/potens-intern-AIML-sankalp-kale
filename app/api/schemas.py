from pydantic import BaseModel, Field
from typing import List, Optional


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


# =====================================================
# Evaluation Endpoint
# =====================================================

class EvaluationItem(BaseModel):
    question: str = Field(..., min_length=1)
    expected_sources: List[str] = Field(default_factory=list)
    expected_answer: Optional[str] = None


class EvaluationRequest(BaseModel):
    dataset: List[EvaluationItem]
    k: int = Field(default=4, ge=1)


class EvaluationBreakdownItem(BaseModel):
    question: str
    precision_at_k: float
    recall_at_k: float
    mrr_at_k: float
    answer_match: float
    retrieved_sources: List[str]


class EvaluationResponse(BaseModel):
    num_items: int
    avg_precision_at_k: float
    avg_recall_at_k: float
    avg_mrr_at_k: float
    avg_answer_match: float
    breakdown: List[EvaluationBreakdownItem]