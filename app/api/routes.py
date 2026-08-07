from fastapi import APIRouter, HTTPException

from app.api.schemas import (
    AskRequest,
    AskResponse,
    ContradictRequest,
    ContradictResponse,
    EvaluationRequest,
    EvaluationResponse,
    HealthResponse,
)

from app.services.rag import RAGService
from app.services.contradiction import ContradictionService
from app.services.evaluation import EvaluationService
from app.services.vectordb import VectorDB

router = APIRouter()

from functools import lru_cache

@lru_cache
def get_rag_service():
    return RAGService()

@lru_cache
def get_contradiction_service():
    return ContradictionService()

@lru_cache
def get_vector_db():
    return VectorDB()

@lru_cache
def get_evaluation_service():
    return EvaluationService()


# ======================================================
# Health Check
# ======================================================

@router.get(
    "/health",
    response_model=HealthResponse,
    tags=["Health"],
)
def health():

    return HealthResponse(
        status="healthy",
        documents_indexed=get_vector_db().document_count(),
    )


# ======================================================
# Ask Endpoint
# ======================================================

@router.post(
    "/ask",
    response_model=AskResponse,
    tags=["RAG"],
)
def ask(request: AskRequest):

    try:

        return get_rag_service().ask(request.question)

    except Exception as e:

        raise HTTPException(
            status_code=500,
            detail=str(e),
        )


# ======================================================
# Evaluation Endpoint
# ======================================================

@router.post(
    "/evaluate",
    response_model=EvaluationResponse,
    tags=["Evaluation"],
)
def evaluate(request: EvaluationRequest):

    try:
        result = get_evaluation_service().evaluate_dataset(
            dataset=[item.model_dump() for item in request.dataset],
            k=request.k,
        )

        return EvaluationResponse(**result)

    except Exception as e:

        raise HTTPException(
            status_code=500,
            detail=str(e),
        )


# ======================================================
# Contradiction Endpoint
# ======================================================

@router.post(
    "/contradict",
    response_model=ContradictResponse,
    tags=["Contradiction"],
)
def contradict(request: ContradictRequest):

    try:

        return get_contradiction_service().compare(
            doc1=request.doc1,
            doc2=request.doc2,
            topic=request.topic,
        )

    except Exception as e:

        raise HTTPException(
            status_code=500,
            detail=str(e),
        )