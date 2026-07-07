from fastapi import APIRouter, HTTPException

from app.api.schemas import (
    AskRequest,
    AskResponse,
    ContradictRequest,
    ContradictResponse,
    HealthResponse,
)

from app.services.rag import RAGService
from app.services.contradiction import ContradictionService
from app.services.vectordb import VectorDB

router = APIRouter()

rag_service = RAGService()
contradiction_service = ContradictionService()
vector_db = VectorDB()


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
        documents_indexed=vector_db.document_count(),
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

        return rag_service.ask(request.question)

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

        return contradiction_service.compare(
            doc1=request.doc1,
            doc2=request.doc2,
            topic=request.topic,
        )

    except Exception as e:

        raise HTTPException(
            status_code=500,
            detail=str(e),
        )