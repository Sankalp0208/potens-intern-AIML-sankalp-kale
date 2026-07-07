from langchain_groq import ChatGroq

from app.api.schemas import AskResponse
from app.core.config import (
    GROQ_API_KEY,
    MODEL_NAME,
)

from app.prompts.qa_prompts import QA_PROMPT
from app.services.reteriver import RetrieverService
from app.services.citation import CitationService
from app.api.schemas import AskResponse

class RAGService:
    """
    Main Retrieval-Augmented Generation service.
    """

    def __init__(self):

        self.retriever = RetrieverService()

        self.llm = ChatGroq(
            groq_api_key=GROQ_API_KEY,
            model_name=MODEL_NAME,
            temperature=0,
        )

    def _build_context(self, documents):

        context_parts = []

        for doc in documents:

            context_parts.append(
                f"""
Source : {doc.metadata['source']}
Page   : {doc.metadata['page']}

{doc.page_content}
"""
            )

        return "\n\n-----------------------------\n\n".join(
            context_parts
        )

    def ask(
        self,
        question: str,
    ):

        retrieved_docs = self.retriever.retrieve(question)

        if not retrieved_docs:

            return {
                "answer": (
                    "I couldn't find enough information "
                    "in the provided documents."
                ),
                "citations": [],
                "confidence": 0.0,
            }

        context = self._build_context(retrieved_docs)

        chain = QA_PROMPT | self.llm

        response = chain.invoke(
    {
        "context": context,
        "question": question,
    }
)

        citations = CitationService.build_citations(
            retrieved_docs
        )

        confidence = max(
            0.0,
            1
            - min(
                doc.metadata.get("score", 1.0)
                for doc in retrieved_docs
            ),
        )

        return AskResponse(
            answer=response.content,
            confidence=round(confidence, 2),
            citations=citations
        )