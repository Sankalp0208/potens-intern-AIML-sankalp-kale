from langchain_groq import ChatGroq
from streamlit import context


from app.core.config import (
    GROQ_API_KEY,
    MODEL_NAME,
)

from app.prompts.contradiction_prompt import CONTRADICTION_PROMPT
from app.services.vectordb import VectorDB
from app.api.schemas import ContradictResponse


class ContradictionService:
    """
    Detect contradictions between two documents
    on a given topic.
    """

    def __init__(self):

        self.vector_db = VectorDB()

        self.llm = ChatGroq(
            groq_api_key=GROQ_API_KEY,
            model_name=MODEL_NAME,
            temperature=0,
        )

        self.structured_llm = self.llm.with_structured_output(
            ContradictResponse
        )
        
    def _retrieve_document_context(
        self,
        doc_id: str,
        topic: str,
    ) -> str:

        docs = self.vector_db.vectorstore.similarity_search(
            query=topic,
            k=5,
            filter={
                "doc_id": doc_id
            }
        )

        if not docs:
            return ""

        context = []

        for doc in docs:

           context.append(
            f"""
Source: {doc.metadata['source']}
Page: {doc.metadata['page']}

{doc.page_content}
"""
        )

        return "\n\n".join(context)

    def compare(
        self,
        doc1: str,
        doc2: str,
        topic: str,
    ):

        context1 = self._retrieve_document_context(
            doc1,
            topic,
        )

        context2 = self._retrieve_document_context(
            doc2,
            topic,
        )

        if not context1 or not context2:

            return ContradictResponse(
                conflict=False,
                reason="The requested topic was not found in one or both documents.",
                evidence="",
            )

        chain = CONTRADICTION_PROMPT | self.structured_llm

        response = chain.invoke(
        {
            "topic": topic,
            "doc1": context1,
            "doc2": context2,
        }
    )

        return response