from typing import List

from langchain_core.documents import Document

from app.core.config import TOP_K
from app.services.vectordb import VectorDB


class RetrieverService:
    """
    Retrieves the most relevant documents from ChromaDB.
    """

    def __init__(self):
        self.vectordb = VectorDB()

    def retrieve(
        self,
        query: str,
        k: int = TOP_K,
    ) -> List[Document]:
        """
        Retrieve top-k relevant documents.

        Similarity scores are attached as metadata.
        Filtering will be handled later in the RAG layer.
        """

        results = self.vectordb.similarity_search_with_score(
            query=query,
            k=k,
        )

        documents = []

        for document, score in results:
            document.metadata["score"] = float(score)
            documents.append(document)

        return documents