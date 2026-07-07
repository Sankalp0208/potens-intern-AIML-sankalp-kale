from typing import List

from langchain_core.documents import Document
from langchain_chroma import Chroma
from chromadb.config import Settings

from app.core.config import (
    CHROMA_DB_PATH,
    TOP_K,
)

from app.services.embeddings import get_embedding_model


class VectorDB:
    """
    Handles all interactions with ChromaDB.
    """

    def __init__(self):

        self.embedding_model = get_embedding_model()

        self.vectorstore = Chroma(
            persist_directory=str(CHROMA_DB_PATH),
            embedding_function=self.embedding_model,
    )

    # ---------------------------------------------------
    # Add Documents
    # ---------------------------------------------------
    def add_documents(self, documents):

        ids = [
            doc.metadata["chunk_id"]
            for doc in documents
    ]

        self.vectorstore.add_documents(
            documents=documents,
            ids=ids,
    )

    # Force persistence
        try:
            self.vectorstore.persist()
        except AttributeError:
        # Newer versions of Chroma persist automatically
            pass

    # ---------------------------------------------------
    # Similarity Search
    # ---------------------------------------------------
    def similarity_search(
        self,
        query: str,
        k: int = TOP_K,
    ) -> List[Document]:

        return self.vectorstore.similarity_search(
            query=query,
            k=k,
        )

    # ---------------------------------------------------
    # Similarity Search With Scores
    # ---------------------------------------------------
    def similarity_search_with_score(
        self,
        query: str,
        k: int = TOP_K,
    ):

        return self.vectorstore.similarity_search_with_score(
            query=query,
            k=k,
        )

    # ---------------------------------------------------
    # Retriever
    # ---------------------------------------------------
    def get_retriever(
        self,
        k: int = TOP_K,
    ):

        return self.vectorstore.as_retriever(
            search_kwargs={
                "k": k
            }
        )

    # ---------------------------------------------------
    # Reset Database
    # ---------------------------------------------------
    def reset_database(self) -> None:
        """
        Deletes all documents from the current collection.
        Useful before rebuilding the vector database.
        """

        collection = self.vectorstore._collection

        ids = collection.get()["ids"]

        if ids:
            collection.delete(ids=ids)

    # ---------------------------------------------------
    # Count Documents
    # ---------------------------------------------------
    def document_count(self) -> int:
        """
        Returns the number of indexed chunks.
        """

        return self.vectorstore._collection.count()