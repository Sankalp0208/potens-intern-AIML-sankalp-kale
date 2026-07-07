from typing import List

from langchain_core.documents import Document


class CitationService:
    """
    Formats citations for retrieved documents.
    """

    @staticmethod
    def build_citations(documents: List[Document]) -> List[dict]:
        """
        Convert retrieved documents into structured citations.
        """

        citations = []

        for doc in documents:

            snippet = doc.page_content.strip()

            # Keep snippet readable
            if len(snippet) > 250:
                snippet = snippet[:250].rstrip() + "..."

            citations.append(
                {
                    "source": doc.metadata.get("source"),
                    "page": doc.metadata.get("page"),
                    "chunk_id": doc.metadata.get("chunk_id"),
                    "score": round(
                        float(doc.metadata.get("score", 0.0)),
                        4,
                    ),
                    "snippet": snippet,
                }
            )

        return citations