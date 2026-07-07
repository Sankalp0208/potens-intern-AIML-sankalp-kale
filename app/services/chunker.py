from typing import List

from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter

from app.core.config import (
    CHUNK_SIZE,
    CHUNK_OVERLAP,
)


class DocumentChunker:
    """
    Splits documents into smaller chunks while
    preserving metadata for citations.
    """

    def __init__(self):

        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=CHUNK_SIZE,
            chunk_overlap=CHUNK_OVERLAP,
            separators=[
                "\n\n",
                "\n",
                ". ",
                " ",
                "",
            ],
        )

    def chunk_documents(
        self,
        documents: List[Document],
    ) -> List[Document]:

        chunked_documents = []

        for document in documents:

            chunks = self.text_splitter.split_documents(
                [document]
            )

            for index, chunk in enumerate(chunks):

                chunk.metadata["chunk_id"] = (
                    f"{chunk.metadata['doc_id']}"
                    f"_page{chunk.metadata['page']}"
                    f"_chunk{index+1}"
                )

                chunk.metadata["chunk_index"] = index + 1

                chunked_documents.append(chunk)

        return chunked_documents