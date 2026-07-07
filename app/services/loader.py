from pathlib import Path
from typing import List

import pymupdf as fitz  # PyMuPDF
from langchain_core.documents import Document

from app.core.config import DOCUMENTS_PATH


class PDFLoader:
    """
    Loads all PDF files from the documents directory
    and converts them into LangChain Document objects.
    """

    def __init__(self, documents_path: Path = DOCUMENTS_PATH):
        self.documents_path = documents_path

    def load_documents(self) -> List[Document]:
        """
        Load every PDF inside the documents folder.

        Returns:
            List[Document]
        """

        documents = []

        pdf_files = sorted(self.documents_path.glob("*.pdf"))

        if not pdf_files:
            raise FileNotFoundError(
                f"No PDF files found inside {self.documents_path}"
            )

        for pdf_file in pdf_files:
            documents.extend(self._load_single_pdf(pdf_file))

        return documents

    def _load_single_pdf(self, pdf_path: Path) -> List[Document]:
        """
        Read a single PDF page by page.
        """

        docs = []

        pdf = fitz.open(pdf_path)

        for page_number in range(len(pdf)):

            page = pdf.load_page(page_number)

            text = page.get_text("text").strip()

            # Skip blank pages
            if not text:
                continue

            docs.append(
                Document(
                    page_content=text,
                    metadata={
                        "source": pdf_path.name,
                        "doc_id": pdf_path.stem,
                        "page": page_number + 1,
                    },
                )
            )

        pdf.close()

        return docs