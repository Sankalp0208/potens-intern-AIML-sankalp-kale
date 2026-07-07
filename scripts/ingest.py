"""
One-time ingestion script.

Loads all PDFs from the documents folder,
splits them into chunks,
creates embeddings,
and stores them in ChromaDB.
"""

from app.services.loader import PDFLoader
from app.services.chunker import DocumentChunker
from app.services.vectordb import VectorDB


def main():

    print("=" * 60)
    print("Starting Document Ingestion...")
    print("=" * 60)

    # ----------------------------------
    # Load PDFs
    # ----------------------------------
    loader = PDFLoader()

    documents = loader.load_documents()

    print(f"Loaded {len(documents)} pages.")

    # ----------------------------------
    # Chunk Documents
    # ----------------------------------
    chunker = DocumentChunker()

    chunks = chunker.chunk_documents(documents)

    print(f"Created {len(chunks)} chunks.")

    # ----------------------------------
    # Initialize VectorDB
    # ----------------------------------
    vectordb = VectorDB()

    # Prevent duplicate indexing
    print("Resetting existing vector database...")

    vectordb.reset_database()

    # ----------------------------------
    # Index Documents
    # ----------------------------------
    print("Indexing chunks...")

    vectordb.add_documents(chunks)

    print(
        f"Indexed {vectordb.document_count()} chunks successfully."
    )

    print("=" * 60)
    print("Document Ingestion Completed Successfully!")
    print("=" * 60)


if __name__ == "__main__":
    main()