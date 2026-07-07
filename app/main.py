from fastapi import FastAPI

from app.api.routes import router


app = FastAPI(
    title="Document Q&A with Citations API",
    description="A multilingual Retrieval-Augmented Generation (RAG) system with citations and contradiction detection.",
    version="1.0.0",
)

app.include_router(router)


@app.get("/")
def root():
    return {
        "message": "Welcome to the Document Q&A API",
        "docs": "/docs",
        "health": "/health",
    }