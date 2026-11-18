"""
FastAPI REST API for Enterprise Document QA RAG
Exposes endpoints:
- POST /upload: Ingests and indexes uploaded PDF documents into FAISS
- POST /ask: Retrieves Top-K chunks and generates answer via Groq LLM
- GET  /status: Reports vector store and document indexing status
- POST /clear: Resets the knowledge base and FAISS index
"""

import os
import shutil
import tempfile
from typing import List, Optional
from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

from rag_engine import RAGEngine

app = FastAPI(
    title="Enterprise Document QA RAG API",
    description="Production-ready REST API for Enterprise Document Ingestion and Retrieval-Augmented Generation with LangChain, FAISS, and Groq.",
    version="1.0.0"
)

# Enable CORS for local dev and Streamlit integration
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Initialize global RAG Engine singleton
engine = RAGEngine()


# Pydantic Schemas
class QueryRequest(BaseModel):
    query: str = Field(..., description="The user's question about the uploaded documents", min_length=1)
    k: int = Field(default=4, description="Number of most relevant chunks to retrieve", ge=1, le=10)
    groq_api_key: Optional[str] = Field(default=None, description="Optional custom Groq API Key")


class SourceItem(BaseModel):
    source: str
    page: int
    snippet: str
    score: Optional[float] = None


class QueryResponse(BaseModel):
    answer: str
    sources: List[SourceItem]


class StatusResponse(BaseModel):
    status: str
    is_indexed: bool
    total_chunks: int
    indexed_files: List[str]
    embedding_model: str
    llm_model: str


@app.get("/", tags=["General"])
def root():
    return {
        "message": "Enterprise Document QA RAG Backend is running.",
        "docs_url": "/docs",
        "endpoints": ["/status", "/upload", "/ask", "/clear"]
    }


@app.get("/status", response_model=StatusResponse, tags=["Status"])
def get_status():
    """Retrieve the current state of the vector store and indexed files."""
    info = engine.get_status()
    return StatusResponse(
        status="ok",
        is_indexed=info["is_indexed"],
        total_chunks=info["total_chunks"],
        indexed_files=info["indexed_files"],
        embedding_model=info["embedding_model"],
        llm_model=info["llm_model"]
    )


@app.post("/upload", tags=["Ingestion"])
async def upload_document(file: UploadFile = File(...)):
    """
    Ingest, recursively chunk, embed, and store a PDF document into FAISS vector database.
    """
    if not file.filename.lower().endswith(".pdf"):
        raise HTTPException(
            status_code=400,
            detail="Unsupported file format. Please upload a PDF file."
        )

    temp_dir = tempfile.mkdtemp()
    temp_file_path = os.path.join(temp_dir, file.filename)

    try:
        # Save file to temporary disk location for PyPDFLoader
        with open(temp_file_path, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)

        result = engine.index_pdf(temp_file_path, file.filename)

        if result.get("status") == "error":
            raise HTTPException(status_code=422, detail=result.get("message"))

        return {
            "message": f"Successfully indexed '{file.filename}'",
            "filename": file.filename,
            "raw_pages": result["raw_pages"],
            "chunks_added": result["chunks_added"],
            "total_chunks_in_index": result["total_chunks_in_index"]
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to process document: {str(e)}")

    finally:
        # Clean up temporary upload file
        if os.path.exists(temp_dir):
            shutil.rmtree(temp_dir, ignore_errors=True)


@app.post("/ask", response_model=QueryResponse, tags=["Retrieval & Generation"])
def ask_question(payload: QueryRequest):
    """
    Query the knowledge base using dense vector search (FAISS) and generate
    an answer with Groq LLM (temperature=0) citing sources and page numbers.
    """
    try:
        result = engine.query(
            question=payload.query,
            k=payload.k,
            custom_api_key=payload.groq_api_key
        )
        return QueryResponse(
            answer=result["answer"],
            sources=[SourceItem(**s) for s in result.get("sources", [])]
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Query error: {str(e)}")


@app.post("/clear", tags=["Maintenance"])
def clear_knowledge_base():
    """Clear all vector embeddings and indexed documents from the system."""
    res = engine.clear()
    return {"message": "Knowledge base successfully cleared.", "status": res}


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("backend:app", host="0.0.0.0", port=8000, reload=True)
