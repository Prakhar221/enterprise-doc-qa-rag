"""
Core LangChain + FAISS + Embedding RAG Engine
Handles document ingestion, chunking, dense vector embeddings,
FAISS vector storage, semantic search, and zero-hallucination answer generation.
"""

import os
import shutil
from typing import List, Dict, Any, Optional
from dotenv import load_dotenv

# LangChain components
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PyPDFLoader
try:
    from langchain_huggingface import HuggingFaceEmbeddings
except ImportError:
    from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

load_dotenv()

INDEX_DIR = os.path.join(os.path.dirname(__file__), "faiss_index")
EMBEDDING_MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"
DEFAULT_GROQ_MODEL = os.getenv("GROQ_MODEL", "openai/gpt-oss-120b")

# Strict enterprise system prompt designed to prevent hallucinations
SYSTEM_PROMPT = """You are an enterprise AI assistant answering questions strictly based on uploaded company documents.

STRICT INSTRUCTIONS:
1. Answer the question using ONLY the factual information in the context below.
2. If the answer cannot be determined from the context, respond EXACTLY:
   "I don't have enough information in the uploaded documents."
3. Do NOT extrapolate, speculate, or draw from general world knowledge beyond the provided context.
4. For every claim or key fact, cite the document name and page number using format: [Doc: <filename>, Page: <page>]
5. Keep your answer professional, concise, and well-structured.

Context:
{context}

Question:
{question}

Answer:"""


class RAGEngine:
    def __init__(self, index_dir: str = INDEX_DIR):
        self.index_dir = index_dir
        self.embedding_model = None
        self.vector_store: Optional[FAISS] = None
        self.indexed_files: List[str] = []
        self.total_chunks: int = 0
        
        # Initialize text splitter as specified: chunk_size=500, chunk_overlap=50
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=500,
            chunk_overlap=50,
            separators=["\n\n", "\n", " ", ""]
        )
        
        self._load_embeddings()
        self._load_existing_index()

    def _load_embeddings(self):
        """Initializes the 384-dimensional dense vector embedding model."""
        if self.embedding_model is None:
            self.embedding_model = HuggingFaceEmbeddings(
                model_name=EMBEDDING_MODEL_NAME,
                model_kwargs={"device": "cpu"},
                encode_kwargs={"normalize_embeddings": True}
            )

    def _load_existing_index(self):
        """Loads a persisted FAISS index if present on disk."""
        if os.path.exists(self.index_dir) and os.path.exists(os.path.join(self.index_dir, "index.faiss")):
            try:
                self.vector_store = FAISS.load_local(
                    folder_path=self.index_dir,
                    embeddings=self.embedding_model,
                    allow_dangerous_deserialization=True
                )
                # Count approximate docs in index
                self.total_chunks = self.vector_store.index.ntotal
            except Exception as e:
                print(f"[Warning] Failed to load local FAISS index: {e}")
                self.vector_store = None

    def save_index(self):
        """Persists FAISS vector store to disk."""
        if self.vector_store is not None:
            os.makedirs(self.index_dir, exist_ok=True)
            self.vector_store.save_local(self.index_dir)

    def index_pdf(self, file_path: str, original_filename: str) -> Dict[str, Any]:
        """
        Phase 1: Ingestion & Indexing
        1. PyPDFLoader extracts raw text + page numbers
        2. RecursiveCharacterTextSplitter splits into 500-char chunks (overlap 50)
        3. all-MiniLM-L6-v2 converts chunks into 384-d dense vectors
        4. FAISS stores vectors and metadata (filename + page number)
        """
        self._load_embeddings()

        # Step 1: Load PDF documents with PyPDFLoader
        loader = PyPDFLoader(file_path)
        raw_docs = loader.load()

        if not raw_docs:
            return {"status": "error", "message": "No text could be extracted from this PDF."}

        # Step 2: Recursive text chunking
        chunks = self.text_splitter.split_documents(raw_docs)

        # Standardize metadata: file name & 1-based page number
        for chunk in chunks:
            chunk.metadata["source"] = original_filename
            # PyPDFLoader provides 0-indexed 'page'; convert to 1-indexed for citations
            raw_page = chunk.metadata.get("page", 0)
            chunk.metadata["page_number"] = raw_page + 1

        # Step 3 & 4: Embed & Add to FAISS Vector Store
        if self.vector_store is None:
            self.vector_store = FAISS.from_documents(chunks, self.embedding_model)
        else:
            self.vector_store.add_documents(chunks)

        self.save_index()
        self.total_chunks = self.vector_store.index.ntotal
        if original_filename not in self.indexed_files:
            self.indexed_files.append(original_filename)

        return {
            "status": "success",
            "filename": original_filename,
            "raw_pages": len(raw_docs),
            "chunks_added": len(chunks),
            "total_chunks_in_index": self.total_chunks
        }

    def query(self, question: str, k: int = 4, custom_api_key: Optional[str] = None, model_name: Optional[str] = None) -> Dict[str, Any]:
        """
        Phase 2: Query & Answer Generation
        1. Embed user question using all-MiniLM-L6-v2
        2. Semantic similarity search (Top-K) via FAISS
        3. Strict system prompt augmentation with retrieved chunks
        4. Groq LLM inference with temperature=0
        """
        if self.vector_store is None or self.total_chunks == 0:
            return {
                "answer": "No documents have been indexed yet. Please upload and index a PDF file first.",
                "sources": []
            }

        api_key = custom_api_key or os.getenv("GROQ_API_KEY")
        if not api_key:
            return {
                "answer": "Error: GROQ_API_KEY is not set. Please provide your Groq API key in the environment or sidebar.",
                "sources": []
            }

        # 1 & 2: Similarity search (Top-K chunks)
        docs_and_scores = self.vector_store.similarity_search_with_score(question, k=k)

        if not docs_and_scores:
            return {
                "answer": "I don't have enough information in the uploaded documents.",
                "sources": []
            }

        # Format context and collect metadata for citations
        context_blocks = []
        sources = []

        for doc, score in docs_and_scores:
            source_file = doc.metadata.get("source", "Unknown Document")
            page_num = doc.metadata.get("page_number", doc.metadata.get("page", 0) + 1)
            content = doc.page_content.strip()

            context_blocks.append(f"[Document: {source_file} | Page: {page_num}]\n{content}")
            sources.append({
                "source": source_file,
                "page": page_num,
                "snippet": content,
                "score": float(round(score, 4))
            })

        formatted_context = "\n\n---\n\n".join(context_blocks)

        chosen_model = model_name or os.getenv("GROQ_MODEL", DEFAULT_GROQ_MODEL)

        # 3 & 4: Prompt Augmentation & Groq LLM Inference (temperature=0)
        try:
            llm = ChatGroq(
                model_name=chosen_model,
                temperature=0.0,
                groq_api_key=api_key,
                max_retries=2
            )

            prompt_template = ChatPromptTemplate.from_template(SYSTEM_PROMPT)
            chain = prompt_template | llm | StrOutputParser()

            answer = chain.invoke({
                "context": formatted_context,
                "question": question
            })

            return {
                "answer": answer.strip(),
                "sources": sources
            }
        except Exception as e:
            return {
                "answer": f"Error communicating with LLM: {str(e)}",
                "sources": sources
            }

    def clear(self):
        """Resets the FAISS index and local storage."""
        self.vector_store = None
        self.indexed_files = []
        self.total_chunks = 0
        if os.path.exists(self.index_dir):
            shutil.rmtree(self.index_dir, ignore_errors=True)
        return {"status": "cleared", "total_chunks": 0}

    def get_status(self) -> Dict[str, Any]:
        """Returns knowledge base stats."""
        return {
            "is_indexed": self.vector_store is not None and self.total_chunks > 0,
            "total_chunks": self.total_chunks,
            "indexed_files": self.indexed_files,
            "embedding_model": EMBEDDING_MODEL_NAME,
            "llm_model": DEFAULT_GROQ_MODEL
        }
