# Enterprise Document QA RAG 📚⚡

> Production-ready Retrieval-Augmented Generation (RAG) system for querying private enterprise documents (PDFs, policies, financial reports) with zero hallucinations and verifiable page-level source citations.

[![FastAPI](https://img.shields.io/badge/Backend-FastAPI-009688?style=flat&logo=fastapi)](https://fastapi.tiangolo.com/)
[![Streamlit](https://img.shields.io/badge/Frontend-Streamlit-FF4B4B?style=flat&logo=streamlit)](https://streamlit.io/)
[![LangChain](https://img.shields.io/badge/Orchestration-LangChain-1C3C3C?style=flat)](https://www.langchain.com/)
[![FAISS](https://img.shields.io/badge/VectorStore-FAISS-0080FF?style=flat)](https://github.com/facebookresearch/faiss)
[![Groq](https://img.shields.io/badge/Inference-Groq%20Llama%203.3-F55036?style=flat)](https://groq.com/)
[![Python](https://img.shields.io/badge/Python-3.10%2B-blue?style=flat&logo=python)](https://www.python.org/)

---

## 🎯 The Problem It Solves

Standard Large Language Models (LLMs) suffer from two fundamental enterprise limitations:
1. **No Access to Private Data:** Off-the-shelf models cannot read internal company handbooks, compliance policies, or proprietary contracts.
2. **Hallucination Risk:** When queried about unknown specifics, LLMs tend to invent convincing but inaccurate answers.

**Retrieval-Augmented Generation (RAG)** solves both challenges. Rather than relying on the LLM's parametric memory, our system extracts relevant text chunks from uploaded documents at query time and forces the model to answer strictly from the retrieved context while citing the exact page numbers.

---

## 🏛️ System Architecture

The pipeline operates in two distinct phases:

```
[Phase 1: Ingestion & Indexing]
Uploaded PDF ──► PyPDFLoader ──► Recursive Chunking ──► all-MiniLM-L6-v2 ──► FAISS Vector Store
                                (500 chars, 50 overlap) (384-dim Dense Vectors)   (In-Memory / Disk)

[Phase 2: Query & Generation]
User Query ──► all-MiniLM-L6-v2 ──► FAISS Top-K Search ──► Grounded Prompt ──► Groq LLM (Temp=0) ──► Answer + Citations
               (Query Embedding)    (Cosine Similarity)     (Strict Context)    (Llama 3.3 70B)
```

### Phase 1: Document Ingestion & Indexing (Upload Time)
1. **Document Loading:** `PyPDFLoader` extracts raw text and tracks source page numbers from uploaded PDFs.
2. **Recursive Text Chunking:** `RecursiveCharacterTextSplitter` splits long documents into smaller chunks (`chunk_size=500`, `chunk_overlap=50`). The 50-character overlap prevents sentences and context from breaking across chunk boundaries.
3. **Dense Vector Embeddings:** Each chunk is converted into a 384-dimensional dense semantic vector using `sentence-transformers/all-MiniLM-L6-v2` from Hugging Face.
4. **Vector Storage:** Embeddings and chunk metadata (document name, page number, raw text) are stored in a local `FAISS` index.

### Phase 2: Query & Answer Generation (Query Time)
1. **Query Embedding:** The user's question is embedded into the same 384-dimensional vector space using `all-MiniLM-L6-v2`.
2. **Semantic Vector Search:** FAISS executes nearest-neighbor similarity search (L2 / Cosine) and returns the Top-K ($k=4$) most relevant chunks.
3. **Prompt Augmentation:** Retrieved chunks and page metadata are injected into a strict system prompt:
   > *"Answer using only the provided context and cite page numbers. If the answer is not present, respond 'I don't have enough information in the uploaded documents.'"*
4. **LLM Inference:** `ChatGroq` (`llama-3.3-70b-versatile`, `temperature=0`) evaluates the augmented prompt with ultra-low latency inference (~300ms).
5. **Interactive Delivery:** FastAPI exposes REST endpoints (`/upload`, `/ask`), and Streamlit provides an enterprise chat UI displaying answers alongside collapsible citation cards with exact document names and page numbers.

---

## 📂 Repository Structure

```
enterprise-doc-qa-rag/
├── requirements.txt       # Project dependencies
├── rag_engine.py         # Core LangChain, FAISS, and Groq RAG engine
├── backend.py            # FastAPI REST endpoints (/upload, /ask, /status, /clear)
├── app.py                # Streamlit chat interface with citation inspector
├── .env.example          # Environment variables template
├── .gitignore            # Git exclusions
└── README.md             # Project documentation & interview guide
```

---

## 🚀 Quickstart Guide

### 1. Clone the Repository
```bash
git clone https://github.com/Prakhar221/enterprise-doc-qa-rag.git
cd enterprise-doc-qa-rag
```

### 2. Create and Activate Virtual Environment
```bash
# Windows
python -m venv venv
venv\Scripts\activate

# macOS / Linux
python3 -m venv venv
source venv/bin/activate
```

### 3. Install Dependencies
```bash
pip install -r requirements.txt
```

### 4. Configure Environment Variables
Copy `.env.example` to `.env` and add your free Groq API key from [Groq Console](https://console.groq.com):
```bash
cp .env.example .env
```
Edit `.env`:
```env
GROQ_API_KEY=gsk_your_groq_api_key_here
BACKEND_API_URL=http://localhost:8000
```

### 5. Launch the Application

In **Terminal 1**, start the FastAPI backend:
```bash
python backend.py
# Or: uvicorn backend:app --reload --port 8000
```
Swagger API docs will be available at: [http://localhost:8000/docs](http://localhost:8000/docs)

In **Terminal 2**, start the Streamlit UI:
```bash
streamlit run app.py
```
Open your browser at: [http://localhost:8501](http://localhost:8501)

---

## 📡 API Endpoints Reference

| Method | Endpoint | Description | Request Body / Params |
|---|---|---|---|
| `GET` | `/status` | Returns index health, chunk counts, and model info | None |
| `POST` | `/upload` | Ingests, chunks, embeds, and indexes a PDF document | `multipart/form-data` (`file`) |
| `POST` | `/ask` | Queries FAISS vector store & runs LLM generation | JSON: `{"query": "...", "k": 4}` |
| `POST` | `/clear` | Clears all stored vectors and resets index | None |

---

## 💼 Common Interview Questions & Answers

### 1. Why use RAG instead of fine-tuning an LLM?
> **Answer:** Fine-tuning modifies the model's internal weights to teach it style, tone, or specific formatting, but it is expensive, prone to catastrophic forgetting, and cannot provide real-time updates. RAG injects current, verifiable external data directly into the prompt context at inference time. This guarantees up-to-date knowledge without retraining and enables verifiable citations (linking answers directly back to specific document pages).

### 2. Why did you choose FAISS instead of managed vector databases like Pinecone or Weaviate?
> **Answer:** FAISS (Facebook AI Similarity Search) runs directly in-memory or on local disk with zero external network overhead, zero cloud infrastructure cost, and sub-millisecond similarity search over tens of thousands of vectors. For enterprise prototypes and internal secure microservices, FAISS avoids sending sensitive document embeddings to third-party SaaS vendors.

### 3. How do you prevent hallucinations in this pipeline?
> **Answer:** Hallucinations are mitigated through three guardrails:
> 1. **Temperature = 0:** Forces the LLM into deterministic decoding, minimizing creative extrapolation.
> 2. **Strict System Prompt Grounding:** The prompt explicitly instructs the LLM: *"If the answer cannot be found in the context, respond EXACTLY: 'I don't have enough information in the uploaded documents.' Never extrapolate or use external world knowledge."*
> 3. **Page-Level Source Citations:** Every retrieved chunk retains its original metadata (`source`, `page_number`), which is passed directly to the UI so users can verify every claim against the original PDF.

### 4. Why is chunk overlap necessary?
> **Answer:** If a document is split strictly at character boundaries without overlap, sentences, semantic units, or numerical tables can be severed in half across adjacent chunks. A 50-character overlap preserves semantic continuity, ensuring neither chunk loses critical context during dense vector embedding.

### 5. How would you scale this system for hundreds of thousands of documents?
> **Answer:**
> - Migrate from `faiss-cpu` flat index to an **IVF-PQ (Inverted File with Product Quantization)** or **HNSW (Hierarchical Navigable Small World)** index for sub-linear search complexity.
> - Decouple ingestion into an asynchronous background task queue (Celery + Redis / RabbitMQ) so large file uploads don't block the API.
> - Add a **Cross-Encoder Reranker** (e.g., `bge-reranker-large`) after Top-20 retrieval to rerank down to Top-4 before feeding to the LLM for higher precision.
