"""
Streamlit Frontend for Enterprise Document QA RAG
Provides an interactive chat interface, document upload workflow,
knowledge base statistics, and source citation display.
"""

import os
import requests
import streamlit as st

# Configuration
API_BASE_URL = os.getenv("BACKEND_API_URL", "http://localhost:8000")

# Page Configuration
st.set_page_config(
    page_title="Enterprise Document QA (RAG)",
    page_icon="📚",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom Styling
st.markdown("""
<style>
    .main-header {
        font-size: 2.2rem;
        font-weight: 700;
        color: #1E3A8A;
        margin-bottom: 0.2rem;
    }
    .sub-header {
        font-size: 1.05rem;
        color: #4B5563;
        margin-bottom: 1.5rem;
    }
    .source-card {
        background-color: #F8FAFC;
        border-left: 4px solid #3B82F6;
        padding: 10px 14px;
        margin-top: 8px;
        border-radius: 4px;
        font-size: 0.88rem;
    }
    .badge {
        display: inline-block;
        background: #DBEAFE;
        color: #1E40AF;
        padding: 2px 8px;
        border-radius: 9999px;
        font-size: 0.75rem;
        font-weight: 600;
        margin-right: 6px;
    }
</style>
""", unsafe_allow_html=True)


def check_backend_status():
    """Queries backend health and vector index statistics."""
    try:
        res = requests.get(f"{API_BASE_URL}/status", timeout=3)
        if res.status_code == 200:
            return res.json()
    except Exception:
        return None
    return None


def upload_pdf_file(uploaded_file):
    """Sends uploaded PDF to FastAPI /upload endpoint."""
    try:
        files = {"file": (uploaded_file.name, uploaded_file.getvalue(), "application/pdf")}
        response = requests.post(f"{API_BASE_URL}/upload", files=files, timeout=60)
        return response
    except Exception as e:
        return None


def query_rag_backend(query_text, k=4, groq_key=None, model=None):
    """Sends user question to FastAPI /ask endpoint."""
    payload = {
        "query": query_text,
        "k": k,
        "groq_api_key": groq_key if groq_key else None,
        "model": model
    }
    try:
        response = requests.post(f"{API_BASE_URL}/ask", json=payload, timeout=60)
        if response.status_code == 200:
            return response.json()
        else:
            return {"answer": f"API Error ({response.status_code}): {response.text}", "sources": []}
    except Exception as e:
        return {"answer": f"Connection error: Ensure backend is running at {API_BASE_URL}. Details: {str(e)}", "sources": []}


# --- SIDEBAR CONTROLS ---
with st.sidebar:
    st.image("https://cdn-icons-png.flaticon.com/512/8649/8649607.png", width=60)
    st.title("Enterprise Doc QA")
    st.caption("Retrieval-Augmented Generation (RAG) System")
    st.markdown("---")

    # API Key Configuration
    st.subheader("🔑 Credentials & Model")
    env_groq_key = os.getenv("GROQ_API_KEY", "")
    user_groq_key = st.text_input(
        "Groq API Key",
        value=env_groq_key,
        type="password",
        help="Free API key from console.groq.com. If set in .env, it auto-fills here."
    )
    selected_model = st.selectbox(
        "Groq LLM Model",
        options=["openai/gpt-oss-120b", "openai/gpt-oss-20b", "qwen/qwen3.8-27b", "llama-3.3-70b-versatile"],
        index=0,
        help="Select the LLM model to query via Groq"
    )

    st.markdown("---")

    # Document Upload Section
    st.subheader("📄 Upload Document")
    uploaded_file = st.file_uploader(
        "Select a PDF (Policy, Report, Manual)",
        type=["pdf"],
        help="Documents are chunked (500 chars, 50 overlap) and embedded into FAISS."
    )

    if uploaded_file is not None:
        if st.button("📥 Index Document", type="primary", use_container_width=True):
            with st.spinner(f"Ingesting & vectorizing '{uploaded_file.name}'..."):
                res = upload_pdf_file(uploaded_file)
                if res and res.status_code == 200:
                    data = res.json()
                    st.success(f"Indexed **{data['filename']}**!")
                    st.info(f"Pages: {data['raw_pages']} | Chunks added: {data['chunks_added']}")
                    st.rerun()
                elif res:
                    st.error(f"Error ({res.status_code}): {res.text}")
                else:
                    st.error(f"Could not reach backend at {API_BASE_URL}.")

    st.markdown("---")

    # Knowledge Base Status
    st.subheader("📊 Knowledge Base Status")
    status_info = check_backend_status()
    if status_info:
        st.success("🟢 Backend Connected")
        col1, col2 = st.columns(2)
        with col1:
            st.metric("Total Chunks", status_info.get("total_chunks", 0))
        with col2:
            st.metric("Indexed", "Yes" if status_info.get("is_indexed") else "No")
        
        st.caption(f"**Embedding Model:** `{status_info.get('embedding_model')}`")
        st.caption(f"**LLM:** `{status_info.get('llm_model')}`")

        if st.button("🗑️ Reset Knowledge Base", use_container_width=True):
            try:
                requests.post(f"{API_BASE_URL}/clear", timeout=5)
                st.session_state.messages = []
                st.warning("Knowledge base cleared.")
                st.rerun()
            except Exception as e:
                st.error(f"Failed to clear: {e}")
    else:
        st.error("🔴 Backend Offline")
        st.caption(f"Start the FastAPI backend at `{API_BASE_URL}` using:")
        st.code("python backend.py", language="bash")

    st.markdown("---")
    # Search parameters
    top_k = st.slider("Top-K Chunks to Retrieve", min_value=1, max_value=8, value=4)


# --- MAIN INTERFACE ---
st.markdown('<div class="main-header">📚 Enterprise Document QA System</div>', unsafe_allow_html=True)
st.markdown('<div class="sub-header">Zero-hallucination semantic search over private company documents with strict page citations.</div>', unsafe_allow_html=True)

# Architecture Banner
with st.expander("ℹ️ How this RAG Architecture Works (2 Phases)", expanded=False):
    col_a, col_b = st.columns(2)
    with col_a:
        st.markdown("""
        **Phase 1: Ingestion & Indexing**
        1. **Load:** `PyPDFLoader` extracts text and page numbers.
        2. **Chunk:** `RecursiveCharacterTextSplitter` creates 500-char segments with 50-char overlap.
        3. **Embed:** `all-MiniLM-L6-v2` converts chunks into 384-d dense vectors.
        4. **Index:** Vectors stored in memory/disk with `FAISS`.
        """)
    with col_b:
        st.markdown("""
        **Phase 2: Retrieval & Inference**
        1. **Query:** User question is embedded with `all-MiniLM-L6-v2`.
        2. **Search:** FAISS performs L2/cosine similarity search for Top-K chunks.
        3. **Augment:** Chunks injected into zero-hallucination prompt.
        4. **Inference:** `ChatGroq` (`llama-3.3-70b`, `temperature=0`) answers with exact page citations.
        """)

# Initialize Chat History
if "messages" not in st.session_state:
    st.session_state.messages = [
        {
            "role": "assistant",
            "content": "Hello! Upload a company policy, report, or handbook in the sidebar, and I'll answer questions strictly using facts from the document with verifiable page citations.",
            "sources": []
        }
    ]

# Display Existing Chat Messages
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])
        
        # Display Sources / Citations if available
        if msg.get("sources"):
            with st.expander(f"📑 View Source Citations ({len(msg['sources'])} retrieved chunks)"):
                for idx, src in enumerate(msg["sources"], 1):
                    st.markdown(f"""
                    <div class="source-card">
                        <span class="badge">Citation #{idx}</span>
                        <span class="badge">📄 {src.get('source')}</span>
                        <span class="badge">📌 Page {src.get('page')}</span>
                        <p style="margin-top: 6px; color: #374151; font-family: monospace;">"{src.get('snippet')}"</p>
                    </div>
                    """, unsafe_allow_html=True)

# User Chat Input
user_query = st.chat_input("Ask a question about your uploaded documents...")

if user_query:
    # 1. Append & Display user message
    st.session_state.messages.append({"role": "user", "content": user_query, "sources": []})
    with st.chat_message("user"):
        st.markdown(user_query)

    # 2. Query RAG backend
    with st.chat_message("assistant"):
        with st.spinner("Searching vector index & generating response..."):
            response_data = query_rag_backend(
                query_text=user_query,
                k=top_k,
                groq_key=user_groq_key,
                model=selected_model
            )
            answer_text = response_data.get("answer", "No answer received.")
            sources_list = response_data.get("sources", [])

            st.markdown(answer_text)

            # Show citations if any
            if sources_list:
                with st.expander(f"📑 View Source Citations ({len(sources_list)} retrieved chunks)"):
                    for idx, src in enumerate(sources_list, 1):
                        st.markdown(f"""
                        <div class="source-card">
                            <span class="badge">Citation #{idx}</span>
                            <span class="badge">📄 {src.get('source')}</span>
                            <span class="badge">📌 Page {src.get('page')}</span>
                            <p style="margin-top: 6px; color: #374151; font-family: monospace;">"{src.get('snippet')}"</p>
                        </div>
                        """, unsafe_allow_html=True)

    # 3. Save assistant message to session state
    st.session_state.messages.append({
        "role": "assistant",
        "content": answer_text,
        "sources": sources_list
    })
