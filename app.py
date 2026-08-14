import streamlit as st
import os
from dotenv import load_dotenv

from src.pdf_processor import PDFProcessor
from src.vector_store import VectorStoreManager
from src.rag_engine import RAGEngine

# Load environment variables silently
load_dotenv()

# Page configuration
st.set_page_config(
    page_title="Document Intelligence Assistant",
    page_icon="⚡",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for enterprise UI styling
st.markdown("""
<style>
    /* Main container padding */
    .block-container {
        padding-top: 1.8rem;
        padding-bottom: 2rem;
    }
    
    /* Header branding */
    .main-header {
        font-size: 2.2rem;
        font-weight: 700;
        background: linear-gradient(90deg, #6366F1 0%, #A855F7 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin-bottom: 0.2rem;
    }
    
    /* Sidebar styling */
    [data-testid="stSidebar"] {
        background-color: #0F172A;
        border-right: 1px solid #1E293B;
    }
    
    /* Document stats container */
    .stat-card {
        background-color: #1E293B;
        border: 1px solid #334155;
        border-radius: 8px;
        padding: 12px 16px;
        margin-top: 10px;
        margin-bottom: 15px;
    }
    
    /* Source Evidence Cards */
    .source-card {
        background-color: #1E293B;
        border-left: 4px solid #6366F1;
        padding: 12px 16px;
        border-radius: 4px;
        margin-bottom: 12px;
    }
    .source-badge {
        display: inline-block;
        background-color: #334155;
        color: #38BDF8;
        font-size: 0.78rem;
        font-weight: 600;
        padding: 2px 8px;
        border-radius: 12px;
        margin-right: 8px;
    }
    .score-badge {
        display: inline-block;
        background-color: #059669;
        color: #FFFFFF;
        font-size: 0.78rem;
        font-weight: 600;
        padding: 2px 8px;
        border-radius: 12px;
    }
</style>
""", unsafe_allow_html=True)

# Initialize persistent session state variables
if "vector_store" not in st.session_state:
    st.session_state.vector_store = VectorStoreManager()
if "pdf_processor" not in st.session_state:
    st.session_state.pdf_processor = PDFProcessor()
if "messages" not in st.session_state:
    st.session_state.messages = []
if "indexed_file" not in st.session_state:
    st.session_state.indexed_file = None
if "doc_stats" not in st.session_state:
    st.session_state.doc_stats = {"pages": 0, "chunks": 0}

# App Header
st.markdown('<div class="main-header">⚡ Document Intelligence Assistant</div>', unsafe_allow_html=True)
st.caption("Strictly grounded Q&A engine with verified page and chunk level citations.")
st.divider()

# Sidebar Setup
with st.sidebar:
    st.markdown("### 📁 Document Workspace")
    st.caption("Upload a PDF document to index its contents into the local vector store.")
    
    uploaded_file = st.file_uploader("Upload PDF Document", type=["pdf"], label_visibility="collapsed")
    
    # Process PDF when a new file is uploaded
    if uploaded_file and (st.session_state.indexed_file != uploaded_file.name):
        with st.spinner("Processing & indexing document..."):
            st.session_state.vector_store.clear_database()
            
            bytes_data = uploaded_file.read()
            pages_data = st.session_state.pdf_processor.extract_text_with_metadata(bytes_data, uploaded_file.name)
            chunks = st.session_state.pdf_processor.chunk_document(pages_data)
            
            st.session_state.vector_store.add_chunks(chunks)
            st.session_state.indexed_file = uploaded_file.name
            st.session_state.doc_stats = {"pages": len(pages_data), "chunks": len(chunks)}
            st.rerun()

    # Active Document Status Card
    if st.session_state.indexed_file:
        st.markdown(f"""
        <div class="stat-card">
            <div style="font-size: 0.85rem; color: #94A3B8;">ACTIVE DOCUMENT</div>
            <div style="font-weight: 600; font-size: 0.95rem; color: #F8FAFC; text-overflow: ellipsis; overflow: hidden; whitespace: nowrap;">
                📄 {st.session_state.indexed_file}
            </div>
            <div style="margin-top: 8px; font-size: 0.8rem; color: #CBD5E1;">
                <b>Pages:</b> {st.session_state.doc_stats['pages']} | <b>Vector Chunks:</b> {st.session_state.doc_stats['chunks']}
            </div>
        </div>
        """, unsafe_allow_html=True)
        
        if st.button("🗑️ Reset Workspace", use_container_width=True, type="secondary"):
            st.session_state.vector_store.clear_database()
            st.session_state.messages = []
            st.session_state.indexed_file = None
            st.session_state.doc_stats = {"pages": 0, "chunks": 0}
            st.rerun()
    else:
        st.info("Upload a PDF above to activate the system.")

# Empty State Welcome Message
if not st.session_state.messages:
    if not st.session_state.indexed_file:
        st.info("👈 Please upload a PDF document from the sidebar to start asking questions.")
    else:
        st.success(f"**Document Ready:** Ask any question about `{st.session_state.indexed_file}` in the input box below.")

# Render Chat History
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])
        
        # Render expandable evidence card if sources exist
        if "sources" in msg and msg["sources"]:
            with st.expander("🔍 View Retrieved Grounding Context"):
                for idx, src in enumerate(msg["sources"], 1):
                    meta = src["metadata"]
                    st.markdown(f"""
                    <div class="source-card">
                        <div>
                            <span class="source-badge">Page {meta['page']}</span>
                            <span class="source-badge">Chunk {meta['chunk_index']}</span>
                            <span class="score-badge">Match: {int(src['score'] * 100)}%</span>
                        </div>
                        <div style="margin-top: 8px; font-size: 0.88rem; color: #E2E8F0; font-style: italic;">
                            "{src['text']}"
                        </div>
                    </div>
                    """, unsafe_allow_html=True)

# Chat Input & Pipeline Trigger
if query := st.chat_input("Ask a question about your uploaded document..."):
    # Silent API Key check
    if not os.getenv("GROQ_API_KEY"):
        st.error("Missing GROQ_API_KEY in .env file. Please configure your key to proceed.")
        st.stop()
        
    if not st.session_state.indexed_file:
        st.warning("Please upload a PDF document first.")
        st.stop()

    # Append & render user message
    st.session_state.messages.append({"role": "user", "content": query})
    with st.chat_message("user"):
        st.markdown(query)

    # Process assistant response
    with st.chat_message("assistant"):
        with st.spinner("Analyzing document vectors..."):
            retrieved_chunks = st.session_state.vector_store.search_similar(query, top_k=4)
            
            rag = RAGEngine()
            answer, sources = rag.generate_grounded_response(query, retrieved_chunks)

            st.markdown(answer)
            
            # Display source drawer if answer was grounded successfully
            if sources and answer != "Information not found in the provided document.":
                with st.expander("🔍 View Retrieved Grounding Context"):
                    for idx, src in enumerate(sources, 1):
                        meta = src["metadata"]
                        st.markdown(f"""
                        <div class="source-card">
                            <div>
                                <span class="source-badge">Page {meta['page']}</span>
                                <span class="source-badge">Chunk {meta['chunk_index']}</span>
                                <span class="score-badge">Match: {int(src['score'] * 100)}%</span>
                            </div>
                            <div style="margin-top: 8px; font-size: 0.88rem; color: #E2E8F0; font-style: italic;">
                                "{src['text']}"
                            </div>
                        </div>
                        """, unsafe_allow_html=True)

            # Store assistant response in session
            st.session_state.messages.append({
                "role": "assistant",
                "content": answer,
                "sources": sources if answer != "Information not found in the provided document." else []
            })