# ⚡ Document Intelligence Assistant (PDF RAG Engine)

> An enterprise-grade, zero-hallucination Retrieval-Augmented Generation (RAG) assistant that delivers precise, grounded answers from uploaded PDFs with precise page and chunk-level superscript citations.

---

## 🚀 Live Demo

[![Streamlit App](https://img.shields.io/badge/Streamlit-Live_Demo-FF4B4B?style=for-the-badge&logo=streamlit&logoColor=white)](https://your-live-app-link.streamlit.app)

👉 **[Click here to access the Live Application](https://p-d-f-r-a-g.streamlit.app/)**

---

## 🎬 Application Demo

![PDF RAG Assistant Demo](assets/demo.gif)

*Note: Replace `assets/demo.gif` with your actual GIF or screen recording demonstrating PDF ingestion, document chunking, and grounded Q&A with superscript citations.*

---

## 📌 Features

* 📄 **Strict PDF Grounding:** Completely eliminates AI hallucinations by enforcing strict reliance on uploaded document context.
* <sup>[Page X, Chunk Y]</sup> **Superscript Web-Style Citations:** Inline references rendered as small, unobtrusive HTML superscripts at the end of cited claims.
* ⚡ **High-Speed Vector Search:** Powered by ChromaDB and Groq's high-throughput LLaMA 3.3 70B inference engine.
* 🛡️ **Robust Error Handling:** Comprehensive exception guards for corrupt PDFs, password protection, rate limits, and missing keys.
* 🎨 **Interactive Interface:** Clean Streamlit UI with source attribution metadata pills and document workspace controls.

---

## 🏗️ System Architecture & Technical Choices

```text
[ User Query ]
       │
       ▼
[ Streamlit UI (app.py) ]
       │
       ▼
[ PyMuPDF Document Ingestion ] ──► Extracted Text + Page Metadata
       │
       ▼
[ Overlapping Text Chunking ] ──► Chunks (Size: 600, Overlap: 100)
       │
       ▼
[ Local Vector Store (ChromaDB) ] ──► Dense Vector Embeddings (all-MiniLM-L6-v2)
       │
       ▼
[ Semantic Similarity Retrieval ] ──► Top K Grounding Context Blocks
       │
       ▼
[ Strict Prompt Injection ] ──► LLaMA 3.3 70B via Groq API
       │
       ▼
[ Grounded Answer + HTML Superscript Citations ]
```



### 1. Document Extraction & Metadata Retention (`src/pdf_processor.py`)
* **PyMuPDF (`pymupdf`)** was selected over heavy frameworks (such as LangChain or LlamaIndex) for low memory overhead, lightning-fast execution, and native page-level metadata tracking.
* Each extracted text block maintains its original `filename`, `page_number`, and sequential `chunk_index`.

### 2. Chunking Strategy
* **Chunk Size:** 600 characters
* **Overlap:** 100 characters
* **Rationale:** A 600-character window isolates coherent ideas while maintaining strong embedding density. A 100-character overlap prevents information loss across chunk boundaries (e.g., splitting a sentence or technical metric across two chunks).

### 3. Vector Embeddings & Similarity Search (`src/vector_store.py`)
* **ChromaDB:** Lightweight, in-memory vector database requiring zero external cloud setup or complex index management.
* **`all-MiniLM-L6-v2`:** SentenceTransformer model generating 384-dimensional dense vectors. Optimized for high speed and accuracy in semantic matching.

### 4. Grounded Prompt Engineering & Citation Engine (`src/rag_engine.py`)
* **Strict Fallback Rules:** If the retrieved context lacks explicit evidence to answer the query, the LLM is instruction-tuned to respond exclusively with: `"Information not found in the provided document."`
* **Inline Source Citations:** Context blocks are injected with explicit document metadata tags (`[Page X, Chunk Y]`). The LLM attaches these exact bracketed references directly following statements derived from the source text.
---

## 🛠️ Local Setup & Installation Guide

Follow these steps to run the application locally on your machine or inside GitHub Codespaces.

### Prerequisites

* **Python 3.10+** installed on your system.
* A **Groq API Key** (Get one for free at [console.groq.com](https://console.groq.com)).

### 1. Clone the Repository

```bash
git clone [https://github.com/qaz1234hoola/pdf-rag.git](https://github.com/qaz1234hoola/pdf-rag.git)
cd pdf-rag
```

### 2. Set Up Virtual Environment

```bash
# Create virtual environment
python3 -m venv .venv

# Activate virtual environment
# On Linux/macOS/Codespaces:
source .venv/bin/activate

# On Windows:
# .venv\Scripts\activate
```

### 3. Install Dependencies

```bash
pip install --upgrade pip
pip install -r requirements.txt
```

### 4. Configure Environment Variables

Create a `.env` file in the root directory:

```bash
touch .env
```

Add your Groq API key inside `.env`:

```env
GROQ_API_KEY=your_actual_groq_api_key_here
```

### 5. Launch the Application

```bash
streamlit run app.py
```

## 💻 Project Structure

```text
pdf-rag/
├── app.py                  # Main Streamlit UI & Chat Application
├── requirements.txt        # Python Dependencies
├── .env                    # Environment Variables (Ignored by Git)
├── .gitignore              # Excluded Git Files
├── README.md               # System Documentation
└── src/
    ├── pdf_processor.py    # PDF Extraction, Validation & Chunking
    ├── vector_store.py     # ChromaDB Vector Storage & Similarity Retrieval
    └── rag_engine.py       # Grounded Prompt Pipeline & Groq Integration
```

## 🛡️ License

This project is open-source and available under the [MIT License](LICENSE).
