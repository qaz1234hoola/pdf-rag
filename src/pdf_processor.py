import pymupdf  # PyMuPDF
from typing import List, Dict, Any

class PDFProcessor:
    def __init__(self, chunk_size: int = 600, overlap: int = 100):
        """
        chunk_size: Target character count per chunk (~100-150 words).
        overlap: Character overlap between consecutive chunks to prevent context loss.
        """
        self.chunk_size = chunk_size
        self.overlap = overlap

    def extract_text_with_metadata(self, pdf_bytes: bytes, filename: str) -> List[Dict[str, Any]]:
        """Extracts text page-by-page while tracking page numbers."""
        doc = pymupdf.open(stream=pdf_bytes, filetype="pdf")
        pages_data = []

        for page_num in range(len(doc)):
            page = doc[page_num]
            text = page.get_text("text")
            
            # Keep non-empty pages
            if text.strip():
                pages_data.append({
                    "page": page_num + 1,  # 1-based indexing for page numbers
                    "text": text,
                    "filename": filename
                })
        return pages_data

    def chunk_document(self, pages_data: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Splits page text into overlapping chunks with precise citation metadata."""
        chunks = []
        global_chunk_idx = 0

        for page_info in pages_data:
            text = page_info["text"]
            page_num = page_info["page"]
            filename = page_info["filename"]

            start = 0
            text_length = len(text)

            while start < text_length:
                end = min(start + self.chunk_size, text_length)
                
                # Boundary optimization: avoid splitting mid-sentence if possible
                if end < text_length:
                    last_period = text.rfind('.', start, end)
                    last_newline = text.rfind('\n', start, end)
                    boundary = max(last_period, last_newline)
                    if boundary > start + (self.chunk_size // 2):
                        end = boundary + 1

                chunk_text = text[start:end].strip()

                if chunk_text:
                    global_chunk_idx += 1
                    chunks.append({
                        "id": f"{filename}_p{page_num}_c{global_chunk_idx}",
                        "text": chunk_text,
                        "metadata": {
                            "filename": filename,
                            "page": page_num,
                            "chunk_index": global_chunk_idx,
                            "citation": f"[Doc: {filename} | Page {page_num}, Chunk {global_chunk_idx}]"
                        }
                    })

                # Move sliding window forward by (chunk_size - overlap)
                start += (self.chunk_size - self.overlap)

        return chunks