import chromadb
from chromadb.utils import embedding_functions
from typing import List, Dict, Any

class VectorStoreManager:
    def __init__(self, collection_name: str = "pdf_rag_store", persist_dir: str = "./chroma_db"):
        """
        Initializes ChromaDB persistent client and sets up local embedding model.
        """
        self.client = chromadb.PersistentClient(path=persist_dir)
        
        # Local, lightweight embedding model (~90MB) running on CPU
        self.embedding_fn = embedding_functions.SentenceTransformerEmbeddingFunction(
            model_name="sentence-transformers/all-MiniLM-L6-v2"
        )
        
        # Get or create the ChromaDB collection using Cosine distance space
        self.collection = self.client.get_or_create_collection(
            name=collection_name,
            embedding_function=self.embedding_fn,
            metadata={"hnsw:space": "cosine"}
        )

    def add_chunks(self, chunks: List[Dict[str, Any]]) -> None:
        """Stores text chunks, generates their embeddings, and links source metadata."""
        if not chunks:
            return

        ids = [c["id"] for c in chunks]
        documents = [c["text"] for c in chunks]
        metadatas = [c["metadata"] for c in chunks]

        # Upsert inserts new vectors or updates existing ones by ID
        self.collection.upsert(
            ids=ids,
            documents=documents,
            metadatas=metadatas
        )

    def search_similar(self, query: str, top_k: int = 4) -> List[Dict[str, Any]]:
        """Retrieves top_k context chunks matching the query using Cosine Similarity."""
        results = self.collection.query(
            query_texts=[query],
            n_results=top_k,
            include=["documents", "metadatas", "distances"]
        )

        retrieved = []
        if results and results["documents"] and results["documents"][0]:
            docs = results["documents"][0]
            metas = results["metadatas"][0]
            dists = results["distances"][0]

            for doc, meta, dist in zip(docs, metas, dists):
                retrieved.append({
                    "text": doc,
                    "metadata": meta,
                    "score": round(1 - dist, 4)  # Convert cosine distance to similarity score (0 to 1)
                })

        return retrieved

    def clear_database(self) -> None:
        """Resets the vector collection when uploading a new document."""
        self.client.delete_collection(self.collection.name)
        self.collection = self.client.get_or_create_collection(
            name=self.collection.name,
            embedding_function=self.embedding_fn,
            metadata={"hnsw:space": "cosine"}
        )