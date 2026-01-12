import os
import faiss
import numpy as np
import pickle
from typing import List, Any
from sentence_transformers import SentenceTransformer
from embeddings import EmbeddingPipeline

class FaissVectorStore:
    def __init__(self, persist_dir: str = "faiss_store", embedding_model: str = "all-MiniLM-L6-v2", chunk_size: int = 1000, chunk_overlap: int = 200):
        
        #  Directory where FAISS index and metadata will be stored
        self.persist_dir = persist_dir

        # Prevents error if folder already exists
        os.makedirs(self.persist_dir, exist_ok=True)

        # FAISS index (will be created later) now keep it empty
        self.index = None

        # Metadata list to store info about each text chunk. Example: source file, page number, chunk text
        self.metadata = []
        
        self.embedding_model = embedding_model
        self.model = SentenceTransformer(embedding_model)
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        print(f"[INFO] Loaded embedding model: {embedding_model}")

    
    # Build vector store from raw documents
    def build_from_documents(self, documents: List[Any]):
        print(f"[INFO] Building vector store from {len(documents)} raw documents...")
        emb_pipe = EmbeddingPipeline(model_name=self.embedding_model, chunk_size=self.chunk_size, chunk_overlap=self.chunk_overlap)
        chunks = emb_pipe.chunk_documents(documents)
        embeddings = emb_pipe.embed_chunks(chunks)

        # Store original chunk text as metadata
        metadatas = [{"text": chunk.page_content} for chunk in chunks]
        
        # [[0.12, -0.45, 0.78, ...],[0.33,  0.91, -0.22, ...]] embeddings - 2D array list
        # Add embeddings to FAISS index (FAISS does NOT accept Python lists,FAISS only supports float32 vectors)
        self.add_embeddings(np.array(embeddings).astype('float32'), metadatas)

        # Persist index and metadata
        self.save()
        print(f"[INFO] Vector store built and saved to {self.persist_dir}")
    

    # Add embeddings (vectors) to the FAISS index
    def add_embeddings(self, embeddings: np.ndarray, metadatas: List[Any] = None):

        # embeddings.shape = (number_of_vectors, number_of_dimensions) = (120, 384)
        # shape[0] = Number of vectors (chunks), shape[1] =	Dimensions per vector
        dim = embeddings.shape[1]

        # Create FAISS index if not exists
        if self.index is None:
            self.index = faiss.IndexFlatL2(dim)

        # Add vectors to index
        self.index.add(embeddings)

        # Store Metadata
        if metadatas:
            self.metadata.extend(metadatas)
        print(f"[INFO] Added {embeddings.shape[0]} vectors to Faiss index.")


    # Save FAISS index and metadata to Disk(presistant Memory)
    def save(self):
        faiss_path = os.path.join(self.persist_dir, "faiss.index")
        meta_path = os.path.join(self.persist_dir, "metadata.pkl")
        faiss.write_index(self.index, faiss_path)
        with open(meta_path, "wb") as f:
            pickle.dump(self.metadata, f)
        print(f"[INFO] Saved Faiss index and metadata to {self.persist_dir}")


    # Load FAISS index and metadata from disk
    def load(self):
        faiss_path = os.path.join(self.persist_dir, "faiss.index")
        meta_path = os.path.join(self.persist_dir, "metadata.pkl")
        self.index = faiss.read_index(faiss_path)
        with open(meta_path, "rb") as f:
            self.metadata = pickle.load(f)
        print(f"[INFO] Loaded Faiss index and metadata from {self.persist_dir}")


    # Search FAISS index using query embedding
    def search(self, query_embedding: np.ndarray, top_k: int = 5):
        D, I = self.index.search(query_embedding, top_k)
        results = []
        for idx, dist in zip(I[0], D[0]):
            meta = self.metadata[idx] if idx < len(self.metadata) else None
            results.append({"index": idx, "distance": dist, "metadata": meta})
        return results


    # Query vector store using plain text
    def query(self, query_text: str, top_k: int = 5):
        print(f"[INFO] Querying vector store for: '{query_text}'")
        query_emb = self.model.encode([query_text]).astype('float32')
        return self.search(query_emb, top_k=top_k)

# Example usage
if __name__ == "__main__":
    from data_loader import load_all_documents
    docs = load_all_documents("data")
    store = FaissVectorStore("faiss_store")
    store.build_from_documents(docs)
    store.load()
    print(store.query("What is deep learning?", top_k=3))