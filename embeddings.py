from typing import List, Any           # Used for type hints (helps readability and IDE support)
from langchain_text_splitters import RecursiveCharacterTextSplitter
from sentence_transformers import SentenceTransformer
import numpy as np
from data_loader import load_all_documents


# This class defines the complete embedding pipeline
class EmbeddingPipeline:

    # Constructor: runs automatically when an object is created
    def __init__(
        self,
        model_name: str = "all-MiniLM-L6-v2",  # embedding model name
        chunk_size: int = 1000,               # max characters per chunk
        chunk_overlap: int = 200               # overlapping characters between chunks
    ):
        # Store chunk configuration inside the object
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap

        # Load the embedding model once
        self.model = SentenceTransformer(model_name)

        # Confirmation message
        print(f"[INFO] Loaded embedding model: {model_name}")


    # Method to split documents into smaller text chunks
    def chunk_documents(self, documents: List[Any]) -> List[Any]:

        # Create a recursive text splitter
        splitter = RecursiveCharacterTextSplitter(
            chunk_size=self.chunk_size,          # chunk size in characters
            chunk_overlap=self.chunk_overlap,    # overlap to preserve context
            length_function=len,                 # measure chunk size using character length
            separators=["\n\n", "\n", " ", ""]   # splitting priority (paragraph → line → word → char)
        )
        chunks = splitter.split_documents(documents)
        print(f"[INFO] Split {len(documents)} documents into {len(chunks)} chunks.")
        return chunks
    

    # Method to generate embeddings from text chunks
    def embed_chunks(self, chunks: List[Any]) -> np.ndarray:

        # Extract only the text content from each chunk
        texts = [chunk.page_content for chunk in chunks]
        print(f"[INFO] Generating embeddings for {len(texts)} chunks...")

        # Convert text chunks into numerical embeddings
        embeddings = self.model.encode(texts, show_progress_bar=True)

        # Print shape of the embedding matrix (num_chunks × embedding_dimension)
        print(f"[INFO] Embeddings shape: {embeddings.shape}")

        # Return the embeddings as a NumPy array
        return embeddings


if __name__ == "__main__":
    docs = load_all_documents("data")

    # Create an instance of the embedding pipeline
    emb_pipe = EmbeddingPipeline()
    chunks = emb_pipe.chunk_documents(docs)
    embeddings = emb_pipe.embed_chunks(chunks)
    print("[INFO] Example embedding:", embeddings[0] if len(embeddings) > 0 else None)
