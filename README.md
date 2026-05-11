# RAG Application using LangChain, FAISS & Groq LLM

A modular Retrieval-Augmented Generation (RAG) application built using **LangChain**, **FAISS Vector Database**, **Sentence Transformers**, and **Groq LLMs**.

This project allows users to query PDF and CSV documents using natural language. The system retrieves the most relevant document chunks using semantic search and generates accurate responses using a Large Language Model (LLM).

---

# Features

- PDF and CSV document support
- Semantic search using embeddings
- FAISS vector database integration
- Recursive text chunking
- Modular programming architecture
- Groq LLM integration
- Fast and context-aware responses
- Persistent vector storage
- Streamlit-based interface

---

# Tech Stack

| Technology | Purpose |
|------------|---------|
| Python | Backend Development |
| LangChain | RAG Pipeline |
| FAISS | Vector Database |
| Sentence Transformers | Text Embeddings |
| HuggingFace | Embedding Model |
| Groq API | LLM Inference |
| Streamlit | Frontend Interface |

---

# Project Structure

```bash
RAG-Application/
│
├── data/                     # PDF and CSV files
├── faiss_store/              # Stored FAISS index & metadata
│
├── app.py                    # Streamlit application
├── rag_search.py             # Retrieval & response generation
├── vector_store.py           # FAISS vector store handling
├── data_loader.py            # Document loading utilities
│
├── requirements.txt
└── README.md
```

---

# How the Application Works

## 1. Document Loading

The application loads documents using LangChain loaders:

- `PyPDFLoader` for PDF files
- `CSVLoader` for CSV files

---

## 2. Text Chunking

Documents are split into smaller chunks using:

```python
RecursiveCharacterTextSplitter
```

### Configuration

```python
chunk_size = 1000
chunk_overlap = 200
```

This improves retrieval accuracy while preserving context.

---

## 3. Embedding Generation

Text chunks are converted into embeddings using the Sentence Transformer model:

```python
all-MiniLM-L6-v2
```

---

## 4. Vector Storage

Generated embeddings are stored inside a FAISS vector database for efficient similarity search.

Stored files:

```bash
faiss_store/
├── faiss.index
└── metadata.pkl
```

---

## 5. Retrieval Process

When a user asks a question:

1. Query is converted into embeddings
2. Similar chunks are retrieved from FAISS
3. Relevant context is collected
4. Context is sent to the LLM
5. Final response is generated

---

## 6. Response Generation

The project uses:

```python
llama-3.1-8b-instant
```

through the Groq API for fast and accurate response generation.

---

# Workflow

```text
User Query
     ↓
Embedding Generation
     ↓
FAISS Similarity Search
     ↓
Top Relevant Chunks Retrieved
     ↓
Context + Prompt Sent to LLM
     ↓
Generated Response
```

---

# Installation

## Clone the Repository

```bash
git clone https://github.com/sanjanaramgarhia/RAG-Application.git

cd RAG-Application
```

---

## Create Virtual Environment

### Windows

```bash
python -m venv venv

venv\Scripts\activate
```

### Linux / Mac

```bash
python3 -m venv venv

source venv/bin/activate
```

---

## Install Dependencies

```bash
pip install -r requirements.txt
```

---

# Environment Variables

Create a `.streamlit/secrets.toml` file and add:

```toml
GROQ_API_KEY = "your_groq_api_key"
```

---

# Run the Application

```bash
streamlit run app.py
```

---

# Example Queries

- What courses are available?
- What is the duration of the Python course?
- Who is the instructor?
- What are the course fees?
- Explain the course curriculum.

---

# Core Components

## FAISS Vector Store

Handles:

- Embedding generation
- Similarity search
- Vector indexing
- Persistent storage

---

## RAGSearch Class

Responsible for:

- Loading vector store
- Retrieving relevant chunks
- Prompt engineering
- Generating final responses

---

# Prompt Engineering

The application dynamically changes prompts based on query type.

### Metadata Queries

For queries related to:

- Fees
- Duration
- Instructor
- Eligibility

The system generates structured professional responses.

### General Queries

For general questions, concise contextual answers are generated.

---

# Why RAG?

Retrieval-Augmented Generation (RAG) improves:

- Accuracy
- Context-awareness
- Hallucination reduction
- Domain-specific answering

Instead of relying only on LLM knowledge, the model answers using your own documents.

---

# Future Improvements

- Conversational memory
- OCR support for scanned PDFs
- Source citations
- Hybrid search (BM25 + Vector Search)
- Pinecone / ChromaDB integration
- Docker deployment
- FastAPI integration
- Authentication system

---

# Requirements

```txt
langchain
faiss-cpu
sentence-transformers
streamlit
groq
pypdf
pandas
```

---

# Learning Outcomes

This project demonstrates practical implementation of:

- Retrieval-Augmented Generation (RAG)
- Vector Databases
- Semantic Search
- Prompt Engineering
- LLM Integration
- Modular Python Development

---

# Author

## Sanjana Ramgarhia

Computer Science Graduate passionate about:

- Artificial Intelligence
- Machine Learning
- Generative AI
- Data Analytics
- Backend Development

GitHub:
https://github.com/sanjanaramgarhia

---

# License

This project is open-source and available under the MIT License.
