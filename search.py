import os
import streamlit as st
from vector_store import FaissVectorStore
from langchain_groq import ChatGroq


class RAGSearch:
    def __init__(
        self,
        persist_dir: str = "faiss_store",
        embedding_model: str = "all-MiniLM-L6-v2",
        llm_model: str = "llama-3.1-8b-instant"
    ):
        # ---- Load API key from Streamlit secrets ----
        if "GROQ_API_KEY" not in st.secrets:
            raise RuntimeError("GROQ_API_KEY not found in Streamlit secrets")

        GROQ_API_KEY = st.secrets["GROQ_API_KEY"]

        # ---- Vector store ----
        self.vectorstore = FaissVectorStore(persist_dir, embedding_model)

        faiss_path = os.path.join(persist_dir, "faiss.index")
        meta_path = os.path.join(persist_dir, "metadata.pkl")

        if not (os.path.exists(faiss_path) and os.path.exists(meta_path)):
            from data_loader import load_all_documents
            docs = load_all_documents("data")
            self.vectorstore.build_from_documents(docs)
        else:
            self.vectorstore.load()

        # ---- Groq LLM ----
        self.llm = ChatGroq(
            groq_api_key=GROQ_API_KEY,
            model_name=llm_model
        )

        print(f"[INFO] Groq LLM initialized: {llm_model}")

    def search_and_summarize(self, query: str, top_k: int = 3) -> str:
        results = self.vectorstore.query(query, top_k=top_k)
        texts = [r["metadata"].get("text", "") for r in results if r.get("metadata")]
    
        context = "\n\n".join(texts)
    
        if not context:
            return "No relevant information found."
    
        short_keywords = [
            "fee", "fees", "cost", "price",
            "duration", "length",
            "eligibility",
            "instructor",
            "timing"
        ]
        is_short_query = any(k in query.lower() for k in short_keywords)
        prompt = None
    
        if is_short_query:
            prompt = f"""You are a professional course advisor. Based on the following context, provide a comprehensive and well-structured response to the query: '{query}'

Context:
{context}

Please format your response professionally with the following structure:

## Course Overview
[Brief introduction and key highlights]

## Course Metadata
- **Course Name:** [Name]
- **Duration:** [Duration]
- **Course Fee:** [Fee]
- **Instructor:** [Name with qualifications and experience]

## Course Details
### Curriculum & Learning Objectives
[Detailed description of what students will learn]

### Course Structure & Methodology
[Information about hands-on practice, assessments, projects, etc.]

### Key Features
[Highlight unique aspects and benefits]

Make the response professional, informative, and well-organized. Use proper formatting with headers, bullet points, and clear sections. If multiple courses are relevant, structure each course separately."""

        response = self.llm.invoke(prompt)
        return response.content

