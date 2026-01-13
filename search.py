import os
from vector_store import FaissVectorStore
from langchain_groq import ChatGroq

load_dotenv()

class RAGSearch:
    def __init__(self, persist_dir: str = "faiss_store", embedding_model: str = "all-MiniLM-L6-v2", llm_model: str = "llama-3.1-8b-instant"):
        
        self.vectorstore = FaissVectorStore(persist_dir, embedding_model)

        # Load or build vectorstore
        faiss_path = os.path.join(persist_dir, "faiss.index")
        meta_path = os.path.join(persist_dir, "metadata.pkl")

        if not (os.path.exists(faiss_path) and os.path.exists(meta_path)):
            from data_loader import load_all_documents
            docs = load_all_documents("data")
            self.vectorstore.build_from_documents(docs)
        else:
            self.vectorstore.load()
        
        GROQ_API_KEY = st.secrets["GROQ_API_KEY"]

        self.llm = ChatGroq(
            groq_api_key=GROQ_API_KEY,
            model_name=llm_model
        )

        print(f"[INFO] Groq LLM initialized: {llm_model}")


    def search_and_summarize(self, query: str, top_k: int = 5) -> str:
        results = self.vectorstore.query(query, top_k=top_k)
        texts = [r["metadata"].get("text", "") for r in results if r["metadata"]]

        context = "\n\n".join(texts)

        if not context:
            return "I apologize, but I couldn't find any relevant information in our course database for your query. Please try rephrasing your question or ask about our available programs."
        
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

        response = self.llm.invoke([prompt])
        return response.content

# Example usage
if __name__ == "__main__":
    rag_search = RAGSearch()
    query = "Who is Vivek sambharwal ?"
    summary = rag_search.search_and_summarize(query, top_k=3)
    print("Summary:", summary)
