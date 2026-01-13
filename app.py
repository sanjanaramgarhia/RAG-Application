import streamlit as st
from search import RAGSearch

# Page configuration
st.set_page_config(
    page_title="RAG Chatbot",
    page_icon="💬",
    layout="wide"
)

# Initialize RAG system
@st.cache_resource
def init_rag_system():
    try:
        return RAGSearch()
        
    except Exception as e:
        st.error(f"Error initializing RAG system: {e}")
        return None

# Initialize session state
if "rag_system" not in st.session_state:
    st.session_state.rag_system = init_rag_system()

if "messages" not in st.session_state:
    st.session_state.messages = []

if "search_history" not in st.session_state:
    st.session_state.search_history = []

# Sidebar for search history
with st.sidebar:
    st.title("🔍 Search History")

    if st.session_state.search_history:
        for i, question in enumerate(reversed(st.session_state.search_history)):
            if st.button(
                f"💬 {question[:40]}{'...' if len(question) > 40 else ''}",
                key=f"history_{i}",
                use_container_width=True
            ):
                st.session_state.messages.append(
                    {"role": "user", "content": question}
                )

                if st.session_state.rag_system:
                    response = st.session_state.rag_system.search_and_summarize(
                        question, top_k=5
                    )
                    st.session_state.messages.append(
                        {"role": "assistant", "content": response}
                    )

                st.rerun()
    else:
        st.write("No search history yet")

    if st.session_state.search_history:
        if st.button("🗑️ Clear History", use_container_width=True):
            st.session_state.search_history = []
            st.rerun()

# Main chat area
st.title("🤖 RAG Chatbot")
st.write("Ask me anything about your documents!")

# Display chat messages
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.write(message["content"])

# Chat input
if prompt := st.chat_input("Type your message here..."):

    if prompt not in st.session_state.search_history:
        st.session_state.search_history.append(prompt)

    st.session_state.messages.append(
        {"role": "user", "content": prompt}
    )

    with st.chat_message("user"):
        st.write(prompt)

    if st.session_state.rag_system:
        with st.chat_message("assistant"):
            with st.spinner("Thinking..."):
                response = st.session_state.rag_system.search_and_summarize(
                    prompt, top_k=5
                )
            st.write(response)

        st.session_state.messages.append(
            {"role": "assistant", "content": response}
        )


             

    

