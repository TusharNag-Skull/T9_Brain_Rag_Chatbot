"""Simple Streamlit chat UI for Think9 Brain.

This file only handles the UI. All RAG logic stays in the backend modules.
"""

import sys
from pathlib import Path

import streamlit as st

# Make the src/ package importable from the project root
sys.path.insert(0, str(Path(__file__).resolve().parent / "src"))

from ingestion.document_loader import load_documents
from preprocessing.chunker import chunk_documents
from embeddings.embedding_service import create_embedding_model
from vectorstore.vector_store import create_vector_store
from rag.rag_pipeline import create_llm, answer_question


# Cache the heavy resources so Streamlit does not rebuild them on every interaction.
@st.cache_resource
def load_brain():
    documents = load_documents()
    chunks = chunk_documents(documents)
    embedding_model = create_embedding_model()
    vector_store = create_vector_store(chunks, embedding_model)
    llm = create_llm()
    return vector_store, llm


st.set_page_config(page_title="Think9 Brain", page_icon="🧠")
st.title("Think9 Brain")
st.caption("AI-powered decision intelligence assistant")

# Small info control — full note opens only when clicked
with st.popover("ℹ️ Read this disclaimer"):
    st.caption(
        "Development Note: This prototype was developed with AI-assisted tools to "
        "accelerate implementation and documentation within the short assignment "
        "timeline. All components have been personally reviewed, tested, and "
        "understood, including the RAG pipeline, embeddings, vector database, "
        "conversational retrieval, query contextualization, Groq integration, and "
        "Streamlit interface. I am prepared to explain the architecture, "
        "implementation decisions, limitations, and future improvements during "
        "the discussion."
    )

# Initialize backend with a simple loading message
with st.spinner("Loading Think9 Brain..."):
    vector_store, llm = load_brain()

# Session chat history used by the RAG backend
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

# Display history so messages stay visible on the page
if "messages" not in st.session_state:
    st.session_state.messages = []

# Show previous conversation turns
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

        # Show sources under assistant replies
        if message["role"] == "assistant" and message.get("sources"):
            with st.expander("Sources"):
                for source in message["sources"]:
                    st.markdown(
                        f"- **source:** {source.get('source')}\n"
                        f"  - brand: {source.get('brand')}\n"
                        f"  - document_type: {source.get('document_type')}\n"
                        f"  - chunk_index: {source.get('chunk_index')}\n"
                        f"  - chunk_id: {source.get('chunk_id')}"
                    )

# User starts the conversation
question = st.chat_input("Ask Think9 Brain...")

if question:
    # Show the user message immediately
    st.session_state.messages.append({"role": "user", "content": question})
    with st.chat_message("user"):
        st.markdown(question)

    # Call the existing conversational RAG backend
    with st.chat_message("assistant"):
        with st.spinner("Thinking..."):
            result = answer_question(
                vector_store,
                llm,
                question,
                chat_history=st.session_state.chat_history,
                k=3,
            )

        answer = result["answer"]
        sources = result["sources"]

        st.markdown(answer)

        with st.expander("Sources"):
            for source in sources:
                st.markdown(
                    f"- **source:** {source.get('source')}\n"
                    f"  - brand: {source.get('brand')}\n"
                    f"  - document_type: {source.get('document_type')}\n"
                    f"  - chunk_index: {source.get('chunk_index')}\n"
                    f"  - chunk_id: {source.get('chunk_id')}"
                )

    # Keep the assistant reply visible on later reruns
    st.session_state.messages.append(
        {
            "role": "assistant",
            "content": answer,
            "sources": sources,
        }
    )