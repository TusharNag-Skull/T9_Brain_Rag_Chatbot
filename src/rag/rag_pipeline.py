"""Simple RAG pipeline for Think9 Brain.

Retrieves relevant chunks from Chroma, then asks Groq
to answer using only that context.
Supports simple in-session follow-up questions.
"""

import os
import sys
from pathlib import Path

from dotenv import load_dotenv
from langchain_core.prompts import ChatPromptTemplate
from langchain_groq import ChatGroq

# Allow importing from src/ when this file is run directly
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from ingestion.document_loader import load_documents
from preprocessing.chunker import chunk_documents
from embeddings.embedding_service import create_embedding_model
from vectorstore.vector_store import create_vector_store, search_vector_store

# Load environment variables from the project root
project_root = Path(__file__).resolve().parents[2]
load_dotenv(project_root / ".env")
load_dotenv(project_root / ".env.example")


def create_llm():
    """Create and return a ChatGroq LLM from environment settings."""
    llm = ChatGroq(
        api_key=os.getenv("GROQ_API_KEY"),
        model=os.getenv("GROQ_MODEL"),
        temperature=0,
    )
    return llm


# Prompt used only to rewrite follow-up questions for vector search
SEARCH_QUERY_PROMPT = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            "Rewrite the user's question as a standalone search query.\n"
            "Use the conversation history only to resolve references such as:\n"
            '"it", "that", "this", "they", "the vendor", etc.\n'
            "Preserve the user's original meaning.\n"
            "Do not answer the question.\n"
            "Return ONLY the standalone search query.\n"
            "Do not add explanations.\n\n"
            "Previous conversation:\n{chat_history}",
        ),
        ("human", "{question}"),
    ]
)


# Prompt includes history, retrieved context, and the current question
RAG_PROMPT = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            "Answer using the provided context.\n"
            "Use conversation history only to understand the user's follow-up question.\n"
            "Do not invent facts.\n"
            "If the answer is not available in the provided context, say that the "
            "information is not available in the provided documents.\n"
            "Keep the answer clear and concise.\n\n"
            "Previous conversation:\n{chat_history}\n\n"
            "Context:\n{context}",
        ),
        ("human", "{question}"),
    ]
)


def _format_chat_history(chat_history):
    """Turn the chat history list into plain text for the prompt."""
    if not chat_history:
        return "No previous conversation."

    lines = []
    for message in chat_history:
        lines.append(f"{message['role']}: {message['content']}")
    return "\n".join(lines)


def create_search_query(llm, question, chat_history):
    """Turn a follow-up question into a standalone search query."""
    # No history yet, so the question is already standalone
    if not chat_history:
        return question

    # Follow-up questions may not contain enough information for vector search.
    # We use conversation history to turn them into standalone search queries.
    messages = SEARCH_QUERY_PROMPT.format_messages(
        chat_history=_format_chat_history(chat_history),
        question=question,
    )
    response = llm.invoke(messages)
    return response.content.strip()


def answer_question(vector_store, llm, question, chat_history=None, k=3):
    """Retrieve relevant chunks and generate a grounded answer.

    Args:
        vector_store: Chroma store from create_vector_store()
        llm: ChatGroq model from create_llm()
        question: user question as a string
        chat_history: optional list of previous messages in this session
        k: number of chunks to retrieve

    Returns:
        dict with "answer" and "sources"
    """
    # Start a fresh history list when none is provided
    if chat_history is None:
        chat_history = []

    # Build a standalone query for retrieval only
    search_query = create_search_query(llm, question, chat_history)
    print(f"Search query: {search_query}")

    # Step 1: find the most relevant chunks using the search query
    retrieved_docs = search_vector_store(vector_store, search_query, k=k)

    # Step 2: combine chunk texts into one context string
    context = "\n\n".join(doc.page_content for doc in retrieved_docs)

    # Step 3: answer with the original question + history + context
    messages = RAG_PROMPT.format_messages(
        chat_history=_format_chat_history(chat_history),
        context=context,
        question=question,
    )
    response = llm.invoke(messages)
    answer = response.content

    # Step 4: save this turn in the session history
    chat_history.append({"role": "user", "content": question})
    chat_history.append({"role": "assistant", "content": answer})

    # Step 5: keep source metadata for transparency
    sources = []
    for doc in retrieved_docs:
        meta = doc.metadata
        sources.append(
            {
                "source": meta.get("source"),
                "brand": meta.get("brand"),
                "document_type": meta.get("document_type"),
                "chunk_index": meta.get("chunk_index"),
                "chunk_id": meta.get("chunk_id"),
            }
        )

    return {
        "answer": answer,
        "sources": sources,
    }


if __name__ == "__main__":
    documents = load_documents()
    chunks = chunk_documents(documents)
    embedding_model = create_embedding_model()
    vector_store = create_vector_store(chunks, embedding_model)
    llm = create_llm()

    # One shared history list for this session
    chat_history = []

    print("Think9 Brain is ready. Type 'exit' to quit.")
    print("=" * 40)

    # Keep asking questions until the user types exit
    while True:
        question = input("Think9 Brain - Ask a question (type 'exit' to quit): ").strip()

        if question.lower() == "exit":
            print("Goodbye!")
            break

        if not question:
            continue

        result = answer_question(
            vector_store,
            llm,
            question,
            chat_history=chat_history,
            k=3,
        )

        print()
        print("Think9 Brain:")
        print(result["answer"])
        print()
        print("Sources:")
        for i, source in enumerate(result["sources"], start=1):
            print(f"  Source {i}: {source}")
        print("=" * 40)




# I use query contextualization only when conversational history is needed to make the retrieval query complete. The resulting standalone query is sent to Chroma, while the original user question remains the question answered by the final RAG generation step