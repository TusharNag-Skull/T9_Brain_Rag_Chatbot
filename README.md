# Think9 Brain

A prototype **decision intelligence assistant** for consumer brands.

It helps people ask natural-language questions about internal documents — meetings, strategy, vendor evaluations, and decisions — and get answers grounded in those documents.

For the assignment write-up and architecture diagrams, see:

- [THINK9_BRAIN_SUBMISSION.md](THINK9_BRAIN_SUBMISSION.md)
- [THINK9_BRAIN_ARCHITECTURE.md](THINK9_BRAIN_ARCHITECTURE.md)

## Problem

Brand knowledge is often spread across many files. Finding *why* a vendor was chosen, or *what* a team decided, takes too long. Think9 Brain centralizes that information so operational questions can be answered quickly, with sources.

## Main features

- Ingests local `.txt` documents from `data/raw/`
- Chunks text and stores embeddings in a local Chroma vector database
- Conversational RAG with Groq (Llama 3.3 70B)
- Query contextualization so follow-up questions like “How much more expensive was it?” still retrieve the right documents
- Streamlit chat UI with source metadata

## Architecture overview

```text
User question
    → conversation history
    → standalone search query (query contextualization)
    → Chroma retrieval
    → Groq answer using original question + context + history
    → answer + sources
```

Heavy resources (documents, embeddings, Chroma, Groq client) are loaded once in Streamlit. Chat history stays in the current session only.

## Tech stack

- Python
- Streamlit
- LangChain (text splitters, HuggingFace embeddings, Chroma, Groq)
- sentence-transformers (`all-MiniLM-L6-v2`)
- Chroma (local vector store)
- Groq API for LLM generation

## Project structure

```text
think9-brain/
├── app.py                          # Streamlit UI
├── requirements.txt
├── .env.example                    # placeholders only
├── data/raw/                       # sample brand documents (tracked)
├── src/
│   ├── ingestion/                  # load documents + metadata
│   ├── preprocessing/              # chunking
│   ├── embeddings/                 # embedding model
│   ├── vectorstore/                # Chroma create/search
│   └── rag/                        # conversational RAG pipeline
├── THINK9_BRAIN_ARCHITECTURE.md
└── THINK9_BRAIN_SUBMISSION.md
```

## Setup

1. Clone the repository.
2. Create a virtual environment and install dependencies:

```bash
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
```

3. Create a local `.env` file from the example:

```bash
copy .env.example .env
```

4. Put your Groq API key in `.env`. **Do not commit `.env`.**

```text
GROQ_API_KEY=your_groq_api_key_here
GROQ_MODEL=llama-3.3-70b-versatile
```

Get a key from [Groq](https://console.groq.com/).

## Run locally

```bash
streamlit run app.py
```

Then open [http://localhost:8501](http://localhost:8501).

The first load may take a little time while documents are chunked, embedded, and stored in Chroma.

You can also run the terminal chat:

```bash
python src/rag/rag_pipeline.py
```

## Example usage

Ask:

- “Why was GreenCan selected?”
- “How much more expensive was it?”
- “Why was that acceptable?”

Follow-up questions can use words like “it” or “that”. The app rewrites them into a standalone search query before retrieving from Chroma.

## Current limitations

- Sample knowledge base only (three fictional brands, `.txt` files)
- Session chat history is not saved after the browser tab closes
- Retrieval is similarity search without reranking
- Answers depend on retrieved chunks; missing context can lead to incomplete answers
- Requires a Groq API key and internet access for generation

## Future improvements

- Broader document types (PDF, DOCX)
- Persistent conversation memory
- Brand/document-type filters
- Evaluation set and retrieval metrics
- Optional human-review step for high-impact recommendations
