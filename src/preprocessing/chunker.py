"""Simple text chunker for Think9 Brain.

Takes documents from the ingestion step and splits them into
smaller overlapping chunks for later retrieval.
"""

import sys
from pathlib import Path

from langchain_text_splitters import RecursiveCharacterTextSplitter


# Add the src directory to Python's import path.
# This allows us to import modules from src/ when
# this file is executed directly.
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from ingestion.document_loader import load_documents


def chunk_documents(documents):
    """Split each document into smaller text chunks.

    Args:
        documents: List of dictionaries returned by load_documents().
                   Each dictionary contains "text" and "metadata".

    Returns:
        A new list of dictionaries containing:
        - chunk text
        - original metadata
        - chunk index
        - chunk ID
    """

    # Create the LangChain text splitter.
    #
    # chunk_size:
    # Approximate maximum size of each chunk.
    #
    # chunk_overlap:
    # Number of characters shared between consecutive chunks.
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=500,
        chunk_overlap=50,
    )

    # Store all generated chunks here.
    chunks = []

    # Process each document individually.
    for doc in documents:

        # Split only the document text.
        # The original metadata is handled separately.
        text_chunks = text_splitter.split_text(doc["text"])

        # Process every chunk created from this document.
        for chunk_index, chunk_text in enumerate(text_chunks):

            # Copy the original metadata so that
            # we do not modify the original document.
            metadata = doc["metadata"].copy()

            # Store the position of this chunk
            # within the original document.
            metadata["chunk_index"] = chunk_index

            # Create a simple identifier for the chunk.
            # Example:
            # retail_launch_decisions.txt::0
            metadata["chunk_id"] = (
                f"{metadata['source']}::{chunk_index}"
            )

            # Store the chunk text and its metadata.
            chunks.append(
                {
                    "text": chunk_text,
                    "metadata": metadata,
                }
            )

    # Return all generated chunks.
    return chunks


# This block runs only when chunker.py is executed directly.
if __name__ == "__main__":

    # Step 1: Load the original documents.
    documents = load_documents()

    # Step 2: Split the documents into smaller chunks.
    chunks = chunk_documents(documents)

    # Display the total number of chunks created.
    print(f"Total chunks: {len(chunks)}")
    print("=" * 40)

    # Display the first three chunks as a quick inspection.
    for chunk in chunks[:3]:

        # Show only the first 120 characters of the chunk.
        # lstrip() removes a possible BOM character.
        preview = chunk["text"].lstrip("\ufeff")[:120]

        print(f"Text: {preview}...")
        print(f"Metadata: {chunk['metadata']}")
        print("-" * 40)