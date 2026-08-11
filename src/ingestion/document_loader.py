"""Simple document loader for Think9 Brain.

Reads .txt files from data/raw/ and returns them as dictionaries
containing the document text and basic metadata.
"""

from pathlib import Path


def load_documents():
    """Load all .txt files under data/raw/ into a list of dictionaries."""

    # Get the root directory of the project.
    # Current file:
    # think9-brain/src/ingestion/document_loader.py
    #
    # parents[2] moves:
    # document_loader.py → ingestion → src → think9-brain
    project_root = Path(__file__).resolve().parents[2]

    # Define the location of our raw documents.
    raw_dir = project_root / "data" / "raw"

    # This list will store all loaded documents.
    documents = []

    # Recursively find all .txt files inside data/raw/.
    # sorted() keeps the processing order consistent.
    for file_path in sorted(raw_dir.rglob("*.txt")):

        # Read the complete text from the file.
        text = file_path.read_text(encoding="utf-8")

        # Get the file path relative to data/raw/.
        #
        # Example:
        # brand_alpha/meetings/q1_campaign_review.txt
        relative_path = file_path.relative_to(raw_dir)

        # The first folder represents the brand.
        #
        # Example:
        # brand_alpha/meetings/file.txt
        #     ↑
        #     brand
        brand = relative_path.parts[0]

        # The second folder represents the document type.
        #
        # Example:
        # brand_alpha/meetings/file.txt
        #                 ↑
        #                 document type
        document_type = relative_path.parts[1]

        # Store the document text and metadata together.
        documents.append(
            {
                "text": text,
                "metadata": {
                    # Name of the original file.
                    "source": file_path.name,

                    # Brand to which the document belongs.
                    "brand": brand,

                    # Category of the document:
                    # meetings, strategy, vendors, or decisions.
                    "document_type": document_type,
                },
            }
        )

    # Return the complete list of loaded documents.
    return documents


# This block runs only when this file is executed directly.
# It does not run when the function is imported into another file.
if __name__ == "__main__":

    # Load all documents from data/raw/.
    docs = load_documents()

    # Display the total number of documents found.
    print(f"Total documents: {len(docs)}")
    print("-" * 40)

    # Display basic information about every document.
    for doc in docs:
        meta = doc["metadata"]

        print(f"Source: {meta['source']}")
        print(f"Brand: {meta['brand']}")
        print(f"Document Type: {meta['document_type']}")
        print("-" * 40)



# I created a lightweight ingestion layer that recursively discovers raw documents, reads their contents, and extracts metadata from the directory structure. I keep the document content separate from metadata because the content will be used for semantic retrieval, while metadata will support filtering, source attribution, and brand-level retrieval.        