# Think9 Brain --- Architecture

## Current POC

``` mermaid
flowchart TD
    A[Internal Documents] --> B[Document Loader]
    B --> C[Metadata + Text]
    C --> D[Text Chunking]
    D --> E[Embedding Model]
    E --> F[(Chroma Vector Store)]

    U[User] --> G[Streamlit Chat UI]
    G --> H[Conversation History]
    G --> I[Current Question]

    H --> J[Query Contextualization]
    I --> J
    J --> K[Standalone Search Query]

    K --> F
    F --> L[Relevant Chunks]

    L --> M[Context]
    I --> N[Original Question]
    H --> O[Conversation Context]

    M --> P[Groq / Llama 3.3 70B]
    N --> P
    O --> P

    P --> Q[Grounded Answer]
    L --> R[Source Metadata]

    Q --> G
    R --> G
```

## Why the flow has two uses of the LLM

The LLM has two different responsibilities:

1.  **Query contextualization**\
    Converts an incomplete follow-up question into a standalone
    retrieval query.

2.  **Answer generation**\
    Uses the retrieved evidence, original question, and conversation
    context to generate the final answer.

Chroma is responsible for retrieval. Groq/Llama is responsible for
language understanding and answer generation.

## Conversational Retrieval Example

``` text
User:
Why was GreenCan selected?

        ↓

Assistant:
GreenCan was selected because...

        ↓

User:
How much more expensive was it?

        ↓

Query Contextualization:

"What was the additional cost of GreenCan
compared with PackRight?"

        ↓

Chroma Retrieval

        ↓

Relevant GreenCan/PackRight chunks

        ↓

Groq

        ↓

"$0.06 more per unit."
```

## Future Production Workflow

``` mermaid
flowchart TD
    A[User Request] --> B[Query Understanding]
    B --> C[Knowledge Retrieval]
    C --> D[Evidence Gathering]
    D --> E[Reasoning / Analysis]
    E --> F{Request Type}

    F -->|Information| G[Grounded Answer]
    F -->|Recommendation| H[Recommendation]
    F -->|High-impact Action| I[Human Review]

    H --> I
    I --> J{Approved?}
    J -->|Yes| K[Execute Workflow]
    J -->|No| L[Return for Revision]

    G --> M[Source Traceability]
    K --> M
    L --> M
```

The future workflow is intentionally designed so that consequential
actions have a human approval checkpoint.
