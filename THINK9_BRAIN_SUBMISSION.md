# Think9 Brain --- AI Decision Intelligence System

## 1. Executive Summary

Think9 Brain is a centralized AI-powered decision intelligence system
designed around the **"Decision Velocity & Institutional Memory"**
track.

The goal is to reduce the time spent searching fragmented internal
knowledge such as meeting notes, decision records, strategy documents,
and vendor evaluations. The proof of concept uses Retrieval-Augmented
Generation (RAG) so users can ask operational questions in natural
language and receive answers grounded in the organization's documents.

The current prototype provides:

-   Document ingestion
-   Metadata extraction
-   Text chunking
-   Local semantic embeddings
-   Chroma vector storage
-   Similarity-based retrieval
-   Groq-hosted Llama 3.3 70B generation
-   Source metadata
-   Conversational question answering
-   Query contextualization for follow-up questions
-   Streamlit chat interface

The assignment describes the opportunity as fragmented institutional
knowledge slowing decision-making and asks for a centralized "Think9
Brain" that can ingest internal information and answer operational
queries quickly.

------------------------------------------------------------------------

## 2. Problem

As the number of brands grows, useful organizational knowledge can
become distributed across many documents and functions.

Examples of information that may need to be found quickly include:

-   Why a business decision was made
-   Which vendor was selected
-   What alternatives were considered
-   What risks were identified
-   What strategy was agreed upon
-   What happened in previous meetings
-   What evidence supports a decision

Without a centralized intelligence layer, employees may need to manually
search documents, ask other team members, or repeatedly reconstruct the
reasoning behind earlier decisions.

This creates an **institutional memory and decision-velocity problem**.

------------------------------------------------------------------------

## 3. Proposed Solution

Think9 Brain provides a natural-language interface over internal
organizational knowledge.

The current prototype follows this flow:

``` text
Internal Documents
       |
       v
Document Ingestion
       |
       v
Metadata + Text
       |
       v
Chunking
       |
       v
Embeddings
       |
       v
Chroma Vector Store
       |
       v
User Question
       |
       v
Query Contextualization
       |
       v
Semantic Retrieval
       |
       v
Relevant Context
       |
       v
Groq / Llama 3.3 70B
       |
       v
Grounded Answer + Sources
```

The system is designed so that the LLM does not need to contain the
organization's private knowledge in its training data. Relevant
information is retrieved from the organization's documents and supplied
as context during answer generation.

------------------------------------------------------------------------

# 4. Current Proof of Concept Architecture

## 4.1 Document Ingestion

The ingestion layer scans the `data/raw/` directory and reads `.txt`
files from nested brand/document-type folders.

Example:

``` text
data/raw/
├── brand_alpha/
│   ├── decisions/
│   ├── meetings/
│   ├── strategy/
│   └── vendors/
├── brand_beta/
└── brand_gamma/
```

Each document is converted into a simple structure containing:

``` text
text
metadata
```

Metadata currently includes fields such as:

-   source
-   brand
-   document_type

This metadata is later stored with the vector representation.

------------------------------------------------------------------------

## 4.2 Chunking

Large documents are split into smaller overlapping chunks using
LangChain's `RecursiveCharacterTextSplitter`.

Current prototype settings:

``` text
chunk_size = 500
chunk_overlap = 50
```

The purpose is to make documents small enough for effective semantic
retrieval while keeping nearby information connected.

Each chunk receives:

``` text
chunk_index
chunk_id
```

The `chunk_id` helps identify the original chunk and prevents ambiguity
when tracing retrieved information.

------------------------------------------------------------------------

## 4.3 Embeddings

Each text chunk is converted into a numerical vector using:

``` text
sentence-transformers/all-MiniLM-L6-v2
```

The prototype produces vectors with:

``` text
384 dimensions
```

Conceptually:

``` text
Text
 ↓
Embedding Model
 ↓
[0.12, -0.04, 0.08, ...]
 ↓
384-dimensional vector
```

These vectors allow semantically similar text to be compared
mathematically.

------------------------------------------------------------------------

## 4.4 Vector Store

The embeddings and associated text/metadata are stored in a persistent
Chroma vector store.

The vector store is used for semantic similarity search.

For example:

``` text
Question:
Why was GreenCan selected?

        ↓

Chroma searches for semantically similar chunks

        ↓

Relevant documents:
retail_launch_decisions.txt
packaging_vendor_evaluation.txt
```

The system does not simply search for exact keywords.

------------------------------------------------------------------------

# 5. Retrieval-Augmented Generation

The core RAG pipeline is:

``` text
User Question
      |
      v
Query Contextualization
      |
      v
Standalone Search Query
      |
      v
Chroma Similarity Search
      |
      v
Top Relevant Chunks
      |
      v
Context + Original Question
      |
      v
Groq LLM
      |
      v
Grounded Answer
```

The LLM is instructed to answer using the retrieved context and to state
when the requested information is not available in the provided
documents.

This is intended to reduce unsupported answers.

------------------------------------------------------------------------

# 6. Conversational RAG

The system supports follow-up questions within the current session.

Example:

``` text
User:
Why was GreenCan selected?

Assistant:
GreenCan was selected because its recyclable specifications
met premium grocery requirements...

User:
How much more expensive was it?

Assistant:
GreenCan was $0.06 more expensive per unit than PackRight.

User:
Why was that acceptable?

Assistant:
The additional cost was accepted because retailer acceptance
and brand trust were considered more important than short-term
packaging savings.
```

The system maintains conversation history so references such as:

-   it
-   that
-   this
-   the vendor

can be understood in context.

------------------------------------------------------------------------

# 7. Query Contextualization

During testing, a retrieval problem was identified.

A follow-up question such as:

``` text
How much more expensive was it?
```

does not contain enough information by itself for reliable vector
search.

A naive retrieval system might retrieve an unrelated document about
another vendor.

To address this, Think9 Brain uses a simple query-contextualization
step.

``` text
Conversation History
        +
Current Follow-up Question
        |
        v
Query Contextualization
        |
        v
Standalone Search Query
        |
        v
Chroma
```

For example:

``` text
Conversation:
Why was GreenCan selected?

Follow-up:
How much more expensive was it?

Standalone retrieval query:
What was the additional cost of GreenCan compared with PackRight?
```

The contextualized query is used only for retrieval.

The original user question is still used for the final answer
generation.

This improvement changed the observed retrieval behavior from an
unrelated logistics-vendor result to the correct GreenCan/PackRight
information.

------------------------------------------------------------------------

# 8. Streamlit Application

The prototype provides a Streamlit chat interface.

The user starts the conversation and can ask arbitrary questions.

The UI displays:

-   User questions
-   Assistant answers
-   Expandable source information
-   Continuous conversation history

The application uses Streamlit session state for current-session
conversation history.

The expensive resources such as the embedding model and Chroma vector
store are reused rather than rebuilt for every user interaction.

Importantly, answers are **not cached**. Every new question performs
fresh retrieval and answer generation.

------------------------------------------------------------------------

# 9. Source Traceability

Each retrieved result retains metadata such as:

``` text
source
brand
document_type
chunk_index
chunk_id
```

The UI exposes this information under a Sources section.

This gives users a way to understand which internal document supported
an answer.

The goal is not only to provide an answer, but also to preserve
traceability back to organizational knowledge.

------------------------------------------------------------------------

# 10. Human-in-the-Loop Design

The current prototype is an information-retrieval and question-answering
system. It does not automatically execute business decisions.

For a future production version, actions should be separated by risk.

### Low-risk information request

``` text
User
 ↓
Retrieve
 ↓
Answer
```

Example:

> Why was GreenCan selected?

No approval is required.

### High-impact business recommendation

``` text
User Request
      |
      v
Retrieve Evidence
      |
      v
Analyze Options
      |
      v
Generate Recommendation
      |
      v
Human Review / Approval
      |
      v
Business Action
```

Example:

> Should we replace the current packaging vendor?

The system could collect historical decisions, vendor evaluations,
risks, and costs, then produce a recommendation.

A human decision-maker should review and approve the recommendation
before a consequential action is taken.

This human-in-the-loop layer is a proposed production capability, not
part of the current POC.

------------------------------------------------------------------------

# 11. Future Agentic Framework

The current POC intentionally keeps the implementation simple.

A production Think9 Brain can evolve into an agentic workflow:

``` text
                         User
                          |
                          v
                  Query Understanding
                          |
                          v
                  Retrieval / Search
                          |
                          v
                  Evidence Gathering
                          |
                          v
                  Reasoning / Analysis
                          |
              +-----------+-----------+
              |                       |
        Information                 Action
          Request                  Request
              |                       |
              v                       v
          Answer                 Recommendation
                                      |
                                      v
                                Human Approval
                                      |
                                      v
                                  Execution
```

Potential future capabilities could include:

-   Multi-source retrieval
-   Structured databases alongside document retrieval
-   Vendor comparison
-   Decision summarization
-   Risk identification
-   Recommendation generation
-   Human approval checkpoints
-   Workflow execution
-   Monitoring and alerts

These are proposed future capabilities, not claims about the current
prototype.

------------------------------------------------------------------------

# 12. Technology Stack

## Current Prototype

  Layer                       Technology
  --------------------------- ------------------------------------------
  Language                    Python
  UI                          Streamlit
  Document loading            Python / pathlib
  Text splitting              LangChain RecursiveCharacterTextSplitter
  Embeddings                  Sentence Transformers `all-MiniLM-L6-v2`
  Vector database             Chroma
  LLM                         Llama 3.3 70B via Groq
  LLM framework               LangChain
  Environment configuration   python-dotenv
  Conversation state          Streamlit session state

------------------------------------------------------------------------

# 13. Current Project Flow

``` text
data/raw/
   |
   v
document_loader.py
   |
   v
chunker.py
   |
   v
embedding_service.py
   |
   v
vector_store.py
   |
   v
rag_pipeline.py
   |
   +--> Query Contextualization
   |
   +--> Chroma Retrieval
   |
   +--> Groq Generation
   |
   v
app.py
   |
   v
Streamlit Chat UI
```

------------------------------------------------------------------------

# 14. 30-Day MVP Roadmap

## Days 1--5 --- Foundation

-   Finalize document ingestion format
-   Define metadata standards
-   Add support for common internal document formats
-   Establish document validation
-   Define brand/document access boundaries

## Days 6--10 --- Knowledge Retrieval

-   Improve chunking strategy
-   Evaluate embedding models
-   Build persistent vector storage
-   Add metadata filtering
-   Create retrieval evaluation questions

## Days 11--15 --- Intelligence Layer

-   Improve conversational retrieval
-   Add query contextualization
-   Improve source traceability
-   Add answer-grounding checks
-   Test hallucination behavior
-   Evaluate retrieval quality

## Days 16--20 --- Decision Intelligence

-   Add structured decision summaries
-   Add vendor comparison workflows
-   Add risk extraction
-   Add recommendation generation
-   Define human approval checkpoints

## Days 21--25 --- Production Workflow

-   Add authentication and role-based access
-   Add monitoring and logging
-   Add document update workflows
-   Add feedback collection
-   Add evaluation dashboards

## Days 26--30 --- Deployment & Validation

-   Deploy the MVP
-   Run end-to-end tests
-   Measure retrieval and answer quality
-   Conduct user testing
-   Document failure cases
-   Prepare rollout plan for additional brands

------------------------------------------------------------------------

# 15. Evaluation Plan

The system should be evaluated on more than whether the answer sounds
good.

Key evaluation dimensions:

### Retrieval quality

Does Chroma retrieve the correct source documents?

### Answer grounding

Does the answer stay within the retrieved evidence?

### Conversational understanding

Can follow-up questions correctly resolve references?

### Source traceability

Can users identify the documents supporting an answer?

### Hallucination behavior

Does the system clearly say when information is unavailable?

### Latency

How quickly can a user receive a response?

A production evaluation set should contain representative operational
questions and expected supporting documents.

------------------------------------------------------------------------

# 16. Key Engineering Decision

The prototype intentionally avoids unnecessary complexity.

The current system does not use:

-   LangGraph
-   Multiple autonomous agents
-   Redis
-   Complex memory frameworks
-   Persistent chat databases
-   Reranking pipelines
-   Tool-calling agents

The goal of the POC is to establish a clear, understandable foundation
first.

The architecture can then be expanded as actual business requirements
justify additional complexity.

------------------------------------------------------------------------

# 17. Business Value

The proposed system can help Think9 reduce the time required to recover
organizational knowledge.

Instead of:

``` text
Question
 ↓
Search folders
 ↓
Open documents
 ↓
Read multiple files
 ↓
Ask colleagues
 ↓
Reconstruct decision
```

Think9 Brain aims for:

``` text
Question
 ↓
Relevant evidence
 ↓
Grounded answer
 ↓
Source
```

For a multi-brand organization, this creates a foundation for
centralized institutional memory while still allowing knowledge to
remain connected to its originating documents.

------------------------------------------------------------------------

# 18. POC Limitations

The current prototype should be considered a proof of concept.

Known limitations include:

-   Current source documents are primarily text files.
-   The embedding model is a lightweight local model.
-   Retrieval quality depends on chunking and embedding quality.
-   The current conversation memory is session-level.
-   The current system is primarily question-answering rather than
    autonomous action execution.
-   Human approval workflows are proposed for future versions.
-   Production authentication and authorization are not implemented in
    the POC.
-   Production-scale monitoring and evaluation are not implemented in
    the POC.

These limitations are intentional for the prototype stage.

------------------------------------------------------------------------

# 19. Final Pitch

**Think9 Brain is a centralized decision-intelligence layer that turns
fragmented organizational knowledge into an accessible conversational
interface.**

The current POC demonstrates the core technical loop:

``` text
Internal Knowledge
      ↓
Semantic Retrieval
      ↓
Contextual Understanding
      ↓
Grounded Generation
      ↓
Source-backed Answer
```

The longer-term opportunity is to evolve this from a question-answering
system into a controlled intelligence and workflow layer that can gather
evidence, analyze business situations, recommend actions, and involve
humans whenever decisions have meaningful business consequences.
