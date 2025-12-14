AI-Powered PySpark Column-Level Lineage with RAG-Based Q&A
Executive Summary

This repository implements a production-grade metadata lineage platform for PySpark batch pipelines, augmented with Retrieval-Augmented Generation (RAG) to support fuzzy natural-language questions about lineage, impact, and downstream effects.

Unlike demo-style RAG systems, this project:

does not rely on LLMs for truth

uses deterministic lineage graphs as the source of record

uses LLMs only for semantic enrichment and explanation

mirrors the internal architecture of enterprise data catalogs (DataHub / Atlan / Amundsen)

The result is a system that can explain lineage, impact, and business meaning while remaining auditable, reproducible, and scalable.

What Problems This Solves (Why This Exists)

Modern data teams struggle with:

understanding column-level lineage in Spark pipelines

impact analysis when business definitions change

answering ad-hoc lineage questions without reading code

avoiding LLM hallucination in metadata systems

This project solves those problems by:

Extracting deterministic lineage from PySpark code

Enriching it with LLM-assisted semantics

Converting lineage into a queryable knowledge base

Enabling RAG-based Q&A over lineage metadata

High-Level Architecture (Interview-Grade)
                    ┌────────────────────────────┐
                    │   PySpark Code Repository   │
                    │  (batch ETL / ELT scripts)  │
                    └──────────────┬─────────────┘
                                   │
                                   ▼
┌────────────────────────────────────────────────────────┐
│ 1. Static Lineage Extraction (Deterministic)            │
│    - Python AST parsing                                 │
│    - DataFrames, columns, joins, reads, writes          │
│    - NO LLM usage                                       │
└──────────────────────────────┬─────────────────────────┘
                               │
                               ▼
┌────────────────────────────────────────────────────────┐
│ 2. Base Lineage JSON                                    │
│    - scripts → dataframes → columns                     │
│    - syntactic truth only                               │
└──────────────────────────────┬─────────────────────────┘
                               │
                               ▼
┌────────────────────────────────────────────────────────┐
│ 3. Semantic Enrichment (Amazon Bedrock)                 │
│    - window functions (lag, lead, rank, rolling)        │
│    - derived column semantics                           │
│    - SQL blocks (CTEs, aggregations)                    │
│    - joins + business meaning                           │
│    - STRICT JSON output                                 │
└──────────────────────────────┬─────────────────────────┘
                               │
                               ▼
┌────────────────────────────────────────────────────────┐
│ 4. Deterministic Lineage Post-Processing                │
│    - Convert semantics → canonical column edges         │
│    - Enforce correctness                                │
│    - Prevent hallucination                              │
└──────────────────────────────┬─────────────────────────┘
                               │
                               ▼
┌────────────────────────────────────────────────────────┐
│ 5. Repository-Wide Lineage Graph                        │
│    - Cross-script stitching via assets                  │
│    - End-to-end lineage                                 │
└──────────────────────────────┬─────────────────────────┘
                               │
                               ▼
┌────────────────────────────────────────────────────────┐
│ 6. Interactive Lineage Visualization                    │
│    - Mermaid-based HTML                                 │
│    - Column-to-column edges                             │
│    - Scripts → DataFrames → Columns                     │
└──────────────────────────────┬─────────────────────────┘
                               │
                               ▼
┌────────────────────────────────────────────────────────┐
│ 7. RAG Knowledge Base                                   │
│    - Lineage → textual facts                            │
│    - Embeddings (Amazon Titan)                          │
│    - FAISS vector index                                 │
└──────────────────────────────┬─────────────────────────┘
                               │
                               ▼
┌────────────────────────────────────────────────────────┐
│ 8. Natural Language Q&A (RAG)                            │
│    - Semantic retrieval                                 │
│    - Context-grounded LLM answers                       │
│    - No hallucination                                   │
└────────────────────────────────────────────────────────┘


Key design principle:

LLMs assist understanding — they never define lineage truth.

Repository Structure
metadata-lineage-ai/
├── extractor/
│   ├── static_extract.py        # AST-based deterministic extraction
│   ├── bedrock_enrich.py        # LLM semantic enrichment
│   ├── lineage_postprocess.py   # Canonical lineage edges
│   ├── stitch_repo.py           # Cross-script stitching
│   └── build_html.py            # Interactive HTML lineage
│
├── examples/                    # Sample PySpark pipelines
│
├── outputs/                     # Generated lineage artifacts
│   ├── *.json                   # Base extraction
│   ├── *.enriched.json          # Enriched lineage
│   ├── repo_graph.json
│   └── lineage_repo.html
│
├── qa/
│   ├── build_corpus.py          # Convert lineage → documents
│   ├── embed_index.py           # Embeddings + FAISS
│   ├── ask.py                   # RAG Q&A
│   ├── corpus.json
│   ├── index.faiss
│   └── index_ids.json
│
├── config.json                  # AWS + model configuration
└── README.md

Prerequisites (Before Cloning)
1. System Requirements

Python 3.9+

macOS / Linux (Windows works with minor changes)

2. AWS Requirements

AWS account with Amazon Bedrock enabled

IAM permissions:

bedrock:InvokeModel

3. AWS Credentials
aws configure

Step-by-Step: Clone → Ask a Question
1️⃣ Clone the Repository
git clone https://github.com/rohit2101991/ai-metadata-lineage-pyspark.git
cd ai-metadata-lineage-pyspark

2️⃣ Create a Virtual Environment
python -m venv .venv
source .venv/bin/activate

3️⃣ Install Dependencies
pip install -r requirements.txt

4️⃣ Configure Bedrock Models

Edit config.json:

{
  "region": "us-east-1",
  "model_id": "amazon.nova-pro-v1:0",
  "embedding_model_id": "amazon.titan-embed-text-v2:0"
}

Lineage Pipeline (Must Run in Order)
5️⃣ Extract Deterministic Lineage
python extractor/static_extract.py examples/ --out outputs


Output:

outputs/script_x.json


This step:

parses Python AST

detects DataFrames, columns, reads/writes

does NOT use AI

6️⃣ Enrich Lineage with LLM Semantics
python extractor/bedrock_enrich.py examples/ outputs/


Output:

outputs/script_x.enriched.json


This step:

infers window functions

derived column semantics

SQL aggregations

join meaning

7️⃣ Canonicalize Lineage
python extractor/lineage_postprocess.py outputs/


This step:

converts semantics → deterministic column edges

guarantees correctness

8️⃣ Stitch Repository & Visualize
python extractor/stitch_repo.py outputs/
python extractor/build_html.py outputs/


Open:

outputs/lineage_repo.html

RAG Q&A Pipeline
9️⃣ Build the Knowledge Corpus
python qa/build_corpus.py


Creates:

qa/corpus.json

🔟 Build Vector Index
python qa/embed_index.py


Creates:

qa/index.faiss
qa/index_ids.json

1️⃣1️⃣ Ask Questions
python qa/ask.py


Example questions:

Where does net_spend come from?

Which window functions depend on event_ts?

If amount changes in bronze ingestion,
which gold marts are impacted?

Which scripts reference abs_amount?


Future Enhancements

Web UI with lineage highlighting

OpenSearch Serverless vector backend

Streaming lineage ingestion

Change-impact scoring

OpenLineage compatibility



