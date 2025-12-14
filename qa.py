#!/usr/bin/env python3
"""
Interactive lineage Q&A using FAISS + Amazon Bedrock.

Fixes:
- Uses NumPy for FAISS search vectors
- Supports multi-line questions
- Correctly resolves FAISS hits using index_ids.json (string IDs) -> corpus docs
- Gracefully handles empty / small corpora and missing IDs
- Prevents hallucinations by instructing LLM to only use retrieved context
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any

import boto3
import faiss
import numpy as np


# -------------------------
# Paths & constants
# -------------------------

CFG_PATH = "config.json"
CORPUS_PATH = Path("qa/corpus.json")
INDEX_PATH = Path("qa/index.faiss")
IDS_PATH = Path("qa/index_ids.json")

TOP_K = 6


# -------------------------
# Config loading
# -------------------------

def load_cfg() -> Tuple[str, str, str]:
    cfg = json.loads(Path(CFG_PATH).read_text())

    region = cfg.get("aws_region") or cfg.get("region")
    embed_model = cfg.get("embedding_model_id")
    llm_model = cfg.get("llm_model_id") or cfg.get("model_id")

    if not region:
        raise ValueError("config.json missing aws_region / region")
    if not embed_model:
        raise ValueError("config.json missing embedding_model_id")
    if not llm_model:
        raise ValueError("config.json missing llm_model_id / model_id")

    return region, embed_model, llm_model


# -------------------------
# Bedrock helpers
# -------------------------

def embed_text(region: str, model_id: str, text: str) -> List[float]:
    client = boto3.client("bedrock-runtime", region_name=region)

    body = {"inputText": text}

    resp = client.invoke_model(
        modelId=model_id,
        body=json.dumps(body).encode("utf-8"),
        accept="application/json",
        contentType="application/json",
    )

    payload = json.loads(resp["body"].read())
    return payload["embedding"]


def ask_llm(region: str, model_id: str, question: str, context: str) -> str:
    client = boto3.client("bedrock-runtime", region_name=region)

    prompt = f"""
You are a metadata lineage Q&A assistant.

RULES:
- Answer ONLY using the provided context.
- If information is missing, say: "Cannot be determined from the available lineage."
- Do NOT invent columns, scripts, tables, or transformations.
- If the question asks for repo-wide impact but the context only covers part of the repo,
  say which parts are covered and which are unknown.
- Be structured and precise.

CONTEXT:
{context}

QUESTION:
{question}

ANSWER:
""".strip()

    body = {
        "messages": [
            {"role": "user", "content": [{"text": prompt}]}
        ],
        "inferenceConfig": {
            "maxTokens": 900,
            "temperature": 0.0,
            "topP": 0.9
        }
    }

    resp = client.invoke_model(
        modelId=model_id,
        body=json.dumps(body).encode("utf-8"),
        accept="application/json",
        contentType="application/json",
    )

    payload = json.loads(resp["body"].read())
    return payload["output"]["message"]["content"][0]["text"]


# -------------------------
# Corpus helpers
# -------------------------

def load_corpus_and_index_ids() -> Tuple[List[dict], List[Any], Dict[str, dict]]:
    """
    Returns:
      docs_list: list of corpus docs
      index_ids: list aligned to FAISS vectors (strings or ints)
      docs_by_id: mapping from doc_id(str) -> doc
    """
    docs = json.loads(CORPUS_PATH.read_text())
    index_ids = json.loads(IDS_PATH.read_text())

    docs_list: List[dict]
    docs_by_id: Dict[str, dict] = {}

    # Corpus might be list or dict; normalize to list
    if isinstance(docs, dict):
        # If corpus is dict keyed by id -> doc
        docs_list = list(docs.values())
        for k, v in docs.items():
            if isinstance(v, dict):
                docs_by_id[str(k)] = v
    elif isinstance(docs, list):
        docs_list = docs
    else:
        raise ValueError("Unsupported corpus.json format; expected list or dict.")

    # Build id map if docs contain an 'id' field
    for i, d in enumerate(docs_list):
        if not isinstance(d, dict):
            continue
        doc_id = d.get("id")
        if doc_id is not None:
            docs_by_id[str(doc_id)] = d
        else:
            # fallback: allow integer indexing id as string
            docs_by_id.setdefault(str(i), d)

    return docs_list, index_ids, docs_by_id


def read_multiline_question() -> str:
    print("\nAsk a lineage question (end with a blank line):")
    lines = []
    while True:
        try:
            line = input()
        except EOFError:
            break
        if not line.strip():
            break
        lines.append(line)
    return "\n".join(lines).strip()


# -------------------------
# Main loop
# -------------------------

def main():
    region, embed_model, llm_model = load_cfg()

    if not CORPUS_PATH.exists() or not INDEX_PATH.exists() or not IDS_PATH.exists():
        print("ERROR: corpus or FAISS index not found. Run build_corpus.py and embed_index.py first.")
        sys.exit(1)

    docs_list, index_ids, docs_by_id = load_corpus_and_index_ids()

    if not docs_list:
        print("Corpus is empty. Nothing to search.")
        sys.exit(1)

    index = faiss.read_index(str(INDEX_PATH))

    print("\nLineage Q&A ready.")
    print("Paste multi-line questions. Press ENTER twice to submit. Ctrl+C to exit.")

    while True:
        try:
            q = read_multiline_question()
            if not q:
                continue

            q_vec = embed_text(region, embed_model, q)

            D, I = index.search(
                np.array(q_vec, dtype="float32").reshape(1, -1),
                TOP_K
            )

            # Convert FAISS indices -> doc IDs using index_ids.json
            hit_ids: List[str] = []
            for idx in I[0]:
                if idx < 0:
                    continue
                if idx >= len(index_ids):
                    continue
                hit_ids.append(str(index_ids[idx]))

            # Resolve IDs -> docs
            context_blocks: List[str] = []
            resolved = 0
            for doc_id in hit_ids:
                doc = docs_by_id.get(doc_id)
                if not doc:
                    continue
                text = doc.get("text")
                if not text:
                    continue
                context_blocks.append(text)
                resolved += 1

            if not context_blocks:
                print("\nANSWER:\nCannot be determined from the available lineage.\n")
                print(f"--- Retrieved ids: {len(hit_ids)}, resolved docs: {resolved}\n")
                continue

            context = "\n---\n".join(context_blocks)

            answer = ask_llm(region, llm_model, q, context)

            print("\nANSWER:\n")
            print(answer)
            print(f"\n--- Retrieved ids: {len(hit_ids)}, resolved docs: {resolved}, context blocks used: {len(context_blocks)}")

        except KeyboardInterrupt:
            print("\nExiting.")
            break


if __name__ == "__main__":
    main()


