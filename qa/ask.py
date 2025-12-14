#!/usr/bin/env python3
"""
Interactive lineage Q&A using FAISS + Amazon Bedrock.

Correctly resolves FAISS hits using index_ids.json -> corpus docs by id.
Supports multi-line questions. Uses numpy for FAISS vectors.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any, Dict, List, Tuple

import boto3
import faiss
import numpy as np


CFG_PATH = "config.json"
CORPUS_PATH = Path("qa/corpus.json")
INDEX_PATH = Path("qa/index.faiss")
IDS_PATH = Path("qa/index_ids.json")

TOP_K = 6


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
- If the answer is not present, say: "Cannot be determined from the available lineage."
- Do NOT invent columns, scripts, tables, or transformations.
- Be concise and precise.

CONTEXT:
{context}

QUESTION:
{question}

ANSWER:
""".strip()

    body = {
        "messages": [{"role": "user", "content": [{"text": prompt}]}],
        "inferenceConfig": {"maxTokens": 500, "temperature": 0.0, "topP": 0.9},
    }

    resp = client.invoke_model(
        modelId=model_id,
        body=json.dumps(body).encode("utf-8"),
        accept="application/json",
        contentType="application/json",
    )
    payload = json.loads(resp["body"].read())
    return payload["output"]["message"]["content"][0]["text"]


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


def load_corpus_maps() -> Tuple[List[Dict[str, Any]], List[Any], Dict[str, Dict[str, Any]]]:
    docs = json.loads(CORPUS_PATH.read_text())
    index_ids = json.loads(IDS_PATH.read_text())

    if not isinstance(docs, list):
        raise ValueError("qa/corpus.json must be a list of documents")

    docs_by_id: Dict[str, Dict[str, Any]] = {}
    for i, d in enumerate(docs):
        if not isinstance(d, dict):
            continue
        doc_id = d.get("id")
        if doc_id is None:
            # fallback id if older docs exist
            doc_id = str(i)
        docs_by_id[str(doc_id)] = d

    return docs, index_ids, docs_by_id


def main():
    region, embed_model, llm_model = load_cfg()

    if not (CORPUS_PATH.exists() and INDEX_PATH.exists() and IDS_PATH.exists()):
        print("ERROR: corpus/index missing. Run build_corpus.py and embed_index.py first.")
        sys.exit(1)

    docs, index_ids, docs_by_id = load_corpus_maps()
    if not docs:
        print("Corpus empty.")
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
            D, I = index.search(np.array(q_vec, dtype="float32").reshape(1, -1), TOP_K)

            # FAISS indices -> doc_ids via index_ids.json
            hit_doc_ids: List[str] = []
            for idx in I[0]:
                if idx < 0 or idx >= len(index_ids):
                    continue
                hit_doc_ids.append(str(index_ids[idx]))

            # doc_id -> doc text
            context_blocks: List[str] = []
            for doc_id in hit_doc_ids:
                d = docs_by_id.get(doc_id)
                if not d:
                    continue
                t = d.get("text", "")
                if t:
                    context_blocks.append(t)

            if not context_blocks:
                print("\nANSWER:\nCannot be determined from the available lineage.\n")
                continue

            context = "\n---\n".join(context_blocks)
            answer = ask_llm(region, llm_model, q, context)

            print("\nANSWER:\n")
            print(answer)
            print(f"\n--- Retrieved context blocks: {len(context_blocks)}")

        except KeyboardInterrupt:
            print("\nExiting.")
            break


if __name__ == "__main__":
    main()


