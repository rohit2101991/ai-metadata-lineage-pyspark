#!/usr/bin/env python3
"""
Hybrid-always lineage Q&A:
- Retrieval: Titan embeddings + FAISS over qa/corpus.json
- Deterministic impact: build column dependency graph from outputs/*.enriched.json
- Generation: Nova (Bedrock) answers strictly using combined evidence

Usage:
  python qa/ask.py
"""

from __future__ import annotations

import json
import re
import sys
from collections import defaultdict, deque
from pathlib import Path
from typing import Any, Dict, List, Set, Tuple

import boto3
import faiss
import numpy as np

CFG_PATH = "config.json"
CORPUS_PATH = Path("qa/corpus.json")
INDEX_PATH = Path("qa/index.faiss")
IDS_PATH = Path("qa/index_ids.json")
OUTPUTS_DIR = Path("outputs")

TOP_K = 25                 # retrieval depth (increase for more evidence)
MAX_IMPACT_COLS = 3        # max columns to run deterministic closure for
MAX_IMPACT_SHOW = 80       # cap printing to keep prompts bounded


# -----------------------------
# Config + Bedrock calls
# -----------------------------

def load_cfg() -> Tuple[str, str, str]:
    """
    Supports both old and new keys:
      region / aws_region
      embed_model_id / embedding_model_id
      model_id / llm_model_id
    """
    cfg = json.loads(Path(CFG_PATH).read_text())

    region = cfg.get("region") or cfg.get("aws_region")
    embed_model = (
        cfg.get("embed_model_id")
        or cfg.get("embedding_model_id")
        or cfg.get("embeddingModelId")
    )
    llm_model = (
        cfg.get("model_id")
        or cfg.get("llm_model_id")
        or cfg.get("llmModelId")
    )

    if not region:
        raise ValueError("config.json missing region (or aws_region)")
    if not embed_model:
        raise ValueError("config.json missing embed_model_id (or embedding_model_id)")
    if not llm_model:
        raise ValueError("config.json missing model_id (or llm_model_id)")

    return region, embed_model, llm_model


def br_client(region: str):
    return boto3.client("bedrock-runtime", region_name=region)


def embed_text(client, model_id: str, text: str) -> np.ndarray:
    body = {"inputText": text}
    resp = client.invoke_model(
        modelId=model_id,
        body=json.dumps(body).encode("utf-8"),
        accept="application/json",
        contentType="application/json",
    )
    payload = json.loads(resp["body"].read())
    return np.array(payload["embedding"], dtype="float32")


def ask_llm(client, model_id: str, question: str, evidence: str) -> str:
    """
    Nova chat style: requires messages[].
    """
    prompt = f"""
You are a metadata lineage Q&A assistant.

RULES:
- Answer ONLY using the provided EVIDENCE.
- If something is not in EVIDENCE, say: "Unknown from available lineage metadata."
- Do NOT invent columns, scripts, tables, or transformations.
- Prefer bullet points.
- When possible, name scripts/dataframes/columns as shown in evidence.

QUESTION:
{question}

EVIDENCE:
{evidence}

ANSWER:
""".strip()

    body = {
        "messages": [{"role": "user", "content": [{"text": prompt}]}],
        "inferenceConfig": {"maxTokens": 900, "temperature": 0.1, "topP": 0.9},
    }
    resp = client.invoke_model(
        modelId=model_id,
        body=json.dumps(body).encode("utf-8"),
        accept="application/json",
        contentType="application/json",
    )
    payload = json.loads(resp["body"].read())
    return payload["output"]["message"]["content"][0]["text"].strip()


# -----------------------------
# Corpus + FAISS loading
# -----------------------------

def load_rag_store():
    if not (CORPUS_PATH.exists() and INDEX_PATH.exists() and IDS_PATH.exists()):
        raise FileNotFoundError("Missing qa/corpus.json or qa/index.faiss or qa/index_ids.json. Run build_corpus.py and embed_index.py first.")

    docs = json.loads(CORPUS_PATH.read_text())
    index_ids = json.loads(IDS_PATH.read_text())
    index = faiss.read_index(str(INDEX_PATH))

    docs_by_id = {str(d["id"]): d for d in docs if isinstance(d, dict) and "id" in d}
    return docs, docs_by_id, index_ids, index


def retrieve_docs(index, q_vec: np.ndarray, index_ids: List[str], docs_by_id: Dict[str, Any], top_k: int) -> List[str]:
    D, I = index.search(q_vec.reshape(1, -1).astype("float32"), top_k)
    hits: List[str] = []
    for idx in I[0]:
        if idx < 0 or idx >= len(index_ids):
            continue
        doc_id = str(index_ids[idx])
        if doc_id in docs_by_id:
            hits.append(doc_id)

    # dedup keep order
    seen = set()
    out = []
    for h in hits:
        if h not in seen:
            seen.add(h)
            out.append(h)
    return out


# -----------------------------
# Deterministic graph from enriched JSONs
# -----------------------------

def load_all_enriched(outputs_dir: Path) -> List[dict]:
    return [json.loads(p.read_text()) for p in outputs_dir.glob("*.enriched.json")]


def build_dependency_graph(enriched_files: List[dict]) -> Tuple[Dict[str, Set[str]], Dict[Tuple[str, str], Set[str]], Set[str]]:
    """
    Build edges from bedrock_enrichment:
      derived_columns: source_cols -> col
      aggregations: source_cols -> alias

    Returns:
      adj: src -> {dst}
      reasons: (src,dst) -> {why strings}
      known_cols: set of all seen column names
    """
    adj: Dict[str, Set[str]] = defaultdict(set)
    reasons: Dict[Tuple[str, str], Set[str]] = defaultdict(set)
    known_cols: Set[str] = set()

    for data in enriched_files:
        script = data.get("script_name", "unknown_script")
        be = data.get("bedrock_enrichment", {}) or {}

        # derived columns
        for item in be.get("derived_columns", []) or []:
            dst = item.get("col")
            srcs = item.get("source_cols", []) or []
            expr = (item.get("expression") or "").strip()
            if dst:
                known_cols.add(dst)
            for s in srcs:
                known_cols.add(s)
            if dst and srcs:
                for s in srcs:
                    adj[s].add(dst)
                    reasons[(s, dst)].add(f"{script}: derived ({expr})" if expr else f"{script}: derived")

        # aggregations
        for agg in be.get("aggregations", []) or []:
            dst = agg.get("alias")
            srcs = agg.get("source_cols", []) or []
            expr = (agg.get("expr") or "").strip()
            if dst:
                known_cols.add(dst)
            for s in srcs:
                known_cols.add(s)
            if dst and srcs:
                for s in srcs:
                    adj[s].add(dst)
                    reasons[(s, dst)].add(f"{script}: agg ({expr})" if expr else f"{script}: agg")

        # ALSO add any explicitly listed dataframe columns (helps candidate extraction)
        for df in data.get("dataframes", []) or []:
            for c in df.get("columns", []) or []:
                name = c.get("name")
                if name:
                    known_cols.add(name)

        # NOTE: we do NOT touch group_by because its schema varies (dict vs list) and
        # it’s not required to build column-to-column dependency edges.

    return adj, reasons, known_cols


def downstream_closure(adj: Dict[str, Set[str]], start: str, limit: int = 2000) -> List[str]:
    seen = {start}
    out: List[str] = []
    q = deque([start])
    while q and len(out) < limit:
        cur = q.popleft()
        for nxt in sorted(adj.get(cur, [])):
            if nxt in seen:
                continue
            seen.add(nxt)
            out.append(nxt)
            q.append(nxt)
    return out


def extract_candidate_columns(question: str, known_cols: Set[str], max_cols: int) -> List[str]:
    """
    Hybrid always needs a start column for graph traversal.
    Heuristic (deterministic):
      1) backticks: `event_ts`
      2) tokens that exist in known_cols
    """
    candidates: List[str] = []

    # backticks first
    for m in re.findall(r"`([^`]+)`", question):
        if m in known_cols and m not in candidates:
            candidates.append(m)
        if len(candidates) >= max_cols:
            return candidates[:max_cols]

    # then plain tokens
    tokens = re.findall(r"[A-Za-z_][A-Za-z0-9_]*", question)
    for t in tokens:
        if t in known_cols and t not in candidates:
            candidates.append(t)
        if len(candidates) >= max_cols:
            break

    return candidates[:max_cols]


# -----------------------------
# UI helpers
# -----------------------------

def read_multiline_question() -> str:
    print("\nAsk a lineage question (end with a blank line):")
    lines = []
    while True:
        try:
            line = input()
        except EOFError:
            return ""
        if not line.strip():
            break
        lines.append(line)
    return "\n".join(lines).strip()


def build_evidence(hits: List[str], docs_by_id: Dict[str, Any], col_impacts: Dict[str, List[str]], reasons: Dict[Tuple[str, str], Set[str]]) -> str:
    parts: List[str] = []

    # Deterministic impact evidence
    parts.append("=== DETERMINISTIC IMPACT (GRAPH TRAVERSAL) ===")
    if not col_impacts:
        parts.append("No candidate column detected in question; impact traversal skipped.")
    else:
        for col, impacted in col_impacts.items():
            parts.append(f"Start column: {col}")
            if not impacted:
                parts.append("  No downstream impacted columns found from available lineage.")
                continue
            parts.append("  Downstream impacted columns:")
            for x in impacted[:MAX_IMPACT_SHOW]:
                parts.append(f"   - {x}")
            if len(impacted) > MAX_IMPACT_SHOW:
                parts.append(f"   - ... ({len(impacted) - MAX_IMPACT_SHOW} more)")

            # show direct edges for transparency
            parts.append("  Direct edges (1-hop) with reasons:")
            direct = sorted([dst for (src, dst) in reasons.keys() if src == col])
            if not direct:
                parts.append("   - None found")
            else:
                for dst in direct[:20]:
                    why = "; ".join(sorted(reasons.get((col, dst), set())))
                    parts.append(f"   - {col} -> {dst}: {why}")

    # Retrieved facts evidence
    parts.append("\n=== RETRIEVED LINEAGE FACTS (FAISS TOP-K) ===")
    for i, doc_id in enumerate(hits, 1):
        txt = docs_by_id[doc_id].get("text", "").strip()
        txt_one = " ".join(txt.split())
        parts.append(f"{i:02d}. {doc_id} :: {txt_one}")

    return "\n".join(parts)


# -----------------------------
# Main
# -----------------------------

def main():
    region, embed_model, llm_model = load_cfg()
    client = br_client(region)

    # Load RAG store
    docs, docs_by_id, index_ids, index = load_rag_store()

    # Load deterministic graph
    if not OUTPUTS_DIR.exists():
        print("ERROR: outputs/ directory not found. Run the extractor pipeline first.")
        sys.exit(1)

    enriched_files = load_all_enriched(OUTPUTS_DIR)
    adj, reasons, known_cols = build_dependency_graph(enriched_files)

    print("\nLineage Q&A ready (HYBRID always).")
    print("Paste multi-line questions. Press ENTER twice to submit. Ctrl+C to exit.")

    while True:
        try:
            q = read_multiline_question()
            if not q:
                continue

            # 1) Candidate columns (for deterministic closure)
            cols = extract_candidate_columns(q, known_cols, MAX_IMPACT_COLS)

            col_impacts: Dict[str, List[str]] = {}
            for c in cols:
                col_impacts[c] = downstream_closure(adj, c)

            # 2) Retrieval (FAISS)
            q_vec = embed_text(client, embed_model, q)
            hits = retrieve_docs(index, q_vec, index_ids, docs_by_id, TOP_K)

            # 3) Evidence (deterministic + retrieved)
            evidence = build_evidence(hits, docs_by_id, col_impacts, reasons)

            # 4) LLM answer
            answer = ask_llm(client, llm_model, q, evidence)

            print("\nANSWER:\n")
            print(answer)
            print("\n--- Debug ---")
            print(f"candidate_cols: {cols}")
            print(f"retrieved_docs: {len(hits)}")
            print("-------------")

        except KeyboardInterrupt:
            print("\nExiting.")
            break


if __name__ == "__main__":
    main()


