#!/usr/bin/env python3
from __future__ import annotations

import json
import re
import sys
from collections import defaultdict, deque
from pathlib import Path
from typing import Dict, List, Set, Tuple

import numpy as np
import faiss
import boto3


# ----------------------------
# Config / Bedrock helpers
# ----------------------------

def load_config(path: str = "config.json") -> dict:
    return json.loads(Path(path).read_text())


def bedrock_client(region: str):
    return boto3.client("bedrock-runtime", region_name=region)


def titan_embed(client, model_id: str, text: str) -> np.ndarray:
    """
    Returns float32 numpy vector.
    """
    body = json.dumps({"inputText": text})
    resp = client.invoke_model(modelId=model_id, body=body)
    payload = json.loads(resp["body"].read())
    vec = np.array(payload["embedding"], dtype="float32")
    return vec


def nova_generate(client, model_id: str, question: str, evidence: str, temperature: float = 0.1) -> str:
    """
    Nova chat-style models require 'messages'. (You already fixed this earlier.)
    """
    req = {
        "messages": [
            {
                "role": "system",
                "content": [
                    {
                        "text": (
                            "You are a metadata lineage Q&A assistant. "
                            "Answer strictly using the provided EVIDENCE. "
                            "If a detail is not in EVIDENCE, say 'Unknown from available lineage metadata.' "
                            "Prefer bullet lists. Be precise and cite script/dataframe/column names exactly as shown."
                        )
                    }
                ],
            },
            {
                "role": "user",
                "content": [
                    {
                        "text": f"QUESTION:\n{question}\n\nEVIDENCE:\n{evidence}\n"
                    }
                ],
            },
        ],
        "temperature": temperature,
        "max_tokens": 800,
    }
    resp = client.invoke_model(modelId=model_id, body=json.dumps(req))
    payload = json.loads(resp["body"].read())

    # Nova responses can vary slightly by model version; handle common shapes
    # Expected: payload["output"]["message"]["content"][0]["text"]
    try:
        return payload["output"]["message"]["content"][0]["text"].strip()
    except Exception:
        return json.dumps(payload)[:2000]


# ----------------------------
# Graph building (deterministic)
# ----------------------------

def load_all_enriched(outputs_dir: Path) -> List[dict]:
    return [json.loads(p.read_text()) for p in outputs_dir.glob("*.enriched.json")]


def collect_known_columns(enriched_files: List[dict]) -> Set[str]:
    """
    Build a set of known column names from enriched JSONs.
    """
    cols: Set[str] = set()
    for data in enriched_files:
        # From dataframes columns
        for df in data.get("dataframes", []) or []:
            for c in df.get("columns", []) or []:
                name = c.get("name")
                if name:
                    cols.add(name)
        # From bedrock enrichment
        be = data.get("bedrock_enrichment", {}) or {}
        for item in be.get("derived_columns", []) or []:
            if item.get("col"):
                cols.add(item["col"])
            for s in item.get("source_cols", []) or []:
                cols.add(s)
        for agg in be.get("aggregations", []) or []:
            if agg.get("alias"):
                cols.add(agg["alias"])
            for s in agg.get("source_cols", []) or []:
                cols.add(s)
        gb = be.get("group_by", {}) or {}
        for k in gb.get("keys", []) or []:
            cols.add(k)
    return cols


def build_dependency_graph(enriched_files: List[dict]) -> Tuple[Dict[str, Set[str]], Dict[Tuple[str, str], Set[str]]]:
    """
    Build global directed graph: src_col -> dst_col
    Uses bedrock_enrichment (derived_columns + aggregations). This is reliable for column-level deps.
    """
    adj: Dict[str, Set[str]] = defaultdict(set)
    reasons: Dict[Tuple[str, str], Set[str]] = defaultdict(set)

    for data in enriched_files:
        script = data.get("script_name", "unknown_script")
        be = data.get("bedrock_enrichment", {}) or {}

        for d in be.get("derived_columns", []) or []:
            dst = d.get("col")
            srcs = d.get("source_cols", []) or []
            expr = (d.get("expression") or "").strip()
            if not dst or not srcs:
                continue
            for s in srcs:
                adj[s].add(dst)
                reasons[(s, dst)].add(f"{script}: derived {dst} from {s} ({expr})" if expr else f"{script}: derived {dst} from {s}")

        for a in be.get("aggregations", []) or []:
            dst = a.get("alias")
            srcs = a.get("source_cols", []) or []
            expr = (a.get("expr") or "").strip()
            if not dst or not srcs:
                continue
            for s in srcs:
                adj[s].add(dst)
                reasons[(s, dst)].add(f"{script}: agg {expr} -> {dst}")

        # You can optionally model group_by keys as affecting metrics, but that requires knowing which metrics.
        # We keep it conservative here.

    return adj, reasons


def downstream_closure(adj: Dict[str, Set[str]], start: str, limit: int = 2000) -> List[str]:
    """
    BFS for downstream impacted columns. Returns stable list excluding start.
    """
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


# ----------------------------
# RAG retrieval
# ----------------------------

def load_rag_store():
    docs = json.loads(Path("qa/corpus.json").read_text())
    index_ids = json.loads(Path("qa/index_ids.json").read_text())
    index = faiss.read_index("qa/index.faiss")

    docs_by_id = {d["id"]: d for d in docs}
    return docs, docs_by_id, index_ids, index


def retrieve_facts(question_vec: np.ndarray, index, index_ids: List[str], docs_by_id: dict, top_k: int = 25):
    D, I = index.search(np.array([question_vec]).astype("float32"), top_k)
    hits = []
    for idx in I[0]:
        if idx < 0:
            continue
        doc_id = index_ids[idx]
        if doc_id in docs_by_id:
            hits.append(doc_id)
    # Dedup but keep order
    seen = set()
    hits2 = []
    for h in hits:
        if h not in seen:
            seen.add(h)
            hits2.append(h)
    return hits2


# ----------------------------
# Column extraction (no hardcoded routing)
# ----------------------------

def extract_candidate_columns(question: str, known_cols: Set[str], max_cols: int = 3) -> List[str]:
    """
    Heuristic extraction of column names without routing.
    1) Prefer backticked tokens: `event_ts`
    2) Else pick snake_case tokens that exist in known_cols
    """
    candidates: List[str] = []

    # backticked
    for m in re.findall(r"`([^`]+)`", question):
        if m in known_cols:
            candidates.append(m)

    if len(candidates) >= max_cols:
        return candidates[:max_cols]

    # snake_case-ish tokens
    tokens = re.findall(r"[A-Za-z_][A-Za-z0-9_]*", question)
    for t in tokens:
        if t in known_cols and t not in candidates:
            candidates.append(t)
        if len(candidates) >= max_cols:
            break

    return candidates[:max_cols]


# ----------------------------
# Evidence builder
# ----------------------------

def build_evidence(question: str, hits: List[str], docs_by_id: dict,
                   col_impacts: Dict[str, List[str]],
                   reasons: Dict[Tuple[str, str], Set[str]]) -> str:
    parts = []

    # Impact evidence
    if col_impacts:
        parts.append("=== DETERMINISTIC IMPACT (GRAPH TRAVERSAL) ===")
        for col, impacted in col_impacts.items():
            parts.append(f"Start column: {col}")
            if not impacted:
                parts.append("  No downstream impacted columns found from available lineage.")
                continue
            parts.append("  Downstream impacted columns:")
            for x in impacted[:60]:
                parts.append(f"   - {x}")
            if len(impacted) > 60:
                parts.append(f"   - ... ({len(impacted)-60} more)")

            # Show 1-hop reasons
            parts.append("  Direct (1-hop) edges with reasons:")
            one_hop = sorted(set(impacted[:]) & set(reasons_dst_for_src(reasons, col)))
            # If not available, fall back to reasons table scan
            if not one_hop:
                one_hop = sorted([dst for (src, dst) in reasons.keys() if src == col])
            for dst in one_hop[:15]:
                why = "; ".join(sorted(reasons.get((col, dst), set())))
                if why:
                    parts.append(f"   - {col} -> {dst}: {why}")
    else:
        parts.append("=== DETERMINISTIC IMPACT (GRAPH TRAVERSAL) ===")
        parts.append("No explicit column found in question to run impact traversal.")

    # Retrieval evidence
    parts.append("\n=== RETRIEVED LINEAGE FACTS (FAISS TOP-K) ===")
    for i, h in enumerate(hits, 1):
        txt = docs_by_id[h]["text"].strip()
        txt_one = " ".join(txt.split())
        parts.append(f"{i:02d}. {h} :: {txt_one}")

    return "\n".join(parts)


def reasons_dst_for_src(reasons: Dict[Tuple[str, str], Set[str]], src: str) -> Set[str]:
    return {dst for (s, dst) in reasons.keys() if s == src}


# ----------------------------
# Main loop
# ----------------------------

def main():
    cfg = load_config("config.json")
    region = cfg.get("region", "us-east-1")
    llm_model = cfg.get("model_id", "amazon.nova-lite-v1:0")
    embed_model = cfg.get("embed_model_id", "amazon.titan-embed-text-v2:0")

    client = bedrock_client(region)

    # Load RAG store
    docs, docs_by_id, index_ids, index = load_rag_store()

    # Load lineage graph for deterministic traversal
    enriched_files = load_all_enriched(Path("outputs"))
    known_cols = collect_known_columns(enriched_files)
    adj, reasons = build_dependency_graph(enriched_files)

    print("\nLineage Q&A ready (HYBRID always).")
    print("Paste multi-line questions. Press ENTER twice to submit. Ctrl+C to exit.\n")

    buf: List[str] = []
    while True:
        try:
            line = input()
        except EOFError:
            break
        except KeyboardInterrupt:
            print("\nExiting.")
            break

        if line.strip() == "" and buf:
            question = "\n".join(buf).strip()
            buf = []

            # 1) Extract candidate columns (if any)
            cols = extract_candidate_columns(question, known_cols, max_cols=3)

            # 2) Deterministic impact for each candidate column
            col_impacts: Dict[str, List[str]] = {}
            for c in cols:
                col_impacts[c] = downstream_closure(adj, c)

            # 3) Vector retrieval
            q_vec = titan_embed(client, embed_model, question)
            hits = retrieve_facts(q_vec, index, index_ids, docs_by_id, top_k=25)

            # 4) Build evidence
            evidence = build_evidence(question, hits, docs_by_id, col_impacts, reasons)

            # 5) LLM answer constrained to evidence
            answer = nova_generate(client, llm_model, question, evidence, temperature=0.1)

            print("\nANSWER:\n")
            print(answer)
            print("\n--- Debug ---")
            print(f"candidate_cols: {cols}")
            print(f"retrieved_docs: {len(hits)}")
            print("-------------\n")
            print("Ask a lineage question (end with a blank line):")
            continue

        buf.append(line)


if __name__ == "__main__":
    main()


