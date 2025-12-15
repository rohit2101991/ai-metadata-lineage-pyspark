#!/usr/bin/env python3
"""
Hybrid-always lineage Q&A:
- Retrieval: Titan embeddings + FAISS over qa/corpus.json
- Deterministic impact: column dependency graph from outputs/*.enriched.json
- Deterministic asset BFS: impacted downstream scripts + impacted gold outputs
- Generation: Nova answers strictly using combined evidence

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

TOP_K = 25                 # retrieval depth
MAX_IMPACT_COLS = 3        # max columns to run deterministic closure for
MAX_IMPACT_SHOW = 80       # cap for prompt size
MAX_SCRIPTS_SHOW = 50      # cap to keep evidence bounded
MAX_GOLD_SHOW = 60         # cap to keep evidence bounded


# -----------------------------
# Config + Bedrock calls
# -----------------------------

def load_cfg() -> Tuple[str, str, str]:
    cfg = json.loads(Path(CFG_PATH).read_text())

    region = cfg.get("region") or cfg.get("aws_region")
    embed_model = (
        cfg.get("embed_model_id")
        or cfg.get("embedding_model_id")
        or cfg.get("embeddingModelId")
        or "amazon.titan-embed-text-v2:0"
    )
    llm_model = (
        cfg.get("model_id")
        or cfg.get("llm_model_id")
        or cfg.get("llmModelId")
        or "amazon.nova-lite-v1:0"
    )

    if not region:
        raise ValueError("config.json missing region (or aws_region)")

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
    prompt = f"""
You are a metadata lineage Q&A assistant.

RULES:
- Answer ONLY using the provided EVIDENCE.
- If something is not in EVIDENCE, say: "Unknown from available lineage metadata."
- Do NOT invent columns, scripts, tables, or transformations.
- Prefer bullet points.
- When possible, name scripts/dataframes/columns and output paths exactly as shown in evidence.

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
        raise FileNotFoundError(
            "Missing qa/corpus.json or qa/index.faiss or qa/index_ids.json. "
            "Run build_corpus.py and embed_index.py first."
        )

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
# Load enriched JSONs
# -----------------------------

def load_all_enriched(outputs_dir: Path) -> List[dict]:
    return [json.loads(p.read_text()) for p in outputs_dir.glob("*.enriched.json")]


# -----------------------------
# Deterministic column graph
# -----------------------------

def build_dependency_graph(enriched_files: List[dict]) -> Tuple[Dict[str, Set[str]], Dict[Tuple[str, str], Set[str]], Set[str]]:
    adj: Dict[str, Set[str]] = defaultdict(set)
    reasons: Dict[Tuple[str, str], Set[str]] = defaultdict(set)
    known_cols: Set[str] = set()

    for data in enriched_files:
        script = data.get("script_name", "unknown_script")
        be = data.get("bedrock_enrichment", {}) or {}

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

        for df in data.get("dataframes", []) or []:
            for c in df.get("columns", []) or []:
                name = c.get("name")
                if name:
                    known_cols.add(name)

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
    candidates: List[str] = []

    # backticks first
    for m in re.findall(r"`([^`]+)`", question):
        if m in known_cols and m not in candidates:
            candidates.append(m)
        if len(candidates) >= max_cols:
            return candidates[:max_cols]

    # then tokens
    tokens = re.findall(r"[A-Za-z_][A-Za-z0-9_]*", question)
    for t in tokens:
        if t in known_cols and t not in candidates:
            candidates.append(t)
        if len(candidates) >= max_cols:
            break

    return candidates[:max_cols]


# -----------------------------
# Deterministic asset BFS (scripts + gold outputs)
# -----------------------------

def norm_path(p: str) -> str:
    # normalize simple trailing slash differences
    return (p or "").strip()


def extract_assets(data: dict) -> Tuple[List[str], List[str]]:
    """
    assets live at top-level: data["assets"]["reads"/"writes"]
    Each entry looks like: {"format": "...", "path": "...", ...}
    """
    a = data.get("assets", {}) or {}
    reads = []
    writes = []
    for r in a.get("reads", []) or []:
        path = r.get("path") if isinstance(r, dict) else None
        if path:
            reads.append(norm_path(path))
    for w in a.get("writes", []) or []:
        path = w.get("path") if isinstance(w, dict) else None
        if path:
            writes.append(norm_path(path))
    return reads, writes


def script_column_set(data: dict) -> Set[str]:
    cols: Set[str] = set()
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
    for df in data.get("dataframes", []) or []:
        for c in df.get("columns", []) or []:
            if c.get("name"):
                cols.add(c["name"])
    return cols


def build_asset_graph(enriched_files: List[dict]) -> Tuple[Dict[str, Set[str]], Dict[str, Set[str]], Dict[str, Set[str]]]:
    """
    Returns:
      writers_by_path[path] = {scriptA, scriptB}
      readers_by_path[path] = {scriptX, scriptY}
      downstream_scripts[script] = {scripts that read what this script writes}
    """
    writers_by_path: Dict[str, Set[str]] = defaultdict(set)
    readers_by_path: Dict[str, Set[str]] = defaultdict(set)

    for data in enriched_files:
        script = data.get("script_name", "unknown_script")
        reads, writes = extract_assets(data)

        for p in reads:
            readers_by_path[p].add(script)
        for p in writes:
            writers_by_path[p].add(script)

    downstream_scripts: Dict[str, Set[str]] = defaultdict(set)
    for path, writers in writers_by_path.items():
        readers = readers_by_path.get(path, set())
        for w in writers:
            for r in readers:
                if r != w:
                    downstream_scripts[w].add(r)

    return writers_by_path, readers_by_path, downstream_scripts


def bfs_downstream_scripts(seed_scripts: Set[str], downstream_scripts: Dict[str, Set[str]], limit: int = 2000) -> List[str]:
    seen = set(seed_scripts)
    out: List[str] = []
    q = deque(sorted(seed_scripts))

    while q and len(out) < limit:
        s = q.popleft()
        for nxt in sorted(downstream_scripts.get(s, set())):
            if nxt in seen:
                continue
            seen.add(nxt)
            out.append(nxt)
            q.append(nxt)

    # include seeds at front for reporting (stable)
    return sorted(seed_scripts) + out


def gold_outputs_for_scripts(enriched_files: List[dict], scripts: Set[str]) -> List[str]:
    gold = []
    for data in enriched_files:
        script = data.get("script_name", "unknown_script")
        if script not in scripts:
            continue
        _, writes = extract_assets(data)
        for p in writes:
            if "/gold/" in p or p.rstrip("/").endswith("/gold") or "gold" in p.split("/"):
                gold.append(p)
    # dedup stable
    seen = set()
    out = []
    for p in gold:
        if p not in seen:
            seen.add(p)
            out.append(p)
    return out


def seed_scripts_for_column(enriched_files: List[dict], col: str) -> Set[str]:
    seeds = set()
    for data in enriched_files:
        script = data.get("script_name", "unknown_script")
        cols = script_column_set(data)
        if col in cols:
            seeds.add(script)
    return seeds


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


def build_evidence(
    hits: List[str],
    docs_by_id: Dict[str, Any],
    col_impacts: Dict[str, List[str]],
    reasons: Dict[Tuple[str, str], Set[str]],
    impacted_scripts: List[str],
    impacted_gold: List[str],
) -> str:
    parts: List[str] = []

    parts.append("=== DETERMINISTIC COLUMN IMPACT (GRAPH TRAVERSAL) ===")
    if not col_impacts:
        parts.append("No candidate column detected in question; column-impact traversal skipped.")
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

            parts.append("  Direct edges (1-hop) with reasons:")
            direct = sorted([dst for (src, dst) in reasons.keys() if src == col])
            if not direct:
                parts.append("   - None found")
            else:
                for dst in direct[:20]:
                    why = "; ".join(sorted(reasons.get((col, dst), set())))
                    parts.append(f"   - {col} -> {dst}: {why}")

    parts.append("\n=== DETERMINISTIC REPO IMPACT (ASSET BFS) ===")
    if not impacted_scripts:
        parts.append("No impacted scripts found from available asset metadata.")
    else:
        parts.append("Impacted scripts (seed + downstream via writes->reads):")
        for s in impacted_scripts[:MAX_SCRIPTS_SHOW]:
            parts.append(f" - {s}")
        if len(impacted_scripts) > MAX_SCRIPTS_SHOW:
            parts.append(f" - ... ({len(impacted_scripts)-MAX_SCRIPTS_SHOW} more)")

    if impacted_gold:
        parts.append("\nImpacted GOLD outputs (deterministic from impacted scripts writes):")
        for p in impacted_gold[:MAX_GOLD_SHOW]:
            parts.append(f" - {p}")
        if len(impacted_gold) > MAX_GOLD_SHOW:
            parts.append(f" - ... ({len(impacted_gold)-MAX_GOLD_SHOW} more)")
    else:
        parts.append("\nImpacted GOLD outputs: none found from available asset metadata.")

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

    docs, docs_by_id, index_ids, index = load_rag_store()

    if not OUTPUTS_DIR.exists():
        print("ERROR: outputs/ directory not found. Run the extractor pipeline first.")
        sys.exit(1)

    enriched_files = load_all_enriched(OUTPUTS_DIR)
    adj, reasons, known_cols = build_dependency_graph(enriched_files)

    # asset stitching
    writers_by_path, readers_by_path, downstream_scripts = build_asset_graph(enriched_files)

    print("\nLineage Q&A ready (HYBRID always).")
    print("Paste multi-line questions. Press ENTER twice to submit. Ctrl+C to exit.")

    while True:
        try:
            q = read_multiline_question()
            if not q:
                continue

            cols = extract_candidate_columns(q, known_cols, MAX_IMPACT_COLS)

            # deterministic column impacts
            col_impacts: Dict[str, List[str]] = {}
            for c in cols:
                col_impacts[c] = downstream_closure(adj, c)

            # deterministic repo impacts (scripts + gold outputs)
            seed_scripts = set()
            for c in cols:
                seed_scripts |= seed_scripts_for_column(enriched_files, c)

            impacted_scripts_list: List[str] = []
            impacted_gold: List[str] = []
            if seed_scripts:
                impacted_scripts_list = bfs_downstream_scripts(seed_scripts, downstream_scripts)
                impacted_gold = gold_outputs_for_scripts(enriched_files, set(impacted_scripts_list))

            # RAG retrieval
            q_vec = embed_text(client, embed_model, q)
            hits = retrieve_docs(index, q_vec, index_ids, docs_by_id, TOP_K)

            evidence = build_evidence(hits, docs_by_id, col_impacts, reasons, impacted_scripts_list, impacted_gold)
            answer = ask_llm(client, llm_model, q, evidence)

            print("\nANSWER:\n")
            print(answer)
            print("\n--- Debug ---")
            print(f"candidate_cols: {cols}")
            print(f"seed_scripts: {sorted(seed_scripts) if seed_scripts else []}")
            print(f"impacted_scripts: {len(impacted_scripts_list)}")
            print(f"impacted_gold: {len(impacted_gold)}")
            print(f"retrieved_docs: {len(hits)}")
            print("-------------")

        except KeyboardInterrupt:
            print("\nExiting.")
            break


if __name__ == "__main__":
    main()



