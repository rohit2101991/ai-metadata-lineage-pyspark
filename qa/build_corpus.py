#!/usr/bin/env python3
"""
Build semantic corpus for lineage Q&A from enriched lineage JSONs.

Reads:
  outputs/*.enriched.json

Writes:
  qa/corpus.json

Corpus strategy:
- Script-level summary docs
- Deterministic lineage edge docs (source -> target)
- Bedrock enrichment docs (derived cols, aggs, joins, group_by, sql)

This dramatically increases recall and ensures new scripts are indexed.
"""

import json
import hashlib
from pathlib import Path
from typing import Dict, Any, List


OUT_DIR = Path("outputs")
CORPUS_PATH = Path("qa/corpus.json")


# -------------------------
# Helpers
# -------------------------

def stable_id(*parts: str) -> str:
    s = "||".join(p for p in parts if p)
    return hashlib.sha1(s.encode("utf-8")).hexdigest()


def safe_list(x):
    return x if isinstance(x, list) else []


def safe_dict(x):
    return x if isinstance(x, dict) else {}


# -------------------------
# Builders
# -------------------------

def build_script_doc(script: str, data: Dict[str, Any], src_file: Path) -> Dict[str, Any]:
    reads = safe_list(data.get("assets", {}).get("reads"))
    writes = safe_list(data.get("assets", {}).get("writes"))
    dfs = sorted({df.get("df_name") for df in safe_list(data.get("dataframes")) if df.get("df_name")})

    lines = [
        f"SCRIPT: {script}",
        f"SOURCE_FILE: {src_file.name}"
    ]

    if reads:
        lines.append("READS:")
        for r in reads:
            lines.append(f"- {r.get('format','')} {r.get('path','')}")

    if writes:
        lines.append("WRITES:")
        for w in writes:
            lines.append(f"- df={w.get('df','')} -> {w.get('format','')} {w.get('path','')}")

    if dfs:
        lines.append("DATAFRAMES:")
        for d in dfs:
            lines.append(f"- {d}")

    return {
        "id": stable_id("script", script),
        "text": "\n".join(lines),
        "meta": {"type": "script", "script": script}
    }


def build_lineage_edge_docs(script: str, data: Dict[str, Any]) -> List[Dict[str, Any]]:
    docs = []

    for df in safe_list(data.get("dataframes")):
        df_name = df.get("df_name")
        if not df_name:
            continue

        for i, edge in enumerate(safe_list(df.get("lineage"))):
            src = edge.get("source")
            tgt = edge.get("target")
            tr = edge.get("transformation", "")
            expr = edge.get("expression", "")

            if not src or not tgt:
                continue

            text = (
                f"SCRIPT: {script}\n"
                f"DATAFRAME: {df_name}\n"
                f"LINEAGE: {src} -> {tgt}\n"
                f"TRANSFORMATION: {tr}\n"
                f"EXPRESSION: {expr}"
            )

            docs.append({
                "id": stable_id("edge", script, df_name, src, tgt, str(i)),
                "text": text,
                "meta": {
                    "type": "lineage_edge",
                    "script": script,
                    "df": df_name,
                    "source": src,
                    "target": tgt,
                    "transformation": tr
                }
            })

    return docs


def build_bedrock_docs(script: str, data: Dict[str, Any]) -> List[Dict[str, Any]]:
    docs = []
    be = safe_dict(data.get("bedrock_enrichment"))

    # Derived columns
    for d in safe_list(be.get("derived_columns")):
        df = d.get("df")
        col = d.get("col")
        expr = d.get("expression", "")
        src_cols = d.get("source_cols", [])

        if not df or not col:
            continue

        text = (
            f"SCRIPT: {script}\n"
            f"DATAFRAME: {df}\n"
            f"DERIVED_COLUMN: {col}\n"
            f"EXPRESSION: {expr}\n"
            f"SOURCE_COLUMNS: {src_cols}"
        )

        docs.append({
            "id": stable_id("derived", script, df, col),
            "text": text,
            "meta": {"type": "derived", "script": script, "df": df, "column": col}
        })

    # Aggregations
    for a in safe_list(be.get("aggregations")):
        df = a.get("df")
        alias = a.get("alias")
        expr = a.get("expr", "")
        src_cols = a.get("source_cols", [])

        if not df or not alias:
            continue

        text = (
            f"SCRIPT: {script}\n"
            f"DATAFRAME: {df}\n"
            f"AGGREGATION: {expr} AS {alias}\n"
            f"SOURCE_COLUMNS: {src_cols}"
        )

        docs.append({
            "id": stable_id("agg", script, df, alias),
            "text": text,
            "meta": {"type": "aggregation", "script": script, "df": df, "column": alias}
        })

    # Joins
    for j in safe_list(be.get("joins")):
        df = j.get("df")
        left = j.get("left_df")
        right = j.get("right_df")
        how = j.get("how")
        on = j.get("on", [])

        if not df or not left or not right:
            continue

        text = (
            f"SCRIPT: {script}\n"
            f"RESULT_DF: {df}\n"
            f"JOIN: {left} {how} JOIN {right}\n"
            f"ON: {on}"
        )

        docs.append({
            "id": stable_id("join", script, df, left, right),
            "text": text,
            "meta": {"type": "join", "script": script, "df": df}
        })

    return docs


# -------------------------
# Main
# -------------------------

def main():
    corpus: List[Dict[str, Any]] = []

    files = sorted(OUT_DIR.glob("*.enriched.json"))
    if not files:
        print("No enriched files found.")
        CORPUS_PATH.write_text("[]")
        return

    for p in files:
        data = json.loads(p.read_text())
        script = data.get("script_name") or p.stem.replace(".enriched", "") + ".py"

        corpus.append(build_script_doc(script, data, p))
        corpus.extend(build_lineage_edge_docs(script, data))
        corpus.extend(build_bedrock_docs(script, data))

    CORPUS_PATH.parent.mkdir(parents=True, exist_ok=True)
    CORPUS_PATH.write_text(json.dumps(corpus, indent=2))

    print(f"Wrote {len(corpus)} corpus documents to {CORPUS_PATH}")


if __name__ == "__main__":
    main()

