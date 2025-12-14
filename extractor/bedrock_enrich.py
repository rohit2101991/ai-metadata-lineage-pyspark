#!/usr/bin/env python3
"""
Bedrock enrichment for PySpark lineage (Amazon Nova via messages schema).

Mode A (single):
  python extractor/bedrock_enrich.py examples/script_x.py outputs/script_x.json

Mode B (batch):
  python extractor/bedrock_enrich.py examples/ outputs/
"""

from __future__ import annotations          # allows forward type annotations

import json                                 # read/write JSON files
import re                                   # strip markdown fences / extract JSON block
import sys                                  # CLI and stderr output
from pathlib import Path                    # file paths
from typing import Any, Dict, List, Tuple   # type hints

import boto3                                # AWS SDK for calling Bedrock


# ----------------------------
# JSON extraction robustness
# ----------------------------

def extract_json_object(text: str) -> Dict[str, Any]:
    """
    Bedrock sometimes returns:
      - pure JSON
      - JSON wrapped in ```json fences
      - extra commentary + JSON block
    This function makes extraction robust.
    """
    if not text or not text.strip():
        raise ValueError("Bedrock returned empty text (cannot parse JSON).")

    t = text.strip()

    # Strip markdown fences like ```json ... ```
    t = re.sub(r"^```json\s*", "", t, flags=re.IGNORECASE).strip()
    t = re.sub(r"^```\s*", "", t).strip()
    t = re.sub(r"\s*```$", "", t).strip()

    # Try direct JSON parse first
    try:
        obj = json.loads(t)
        if isinstance(obj, dict):
            return obj
    except Exception:
        pass

    # If direct parse fails, find first {...} block
    start = t.find("{")
    end = t.rfind("}")
    if start == -1 or end == -1 or end <= start:
        preview = t[:400].replace("\n", "\\n")
        raise ValueError(f"Model response does not contain a JSON object. Preview: {preview}")

    return json.loads(t[start:end + 1])


# ----------------------------
# Config
# ----------------------------

def load_cfg(cfg_path: str = "config.json") -> Tuple[str, str]:
    """
    Supports either:
      {"aws_region": "...", "llm_model_id": "..."}
    or older:
      {"region": "...", "model_id": "..."}
    """
    cfg = json.loads(Path(cfg_path).read_text(encoding="utf-8"))

    region = cfg.get("aws_region") or cfg.get("region")
    model_id = cfg.get("llm_model_id") or cfg.get("model_id")

    if not region:
        raise ValueError("config.json missing aws_region (or region).")
    if not model_id:
        raise ValueError("config.json missing llm_model_id (or model_id).")

    return region, model_id


# ----------------------------
# Bedrock invocation (Nova messages schema)
# ----------------------------

def bedrock_nova_invoke_text(
    region: str,
    model_id: str,
    prompt: str,
    max_tokens: int = 2500,
    temperature: float = 0.0,
    top_p: float = 0.9
) -> str:
    """
    Nova requires a `messages` schema:
      {"messages":[{"role":"user","content":[{"text":"..."}]}], "inferenceConfig":{...}}
    """
    client = boto3.client("bedrock-runtime", region_name=region)

    body = {
        "messages": [
            {
                "role": "user",
                "content": [{"text": prompt}]
            }
        ],
        "inferenceConfig": {
            "maxTokens": max_tokens,
            "temperature": temperature,
            "topP": top_p
        }
    }

    resp = client.invoke_model(
        modelId=model_id,
        body=json.dumps(body).encode("utf-8"),
        accept="application/json",
        contentType="application/json",
    )

    raw = resp["body"].read()
    payload = json.loads(raw)

    # Expected Nova output shape
    try:
        return payload["output"]["message"]["content"][0]["text"]
    except Exception:
        pass

    # Defensive fallbacks
    try:
        return payload["message"]["content"][0]["text"]
    except Exception:
        pass

    # Another fallback: join multiple text blocks
    try:
        blocks = payload.get("output", {}).get("message", {}).get("content", [])
        texts: List[str] = []
        for b in blocks:
            if isinstance(b, dict) and "text" in b:
                texts.append(b["text"])
        if texts:
            return "\n".join(texts)
    except Exception:
        pass

    # Last resort: return entire payload as string for debugging
    return json.dumps(payload)


def bedrock_invoke_json(region: str, model_id: str, prompt: str) -> Dict[str, Any]:
    """
    Invoke Nova, then parse JSON from its text response.
    """
    text = bedrock_nova_invoke_text(region, model_id, prompt)
    return extract_json_object(text)


# ----------------------------
# Enrichment logic
# ----------------------------

def build_prompt(script_text: str, base_json: Dict[str, Any]) -> str:
    """
    Create a strict prompt that asks Nova to return a specific JSON schema.
    We provide:
      - the PySpark script text
      - the deterministic base JSON extracted by static_extract.py
    """
    schema = {
        "group_by": [{"df": "df_name", "keys": ["col1", "col2"]}],
        "aggregations": [{"df": "df_name", "expr": "sum(amount)", "source_cols": ["amount"], "alias": "total_amount"}],
        "derived_columns": [{"df": "df_name", "col": "new_col", "expression": "colA + colB", "source_cols": ["colA", "colB"]}],
        "joins": [{"df": "result_df", "left_df": "dfL", "right_df": "dfR", "how": "left",
                   "on": ["key1"], "conditions": "dfL.key1 = dfR.key1"}],
        "sql_blocks": [{"name": "block_name", "sql": "WITH ... SELECT ...", "tables": ["t1"], "columns": ["c1"]}]
    }

    script_excerpt = script_text if len(script_text) <= 12000 else (script_text[:12000] + "\n\n# TRUNCATED\n")

    base_excerpt = json.dumps(base_json, indent=2)
    if len(base_excerpt) > 12000:
        base_excerpt = base_excerpt[:12000] + "\n\n# TRUNCATED\n"

    return f"""
You are a metadata lineage enrichment engine for PySpark.

Return ONLY valid JSON. No markdown fences. No commentary.

Infer and output:
1) groupBy keys (df, keys)
2) aggregations (df, expr, source_cols, alias)
3) derived columns (df, col, expression, source_cols)
4) joins (result df, left/right df, how, on keys, conditions)
5) spark.sql blocks (name/sql + best-effort tables/columns)

Output must match this JSON schema shape:
{json.dumps(schema, indent=2)}

PYSPARK SCRIPT:
{script_excerpt}

STATIC EXTRACTION JSON:
{base_excerpt}

Now return the enrichment JSON only.
""".strip()


def enrich_one(script_path: Path, base_json_path: Path, out_dir: Path) -> Path:
    """
    Enrich one script:
      - read config for Bedrock region/model
      - read script text
      - read static extraction json
      - call Bedrock to get enrichment json
      - merge enrichment into base json
      - write <script>.enriched.json
    """
    region, model_id = load_cfg("config.json")

    script_text = script_path.read_text(encoding="utf-8", errors="ignore")
    base_json = json.loads(base_json_path.read_text(encoding="utf-8"))

    prompt = build_prompt(script_text, base_json)
    enrichment = bedrock_invoke_json(region=region, model_id=model_id, prompt=prompt)

    # Merge: keep static truth, add enrichment as a nested object
    merged = dict(base_json)
    merged["bedrock_enrichment"] = enrichment
    merged["enriched_from"] = {
        "script": str(script_path),
        "base_json": str(base_json_path),
        "model_id": model_id,
        "region": region
    }

    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"{script_path.stem}.enriched.json"
    out_path.write_text(json.dumps(merged, indent=2), encoding="utf-8")
    print(f"Wrote {out_path}")
    return out_path


# ----------------------------
# CLI
# ----------------------------

def iter_py_files(p: Path) -> List[Path]:
    """
    Get .py files (non-recursive) from a directory, or return the file itself.
    """
    if p.is_file() and p.suffix == ".py":
        return [p]
    if p.is_dir():
        return sorted([x for x in p.glob("*.py") if x.is_file()])
    return []


def main(argv: List[str]) -> int:
    """
    Supports:
      python extractor/bedrock_enrich.py <script.py> <outputs/base.json>
      python extractor/bedrock_enrich.py <examples_dir> <outputs_dir>
    """
    if len(argv) < 3:
        print(
            "Usage:\n"
            "  python extractor/bedrock_enrich.py <script.py> <outputs/base.json>\n"
            "  python extractor/bedrock_enrich.py <examples_dir> <outputs_dir>\n",
            file=sys.stderr,
        )
        return 2

    a1 = Path(argv[1])

    # Mode A: single script + base json
    if a1.is_file() and a1.suffix == ".py":
        script = a1
        base_json = Path(argv[2])
        if not base_json.exists():
            print(f"ERROR: base json not found: {base_json}", file=sys.stderr)
            return 1
        enrich_one(script, base_json, base_json.parent)
        return 0

    # Mode B: batch (directory)
    if a1.is_dir():
        examples_dir = a1
        outputs_dir = Path(argv[2])
        if not outputs_dir.exists():
            print(f"ERROR: outputs dir not found: {outputs_dir}", file=sys.stderr)
            return 1

        scripts = iter_py_files(examples_dir)
        if not scripts:
            print(f"ERROR: no .py scripts found in {examples_dir}", file=sys.stderr)
            return 1

        ok = 0
        fail = 0
        for s in scripts:
            base_json = outputs_dir / f"{s.stem}.json"
            if not base_json.exists():
                print(f"SKIP: missing base json for {s.name} at {base_json}")
                continue
            try:
                enrich_one(s, base_json, outputs_dir)
                ok += 1
            except Exception as e:
                print(f"FAILED: {s.name} -> {e}", file=sys.stderr)
                fail += 1

        print(f"Done. Enriched OK={ok}, failed={fail}.")
        return 0 if ok > 0 and fail == 0 else (0 if ok > 0 else 1)

    print("ERROR: invalid arguments.", file=sys.stderr)
    return 2


if __name__ == "__main__":
    raise SystemExit(main(sys.argv))


