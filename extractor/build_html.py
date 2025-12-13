import json
from pathlib import Path

HTML_HEAD = """<!DOCTYPE html>
<html lang="en"><head>
<meta charset="UTF-8"/>
<title>Repo Lineage</title>
<script src="https://cdn.jsdelivr.net/npm/mermaid@10/dist/mermaid.min.js"></script>
<script>mermaid.initialize({startOnLoad:true,theme:"dark"});</script>
<style>
body{font-family:system-ui;background:#0f172a;color:#e5e7eb;margin:0}
header{padding:1rem 1.5rem;background:#020617;border-bottom:1px solid #1f2937}
main{padding:1rem 1.5rem}
.mermaid{background:#020617;padding:1rem;border-radius:8px;margin:.75rem 0; overflow:auto}
details{margin:.5rem 0}
summary{cursor:pointer;font-weight:600}
hr{border:0;border-top:1px solid #1f2937;margin:1rem 0}
small{color:#9ca3af}
</style>
</head><body>
<header>
<h1>Repo Metadata Lineage</h1>
<p><small>Expand a dataframe to see groupBy, aggregations, derived columns, joins, and column-to-column lineage.</small></p>
</header>
<main>
"""

HTML_TAIL = "</main></body></html>"


def safe_id(s: str) -> str:
    return "".join(ch if ch.isalnum() else "_" for ch in (s or ""))


def load_repo_links() -> dict:
    p = Path("outputs/repo_graph.json")
    if not p.exists():
        return {"links": []}
    return json.loads(p.read_text())


def get_script_name(item: dict, fallback_name: str) -> str:
    return item.get("script_name") or item.get("script") or fallback_name


def _asset_label(a: dict) -> str:
    return a.get("path") or a.get("table") or ""


def overview_graph(items, repo_links):
    lines = ["flowchart LR", ""]

    # script + asset nodes
    for item, fallback_name in items:
        script_name = get_script_name(item, fallback_name)
        sid = safe_id(script_name)
        lines.append(f'{sid}["{script_name}"]')

        for w in item.get("assets", {}).get("writes", []):
            lbl = _asset_label(w)
            if lbl:
                aid = safe_id("asset_" + lbl)
                lines.append(f'{aid}["WRITE\\n{lbl}"]')
                lines.append(f"{sid} --> {aid}")

        for r in item.get("assets", {}).get("reads", []):
            lbl = _asset_label(r)
            if lbl:
                aid = safe_id("asset_" + lbl)
                lines.append(f'{aid}["READ\\n{lbl}"]')
                lines.append(f"{aid} --> {sid}")

    # stitched cross-script links (writes -> reads)
    for lk in repo_links.get("links", []):
        asset = lk.get("asset", "")
        if not asset:
            continue
        a = safe_id("asset_" + asset)
        s_from = safe_id(lk.get("from_script", "from"))
        s_to = safe_id(lk.get("to_script", "to"))
        lines.append(f'{a}["ASSET\\n{asset}"]')
        lines.append(f"{s_from} --> {a} --> {s_to}")

    return "\n".join(lines)


def _join_key_to_str(j: dict) -> str:
    key = j.get("join_key", "")
    if isinstance(key, list):
        key_str = ",".join([str(x) for x in key])
    else:
        key_str = str(key) if key else ""

    if not key_str:
        jk = j.get("join_keys", [])
        if isinstance(jk, list):
            key_str = ",".join([str(x) for x in jk])
        else:
            key_str = str(jk) if jk else ""

    return key_str


def df_graph(df: dict) -> str:
    """
    Column-level graph for a single dataframe.

    Expected (from Bedrock enrichment) optional fields:
      - group_by: [col,...]
      - aggregations: [{out_col, func, in_col, expression},...]
      - derived: [{out_col, expression, derived_from},...]
      - joins: [{right_df, join_key|join_keys, join_type},...]
      - lineage: [
          {"target_col":"x", "sources":[{"df":"df1","col":"y"}, ...]}
        ]
      - columns: baseline fallback [{name, expression}, ...]
    """
    lines = ["flowchart TB"]
    dfname = df.get("df_name", "df")
    dfid = safe_id("DF_" + dfname)
    lines.append(f'{dfid}["{dfname}"]')

    # Track what we already rendered as a column to avoid duplicates
    seen_cols = set()

    # 1) Column-to-column lineage edges (preferred, if present)
    lineage = df.get("lineage") or []
    for e in lineage:
        tgt = str(e.get("target_col", "") or "")
        if not tgt:
            continue

        tgt_node = safe_id(dfname + "_col_" + tgt)
        lines.append(f'{tgt_node}["{tgt}"]')
        lines.append(f"{tgt_node} --> {dfid}")
        seen_cols.add(tgt)

        for src in (e.get("sources") or []):
            s_df = str(src.get("df", "") or "")
            s_col = str(src.get("col", "") or "")
            if not s_df or not s_col:
                continue
            src_node = safe_id(s_df + "_col_" + s_col)
            lines.append(f'{src_node}["{s_df}.{s_col}"]')
            lines.append(f"{src_node} --> {tgt_node}")

    # 2) groupBy operation + explicit group key columns (if provided)
    gb = df.get("group_by") or []
    if gb:
        gid = safe_id(dfname + "_groupby")
        gb_str = ", ".join([str(x) for x in gb])
        lines.append(f'{gid}["groupBy({gb_str})"]')

        for k in gb:
            k = str(k)
            if k in seen_cols:
                continue
            kid = safe_id(dfname + "_col_" + k)
            lines.append(f'{kid}["{k}"]')
            lines.append(f"{kid} --> {dfid}")
            seen_cols.add(k)

        lines.append(f"{gid} --> {dfid}")

    # 3) Aggregations (sum/avg/etc.)
    aggs = df.get("aggregations") or []
    for a in aggs:
        out_col = str(a.get("out_col", "") or "")
        if not out_col:
            continue

        expr = a.get("expression") or f'{a.get("func","")}({a.get("in_col","")})'
        expr = str(expr)

        aid = safe_id(dfname + "_agg_" + out_col)
        cid = safe_id(dfname + "_col_" + out_col)

        lines.append(f'{aid}["{expr}"]')
        lines.append(f'{cid}["{out_col}"]')
        lines.append(f"{aid} --> {cid} --> {dfid}")
        seen_cols.add(out_col)

    # 4) Derived columns
    deriv = df.get("derived") or []
    for d in deriv:
        out_col = str(d.get("out_col", "") or "")
        if not out_col:
            continue

        expr = str(d.get("expression", "") or "")
        did = safe_id(dfname + "_der_" + out_col)
        cid = safe_id(dfname + "_col_" + out_col)

        lines.append(f'{did}["{out_col} = {expr}"]')
        lines.append(f'{cid}["{out_col}"]')
        lines.append(f"{did} --> {cid} --> {dfid}")
        seen_cols.add(out_col)

    # 5) Joins
    joins = df.get("joins") or []
    for j in joins:
        right = str(j.get("right_df", "") or "")
        how = str(j.get("join_type", "") or "")
        key_str = _join_key_to_str(j)

        jid = safe_id(dfname + "_join_" + right + "_" + key_str)
        lines.append(f'{jid}["join {right} on {key_str} ({how})"]')
        lines.append(f"{jid} --> {dfid}")

    # 6) Fallback baseline columns (only for those not already rendered)
    for c in (df.get("columns") or []):
        name = str(c.get("name", "") or "")
        if not name or name in seen_cols:
            continue

        expr = str(c.get("expression", "") or "")
        cid = safe_id(dfname + "_basic_" + name)

        if expr:
            eid = safe_id(dfname + "_expr_" + name)
            lines.append(f'{eid}["{expr}"]')
            lines.append(f'{cid}["{name}"]')
            lines.append(f"{eid} --> {cid} --> {dfid}")
        else:
            lines.append(f'{cid}["{name}"]')
            lines.append(f"{cid} --> {dfid}")

        seen_cols.add(name)

    return "\n".join(lines)


def build():
    enriched_files = sorted(Path("outputs").glob("*.enriched.json"))
    if not enriched_files:
        raise SystemExit("No outputs/*.enriched.json found. Run bedrock_enrich first.")

    items = []
    for p in enriched_files:
        item = json.loads(p.read_text())
        fallback = p.stem.replace(".enriched", "") + ".py"
        items.append((item, fallback))

    repo_links = load_repo_links()

    html = [HTML_HEAD]

    html.append("<h2>1. Overview (Scripts + Assets + Cross-Script Links)</h2>")
    html.append('<div class="mermaid">')
    html.append(overview_graph(items, repo_links))
    html.append("</div>")

    html.append("<hr/>")
    html.append("<h2>2. DataFrames (expand)</h2>")

    for item, fallback_name in items:
        script_name = get_script_name(item, fallback_name)
        html.append(f"<h3>{script_name}</h3>")

        for df in item.get("dataframes", []):
            df_name = df.get("df_name", "df")
            html.append("<details>")
            html.append(f"<summary>{df_name} ({df.get('type','')})</summary>")
            html.append('<div class="mermaid">')
            html.append(df_graph(df))
            html.append("</div>")
            html.append("</details>")

    html.append(HTML_TAIL)
    Path("outputs/lineage_repo.html").write_text("\n".join(html))
    print("Wrote outputs/lineage_repo.html")


if __name__ == "__main__":
    build()
