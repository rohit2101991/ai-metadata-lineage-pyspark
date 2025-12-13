import json
from pathlib import Path
from collections import defaultdict

def _as_list(x):
    if x is None:
        return []
    if isinstance(x, list):
        return x
    return [x]

def postprocess_file(path: Path):
    data = json.loads(path.read_text())

    # Build quick lookup of dataframes
    dfs = {d["df_name"]: d for d in data.get("dataframes", [])}
    df_order = [d["df_name"] for d in data.get("dataframes", [])]

    # Heuristic: if df has assets.reads, treat that as its upstream dataset label.
    # We'll map "df_name" -> "upstream_df_label" to reference in lineage edges.
    # Example: df3 reads curated/customer_agg => upstream label "df3" still, but this helps if you later want dataset-level links.
    upstream_reads = {dfn: (dfs[dfn].get("assets", {}).get("reads", []) if dfn in dfs else []) for dfn in dfs}

    # Determine "inputs" for each dataframe based on transformations:
    # - If df has joins with right_df, inputs include right_df and some left df.
    # - Left df is ambiguous; we assume previous df in script order is the left input.
    inputs = defaultdict(list)  # df_name -> [input_df_names]
    for i, dfn in enumerate(df_order):
        d = dfs[dfn]
        joins = d.get("joins") or []
        if joins:
            # assume left is previous df in order (common in scripts)
            if i > 0:
                inputs[dfn].append(df_order[i-1])
            for j in joins:
                r = j.get("right_df")
                if r:
                    inputs[dfn].append(r)

    # Default: if no explicit inputs, carry from previous df if it exists and df is not a pure source
    for i, dfn in enumerate(df_order):
        d = dfs[dfn]
        if d.get("type") != "source" and not inputs[dfn] and i > 0:
            inputs[dfn].append(df_order[i-1])

    # Build lineage per dataframe
    for dfn in df_order:
        d = dfs[dfn]
        lineage = []

        # 1) Derived columns based on columns[].derived_from
        for c in d.get("columns") or []:
            tgt = c.get("name")
            if not tgt:
                continue
            sources = []
            for src_col in _as_list(c.get("derived_from")):
                # Find which input df likely contains this src_col.
                # Heuristic: prefer same df if already created (chained derivations),
                # else search input dfs.
                src_col = str(src_col)

                # same df (for chained derived cols like margin -> is_profitable)
                sources.append({"df": dfn, "col": src_col})

            lineage.append({"target_col": tgt, "sources": sources})

        # 2) Join keys: add mapping from left & right input dfs join keys to target join key
        for j in d.get("joins") or []:
            key = j.get("join_key") or j.get("join_keys")
            keys = _as_list(key)
            right_df = j.get("right_df")
            # left df assumed first input if present
            left_df = inputs[dfn][0] if inputs[dfn] else None

            for k in keys:
                k = str(k)
                sources = []
                if left_df:
                    sources.append({"df": left_df, "col": k})
                if right_df:
                    sources.append({"df": right_df, "col": k})
                lineage.append({"target_col": k, "sources": sources})

        # 3) Group by keys (if present): carry-through columns
        gb = d.get("group_by") or []
        for k in gb:
            k = str(k)
            srcs = []
            # group keys usually come from first input
            if inputs[dfn]:
                srcs.append({"df": inputs[dfn][0], "col": k})
            lineage.append({"target_col": k, "sources": srcs})

        # 4) Aggregations: map out_col to in_col from first input
        for a in d.get("aggregations") or []:
            out_col = a.get("out_col")
            in_col = a.get("in_col")
            if not out_col or not in_col:
                continue
            srcs = []
            if inputs[dfn]:
                srcs.append({"df": inputs[dfn][0], "col": str(in_col)})
            lineage.append({"target_col": str(out_col), "sources": srcs})

        # De-duplicate lineage entries by target_col+sources
        seen = set()
        dedup = []
        for e in lineage:
            tgt = e.get("target_col","")
            srcs = tuple(sorted((s.get("df",""), s.get("col","")) for s in e.get("sources") or []))
            key = (tgt, srcs)
            if key in seen:
                continue
            seen.add(key)
            dedup.append(e)

        d["lineage"] = dedup

    path.write_text(json.dumps(data, indent=2))
    print(f"Updated lineage in {path}")

def main():
    for p in Path("outputs").glob("*.enriched.json"):
        postprocess_file(p)

if __name__ == "__main__":
    main()
