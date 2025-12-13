import json
from pathlib import Path

docs = []

for p in Path("outputs").glob("*.enriched.json"):
    data = json.loads(p.read_text())
    script = data.get("script_name")

    for df in data.get("dataframes", []):
        df_name = df["df_name"]

        # Columns
        for c in df.get("columns", []):
            docs.append({
                "id": f"{script}:{df_name}:{c['name']}",
                "text": f"""
Script {script}, dataframe {df_name}, column {c['name']}.
Expression: {c.get('expression','')}.
Derived from: {c.get('derived_from','')}.
""",
                "meta": {"script": script, "df": df_name, "column": c["name"]}
            })

        # Aggregations
        for a in df.get("aggregations", []):
            docs.append({
                "id": f"{script}:{df_name}:agg:{a['out_col']}",
                "text": f"""
Script {script} aggregates {a['in_col']} using {a['func']}
to produce {a['out_col']} in dataframe {df_name}.
""",
                "meta": {"script": script, "df": df_name}
            })

        # Assets
        for r in df.get("assets", {}).get("reads", []):
            docs.append({
                "id": f"{script}:read:{r['path']}",
                "text": f"Script {script} reads data from {r['path']}",
                "meta": {"script": script, "asset": r["path"]}
            })

        for w in df.get("assets", {}).get("writes", []):
            docs.append({
                "id": f"{script}:write:{w['path']}",
                "text": f"Script {script} writes data to {w['path']}",
                "meta": {"script": script, "asset": w["path"]}
            })

Path("qa/corpus.json").write_text(json.dumps(docs, indent=2))
print(f"Wrote {len(docs)} corpus documents")
