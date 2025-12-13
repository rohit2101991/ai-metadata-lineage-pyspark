import json
from pathlib import Path
import boto3

SYSTEM = """You are a data lineage expert for PySpark.
Return STRICT JSON only. No markdown. No commentary.

You will be given:
1) The PySpark script
2) A baseline lineage JSON extracted by a static parser

Enrich + correct the baseline JSON with:
- group_by keys for each df (if applicable)
- aggregations: [{out_col, func, in_col, expression}]
- derived columns: [{out_col, expression, derived_from}]
- join details: right_df, join_key(s), join_type
- assets.reads/writes (S3 paths or table names) in canonical form
Preserve existing fields, add new ones.

Add these fields per dataframe (if applicable):
{
  "group_by": [],
  "aggregations": [],
  "derived": [],
  "joins": [],
  "lineage": [
    {"target_col": "col_name", "sources": [{"df":"dfX","col":"colY"}]}
  ]
}

Rules for lineage:
- For carried-through columns, map to the upstream df/col (e.g., df1.customer_id -> dfAgg.customer_id).
- For aggregated columns, map to the input metric column(s) used (e.g., sum(amount) -> sources:[df1.amount]).
- For derived columns, map to the columns referenced in the expression (e.g., big_amount -> sources:[dfAgg.total_amount, dfAgg.total_revenue]).

{
  "group_by": [],
  "aggregations": [],
  "derived": [],
  "joins": []
}

Return STRICT JSON only.
"""

def enrich(script_path: str, base_json_path: str, config_path: str = "config.json"):
    cfg = json.loads(Path(config_path).read_text())
    region = cfg["region"]
    model_id = cfg["model_id"]

    code = Path(script_path).read_text()
    base = json.loads(Path(base_json_path).read_text())

    client = boto3.client("bedrock-runtime", region_name=region)

    user_text = (
        "BASELINE_JSON:\n"
        + json.dumps(base)
        + "\n\nPYSPARK_SCRIPT:\n"
        + code
        + "\n\nReturn STRICT JSON only."
    )

    resp = client.converse(
        modelId=model_id,
        system=[{"text": SYSTEM}],
        messages=[{"role": "user", "content": [{"text": user_text}]}],
        inferenceConfig={"maxTokens": 2500, "temperature": 0}
    )

    text = resp["output"]["message"]["content"][0]["text"]
    return json.loads(text)

if __name__ == "__main__":
    import sys
    script = sys.argv[1]
    base_json = sys.argv[2]
    enriched = enrich(script, base_json)
    out_path = Path("outputs") / (Path(script).stem + ".enriched.json")
    out_path.write_text(json.dumps(enriched, indent=2))
    print("Wrote", out_path)
