import json
from pathlib import Path

def norm(p: str) -> str:
    return (p or "").strip().rstrip("/")

def main():
    enriched_files = sorted(Path("outputs").glob("*.enriched.json"))
    if not enriched_files:
        raise SystemExit("No outputs/*.enriched.json found. Run bedrock_enrich first.")

    items = [json.loads(p.read_text()) for p in enriched_files]

    writes = {}
    for it in items:
        for w in it.get("assets", {}).get("writes", []):
            p = norm(w.get("path",""))
            if p:
                writes[p] = it.get("script_name","")

    links = []
    for it in items:
        for r in it.get("assets", {}).get("reads", []):
            p = norm(r.get("path",""))
            if p and p in writes:
                links.append({"from_script": writes[p], "to_script": it.get("script_name",""), "asset": p})

    out = {"links": links}
    Path("outputs/repo_graph.json").write_text(json.dumps(out, indent=2))
    print(f"Wrote outputs/repo_graph.json with {len(links)} links")

if __name__ == "__main__":
    main()
