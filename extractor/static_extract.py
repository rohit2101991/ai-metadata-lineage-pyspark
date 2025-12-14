import ast
import json
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional

# ---------- Helpers ----------

def _is_name(node: ast.AST, name: str) -> bool:
    return isinstance(node, ast.Name) and node.id == name

def _get_str(node: ast.AST) -> Optional[str]:
    if isinstance(node, ast.Constant) and isinstance(node.value, str):
        return node.value
    return None

def _attr_chain(node: ast.AST) -> List[str]:
    """
    For an expression like spark.read.parquet, returns ["spark", "read", "parquet"].
    For chained calls, returns the attribute names for the callable part.
    """
    parts = []
    cur = node
    while isinstance(cur, ast.Attribute):
        parts.append(cur.attr)
        cur = cur.value
    if isinstance(cur, ast.Name):
        parts.append(cur.id)
    return list(reversed(parts))

def _flatten_calls(expr: ast.AST) -> List[ast.Call]:
    """Collect all Call nodes inside an expression."""
    calls = []
    for n in ast.walk(expr):
        if isinstance(n, ast.Call):
            calls.append(n)
    return calls

def _safe_unparse(node: ast.AST) -> str:
    try:
        return ast.unparse(node)  # Python 3.9+
    except Exception:
        return ""

def _read_text_best_effort(p: Path) -> str:
    # Most PySpark scripts will be UTF-8; keep a safe fallback.
    try:
        return p.read_text(encoding="utf-8")
    except UnicodeDecodeError:
        return p.read_text(errors="ignore")

def _iter_scripts(target: str, recursive: bool) -> List[Path]:
    p = Path(target)
    if p.is_file():
        return [p] if p.suffix == ".py" else []
    if p.is_dir():
        globber = p.rglob("*.py") if recursive else p.glob("*.py")
        return sorted([x for x in globber if x.is_file()])
    return []


# ---------- Extractors ----------

def extract_lineage(script_path: str) -> Dict[str, Any]:
    text = _read_text_best_effort(Path(script_path))
    tree = ast.parse(text)

    script_name = Path(script_path).name
    reads: List[Dict[str, str]] = []
    writes: List[Dict[str, str]] = []

    # dataframes map: df_name -> record
    dfs: Dict[str, Dict[str, Any]] = {}

    # Track assignment order to mark source/sink heuristically
    assigned_order: List[str] = []

    # 1) Find dataframe assignments: dfX = <expr>
    for node in tree.body:
        if isinstance(node, ast.Assign) and len(node.targets) >= 1:
            # Only handle simple names: df = ...
            if isinstance(node.targets[0], ast.Name):
                df_name = node.targets[0].id
                assigned_order.append(df_name)
                if df_name not in dfs:
                    dfs[df_name] = {
                        "df_name": df_name,
                        "type": "intermediate",
                        "columns": []
                    }

                # Look inside RHS for Spark transformations
                rhs = node.value
                calls = _flatten_calls(rhs)

                # Detect spark.read.<fmt>("path")
                for c in calls:
                    if isinstance(c.func, ast.Attribute):
                        chain = _attr_chain(c.func)
                        # spark.read.parquet(...)
                        if len(chain) >= 3 and chain[0] == "spark" and chain[1] == "read":
                            fmt = chain[2]
                            if c.args:
                                p = _get_str(c.args[0])
                                if p:
                                    reads.append({"format": fmt, "path": p})

                # Detect withColumn("newcol", expr)
                for c in calls:
                    if isinstance(c.func, ast.Attribute) and c.func.attr == "withColumn":
                        # args: ("colname", expr)
                        if len(c.args) >= 2:
                            colname = _get_str(c.args[0])
                            expr_str = _safe_unparse(c.args[1])
                            if colname:
                                dfs[df_name]["columns"].append({
                                    "name": colname,
                                    "derived_from": [],
                                    "expression": expr_str,
                                    "transformation": "withColumn"
                                })

                # Detect groupBy keys
                for c in calls:
                    if isinstance(c.func, ast.Attribute) and c.func.attr == "groupBy":
                        keys = []
                        for a in c.args:
                            s = _get_str(a)
                            if s:
                                keys.append(s)
                        for k in keys:
                            dfs[df_name]["columns"].append({
                                "name": k,
                                "derived_from": [k],
                                "expression": "",
                                "transformation": "groupByKey"
                            })

                # Detect agg(sum(...).alias("x"), ...)
                for c in calls:
                    if isinstance(c.func, ast.Attribute) and c.func.attr == "agg":
                        # args can be: F.sum("amount").alias("total_amount")
                        for a in c.args:
                            # look for .alias("name")
                            if isinstance(a, ast.Call) and isinstance(a.func, ast.Attribute) and a.func.attr == "alias":
                                alias_name = _get_str(a.args[0]) if a.args else None
                                inner = a.func.value  # the thing being aliased
                                # inner might be F.sum("amount")
                                src_col = None
                                expr_str = _safe_unparse(inner)
                                if isinstance(inner, ast.Call) and isinstance(inner.func, ast.Attribute):
                                    if inner.func.attr == "sum" and inner.args:
                                        src_col = _get_str(inner.args[0])
                                if alias_name:
                                    dfs[df_name]["columns"].append({
                                        "name": alias_name,
                                        "derived_from": [src_col] if src_col else [],
                                        "expression": expr_str,
                                        "transformation": "aggregation"
                                    })

                # Detect join(dfRight, on="key", how="left")
                for c in calls:
                    if isinstance(c.func, ast.Attribute) and c.func.attr == "join":
                        right_df = None
                        on_key = None
                        how = None
                        if c.args:
                            if isinstance(c.args[0], ast.Name):
                                right_df = c.args[0].id
                        for kw in c.keywords:
                            if kw.arg == "on":
                                on_key = _get_str(kw.value)
                            if kw.arg == "how":
                                how = _get_str(kw.value)
                        dfs[df_name]["columns"].append({
                            "name": f"__join__{right_df or 'unknown'}",
                            "derived_from": [on_key] if on_key else [],
                            "expression": f"join({right_df}) on {on_key} ({how})",
                            "transformation": "join"
                        })

        # 2) Capture writes: df.write.mode(...).parquet("path")
        # Writes often appear as Expr statements: dfCurated.write.mode(...).parquet("s3://...")
        if isinstance(node, ast.Expr) and isinstance(node.value, ast.Call):
            c = node.value
            if isinstance(c.func, ast.Attribute):
                fmt = c.func.attr  # parquet/csv/json/etc
                chain = _attr_chain(c.func)
                if c.args:
                    p = _get_str(c.args[0])
                    if p:
                        df_writer = chain[0] if chain else ""
                        writes.append({"format": fmt, "path": p, "df": df_writer})

    # mark source/sink heuristics
    if assigned_order:
        if assigned_order[0] in dfs:
            dfs[assigned_order[0]]["type"] = "source"
        if assigned_order[-1] in dfs:
            dfs[assigned_order[-1]]["type"] = "sink"

    return {
        "script_name": script_name,
        "dataframes": list(dfs.values()),
        "assets": {"reads": reads, "writes": writes}
    }


def main(argv: List[str]) -> int:
    # Compatible with your current CLI pattern:
    #   python extractor/static_extract.py examples/
    # but also supports:
    #   python extractor/static_extract.py examples/ --recursive --out outputs
    target = "examples"
    recursive = False
    out_dir = Path("outputs")

    args = argv[1:]
    i = 0
    while i < len(args):
        a = args[i]
        if a == "--recursive":
            recursive = True
        elif a == "--out":
            if i + 1 >= len(args):
                print("ERROR: --out requires a directory path", file=sys.stderr)
                return 2
            out_dir = Path(args[i + 1])
            i += 1
        else:
            # first positional target
            target = a
        i += 1

    scripts = _iter_scripts(target, recursive)
    if not scripts:
        print(f"No .py scripts found under: {target}", file=sys.stderr)
        return 1

    out_dir.mkdir(parents=True, exist_ok=True)

    wrote = 0
    for script_path in scripts:
        out = extract_lineage(str(script_path))
        out_path = out_dir / (script_path.stem + ".json")
        out_path.write_text(json.dumps(out, indent=2), encoding="utf-8")
        wrote += 1
        print("Wrote", out_path)

    print(f"Done. Extracted {wrote} script(s).")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv))


