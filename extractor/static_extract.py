import ast                      # Parse Python code into an AST (Abstract Syntax Tree)
import json                     # Write extracted lineage to JSON
import sys                      # Read CLI args and exit codes
from pathlib import Path        # File path utilities (cross-platform)
from typing import Any, Dict, List, Optional  # Type hints for clarity and safety


# ---------- Helpers ----------
# Helper functions keep the extraction logic clean and testable.

def _is_name(node: ast.AST, name: str) -> bool:
    """True if node is a variable name like `spark`."""
    return isinstance(node, ast.Name) and node.id == name


def _get_str(node: ast.AST) -> Optional[str]:
    """Return the Python string literal from AST node, else None."""
    if isinstance(node, ast.Constant) and isinstance(node.value, str):
        return node.value
    return None


def _attr_chain(node: ast.AST) -> List[str]:
    """
    For an expression like spark.read.parquet, returns ["spark", "read", "parquet"].
    For sess_agg.write.mode("overwrite").partitionBy(...).parquet(out), when called on
    the final .parquet attribute, returns ["sess_agg", "write", "mode", "partitionBy", "parquet"].

    This is used to recognize patterns like:
      - spark.read.parquet(...)
      - df.write....parquet(...)
    """
    parts: List[str] = []
    cur = node
    while isinstance(cur, ast.Attribute):
        parts.append(cur.attr)
        cur = cur.value
    if isinstance(cur, ast.Name):
        parts.append(cur.id)
    return list(reversed(parts))


def _flatten_calls(expr: ast.AST) -> List[ast.Call]:
    """
    Collect all Call nodes inside an expression.
    Example: df.write.mode(...).partitionBy(...).parquet(out)
    contains multiple calls; ast.walk finds them all.
    """
    calls: List[ast.Call] = []
    for n in ast.walk(expr):
        if isinstance(n, ast.Call):
            calls.append(n)
    return calls


def _safe_unparse(node: ast.AST) -> str:
    """
    Convert AST back into source-like string.
    Python 3.9+ supports ast.unparse; if anything fails, return empty string.
    """
    try:
        return ast.unparse(node)
    except Exception:
        return ""


def _read_text_best_effort(p: Path) -> str:
    """
    Read script text. Most are UTF-8. If decode fails, fall back to ignoring errors.
    """
    try:
        return p.read_text(encoding="utf-8")
    except UnicodeDecodeError:
        return p.read_text(errors="ignore")


def _iter_scripts(target: str, recursive: bool) -> List[Path]:
    """
    Supports:
      - file: return [file] if .py
      - dir: return *.py or recursive **/*.py depending on `recursive`
    """
    p = Path(target)
    if p.is_file():
        return [p] if p.suffix == ".py" else []
    if p.is_dir():
        globber = p.rglob("*.py") if recursive else p.glob("*.py")
        return sorted([x for x in globber if x.is_file()])
    return []


def _resolve_possible_path_arg(arg: ast.AST, assignments: Dict[str, str]) -> Optional[str]:
    """
    If the write path is a string literal, return it.
    If the write path is a variable name like `out`, try to resolve it from earlier
    assignments like: out = "s3a://..."
    Otherwise return None.
    """
    s = _get_str(arg)                 # case 1: parquet("s3a://...")
    if s:
        return s
    if isinstance(arg, ast.Name):     # case 2: parquet(out)
        return assignments.get(arg.id)
    return None


# ---------- Extractors ----------

def extract_lineage(script_path: str) -> Dict[str, Any]:
    """
    Extract a deterministic base lineage structure:
      - script_name
      - dataframe assignments (df_name)
      - simple column derivations (withColumn, agg alias, groupBy keys)
      - asset reads/writes when detectable
    """
    text = _read_text_best_effort(Path(script_path))  # read file contents
    tree = ast.parse(text)                            # parse into AST

    script_name = Path(script_path).name              # e.g., script_06_web_sessionization.py

    reads: List[Dict[str, str]] = []                  # list of {"format","path"}
    writes: List[Dict[str, str]] = []                 # list of {"format","path","df"}

    dfs: Dict[str, Dict[str, Any]] = {}               # df_name -> dataframe record

    assigned_order: List[str] = []                    # used to mark first df as source, last as sink

    # Track assignments of string literals like:
    #   out = "s3a://my-bucket/gold/web_sessions/"
    # so that parquet(out) can still be captured as a write path.
    assignments: Dict[str, str] = {}                  # var_name -> string value

    # Walk top-level statements in the script (not inside functions).
    # Note: If your scripts define everything inside def run(spark):, this only sees the def,
    # not the inside body. Your scripts DO define logic inside a function (run), so ideally
    # we should walk the entire tree, not just tree.body. We'll handle that below by walking tree.
    #
    # But we still want "assignment order" in a reasonable way, so we collect df assignments
    # across the whole tree.
    for node in ast.walk(tree):

        # (A) Capture simple string assignments: out = "s3a://..."
        if isinstance(node, ast.Assign) and len(node.targets) >= 1:
            if isinstance(node.targets[0], ast.Name):
                var = node.targets[0].id
                s = _get_str(node.value)
                if s:
                    assignments[var] = s

        # (B) Capture dataframe assignments: dfX = <expr>
        if isinstance(node, ast.Assign) and len(node.targets) >= 1:
            if isinstance(node.targets[0], ast.Name):
                df_name = node.targets[0].id
                assigned_order.append(df_name)

                # Initialize record for this df if missing
                if df_name not in dfs:
                    dfs[df_name] = {
                        "df_name": df_name,
                        "type": "intermediate",   # will be overwritten later for first/last df
                        "columns": []
                    }

                rhs = node.value
                calls = _flatten_calls(rhs)

                # (1) Detect reads: spark.read.<fmt>("path")
                for c in calls:
                    if isinstance(c.func, ast.Attribute):
                        chain = _attr_chain(c.func)
                        # Example chain: ["spark","read","parquet"]
                        if len(chain) >= 3 and chain[0] == "spark" and chain[1] == "read":
                            fmt = chain[2]
                            if c.args:
                                p = _resolve_possible_path_arg(c.args[0], assignments)
                                if p:
                                    reads.append({"format": fmt, "path": p})

                # (2) Detect withColumn("newcol", expr)
                for c in calls:
                    if isinstance(c.func, ast.Attribute) and c.func.attr == "withColumn":
                        if len(c.args) >= 2:
                            colname = _get_str(c.args[0])          # first arg is column name
                            expr_str = _safe_unparse(c.args[1])     # second arg is expression
                            if colname:
                                dfs[df_name]["columns"].append({
                                    "name": colname,
                                    "derived_from": [],            # filled later by postprocess/LLM
                                    "expression": expr_str,
                                    "transformation": "withColumn"
                                })

                # (3) Detect groupBy("k1","k2") keys
                for c in calls:
                    if isinstance(c.func, ast.Attribute) and c.func.attr == "groupBy":
                        keys: List[str] = []
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

                # (4) Detect agg(F.sum("amount").alias("total_amount"), ...)
                for c in calls:
                    if isinstance(c.func, ast.Attribute) and c.func.attr == "agg":
                        for a in c.args:
                            if isinstance(a, ast.Call) and isinstance(a.func, ast.Attribute) and a.func.attr == "alias":
                                alias_name = _get_str(a.args[0]) if a.args else None
                                inner = a.func.value                        # expression before .alias()
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

                # (5) Detect join(dfRight, on="key", how="left")
                for c in calls:
                    if isinstance(c.func, ast.Attribute) and c.func.attr == "join":
                        right_df = None
                        on_key = None
                        how = None
                        if c.args and isinstance(c.args[0], ast.Name):
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

        # (C) Capture writes even if inside function bodies
        #
        # We look for patterns like:
        #   sess_agg.write.mode("overwrite").partitionBy("customer_id").parquet(out)
        #
        # In AST, that is an Expr(Call(...)) where the final call is parquet(out).
        if isinstance(node, ast.Expr) and isinstance(node.value, ast.Call):
            c = node.value
            if isinstance(c.func, ast.Attribute):
                fmt = c.func.attr                    # parquet/csv/json/etc
                chain = _attr_chain(c.func)          # e.g., ["sess_agg","write","mode","partitionBy","parquet"]
                if c.args:
                    p = _resolve_possible_path_arg(c.args[0], assignments)
                    if p:
                        df_writer = chain[0] if chain else ""
                        writes.append({"format": fmt, "path": p, "df": df_writer})

    # Heuristic: first df assignment is "source", last is "sink"
    if assigned_order:
        if assigned_order[0] in dfs:
            dfs[assigned_order[0]]["type"] = "source"
        if assigned_order[-1] in dfs:
            dfs[assigned_order[-1]]["type"] = "sink"

    # Final deterministic base JSON for this script
    return {
        "script_name": script_name,
        "dataframes": list(dfs.values()),
        "assets": {"reads": reads, "writes": writes}
    }


def main(argv: List[str]) -> int:
    """
    CLI supports:
      python extractor/static_extract.py examples/
      python extractor/static_extract.py examples/ --recursive --out outputs
      python extractor/static_extract.py examples/script_x.py
    """
    target = "examples"              # default directory
    recursive = False                # default: not recursive
    out_dir = Path("outputs")        # default output folder

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
            target = a              # positional: file or folder
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


