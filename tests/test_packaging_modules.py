"""Packaging guards: every local module the app imports is copied into the
desktop bundle, and running from source never sends Streamlit usage stats."""
import ast
import pathlib
import re
import tomllib

REPO = pathlib.Path(__file__).resolve().parent.parent
BUILD = (REPO / "desktop" / "build_dmg.sh").read_text()
LOCAL = {p.stem for p in REPO.glob("*.py")}


def _local_imports(path):
    """Top-level local modules a file imports, including inside functions."""
    found = set()
    for node in ast.walk(ast.parse(path.read_text())):
        if isinstance(node, ast.Import):
            names = [a.name for a in node.names]
        elif isinstance(node, ast.ImportFrom) and node.level == 0 and node.module:
            names = [node.module]
        else:
            continue
        found |= {n.split(".")[0] for n in names} & LOCAL
    return found


def _app_modules():
    seen, todo = set(), ["app"]
    while todo:
        mod = todo.pop()
        if mod in seen:
            continue
        seen.add(mod)
        todo += _local_imports(REPO / f"{mod}.py")
    return seen


def _packaged():
    """File names build_dmg.sh copies, however it lists them: a `for f in` list
    or individual cp lines."""
    return set(re.findall(r"[\w./-]+\.py\b", BUILD))


def test_every_imported_module_is_packaged():
    missing = {m + ".py" for m in _app_modules()} - _packaged()
    assert not missing, f"build_dmg.sh does not copy: {sorted(missing)}"


def test_scanner_sees_the_insights_chain():
    assert {"ui_insights", "learned", "insights"} <= _app_modules()


def test_build_script_ships_the_sample_data():
    # Run by path, never imported, so the import scan cannot see it.
    assert "insights_worker.py" in _packaged()
    SHIPPED_DATA = [
        "data/ingredients_template.csv",
        "data/samples/yogurt_pilot/Yogurt pilot - Round 6 to score.json",
        "data/samples/yogurt_pilot/Yogurt pilot - all 10 Rounds.json",
        "data/samples/yogurt_pilot/Round 6 results (pilot).xlsx",
    ]
    for rel in SHIPPED_DATA:
        assert rel.split("/")[-1] in BUILD, rel
    assert "sample_ingredients.csv" not in BUILD


def test_source_runs_disable_usage_stats():
    cfg = tomllib.loads((REPO / ".streamlit" / "config.toml").read_text())
    assert cfg["browser"]["gatherUsageStats"] is False
    assert cfg["server"]["address"] == "127.0.0.1"
