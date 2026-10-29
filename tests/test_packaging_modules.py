"""Packaging guards: desktop/shared/bundle_manifest.txt lists every local
module the app imports and nothing it doesn't, the Mac bundle's golden
listing follows from it, and running from source never sends Streamlit
usage stats."""
import ast
import pathlib
import re
import sys
import tomllib

REPO = pathlib.Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO / "desktop"))
from bundle_listing import expected_bundle_listing, read_manifest  # noqa: E402
BUILD = (REPO / "desktop" / "build_dmg.sh").read_text(encoding="utf-8")
LOCAL = {p.stem for p in REPO.glob("*.py")}
MANIFEST = REPO / "desktop" / "shared" / "bundle_manifest.txt"
GOLDEN = REPO / "desktop" / "mac_bundle_files.txt"
SHIPPED_DATA = [
    "data/ingredients_template.csv",
    "data/samples/yogurt_pilot/Yogurt pilot - Round 6 to score.json",
    "data/samples/yogurt_pilot/Yogurt pilot - all 10 Rounds.json",
    "data/samples/yogurt_pilot/Round 6 results (pilot).xlsx",
    "data/samples/burger_setup/Example setup - plant-based burger.json",
]


def _local_imports(path):
    """Top-level local modules a file imports, including inside functions."""
    found = set()
    for node in ast.walk(ast.parse(path.read_text(encoding="utf-8"))):
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
    """The files both builds ship: the manifest's lines."""
    return set(read_manifest())


def test_every_imported_module_is_packaged():
    missing = {m + ".py" for m in _app_modules()} - _packaged()
    assert not missing, f"bundle_manifest.txt does not list: {sorted(missing)}"


def test_manifest_ships_only_what_the_app_needs():
    """theory.py is deliberately not shipped: nothing in the app imports it,
    and it needs matplotlib, which left requirements.txt with the deploy
    cleanup."""
    shipped = {e for e in read_manifest() if e.endswith(".py")}
    assert shipped == {m + ".py" for m in _app_modules()}


def test_scanner_sees_the_insights_chain():
    # v3: nothing on screen reads learned.py's verdicts any more.
    assert {"ui_insights", "insights"} <= _app_modules()


def test_manifest_is_plain_sorted_lines():
    raw = MANIFEST.read_bytes().decode("utf-8")
    assert raw.endswith("\n") and "\r" not in raw
    lines = raw.splitlines()
    assert lines == sorted(set(lines))
    assert not [l for l in lines if not l or l.startswith("#") or l != l.strip()]


def test_manifest_entries_exist_inside_the_repo():
    for entry in read_manifest():
        path = pathlib.PurePosixPath(entry)
        assert not path.is_absolute() and ".." not in path.parts, entry
        assert (REPO / entry).is_file(), entry


def test_build_script_ships_the_sample_data():
    # 0.8.0 cleanup: the Review fits nothing, so the worker engine is gone.
    assert not {"insights_worker.py", "insights_jobs.py", "insights_cache.py",
                "learned.py"} & _packaged()
    assert set(SHIPPED_DATA) <= _packaged()
    assert not [e for e in read_manifest() if e.endswith("sample_ingredients.csv")]


def test_golden_bundle_list_matches_the_manifest():
    assert GOLDEN.read_text(encoding="utf-8").splitlines() == \
        expected_bundle_listing(read_manifest())


def test_build_script_lists_no_files_itself():
    """build_dmg.sh copies the manifest's lines and names no app file of its
    own, so a leftover or commented-out list can't satisfy the guards."""
    assert not re.findall(r"[\w./-]+\.py\b", BUILD)
    assert not [d for d in SHIPPED_DATA if d.split("/")[-1] in BUILD]
    assert "bundle_manifest.txt" in BUILD and "while IFS= read -r" in BUILD


def test_source_runs_disable_usage_stats():
    cfg = tomllib.loads((REPO / ".streamlit" / "config.toml").read_text(encoding="utf-8"))
    assert cfg["browser"]["gatherUsageStats"] is False
    assert cfg["server"]["address"] == "127.0.0.1"
