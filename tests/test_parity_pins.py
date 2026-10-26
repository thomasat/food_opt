"""Parity pins: hashes of the Mac sections and the shared ports that mirror
them. Changing one side of a pair fails here until the other side is
mirrored and the pair is re-pinned in the same commit."""
import importlib.util
import json
import shutil

from tests.shared_texts import REPO, read_lf

PINS = REPO / "desktop" / "shared" / "parity_pins.json"


def _module():
    spec = importlib.util.spec_from_file_location(
        "parity_pins", REPO / "desktop" / "shared" / "parity_pins.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _doc():
    return json.loads(read_lf(PINS))


def test_parity_pins_are_current():
    doc = _doc()
    stale = _module().stale_sections(doc, REPO)
    pair_of = {s: p for p, spec in doc["pairs"].items() for s in spec["sections"]}
    assert stale == [], "\n".join(
        f"{s} changed (pair {pair_of.get(s)}): mirror the change on the other side first, "
        f"then run: python desktop/shared/parity_pins.py --pair {pair_of.get(s)} "
        f'--note "<what was mirrored>"' for s in stale)


def test_every_anchor_occurs_once():
    for name, spec in _doc()["sections"].items():
        if "file" not in spec:
            continue
        lines = read_lf(REPO / spec["file"]).split("\n")
        for anchor in (spec["start"], spec["end"]):
            if anchor is None:
                continue
            assert sum(1 for line in lines if line.startswith(anchor)) == 1, (name, anchor)


def test_every_section_is_in_exactly_one_pair():
    doc = _doc()
    listed = [s for spec in doc["pairs"].values() for s in spec["sections"]]
    assert sorted(listed) == sorted(doc["sections"])
    assert len(listed) == len(set(listed))
    assert all(spec["note"] for spec in doc["pairs"].values())


def test_a_change_on_one_side_fails(tmp_path):
    doc = _doc()
    for spec in doc["sections"].values():
        rel = spec.get("file") or spec["dir"]
        src = REPO / rel
        dst = tmp_path / rel
        dst.parent.mkdir(parents=True, exist_ok=True)
        if src.is_dir():
            if not dst.exists():
                shutil.copytree(src, dst, ignore=shutil.ignore_patterns("__pycache__"))
        else:
            shutil.copyfile(src, dst)
    mod = _module()
    assert mod.stale_sections(doc, tmp_path) == []
    swift = tmp_path / "desktop" / "FoodOptimizerApp.swift"
    text = read_lf(swift)
    line = "    func showStepsPage() {"
    assert line in text
    swift.write_text(text.replace(line, line + " ", 1), encoding="utf-8", newline="\n")
    assert mod.stale_sections(doc, tmp_path) == ["mac.swift.page"]


def test_a_dotfile_does_not_change_a_dir_pin(tmp_path):
    pages = tmp_path / "desktop" / "shared" / "pages"
    shutil.copytree(REPO / "desktop" / "shared" / "pages", pages,
                    ignore=shutil.ignore_patterns("__pycache__"))
    mod = _module()
    spec = {"dir": "desktop/shared/pages", "glob": "*"}
    before = mod.section_hash(spec, tmp_path)
    (pages / ".DS_Store").write_bytes(b"\0\1")
    (pages / ".cache").mkdir()
    (pages / ".cache" / "x.js").write_text("x", encoding="utf-8")
    assert mod.section_hash(spec, tmp_path) == before


def test_repinning_needs_a_note():
    assert _module().main(["--pair", "page"]) == 2
