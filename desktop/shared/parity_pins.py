"""Parity pins: each pair ties a Mac section (Swift page code, launcher.sh's
run-time half) to the shared port that mirrors it. A change to one side
makes the pins stale until the other side is mirrored and the pair re-pinned.

    python desktop/shared/parity_pins.py --check
    python desktop/shared/parity_pins.py --pair page --note "what was mirrored"

Build-time only; never shipped.
"""
import argparse
import hashlib
import json
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
PINS = Path(__file__).resolve().parent / "parity_pins.json"


def _read_lf(path: Path) -> str:
    return Path(path).read_bytes().decode("utf-8").replace("\r\n", "\n")


def _line_of(lines, anchor, name):
    hits = [i for i, line in enumerate(lines) if line.startswith(anchor)]
    if len(hits) != 1:
        raise ValueError(f"{name}: anchor {anchor!r} occurs {len(hits)} times")
    return hits[0]


def section_hash(spec: dict, repo: Path) -> str:
    """sha256 of LF text: a file's lines from `start` up to (not including)
    `end` (or its end), or every file of a dir matching `glob`, each as
    relpath + "\\n" + content, in sorted relpath order."""
    repo = Path(repo)
    digest = hashlib.sha256()
    if "file" in spec:
        lines = _read_lf(repo / spec["file"]).split("\n")
        start = _line_of(lines, spec["start"], spec["file"])
        end = len(lines) if spec.get("end") is None else _line_of(lines, spec["end"], spec["file"])
        digest.update("\n".join(lines[start:end]).encode("utf-8"))
        return digest.hexdigest()
    root = repo / spec["dir"]
    paths = [p for p in root.glob(spec["glob"])
             if p.is_file() and "__pycache__" not in p.relative_to(root).parts]
    for rel in sorted(p.relative_to(root).as_posix() for p in paths):
        digest.update((rel + "\n" + _read_lf(root / rel)).encode("utf-8"))
    return digest.hexdigest()


def stale_sections(doc: dict, repo: Path) -> list:
    return [name for name, spec in doc["sections"].items()
            if section_hash(spec, repo) != spec["sha256"]]


def _write(doc: dict) -> None:
    PINS.write_bytes((json.dumps(doc, indent=2, ensure_ascii=False) + "\n").encode("utf-8"))


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--check", action="store_true")
    parser.add_argument("--pair")
    parser.add_argument("--note")
    try:
        args = parser.parse_args(argv)
    except SystemExit as exit_:
        return int(exit_.code or 0)
    doc = json.loads(_read_lf(PINS))
    if args.pair:
        if not args.note or not args.note.strip():
            print("--pair needs --note \"<what was mirrored on the other side>\"")
            return 2
        if args.pair not in doc["pairs"]:
            print(f"no pair {args.pair!r}; pairs: {', '.join(doc['pairs'])}")
            return 2
        for name in doc["pairs"][args.pair]["sections"]:
            doc["sections"][name]["sha256"] = section_hash(doc["sections"][name], REPO)
        doc["pairs"][args.pair]["note"] = args.note.strip()
        _write(doc)
        return 0
    stale = stale_sections(doc, REPO)
    for name in stale:
        pair = next(p for p, spec in doc["pairs"].items() if name in spec["sections"])
        print(f"stale: {name} (pair {pair}) - mirror the change on the other side, then "
              f"python desktop/shared/parity_pins.py --pair {pair} --note \"...\"")
    return 1 if stale else 0


if __name__ == "__main__":
    sys.exit(main())
