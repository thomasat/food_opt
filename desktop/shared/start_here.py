"""Render Start Here for each OS from desktop/shared/start_here.src.md.

    python desktop/shared/start_here.py --write   # rewrite both renders
    python desktop/shared/start_here.py --check   # exit 1 if one is stale

Source syntax, one directive alone on its line:
    <!-- mac -->, <!-- windows --> ... <!-- end -->   a block only that OS keeps
    <!-- any other words -->                         a comment, dropped
Text outside a block goes to both. Build-time only; never shipped.
"""
import argparse
import re
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
SOURCE = "desktop/shared/start_here.src.md"
OUTPUTS = {"mac": "desktop/start_here.txt", "windows": "desktop/windows/start_here.txt"}

_COMMENT = re.compile(r"^<!--(.*)-->$")
_NAME = re.compile(r"^[a-z]+$")


def render(source: str, os_name: str) -> str:
    """The text one OS gets. Raises ValueError on an unknown directive, a
    nested or unclosed block, or a stray <!-- end -->."""
    out = []
    block = None
    for line in source.splitlines(keepends=True):
        m = _COMMENT.match(line.strip())
        if not m:
            if block is None or block == os_name:
                out.append(line)
            continue
        word = m.group(1).strip()
        if not _NAME.match(word):
            continue   # a comment
        if word == "end":
            if block is None:
                raise ValueError("<!-- end --> with no open block")
            block = None
        elif word in OUTPUTS:
            if block is not None:
                raise ValueError(f"<!-- {word} --> inside an open <!-- {block} --> block")
            block = word
        else:
            raise ValueError(f"unknown directive <!-- {word} -->")
    if block is not None:
        raise ValueError(f"<!-- {block} --> block is never closed")
    return "".join(out)


def _path(rel: str) -> Path:
    p = Path(rel)
    return p if p.is_absolute() else REPO / p


def _read_lf(path: Path) -> str:
    return path.read_bytes().decode("utf-8").replace("\r\n", "\n")


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--write", action="store_true")
    mode.add_argument("--check", action="store_true")
    args = parser.parse_args(argv)
    source = _read_lf(REPO / SOURCE)
    stale = []
    for os_name, rel in OUTPUTS.items():
        text = render(source, os_name)
        path = _path(rel)
        if args.write:
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_bytes(text.encode("utf-8"))
        elif not path.exists() or _read_lf(path) != text:
            stale.append(rel)
    for rel in stale:
        print(f"stale: {rel} (run python {SOURCE.replace('.src.md', '.py')} --write)")
    return 1 if stale else 0


if __name__ == "__main__":
    sys.exit(main())
