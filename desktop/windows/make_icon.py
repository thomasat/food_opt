"""Builds desktop/windows/icon.ico (the exe, window, shortcut and installer
icon) from the Food Intelligence Lab logo in desktop/logo.

    python desktop/windows/make_icon.py --write   # regenerate icon.ico
    python desktop/windows/make_icon.py --check   # 0 when icon.ico matches

Needs Pillow (a dev tool only; the app never runs this)."""
import io
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
LOGO = HERE.parent / "logo" / "Food Intelligence Lab Icon_FIL Green.png"
ICON = HERE / "icon.ico"
SIZES = (16, 20, 24, 32, 40, 48, 64, 256)


def render(logo: Path) -> bytes:
    """A multi-size ICO, each size resampled (LANCZOS) from the square logo."""
    from PIL import Image

    with Image.open(logo) as src:
        img = src.convert("RGBA")
    frames = [img.resize((s, s), Image.Resampling.LANCZOS) for s in SIZES]
    out = io.BytesIO()
    frames[-1].save(out, format="ICO", sizes=[(s, s) for s in SIZES], append_images=frames[:-1])
    return out.getvalue()


def _frames(data: bytes) -> dict:
    """Each size's decoded RGBA pixels: what --check compares, so a
    different Pillow's PNG compression does not count as a change."""
    from PIL import Image

    out = {}
    with Image.open(io.BytesIO(data)) as ico:
        for size in ico.info["sizes"]:
            ico.size = size
            out[size] = ico.convert("RGBA").tobytes()
    return out


def main(argv) -> int:
    if argv == ["--write"]:
        ICON.write_bytes(render(LOGO))
        print(f"wrote {ICON}")
        return 0
    if argv == ["--check"]:
        if not ICON.exists():
            print(f"missing {ICON}")
            return 1
        same = _frames(ICON.read_bytes()) == _frames(render(LOGO))
        print("icon.ico matches the logo" if same else "icon.ico differs from the logo: run --write")
        return 0 if same else 1
    print(__doc__)
    return 2


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
