"""Builds desktop/windows/icon.ico (the exe, window, shortcut and installer
icon) and the installer's wizard pictures (installer_art/*.bmp) from the
Food Intelligence Lab logo in desktop/logo.

    python desktop/windows/make_icon.py --write   # regenerate icon.ico and the art
    python desktop/windows/make_icon.py --check   # 0 when both match the logo

Needs Pillow (a dev tool only; the app never runs this)."""
import io
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
LOGO = HERE.parent / "logo" / "Food Intelligence Lab Icon_FIL Green.png"
ICON = HERE / "icon.ico"
SIZES = (16, 20, 24, 32, 40, 48, 64, 256)
ART = HERE / "installer_art"
# Inno's wizard pictures at 100/150/200% display scaling: the tall one on
# the Welcome and Finish pages, the small one at the top of the others.
WIZARD_SIZES = {"wizard_100": (164, 314), "wizard_150": (246, 471), "wizard_200": (328, 628),
                "small_100": (55, 55), "small_150": (83, 83), "small_200": (110, 110)}
BACKGROUND = (0xF7, 0xF6, 0xF2)     # the window's background, #f7f6f2
LOGO_SHARE = 0.8                    # the logo's width as a share of the picture's narrow side


def render(logo: Path) -> bytes:
    """A multi-size ICO, each size resampled (LANCZOS) from the square logo."""
    from PIL import Image

    with Image.open(logo) as src:
        img = src.convert("RGBA")
    frames = [img.resize((s, s), Image.Resampling.LANCZOS) for s in SIZES]
    out = io.BytesIO()
    frames[-1].save(out, format="ICO", sizes=[(s, s) for s in SIZES], append_images=frames[:-1])
    return out.getvalue()


def wizard_images(logo: Path) -> dict[str, bytes]:
    """24-bit BMPs: the logo, centred on the window's background colour."""
    from PIL import Image

    with Image.open(logo) as src:
        img = src.convert("RGBA")
    out = {}
    for name, (w, h) in WIZARD_SIZES.items():
        side = round(min(w, h) * LOGO_SHARE)
        mark = img.resize((side, side), Image.Resampling.LANCZOS)
        canvas = Image.new("RGBA", (w, h), BACKGROUND + (255,))
        canvas.alpha_composite(mark, ((w - side) // 2, (h - side) // 2))
        buf = io.BytesIO()
        canvas.convert("RGB").save(buf, format="BMP")
        out[name] = buf.getvalue()
    return out


def _pixels(data: bytes) -> bytes:
    from PIL import Image

    with Image.open(io.BytesIO(data)) as im:
        return im.convert("RGB").tobytes()


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
        ART.mkdir(exist_ok=True)
        for name, data in wizard_images(LOGO).items():
            (ART / f"{name}.bmp").write_bytes(data)
            print(f"wrote {ART / name}.bmp")
        return 0
    if argv == ["--check"]:
        if not ICON.exists():
            print(f"missing {ICON}")
            return 1
        same = _frames(ICON.read_bytes()) == _frames(render(LOGO))
        print("icon.ico matches the logo" if same else "icon.ico differs from the logo: run --write")
        for name, data in wizard_images(LOGO).items():
            path = ART / f"{name}.bmp"
            if not path.exists() or _pixels(path.read_bytes()) != _pixels(data):
                print(f"{path.name} differs from the logo: run --write")
                same = False
        return 0 if same else 1
    print(__doc__)
    return 2


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
