"""0.8.0 e2e fixes, W3 (spec 2026-10-01 §D; review M7): the charts'
meaning-carrying marks have theme tokens at 3:1 in both themes. (The
Insights cache, worker-pool and fitted-chart tests went with that code.)
The colour helpers are imported by the chart tests."""
import ui_insights


PAGES = {"light": "#ffffff", "dark": "#0e1117"}


def _rgb(hex_colour):
    h = hex_colour.lstrip("#")
    return [int(h[k:k + 2], 16) for k in (0, 2, 4)]


def _luminance(rgb):
    def lin(c):
        c /= 255
        return c / 12.92 if c <= 0.03928 else ((c + 0.055) / 1.055) ** 2.4
    r, g, b = rgb
    return 0.2126 * lin(r) + 0.7152 * lin(g) + 0.0722 * lin(b)


def _contrast_on(colour, opacity, page):
    """The mark blended onto the page at its opacity, against the page."""
    fg, bg = _rgb(colour), _rgb(page)
    seen = [opacity * f + (1 - opacity) * b for f, b in zip(fg, bg)]
    la, lb = sorted((_luminance(seen), _luminance(bg)), reverse=True)
    return (la + 0.05) / (lb + 0.05)


# Vega's own "blues" scheme (vega-scale): eleven stops, linear between them.
_VEGA_BLUES = ("cfe1f2bed8eca8cee58fc1de74b2d75ba3cf4592c63181bd206fb2125ca40a4a90")


def _vega_blues(t):
    stops = [_rgb(_VEGA_BLUES[i:i + 6]) for i in range(0, len(_VEGA_BLUES), 6)]
    x = t * (len(stops) - 1)
    k = min(int(x), len(stops) - 2)
    f = x - k
    return "#" + "".join(f"{round(a + f * (b - a)):02x}" for a, b in zip(stops[k], stops[k + 1]))


def test_every_token_reaches_3_to_1_in_its_theme():
    for theme, page in PAGES.items():
        for name, (colour, opacity) in ui_insights._MARKS[theme].items():
            assert _contrast_on(colour, opacity, page) >= 3.0, (theme, name)


def _all_layers(spec):
    out = []
    for key in ("concat", "layer"):
        for part in spec.get(key, []):
            out += _all_layers(part)
    if "spec" in spec:
        out += _all_layers(spec["spec"])
    if "mark" in spec:
        out.append(spec)
    return out


def _mark(layer):
    return layer["mark"] if isinstance(layer["mark"], dict) else {"type": layer["mark"]}
