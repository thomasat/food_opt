"""A Review you can read in one glance (owner-approved 2026-10-14, spec §10):
the "Are we improving?" Round panel.

- Zoom-proof layout: the score chart (titled "Best so far") over What
  changed, and the sensory profile, share the width from each one's least
  width; zoomed in past both, the profile wraps under them, so the score
  chart is never narrower than at 100 % (534 px at 1440 × 900).
- The profile is sized to its column (its rings stay round).
- What changed is a picture: one row per ingredient on its own
  Lowest–Highest bar, a grey dot at Start, a blue dot at the best so far,
  the change at the right ("at Highest" when it sits there), largest move
  first; the exact amounts in the row's tooltip.

The yogurt pilot samples (sample_projects.ensure in a tmp cwd)."""
import importlib.util
import inspect
import json
import re

import pytest

import sample_projects as sp
import storage as storage_backend
import ui_insights as U
import wording
from food_bo import FoodOptimizer
from tests.test_food_bo import _BANNED

WIDTH_AT_100 = 534      # px: the score chart at 1440 × 900 before this change
CONTENT_1152 = 961      # px: the panel's inside at 1152 × 720 (1440 × 900 at 125 %)
CONTENT_960 = 769       # px: at 960 × 600 (1440 × 900 at 150 %)
GAP = 16                # px: the panel row's gap ("small")


def _builder():
    spec = importlib.util.spec_from_file_location("yogurt_build", sp.SAMPLE_DIR / "build.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


@pytest.fixture(scope="module")
def samples(tmp_path_factory):
    """(Round 6 sample with Round 6 recorded, the all-10 sample)."""
    import os
    here = os.getcwd()
    os.chdir(tmp_path_factory.mktemp("samples"))
    try:
        store = storage_backend.LocalStorage()
        sp.ensure(store)
        r6 = FoodOptimizer(sp.ROUND6, storage=store)
        b = _builder()
        b.score_round(r6, b.pilot()[6])
        all10 = FoodOptimizer(sp.ALL10, storage=store)
    finally:
        os.chdir(here)
    return r6, all10


def _pic(opt, r):
    ch = U.what_changed(opt, r)
    return ch, U.changes_picture(opt, ch)


def _banned(text):
    return [p.pattern for p in _BANNED if p.search(str(text))]


# ------------------------------------------------------------------ #
#  What changed: the picture's rows
# ------------------------------------------------------------------ #

def test_round_6_rows_sorted_by_move_with_start_now_and_limits(samples):
    r6, _ = samples
    ch, pic = _pic(r6, 6)
    assert (pic["start"], pic["best"]) == ("Start (F3)", "Best after Round 6: F16")
    rows = pic["rows"]
    assert [r["name"] for r in rows] == [
        "Coconut cream", "Xanthan gum", "Nutritional yeast", "Salt", "Lemon juice",
        "Tap water", "Tapioca starch"]
    assert [r["label"] for r in rows] == [
        "−6.3 g", "+0.23 g", "+0.47 g", "+0.32 g", "+0.49 g",
        "+2.0 g, at Highest", "+0.12 g, at Highest"]
    assert pic["unchanged"] == []
    # Largest move first, as a share of each row's own Lowest–Highest bar.
    moves = [abs(r["now"] - r["start"]) for r in rows]
    assert moves == sorted(moves, reverse=True)
    # Start and now: the amounts as shares of the bar.
    by = {r.name: r for r in ch.rows}
    for row in rows:
        c = by[row["name"]]
        assert row["start"] == pytest.approx((c.a - c.low) / (c.high - c.low))
        assert row["now"] == pytest.approx(min(1.0, (c.b - c.low) / (c.high - c.low)))
    coconut = rows[0]
    assert (coconut["start_text"], coconut["best_text"], coconut["low_text"],
            coconut["high_text"], coconut["change_text"]) == (
        "57.6 g", "51.3 g", "48 g", "61 g", "−6.3 g (−11 %)")
    assert coconut["start"] == pytest.approx(0.74, abs=0.005)
    assert coconut["now"] == pytest.approx(0.252, abs=0.005)
    # At a limit: the dot at the bar's end, said on the face.
    ends = {r["name"]: r["at_end"] for r in rows}
    assert ends == {"Coconut cream": None, "Xanthan gum": None, "Nutritional yeast": None,
                    "Salt": None, "Lemon juice": None, "Tap water": "Highest",
                    "Tapioca starch": "Highest"}
    assert [r["now"] for r in rows if r["at_end"]] == pytest.approx([1.0, 1.0])


def test_round_4_follows_the_panels_round(samples):
    r6, _ = samples
    ch, pic = _pic(r6, 4)
    assert pic["best"] == "Best after Round 4: F10"
    assert [(r["name"], r["label"]) for r in pic["rows"]] == [
        ("Coconut cream", "−5.8 g"), ("Xanthan gum", "+0.21 g"),
        ("Nutritional yeast", "+0.92 g"), ("Lemon juice", "+0.54 g"), ("Salt", "+0.15 g"),
        ("Tap water", "+1.2 g"), ("Tapioca starch", "+0.04 g")]
    assert not any(r["at_end"] for r in pic["rows"])
    # Start stays Round 1's best at every Round.
    assert [r["start"] for r in pic["rows"]] == pytest.approx(
        [r["start"] for r in sorted(_pic(r6, 6)[1]["rows"],
                                    key=lambda x: [y["name"] for y in pic["rows"]]
                                    .index(x["name"]))])
    # Rounds 1 and 2: the best is still the start; no picture.
    for r in (1, 2):
        ch = U.what_changed(r6, r)
        assert ch.same and wording.changes_same(ch.start_round) == "Best so far is still Round 1's best."


def test_all_10_rounds(samples):
    _, all10 = samples
    _, pic = _pic(all10, 10)
    assert pic["best"] == "Best after Round 10: F21 (Round 7)"
    assert [(r["name"], r["label"]) for r in pic["rows"]][:2] == [
        ("Lemon juice", "+2.14 g"), ("Coconut cream", "−8.6 g")]
    assert {r["name"] for r in pic["rows"] if r["at_end"] == "Highest"} == {
        "Tap water", "Tapioca starch"}
    # Round 3: Lemon juice at its Lowest.
    _, early = _pic(all10, 3)
    lemon = next(r for r in early["rows"] if r["name"] == "Lemon juice")
    assert (lemon["label"], lemon["at_end"], lemon["now"]) == ("−0.06 g, at Lowest", "Lowest", 0.0)


def test_unchanged_rows_fold_into_one_line():
    opt = FoodOptimizer("panel_unchanged")
    opt.set_amount_unit("g")
    for name in ("Oil", "Salt", "Sugar"):
        opt.add_ingredient(name, 0, 10)
    opt.add_objective("Firmness", 100, goal="target", target=6, min_val=0, max_val=10)
    opt.tell({"Oil": 2.0, "Salt": 1.0, "Sugar": 3.0}, {"Firmness": 3}, formulation_no=1,
             batch_no=1)
    opt.tell({"Oil": 6.0, "Salt": 1.0, "Sugar": 3.0}, {"Firmness": 6}, formulation_no=2,
             batch_no=2)
    _, pic = _pic(opt, 2)
    assert [r["name"] for r in pic["rows"]] == ["Oil"]
    assert pic["unchanged"] == ["Salt", "Sugar"]
    assert wording.changes_unchanged(pic["unchanged"]) == "Unchanged: Salt and Sugar."
    assert wording.changes_unchanged(["A", "B", "C", "D", "E"]) == "Unchanged: A, B and 3 more."


# ------------------------------------------------------------------ #
#  What changed: the chart
# ------------------------------------------------------------------ #

def _datasets(spec):
    return [row for rows in spec.get("datasets", {}).values() for row in rows]


def _flat_layers(spec):
    out = []
    for layer in spec.get("layer", []):
        out += _flat_layers(layer) if "layer" in layer else [layer]
    return out


def test_the_chart_draws_bars_dots_and_labels_and_keeps_numbers_in_the_tooltip(samples):
    r6, _ = samples
    _, pic = _pic(r6, 6)
    spec = U.changes_chart(r6, pic).to_dict()
    assert spec["height"] == U.CHANGES_ROW_H * 7
    layers = _flat_layers(spec)
    kinds = [(l["mark"]["type"] if isinstance(l["mark"], dict) else l["mark"]) for l in layers]
    assert kinds.count("rule") == 2 and "tick" in kinds and kinds.count("text") == 2
    # Sized to its column: the x domain is worked out from the chart's width.
    x = next(l["encoding"]["x"] for l in layers if "x" in l["encoding"])
    assert all("width" in d["expr"] for d in x["scale"]["domain"])
    # The header: Lowest at the left end of every bar, Highest at the right.
    assert x["axis"]["values"] == [0, 1] and x["axis"]["orient"] == "top"
    assert "'Lowest'" in x["axis"]["labelExpr"] and "'Highest'" in x["axis"]["labelExpr"]
    # The legend: grey Start, blue best so far.
    dots = next(l for l in layers if "color" in l["encoding"])
    assert dots["encoding"]["color"]["scale"]["domain"] == ["Start (F3)",
                                                           "Best after Round 6: F16"]
    # The face says only the change; the exact amounts are in the tooltip.
    text = [l["encoding"]["text"]["field"] for l in layers if "text" in l["encoding"]]
    assert set(text) == {"label"}
    tips = {t["title"] for t in dots["encoding"]["tooltip"]}
    assert {"Start (F3)", "Best after Round 6: F16", "Lowest", "Highest", "Change"} <= tips
    # The rows in the picture's order; arrowheads only where the move shows.
    names = next(l["encoding"]["y"]["sort"] for l in layers if "y" in l["encoding"])
    assert names == [r["name"] for r in pic["rows"]]
    arrows = [l for l in layers if (l["mark"].get("shape") or "") == U._CHANGES_ARROW]
    assert len(arrows) == 1
    drawn = {row["name"] for row in _datasets(spec) if "angle" in row}
    assert drawn == {r["name"] for r in pic["rows"]
                     if abs(r["now"] - r["start"]) >= U._CHANGES_ARROW_FROM}


def test_no_chart_with_no_changed_row():
    assert U.changes_chart(None, {"rows": [], "unchanged": ["Salt"], "start": "S",
                                  "best": "B"}) is None


def test_the_block_keeps_one_height_and_one_element_count(samples, monkeypatch):
    from tests.test_final_demo import _FakeSt
    r6, _ = samples
    assert U.changes_height(r6) == U.CHANGES_BASE_H + 7 * U.CHANGES_ROW_H
    counts = set()
    for r in (1, 2, 4, 6):
        fake = _FakeSt()
        monkeypatch.setattr(U, "st", fake)
        U._changes_block(r6, r)
        styles = [c[1][0] for c in fake.calls if c[0] == "html"]
        assert styles == [f"<style>.st-key-insights_changes {{ min-height: "
                          f"{U.changes_height(r6)}px; }}</style>"]
        counts.add(len([c for c in fake.calls if c[0] in (
            "markdown", "empty", "vega_lite_chart", "caption")]))
        titles = [c[1][0] for c in fake.calls if c[0] == "markdown"]
        assert titles[0] == f"**{wording.changes_title(r)}**"
        assert not [t for t in titles if t.startswith("| ")]        # no table
    assert counts == {3}


# ------------------------------------------------------------------ #
#  The layout
# ------------------------------------------------------------------ #

def test_the_panel_shares_the_width_and_stacks_when_zoomed_in():
    body = inspect.getsource(U._progress_fragment)
    assert "st.columns(" not in body and "horizontal=True" in body
    assert body.index("PANEL_LEFT") < body.index("PANEL_RIGHT")         # score chart first
    assert body.index("_improving_part(") < body.index("_changes_block(") \
        < body.index("_profile_part(")
    assert "flex-wrap: wrap" in U.REPLAY_STYLE
    left, right = U.PANEL_MIN_W
    grow_l, grow_r = U.PANEL_GROW
    assert f"flex: {grow_l} 1 {left}px" in U.REPLAY_STYLE
    assert f"flex: {grow_r} 1 {right}px" in U.REPLAY_STYLE
    # Never narrower than at 100 %; side by side at 125 %, stacked at 150 %.
    assert left >= WIDTH_AT_100
    assert left + right + GAP <= CONTENT_1152 < 2 * CONTENT_1152
    assert left + right + GAP > CONTENT_960


def test_the_score_chart_is_titled_plainly():
    assert wording.PANEL_SCORE_TITLE == "Best so far"
    assert "PANEL_SCORE_TITLE" in inspect.getsource(U._improving_part)


def test_the_profile_fits_its_column(samples):
    r6, _ = samples
    data = U.spider_data(r6)
    spec = U.spider_chart(r6, data, 6, size=U.PANEL_SPIDER_H, fit=True).to_dict()
    assert "width" not in spec and spec["height"] == U.PANEL_SPIDER_H + U._SPIDER_KEY_H
    assert spec["autosize"]["type"] == "fit-x"
    flat = json.dumps(spec)
    assert '"expr": "-width / 2 / max(30' in flat
    # Fix round: the key centred under the rings, in the room kept for it.
    legends = [l["encoding"]["color"]["legend"] for l in _flat_layers(spec)
               if "legend" in l.get("encoding", {}).get("color", {})]
    assert legends and all(g["orient"] == "none" and g["legendX"]["expr"].startswith("width / 2 - ")
                           and g["legendY"] == U.PANEL_SPIDER_H + 10 for g in legends)
    # Side labels wrapped short enough for a narrow column.
    sides = {row["text"] for row in _datasets(spec)
             if "x1" in row and row.get("align") in ("left", "right")}
    assert sides == {"Thickness\nuniformity\ntarget 13", "Creaminess\ntarget 5"}
    assert "fit=True" in inspect.getsource(U._spider_block)


# ------------------------------------------------------------------ #
#  Wording
# ------------------------------------------------------------------ #

def _words(text):
    return len([w for w in re.sub(r"\*\*", "", str(text)).split() if re.search(r"\w", w)])


def test_succinct_and_no_banned_word(samples):
    r6, all10 = samples
    labels = [wording.PANEL_SCORE_TITLE, wording.changes_title(10),
              wording.changes_at_end("+0.12 g", "Highest"), wording.LOWEST_LABEL,
              wording.HIGHEST_LABEL]
    for opt, r in ((r6, 6), (r6, 4), (all10, 10), (all10, 3)):
        labels += [row["label"] for row in _pic(opt, r)[1]["rows"]]
    assert [t for t in labels if _words(t) > 5] == []
    captions = [wording.CHANGES_CAPTION, wording.changes_unchanged(["Salt", "Sugar"]),
                wording.changes_unchanged(["Nutritional yeast", "Lemon juice", "A", "B"])]
    assert [t for t in captions if _words(t) > 8] == []
    ch, pic = _pic(r6, 6)
    texts = labels + captions + [
        wording.changes_picture_help(pic["start"], pic["best"], 1, fill="Water", fixed=True,
                                     unchanged=["A", "B", "C", "D"]),
        wording.changes_cause_help(3, 16)]
    texts += [row[k] for row in pic["rows"] for k in ("start_text", "best_text", "low_text",
                                                      "high_text", "change_text")]
    assert [(t, _banned(t)) for t in texts if _banned(t)] == []
    # Specific: the help names both formulations and the Round.
    help_ = wording.changes_picture_help(pic["start"], pic["best"], ch.start_round)
    assert "Start (F3), the best of Round 1" in help_ and "Best after Round 6: F16" in help_
    assert wording.changes_cause_help(3, 16).startswith(
        "Several things changed at once between Formulation 3 and Formulation 16")


def test_the_score_chart_draws_no_noise_band_and_keeps_the_top_zone(samples):
    """Fix round (owner, 2026-10-14): tasting noise is no longer a Review
    concept; the grey zone near the top (what the stop refers to) stays."""
    r6, all10 = samples
    for opt in (r6, all10):
        text = json.dumps(U.improving_chart(opt, height=U.PANEL_CHART_H).to_dict(),
                          ensure_ascii=False)
        assert "assumed noise" not in text and "Noise from Repeats" not in text
        assert wording.TOP_ZONE in text
    assert "Blue band" not in wording.story_improving_help("84.2", "96.1", 1, 6, "2.3")
    assert not hasattr(wording, "noise_band")
