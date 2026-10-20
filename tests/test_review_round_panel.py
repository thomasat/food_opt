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


@pytest.fixture(autouse=True)
def _isolate_cwd(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)          # a project saves into the cwd


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


# ------------------------------------------------------------------ #
#  The layout
# ------------------------------------------------------------------ #

def test_the_panel_shares_the_width_and_stacks_when_zoomed_in():
    # Review v3: Best so far beside Best so far vs targets, Amounts in the
    # top formulations under both; the two columns side by side at 1152 px.
    body = inspect.getsource(U._progress_fragment)
    assert "st.columns(" not in body and "horizontal=True" in body
    assert body.index("PANEL_LEFT") < body.index("PANEL_RIGHT")         # score chart first
    assert body.index("_improving_part(") < body.index("_targets_part(") \
        < body.index("_amounts_part(")
    assert "flex-wrap: wrap" in U.REPLAY_STYLE
    left, right = U.PANEL_MIN_W
    grow_l, grow_r = U.PANEL_GROW
    assert f"flex: {grow_l} 1 {left}px" in U.REPLAY_STYLE
    assert f"flex: {grow_r} 1 {right}px" in U.REPLAY_STYLE
    assert left + right + GAP <= CONTENT_1152


def test_the_panel_style_has_no_tag_in_it():
    # st.html reads anything like "<key>" inside the <style> as a tag and
    # drops the rules after it (fix round 2): only the style tags themselves.
    body = U.REPLAY_STYLE.strip()
    assert body.startswith("<style>") and body.endswith("</style>")
    assert "<" not in body[len("<style>"):-len("</style>")]
    assert "/*" not in body


def test_the_score_chart_is_titled_plainly():
    assert wording.PANEL_SCORE_TITLE == "Best so far"
    assert "PANEL_SCORE_TITLE" in inspect.getsource(U._improving_part)


# ------------------------------------------------------------------ #
#  Wording
# ------------------------------------------------------------------ #

def _words(text):
    return len([w for w in re.sub(r"\*\*", "", str(text)).split() if re.search(r"\w", w)])


# ------------------------------------------------------------------ #
#  Fix round 2: limits on several rows, beyond an end, choices
# ------------------------------------------------------------------ #


