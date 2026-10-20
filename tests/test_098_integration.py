"""0.8.0 BO views, Task 8: the wave as a whole (spec 2026-09-30-bo-views §5,
§7; 04 §5; review I10). At most one link to Insights on any screen, never
lit; the new screens say no retired word; the lines the wave retired are gone."""
import re

import pytest
from streamlit.testing.v1 import AppTest

import ui_batch
import ui_insights
import wording
from tests.test_app_ui import APP_PATH
from tests.test_food_bo import _BANNED, _FTUE_ALLOWED
from tests.test_insights import _burger
from ui_helpers import INSIGHTS_OPENED_KEY


@pytest.fixture(autouse=True)
def _isolate(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)


def _run(at, tab):
    at.session_state["main_tab"] = tab
    at.run()
    assert not at.exception
    return at


def _links(at):
    return [(i, b) for i, t in enumerate(at.tabs) for b in t.button
            if str(b.key or "").startswith("link_")]


# ------------------------------------------------------------------ #
#  04 §5: at most one link on a screen, never the lit button
# ------------------------------------------------------------------ #

def test_each_screen_offers_at_most_one_link():
    opt = _burger(rounds=4)
    at = AppTest.from_file(APP_PATH, default_timeout=300)
    at.session_state["_loaded_project"] = opt.project_name
    at.run()
    _run(at, wording.TAB_INSIGHTS)
    del at.session_state[INSIGHTS_OPENED_KEY]
    seen = set()
    for index, tab in ((2, wording.TAB_RESULTS), (1, wording.TAB_BATCH),
                       (3, wording.TAB_INSIGHTS)):
        _run(at, tab)
        seen |= {b.key for i, b in _links(at) if i == index}
        per_tab = {}
        for i, b in _links(at):
            per_tab.setdefault(i, []).append(b)
            assert b.proto.type == "tertiary", b.key
        assert all(len(v) <= 1 for v in per_tab.values()), per_tab
    # v3: no screen links anywhere: Results has no link to the forecasts
    # (no "measured against predicted"), Make a round no status line, the
    # Review no Why or Repeat link.
    assert seen == set()


def test_the_new_screens_say_no_retired_word():
    """03 §3: no "likely range", "On track", "though", hedges or advice
    words; and the vocabulary guard over everything the two screens say."""
    opt = _burger(rounds=4, noise=0.4)
    ui_batch.ask_alone(opt, n_suggestions=3)
    at = AppTest.from_file(APP_PATH, default_timeout=300)
    at.session_state["_loaded_project"] = opt.project_name
    at.run()
    at.session_state[ui_insights.HOW_KEY] = True
    said = []
    for tab in (wording.TAB_INSIGHTS, wording.TAB_BATCH):
        _run(at, tab)
        said += [m.value for m in at.markdown] + [c.value for c in at.caption]
    # T8 review: the audit's "keep in mind", "rough" and "a little" too.
    retired = re.compile(r"\b(likely|On track|though|treat|Barely|Not lately|stands out|"
                         r"worth|some luck|too few to be sure|[Kk]eep in mind|[Rr]ough|[Aa] little)\b")
    assert [t for t in said if retired.search(t)] == []
    # The one scoped allowance (first-time wave, review R-B2): Model and
    # method's lead, said as the exact string the guard allows.
    assert [t for t in said if t not in _FTUE_ALLOWED
            and any(p.search(t) for p in _BANNED)] == []


def test_the_lines_the_wave_retired_are_gone():
    for name in ("bias_tail", "PREDICTIONS_CAPTION"):
        assert not hasattr(wording, name), name
    assert not hasattr(ui_insights, "track_line")
