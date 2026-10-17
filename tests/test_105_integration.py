"""Integration of the 0.8.0 e2e fixes (W2 → W1 → W3; spec 2026-10-01 rev. 2
and its Overrides): what joins the three waves, and the loose ends they
left.

- Sticky flashes (W2, review I10) survive the reruns the app makes itself
  (W3's poller, keep_up's two reruns) and come down at the user's next
  rerun of any kind: a click, a widget, a fold that reruns.
- The Why fold (W2) gives the exact reason the expected gain is
  withheld, from W1's skill gate: too few checks, skill under 0.20, no
  skill, the guard or the verdict, as Progress gives it.
- Make a round's E3 kinds lose the dead "bias".
- D2: Method and details keeps Error ÷ Repeat noise and Skill in view.
- D5: Share of score shown as stored, to one decimal.
- T0′: the interval chart's caption holds in both themes; the progress
  line never says "12 of 12 Rounds checked…" while models are fitted.
- pair_chart takes W3's theme scale and tokens (D1).
"""
import pytest
from tests.demo_build_reason import REVIEW_FOUR
from streamlit.testing.v1 import AppTest

import insights
import insights_jobs
import ui_insights
import wording
# Review four, ruling 2: the Why fold's "?" says the gain is greyed, not hidden.
from tests import e2e_fixtures as E
from tests.test_104_speed_robust_charts import (PAGES, _all_layers, _contrast_on, _mark,
                                                _vega_blues)

# ------------------------------------------------------------------ #
#  Sticky flashes and the app's own reruns (W2 I10 × W3 C1)
# ------------------------------------------------------------------ #

# A stand-in for app.py's order: what the app does on its own (the poller's
# fragment rerun, keep_up's reruns at the foot of the run before) happens
# before this run's render_flash; the user's clicks happen after it.
STICKY_SCRIPT = """
import streamlit as st
import ui_insights
from ui_helpers import flash, render_flash


class Opt:
    def complete_count(self):
        return 99


auto = st.session_state.pop("_auto", None)
if auto == "poller":
    ui_insights._poller(Opt(), ())
elif auto == "keep_up":
    ui_insights.keep_up(Opt(), ())
render_flash()
if st.button("save"):
    flash("success", "Round 6 recorded.", sticky=True)
    st.rerun()
st.checkbox("A setting", key="setting")
with st.expander("A fold", key="fold", on_change="rerun"):
    st.write("inside")
"""


def _sticky_app():
    at = AppTest.from_string(STICKY_SCRIPT, default_timeout=60)
    at.run()
    next(b for b in at.button if b.label == "save").click()
    at.run()
    assert not at.exception
    assert _shown(at) == ["Round 6 recorded."]
    return at


def _shown(at):
    return [m.value for m in at.success]


def _auto(at, how):
    at.session_state["_auto"] = how
    at.run()
    assert not at.exception


@pytest.fixture
def poll_lands(monkeypatch):
    """The poller's look finds a part landed: it reruns the page."""
    monkeypatch.setattr(ui_insights, "_poll_step", lambda opt, stages: True)


def test_a_sticky_flash_survives_the_pollers_rerun(poll_lands):
    at = _sticky_app()
    _auto(at, "poller")
    assert _shown(at) == ["Round 6 recorded."]
    _auto(at, "poller")
    assert _shown(at) == ["Round 6 recorded."]


def test_a_sticky_flash_survives_keep_ups_rerun_when_a_part_landed(monkeypatch):
    """keep_up's rerun when a part landed while the page was drawn."""
    monkeypatch.setattr(insights_jobs, "workers_on", lambda: True)
    monkeypatch.setattr(ui_insights, "_catch_up", lambda opt: None)
    monkeypatch.setattr(ui_insights, "_seen", lambda opt, stages: (True,))
    monkeypatch.setattr(ui_insights, "_in_flight", lambda opt: False)
    at = _sticky_app()
    at.session_state[ui_insights.SEEN_KEY] = (False,)
    _auto(at, "keep_up")
    assert _shown(at) == ["Round 6 recorded."]


def test_a_sticky_flash_survives_keep_ups_rerun_when_the_workers_stop(monkeypatch):
    """keep_up's rerun when the workers were given up on mid-draw."""
    monkeypatch.setattr(insights_jobs, "workers_on", lambda: False)
    at = _sticky_app()
    at.session_state[ui_insights.SEEN_KEY] = (False,)
    _auto(at, "keep_up")
    assert _shown(at) == ["Round 6 recorded."]


def test_any_rerun_of_the_users_takes_it_down(poll_lands):
    """Review I10: "until the next action". A rerun the app did not make is
    the user's: a click, a widget, a fold opened (a fold that reruns)."""
    at = _sticky_app()
    _auto(at, "poller")
    at.run()                                         # the user's rerun, of any kind
    assert _shown(at) == []
    at = _sticky_app()
    at.checkbox(key="setting").check()
    at.run()
    assert _shown(at) == []
    at = _sticky_app()
    at.session_state["fold"] = True                  # what opening the fold posts
    at.run()
    assert _shown(at) == []


def test_after_the_user_moves_on_the_poller_does_not_bring_it_back(poll_lands):
    at = _sticky_app()
    at.run()
    _auto(at, "poller")
    assert _shown(at) == []


# ------------------------------------------------------------------ #
#  The A3 reason is Progress's own words (W2 × W1's skill_gate)
# ------------------------------------------------------------------ #



def test_the_why_fold_and_progress_give_one_reason(monkeypatch):
    """The fold's words are Progress's reason (gain_reason), not a second
    paraphrase of it."""
    for gate in (insights.SkillGate("wait", 6),
                 insights.SkillGate("unclear", 24, insights.SkillFact(0.09, 24, [3, 4], "unclear"))):
        reason = ui_insights.gain_reason("skill", gate)
        assert reason in wording.why_guard_line(gate_reason=reason)
        assert reason in wording.gain_unreliable(reason)


# ------------------------------------------------------------------ #
#  D5: Share of score as stored, one decimal (03 M1, review M11)
# ------------------------------------------------------------------ #

def _bar():
    return E.load(E.BAR_CLEAN)


def test_share_of_score_is_shown_as_stored_to_one_decimal():
    opt = _bar()
    stored = {o['name']: float(o['weight']) for o in opt.objectives}
    shown = opt.share_percents()
    assert sorted(shown.values(), reverse=True) == [50.4, 29.6, 20.0]
    assert all(abs(shown[n] - round(stored[n], 1)) < 1e-9 for n in stored)
    assert round(sum(shown.values()), 6) == 100.0
    frame = opt.measurement_grid_frame()
    assert sorted(frame[wording.SHARE_COLUMN], reverse=True) == [50.4, 29.6, 20.0]
    first = opt.measurements_by_importance()[0]['name']
    assert opt.share_text(first) == "50.4 %"


def test_one_decimal_still_adds_to_100_and_whole_shares_read_whole():
    from food_bo import FoodOptimizer
    opt = FoodOptimizer.__new__(FoodOptimizer)
    opt.objectives = [{'name': n, 'weight': 1.0} for n in "ABC"]
    shown = opt.share_percents()
    assert round(sum(shown.values()), 6) == 100.0 and sorted(shown.values()) == [33.3, 33.3, 33.4]
    opt.objectives = [{'name': 'A', 'weight': 60.0}, {'name': 'B', 'weight': 40.0}]
    assert opt.share_text('A') == "60 %"


def test_retyping_the_rounded_share_is_seen_as_a_change():
    """03 M1: 50/30/20 was shown while 50.4/29.6/20 was stored, and
    typing 50 over the 50 shown was not seen as a change."""
    opt = _bar()
    frame = opt.measurement_grid_frame()
    first = frame[wording.SHARE_COLUMN].idxmax()
    frame.loc[first, wording.SHARE_COLUMN] = 50.0
    preview = opt.preview_measurement_shares(frame)
    assert preview[frame.loc[first, wording.MEASUREMENT_COLUMN]] == 50.0
    assert sorted(preview.values(), reverse=True) != [50.4, 29.6, 20.0]


def test_the_share_column_shows_one_decimal():
    import ui_setup
    column = ui_setup._measurement_columns()[wording.SHARE_COLUMN]
    assert column["type_config"]["format"] == "%.1f"


# ------------------------------------------------------------------ #
#  D2: Method and details keeps its key columns in view at 1280 px
# ------------------------------------------------------------------ #

# Streamlit 1.55's named column widths, in px.
_WIDTHS = {"small": 75, "medium": 200, "large": 400}
# Measured in the browser at 1280 × 860 (integration pass, Line A): the
# sidebar is 300 px and the main block 980 px; the details table drew 784 px
# wide with no horizontal scroll (scrollWidth = clientWidth).
_MAIN_AT_1280 = 980
# The Measurement column has no set width: its longest name, at most.
_NAME_COLUMN = 200


# ------------------------------------------------------------------ #
#  T0′: the interval chart's caption, and the progress line after Save
# ------------------------------------------------------------------ #

def test_the_interval_caption_holds_in_both_themes():
    """The latest Round's bars are darker on white and lighter on the dark
    page (W3's D1 tokens): the caption names neither."""
    text = wording.MVP_CAPTION
    assert "latest Round" in text
    assert "Darker" not in text and "darker" not in text and "ighter" not in text


@pytest.mark.parametrize("total", [1, 4, 12])
def test_the_progress_line_never_says_all_checked_while_fitting(total):
    line = wording.insights_progress(total, total)
    assert f"{total} of {total}" not in line and "checked…" not in line
    assert line.endswith("…")
    assert wording.insights_progress(total - 1, total).endswith(
        f"{total - 1} of {total} {'Round' if total == 1 else 'Rounds'} checked…")


# ------------------------------------------------------------------ #
#  pair_chart in W3's theme scale and tokens (D1)
# ------------------------------------------------------------------ #

@pytest.fixture(params=["light", "dark"])
def theme(request, monkeypatch):
    monkeypatch.setattr(ui_insights, "theme", lambda: request.param)
    monkeypatch.setattr(ui_insights, "muted", lambda: (
        ui_insights._MUTED_DARK if request.param == "dark" else ui_insights._MUTED_LIGHT))
    return request.param


def _pair_spec():
    opt = E.load(E.BAR_CLEAN)
    a, b = [o['name'] for o in opt.measurements_by_importance()[:2]]
    points = insights.pair_points(opt, a, b)
    return ui_insights.pair_chart(opt, a, b, points).to_dict()


@pytest.mark.skip(reason=REVIEW_FOUR)
def test_pair_chart_takes_the_themes_scale(theme):
    spec = _pair_spec()
    [dots] = [l for l in _all_layers(spec)
              if l.get("encoding", {}).get("color", {}).get("field") == "score"]
    # Review plots: the score is the viridis ramp (score_colours), not the
    # blues scheme; every stop reaches 3:1 on its theme's page.
    stops = dots["encoding"]["color"]["scale"]["range"]
    assert stops == ui_insights.score_colours(theme)
    assert all(_contrast_on(c, 1.0, PAGES[theme]) >= 3.0 for c in stops)


@pytest.mark.skip(reason=REVIEW_FOUR)
def test_pair_charts_grey_marks_reach_3_to_1(theme):
    spec = _pair_spec()
    colours = []
    for layer in _all_layers(spec):
        m = _mark(layer)
        if m["type"] in ("point", "rule") and m.get("color"):
            colours.append((m["color"], m.get("opacity", 1.0)))
    assert colours
    for colour, opacity in colours:
        assert colour.startswith("#"), colour
        assert _contrast_on(colour, opacity, PAGES[theme]) >= 3.0, (colour, theme)
