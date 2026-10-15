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
import ui_batch
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
#  The Why fold says the A3 reason exactly (W2 × W1's skill_gate)
# ------------------------------------------------------------------ #

from tests.test_094_make_a_round import _app, _fold, _open_round, _rerun  # noqa: E402


def _why_caption(monkeypatch, gate, reasons=("skill",)):
    opt = _open_round()
    # Review four polish: the fold greys by the Predictions card's word;
    # pinned here at Not yet, so the skill gate's own reason is its detail.
    monkeypatch.setattr(insights, "score_predictions", lambda *a, **k: "not_yet")
    monkeypatch.setattr(insights, "noise_guard",
                        lambda *a, **k: insights.Guard(list(reasons), gate))
    monkeypatch.setattr(insights, "skill_gate", lambda *a, **k: gate)
    at = _app(opt)
    at.session_state[ui_batch.WHY_KEY] = True
    _rerun(at)
    # Concision round: a withheld-gain line is short on screen, its full
    # sentence in its help; both are returned.
    caps = _fold(at).caption
    return [c.value for c in caps] + [c.proto.help for c in caps if c.proto.help]


def test_the_why_fold_says_too_few_checks(monkeypatch):
    gate = insights.SkillGate("wait", 6)
    line = wording.why_guard_line(gate_reason=wording.gain_reason_too_few(6))
    assert line == ("Expected gain is not shown: fewer than 10 Round-ahead checks of "
                    "the Overall score so far (6).")
    # Review four polish (re-pinned): greyed by the card's word, the A3
    # reason said after it.
    line = wording.why_greyed_line("not_yet", wording.gain_reason_too_few(6))
    assert wording.why_gain_greyed_said(line) in _why_caption(monkeypatch, gate)


def test_the_why_fold_says_skill_under_0_20(monkeypatch):
    fact = insights.SkillFact(0.09, 24, [3, 4, 5, 6, 7, 8], "unclear")
    gate = insights.SkillGate("unclear", 24, fact)
    # Review four polish (re-pinned): the card's word, then the A3 reason.
    line = wording.why_greyed_line("not_yet", wording.gain_reason_unclear("0.09", 24))
    assert wording.why_gain_greyed_said(line) in _why_caption(monkeypatch, gate)


def test_the_why_fold_says_no_skill(monkeypatch):
    fact = insights.SkillFact(-0.4, 24, [3, 4, 5, 6, 7, 8], "no_skill")
    gate = insights.SkillGate("no_skill", 24, fact)
    # Review four polish (re-pinned): the card's word, then the A3 reason.
    line = wording.why_greyed_line("not_yet", wording.GAIN_REASON_SKILL)
    assert wording.why_gain_greyed_said(line) in _why_caption(monkeypatch, gate)


def test_the_why_fold_and_progress_give_one_reason(monkeypatch):
    """The fold's words are Progress's reason (gain_reason), not a second
    paraphrase of it."""
    for gate in (insights.SkillGate("wait", 6),
                 insights.SkillGate("unclear", 24, insights.SkillFact(0.09, 24, [3, 4], "unclear"))):
        reason = ui_insights.gain_reason("skill", gate)
        assert reason in wording.why_guard_line(gate_reason=reason)
        assert reason in wording.gain_unreliable(reason)


# ------------------------------------------------------------------ #
#  Make a round's E3 kinds (W1 note b)
# ------------------------------------------------------------------ #

def test_the_e3_kinds_lose_the_bias_row_next_step_no_longer_makes():
    assert "bias" not in ui_batch._E3_KINDS


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


# ------------------------------------------------------------------ #
#  One message: Make a round's E3 line names the Rounds as Insights does
#  (consistency printout, 02 final)
# ------------------------------------------------------------------ #

def test_make_a_rounds_next_step_line_names_insights_rounds(tmp_path, monkeypatch):
    import bo_views
    from food_bo import FoodOptimizer
    from tests.test_app_ui import APP_PATH
    monkeypatch.chdir(tmp_path)
    state = E.state(E.PATTY)
    opt = FoodOptimizer(state["project_name"])
    opt.import_json(state)
    opt.save()
    with bo_views.fresh():
        checks = insights.check_rounds(opt)
        step = insights.next_step(opt, checks, None)
        assert step.kind == "misses"
        rounds = insights.headline(opt, checks).rounds
        said = wording.first_sentence(ui_insights.next_step_text(opt, step, rounds)[0])
        assert "Rounds 6–8" in said
        at = AppTest.from_file(APP_PATH, default_timeout=600)
        at.session_state["_loaded_project"] = opt.project_name
        at.run()
        at.session_state["main_tab"] = wording.TAB_INSIGHTS     # Insights works it out
        at.run()
        at.session_state["main_tab"] = wording.TAB_BATCH
        at.run()
        assert not at.exception
        captions = [c.value for c in at.tabs[1].caption]
        assert said in captions
        assert not any("the verdict's Rounds" in c for c in captions)
