"""Integration of the 0.8.0 e2e fixes (W2 → W1 → W3; spec 2026-10-01 rev. 2
and its Overrides): what joins the three waves, and the loose ends they
left.

- Sticky flashes (W2, review I10) come down at the user's next rerun of
  any kind: a click, a widget, a fold that reruns.
- D5: Share of score shown as stored, to one decimal.
"""
from streamlit.testing.v1 import AppTest

import wording
from tests import e2e_fixtures as E

# ------------------------------------------------------------------ #
#  Sticky flashes and the app's own reruns (W2 I10 × W3 C1)
# ------------------------------------------------------------------ #

# A stand-in for the app's order: the flash is drawn first, then the user's
# clicks happen after it.
STICKY_SCRIPT = """
import streamlit as st
from ui_helpers import flash, render_flash


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


def test_any_rerun_of_the_users_takes_it_down():
    """Review I10: "until the next action". A rerun the app did not make is
    the user's: a click, a widget, a fold opened (a fold that reruns)."""
    at = _sticky_app()
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
