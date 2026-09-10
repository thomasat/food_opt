"""Tests for ui_helpers: the two UI patterns every destructive/saving action uses."""
from streamlit.testing.v1 import AppTest

FLASH_SCRIPT = """
import streamlit as st
from ui_helpers import flash, render_flash
render_flash()
if st.button("go"):
    flash("success", "Done.")
    st.rerun()
"""


def test_flash_is_shown_on_the_run_after_rerun():
    at = AppTest.from_string(FLASH_SCRIPT)
    at.run()
    assert [s.value for s in at.success] == []
    at.button[0].click()
    at.run()
    assert [s.value for s in at.success] == ["Done."]


def test_flash_is_shown_only_once():
    at = AppTest.from_string(FLASH_SCRIPT)
    at.run()
    at.button[0].click()
    at.run()
    at.run()
    assert [s.value for s in at.success] == []


CONFIRM_SCRIPT = """
import streamlit as st
from ui_helpers import confirm_action
st.session_state.setdefault("done", 0)
if confirm_action("del", "Delete thing", "Really delete?", confirm_label="Yes, delete"):
    st.session_state["done"] += 1
st.write(f"done={st.session_state['done']}")
"""


def _btn(at, label):
    return next(b for b in at.button if b.label == label)


def test_confirm_action_needs_two_clicks():
    at = AppTest.from_string(CONFIRM_SCRIPT)
    at.run()
    _btn(at, "Delete thing").click()
    at.run()
    assert any("Really delete?" in w.value for w in at.warning)
    assert at.session_state["done"] == 0
    # The warning must sit above the confirm/cancel buttons, not below them:
    # walk main's ordered children and check the Warning element's position
    # precedes the Block (the st.columns row) holding the Yes/Cancel buttons.
    children = list(at.main.children.values())
    warning_idx = next(i for i, c in enumerate(children) if type(c).__name__ == "Warning")
    buttons_block_idx = next(
        i for i, c in enumerate(children)
        if type(c).__name__ == "Block" and any(b.label == "Yes, delete" for b in c.get("button"))
    )
    assert warning_idx < buttons_block_idx
    _btn(at, "Yes, delete").click()
    at.run()
    assert at.session_state["done"] == 1
    assert not any("Really delete?" in w.value for w in at.warning)


TWO_CONFIRM_SCRIPT = """
import streamlit as st
from ui_helpers import confirm_action, other_confirmation
for key in ("first", "last"):
    if confirm_action(key, f"Arm {key}", f"Really {key}?",
                      confirm_label=f"Yes, {key}",
                      disabled=other_confirmation(key)):
        st.session_state[f"{key}_done"] = True
"""


def test_arming_the_first_site_greys_the_last_one_on_the_same_frame():
    """The gap a scan of `{key}__pending` flags left open: the flag is set at
    the point in the script where its own confirm_action runs, so anything
    drawn before it answered 'nothing is armed'. One key answers everywhere."""
    at = AppTest.from_string(TWO_CONFIRM_SCRIPT)
    at.run()
    assert not at.button(key="first__btn").disabled
    assert not at.button(key="last__btn").disabled
    _btn(at, "Arm first").click()
    at.run()
    assert at.session_state["_armed_confirmation"] == "first"
    assert at.button(key="last__btn").disabled
    assert not at.button(key="first__btn").disabled
    assert [b.label for b in at.button if b.proto.type == "primary"] == ["Yes, first"]


def test_two_arming_clicks_in_one_run_leave_exactly_one_armed():
    """Both buttons were live on the frame the user clicked, so both clicks
    arrive together. The second must be dropped, not light a second Yes."""
    at = AppTest.from_string(TWO_CONFIRM_SCRIPT)
    at.run()
    _btn(at, "Arm first").click()
    _btn(at, "Arm last").click()
    at.run()
    assert at.session_state["_armed_confirmation"] == "first"
    assert [b.label for b in at.button if b.proto.type == "primary"] == ["Yes, first"]
    assert [w.value for w in at.warning] == ["Really first?"]
    # Answering the one that is armed frees the other.
    _btn(at, "Cancel").click()
    at.run()
    at.run()
    assert "_armed_confirmation" not in at.session_state
    assert not at.button(key="last__btn").disabled


def test_confirm_action_cancel():
    at = AppTest.from_string(CONFIRM_SCRIPT)
    at.run()
    _btn(at, "Delete thing").click()
    at.run()
    _btn(at, "Cancel").click()
    at.run()
    assert at.session_state["done"] == 0
    assert not any("Really delete?" in w.value for w in at.warning)


import pytest

from ui_helpers import (
    TAB_BATCH, TAB_RESULTS, TAB_SETUP, fmt_amount, goal_line, join_unit,
    landing_tab, number_list, plural, readiness, saved_line, scale_error,
    table_height,
)


def test_plural_uses_real_grammar():
    assert plural(0, "formulation") == "0 formulations"
    assert plural(1, "formulation") == "1 formulation"
    assert plural(3, "result") == "3 results"


def test_number_list_reads_as_english():
    assert number_list([1]) == "1"
    assert number_list([1, 2]) == "1 and 2"
    assert number_list([7, 8, 9]) == "7, 8 and 9"


def test_fmt_amount_always_shows_two_decimals_with_the_unit():
    """A weighing sheet is read down the column: 0.30 g, 33.90 g, 68.00 g."""
    assert fmt_amount(12.5, "g") == "12.50 g"
    assert fmt_amount(12.0, "g") == "12.00 g"
    assert fmt_amount(0.804, "g") == "0.80 g"
    assert fmt_amount(0.3, "g") == "0.30 g"
    assert fmt_amount(-0.001, "g") == "0.00 g"     # never '-0.00 g'
    assert fmt_amount(12.5, "") == "12.50"
    assert fmt_amount(None, "g") == ""


def test_fmt_setting_rounds_a_dial_to_what_a_dial_can_hold():
    """188.494 °C is a precision no oven has, and 180.00 °C is one nobody
    typed. Two decimals at most, trailing zeros stripped."""
    from ui_helpers import fmt_setting
    assert fmt_setting(188.4936, "°C") == "188.49 °C"
    assert fmt_setting(180.0, "°C") == "180 °C"
    assert fmt_setting(12.5, "min") == "12.5 min"
    assert fmt_setting(0.0, "°C") == "0 °C"
    assert fmt_setting(-0.001, "°C") == "0 °C"
    assert fmt_setting(180.0, "") == "180"
    assert fmt_setting(None, "°C") == ""


def test_join_unit_and_goal_line_are_re_exported():
    assert join_unit("7", "/10") == "7/10"
    assert goal_line({"goal": "target", "target": 6, "unit": "N"}) == "target 6 N"


def test_scale_error_names_the_value_the_scale_and_the_fix():
    obj = {"name": "Firmness", "min_val": 0.0, "max_val": 10.0, "unit": "N"}
    assert scale_error(obj, 6.0) == ""
    assert scale_error(obj, None) == ""
    assert scale_error(obj, 12.0) == (
        "Firmness 12 N is outside your scale of 0 to 10 N. Widen the scale in "
        "Set up, or check the value."
    )


def test_table_height_grows_with_rows_and_stops_at_the_cap():
    assert table_height(1) == 75
    assert table_height(3) == 145
    assert table_height(0) == table_height(1)          # never a zero-height table
    assert table_height(50, max_rows=12) == table_height(12, max_rows=12)


def test_saved_line_shows_the_time_today_and_the_date_before_that():
    from datetime import datetime, timedelta
    now = datetime.now().astimezone()
    today = now.replace(hour=14, minute=32)
    assert saved_line(today) == "Saved 14:32 · automatically, to this Mac"
    earlier = today - timedelta(days=3)
    line = saved_line(earlier)
    assert line.startswith("Saved ") and line.endswith("· automatically, to this Mac")
    assert earlier.strftime("%b") in line


class _FakeOpt:
    def __init__(self, variables, objectives, pending_batch=None,
                 X_history=None, skipped=None):
        self.variables = variables
        self.objectives = objectives
        self.pending_batch = pending_batch
        self.X_history = X_history or []
        self.skipped = skipped or []


def test_readiness_names_the_one_missing_thing():
    assert readiness(_FakeOpt([], [])) == (
        False, "Add at least one ingredient or process setting.")
    ing = [{"name": "Water", "category": "ingredient"}]
    assert readiness(_FakeOpt(ing, [])) == (False, "Add at least one measurement.")
    assert readiness(_FakeOpt(ing, [{"name": "Firmness", "weight": 1.0}])) == (True, "")


def test_a_project_of_process_settings_alone_is_ready():
    """A fermentation project varies incubation temperature, time and culture
    dose and weighs nothing out. It is a project, not an incomplete one."""
    setting = [{"name": "Incubation temperature", "category": "process"}]
    meas = [{"name": "Acidity", "weight": 1.0}]
    assert readiness(_FakeOpt(setting, [])) == (
        False, "Add at least one measurement.")
    assert readiness(_FakeOpt(setting, meas)) == (True, "")
    # ...and the landing rule reads it the same way: complete, nothing made
    # yet, so it opens on its set-up.
    assert landing_tab(_FakeOpt(setting, meas)) == TAB_SETUP
    assert landing_tab(_FakeOpt(setting, meas, X_history=[[0.5]])) == TAB_RESULTS


def test_landing_tab_follows_the_spec_rule():
    assert landing_tab(_FakeOpt([], [])) == TAB_SETUP
    ing = [{"name": "Water", "category": "ingredient"}]
    meas = [{"name": "Firmness", "weight": 1.0}]
    assert landing_tab(_FakeOpt(ing, [])) == TAB_SETUP
    # Set up is complete but nothing has been made: the set-up is what there
    # is to look at, and Results would be an empty tab.
    assert landing_tab(_FakeOpt(ing, meas)) == TAB_SETUP
    scored = _FakeOpt(ing, meas, X_history=[[0.5]])
    assert landing_tab(scored) == TAB_RESULTS
    # A batch nobody managed to make still counts as results held.
    left_out = _FakeOpt(ing, meas, skipped=[{"formulation": 1}])
    assert landing_tab(left_out) == TAB_RESULTS
    # An unrecorded batch outranks both.
    mid = _FakeOpt(ing, meas, X_history=[[0.5]],
                   pending_batch=[{"formulation": 1, "recipe": {"Water": 1.0}}])
    assert landing_tab(mid) == TAB_BATCH
    fresh_batch = _FakeOpt(ing, meas,
                           pending_batch=[{"formulation": 1,
                                           "recipe": {"Water": 1.0}}])
    assert landing_tab(fresh_batch) == TAB_BATCH


# The two halves of the move, in the same order as app.py: the pending target
# is drained into the widget key BEFORE st.tabs, and on_change="rerun" is what
# binds that key to the frontend at all.
GO_TO_TAB_SCRIPT = """
import streamlit as st
from ui_helpers import TAB_BATCH, TAB_SETUP, TAB_RESULTS, go_to_tab
if "_pending_tab" in st.session_state:
    st.session_state["main_tab"] = st.session_state.pop("_pending_tab")
st.tabs([TAB_SETUP, TAB_BATCH, TAB_RESULTS], key="main_tab",
        on_change="rerun")
if st.button("go"):
    go_to_tab(TAB_BATCH)
st.write(f"tab={st.session_state.get('main_tab')}")
"""

# go_to_tab on its own, with no tabs widget to drain it: what it leaves behind.
GO_TO_TAB_ALONE_SCRIPT = """
import streamlit as st
from ui_helpers import TAB_BATCH, go_to_tab
if st.button("go"):
    go_to_tab(TAB_BATCH)
"""


def test_go_to_tab_switches_the_open_tab():
    at = AppTest.from_string(GO_TO_TAB_SCRIPT)
    at.run()
    at.button[0].click()
    at.run()
    assert at.session_state["main_tab"] == "2 · Make a batch"


def test_go_to_tab_defers_the_target_instead_of_writing_the_widget_key():
    """The tabs widget already exists when a handler runs, so go_to_tab must
    not touch its key: it parks the target for the next run to drain."""
    at = AppTest.from_string(GO_TO_TAB_ALONE_SCRIPT)
    at.run()
    at.button[0].click()
    at.run()
    assert at.session_state["_pending_tab"] == "2 · Make a batch"
    assert "main_tab" not in at.session_state


# A select box that a button empties. Popping the widget's key does not reach
# the browser, so the clear is parked and honoured before the widget is built.
CLEAR_SELECTION_SCRIPT = """
import streamlit as st
from ui_helpers import clear_selection, take_clear
take_clear("pick")
st.selectbox("Pick", [1, 2, 3], index=None, key="pick")
if st.button("Done"):
    clear_selection("pick")
    st.rerun()
"""


def test_clear_selection_empties_the_box_on_the_next_run():
    at = AppTest.from_string(CLEAR_SELECTION_SCRIPT)
    at.run()
    at.selectbox(key="pick").set_value(2)
    at.run()
    assert at.session_state["pick"] == 2
    at.button[0].click()
    at.run()
    # Assigned, not popped: an assignment is sent to the browser, so the value
    # cannot come back with the next click.
    assert at.session_state["pick"] is None
    assert "_clear_pick" not in at.session_state
    assert at.selectbox(key="pick").value is None


def test_take_clear_is_a_no_op_when_nothing_asked_for_it():
    at = AppTest.from_string(CLEAR_SELECTION_SCRIPT)
    at.run()
    at.selectbox(key="pick").set_value(3)
    at.run()
    at.run()
    assert at.session_state["pick"] == 3
