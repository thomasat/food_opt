"""W2 of the 0.8.0 e2e fixes, Results and the flashes (spec 2026-10-01 rev.
2; review I7, I8, I10, I13 M5; 04 M-3): the View formulation box back on
Best so far after a save; sticky flashes that outlast the app's own
reruns; this tab's confirmations above the View formulation box, scrolled
to; the restore warning naming the open Round; Best so far's Repeats;
"Round 1 · Make 2 trials"; Save uploaded results lit once its preview
shows; Set up's callout for "Change X's Highest on Set up"."""
import io

import pytest
from streamlit.testing.v1 import AppTest

import insights
import ui_batch
import ui_results
import ui_setup
import wording
from food_bo import FoodOptimizer
from tests import e2e_fixtures as E
from tests.test_app_ui import (  # noqa: F401
    APP_PATH, _submit_button, _tab_primaries, burger, scored, scored_open_batch)
from tests.test_complete_results_ui import one_waiting  # noqa: F401
from ui_helpers import SCROLL_KEY, SETUP_CALLOUT_KEY

# ------------------------------------------------------------------ #
#  Sticky flashes (review I10)
# ------------------------------------------------------------------ #

FLASH_SCRIPT = """
import streamlit as st
from ui_helpers import app_rerun, flash, go_to_tab, render_flash
st.session_state.setdefault("main_tab", "A")
# The app's own rerun (the poller's fragment) comes before this run's
# render_flash, as in app.py; a click comes after it.
if st.session_state.pop("_poll", False):
    app_rerun()
render_flash()
with st.container(key="slot"):
    render_flash(place="here")
if st.button("save"):
    flash("success", "Round 6 recorded.", sticky=True)
    st.rerun()
if st.button("here"):
    flash("success", "Moved.", sticky=True, place="here")
    st.rerun()
if st.button("plain"):
    flash("info", "Once.")
    st.rerun()
if st.button("move"):
    go_to_tab("A")      # a move the app makes, even to the tab on screen
"""


def _flash_app():
    at = AppTest.from_string(FLASH_SCRIPT)
    at.run()
    return at


def _click(at, label):
    next(b for b in at.button if b.label == label).click()
    at.run()
    assert not at.exception


def _values(block):
    """Every message drawn inside `block`, nested containers included."""
    out = []
    for child in block.children.values():
        out += _values(child) if hasattr(child, "children") and not hasattr(child, "value") \
            else [child.value]
    return out


def _poll(at):
    """The poller's rerun: the app's own, no action of the user's."""
    at.session_state["_poll"] = True
    at.run()
    assert not at.exception


def _top(at):
    return _values(at.main.children[0])


def _slot(at):
    return _values(at.main.children[1])


def test_a_sticky_flash_outlasts_the_apps_own_reruns():
    """01 O3: "Round 6 recorded." vanished about 2.8 s after Save, on the
    first rerun the Insights poller made."""
    at = _flash_app()
    _click(at, "save")
    assert _top(at) == ["Round 6 recorded."]
    _poll(at)                                # the poller's rerun
    assert _top(at) == ["Round 6 recorded."]
    _poll(at)
    assert _top(at) == ["Round 6 recorded."]
    # Integration (review I10, "until the next action"): any rerun the
    # app did not make is the user's, and takes it down.
    at.run()
    assert _top(at) == []


def test_the_next_flash_replaces_it_and_a_plain_flash_shows_once():
    at = _flash_app()
    _click(at, "save")
    _click(at, "plain")
    assert _top(at) == ["Once."]
    at.run()
    assert _top(at) == []


def test_go_to_tab_and_a_move_to_another_tab_take_it_down():
    at = _flash_app()
    _click(at, "save")
    _click(at, "move")
    at.run()            # AppTest keeps the rerun's first pass on screen
    assert _top(at) == []
    at = _flash_app()
    _click(at, "save")
    at.session_state["main_tab"] = "B"       # the reader clicked another tab
    at.run()
    assert _top(at) == []


def test_a_placed_flash_is_drawn_in_its_slot_only():
    at = _flash_app()
    _click(at, "here")
    assert _top(at) == [] and _slot(at) == ["Moved."]
    _poll(at)
    assert _slot(at) == ["Moved."]


def test_a_message_queued_before_sticky_flashes_still_shows():
    """A session from before this change queued (kind, message) pairs."""
    at = _flash_app()
    at.session_state["_flash_messages"] = [("success", "Old shape.")]
    at.run()
    assert _top(at) == ["Old shape."]


# ------------------------------------------------------------------ #
#  Results on the app
# ------------------------------------------------------------------ #

def _app(tab):
    at = AppTest.from_file(APP_PATH, default_timeout=180)
    at.run()
    at.session_state["main_tab"] = tab
    at.run()
    assert not at.exception
    return at


def _order(tab):
    """The Results tab's top-level elements in drawing order, flattened one
    level: (kind, key or value)."""
    out = []
    for child in tab.children.values():
        key = getattr(child.proto, "id", "") if hasattr(child, "proto") else ""
        out.append((type(child).__name__, key, child))
    return out


def test_after_save_results_the_view_box_is_back_on_best_so_far(scored_open_batch):
    """02 I5 (review I10): once F8 had been viewed, every later Save landed
    on "Formulation 8 (Round 2) · Not scored", hiding the Round summary."""
    at = _app(wording.TAB_RESULTS)
    at.selectbox(key="result_view_burger").set_value(1)
    at.run()
    assert at.selectbox(key="result_view_burger").value == 1
    at.session_state["main_tab"] = wording.TAB_BATCH
    at.run()
    no = FoodOptimizer("burger").pending_batch[0]["formulation"]
    at.number_input(key=f"f{no}_Firmness").set_value(6.0)
    at.number_input(key=f"f{no}_Juiciness").set_value(7.0)
    at.run()
    _submit_button(at, "Save results").click()
    at.run()
    assert not at.exception
    box = at.selectbox(key="result_view_burger")
    assert box.value == ui_results.BEST_CHOICE
    assert box.proto.set_value                 # sent to the browser again
    assert box.proto.options[0] == wording.result_view_option(no, 2, best=True)
    assert wording.batch_recorded_flash(2) in [s.value for s in at.success]


def test_move_to_not_scored_confirms_above_the_view_box_and_scrolls_there(one_waiting):
    """01 O3: "Formulation 19 moved to Not scored." rendered 501 px above
    the button just pressed."""
    at = _app(wording.TAB_RESULTS)
    at.selectbox(key="result_view_burger").set_value(4)
    at.run()
    _submit_button(at, "Move to Not scored").click()
    at.run()
    at.button(key="set_aside_4__yes").click()
    at.run()
    assert not at.exception
    said = wording.moved_to_not_scored(4)
    results = at.tabs[2]
    slot = next(c for c in results.children.values()
                if said in [s.value for s in getattr(c, "success", [])])
    kinds = list(results.children.values())
    box = next(k for k, c in enumerate(kinds)
               if "result_view_burger" in [s.key for s in getattr(c, "selectbox", [])]
               or getattr(c, "key", None) == "result_view_burger")
    assert kinds.index(slot) < box
    # Scrolled to: the slot's key is the scroll target app.py ran.
    scripts = [h.proto.body for h in at.get("html")]
    assert any(".st-key-results_flash" in body for body in scripts), scripts


def test_the_restore_warning_names_the_open_round(scored_open_batch):
    """02 I7: "Replace Pea burger patty, which has 24 formulations?" said
    nothing of the open Round 7 and its 4 suggestions."""
    opt = FoodOptimizer("burger")
    no = opt.pending_batch[0]["formulation"]
    opt.save_result_drafts({no: {"results": {"Firmness": 6.0}, "note": ""}})
    at = _app(wording.TAB_RESULTS)
    at.session_state["_restore_candidate"] = FoodOptimizer("burger").export_json()
    at.run()
    assert not at.exception
    lost = wording.open_round_lost(2, 1, 1)
    assert any(lost in w.value for w in at.sidebar.warning), [w.value for w in at.sidebar.warning]


def test_best_so_far_lists_its_repeats_with_the_mean():
    """03 noisy (review I7): F14 91.45 with Repeats 67.30, 66.34, 75.61; the
    mean from the stored scores reads 75.17 (stage-1 report §1.2 #4)."""
    opt = E.load(E.BAR_NOISY)
    assert ui_results.best_repeats_text(opt, checks=()) == (
        "Repeats of Formulation 14: 67.30, 66.34, 75.61 (mean with Formulation 14: 75.17).")


def test_a_disagreeing_repeat_is_listed_and_marked(monkeypatch):
    opt = E.load(E.BAR_NOISY)
    row = opt.index_of_formulation(20)
    monkeypatch.setattr(insights, "disagreeing_repeat",
                        lambda o, c, m=None: insights.Disagreement(20, 14, -24.15, row))
    assert ui_results.best_repeats_text(opt, checks=[object()]) == (
        "Repeats of Formulation 14: 67.30 (disagrees), 66.34, 75.61 "
        "(mean with Formulation 14: 75.17).")


def test_no_repeats_no_line():
    assert ui_results.best_repeats_text(E.load(E.PATTY), checks=()) == ""


def test_results_shows_the_repeats_under_best_so_far(scored):
    """Display only: Best so far is still F2's single score."""
    scored.tell({"Pea protein": 20.0, "Methylcellulose": 2.0},
                {"Juiciness": 7.0, "Firmness": 9.0}, formulation_no=4, batch_no=2,
                note="Repeat of Formulation 2")
    at = _app(wording.TAB_RESULTS)
    line = ui_results.best_repeats_text(FoodOptimizer("burger"), checks=())
    assert line.startswith("Repeat of Formulation 2: ")
    assert line in [c.value for c in at.tabs[2].caption]
    assert FoodOptimizer("burger").best_formulation_no() == 2


# ------------------------------------------------------------------ #
#  Make a round
# ------------------------------------------------------------------ #

def test_a_process_only_round_says_trials(tmp_path, monkeypatch):
    """04 M-3: "Round 4 · Make 3 formulations" in a study of settings."""
    monkeypatch.chdir(tmp_path)
    opt = FoodOptimizer("ferment")
    opt.add_process_parameter("Temperature", 20, 40, unit="°C")
    opt.add_objective("Acidity", 1.0, goal="max", min_val=0, max_val=10)
    opt.set_pending_batch([{"Temperature": 25.0}, {"Temperature": 35.0}])
    at = AppTest.from_file(APP_PATH, default_timeout=180)
    at.session_state["_loaded_project"] = "ferment"
    at.run()
    at.session_state["main_tab"] = wording.TAB_BATCH
    at.run()
    assert not at.exception
    title = wording.for_project(opt, wording.make_these(1, 2))
    assert title == "**Round 1 · Make 2 trials**"
    assert title in [m.value for m in at.tabs[1].markdown]


def test_save_uploaded_results_is_lit_once_the_preview_shows(scored_open_batch):
    """02 M5: after the upload the only lit button was Download."""
    no = FoodOptimizer("burger").pending_batch[0]["formulation"]
    filled = io.BytesIO(f"Formulation,Juiciness,Firmness\n{no},7,6\n".encode())
    filled.name = "round.csv"
    at = _app(wording.TAB_BATCH)
    assert wording.DOWNLOAD_BATCH_SHEETS in _tab_primaries(at, 1)
    parsed = ui_batch._read_results_file(at.session_state["optimizer"], filled)
    at.session_state["_results_upload"] = parsed
    at.session_state["main_tab"] = wording.TAB_BATCH
    at.run()
    assert not at.exception
    assert _tab_primaries(at, 1) == [wording.SAVE_UPLOADED_RESULTS]


# ------------------------------------------------------------------ #
#  Set up's callout (A7, review I8)
# ------------------------------------------------------------------ #

def test_the_callout_names_the_row_and_its_highest(burger):
    text = ui_setup.callout_text(burger, wording.change_on_setup("Pea protein"))
    assert text == wording.setup_callout("Pea protein", "25 g")
    assert ui_setup.callout_text(burger, wording.change_on_setup(
        "Methylcellulose", highest=False)) == wording.setup_callout(
        "Methylcellulose", "0 g", highest=False)
    assert ui_setup.callout_text(burger, "Something else") == ""


def test_the_callout_lands_above_the_grid_and_clears_on_a_tab_move(burger):
    at = AppTest.from_file(APP_PATH, default_timeout=180)
    at.session_state["_loaded_project"] = "burger"
    at.run()
    at.session_state[SETUP_CALLOUT_KEY] = wording.change_on_setup("Pea protein")
    at.session_state["main_tab"] = wording.TAB_SETUP
    at.session_state[SCROLL_KEY] = "setup_callout"
    at.run()
    assert not at.exception
    said = wording.setup_callout("Pea protein", "25 g")
    assert said in [i.value for i in at.tabs[0].info]
    scripts = [h.proto.body for h in at.get("html")]
    assert any(".st-key-setup_callout" in body for body in scripts)
    at.session_state["main_tab"] = wording.TAB_RESULTS
    at.run()
    assert said not in [i.value for i in at.info]
    assert SETUP_CALLOUT_KEY not in at.session_state


