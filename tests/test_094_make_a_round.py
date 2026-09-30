"""0.8.0 BO views, Task 4 (spec 2026-09-30-bo-views §2.1, §3.1, §5; review
B4, I1, I4, I5, I12, M4, M5, M9, M12, M15): Make a round records how each
Round was generated, says a no-skill record on its track line, links a
warning to Insights (E2), says Next step above Generate (E3), and explains
its suggestions in a fold that is worked out only when opened."""
import json

import pytest
from streamlit.testing.v1 import AppTest

import bo_views
import insights
import ui_batch
import ui_insights
import wording
from food_bo import FoodOptimizer
from tests.test_app_ui import APP_PATH
from tests.test_insights import _burger
from ui_helpers import INSIGHTS_OPENED_KEY


@pytest.fixture(autouse=True)
def _isolate(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    with bo_views.fresh():
        yield


def _open_round(rounds=4, n=3):
    opt = _burger(rounds=rounds)
    ui_batch.ask_alone(opt, n_suggestions=n)
    return opt


def _app(opt, tab=wording.TAB_BATCH):
    at = AppTest.from_file(APP_PATH, default_timeout=300)
    at.session_state["_loaded_project"] = opt.project_name
    at.run()
    at.session_state["main_tab"] = tab
    at.run()
    assert not at.exception
    return at


def _links(at, tab=1):
    """The links on one tab (AppTest draws every tab's content)."""
    return [b for b in at.tabs[tab].button if str(b.key or "").startswith("link_")]


def _rerun(at, tab=wording.TAB_BATCH):
    """AppTest keeps no tab choice between runs: say it again."""
    at.session_state["main_tab"] = tab
    at.run()
    assert not at.exception


def _fold(at):
    return next(e for e in at.tabs[1].expander if e.label == wording.WHY_THESE)


# ------------------------------------------------------------------ #
#  The generation record (review B4)
# ------------------------------------------------------------------ #

def test_generate_records_how_the_round_was_made():
    opt = _open_round()
    again = FoodOptimizer(opt.project_name)
    record = again.round_generation[again.pending_batch_no]
    assert record["regime"] == "warm" and record["n"] == 3
    assert record["seed"] == min(r["formulation"] for r in again.pending_batch)
    assert record["rows"] == [int(n) for n in again.formulation_ids]
    assert bo_views.generation(again, again.pending_batch_no).exact


def test_a_cold_round_is_recorded_as_cold():
    opt = _burger(rounds=1)
    ui_batch.ask_alone(opt, n_suggestions=2)
    assert opt.round_generation[opt.pending_batch_no]["regime"] == "cold"


def test_recording_never_moves_a_suggestion():
    """The record is made before ask() and changes nothing it reads."""
    a, b = _burger(rounds=4, seed=3), _burger(rounds=4, seed=3)
    ui_batch.ask_alone(a, n_suggestions=3)
    with insights.RNG_LOCK:
        b.ask(n_suggestions=3)
    assert [r["recipe"] for r in a.pending_batch] == [r["recipe"] for r in b.pending_batch]


# ------------------------------------------------------------------ #
#  Why these suggestions (§3.1)
# ------------------------------------------------------------------ #

def test_the_fold_is_collapsed_and_costs_nothing_until_opened(monkeypatch):
    """Binding I15, review M15."""
    opt = _open_round()
    calls = []
    monkeypatch.setattr(bo_views, "why", lambda *a, **k: calls.append(1))
    at = _app(opt)
    fold = _fold(at)
    assert calls == [] and not fold.dataframe
    assert not [b for b in _links(at) if b.proto.type == "primary"]


def test_the_opened_fold_explains_each_row():
    opt = _open_round()
    at = _app(opt)
    at.session_state[ui_batch.WHY_KEY] = True
    _rerun(at)
    fold = _fold(at)
    [frame] = [d.value for d in fold.dataframe]
    assert list(frame.columns) == [
        wording.FORMULATION_CAP, wording.PREDICTED_SCORE_COLUMN, wording.MODEL_SD,
        wording.EXPECTED_IMPROVEMENT, wording.NEAREST_TESTED, wording.LARGEST_DIFFERENCES,
        wording.ROLE]
    numbers = [r["formulation"] for r in opt.pending_batch]
    assert list(frame[wording.FORMULATION_CAP]) == [f"Formulation {n}" for n in numbers]
    ei = list(frame[wording.EXPECTED_IMPROVEMENT])
    assert not ei[0].endswith(wording.PENDING_MARK)
    assert all(v.endswith(wording.PENDING_MARK) for v in ei[1:])       # picks 2+
    assert set(frame[wording.ROLE]) <= {wording.ROLE_REFINES, wording.ROLE_BALANCED,
                                        wording.ROLE_EXPLORES}
    assert all(" g" in cell for cell in frame[wording.LARGEST_DIFFERENCES] if cell)
    captions = [c.value for c in fold.caption]
    assert wording.WHY_PENDING in captions
    assert wording.WHY_RECOMPUTED not in captions                        # exact
    assert wording.why_chart_caption(f"{bo_views.POOL_SIZE:,}") in captions
    [chart] = [c for c in fold.children.values() if getattr(c, "type", "") == "vega_lite_chart"]
    spec = json.loads(chart.proto.spec)
    assert spec["usermeta"] == {"embedOptions": {"actions": False}}      # private


def test_an_own_formulation_is_labelled_and_valued_alone():
    opt = _open_round()
    no = opt.add_to_pending_batch({"Pea": 6.0, "Oil": 4.0}, note="Own formulation")
    w = bo_views.why(opt)
    own = next(r for r in w.rows if r.formulation == no)
    assert own.own and own.pick is None and not own.pending
    frame, _ = ui_batch.why_frame(opt, w, None)
    assert wording.why_row_label(no, own=True) in list(frame[wording.FORMULATION_CAP])


def test_the_guard_takes_the_roles_and_the_expected_improvement_away(monkeypatch):
    """Review I4: when the model is fitting noise the fold keeps its
    model-free columns and says why."""
    opt = _open_round()
    monkeypatch.setattr(insights, "noise_guard",
                        lambda *a, **k: insights.Guard(["floor", "skill"]))
    at = _app(opt)
    at.session_state[ui_batch.WHY_KEY] = True
    _rerun(at)
    fold = _fold(at)
    [frame] = [d.value for d in fold.dataframe]
    assert wording.ROLE not in frame.columns
    assert wording.EXPECTED_IMPROVEMENT not in frame.columns
    assert wording.guard_line([wording.GUARD_FLOOR]) in [c.value for c in fold.caption]


def test_a_cold_round_says_why_it_has_no_values():
    opt = _burger(rounds=1)
    ui_batch.ask_alone(opt, n_suggestions=2)
    w = bo_views.why(opt)
    assert w.cold


def test_qucb_names_its_optimistic_score():
    """Review I5: never "upper bound" beside a 95 % interval."""
    opt = _burger(rounds=4)
    opt.set_bo_config({"acquisition": "qucb"})
    ui_batch.ask_alone(opt, n_suggestions=2)
    frame, column = ui_batch.why_frame(opt, bo_views.why(opt), None)
    assert column == wording.OPTIMISTIC_SCORE and column in frame.columns


# ------------------------------------------------------------------ #
#  The track line and E2 (review I1, I7)
# ------------------------------------------------------------------ #

def _track(no_skill=None, rough=False, gap=None, see=False, record=True):
    fact = None if no_skill is None else insights.SkillFact(no_skill, 16, [2, 3, 4, 5],
                                                            "no_skill")
    return insights.RoundTrack(rough, 11, 12, gap, see, fact, [3, 4, 5], record)


def test_the_track_line_says_no_skill_after_the_record():
    text = ui_insights.track_text(_track(no_skill=-0.47))
    assert text == (wording.round_track_holding(11, 12, rounds=[3, 4, 5])
                    + wording.round_track_no_skill("−0.47", 16))
    alone = ui_insights.track_text(_track(no_skill=-0.47, record=False))
    assert alone == wording.round_track_no_skill("−0.47", 16).strip()


@pytest.mark.parametrize("t,warns", [(_track(), False), (_track(rough=True), True),
                                     (_track(gap=-1.2), True), (_track(see=True), True),
                                     (_track(no_skill=-0.4), True)])
def test_which_lines_are_warnings(t, warns):
    assert ui_batch._warns(t) == warns


def test_e2_links_a_warning_to_the_prediction_check(monkeypatch):
    opt = _open_round()
    monkeypatch.setattr(insights, "round_track", lambda o, c: _track(no_skill=-0.47))
    at = _app(opt, wording.TAB_INSIGHTS)               # the checks are worked out
    _rerun(at)
    [e2] = _links(at)
    assert (e2.key, e2.label, e2.proto.type) == ("link_trust", wording.LINK_PREDICTION_CHECK,
                                                 "tertiary")
    e2.click()
    at.run()
    assert at.session_state["main_tab"] == wording.TAB_INSIGHTS


def test_no_link_when_the_line_is_good_news(monkeypatch):
    opt = _open_round()
    monkeypatch.setattr(insights, "round_track", lambda o, c: _track())
    at = _app(opt, wording.TAB_INSIGHTS)
    _rerun(at)
    assert wording.round_track_holding(11, 12, rounds=[3, 4, 5]) in [
        c.value for c in at.tabs[1].caption]
    assert _links(at) == []


# ------------------------------------------------------------------ #
#  E3 above Generate (04 §3; review I1, I10, M12)
# ------------------------------------------------------------------ #

def _generate_screen(opt, monkeypatch, step):
    """Make a round with no Round open, Insights worked out in this session
    (E3 reads only what is worked out), Next step forced to `step`."""
    monkeypatch.setattr(insights, "next_step", lambda *a, **k: step)
    at = _app(opt, wording.TAB_INSIGHTS)
    del at.session_state[INSIGHTS_OPENED_KEY]
    _rerun(at)
    return at


@pytest.mark.parametrize("kind,label", [("close", wording.LINK_NEXT_STEP),
                                        ("untested", wording.LINK_NEXT_STEP),
                                        ("misses", wording.LINK_PREDICTION_CHECK)])
def test_e3_says_next_step_above_generate(monkeypatch, kind, label):
    opt = _burger(rounds=4)
    at = _generate_screen(opt, monkeypatch, insights.NextStep(kind))
    [e3] = _links(at)
    assert (e3.key, e3.label, e3.proto.type) == ("link_next", label, "tertiary")
    caps = [c.value for c in at.tabs[1].caption]
    text, _ = ui_insights.next_step_text(opt, insights.NextStep(kind))
    assert wording.first_sentence(text) in caps
    [generate] = [b for b in at.tabs[1].button if b.key == "generate"]
    assert generate.proto.type == "primary"


def test_e3_says_no_skill(monkeypatch):
    opt = _burger(rounds=4)
    fact = insights.SkillFact(-0.47, 16, [2, 3, 4], "no_skill")
    at = _generate_screen(opt, monkeypatch, insights.NextStep("no_skill", skill=fact))
    assert [b.label for b in _links(at)] == [wording.LINK_PREDICTION_CHECK]
    assert wording.first_sentence(wording.next_no_skill("−0.47", 16)) in [
        c.value for c in at.tabs[1].caption]


@pytest.mark.parametrize("kind", ["rising", "see", "early"])
def test_e3_is_silent_when_the_next_step_is_another_round(monkeypatch, kind):
    at = _generate_screen(_burger(rounds=4), monkeypatch, insights.NextStep(kind))
    assert _links(at) == []


def test_e3_keeps_its_line_but_drops_the_link_once_insights_was_opened(monkeypatch):
    """04 §3: no ping-pong."""
    opt = _burger(rounds=4)
    monkeypatch.setattr(insights, "next_step", lambda *a, **k: insights.NextStep("close"))
    at = _app(opt, wording.TAB_INSIGHTS)
    _rerun(at)
    assert _links(at) == []
    text, _ = ui_insights.next_step_text(opt, insights.NextStep("close"))
    assert wording.first_sentence(text) in [c.value for c in at.tabs[1].caption]


def test_e3_never_fits_or_waits(monkeypatch):
    """Review I10: with nothing worked out, E3 says nothing."""
    opt = _burger(rounds=4)
    fits = []
    monkeypatch.setattr(insights, "fit_all", lambda o: fits.append(1) or {})
    at = _app(opt)
    assert _links(at) == [] and fits == []


def test_the_column_help_names_its_basis():
    opt = _open_round()
    at = _app(opt)
    table = next(d for d in at.tabs[1].dataframe if wording.PREDICTED_SCORE_COLUMN in d.value)
    assert f"fitted to the {opt.complete_count()} scored formulations" in str(table.proto)


# ------------------------------------------------------------------ #
#  Review Focus (plan)
# ------------------------------------------------------------------ #

def test_a_regenerated_round_is_recorded_again_and_stays_exact():
    """Generate a different Round keeps its number and asks again with a
    new seed: the record follows it."""
    opt = _open_round()
    no, old = opt.pending_batch_no, dict(opt.round_generation[opt.pending_batch_no])
    opt.set_pending_batch(None)
    ui_batch.ask_alone(opt, n_suggestions=3, batch_no=no)
    new = opt.round_generation[no]
    assert new["seed"] != old["seed"] and new["seed"] == min(
        r["formulation"] for r in opt.pending_batch)
    assert bo_views.why(opt).exact


def test_a_process_only_study_explains_its_trials():
    """No ingredients and no Formulation total: the fold speaks of trials
    and settings, and the pool is the settings' own box."""
    opt = FoodOptimizer("ferment_why")
    opt.add_process_parameter("Mixing time", 1, 10, unit="min")
    opt.add_process_parameter("Temperature", 20, 40, unit="°C")
    opt.add_objective("Acidity", 1.0, goal="max", min_val=0, max_val=10, unit="pH")
    for k in range(6):
        opt.tell({"Mixing time": 1 + 1.5 * k, "Temperature": 20 + (7 * k) % 20},
                 {"Acidity": 3 + 0.5 * k - 0.02 * k * k}, formulation_no=k + 1, batch_no=1)
    ui_batch.ask_alone(opt, n_suggestions=2)
    w = bo_views.why(opt)
    assert not w.cold and len(w.rows) == 2 and w.pool_mean is not None
    frame, _ = ui_batch.why_frame(opt, w, None)
    labels = [wording.for_project(opt, t) for t in frame[wording.FORMULATION_CAP]]
    assert all(t.startswith("Trial ") for t in labels)
    units = {u for r in w.rows for _, _, u in r.differences}
    assert units <= {"min", "°C"}
    assert "trial" in wording.for_project(opt, wording.why_chart_caption("3,000"))
