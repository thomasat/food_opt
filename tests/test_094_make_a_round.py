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
# Review four, ruling 2: the Why fold's "?" says the gain is greyed, not hidden.
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
    # Review guide: the status line's "Open Review" is not E2's or E3's.
    return [b for b in at.tabs[tab].button if str(b.key or "").startswith("link_")
            and b.key != "link_review_status"]


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
    """T47 review minor 2: one row still waiting for a result is not among
    the rows ask() learns from, so it is not in the record either."""
    opt = _burger(rounds=4)
    last = len(opt.Y_history) - 1
    waiting = int(opt.formulation_ids[last])
    opt.results_history[last] = {k: v for k, v in list(opt.results_history[last].items())[:1]}
    assert not opt.fully_measured(last)
    ui_batch.ask_alone(opt, n_suggestions=3)
    again = FoodOptimizer(opt.project_name)
    record = again.round_generation[again.pending_batch_no]
    assert record["regime"] == "warm" and record["n"] == 3
    assert record["seed"] == min(r["formulation"] for r in again.pending_batch)
    assert record["rows"] == [int(n) for n in again.formulation_ids if int(n) != waiting]
    assert record["picks"] == [r["formulation"] for r in again.pending_batch]
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
    assert calls == [] and not fold.dataframe and not fold.table
    assert not [b for b in _links(at) if b.proto.type == "primary"]


def test_the_opened_fold_explains_each_row(monkeypatch):
    # A model with skill: the A3 gate (0.8.0 e2e fixes, review I4) would
    # withhold the fold's model columns on this small project.
    monkeypatch.setattr(insights, "skill_gate", lambda opt, checks, fact=None: None)
    opt = _open_round()
    at = _app(opt)
    at.session_state[ui_batch.WHY_KEY] = True
    _rerun(at)
    fold = _fold(at)
    # Review critique (fix 9): a static table (headers wrap at 1280 px), short
    # headers; Role holds the role and Source where the row came from; the
    # model sd is the map's y axis.
    [table] = list(fold.table)
    assert table.value.index.name == wording.FORMULATION_CAP
    frame = table.value.reset_index()
    assert list(frame.columns) == [
        wording.FORMULATION_CAP, wording.ROLE, wording.WHY_SOURCE, wording.WHY_COL_PREDICTED,
        wording.WHY_COL_GAIN, wording.WHY_COL_CLOSEST, wording.LARGEST_DIFFERENCES]
    numbers = [r["formulation"] for r in opt.pending_batch]
    assert list(frame[wording.FORMULATION_CAP]) == [f"Formulation {n}" for n in numbers]
    ei = list(frame[wording.WHY_COL_GAIN])
    assert not ei[0].endswith(wording.PENDING_MARK)
    assert all(v.endswith(wording.PENDING_MARK) for v in ei[1:])       # picks 2+
    assert set(frame[wording.ROLE]) <= {wording.ROLE_REFINES, wording.ROLE_BALANCED,
                                        wording.ROLE_EXPLORES, wording.OWN_FORMULATION_KIND}
    assert all(" g" in cell for cell in frame[wording.LARGEST_DIFFERENCES] if cell)
    captions = [c.value for c in fold.caption]
    # Ftue fix C: the picks marked · named by number; the plain recalculated line.
    assert wording.why_picked_after(numbers[1:]) in captions
    assert wording.WHY_RECALCULATED not in captions                      # exact
    assert wording.PLOT_CAPTION_MAP in captions        # demo build: the marks in the legend
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
    # Its role is the Compared with cell's own kind (0.8.0 e2e fixes A2).
    assert own.role == wording.OWN_FORMULATION_KIND


@pytest.mark.parametrize("reasons,state,line", [
    (["floor"], "within", wording.why_guard_line([wording.GUARD_FLOOR])),
    # Integration of the 0.8.0 e2e fixes: the skill reason is said exactly,
    # from W1's skill_gate. This 4-Round project has 6 Overall-score checks.
    (["floor", "skill"], "within",
     wording.why_guard_line(gate_reason=wording.gain_reason_too_few(6))),
    ([], "unreliable", wording.why_guard_line(verdict=True)),
])
def test_the_fold_withholds_by_progresss_rule(monkeypatch, reasons, state, line):
    """Review I4, fix round 1: the fold keeps its model-free columns and
    withholds the expected gain exactly when Progress withholds its
    expected gain, with the same reason. The role is model-free from
    the 0.8.0 e2e fixes (A2, review I3), so it stays."""
    opt = _open_round()
    monkeypatch.setattr(insights, "noise_guard", lambda *a, **k: insights.Guard(reasons))
    real = insights.verdict_state
    monkeypatch.setattr(insights, "verdict_state",
                        lambda *a: state if state != "within" else real(*a))
    at = _app(opt)
    at.session_state[ui_batch.WHY_KEY] = True
    _rerun(at)
    fold = _fold(at)
    [frame] = [d.value for d in fold.dataframe]
    assert wording.ROLE in frame.columns
    assert wording.EXPECTED_GAIN not in frame.columns
    # Ftue fix C: said under the fold's own column name.
    assert wording.why_gain_greyed_said(wording.why_gain_said(line)) in [c.value for c in fold.caption]
    assert not any("(marked ·)" in c.value for c in fold.caption)


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
                    + wording.round_track_no_skill("−0.47", 16, [2, 3, 4, 5]))
    # T47 review I-3: the skill names its own Rounds, not the verdict's.
    assert text.endswith(" Skill −0.47 (n = 16, Rounds 2–5): no better than the average of "
                         "earlier results.")
    alone = ui_insights.track_text(_track(no_skill=-0.47, record=False))
    assert alone == wording.round_track_no_skill("−0.47", 16, [2, 3, 4, 5]).strip()


@pytest.mark.parametrize("t,warns", [(_track(), False), (_track(rough=True), True),
                                     (_track(gap=-1.2), True), (_track(see=True), True),
                                     (_track(no_skill=-0.4), True)])
def test_which_lines_are_warnings(t, warns):
    assert ui_batch._warns(t) == warns


@pytest.mark.skip(reason="review guide prototype: Make a round's status line carries the one link to Review, so E2 and E3 keep their sentences without links; reconcile at the final merge")
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


@pytest.mark.skip(reason="review guide prototype: Make a round's status line carries the one link to Review, so E2 and E3 keep their sentences without links; reconcile at the final merge")
@pytest.mark.parametrize("kind,label", [("close", wording.LINK_NEXT_STEP),
                                        ("untested", wording.LINK_NEXT_STEP),
                                        ("misses", wording.LINK_PREDICTION_CHECK)])
def test_e3_says_next_step_above_generate(monkeypatch, kind, label):
    opt = _burger(rounds=4)
    at = _generate_screen(opt, monkeypatch, insights.NextStep(kind))
    [e3] = _links(at)
    assert (e3.key, e3.label, e3.proto.type) == ("link_next", label, "tertiary")
    caps = [c.value for c in at.tabs[1].caption]
    # Integration: the verdict's Rounds named as Insights names them.
    checks = ui_insights.checks_for(opt, "trust") or insights.check_rounds(opt)
    text, _ = ui_insights.next_step_text(opt, insights.NextStep(kind),
                                         insights.headline(opt, checks).rounds)
    assert wording.first_sentence(text) in caps
    [generate] = [b for b in at.tabs[1].button if b.key == "generate"]
    assert generate.proto.type == "primary"


@pytest.mark.skip(reason="review guide prototype: Make a round's status line carries the one link to Review, so E2 and E3 keep their sentences without links; reconcile at the final merge")
def test_e3_says_no_skill(monkeypatch):
    opt = _burger(rounds=4)
    fact = insights.SkillFact(-0.47, 16, [2, 3, 4], "no_skill")
    at = _generate_screen(opt, monkeypatch, insights.NextStep("no_skill", skill=fact))
    assert [b.label for b in _links(at)] == [wording.LINK_PREDICTION_CHECK]
    said = wording.first_sentence(wording.next_no_skill("−0.47", 16, rounds=[2, 3, 4]))
    assert "(skill −0.47, n = 16, Rounds 2–4)" in said        # T47 review I-3
    assert said in [c.value for c in at.tabs[1].caption]


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
    """Review I10 (T47 minor 1): with nothing worked out, E3 says nothing,
    fits nothing and works out no check — even with a Next step that would
    show a link."""
    opt = _burger(rounds=4)
    calls = []
    monkeypatch.setattr(insights, "next_step", lambda *a, **k: insights.NextStep("close"))
    monkeypatch.setattr(insights, "fit_all", lambda o: calls.append("fit_all") or {})
    real_fit, real_checks = insights.fit_xy, insights.check_rounds
    monkeypatch.setattr(insights, "fit_xy", lambda *a, **k: calls.append("fit") or real_fit(
        *a, **k))
    monkeypatch.setattr(insights, "check_rounds", lambda *a, **k: calls.append(
        "checks") or real_checks(*a, **k))
    at = _app(opt)
    assert _links(at) == [] and calls == []


@pytest.mark.skip(reason="review guide prototype: Make a round's status line carries the one link to Review, so E2 and E3 keep their sentences without links; reconcile at the final merge")
@pytest.mark.parametrize("gain,shown", [(1.5, False), (0.4, True), (None, True)])
def test_e3_weighs_the_expected_improvement_insights_shows(monkeypatch, gain, shown):
    """Fix round 1 (ledger ruling C1): "close to the top" above Generate
    only when the expected gain Insights shows is at or under the
    noise threshold too; above it Next step is the explore row, which is
    another Round, so E3 says nothing."""
    opt = _burger(rounds=4)

    def step(o, c, standing=None, gain=None):
        return (insights.NextStep("explore", gain=gain) if gain is not None and gain > 1.0
                else insights.NextStep("close"))
    monkeypatch.setattr(insights, "next_step", step)
    monkeypatch.setattr(ui_insights, "shown_gain", lambda *a, **k: gain)
    at = _app(opt, wording.TAB_INSIGHTS)
    del at.session_state[INSIGHTS_OPENED_KEY]
    _rerun(at)
    assert bool(_links(at)) is shown


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


def _ferment():
    opt = FoodOptimizer("ferment_why")
    opt.add_process_parameter("Mixing time", 1, 10, unit="min")
    opt.add_process_parameter("Temperature", 20, 40, unit="°C")
    opt.add_objective("Acidity", 1.0, goal="max", min_val=0, max_val=10, unit="pH")
    for k in range(6):
        opt.tell({"Mixing time": 1 + 1.5 * k, "Temperature": 20 + (7 * k) % 20},
                 {"Acidity": 3 + 0.5 * k - 0.02 * k * k}, formulation_no=k + 1, batch_no=1)
    return opt


def test_a_process_only_study_explains_its_trials():
    """No ingredients and no Formulation total: the fold speaks of trials
    and settings, and the pool is the settings' own box. T47 review I-1:
    the cells themselves, as the fold draws them, say "Trial"."""
    opt = _ferment()
    ui_batch.ask_alone(opt, n_suggestions=2)
    w = bo_views.why(opt)
    assert not w.cold and len(w.rows) == 2 and w.pool_mean is not None
    frame, _ = ui_batch.why_frame(opt, w, None)
    assert all(t.startswith("Trial ") for t in frame[wording.FORMULATION_CAP])
    assert all(t.startswith("Trial ") for t in frame[wording.CLOSEST_MADE] if t)
    units = {u for r in w.rows for _, _, u in r.differences}
    assert units <= {"min", "°C"}
    assert "trial" in wording.for_project(opt, wording.why_chart_caption("3,000"))
    at = _app(opt)
    at.session_state[ui_batch.WHY_KEY] = True
    _rerun(at)
    [drawn] = [d.value for d in _fold(at).dataframe]
    assert all(t.startswith("Trial ") for t in drawn[wording.FORMULATION_CAP])
    assert all(t.startswith("Trial ") for t in drawn[wording.CLOSEST_MADE] if t)


@pytest.mark.parametrize("line,tail", [
    (True, "; dashed line, Best so far (predicted Repeat)."),
    (False, "labelled, this Round's suggestions.")])
def test_the_map_caption_is_one_line_naming_every_mark(line, tail):
    """T47 review minor 4: one sentence; the grey rings named; the dashed
    line, when drawn, is Best so far's predicted Repeat (answer-key
    refresh item 6)."""
    text = wording.why_chart_caption("3,000", line=line)
    assert text.endswith(tail) and text.count(". ") == 0
    assert "grey rings, tested formulations" in text


def test_the_cold_line_says_values_in_a_process_only_study():
    """T0 review M-8."""
    assert wording.why_cold(True) == "Chosen to cover the allowed values; no model yet."
    assert wording.gain_cold(2, values=True).endswith("the allowed values: no expected "
                                                      "gain yet.")


@pytest.mark.skip(reason="review guide prototype: Make a round's status line carries the one link to Review, so E2 and E3 keep their sentences without links; reconcile at the final merge")
def test_e3_sits_under_generate(monkeypatch):
    """T47 review minor 8: a line that lands once the checks are in never
    moves the lit button."""
    opt = _burger(rounds=4)
    at = _generate_screen(opt, monkeypatch, insights.NextStep("close"))
    order = []

    def walk(node):
        for child in getattr(node, "children", {}).values():
            key = getattr(child, "key", None) if hasattr(child, "proto") else None
            if key in ("generate", "link_next"):
                order.append(key)
            walk(child)
    walk(at.tabs[1])
    assert order == ["generate", "link_next"]


def test_the_why_fold_does_not_refit_on_a_rerun(monkeypatch):
    """Final review M-1: a recomputed Round's display model comes from the
    memo; reruns with the fold open fit nothing again."""
    opt = _open_round()
    opt.round_generation = {}                 # no record: the Round is recomputed
    opt.save()
    at = _app(opt)
    at.session_state[ui_batch.WHY_KEY] = True
    _rerun(at)
    assert not bo_views.why(opt).exact
    fits = []
    real = insights._fit_gp
    monkeypatch.setattr(insights, "_fit_gp", lambda *a: fits.append(1) or real(*a))
    for _ in range(2):
        at.session_state[ui_batch.WHY_KEY] = True      # AppTest does not keep it open
        _rerun(at)
        captions = [c.value for c in _fold(at).caption]
        assert _fold(at).table                          # the recomputed fold is drawn
        # 01 O9 and last-fix review R-1: one Recomputed line, about what the
        # fold shows (tests/test_106_last_fixes.py).
        # Ftue fix C: one plain line either way, its reason in the tooltip.
        assert captions.count(wording.WHY_RECALCULATED) == 1, captions
    assert fits == []
