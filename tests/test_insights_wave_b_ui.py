"""Insights wave B on screen: the predictions on Make a round, the answer and
compare chart of question 2, and question 4. AppTest, like
tests/test_insights_ui.py."""
import io

import openpyxl
import pytest
from tests.demo_build_reason import REVIEW_FOUR
from streamlit.testing.v1 import AppTest

import insights
import ui_insights
import wording
from food_bo import FoodOptimizer
from tests.test_app_ui import APP_PATH, _submit_button, burger, ferment, scored  # noqa: F401
from tests.test_insights_ui import _open_how, _open_insights, _tab_text, insights_scored  # noqa: F401


def _app():
    return AppTest.from_file(APP_PATH, default_timeout=180)


def _counting(monkeypatch, name):
    calls = []
    real = getattr(insights, name)
    monkeypatch.setattr(insights, name, lambda *a, **k: calls.append(1) or real(*a, **k))
    return calls


def _rerun_on(at, tab):
    """Run again with `tab` open. AppTest does not replay the tab bar: a
    plain at.run() after a tab was chosen draws the first tab again."""
    at.session_state["main_tab"] = tab
    return at.run()


def _batch_table(at):
    return next(d for d in at.dataframe if wording.FORMULATION_CAP in d.value.columns
                and "Pea protein (g)" in d.value.columns)


# ------------------------------------------------------------------ #
#  Task 2: predictions on Make a round
# ------------------------------------------------------------------ #

@pytest.fixture
def warm_round(insights_scored):
    """Nine formulations with all their results (the spread-out start is
    over) and Round 4 open with three formulations."""
    insights_scored.set_pending_batch([
        {"Pea protein": 12.0, "Methylcellulose": 1.0},
        {"Pea protein": 18.0, "Methylcellulose": 2.0},
        {"Pea protein": 22.0, "Methylcellulose": 2.5}])
    return insights_scored


def _expected(project):
    opt = FoodOptimizer(project)
    from ui_helpers import open_rows
    return insights.round_predictions(opt, insights.fit_all(opt), open_rows(opt))


def test_the_round_table_gains_the_predicted_score_column(warm_round):
    at = _app()
    at.run()                                   # opens on Make a round
    assert not at.exception
    table = _batch_table(at)
    cols = list(table.value.columns)
    # Compared with third, on screen only (spec 2026-09-30 §3, review M14).
    assert cols == [wording.FORMULATION_CAP, wording.PREDICTED_SCORE_COLUMN,
                    wording.compared_with_column(warm_round.best_formulation_no()),
                    "Pea protein (g)", "Methylcellulose (g)", "Total (g)"]
    expected = _expected("burger")
    shown = list(table.value[wording.PREDICTED_SCORE_COLUMN])
    numbers = list(table.value[wording.FORMULATION_CAP])
    assert shown == [ui_insights.score_text(expected[no][insights.SCORE]) for no in numbers]
    assert all(" (" in cell and "–" in cell for cell in shown)


def test_the_column_explains_itself_in_its_tooltip(warm_round):
    at = _app()
    at.run()
    # With its basis (BO views, 03 §5.8): the scored formulations it rests on.
    opt = FoodOptimizer("burger")
    assert (f"fitted to the {opt.complete_count()} scored formulations"
            in str(_batch_table(at).proto))
    # Ledger, statistician cold read I2 (fix round B): no "19 in 20" claim; the
    # plain reading gives the 95 % aim and the coverage measured on simulated projects.
    # Ftue fix A's re-run (I9 folds): 90 % of Overall scores (was 89 %).
    # Round 2: "95 %" has a non-breaking space, escaped in the proto.
    assert ("measured about 91 % (measurements) / 90 % (Overall score) on "
            "simulated projects.") in str(_batch_table(at).proto)


def _scores(monkeypatch, below=()):
    """The real predictions, with each Overall score moved just above Best
    so far's measured one, or well below it for the formulations in
    `below`."""
    real = insights.round_predictions

    def fake(opt, fitted, rows):
        got = real(opt, fitted, rows)
        top = float(opt.Y_history[opt.best_index()])
        for no, per in got.items():
            m = top - 5.0 if no in below else top + 0.5
            per[insights.SCORE] = insights.Prediction(m, m - 3.0, m + 3.0)
        return got
    monkeypatch.setattr(insights, "round_predictions", fake)


def test_one_caption_and_the_predicted_measurements(warm_round, monkeypatch):
    _scores(monkeypatch, below={warm_round.pending_batch[0]['formulation']})
    at = _app()
    at.run()
    captions = [c.value for c in at.caption]
    assert captions.count(wording.PREDICTIONS_ROUND_CAPTION) == 1
    assert wording.PREDICTED_SCORE_HELP not in captions        # tooltip, not a caption
    [exp] = [e for e in at.expander if e.label == wording.PREDICTED_MEASUREMENTS]
    # Formulation is the table's index, so no 0-3 column is drawn before it
    # (0.8.0 e2e fixes D3; 01 O4).
    assert exp.table[0].value.index.name == wording.FORMULATION_CAP
    frame = exp.table[0].value.reset_index()
    # Results' order: Firmness carries the larger Share of score.
    assert list(frame.columns) == [wording.FORMULATION_CAP, "Firmness (N)", "Juiciness (/10)"]
    expected = _expected("burger")
    for _, row in frame.iterrows():
        p = expected[int(row[wording.FORMULATION_CAP])]["Firmness"]
        assert row["Firmness (N)"] == ui_insights.value_text(p, 1)


def test_no_round_caption_when_every_suggestion_is_predicted_above_best(warm_round, monkeypatch):
    _scores(monkeypatch)
    at = _app()
    at.run()
    assert not at.exception
    assert wording.PREDICTED_SCORE_COLUMN in _batch_table(at).value.columns
    assert wording.PREDICTIONS_ROUND_CAPTION not in [c.value for c in at.caption]
    assert wording.PREDICTED_MEASUREMENTS in [e.label for e in at.expander]


def test_an_own_formulation_predicted_below_best_needs_no_caption(warm_round, monkeypatch):
    """The caption explains why a SUGGESTION can be predicted lower; a
    formulation of the user's own was not chosen to try anything."""
    own = warm_round.add_to_pending_batch({"Pea protein": 5.0, "Methylcellulose": 0.2},
                                          note="our standard")
    _scores(monkeypatch, below={own})
    at = _app()
    at.run()
    assert not at.exception
    assert wording.PREDICTIONS_ROUND_CAPTION not in [c.value for c in at.caption]
    _scores(monkeypatch, below={own, warm_round.pending_batch[1]['formulation']})
    _rerun_on(at, wording.TAB_BATCH)
    assert [c.value for c in at.caption].count(wording.PREDICTIONS_ROUND_CAPTION) == 1


@pytest.fixture
def warm_ferment(ferment):
    """Five scored trials (enough for a model) and one more trial open, on a
    project made of process settings alone: no ingredients, so every one of
    the Round-screen strings must read "trial", never "formulation"."""
    for no, (temp, hours) in enumerate(
            ((32.0, 5.0), (34.0, 7.0), (36.0, 9.0), (38.0, 11.0), (40.0, 13.0)), start=1):
        ferment.tell({"Incubation temperature": temp, "Incubation time": hours},
                     {"Acidity": 3.0 + 0.1 * temp}, formulation_no=no, batch_no=1)
    ferment.set_pending_batch([{"Incubation temperature": 37.0, "Incubation time": 8.0}])
    return ferment


def _ferment_table(at):
    return next(d for d in at.dataframe
                if "Incubation temperature (°C)" in d.value.columns)


def test_the_round_screen_reads_trial_on_a_process_only_project(warm_ferment):
    at = _app()
    at.run()
    assert not at.exception
    table = _ferment_table(at)
    assert wording.PREDICTED_SCORE_COLUMN in table.value.columns
    proto = str(table.proto)
    trial_help = wording.for_project(
        warm_ferment, wording.predicted_score_help(warm_ferment.complete_count()))
    assert "trial" in trial_help and "formulation" not in trial_help
    assert "fitted to the 5 scored trials" in proto and "formulation" not in trial_help
    captions = [c.value for c in at.caption]
    trial_caption = wording.for_project(warm_ferment, wording.predictions_round_caption(values=True))
    # Concision round: the short caption names neither; its help says trials.
    assert trial_caption == "Exploring suggestions can predict below Best so far."
    assert captions.count(trial_caption) == 1
    [line] = [c for c in at.caption if c.value == trial_caption]
    assert line.proto.help == ("Suggestions favour trials the model is unsure about as well "
                               "as ones it predicts high, so one can be predicted below Best "
                               "so far.")
    [exp] = [e for e in at.expander if e.label == wording.PREDICTED_MEASUREMENTS]
    frame = exp.table[0].value.reset_index()
    assert list(frame.columns) == [wording.TRIAL_CAP, "Acidity (pH)"]


def test_a_measurement_with_too_few_values_shows_a_dash(warm_round, monkeypatch):
    real = insights.fit_all

    def fewer(opt):
        fitted = real(opt)
        fitted["Juiciness"] = insights.Fitted("Juiciness", None, None, 3, 0.0, 10.0)
        return fitted
    monkeypatch.setattr(insights, "fit_all", fewer)
    at = _app()
    at.run()
    assert not at.exception
    [exp] = [e for e in at.expander if e.label == wording.PREDICTED_MEASUREMENTS]
    frame = exp.table[0].value
    assert set(frame["Juiciness (/10)"]) == {"—"}
    assert "—" not in set(frame["Firmness (N)"])


def test_the_cold_start_round_has_no_predictions(scored, monkeypatch):
    scored.set_pending_batch([{"Pea protein": 12.0, "Methylcellulose": 1.0}])
    calls = _counting(monkeypatch, "fit_all")
    at = _app()
    at.run()
    assert not at.exception
    assert wording.PREDICTED_SCORE_COLUMN not in _batch_table(at).value.columns
    assert wording.PREDICTIONS_ROUND_CAPTION not in [c.value for c in at.caption]
    assert wording.PREDICTED_MEASUREMENTS not in [e.label for e in at.expander]
    assert calls == []


def test_editing_the_round_hides_the_column(warm_round):
    at = _app()
    at.run()
    at.button(key=f"edit_round_burger_{warm_round.pending_batch_no}_start").click().run()
    assert not at.exception
    assert any(i.value == wording.EDIT_FORMULATIONS_ACTIVE for i in at.info)
    [grid] = [d for d in at.dataframe if "Pea protein (g)" in d.value.columns]   # the editor
    assert wording.PREDICTED_SCORE_COLUMN not in grid.value.columns
    assert wording.PREDICTIONS_ROUND_CAPTION not in [c.value for c in at.caption]
    assert wording.PREDICTED_MEASUREMENTS not in [e.label for e in at.expander]


def test_recorded_and_not_scored_rows_show_nothing(warm_round):
    a, b, c = (r['formulation'] for r in warm_round.pending_batch)
    warm_round.tell(warm_round.pending_batch[0]['recipe'], {"Juiciness": 6.0, "Firmness": 5.0},
                    formulation_no=a, batch_no=warm_round.pending_batch_no)
    warm_round.record_skipped(b, warm_round.pending_batch_no, warm_round.pending_batch[1]['recipe'])
    at = _app()
    at.run()
    assert not at.exception
    table = _batch_table(at).value
    cells = dict(zip(table[wording.FORMULATION_CAP], table[wording.PREDICTED_SCORE_COLUMN]))
    assert cells[a] == "" and cells[b] == "" and cells[c] != ""
    [exp] = [e for e in at.expander if e.label == wording.PREDICTED_MEASUREMENTS]
    assert list(exp.table[0].value.reset_index()[wording.FORMULATION_CAP]) == [c]


def test_the_round_tab_closed_fits_nothing(warm_round, monkeypatch):
    calls = _counting(monkeypatch, "fit_all")
    at = _app()
    at.session_state["_loaded_project"] = "burger"   # no landing on Make a round
    at.session_state["main_tab"] = wording.TAB_RESULTS
    at.run()
    assert not at.exception
    assert calls == []
    assert ui_insights.FITTED_KEY not in at.session_state


@pytest.mark.skip(reason=REVIEW_FOUR)
def test_the_models_are_fitted_once_for_both_screens(warm_round, monkeypatch):
    calls = _counting(monkeypatch, "fit_all")
    at = _app()
    at.run()                                          # Make a round
    assert calls == [1]
    _rerun_on(at, wording.TAB_INSIGHTS)
    assert not at.exception
    assert calls == [1] and ui_insights.CACHE_KEY in at.session_state
    assert wording.Q_TRUST in " ".join(_tab_text(at)[0])
    _rerun_on(at, wording.TAB_BATCH)
    assert calls == [1]
    assert wording.PREDICTED_SCORE_COLUMN in _batch_table(at).value.columns


def test_the_round_screen_waits_with_dots_not_a_spinner(warm_round, monkeypatch):
    """Spec 2026-09-30 §1 (V28), review I1: the column's "…" is the one
    indicator while its models are fitted; no spinner of its own."""
    said = []
    real = ui_insights.fitted_models
    monkeypatch.setattr(ui_insights, "fitted_models",
                        lambda opt, working=wording.INSIGHTS_WORKING: said.append(working) or real(opt, working))
    at = _app()
    at.run()
    assert said == [None]


def test_a_failed_fit_leaves_the_table_as_it_was(warm_round, monkeypatch):
    calls = []
    def boom(opt):
        calls.append(1)
        raise RuntimeError("no fit")
    monkeypatch.setattr(insights, "fit_all", boom)
    at = _app()
    at.run()
    assert not at.exception
    assert wording.PREDICTED_SCORE_COLUMN not in _batch_table(at).value.columns
    _rerun_on(at, wording.TAB_BATCH)
    assert calls == [1]


def test_the_round_screen_is_drawn_before_the_fit(warm_round, monkeypatch):
    """The table, the Save results button and the record and upload steps
    are all on screen before the models are fitted; the table is then drawn
    again in its place with the column."""
    import ui_batch
    order = []
    for name in ("_record_results", "_upload"):
        real = getattr(ui_batch, name)
        monkeypatch.setattr(ui_batch, name,
                            lambda *a, _real=real, _name=name, **k: order.append(_name) or _real(*a, **k))
    real_fit = insights.fit_all
    monkeypatch.setattr(insights, "fit_all", lambda opt: order.append("fit") or real_fit(opt))
    at = _app()
    at.run()
    assert not at.exception
    assert order == ["_record_results", "_upload", "fit"]
    assert wording.SAVE_RESULTS in [b.label for b in at.button]
    [table] = [d for d in at.dataframe if "Pea protein (g)" in d.value.columns
               and wording.FORMULATION_CAP in d.value.columns]      # drawn once, in place
    assert wording.PREDICTED_SCORE_COLUMN in table.value.columns
    [exp] = [e for e in at.expander if e.label == wording.PREDICTED_MEASUREMENTS]


def test_no_open_rows_no_fit(warm_round, monkeypatch):
    rows = warm_round.pending_batch
    for row in rows[:2]:
        warm_round.tell(row['recipe'], {"Juiciness": 6.0, "Firmness": 5.0},
                        formulation_no=row['formulation'], batch_no=warm_round.pending_batch_no)
    warm_round.record_skipped(rows[2]['formulation'], warm_round.pending_batch_no, rows[2]['recipe'])
    calls = _counting(monkeypatch, "fit_all")
    at = _app()
    at.run()
    assert not at.exception
    assert calls == []
    assert ui_insights.FITTED_KEY not in at.session_state
    assert wording.PREDICTED_SCORE_COLUMN not in _batch_table(at).value.columns


def _cells(book):
    return [[list(r) for r in ws.iter_rows(values_only=True)] for ws in book.worksheets]


def test_the_workbook_is_unchanged(warm_round):
    opt = FoodOptimizer("burger")
    before = _cells(openpyxl.load_workbook(io.BytesIO(
        opt.workbook_bytes(opt.pending_batch, opt.open_round_size()))))
    at = _app()
    at.run()
    assert wording.PREDICTED_SCORE_COLUMN in _batch_table(at).value.columns
    opt = at.session_state["optimizer"]
    after = _cells(openpyxl.load_workbook(io.BytesIO(
        opt.workbook_bytes(opt.pending_batch, opt.open_round_size()))))
    assert after == before
    assert wording.PREDICTED_SCORE_COLUMN not in opt.batch_frame(opt.pending_batch).columns


def test_a_project_switch_drops_the_shared_models(warm_round):
    at = _open_insights(_app())
    assert ui_insights.FITTED_KEY in at.session_state
    FoodOptimizer("second").add_ingredient("Flour", 0, 100)
    at.run()
    at.selectbox(key="project_select").set_value("second")
    at.run()
    _submit_button(at.sidebar, "Open").click()
    at.run()
    assert not at.exception
    assert ui_insights.FITTED_KEY not in at.session_state
    assert ui_insights.CACHE_KEY not in at.session_state


@pytest.mark.parametrize("low, high, text", [(70.2, 85.4, "70–85"), (-1.25, 0.4, "-1.25 to 0.4"),
                                             (-3.0, -1.0, "-3 to -1")])
def test_range_text_uses_to_beside_a_minus_sign(low, high, text):
    fmt = (lambda v: f"{v:.0f}") if text in ("70–85", "-3 to -1") else None
    assert ui_insights.range_text(low, high, fmt) == text


def test_score_and_value_text():
    p = insights.Prediction(78.4, 70.2, 85.49)
    assert ui_insights.score_text(p) == "78 (70–85)"
    assert ui_insights.value_text(insights.Prediction(5.81, 5.1, 6.54)) == "5.81 (5.1–6.54)"


# ------------------------------------------------------------------ #
#  Task 3: which ingredients matter most
# ------------------------------------------------------------------ #

@pytest.mark.skip(reason="review profiler prototype: pins Try one's path and Compare all, which the profiler replaces; reconcile at the final merge")
def test_a_failed_ranking_leaves_the_path(insights_scored, monkeypatch):
    """tested_near_changes failing drops the fold's first line; the path
    and its captions still draw (ui_insights._change's fallback)."""
    def boom(*a, **k):
        raise RuntimeError("no ranking")
    monkeypatch.setattr(insights, "tested_near_changes", boom)
    at = _open_insights(_app())
    assert not at.exception
    md, caps = _tab_text(at)
    assert not [m for m in md if m.startswith(("Near what you have tested",
                                                "Nothing has been tested near"))]
    assert [c for c in caps if c.startswith("More Pea protein")]


# ------------------------------------------------------------------ #
#  Task 5: which formulation is predicted to score highest
# ------------------------------------------------------------------ #

def _best_frames(at):
    return [d.value for d in at.tabs[3].dataframe
            if wording.FORMULATION_CAP in d.value.columns
            and wording.BEST_ROW in list(d.value[wording.FORMULATION_CAP])]


def _best(monkeypatch, close):
    def fake(opt, fitted, checks=()):
        return insights.PredictedBest(
            dict(opt.recipe_history[0]), insights.Prediction(88.2, 80.4, 95.1),
            close, close if close is not None else opt.best_index())
    monkeypatch.setattr(insights, "predicted_best", fake)


@pytest.mark.skip(reason=REVIEW_FOUR)
def test_section_4_shows_the_reason_a_round_could_not_be_made(insights_scored, monkeypatch):
    monkeypatch.setattr(insights, "predicted_best",
                        lambda opt, fitted, checks=(): wording.EVERYTHING_IS_FIXED)
    at = _open_insights(_app())
    md, caps = _tab_text(at)
    assert wording.EVERYTHING_IS_FIXED in caps
    assert not [m for m in md if m.startswith("Predicted to score highest")]


@pytest.mark.skip(reason=REVIEW_FOUR)
def test_section_4_failure_is_one_line_and_not_retried(insights_scored, monkeypatch):
    calls = []
    def boom(opt, fitted, checks=()):
        calls.append(1)
        raise RuntimeError("no best")
    monkeypatch.setattr(insights, "predicted_best", boom)
    at = _open_insights(_app())
    assert not at.exception
    md, caps = _tab_text(at)
    assert caps.count(wording.BEST_FAILED) == 1
    assert any("Best so far:" in m for m in md)
    assert wording.Q_TRUST in " ".join(md)
    _rerun_on(at, wording.TAB_INSIGHTS)
    assert caps.count(wording.BEST_FAILED) == 1 and calls == [1]


@pytest.mark.skip(reason="review cards prototype: pins the old Review layout (status lines, Explored fold, header); reconcile at the final merge")
def test_section_4_gate(scored):          # 2 formulations with all their results
    at = _open_insights(_app())
    md, caps = _tab_text(at)
    assert f"##### {wording.Q_NEXT}" not in md               # under 5: the gate line instead
    # Since ftue W1 the gate is the Predictions status line's (R-B1 row 3).
    assert md.count(wording.status_line(wording.STATUS_PREDICTIONS, wording.short_gate(2), wording.PREDICTIONS_WORDS["not_yet"])) == 1
    assert wording.insights_gate(2) not in caps


@pytest.mark.skip(reason=REVIEW_FOUR)
def test_a_limit_edit_works_out_the_best_again(insights_scored, monkeypatch):
    calls = _counting(monkeypatch, "predicted_best")
    at = _open_insights(_app())
    _rerun_on(at, wording.TAB_INSIGHTS)
    assert calls == [1]
    at.session_state["optimizer"].add_quantity_constraint(
        ["Pea protein", "Methylcellulose"], max_val=15.0)
    _rerun_on(at, wording.TAB_INSIGHTS)
    assert not at.exception
    assert calls == [1, 1]
    assert wording.Q_STAND in " ".join(_tab_text(at)[0])


def test_a_project_switch_drops_the_best(warm_round):
    at = _open_insights(_app())
    assert ui_insights.BEST_KEY in at.session_state
    FoodOptimizer("second").add_ingredient("Flour", 0, 100)
    at.run()
    at.selectbox(key="project_select").set_value("second")
    at.run()
    _submit_button(at.sidebar, "Open").click()
    at.run()
    assert not at.exception
    assert ui_insights.BEST_KEY not in at.session_state


@pytest.mark.skip(reason=REVIEW_FOUR)
def test_the_models_are_shared_with_question_4(warm_round, monkeypatch):
    calls = _counting(monkeypatch, "fit_all")
    at = _app()
    at.run()                                          # Make a round
    _rerun_on(at, wording.TAB_INSIGHTS)
    assert wording.Q_STAND in " ".join(_tab_text(at)[0])
    assert calls == [1]


def test_how_models_names_both_places():
    """03 §5.7: the models, the kernel and the training rows named."""
    assert wording.HOW_MODELS == (
        "Models: one Gaussian process per measurement and one for the Overall score, each "
        "fitted to the formulations that have that value (default kernel: squared exponential, "
        "one correlation length per varying row; noise fitted). Suggestions use the "
        "Overall-score model only. It is fitted to the scores themselves, not built from the "
        "measurement models, so the two can disagree slightly.")


def test_every_wave_b_string_passes_the_vocabulary_guard():
    import re
    from tests.test_food_bo import _BANNED
    # "the model" is allowed from the BO-views wave on (review I5: the
    # caption says what the suggestions weigh).
    also = re.compile(r"\b(optimum|optimal|maximi[sz]e|confidence|probability)\b", re.I)
    for s in (wording.PREDICTIONS_WORKING, wording.PREDICTED_SCORE_COLUMN,
              wording.PREDICTED_SCORE_HELP, wording.PREDICTIONS_ROUND_CAPTION,
              wording.PREDICTED_MEASUREMENTS,
              wording.compare_all(True), wording.compare_all(False),
              wording.BEST_ROW, wording.BEST_FAILED, wording.HOW_MODELS,
              wording.held_line("Best so far", ["Wheat gluten", "Sunflower oil"]),
              wording.held_all("Formulation 9"), wording.held_all("Best so far", True),
              wording.compare_all(False, False), wording.predictions_round_caption(True)):
        assert not any(p.search(s) for p in _BANNED), s
        assert not also.search(s), s


def test_exact_wave_b_strings():
    assert wording.PREDICTED_SCORE_COLUMN == "Predicted Overall score (95 % range)"
    # Ledger, statistician cold read I2 (fix round B): no "19 in 20" claim; the
    # plain reading gives the 95 % aim and the coverage measured on simulated projects.
    # Ftue fix A's re-run (I9 folds): 90 % of Overall scores (was 89 %).
    assert wording.PREDICTED_SCORE_HELP == (
        "Model mean and 95 % prediction interval for one measured Overall score, from a "
        "Gaussian process fitted to the scored formulations. Aims for 95 %; measured about "
        "91 % (measurements) / 90 % (Overall score) on simulated projects.")
    assert wording.PREDICTIONS_ROUND_CAPTION == "Exploring suggestions can predict below Best so far."
    assert wording.predictions_round_help() == (
        "Suggestions favour formulations the model is unsure about as well as ones it "
        "predicts high, so one can be predicted below Best so far.")
    assert wording.spread_from(False, 9) == "Formulation 9"


# ------------------------------------------------------------------ #
#  One vocabulary across Insights: trials, process settings, values
# ------------------------------------------------------------------ #

def _everything_said(tab):
    said = ([m.value for m in tab.markdown] + [c.value for c in tab.caption]
            + [e.label for e in tab.expander] + [s.label for s in tab.selectbox])
    for box in tab.selectbox:
        said += [str(o) for o in box.options]
    for d in tab.dataframe:
        said += [str(c) for c in d.value.columns]
        said += [str(v) for v in d.value.astype(str).to_numpy().ravel()]
    return said


@pytest.mark.skip(reason="review cards prototype: pins the old Review layout (status lines, Explored fold, header); reconcile at the final merge")
def test_insights_reads_trial_and_setting_on_a_process_only_project(warm_ferment):
    at = _open_insights(_app())
    assert not at.exception
    tab = at.tabs[3]
    said = _everything_said(tab)
    assert not [t for t in said if "ormulation" in t], [t for t in said if "ormulation" in t]
    assert not [t for t in said if "ngredient" in t], [t for t in said if "ngredient" in t]
    md, caps = _tab_text(at)
    # The three answers and the folds (spec 2026-09-30 §2, review I14).
    assert f"##### {wording.Q_STAND}" in md
    assert "Try one process setting" in [e.label for e in tab.expander]
    assert tab.selectbox(key=ui_insights.CHOICE_KEY).label == "Process setting"
    assert [c for c in caps if c.startswith("5 scored trials")]


def _best_frames_for(tab, first):
    return [d.value for d in tab.dataframe
            if first in d.value.columns and wording.BEST_ROW in list(d.value[first])]


@pytest.fixture
def mixed_scored(burger):
    """The burger with a process setting as well, nine formulations scored."""
    burger.add_process_parameter("Mixing time", 2, 10, unit="min")
    no = 0
    for r in (1, 2, 3):
        for pea, mc, mix in ((8.0, 0.5, 3.0), (14.0, 1.5, 6.0), (20.0, 2.5, 9.0)):
            no += 1
            pea += r * 0.7
            burger.tell({"Pea protein": pea, "Methylcellulose": mc, "Mixing time": mix},
                        {"Juiciness": 3 + 0.2 * pea, "Firmness": 1 + 0.25 * pea + mc},
                        formulation_no=no, batch_no=r)
    return burger


def test_process_setting_wording():
    assert wording.compare_all(True, False) == "Compare all process settings"
    assert wording.path_head_process("Incubation time") == "Incubation time changes"
    assert wording.PREDICTIONS_ROUND_CAPTION == wording.predictions_round_caption()


# ------------------------------------------------------------------ #
#  Final fix wave (waves B + C): carried from wave B's final review
# ------------------------------------------------------------------ #

def test_a_warm_round_table_is_drawn_once_with_its_column(warm_round, monkeypatch):
    """With the models already fitted for this data, the table is drawn
    with the Predicted Overall score column on the first pass: no draw
    without it first, so the column does not blink on every rerun."""
    import ui_batch
    at = _app()
    at.run()                                   # cold: fits, then fills the column
    assert wording.PREDICTED_SCORE_COLUMN in _batch_table(at).value.columns
    drawn = []
    real = ui_batch._draw_table
    monkeypatch.setattr(ui_batch, "_draw_table",
                        lambda opt, slot, frame: drawn.append(list(frame.columns))
                        or real(opt, slot, frame))
    calls = _counting(monkeypatch, "fit_all")
    _rerun_on(at, wording.TAB_BATCH)
    assert not at.exception
    assert calls == []
    assert len(drawn) == 1 and wording.PREDICTED_SCORE_COLUMN in drawn[0]
    assert wording.PREDICTED_SCORE_COLUMN in _batch_table(at).value.columns
    assert wording.PREDICTED_MEASUREMENTS in [e.label for e in at.expander]


def test_a_cold_round_table_is_drawn_first_and_filled_after(warm_round, monkeypatch):
    import ui_batch
    drawn = []
    real = ui_batch._draw_table
    monkeypatch.setattr(ui_batch, "_draw_table",
                        lambda opt, slot, frame: drawn.append(
                            list(frame[wording.PREDICTED_SCORE_COLUMN]))
                        or real(opt, slot, frame))
    at = _app()
    at.run()
    assert not at.exception
    # Review I1: drawn first with "…" in the column (its width kept), then
    # again with the numbers.
    assert [cells[0] == wording.PREDICTION_PENDING for cells in drawn] == [True, False]


def test_a_hidden_round_tab_predicts_nothing_even_when_warm(warm_round, monkeypatch):
    at = _app()
    at.run()
    calls = _counting(monkeypatch, "round_predictions")
    _rerun_on(at, wording.TAB_INSIGHTS)
    assert not at.exception
    assert calls == []


@pytest.mark.skip(reason="review profiler prototype: pins Try one's path and Compare all, which the profiler replaces; reconcile at the final merge")
def test_question_2_keeps_ingredient_when_nothing_varies(insights_scored, monkeypatch):
    monkeypatch.setattr(insights, "path_choices", lambda opt: [])
    at = _open_insights(_app())
    assert not at.exception
    md, caps = _tab_text(at)
    assert wording.try_one_fold(False) in [e.label for e in at.tabs[3].expander]
    assert wording.CHANGE_NONE in caps


@pytest.mark.skip(reason="review profiler prototype: pins Try one's path and Compare all, which the profiler replaces; reconcile at the final merge")
def test_a_process_only_project_says_values_and_settings(warm_ferment, monkeypatch):
    monkeypatch.setattr(insights, "path_choices", lambda opt: [])
    at = _open_how(_open_insights(_app()))
    assert not at.exception
    md, caps = _tab_text(at)
    assert "Try one process setting" in [e.label for e in at.tabs[3].expander]
    assert "Nothing to change yet: every process setting is fixed." in caps
    [how] = [e for e in at.tabs[3].expander if e.label == wording.FOLD_METHOD]
    checks = next(m.value for m in how.markdown if m.value.startswith("Round-ahead check:"))
    assert "at the values it was run at" in checks and "amounts" not in checks
    assert "ngredient" not in " ".join(md + caps)


def test_the_process_only_lines_pass_the_word_guards():
    import re
    from tests.test_food_bo import _BANNED
    also = re.compile(r"\b(optimum|optimal|maximi[sz]e|confidence|probability|the model)\b", re.I)
    for s in (wording.how_checks(values=True), wording.change_none(settings_only=True)):
        assert not any(p.search(s) for p in _BANNED), s
        assert not also.search(s), s
    assert wording.how_checks() == wording.HOW_CHECKS
    assert wording.change_none() == wording.CHANGE_NONE
