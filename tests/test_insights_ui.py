import pytest
import insights
import ui_insights
import wording
from tests.test_insights import _burger


@pytest.fixture(autouse=True)
def _isolate_cwd(tmp_path, monkeypatch):
    # ui_insights builders call into FoodOptimizer via _burger, which saves a
    # "<project_name>.pkl" into the current working directory on every
    # tell()/set_amount_unit() (see tests/test_insights.py's own fixture of
    # the same name). Isolate every test here into its own tmp_path so this
    # file does not litter the worktree root with test .pkl files.
    monkeypatch.chdir(tmp_path)


def _spec(chart):
    return chart.to_dict()


def test_track_chart_has_one_panel_per_output_and_no_caps():
    opt = _burger(rounds=4, per_round=4, noise=0.1)
    chart = ui_insights.track_chart(opt, insights.check_rounds(opt))
    spec = _spec(chart)
    panels = {row["panel"] for row in spec["datasets"][next(iter(spec["datasets"]))]}
    assert panels == {ui_insights.panel_label(opt, n) for n in ("Firmness", "Juiciness", insights.SCORE)}
    marks = str(spec)
    assert "errorbar" not in marks and "tick" not in marks     # no end caps
    assert spec["facet"] and spec.get("columns") == 3


def test_path_chart_panels_score_first_and_grey_where_untested():
    opt = _burger(rounds=3)
    fitted = insights.fit_all(opt)
    p = insights.build_path(opt, "Oil", insights.start_recipe(opt, opt.best_index()))
    chart = ui_insights.path_chart(opt, p, insights.path_predictions(opt, fitted, p))
    assert ui_insights.path_panels(opt)[0] == insights.SCORE
    spec = str(_spec(chart))
    assert "rect" in spec                                     # untested shading layer


def test_track_dots_share_the_theme_colour():
    """Rings for every measured value, a filled dot of the same (theme) colour
    only where it landed inside: never the text colour."""
    opt = _burger(rounds=4, per_round=4, noise=0.1)
    spec = _spec(ui_insights.track_chart(opt, insights.check_rounds(opt)))
    assert "currentColor" not in str(spec)
    points = [layer for layer in spec["spec"]["layer"]
              if (layer.get("mark") or {}).get("type") == "point"]
    assert [bool(p["mark"].get("filled")) for p in points] == [False, True]
    assert points[1]["transform"] == [{"filter": {"field": wording.TRACK_INSIDE}}] \
        or wording.TRACK_INSIDE in str(points[1]["transform"])


def test_path_chart_x_axis_is_not_zero_based():
    opt = _burger(rounds=3)
    fitted = insights.fit_all(opt)
    p = insights.build_path(opt, "Oil", insights.start_recipe(opt, opt.best_index()))
    spec = _spec(ui_insights.path_chart(opt, p, insights.path_predictions(opt, fitted, p)))
    assert "'zero': False" in str(spec)


def test_path_chart_none_without_points():
    opt = _burger(rounds=3)
    opt.add_quantity_constraint(["Pea", "Oil"], max_val=8.0)
    p = insights.build_path(opt, "Oil", opt.fill_formulas({"Pea": 7.0, "Oil": 3.0}))
    assert ui_insights.path_chart(opt, p, {}) is None


def test_every_insights_string_passes_the_vocabulary_guard():
    """Scans only the Insights strings added in Task 3 (path naming) and
    Task 5 (this task) — not every wording.py constant. The repo-wide guard
    `test_no_old_vocabulary_reaches_the_user` in tests/test_food_bo.py already
    covers wording.py in full, with its own owner-approved exceptions."""
    from tests.test_food_bo import _BANNED

    def _check(s):
        assert not any(p.search(s) for p in _BANNED), s

    # Task 3
    _check(wording.path_past_lowest("Pea"))
    _check(wording.path_past_highest("Pea"))
    _check(wording.path_below_zero("Total fat"))
    _check(wording.PATH_NO_ROOM_DOWN)
    _check(wording.PATH_NO_ROOM_UP)

    # Task 5
    _check(wording.TAB_INSIGHTS)
    _check(wording.INSIGHTS_NONE)
    _check(wording.INSIGHTS_WORKING)
    _check(wording.insights_header(1, None))
    _check(wording.insights_header(17, "5"))
    _check(wording.insights_gate(3))
    _check(wording.Q_PREDICTIONS)
    _check(wording.Q_CHANGE)
    _check(wording.Q_IMPROVING)
    _check(wording.PREDICTIONS_NONE)
    _check(wording.predictions_early(1))
    _check(wording.predictions_early(3))
    _check(wording.rounds_span(["3"]))
    _check(wording.rounds_span(["3", "4", "5"]))
    _check(wording.predictions_count(3, ["3", "4", "5"], 26, 27))
    _check(wording.predictions_count(1, ["5"], 1, 1))
    for v in wording.VERDICT.values():
        _check(v)
    _check(wording.SCORE_BELOW)
    _check(wording.PREDICTIONS_CAPTION)
    _check(wording.SHOW_PREDICTIONS)
    _check(wording.INGREDIENT_OR_SETTING)
    _check(wording.START_FROM)
    _check(wording.start_best(14))
    _check(wording.start_formulation(3))
    _check(wording.path_fill("Coconut oil", "Remaining water", 14))
    _check(wording.path_proportional("Coconut oil", "100 g"))
    _check(wording.path_free("Coconut oil", 14))
    _check(wording.path_process("Mixing speed", 14))
    _check(wording.path_stops("9.1 g", "Fat: at most 16 g"))
    _check(wording.path_fixed("Salt 2 g"))
    _check(wording.path_follows("Total fat"))
    _check(wording.path_outside(3, "9 g"))
    _check(wording.path_start_breaks(3, "Fat: at most 16 g"))
    _check(wording.path_not_checked("Fat"))
    _check(wording.PATH_GREY)
    _check(wording.panel_missing("Firmness"))
    _check(wording.panel_failed("Firmness"))
    _check(wording.best_line("81.0", 14, "4"))
    _check(wording.best_line("81.0", 14, None))
    _check(wording.BEST_LATEST)
    _check(wording.best_since(1))
    _check(wording.best_since(3))
    _check(wording.repeats_line("2.5"))
    _check(wording.SHOW_PROGRESS)
    _check(wording.HOW_WORKED_OUT)
    _check(wording.HOW_MODELS)
    _check(wording.HOW_RANGE)
    _check(wording.HOW_RANGE_FIXED_TINY)
    _check(wording.HOW_CHECKS)
    _check(wording.HOW_NO_P)
    for c in wording.DETAILS_COLUMNS:
        _check(c)
    _check(wording.TRACK_PREDICTED)
    _check(wording.TRACK_MEASURED)
    _check(wording.TRACK_LOW)
    _check(wording.TRACK_HIGH)
    _check(wording.TRACK_INSIDE)


# ------------------------------------------------------------------ #
#  The tab in the app (Task 6)
# ------------------------------------------------------------------ #

from streamlit.testing.v1 import AppTest                     # noqa: E402

from food_bo import FoodOptimizer                            # noqa: E402
from tests.test_app_ui import (                               # noqa: E402
    APP_PATH, _submit_button, burger, scored)  # noqa: F401


@pytest.fixture
def insights_scored(burger):
    """Nine scored formulations over three rounds: enough for every section."""
    no = 0
    for r in (1, 2, 3):
        for pea, mc in ((8.0, 0.5), (14.0, 1.5), (20.0, 2.5)):
            no += 1
            pea += r * 0.7
            burger.tell({"Pea protein": pea, "Methylcellulose": mc},
                        {"Juiciness": 3 + 0.2 * pea, "Firmness": 1 + 0.25 * pea + mc},
                        formulation_no=no, batch_no=r)
    return burger


@pytest.fixture
def insights_fixed_tiny(insights_scored):
    """The expert noise setting that assumes no variation between Repeats;
    set_bo_config saves, so the app opens the project with it."""
    insights_scored.set_bo_config({"kernel": "matern52", "lengthscale_prior": "default",
                                   "noise": "fixed_tiny", "acquisition": "qlognei"})
    return insights_scored


def _open_insights(at):
    """Open the project first, then the tab: the run that opens a project
    moves to its landing tab (never Insights), over any tab set before it."""
    at.run()
    at.session_state["main_tab"] = wording.TAB_INSIGHTS
    at.run()
    return at


def _app():
    return AppTest.from_file(APP_PATH, default_timeout=180)


def test_insights_is_the_last_tab(insights_scored):
    at = _app()
    at.run()
    assert [t.label for t in at.tabs][-1] == wording.TAB_INSIGHTS


def test_insights_sections_and_expanders(insights_scored):
    at = _open_insights(_app())
    assert not at.exception
    tab = at.tabs[3]
    text = " ".join(m.value for m in tab.markdown)
    for q in (wording.Q_PREDICTIONS, wording.Q_CHANGE, wording.Q_IMPROVING):
        assert q in text
    labels = [e.label for e in tab.expander]
    for lab in (wording.SHOW_PREDICTIONS, wording.SHOW_PROGRESS, wording.HOW_WORKED_OUT):
        assert lab in labels, labels
    assert not [b for b in tab.button if b.proto.type == "primary"]
    # The two controls of "What happens if you change one ingredient?"
    assert tab.selectbox(key="insights_choice").value == "Pea protein"
    best = insights_scored.best_index()
    assert tab.selectbox(key="insights_start").value == int(insights_scored.formulation_ids[best])
    assert [c.value for c in tab.caption if c.value.startswith("More Pea protein")]


def test_progress_chart_moved_from_results(insights_scored):
    at = _app()
    at.session_state["main_tab"] = wording.TAB_RESULTS
    at.run()
    labels = [e.label for e in at.expander]
    assert "Progress chart" not in labels and wording.SHOW_PROGRESS not in labels
    assert not any(c.value == wording.PROGRESS_CHART_CAPTION for c in at.caption)
    at = _open_insights(_app())
    assert any(c.value == wording.PROGRESS_CHART_CAPTION for c in at.tabs[3].caption)


def test_insights_not_computed_while_another_tab_is_open(insights_scored, monkeypatch):
    calls = []
    monkeypatch.setattr(insights, "fit_all", lambda opt: calls.append(1) or {})
    at = _app()
    at.session_state["main_tab"] = wording.TAB_RESULTS
    at.run()
    assert not at.exception
    assert calls == []
    assert ui_insights.CACHE_KEY not in at.session_state


def test_insights_computed_once_then_cached(insights_scored, monkeypatch):
    calls = []
    real = insights.fit_all
    monkeypatch.setattr(insights, "fit_all", lambda opt: calls.append(1) or real(opt))
    at = _open_insights(_app())
    assert calls == [1] and ui_insights.CACHE_KEY in at.session_state
    at.run()
    assert calls == [1]


def test_no_scored_formulation_one_line(burger):
    at = _open_insights(_app())
    tab = at.tabs[3]
    assert [c.value for c in tab.caption] == [wording.INSIGHTS_NONE]
    assert not tab.markdown and not tab.expander


def test_gate_before_five_scored(scored):     # 2 scored rows
    at = _open_insights(_app())
    assert not at.exception
    tab = at.tabs[3]
    captions = [c.value for c in tab.caption]
    assert captions.count(wording.insights_gate(2)) == 2       # sections 1 and 2
    assert any(m.value.startswith("Best so far:") for m in tab.markdown)   # section 3 shows


def test_how_section_default_wording(insights_scored):
    # A fourth round, so each measurement has 5 or more values checked
    # (Rounds 3 and 4) and the details table has its rows.
    for no, (pea, mc) in enumerate(((10.0, 1.0), (16.0, 2.0), (22.0, 3.0)), start=10):
        insights_scored.tell({"Pea protein": pea, "Methylcellulose": mc},
                             {"Juiciness": 3 + 0.2 * pea, "Firmness": 1 + 0.25 * pea + mc},
                             formulation_no=no, batch_no=4)
    at = _open_insights(_app())
    body = " ".join(m.value for m in at.tabs[3].markdown)
    assert wording.HOW_RANGE in body and wording.HOW_RANGE_FIXED_TINY not in body
    frame = at.tabs[3].dataframe[0].value
    assert list(frame.columns) == list(wording.DETAILS_COLUMNS)
    assert frame.iloc[-1][wording.DETAILS_COLUMNS[0]] == wording.OVERALL_SCORE_COLUMN


def test_how_section_fixed_tiny_wording(insights_fixed_tiny):
    at = _open_insights(_app())
    assert not at.exception
    body = " ".join(m.value for m in at.tabs[3].markdown)
    assert wording.HOW_RANGE_FIXED_TINY in body and wording.HOW_RANGE not in body


def test_insights_cache_does_not_follow_a_project_switch(insights_scored):
    at = _open_insights(_app())
    assert ui_insights.CACHE_KEY in at.session_state
    FoodOptimizer("second").add_ingredient("Flour", 0, 100)
    at.run()
    at.selectbox(key="project_select").set_value("second")
    at.run()
    _submit_button(at.sidebar, "Open").click()
    at.run()
    assert not at.exception
    assert ui_insights.CACHE_KEY not in at.session_state
    assert "insights_choice" not in at.session_state
    assert "insights_start" not in at.session_state
