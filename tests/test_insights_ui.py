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
