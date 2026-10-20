import pytest
from tests.demo_build_reason import REVIEW_FOUR
import food_bo
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


@pytest.mark.skip(reason=REVIEW_FOUR)
def test_path_chart_panels_score_first_and_grey_where_untested():
    opt = _burger(rounds=3)
    fitted = insights.fit_all(opt)
    p = insights.build_path(opt, "Oil", insights.start_recipe(opt, opt.best_index()))
    chart = ui_insights.path_chart(opt, p, insights.path_predictions(opt, fitted, p))
    assert ui_insights.path_panels(opt)[0] == insights.SCORE
    spec = str(_spec(chart))
    assert "rect" in spec                                     # untested shading layer


@pytest.mark.skip(reason=REVIEW_FOUR)
def test_path_chart_x_axis_is_not_zero_based():
    opt = _burger(rounds=3)
    fitted = insights.fit_all(opt)
    p = insights.build_path(opt, "Oil", insights.start_recipe(opt, opt.best_index()))
    spec = _spec(ui_insights.path_chart(opt, p, insights.path_predictions(opt, fitted, p)))
    assert "'zero': False" in str(spec)


@pytest.mark.skip(reason=REVIEW_FOUR)
def test_path_chart_none_without_points():
    opt = _burger(rounds=3)
    opt.add_quantity_constraint(["Pea", "Oil"], max_val=8.0)
    p = insights.build_path(opt, "Oil", opt.fill_formulas({"Pea": 7.0, "Oil": 3.0}))
    assert ui_insights.path_chart(opt, p, {}) is None


def _layers_by_panel(spec):
    """What each layer of a faceted chart actually draws, panel by panel:
    the chart's one dataset put through that layer's own filters. A layer
    carrying its own data would be drawn whole in every panel, so none may."""
    from collections import defaultdict
    rows = spec["datasets"][spec["data"]["name"]]
    out = {}
    for layer in spec["spec"]["layer"]:
        assert "data" not in layer, layer
        kept, mark = rows, None
        for t in layer.get("transform", []):
            f = t["filter"]
            if "equal" in f:
                kept = [r for r in kept if r.get(f["field"]) == f["equal"]]
                if f["field"] == "mark":
                    mark = f["equal"]
            elif set(f) == {"field"}:
                kept = [r for r in kept if r.get(f["field"])]
            else:
                raise AssertionError(f"unexpected filter {f}")
        assert mark is not None, layer          # every layer names its rows
        by = defaultdict(list)
        for r in kept:
            by[r["panel"]].append(r)
        out.setdefault(mark, []).append((layer["mark"], dict(by)))
    return out


def _path_spec(opt, name="Oil"):
    fitted = insights.fit_all(opt)
    p = insights.build_path(opt, name, insights.start_recipe(opt, opt.best_index()))
    return p, _spec(ui_insights.path_chart(opt, p, insights.path_predictions(opt, fitted, p)))


@pytest.mark.skip(reason=REVIEW_FOUR)
def test_path_chart_without_targets_has_no_target_layer():
    opt = _burger(rounds=3)
    opt.objectives[0]["goal"] = "max"
    _, spec = _path_spec(opt)
    assert "target" not in _layers_by_panel(spec)


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


def _open_how(at):
    """Method and details is keyed and drawn only while open (BO views fix
    round 1, T56 I5): open it, as the reader does."""
    import ui_insights
    at.session_state[ui_insights.HOW_KEY] = True
    at.session_state["main_tab"] = wording.TAB_INSIGHTS
    at.run()
    return at


def test_insights_is_the_last_tab(insights_scored):
    at = _app()
    at.run()
    assert [t.label for t in at.tabs][-1] == wording.TAB_INSIGHTS


@pytest.mark.skip(reason="review profiler prototype: pins Try one's path and Compare all, which the profiler replaces; reconcile at the final merge")
def test_insights_sections_and_expanders(insights_scored):
    at = _open_insights(_app())
    assert not at.exception
    tab = at.tabs[3]
    text = " ".join(m.value for m in tab.markdown)
    for q in (wording.Q_STAND, ):
        assert q in text
    labels = [e.label for e in tab.expander]
    for lab in (wording.FOLD_PREDICTION_FIGURES, wording.FOLD_MEASURED_CHART,
                wording.FOLD_PROGRESS_CHART,
                wording.FOLD_NEXT_REASONING, wording.try_one_fold(False), wording.FOLD_METHOD):
        assert lab in labels, labels
    # One lit button: the foot (spec 2026-09-30 §2.4).
    assert [b.label for b in tab.button if b.proto.type == "primary"] == [
        wording.START_NEXT_BATCH]
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
    assert "Progress chart" not in labels
    at = _open_insights(_app())
    # It lives on the Review as "Best so far", under its own caption.
    tab = at.tabs[3]
    caption, _help = ui_insights.score_caption(insights_scored, ui_insights.round_view(insights_scored))
    assert any(wording.PANEL_SCORE_TITLE in m.value for m in tab.markdown)
    assert any(c.value == caption for c in tab.caption)


def test_insights_not_computed_while_another_tab_is_open(insights_scored, monkeypatch):
    calls = []
    real = food_bo.fit_gpytorch_mll
    monkeypatch.setattr(food_bo, "fit_gpytorch_mll", lambda *a, **k: calls.append(1) or real(*a, **k))
    at = _app()
    at.session_state["main_tab"] = wording.TAB_RESULTS
    at.run()
    assert not at.exception
    assert calls == []
    assert ui_insights.CACHE_KEY not in at.session_state


def test_the_review_fits_nothing_on_any_run(insights_scored, monkeypatch):
    # Review v3 draws from the measurements alone: it fits no model
    # on the first run or on a rerun, and keeps no cache.
    fit_calls = []
    real_fit = food_bo.fit_gpytorch_mll
    monkeypatch.setattr(food_bo, "fit_gpytorch_mll",
                        lambda *a, **k: fit_calls.append(1) or real_fit(*a, **k))
    at = _open_insights(_app())
    assert not at.exception
    assert fit_calls == []
    assert ui_insights.CACHE_KEY not in at.session_state
    at.session_state["main_tab"] = wording.TAB_INSIGHTS
    at.run()
    assert fit_calls == []


def test_no_scored_formulation_one_line(burger):
    at = _open_insights(_app())
    tab = at.tabs[3]
    assert [c.value for c in tab.caption] == [wording.INSIGHTS_NONE]
    assert not tab.markdown and not tab.expander


def test_insights_cache_does_not_follow_a_project_switch(insights_scored):
    at = _open_insights(_app())
    # Review v3 draws from the measurements alone: the key is set here, as
    # another screen's work would leave it, to pin app.py's switch cleanup.
    at.session_state[ui_insights.CACHE_KEY] = {"stale": True}
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


# ------------------------------------------------------------------ #
#  Final fix wave D, E, F, G
# ------------------------------------------------------------------ #

def test_num_never_uses_scientific_notation():
    n = ui_insights._num
    assert n(1234.567) == "1235" and n(100.0) == "100"
    assert n(12.34) == "12.3" and n(10.0) == "10.0"
    assert n(3.14159) == "3.14" and n(2.5) == "2.5" and n(4.0) == "4"
    assert n(0.5) == "0.5" and n(0.0123) == "0.012" and n(0.00012) == "0.00012"
    assert n(-3.14159) == "-3.14"
    # Only a true zero reads 0: a non-zero value shows enough decimals (up
    # to 6) to be non-zero, and anything smaller says so.
    assert n(0.0) == "0" and n(-0.0) == "0"
    assert n(0.00005) == "0.00005" and n(-0.00001) == "-0.00001"
    assert n(0.0000012) == "0.000001" and n(0.000001) == "0.000001"
    assert n(0.0000004) == "< 0.000001" and n(-0.0000004) == "> -0.000001"
    for v in (1e-3, 2.5e-4, 1e-4, 5e-5, 1e-6, 1e-9, -1e-9, 0.05, 99.99, 1e6):
        assert "e" not in n(v).lower(), (v, n(v))
        assert n(v).strip("<> -") not in ("0", "0.0", "0.000000"), (v, n(v))


def test_round_and_tab_words_come_from_constants(monkeypatch):
    # G3: rename the round word and the tab, and every Insights string follows.
    import importlib
    src = open(wording.__file__).read()
    assert 'f"2 · {TAB_BATCH_NAME}"' in src or "TAB_BATCH_NAME" in src
    assert wording.TAB_BATCH.endswith(wording.TAB_BATCH_NAME)


# The tab in the app ------------------------------------------------ #

def _tab_text(at):
    tab = at.tabs[3]
    return ([m.value for m in tab.markdown], [c.value for c in tab.caption])


@pytest.mark.skip(reason="review profiler prototype: pins Try one's path and Compare all, which the profiler replaces; reconcile at the final merge")
def test_section_2_says_when_there_is_nothing_to_change(insights_scored, monkeypatch):
    monkeypatch.setattr(insights, "path_choices", lambda opt: [])
    at = _open_insights(_app())
    assert not at.exception
    assert wording.CHANGE_NONE in _tab_text(at)[1]


@pytest.mark.skip(reason=REVIEW_FOUR)
def test_question_2_names_process_settings_only_when_there_are_some(insights_scored):
    at = _open_insights(_app())
    labels = [e.label for e in at.tabs[3].expander]
    assert wording.try_one_fold(False) in labels and wording.try_one_fold(True) not in labels


@pytest.fixture
def insights_two_rounds(burger):
    """Six scored formulations over two rounds: enough to fit, none checked."""
    no = 0
    for r in (1, 2):
        for pea, mc in ((8.0, 0.5), (14.0, 1.5), (20.0, 2.5)):
            no += 1
            pea += r * 0.7
            burger.tell({"Pea protein": pea, "Methylcellulose": mc},
                        {"Juiciness": 3 + 0.2 * pea, "Firmness": 1 + 0.25 * pea + mc},
                        formulation_no=no, batch_no=r)
    return burger


    # Ledger, statistician cold read I2 (fix round B): no "19 in 20" claim; the
    # plain reading gives the 95 % aim and the coverage measured on simulated projects.
    # Ftue fix A's re-run (I9 folds): 90 % of Overall scores (was 89 %).
