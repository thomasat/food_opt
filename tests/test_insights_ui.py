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
    marks = {layer["mark"]["type"] for layer in spec["spec"]["layer"]}
    assert not marks & {"errorbar", "tick"}                   # no end caps
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


def test_track_chart_target_only_in_its_own_panel():
    opt = _burger(rounds=4, per_round=4, noise=0.1)
    spec = _spec(ui_insights.track_chart(opt, insights.check_rounds(opt)))
    layers = _layers_by_panel(spec)
    [(mark, panels)] = layers["target"]
    firm = ui_insights.panel_label(opt, "Firmness")
    assert list(panels) == [firm]
    assert [r["target"] for r in panels[firm]] == [6.0]
    # Dashed and lighter than the likely-range bars, so it never reads as one.
    assert mark["strokeDash"] == [4, 3]
    bars = [m for m, _ in layers["check"] if m["type"] == "rule"]
    assert bars and all("strokeDash" not in m for m in bars)
    assert mark.get("strokeWidth", 1) < bars[0]["strokeWidth"]
    # Every check in its own panel, once.
    for m, by in layers["check"]:
        assert sum(len(v) for v in by.values()) <= len(insights.check_rounds(opt))


def test_track_chart_width_is_bounded():
    opt = _burger(rounds=1, per_round=1)
    checks = [insights.Check(no, 1 + no // 4, name, 5.0, 4.0, 6.0, 5.2, True, True)
              for no in range(1, 121) for name in ("Firmness", "Juiciness", insights.SCORE)]
    spec = _spec(ui_insights.track_chart(opt, checks))
    assert spec["spec"]["width"] <= 240 and spec["spec"]["height"] <= 240
    x = [layer["encoding"]["x"] for layer in spec["spec"]["layer"] if "x" in layer["encoding"]]
    assert x and all(e["type"] == "quantitative" for e in x)


def _path_spec(opt, name="Oil"):
    fitted = insights.fit_all(opt)
    p = insights.build_path(opt, name, insights.start_recipe(opt, opt.best_index()))
    return p, _spec(ui_insights.path_chart(opt, p, insights.path_predictions(opt, fitted, p)))


def test_path_chart_marks_stay_in_their_own_panel():
    opt = _burger(rounds=3)
    p, spec = _path_spec(opt)
    layers = _layers_by_panel(spec)
    labels = [ui_insights.panel_label(opt, n) for n in ui_insights.path_panels(opt)]
    firm = ui_insights.panel_label(opt, "Firmness")
    # Target: Firmness's panel only, once; the Overall score panel has none.
    [(mark, panels)] = layers["target"]
    assert list(panels) == [firm] and [r["target"] for r in panels[firm]] == [6.0]
    assert mark["strokeDash"] == [4, 3]
    # Start marker: once per panel.
    [(_, starts)] = layers["start"]
    assert sorted(starts) == sorted(labels)
    assert all([r["x"] for r in v] == [p.start_amount] for v in starts.values())
    # Grey: exactly this chart's untested intervals, once in each panel.
    want = []
    if p.amounts[0] < p.tested[0]:
        want.append((p.amounts[0], p.tested[0]))
    if p.amounts[-1] > p.tested[1]:
        want.append((p.tested[1], p.amounts[-1]))
    assert want                                   # this fixture has some
    [(_, grey)] = layers["grey"]
    assert sorted(grey) == sorted(labels)
    for v in grey.values():
        assert sorted((r["x"], r["x2"]) for r in v) == sorted(want)
    # The curve: one point per amount in each panel, for band and line alike.
    for _, by in layers["curve"]:
        assert sorted(by) == sorted(labels)
        assert all(len(v) == len(p.amounts) for v in by.values())


def test_path_chart_without_targets_has_no_target_layer():
    opt = _burger(rounds=3)
    opt.objectives[0]["goal"] = "max"
    _, spec = _path_spec(opt)
    assert "target" not in _layers_by_panel(spec)


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
    _check(wording.insights_header(17, "5", 2))
    _check(wording.insights_gate(3))
    _check(wording.Q_PREDICTIONS)
    _check(wording.q_change(True))
    _check(wording.q_change(False))
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
    _check(wording.path_not_checked("Fat: at most 16 g", "Fat"))
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

    # Final fix wave
    _check(wording.INSIGHTS_FAILED)
    _check(wording.CHANGE_NONE)
    _check(wording.DETAILS_NONE)
    _check(wording.rounds_not_checked([1, 2]))
    _check(wording.rounds_not_checked([1]))
    _check(wording.rounds_span([3, 5, 6]))


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
    for q in (wording.Q_PREDICTIONS, wording.q_change(False), wording.Q_IMPROVING):
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
    at.session_state["main_tab"] = wording.TAB_INSIGHTS
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


def test_rounds_span_contiguous_and_not():
    assert wording.rounds_span([3]) == "Round 3"
    assert wording.rounds_span([3, 4, 5]) == "Rounds 3–5"
    assert wording.rounds_span([3, 4]) == "Rounds 3–4"
    assert wording.rounds_span([3, 5]) == "Rounds 3 and 5"
    assert wording.rounds_span([3, 5, 6]) == "Rounds 3, 5 and 6"


def test_rounds_not_checked_wording():
    assert wording.rounds_not_checked([1, 2]) == (
        "Rounds 1–2 are not checked: fewer than 5 formulations were recorded before them.")
    assert wording.rounds_not_checked([1]) == (
        "Round 1 is not checked: fewer than 5 formulations were recorded before it.")


def test_final_wave_wording():
    assert wording.INSIGHTS_WORKING == "Preparing the insights…"
    assert wording.HOW_WORKED_OUT == "How these insights are made"
    assert wording.panel_failed("Firmness") == "Could not be predicted for Firmness."
    assert wording.q_change(True) == "What happens if you change one ingredient or process setting?"
    assert wording.q_change(False) == "What happens if you change one ingredient?"
    assert wording.path_not_checked("Fat: at most 16 g", "Fat") == (
        "Not checked: Fat: at most 16 g — some ingredients have no Fat value.")
    assert wording.INSIGHTS_FAILED == "Insights could not be prepared for this project."
    assert wording.CHANGE_NONE == (
        "Nothing to change yet: every ingredient and process setting is fixed.")
    assert wording.DETAILS_NONE == "This table fills in once a round has been checked."
    for text in (wording.INSIGHTS_WORKING, wording.HOW_WORKED_OUT, wording.panel_failed("X"),
                 wording.PREDICTIONS_NONE, wording.BEST_LATEST, wording.best_since(1),
                 wording.best_since(3), wording.HOW_MODELS, wording.q_change(True),
                 wording.q_change(False), wording.INSIGHTS_FAILED, wording.CHANGE_NONE,
                 wording.DETAILS_NONE, wording.rounds_not_checked([1, 2])):
        assert "worked out" not in text.lower() and "working out" not in text.lower(), text


def test_round_and_tab_words_come_from_constants(monkeypatch):
    # G3: rename the round word and the tab, and every Insights string follows.
    import importlib
    src = open(wording.__file__).read()
    assert 'f"2 · {TAB_BATCH_NAME}"' in src or "TAB_BATCH_NAME" in src
    assert wording.TAB_BATCH.endswith(wording.TAB_BATCH_NAME)
    assert wording.TAB_BATCH_NAME in wording.HOW_MODELS
    for fn in ("PREDICTIONS_NONE", "BEST_LATEST", "best_since", "HOW_MODELS"):
        block = src[src.index(fn + (" = " if fn.isupper() else "(")):][:400]
        assert " round" not in block.split("\n\n")[0].replace("{ROUND}", ""), (fn, block)


def test_path_skipped_names_the_property():
    opt = _burger(rounds=2)
    opt.add_property("Fat")
    opt.set_property_value("Oil", "Fat", 100.0)
    opt.add_constraint("Fat", max_val=5.0)
    p = insights.build_path(opt, "Oil", insights.start_recipe(opt, opt.best_index()))
    assert p.skipped == [(opt.property_limit_text(opt.constraints[0]), "Fat")]


def test_band_and_amount_text_use_num():
    opt = _burger(rounds=2)
    assert ui_insights._amount(0.00012, "g") == "0.00012 g"
    assert ui_insights._amount(123.4, "g") == "123 g"


# The tab in the app ------------------------------------------------ #

def _tab_text(at):
    tab = at.tabs[3]
    return ([m.value for m in tab.markdown], [c.value for c in tab.caption])


def test_insights_failure_shows_one_line_and_is_not_retried(insights_scored, monkeypatch):
    calls = []
    def boom(opt):
        calls.append(1)
        raise RuntimeError("no checks")
    monkeypatch.setattr(insights, "check_rounds", boom)
    at = _open_insights(_app())
    assert not at.exception
    md, caps = _tab_text(at)
    assert caps.count(wording.INSIGHTS_FAILED) == 1
    assert any(m.startswith("Best so far:") for m in md)       # section 3 still shows
    at.session_state["main_tab"] = wording.TAB_INSIGHTS
    at.run()
    assert not at.exception and calls == [1]


@pytest.mark.parametrize("where", ["build_path", "path_predictions", "start_recipe"])
def test_section_2_failure_keeps_sections_1_and_3(insights_scored, monkeypatch, where):
    real = getattr(insights, where)
    def boom(*a, **k):
        import inspect
        caller = inspect.stack()[1].function
        if caller == "_change":
            raise RuntimeError("no path")
        return real(*a, **k)
    monkeypatch.setattr(insights, where, boom)
    at = _open_insights(_app())
    assert not at.exception
    md, caps = _tab_text(at)
    assert wording.panel_failed("Pea protein") in caps
    assert wording.Q_PREDICTIONS in " ".join(md)
    assert any(m.startswith("Best so far:") for m in md)


def test_section_2_says_when_there_is_nothing_to_change(insights_scored, monkeypatch):
    monkeypatch.setattr(insights, "path_choices", lambda opt: [])
    at = _open_insights(_app())
    assert not at.exception
    assert wording.CHANGE_NONE in _tab_text(at)[1]


def test_question_2_names_process_settings_only_when_there_are_some(insights_scored):
    at = _open_insights(_app())
    body = " ".join(_tab_text(at)[0])
    assert wording.q_change(False) in body and wording.q_change(True) not in body


@pytest.mark.parametrize("spread, shown", [(0.04, False), (0.0, False), (0.06, True)])
def test_repeats_line_needs_a_spread_that_shows(insights_scored, monkeypatch, spread, shown):
    monkeypatch.setattr(insights, "repeat_spread", lambda opt: spread)
    at = _open_insights(_app())
    md = " ".join(_tab_text(at)[0])
    assert ("Repeats of the same formulation" in md) == shown
    if shown:
        assert wording.repeats_line("0.06") in md


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


def test_details_table_empty_says_so(insights_two_rounds):
    at = _open_insights(_app())
    assert not at.exception
    assert wording.DETAILS_NONE in _tab_text(at)[1]
    assert not at.tabs[3].dataframe


def test_details_table_shows_rows_after_one_checked_round(insights_scored):
    at = _open_insights(_app())
    frame = at.tabs[3].dataframe[0].value
    assert len(frame) == 3                                        # 3 checked each
    assert set(frame[wording.DETAILS_COLUMNS[5]]) == {"—"}
    assert wording.DETAILS_NONE not in _tab_text(at)[1]


def test_unchecked_rounds_caption(insights_scored):
    at = _open_insights(_app())
    assert wording.rounds_not_checked([1, 2]) in _tab_text(at)[1]


def test_how_range_wording_says_the_range_is_widened():
    assert wording.HOW_RANGE == (
        "Likely range: where about 19 in 20 results should land, including the variation "
        "seen between Repeats (a 95 % prediction interval). When earlier predictions missed "
        "by more than their range allowed, the range is widened to match.")
    assert wording.HOW_RANGE_FIXED_TINY == (
        "Likely range: your model settings assume no variation between Repeats, so the range "
        "comes from how sure the prediction is, widened when earlier predictions missed by "
        "more than it allowed.")
