"""Insights wave C on screen: the "More charts" expander after question 4 —
"Can you have both?", "What happens when three ingredients trade places?"
and "How far did each prediction miss?". AppTest, like
tests/test_insights_ui.py, plus the chart builders on their own."""
import json
import pickle
import re

import pytest
import torch
from streamlit.testing.v1 import AppTest

import insights
import ui_insights
import wording
from food_bo import FoodOptimizer
from tests.test_app_ui import APP_PATH, _submit_button, burger, scored  # noqa: F401
from tests.test_insights_ui import _open_insights, _tab_text, insights_scored  # noqa: F401


def _app():
    return AppTest.from_file(APP_PATH, default_timeout=180)


def _rerun_on(at, tab=wording.TAB_INSIGHTS):
    """Run again with `tab` open. AppTest does not replay the tab bar: a
    plain at.run() after a tab was chosen draws the first tab again."""
    at.session_state["main_tab"] = tab
    return at.run()


def _counting(monkeypatch, name):
    calls = []
    real = getattr(insights, name)
    monkeypatch.setattr(insights, name, lambda *a, **k: calls.append(1) or real(*a, **k))
    return calls


def _more(at):
    found = [e for e in at.tabs[3].expander if e.label == wording.MORE_CHARTS]
    return found[0] if found else None


def _charts(block):
    """The Altair charts drawn directly in `block`, as Vega-Lite dicts."""
    return [json.loads(c.proto.spec) for c in block.children.values()
            if getattr(c, "type", "") == "vega_lite_chart"]


def _state(opt):
    return pickle.dumps({k: v for k, v in opt.__dict__.items() if k != "_forms_cache"})


_WAVE_C_WORDS = re.compile(
    r"\b(held|hold|scales?|ranges|residuals?|posterior|probability|Round-ahead|optimum|"
    r"optimal|maximi[sz]e|confidence|the model)\b", re.I)
_CASED = re.compile(r"\b(Min|Max)\b")


def _assert_clean(*strings):
    """The repo-wide guard plus the words wave C must never put on screen."""
    from tests.test_food_bo import _BANNED
    for s in strings:
        assert not any(p.search(s) for p in _BANNED), s
        assert not _WAVE_C_WORDS.search(s), s
        assert not _CASED.search(s), s


# ------------------------------------------------------------------ #
#  Task 3: the expander and "Can you have both?"
# ------------------------------------------------------------------ #

def test_more_charts_is_collapsed_after_question_4_and_before_how(insights_scored):
    at = _open_insights(_app())
    assert not at.exception
    labels = [e.label for e in at.tabs[3].expander]
    assert labels.index(wording.MORE_CHARTS) == labels.index(wording.HOW_WORKED_OUT) - 1
    assert _more(at).proto.expanded is False
    md, _ = _tab_text(at)
    assert md.index(f"**{wording.Q_BEST}**") < md.index(f"**{wording.Q_PAIR}**")


def test_can_you_have_both_opens_on_the_two_largest_shares(insights_scored):
    at = _open_insights(_app())
    more = _more(at)
    assert [m.value for m in more.markdown][0] == f"**{wording.Q_PAIR}**"
    [box] = more.multiselect
    assert box.label == wording.PAIR_PICK
    assert box.value == ["Firmness", "Juiciness"]          # Share 1.5 before 1.0
    assert wording.PAIR_CAPTION in [c.value for c in more.caption]
    [spec] = _charts(more)
    titles = [layer["encoding"]["x"].get("title") for layer in spec["layer"]
              if "x" in layer.get("encoding", {})]
    assert wording.pair_axis("Firmness (N)", "target", 6) in titles


def test_one_measurement_picked_asks_for_two(insights_scored):
    at = _open_insights(_app())
    _more(at).multiselect[0].set_value(["Juiciness"])
    _rerun_on(at)
    assert not at.exception
    more = _more(at)
    assert wording.PICK_TWO in [c.value for c in more.caption]
    assert _charts(more) == []


def test_the_pick_is_kept_across_reruns(insights_scored):
    at = _open_insights(_app())
    _more(at).multiselect[0].set_value(["Juiciness", "Firmness"])
    _rerun_on(at)
    _rerun_on(at)
    assert _more(at).multiselect[0].value == ["Juiciness", "Firmness"]


def test_no_more_charts_before_five_formulations_have_all_their_results(scored):
    at = _open_insights(_app())
    assert not at.exception
    assert _more(at) is None


def test_no_more_charts_with_nothing_to_show(tmp_path, monkeypatch):
    """One measurement, no Formulation total and too few checks: every chart
    is past its gate, so the expander is not drawn empty."""
    monkeypatch.chdir(tmp_path)
    opt = FoodOptimizer("one")
    opt.set_amount_unit("g")
    opt.add_ingredient("Pea protein", 0, 25)
    opt.add_objective("Firmness", 1.0, goal="max", min_val=0, max_val=10, unit="N")
    for no in range(1, 7):
        opt.tell({"Pea protein": 3.0 * no}, {"Firmness": 0.3 * no}, formulation_no=no, batch_no=1)
    at = _open_insights(_app())
    assert not at.exception
    assert _more(at) is None


def test_more_charts_write_nothing_to_the_project(insights_scored):
    at = _open_insights(_app())
    opt = at.session_state["optimizer"]
    before = _state(opt)
    _more(at).multiselect[0].set_value(["Juiciness", "Firmness"])
    _rerun_on(at)
    assert _state(at.session_state["optimizer"]) == before


def test_a_project_switch_drops_the_pair_pick(insights_scored):
    at = _open_insights(_app())
    assert ui_insights.PAIR_KEY in at.session_state
    FoodOptimizer("second").add_ingredient("Flour", 0, 100)
    at.run()
    at.selectbox(key="project_select").set_value("second")
    at.run()
    _submit_button(at.sidebar, "Open").click()
    at.run()
    assert not at.exception
    assert ui_insights.PAIR_KEY not in at.session_state


def test_a_deleted_measurement_drops_out_of_the_pick(insights_scored):
    insights_scored.add_objective("Colour", 1.0, goal="max", min_val=0, max_val=10)
    at = _open_insights(_app())
    _more(at).multiselect[0].set_value(["Firmness", "Colour"])
    _rerun_on(at)
    at.session_state["optimizer"].remove_objective("Colour")
    _rerun_on(at)
    assert not at.exception
    more = _more(at)
    assert more.multiselect[0].value == ["Firmness"]
    assert wording.PICK_TWO in [c.value for c in more.caption]


def test_a_failed_chart_is_one_line_and_the_tab_stands(insights_scored, monkeypatch):
    def boom(*a):
        raise RuntimeError("no pair")
    monkeypatch.setattr(insights, "pair_points", boom)
    at = _open_insights(_app())
    assert not at.exception
    assert [c.value for c in _more(at).caption] == [wording.CHART_FAILED]
    assert wording.HOW_WORKED_OUT in [e.label for e in at.tabs[3].expander]


def _pair_opt(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    opt = FoodOptimizer("pair")
    opt.set_amount_unit("g")
    opt.add_ingredient("Pea protein", 0, 25)
    opt.add_objective("Firmness", 60, goal="target", target=6, min_val=0, max_val=10, unit="N")
    opt.add_objective("Cook loss", 40, goal="min", min_val=0, max_val=30, unit="%")
    for no, (f, c) in enumerate(((5.0, 20.0), (6.0, 25.0), (12.5, 10.0)), start=1):
        opt.tell({"Pea protein": 5.0 * no}, {"Firmness": f, "Cook loss": c},
                 formulation_no=no, batch_no=1)
    opt.add_objective("Juiciness", 20, goal="max", min_val=0, max_val=10)
    opt.tell({"Pea protein": 22.0}, {"Firmness": 7.0, "Cook loss": 15.0},
             formulation_no=4, batch_no=2)                 # waiting for Juiciness
    return opt


def _layers(spec, mark):
    return [l for l in spec["layer"]
            if (l["mark"]["type"] if isinstance(l["mark"], dict) else l["mark"]) == mark]


def test_pair_chart_axes_name_the_goal_and_span_range_plus_data(tmp_path, monkeypatch):
    opt = _pair_opt(tmp_path, monkeypatch)
    spec = ui_insights.pair_chart(opt, "Firmness", "Cook loss",
                                  insights.pair_points(opt, "Firmness", "Cook loss")).to_dict()
    [dots] = [l for l in _layers(spec, "circle")]
    assert dots["encoding"]["x"]["title"] == "Firmness (N) · target 6"
    assert dots["encoding"]["y"]["title"] == "Cook loss (%) · lower is better"
    assert dots["encoding"]["x"]["scale"]["domain"] == [0.0, 12.5]   # 12.5 is past the Range
    assert dots["encoding"]["y"]["scale"]["domain"] == [0.0, 30.0]


def test_pair_chart_target_lines_only_for_targets(tmp_path, monkeypatch):
    opt = _pair_opt(tmp_path, monkeypatch)
    spec = ui_insights.pair_chart(opt, "Firmness", "Cook loss",
                                  insights.pair_points(opt, "Firmness", "Cook loss")).to_dict()
    rules = _layers(spec, "rule")
    assert [list(r["encoding"]) for r in rules] == [["x"]]          # Firmness only
    [rows] = [v for k, v in spec["datasets"].items() if v and "t" in v[0]]
    assert rows == [{"t": 6.0}]


def test_pair_chart_colours_by_score_and_draws_waiting_rows_hollow(tmp_path, monkeypatch):
    opt = _pair_opt(tmp_path, monkeypatch)
    points = insights.pair_points(opt, "Firmness", "Cook loss")
    assert [p.score is None for p in points] == [False, False, False, True]
    spec = ui_insights.pair_chart(opt, "Firmness", "Cook loss", points).to_dict()
    [dots] = _layers(spec, "circle")
    colour = dots["encoding"]["color"]
    assert colour["field"] == "score"
    assert colour["scale"]["domain"] == [0, opt.utility_ceiling()]
    assert colour["scale"]["scheme"] == {"name": "blues", "extent": [0.25, 1]}
    hollow = [l for l in _layers(spec, "point") if l["mark"].get("filled") is False
              and l["mark"].get("color") == "gray" and l["mark"].get("size") == 60]
    assert hollow and hollow[0]["transform"] == [{"filter": {"field": "waiting", "equal": True}}]
    [label] = _layers(spec, "text")
    assert label["encoding"]["text"] == {"value": wording.PAIR_BEST}


def test_pair_axis_wording():
    assert wording.pair_axis("Cook loss (%)", "min") == "Cook loss (%) · lower is better"
    assert wording.pair_axis("Juiciness", "max") == "Juiciness · higher is better"
    assert wording.pair_axis("Firmness (N)", "target", 6) == "Firmness (N) · target 6"
    assert wording.pair_axis("Firmness (N)", "target", None) == "Firmness (N)"


def test_exact_wave_c_strings_for_the_expander_and_the_pair():
    assert wording.MORE_CHARTS == "More charts"
    assert wording.Q_PAIR == "Can you have both?"
    assert wording.PAIR_CAPTION == (
        "Each dot is a formulation you measured; lines mark your targets. If no dot sits near "
        "both targets, the two may pull against each other.")


def test_the_pair_strings_pass_the_word_guards():
    _assert_clean(wording.MORE_CHARTS, wording.Q_PAIR, wording.PAIR_PICK, wording.PICK_TWO,
                  wording.PAIR_CAPTION, wording.PAIR_NONE, wording.PAIR_BEST,
                  wording.CHART_FAILED,
                  wording.pair_axis("Cook loss (%)", "min"),
                  wording.pair_axis("Juiciness", "max"),
                  wording.pair_axis("Firmness (N)", "target", 6))


def test_no_formulation_has_both_values_says_so(insights_scored):
    insights_scored.add_objective("Colour", 1.0, goal="max", min_val=0, max_val=10)
    at = _open_insights(_app())
    _more(at).multiselect[0].set_value(["Firmness", "Colour"])
    _rerun_on(at)
    assert not at.exception
    more = _more(at)
    assert wording.PAIR_NONE in [c.value for c in more.caption]
    assert _charts(more) == []
