"""Helpers left from Insights wave C (the "More charts" expander tests went
with the old Insights tab). Other test files still import these fixtures
and helpers (ferment_scored, mixture_scored, _rerun_on, _mix_parts, ...);
delete this file when none does."""
import dataclasses
import itertools
import json
import pickle
import re

import pytest
from tests.demo_build_reason import REVIEW_FOUR
import torch
from streamlit.testing.v1 import AppTest

import insights
import ui_insights
import wording
from food_bo import FoodOptimizer
from tests.test_app_ui import APP_PATH, _submit_button, burger, scored  # noqa: F401
from tests.test_insights_ui import insights_scored  # noqa: F401


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
    """Charts proto: the two-measurement fold (Two measurements, was
    Trade-offs) and the triangle, which now sits in Explored and
    unexplored, read as one block (pair first, as these tests were
    written); None when neither is drawn."""
    folds = {e.label: e for e in at.tabs[3].expander}
    found = [folds[k] for k in (wording.PAIR_FOLD, wording.FOLD_EXPLORED) if k in folds]
    pair = wording.PAIR_FOLD in folds
    tri = wording.FOLD_EXPLORED in folds and any(
        m.value == f"##### {wording.Q_MIXTURE}" for m in folds[wording.FOLD_EXPLORED].markdown)
    if not pair and not tri:
        return None
    return _Folds(found if tri else found[:1])


class _Folds:
    """Several folds read as one: list attributes joined, in order."""

    def __init__(self, folds):
        self.folds = folds

    @property
    def proto(self):
        return self.folds[0].proto

    @property
    def label(self):
        return self.folds[0].label

    @property
    def children(self):
        out = {}
        for k, f in enumerate(self.folds):
            out.update({(k, key): c for key, c in f.children.items()})
        return out

    def __getattr__(self, name):
        out = []
        for f in self.folds:
            out += list(getattr(f, name))
        return out


def _open_insights(at):
    """test_insights_ui's, with Explored and unexplored open: the triangle
    is drawn there now (charts proto), and only while it is open."""
    at.run()
    at.session_state[ui_insights.EXPLORED_KEY] = True
    at.session_state["main_tab"] = wording.TAB_INSIGHTS
    at.run()
    return at


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


# ------------------------------------------------------------------ #
#  Task 4: "What happens when three ingredients trade places?"
# ------------------------------------------------------------------ #

@pytest.fixture
def mixture_scored(tmp_path, monkeypatch):
    """Pea protein 10–20 g, Oil 2–6 g, Starch 1–5 g and Water filling to
    100 g; two measurements; twelve formulations over four Rounds, all with
    their results (enough for all three charts)."""
    monkeypatch.chdir(tmp_path)
    opt = FoodOptimizer("mix")
    opt.set_amount_unit("g")
    opt.add_ingredient("Pea protein", 10, 20)
    opt.add_ingredient("Oil", 2, 6)
    opt.add_ingredient("Starch", 1, 5)
    opt.add_ingredient("Water", 0, 100)
    opt.set_formulation_total(100)
    opt.set_formula("Water", "= rest")
    opt.add_objective("Firmness", 1.5, goal="target", target=6, min_val=0, max_val=10, unit="N")
    opt.add_objective("Juiciness", 1.0, goal="max", min_val=0, max_val=10, unit="/10")
    g = torch.Generator().manual_seed(4)
    for no in range(1, 13):
        u = torch.rand(3, generator=g).tolist()
        pea, oil, starch = 10 + 10 * u[0], 2 + 4 * u[1], 1 + 4 * u[2]
        opt.tell(opt.fill_formulas({"Pea protein": pea, "Oil": oil, "Starch": starch}),
                 {"Firmness": 0.4 * pea + 0.3 * starch, "Juiciness": 1 + 1.2 * oil},
                 formulation_no=no, batch_no=1 + (no - 1) // 3)
    return opt


def _triangles(block):
    # The one chart drawn without a view frame (configure_view(stroke=None)).
    return [c for c in _charts(block)
            if "stroke" in ((c.get("config") or {}).get("view") or {})
            and c["config"]["view"]["stroke"] is None]


def _corner_texts(mix):
    """The three labels as mixture_chart draws them: each one's largest
    amount inside the region, one shared number of decimals across all
    three (the most any of them needs, up to 2)."""
    amounts = mix.top
    decimals = ui_insights._shared_decimals(amounts)
    return [wording.mixture_corner(mix.names[k],
                                   ui_insights.join_unit(f"{amounts[k]:.{decimals}f}", mix.units[k]))
            for k in range(3)]


def _most_room(opt, order):
    """The first combination of three, in `order`, with the largest allowed
    share (the default rule, worked out here independently)."""
    start = insights.start_recipe(opt, opt.best_index())
    combos = list(itertools.combinations(order, 3))[:ui_insights._THREE_TRIES]
    shares = [insights.allowed_share(opt, list(c), start) for c in combos]
    return list(combos[shares.index(max(shares))])


def _full_slices(monkeypatch):
    """Counts the triangle's own slices: the default three's room check
    (steps=1, three points) is not one."""
    calls = []
    real = insights.mixture_slice

    def counted(*a, **k):
        if k.get("steps", insights.GRID_STEPS) != 1:
            calls.append(1)
        return real(*a, **k)
    monkeypatch.setattr(insights, "mixture_slice", counted)
    return calls


def _refused(monkeypatch, **kw):
    def fake(opt, names, start, steps=insights.GRID_STEPS):
        base = dict(names=list(names), units=["g"] * 3, start=dict(start), total=20.0,
                    lower=[0.0] * 3, upper=[1.0] * 3, span=0.0, points=[], dots=[],
                    refusal=None, start_breaks=None, follows=[], fill_moves=None, skipped=[])
        return insights.Mixture(**{**base, **kw})
    monkeypatch.setattr(insights, "mixture_slice", fake)


def _mix_parts(opt):
    fitted = insights.fit_all(opt)
    start = insights.start_recipe(opt, opt.best_index())
    mix = insights.mixture_slice(opt, ["Pea protein", "Oil", "Water"], start)
    return mix, insights.mixture_predictions(opt, fitted, mix)


# ------------------------------------------------------------------ #
#  Task 4 fix round 1: the triangle as Streamlit actually renders it
# ------------------------------------------------------------------ #

def _mark_type(layer):
    mark = layer["mark"]
    return mark["type"] if isinstance(mark, dict) else mark


# ------------------------------------------------------------------ #
#  Task 5: "How far did each prediction miss?"
# ------------------------------------------------------------------ #

def _checks(opt):
    return insights.check_rounds(opt)


# ------------------------------------------------------------------ #
#  On wave B's final code: the triangle's default three read the moved
#  spreads, and every wave C line is in the project's own words.
# ------------------------------------------------------------------ #

_OFFERED = ["Pea protein", "Oil", "Starch", "Water"]


def _three_with(monkeypatch, found):
    def fake(opt, fitted, start):
        if isinstance(found, Exception):
            raise found
        return found
    monkeypatch.setattr(insights, "spreads", fake)
    return ui_insights._three_default(None, {}, _OFFERED, {})


@pytest.fixture
def ferment_scored(tmp_path, monkeypatch):
    """Process settings alone, two measurements, twelve trials over four
    Rounds with their results: "Can you have both?" passes its gate."""
    monkeypatch.chdir(tmp_path)
    opt = FoodOptimizer("ferment")
    opt.add_process_parameter("Incubation temperature", 30, 42, unit="°C")
    opt.add_process_parameter("Incubation time", 4, 16, unit="h")
    opt.add_objective("Acidity", 1.0, goal="target", target=4.5, min_val=3.0, max_val=7.0,
                      unit="pH")
    opt.add_objective("Thickness", 1.0, goal="max", min_val=0, max_val=10, unit="/10")
    g = torch.Generator().manual_seed(7)
    for no in range(1, 13):
        u = torch.rand(2, generator=g).tolist()
        temp, hours = 30 + 12 * u[0], 4 + 12 * u[1]
        opt.tell({"Incubation temperature": temp, "Incubation time": hours},
                 {"Acidity": 3.0 + 0.08 * temp, "Thickness": 1 + 0.5 * hours},
                 formulation_no=no, batch_no=1 + (no - 1) // 3)
    return opt


# ------------------------------------------------------------------ #
#  Final fix wave (waves B + C)
# ------------------------------------------------------------------ #

def _layer_rows(spec, layer):
    return spec["datasets"][layer["data"]["name"]]


def _assert_offset_not_haloed(spec, label_layer):
    """No second, white-stroked layer duplicates the label (a halo glows in
    dark mode and doubles the text for screen readers): a plain grey label,
    moved clear of the mark it names by a `dy` offset, stands alone."""
    assert not label_layer["mark"].get("stroke")
    assert label_layer["mark"].get("dy")
    same_data = [l for l in spec["layer"] if _mark_type(l) == "text"
                 and l["data"] == label_layer["data"]]
    assert same_data == [label_layer]


def _pair_caption_for(opt, a, b):
    return ui_insights.pair_caption(opt, a, b)


