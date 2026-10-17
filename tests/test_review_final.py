"""Review round 2 (2026-10-05): the fix plan from four critics (visual,
statistics, demo walkthrough, wording), P1 → P4.

- One widening everywhere: each measurement's Round-ahead checks widened by
  its own record from earlier Rounds (insights.nest_measure_widen), so Make a
  round's Predicted measurements, the Forecast check and Results' In / Above
  / Below read one 95 % range per forecast.
- Holm across the results checked before Predicts; the chance bar drawn at
  each result's Holm level, so dot and verdict agree.
- Two forecasts both at the top of the Range count as too close to call.
- Implied (expected): E[score] over each measurement's forecast.
- The stop: a concrete line under the headline, its "?" in numbers, the
  noisy-perfect score; "3 Repeats would measure the noise".
- Are we improving?: the zone within noise of the top; Start vs best's labels;
  the suggestions folded under Stop and cached; the Forecast check its own
  section; the profiler's cramped columns named, not drawn; secondary
  buttons under Stop; one form per idea.

Synthetic projects and hand-made checks only; no pilot data."""
import ast
import contextlib
import json
import pathlib
import re
from types import SimpleNamespace as NS

import pytest

import insights
import learned
import wording
from food_bo import FoodOptimizer
from insights import Check
from tests.test_food_bo import _BANNED

NBSP = " "


def _check(no, r, name, pred, meas, sd=0.5, raw=None, low=None, high=None, full=True):
    raw = pred if raw is None else raw
    low = raw - insights.Z95 * sd if low is None else low
    high = raw + insights.Z95 * sd if high is None else high
    return Check(no, r, name, pred, low, high, meas, low <= meas <= high, full,
                 raw_mean=raw, sd=sd)


def _stub(names=("Firmness",), top=15.0):
    return NS(objectives=[{"name": n, "goal": "target", "target": 7, "weight": 100 / len(names),
                           "min_val": 0.0, "max_val": top} for n in names],
              utility_ceiling=lambda: 100.0)


def _project(tmp_path, monkeypatch, name="final", k=4):
    """k target measurements on 0–15, equal shares, targets 12/5/5/13."""
    monkeypatch.chdir(tmp_path)
    opt = FoodOptimizer(name)
    opt.set_amount_unit("g")
    opt.add_ingredient("Oat", 0, 100)
    opt.add_ingredient("Water", 0, 100)
    targets = [("Consistency", 12), ("Creaminess", 5), ("Tanginess", 5), ("Thickness", 13)][:k]
    for n, t in targets:
        opt.add_objective(n, 100.0 / k, goal="target", target=t, min_val=0, max_val=15)
    return opt


# ------------------------------------------------------------------ #
#  P1 #1 — one widening everywhere
# ------------------------------------------------------------------ #

def _measure_checks():
    """Rounds 3–6 of one measurement; Rounds 3–5 miss by 2 sd each, so
    the record widens Round 6's range by 2."""
    out = []
    no = 1
    for r in (3, 4, 5):
        for k in range(4):
            out.append(_check(no, r, "Firmness", 7.0, 7.0 + (1.0 if k % 2 else -1.0)))
            no += 1
    out.append(_check(no, 6, "Firmness", 7.0, 8.5))     # 3 sd above: outside 1.96 sd, inside 3.92
    return out


def test_one_widening_is_the_same_range_on_make_forecast_check_and_results():
    """Re-pinned (one-glance Review, fix round 1) without the removed
    Forecast check panels (check_frame): the measurement widening Make a
    round uses is the one each Round's check carries."""
    opt = _stub()
    checks = insights.nest_measure_widen(opt, _measure_checks())
    r6 = [c for c in checks if c.round == 6][0]
    # Make a round widens the open Round by every other Round's record:
    make = insights.measure_widen(checks, "Firmness", skip_round=6)
    assert make == pytest.approx(2.0)
    assert r6.widen == pytest.approx(make)
    p = insights.widened(NS(low=0.0, high=15.0),
                         insights.Prediction(7.0, 6.0, 8.0, 7.0, 0.5), make)
    assert (p.low, p.high) == pytest.approx((r6.low, r6.high))
    # Round 6 was a miss unwidened; inside the one range it is not.
    assert r6.inside and r6.low <= 8.5 <= r6.high
    # The earliest Round has no record: unwidened.
    assert [c for c in checks if c.round == 3][0].widen == 1.0


def test_measure_widen_reads_the_models_own_sd_not_the_nested_widening():
    checks = insights.nest_measure_widen(_stub(), _measure_checks())
    plain = [c for c in _measure_checks() if c.round < 6]
    assert insights.measure_widen(checks, "Firmness", 6) == insights.measure_widen(plain,
                                                                                    "Firmness")


# ------------------------------------------------------------------ #
#  P1 #2 — Holm
# ------------------------------------------------------------------ #

def _ordering(name, p_exact, pairs=6, right=6):
    o = learned.Ordering(name, right=right, pairs=pairs)
    # A null whose tail at the observed stat is p_exact: S = stat with
    # probability p_exact, else 0.
    stat = 2 * right - pairs
    o.nulls = [{stat: p_exact, -stat: 1.0 - p_exact}]
    o.rounds = [3]
    return o


def test_holm_adjusts_before_predicts_and_the_bar_agrees():
    ords = {"a": _ordering("a", 0.004), "b": _ordering("b", 0.03), "c": _ordering("c", 0.3)}
    learned.holm(ords)
    assert ords["a"].p_holm == pytest.approx(0.012)
    assert ords["b"].p_holm == pytest.approx(0.06)
    assert ords["a"].verdict == learned.PREDICTABLE
    # p 0.03 alone would be Predictable; adjusted for 3 results it is not.
    assert ords["b"].verdict == learned.PARTLY
    assert ords["c"].verdict == learned.NOT_YET
    assert {o.tested for o in ords.values()} == {3}
    # The dot clears its bar exactly when the verdict is Predicts.
    for o in ords.values():
        lo, hi = o.chance_share()
        assert (o.share > hi + 1e-9) == (o.verdict == learned.PREDICTABLE), o.name


def test_holm_keeps_a_later_result_inside_its_bar_after_one_fails():
    ords = {"a": _ordering("a", 0.03), "b": _ordering("b", 0.04)}
    learned.holm(ords)
    assert ords["a"].verdict != learned.PREDICTABLE and ords["b"].verdict != learned.PREDICTABLE
    # b alone (0.04 < 0.05) would clear 0.05; held at the failed level it does not.
    assert ords["b"].level == ords["a"].level == pytest.approx(0.025)
    lo, hi = ords["b"].chance_share()
    assert not ords["b"].share > hi + 1e-9


def test_one_result_alone_is_not_adjusted():
    ords = learned.holm({"a": _ordering("a", 0.03)})
    assert ords["a"].p_holm == pytest.approx(0.03) and ords["a"].verdict == learned.PREDICTABLE


# ------------------------------------------------------------------ #
#  P1 #3 — both at the cap: too close to call
# ------------------------------------------------------------------ #

def test_two_forecasts_at_the_top_are_too_close_to_call():
    a = _check(1, 2, insights.SCORE, 100.0, 97.0, raw=104.0)
    b = _check(2, 2, insights.SCORE, 100.0, 95.0, raw=101.0)
    c = _check(3, 2, insights.SCORE, 92.0, 90.0)
    assert learned.at_cap(a) and learned.at_cap(b) and not learned.at_cap(c)
    assert learned.too_close(a, b, 1.0)
    assert learned.too_close(a, b, 0.0)
    assert not learned.too_close(a, c, 1.0)
    call = learned.round_call([a, b, c], insights.SCORE, 2, 1.0)
    assert call.close == 1 and call.counted == 2


# ------------------------------------------------------------------ #
#  P1 #4 — Implied (expected)
# ------------------------------------------------------------------ #


# ------------------------------------------------------------------ #
#  P1 #5 — the stop's concrete line and its "?" numbers
# ------------------------------------------------------------------ #


# ------------------------------------------------------------------ #
#  P2 #7 — the top zone on Are we improving?
# ------------------------------------------------------------------ #

def _near_top(tmp_path, monkeypatch):
    opt = _project(tmp_path, monkeypatch, k=1)
    values = [[9, 10, 8], [11, 12, 11.5]]
    no = 1
    for r, row in enumerate(values, 1):
        for v in row:
            opt.tell({"Oat": 10.0 * no, "Water": 50.0}, {"Consistency": v},
                     formulation_no=no, batch_no=r)
            no += 1
    return opt


# ------------------------------------------------------------------ #
#  P2 #8 — the suggestions folded under Stop; the preview cached
# ------------------------------------------------------------------ #

class _FakeSt:
    """Records what a builder draws: markdown, captions, buttons, folds."""

    def __init__(self, fold_open=False):
        self.calls, self.fold_open = [], fold_open
        self.session_state = {}

    def __getattr__(self, name):
        def call(*a, **kw):
            self.calls.append((name, a, kw))
            if name == "button":
                return False
            if name in ("container", "empty", "spinner"):
                return contextlib.nullcontext(self)
            return None
        return call

    @contextlib.contextmanager
    def expander(self, label, **kw):
        self.calls.append(("expander", (label,), kw))
        yield NS(open=self.fold_open)


# ------------------------------------------------------------------ #
#  P2 #9 — the Forecast check is its own section
# ------------------------------------------------------------------ #

def _round6():
    out = []
    for no, sp, sm in zip((16, 17, 18), (94.3, 94.7, 93.7), (98.3, 98.3, 91.7)):
        out.append(_check(no, 6, insights.SCORE, sp, sm, sd=1.2))
    return out


class _Obj:
    def __init__(self):
        self.objectives = [{"name": "Consistency", "goal": "target", "target": 12,
                            "weight": 100, "min_val": 0, "max_val": 15}]
        self.formulation_ids = list(range(1, 40))

    def measurements_by_importance(self):
        return list(self.objectives)


# ------------------------------------------------------------------ #
#  P2 #10 — Start vs best labels
# ------------------------------------------------------------------ #

def _rows():
    def r(name, start, best, target):
        return {"name": name, "goal": "target", "target": target, "low": 0.0, "high": 15.0,
                "start": start, "best": best, "unit": "/15"}
    return [r("Consistency", 8, 12, 12), r("Creaminess", 4, 5, 5), r("Tanginess", 5, 4, 5),
            r("Thickness", 13, 13, 13)]


# ------------------------------------------------------------------ #
#  P2 #13 — the profiler's cramped columns
# ------------------------------------------------------------------ #

# ------------------------------------------------------------------ #
#  P2 #16 — the primaries go secondary under Stop
# ------------------------------------------------------------------ #


# ------------------------------------------------------------------ #
#  P3 #17–19 — one form per idea
# ------------------------------------------------------------------ #


def test_captions_and_labels_keep_to_their_word_budgets():
    def words(t):
        return len(re.sub(r"\*\*", "", t).split())
    for text, limit in ((wording.CARD_CEILING, 5), (wording.back_to_batch_label(6, 3), 5),
                        (wording.OFF_BY_TITLE, 5)):
        assert words(text) <= limit, text


# ------------------------------------------------------------------ #
#  _BANNED
# ------------------------------------------------------------------ #

def _round2_source():
    src = pathlib.Path(wording.__file__).read_text()
    return src[src.index("# --- round 2 fixes ---"):]


def test_no_round_two_string_says_a_banned_word():
    tree = ast.parse(_round2_source())
    # Docstrings say what a function returns, not what reaches the screen;
    # _GLOSSARY_GONE names rows taken off the screen.
    skip = set()
    for n in ast.walk(tree):
        if isinstance(n, ast.FunctionDef) and n.body and isinstance(n.body[0], ast.Expr):
            skip.add(id(n.body[0].value))
        if isinstance(n, ast.Assign) and any(getattr(t, "id", "") == "_GLOSSARY_GONE"
                                             for t in n.targets):
            skip |= {id(c) for c in ast.walk(n.value)}
    shown = [n.value for n in ast.walk(tree) if isinstance(n, ast.Constant)
             and isinstance(n.value, str) and id(n) not in skip]
    bad = [t for t in shown if any(p.search(t) for p in _BANNED)]
    assert not bad, bad
    rendered = [
                wording.METHOD_STOP]
    assert not [t for t in rendered if any(p.search(t) for p in _BANNED)]
