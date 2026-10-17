"""Review critique (prototype, 2026-10-04): one fix pass from three
critiques of the Review (clarity, statistics, confidence).

- Forecast check: one shared axis per result — a row per formulation in
  forecast order, the 95 % range a light bar, the forecast a tick, the
  measured value a dot, a miss in the accent colour labelled above/below;
  measured ties counted aloud.
- Stop signal: within the tie threshold of the top score, "Within noise of
  the top score. Stop, or change the target." and no widening advice;
  else every binding limit named, a moved-together ingredient said.
- Order: an exact within-Round permutation test; the chart's chance range
  is the same null, so the dot clears it exactly when the verdict is
  Predictable; under 5 pairs: "too few pairs", no dot.
- Noise: a measurement's share of the Overall score's assumed 1 point,
  σ × w ÷ (s × √k): 0.3 on a 0–15 target worth 25 of 4.
- Page order; "Show as lines" gone; Start vs best, against target; range
  notation; Results' In / Above / Below; tooltips of about 120 words.

Synthetic projects and hand-made checks only; no pilot data."""
import ast
import json
import os
import pathlib
import re
from types import SimpleNamespace as NS

import pytest

import bo_views
import insights
import learned
import ui_insights as U
import wording
from insights import Check
from tests.test_food_bo import _BANNED


def _check(no, r, name, pred, meas, low=None, high=None, full=True):
    low = pred - 1 if low is None else low
    high = pred + 1 if high is None else high
    return Check(no, r, name, pred, low, high, meas, low <= meas <= high, full,
                 raw_mean=pred, sd=0.5)


def _words(text):
    plain = re.sub(r":\w+\[(.*?)\]", r"\1", text).replace("**", "")
    return len([w for w in plain.split() if re.search(r"\w", w)])


@pytest.fixture
def plain(monkeypatch):
    monkeypatch.setattr(U, "_say", lambda o, t: t)
    monkeypatch.setattr(U.insights, "_range_of", lambda o, n: (0.0, 100.0 if n == insights.SCORE
                                                                else 15.0))
    monkeypatch.setattr(U, "decimals", lambda o, n: 1)


class _Obj:
    """A project the helpers read: one target measurement."""

    def __init__(self):
        self.objectives = [{"name": "Consistency", "goal": "target", "target": 12,
                            "weight": 100, "min_val": 0, "max_val": 15}]
        self.formulation_ids = list(range(1, 40))

    def measurements_by_importance(self):
        return list(self.objectives)


def _round6():
    """Round 6: F16 and F17 measured equal (a tie), both above their range;
    F18 inside; forecasts 84.3 / 84.7 / 83.7."""
    checks = []
    for r, base in ((4, 10), (5, 13)):
        checks += [_check(base + k, r, insights.SCORE, 95 - 3 * k, 95 - 3 * k, low=92 - 3 * k,
                          high=98 - 3 * k) for k in range(3)]
    for no, sp, sm in zip((16, 17, 18), (84.3, 84.7, 83.7), (88.0, 88.0, 81.5)):
        checks.append(_check(no, 6, insights.SCORE, sp, sm, low=sp - 2.5, high=sp + 2.5))
    return checks


# ------------------------------------------------------------------ #
#  1. Forecast check: shared axis, misses flagged, ties counted
# ------------------------------------------------------------------ #


# ------------------------------------------------------------------ #
#  2. The stop signal at the top score
# ------------------------------------------------------------------ #

def _yard(value=1.0, assumed=True):
    return insights.Yardstick(value, None, 0, False, assumed=assumed, df=0, groups=0, left_out=0)


# ------------------------------------------------------------------ #
#  3. Ordering: the exact within-Round permutation test
# ------------------------------------------------------------------ #

def _perfect(rounds, n=3, name="Consistency"):
    checks = []
    for r in range(rounds):
        checks += [_check(10 * r + k, r + 1, name, 10 - 2 * k, 10 - 2 * k) for k in range(n)]
    return checks


def test_permutation_p_is_exact():
    # Each Round of 3 in perfect order: P(S = 3) = 1/6; two Rounds 1/36.
    o = learned.ordering(_perfect(2), "Consistency", 0.0)
    assert (o.right, o.pairs) == (6, 6)
    assert o.p == pytest.approx(1 / 36)
    # The binomial on pairs claimed 1/64 (pairs within a Round treated as
    # independent).
    assert learned.chance_tail(6, 6) == pytest.approx(1 / 64)
    one = learned.ordering(_perfect(1), "Consistency", 0.0)
    assert one.p == pytest.approx(1 / 6)


def test_round_null_sums_to_one_and_is_symmetric():
    cks = _perfect(1, n=4)
    null = learned.round_null(cks)
    assert sum(null.values()) == pytest.approx(1.0)
    assert null[6] == pytest.approx(1 / 24) and null[-6] == pytest.approx(1 / 24)


@pytest.mark.parametrize("seed", range(12))
def test_the_chance_range_and_the_verdict_always_agree(seed):
    import random
    rng = random.Random(seed)
    checks = []
    for r in range(rng.randint(2, 6)):
        preds = [rng.uniform(0, 10) for _ in range(3)]
        for k, p in enumerate(preds):
            meas = p + rng.gauss(0, rng.choice([0.5, 2.0, 5.0]))
            checks.append(_check(10 * r + k, r + 1, "Consistency", p, round(meas)))
    o = learned.ordering(checks, "Consistency", 0.0)
    if o.pairs < learned.MIN_PAIRS:
        return
    lo, hi = o.chance_share()
    beyond = o.right / o.pairs > hi + 1e-12
    assert beyond == (o.p < learned.VERDICT_P)
    if o.verdict == learned.PREDICTABLE:
        assert beyond


    # Specific wording (2026-10-12), re-pinned: "under 5 pairs", not "too few".


# ------------------------------------------------------------------ #
#  Noise: each measurement's share of the Overall score's 1 point
# ------------------------------------------------------------------ #

def test_noise_derivation_gives_0_3(monkeypatch):
    monkeypatch.setattr(learned, "_repeat_noise", lambda opt, name: None)
    objs = [{"name": n, "goal": "target", "target": 5, "weight": 25, "min_val": 0,
             "max_val": 15} for n in ("A", "B", "C", "D")]
    opt = NS(objectives=objs, measurements_by_importance=lambda: objs)
    noise = learned.pair_noise(opt)
    assert noise[insights.SCORE] == 1.0
    assert all(noise[n] == pytest.approx(0.3) for n in "ABCD")
    # Two measurements, 60 / 40 on 0–10: 1 × 10 ÷ (60 × √2), 1 × 10 ÷ (40 × √2).
    two = [dict(objs[0], weight=60, max_val=10), dict(objs[1], weight=40, max_val=10)]
    opt2 = NS(objectives=two, measurements_by_importance=lambda: two)
    got = learned.pair_noise(opt2)
    assert got["A"] == pytest.approx(10 / (60 * 2 ** 0.5))
    assert got["B"] == pytest.approx(10 / (40 * 2 ** 0.5))


# ------------------------------------------------------------------ #
#  Start vs best, against target
# ------------------------------------------------------------------ #

def _rows():
    def r(name, start, best, target=None, goal="target"):
        return {"name": name, "goal": goal, "target": target, "low": 0.0, "high": 15.0,
                "start": start, "best": best, "unit": "/15"}
    return [r("Consistency", 8, 12, 12), r("Creaminess", 4, 5, 5), r("Tanginess", 5, 5, 5),
            r("Thickness", 13, 12.5, 13)]


# ------------------------------------------------------------------ #
#  Range notation; Results' flags
# ------------------------------------------------------------------ #

def test_range_never_prints_a_top_below_its_point():
    p = insights.Prediction(99.5, 93.4, 100.0, 99.5, 3.0)
    # Round 2: the range's top with the point's decimals.
    assert U.capped_range(p, 100.0) == "99.5 (93.4–100.0)"
    assert U.capped_range(p, 100.0, 0) == "≥ 99 (93–100)"
    assert U.capped_range(insights.Prediction(13.5, 11.1, 15.0, 13.5, 1.0), 15.0) == (
        "13.5 (11.1–15.0)")
    assert U.capped_range(insights.Prediction(94.3, 91.9, 96.8, 94.3, 1.0), 100.0) == (
        "94.3 (91.9–96.8)")
    assert U.score_text(insights.Prediction(99.4, 91.0, 100.0, 99.4, 3.0)) == "99 (91–100)"
    for text in (U.capped_range(p, 100.0), U.score_text(p)):
        assert "–≥" not in text


# ------------------------------------------------------------------ #
#  Tooltips, words, _BANNED
# ------------------------------------------------------------------ #


    # The tried gaps (owner, 2026-10-05): the count first, then the largest.


def _critique_strings():
    """Every string literal in wording.py's critique-fixes block."""
    text = pathlib.Path(wording.__file__).read_text()
    block = text[text.index("# --- critique fixes ---"):]
    out = []
    for node in ast.walk(ast.parse(block)):
        if isinstance(node, ast.Constant) and isinstance(node.value, str):
            out.append(node.value)
        elif isinstance(node, ast.JoinedStr):
            out.append("".join(v.value for v in node.values if isinstance(v, ast.Constant)))
    return out


def test_critique_strings_pass_banned():
    texts = [t for t in _critique_strings() if " " in t and not t.startswith(("'", '"'))]
    texts = [t for t in texts if "\n" not in t[:1]]     # docstrings start with a word too
    bad = [t for t in texts if any(p.search(t) for p in _BANNED)
           and not t.lstrip().startswith(("Every", "The", "A ", "'", "Review"))]
    assert bad == []
    for t in (
              wording.CARD_CEILING, 
              wording.implied_line("90", "94"), ):
        assert not any(p.search(t) for p in _BANNED), t
        assert "spread" not in t.lower() and "interval" not in t.lower()


def test_show_as_lines_is_gone():
    assert not hasattr(wording, "STORY_SHOW_LINES")
    assert not hasattr(U, "tried_chart") and not hasattr(U, "PATH_SHOW_LINES_KEY")


# ------------------------------------------------------------------ #
#  Page order (a synthetic project through the app)
# ------------------------------------------------------------------ #


@pytest.fixture(scope="module")
def line_a(tmp_path_factory):
    from tests import demo_projects as D
    from tests.review_screens import _install, _open
    here = os.getcwd()
    os.chdir(tmp_path_factory.mktemp("review-critique"))
    try:
        with bo_views.fresh():
            _install(D.LINE_A)
            yield _open(D.LINE_A), D.load(D.LINE_A)
    finally:
        os.chdir(here)


# ------------------------------------------------------------------ #
#  Follow-up (owner): the stop signal after a rise too
# ------------------------------------------------------------------ #

