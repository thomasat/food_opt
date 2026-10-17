"""Persona fixes 2 (prototype, 2026-10-02; the senior food scientist's
re-test: clearer, but not yet trusted):

- the "Called it?" block is an honest "Forecast check": a measured value
  outside its 95 % range is a miss on either side; a pair counts only when
  its forecasts differ by at least one noise SD of that result (else "too
  close to call", left out of every count, drawn faint); a steady bias is
  said in one line;
- a Round of entered formulations says its notes are the source;
- the widen advice says why; a suggestion moving the same ingredient the
  other way says it "checks the other side";
- the score recipe on screen under Closer to target;
- Flat after a rise reads "Plateau since Round N" wherever the word shows.

Synthetic projects and hand-made checks only; no pilot data."""
import re
from types import SimpleNamespace as NS

import pytest

import insights
import learned
import ui_batch
import ui_insights as U
import wording
from food_bo import FoodOptimizer
from insights import Check
from tests.test_food_bo import _BANNED


def _check(no, r, name, pred, meas, low=None, high=None, full=True):
    low = pred - 1 if low is None else low
    high = pred + 1 if high is None else high
    return Check(no, r, name, pred, low, high, meas, low <= meas <= high, full,
                 raw_mean=pred, sd=0.5)


def _words(text, numbers=True):
    plain = re.sub(r":\w+\[(.*?)\]", r"\1", text).replace("**", "")
    words = [w for w in plain.split() if re.search(r"\w", w)]
    if not numbers:
        words = [w for w in words if not re.fullmatch(r"[\d.,:;()F]+", w)]
    return len(words)


class _Obj:
    """A project the Forecast check helpers read: one target measurement."""

    def __init__(self, goal="target"):
        self.objectives = [{"name": "Consistency", "goal": goal, "target": 12}]

    def measurements_by_importance(self):
        return [{"name": "Consistency"}]


@pytest.fixture
def plain(monkeypatch):
    monkeypatch.setattr(U, "_say", lambda o, t: t)
    monkeypatch.setattr(U.insights, "_range_of", lambda o, n: (0.0, 100.0 if n == insights.SCORE
                                                                else 15.0))
    monkeypatch.setattr(U, "decimals", lambda o, n: 1)


def _earlier():
    """Rounds 4-5: Consistency forecast 3 points apart, ordered right."""
    checks = []
    for r, base in ((4, 10), (5, 13)):
        checks += [_check(base + k, r, "Consistency", 13 - 3 * k, 13 - 3 * k) for k in range(3)]
        checks += [_check(base + k, r, insights.SCORE, 95 - 3 * k, 95 - 3 * k, low=92 - 3 * k,
                          high=98 - 3 * k) for k in range(3)]
    return checks


def _round6(score_pred=(84.3, 84.7, 83.7), score_meas=(88.0, 88.0, 81.5),
            cons_pred=(8.5, 8.4, 8.6), cons_meas=(7, 7, 8)):
    checks = _earlier()
    for no, sp, sm, cp, cm in zip((16, 17, 18), score_pred, score_meas, cons_pred, cons_meas):
        checks.append(_check(no, 6, insights.SCORE, sp, sm, low=sp - 2.5, high=sp + 2.5))
        checks.append(_check(no, 6, "Consistency", cp, cm, low=cp - 2.5, high=cp + 2.5))
    return checks


NOISE = {insights.SCORE: 1.0, "Consistency": 1.0}


# ------------------------------------------------------------------ #
#  1. Forecast check
# ------------------------------------------------------------------ #

def test_too_close_pairs_are_not_counted():
    checks = _round6()
    call = learned.round_call(checks, insights.SCORE, 6, 1.0)
    # Forecast gaps: F17-F16 0.4 (a measured tie), F16-F18 0.6, F17-F18 1.0.
    assert call.counted == 1 and call.right == 1 and call.close == 1 and call.ties == 1
    # A larger noise: nothing counts, the top one is not claimed either.
    call = learned.round_call(checks, insights.SCORE, 6, 3.0)
    assert call.counted == 0 and call.close == 2 and call.top_right is None
    assert not learned.order_held(call)
    # No noise: as before.
    assert learned.round_call(checks, insights.SCORE, 6, 0.0).counted == 2


def test_too_close_pairs_are_left_out_of_the_record_and_verdict():
    checks = _round6()
    before = learned.ordering(checks, "Consistency", 0.0)
    after = learned.ordering(checks, "Consistency", 1.0)
    # Round 6's Consistency forecasts are 0.1-0.2 apart: too close.
    assert before.pairs == 8 and after.pairs == 6 and after.close == 2
    assert after.verdict == learned.PREDICTABLE
    # A noise above every gap: nothing to count, Not yet (too few).
    far = learned.ordering(checks, "Consistency", 7.0)
    assert far.pairs == 0 and far.verdict == learned.NOT_YET and far.reason == "few"


# Review critique (fix 1): no slopegraph; too close to call is said in the panel's
# subtitle and the headline (tests/test_review_critique.py).


def test_pair_noise_is_the_tasting_noise(tmp_path, monkeypatch):
    """Coordinator's ruling: the noise Progress uses — Repeats' SD from the
    same 3 df Progress needs, else the assumed 1 point; a measurement, that
    point in its own units through its share of the score (σ × Range width
    ÷ share)."""
    opt = _opt(tmp_path, monkeypatch)
    opt.add_objective("Sour", 25.0, goal="target", target=5, min_val=0, max_val=15)
    opt.add_objective("Crisp", 50.0, goal="max", min_val=0, max_val=10)
    noise = learned.pair_noise(opt)
    assert learned.overall_noise(opt) == (1.0, True)
    assert noise[insights.SCORE] == 1.0
    # Review critique (owner's decision): each scored measurement carries an
    # equal share of the Overall variance: σ × width ÷ (share × √k).
    k = len(learned.scored_measurements(opt))
    for name in ("Sour", "Crisp", "Firmness"):
        obj = next(o for o in opt.objectives if o["name"] == name)
        width = obj["max_val"] - obj["min_val"]
        assert noise[name] == pytest.approx(width / (obj["weight"] * k ** 0.5)), name
    # 25 points on a 0-15 scale: 25 / 15 points per unit (the yogurt's case;
    # with its 4 measurements, 0.3).
    assert learned.points_per_unit({"goal": "target", "weight": 25.0, "min_val": 0,
                                    "max_val": 15}) == pytest.approx(25 / 15)
    # Repeats with fewer than Progress's 3 df: still the assumed 1 point.
    spread = lambda df: (lambda o, n, m=None, w=(): NS(
        sd=2.0 if n == insights.SCORE else None, df=df, pairs=df, groups=df))
    monkeypatch.setattr(learned.insights, "repeat_sd", spread(2))
    assert learned.overall_noise(opt) == (1.0, True)
    assert insights.noise_yardstick(opt).assumed
    # From 3 df, as Progress: the Repeats' SD.
    monkeypatch.setattr(learned.insights, "repeat_sd", spread(insights.YARDSTICK_DF))
    assert learned.overall_noise(opt) == (2.0, False)
    assert learned.pair_noise(opt)["Sour"] == pytest.approx(2.0 * 15 / (25 * k ** 0.5))


# ------------------------------------------------------------------ #
#  2. Provenance from the notes
# ------------------------------------------------------------------ #

def test_round_caption_respects_notes():
    own = [{"note": wording.OWN_FORMULATION_NOTE}, {"note": wording.OWN_FORMULATION_NOTE}]
    assert ui_batch.own_round_caption(own) == wording.ONLY_OWN_FORMULATIONS_CAPTION
    noted = [{"note": "Pilot batch 6.1 · EGBO"}, {"note": wording.OWN_FORMULATION_NOTE}]
    caption = ui_batch.own_round_caption(noted)
    assert caption == wording.ENTERED_ROUND_CAPTION
    assert "only your own" not in caption and "notes" in caption
    # Generic: any note.
    assert ui_batch.own_round_caption([{"note": "Bench control B"}]) == \
        wording.ENTERED_ROUND_CAPTION


# ------------------------------------------------------------------ #
#  3. The widen reason; checks the other side
# ------------------------------------------------------------------ #

def _opt(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    opt = FoodOptimizer("persona2")
    opt.set_amount_unit("g")
    opt.add_ingredient("Water", 30, 100)
    opt.add_ingredient("Wheat gluten", 0, 50)
    opt.add_objective("Firmness", 1.0, goal="target", target=6, min_val=0, max_val=10)
    for k, (w, g, f) in enumerate([(40, 20, 5), (50, 10, 6), (30, 30, 4), (60, 5, 7),
                                   (45, 25, 5.5), (35, 15, 6.2)], 1):
        opt.tell({"Water": float(w), "Wheat gluten": float(g)}, {"Firmness": f},
                 formulation_no=k, batch_no=1 + (k > 3))
    return opt


# ------------------------------------------------------------------ #
#  4. The score caption
# ------------------------------------------------------------------ #


# ------------------------------------------------------------------ #
#  5. Plateau
# ------------------------------------------------------------------ #


# ------------------------------------------------------------------ #
#  Wording limits and _BANNED
# ------------------------------------------------------------------ #

def _strings():
    src = open(wording.__file__).read().split("# --- persona fixes 2 ---", 1)[1]
    return re.findall(r'"((?:[^"\\\n]|\\.)*)"', src)


