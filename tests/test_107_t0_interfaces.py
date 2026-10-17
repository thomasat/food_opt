"""Insights for a first-time user, T0 (spec 2026-10-01 rev. 2; the review's
"Proposed task split"): the functions W1, W2 and W3 fill in exist with
their final signatures, so the three work in parallel without editing one
another's functions, and the data they hand one another has its fields.

Only signatures and shapes are pinned here, never a stub's return value:
each wave replaces its own stub's body and keeps this file green."""
import dataclasses
import inspect

import pytest
from tests.demo_build_reason import REVIEW_FOUR

import bo_views
import insights
import ui_insights
import wording

# (module, function, its parameters in order), owner in the comment.
_SIGNATURES = [
    # W1: status logic and the first screen.
    (insights, "status_facts", ["opt", "checks", "standing", "gain"]),
    (insights, "tie_partner", ["opt", "best"]),
    (ui_insights, "_status_lines", ["opt", "facts"]),
    (ui_insights, "_why_link", ["opt"]),
    (ui_insights, "_repeat_link", ["opt", "facts"]),
    # W2: the expert layer.
    (insights, "model_summary", ["opt"]),
    (insights, "status_rules", ["opt", "checks"]),
    (ui_insights, "tooltip_for", ["opt", "line_id", "facts"]),
    # W3: explored and unexplored.
    (insights, "tested_strips", ["opt", "pool"]),
    (insights, "strip_gaps", ["opt", "strips", "k"]),
    (insights, "limit_hits", ["opt"]),
    (bo_views, "map_values", ["opt", "model"]),
    (ui_insights, "explored_fold", ["opt", "computed"]),
    # demo build: strips_chart deleted (nothing drew it)
    (ui_insights, "explore_map_chart", ["pool_mean", "pool_sd", "tested", "picks", "ref", "opt"]),
]


@pytest.mark.parametrize("module,name,params", _SIGNATURES,
                         ids=[f"{m.__name__}.{n}" for m, n, _ in _SIGNATURES])
@pytest.mark.skip(reason=REVIEW_FOUR)
def test_each_interface_has_its_final_signature(module, name, params):
    func = getattr(module, name)
    assert list(inspect.signature(func).parameters) == params


def _fields(cls):
    return [f.name for f in dataclasses.fields(cls)]


def test_the_shapes_the_waves_hand_one_another():
    """StatusFacts (W1 → W2's tooltips), ModelFacts (W2 → wording), the strip
    shapes (W3)."""
    assert {"prediction_row", "clauses", "headline", "verdict", "skill", "disagreement",
            "bias", "guard", "yardstick", "progress", "improvement", "best", "tie", "luck",
            "next", "scored"} <= set(_fields(insights.StatusFacts))
    facts = insights.StatusFacts(scored=20, prediction_row="reliable")
    assert facts.predictions == "reliable"
    assert insights.StatusFacts(scored=3).predictions is None
    assert {"value", "formulation", "round", "score", "state", "latest"} == set(
        _fields(insights.StripTick))
    assert {"name", "unit", "low", "high", "reach", "ticks", "fill", "setting", "options"} == set(
        _fields(insights.Strip))
    assert {"name", "kind", "low", "high", "size", "option"} == set(_fields(insights.StripGap))
    assert {"round", "hits", "picks", "binding"} == set(_fields(insights.LimitHit))
    # wording.model_summary reads every ModelFacts field it needs, and no
    # other (a facts object with the defaults writes a whole summary).
    facts = insights.ModelFacts(cold=False, scored=20, cold_start=5, values=False, total=True)
    assert wording.model_summary(facts)[0] == wording.MODEL_SUMMARY_LEAD
    assert facts.folds == insights.FOLDS and facts.fit_tolerance == insights.FIT_OPTIONS["ftol"]


def test_the_rule_keys_are_the_words_keys():
    """R-B1's rows, worst first, in the order the rule table lists them."""
    assert insights.PREDICTION_ROWS[0] == "off_repeat"
    assert insights.PREDICTION_ROWS[-1] == "reliable"
    assert set(insights.PREDICTION_STATUS) == set(insights.PREDICTION_ROWS)
    assert set(insights.PREDICTION_STATUS.values()) == set(wording.PREDICTIONS_WORDS)
    assert set(insights.PROGRESS_STATES) == set(wording.PROGRESS_WORDS)
    table = [r[1].split(" (")[0] for r in wording.status_rule_rows()
             if r[0] == wording.STATUS_PREDICTIONS and not r[1].startswith("Clause")]
    assert table == [wording.PREDICTIONS_WORDS[insights.PREDICTION_STATUS[k]]
                     for k in insights.PREDICTION_ROWS]


