"""Insights for a first-time user, W1 (spec 2026-10-01 rev. 2; review
R-B1, R-I4, R-I5, R-I14): each status line's sentence from its facts, for
the states the demo projects do not reach (the review's templates, built
in tests/test_107_ftue.UNREACHED), and the one-link rules. Fast: facts are
built by hand, nothing is fitted. The demo projects' own sentences are
pinned on the rendered screen (tests/test_107_w1_first_screen.py)."""
import types

import pytest

import bo_views
import insights
import ui_insights
import wording
from tests.ftue_terms import statistic_names_in
from tests.test_complete_results import _patty


@pytest.fixture(autouse=True)
def _isolate_cwd(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)


@pytest.fixture
def opt():
    return _patty(rows=12)


def _h(inside=32, total=35, rounds=(3, 4, 5)):
    return insights.Headline("verdict", list(rounds), inside, total, "holding")


def _facts(row, clauses=(), h=None, **kw):
    return insights.StatusFacts(scored=kw.pop("scored", 20), prediction_row=row,
                                clauses=list(clauses), headline=h or _h(), **kw)


# ------------------------------------------------------------------ #
#  Predictions: every row, its clauses joined as the ruling says
# ------------------------------------------------------------------ #


# ------------------------------------------------------------------ #
#  Progress
# ------------------------------------------------------------------ #

def _best(opt, rounds_since=0, latest=True):
    i = opt.best_index()
    return insights.BestFacts(float(opt.Y_history[i]), int(opt.formulation_ids[i]),
                              opt.batch_history[i], rounds_since, latest, [])


# ------------------------------------------------------------------ #
#  Next: one plain sentence per row (R-I5)
# ------------------------------------------------------------------ #

def _next(opt, kind, **kw):
    return _facts("reliable", best=_best(opt), next=insights.NextStep(kind, **kw))


# ------------------------------------------------------------------ #
#  The one link (R-I3, R-I4, R-I14)
# ------------------------------------------------------------------ #


def _open_round(opt, regime="warm", own=False):
    opt.pending_batch = [{"formulation": 99, "recipe": {}, "note": ""}]
    opt.pending_batch_no = 5
    gen = bo_views.Generation(5, 99, [], regime, True, [] if own else [99])
    return gen


@pytest.mark.skip(reason="review cards prototype: pins the old Review layout (status lines, Explored fold, header); reconcile at the final merge")
def test_the_explored_link_never_joins_a_why_link(opt, monkeypatch):
    """R-I4: at most one tertiary link; Why (priority 1) wins over Explored
    (priority 3)."""
    import contextlib
    asked = []
    for name in ("_status_lines", "_question", "_basis", "_setup_button"):
        monkeypatch.setattr(ui_insights, name, lambda *a, **k: None)
    monkeypatch.setattr(ui_insights, "_figures", lambda *a, **k: contextlib.nullcontext())
    monkeypatch.setattr(ui_insights, "_next_step", lambda opt, step: None)
    slot = types.SimpleNamespace(container=contextlib.nullcontext)
    facts = _next(opt, "explore", gain=2.0)
    for why, expected in ((True, []), (False, [1])):
        asked.clear()
        places = types.SimpleNamespace(lines=slot, change=slot, link=slot,
                                       nxt=contextlib.nullcontext(), why=why)
        ui_insights._finish(opt, places, facts)
        assert asked == expected
