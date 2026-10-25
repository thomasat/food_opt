"""Suggestions must not hinge on the last bit of a float (0.9.0, owner
2026-10-24). Windows rounds floats differently from the Mac, and before
this change a difference of 1e-15 in the scores handed to the model moved a
suggestion by 4-17 %: BoTorch solved all its restarts as one summed scipy
problem, so one restart's rounding moved every other's stopping point, and
the best of near-equal restarts was picked by the last bits of its value.

Each case below is one that flipped in the 2026-10-24 investigation. It is
asked twice in one process, the second time with every score the model
sees scaled by (1 + 1e-15 · a fixed pattern), and the two must agree to
0.1 % relative — the tolerance the golden tests allow on Windows."""
import math

import pytest
import torch

import food_bo
from food_bo import FoodOptimizer
from tests import test_ask_total_unchanged as total_cases
from tests import test_complete_results_golden as golden_cases

LAST_BIT = 1e-15


def _nudged(train_Y):
    """train_Y × (1 + LAST_BIT · a fixed pattern in [-1, 1])."""
    k = torch.arange(train_Y.numel(), dtype=torch.double)
    pattern = torch.sin(1.0 + 2.3 * k).reshape(train_Y.shape)
    return train_Y * (1.0 + LAST_BIT * pattern)


def _nudge_every_score(monkeypatch):
    """Every ask from here on, the cases' own setup asks included, models
    the nudged scores: the GP and the acquisition both see them."""
    build_gp, build_acqf = FoodOptimizer._build_gp, FoodOptimizer._build_acqf

    def gp(self, train_X_norm, train_Y, dim, input_tf):
        return build_gp(self, train_X_norm, _nudged(train_Y), dim, input_tf)

    def acqf(self, model, train_X_norm, train_Y):
        return build_acqf(self, model, train_X_norm, _nudged(train_Y))

    monkeypatch.setattr(FoodOptimizer, "_build_gp", gp)
    monkeypatch.setattr(FoodOptimizer, "_build_acqf", acqf)


def _band():
    return total_cases.band().ask(3)


def _round_at_default():
    return total_cases.round_at_default().ask(3)


def _robust():
    return golden_cases._project("stability_robust", rows=7, robust=True).ask(2)


CASES = {"band": _band, "round_at_default": _round_at_default, "robust": _robust}


@pytest.fixture(autouse=True)
def _isolate_cwd(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)


@pytest.mark.parametrize("case", sorted(CASES))
def test_a_last_bit_change_in_the_scores_moves_no_suggestion(case, tmp_path,
                                                            monkeypatch):
    base = CASES[case]()
    # A fresh folder: the same project name would reopen the saved first run.
    (tmp_path / "nudged").mkdir()
    monkeypatch.chdir(tmp_path / "nudged")
    _nudge_every_score(monkeypatch)
    nudged = CASES[case]()
    assert len(nudged) == len(base)
    for i, (want, got) in enumerate(zip(base, nudged)):
        assert set(got) == set(want), i
        for name in want:
            assert math.isclose(got[name], want[name], rel_tol=1e-3, abs_tol=1e-9), \
                (case, i, name, got[name], want[name])


# ---------------------------------------------------------------------------
# The rule that picks among restarts
# ---------------------------------------------------------------------------

def _restarts(*candidates):
    """Candidates as optimize_acqf(return_best_only=False) returns them."""
    return torch.tensor([[list(c)] for c in candidates], dtype=torch.double)


@pytest.mark.parametrize("best", [-0.1389, 77.37])
def test_restarts_within_the_tolerance_go_to_the_smallest_candidate(best):
    """qLogNEI's values are small and negative, qUCB's are on the score
    scale: the tolerance is relative, with a floor of 1 in absolute terms."""
    tol = food_bo.ACQ_TIE_RTOL * max(1.0, abs(best))
    cands = _restarts((0.6, 0.2), (0.4, 0.9), (0.1, 0.1))
    vals = torch.tensor([best, best - 0.5 * tol, best - 1.0], dtype=torch.double)
    # Restarts 0 and 1 are tied; 1 is the smaller. 2 is smaller still but
    # far behind, so it is never in the running.
    assert food_bo._pick_restart(cands, vals) == 1


def test_a_clear_winner_beyond_the_tolerance_always_wins():
    cands = _restarts((0.6, 0.2), (0.4, 0.9), (0.1, 0.1))
    vals = torch.tensor([1.0, 1.0 - 2 * food_bo.ACQ_TIE_RTOL, 0.0],
                        dtype=torch.double)
    assert food_bo._pick_restart(cands, vals) == 0


def test_candidates_are_compared_at_three_decimals():
    """Two restarts at the same point up to the optimiser's own noise are
    the same point: the next coordinate decides."""
    cands = _restarts((0.40004, 0.9), (0.40001, 0.1))
    vals = torch.tensor([1.0, 1.0], dtype=torch.double)
    assert food_bo.ACQ_TIE_DECIMALS == 3
    assert food_bo._pick_restart(cands, vals) == 1
