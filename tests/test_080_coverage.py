"""0.8.0 speed, Plan A Task 7 (spec 2026-09-30 §8, §10; review I1, M20):
Round-ahead coverage on 48 simulated projects built with ask(), measured
before the fact. The fast tests read tests/fixtures/coverage_checks.json
(see tests/coverage_sim.py); FOODOPT_SLOW=1 works all 48 out again."""
import os

import pytest

import insights
from tests import coverage_sim as sim

SLOW = os.environ.get("FOODOPT_SLOW") == "1"


@pytest.fixture(autouse=True)
def _isolate_cwd(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)


@pytest.fixture(scope="module")
def entries():
    return sim.fixture()["projects"]


def test_the_fixture_was_made_by_this_fit(entries):
    """The fixture carries this MODEL_SALT: a salt bump without
    `python -m tests.coverage_sim` fails here. A change to the fit without
    a salt bump is caught only when it moves the smallest project's checks
    (test_one_project_worked_out_again_matches_the_fixture) or any
    project's (the slow rerun)."""
    assert sim.fixture()["salt"] == insights.MODEL_SALT
    assert len(entries) == len(sim.projects()) == 48


# The fixture and coverage() are deterministic, so the counts are pinned
# exactly (review A I-1): a widening that looks at a Round's own checks or
# later ones (hindsight, which I1 forbids) only raises coverage, and would
# pass every lower bound below. A fixture regenerated with a MODEL_SALT
# bump updates these on purpose.
#
# Re-pinned by ftue fix A (statistician's cold read I9, MODEL_SALT
# "0.8.0-speed-1" -> "0.8.0-ftue-fa-folds-1"): the cross-validation folds
# behind the first widening now hold out a Round, and a Repeat group,
# whole (before, row k went to fold k % 5, so a held-out row kept its
# Round's near neighbours and its Repeat in training). Old -> new:
#   measured          4271 -> 4262 of 4702  (90.8 % -> 90.6 %)
#   score_raw         1406 -> 1425 of 1584  (88.8 % -> 90.0 %)
#   score             1463 -> 1465 of 1584  (92.4 % -> 92.5 %)
#   window_measured   1394 -> 1415 of 1497  (93.1 % -> 94.5 %)
#   window_score_raw   468 ->  472 of 504   (92.9 % -> 93.7 %)
#   window_score       484 ->  487 of 504   (96.0 % -> 96.6 %)
PINNED = {"measured": (4262, 4702), "score_raw": (1425, 1584), "score": (1465, 1584),
          "window_measured": (1415, 1497), "window_score_raw": (472, 504),
          "window_score": (487, 504)}


def test_coverage_is_exactly_the_before_the_fact_figure(entries):
    cov = sim.coverage(entries)
    assert {k: (round(share * n), n) for k, (share, n) in cov.items()} == PINNED


def test_coverage_at_the_verdict_window_is_at_least_92_percent(entries):
    cov = sim.coverage(entries)
    assert cov["window_measured"][0] >= 0.92, cov
    assert cov["window_score"][0] >= 0.92, cov


def test_the_nested_widening_lifts_the_overall_score_and_leaves_measurements(entries):
    cov = sim.coverage(entries)
    assert cov["score"][0] > cov["score_raw"][0], cov
    assert cov["score"][0] >= 0.91, cov
    assert cov["measured"][0] >= 0.90, cov


def test_widening_never_turns_an_inside_check_outside(entries):
    for entry in entries:
        raw = sim.checks_of(entry)
        nested = insights.nest_score_widen(sim.stub(entry), raw)
        assert all(n.inside for c, n in zip(raw, nested) if c.inside)


def _same(got, want):
    assert [(c.round, c.name, c.formulation) for c in got] == [
        (c.round, c.name, c.formulation) for c in want]
    for c, w in zip(got, want):
        assert (c.low, c.high, c.raw_mean, c.sd) == pytest.approx(
            (w.low, w.high, w.raw_mean, w.sd), abs=1e-6)
        assert c.inside == w.inside


def test_one_project_worked_out_again_matches_the_fixture(entries):
    entry = min(entries, key=lambda e: len(e["checks"]))
    [path] = [p for p in sim.projects() if p.stem == entry["project"]]
    opt, _ = sim.load(path)
    _same(sim.raw_checks(opt), sim.checks_of(entry))


@pytest.mark.skipif(not SLOW, reason="slow: set FOODOPT_SLOW=1 (about a minute on 8 workers)")
def test_every_project_worked_out_again_matches_the_fixture(entries):
    fresh = sim.work_out(sim.projects())
    for got, want in zip(fresh, entries):
        assert got["project"] == want["project"]
        _same(sim.checks_of(got), sim.checks_of(want))
    assert sim.coverage(fresh) == sim.coverage(entries)
