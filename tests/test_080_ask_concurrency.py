"""0.8.0 speed, Plan A Task 5 (review B2, M20): the suggestions are exactly
the fixture's while Insights' fits run in the workers. The fixture is
tests/fixtures/ask_unchanged.json, never regenerated."""
import pytest

import insights
import insights_jobs
from tests.test_ask_unchanged import CASES, fixture, run, same
from tests.test_insights import _burger


@pytest.fixture(autouse=True)
def _isolate_cwd(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)


@pytest.fixture(scope="module")
def pool():
    p = insights_jobs.Pool(2)
    assert p.wait_ready(120)
    yield p
    p.close()


@pytest.mark.parametrize("case", sorted(CASES))
def test_suggestions_are_bit_identical_while_insights_fits_run(case, pool):
    busy = _burger(rounds=8, noise=0.4)
    memo = {}
    job = insights_jobs.Job(busy, pool, memo)
    opt = CASES[case]()
    own = insights_jobs.Job(opt, pool, memo)
    assert job.futures, "nothing was in flight"
    got = run(opt)
    assert not job.poll(memo), "the fits finished before ask() ran: make the busy project bigger"
    same(got, fixture()[case])
    job.wait(memo)
    own.wait(memo)


@pytest.mark.parametrize("case", sorted(CASES))
def test_suggestions_are_bit_identical_after_models_are_rebuilt_here(case, pool):
    opt = CASES[case]()
    memo = {}
    insights_jobs.Job(opt, pool, memo).wait(memo)
    with insights.memo_in_use(memo):
        insights.fit_all(opt)
        insights.check_rounds(opt)
    same(run(opt), fixture()[case])
