"""0.8.0 speed, Plan A Task 11 (spec 2026-09-30 §8, §10): the Insights
work at 60 formulations x 4 measurements (the owner's bench shape), with
4 workers, inside generous bounds — the first answers (the verdict, the
predicted best and the next step) under 5 s, the whole tab under 10 s,
a reopened project (the next Round's
expected gain included) under 1 s, and a saved Round under 6 s. The pool's own
start (about 2 s, once per app launch) is not counted. On the owner's Mac
(M4 Pro) the figures are about 3.9, 4.3, 0.01 and 1.0 s. The first answers
were 2.4 s under a 3 s bound until 633c531 (2026-10-07, "one 95 % range per
forecast, whatever is cached") made "trust" and "stand" wait for every check
their widening reads, a cost that commit states (about 3.9 s against 2.4 s;
whole tab unchanged); the bound moved to 5 s with it.
FOODOPT_SKIP_TIMING=1 skips it; FOODOPT_TIMING_WORKERS sets the pool."""
import os
import time

import pytest
import torch

import bo_views
import insights
import insights_cache
import insights_jobs
from tests import speed_projects

pytestmark = pytest.mark.skipif(os.environ.get("FOODOPT_SKIP_TIMING") == "1",
                                reason="FOODOPT_SKIP_TIMING=1")


@pytest.fixture(autouse=True)
def _isolate(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("FOODOPT_INSIGHTS_CACHE", "1")
    monkeypatch.setattr(insights_jobs, "_JOBS", {})
    monkeypatch.setattr(insights_jobs, "_INFLIGHT", {})


@pytest.fixture(scope="module")
def pool():
    p = insights_jobs.Pool(int(os.environ.get("FOODOPT_TIMING_WORKERS", "4")))
    assert p.wait_ready(120)
    yield p
    p.close()


def _wait(opt, memo, stage, job):
    specs = insights_jobs.stage_specs(opt, insights_jobs.plan(opt), stage)
    while not insights_jobs.ready(specs, memo):
        job.poll(memo)
        time.sleep(0.01)
    return specs


def _score_parts(opt, memo, specs):
    with insights.memo_in_use(memo):
        checks = insights.check_rounds(opt, only=insights_jobs.only_pairs(specs))
        score = insights.fit(opt, insights.SCORE, list(range(len(opt.Y_history))))
    return score, checks


def _best_early(opt, memo, job):
    """What ui_insights.warm_best does once the Overall score's models are
    in, while the workers fit the rest: the predicted best."""
    specs = _wait(opt, memo, "column", job)
    score, checks = _score_parts(opt, memo, specs)
    return insights.predicted_best(opt, {insights.SCORE: score}, checks)


def _answers(opt, memo, specs):
    """What answers 1-3 read (what _render_filling works out), the
    predicted best already there."""
    score, checks = _score_parts(opt, memo, specs)
    assert insights_jobs.settled(opt, checks)
    insights.headline(opt, checks)
    insights.noise_yardstick(opt, checks)
    return checks


def _whole(opt, memo, rounds):
    with insights.memo_in_use(memo):
        fitted = insights.fit_all(opt)
        checks = insights.check_rounds(opt, cache=rounds)
    return fitted, checks


def test_sixty_by_four(pool):
    opt = speed_projects.bench(60, 4)
    torch.manual_seed(0)
    memo, rounds = {}, {}
    t0 = time.perf_counter()
    job = insights_jobs.Job(opt, pool, memo)
    best = _best_early(opt, memo, job)
    specs = _wait(opt, memo, "stand", job)
    _answers(opt, memo, specs)
    first = time.perf_counter() - t0
    _wait(opt, memo, "all", job)
    fitted, checks = _whole(opt, memo, rounds)
    whole = time.perf_counter() - t0

    # The next Round's expected gain is saved with them (spec
    # 2026-10-01 C1): a reopen reads it instead of fitting and searching.
    gain = bo_views.next_gain_for(opt)
    insights_cache.save(opt.project_name, insights_cache.Saved(
        memo, rounds, (insights.setup_key(opt), best), (bo_views.next_gain_key(opt), gain)))
    bo_views._CACHE.clear()
    t1 = time.perf_counter()
    saved = insights_cache.load(opt.project_name)
    again, again_checks = _whole(opt, saved.memo, saved.rounds)
    again_gain = bo_views.saved_next_gain(opt, saved.gain)
    reopen = time.perf_counter() - t1
    assert again_checks == checks and again_gain == gain

    # A Round of six results saved: its check is the model already fitted
    # for the Round before it; the current models are fitted anew.
    g = torch.Generator().manual_seed(99)
    for no in range(61, 67):
        speed_projects.tell(opt, no, g, 4)
    t2 = time.perf_counter()
    job = insights_jobs.Job(opt, pool, memo)
    _best_early(opt, memo, job)
    _wait(opt, memo, "all", job)
    _whole(opt, memo, rounds)
    after_save = time.perf_counter() - t2

    print(f"\n60x4, {pool.n} workers: first answers {first:.1f} s, whole tab {whole:.1f} s, "
          f"reopen {reopen:.2f} s, after Save results {after_save:.1f} s")
    assert first < 5.0
    assert whole < 10.0
    assert reopen < 1.0
    assert after_save < 6.0
