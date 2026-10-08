"""0.8.0 speed, Plan A Task 5: what to fit, in the order the screens need
it, sent to the workers as one Job per project state (spec 2026-09-30
§10.1, §10.3; review I15)."""
import pytest

import insights
import insights_jobs
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


def _no_fits_here(monkeypatch):
    monkeypatch.setattr(insights, "_fit_gp", lambda *a: pytest.fail("fitted in the app's process"))


def test_the_plan_puts_the_overall_score_first_then_the_verdicts_rounds():
    opt = _burger(rounds=6, noise=0.4)
    specs = insights_jobs.plan(opt)
    order = [(s.stage, s.name) for s in specs]
    checked = [r for r, first in insights.round_order(opt) if first >= insights.MIN_ROWS]
    window = insights_jobs.window_rounds(opt)
    score = [(insights_jobs.NOW, insights.SCORE)] + [(r, insights.SCORE) for r in reversed(checked)]
    assert order[:len(score)] == score
    rest = order[len(score):]
    head = [(r, n) for r in reversed(window) for n in ("Firmness", "Juiciness")]
    assert rest[:len(head)] == head
    # Then the Rounds before the window, which the window's widening reads
    # (2026-10-07), then the current measurement models.
    before = [(r, n) for r in reversed(checked) if r < min(window)
              for n in ("Firmness", "Juiciness")]
    assert before
    assert rest[len(head):len(head) + len(before)] == before
    assert rest[len(head) + len(before):len(head) + len(before) + 2] == [
        (insights_jobs.NOW, "Firmness"), (insights_jobs.NOW, "Juiciness")]
    assert len(set(order)) == len(order)


def test_the_planned_window_is_the_one_the_verdict_reads():
    for opt in (_burger(rounds=6, noise=0.4), _burger(rounds=3, per_round=2),
                _burger(rounds=8, per_round=2, noise=0.2)):
        assert insights_jobs.window_rounds(opt) == insights.headline(
            opt, insights.check_rounds(opt)).rounds


def test_a_spec_with_too_few_rows_is_left_out():
    opt = _burger(rounds=3, per_round=2)       # Round 3 is checked on 4 rows: too few
    assert all(len(s.rows) >= insights.MIN_ROWS for s in insights_jobs.plan(opt))
    assert all(s.stage in (insights_jobs.NOW,) for s in insights_jobs.plan(opt))


def test_each_part_of_the_screen_waits_for_its_own_specs():
    opt = _burger(rounds=6, noise=0.4)
    specs = insights_jobs.plan(opt)
    window = set(insights_jobs.window_rounds(opt))
    part = lambda stage: {(s.stage, s.name) for s in insights_jobs.stage_specs(opt, specs, stage)}
    score = {(s.stage, s.name) for s in specs if s.name == insights.SCORE}
    # The verdict's Rounds and every Round before them: the checks their
    # widening reads (2026-10-07).
    in_window = {(s.stage, s.name) for s in specs
                 if s.name != insights.SCORE and s.stage != insights_jobs.NOW
                 and s.stage <= max(window)}
    assert {r for r, n in in_window} - window
    assert part("column") == score
    assert part("trust") == (score - {(insights_jobs.NOW, insights.SCORE)}) | in_window
    assert part("stand") == score | in_window
    assert part("all") == {(s.stage, s.name) for s in specs}


def test_a_job_fills_the_memo_and_the_screens_then_fit_nothing(pool, monkeypatch):
    opt = _burger(rounds=5, noise=0.4)
    inline = {}
    with insights.memo_in_use(inline):
        want_checks = insights.check_rounds(opt)
        want = insights.fit_all(opt)
    memo = {}
    job = insights_jobs.Job(opt, pool, memo)
    job.wait(memo)
    assert set(memo) == set(inline)
    for key in memo:
        assert memo[key]["widen"] == pytest.approx(inline[key]["widen"], abs=1e-12)
    _no_fits_here(monkeypatch)
    with insights.memo_in_use(memo):
        got_checks = insights.check_rounds(opt)
        got = insights.fit_all(opt)
    for a, b in zip(got_checks, want_checks):
        assert (a.low, a.high, a.inside) == (pytest.approx(b.low, abs=1e-9),
                                             pytest.approx(b.high, abs=1e-9), b.inside)
    assert got[insights.SCORE].widen == pytest.approx(want[insights.SCORE].widen)


def test_what_is_already_in_the_memo_is_not_sent(pool):
    opt = _burger(rounds=4, noise=0.4)
    memo = {}
    with insights.memo_in_use(memo):
        insights.fit_all(opt)
    job = insights_jobs.Job(opt, pool, memo)
    assert not set(job.futures) & set(memo)


def test_one_job_per_project_state_and_a_new_state_shares_what_is_in_flight(pool):
    opt = _burger(rounds=5, noise=0.4)
    memo = {}
    first = insights_jobs.job_for(opt, pool, memo)
    assert insights_jobs.job_for(opt, pool, memo) is first
    opt.results_history[-1]["Juiciness"] += 0.3      # one result changed: a new state
    second = insights_jobs.job_for(opt, pool, memo)
    assert second is not first
    shared = set(first.futures) & set(second.futures)
    assert shared and all(first.futures[k] is second.futures[k] for k in shared)
    second.wait(memo)


def test_a_broken_pool_fails_the_job_and_the_rest_is_fitted_here(monkeypatch, tmp_path):
    dies = tmp_path / "dies_worker.py"
    dies.write_text("import json, sys\nprint(json.dumps({'ready': True}), flush=True)\n"
                    "sys.stdin.readline()\nsys.exit(1)\n")
    monkeypatch.setattr(insights_jobs, "WORKER", str(dies))
    broken = insights_jobs.Pool(1)
    opt = _burger(rounds=4, noise=0.4)
    memo = {}
    job = insights_jobs.Job(opt, broken, memo)
    job.wait(memo)
    assert job.failed and broken.broken
    specs = insights_jobs.plan(opt)
    steps = []
    insights_jobs.compute_inline(opt, memo, specs, step=lambda k, n: steps.append((k, n)))
    assert insights_jobs.ready(specs, memo)
    assert steps and steps[-1][0] == steps[-1][1]


def test_settled_tells_a_complete_window_from_a_short_one():
    opt = _burger(rounds=6, noise=0.4)
    checks = insights.check_rounds(opt)
    assert insights_jobs.settled(opt, checks)
    last = insights_jobs.window_rounds(opt)[-1]
    short = [c for c in checks if not (c.round == last and c.name == "Firmness")]
    assert not insights_jobs.settled(opt, short)


def test_rounds_done_counts_rounds_whose_checks_are_all_in():
    opt = _burger(rounds=5, noise=0.4)
    specs = insights_jobs.plan(opt)
    rounds = {s.stage for s in specs if s.stage != insights_jobs.NOW}
    assert insights_jobs.rounds_done(opt, specs, {}) == (0, len(rounds))
    memo = {s.key: {} for s in specs}
    assert insights_jobs.rounds_done(opt, specs, memo) == (len(rounds), len(rounds))


# ------------------------------------------------------------------ #
#  Plan A review B I-1, M-1: a Job covers what every caller's memo lacks
# ------------------------------------------------------------------ #

def test_a_shared_job_fits_what_a_later_callers_memo_lacks(pool):
    """Session 1 already had some models, so its Job left them out; a
    second memo without them (a second tab, or the project reopened from
    disk) gets the same Job and must still end up with every model."""
    opt = _burger(rounds=4, noise=0.4)
    specs = insights_jobs.plan(opt)
    first = {}
    insights_jobs.compute_inline(opt, first, specs[:2])
    job = insights_jobs.job_for(opt, pool, first)
    second = {}
    assert insights_jobs.job_for(opt, pool, second) is job
    job.wait(second)
    assert insights_jobs.ready(specs, second)
    job.wait(first)
    assert insights_jobs.ready(specs, first)


def test_two_tabs_polling_one_job_never_raise(pool):
    import threading
    opt = _burger(rounds=5, noise=0.4)
    memos, errors = [{}, {}], []
    job = insights_jobs.job_for(opt, pool, memos[0])

    def poll(memo):
        try:
            while not job.poll(memo):
                pass
        except Exception as exc:
            errors.append(exc)
    threads = [threading.Thread(target=poll, args=(m,)) for m in memos]
    for t in threads:
        t.start()
    for t in threads:
        t.join(300)
    assert errors == []
    assert all(insights_jobs.ready(insights_jobs.plan(opt), m) for m in memos)


def test_a_failed_fold_is_skipped_as_in_process():
    """Review B M-7."""
    entry = insights_jobs._entry([{"state": {"a": ["float64", 1.0]}},
                                  {"z": [0.5, -1.0]}, {"error": "no fit"}, {"z": [2.0]}])
    assert entry == {"state": {"a": ["float64", 1.0]},
                     "widen": insights.widen_from([0.5, -1.0, 2.0])}
    assert insights_jobs._entry([{"error": "no fit"}, {"z": [1.0]}]) == {"error": "no fit"}


class _Never:
    """A pool whose work never starts, so nothing finishes on its own."""

    def submit(self, priority, request):
        import concurrent.futures as cf
        return cf.Future()


def test_a_new_state_cancels_what_only_the_old_one_needed(monkeypatch):
    """Review B M-7: job_for gives up an older Job's specs that the new
    state does not need, and shares the rest."""
    monkeypatch.setattr(insights_jobs, "_JOBS", {})
    monkeypatch.setattr(insights_jobs, "_INFLIGHT", {})
    opt = _burger(rounds=5, noise=0.4)
    pool = _Never()
    old = insights_jobs.job_for(opt, pool, {})
    opt.results_history[-1]["Juiciness"] += 0.3
    new = insights_jobs.job_for(opt, pool, {})
    gone = set(old.futures) - set(new.futures)
    assert gone and all(f.cancelled() for k in gone for f in old.futures[k])
    kept = set(old.futures) & set(new.futures)
    assert kept and not any(f.cancelled() for k in kept for f in new.futures[k])
