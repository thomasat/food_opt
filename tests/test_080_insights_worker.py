"""0.8.0 speed, Plan A Task 4: Insights' fits in worker processes (spec
2026-09-30 §10.1, §10.5). A worker is `python insights_worker.py`, one
torch thread, JSON lines in and out; its answers equal the fits made in
the app's process; a pool that cannot start or loses a worker says so, and
the caller fits in its own process."""
import json
import os
import subprocess
import sys

import pytest

import insights
import insights_jobs
from tests.test_insights import _burger


@pytest.fixture(autouse=True)
def _isolate_cwd(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)


def _requests(opt, name):
    X, Y = insights.xy(opt, name, list(range(len(opt.Y_history))))
    main = {"id": 1, "kind": "main", "cfg": None, "X": X.tolist(), "Y": Y.tolist()}
    kept, held = insights.folds(len(opt.Y_history))[0]
    fold = {"id": 2, "kind": "fold", "cfg": None, "X": X[kept].tolist(), "Y": Y[kept].tolist(),
            "Xt": X[held].tolist(), "Yt": Y[held].tolist()}
    return main, fold, (X, Y, kept, held)


def _talk(lines, cwd):
    """Run a worker by path from `cwd` and feed it `lines`."""
    proc = subprocess.run([sys.executable, insights_jobs.WORKER], cwd=cwd,
                          input="".join(json.dumps(x) + "\n" for x in lines),
                          capture_output=True, text=True, timeout=120)
    return [json.loads(line) for line in proc.stdout.splitlines()]


def test_a_worker_answers_what_this_process_fits(tmp_path):
    opt = _burger(rounds=3, noise=0.3)
    main, fold, (X, Y, kept, held) = _requests(opt, "Firmness")
    hello, a, b = _talk([main, fold], cwd=tmp_path)
    assert hello == {"ready": True}
    here = insights.hyper_state(insights._fit_gp(opt, "Firmness", list(range(len(opt.Y_history)))))
    assert a == {"id": 1, "state": json.loads(json.dumps(here))}
    z = insights.fold_z(insights.builder(None), X[kept], Y[kept], X[held], Y[held])
    assert b["id"] == 2 and b["z"] == pytest.approx(z, abs=1e-12)


def test_a_failed_fit_answers_its_error(tmp_path):
    bad = {"id": 7, "kind": "main", "cfg": None, "X": [[0.5]], "Y": []}
    hello, answer = _talk([bad], cwd=tmp_path)
    assert answer["id"] == 7 and answer["error"]


def test_a_worker_runs_from_any_folder_with_one_torch_thread(tmp_path):
    hello, ping = _talk([{"id": 3, "kind": "ping"}], cwd=tmp_path)
    assert ping["id"] == 3 and ping["threads"] == 1 and ping["pid"] != os.getpid()


def test_the_pool_answers_lowest_priority_first():
    opt = _burger(rounds=3, noise=0.3)
    main, fold, _ = _requests(opt, "Juiciness")
    pool = insights_jobs.Pool(1)
    try:
        assert pool.wait_ready(120)
        # One worker: hold it with a first request, then queue two more.
        first = pool.submit(0, main)
        late = pool.submit(9, dict(fold, id="late"))
        soon = pool.submit(1, dict(fold, id="soon"))
        order = []
        for fut in (first, late, soon):
            fut.add_done_callback(lambda f: order.append(f.result()["id"]))
        late.result(120)
        soon.result(120)
        assert order.index("soon") < order.index("late")
    finally:
        pool.close()


def test_with_workers_off_there_is_no_pool(monkeypatch):
    monkeypatch.setattr(insights_jobs, "_POOL", None)
    monkeypatch.setenv("FOODOPT_INSIGHTS_WORKERS", "0")
    assert insights_jobs.pool() is None


def test_workers_on_follows_the_setting_and_a_failure(monkeypatch):
    monkeypatch.setattr(insights_jobs, "_GAVE_UP", False)
    monkeypatch.setenv("FOODOPT_INSIGHTS_WORKERS", "2")
    assert insights_jobs.workers_on()
    monkeypatch.setattr(insights_jobs, "_GAVE_UP", True)
    assert not insights_jobs.workers_on()
    monkeypatch.setattr(insights_jobs, "_GAVE_UP", False)
    monkeypatch.setenv("FOODOPT_INSIGHTS_WORKERS", "0")
    assert not insights_jobs.workers_on()


def test_a_pool_that_broke_is_not_started_again(monkeypatch, tmp_path):
    bad = tmp_path / "broken_worker.py"
    bad.write_text("raise SystemExit(3)\n")
    monkeypatch.setattr(insights_jobs, "WORKER", str(bad))
    monkeypatch.setattr(insights_jobs, "_POOL", None)
    monkeypatch.setattr(insights_jobs, "_GAVE_UP", False)
    monkeypatch.setenv("FOODOPT_INSIGHTS_WORKERS", "1")
    first = insights_jobs.pool()
    assert not first.wait_ready(60)
    assert insights_jobs.pool() is None
    assert insights_jobs.pool() is None


def test_a_worker_that_hangs_on_start_breaks_the_pool(monkeypatch, tmp_path):
    hangs = tmp_path / "hangs_worker.py"
    hangs.write_text("import time\ntime.sleep(600)\n")
    monkeypatch.setattr(insights_jobs, "WORKER", str(hangs))
    monkeypatch.setattr(insights_jobs, "START_SECS", 0.5)
    monkeypatch.setattr(insights_jobs, "_POOL", None)
    monkeypatch.setattr(insights_jobs, "_GAVE_UP", False)
    monkeypatch.setenv("FOODOPT_INSIGHTS_WORKERS", "1")
    pool = insights_jobs.pool()
    waiting = pool.submit(0, {"id": 1})
    import time
    time.sleep(1.0)
    assert insights_jobs.pool() is None
    with pytest.raises(RuntimeError):
        waiting.result(5)


def test_a_worker_that_will_not_start_breaks_the_pool(monkeypatch, tmp_path):
    bad = tmp_path / "broken_worker.py"
    bad.write_text("raise SystemExit(3)\n")
    monkeypatch.setattr(insights_jobs, "WORKER", str(bad))
    pool = insights_jobs.Pool(2)
    assert not pool.wait_ready(60)
    assert pool.broken
    with pytest.raises(RuntimeError):
        pool.submit(0, {"id": 1}).result(5)


def test_a_worker_that_dies_fails_what_was_waiting(monkeypatch, tmp_path):
    dies = tmp_path / "dies_worker.py"
    dies.write_text("import json, sys\nprint(json.dumps({'ready': True}), flush=True)\n"
                    "sys.stdin.readline()\nsys.exit(1)\n")
    monkeypatch.setattr(insights_jobs, "WORKER", str(dies))
    pool = insights_jobs.Pool(1)
    assert pool.wait_ready(60)
    first, second = pool.submit(0, {"id": 1}), pool.submit(1, {"id": 2})
    for fut in (first, second):
        with pytest.raises(Exception):
            fut.result(30)
    assert pool.broken


def test_worker_count_follows_the_setting(monkeypatch):
    monkeypatch.setenv("FOODOPT_INSIGHTS_WORKERS", "3")
    assert insights_jobs.worker_count() == 3
    monkeypatch.setenv("FOODOPT_INSIGHTS_WORKERS", "nonsense")
    assert insights_jobs.worker_count() == 0
    monkeypatch.delenv("FOODOPT_INSIGHTS_WORKERS")
    assert 1 <= insights_jobs.worker_count() <= 6
    monkeypatch.setattr(insights_jobs.os, "cpu_count", lambda: 8)
    monkeypatch.setattr(insights_jobs.os, "sysconf",
                        lambda k: {"SC_PHYS_PAGES": 2 * 2 ** 20, "SC_PAGE_SIZE": 4096}[k])
    assert insights_jobs.worker_count() == 2          # an 8 GB laptop
    big = lambda k: {"SC_PHYS_PAGES": 16 * 2 ** 20, "SC_PAGE_SIZE": 4096}[k]   # 64 GB
    monkeypatch.setattr(insights_jobs.os, "sysconf", big)
    monkeypatch.setattr(insights_jobs.os, "cpu_count", lambda: 4)
    assert insights_jobs.worker_count() == 2          # cores - 2 (review B M-7)
    monkeypatch.setattr(insights_jobs.os, "cpu_count", lambda: 16)
    assert insights_jobs.worker_count() == 6          # never more than 6


# ------------------------------------------------------------------ #
#  Plan A review B I-2, M-8: an idle pool stops by itself; a closed pool
#  refuses work and is replaced, never given up on
# ------------------------------------------------------------------ #

def test_an_idle_pool_stops_by_itself(monkeypatch):
    monkeypatch.setattr(insights_jobs, "IDLE_SECS", 0.5)
    monkeypatch.setattr(insights_jobs, "_POOL", None)
    monkeypatch.setattr(insights_jobs, "_GAVE_UP", False)
    monkeypatch.setenv("FOODOPT_INSIGHTS_WORKERS", "2")
    import time
    pool = insights_jobs.pool()
    assert pool.wait_ready(120)
    procs = list(pool._procs)
    end = time.monotonic() + 30
    while any(p.poll() is None for p in procs) and time.monotonic() < end:
        time.sleep(0.1)                      # no call to pool() here
    assert all(p.poll() is not None for p in procs)
    assert insights_jobs._POOL is None
    assert pool.closed and not pool.broken
    again = insights_jobs.pool()             # the next fit starts a new one
    assert again is not None and again is not pool and not insights_jobs._GAVE_UP
    again.close()


def test_a_busy_pool_is_not_stopped_and_pool_hands_out_the_same_one(monkeypatch):
    monkeypatch.setattr(insights_jobs, "IDLE_SECS", 0.5)
    monkeypatch.setattr(insights_jobs, "_POOL", None)
    monkeypatch.setattr(insights_jobs, "_GAVE_UP", False)
    monkeypatch.setenv("FOODOPT_INSIGHTS_WORKERS", "1")
    import time
    opt = _burger(rounds=3, noise=0.3)
    main, _, _ = _requests(opt, "Juiciness")
    pool = insights_jobs.pool()
    assert pool.wait_ready(120)
    for _ in range(4):                       # used every 0.3 s for over a second
        assert pool.submit(0, main).result(120)["id"]
        assert insights_jobs.pool() is pool
        time.sleep(0.3)
    assert not pool.closed and all(p.poll() is None for p in pool._procs)
    pool.close()


def test_a_closed_pool_refuses_work():
    pool = insights_jobs.Pool(1)
    pool.close()
    with pytest.raises(RuntimeError):
        pool.submit(0, {"id": 1}).result(5)


def test_a_worker_that_hangs_mid_fit_is_stopped_and_the_pool_gives_way(monkeypatch, tmp_path):
    """Re-review N-1: a request running past REQUEST_SECS kills its worker;
    what was waiting fails, the pool breaks, and the screens fall back to
    fitting here behind their own indicator."""
    hangs = tmp_path / "hangs_mid_fit.py"
    hangs.write_text("import json, sys, time\nprint(json.dumps({'ready': True}), flush=True)\n"
                     "sys.stdin.readline()\ntime.sleep(600)\n")
    monkeypatch.setattr(insights_jobs, "WORKER", str(hangs))
    monkeypatch.setattr(insights_jobs, "REQUEST_SECS", 0.5)
    import time
    pool = insights_jobs.Pool(1)
    assert pool.wait_ready(60)
    [proc] = pool._procs
    fut = pool.submit(0, {"id": 1})
    with pytest.raises(Exception):
        fut.result(30)
    end = time.monotonic() + 10
    while proc.poll() is None and time.monotonic() < end:
        time.sleep(0.05)
    assert proc.poll() is not None
    assert pool.broken


def test_a_slow_fit_within_the_limit_is_left_alone(monkeypatch):
    monkeypatch.setattr(insights_jobs, "REQUEST_SECS", 60)
    opt = _burger(rounds=3, noise=0.3)
    main, _, _ = _requests(opt, "Juiciness")
    pool = insights_jobs.Pool(1)
    try:
        assert pool.wait_ready(120)
        assert "state" in pool.submit(0, main).result(120)
        assert not pool.broken
    finally:
        pool.close()
