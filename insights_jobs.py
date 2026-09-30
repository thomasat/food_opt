"""Insights' model fits, planned in the order the screens need them and
fitted in worker processes (spec 2026-09-30 §10). No Streamlit here.

Every model Insights and Make a round show is one *spec*: an output and its
training rows — all rows for the current models ("now"), or the rows
recorded before a Round for that Round's check. A spec costs one main fit
plus one fit per fold. plan() lists them, most needed first; a Job sends
the ones not yet in the caller's memo to the Pool, and poll() moves each
finished one into the memo, where insights.fit() finds it and only
conditions (insights.memo_in_use).

The Pool is a few `insights_worker.py` processes (one torch thread each),
started once per app process. Nothing crosses to them but numbers as JSON,
and each has its own torch random state, so no fit can interleave with
ask() in the app's process. When no worker can start (or FOODOPT_INSIGHTS_WORKERS
is 0) pool() is None and the screens fit in the app's own process, as
before, with compute_inline()."""
import atexit
import concurrent.futures as cf
import dataclasses
import itertools
import json
import os
import queue
import subprocess
import sys
import threading
import time

import insights

WORKER = os.path.join(os.path.dirname(os.path.abspath(__file__)), "insights_worker.py")
# Workers stop after this long with nothing to do; the next fit starts them again.
IDLE_SECS = 600
# How long pool() waits for a worker's imports before giving up on workers.
START_SECS = 60


def worker_count():
    """FOODOPT_INSIGHTS_WORKERS when set (0: none), else up to 4 — about
    400 MB each — leaving two cores for the app itself."""
    env = os.environ.get("FOODOPT_INSIGHTS_WORKERS")
    if env is not None:
        try:
            return max(0, int(env))
        except ValueError:
            return 0
    return max(1, min(4, (os.cpu_count() or 2) - 2))


# ------------------------------------------------------------------ #
#  The pool
# ------------------------------------------------------------------ #

class Pool:
    """`n` worker processes fed from one priority queue: the lowest
    priority number is sent first. If a worker stops, the pool is broken:
    every waiting request fails and the caller fits in its own process."""

    def __init__(self, n, python=None):
        self._queue = queue.PriorityQueue()
        self._seq = itertools.count()
        self._lock = threading.Lock()
        self._procs = []
        self._ready = threading.Event()
        self.broken = False
        self.last_used = time.monotonic()
        env = {**os.environ, "OMP_NUM_THREADS": "1", "PYTHONDONTWRITEBYTECODE": "1"}
        for _ in range(n):
            proc = subprocess.Popen(
                [python or sys.executable, WORKER], stdin=subprocess.PIPE,
                stdout=subprocess.PIPE, stderr=subprocess.DEVNULL, text=True, bufsize=1,
                env=env)
            self._procs.append(proc)
            threading.Thread(target=self._serve, args=(proc,), daemon=True).start()

    def wait_ready(self, timeout):
        """True once a worker has finished its imports; False if none did
        in time or the pool broke."""
        return self._ready.wait(timeout) and not self.broken

    def submit(self, priority, request):
        """A Future of the worker's answer (a dict) to `request`."""
        fut = cf.Future()
        with self._lock:
            if self.broken:
                fut.set_exception(RuntimeError("the Insights workers stopped"))
                return fut
            self.last_used = time.monotonic()
            self._queue.put((priority, next(self._seq), request, fut))
        return fut

    def idle(self):
        return self._queue.empty() and time.monotonic() - self.last_used > IDLE_SECS

    def _serve(self, proc):
        try:
            if json.loads(proc.stdout.readline() or "{}").get("ready") is not True:
                raise RuntimeError("the Insights worker did not start")
            self._ready.set()
            while True:
                _, _, request, fut = self._queue.get()
                if request is None:
                    return
                if not fut.set_running_or_notify_cancel():
                    continue
                try:
                    proc.stdin.write(json.dumps(request) + "\n")
                    proc.stdin.flush()
                    line = proc.stdout.readline()
                    if not line:
                        raise RuntimeError("the Insights worker stopped")
                    fut.set_result(json.loads(line))
                except BaseException as exc:
                    fut.set_exception(exc)
                    raise
                self.last_used = time.monotonic()
        except Exception:
            self._break()

    def _break(self):
        with self._lock:
            self.broken = True
            self._ready.set()
            while True:
                try:
                    _, _, request, fut = self._queue.get_nowait()
                except queue.Empty:
                    break
                if fut is not None and fut.set_running_or_notify_cancel():
                    fut.set_exception(RuntimeError("the Insights workers stopped"))
        self.close()

    def close(self):
        for proc in self._procs:
            try:
                proc.kill()
            except Exception:
                pass
        for _ in self._procs:
            self._queue.put((-1, -1, None, None))


_POOL = None
_POOL_LOCK = threading.Lock()


def pool():
    """The app process's Pool, started on first use; None when workers are
    off (worker_count() is 0) or none could start (the caller then fits in
    its own process). A pool idle for IDLE_SECS is closed and started
    again when next needed."""
    global _POOL
    with _POOL_LOCK:
        if _POOL is not None and (_POOL.broken or _POOL.idle()):
            _POOL.close()
            _POOL = None
            _INFLIGHT.clear()
        n = worker_count()
        if _POOL is None and n > 0:
            try:
                _POOL = Pool(n)
            except Exception:
                return None
        return _POOL


def pool_started(timeout=START_SECS):
    """pool(), once a worker is ready; None when there is none."""
    p = pool()
    return p if p is not None and p.wait_ready(timeout) else None


@atexit.register
def _close_pool():
    if _POOL is not None:
        _POOL.close()


# ------------------------------------------------------------------ #
#  What to fit, most needed first
# ------------------------------------------------------------------ #

NOW = "now"     # a spec of the current models (insights.fit_all), not a Round's check


@dataclasses.dataclass(frozen=True)
class Spec:
    stage: object           # NOW or the Round number whose check it makes
    name: str               # output: a measurement or insights.SCORE
    rows: tuple             # its training rows, measured ones only
    key: str                # insights.model_key


def _counts(opt):
    """[(Round, how many measurement checks it will have)] for every Round
    with any, in Round order, from the results alone: a measurement is
    checked in a Round when MIN_ROWS rows before it have it."""
    names = [o['name'] for o in opt.objectives]
    out = []
    for r, first in insights.round_order(opt):
        n = 0
        for name in names:
            if sum(insights.measured(opt, name, i) is not None
                   for i in range(first)) < insights.MIN_ROWS:
                continue
            n += sum(insights.measured(opt, name, i) is not None
                     for i, b in enumerate(opt.batch_history) if b == r)
        if n:
            out.append((r, n))
    return out


def window_rounds(opt):
    """The Rounds the verdict will read (insights.headline's rule: the last
    three with measurement checks, more until they hold 10), worked out
    before any model is fitted. Exact unless a fit fails; settled() tells."""
    counted = _counts(opt)
    out = []
    for r, n in reversed(counted):
        out.append(r)
        if len(out) >= 3 and sum(m for q, m in counted if q in out) >= 10:
            break
    return list(reversed(out))


def settled(opt, checks):
    """Whether `checks` hold every measurement check the verdict's Rounds
    were planned to have: then the verdict read from them is the one the
    whole set gives. False after a failed fit (wait for everything)."""
    expected = dict(_counts(opt))
    got = {}
    for c in checks:
        if c.name != insights.SCORE:
            got[c.round] = got.get(c.round, 0) + 1
    return all(got.get(r, 0) == expected[r] for r in window_rounds(opt))


def plan(opt):
    """Every spec the screens need, in the order they need them (review
    I15): the current Overall-score model and every Round's Overall-score
    check (Make a round's column and its widening), the measurements'
    checks in the verdict's Rounds (answer 1), the current measurement
    models, then every other check, latest Round first. Specs with fewer
    than MIN_ROWS measured rows fit nothing and are left out."""
    names = [n for n, _, _ in insights.outputs(opt)]
    measurements = [n for n in names if n != insights.SCORE]
    checked = [(r, first) for r, first in insights.round_order(opt) if first >= insights.MIN_ROWS]
    window = set(window_rounds(opt))
    every = range(len(opt.Y_history))
    wanted = [(NOW, insights.SCORE, every)]
    wanted += [(r, insights.SCORE, range(first)) for r, first in reversed(checked)]
    wanted += [(r, n, range(first)) for r, first in reversed(checked) if r in window
               for n in measurements]
    wanted += [(NOW, n, every) for n in measurements]
    wanted += [(r, n, range(first)) for r, first in reversed(checked) if r not in window
               for n in measurements]
    out = []
    for stage, name, rows in wanted:
        use = tuple(i for i in rows if insights.measured(opt, name, i) is not None)
        if len(use) >= insights.MIN_ROWS:
            out.append(Spec(stage, name, use, insights.model_key(opt, name, list(use))))
    return out


def stage_specs(opt, specs, stage):
    """The specs one part of the screens waits for:
    "column" — Make a round's Overall-score column (the current Overall-score
    model and every Round's Overall-score check);
    "trust"  — answer 1 and Make a round's track-record line (every Round's
    Overall-score check and the measurements' checks in the verdict's Rounds);
    "stand"  — answers 2 and 3 ("trust" and the current Overall-score model);
    "all"    — everything."""
    window = set(window_rounds(opt)) if stage in ("trust", "stand") else set()
    keep = {
        "column": lambda s: s.name == insights.SCORE,
        "trust": lambda s: s.stage != NOW and (s.name == insights.SCORE or s.stage in window),
        "stand": lambda s: s.name == insights.SCORE or (s.stage != NOW and s.stage in window),
        "all": lambda s: True,
    }[stage]
    return [s for s in specs if keep(s)]


def ready(specs, memo):
    return all(s.key in memo for s in specs)


def only_pairs(specs):
    """check_rounds' `only` for `specs`: their (Round, output) pairs."""
    return {(s.stage, s.name) for s in specs if s.stage != NOW}


def rounds_done(opt, specs, memo):
    """(Rounds whose checks are all fitted, Rounds to check): the count on
    the one progress line."""
    by_round = {}
    for s in specs:
        if s.stage != NOW:
            by_round.setdefault(s.stage, []).append(s.key in memo)
    return sum(all(v) for v in by_round.values()), len(by_round)


def compute_inline(opt, memo, specs, step=None):
    """Fit `specs` not yet in `memo` here, in the app's process, most
    needed first: the fallback when there are no workers. `step(done,
    total)` is called after each one, for the progress line."""
    todo = [s for s in specs if s.key not in memo]
    with insights.memo_in_use(memo):
        for k, s in enumerate(todo):
            insights.fit(opt, s.name, list(s.rows))
            if step is not None:
                step(k + 1, len(todo))


# ------------------------------------------------------------------ #
#  A Job: one project state's specs, in the pool
# ------------------------------------------------------------------ #

_INFLIGHT = {}      # model key -> [Future]: shared by every Job that wants it
_JOBS = {}          # (project, insights.data_key) -> Job
_JOBS_LOCK = threading.Lock()


def _requests(opt, spec):
    """The main fit and the fold fits of one spec, as worker requests."""
    X, Y = insights.xy(opt, spec.name, list(spec.rows))
    cfg = getattr(opt, "bo_config", None)
    out = [{"kind": "main", "cfg": cfg, "X": X.tolist(), "Y": Y.tolist()}]
    for kept, held in insights.folds(len(spec.rows)):
        out.append({"kind": "fold", "cfg": cfg, "X": X[kept].tolist(), "Y": Y[kept].tolist(),
                    "Xt": X[held].tolist(), "Yt": Y[held].tolist()})
    return out


def _entry(answers):
    """A memo entry from a spec's answers: the main fit's state and the
    widening from its folds (a failed fold is skipped, as in-process)."""
    main, folds = answers[0], answers[1:]
    if "error" in main:
        return {"error": main["error"]}
    z = [v for f in folds if "z" in f for v in f["z"]]
    return {"state": main["state"], "widen": insights.widen_from(z)}


class Job:
    """One project state's specs not yet in `memo`, sent to `pool` in plan
    order. A spec already in flight for an earlier state (a result saved
    while it ran) is shared, not sent again; a spec the new state no longer
    needs is cancelled if it has not started."""

    def __init__(self, opt, pool, memo, specs=None):
        self.specs = plan(opt) if specs is None else specs
        self.results = {}           # model key -> memo entry, for every session polling
        self.futures = {}           # model key -> [Future]
        self.failed = False
        for k, s in enumerate(self.specs):
            if s.key in memo or s.key in self.futures:
                continue
            futs = _INFLIGHT.get(s.key)
            if futs is None or any(f.cancelled() for f in futs):
                futs = [pool.submit(k, dict(r, id=f"{s.key}:{j}"))
                        for j, r in enumerate(_requests(opt, s))]
                _INFLIGHT[s.key] = futs
            self.futures[s.key] = futs

    def poll(self, memo):
        """Move every finished spec into `memo` (and self.results); True
        when nothing is left in flight. A spec whose worker failed marks the
        Job failed; the screens then fit what is missing in-process."""
        for key, futs in list(self.futures.items()):
            if not all(f.done() for f in futs):
                continue
            del self.futures[key]
            _INFLIGHT.pop(key, None)
            try:
                self.results[key] = _entry([f.result() for f in futs])
            except Exception:
                self.failed = True
        for key, entry in self.results.items():
            memo.setdefault(key, entry)
        return not self.futures

    def wait(self, memo, timeout=600):
        """poll() until nothing is left in flight (tests, timing)."""
        end = time.monotonic() + timeout
        while not self.poll(memo):
            if time.monotonic() > end:
                raise TimeoutError("the Insights workers took too long")
            time.sleep(0.02)
        return self

    def cancel_others(self, keep):
        for key, futs in list(self.futures.items()):
            if key not in keep:
                for f in futs:
                    f.cancel()
                _INFLIGHT.pop(key, None)


def job_for(opt, pool, memo, specs=None):
    """The Job for this project's data, started on first call; an earlier
    Job of the same project gives up what the new state does not need."""
    key = (opt.project_name, insights.data_key(opt))
    with _JOBS_LOCK:
        job = _JOBS.get(key)
        if job is None:
            job = Job(opt, pool, memo, specs)
            for old in [k for k in _JOBS if k[0] == opt.project_name]:
                _JOBS.pop(old).cancel_others({s.key for s in job.specs})
            _JOBS[key] = job
        return job
