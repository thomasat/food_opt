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
# Workers stop after this long with nothing to do (the pool's own watch
# closes it); the next fit starts them again.
IDLE_SECS = 600
# How long pool() waits for a worker's imports before giving up on workers.
START_SECS = 60
# How long one request (a main fit or a fold fit) may run before its worker
# is taken to have hung and is stopped (re-review N-1). The slowest measured
# fit, 120 rows x 6 measurements, takes a few seconds: this is generous.
REQUEST_SECS = 120
# C2 (spec 2026-10-01; 04 I-3, review M5): a pool that broke after a worker
# had started (one killed by memory pressure, say) is replaced on the next
# need, at most RESTART_CAP times in RESTART_WINDOW seconds; after that the
# app process fits in-process, and Insights says so (INSIGHTS_IN_PROCESS).
RESTART_CAP = 3
RESTART_WINDOW = 600


def worker_count():
    """FOODOPT_INSIGHTS_WORKERS when set (0: none), else up to 6, leaving
    two cores for the app itself and at least 4 GB of memory per worker
    (each holds about 400 MB): 6 on a 14-core 48 GB Mac, 4 on a 16 GB
    laptop, 2 on an 8 GB one."""
    env = os.environ.get("FOODOPT_INSIGHTS_WORKERS")
    if env is not None:
        try:
            return max(0, int(env))
        except ValueError:
            return 0
    try:
        memory_gb = os.sysconf("SC_PHYS_PAGES") * os.sysconf("SC_PAGE_SIZE") / 2 ** 30
    except (AttributeError, OSError, ValueError):
        memory_gb = 8
    return max(1, min(6, (os.cpu_count() or 2) - 2, int(memory_gb // 4)))


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
        self.n = n
        self.broken = False
        self.closed = False
        self.became_ready = False  # a worker finished its imports (not set by _break)
        self._running = 0          # requests a worker is fitting now
        self._since = {}           # worker process -> when its request started
        self.started = self.last_used = time.monotonic()
        env = {**os.environ, "OMP_NUM_THREADS": "1", "PYTHONDONTWRITEBYTECODE": "1"}
        for _ in range(n):
            proc = subprocess.Popen(
                [python or sys.executable, WORKER], stdin=subprocess.PIPE,
                stdout=subprocess.PIPE, stderr=subprocess.DEVNULL, text=True, bufsize=1,
                env=env)
            self._procs.append(proc)
            threading.Thread(target=self._serve, args=(proc,), daemon=True).start()
        threading.Thread(target=self._watch, daemon=True).start()

    def wait_ready(self, timeout):
        """True once a worker has finished its imports; False if none did
        in time or the pool broke."""
        return self._ready.wait(timeout) and not self.broken

    def submit(self, priority, request):
        """A Future of the worker's answer (a dict) to `request`."""
        fut = cf.Future()
        with self._lock:
            if self.broken or self.closed:
                fut.set_exception(RuntimeError("the Insights workers stopped"))
                return fut
            self.last_used = time.monotonic()
            self._queue.put((priority, next(self._seq), request, fut))
        return fut

    def idle(self):
        """Started, nothing queued, nothing being fitted, and nothing asked
        of it for IDLE_SECS (pool() handing it out counts as being asked)."""
        return (self._ready.is_set() and self._queue.empty() and not self._running
                and time.monotonic() - self.last_used > IDLE_SECS)

    def _watch(self):
        """Close this pool once it has been idle for IDLE_SECS, without
        waiting for anyone to call pool() (review B I-2): six idle workers
        hold about 2.4 GB. And stop a worker whose request has run past
        REQUEST_SECS (re-review N-1): its reader then fails what was
        waiting and breaks the pool, and the screens fit here behind their
        own indicator instead of waiting for ever."""
        while not (self.closed or self.broken):
            time.sleep(max(0.05, min(5.0, IDLE_SECS / 4, REQUEST_SECS / 4)))
            now = time.monotonic()
            with self._lock:
                hung = [p for p, t in self._since.items() if now - t > REQUEST_SECS]
            for proc in hung:
                try:
                    proc.kill()
                except Exception:
                    pass
            _close_if_idle(self)

    def check_start(self):
        """Break the pool if no worker finished its imports within
        START_SECS: a worker that hangs must not leave a screen waiting."""
        if not self._ready.is_set() and time.monotonic() - self.started > START_SECS:
            self._break()

    def _serve(self, proc):
        try:
            if json.loads(proc.stdout.readline() or "{}").get("ready") is not True:
                raise RuntimeError("the Insights worker did not start")
            with self._lock:
                self.last_used = max(self.last_used, time.monotonic())
                self.became_ready = True
            self._ready.set()
            while True:
                _, _, request, fut = self._queue.get()
                if request is None:
                    return
                if not fut.set_running_or_notify_cancel():
                    continue
                with self._lock:
                    self._running += 1
                    self._since[proc] = time.monotonic()
                try:
                    proc.stdin.write(json.dumps(request) + "\n")
                    proc.stdin.flush()
                    line = proc.stdout.readline()
                    if not line:
                        raise RuntimeError("the Insights worker stopped")
                    fut.set_result(json.loads(line))
                except BaseException as exc:
                    # Broken before the failure is seen: a Job that polls it
                    # finds the pool broken, and start() replaces it at once
                    # rather than leaving the screen to fit in-process (C2).
                    with self._lock:
                        self.broken = True
                    fut.set_exception(exc)
                    raise
                finally:
                    with self._lock:
                        self._running -= 1
                        self._since.pop(proc, None)
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
        with self._lock:
            self.closed = True
        for proc in self._procs:
            try:
                proc.kill()
            except Exception:
                pass
        for _ in self._procs:
            self._queue.put((-1, -1, None, None))


_POOL = None
_POOL_LOCK = threading.Lock()
# model key -> [Future] of every spec in flight, shared by every Job that
# wants it (a result saved while one runs); emptied with the pool.
_INFLIGHT = {}
# Set once workers are given up on (a worker would not start, or pools
# broke more than RESTART_CAP times in RESTART_WINDOW): this app process
# fits in-process from then on rather than starting workers again.
_GAVE_UP = False
# When each pool that had started broke, for the restart cap.
_BREAKS = []
# Given up on by the cap (not because no worker could start): the one case
# Insights names with its indicator.
_STOPPED = False


def _broke(p):
    """pool()'s step for a pool found broken: given up on at once if no
    worker ever started (as before); else counted, and given up on only
    past RESTART_CAP breaks in RESTART_WINDOW seconds."""
    global _GAVE_UP, _STOPPED
    if not p.became_ready:
        _GAVE_UP = True
        return
    now = time.monotonic()
    _BREAKS[:] = [t for t in _BREAKS if now - t < RESTART_WINDOW] + [now]
    if len(_BREAKS) > RESTART_CAP:
        _GAVE_UP = _STOPPED = True


def stopped():
    """Whether the workers were given up on after repeated failures."""
    return _STOPPED


def pool():
    """The app process's Pool, started on first use; None when workers are
    off (worker_count() is 0) or were given up on in this process (the
    caller then fits in its own process). A pool closes itself after
    IDLE_SECS idle (Pool._watch); the next call starts a new one. A broken
    pool is replaced here, on the next need, within the restart cap
    (_broke). A healthy pool is never closed here: handing it out counts
    as using it."""
    global _POOL, _GAVE_UP
    with _POOL_LOCK:
        if _POOL is not None:
            _POOL.check_start()
            if _POOL.broken:
                _broke(_POOL)
            if _POOL.broken or _POOL.closed:
                _POOL.close()
                _POOL = None
                _INFLIGHT.clear()
        n = worker_count()
        if _POOL is None and n > 0 and not _GAVE_UP:
            try:
                _POOL = Pool(n)
            except Exception:
                _GAVE_UP = True
        if _POOL is not None:
            _POOL.last_used = time.monotonic()
        return _POOL


def _close_if_idle(p):
    """Pool._watch's step: close `p` if it is idle, under pool()'s lock so
    pool() never hands out a pool that is closing."""
    global _POOL
    with _POOL_LOCK:
        if p.closed or p.broken or not p.idle():
            return
        if _POOL is p:
            _POOL = None
            _INFLIGHT.clear()
    p.close()


def workers_on():
    """Whether fits go to worker processes in this app process: workers are
    not set off (worker_count() > 0) and none has failed here. It starts
    nothing (pool() does, when there is something to fit)."""
    return worker_count() > 0 and not _GAVE_UP


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
    checks in the verdict's Rounds (answer 1), those of the Rounds before
    them (their widening), the current measurement models, then every
    other check, latest Round first. Specs with fewer
    than MIN_ROWS measured rows fit nothing and are left out."""
    names = [n for n, _, _ in insights.outputs(opt)]
    measurements = [n for n in names if n != insights.SCORE]
    checked = [(r, first) for r, first in insights.round_order(opt) if first >= insights.MIN_ROWS]
    window = set(window_rounds(opt))
    # The checks the verdict's checks are widened by (every Round before
    # them, insights.widen_pairs): answer 1 waits for them too (2026-10-07).
    reads = {r for r, _ in insights.widen_pairs(opt, {(r, None) for r in window})}
    every = range(len(opt.Y_history))
    wanted = [(NOW, insights.SCORE, every)]
    wanted += [(r, insights.SCORE, range(first)) for r, first in reversed(checked)]
    wanted += [(r, n, range(first)) for r, first in reversed(checked) if r in window
               for n in measurements]
    wanted += [(r, n, range(first)) for r, first in reversed(checked)
               if r in reads - window for n in measurements]
    wanted += [(NOW, n, every) for n in measurements]
    wanted += [(r, n, range(first)) for r, first in reversed(checked) if r not in reads
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
    "trust"  — answer 1, Make a round's track-record line and the
    measurements' 95 % ranges on Results and Make a round (every Round's
    Overall-score check, the measurements' checks in the verdict's Rounds
    and every check their widening reads, insights.widen_pairs);
    "stand"  — answers 2 and 3 ("trust" and the current Overall-score model);
    "now"    — the current models (Make a round's predicted measurements,
    shown once "trust"'s checks, their widening, are in too);
    "all"    — everything.
    A part never lacks a check its own checks' widening reads, so a range
    worked out from a part is the one every check gives (2026-10-07)."""
    pairs = set()
    if stage in ("trust", "stand"):
        window = set(window_rounds(opt))
        pairs = insights.widen_pairs(opt, {(s.stage, s.name) for s in specs
                                           if s.stage != NOW and s.stage in window})
    keep = {
        "column": lambda s: s.name == insights.SCORE,
        "now": lambda s: s.stage == NOW,
        "trust": lambda s: s.stage != NOW and (s.name == insights.SCORE
                                               or (s.stage, s.name) in pairs),
        "stand": lambda s: s.name == insights.SCORE or (s.stage != NOW
                                                         and (s.stage, s.name) in pairs),
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

_JOBS = {}          # (project, insights.data_key) -> Job
_JOBS_LOCK = threading.Lock()


def _requests(opt, spec):
    """The main fit and the fold fits of one spec, as worker requests."""
    X, Y = insights.xy(opt, spec.name, list(spec.rows))
    cfg = getattr(opt, "bo_config", None)
    out = [{"kind": "main", "cfg": cfg, "X": X.tolist(), "Y": Y.tolist()}]
    for kept, held in insights.cv_folds(opt, spec.rows):
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
        self.pool = pool
        self.results = {}           # model key -> memo entry, for every session polling
        self.futures = {}           # model key -> [Future]
        self.failed = False
        # Two tabs poll one Job from their own threads (review B M-1).
        self._lock = threading.RLock()
        self.top_up(opt, pool, memo)

    def top_up(self, opt, pool, memo):
        """Send every spec of this state that is in neither `memo`, the
        results nor the specs in flight (review B I-1): a later caller —
        a second tab, or the project reopened from disk — may lack models
        the first caller's memo already had."""
        with self._lock:
            for k, s in enumerate(self.specs):
                if s.key in memo or s.key in self.futures or s.key in self.results:
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
        with self._lock:
            for key, futs in list(self.futures.items()):
                if not all(f.done() for f in futs):
                    continue
                del self.futures[key]
                _INFLIGHT.pop(key, None)
                try:
                    self.results[key] = _entry([f.result() for f in futs])
                except Exception:
                    self.failed = True
            for key, entry in list(self.results.items()):
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
        with self._lock:
            for key, futs in list(self.futures.items()):
                if key not in keep:
                    for f in futs:
                        f.cancel()
                    _INFLIGHT.pop(key, None)


def current_job(opt):
    """The Job for this project's data if one was started, else None."""
    return _JOBS.get((opt.project_name, insights.data_key(opt)))


def job_for(opt, pool, memo, specs=None):
    """The Job for this project's data, started on first call and topped up
    with what this caller's memo lacks on every later one; an earlier Job of
    the same project gives up what the new state does not need. A Job that
    failed with a pool since replaced (C2) is started again on the new one."""
    key = (opt.project_name, insights.data_key(opt))
    with _JOBS_LOCK:
        job = _JOBS.get(key)
        if job is not None and job.failed and job.pool is not pool:
            del _JOBS[key]
            job = None
        if job is None:
            job = Job(opt, pool, memo, specs)
            for old in [k for k in _JOBS if k[0] == opt.project_name]:
                _JOBS.pop(old).cancel_others({s.key for s in job.specs})
            _JOBS[key] = job
        elif not job.failed:
            job.top_up(opt, pool, memo)
        return job


def ask_workers(request, timeout=60):
    """One request answered by the pool, waited for (the BO views'
    acquisition, kind "acq", spec 2026-09-30-bo-views §3.4, review I10): the
    worker's answer, or None when there are no workers, they failed, or
    the answer took longer than `timeout` seconds — the caller then works
    it out in its own process. Sent ahead of the background fits: a screen
    is waiting for it."""
    p = pool()
    if p is None:
        return None
    try:
        return p.submit(-1, dict(request, id="acq")).result(timeout)
    except Exception:
        return None
