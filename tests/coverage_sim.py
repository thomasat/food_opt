"""Round-ahead coverage on 48 simulated projects (spec 2026-09-30 §8, M20).

tests/fixtures/coverage_sims/ holds 48 projects whose every formulation was
chosen by the app's own ask(), with the 2026-09-29 trust audit's truths,
noise, measurement counts and Round sizes (perf.md §2.1): 40 of 36 rows
(good 1-16, shift 101-108, shift8 301-308, step 201-208) and 8 of 60 rows
(good 21-28).

tests/fixtures/coverage_checks.json holds each project's Round-ahead checks
before I1's widening, as insights.check_rounds works them out. It is
written by

    python -m tests.coverage_sim

(about a minute, in worker processes) and rewritten only when a change is
MEANT to change a fitted model — the same change bumps insights.MODEL_SALT.
test_080_coverage.py reads it fast; its slow variant works every check out
again and compares."""
import json
import os
import pathlib
import sys
import tempfile
from types import SimpleNamespace

import insights
import insights_jobs
from food_bo import FoodOptimizer

HERE = pathlib.Path(__file__).parent
SIMS = HERE / "fixtures" / "coverage_sims"
FIXTURE = HERE / "fixtures" / "coverage_checks.json"
FIELDS = ["formulation", "round", "name", "predicted", "low", "high", "measured",
          "inside", "fully_measured", "raw_mean", "sd"]


def projects():
    return sorted(SIMS.glob("*.json"))


def load(path):
    """The simulated project at `path`, opened in a fresh folder."""
    payload = json.loads(pathlib.Path(path).read_text())
    os.chdir(tempfile.mkdtemp(prefix="coverage-"))
    state = payload["state"]
    opt = FoodOptimizer(state["project_name"])
    opt.save = lambda: None
    opt.import_json(state)
    opt.save = lambda: None
    return opt, payload["meta"]


def raw_checks(opt, memo=None):
    """Every Round-ahead check of `opt` before I1's widening, in Round order."""
    rounds = {}
    with insights.memo_in_use({} if memo is None else memo):
        insights.check_rounds(opt, cache=rounds)
    keys = insights.round_keys(opt)
    return [c for r, _ in insights.round_order(opt) for c in rounds.get(keys[r], [])]


def stub(entry):
    """What insights.headline and nest_score_widen read of a project."""
    return SimpleNamespace(objectives=entry["objectives"],
                           measurements_by_importance=lambda: entry["objectives"],
                           utility_ceiling=lambda: entry["ceiling"])


def checks_of(entry):
    return [insights.Check(**dict(zip(FIELDS, row))) for row in entry["checks"]]


def entry_for(opt, meta, checks):
    return {"project": opt.project_name, "meta": meta, "objectives": opt.objectives,
            "ceiling": float(opt.utility_ceiling()),
            "checks": [[getattr(c, f) for f in FIELDS] for c in checks]}


def coverage(entries):
    """The share inside, pooled over every project: every check, and the
    checks in the Rounds each project's verdict reads at its end, for the
    measurements and for the Overall score before and after I1."""
    tally = {k: [0, 0] for k in ("measured", "score_raw", "score", "window_measured",
                                 "window_score_raw", "window_score")}

    def add(key, inside):
        tally[key][0] += bool(inside)
        tally[key][1] += 1

    for entry in entries:
        raw = checks_of(entry)
        nested = insights.nest_score_widen(stub(entry), raw)
        window = set(insights.headline(stub(entry), nested).rounds)
        for c, n in zip(raw, nested):
            inside_window = c.round in window
            if c.name != insights.SCORE:
                add("measured", c.inside)
                if inside_window:
                    add("window_measured", c.inside)
            elif c.fully_measured:
                add("score_raw", c.inside)
                add("score", n.inside)
                if inside_window:
                    add("window_score_raw", c.inside)
                    add("window_score", n.inside)
    return {k: (a / b if b else None, b) for k, (a, b) in tally.items()}


def work_out(paths, workers=8):
    """Every project's entry, its models fitted in `workers` processes."""
    pool = insights_jobs.Pool(workers)
    assert pool.wait_ready(180), "no Insights worker started"
    try:
        opened = [load(p) for p in paths]
        memo = {}
        jobs = [insights_jobs.Job(opt, pool, memo) for opt, _ in opened]
        for job in jobs:
            job.wait(memo)
        return [entry_for(opt, meta, raw_checks(opt, memo)) for opt, meta in opened]
    finally:
        pool.close()


def fixture():
    return json.loads(FIXTURE.read_text())


if __name__ == "__main__":
    here = os.getcwd()
    entries = work_out(projects())
    os.chdir(here)
    print(json.dumps(coverage(entries), indent=1), file=sys.stderr)
    FIXTURE.write_text(json.dumps({"salt": insights.MODEL_SALT, "projects": entries}) + "\n")
    print(f"wrote {FIXTURE}", file=sys.stderr)
