"""Insights' fitted models and checks, kept beside the project so reopening
it fits nothing (spec 2026-09-30 §9 I15, §10.2).

One file per project: `.insights/<project>.insights.json` in the projects'
folder. JSON data only — hyperparameters as numbers, the Round-by-Round
checks, the predicted best, the next Round's expected improvement — never a pickled object, and never a name that
matches `*.pkl`, so it is never listed as a project, never backed up with
the projects and never part of a Saved copy (those copy `<name>.pkl`
alone). Every entry is keyed by a digest of everything it depends on, so a
stale entry is simply never looked up. A missing, damaged or older file
reads as empty. Nothing here leaves the machine or touches the suggestions.
"""
import dataclasses
import json
import math
import os
import tempfile

import bo_views
import insights

FOLDER = ".insights"
SUFFIX = ".insights.json"
VERSION = 1


@dataclasses.dataclass
class Saved:
    memo: dict              # insights.model_key -> {"state": ..., "widen": ...}
    rounds: dict            # insights.round_keys key -> [insights.Check], before I1's widening
    best: tuple | None      # (insights.setup_key, PredictedBest or refusal text), or None
    # (bo_views.next_gain_key, RoundGain), or None: the next Round's expected
    # improvement (spec 2026-10-01 C1). Optional, like `best`: a file from
    # before it reads with None, and the figure is worked out once.
    gain: tuple | None = None


def empty():
    return Saved({}, {}, None)


def enabled():
    """Off only when FOODOPT_INSIGHTS_CACHE is "0": the test suite's
    default (tests/conftest.py), so one test's app never reads what
    another wrote. Its own tests turn it on."""
    return os.environ.get("FOODOPT_INSIGHTS_CACHE", "1") != "0"


def path(project, folder="."):
    return os.path.join(folder, FOLDER, f"{project}{SUFFIX}")


# ------------------------------------------------------------------ #
#  C3 (spec 2026-10-01; 04 M-2, review M6): a number that is not finite
#  never comes back as a model, a check or a figure. JSON reads NaN and
#  Infinity bare, so a damaged but well-formed file would otherwise load;
#  the entry holding one is dropped (and so fitted again), the rest of the
#  file still loads, and nothing dropped is written back.
# ------------------------------------------------------------------ #

def _finite(value):
    """Every number in `value` (nested dicts and lists) is finite."""
    if value is None or isinstance(value, (bool, str)):
        return True
    if isinstance(value, (int, float)):
        return math.isfinite(value)
    if isinstance(value, dict):
        return all(_finite(v) for v in value.values())
    if isinstance(value, (list, tuple)):
        return all(_finite(v) for v in value)
    return False


def _number(value, optional=False):
    """A finite number; None only where the dataclass allows it."""
    if value is None:
        return optional
    return (isinstance(value, (int, float)) and not isinstance(value, bool)
            and math.isfinite(value))


def _no_nan(value):
    """No NaN anywhere in `value` (nested lists); infinities allowed."""
    if isinstance(value, float):
        return not math.isnan(value)
    if isinstance(value, (list, tuple)):
        return all(_no_nan(v) for v in value)
    return True


# A GP's state holds infinite numbers of its own: a constraint with no
# upper (or lower) bound stores it as a buffer of inf.
_BOUNDS = ("_constraint.lower_bound", "_constraint.upper_bound")


def _good_state(state):
    """hyper_state's {name: [dtype, values]}: finite values, but for a
    constraint's open bound, which only may be infinite (never NaN)."""
    for name, item in state.items():
        if not (isinstance(item, (list, tuple)) and len(item) == 2):
            return False
        values = item[1]
        if not (_no_nan(values) if name.endswith(_BOUNDS) else _finite(values)):
            return False
    return True


def _good_model(entry):
    return (isinstance(entry, dict) and isinstance(entry.get("state"), dict)
            and _number(entry.get("widen")) and _good_state(entry["state"]))


def _good_check(c):
    return (isinstance(c, dict)
            and all(_number(c.get(k)) for k in ("predicted", "low", "high", "measured"))
            and all(_number(c.get(k), optional=True) for k in ("raw_mean", "sd"))
            and _number(c.get("widen", 1.0)) and _finite(c))


def _good_prediction(d, optional=False):
    if d is None:
        return optional
    return (isinstance(d, dict) and all(_number(d.get(k)) for k in ("mean", "low", "high"))
            and all(_number(d.get(k), optional=True) for k in ("raw_mean", "sd")))


def _good_best(d):
    if not isinstance(d, dict):
        return False
    if "refusal" in d:
        return isinstance(d["refusal"], str)
    return (_good_prediction(d.get("score")) and _good_prediction(d.get("repeat"), optional=True)
            and _number(d.get("factor", 1.0)) and isinstance(d.get("recipe"), dict)
            and _finite(d) and isinstance(d.get("compare"), int)
            and (d.get("close") is None or isinstance(d["close"], int)))


def _good_gain(d):
    """A bo_views.RoundGain as JSON: its numbers finite, None where it allows."""
    return (isinstance(d, dict) and set(d) == {f.name for f in
                                               dataclasses.fields(bo_views.RoundGain)}
            and (d["round"] is None or (isinstance(d["round"], int)
                                        and not isinstance(d["round"], bool)))
            and all(_number(d[k], optional=True) for k in ("best_measured", "highest", "gain"))
            and isinstance(d["exact"], bool)
            and (d["why_none"] is None or isinstance(d["why_none"], str)))


def _gain_to_json(gain):
    if gain is None or not isinstance(gain[1], bo_views.RoundGain):
        return None     # a failed search is worked out again next time
    item = [gain[0], dataclasses.asdict(gain[1])]
    return item if _good_gain(item[1]) else None


def _gain_from_json(item):
    if not item or len(item) != 2 or not _good_gain(item[1]):
        return None
    return item[0], bo_views.RoundGain(**item[1])


def _good_rounds(rounds):
    """{key: [check dicts]} with every Round holding a bad check left out."""
    return {k: v for k, v in rounds.items()
            if isinstance(v, list) and all(_good_check(c) for c in v)}


def _best_to_json(best):
    if best is None:
        return None
    key, value = best
    if isinstance(value, str):
        return [key, {"refusal": value}]
    if isinstance(value, insights.PredictedBest):
        return [key, dataclasses.asdict(value)]
    return None     # a failed search is worked out again next time


def _best_from_json(item):
    if not item:
        return None
    key, d = item
    if "refusal" in d:
        return key, d["refusal"]
    d = dict(d)
    d["score"] = insights.Prediction(**d["score"])
    if d.get("repeat") is not None:
        d["repeat"] = insights.Prediction(**d["repeat"])
    return key, insights.PredictedBest(**d)


def load(project, folder="."):
    """The Saved models, checks and figures for `project`, or empty() when the file
    is missing, unreadable, from another version or another MODEL_SALT."""
    if not enabled():
        return empty()
    try:
        with open(path(project, folder), encoding="utf-8") as fh:
            payload = json.load(fh)
        if payload.get("version") != VERSION or payload.get("salt") != insights.MODEL_SALT:
            return empty()
        memo = {k: v for k, v in payload["memo"].items() if _good_model(v)}
        rounds = {k: [insights.Check(**c) for c in v]
                  for k, v in _good_rounds(payload["rounds"]).items()}
        best = payload.get("best")
        best = _best_from_json(best) if best and _good_best(best[1]) else None
        return Saved(memo, rounds, best, _gain_from_json(payload.get("gain")))
    except Exception:
        return empty()


def save(project, saved, folder="."):
    """Write `saved` for `project`, failed fits left out, through a temp file
    moved into place so a crash never leaves half a file. Returns the path,
    or None when the cache is off."""
    if not enabled():
        return None
    target = path(project, folder)
    os.makedirs(os.path.dirname(target), exist_ok=True)
    best = _best_to_json(saved.best)
    payload = {"version": VERSION, "salt": insights.MODEL_SALT,
               "memo": {k: v for k, v in saved.memo.items() if _good_model(v)},
               "rounds": _good_rounds({k: [dataclasses.asdict(c) for c in v]
                                       for k, v in saved.rounds.items()}),
               "best": best if best and _good_best(best[1]) else None,
               "gain": _gain_to_json(saved.gain)}
    # A temp file of its own: two tabs of one project may save at once
    # (review B M-2).
    fd, tmp = tempfile.mkstemp(dir=os.path.dirname(target),
                               prefix=os.path.basename(target) + ".", suffix=".tmp")
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as fh:
            json.dump(payload, fh)
        os.replace(tmp, target)
    finally:
        if os.path.exists(tmp):
            os.remove(tmp)
    return target


def remove(project, folder="."):
    """Take `project`'s file off the disk, if there is one."""
    try:
        os.remove(path(project, folder))
    except OSError:
        pass
