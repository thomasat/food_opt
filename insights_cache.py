"""Insights' fitted models and checks, kept beside the project so reopening
it fits nothing (spec 2026-09-30 §9 I15, §10.2).

One file per project: `.insights/<project>.insights.json` in the projects'
folder. JSON data only — hyperparameters as numbers, the Round-by-Round
checks, the predicted best — never a pickled object, and never a name that
matches `*.pkl`, so it is never listed as a project, never backed up with
the projects and never part of a Saved copy (those copy `<name>.pkl`
alone). Every entry is keyed by a digest of everything it depends on, so a
stale entry is simply never looked up. A missing, damaged or older file
reads as empty. Nothing here leaves the machine or touches the suggestions.
"""
import dataclasses
import json
import os

import insights

FOLDER = ".insights"
SUFFIX = ".insights.json"
VERSION = 1


@dataclasses.dataclass
class Saved:
    memo: dict              # insights.model_key -> {"state": ..., "widen": ...}
    rounds: dict            # insights.round_keys key -> [insights.Check], before I1's widening
    best: tuple | None      # (insights.setup_key, PredictedBest or refusal text), or None


def empty():
    return Saved({}, {}, None)


def path(project, folder="."):
    return os.path.join(folder, FOLDER, f"{project}{SUFFIX}")


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
    """The Saved models and checks for `project`, or empty() when the file
    is missing, unreadable, from another version or another MODEL_SALT."""
    try:
        with open(path(project, folder), encoding="utf-8") as fh:
            payload = json.load(fh)
        if payload.get("version") != VERSION or payload.get("salt") != insights.MODEL_SALT:
            return empty()
        memo = {k: v for k, v in payload["memo"].items()
                if isinstance(v, dict) and "state" in v and "widen" in v}
        rounds = {k: [insights.Check(**c) for c in v] for k, v in payload["rounds"].items()}
        return Saved(memo, rounds, _best_from_json(payload.get("best")))
    except Exception:
        return empty()


def save(project, saved, folder="."):
    """Write `saved` for `project`, failed fits left out, through a temp file
    moved into place so a crash never leaves half a file. Returns the path."""
    target = path(project, folder)
    os.makedirs(os.path.dirname(target), exist_ok=True)
    payload = {"version": VERSION, "salt": insights.MODEL_SALT,
               "memo": {k: v for k, v in saved.memo.items() if "state" in v},
               "rounds": {k: [dataclasses.asdict(c) for c in v] for k, v in saved.rounds.items()},
               "best": _best_to_json(saved.best)}
    tmp = f"{target}.tmp"
    try:
        with open(tmp, "w", encoding="utf-8") as fh:
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
