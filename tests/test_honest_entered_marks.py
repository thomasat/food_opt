"""Rows the team entered are never drawn as the app's suggestion."""
import dataclasses
import json

import pytest

import bo_views
import sample_projects as sp
import storage as storage_backend
import ui_insights
import wording
from food_bo import FoodOptimizer
from tests import demo_projects as D

SUGG, ENT = wording.PLOT_SUGGESTION, wording.PLOT_ENTERED


def _whats(chart, domain=False):
    spec = json.loads(json.dumps(chart.to_dict()))
    found = {"rows": [], "domains": []}

    def walk(x):
        if isinstance(x, dict):
            if x.get("encoding", {}).get("shape", {}).get("scale", {}).get("domain"):
                found["domains"].append(x["encoding"]["shape"]["scale"]["domain"])
            if isinstance(x.get("data"), dict) and False:
                pass
            for v in x.values():
                walk(v)
        elif isinstance(x, list):
            for v in x:
                walk(v)
    walk(spec)
    for rows in spec["datasets"].values():
        if rows and "what" in rows[0] and "no" in rows[0] and "bo" in rows[0]:
            found["rows"] += [r["what"] for r in rows]
    return found


def _chart(opt, why):
    """The explore map as Make a round drew it before v3 (ui_batch.why_chart,
    gone with the map): ui_insights.explore_map_chart, which Review keeps."""
    tested = [{"mean": m, "sd": s}
              for m, s in zip(why.tested_mean.tolist(), why.tested_sd.tolist())]
    picks = [{"mean": r.mean, "sd": r.sd, "no": str(r.formulation), "own": bool(r.own)}
             for r in why.rows]
    return ui_insights.explore_map_chart(why.pool_mean, why.pool_sd, tested, picks, None, opt,
                                         gain=getattr(why, "pool_value", None))


def test_entered_round_has_no_suggestion_mark(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    store = storage_backend.LocalStorage()
    sp.ensure(store)
    opt = FoodOptimizer(sp.ROUND6, storage=store)
    why = bo_views.why(opt)
    assert why.rows and all(r.own for r in why.rows)
    got = _whats(_chart(opt, why))
    assert set(got["rows"]) == {ENT}
    assert all(SUGG not in d and ENT in d for d in got["domains"])


def test_generated_round_is_all_suggestions_and_mixed_shows_both():
    opt = D.load(D.LINE_A)
    why = bo_views.why(opt)
    got = _whats(_chart(opt, why))
    assert set(got["rows"]) == {SUGG}
    assert all(ENT not in d for d in got["domains"])
    rows = [dataclasses.replace(r, own=(i == 0)) for i, r in enumerate(why.rows)]
    got = _whats(_chart(opt, dataclasses.replace(why, rows=rows)))
    assert set(got["rows"]) == {SUGG, ENT}
    assert all(SUGG in d and ENT in d for d in got["domains"])
