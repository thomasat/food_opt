import json

import pytest

from egbo import Optimizer


def _build(seed=3):
    opt = Optimizer(seed=seed)
    opt.add_ingredient("Pea protein", 0, 5)
    opt.add_ingredient("Oil", 0, 5)
    opt.add_process_parameter("Temp", 60, 90, baseline=70, step=5)
    opt.add_objective("Taste", weight=1)
    return opt


def _taste(r):
    return {"Taste": max(0.0, min(10.0, 10 - (r["Pea protein"] - 3) ** 2
                                  - (r.get("Oil", 0) - 1) ** 2))}


def _round(opt, n):
    for r in opt.ask(n):
        opt.tell(r, _taste(r))


def test_roundtrip_then_ask_identical(tmp_path):
    opt = _build()
    _round(opt, 3)
    _round(opt, 3)
    path = tmp_path / "o.json"
    opt.save(path)
    loaded = Optimizer.load(path)
    assert loaded.to_dict() == opt.to_dict()
    assert loaded.ask(2) == opt.ask(2)


def test_open_rows_survive_save(tmp_path):
    opt = _build()
    opt.ask(2)
    path = tmp_path / "o.json"
    opt.save(path)
    loaded = Optimizer.load(path)
    assert loaded.pending() == opt.pending() and len(loaded.pending()) == 2
    recipe = loaded.pending()[0]["recipe"]
    assert loaded.tell(recipe, _taste(recipe)) == opt.pending()[0]["formulation"]


def test_refuses_foreign_file(tmp_path):
    with pytest.raises(ValueError):
        Optimizer.from_dict({"format": "other", "format_version": 1})
    with pytest.raises(ValueError):
        Optimizer.from_dict({"format": "egbo", "format_version": 2})
    path = tmp_path / "x.json"
    path.write_text(json.dumps({"hello": 1}), encoding="utf-8")
    with pytest.raises(ValueError):
        Optimizer.load(path)


def test_no_file_written_without_save(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    opt = _build()
    _round(opt, 3)
    _round(opt, 3)
    opt.to_dict()
    assert list(tmp_path.iterdir()) == []
