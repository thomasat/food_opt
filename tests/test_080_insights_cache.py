"""0.8.0 speed, Plan A Task 3: the fitted models and checks kept beside the
project (spec 2026-09-30 §9 I15, §10.2). JSON data only, in
.insights/<project>.insights.json, never *.pkl, never a Saved copy, and a
damaged or older file reads as empty."""
import glob
import json
import os

import pytest

import insights
import insights_cache
from storage import LocalStorage
from tests.test_insights import _burger


@pytest.fixture(autouse=True)
def _isolate_cwd(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("FOODOPT_INSIGHTS_CACHE", "1")


def test_the_cache_can_be_turned_off(monkeypatch):
    monkeypatch.setenv("FOODOPT_INSIGHTS_CACHE", "0")
    assert insights_cache.save("Patty", insights_cache.Saved({"b": {"state": {}, "widen": 1.0}},
                                                             {}, None)) is None
    assert not os.path.exists(".insights")
    assert insights_cache.load("Patty").memo == {}


def _saved(opt):
    memo, rounds = {}, {}
    with insights.memo_in_use(memo):
        insights.check_rounds(opt, cache=rounds)
        fitted = insights.fit_all(opt)
    best = insights.predicted_best(opt, fitted, insights.check_rounds(opt, cache=rounds))
    return insights_cache.Saved(memo, rounds, (insights.setup_key(opt), best))


def test_the_file_sits_in_its_own_folder_and_never_matches_pkl():
    path = insights_cache.path("Patty")
    assert path == os.path.join(".", ".insights", "Patty.insights.json")
    assert not glob.fnmatch.fnmatch(os.path.basename(path), "*.pkl")


def test_what_is_saved_loads_back_the_same():
    opt = _burger(rounds=4, noise=0.4)
    saved = _saved(opt)
    insights_cache.save(opt.project_name, saved)
    back = insights_cache.load(opt.project_name)
    assert back.memo == json.loads(json.dumps(saved.memo))
    assert back.rounds == saved.rounds
    assert back.best == saved.best


def test_the_file_is_plain_json():
    opt = _burger(rounds=3)
    path = insights_cache.save(opt.project_name, _saved(opt))
    with open(path, "rb") as fh:
        head = fh.read(1)
    assert head == b"{"
    with open(path, encoding="utf-8") as fh:
        payload = json.load(fh)
    assert payload["version"] == insights_cache.VERSION
    assert payload["salt"] == insights.MODEL_SALT


def test_a_reopened_project_fits_nothing(monkeypatch):
    opt = _burger(rounds=4, noise=0.4)
    saved = _saved(opt)
    insights_cache.save(opt.project_name, saved)
    back = insights_cache.load(opt.project_name)
    monkeypatch.setattr(insights, "_fit_gp", lambda *a: pytest.fail("fitted again"))
    with insights.memo_in_use(back.memo):
        fitted = insights.fit_all(opt)
        checks = insights.check_rounds(opt, cache=back.rounds)
    assert fitted[insights.SCORE].model is not None
    assert checks == insights.nest_score_widen(opt, [c for v in saved.rounds.values() for c in v])


@pytest.mark.parametrize("content", ["", "{", "[]", '{"version": 999}', "\x00\x01"])
def test_a_damaged_or_other_version_file_reads_as_empty(content):
    os.makedirs(".insights")
    with open(insights_cache.path("Patty"), "w") as fh:
        fh.write(content)
    back = insights_cache.load("Patty")
    assert (back.memo, back.rounds, back.best) == ({}, {}, None)


def test_another_salt_reads_as_empty(monkeypatch):
    opt = _burger(rounds=3)
    insights_cache.save(opt.project_name, _saved(opt))
    monkeypatch.setattr(insights, "MODEL_SALT", "later")
    assert insights_cache.load(opt.project_name).memo == {}


def test_a_missing_file_reads_as_empty():
    assert insights_cache.load("Nobody").memo == {}


def test_failed_fits_are_not_written():
    saved = insights_cache.Saved({"a": {"error": "no fit"}, "b": {"state": {}, "widen": 1.0}},
                                 {}, None)
    insights_cache.save("Patty", saved)
    assert list(insights_cache.load("Patty").memo) == ["b"]


def test_a_failed_search_is_not_written_and_a_refusal_is():
    insights_cache.save("A", insights_cache.Saved({}, {}, ("k", {"failed": True})))
    insights_cache.save("B", insights_cache.Saved({}, {}, ("k", "Add an ingredient first.")))
    assert insights_cache.load("A").best is None
    assert insights_cache.load("B").best == ("k", "Add an ingredient first.")


def test_it_is_never_listed_as_a_project_nor_copied_with_one():
    opt = _burger(rounds=3)
    insights_cache.save(opt.project_name, _saved(opt))
    store = LocalStorage()
    assert store.list_projects() == [opt.project_name]
    copy = store.archive(opt.project_name, "pre_edit", copy=True)
    assert not os.path.exists(insights_cache.path(copy))
    assert sorted(os.listdir(".insights")) == [f"{opt.project_name}.insights.json"]


def test_deleting_a_saved_copy_deletes_its_file_too():
    opt = _burger(rounds=3)
    store = LocalStorage()
    copy = store.archive(opt.project_name, "pre_edit", copy=True)
    insights_cache.save(copy, insights_cache.Saved({}, {}, None))
    assert store.delete_archive(copy)
    assert not os.path.exists(insights_cache.path(copy))


def test_a_crash_while_writing_leaves_the_old_file(monkeypatch):
    insights_cache.save("Patty", insights_cache.Saved({"b": {"state": {}, "widen": 2.0}}, {}, None))

    real = json.dump

    def boom(*a, **k):
        raise OSError("disk full")
    monkeypatch.setattr(json, "dump", boom)
    with pytest.raises(OSError):
        insights_cache.save("Patty", insights_cache.Saved({}, {}, None))
    monkeypatch.setattr(json, "dump", real)
    assert insights_cache.load("Patty").memo == {"b": {"state": {}, "widen": 2.0}}
    assert not os.path.exists(insights_cache.path("Patty") + ".tmp")
