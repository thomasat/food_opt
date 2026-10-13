"""The yogurt pilot samples: shipped data, provenance and install/restore."""
import json, os, re, importlib.util
from pathlib import Path
import pytest
import sample_projects as sp
import storage as storage_backend
from food_bo import FoodOptimizer

# A timestamp, or a bare date (pending_batch_created is date-only).
TS = re.compile(r'"20\d\d-\d\d-\d\d(T[^"]*)?"')


def _norm(text):
    return TS.sub('"<ts>"', text)


def _builder():
    spec = importlib.util.spec_from_file_location("yogurt_build", sp.SAMPLE_DIR / "build.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


@pytest.fixture
def store(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)          # LocalStorage writes into the cwd
    return storage_backend.LocalStorage()


def test_every_shipped_file_exists():
    assert sp.available()
    for name in sp.NAMES:
        assert (sp.SAMPLE_DIR / f"{name}.json").is_file()
    assert sp.ROUND6_RESULTS.is_file()
    assert sp.ROUND6_RESULTS.name == sp.ROUND6_RESULTS_FILE_NAME == "Round 6 results (pilot).xlsx"


@pytest.mark.parametrize("name,scored,open_round", [
    (sp.ROUND6, 5, 6), (sp.ALL10, 10, None)])
def test_shipped_project_equals_a_rebuild_from_the_csvs(tmp_path, name, scored, open_round):
    _builder().build(name, scored, open_round, out_dir=str(tmp_path))
    rebuilt = (tmp_path / f"{name}.pkl").read_text()
    shipped = (sp.SAMPLE_DIR / f"{name}.json").read_text()
    assert _norm(rebuilt) == _norm(shipped)


@pytest.mark.parametrize("name,n,open_rows", [(sp.ROUND6, 15, 3), (sp.ALL10, 30, 0)])
def test_shipped_projects_load_at_this_class_version(store, name, n, open_rows):
    FoodOptimizer.validate_state(sp.shipped_state(name))
    sp.ensure(store)
    opt = FoodOptimizer(name, storage=store)
    assert not opt.load_error
    assert len(opt.Y_history) == n
    assert len(opt.pending_batch or []) == open_rows
    assert opt.tasting_noise is None


def test_is_sample_is_an_exact_match():
    assert sp.is_sample(sp.ROUND6) and sp.is_sample(sp.ALL10)
    for other in ("Sample project", sp.ROUND6 + " copy", sp.ROUND6 + "_pre_restore",
                  "yogurt pilot - round 6 to score", ""):
        assert not sp.is_sample(other)


def test_ensure_writes_only_what_is_missing(store):
    assert sorted(sp.ensure(store)) == sorted(sp.NAMES)
    Path(f"{sp.ALL10}.pkl").unlink()
    before = Path(f"{sp.ROUND6}.pkl").stat().st_mtime_ns
    assert sp.ensure(store) == [sp.ALL10]
    assert Path(f"{sp.ROUND6}.pkl").stat().st_mtime_ns == before
    assert sp.ensure(store) == []


def test_ensure_never_overwrites_user_work(store):
    sp.ensure(store)
    opt = FoodOptimizer(sp.ROUND6, storage=store)
    opt.tasting_noise = 2.0
    opt.save()
    sp.ensure(store)
    assert FoodOptimizer(sp.ROUND6, storage=store).tasting_noise == 2.0


def test_restore_archives_then_writes_the_shipped_copy(store):
    sp.ensure(store)
    opt = FoodOptimizer(sp.ROUND6, storage=store)
    opt.tasting_noise = 3.0
    opt.save()
    Path("My own project.pkl").write_text('{"keep": true}')
    first = sp.restore(sp.ROUND6, store)
    second = sp.restore(sp.ROUND6, store)
    assert first == f"{sp.ROUND6}_pre_restore"
    assert second == f"{sp.ROUND6}_pre_restore_1"
    assert FoodOptimizer(sp.ROUND6, storage=store).tasting_noise is None
    assert json.loads(Path(f"{first}.pkl").read_text())["tasting_noise"] == 3.0
    assert Path("My own project.pkl").read_text() == '{"keep": true}'
    assert Path(f"{sp.ALL10}.pkl").exists()


def test_restore_without_a_file_just_writes_it(store):
    assert sp.restore(sp.ALL10, store) is None
    assert Path(f"{sp.ALL10}.pkl").exists()


def test_restore_refuses_a_non_sample(store):
    with pytest.raises(ValueError):
        sp.restore("My own project", store)


def test_round6_download_offer(store):
    sp.ensure(store)
    r6 = FoodOptimizer(sp.ROUND6, storage=store)
    assert sp.offers_round6_download(r6)
    assert not sp.offers_what_next(r6)
    assert not sp.offers_round6_download(FoodOptimizer(sp.ALL10, storage=store))
    assert sp.round6_results_bytes()[:2] == b"PK"      # an xlsx is a zip


def test_after_round6_is_recorded(store):
    sp.ensure(store)
    r6 = FoodOptimizer(sp.ROUND6, storage=store)
    b = _builder()
    b.score_round(r6, b.pilot()[6])
    assert not sp.offers_round6_download(r6)
    assert sp.offers_what_next(r6)
    b.add_round(r6, b.pilot()[7])           # the user goes on to Round 7
    assert not sp.offers_round6_download(r6)
    assert sp.offers_what_next(r6)
