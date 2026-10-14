"""The yogurt pilot samples: shipped data, provenance and install/restore."""
import json, re, importlib.util
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



@pytest.mark.parametrize("name", [sp.ROUND6, sp.ALL10])
def test_shipped_projects_carry_no_date(name):
    """The pilot's CSVs date no Batch, so the shipped files carry no date at
    all: not the day build.py ran as the day a pilot row was recorded, nor
    as the day the open Round 6 was opened."""
    text = (sp.SAMPLE_DIR / f"{name}.json").read_text()
    assert not TS.search(text)
    state = json.loads(text)
    assert state["timestamps_history"] == [None] * len(state["results_history"])
    assert state["pending_batch_created"] is None


def _today_range(before, after):
    return {before.astimezone().strftime("%Y-%m-%d"),
            after.astimezone().strftime("%Y-%m-%d")}


@pytest.mark.parametrize("name,n", [(sp.ROUND6, 15), (sp.ALL10, 30)])
def test_pilot_rows_show_a_blank_date_recorded(store, name, n):
    """3 · Results (history_frame, as ui_results draws it), the workbook and
    the CSV (history_export_frame) show every pilot row's Date recorded as
    a blank cell, never the build date, None or NaT."""
    import wording
    sp.ensure(store)
    opt = FoodOptimizer(name, storage=store)
    shown = opt.history_frame(include_amounts=True)
    assert len(shown) == n
    assert list(shown[wording.DATE_RECORDED_COLUMN]) == [""] * n
    exported = opt.history_export_frame()
    assert list(exported[wording.DATE_RECORDED_COLUMN]) == [""] * n
    csv = opt.history_csv()
    for bad in ("None", "NaT", "nan"):
        assert bad not in csv
    assert not re.search(r"20\d\d-\d\d-\d\d", csv)
    assert opt.all_formulations_workbook()[:2] == b"PK"


def test_a_result_recorded_now_gets_todays_date(store):
    """Results the user records in the app are stamped as before: Round 6
    scored now shows today on 3 · Results, beside 15 blank pilot dates.
    Round 6's sheets, opened by the pilot on no known day, print today's
    date, as does a formulation of the user's own added to it."""
    from datetime import datetime, timezone
    import wording
    sp.ensure(store)
    r6 = FoodOptimizer(sp.ROUND6, storage=store)
    before = datetime.now(timezone.utc)
    assert r6._sheet_date() in _today_range(before, datetime.now(timezone.utc))
    r6.add_to_pending_batch(dict(r6.pending_batch[0]["recipe"]), "My own")
    assert r6.pending_batch_created in _today_range(before, datetime.now(timezone.utc))
    b = _builder()
    by_note = {b.note_of(x): res for x, _, res, _ in b.pilot()[6]}
    by_note["My own"] = by_note[r6.pending_batch[0]["note"]]
    for row in list(r6.pending_batch):
        r6.tell(row["recipe"], dict(by_note[row["note"]]),
                formulation_no=row["formulation"], batch_no=6, note=row["note"])
    r6.set_pending_batch(None)
    today = _today_range(before, datetime.now(timezone.utc))
    frame = r6.history_frame(order=wording.SORT_BATCH_ORDER)
    dates = dict(zip(frame["Formulation"], frame[wording.DATE_RECORDED_COLUMN]))
    assert all(dates[no] == "" for no in range(1, 16))
    assert len(dates) == 19 and all(dates[no] in today for no in range(16, 20))
    reloaded = FoodOptimizer(sp.ROUND6, storage=store)
    exported = reloaded.history_export_frame()
    assert sorted(no for no, d in zip(exported["Formulation"],
                                      exported[wording.DATE_RECORDED_COLUMN])
                  if d in today) == [16, 17, 18, 19]


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
                  sp.ROUND6 + "_sample_reset", "yogurt pilot - round 6 to score", ""):
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
    assert first == f"{sp.ROUND6}_sample_reset"
    assert second == f"{sp.ROUND6}_sample_reset_1"
    assert FoodOptimizer(sp.ROUND6, storage=store).tasting_noise is None
    assert json.loads(Path(f"{first}.pkl").read_text())["tasting_noise"] == 3.0
    assert Path("My own project.pkl").read_text() == '{"keep": true}'
    assert Path(f"{sp.ALL10}.pkl").exists()


def test_the_restored_copy_is_a_saved_copy_not_a_project(store):
    """The copy Start this sample over keeps is listed with the project's
    Saved copies, under its own reason, and never as a project of its own."""
    import wording
    sp.ensure(store)
    archived = sp.restore(sp.ROUND6, store)
    assert archived in store.list_archives()
    assert archived not in store.list_projects()
    assert sorted(store.list_projects()) == sorted(sp.NAMES)
    match = storage_backend.ARCHIVE_SUFFIX_RE.search(archived)
    assert storage_backend.ARCHIVE_SUFFIX_RE.sub("", archived) == sp.ROUND6
    assert wording.SAFETY_COPY_REASONS[match.group(1)] == "Before Start this sample over"


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


def test_an_own_formulation_added_to_round6_keeps_the_download(store):
    sp.ensure(store)
    r6 = FoodOptimizer(sp.ROUND6, storage=store)
    r6.add_to_pending_batch(dict(r6.pending_batch[0]["recipe"]), "My own")
    assert r6.pending_batch_no == 6 and len(r6.pending_batch) == 4
    assert sp.offers_round6_download(r6)
    assert not sp.offers_what_next(r6)


def test_a_round6_row_not_scored_still_offers_what_next(store):
    sp.ensure(store)
    r6 = FoodOptimizer(sp.ROUND6, storage=store)
    b = _builder()
    by_note = {b.note_of(x): res for x, _, res, _ in b.pilot()[6]}
    *scored, last = list(r6.pending_batch)
    for row in scored:
        r6.tell(row["recipe"], dict(by_note[row["note"]]),
                formulation_no=row["formulation"], batch_no=6, note=row["note"])
    assert not sp.offers_what_next(r6)          # Round 6 still open
    r6.record_skipped(last["formulation"], 6, last["recipe"])
    r6.set_pending_batch(None)
    assert len(r6.X_history) == 17
    assert not sp.offers_round6_download(r6)
    assert sp.offers_what_next(r6)


def test_round7_open_hides_the_download(store):
    sp.ensure(store)
    r6 = FoodOptimizer(sp.ROUND6, storage=store)
    b = _builder()
    b.score_round(r6, b.pilot()[6])
    b.add_round(r6, b.pilot()[7])
    assert r6.pending_batch_no == 7 and r6.pending_batch
    assert not sp.offers_round6_download(r6)
    assert sp.offers_what_next(r6)


def test_the_all10_sample_offers_neither(store):
    sp.ensure(store)
    a10 = FoodOptimizer(sp.ALL10, storage=store)
    assert not sp.offers_round6_download(a10)
    assert not sp.offers_what_next(a10)


def test_uploading_the_shipped_round6_file_records_the_pilots_results(store):
    """Spec 4.4: the bundled xlsx, read and saved the way 2 · Make a round's
    Or upload results from a file > Save uploaded results does
    (ui_batch._read_results_file, parse_batch_results, tell)."""
    from food_bo import uploaded_parts
    sp.ensure(store)
    opt = FoodOptimizer(sp.ROUND6, storage=store)
    with open(sp.ROUND6_RESULTS, "rb") as fh:
        upload = opt.results_from_workbook(fh)
    sheet, weighed, lots = uploaded_parts(upload)
    parsed, left_out = opt.parse_batch_results(
        sheet, opt.pending_batch, with_skipped=True, weighed=weighed)
    assert [no for no, _, _ in parsed] == [16, 17, 18] and left_out == []
    batch_no = opt.pending_batch_no
    by_number = {r["formulation"]: r["recipe"] for r in opt.pending_batch}
    for number, results, note in parsed:
        opt.tell(opt.amounts_as_weighed(by_number[number], weighed.get(number)),
                 results, formulation_no=number, batch_no=batch_no, note=note)
    opt.store_lots(batch_no, lots)
    opt.set_pending_batch(None)

    names = ("Consistency", "Creaminess", "Tangy", "Thickness uniformity")
    opt = FoodOptimizer(sp.ROUND6, storage=store)
    got = {no: tuple(opt.results_history[opt.index_of_formulation(no)][n] for n in names)
           for no in (16, 17, 18)}
    assert got == {16: (12, 5, 4, 13), 17: (12, 5, 4, 13), 18: (13, 7, 4, 12)}
    assert [opt.batch_history[opt.index_of_formulation(no)] for no in (16, 17, 18)] == [6, 6, 6]
    assert not opt.pending_batch                # Round 6 closed
    assert sp.offers_what_next(opt)
    assert not sp.offers_round6_download(opt)


# The Round 6 results file beside the projects, so Browse files lands on it.
_R6_COPY = "Round 6 results (pilot).xlsx"


def test_ensure_writes_the_round6_results_file_into_the_projects_folder(store, tmp_path):
    assert store.folder() == str(tmp_path.resolve())
    sp.ensure(store)
    copy = Path(store.folder()) / _R6_COPY
    assert copy.read_bytes() == sp.ROUND6_RESULTS.read_bytes()


def test_ensure_never_overwrites_a_changed_results_file(store):
    sp.ensure(store)
    Path(_R6_COPY).write_bytes(b"changed")
    Path(f"{sp.ALL10}.pkl").unlink()              # ensure has work to do
    assert sp.ensure(store) == [sp.ALL10]
    assert Path(_R6_COPY).read_bytes() == b"changed"
    sp.ensure(store)
    assert Path(_R6_COPY).read_bytes() == b"changed"


def test_ensure_writes_the_results_file_when_only_it_is_missing(store):
    sp.ensure(store)
    Path(_R6_COPY).unlink()
    assert sp.ensure(store) == []
    assert Path(_R6_COPY).read_bytes() == sp.ROUND6_RESULTS.read_bytes()


def test_restore_round6_rewrites_the_results_file(store):
    sp.ensure(store)
    Path(_R6_COPY).write_bytes(b"changed")
    sp.restore(sp.ROUND6, store)
    assert Path(_R6_COPY).read_bytes() == sp.ROUND6_RESULTS.read_bytes()
    Path(_R6_COPY).unlink()
    sp.restore(sp.ROUND6, store)
    assert Path(_R6_COPY).read_bytes() == sp.ROUND6_RESULTS.read_bytes()


def test_restore_all10_leaves_the_results_file(store):
    sp.ensure(store)
    Path(_R6_COPY).write_bytes(b"changed")
    sp.restore(sp.ALL10, store)
    assert Path(_R6_COPY).read_bytes() == b"changed"
    Path(_R6_COPY).unlink()
    sp.restore(sp.ALL10, store)
    assert not Path(_R6_COPY).exists()


def test_the_results_file_is_never_a_project_or_a_copy(store):
    sp.ensure(store)
    sp.restore(sp.ROUND6, store)
    assert Path(_R6_COPY).exists()
    listed = store.list_projects() + store.list_archives()
    assert not [n for n in listed if "results" in n.lower()]
    assert sorted(store.list_projects()) == sorted(sp.NAMES)


@pytest.mark.parametrize("doc", ["README.md", "desktop/start_here.txt", "desktop/README.md"])
def test_the_docs_say_what_the_screen_says(doc):
    """The walkthrough names the stop by the callout's own headline and the
    results file by the name it has in the FoodOptimizer folder."""
    import wording
    text = " ".join((Path(sp._ROOT) / doc).read_text().split())
    assert wording.STOP_CALLOUT_HEAD in text
    assert sp.ROUND6_RESULTS_FILE_NAME in text
    assert "Browse files" in text and "FoodOptimizer" in text
    assert "it says to stop, or change the target" not in text


def _xlsx_write_fails(monkeypatch):
    """Every write of the results file fails (a full or read-only disk);
    the project files are written by LocalStorage.save, not this."""
    def boom(self, data):
        raise PermissionError("read-only")
    monkeypatch.setattr(Path, "write_bytes", boom)


def test_a_failed_results_file_write_never_blocks_the_samples(store, monkeypatch):
    _xlsx_write_fails(monkeypatch)
    assert sorted(sp.ensure(store)) == sorted(sp.NAMES)
    assert all(Path(f"{n}.pkl").exists() for n in sp.NAMES)
    assert not Path(_R6_COPY).exists()


def test_a_failed_results_file_write_never_fails_a_restore(store, monkeypatch):
    sp.ensure(store)
    opt = FoodOptimizer(sp.ROUND6, storage=store)
    opt.tasting_noise = 3.0
    opt.save()
    Path(_R6_COPY).unlink()
    Path(_R6_COPY).mkdir()                      # a folder in its place
    assert sp.restore(sp.ROUND6, store) == f"{sp.ROUND6}_sample_reset"
    assert FoodOptimizer(sp.ROUND6, storage=store).tasting_noise is None
    Path(_R6_COPY).rmdir()
    _xlsx_write_fails(monkeypatch)              # or a disk that refuses it
    assert sp.restore(sp.ROUND6, store) == f"{sp.ROUND6}_sample_reset_1"
    assert FoodOptimizer(sp.ROUND6, storage=store).tasting_noise is None


def test_a_project_write_failure_still_raises(store, monkeypatch):
    def boom(name, state):
        raise OSError("disk full")
    monkeypatch.setattr(store, "save", boom)
    with pytest.raises(OSError):
        sp.ensure(store)
