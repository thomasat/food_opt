"""App-level flows for the yogurt samples."""
import json
import os
from pathlib import Path
import pytest
from streamlit.testing.v1 import AppTest
import sample_projects as sp
import wording

APP = str(Path(__file__).resolve().parent.parent / "app.py")


@pytest.fixture
def at(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    app = AppTest.from_file(APP, default_timeout=120)
    app.run()
    return app


def _click(app, label):
    [b for b in app.button if b.label == label][0].click().run()


def _write_own(name="Mine"):
    """An own project with the pilot's contents. The file carries its own
    name: a project's stored name is the one it opens under, so a copy that
    kept the sample's name would still be the sample."""
    state = sp.shipped_state(sp.ALL10)
    state["project_name"] = name
    Path(f"{name}.pkl").write_text(json.dumps(state))


def _open(app, name):
    """Choose `name` in Open project and press Open, as the sidebar does."""
    app.selectbox(key="project_select").set_value(name).run()
    # Open is drawn only while it would switch: a project already open
    # (the only one on disk is opened at launch) has none.
    if any(b.label == wording.OPEN_BUTTON for b in app.button):
        _click(app, wording.OPEN_BUTTON)


def test_welcome_try_sample_creates_both_and_opens_round6_on_review(at):
    _click(at, wording.OPEN_YOGURT_LABEL)
    assert Path(f"{sp.ROUND6}.pkl").exists() and Path(f"{sp.ALL10}.pkl").exists()
    assert at.session_state["_loaded_project"] == sp.ROUND6
    assert at.session_state["main_tab"] == wording.TAB_INSIGHTS


def test_try_sample_keeps_a_used_sample_as_it_is(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    state = sp.shipped_state(sp.ROUND6)
    state["tasting_noise"] = 2.0
    Path(f"{sp.ROUND6}.pkl").write_text(json.dumps(state))
    before = Path(f"{sp.ROUND6}.pkl").read_text()
    app = AppTest.from_file(APP, default_timeout=120); app.run()
    _click(app, wording.OPEN_YOGURT_LABEL)
    assert app.session_state["_loaded_project"] == sp.ROUND6
    assert Path(f"{sp.ALL10}.pkl").exists()          # written alongside
    assert Path(f"{sp.ROUND6}.pkl").read_text() == before
    assert json.loads(Path(f"{sp.ROUND6}.pkl").read_text())["tasting_noise"] == 2.0


def test_start_this_sample_over_restores_and_archives(at):
    _click(at, wording.OPEN_YOGURT_LABEL)
    st = json.loads(Path(f"{sp.ROUND6}.pkl").read_text())
    st["tasting_noise"] = 3.0
    Path(f"{sp.ROUND6}.pkl").write_text(json.dumps(st))
    at.run()
    _click(at, wording.START_SAMPLE_OVER_LABEL)
    _click(at, wording.YES_START_SAMPLE_OVER)
    assert "tasting_noise" not in json.loads(Path(f"{sp.ROUND6}.pkl").read_text())
    assert Path(f"{sp.ROUND6}_sample_reset.pkl").exists()
    assert at.session_state["main_tab"] == wording.TAB_INSIGHTS
    # One line: what happened and where the reader's version went.
    assert [s.value for s in at.success] == [wording.sample_restored(sp.ROUND6)]
    assert [i.value for i in at.info] == []
    assert Path(sp.ROUND6_RESULTS_FILE_NAME).read_bytes() == sp.round6_results_bytes()


def test_the_restored_line_names_the_sample_and_the_copy():
    line = wording.sample_restored(sp.ROUND6)
    assert line == (f"{sp.ROUND6} is back as shipped. "
                    "Your version is in Saved copies.")
    # Owner-approved wording: two short statements, each within the status cap.
    for sentence in line.replace(sp.ROUND6, "X").split(". "):
        assert len(sentence.split()) <= 10


def test_own_project_keeps_clear_contents_not_sample_restore(at):
    _write_own()
    at.run()
    _open(at, "Mine")
    assert not at.exception
    assert at.session_state["_loaded_project"] == "Mine"
    labels = [b.label for b in at.button]
    assert wording.START_OVER_LABEL in labels
    assert wording.START_SAMPLE_OVER_LABEL not in labels


def test_provenance_caption_only_on_samples(at):
    _click(at, wording.OPEN_YOGURT_LABEL)
    assert wording.SAMPLE_PROVENANCE_CAPTION in [c.value for c in at.sidebar.caption]
    _write_own()
    at.run()
    _open(at, "Mine")
    assert at.session_state["_loaded_project"] == "Mine"
    assert wording.SAMPLE_PROVENANCE_CAPTION not in [c.value for c in at.sidebar.caption]


def test_welcome_still_offers_the_template(at):
    assert wording.DOWNLOAD_TEMPLATE in [b.label for b in at.get("download_button")]


def test_open_next_opens_all10_on_review(at):
    _click(at, wording.OPEN_YOGURT_LABEL)
    at.session_state[sp.OPEN_NEXT_KEY] = True
    at.run()
    assert not at.exception
    assert at.session_state["_loaded_project"] == sp.ALL10
    assert at.session_state["main_tab"] == wording.TAB_INSIGHTS
    assert sp.OPEN_NEXT_KEY not in at.session_state


# A sample reopens where the demo starts: 4 · Review, whatever Round is open.

@pytest.mark.parametrize("name", sp.YOGURT_NAMES)
def test_reopening_a_sample_from_the_sidebar_lands_on_review(at, name):
    _click(at, wording.OPEN_YOGURT_LABEL)
    _write_own()
    at.run()
    _open(at, "Mine")
    assert at.session_state["_loaded_project"] == "Mine"
    _open(at, name)
    assert not at.exception
    assert at.session_state["_loaded_project"] == name
    assert at.session_state["main_tab"] == wording.TAB_INSIGHTS


@pytest.mark.parametrize("name", sp.YOGURT_NAMES)
def test_launching_on_a_sample_lands_on_review(tmp_path, monkeypatch, name):
    """A fresh run with the sample as the project to open, the way launch
    sets _loaded_project and _land_on_open for the most recent project."""
    monkeypatch.chdir(tmp_path)
    sp.ensure(__import__("storage").LocalStorage())
    app = AppTest.from_file(APP, default_timeout=120)
    app.session_state["_loaded_project"] = name
    app.session_state["_land_on_open"] = True
    app.run()
    assert not app.exception
    assert app.session_state["main_tab"] == wording.TAB_INSIGHTS


def test_launch_on_the_most_recent_sample_lands_on_review(tmp_path, monkeypatch):
    import os
    monkeypatch.chdir(tmp_path)
    sp.ensure(__import__("storage").LocalStorage())
    os.utime(f"{sp.ROUND6}.pkl", (2e9, 2e9))      # the most recent project
    app = AppTest.from_file(APP, default_timeout=120)
    app.run()
    assert not app.exception
    assert app.session_state["_loaded_project"] == sp.ROUND6
    assert app.session_state["main_tab"] == wording.TAB_INSIGHTS


def test_own_project_with_an_open_round_still_lands_on_make_a_round(at):
    """The same contents under an own name keep the landing rule."""
    from ui_helpers import TAB_BATCH
    _click(at, wording.OPEN_YOGURT_LABEL)
    state = sp.shipped_state(sp.ROUND6)           # Round 6 open, 5 recorded
    state["project_name"] = "Mine"
    Path("Mine.pkl").write_text(json.dumps(state))
    at.run()
    _open(at, "Mine")
    assert not at.exception
    assert at.session_state["_loaded_project"] == "Mine"
    assert at.session_state["main_tab"] == TAB_BATCH


def test_try_sample_opens_round6_when_the_results_file_cannot_be_written(at, monkeypatch):
    def boom(self, data):
        raise PermissionError("read-only")
    monkeypatch.setattr(Path, "write_bytes", boom)
    _click(at, wording.OPEN_YOGURT_LABEL)
    assert not at.exception and not at.error
    assert at.session_state["_loaded_project"] == sp.ROUND6
    assert at.session_state["main_tab"] == wording.TAB_INSIGHTS
    assert Path(f"{sp.ALL10}.pkl").exists()
    assert not Path(sp.ROUND6_RESULTS_FILE_NAME).exists()


def test_start_this_sample_over_when_the_results_file_cannot_be_written(at):
    _click(at, wording.OPEN_YOGURT_LABEL)
    st = json.loads(Path(f"{sp.ROUND6}.pkl").read_text())
    st["tasting_noise"] = 3.0
    Path(f"{sp.ROUND6}.pkl").write_text(json.dumps(st))
    Path(sp.ROUND6_RESULTS_FILE_NAME).unlink()
    Path(sp.ROUND6_RESULTS_FILE_NAME).mkdir()
    at.run()
    _click(at, wording.START_SAMPLE_OVER_LABEL)
    _click(at, wording.YES_START_SAMPLE_OVER)
    assert not at.exception and not at.error
    assert "tasting_noise" not in json.loads(Path(f"{sp.ROUND6}.pkl").read_text())
    assert [s.value for s in at.success] == [wording.sample_restored(sp.ROUND6)]
    assert at.session_state["main_tab"] == wording.TAB_INSIGHTS


def test_first_launch_brings_an_untouched_earlier_sample_up_to_date(tmp_path, monkeypatch):
    """A copy an earlier build wrote, never changed, is the current sample
    from the first run on: the project the app reopens shows the pilot's
    own limits. A changed copy is left as it is."""
    from tests.test_sample_projects import _earlier_build
    monkeypatch.chdir(tmp_path)
    Path(f"{sp.ROUND6}.pkl").write_text(json.dumps(_earlier_build(sp.ROUND6), indent=2))
    changed = _earlier_build(sp.ALL10)
    changed["tasting_noise"] = 2.0
    Path(f"{sp.ALL10}.pkl").write_text(json.dumps(changed, indent=2))
    os.utime(f"{sp.ROUND6}.pkl", (1000, 1000))
    os.utime(f"{sp.ALL10}.pkl", (2000, 2000))   # the reader's last project
    app = AppTest.from_file(APP, default_timeout=120)
    app.run()
    assert not app.exception
    assert json.loads(Path(f"{sp.ROUND6}.pkl").read_text()) == sp.shipped_state(sp.ROUND6)
    assert json.loads(Path(f"{sp.ALL10}.pkl").read_text()) == changed
    # The replaced copy is newer on disk now, but the project the reader
    # last worked in is the one that reopens.
    assert app.session_state["_loaded_project"] == sp.ALL10
    _open(app, sp.ROUND6)
    assert app.session_state["_loaded_project"] == sp.ROUND6
    opt = app.session_state["optimizer"]
    assert opt.quantity_constraints[0]["max"] == 98.69051


def test_own_newest_project_reopens_after_an_earlier_sample_is_replaced(tmp_path, monkeypatch):
    """Replacing an earlier build's sample gives its file a new modified
    time; the project the reader last worked in still reopens, not it."""
    from tests.test_sample_projects import _earlier_build
    monkeypatch.chdir(tmp_path)
    Path(f"{sp.ROUND6}.pkl").write_text(json.dumps(_earlier_build(sp.ROUND6), indent=2))
    _write_own("Mine")
    os.utime(f"{sp.ROUND6}.pkl", (1000, 1000))
    os.utime("Mine.pkl", (2000, 2000))
    app = AppTest.from_file(APP, default_timeout=120)
    app.run()
    assert not app.exception
    assert json.loads(Path(f"{sp.ROUND6}.pkl").read_text()) == sp.shipped_state(sp.ROUND6)
    assert app.session_state["_loaded_project"] == "Mine"


def test_a_damaged_shipped_sample_does_not_stop_launch(tmp_path, monkeypatch):
    """The launch-time refresh never crashes the app: a shipped sample file
    that cannot be read leaves the earlier copy on disk, opening as it is."""
    from tests.test_sample_projects import _earlier_build
    monkeypatch.chdir(tmp_path)
    earlier = _earlier_build(sp.ROUND6)
    Path(f"{sp.ROUND6}.pkl").write_text(json.dumps(earlier, indent=2))
    damaged = tmp_path / "damaged.json"
    damaged.write_text("{ not json")
    monkeypatch.setattr(sp, "_shipped_path", lambda name: damaged)
    app = AppTest.from_file(APP, default_timeout=120)
    app.run()
    assert not app.exception
    assert json.loads(Path(f"{sp.ROUND6}.pkl").read_text()) == earlier
    assert app.session_state["_loaded_project"] == sp.ROUND6
