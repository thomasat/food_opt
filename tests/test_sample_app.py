"""App-level flows for the yogurt samples."""
import json
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
    _click(app, wording.OPEN_BUTTON)


def test_welcome_try_sample_creates_both_and_opens_round6_on_review(at):
    _click(at, wording.TRY_SAMPLE_LABEL)
    assert Path(f"{sp.ROUND6}.pkl").exists() and Path(f"{sp.ALL10}.pkl").exists()
    assert at.session_state["_loaded_project"] == sp.ROUND6
    assert at.session_state["main_tab"] == wording.TAB_INSIGHTS


def test_try_sample_keeps_a_used_sample_as_it_is(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    state = sp.shipped_state(sp.ROUND6)
    state["tasting_noise"] = 2.0
    Path(f"{sp.ROUND6}.pkl").write_text(json.dumps(state))
    app = AppTest.from_file(APP, default_timeout=120); app.run()
    _click(app, wording.TRY_SAMPLE_LABEL)
    assert json.loads(Path(f"{sp.ROUND6}.pkl").read_text())["tasting_noise"] == 2.0


def test_start_this_sample_over_restores_and_archives(at):
    _click(at, wording.TRY_SAMPLE_LABEL)
    st = json.loads(Path(f"{sp.ROUND6}.pkl").read_text())
    st["tasting_noise"] = 3.0
    Path(f"{sp.ROUND6}.pkl").write_text(json.dumps(st))
    at.run()
    _click(at, wording.START_SAMPLE_OVER_LABEL)
    _click(at, wording.YES_START_SAMPLE_OVER)
    assert "tasting_noise" not in json.loads(Path(f"{sp.ROUND6}.pkl").read_text())
    assert Path(f"{sp.ROUND6}_pre_restore.pkl").exists()
    assert at.session_state["main_tab"] == wording.TAB_INSIGHTS


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
    _click(at, wording.TRY_SAMPLE_LABEL)
    assert wording.SAMPLE_PROVENANCE_CAPTION in [c.value for c in at.sidebar.caption]
    _write_own()
    at.run()
    _open(at, "Mine")
    assert at.session_state["_loaded_project"] == "Mine"
    assert wording.SAMPLE_PROVENANCE_CAPTION not in [c.value for c in at.sidebar.caption]


def test_welcome_still_offers_the_template(at):
    assert wording.DOWNLOAD_TEMPLATE in [b.label for b in at.get("download_button")]


def test_open_next_opens_all10_on_review(at):
    _click(at, wording.TRY_SAMPLE_LABEL)
    at.session_state[sp.OPEN_NEXT_KEY] = True
    at.run()
    assert not at.exception
    assert at.session_state["_loaded_project"] == sp.ALL10
    assert at.session_state["main_tab"] == wording.TAB_INSIGHTS
    assert sp.OPEN_NEXT_KEY not in at.session_state
