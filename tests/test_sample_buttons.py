"""The yogurt pilot samples' two buttons (0.8.0, Task 3): on the Round 6
sample, Download the pilot's Round 6 results inside the upload fold of
2 · Make a round while the pilot's Round 6 is open; and, once Round 6 is
recorded, See what happened next at the end of What next? on 4 · Review,
which asks app.py to open the all-10 sample (sample_projects.OPEN_NEXT_KEY).

The pilot data is public; the samples are the shipped files."""
import importlib.util
import json

import pytest
from streamlit.testing.v1 import AppTest

import streamlit as st

import sample_projects as sp
import storage as storage_backend
import ui_insights as U
import ui_setup
import wording
from food_bo import WORKBOOK_MIME, FoodOptimizer
from tests.test_app_ui import APP_PATH, _unknowns


@pytest.fixture(autouse=True)
def _samples(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)          # LocalStorage writes into the cwd
    sp.ensure(storage_backend.LocalStorage())


def _builder():
    spec = importlib.util.spec_from_file_location("yogurt_build", sp.SAMPLE_DIR / "build.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _record_round6(make_round7=False):
    """Record results > Save results for the pilot's Round 6, as the builder
    does; then, if asked, the pilot's Round 7 added as the open Round."""
    b = _builder()
    rounds = b.pilot()
    opt = FoodOptimizer(sp.ROUND6)
    b.score_round(opt, rounds[6])
    if make_round7:
        b.add_round(opt, rounds[7])
    opt.save()
    return opt


def _app(name, tab):
    at = AppTest.from_file(APP_PATH, default_timeout=900)
    at.session_state["_loaded_project"] = name
    at.run()
    at.session_state["main_tab"] = tab
    at.run()
    assert not at.exception, [e.message for e in at.exception]
    return at


def _downloads(at):
    return [d for d in _unknowns(at.tabs[1], "download_button")
            if d.label == wording.DOWNLOAD_ROUND6_RESULTS]


def _in_order(node):
    """(type, key or text) for every element under node, in render order."""
    out = []

    def walk(n):
        children = getattr(n, "children", None) or {}
        if hasattr(children, "values"):
            children = children.values()
        for c in children:
            kind = getattr(c, "type", None)
            if kind in ("button", "selectbox"):
                out.append((kind, getattr(c, "key", None)))
            elif kind == "caption":
                out.append((kind, c.value))
            walk(c)
    walk(node)
    return out


def _what_next(at):
    return [b for b in at.tabs[3].button if b.label == wording.SEE_WHAT_HAPPENED_NEXT]


def test_the_wording():
    assert wording.DOWNLOAD_ROUND6_RESULTS == "Download the pilot's Round 6 results"
    # Review v3: the button says it; no caption under it.
    assert wording.SEE_WHAT_HAPPENED_NEXT == "See the pilot's Rounds 7–10"
    assert not hasattr(wording, "SEE_WHAT_HAPPENED_NEXT_CAPTION")


def test_round6_open_offers_the_pilots_results_file(monkeypatch):
    # The proto carries no file name: a spy on st.download_button reads
    # what the app passed.
    calls = []
    real = st.download_button

    def spy(*a, **k):
        calls.append((a, k))
        return real(*a, **k)

    monkeypatch.setattr(st, "download_button", spy)
    at = _app(sp.ROUND6, wording.TAB_BATCH)
    [dl] = _downloads(at)
    assert "sample_round6_results" in dl.proto.id
    ours = [c for c in calls if c[1].get("key") == "sample_round6_results"]
    assert ours                          # once per run, the same each time
    a, k = ours[-1]
    assert a[0] == wording.DOWNLOAD_ROUND6_RESULTS
    assert k["file_name"] == sp.ROUND6_RESULTS_FILE_NAME
    assert k["data"] == sp.round6_results_bytes()
    assert k["mime"] == WORKBOOK_MIME
    assert dl.proto.type != "primary"
    # Inside the upload fold, right after the uploader.
    fold = next(e for e in at.tabs[1].expander if e.label == wording.UPLOAD_EXPANDER)
    kinds = [type(c).__name__ == "UnknownElement" and c.type
             for c in fold.children.values()]
    up = kinds.index("file_uploader")
    assert kinds[up + 1] == "download_button"


def test_the_download_points_at_the_shipped_file():
    assert sp.round6_results_bytes()[:2] == b"PK"           # an xlsx
    assert sp.ROUND6_RESULTS_FILE_NAME.endswith(".xlsx")


def test_the_all10_sample_has_no_download():
    at = _app(sp.ALL10, wording.TAB_BATCH)
    assert not _downloads(at)


def test_no_what_next_button_before_round6_is_recorded():
    at = _app(sp.ROUND6, wording.TAB_INSIGHTS)
    assert not _what_next(at)


def test_after_round6_see_what_happened_next_asks_for_the_all10_sample():
    _record_round6()
    at = _app(sp.ROUND6, wording.TAB_INSIGHTS)
    [btn] = _what_next(at)
    assert btn.proto.type != "primary"
    assert btn.key == "sample_what_next"
    # Review v3: What next? is buttons only — the primary, then this one.
    order = _in_order(at.tabs[3])
    i = order.index(("button", "sample_what_next"))
    assert not [k for k in order if k[0] == "selectbox" and "noise" in str(k[1])]
    assert order[i - 1] == ("button", "insights_foot")
    # AppTest sends no tab with a click: the tab is set again, as
    # tests/test_final_demo.py does after its selectbox.
    btn.click()
    at.session_state["main_tab"] = wording.TAB_INSIGHTS
    at.run()
    assert not at.exception, [e.message for e in at.exception]
    assert at.session_state["_loaded_project"] == sp.ALL10
    assert at.session_state["main_tab"] == wording.TAB_INSIGHTS
    assert sp.OPEN_NEXT_KEY not in at.session_state


def test_round7_made_drops_the_download_and_keeps_what_next():
    _record_round6(make_round7=True)
    at = _app(sp.ROUND6, wording.TAB_BATCH)
    assert not _downloads(at)
    at.session_state["main_tab"] = wording.TAB_INSIGHTS
    at.run()
    assert not at.exception, [e.message for e in at.exception]
    assert len(_what_next(at)) == 1


def test_what_next_recreates_a_deleted_all10_sample(tmp_path):
    _record_round6()
    (tmp_path / f"{sp.ALL10}.pkl").unlink()
    at = _app(sp.ROUND6, wording.TAB_INSIGHTS)
    [btn] = _what_next(at)
    btn.click()
    at.session_state["main_tab"] = wording.TAB_INSIGHTS
    at.run()
    assert not at.exception, [e.message for e in at.exception]
    assert (tmp_path / f"{sp.ALL10}.pkl").is_file()
    assert at.session_state["_loaded_project"] == sp.ALL10
    assert at.session_state["main_tab"] == wording.TAB_INSIGHTS


def _stop_line_help(at, line):
    return next(m for m in at.tabs[3].markdown if m.value == line).proto.help


def test_no_wording_string_says_more_settings():
    """The Set up tab's second tier is labelled MORE_SETTINGS_EXPANDER on
    screen; no wording text may call it anything else."""
    for name, value in vars(wording).items():
        if isinstance(value, str) and name.isupper():
            assert "More settings" not in value, name
    assert wording.MORE_SETTINGS_EXPANDER in wording.SAMPLE_PROVENANCE_HELP
