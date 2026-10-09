"""Layout study (owner, 2026-10-09: "Doesn't the layout require us to scroll
up and down while using the slider, making it confusing?").

"Are we improving?" is one Replay panel: its Round control at the top
(◀ Round N of M ▶, Replay Rounds, then Latest, or Earlier Round and Back to
latest), the slim score chart over What changed on the left, the sensory
profile (a 340 px square) on the right — all of it inside one 1152 × 720
viewport, so a step never needs a scroll. An earlier Round is framed and
tinted. Replay steps Round 1 to the latest, one a second, and stops on the
latest. Leaving the tab, or another project, is the latest again; the
section's answer line and What next? stay about today.

Synthetic projects only."""
import inspect
import itertools
import json

import pytest
from streamlit.testing.v1 import AppTest

import ui_insights as U
import wording
from tests.test_app_ui import APP_PATH
from tests.test_review_slim import ROWS, _band_and_marker, _spider_opt, _specs

_n = itertools.count(1)


@pytest.fixture(autouse=True)
def _isolate_cwd(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)


@pytest.fixture
def state(monkeypatch):
    """A plain dict for st.session_state (the pure helpers, outside a run)."""
    d = {}
    monkeypatch.setattr(U.st, "session_state", d)
    return d


class _Opt:
    def __init__(self, name):
        self.project_name = name


def _review(opt):
    opt.save()
    at = AppTest.from_file(APP_PATH, default_timeout=600)
    at.session_state["_loaded_project"] = opt.project_name
    at.run()
    at.session_state["main_tab"] = wording.TAB_INSIGHTS
    at.run()
    assert not at.exception, [e.message for e in at.exception]
    return at


def _click(at, key, tab=wording.TAB_INSIGHTS):
    at.button(key=key).click()
    at.session_state["main_tab"] = tab
    at.run()
    assert not at.exception, [e.message for e in at.exception]
    return [m.value for m in at.tabs[3].markdown]


def _keys(at):
    return {b.key for b in at.tabs[3].button}


# ------------------------------------------------------------------ #
#  The Round on view: latest by default, per project
# ------------------------------------------------------------------ #

def test_the_latest_unless_an_earlier_round_is_chosen(state):
    a, rounds = _Opt("A"), [1, 2, 3, 4]
    assert U.shown_round(a, rounds) == 4
    U.show_round(a, rounds, 2)
    assert U.shown_round(a, rounds) == 2
    U.show_round(a, rounds, 4)                     # the latest is stored as nothing
    assert U.REPLAY_STATE not in state and U.shown_round(a, rounds) == 4
    U.show_round(a, rounds, 3)
    U.show_round(a, rounds, None)                  # Back to latest
    assert U.shown_round(a, rounds) == 4
    assert U.shown_round(a, []) is None


def test_another_project_is_the_latest_again(state):
    a, b = _Opt("A"), _Opt("B")
    U.show_round(a, [1, 2, 3, 4], 2)
    assert U.shown_round(b, [1, 2, 3, 4]) == 4     # the reset rule: per project
    assert U.shown_round(a, [1, 2, 3, 4]) == 2
    # A Round the project no longer has (fewer Rounds now): the latest.
    assert U.shown_round(a, [1]) == 1


def test_a_newly_scored_round_shows_when_on_the_latest(state):
    a = _Opt("A")
    assert U.shown_round(a, [1, 2, 3, 4, 5]) == 5     # nothing stored: follows the data
    U.show_round(a, [1, 2, 3, 4, 5], 5)
    assert U.shown_round(a, [1, 2, 3, 4, 5, 6]) == 6


def test_leaving_the_review_resets_and_stops(state):
    a, rounds = _Opt("A"), [1, 2, 3]
    U._on_play(a, rounds)
    assert U.shown_round(a, rounds) == 1 and U.replaying(a)
    U.leave_review()
    assert U.shown_round(a, rounds) == 3 and not U.replaying()


# ------------------------------------------------------------------ #
#  Replay: Round by Round, then stops on the latest
# ------------------------------------------------------------------ #

def test_replay_steps_every_round_and_stops_on_the_latest():
    for rounds in ([1, 2], [1, 2, 3, 4, 5, 6], list(range(1, 11)), [3, 4, 5]):
        r, playing, seen = rounds[0], True, [rounds[0]]
        while playing:
            r, playing = U.replay_next(rounds, r)
            seen.append(r)
            assert len(seen) <= len(rounds)
        assert seen == rounds            # each Round once, in order, nothing between
        assert U.replay_next(rounds, rounds[-1]) == (rounds[-1], False)


def test_replay_tick_waits_then_reruns_only_the_fragment(state, monkeypatch):
    a, rounds = _Opt("A"), [1, 2, 3]
    slept, reruns = [], []

    class Rerun(Exception):
        pass

    def rerun(scope="app"):
        reruns.append(scope)
        raise Rerun

    monkeypatch.setattr(U.time, "sleep", slept.append)
    monkeypatch.setattr(U.st, "rerun", rerun)
    monkeypatch.setattr(U, "_fragment_rerun", lambda: True)
    U._on_play(a, rounds)
    r = U.shown_round(a, rounds)
    path = [r]
    while U.replaying(a):
        with pytest.raises(Rerun):
            U._replay_tick(a, rounds, r)
        r = U.shown_round(a, rounds)
        path.append(r)
    assert path == [1, 2, 3]
    assert slept == [U.REPLAY_STEP_SECS] * 2 and U.REPLAY_STEP_SECS == 1.0
    assert reruns == ["fragment", "fragment"]
    assert U.shown_round(a, rounds) == 3 and U.REPLAY_STATE not in state
    U._replay_tick(a, rounds, 3)                   # stopped: nothing more
    assert reruns == ["fragment", "fragment"]


def test_a_full_run_stops_a_replay_where_it_is(state, monkeypatch):
    a, rounds = _Opt("A"), [1, 2, 3]
    monkeypatch.setattr(U, "_fragment_rerun", lambda: False)
    monkeypatch.setattr(U.time, "sleep", lambda s: pytest.fail("no wait in a full run"))
    U._on_play(a, rounds)
    U._replay_tick(a, rounds, 1)
    assert not U.replaying() and U.shown_round(a, rounds) == 1


def test_a_step_or_back_to_latest_stops_a_replay(state):
    a, rounds = _Opt("A"), [1, 2, 3, 4]
    U._on_play(a, rounds)
    U._on_step(a, rounds, 1)
    assert not U.replaying() and U.shown_round(a, rounds) == 2
    U._on_step(a, rounds, -5)                      # clamped to the first
    assert U.shown_round(a, rounds) == 1
    U._on_play(a, rounds)
    U._on_latest(a, rounds)
    assert not U.replaying() and U.shown_round(a, rounds) == 4
    U._on_play(a, rounds)
    U._on_stop()
    assert not U.replaying() and U.shown_round(a, rounds) == 1


def test_the_poller_never_reruns_the_page_during_a_replay():
    src = inspect.getsource(U._poller)
    assert "not replaying()" in src and "app_rerun()" in src


def test_a_replay_step_draws_as_many_elements_as_any_round():
    # Nothing of the Round before lingers (dimmed) while a step waits.
    assert "_hold(2)" in inspect.getsource(U._changes_block)
    assert "_hold(1)" in inspect.getsource(U._changes_block)
    assert "_hold(1)" in inspect.getsource(U._replay_bar)


# ------------------------------------------------------------------ #
#  The panel in the app
# ------------------------------------------------------------------ #

def test_the_control_frame_label_and_back_to_latest():
    opt = _spider_opt()
    at = _review(opt)
    texts = [m.value for m in at.tabs[3].markdown]
    today = next(t for t in texts if t.startswith("**") and "→" in t)
    assert wording.replay_label(4, 4) in texts
    assert f":gray-badge[{wording.REPLAY_LATEST}]" in texts
    assert "insights_replay_latest" not in _keys(at)
    assert at.button(key="insights_replay_play").label == wording.REPLAY_PLAY
    assert at.button(key="insights_replay_next").disabled
    # An earlier Round: labelled, framed, Back to latest offered.
    texts = _click(at, "insights_replay_prev")
    assert wording.replay_label(3, 4) in texts
    assert f":orange-badge[{wording.REPLAY_EARLIER}]" in texts
    assert f":gray-badge[{wording.REPLAY_LATEST}]" not in texts
    assert "insights_replay_latest" in _keys(at)
    assert ".st-key-insights_replay_panel_past" in U.REPLAY_STYLE
    assert any(".st-key-insights_replay_panel_past" in h.proto.body
               for h in at.get("html"))
    # The section's answer line and What next? stay about today.
    assert today in texts
    # Back to latest.
    texts = _click(at, "insights_replay_latest")
    assert wording.replay_label(4, 4) in texts
    assert f":orange-badge[{wording.REPLAY_EARLIER}]" not in texts
    assert "insights_replay_latest" not in _keys(at)


def test_replay_starts_on_the_first_round():
    opt = _spider_opt()
    at = _review(opt)
    # (AppTest reruns the whole script, which stops a Replay where it is:
    # the browser reruns only the fragment and steps on, test_replay_tick.)
    texts = _click(at, "insights_replay_play")
    assert wording.replay_label(1, 4) in texts
    assert any(t.startswith("Round 1: on target") for t in texts)
    band, marker = next((b, m) for b, m in map(_band_and_marker, _specs(at)) if b)
    assert band == (0.5, 3.5) and marker["f"] == "F2"
    assert at.button(key="insights_replay_prev").disabled


def test_leaving_the_tab_shows_the_latest_again():
    opt = _spider_opt()
    at = _review(opt)
    _click(at, "insights_replay_prev")
    texts = _click(at, "insights_replay_prev")
    assert wording.replay_label(2, 4) in texts
    at.session_state["main_tab"] = wording.TAB_RESULTS
    at.run()
    assert U.REPLAY_STATE not in at.session_state
    at.session_state["main_tab"] = wording.TAB_INSIGHTS
    at.run()
    texts = [m.value for m in at.tabs[3].markdown]
    assert wording.replay_label(4, 4) in texts


def test_one_round_has_no_control_and_no_frame():
    opt = _spider_opt([r for r in ROWS if r[0] == 1])
    at = _review(opt)
    assert not {k for k in _keys(at) if (k or "").startswith("insights_replay")}
    texts = [m.value for m in at.tabs[3].markdown]
    assert f"**{wording.SPIDER_TITLE}**" in texts


# ------------------------------------------------------------------ #
#  Geometry: legible beside each other, no horizontal scroll
# ------------------------------------------------------------------ #

def test_the_profile_stays_legible_beside_the_chart():
    assert U.PANEL_SPIDER >= 340
    opt = _spider_opt()
    data = U.spider_data(opt)
    spec = U.spider_chart(opt, data, 4, size=U.PANEL_SPIDER, wrap=True).to_dict()
    flat = json.dumps(spec)
    assert f'"width": {U.PANEL_SPIDER}' in flat and f'"height": {U.PANEL_SPIDER}' in flat
    # The side spokes' labels on two lines (name, then target); top and
    # bottom stay on one.
    texts = [row["text"] for name, rows in spec["datasets"].items() for row in rows
             if isinstance(row, dict) and "text" in row and "align" in row
             and "x1" in row]
    sides = {t for t in texts if "\n" in t}
    assert len(sides) == 2 and all(t.split("\n")[1].startswith("target") for t in sides)
    drawn = {t for t in texts if U.SPOKE_SEP in t} - {t.replace("\n", U.SPOKE_SEP) for t in sides}
    assert len(drawn) == 2                        # the top and bottom spokes, one line


def test_the_table_header_wraps_only_before_its_bracket():
    opt = _spider_opt()
    ch = U.what_changed(opt, 2)
    head = U.changes_table(opt, ch, compact=True).splitlines()[0]
    assert "Best so far (F4, Round 2)" in head
    assert "Best so far (F4, Round 2)" in U.changes_table(opt, ch).splitlines()[0]
    # The cells never wrap (no horizontal scroll: the column fits them).
    assert "white-space: nowrap" in U.REPLAY_STYLE
    # No chart toolbar pops over the control (a click meant for Replay).
    assert '[class*="st-key-insights_replay_panel"] [data-testid="stElementToolbar"]' \
        in U.REPLAY_STYLE


def test_the_slim_score_chart_keeps_its_marks():
    opt = _spider_opt()
    slim = U.improving_spec(opt, 2, height=U.PANEL_CHART_H)
    full = U.improving_spec(opt, 2)
    assert slim is not full
    assert _band_and_marker(slim) == _band_and_marker(full)
    assert len(U._plot_layers(slim)) == len(U._plot_layers(full))
    assert U.PANEL_CHART_H < U.IMPROVING_HEIGHT


def test_no_banned_word_in_the_control():
    from tests.test_food_bo import _BANNED
    texts = [wording.REPLAY_PLAY, wording.REPLAY_STOP, wording.REPLAY_BACK,
             wording.REPLAY_EARLIER, wording.REPLAY_LATEST, wording.REPLAY_HELP,
             wording.replay_label(4, 10), wording.replay_to(3), wording.replay_play_help(1, 6)]
    assert [t for t in texts if any(p.search(t) for p in _BANNED)] == []
