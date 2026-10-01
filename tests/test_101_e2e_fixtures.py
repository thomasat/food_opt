"""T0b of the 0.8.0 e2e fixes (spec 2026-10-01 rev. 2; review I12): the
campaign's own projects load as saved, the Round truncation behaves, and
the model-free facts the later pins rest on are fixed here. The last test
was T0b's pin of the 500 g Round read off its stored amounts; W2 flipped
it (suggestion_role reads both sides at the Formulation total)."""
import json

import pytest

import insights
import wording
from tests import e2e_fixtures as E


def _best(opt):
    rows = opt.best_candidates()
    i = max(rows, key=lambda r: opt.Y_history[r])
    return int(opt.formulation_ids[i]), round(float(opt.Y_history[i]), 2)


@pytest.mark.parametrize("name,rows,rounds", [
    (E.PATTY, 32, 8), (E.PATTY_ROUND3, 12, 3), (E.PATTY_ROUND7_OPEN, 24, 6),
    (E.BAR_CLEAN, 32, 8), (E.BAR_DRIFT, 32, 8), (E.BAR_NOISY, 28, 7)])
def test_every_fixture_loads_as_saved(name, rows, rounds):
    opt = E.load(name)
    assert len(opt.recipe_history) == rows
    assert max(opt.batch_history) == rounds
    assert opt.formulation_total == 100.0


def test_the_loader_never_writes_back(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    opt = E.load(E.BAR_CLEAN)
    opt.save()
    assert list(tmp_path.iterdir()) == []


def test_the_fixtures_are_frozen_json():
    """No NaN or Infinity: every file is plain JSON a browser could read."""
    for name in E.NAMES:
        text = (E.HERE / f"{name}.json").read_text()
        assert "NaN" not in text and "Infinity" not in text
        assert json.loads(text)["CLASS_VERSION"] == 17


def test_the_02_facts():
    final = E.load(E.PATTY)
    assert [o["name"] for o in final.objectives] == ["Firmness", "Juiciness", "Cook loss",
                                                     "Redness"]
    assert final.set_aside == [8]
    assert final.batch_totals[3] == 500.0
    assert _best(final) == (30, 82.2)
    assert final.bo_config is None
    r3 = E.load(E.PATTY_ROUND3)
    assert [(o["name"], o["weight"]) for o in r3.objectives] == [
        ("Firmness", 40.0), ("Juiciness", 35.0), ("Cook loss", 25.0)]
    assert _best(r3) == (11, 83.01)
    assert insights.repeat_groups(r3, r3.best_candidates()) == []
    open7 = E.load(E.PATTY_ROUND7_OPEN)
    assert (open7.pending_batch_no, len(open7.pending_batch)) == (7, 4)


def test_the_03_facts():
    clean, drift = E.load(E.BAR_CLEAN), E.load(E.BAR_DRIFT)
    assert _best(clean) == (23, 91.69)
    assert clean.bo_config["kernel"] == "matern52"
    assert drift.bo_config["kernel"] == "rbf"
    assert drift.notes_history[19] == "Repeat of Formulation 14"
    noisy = E.load(E.BAR_NOISY)
    assert noisy.bo_config["noise"] == "fixed_tiny"
    assert E.load(E.BAR_NOISY, recommended=True).bo_config is None
    assert _best(noisy) == (14, 91.45)
    groups = insights.repeat_groups(noisy, noisy.best_candidates())
    f14 = next(g for g in groups if int(noisy.formulation_ids[sorted(g)[0]]) == 14)
    scores = [round(float(noisy.Y_history[i]), 2) for i in sorted(f14)]
    assert scores == [91.45, 67.3, 66.34, 75.61]
    # Review I7 printed 75.18, the mean of the four scores as shown (75.175);
    # the stored scores' mean is 75.173, so a mean from them reads 75.17.
    assert f"{sum(noisy.Y_history[i] for i in f14) / 4:.2f}" == "75.17"


def test_truncation_keeps_the_rounds_asked_for():
    opt = E.load(E.BAR_CLEAN, last_round=6)
    assert len(opt.recipe_history) == 24
    assert set(opt.batch_history) == {1, 2, 3, 4, 5, 6}
    assert (opt.next_batch_number, opt.next_formulation_no) == (7, 25)
    assert opt.pending_batch is None
    assert set(opt.round_generation) == {1, 2, 3, 4, 5, 6}
    assert _best(opt) == (23, 91.69)
    # The Overall scores of the rows kept are the saved file's own.
    full = E.load(E.BAR_CLEAN)
    assert opt.Y_history == full.Y_history[:24]


def test_truncation_drops_an_open_round_and_its_not_scored_rows():
    opt = E.load(E.PATTY_ROUND7_OPEN, last_round=5)
    assert opt.pending_batch is None and max(opt.batch_history) == 5
    assert opt.set_aside == [8]
    assert E.load(E.PATTY, last_round=1).set_aside == []


def test_state_is_a_fresh_copy():
    a = E.state(E.BAR_DRIFT)
    a["recipe_history"].clear()
    assert len(E.state(E.BAR_DRIFT)["recipe_history"]) == 32
    saved = E.state(E.BAR_DRIFT)
    E.truncate_state(saved, 2)
    assert len(saved["recipe_history"]) == 32


def test_the_500g_round_reads_at_the_formulation_total():
    """W2 (review I3): suggestion_role compares Round 3's rows, made at
    500 g, with Best so far at the Formulation total. T0b pinned all four
    as "explores" (02 I1: "all 4 'Trying something different'"): the
    stored 500 g amounts against Best so far at 100 g. F12 moved Textured
    pea protein −9.3 g (more than 0.30 of its span from everything made):
    it explores, and the other three are balanced."""
    before = E.load(E.PATTY_ROUND3, last_round=2)
    r3 = E.load(E.PATTY_ROUND3)
    assert r3.batch_totals[3] == 500.0
    kinds = [before.suggestion_kind(r3.recipe_history[i]) for i in range(8, 12)]
    assert kinds == [wording.SUGGESTION_BALANCED] * 3 + [wording.SUGGESTION_DIFFERENT]
