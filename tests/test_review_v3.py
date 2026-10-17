"""Review v3 (owner-approved 2026-10-16; final-v3/wording.md §4): Are we
improving? as measurements only — the headline, Best so far, Best so far vs
targets (the strip that replaced the radar) and Amounts in the top
formulations — then What next? (buttons only) and five Definitions. No
Forecast check, no stop, no verdict anywhere on the tab.

Checked on the yogurt pilot's Round 6 sample after its Round 6 results are
uploaded: F16 and F17 tie at 98.33 (Tangy 4 against 5), F18 91.67; the best
before is F10's 95.0 (Round 4); the top 4 within 5 points are F16, F17, F10
and F13 (93.33: exactly 5 below, kept by the score tolerance)."""
import io
import json
import os
from pathlib import Path

import pytest
from streamlit.testing.v1 import AppTest

import insights
import sample_projects as sp
import storage as storage_backend
import ui_insights as U
import wording
from food_bo import FoodOptimizer

APP = str(Path(__file__).resolve().parent.parent / "app.py")


def _round6(tmp_path, name=sp.ROUND6):
    """The Round 6 sample with the pilot's Round 6 results saved and the
    Round closed, as Results' upload leaves it."""
    os.chdir(tmp_path)
    store = storage_backend.LocalStorage()
    sp.ensure(store)
    opt = FoodOptimizer(name, storage=store)
    state = store.load(sp.ROUND6)
    state["project_name"] = name
    opt.import_json(state)
    up = opt.results_from_workbook(io.BytesIO(sp.round6_results_bytes()))
    parsed, _left = opt.parse_batch_results(up.frame, opt.pending_batch, with_skipped=True,
                                            weighed=up.actual)
    by = {r['formulation']: r['recipe'] for r in opt.pending_batch}
    for no, res, note in parsed:
        opt.tell(opt.amounts_as_weighed(by[no], up.actual.get(no)), res, formulation_no=no,
                 batch_no=opt.pending_batch_no, note=note)
    opt.set_pending_batch(None)
    return opt


@pytest.fixture
def r6(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    return _round6(tmp_path)


def _nos(opt, positions):
    return [int(opt.formulation_ids[i]) for i in positions]


# ------------------------------------------------------------------ #
#  The facts at a Round
# ------------------------------------------------------------------ #

def test_round_view_names_the_tie_the_best_before_and_the_top_four(r6):
    v = U.round_view(r6)
    assert round(v.best, 2) == 98.33
    assert _nos(r6, v.bests) == [16, 17]
    assert v.rounds == [6] and v.new
    assert round(v.was, 2) == 95.0 and _nos(r6, v.was_bests) == [10] and v.was_rounds == [4]
    # F13 is 93.33: 98.33 − 93.33 is 5 up to float error, kept by SCORE_TOL.
    assert _nos(r6, v.top) == [16, 17, 10, 13]
    assert len(v.tried) == 18


def test_the_top_edge_uses_the_score_tolerance(r6):
    """A score 5 below the best up to float error (93.33 against 98.33,
    each a sum of thirds) is in the top; one clearly further is not."""
    best = max(r6.Y_history)
    i13 = list(r6.formulation_ids).index(13)
    r6.Y_history[i13] = best - U.TOP_WITHIN - 1e-9
    assert 13 in _nos(r6, U.round_view(r6).top)
    r6.Y_history[i13] = best - U.TOP_WITHIN - 1e-3
    assert 13 not in _nos(r6, U.round_view(r6).top)


def test_an_earlier_round_reads_only_what_was_measured_by_then(r6):
    v4 = U.round_view(r6, 4)
    assert _nos(r6, v4.bests) == [10] and v4.new and len(v4.tried) == 12
    assert _nos(r6, v4.was_bests) == [7]
    v5 = U.round_view(r6, 5)
    assert _nos(r6, v5.bests) == [10] and not v5.new


# ------------------------------------------------------------------ #
#  The headline
# ------------------------------------------------------------------ #

def test_the_headline_on_the_round6_sample(r6):
    assert U.headline_text(r6, U.round_view(r6)) == \
        "Best 98.3: F16, F17 tied (Round 6). Was 95.0 (F10, Round 4)."
    assert U.headline_text(r6, U.round_view(r6, 5)) == "Best 95.0: F10 (Round 4). No new best since."
    assert U.headline_text(r6, U.round_view(r6, 1)) == "Best 91.7: F3 (Round 1)."
    assert U.headline_text(r6, U.round_view(r6, 4)) == \
        "Best 95.0: F10 (Round 4). Was 92.5 (F7, Round 3)."


@pytest.mark.parametrize("args, kw, want", [
    (("99.2", ["F21"], [7]), {"new": False}, "Best 99.2: F21 (Round 7). No new best since."),
    (("95.0", ["F10", "F14"], [4, 5]), {"new": False},
     "Best 95.0: F10, F14 tied (Rounds 4, 5). No new best since."),
    (("98.3", ["F16", "F17"], [6]), {"was": "95.0", "was_names": ["F10"], "was_rounds": [4]},
     "Best 98.3: F16, F17 tied (Round 6). Was 95.0 (F10, Round 4)."),
])
def test_the_headline_wording(args, kw, want):
    text = wording.review_headline(*args, **kw)
    assert text == want
    for word in ("Rising", "Plateau", "noise", "(?)", "Stop", "stop"):
        assert word not in text


# ------------------------------------------------------------------ #
#  Best so far
# ------------------------------------------------------------------ #

def _spec_text(spec):
    return json.dumps(spec, default=str, ensure_ascii=False)


def test_the_score_chart_names_the_tie_the_top_and_counts_the_dots(r6):
    text = _spec_text(U.improving_chart(r6, height=U.PANEL_CHART_H).to_dict())
    assert "Best: 98.3 (F16, F17 tied)" in text
    assert "100 = every attribute on target" in text
    assert "Each formulation (18)" in text
    assert "Round average" in text and wording.BEST_SO_FAR_COLUMN in text
    for gone in ("Stop", "Within noise", "within noise", 'Within noise of top', "Start:"):
        assert gone not in text, gone


def test_the_score_caption_and_its_question_mark(r6):
    caption, help_ = U.score_caption(r6, U.round_view(r6))
    assert caption == "Each formulation made once, rated once by the panel (n = 5)."
    assert help_ == ("Tasting the same formulation again could give a slightly different "
                     "score, so small differences may not be real. Example: Tangy 4 instead "
                     "of 5 costs 1.7 points.")


def test_no_panel_size_on_a_project_that_records_none(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    opt = _round6(tmp_path, name="Mine")
    caption, _ = U.score_caption(opt, U.round_view(opt))
    assert caption == "Each formulation made once, rated once by the panel."


def test_no_example_when_every_attribute_is_on_target(r6):
    v = U.round_view(r6)
    for i in v.bests:
        r6.results_history[i]["Tangy"] = 5.0
    assert U.cost_example(r6, v) is None
    assert "Example" not in U.score_caption(r6, v)[1]


# ------------------------------------------------------------------ #
#  Best so far vs targets (the radar is gone)
# ------------------------------------------------------------------ #

def test_the_strip_has_one_row_per_attribute_with_its_value_text(r6):
    rows = U.target_strip_rows(r6, U.round_view(r6))
    assert [r["name"] for r in rows] == ["Consistency", "Creaminess", "Tangy",
                                         "Thickness uniformity"]
    texts = {r["name"]: r["text"] for r in rows}
    assert texts == {"Consistency": "12 (target 12)", "Creaminess": "5 (target 5)",
                     "Tangy": "4 (target 5, −1)", "Thickness uniformity": "13 (target 13)"}
    assert [r["off"] for r in rows] == [False, False, True, False]
    # F16 and F17 agree on every attribute: one dot a row.
    assert all(len(r["dots"]) == 1 for r in rows)


def test_the_strip_spec(r6, monkeypatch):
    spec = U.targets_chart(r6, U.round_view(r6)).to_dict()
    text = _spec_text(spec)
    assert "Panel median (0–15)" in text
    for entry in ("Target", "Best, on target", "Best, off target"):
        assert entry in text
    assert U.outline_colour() in text and U.model_blue() in text
    assert "theta" not in text and "arc" not in text        # no radar
    for gone in ("spider_chart", "spider_data", "_spider_block", "target_chart"):
        assert not hasattr(U, gone), gone


def test_tied_bests_that_differ_get_a_dot_each(r6):
    v = U.round_view(r6)
    r6.results_history[v.bests[1]]["Tangy"] = 5.0      # F17 on target, F16 not
    row = next(r for r in U.target_strip_rows(r6, v) if r["name"] == "Tangy")
    assert [d["who"] for d in row["dots"]] == ["F16", "F17"]
    assert [d["off"] for d in row["dots"]] == [True, False]
    assert row["text"] == "F16 4, F17 5 (target 5)"
    assert row["dots"][0]["shape"] != row["dots"][1]["shape"]


def test_a_max_goal_has_no_target_tick(r6):
    obj = next(o for o in r6.objectives if o["name"] == "Creaminess")
    obj["goal"], obj["target"] = "max", None
    v = U.round_view(r6)
    row = next(r for r in U.target_strip_rows(r6, v) if r["name"] == "Creaminess")
    assert row["target"] is None and row["text"] == "5 (higher is better)"
    assert not row["off"]
    text = _spec_text(U.targets_chart(r6, v).to_dict())
    assert wording.REVIEW_KEY_NO_TARGET in text


# ------------------------------------------------------------------ #
#  Amounts in the top formulations (What changed is gone)
# ------------------------------------------------------------------ #

def test_amount_rows_tag_the_limits_the_best_sits_at(r6):
    rows = {r["name"]: r for r in U.amounts_rows(r6, U.round_view(r6))}
    assert list(rows) == ["Coconut cream", "Tap water", "Lemon juice", "Salt",
                          "Nutritional yeast", "Tapioca starch", "Xanthan gum"]
    assert {n for n, r in rows.items() if r["tag"]} == {"Tap water", "Tapioca starch"}
    assert rows["Tap water"]["tag"] == "upper"
    assert rows["Tap water"]["at"][:2] == ["F16", "F17"]
    # Lemon juice: F13 sits at 0, but no best does: no tag.
    assert rows["Lemon juice"]["tag"] is None
    assert wording.review_limit_help(["F16", "F17"]) == (
        "F16, F17 use the most this limit allows. You can raise it in Set up.")


def test_the_top_band_spans_the_top_four(r6):
    row = next(r for r in U.amounts_rows(r6, U.round_view(r6)) if r["name"] == "Tapioca starch")
    lo, hi = row["top_text"]
    assert round(lo, 3) == 2.102 and round(hi, 3) == 2.6        # F13 … F16/F17


def test_equal_amounts_are_spread_so_no_marker_hides_another():
    offs = U.best_offsets([1.0, 1.0])
    assert offs[0] != offs[1] and abs(offs[0] + offs[1]) < 1e-9
    assert U.best_offsets([0.2, 0.6]) == [0.0, 0.0]


def test_the_amounts_key_and_the_linked_total(r6):
    v = U.round_view(r6)
    text = _spec_text(U.amounts_chart(r6, v).to_dict())
    for entry in ("Allowed, not tried", "Tried (18)", "Top 4 (F16, F17, F10, F13)", "F16",
                  "F17 (tied)", "at upper limit  (?)"):
        assert entry in text, entry
    assert U.amounts_linked_text(r6) == (
        "The 7 add to 97.7–98.7 g (the allowed total), so amounts are linked.")
    assert wording.review_amounts_title(4) == "Amounts in the top 4 formulations"


def test_no_linked_line_without_a_total(r6):
    r6.quantity_constraints = []
    assert U.amounts_linked_text(r6) is None


def test_dark_theme_uses_the_dark_tokens(r6, monkeypatch):
    monkeypatch.setattr(U, "muted", lambda: U._MUTED_DARK)
    assert U.theme() == "dark"
    v = U.round_view(r6)
    amounts = _spec_text(U.amounts_chart(r6, v).to_dict())
    for colour in U._AMOUNT_COLOURS["dark"].values():
        assert colour in amounts
    for colour in U._AMOUNT_COLOURS["light"].values():
        assert colour not in amounts
    strip = _spec_text(U.targets_chart(r6, v).to_dict())
    assert U._TRACK_COLOURS["dark"][0] in strip and U._TRACK_COLOURS["light"][0] not in strip
    assert U.model_blue("dark") in strip and U._OUTLINE_DARK in strip
    score = _spec_text(U.improving_chart(r6, height=U.PANEL_CHART_H).to_dict())
    assert U.model_blue("dark") in score and "#0e1117" in score


# ------------------------------------------------------------------ #
#  Definitions
# ------------------------------------------------------------------ #

def test_the_five_definitions(r6):
    assert U.definitions(r6) == [
        "Overall score: 100 = every attribute on target. Each attribute earns its share; "
        "each point off costs 1/15 of it.",
        "Round: the formulations made and tasted together.",
        "Best so far: the highest Overall score so far. Each formulation is made once and "
        "rated once.",
        "Top formulations: within 5 points of the best so far.",
        "Suggestions: best bets to improve on the best so far, weighing how good they look "
        "and how unsure the model is.",
    ]
    assert len(wording.method_definitions()) == 5


def test_mixed_scales_say_the_rule_generically():
    lines = wording.method_definitions(span=None)
    assert "1/(its Range)" in lines[0]


# ------------------------------------------------------------------ #
#  The tab itself
# ------------------------------------------------------------------ #

def _walk(node):
    yield node
    for child in getattr(node, "children", {}).values():
        yield from _walk(child)


def _texts(tab):
    out = []
    for el in _walk(tab):
        for attr in ("value", "label", "help", "body"):
            try:
                v = getattr(el, attr, None)
            except Exception:       # a widget whose value lives in session state
                v = None
            if isinstance(v, str):
                out.append(v)
        proto = getattr(el, "proto", None)
        spec = getattr(proto, "spec", None)
        if isinstance(spec, str):
            out.append(spec)
    return out


@pytest.fixture
def review(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    _round6(tmp_path)
    at = AppTest.from_file(APP, default_timeout=180)
    at.session_state["_loaded_project"] = sp.ROUND6
    at.session_state["_land_on_open"] = True
    at.run()
    assert not at.exception
    return at


def test_the_review_after_uploading_round6(review):
    tab = review.tabs[3]
    md = [m.value for m in tab.markdown]
    assert md[0] == f"#### {wording.Q2_TITLE}"
    assert "**Best 98.3: F16, F17 tied (Round 6). Was 95.0 (F10, Round 4).**" in md
    assert "**Best so far vs targets (F16, F17 tied)**" in md
    assert "**Amounts in the top 4 formulations**" in md
    assert f"#### {wording.Q5_TITLE}" in md
    # What next?: buttons only.
    labels = [b.label for b in tab.button]
    assert "Start Round 7" in labels and "See the pilot's Rounds 7–10" in labels
    assert [e.label for e in tab.expander] == [wording.FOLD_METHOD]
    charts = [s for s in _texts(tab) if s.startswith("{")]
    assert len(charts) == 3


def test_no_forecast_check_stop_or_verdict_on_the_review(review):
    text = "\n".join(_texts(review.tabs[3]))
    for gone in ("Forecast check", "forecast", "Predicts",
                 "Partly", "Not yet", "Rising", "Plateau", "Stop:", "stop signal",
                 "Within noise of top", "too close", "called", "Repeat F", "theta"):
        assert gone not in text, gone
    assert not any(b.label in (wording.REPLAY_STOP,) for b in review.tabs[3].button)


def test_stepping_back_redraws_the_headline_for_that_round(review):
    review.button(key="insights_replay_prev").click().run()
    md = [m.value for m in review.tabs[3].markdown]
    assert "**Best 95.0: F10 (Round 4). No new best since.**" in md
