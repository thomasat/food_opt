"""Review four (prototype, 2026-10-02; owner: "clean up the Review tab to
cleanly (and visually) answer these 4 questions"). Under the header and
the three cards, four question sections, always open, in this order: Q1
"Can we trust the predictions?" beside Q2 "Are we improving?", then Q3
"What drives each result?" and Q4 "What have we tried?". Each is a small
heading, one answer line opening with its card's bold word, ONE chart and
its caption. Each card links to its question ("See chart ↓"). The hero,
"Can you have both?", the triangle and their folds are gone.

Ruling 1: the Predictions word reads the Overall score's Round-ahead record
(coverage + skill); the measurements' misses are a secondary, non-red line.
Ruling 2: Expected gain is never hidden: greyed, with its reason."""
import json
import os
import re
import shutil

import pytest
from streamlit.testing.v1 import AppTest

import bo_views
import insights
import insights_jobs
import ui_insights as U
import wording
from tests import demo_projects as D
from tests.test_app_ui import APP_PATH
from tests.test_food_bo import _BANNED
from tests.test_review_cards import _leaves, _text, cards, screens  # noqa: F401 (fixture)
from tests.test_review_diag import ALL10, PILOT, R5, R6, needs_pilot

# Review learned: "What has the model learned?" took Q1's place.
QUESTIONS = [wording.LEARNED_TITLE, wording.Q2_TITLE, wording.Q3_TITLE, wording.Q4_TITLE]
# Review story: the screen asks them as a story — improving first, then
# trust, drivers, what was tried.
STORY = [wording.Q2_TITLE, wording.LEARNED_TITLE, wording.Q3_TITLE, wording.Q4_TITLE]
# Review story 2: What next? closes the story (the Next card's content).
STORY += [wording.Q5_TITLE]
CHART = ("arrow_vega_lite_chart", "vega_lite_chart")


def _words(text):
    plain = re.sub(r":\w+\[(.*?)\]", r"\1", text).replace("**", "")
    return len([w for w in plain.split() if re.search(r"\w", w)])


def sections(at):
    """{question: [(type, text)]} in screen order, from each heading to the
    next (the last one to Method and definitions)."""
    out, cur = {}, None
    for el, folds, _ in _leaves(at.tabs[3]):
        text = _text(el)
        if el.type == "markdown" and text.startswith("#### "):
            cur = text[5:]
            out[cur] = []
            continue
        if el.type == "expander":
            cur = None
        if cur is not None and (text or el.type in CHART):
            out[cur].append((el.type, text))
    return out


# ------------------------------------------------------------------ #

@pytest.mark.parametrize("name", D.ALL)
def test_four_questions_in_order_one_chart_each(screens, name):  # noqa: F811
    at = screens[name]
    got = sections(at)
    assert list(got) == [wording.for_project(D.load(name), q) for q in STORY]
    gate = name == D.SNACK
    for q, items in got.items():
        charts = sum(t in CHART for t, _ in items)
        if gate and q in (wording.LEARNED_TITLE, wording.Q3_TITLE):
            # Before 5 scored: one line, no chart.
            assert charts == 0 and ("caption", wording.Q_WAIT) in items, (q, items)
        elif q == wording.Q2_TITLE:
            # Review story: the best so far and Closer to target, side by side.
            assert charts == 2, (q, items)
        elif q == wording.Q5_TITLE:
            assert charts == 0, (q, items)       # review story 2: words and buttons
        else:
            assert charts == 1, (q, items)


@pytest.mark.parametrize("name", D.ALL)
def test_answer_lines_open_with_the_cards_word(screens, name):  # noqa: F811
    at = screens[name]
    p, g, n = cards(at)
    got = sections(at)
    plain = lambda w: re.sub(r":\w+\[(.*?)\]", r"\1", w)   # noqa: E731
    firsts = {q: next((t for k, t in items if k == "markdown"), None)
              for q, items in got.items()}
    opt = D.load(name)
    q1, q2, q3, q4 = (firsts[wording.for_project(opt, k)] for k in QUESTIONS)
    if name != D.SNACK:
        # Review learned: the section's answer names what the model predicts.
        assert re.match(r"\*\*(Predicts|Partly predicts|Not predictable yet)", q1), q1
        # Review story: Q3's answer is a finding, never the Next card's word.
        assert not q3.startswith(f"**{plain(n['word'])}.**")
        # Review story 3: no clear leader over tested amounts reads
        # 'No single ingredient (or setting) clearly leads yet'.
        # Review learned: or a stable effect, 'Methylcellulose raises Firmness'.
        assert re.match(r"\*\*.+ (matters most|raises .+|lowers .+)\.\*\*|"
                        r"\*\*No single (ingredient|setting)", q3), q3
    assert q2.startswith(f"**{plain(g['word'])}.**")
    assert re.match(r"\*\*\d+ tested\.\*\* ", q4), q4
    for line in (q1, q2, q3, q4):
        if line:
            assert _words(line) <= 10, line


@pytest.mark.parametrize("name", D.ALL)
def test_captions_and_labels_within_the_limits(screens, name):  # noqa: F811
    for q, items in sections(screens[name]).items():
        assert _words(q) <= 5
        for kind, text in items:
            if kind == "caption" and text != wording.Q_WAIT:
                assert _words(text) <= 10, text       # captions 8, a greyed reason 10
            if kind == "button":
                assert _words(text) <= 5, text


@pytest.mark.parametrize("name", D.ALL)
def test_each_card_links_to_its_question(screens, name):  # noqa: F811
    at = screens[name]
    # Review story 2: each strip item is itself the link (its word and line,
    # one tertiary button, keyed as the old "See chart ↓").
    buttons = {b.key: b for b in at.tabs[3].button}
    for which in ("predictions", "progress", "next"):
        b = buttons[f"{U.SEE_CHART_KEY}_{which}"]
        assert b.proto.type == "tertiary"
        assert b.proto.help.startswith("Scrolls to: ")
    # Predictions -> Q1, Progress -> Q2, Next -> What next?, each landing
    # on the keyed place its section is drawn in (the links from Results
    # and Make a round land on the first two).
    assert U.SECTION_KEYS == {"predictions": U.TRUST_SECTION, "progress": U.STAND_SECTION,
                              "drives": U.DRIVES_SECTION, "tried": U.EXPLORED_PLACE,
                              "next": U.NEXT_SECTION}
    assert U._QUESTIONS["predictions"] == wording.LEARNED_TITLE
    assert U._QUESTIONS["drives"] == wording.Q3_TITLE
    assert U._QUESTIONS["next"] == wording.Q5_TITLE


def test_a_card_link_lands_on_its_question(screens):  # noqa: F811
    at = screens[D.LINE_A]
    landed = {}
    real = U.ask_tab

    def spy(label, section="top", fold=None, state=None):
        landed["section"] = section
        return real(label, section, fold, state)
    for which, key in (("progress", U.STAND_SECTION), ("next", U.NEXT_SECTION)):
        [b] = [b for b in at.tabs[3].button if b.key == f"{U.SEE_CHART_KEY}_{which}"]
        U.ask_tab = spy
        try:
            b.click().run()
        finally:
            U.ask_tab = real
        assert not at.exception
        # The callback is bound with its section (on_click args).
        assert at.session_state["main_tab"] == wording.TAB_INSIGHTS
    src = U._strip_item.__code__.co_names
    assert "SECTION_KEYS" in src and "ask_tab" in src


@pytest.mark.parametrize("name", D.ALL)
def test_one_method_link_and_one_fold(screens, name):  # noqa: F811
    at = screens[name]
    folds = [el.label for el, f, _ in _leaves(at.tabs[3]) if el.type == "expander" and not f]
    assert folds == [wording.FOLD_METHOD]
    links = [b for b in at.tabs[3].button if b.label == wording.METHOD_LINK]
    assert len(links) == 1 and links[0].proto.type == "tertiary"


@pytest.mark.parametrize("name", D.ALL)
def test_one_primary_on_the_screen(screens, name):  # noqa: F811
    at = screens[name]
    primaries = [el for el, _, _ in _leaves(at.tabs[3])
                 if el.type == "button" and el.proto.type == "primary"]
    assert len(primaries) == 1


REMOVED_TEXT = (wording.PLOT_PAIR, wording.PLOT_TRIANGLE, wording.PLOT_CHOOSES,
                "How the app chose", "Where the app looks next", wording.PAIR_FOLD,
                wording.FOLD_EXPLORED, wording.FOLD_MORE, wording.FOLD_TRIED,
                wording.PLOT_MAP)


@pytest.mark.parametrize("name", D.ALL)
def test_the_removed_charts_are_gone(screens, name):  # noqa: F811
    at = screens[name]
    for el, _, _ in _leaves(at.tabs[3]):
        text = _text(el) or ""
        spec = el.proto.spec if el.type in CHART else ""
        for gone in REMOVED_TEXT:
            assert gone not in text and gone not in spec, (gone, text[:80])
    for fn in ("chooses_chart", "chooses_data", "pair_chart", "mixture_chart", "_hero",
               "_triangle", "_more_charts", "explored_fold", "tried_fold", "path_chart",
               "change_chart", "gain_chart"):
        assert not hasattr(U, fn), fn
    # The map stays on Make a round's Why these.
    assert hasattr(U, "explore_map_chart")


# ------------------------------------------------------------------ #
#  Ruling 1: the verdict on the Overall score
# ------------------------------------------------------------------ #

# (row, measurements' verdict) per demo project, from its own checks.
VERDICT = {
    D.SNACK: ("gate", "none"),
    D.LINE_A: ("reliable", "within"),        # score 13/15, skill 0.36
    D.LINE_B: ("caution", "unreliable"),     # score 13/16 (tail 0.04); measurements 28/36
    D.LINE_C: ("no_skill", "within"),        # score 14/16, skill -0.47
    D.UPGRADE: ("too_few", "early"),
    D.TIMING: ("reliable", "within"),        # score 54/55
    D.PROCESS_ONLY: ("off_coverage", "within"),   # score 12/16 (tail 0.007)
}


@pytest.mark.parametrize("name", D.ALL)
def test_the_word_reads_the_overall_score(name):
    opt = D.load(name)
    checks = insights.check_rounds(opt) if opt.complete_count() >= insights.MIN_ROWS else []
    f = insights.status_facts(opt, checks)
    assert (f.prediction_row, f.measure_verdict) == VERDICT[name]
    assert f.measures_miss == (f.measure_verdict in ("unreliable", "caution"))


def test_score_verdict_rule():
    h = insights.Headline("verdict", [3], 30, 36, "unreliable",
                          insights.ScoreRecord(14, 15, 0.0, 1.0, [2, 3]), [])
    assert insights.score_verdict_state(None, h, []) == "within"
    h.score = insights.ScoreRecord(9, 9, 0.0, 1.0, [2, 3])
    assert insights.score_verdict_state(None, h, []) == "early"    # under SKILL_MIN checks
    h.score = insights.ScoreRecord(12, 16, 0.0, 1.0, [2, 3])
    assert insights.score_verdict_state(None, h, []) == "unreliable"
    h.score = insights.ScoreRecord(13, 16, 0.0, 1.0, [2, 3])
    assert insights.score_verdict_state(None, h, []) == "caution"
    # A disagreeing Repeat is no Predictions row any more: the Next card's.
    assert insights.score_prediction_row(20, "within", None, 0) == "too_few"
    assert "off_repeat" not in {insights.score_prediction_row(20, v, None, 12)
                                for v in ("unreliable", "caution", "within", "wide")}


# Review story 2: the strip's line, without "Overall score:" (the label
# says Predictions; the answer line and the "?" still name the score).
# Review learned: the item is Model, its word the model's verdicts; the
# Overall score's red miss line is kept (never folded).
CARDS = {
    D.LINE_A: ("Predicts", ["Firmness"]),
    # Review learned 2: the section repeats no old-card line.
    D.LINE_B: ("Predicts", [":red[**3 / 16 missed**]"]),
    D.PROCESS_ONLY: ("Predicts", [":red[**4 / 16 missed**]"]),
    D.TIMING: ("Predicts", ["Firmness, Juiciness +2"]),
}


@pytest.mark.parametrize("name", list(CARDS))
def test_the_predictions_card(screens, name):  # noqa: F811
    p, _, n = cards(screens[name])
    word, first = CARDS[name]
    assert p["word"] == word
    assert p["lines"][:len(first)] == first
    assert not any(line.startswith(":red[") and "Measurements" in line for line in p["lines"])
    # Review learned: the "?" is the sentence, the evidence (the ordering
    # counts, then the Overall score's coverage and skill), then the rule.
    ev, rule = p["help"].split("\n\n")[1:3]
    assert ev.startswith("Evidence: pairs ordered right before tasting")
    assert rule.startswith(wording.for_project(D.load(name), "Each Round's formulations are "))
    assert "Overall score: " in p["help"] and "inside the 95 % predicted range" in p["help"]


def test_a_disagreeing_repeat_stays_red_on_next(screens):  # noqa: F811
    p, _, n = cards(screens[D.LINE_B])
    assert n["word"] == ":red[Check panel first]"
    assert ":red[**Before Round 6**]" in n["lines"]
    assert all("Repeat came out" not in line for line in p["lines"])


def test_evidence_reports_both():
    assert wording.ev_predictions(10, 12, "0.69", 12, 27, 36) == (
        "Overall score 10/12 in range, skill 0.69 (n = 12); measurements 27/36")


# ------------------------------------------------------------------ #
#  Ruling 2: Expected gain greyed, never hidden
# ------------------------------------------------------------------ #

@pytest.fixture(scope="class")
def line_a(tmp_path_factory):
    mp = pytest.MonkeyPatch()
    mp.chdir(tmp_path_factory.mktemp("review-four"))
    mp.setattr(insights_jobs, "_JOBS", {})
    mp.setattr(insights_jobs, "_INFLIGHT", {})
    mp.setattr(U.st, "session_state", {})
    with bo_views.fresh():
        opt = D.load(D.LINE_A)
        computed = U._computed(opt, quiet=True)
        mp.setattr(U, "checks_for", lambda o, stage: computed["checks"])
        yield opt, computed
    mp.undo()


def _gain_rows(spec):
    return [r for r in spec["vconcat"]
            if wording.PROFILER_GAIN_ROW in json.dumps(r["spec"]["layer"][0].get("encoding", {}))
            or wording.PROFILER_GAIN_ROW in json.dumps(r)]


@pytest.mark.usefixtures("line_a")
class TestGain:
    """One Line A fit for the class (its session_state patch is undone
    before the AppTest tests below)."""

    # Review four polish (re-pinned): the greyed reasons are the noise
    # guard's and the Predictions card's two words short of Reliable (was
    # the measurements' verdict).
    @pytest.mark.parametrize("reason", [wording.CHOOSES_REASON_NOISE,
                                        wording.GAIN_REASON_NOT_YET, wording.GAIN_REASON_OFF])
    def test_gain_is_greyed_not_hidden(self, line_a, reason):
        _greyed(line_a, reason)

    def test_profiler_defaults_to_suggestion_one(self, line_a):
        _defaults(line_a)


def _greyed(line_a, reason):
    opt, computed = line_a
    fitted = computed["fitted"]
    refs = U.profile_refs(opt)
    label, number, recipe, is_pick = refs[-1]
    prof = U.profile_data(opt, fitted, recipe, label, number, is_pick, reason)
    assert prof.greyed and prof.gain_drawn
    assert all(c.gain is not None for c in prof.columns)
    spec = U.profile_chart(opt, prof).to_dict()
    text = json.dumps(spec)
    assert wording.PROFILER_GAIN_ROW in text
    grey, _ = U.mark("neutral")
    gain_row = spec["vconcat"][-1]
    lines = [lay["mark"] for lay in gain_row["spec"]["layer"]
             if isinstance(lay["mark"], dict) and lay["mark"]["type"] in ("line", "area")]
    # The gain's line and area in grey, the line dashed: drawn, not claimed.
    assert lines and all(m["color"] == grey for m in lines), lines
    assert any(m.get("strokeDash") for m in lines if m["type"] == "line")
    assert U.gain_greyable(reason)


def test_no_gain_to_grey_for_a_cold_round_or_ucb():
    assert not U.gain_greyable(wording.CHOOSES_REASON_COLD)
    assert not U.gain_greyable(wording.CHOOSES_REASON_UCB)
    assert not U.gain_greyable(None)


def test_the_greyed_lines_say_why_and_that_suggestions_are_the_same():
    for text in wording.GAIN_GREYED.values():
        assert _words(text) <= 10 and "greyed" in text
    assert "Suggestions are made the same way regardless" in wording.tip_gain_greyed("x")
    # Review four polish: Why these says the profiler's own line.
    for reason in ("noise", "not_yet", "off"):
        assert wording.why_gain_greyed(reason) == wording.GAIN_GREYED[reason]


def test_make_a_round_greys_the_gain_column():
    import inspect
    import ui_batch
    src = inspect.getsource(ui_batch._why_fold)
    assert "hide=hide" not in src and "GREYED_GAIN" in src
    assert "why_gain_greyed" in src


def _defaults(line_a):
    opt, _ = line_a
    refs = U.profile_refs(opt)
    assert refs[-1][0] == wording.PROFILER_REF_PICK and refs[-1][3]
    import inspect
    assert "default=labels[-1]" in inspect.getsource(U._profiler)
    assert _words(wording.Q3_CAPTION_PICK) <= 8


# ------------------------------------------------------------------ #
#  The yogurt pilot (skipped without it)
# ------------------------------------------------------------------ #

@pytest.fixture(scope="module")
def pilot(tmp_path_factory):
    here = os.getcwd()
    work = tmp_path_factory.mktemp("review-four-pilot")
    for n in (R5, R6, ALL10):
        shutil.copy(PILOT / f"{n}.pkl", work)
    os.chdir(work)
    try:
        with bo_views.fresh():
            yield work
    finally:
        os.chdir(here)


def _open(name):
    at = AppTest.from_file(APP_PATH, default_timeout=900)
    at.session_state["_loaded_project"] = name
    at.run()
    at.session_state["main_tab"] = wording.TAB_INSIGHTS
    at.run()
    assert not at.exception, [e.message for e in at.exception]
    return at


@needs_pilot
def test_pilot_cards_on_the_overall_score(pilot):
    p = cards(_open(R5))[0]
    # Review learned: the Model item, by the ordering rule (no pilot names
    # here); no red line.
    assert p["word"] == "Predicts"
    assert not any(line.startswith(":red[") for line in p["lines"])
    at = _open(ALL10)
    assert cards(at)[0]["word"] == "Predicts"
    assert list(sections(at)) == STORY


def test_new_strings_pass_the_vocabulary_guard():
    texts = [*QUESTIONS, wording.SEE_CHART, wording.SEE_CHART_HELP, wording.Q_WAIT,
             wording.Q3_CAPTION_PICK, wording.PROFILER_REF_HELP,
             wording.answer_line("Rising", "Best 98.33"), wording.q3_answer("Salt"),
             wording.Q3_FLAT, wording.q4_word(30), wording.q4_gap("Salt below 1 g"),
             wording.Q4_NO_GAP, wording.gap_span("Salt", "0.6", "1.0 g"),
             wording.CARD_MEASURES_MISS, wording.card_score_missed(3, 16),
             wording.on_score("In Round 3, 9 of 10 results landed."),
             wording.measures_sentence(27, 36, 4, True), wording.measures_sentence(27, 36, 4),
             wording.ev_predictions(10, 12, "0.69", 12, 27, 36), *wording.GAIN_GREYED.values(),
             wording.tip_gain_greyed("x"), wording.why_gain_greyed("not_yet"),
             wording.why_greyed_line("off"), wording.why_greyed_line("not_yet"),
             wording.GAIN_REASON_NOT_YET, wording.GAIN_REASON_OFF, wording.TRIED_NEXT_BEST,
             wording.tried_caption_roles(), wording.tip_tried_roles(),
             wording.tip_tried_roles(True), wording.Q2_X_TITLE, wording.round_tick(3),
             *[f(k) for f in (wording.profiler_more_label, wording.profiler_more_placeholder)
               for k in ("ingredients", "settings", "both", "rows")],
             wording.ev_profiler(20, "Suggestion 1", 21), wording.ev_tried(20)]
    for text in texts:
        assert not any(p.search(text) for p in _BANNED), text
        assert "spread" not in text.lower(), text
    # "Expected gain" is the gain's one name.
    for text in (*wording.GAIN_GREYED.values(), wording.Q3_CAPTION_PICK):
        assert "Expected gain" in text or "expected gain" in text
    # Process-only studies say Trial.
    opt = D.load(D.PROCESS_ONLY)
    assert "trials" in wording.for_project(opt, wording.Q_WAIT)


# ------------------------------------------------------------------ #
#  Review four polish (2026-10-02)
# ------------------------------------------------------------------ #

@pytest.mark.parametrize("word, reasons, want", [
    ("reliable", [], None),
    # The measurements' own verdict no longer greys: only the card's word.
    ("not_yet", [], "not_yet"),
    ("off", [], "off"),
    ("reliable", ["noise"], "noise"),
    ("reliable", ["floor", "skill"], "noise"),
    ("reliable", ["skill"], "not_yet"),
    ("not_yet", ["noise"], "not_yet"),     # the card's word first: never contradicted
])
def test_gain_greying_follows_the_predictions_card(monkeypatch, word, reasons, want):
    monkeypatch.setattr(insights, "score_predictions", lambda opt, checks: word)
    monkeypatch.setattr(insights, "noise_guard",
                        lambda opt, score, checks=(), *a, **k: insights.Guard(list(reasons)))
    assert U.gain_grey(object(), None, ()) == want


def test_the_profiler_and_why_these_read_one_rule():
    import inspect
    import ui_batch
    assert "gain_grey(" in inspect.getsource(U.chooses_reason)
    src = inspect.getsource(ui_batch._why_fold)
    assert "ui_insights.gain_grey(opt, score, checks)" in src
    assert "ui_insights.withheld(" not in src
    # Each reason the rule gives has its profiler line, and its key back.
    for key, reason in U._GREY_REASON.items():
        assert U.hero_reason_key(reason) == key and U.gain_greyable(reason)
        assert key in wording.GAIN_GREYED


@pytest.mark.parametrize("name", [D.LINE_A, D.LINE_B])
def test_score_predictions_is_the_cards_word(tmp_path, monkeypatch, name):
    """insights.score_predictions reads what status_facts reads (Line A
    Reliable, Line B Not yet: Line B's gain stays greyed, its line now the
    card's word, not the measurements')."""
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(insights_jobs, "_JOBS", {})
    monkeypatch.setattr(insights_jobs, "_INFLIGHT", {})
    monkeypatch.setattr(U.st, "session_state", {})
    with bo_views.fresh():
        opt = D.load(name)
        checks = U.checks_here(opt)
        facts = insights.status_facts(opt, checks)
        assert insights.score_predictions(opt, checks) == facts.predictions
        assert facts.predictions == ("reliable" if name == D.LINE_A else "not_yet")


def test_q4_is_coloured_by_role():
    opt = D.load(D.LINE_A)
    lines = insights.tried_lines(opt)
    df, _, _ = U.tried_frame(opt, insights.tried_axes(opt), lines)
    roles = set(df["role"])
    assert {wording.TRIED_BEST, wording.TRIED_NEXT_BEST, wording.TRIED_SCORED} <= roles
    tested = [l for l in lines if l.state == "tested"]
    nexts = set(df.loc[df["role"] == wording.TRIED_NEXT_BEST, "no"])
    assert len(nexts) == U.next_best_count(len(tested))
    assert 3 <= U.next_best_count(5) <= U.next_best_count(30) == 5
    # The next best are the highest scores after Best so far.
    ranked = sorted((l for l in tested if not l.best), key=lambda l: -l.score)
    assert nexts == {l.formulation for l in ranked[:len(nexts)]}
    spec = json.dumps(U.tried_chart(opt, insights.tried_axes(opt), lines).to_dict())
    light, mid = U._TRIED_GREYS[U.theme()]
    for colour in (light, mid, U.mark("bar_latest")[0], U.outline_colour()):
        assert colour in spec
    # No score ramp, no gradient legend: the Overall score is the last axis.
    assert not any(c in spec for c in U.score_colours())
    assert '"gradient' not in spec
    assert _words(wording.tried_caption_roles()) <= 8


def test_q2_marks_its_rounds():
    opt = D.load(D.LINE_A)
    spec = U.progress_chart(opt).to_dict()
    text = json.dumps(spec)
    assert wording.Q2_X_TITLE in text and _words(wording.Q2_X_TITLE) <= 5
    rules, labels = U.round_marks(opt, [int(n) for n in opt.formulation_ids])
    rounds = [r for r in dict.fromkeys(opt.batch_history) if r is not None]
    assert [t for _, t in labels] == [wording.round_tick(r) for r in rounds]
    assert len(rules) == len(rounds) - 1
    assert "'R1'" in text
    # One Round: nothing to mark.
    assert U.round_marks(D.load(D.SNACK), [1, 2, 3]) is None


def test_the_more_pickers_have_a_label_and_placeholder():
    import inspect
    src = inspect.getsource(U._profiler)
    assert src.count("placeholder=wording.profiler_more_placeholder(") == 2
    for kind in ("ingredients", "settings", "both", "rows"):
        assert _words(wording.profiler_more_label(kind)) <= 5
        assert _words(wording.profiler_more_placeholder(kind)) <= 4
    opt = D.load(D.PROCESS_ONLY)
    assert U._more_kind(opt, insights.path_choices(opt)) == "settings"


def test_the_strip_is_one_row_each_item_its_own_link():
    import inspect
    src = inspect.getsource(U._places)
    # Review story 2: the cards became one strip, one row, the primary at
    # its right end; each item is itself the link to its section.
    assert 'height="stretch"' not in src and "horizontal=True" in src
    assert src.index("_strip_item(opt, None, None, which)") < src.index("_foot(opt)")
    assert "on_click=ask_tab" in inspect.getsource(U._strip_item)
