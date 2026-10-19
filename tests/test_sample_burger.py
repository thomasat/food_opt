"""Example setup - plant-based burger (spec 2026-10-13 §9): a second built-in
project that shows how to set one up — calculated rows, pre-mixes, a process
setting, records, the formulation total — with illustrative values and no
results. It opens on 1 · Set up; the yogurt pilot stays the only demo."""
import importlib.util
import json
import re
from pathlib import Path

import pytest
from streamlit.testing.v1 import AppTest

import sample_projects as sp
import storage as storage_backend
import wording
from food_bo import FoodOptimizer
from ui_helpers import TAB_SETUP

APP = str(Path(__file__).resolve().parent.parent / "app.py")
TS = re.compile(r'"20\d\d-\d\d-\d\d(T[^"]*)?"')
SHIPPED = sp.BURGER_DIR / f"{sp.BURGER}.json"


def _builder():
    spec = importlib.util.spec_from_file_location("burger_build", sp.BURGER_DIR / "build.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


@pytest.fixture
def store(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)          # LocalStorage writes into the cwd
    return storage_backend.LocalStorage()


@pytest.fixture
def at(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    app = AppTest.from_file(APP, default_timeout=180)
    app.run()
    return app


def _click(where, label):
    [b for b in where.button if b.label == label][0].click()


def _burger(store):
    sp.ensure(store, (sp.BURGER,))
    return FoodOptimizer(sp.BURGER, storage=store)


# ------------------------------------------------------------------ #
#  The shipped file
# ------------------------------------------------------------------ #

def test_the_names_are_exact_and_say_what_each_is_for():
    assert sp.BURGER == "Example setup - plant-based burger"
    assert sp.YOGURT_NAMES == ("Yogurt pilot - Round 6 to score", "Yogurt pilot - all 10 Rounds")
    assert sp.NAMES == sp.YOGURT_NAMES + (sp.BURGER,)
    for name in sp.NAMES:
        assert "Advanced" not in name and "Demo" not in name
    assert SHIPPED.is_file() and sp.burger_available()
    assert sp.is_sample(sp.BURGER) and not sp.is_yogurt(sp.BURGER)
    assert all(sp.is_yogurt(n) for n in sp.YOGURT_NAMES)
    for other in (sp.BURGER + " copy", sp.BURGER + "_sample_reset",
                  "example setup - plant-based burger", "Sample project"):
        assert not sp.is_sample(other)


def test_the_shipped_burger_equals_a_rebuild(tmp_path):
    _builder().build(out_dir=str(tmp_path))
    rebuilt = (tmp_path / f"{sp.BURGER}.pkl").read_text()
    assert TS.sub('"<ts>"', rebuilt) == TS.sub('"<ts>"', SHIPPED.read_text())


def test_the_shipped_burger_carries_no_date_no_results_and_no_open_round():
    text = SHIPPED.read_text()
    assert not TS.search(text)
    state = json.loads(text)
    for key in ("X_history", "Y_history", "results_history", "recipe_history",
                "timestamps_history", "batch_history", "skipped"):
        assert not state[key], key
    assert state["pending_batch"] is None and state["pending_batch_created"] is None
    assert state["custom_records"]["values"] == {} and not state["lots"]


_MONTHS = re.compile(
    r"\b(January|February|March|April|May|June|July|August|September|October|"
    r"November|December|Jan|Feb|Mar|Apr|Jun|Jul|Aug|Sep|Sept|Oct|Nov|Dec)\b")


def test_nothing_in_the_burger_reads_as_measured_or_sourced():
    """Beside the real yogurt pilot, the example never reads like real data
    (controller ruling, 2026-10-14): no date, no year, no month, no panel
    or panellists, no study."""
    text = SHIPPED.read_text()
    assert not TS.search(text)

    def strings(node):
        if isinstance(node, str):
            yield node
        elif isinstance(node, dict):
            for k, v in node.items():
                yield k
                yield from strings(v)
        elif isinstance(node, list):
            for v in node:
                yield from strings(v)

    # Every text the project holds (an amount such as 2000 mg is a number,
    # not a year).
    words = "\n".join(strings(json.loads(text)))
    assert not re.search(r"\b(19|20)\d\d\b", words)
    assert not re.search(r"\d{1,4}[-/.]\d{1,2}[-/.]\d{1,4}", words)
    assert not _MONTHS.search(words)
    assert not re.search(r"panell?ists?", text, re.I)
    for word in ("our panel", "et al", "doi.org", "study", "sessions"):
        assert word not in text.lower(), word
    state = json.loads(text)
    assert state["targets_source"] == "Illustrative targets for this example; not from a panel."
    assert state["method"].startswith("Example method (illustrative)\n")


def test_the_burger_loads_at_this_class_version(store):
    state = sp.shipped_state(sp.BURGER)
    assert state["CLASS_VERSION"] == FoodOptimizer.CLASS_VERSION
    FoodOptimizer.validate_state(state)
    opt = _burger(store)
    assert not opt.load_error
    assert not opt.X_history and not opt.pending_batch


def test_the_burger_shows_every_set_up_feature(store):
    opt = _burger(store)
    rows = {v["name"]: v for v in opt.variables}
    assert list(opt.ingredient_grid_frame()[wording.NAME_LABEL]) == [
        "Textured pea protein", "Textured soy protein", "Hydration water", "Remaining water",
        "Dry blend", "Wheat gluten", "Seasoning blend", "Fats and oils", "Mixing time after fat"]
    # Calculated rows: hydration as 2.2 x protein, the water "= rest".
    assert rows["Hydration water"]["formula"] == (
        "= 2.2 * (Textured pea protein + Textured soy protein)")
    assert rows["Remaining water"]["formula"] == "= rest"
    # Pre-mixes, the process setting and the formulation total.
    assert {n: p["mode"] for n, p in opt.premixes.items()} == {
        "Dry blend": "portioned", "Fats and oils": "weighed", "Seasoning blend": "portioned"}
    assert opt.premixes["Dry blend"]["smallest"] == 150
    assert rows["Mixing time after fat"]["category"] == "process"
    assert opt.formulation_total == 100
    # Measurements, their targets and shares.
    assert {o["name"]: o.get("target") for o in opt.objectives} == {
        "Juiciness": 7, "Firmness": 6, "Cook loss": None}
    shares = {o["name"]: round(o["weight"]) for o in opt.objectives}
    assert shares == {"Firmness": 45, "Juiciness": 35, "Cook loss": 20}
    # The fat limit, the solids limit, the records and the texts.
    assert [(c["metric"].lower(), c.get("max")) for c in opt.constraints] == [("fat per 100 g", 16)]
    solids = [qc for qc in opt.quantity_constraints if qc.get("source") is None]
    assert [(sorted(q["ingredients"]), q["min"], q["max"]) for q in solids] == [
        (sorted(_builder().SOLIDS), 36.0, 43.0)]
    assert opt.records("lot") and opt.records("actual")
    assert [f["name"] for f in opt.custom_records["fields"]] == _builder().BENCH_RECORDS
    assert opt.targets_source == "Illustrative targets for this example; not from a panel."
    method = "\n".join(opt.method_lines())
    assert method.splitlines()[0] == "Example method (illustrative)"
    assert method.splitlines()[1].startswith("1. Hydrate")
    for fact in ("45 °C", "≤ 5 °C", "180 °C", "74 °C core", "60 s on low",
                 "Cook loss = (raw − cooked)/raw × 100"):
        assert fact in method, fact
    assert "illustrative" in method.splitlines()[-1]


def test_suggestions_for_the_burger_add_up_and_keep_its_rules(store):
    """The example's own rules hold in what the app suggests: the total, the
    rest row, the hydration ratio, the fixed seasoning, the solids and the
    mixer's 5 s steps."""
    opt = _burger(store)
    solids = _builder().SOLIDS
    for recipe in opt.ask(3):
        ingredients = {k: v for k, v in recipe.items() if k != "Mixing time after fat"}
        assert sum(ingredients.values()) == pytest.approx(100)
        assert recipe["Seasoning blend"] == 2.2
        assert recipe["Remaining water"] >= 0
        assert recipe["Hydration water"] == pytest.approx(
            2.2 * (recipe["Textured pea protein"] + recipe["Textured soy protein"]))
        assert 45 <= recipe["Mixing time after fat"] <= 150
        assert 36.0 <= sum(recipe[n] for n in solids) <= 43.0 + 1e-6


# ------------------------------------------------------------------ #
#  ensure / restore: one never writes the other
# ------------------------------------------------------------------ #

def test_ensure_writes_the_yogurt_only_and_ensure_burger_the_burger_only(store):
    assert sorted(sp.ensure(store)) == sorted(sp.YOGURT_NAMES)
    assert not Path(f"{sp.BURGER}.pkl").exists()
    assert sp.ensure(store, (sp.BURGER,)) == [sp.BURGER]
    assert sp.ensure(store, (sp.BURGER,)) == []
    assert sorted(store.list_projects()) == sorted(sp.NAMES)


def test_the_burger_alone_writes_no_yogurt_and_no_results_file(store):
    assert sp.ensure(store, (sp.BURGER,)) == [sp.BURGER]
    assert sorted(store.list_projects()) == [sp.BURGER]
    assert not Path(sp.ROUND6_RESULTS_FILE_NAME).exists()


def test_ensure_burger_never_overwrites_it(store):
    opt = _burger(store)
    opt.set_formulation_total(250)
    sp.ensure(store, (sp.BURGER,))
    assert FoodOptimizer(sp.BURGER, storage=store).formulation_total == 250


def test_restore_burger_archives_then_writes_the_shipped_copy(store):
    opt = _burger(store)
    opt.set_formulation_total(250)
    archived = sp.restore(sp.BURGER, store)
    assert archived == f"{sp.BURGER}_{sp.RESET_LABEL}"
    assert FoodOptimizer(sp.BURGER, storage=store).formulation_total == 100
    assert json.loads(Path(f"{archived}.pkl").read_text())["formulation_total"] == 250
    assert sorted(store.list_projects()) == [sp.BURGER]
    assert not Path(sp.ROUND6_RESULTS_FILE_NAME).exists()


# ------------------------------------------------------------------ #
#  The app
# ------------------------------------------------------------------ #

def test_welcome_offers_both_with_their_captions_burger_primary(at):
    # Owner, 2026-10-19: the burger example is the default, the yogurt
    # pilot the option beside it, burger first.
    assert not at.exception
    buttons = {b.label: b for b in at.main.button}
    yogurt = buttons["Try the yogurt pilot"]
    example = buttons["Open the burger example"]
    assert example.proto.type == "primary"
    assert yogurt.proto.type == "secondary"
    order = [b.label for b in at.main.button]
    assert order.index("Open the burger example") < order.index("Try the yogurt pilot"), order
    captions = [c.value for c in at.main.caption]
    assert "Real tasting data; shows results and the Review." in captions
    assert "Illustrative values; shows calculations and pre-mixes." in captions
    assert wording.OPEN_YOGURT_LABEL == "Try the yogurt pilot"
    assert wording.SEE_EXAMPLE_SETUP_LABEL == "Open the burger example"
    assert "Try the sample project" not in buttons
    # The template stays, as a quiet link-style download.
    [template] = [d for d in at.get("download_button") if d.label == wording.DOWNLOAD_TEMPLATE]
    assert template.proto.type == "tertiary"


def test_opening_the_yogurt_never_writes_the_burger(at, tmp_path):
    _click(at.main, wording.OPEN_YOGURT_LABEL)
    at.run()
    assert not at.exception
    assert at.session_state["_loaded_project"] == sp.ROUND6
    assert at.session_state["main_tab"] == wording.TAB_INSIGHTS
    assert not (tmp_path / f"{sp.BURGER}.pkl").exists()
    assert sp.BURGER not in at.sidebar.selectbox(key="project_select").options


def test_the_example_setup_opens_on_set_up_and_writes_no_yogurt(at, tmp_path):
    _click(at.main, wording.SEE_EXAMPLE_SETUP_LABEL)
    at.run()
    assert not at.exception
    assert at.session_state["_loaded_project"] == sp.BURGER
    assert at.session_state["main_tab"] == TAB_SETUP
    assert sorted(p.stem for p in tmp_path.glob("*.pkl")) == [sp.BURGER]
    assert not (tmp_path / sp.ROUND6_RESULTS_FILE_NAME).exists()
    # The sidebar says what it is: illustrative, no results.
    captions = {c.value: c for c in at.sidebar.caption}
    assert "Example setup; illustrative values, no results." in captions
    assert wording.SAMPLE_PROVENANCE_CAPTION not in captions
    assert "illustrative" in captions["Example setup; illustrative values, no results."].proto.help


def test_the_example_setup_shows_its_pre_mixes_total_and_targets(at):
    _click(at.main, wording.SEE_EXAMPLE_SETUP_LABEL)
    at.run()
    assert not at.exception
    setup = at.tabs[0]
    assert any(c.value == wording.BLEND_OILS_EXAMPLE for c in at.caption)
    total = next(n for n in at.number_input if n.key == "formulation_total")
    assert total.value == 100.0
    opt = at.session_state["optimizer"]
    assert (wording.targets_from_caption(opt.targets_source)
            in [m.value for m in setup.markdown])


def test_the_example_setups_review_says_no_results_and_opens_the_yogurt(at, tmp_path):
    _click(at.main, wording.SEE_EXAMPLE_SETUP_LABEL)
    at.run()
    at.session_state["main_tab"] = wording.TAB_INSIGHTS
    at.run()
    assert not at.exception
    review = at.tabs[3]
    assert "No results: this example shows set-up only." in [c.value for c in review.caption]
    [button] = [b for b in review.button if b.label == "Try the yogurt pilot"]
    assert button.proto.type == "secondary"
    # AppTest sends no tab with a click: the tab is set again, as
    # tests/test_sample_buttons.py does for See what happened next.
    button.click()
    at.session_state["main_tab"] = wording.TAB_INSIGHTS
    at.run()
    assert not at.exception
    assert at.session_state["_loaded_project"] == sp.ROUND6
    assert at.session_state["main_tab"] == wording.TAB_INSIGHTS
    assert all((tmp_path / f"{n}.pkl").exists() for n in sp.YOGURT_NAMES)


def test_manage_project_has_a_samples_group_with_both(at):
    _click(at.main, wording.SEE_EXAMPLE_SETUP_LABEL)
    at.run()
    [manage] = [e for e in at.sidebar.expander if e.label == wording.MANAGE_PROJECT]
    assert "**Samples**" in [m.value for m in manage.markdown]
    labels = [b.label for b in manage.button]
    assert "Try the yogurt pilot" in labels and "Open the burger example" in labels
    captions = [c.value for c in manage.caption]
    assert "Real tasting data; shows results and the Review." in captions
    assert "Illustrative values; shows calculations and pre-mixes." in captions
    # Start this sample over is offered on the example setup too.
    assert wording.START_SAMPLE_OVER_LABEL in labels
    assert wording.START_OVER_LABEL not in labels
    _click(manage, wording.OPEN_YOGURT_LABEL)
    at.run()
    assert at.session_state["_loaded_project"] == sp.ROUND6
    assert at.session_state["main_tab"] == wording.TAB_INSIGHTS


def test_start_this_sample_over_restores_the_burger_on_set_up(at, tmp_path):
    _click(at.main, wording.SEE_EXAMPLE_SETUP_LABEL)
    at.run()
    state = json.loads((tmp_path / f"{sp.BURGER}.pkl").read_text())
    state["formulation_total"] = 250.0
    (tmp_path / f"{sp.BURGER}.pkl").write_text(json.dumps(state))
    at.session_state["main_tab"] = wording.TAB_INSIGHTS
    at.run()
    _click(at.sidebar, wording.START_SAMPLE_OVER_LABEL)
    at.run()
    _click(at.sidebar, wording.YES_START_SAMPLE_OVER)
    at.run()
    assert not at.exception
    assert json.loads((tmp_path / f"{sp.BURGER}.pkl").read_text())["formulation_total"] == 100.0
    assert (tmp_path / f"{sp.BURGER}_sample_reset.pkl").exists()
    assert [s.value for s in at.success] == [wording.sample_restored(sp.BURGER)]
    assert at.session_state["main_tab"] == TAB_SETUP
    assert sorted(p.stem for p in tmp_path.glob("*.pkl")) == sorted(
        [sp.BURGER, f"{sp.BURGER}_sample_reset"])


def test_launching_and_reopening_the_burger_land_on_set_up(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    store = storage_backend.LocalStorage()
    sp.ensure(store, (sp.BURGER,))
    app = AppTest.from_file(APP, default_timeout=180)
    app.run()                                       # the most recent project
    assert not app.exception
    assert app.session_state["_loaded_project"] == sp.BURGER
    assert app.session_state["main_tab"] == TAB_SETUP
    # Reopened from the sidebar after another project.
    sp.ensure(store)
    app.run()
    app.selectbox(key="project_select").set_value(sp.ALL10).run()
    _click(app.sidebar, wording.OPEN_BUTTON)
    app.run()
    assert app.session_state["main_tab"] == wording.TAB_INSIGHTS
    app.selectbox(key="project_select").set_value(sp.BURGER).run()
    _click(app.sidebar, wording.OPEN_BUTTON)
    app.run()
    assert not app.exception
    assert app.session_state["_loaded_project"] == sp.BURGER
    assert app.session_state["main_tab"] == TAB_SETUP


def test_the_burger_never_enters_the_yogurt_demo_flow(at, tmp_path):
    """Try the yogurt pilot, start it over, See what happened next: no step
    writes, lists or opens the example setup."""
    _click(at.main, wording.OPEN_YOGURT_LABEL)
    at.run()
    _click(at.sidebar, wording.START_SAMPLE_OVER_LABEL)
    at.run()
    _click(at.sidebar, wording.YES_START_SAMPLE_OVER)
    at.run()
    at.session_state[sp.OPEN_NEXT_KEY] = True
    at.run()
    assert not at.exception
    assert at.session_state["_loaded_project"] == sp.ALL10
    assert not (tmp_path / f"{sp.BURGER}.pkl").exists()
    assert sp.BURGER not in at.sidebar.selectbox(key="project_select").options
    assert wording.BURGER_PROVENANCE_CAPTION not in [c.value for c in at.sidebar.caption]


def test_the_new_wording_keeps_the_owners_caps():
    for label in (wording.OPEN_YOGURT_LABEL, wording.SEE_EXAMPLE_SETUP_LABEL):
        assert len(label.split()) <= 5
    for caption in (wording.OPEN_YOGURT_CAPTION, wording.SEE_EXAMPLE_SETUP_CAPTION,
                    wording.BURGER_PROVENANCE_CAPTION, wording.BURGER_PROVENANCE_CAPTION_USED):
        assert len(caption.split()) <= 8
    assert len(wording.BURGER_NO_RESULTS.split()) <= 10
    assert wording.BURGER_PROVENANCE_HELP.count(". ") <= 2


# ------------------------------------------------------------------ #
#  Fix round 2: every line stays true once the example is used
# ------------------------------------------------------------------ #

def _used_burger(state):
    """The example with a Round open ("round") or one result recorded
    ("result"), written to the cwd before the app opens it."""
    store = storage_backend.LocalStorage()
    opt = _burger(store)
    recipes = opt.ask(3)                           # opens Round 1
    assert opt.pending_batch and not sp.untouched_example(opt)
    if state == "result":
        row = opt.pending_batch[0]
        opt.tell(row["recipe"], {"Juiciness": 6.5, "Firmness": 6.0, "Cook loss": 20.0},
                 formulation_no=row["formulation"], batch_no=opt.pending_batch_no)
        opt.set_pending_batch(None)
        assert opt.X_history and not opt.pending_batch
    opt.save()
    return recipes


def _app_on(name, tab):
    app = AppTest.from_file(APP, default_timeout=300)
    app.session_state["_loaded_project"] = name
    app.run()
    app.session_state["main_tab"] = tab
    app.run()
    assert not app.exception, [e.message for e in app.exception]
    return app


def test_untouched_example_is_the_shipped_state_only(store):
    opt = _burger(store)
    assert sp.untouched_example(opt)
    sp.ensure(store)
    assert not sp.untouched_example(FoodOptimizer(sp.ROUND6, storage=store))


@pytest.mark.parametrize("state", ["round", "result"])
def test_a_used_example_drops_no_results_and_shows_the_normal_review(
        tmp_path, monkeypatch, state):
    monkeypatch.chdir(tmp_path)
    _used_burger(state)
    at = _app_on(sp.BURGER, wording.TAB_INSIGHTS)
    captions = {c.value: c for c in at.sidebar.caption}
    assert "Example setup; illustrative values." in captions
    assert wording.BURGER_PROVENANCE_CAPTION not in captions
    assert "no results" not in captions["Example setup; illustrative values."].proto.help.lower()
    review = at.tabs[3]
    assert wording.BURGER_NO_RESULTS not in [c.value for c in review.caption]
    assert "Try the yogurt pilot" not in [b.label for b in review.button]
    if state == "round":
        # The normal no-results Review, with its foot back to the open Round.
        assert wording.INSIGHTS_NONE in [c.value for c in review.caption]
        foot = next(b for b in review.button if b.key == "insights_foot")
        assert foot.label == wording.back_to_batch_label(1, 3)
