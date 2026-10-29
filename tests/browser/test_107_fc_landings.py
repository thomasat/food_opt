"""Insights for a first-time user, fix round C, in the browser at 1280 x
800: where each Insights link lands (ledger, "Early walkthrough" rulings;
05-walkthrough-retest-early.md §3, §5 items 2, 5, 9).

- "Why Round 6 was suggested" (Lines A and C): Make a round, the Why
  fold open, its heading just under the app's header bar and its first line
  the role key, all in view.
- "Repeat Formulation 24", with no Round open (a copy of the
  Timing check): Make a round at the top with the one-line instruction;
  after Generate, "Add a formulation to this round" open, in view, filled
  in from Formulation 24 with the note "Repeat of Formulation 24", and
  nothing added to the Round.
- The same link with a Round of the user's own open (another copy, its
  Round 13 holding one own formulation, so no Why link): the Add fold open
  and filled in at once, in view.
- "See untried amounts" (a copy of the Timing check whose
  noise threshold is measured, so Next asks for no Repeat and the Explored
  link has the place): the Explored and unexplored fold open, in view.

The served copy needs the demo projects and the three copies:

    python tests/browser/test_107_fc_landings.py install <folder>
    cd <folder> && HOME=<temp> python -m streamlit run <worktree>/app.py --server.port 8890 …
    APP_URL=http://127.0.0.1:8890 pytest tests/browser/test_107_fc_landings.py

The no-Round test generates a Round in its copy: install again before a
second run. Skipped, never errored, without APP_URL."""
import os
import re
import sys
import time

import pytest

pytestmark = pytest.mark.skipif(
    not os.environ.get("APP_URL"),
    reason="set APP_URL to a served copy of the app")

if __name__ != "__main__":
    pytest.importorskip("playwright.sync_api")
    from grid_helpers import FIRST_SCREEN, URL, settle   # noqa: E402

LINE_A, LINE_C = "Plant patty - Line A", "Plant patty - Line C"
TIMING = "Timing check - 60 formulations"
# The three copies install() writes.
REPEAT_COPY = "Timing check - Repeat link"
OWN_ROUND_COPY = "Timing check - own Round open"
EXPLORED_COPY = "Timing check - Repeats measured"
ROLE_KEY = ("Refines the best = a small change to your best. Explores = far from anything "
            "made so far. Balanced = in between.")
# Streamlit's header bar is 60 px; a landing puts its place just under it.
_HEADER = 60
_LANDED = (_HEADER, _HEADER + 40)


@pytest.fixture(scope="module")
def browser(playwright_session):
    b = playwright_session.chromium.launch(channel="chrome", headless=True)
    yield b
    b.close()


@pytest.fixture
def page(browser):
    p = browser.new_context(viewport=FIRST_SCREEN).new_page()
    p.goto(URL, wait_until="domcontentloaded")
    settle(p, 4)
    yield p
    p.context.close()


def _open_project(page, name):
    sidebar = page.locator('[data-testid="stSidebar"]')
    page.get_by_text("Open project").first.wait_for(timeout=60000)
    sidebar.locator('[data-testid="stSelectbox"]').first.click()
    page.get_by_role("option", name=name, exact=True).click()
    settle(page, 1.5)
    button = sidebar.get_by_role("button", name="Open", exact=True)
    if button.is_enabled():
        button.click()
    settle(page, 3)


def _panel(page):
    return page.locator('[data-testid="stMain"] [role="tabpanel"]:visible').first


def _insights_link(page, name, label):
    """Open `name`, then Insights, and wait (up to 5 min) for the link."""
    _open_project(page, name)
    page.get_by_role("tab", name="Review").first.click()
    link = _panel(page).get_by_role("button", name=label)
    link.first.wait_for(timeout=300000)
    settle(page, 2)
    return link.first


def _place(page, key):
    return page.locator(f".st-key-{key}").first


def _landed(page, key, label):
    """The place `key` scrolled to just under the header, holding the fold
    `label`, open. Retries while the page settles."""
    deadline = time.time() + 20
    while True:
        box = _place(page, key).bounding_box()
        if box is not None and _LANDED[0] <= box["y"] <= _LANDED[1]:
            break
        assert time.time() < deadline, (key, box)
        page.wait_for_timeout(500)
    fold = _place(page, key).locator("details").first
    assert fold.locator("summary").inner_text().strip().endswith(label)
    assert fold.get_attribute("open") is not None, key
    return box


def _in_view(locator):
    box = locator.bounding_box()
    assert box is not None
    assert box["y"] >= _HEADER and box["y"] + box["height"] <= FIRST_SCREEN["height"], box
    return box


@pytest.mark.parametrize("name", (LINE_A, LINE_C))
def test_the_why_link_lands_on_the_open_why_fold(page, name):
    _insights_link(page, name, "Why Round 6 was suggested").click()
    box = _landed(page, "why_these_place", "Why these suggestions")
    key = _place(page, "why_these_place").get_by_text(ROLE_KEY)
    _in_view(key)
    print(f"{name}: Why place at {box['y']:.0f} px; role key at "
          f"{key.bounding_box()['y']:.0f} px")


def _own_form(page):
    """The Add fold's note box and the amounts' boxes, as the page holds them."""
    place = _place(page, "add_own_place")
    note = place.get_by_label("Note").first
    return place, note


def test_the_repeat_link_with_no_round_open_waits_for_generate(page):
    _insights_link(page, REPEAT_COPY, "Repeat Formulation 24").click()
    settle(page, 3)
    panel = _panel(page)
    line = panel.get_by_text(re.compile(r"^Click Generate first: the Repeat of Formulation 24"))
    _in_view(line.first)
    place, note = _own_form(page)
    assert place.locator("details").first.get_attribute("open") is None   # not yet
    panel.get_by_role("button", name=re.compile(r"^Generate \d")).first.click()
    settle(page, 6)
    _landed(page, "add_own_place", "Add a formulation to this round")
    place, note = _own_form(page)
    # The fold's first box is in view; with seven boxes, the note and Add
    # sit just under the fold line (the fold is about 820 px tall).
    boxes = place.locator('[data-testid="stNumberInput"] input')
    _in_view(boxes.first)
    assert note.input_value() == "Repeat of Formulation 24"
    assert boxes.count() >= 2
    assert all(boxes.nth(k).input_value() != "" for k in range(boxes.count())
               if boxes.nth(k).is_enabled())
    # Nothing added: no "added" flash, and Add is still there to click.
    assert not page.get_by_text(re.compile(r"added to Round")).count()
    assert place.get_by_role("button", name="Add to this round").count() == 1


def test_the_repeat_link_with_a_round_open_fills_the_add_fold(page):
    _insights_link(page, OWN_ROUND_COPY, "Repeat Formulation 24").click()
    _landed(page, "add_own_place", "Add a formulation to this round")
    place, note = _own_form(page)
    _in_view(place.locator('[data-testid="stNumberInput"] input').first)
    assert note.input_value() == "Repeat of Formulation 24"
    # Nothing added: the Round still holds the one formulation of its own.
    assert not page.get_by_text(re.compile(r"added to Round")).count()


def test_the_explored_link_lands_on_the_open_fold(page):
    _insights_link(page, EXPLORED_COPY, re.compile(r"^See untried amounts")).click()
    box = _landed(page, "insights_explored_place", "Explored and unexplored")
    strips = _place(page, "insights_explored_place").get_by_text("Where you have tested")
    _in_view(strips.first)
    print(f"Explored place at {box['y']:.0f} px")


def install(folder):
    """The demo projects (test_107_first_screen.install) and three copies of
    the Timing check: as it is; with Round 13 open holding one formulation
    of the user's own; and with three Repeats of Formulation 24 recorded in
    Round 13, so the noise threshold is measured and Next asks for no
    Repeat."""
    here = os.path.dirname(os.path.abspath(__file__))
    sys.path.insert(0, here)
    sys.path.insert(0, os.path.dirname(os.path.dirname(here)))
    import json
    from food_bo import FoodOptimizer
    from tests import demo_projects as D
    from test_107_first_screen import install as install_demos
    install_demos(folder)
    os.chdir(folder)
    state = json.loads((D.HERE / f"{D.TIMING}.json").read_text(encoding="utf-8"))

    def copy(name):
        opt = FoodOptimizer(name)
        opt.import_json({**state, "project_name": name})
        opt.project_name = name
        return opt

    copy(REPEAT_COPY).save()
    opt = copy(OWN_ROUND_COPY)
    best = opt.best_formulation_no()
    opt.add_to_pending_batch(dict(opt.recipe_history[opt.index_of_formulation(best)]),
                             "My own: the bench standard")
    opt.save()
    opt = copy(EXPLORED_COPY)
    index = opt.index_of_formulation(opt.best_formulation_no())
    recipe, results = dict(opt.recipe_history[index]), dict(opt.results_history[index])
    for shift in (-0.03, -0.01, 0.01):
        opt.tell(recipe, {m: v * (1 + shift) for m, v in results.items()},
                 note=f"Repeat of Formulation {opt.best_formulation_no()}", batch_no=13)
    opt.save()


if __name__ == "__main__" and sys.argv[1:2] == ["install"]:
    install(sys.argv[2])
