"""Insights for a first-time user, W1 (spec 2026-10-01 rev. 2 §1, §6;
review R-I8, R-I17): on every demo project and the process-only study, at
1280 x 800 with nothing scrolled, the three status lines and Next's
primary button sit wholly above the fold once the tab has filled in, and
the Explored and unexplored fold row sits under them, ending by 640 px, in
light and dark (review R-I14).

The served copy must have the seven projects in its working folder:

    python tests/browser/test_107_first_screen.py install <folder>
    cd <folder> && HOME=<temp> python -m streamlit run <worktree>/app.py --server.port 8860 …
    APP_URL=http://127.0.0.1:8860 pytest tests/browser/test_107_first_screen.py

Skipped, never errored, without APP_URL (tests/browser/conftest.py)."""
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
    from grid_helpers import FIRST_SCREEN, URL, above_the_fold, settle   # noqa: E402

# The six demo projects and the process-only study (tests/demo_projects.ALL).
PROJECTS = ("Snack bar - first round", "Plant patty - Line A", "Plant patty - Line B",
            "Plant patty - Line C", "Upgrade check - patty 0.7",
            "Timing check - 60 formulations", "Oven trial - process settings")
# The places' captions (wording.CHECKING_PREDICTIONS, …): gone once filled.
_FILLING = ("Checking predictions…", "Checking progress…", "Preparing the next step…")
_LINES = ("Predictions:", "Progress:", "Next:")
# Review R-I14: the fold row sits under the button and its one link, before
# the detail sections. Its table estimated the bottom at <= 620 px from a tab
# that starts at 227 px; measured at integration the tab starts at 243 px, so
# the same rows end at 635 px on the projects with a three-line status line
# (Line B, Line C, Timing check) and at 527-608 px on the rest.
_EXPLORED = "Explored and unexplored"
_EXPLORED_BOTTOM = 640
# Fix round B (ledger: early five-second re-test, early walkthrough): the
# ruled sentences lengthen Line A's Progress line (the tie and "little
# more", +2 lines) and Line C's Predictions line (the skill with its
# meaning, the cause and the check, +3 lines); measured at 1280 x 800 the
# fold row then ends at 659 and 685 px, still above the fold.
# Concision round (2026-10-01): every status line is one short line again,
# so no project needs more room.
_EXPLORED_BOTTOM_LONGER = {}


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


def _main(page):
    """The Insights tab's own panel: every tab is in the page, the others
    hidden, and their buttons and lines must not count."""
    return page.locator('[data-testid="stMain"] [role="tabpanel"]:visible').first


def _filled(page):
    text = _main(page).inner_text()
    return all(f not in text for f in _FILLING) and all(
        _status(page, label).count() for label in _LINES)


def _status(page, label):
    return _main(page).locator('[data-testid="stMarkdownContainer"]').filter(
        has_text=re.compile("^" + re.escape(label)))


@pytest.fixture(scope="module")
def browser(playwright_session):
    b = playwright_session.chromium.launch(channel="chrome", headless=True)
    yield b
    b.close()


@pytest.mark.parametrize("theme", ("light", "dark"))
@pytest.mark.parametrize("name", PROJECTS)
def test_the_lines_and_the_button_are_above_the_fold(browser, name, theme):
    page = browser.new_context(viewport=FIRST_SCREEN, color_scheme=theme).new_page()
    try:
        page.goto(URL, wait_until="domcontentloaded")
        settle(page, 4)
        _open_project(page, name)
        page.get_by_role("tab", name="Review").first.click()
        deadline = time.time() + 300
        while not _filled(page):
            assert time.time() < deadline, f"{name}: the tab never filled in"
            page.wait_for_timeout(1000)
        settle(page, 2)
        for label in _LINES:
            line = _status(page, label).first
            assert above_the_fold(line), (name, label, line.bounding_box())
        primary = _main(page).locator('[data-testid="stBaseButton-primary"]')
        assert primary.count() == 1, name
        assert above_the_fold(primary.first), (name, primary.first.bounding_box())
        # The one link, when there is one, is above the fold too (R-I4).
        links = _main(page).locator('[data-testid="stBaseButton-tertiary"]').filter(
            has_text=re.compile(r"^(Why Round \d+ was suggested|Repeat (Formulation|Trial) \d+"
                                r"|See untried (amounts|values))"))
        for k in range(links.count()):
            assert above_the_fold(links.nth(k)), (name, links.nth(k).inner_text())
        # The Explored fold row, collapsed, under them (R-I14).
        fold = _main(page).locator('[data-testid="stExpander"] summary').filter(
            has_text=_EXPLORED).first
        box = fold.bounding_box()
        assert box is not None, name
        assert box["y"] + box["height"] <= _EXPLORED_BOTTOM_LONGER.get(
            name, _EXPLORED_BOTTOM), (name, theme, box)
        assert box["y"] > primary.first.bounding_box()["y"], (name, box)
        print(f"{name} [{theme}]: button bottom "
              f"{primary.first.bounding_box()['y'] + primary.first.bounding_box()['height']:.0f}"
              f" px, Explored fold bottom {box['y'] + box['height']:.0f} px")
    finally:
        page.context.close()


def install(folder):
    """Save the seven fixtures into `folder`, as the app opens them."""
    here = os.path.dirname(os.path.abspath(__file__))
    sys.path.insert(0, os.path.dirname(os.path.dirname(here)))
    import json
    from food_bo import FoodOptimizer
    from tests import demo_projects as D
    os.makedirs(folder, exist_ok=True)
    os.chdir(folder)
    for name in D.ALL:
        state = json.loads((D.HERE / f"{name}.json").read_text(encoding="utf-8"))
        opt = FoodOptimizer(state["project_name"])
        opt.import_json(state)
        opt.save()


if __name__ == "__main__" and sys.argv[1:2] == ["install"]:
    install(sys.argv[2])
