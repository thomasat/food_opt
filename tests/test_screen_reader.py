"""0.9.0: what a screen reader hears that the screen does not show. The
Windows UI Automation walk heard every fold as "keyboard_arrow_right New
project" (Streamlit's Material icon is ligature text). A script in the
sidebar hides the folds' icons from assistive tech; the arrow still shows.
The browser side (aria-hidden on the icon, the slot's display: none) was
checked in headless Chromium; these pin what reaches the page."""
import pytest
from streamlit.testing.v1 import AppTest

import ui_helpers
from tests.test_app_ui import APP_PATH
from tests.test_insights import _burger


@pytest.fixture(autouse=True)
def _isolate_cwd(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)


def _sidebar_scripts(at):
    return [e.proto.body for e in at.sidebar.get("html")
            if getattr(e.proto, "unsafe_allow_javascript", False)]


def test_the_script_hides_only_the_folds_icons():
    script = ui_helpers.A11Y_SCRIPT
    assert 'summary [data-testid="stIconMaterial"]' in script
    assert 'setAttribute("aria-hidden", "true")' in script
    # Folds drawn later (another tab, a rerun) are caught too, and the
    # observer is installed once per page however often the script runs.
    assert "MutationObserver" in script and "dataset.foA11y" in script
    # Its own slot takes no room in the sidebar.
    assert ('[data-testid="stElementContainer"]:has([data-fo-a11y]) { display: none; }'
            in script)


def test_the_welcome_screen_carries_the_script():
    at = AppTest.from_file(APP_PATH, default_timeout=180)
    at.run()
    assert not at.exception
    assert _sidebar_scripts(at) == [ui_helpers.A11Y_SCRIPT]


def test_an_open_project_carries_the_script_on_every_run():
    opt = _burger(rounds=2)
    at = AppTest.from_file(APP_PATH, default_timeout=180)
    at.session_state["_loaded_project"] = opt.project_name
    at.run()
    assert not at.exception
    assert _sidebar_scripts(at) == [ui_helpers.A11Y_SCRIPT]
    at.run()
    assert _sidebar_scripts(at) == [ui_helpers.A11Y_SCRIPT]
