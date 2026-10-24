"""The shared page in a real browser engine, state by state, against the
Swift window it ports (FoodOptimizerApp.swift showInitialStatus …
updateStepsPage, readStatusFile)."""
import json

import pytest

from tests.desktop_page.conftest import call, text, visible
from tests.shared_texts import texts

WARM = {"firstRun": False}
FIRST = {"firstRun": True}


def run_for(page, ms):
    page.clock.run_for(ms)


def line(page, raw):
    call(page, f"statusLine({json.dumps(raw)})")


def width(page, sel="#bar"):
    return page.eval_on_selector(sel, "e => e.style.width")


def opacity(page, sel):
    return page.eval_on_selector(sel, "e => getComputedStyle(e).opacity")


def rows(page):
    return page.eval_on_selector_all("#steps li", "els => els.length")


# ---------- steps view ----------

@pytest.mark.parametrize("os_name", ["mac", "windows"])
def test_warm_launch_shows_two_steps_with_load_running(open_page, os_name):
    M = texts(os_name)
    page = open_page(os_name, view=WARM)
    assert page.query_selector("#row0") and page.query_selector("#row1")
    assert page.query_selector("#row2") is None
    assert text(page, "#label0") == M["steps.load"]
    assert text(page, "#label1") == M["steps.open"]
    assert visible(page, "#spin0") and visible(page, "#dot1")
    assert opacity(page, "#row1") == "0.45"
    assert text(page, "#title") == M["steps.title.starting"]
    assert text(page, "#note") == M["steps.note.usual"]
    assert text(page, "#elapsed") == "0 s"
    assert page.query_selector("#barwrap") is None or not visible(page, "#barwrap")
    assert text(page, "#detail") == ""
    assert text(page, "#nextup") == M["steps.next_up"]


def _counted(open_page):
    page = open_page("mac", view=WARM)
    run_for(page, 5000)
    assert text(page, "#elapsed") == "5 s"
    line(page, "Loading the app's components|")
    assert text(page, "#elapsed") == "0 s"
    run_for(page, 3000)
    assert text(page, "#elapsed") == "3 s"
    return page


def test_counter_counts_and_restarts_on_the_first_launcher_line(open_page):
    _counted(open_page)


def test_same_line_again_does_not_restart_the_counter(open_page):
    page = _counted(open_page)
    line(page, "Loading the app's components|")
    run_for(page, 1000)
    assert text(page, "#elapsed") == "4 s"


def test_starting_line_moves_to_the_open_step(open_page):
    page = open_page("mac", view=WARM)
    run_for(page, 2000)
    line(page, "Starting the app|")
    assert visible(page, "#tick0") and visible(page, "#spin1")
    assert text(page, "#elapsed") == "0 s"


def test_thirty_seconds_latches_still_loading_for_the_launch(open_page):
    M = texts("mac")
    page = open_page("mac", view=WARM)
    run_for(page, 30000)
    assert text(page, "#note") == M["steps.note.still_loading"]
    line(page, "Starting the app|")
    run_for(page, 1000)
    assert text(page, "#note") == M["steps.note.still_loading"]


def test_first_run_shows_three_steps_bar_and_download_sentence(open_page):
    M = texts("mac")
    page = open_page("mac", view=FIRST)
    assert rows(page) == 3
    assert text(page, "#label0") == M["steps.download"]
    assert text(page, "#title") == M["steps.title.setting_up"]
    assert text(page, "#note") == M["steps.note.first_run"]
    assert visible(page, "#barwrap")
    assert width(page) == "0%"


def _downloading(open_page):
    page = open_page("mac", view=FIRST)
    line(page, "Downloading Python (step 1 of 3)|2")
    assert text(page, "#detail") == "Downloading Python (step 1 of 3)"
    assert width(page) == "2%"
    return page


def test_download_line_moves_the_bar_and_shows_its_detail(open_page):
    _downloading(open_page)


def test_download_never_latches(open_page):
    page = _downloading(open_page)
    run_for(page, 31000)
    assert text(page, "#note") == texts("mac")["steps.note.first_run"]


def test_leaving_the_download_hides_bar_and_detail(open_page):
    M = texts("mac")
    page = _downloading(open_page)
    line(page, "Loading the app's components|")
    assert visible(page, "#tick0")
    assert not visible(page, "#barwrap")
    assert text(page, "#detail") == ""
    assert text(page, "#title") == M["steps.title.starting"]
    assert text(page, "#note") == M["steps.note.usual"]


def test_upgrade_line_adds_the_download_row(open_page):
    M = texts("mac")
    page = open_page("mac", view=WARM)
    line(page, "Updating components|2")
    assert rows(page) == 3
    assert text(page, "#title") == M["steps.title.updating"]
    assert text(page, "#note") == M["steps.note.upgrade"]
    assert visible(page, "#spin0")


def test_status_line_splits_on_the_last_bar_and_escapes_markup(open_page):
    page = open_page("mac", view=FIRST)
    line(page, "Installing a|b|38")
    assert text(page, "#detail") == "Installing a|b"
    assert width(page) == "38%"
    line(page, "Installing <b>x</b>|39")
    assert page.eval_on_selector("#detail", "e => e.textContent") == "Installing <b>x</b>"
    assert page.eval_on_selector("#detail", "e => e.children.length") == 0


def test_percent_is_clamped_and_empty_text_ignored(open_page):
    page = open_page("mac", view=FIRST)
    line(page, "Installing components|150")
    assert width(page) == "100%"
    before = page.eval_on_selector("#steps-view", "e => e.innerHTML")
    line(page, "|5")
    assert page.eval_on_selector("#steps-view", "e => e.innerHTML") == before
    line(page, "Installing components|abc")
    assert width(page) == "100%"


def test_server_started_makes_open_active(open_page):
    page = open_page("mac", view=WARM)
    call(page, "serverStarted()")
    assert visible(page, "#spin1") and visible(page, "#tick0")


def test_windows_first_opening_says_setting_up(open_page):
    M = texts("windows")
    page = open_page("windows", view={"firstRun": False, "firstOpening": True})
    assert rows(page) == 2
    assert text(page, "#title") == M["steps.title.setting_up"]
    assert text(page, "#note") == M["steps.note.first_opening"]
    run_for(page, 30000)
    assert text(page, "#note") == M["steps.note.still_loading"]


@pytest.mark.parametrize("version,shown", [("0.9.0", "Food Optimizer 0.9.0"),
                                           ("", "Food Optimizer")])
def test_footer_shows_the_version_or_just_the_name(open_page, version, shown):
    page = open_page("mac", view=WARM, version=version)
    assert text(page, "#footer") == shown


# ---------- error, stalled and Try again ----------

KINDS = {"could_not_start": True, "missing_part": False, "unsupported": False,
         "needs_internet": True, "no_space": True, "stopped": True}


def _error_cases():
    cases = [("mac", k) for k in KINDS]
    w = texts("windows")
    cases += [("windows", k) for k in KINDS
              if f"error.{k}.title" in w and f"error.{k}.body" in w]
    return cases


@pytest.mark.parametrize("os_name,kind", _error_cases())
def test_error_page_texts_and_retry(open_page, os_name, kind):
    M = texts(os_name)
    page = open_page(os_name, view=WARM)
    call(page, f"showError({json.dumps(kind)})")
    assert visible(page, "#message-view") and not visible(page, "#steps-view")
    assert text(page, "#msg-title") == M[f"error.{kind}.title"]
    assert text(page, "#msg-body") == M[f"error.{kind}.body"]
    assert visible(page, "#retry") == KINDS[kind]
    if KINDS[kind]:
        assert text(page, "#retry") == M["button.try_again"]
    assert text(page, "#msg-footer") == "Food Optimizer 0.9.0"
    for sel in ("#msg-spinner", "#msg-step", "#msg-barwrap"):
        assert not visible(page, sel), sel


def test_windows_cases_cover_the_windows_errors():
    assert {k for o, k in _error_cases() if o == "windows"} == {
        "could_not_start", "missing_part", "unsupported", "stopped"}


def test_could_not_start_with_the_another_copy_detail(open_page):
    M = texts("mac")
    page = open_page("mac", view=WARM)
    call(page, 'showError("could_not_start", "error.could_not_start.another_copy")')
    assert text(page, "#msg-body") == (M["error.could_not_start.another_copy"] + " "
                                       + M["error.could_not_start.body"])


def test_try_again_posts_retry_to_the_shell(open_page):
    page = open_page("mac", view=WARM)
    call(page, 'showError("stopped")')
    page.click("#retry")
    assert page.evaluate("window.__posted") == ["retry"]


def test_try_again_without_a_shell_bridge_does_nothing(open_page):
    page = open_page("mac", view=WARM, bridge=False)
    call(page, 'showError("stopped")')
    page.click("#retry")
    assert page.errors == []


def test_stalled_page_keeps_its_step_line_moving(open_page):
    M = texts("mac")
    page = open_page("mac", view=FIRST)
    first = "Installing components (step 3 of 3): 500 MB of about 1,750 MB"
    line(page, first + "|38")
    call(page, "showStalled()")
    assert text(page, "#msg-title") == M["stalled.title"]
    assert text(page, "#msg-body") == M["stalled.body"]
    assert text(page, "#msg-step") == first
    assert width(page, "#msg-bar") == "38%"
    assert visible(page, "#msg-spinner")
    second = "Installing components (step 3 of 3): 600 MB of about 1,750 MB"
    line(page, second + "|44")
    assert text(page, "#msg-step") == second
    assert width(page, "#msg-bar") == "44%"
    assert not visible(page, "#steps-view")


def test_stalled_page_without_a_percent_slides(open_page):
    M = texts("mac")
    page = open_page("mac", view=WARM)
    call(page, "showStalled()")
    assert visible(page, "#msg-slide")
    assert not visible(page, "#msg-bar")
    assert text(page, "#msg-step") == M["stalled.step_fallback"]


def test_steps_after_an_error_rebuild_cleanly(open_page):
    M = texts("mac")
    page = open_page("mac", view=WARM)
    run_for(page, 4000)
    call(page, 'showError("stopped")')
    call(page, "showSteps({firstRun: false})")
    assert visible(page, "#steps-view") and not visible(page, "#message-view")
    assert text(page, "#elapsed") == "0 s"
    assert text(page, "#note") == M["steps.note.usual"]
