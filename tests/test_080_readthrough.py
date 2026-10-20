"""0.8.0 answer-key read-through fixes (docs/superpowers/research/
2026-09-29-speed-ux/answer-key-refresh.md §7): advice that must hold on the
numbers shown, one bias rule for Insights and Make a round, Repeats left out
said as left out, the width caution beside "still rising", and Results' and
Set up's words."""
from types import SimpleNamespace

import pytest
from streamlit.testing.v1 import AppTest

import wording
from tests.test_app_ui import APP_PATH
from tests.test_complete_results import _patty


@pytest.fixture(autouse=True)
def _isolate_cwd(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)


def test_a_custom_fields_own_name_is_kept_as_typed():
    """Re-review R6: only the scope is said in the project's words."""
    trials = SimpleNamespace(has_ingredients=lambda: False,
                             _process_settings=lambda: ["Incubation time"])
    assert wording.custom_record_option("Formulation lead", "formulation", trials) == (
        "Formulation lead · Each trial")
    assert wording.custom_record_option("Formulation lead", "formulation") == (
        "Formulation lead · Each formulation")


# ------------------------------------------------------------------ #
#  9. "trial" in an ingredient project's Preparation and records
# ------------------------------------------------------------------ #

def test_custom_records_say_formulation_and_trial_by_project():
    assert wording.CUSTOM_FIELD_HELP == ("For example, Operator for each formulation, or Expiry "
                                         "date for each ingredient. These records do not affect "
                                         "scores.")
    assert wording.CUSTOM_SCOPE_NAMES["formulation"] == "Each formulation"
    assert wording.CUSTOM_SECTION_NAMES["formulation"] == "Formulation records"
    trials = SimpleNamespace(has_ingredients=lambda: False,
                             _process_settings=lambda: ["Incubation time"])
    assert wording.for_project(trials, wording.CUSTOM_SCOPE_NAMES["formulation"]) == "Each trial"
    assert "Operator for each trial" in wording.for_project(trials, wording.CUSTOM_FIELD_HELP)


def test_set_up_never_says_trial_on_an_ingredient_project(tmp_path):
    opt = _patty(rows=3)
    at = AppTest.from_file(APP_PATH, default_timeout=300)
    at.session_state["_loaded_project"] = opt.project_name
    at.run()
    at.session_state["main_tab"] = wording.TAB_SETUP
    at.run()
    assert not at.exception
    said = ([c.value for c in at.tabs[0].caption] + [m.value for m in at.tabs[0].markdown]
            + [o for s in at.tabs[0].selectbox for o in s.proto.options])
    assert wording.CUSTOM_FIELD_HELP in said
    assert not [t for t in said if "trial" in t.lower()], [t for t in said if "trial" in t.lower()]
