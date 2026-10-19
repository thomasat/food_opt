"""0.8.0 speed and simplicity, Plan B, Task 6: the Review's foot button
(spec 2026-09-30 §2; review M13). The old Insights questions, folds and
status lines went with the model-free Review (v3)."""
import pytest
from streamlit.testing.v1 import AppTest

import wording
from tests.test_app_ui import APP_PATH, burger  # noqa: F401
from tests.test_complete_results import _patty, _tell
from tests.test_insights_ui import _open_how, _open_insights, insights_scored  # noqa: F401


@pytest.fixture(autouse=True)
def _isolate_cwd(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)


def _app():
    return AppTest.from_file(APP_PATH, default_timeout=300)


def _tab(project):
    at = _app()
    at.session_state["_loaded_project"] = project
    return _open_insights(at)


def test_the_foot_names_the_open_round_and_both_feet_draw_in_one_run(insights_scored):
    insights_scored.set_pending_batch([{"Pea protein": 12.0, "Methylcellulose": 1.0}])
    at = _open_insights(_app())
    assert not at.exception
    keys = [b.key for b in at.button]
    assert "insights_foot" in keys and "foot_batch" in keys          # review M13
    foot = next(b for b in at.tabs[3].button if b.key == "insights_foot")
    assert foot.label == wording.back_to_batch_label(4, 1)


def test_nothing_scored_still_offers_the_foot(tmp_path):
    opt = _patty(rows=0)
    _tell(opt, 1, 1, drop=("Cook loss",))
    opt.set_aside_formulation(1)
    at = _tab(opt.project_name)
    assert not at.exception
    assert [c.value for c in at.tabs[3].caption] == [wording.INSIGHTS_NONE]
    assert "insights_foot" in [b.key for b in at.tabs[3].button]
