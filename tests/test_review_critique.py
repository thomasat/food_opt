"""Review critique (2026-10-04): "Show as lines" stays gone from the Review.

The rest of the critique pass (forecast check, stop signal, ordering test,
noise share) belonged to the old Insights engine, removed in 0.8.0."""
import ui_insights as U
import wording


def test_show_as_lines_is_gone():
    assert not hasattr(wording, "STORY_SHOW_LINES")
    assert not hasattr(U, "tried_chart") and not hasattr(U, "PATH_SHOW_LINES_KEY")
