"""One-glance Review (owner, 2026-10-14): the README describes the v3
Review in at most 25 lines. (The forecast-check and definitions tests of
the old Insights tab went with that code.)"""
import os


README = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "README.md")


def test_the_readme_says_the_v3_review_in_25_lines():
    """v3: four tabs; the Review shows best so far, best vs targets and the
    amounts in the top formulations; single tastings; no stop rule and no
    forecasts on screen."""
    text = open(README, encoding="utf-8").read()
    assert "## How the Review works" in text
    section = text.split("## How the Review works", 1)[1].split("\n## ", 1)[0]
    lines = [line for line in section.splitlines() if line.strip()]
    assert 3 <= len(lines) <= 25
    for term in ("four tabs", "Best so far", "Best vs targets",
                 "Amounts in the 3 best formulations", "single tasting", "no stop rule",
                 "not shown on screen", "Gaussian process"):
        assert term in section, term
    for gone in ("Holm", "calibration factor", "No forecast", "Tasting noise", "**Stop.**",
                 "95 % range", "Round-ahead"):
        assert gone not in section, gone


