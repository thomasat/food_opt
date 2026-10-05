"""What have we tried? — every gap counted (owner, 2026-10-05: "Why do we
say '15 tested. X between a and b g untested.'? What
about the other ingredients?").

- The answer counts the rows with an untested gap (15 % of the span or
  more, within the reachable part), then names the largest: "Gaps in 6 of
  7 ingredients; largest: Salt 1–5 g"; with none, "No untested
  stretch in any ingredient". At most 14 words.
- Its "?" lists every gap, one line per row, low to high, then the caveat
  that one row at a time leaves combinations mostly untested; over about
  120 words, each row's largest and "+N more".
- The search path shades every counted gap, not only the named one.

Synthetic strips and the synthetic demo projects only; no pilot data."""
from types import SimpleNamespace as NS

import pytest

import insights
import ui_insights as U
import wording
from tests import demo_projects as D
from tests.test_food_bo import _BANNED


def _strip(name, values, setting=False, fill=False, unit="g", options=None):
    ticks = [insights.StripTick(v, k + 1, 1, 50.0, "tested") for k, v in enumerate(values)]
    return insights.Strip(name, "" if options else unit, 0.0, 10.0, (0.0, 10.0), ticks,
                          fill, setting, options)


def _opt(n=15):
    return NS(complete_count=lambda: n)


def _words(text):
    return len(text.replace("*", "").split())


# Covered: no stretch of 1.5 or more anywhere on 0–10.
_COVERED = [0.0, 1.4, 2.8, 4.2, 5.6, 7.0, 8.4, 9.8]


def test_the_answer_counts_every_row_with_a_gap_then_names_the_largest():
    strips = [_strip("Oil", [0.0, 2.0, 4.0, 6.0, 8.0, 10.0]),   # gaps 2.0 each
              _strip("Salt", [0.0, 1.0, 5.0, 6.4, 7.8, 9.2]),     # 1–5 is the largest
              _strip("Sugar", _COVERED)]
    line = U.tried_answer(_opt(), strips)
    assert line == "**15 tested.** Gaps in 2 of 3 ingredients; largest: Salt 1–5 g."
    assert _words(line) <= 14


def test_every_row_gapped_reads_all():
    strips = [_strip(n, [0.0, 5.0, 10.0]) for n in ("Oil", "Salt", "Sugar")]
    assert U.tried_answer(_opt(30), strips).startswith(
        "**30 tested.** Gaps in all 3 ingredients; largest: ")


def test_no_gap_says_so():
    strips = [_strip("Oil", _COVERED), _strip("Salt", _COVERED)]
    assert U.tried_answer(_opt(12), strips) == (
        "**12 tested.** No untested stretch in any ingredient.")
    lines = U.tried_gap_lines(_opt(12), strips)
    assert lines == [wording.strips_no_gap(), wording.tried_caveat("ingredient")]


def test_the_fill_row_and_choice_rows_are_not_counted():
    strips = [_strip("Oil", [0.0, 5.0, 10.0]), _strip("Salt", _COVERED),
              _strip("Water", [0.0], fill=True),
              _strip("Flour", [], options={"A": 2, "B": 0})]
    assert "Gaps in 1 of 2 ingredients" in U.tried_answer(_opt(), strips)


def test_settings_alone_say_settings():
    strips = [_strip("Oven", [0.0, 5.0, 10.0], setting=True, unit="°C"),
              _strip("Time", _COVERED, setting=True, unit="min")]
    line = U.tried_answer(_opt(), strips)
    assert line == "**15 tested.** Gaps in 1 of 2 settings; largest: Oven 0–5 °C."
    assert U.tried_gap_lines(_opt(), strips)[-1] == wording.tried_caveat("setting")
    assert "setting" in U.tried_answer(_opt(), [_strip("Oven", _COVERED, setting=True)])


def test_the_help_lists_every_gap_one_line_per_row_and_the_caveat():
    strips = [_strip("Oil", [0.0, 2.0, 4.0, 6.0, 8.0, 10.0]),
              _strip("Salt", [0.0, 1.0, 5.0, 6.4, 7.8, 9.2]),
              _strip("Sugar", _COVERED)]
    listed, caveat = U.tried_gap_lines(_opt(), strips)
    assert listed.split("\n") == [
        "- Oil: 0.0–2.0, 2.0–4.0, 4.0–6.0, 6.0–8.0, 8.0–10.0 g",
        "- Salt: 1–5 g",
    ]
    assert caveat == "One ingredient at a time: combinations of amounts are mostly untested."
    # Every gap the rule finds is listed.
    assert listed.count("–") == len(insights.all_stretches(strips))


def test_an_overlong_list_keeps_each_rows_largest_and_counts_the_rest():
    many = [_strip(f"Ingredient number {k}", [0.0, 2.0, 4.0, 6.0, 8.0, 10.0])
            for k in range(12)]
    listed, caveat = U.tried_gap_lines(_opt(), many)
    assert "+4 more" in listed
    assert _words(listed) + _words(caveat) <= wording.TOOLTIP_WORDS
    many = [_strip(f"Ingredient number {k}", [0.0, 2.0, 4.0, 6.0, 8.0, 10.0])
            for k in range(40)]
    listed, caveat = U.tried_gap_lines(_opt(), many)
    assert listed.split("\n")[-1].startswith("- +") and listed.endswith("more ingredients")
    assert _words(listed) + _words(caveat) <= wording.TOOLTIP_WORDS


def test_the_wording():
    assert wording.tried_gaps(7, 7, "ingredient", "Salt 0.6–1.0 g") == (
        "Gaps in all 7 ingredients; largest: Salt 0.6–1.0 g")
    assert wording.tried_gaps(2, 2, "setting", "Oven 160–175 °C").startswith(
        "Gaps in both settings")
    assert wording.tried_gaps(1, 1, "ingredient", "Salt 1–2 g") == "Largest gap: Salt 1–2 g"
    assert wording.tried_gaps(3, 5, "mixed", "Salt 1–2 g") == (
        "Gaps in 3 of 5 ingredients/settings; largest: Salt 1–2 g")
    assert wording.tried_no_gap("mixed") == "No untested stretch in any ingredient or setting"
    texts = [wording.tried_gaps(3, 5, k, "Salt 1–2 g") for k in ("ingredient", "setting",
                                                                    "mixed")]
    texts += [wording.tried_no_gap(k) for k in ("ingredient", "setting", "mixed")]
    texts += [wording.tried_caveat(k) for k in ("ingredient", "setting", "mixed")]
    texts += [wording.gap_row_line("Salt", "1–2 g", 2), wording.gap_rows_more(3, "mixed"),
              wording.story_path_help(), wording.story_path_help(True)]
    for t in texts:
        assert not any(p.search(t) for p in _BANNED), t
    assert _words(wording.story_path_help()) <= wording.TOOLTIP_WORDS


def _shaded(chart):
    """The shaded gaps of every panel: [(panel title, lo, hi)]."""
    spec = chart.to_dict()
    out = []
    for panel in spec["concat"]:
        for layer in panel["layer"]:
            mark = layer["mark"]["type"] if isinstance(layer["mark"], dict) else layer["mark"]
            if mark == "rect":
                rows = spec["datasets"][layer["data"]["name"]]
                out += [(panel["title"]["text"], r["lo"], r["hi"]) for r in rows]
    return out


@pytest.mark.parametrize("name", [D.LINE_A, D.PROCESS_ONLY])
def test_the_search_path_shades_every_counted_gap(name):
    import bo_views
    opt = D.load(name)
    strips = insights.tested_strips(opt, bo_views.feasible_pool(opt))
    gaps = insights.all_stretches(strips)
    axes, lines = insights.tried_axes(opt), insights.tried_lines(opt)
    shaded = _shaded(U.search_path_chart(opt, axes, lines, gaps))
    assert len(shaded) == len(gaps)
    # More than the one the answer names, on more than one panel.
    assert len({t for t, _, _ in shaded}) == len({g.name for g in gaps})
    rows, found = U._found_gaps(strips)
    assert len(found) == len({g.name for g in gaps})
    line = wording.for_project(opt, U.tried_answer(opt, strips))
    assert _words(line) <= 14, line
    if name == D.PROCESS_ONLY:
        assert "settings" in line and "ingredient" not in line


def test_choice_rows_alone_give_no_gap_line():
    strips = [_strip("Flour", [], options={"A": 2, "B": 1})]
    assert U.tried_answer(_opt(4), strips) == "**4 tested.**"
    assert U.tried_gap_lines(_opt(4), strips) == []


def test_a_long_name_keeps_the_count_within_14_words():
    strips = [_strip("Mixing time after the fat goes in", [0.0, 5.0, 10.0], setting=True,
                     unit="s"),
              _strip("Oil", _COVERED)]
    line = U.tried_answer(_opt(21), strips)
    assert line == "**21 tested.** Gaps in 1 of 2 ingredients/settings."
    assert wording.tried_gaps(1, 1, "ingredient") == "Gaps in the one ingredient"
