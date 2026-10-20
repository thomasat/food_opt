"""Review specifics: the runner-up line is short and says no noise. The rest
of this file (value misses, Predicts/Partly pairs, greyed gain) belonged to
the old Insights engine, removed in 0.8.0."""
import re

import wording


def _words(text):
    plain = re.sub(r":\w+\[(.*?)\]", r"\1", text).replace("**", "")
    return len([w for w in plain.split() if re.search(r"\w", w)])


def test_the_runner_up_line_is_short_and_says_no_noise():
    """Replaced the tie line that said what a rise needs (owner, v3 final
    cleanup: no tasting noise on screen)."""
    line = wording.runner_up_line(13, "93.3", "1.7")
    assert line == "Next best: Formulation 13, 93.3 (1.7 lower)."
    assert _words(line) <= 10
