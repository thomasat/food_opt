"""Review polish (2026-10-02, demo prep): the header says what the Review
is about; an Off card names which numbers missed; each card's "?" opens
with its plain sentence, then one evidence line whose numbers are the
card's; a hidden gain is a deliberate line; the hero's two panels are two
full-width charts that line up; the method is one click away; the three
Make a round lines over the owner's limits are trimmed. Wording and
layout only."""
import re


import ui_insights as U
import wording
from tests.test_food_bo import _BANNED


def _words(text):
    plain = re.sub(r":\w+\[(.*?)\]", r"\1", text).replace("**", "")
    return len([w for w in plain.split() if re.search(r"\w", w)])


def test_hidden_gain_is_one_deliberate_line():
    for text in wording.HERO_HIDDEN.values():
        assert _words(text) <= 10, text
    assert wording.HERO_HIDDEN_ICON.startswith(":material/")


def test_new_strings_pass_the_vocabulary_guard():
    texts = [
             wording.card_values_missed(9, 36, 4), 
             wording.hero_axis(), wording.hero_axis(True), *wording.HERO_HIDDEN.values(),
             wording.tip_hero_hidden("x"), wording.TIP_HERO_TOP,
             wording.ev_rule("Misses")]
    for text in texts:
        assert not any(p.search(text) for p in _BANNED), text
        assert "spread" not in text.lower(), text
