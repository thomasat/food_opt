"""Fix round B (research/2026-10-01-insights-ftue/ledger.md: "Statistician
cold read"): the level-0 list allows the skill number only with its
meaning."""
import pytest

from tests.ftue_terms import statistic_names_in


@pytest.fixture(autouse=True)
def _isolate_cwd(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)


_SKILL = "(skill −0.47, where 0 matches that average and 1 is exact)"


# ------------------------------------------------------------------ #
#  The level-0 list allows the skill number only with its meaning
# ------------------------------------------------------------------ #

def test_the_skill_number_is_allowed_only_with_its_meaning():
    assert statistic_names_in(f"Score predictions … {_SKILL}, and …") == []
    assert statistic_names_in("skill −0.47") != []
    assert statistic_names_in("the skill is low") != []
