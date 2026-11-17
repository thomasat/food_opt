"""The owner's rule for the golden tests (2026-10-24): on the Mac a golden
result is bit-identical; on Windows (other BLAS, other float rounding) the
numbers may differ by at most 0.1 % relative. tests.golden.same is the one
comparison every golden test uses."""
import pytest

from tests import golden

WANT = {"Y": [1.0, 0.0, 77.5], "ask": [{"Pea": 4.25, "Oil": 2.5, "note": "x"}],
        "gp_X": [[0.5, 0.25]], "gp_Y": [[1.5]]}


def _scaled(rel, key="Y"):
    got = {"Y": list(WANT["Y"]), "ask": [dict(WANT["ask"][0])],
           "gp_X": [list(WANT["gp_X"][0])], "gp_Y": [list(WANT["gp_Y"][0])]}
    if key == "Y":
        got["Y"][2] *= 1 + rel
    elif key == "ask":
        got["ask"][0]["Pea"] *= 1 + rel
    else:
        got[key][0][0] *= 1 + rel
    return got


@pytest.mark.parametrize("key", ["Y", "ask", "gp_X", "gp_Y"])
def test_off_windows_a_tiny_difference_fails(monkeypatch, key):
    monkeypatch.setattr(golden.sys, "platform", "darwin")
    with pytest.raises(AssertionError):
        golden.same(_scaled(1e-12, key), WANT)


@pytest.mark.parametrize("key", ["Y", "ask", "gp_X", "gp_Y"])
def test_on_windows_a_difference_within_a_tenth_of_a_percent_passes(monkeypatch, key):
    monkeypatch.setattr(golden.sys, "platform", "win32")
    golden.same(_scaled(1e-5, key), WANT)


@pytest.mark.parametrize("key", ["Y", "ask", "gp_X", "gp_Y"])
def test_on_windows_a_one_percent_difference_fails(monkeypatch, key):
    monkeypatch.setattr(golden.sys, "platform", "win32")
    with pytest.raises(AssertionError):
        golden.same(_scaled(1e-2, key), WANT)


@pytest.mark.parametrize("platform", ["darwin", "win32"])
def test_identical_results_pass_everywhere(monkeypatch, platform):
    monkeypatch.setattr(golden.sys, "platform", platform)
    golden.same(_scaled(0.0), WANT)


def test_on_windows_a_zero_allows_only_a_tiny_absolute_difference(monkeypatch):
    monkeypatch.setattr(golden.sys, "platform", "win32")
    got = _scaled(0.0)
    got["Y"][1] = 1e-15
    golden.same(got, WANT)
    got["Y"][1] = 1e-3
    with pytest.raises(AssertionError):
        golden.same(got, WANT)


@pytest.mark.parametrize("platform", ["darwin", "win32"])
def test_a_different_string_or_shape_fails_everywhere(monkeypatch, platform):
    monkeypatch.setattr(golden.sys, "platform", platform)
    got = _scaled(0.0)
    got["ask"][0]["note"] = "y"
    with pytest.raises(AssertionError):
        golden.same(got, WANT)
    got = _scaled(0.0)
    got["Y"].append(1.0)
    with pytest.raises(AssertionError):
        golden.same(got, WANT)
