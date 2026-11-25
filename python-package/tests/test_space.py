import pytest

from egbo._space import (Variable, active_names, bounds_tensor, decode, encode,
                         fixed_features, ingredient_total, search_bounds,
                         snap_settings)


def test_encode_missing_ingredient_is_zero():
    v = [Variable("Oil", 0, 5, absent=0.0), Variable("Flour", 1, 9, absent=0.0)]
    assert encode(v, {"Flour": 4}) == [0.0, 4.0]


def test_encode_missing_setting_is_baseline():
    v = [Variable("Temp", 100, 120, kind="setting", step=5, absent=110.0)]
    assert encode(v, {}) == [110.0]


def test_encode_missing_with_no_absent_is_zero():
    assert encode([Variable("Temp", 100, 120, kind="setting")], {}) == [0.0]


def test_decode_roundtrip():
    v = [Variable("Oil", 0, 5, absent=0.0), Variable("Flour", 1, 9, absent=0.0)]
    r = {"Oil": 2.5, "Flour": 4.0}
    assert decode(v, encode(v, r)) == r


def test_fixed_column_widened_to_history_span():
    v = [Variable("Salt", 3, 3, absent=0.0)]
    spans = search_bounds(v, [{"Salt": 3}] * 3, [[1.0], [3.0], [5.0]])
    assert spans == [(1.0, 5.0)]
    assert fixed_features(v, spans) == {0: 0.5}


def test_fixed_column_no_history_gets_unit_span():
    v = [Variable("Salt", 3, 3, absent=0.0)]
    spans = search_bounds(v, [], [])
    assert spans == [(3.0, 4.0)]
    assert fixed_features(v, spans) == {0: 0.0}


def test_mid_run_ingredient_span_includes_zero():
    v = [Variable("Gum", 2, 5, absent=0.0)]
    spans = search_bounds(v, [{"Gum": 3}, {}], [[3.0], [0.0]])
    assert spans == [(0.0, 5.0)]


def test_setting_without_absent_not_widened():
    v = [Variable("Temp", 100, 120, kind="setting", step=5)]
    assert search_bounds(v, [{}], [[0.0]]) == [(100.0, 120.0)]


def test_snap_settings_to_step():
    v = [Variable("Temp", 100, 120, kind="setting", step=5),
         Variable("Oil", 0, 5, absent=0.0)]
    out = snap_settings(v, {"Temp": 108.7, "Oil": 1.234})
    assert out == {"Temp": 110.0, "Oil": 1.234}
    v2 = [Variable("Temp", 100, 118, kind="setting", step=5)]
    assert snap_settings(v2, {"Temp": 117.9})["Temp"] == 118.0  # 120 clamped
    assert snap_settings(v2, {"Temp": 99.0})["Temp"] == 100.0


def test_bounds_tensor_shape_and_values():
    t = bounds_tensor([(0.0, 5.0), (1.0, 2.0)])
    assert t.shape == (2, 2) and str(t.dtype) == "torch.float64"
    assert t.tolist() == [[0.0, 1.0], [5.0, 2.0]]


def test_active_names_and_total():
    v = [Variable("A", 1, 1), Variable("B", 0, 3),
         Variable("T", 1, 2, kind="setting")]
    assert active_names(v) == ["B", "T"]
    assert ingredient_total(v, {"A": 1, "B": 2, "T": 150}) == 3.0
