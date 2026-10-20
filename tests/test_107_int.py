"""Insights for a first-time user, integration: the guards that still apply
(helped lines are st.markdown, never st.write; two old help constants are
gone). The status-line, Explored-fold and Try-one tests went with the old
Insights tab."""
import ui_insights
import wording


def test_lines_with_help_are_never_st_write():
    """R-I6: st.write takes no help=; every helped line is st.markdown."""
    import inspect
    src = inspect.getsource(ui_insights)
    assert "st.write(" not in src


# ------------------------------------------------------------------ #
#  One status logic; the ledger's row-order ruling
# ------------------------------------------------------------------ #


# ------------------------------------------------------------------ #
#  One model with no Round open (W3's note; R-I16)
# ------------------------------------------------------------------ #


# ------------------------------------------------------------------ #
#  The Explored fold's tooltips (W3's note 4)
# ------------------------------------------------------------------ #

_DESIGN = "design space"
_POLYTOPE = "get_polytope_samples"
_LINF = "L∞"
_SD = "posterior sd"
_NEEDS = {"basis_explored": (_DESIGN, _POLYTOPE, _LINF, _SD),
          "strips": (_DESIGN, _POLYTOPE, _LINF),
          "gaps": (_DESIGN, _POLYTOPE, _LINF),
          "map": (_DESIGN, _POLYTOPE, _SD),
          "limit_hits": (_DESIGN,)}


# ------------------------------------------------------------------ #
#  Fold renames (§2; T0 report §5) and the pointer
# ------------------------------------------------------------------ #


# ------------------------------------------------------------------ #
#  T0′ wording
# ------------------------------------------------------------------ #


# ------------------------------------------------------------------ #
#  Cleanup
# ------------------------------------------------------------------ #


def test_the_old_help_constants_are_gone():
    for gone in ("MODEL_SD_HELP", "NEAREST_TESTED_HELP"):
        assert not hasattr(wording, gone)
