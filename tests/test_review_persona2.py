"""Persona fixes 2 (2026-10-02): a Round of entered formulations says its
notes are the source. The rest of this pass (Forecast check, widen advice,
score recipe, Plateau) belonged to the old Insights engine, removed in
0.8.0."""
import ui_batch
import wording


def test_round_caption_respects_notes():
    own = [{"note": wording.OWN_FORMULATION_NOTE}, {"note": wording.OWN_FORMULATION_NOTE}]
    assert ui_batch.own_round_caption(own) == wording.ONLY_OWN_FORMULATIONS_CAPTION
    noted = [{"note": "Pilot batch 6.1 · EGBO"}, {"note": wording.OWN_FORMULATION_NOTE}]
    caption = ui_batch.own_round_caption(noted)
    assert caption == wording.ENTERED_ROUND_CAPTION
    assert "only your own" not in caption and "notes" in caption
    # Generic: any note.
    assert ui_batch.own_round_caption([{"note": "Bench control B"}]) == \
        wording.ENTERED_ROUND_CAPTION
