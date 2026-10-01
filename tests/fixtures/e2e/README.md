# End-to-end campaign projects (2026-10-01), frozen

These are the project files the 2026-10-01 end-to-end campaign left behind, on build
`insights-080` @ 111e658 (`CLASS_VERSION` 17). The campaign's reports are in
`docs/superpowers/research/2026-10-01-e2e/`.

Each file was opened with `FoodOptimizer.import_json` and written back with `export_json`.
The result has the same JSON content as its source, and it is never regenerated. A replay
of `ask()` would not reproduce these runs: 03 changed acquisition mid-run, and both runs
made Repeats by hand.

Load the files with `tests/e2e_fixtures.py`:
- `load(name, last_round=None, recommended=False)`
- `state(name)`
- `truncate_state(saved, last_round)`

The session scratchpad named below is
`/private/tmp/claude-501/-Users-sohumpatnaik-Documents-GitHub-food-opt/440bf566-66ce-4395-9cc4-41d02bfae1a0/scratchpad/`.

| Fixture | Source (scratchpad) | Report | Contents |
|---|---|---|---|
| `02_patty_final.json` | `e2e-fs/project/Pea burger patty.pkl` | 02, food scientist | Pea burger patty, final state. 8 Rounds × 4 = 32 formulations. Redness was added before Round 4 (shares 34 / 29.75 / 21.25 / 15). F8 is Not scored. Round 3 was made at 500 g. F16 is a Repeat of F11 and F32 of F26. Best so far is F30 (82.20). Recommended model settings. |
| `02_patty_round3.json` | `e2e-fs/project/Pea burger patty_pre_edit.pkl` | 02 | The app's own "Before editing" copy, taken when Redness was added: the state after Round 3. 12 formulations, three measurements at 40 / 35 / 25 %. F6's Cook loss is filled in and F8 is Not scored. Best so far is F11 (83.01). The 02 Round 3 pins (A2, A4) use this file. |
| `02_patty_round7_open.json` | `e2e-fs/project/Pea burger patty_pre_restore.pkl` | 02 (I7) | The copy the app saved before "Open a saved copy": 24 formulations, with Round 7 open (4 suggestions, none recorded). Used for B3's restore warning. |
| `03_bar_clean.json` | `e2e-doe/cwd/bar_clean.pkl` | 03, DoE expert, clean panel | Snack bar, 8 Rounds × 4. Custom matern52; qLogNEI R1–4, qLogEI R5–6, qUCB R7–8 (the file is left at qLogNEI). Repeats F16/F12, F20/F13, F24/F18 and F28/F23. Best so far is F23 (91.69). |
| `03_bar_drift.json` | `e2e-doe/cwd/bar_drift.pkl` | 03, drift panel | The same bar, 8 Rounds × 4, Custom rbf, qLogNEI. Liking was +5 from Round 5 on. F20 is a Repeat of F14, made across the drift. |
| `03_bar_noisy.json` | `e2e-doe/cwd/bar_noisy.pkl` | 03, noisy panel | The same bar, 7 Rounds × 4, run under Recommended settings and left at Custom rbf / short / fixed_tiny. Load it with `recommended=True` for the A3 pins. F14 scored 91.45; its Repeats F20, F24 and F28 scored 67.30, 66.34 and 75.61. |

The noise and drift that produced the results are described in
`e2e-fs/truth.py`, `e2e-fs/sim.py` and `e2e-doe/truth.py`, and in each report's
"Truth function" section.
