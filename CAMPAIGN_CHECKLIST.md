# Campaign checklist: adaptive EGBO in the Formulation Assistant

For the team running a campaign: 10 rounds of 5 recipes, with expert checks
after rounds 3 and 6 (and 9).

## The one rule: the downloaded backup is the record

The app goes to sleep when nobody uses it, and when it wakes up **its saved
data is usually gone**. Expect this between most rounds.

- **Start of every session:** if the sidebar says the project has no
  experiments, **restore the latest backup** before doing anything else.
- **End of every step:** download a backup. The app reminds you after each
  change. Files are named by round and date, e.g. `mozz_R03_20261202-1415.json`.
- Keep **every** backup file, in a shared folder. Never overwrite or delete them.

Restoring checks the file first. A broken file is refused and changes nothing,
so try the next most recent one.

## Before round 1 (once)

**Dry run** (about 15 minutes, in a throwaway project, e.g. `dryrun`)
- [ ] Set up a couple of ingredients and attributes, then generate and score 2 rounds
      with made-up numbers.
- [ ] Download the backup, reload the app (or wait for it to sleep), and
      restore it. Check that both rounds, the round counter and any pending
      batch are back.

**Real project setup** (Setup tab)
- [ ] **Ingredients** (A) and **process parameters** (A2), with makeable ranges.
      Add process parameters now rather than mid-run if possible.
- [ ] **Objectives** (B): all attributes, with weight, scale (min–max) and goal
      (max / min / target value). Don't change these during the campaign. If an
      attribute has to be added later, every earlier recipe needs a score for it first.
- [ ] **BO hyperparameters** (F): click **Use campaign config (recommended)**.
      This setting locks once the first result is saved.
- [ ] **Campaign settings** (G):
  - [ ] Expert check every **3** rounds.
  - [ ] **Candidate list**: ingredients or process changes the expert may add, with
        plausible ranges.
  - [ ] **Reference recipe**: the same recipe, re-tasted every session.
- [ ] Download a backup.

## Every round

1. Restore the latest backup if the project shows no experiments.
2. **Generate 5 recipes** (batch size 5). Download the batch CSV **and** a backup.
3. Make and taste the 5 recipes, plus the **reference recipe**, in the same session.
4. Enter the scores (panel means) and **Save Results**.
   - Keep the per-panelist score sheets on your side.
5. Download a backup.

Note anything done differently from the app's recipe (a substitution, a
process deviation, a remade batch). Add a short note in the expert check or
send it to the study team.

## Expert check (after rounds 3, 6, 9)

The app shows **"Expert check due"** and won't generate the next round until the
check is recorded.

1. In **Adaptive EGBO**, click **Prepare expert report** and download it.
   It shows each attribute's gap, the best recipes, the ranges, and which
   variables are stuck at a limit.
2. The expert (FSA, reviewed by a food scientist) decides whether to:
   - **add** an ingredient or process parameter (a process parameter needs the
     setting used so far, its *baseline*);
   - **widen or narrow** a range;
   - **deactivate** a variable. Data is kept, and it can be reactivated later.
     Don't use permanent deletion.
   - or make **no change**.
3. Make the changes in the panel.
4. **Record the check**: the decision (write *no change* if nothing changed), the
   reasoning (which attributes and evidence), and who decided. This is
   required even when nothing changes.
5. Download a backup, then generate the next round.

## End of the campaign

- [ ] Open **Finalists** and **Compute finalists**. The top recipes are ranked by
      the model's predicted score, not by their best single tasting, which is
      often a lucky one.
- [ ] Re-taste the top 2–3 with more panelists before choosing the final recipe.
- [ ] Send the final backup and all per-panelist sheets to the study team.
