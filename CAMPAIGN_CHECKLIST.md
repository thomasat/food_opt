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
- [ ] **Ingredients** (A): upload a CSV with **every** ingredient of the starting
      recipe, including water, with columns `Name, Min, Max, Sodium, Cost`
      (sodium and cost per gram). Ranges are in grams and must be makeable.
- [ ] **Balance ingredient** (F): set **Tap water** as the balance, with the batch
      total (e.g. 1002 g) and a min/max amount of water. Water is then filled up
      to the batch total instead of being searched over.
- [ ] **Process parameters** (A2): cook temperature (78–92 °C) and cook time
      (6–12 min).
- [ ] **Ingredients the FSA does not want varied** (e.g. potassium sorbate,
      lecithin): in the *Adaptive EGBO* panel, **Deactivate** each one and **hold it
      at its current amount**. It stays in every recipe at that amount and can be
      reactivated at a check.
- [ ] **Property constraints** (C): sodium and cost **maximums** per batch.
- [ ] **Objectives** (B): every attribute, with weight, scale **0–15** (the app
      defaults to 0–10, so change it), and goal (max / min / target value).
      Don't change these during the campaign. If an attribute has to be added
      later, every earlier recipe needs a score for it first.
- [ ] **BO hyperparameters** (E): click **Use campaign config (recommended)**.
      This setting locks once the first result is saved.
- [ ] **Campaign settings** (F): expert check every **3** rounds. Optionally, a
      **candidate list** of ingredients or process changes the FSA may add (with
      plausible ranges), e.g. what's in stock or can arrive before the next round.
      Leave it blank to let the FSA suggest freely.
- [ ] **Starting recipe:** after the first session, enter the starting recipe and
      its scores with **Import Historical Experiments (CSV)** on the Optimization
      Loop tab: one row with a column for every ingredient and process parameter
      (not water) and every attribute. Earlier tastings of the same recipe family
      can go in the same file.
- [ ] The **dairy reference** (Arla, served twice per session) is not a
      recipe: keep its scores on the panel sheets. Don't enter it as a recipe in the app.
- [ ] Download a backup.

## Every round

1. Restore the latest backup if the project shows no experiments.
2. **Generate 5 recipes** (batch size 5). Download the batch CSV **and** a backup.
3. Make and taste the 5 recipes, plus the dairy reference (twice), in the same session.
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
2. The FSA decides whether to:
   - **add** an ingredient or process parameter, giving the **amount or setting
     used so far** (0 for a brand-new ingredient);
   - **widen or narrow** a range;
   - **deactivate** a variable and **hold it at** a chosen amount (it defaults to
     the amount in the best recipe; 0 drops an ingredient). Data is kept, and it can
     be reactivated later. Don't use permanent deletion.
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
