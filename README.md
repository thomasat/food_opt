Food Optimizer helps food product developers find better formulations in fewer rounds of lab work. You type your ingredients and your measurements into two tables; it suggests the next round of formulations to make, prints the sheets for the bench, and learns from what you measured. No knowledge of statistics or machine learning is needed.

Not every ingredient has to be searched for. Give a row a **Rule** — `= batch size − Water − Salt`, `= 1.5 % of batch size`, or `= rest` for the one row that takes whatever is left of the batch size — and the app works that amount out from the other rows every time. A limit over several ingredients can say **Exactly** as well as At least, At most and Between, in grams or as a **% of default batch size**.

An ingredient can also be a **pre-mix**. Choose **Made as** on its row: make one pre-mix for the round and portion it into each formulation, or weigh its parts separately into each formulation. Open the parts underneath the grid to edit them. The Excel workbook puts preparation pages first for portioned pre-mixes, groups separately weighed parts on each formulation page, and totals the ingredients needed for the round. Lot numbers and actual amounts entered in the shaded cells come back with the results.

The sample burger shows both ways: Dry blend made once, two oils weighed separately in Fat phase, Seasoning blend fixed at 2.5 g, and Water taking the rest of each 100 g formulation. Four rows open into nine parts.

**Desktop app (macOS):** download the latest `.dmg` from the
[releases page](https://github.com/thomasat/food_opt/releases), drag the app to
your Applications folder, and open it. The first time it opens it downloads
about 1 GB. This usually takes under a minute; on a slow network, a few
minutes. After that it works fully offline.

**Run it from source:**

pip install -r requirements.txt

streamlit run app.py
