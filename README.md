Food Optimizer helps food product developers find better formulations in fewer rounds of lab work. You type your ingredients and your measurements into two tables; it suggests the next round of formulations to make, prints the sheets for the bench, and learns from what you measured. No knowledge of statistics or machine learning is needed.

Not every ingredient has to be searched for. Give a row a **Rule** — `= batch size − Water − Salt`, `= 1.5 % of batch size`, or `= rest` for the one row that takes whatever is left of the batch size — and the app works that amount out from the other rows every time. A limit over several ingredients can say **Exactly** as well as At least, At most and Between, in grams or as a **% of default batch size**.

An ingredient can also be a **pre-mix**. Choose **Made as** on its row: `bought in` for an ordinary ingredient, `portioned from one pre-mix` to make one lot for the round and portion it into each formulation, or `weighed into each formulation` to weigh its parts separately every time. Open the parts underneath the grid to edit them. An ingredient the project already has can be a part too. The Excel workbook puts preparation pages first for portioned pre-mixes, groups a pre-mix's separately weighed parts on each formulation page, says what to **make for this round** and what to **have on hand**, and prints the project's **Method** under the round's title. Lot numbers and actual amounts entered in the boxed cells come back with the results.

The sample burger shows both ways: Dry blend made once, two oils weighed separately in Fat phase, Seasoning blend fixed at 2.2 g, and Water taking the rest of each 100 g formulation. Six rows open into ten parts, with one process setting (Mixing time after fat), one finished-product limit (Fat per 100 g, at most 16) and three measurements: Firmness, Juiciness and Cook loss.

A **formulation** is one thing you make and measure: something you weigh out, a process run, a fermentation. A project whose rows are all process settings weighs nothing out, and the app asks it for no batch size and prints its settings in their own units.

**Desktop app (macOS):** download the latest `.dmg` from the
[releases page](https://github.com/thomasat/food_opt/releases), drag the app to
your Applications folder, and open it. The first time it opens it downloads
about 1 GB. This usually takes under a minute; on a slow network, a few
minutes. After that it works fully offline.

**Run it from source:**

pip install -r requirements.txt

streamlit run app.py


Optional recording and returning the sheets

In Set up → More settings, “Also record” turns Vendor, SKU, Lot and Actual
amounts on or off. New projects start with them off; the sample records Lot
and Actual amounts. Turning a field off keeps values already saved.
After filling in the workbook, upload it in 2 · Make a round → Save results
→ Or upload results from a file, check the preview, then save. Editing Excel
alone does not update the app. Preparation amounts and other filled-in boxes
are preserved as Bench records in the All formulations export; preparation
records do not change a pre-mix's percentages for future rounds. Formulation
Actual amounts are used as the amounts made. The Set-up and All formulations
sheets are records, rather than forms to fill in and send back.
Saved copies shows the three newest copies and puts the rest under Older
copies. Its confirmed cleanup keeps the newest three and the last seven days.
For a process or fermentation study, use numeric process settings. Record
separate measurements for different time points, such as pH at 6 h and pH at
24 h. Named categories such as strain or vessel type are not varied by this
version. The sample targets and method are illustrative, not measured results.
