# Food Optimizer

[**Download Food Optimizer 0.7.1 for Mac**](https://github.com/thomasat/food_opt/releases/download/v0.7.1/FoodOptimizer-0.7.1.dmg)

Food Optimizer helps food scientists explore ingredient compositions and process settings. Define what can vary, choose measurements and targets, then generate formulations to prepare and evaluate. The app uses recorded results to suggest what to try next.

## Install on your Mac

Requires **Apple silicon (M1 or newer)** and **macOS 13 Ventura or newer**. Check your Mac in **Apple menu → About This Mac**.

1. Download and open **FoodOptimizer-0.7.1.dmg**.
2. Drag **Food Optimizer** onto **Applications** in the window that opens.
3. Open **Applications**, then double-click **Food Optimizer**.
4. Keep the app open and stay connected to the internet while setup completes. The window shows progress.

First-time setup downloads about **1 GB** and needs about **6 GB of free space**. Setup time depends on your connection and computer. You can use the app offline afterward; a future update may need another download.

**Updating?** Quit Food Optimizer before replacing it in Applications. Your saved projects are kept.

## Start with the example

Choose **Open the yogurt pilot**. It opens **Yogurt pilot - Round 6 to score** on **4 · Review**: a real yogurt-style pilot with 15 formulations scored and Round 6 made, waiting for its results. The pilot's data is public.

1. Go to **2 · Make a round** and open **Or upload results from a file**.
2. Choose **Browse files** and open **Round 6 results (pilot).xlsx**: it is already in your **FoodOptimizer** folder. Or choose **Download the pilot's Round 6 results**, then upload that file.
3. Review the imported values and choose **Save uploaded results**.
4. Open **4 · Review**. After Round 6, it says **Stop: score within noise of 100**.
5. Choose **See what happened next** to open **Yogurt pilot - all 10 Rounds**: the pilot's Rounds 7–10.

To try it again, choose **Manage project → Start this sample over**. Your version is kept in **Saved copies**.

To see how a project is set up, choose **Open the burger example**. It opens **Example setup - plant-based burger** on **1 · Set up**. Its values are illustrative and it has no results; it shows calculations (Hydration water as 2.2 × the textured proteins, Remaining water as the rest), pre-mixes, a process setting, records and a formulation total.

For your own project, generate formulations in **Make a round** and download the workbook. Use **Round overview** for the plan, **Preparation** for pre-mixes, and **Results** for measurements. Upload the completed workbook, review the imported values, and save the results. Editing the workbook alone does not update the app. You can also enter measurements directly in the app.

The included **Start Here.txt** explains installation and the first-round workflow. In Numbers, select the sheet tabs if workbook links do not open.

## Review results

**Results** opens on **Best so far**, based on your current measurements and targets. Use **View formulation** to inspect any other recorded formulation, including unscored entries. Its measurements, ingredient amounts and **% of formulation** appear together. Percentages exclude process settings and are unavailable for ingredient amounts in different units. The **All formulations** table below compares results in the app; **Show amounts** adds ingredient amounts to that table.

## How the Review works

The full method behind **4 · Review**; the app's **Definitions** keeps one line per term.

- **Models.** One Gaussian process for the Overall score and one per measurement, each fitted to the formulations that have that value (kernel from your model settings; noise fitted). Suggestions use the Overall-score model only; it is fitted to the scores, not built from the measurement models, so the two can differ slightly.
- **95 % range.** Mean ± 1.96 × predictive SD × a calibration factor, cut at the measurement's Range (0–100 for the Overall score). The calibration factor is the root-mean-square of standardised errors in 5-fold cross-validation, folds grouped by Round, at least 1; the Overall score's range for the predicted best and new formulations is widened again by a second factor, the root-mean-square of earlier Rounds' standardised Round-ahead errors (at least 1). The range covers one new measured value; on simulated projects it covered about 91 % of measurements and 90 % of Overall scores.
- **Round-ahead check (the Forecast check).** Each Round is forecast by models fitted only to the formulations recorded before it, scored with today's targets, Share of score and model settings. The forecasts are rebuilt today, not saved from when each Round was made. Misses are counted against a binomial at 5 % per value: more than expected below 1 in 10, many more below 1 in 100.
- **Pairs and verdicts.** Within a Round, a pair is ordered right when its forecasts put it in the measured order. Pairs measured equal, forecast less than one tasting noise apart, or both forecast at the top are not called. Predicts: called pairs right more often than chance (within-Round permutation test, Holm-adjusted across the results checked, p < 0.05, at least 5 pairs). Partly: right in more than half, p < 0.20 unadjusted. Not yet: otherwise, under 5 pairs called, or over half the pairs measured equal.
- **No forecast.** A 95 % range covering the whole Range (each end within 1 % of its width of the Range's end) forecasts nothing; it is counted neither in nor out.
- **Tasting noise.** The pooled SD of Repeats of the Overall score once it has 3 degrees of freedom (at least 0.05), from Repeats; until then the project's Tasting noise setting (Set up › Preparation and records; 1 point unless changed), assumed. A measurement's noise is its own Repeat SD, else derived from the Overall score's by its Range and Share of score. A pair is too close to call within one noise; Rising and the stop's tie use a 1-in-20 threshold on it (√2 × noise × a one-sided quantile for the top of the Round's results).
- **Stop.** Best so far is within the tie threshold of the top score (the rise one new result needs to be told apart from the noise), so no higher score could be told apart.
- **Limits.** No term p-values or coefficient table: formulations were chosen Round by Round, so term tests would not keep their error rates. Predictions assume your ingredients, suppliers and equipment behave as before. Best so far is one noisy maximum and tends to read high. Changing model settings re-scores every earlier Round. A noise threshold from 3 degrees of freedom has a 95 % interval of about 0.57 to 3.7 times the estimate. The first 5 scored formulations come from a space-filling design, not the model.

## Ingredients, calculations and limits

Use **Lowest** and **Highest** to define allowed amounts or settings. Enter the same value in both cells to keep a value fixed. A **Calculation** calculates an ingredient amount—for example, `= 0.2 * (Tapioca starch + Xanthan gum)`. Choose **Edit calculation → Fill to total** for the ingredient that supplies the remaining amount. Existing `= rest` calculations remain supported. **Edit calculation** provides ingredient suggestions, insertion buttons and validation in a larger panel. Choose **Use calculation**, then **Save changes** in the table. **Supported calculations** lists the accepted syntax. Examples explain the arithmetic; select relationships and values for your own protocol.

The **Preparation** column offers **Single ingredient**, **Pre-mix: keep proportions fixed**, and **Blend: vary each ingredient**. A pre-mix keeps the same ingredient percentages while the amount used can change. For a blend, set Lowest and Highest for each ingredient; the app chooses each amount separately. Both the proportions and the combined amount can change between formulations.

Ingredient limits can restrict combined amounts or finished-product properties. An empty minimum or maximum imposes no restriction on that side. Missing property values are unknown, rather than zero; complete the values before generating formulations with a limit on that property.

## Optional records

In **Preparation and records**, choose the records you need:

- **Vendor and SKU:** Enter values for each ingredient in the Ingredients table.
- **Lot:** Enter a lot number for each ingredient in **Make a round → Ingredient lot numbers** or in the workbook.
- **Custom fields:** Choose **Add your own recording field**, name the field, and choose each formulation or each ingredient in the round. Enter values in the app or workbook.

These records do not affect scores, and hiding a field preserves its values. The separate **Record changes from the planned amounts or settings** option adds workbook cells for what you actually used in each formulation. Imported corrections are used when the app learns from the results.

## Process studies

For a fermentation study, use numeric process settings such as temperature and time. A project containing only process settings does not need a batch size. Use separate measurements for time points, such as pH at 6 h and pH at 24 h. Named categories such as strain are not varied by this version.

## Saved projects and help

Projects are saved locally. Use **Save a copy of this project** before major changes. **Saved copies** shows recent copies and keeps older ones available separately.

[Report a problem](https://github.com/thomasat/food_opt/issues), describing what happened and what you expected. Do not include confidential project data in a public issue.

## Run from source

```sh
pip install -r requirements.txt
streamlit run app.py
```
