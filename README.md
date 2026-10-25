# Food Optimizer

[**Download Food Optimizer 0.8.0 for Mac**](https://github.com/thomasat/food_opt/releases/download/v0.8.0/FoodOptimizer-0.8.0.dmg)

Food Optimizer helps food scientists explore ingredient compositions and process settings. Define what can vary, choose measurements and targets, then generate formulations to prepare and evaluate. The app uses recorded results to suggest what to try next.

## Install on your Mac

Requires **Apple silicon (M1 or newer)** and **macOS 13 Ventura or newer**. Check your Mac in **Apple menu → About This Mac**.

1. Download and open **FoodOptimizer-0.8.0.dmg**.
2. Drag **Food Optimizer** onto **Applications** in the window that opens.
3. Open **Applications**, then double-click **Food Optimizer**.
4. Keep the app open and stay connected to the internet while setup completes. The window shows progress.

First-time setup downloads about **1 GB** and needs about **6 GB of free space**. Setup time depends on your connection and computer. You can use the app offline afterward; a future update may need another download.

**Updating?** Quit Food Optimizer before replacing it in Applications. Your saved projects are kept.

## Start with the example

Choose **Open the burger example** to see how a project is set up. It opens **Example setup - plant-based burger** on **1 · Set up**. Its values are illustrative and it has no results; it shows calculations (Hydration water as 2.2 × the textured proteins, Remaining water as the rest), pre-mixes, a process setting, records and a formulation total.

To see real results, choose **Try the yogurt pilot**. It opens **Yogurt pilot - Round 6 to score** on **4 · Review**: a real yogurt-style pilot with 15 formulations scored and Round 6 made, waiting for its results. The pilot's data is public.

1. Go to **2 · Make a round** and open **Or upload results from a file**.
2. Choose **Browse files** and open **Round 6 results (pilot).xlsx**: it is already in your **FoodOptimizer** folder. Or choose **Download the pilot's Round 6 results**, then upload that file.
3. Review the imported values and choose **Save uploaded results**.
4. Open **3 · Results** for Round 6's ratings against their targets, then **4 · Review** for the best so far.
5. Under **What next?**, choose **Start Round 7** to have the app suggest Round 7, or **See the pilot's Rounds 7–10** to open **Yogurt pilot - all 10 Rounds**: what the pilot made next.

To try it again, choose **Manage project → Start this sample over**. Your version is kept in **Saved copies**.

For your own project, generate formulations in **Make a round** and download the workbook. Use **Round overview** for the plan and **Results** for measurements; a Round with pre-mixes also has a **Preparation** sheet. Upload the completed workbook, review the imported values, and save the results. Editing the workbook alone does not update the app. You can also enter measurements directly in the app.

The included **Start Here.txt** explains installation and the first-round workflow. In Numbers, select the sheet tabs if workbook links do not open.

## Review results

**Results** opens on the last Round's results: each formulation's **Overall score** and its ratings, each measurement's target in the header. Off-target ratings are orange above the target and purple below it; a new best is blue. **How the score is worked out** shows one formulation's points per attribute: each point off target costs a fixed part of its share (1/15 when ratings run 0–15). **View formulation** shows any recorded formulation's amounts and **% of formulation**; **All formulations** compares them all.

## How the Review works

The app has four tabs: **Set up**, **Make a round**, **Results** and **Review**. **4 · Review** shows measurements, not forecasts:

- **Best so far.** Every Overall score by Round, the best so far and each Round's average. 100 means every attribute on target.
- **Best vs targets.** The best formulation's rating for each attribute against its target.
- **Amounts in the 3 best formulations.** The ingredient amounts of the 3 highest-scoring formulations so far (more when the 3rd place is tied), against the allowed amounts.

Each score is a single tasting: each formulation is made once and rated once, so small differences may not be real. There is no stop rule; you decide when to stop. **Make a round** suggests the next formulations: a Gaussian process model fitted to the Overall scores picks the best bets to improve on the best so far, weighing how good they look and how unsure the model is. Its predictions are used to choose, not shown on screen.

**Limits.** Predictions assume your ingredients, suppliers and equipment behave as before. Best so far is one noisy maximum and tends to read high. The first 5 scored formulations come from a space-filling design, not the model.

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

Email sohum.patnaik@fsi.org, describing what happened and what you expected.

## Run from source

```sh
pip install -r requirements.txt
streamlit run app.py
```
