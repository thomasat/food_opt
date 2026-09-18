# Food Optimizer

[**Download Food Optimizer for Mac**](https://github.com/thomasat/food_opt/releases)

Food Optimizer helps food scientists explore ingredient compositions and process settings. Define what can vary, choose measurements and targets, then generate formulations to prepare and evaluate. The app uses recorded results to suggest what to try next.

For Macs with Apple silicon (M1 or newer), running macOS 13 Ventura or newer. Open the DMG and drag Food Optimizer into Applications. First-time setup downloads about 1 GB and needs internet and about 6 GB of free space. Setup time depends on your connection and computer. The app works offline after setup.

## Start with the example

Choose **Try the sample project**. The 100 g burger example demonstrates pre-mixes with fixed ingredient percentages, oils whose amounts vary separately, water calculated from two protein ingredients, remaining water, mixing time, measurements and a finished-product limit. Its ingredient amounts, targets and property values are illustrative.

1. Review the tables in **Set up**. Open **Blend compositions** to review component ingredients, and **Preparation and records** for the shared preparation method.
2. Generate formulations in **Make a round** and download the workbook.
3. Use **Round overview** for the plan, **Preparation** for pre-mixes, and **Results** for measurements.
4. Upload the completed workbook, review the imported values, and save the results. Editing the workbook alone does not update the app. You can also enter measurements directly in the app.

The included **Start Here.txt** explains installation and the first-round workflow. In Numbers, select the sheet tabs if workbook links do not open.

## Review results

**Results** opens on **Best so far**, based on your current measurements and targets. Use **View formulation** to inspect any other recorded formulation, including unscored entries. Its measurements, ingredient amounts and **% of formulation** appear together. Percentages exclude process settings and are unavailable for ingredient amounts in different units. The **All formulations** table below compares results in the app; **Show amounts** adds ingredient amounts to that table.

## Ingredients, rules and limits

Use **Lowest** and **Highest** to define allowed amounts or settings. Enter the same value in both cells to keep a value fixed. A **Rule** calculates an ingredient amount—for example, `= 2.2 * (Textured pea protein + Textured soy protein)`. Use `= rest` for the ingredient that fills the remaining batch size. **Calculation help** explains these options.

The **Preparation** column offers **Single ingredient**, **Pre-mix: keep proportions fixed**, and **Blend: vary each ingredient**. A pre-mix keeps the same ingredient percentages while the amount used can change. For a blend, set Lowest and Highest for each ingredient; the app chooses each amount separately. In the sample, Fats and oils groups coconut oil and sunflower oil. Their proportions and combined amount can change between formulations.

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
