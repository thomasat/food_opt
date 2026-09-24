Food Optimizer helps food product developers find better formulations in fewer rounds of lab work. You type your ingredients and your measurements into two tables; it suggests the next round of formulations to make, prints the sheets for the bench, and learns from what you measured. No knowledge of statistics or machine learning is needed.

Not every ingredient has to be searched for. Give a row a **Rule** — `= batch size − Water − Salt`, `= 1.5 % of batch size`, or `= rest` for the one row that takes whatever is left of the batch size — and the app works that amount out from the other rows every time. A limit over several ingredients can say **Exactly** as well as At least, At most and Between, in grams or as a **% of default batch size**.

**Desktop app (macOS):** download the latest `.dmg` from the
[releases page](https://github.com/thomasat/food_opt/releases), drag the app to
your Applications folder, and open it. The first time it opens it downloads
about 1 GB. This usually takes under a minute; on a slow network, a few
minutes. After that it works fully offline.

**Run it from source:**

pip install -r requirements.txt

streamlit run app.py
