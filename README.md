Food Optimizer helps food product developers find better formulations in fewer rounds of lab work. You type your ingredients and your measurements into two tables; it suggests the next round of formulations to make, prints the sheets for the bench, and learns from what you measured. No knowledge of statistics or machine learning is needed.

**Desktop app (macOS):** download the latest `.dmg` from the
[releases page](https://github.com/thomasat/food_opt/releases), drag the app to
your Applications folder, and open it. The first time it opens it downloads
about 1 GB. This usually takes under a minute; on a slow network, a few
minutes. After that it works fully offline.

**Run it from source:**

pip install -r requirements.txt

streamlit run app.py
