# Demo walkthrough

The walkthrough both release checklists run: the Mac's (`desktop/README.md`,
item 6) and Windows' (`desktop/windows/README.md`, item W10). The quoted
words are the screen's own; only OS words differ between the two.

## The yogurt pilot

Sample walkthrough: click "Try the yogurt pilot". "Yogurt pilot -
Round 6 to score" opens on "4 · Review" (15 formulations scored, Round
6 made and waiting). On "2 · Make a round", open "Or upload results
from a file", click "Browse files" and choose "Round 6 results
(pilot).xlsx" in the FoodOptimizer folder (Mac: in your home folder; Windows: Quick access) (it is already there; "Download the
pilot's Round 6 results" is the other way), then click "Save uploaded
results". "3 · Results" shows "Round 6 results" (F16, F17 new best, tied);
on "4 · Review", "Start Round 7" opens "2 · Make a round" with
Suggestion 1–3, and "See the pilot's Rounds 7–10" opens "Yogurt pilot
- all 10 Rounds" on "4 · Review". Manage project → "Start this example over" puts Round 6 back
to waiting and the results file back in the FoodOptimizer folder (Mac: in your home folder; Windows: Quick access), shows one
line ("… is back as shipped. Your version is in Saved copies."), and
the old version is listed in Saved copies. Close and reopen the app:
a sample opens on "4 · Review" again.
Example setup: "Open the burger example" opens "Example setup -
plant-based burger" on "1 · Set up" (illustrative values, no results:
calculations, pre-mixes, a process setting, records, a formulation
total); its "4 · Review" says "No results: this example shows set-up
only." with "Try the yogurt pilot". It never appears unless opened.

## A project of your own

UI walkthrough on a project of your own: create one with a default
batch size of 100 g, Lot and actual amounts recorded (Preparation and
records), and a Water row whose Rule cell says "= rest", so its
Highest reads "calculated" and one line under the grid gives
what it comes to in numbers. On "1 · Set up", type
into the ingredients grid and check that "Save changes" lights up and
"Discard changes" puts the grid back; open "Preparation and records" and
"Advanced" and check nothing else sits on the tab. In "Preparation and records",
check that an ingredient limit offers a "Limit is" choice of "At
least", "At most", "Between" or "Exactly" - one box, or two for
"Between" -
and that "Write it as" offers "% of default batch size" (it is
there only while a default batch size is set), and that picking an
ingredient does not close "Preparation and records" under you. Go to "2 · Make a
round", generate one, check every row of the round table totals the
batch size and that Water's column is headed "Water · calculated (g)",
change "Batch size (g)" to 250 and watch every amount
in the round table follow it, download the round sheets (Excel) and
open it: the sheets
are protected, only the cells the instruction line names take a value,
and a printed greyscale copy makes clear which they are — every one of
them is boxed, and boxed and shaded are the one treatment: there is no
second, shaded-but-locked kind of cell. Each formulation page has an
"Actual (g)" column beside its amounts, with a "Lot" cell per
ingredient on the "Round 1" page. Water is printed as "Water ·
calculated" on the formulation pages and "Water · calculated (g)" on the
round sheet — the mark on the name, the unit behind it, one shape
everywhere, with the line "Water is calculated: = rest
(batch size − every other ingredient). Weigh the amount printed."
under the amounts. Record a result (or mark one Not
scored and score it later from Results), and confirm the `.pkl` appears
in the FoodOptimizer folder (Mac: in your home folder; Windows: Quick access).

## The all-formulations workbook

On "3 · Results", click "Download all formulations (Excel)" and open
it: the "Set up" sheet in THAT file is the one with the Rule column,
and it spells Water out as "= rest (batch size − every other
ingredient)" with "calculated" in its Status cell. (The round sheets
in "A project of your own" are a different download and carry no Set-up sheet.)
