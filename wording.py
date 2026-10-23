"""Every word the user reads. Change a word here and it changes everywhere;
the tests read this module too.

Use sentence case for headings, buttons and standalone statuses. Start a
separate heading or command after a middle dot with a capital letter. Keep
user-entered names, units, calculation tokens and legacy import labels unchanged.
Prefer direct instructions and current control names over idioms.
"""
import sys


def on_windows():
    """True on Windows. Read at call time, never frozen at import, so a
    sentence names the computer the reader is on (and a test can say which)."""
    return sys.platform == "win32"


# ------------------------------------------------------------------ #
# Concepts. The set of formulations issued together is a "round" on
# screen. It was a "batch" until 0.5.0, when "batch size" took that word
# for the weight of ONE formulation — the sense the bench already uses it
# in — and one word could not be both. Every round-naming sentence below
# is built from these two constants, so a name change is a one-line edit.
# Stored field names keep the old spelling (`pending_batch`, `batch_totals`);
# they never reach a screen.
# ------------------------------------------------------------------ #
ROUND = "round"
ROUND_CAP = "Round"
FORMULATION = "formulation"
FORMULATION_CAP = "Formulation"

# The weight of ONE formulation. Two boxes ask it — the round screen's
# Batch size and Set up's Default batch size (FORMULATION_TOTAL_NAME,
# further down, is built from this noun) — so the words are spelled once
# here and every sentence about either box reads from them.
BATCH_SIZE_NOUN = "batch size"
BATCH_SIZE_NAME = "Batch size"

# Plain nouns handed to ui_helpers.plural(n, word) at more than one call
# site, so the word is spelled once.
INGREDIENT = "ingredient"
PROCESS_SETTING = "process setting"
LIMIT = "limit"
MEASUREMENT = "measurement"
ROW = "row"

# "Note" is both a table column header (food_bo's own dataframes carry the
# same header) and a form field label; one spelling serves both.
NOTE = "Note"

# A formulation with no result yet: never made, or made and never measured.
# It can be scored later from the Results tab. A concept, not a screen
# label: the results grid, the record, the workbook and the counter all
# say it, so it is spelled once and spelled here.
NOT_SCORED = "Not scored"


# ------------------------------------------------------------------ #
# Shell: the page title, the three tabs, the sentence every irreversible
# action's confirmation ends with, the project-name rule.
# ------------------------------------------------------------------ #
APP_TITLE = "Food Optimizer"

# The first line the window ever shows from the app itself. Streamlit answers
# the wrapper's health check as soon as its server is up, seconds before
# app.py has finished importing what it runs on, so app.py draws this line
# before those imports and clears it after them. Without it the window is
# blank for those seconds.
STARTING_APP = f"Starting {APP_TITLE}. Please wait while the app loads."

# The three tabs, in loop order. The separator is U+00B7 MIDDLE DOT.
TAB_SETUP = "1 · Set up"
# The tab's name without its number, for prose that points at it.
TAB_BATCH_NAME = f"Make a {ROUND}"
TAB_BATCH = f"2 · {TAB_BATCH_NAME}"
TAB_RESULTS = "3 · Results"

# Archived copies are written beside the project's own file, which on the
# desktop app is the FoodOptimizer folder. Every tab and the sidebar use it,
# so the sentence exists once.
COPY_KEPT = ("A copy is saved first. To bring it back, use Saved copies "
             "in the sidebar.")

NAME_RULE = "Up to 64 characters. Start with a letter or a number."
# The characters a project name may hold after its first (app._NAME_RE).
_NAME_EXTRA = " _.-"


def name_not_allowed(name):
    """Create project with a name the rule refuses: which part breaks it and
    the fix, worked out from the name typed (the rule alone named no
    problem: "bad/name" was told it was "Up to 64 characters")."""
    if len(name) > 64:
        return f"Shorten the name to 64 characters; it has {len(name)}."
    def ok(c):
        return c.isascii() and (c.isalnum() or c in _NAME_EXTRA)
    if not (name[0].isascii() and name[0].isalnum()):
        return "Start the name with a letter or a number."
    bad = "".join(dict.fromkeys(c for c in name if not ok(c)))
    if bad:
        shown = " ".join(f"“{c}”" for c in bad)
        return (f"Take {shown} out of the name. Use letters, numbers, "
                "spaces, dots, dashes or underscores.")
    return NAME_RULE
# Create project with nothing in the box: the rule above names no problem.
NAME_EMPTY = "Type a project name first."


# ------------------------------------------------------------------ #
# ui_helpers: generic building blocks used across every tab.
# ------------------------------------------------------------------ #
YES_CONTINUE = "Yes, continue"
CANCEL = "Cancel"

NEED_A_VARIABLE = "Add at least one ingredient or process setting."
NEED_A_MEASUREMENT = "Add at least one measurement."

# scale_error's and bounds_caution's arguments to outside_message: what a
# value is outside of, and the hint scale_error appends that bounds_caution
# does not.
YOUR_RANGE = "your range"
ALLOWED_AMOUNTS = "its allowed amounts"
# The half of the scaled caution that names what was exceeded. One spelling,
# whether the line lists the ingredients or counts them.
AMOUNTS_YOU_ALLOWED = "the amounts you allowed"
# The value is outside the measurement's own span, so the hint names the
# control that sets it. "Widen the range in Set up" named nothing on that
# screen: the grid's columns are Scale minimum and Scale maximum.
# HIGHEST_MEASURABLE_LABEL is the same words, defined with the grid's other
# column headers further down; this one cannot read it because it is needed
# above them. The guard below the grid pins the two together.
WIDEN_RANGE_HINT = " Check the value, or adjust Scale minimum or Scale maximum in Set up."
# A unit that is a word for a score rather than a unit of anything: '12
# score' and '0 to 10 points' do not read, so a result outside its scale is
# said without it.
SCORE_WORD_UNITS = ("score", "scores", "point", "points", "pts", "rating")


def outside_scale(name, value_text, range_text, above=True):
    """'Suggestion 1: Crunch 12 is outside 0–10. Fix it, or raise Scale
    maximum in Set up.' — a result typed outside its measurement's range,
    naming the one Set up cell that would let it in (`above`: past the
    top). The texts carry their unit already ('12 N', '0–20 N')."""
    fix = (f"raise {HIGHEST_MEASURABLE_LABEL}" if above
           else f"lower {LOWEST_MEASURABLE_LABEL}")
    return (f"{name} {value_text} is outside {range_text}. Fix it, or "
            f"{fix} in Set up.")


SMALLER_TOTAL_HINT = (f"Print at a smaller {BATCH_SIZE_NOUN}, or widen them "
                      "in Set up.")
# The same two ways out for a limit rather than an ingredient's own amounts:
# a limit is changed in Set up, not widened there.
SMALLER_TOTAL_OR_LIMIT_HINT = (f"Print at a smaller {BATCH_SIZE_NOUN}, or "
                               "change the limit in Set up.")


def scaled_limit_caution(total_text, limit_text):
    """'At 150 g, the limit Water + Oil: at most 20 g is not met. Print at a
    smaller total, or change the limit in Set up.'

    An amount scaled to a total can carry a limit over with it, and a limit
    is a hard calculation: the sheet that breaks one has to say which, in the same
    words the Limits list writes it in.
    """
    return (f"At {total_text}, the {LIMIT} {limit_text} is not met. "
            + SMALLER_TOTAL_OR_LIMIT_HINT)


def scaled_amounts_note(total_text, factor_text, size_text):
    """'Made to 250 g — every amount is 2.5 × the amounts you set per
    100 g.' — the line a round made bigger than one formulation carries.

    It used to read `At 250 g, 5 of 5 ingredients go past the amounts you
    allowed`, on screen, on every formulation page and on the Round sheet.
    Making a bigger lot of the same formula is ordinary bench work, and the
    app called all five of them mistakes.
    """
    return (f"Made to {total_text} — every amount is {factor_text} × the "
            f"amounts you set per {size_text}.")


def scaled_amounts_caution(total_text, names_text="", n_outside=0, n_total=0):
    """'At 150 g, Pea protein isolate and Water go past the amounts you
    allowed. Print at a smaller total, or widen them in Set up.' — the one
    line for a formulation total that pushes amounts past what the project
    allows.

    Up to three ingredients are named. Above that the line counts them: an
    eight-name list is unreadable, and this same sentence is printed on the
    sheet the technician weighs out from. Either way it ends with the two
    ways out, because the sheet carries no other instruction.

    One sentence, however many ingredients and however many formulations are
    outside: the fix is the same one every time, and a caption per ingredient
    per row buried the step below it under eight lines of raw numbers. The
    total is named because the total is what did it.
    """
    who = (names_text if names_text
           else f"{n_outside} of {n_total} {INGREDIENT}s")
    goes = "go" if (n_outside > 1 or not names_text) else "goes"
    return (f"At {total_text}, {who} {goes} past {AMOUNTS_YOU_ALLOWED}. "
            + SMALLER_TOTAL_HINT)


def saved_line(when):
    """'Saved automatically at 14:32, on this Mac' — `when` is the already
    formatted time or date+time."""
    return f"Saved automatically at {when}, on this {'PC' if on_windows() else 'Mac'}"


def best_moved(before, after):
    """'Best moved from Formulation 3 to Formulation 7.'"""
    return (f"The highest-scoring formulation changed from {FORMULATION_CAP} {before} to "
            f"{FORMULATION_CAP} {after}.")


# ------------------------------------------------------------------ #
# Sidebar: projects, saved copies, manage project.
# ------------------------------------------------------------------ #
PROJECTS_HEADER = "Projects"

NEW_PROJECT_NAME_LABEL = "New project name"
NEW_PROJECT_PLACEHOLDER = "e.g. Oat cookie v2"
CREATE_PROJECT = "Create project"


def name_taken(name):
    """Create project with the name of a project on the list. "Open it
    below" pointed down from a fold that sits under Open project."""
    return (f"A project named {name} already exists. Open it from "
            f"{OPEN_PROJECT_LABEL}, or choose another name.")


def name_used_by_copy(name, suffix=None):
    """Create project with a name a saved copy wears, or one ending the way
    a saved copy's name ends (`suffix`, e.g. "_deleted")."""
    if suffix:
        return (f"Names ending in {suffix} are kept for saved copies. "
                "Choose another name.")
    return f"A saved copy is already named {name}. Choose another name."


OPEN_PROJECT_LABEL = "Open project"

# The built-in samples (spec 2026-10-13 §9, owner wording 2026-10-14): one
# job each, named for that job, on the welcome screen and in Manage
# project's Samples group alike. The yogurt pilot is the demo: real results,
# opened on 4 · Review. The example setup is how to set a project up:
# illustrative values, no results, opened on 1 · Set up.
OPEN_YOGURT_LABEL = "Try the yogurt pilot"
OPEN_YOGURT_CAPTION = "Real tasting data; shows results and the Review."
OPEN_YOGURT_HELP = (
    "A real yogurt-style pilot: 15 formulations scored, Round 6 made and "
    "waiting for its results.")
SEE_EXAMPLE_SETUP_LABEL = "Open the burger example"
SEE_EXAMPLE_SETUP_CAPTION = "Illustrative values; shows calculations and pre-mixes."
SAMPLES_HEADING = "**Examples**"   # the welcome screen says "example" too
# The sidebar fold that holds the New project box while a project is open.
NEW_PROJECT_FOLD = "New project"
BURGER_PROVENANCE_CAPTION = "Example setup; illustrative values, no results."
BURGER_PROVENANCE_HELP = (
    "Every amount, property, limit and target here is illustrative, not "
    "measured. It shows calculated rows (hydration water as 2.2 × the "
    "textured proteins, water as the rest), pre-mixes, a process setting, "
    "records and "
    "a formulation total. It has no results; the yogurt pilot has real ones.")
# Once the example holds a Round or a result of the reader's own, "no
# results" would be false: the caption and its "?" drop it.
BURGER_PROVENANCE_CAPTION_USED = "Example setup; illustrative values."
BURGER_PROVENANCE_HELP_USED = (
    "Every amount, property, limit and target here is illustrative, not "
    "measured. It shows calculated rows (hydration water as 2.2 × the "
    "textured proteins, water as the rest), pre-mixes, a process setting, "
    "records and "
    "a formulation total.")
BURGER_NO_RESULTS = "No results: this example shows set-up only."
# The label of the Set up tab's second tier, on screen and in every text that
# points to it.
MORE_SETTINGS_EXPANDER = "Preparation and records"
def sample_provenance_caption(n_scored):
    """The yogurt pilot's sidebar line: 'Real tasting data; 15 formulations
    scored.' — `n_scored` is what the project holds now (15 in the Round 6
    sample as shipped, 18 once its Round 6 is saved, 30 in the all-10
    one)."""
    return f"Real tasting data; {n_scored} formulations scored."


SAMPLE_PROVENANCE_CAPTION = sample_provenance_caption(15)   # the Round 6 sample
# The pilot as it was run: its own limits, its panel, its process and its
# fixed budget, each said once. The total keeps four decimals: the limit is
# 97.69051–98.69051 g, and 97.69 g fails its check (limit_text says 97.6905
# on Set up for the same reason).
_SAMPLE_PROVENANCE_FACTS = (
    "Rounds 1–2 were a random starting set drawn by the engine; Rounds 3–10 "
    "were chosen by an earlier version of this engine. "
    "The limits are the pilot's own: each ingredient's Lowest and Highest, "
    "and their total at 97.6905–98.6905 g per 100 g. Soy protein isolate (1 %) "
    "and tricalcium phosphate (0.31 %) were fixed in every formulation, so "
    "they are not listed. Each result is the median of a trained panel of "
    "five; no repeats. Each formulation was mixed, cooked at 90 °C for 8 min "
    "and chilled. The budget was fixed at 10 Rounds of 3")
# The Round 6 sample shows Rounds 1–5 and 6, so its "?" says where the rest are.
SAMPLE_PROVENANCE_HELP = (_SAMPLE_PROVENANCE_FACTS
                          + " (Rounds 7–10 are in Yogurt pilot - all 10 Rounds).")
SAMPLE_PROVENANCE_HELP_ALL10 = _SAMPLE_PROVENANCE_FACTS + "."


# The project-level Method: one text area in Preparation and records, printed on the
# Round sheet under the title.
# The four things a project can record beside what it is asked to. All four
# are off for a new project: Vendor and SKU are specification data typed
# once and never looked at again, and Lot and Actual are two more columns
# down every printed page of a project that does not want them.
def no_such_field(name):
    """A caller asking the project to record something it has no box for."""
    return f"{name} is not an available recording field."


ALSO_RECORD_LABEL = "Additional records (optional)"
# Where each option is entered lives here, not on its chip: a chip reading
# "Lot numbers · Enter in Make a round" was cut to "Lot numbers · En…" (S13).
ALSO_RECORD_HELP = ("These options add entry fields in the app or "
                    "workbook: Vendor and SKU for each ingredient, in the "
                    "Ingredients table; Lot numbers in Make a round; your "
                    "own fields for each formulation or each ingredient. "
                    "Saved values are retained when an option is "
                    "turned off.")
RECORD_FIELD_LABELS = {
    'vendor': "Vendor",
    'sku': "SKU",
    'lot': "Lot",
    'actual': "Actual amounts",
}


METHOD_LABEL = "Preparation method"
METHOD_HELP = "Steps shared by every formulation or trial. Included on Round overview in the workbook. Changes save when you leave this field."
METHOD_PLACEHOLDER = ("e.g. Mix the dry blend into the water, 60 s. Add the "
                      "oils. Form 100 g patties.")
METHOD_SAVED = "Method saved."
METHOD_CLEARED = "Method cleared."
# The heading it prints under on the Round sheet: every formulation of a
# round is made the same way, and the sheet says so once.
METHOD_SHEET_HEADING = "Method — the same for every formulation"

PROJECT_LOAD_ERROR_SIDEBAR_NOTE = 'This project could not be opened. See the explanation on the main screen.'

SAVED_COPIES_HEADING = "**Saved copies**"
SAVED_COPIES_CAPTION = "Includes every ingredient, formulation and result."
SAVED_COPIES_HELP = ("A saved copy includes ingredients, "
                     "measurements, every formulation and result.")

COPY_UNAVAILABLE = ("Saving a copy is unavailable while the project "
                    "file cannot be read.")
SAVE_A_COPY = "Save a copy of this project"

OPEN_A_SAVED_COPY = "Open a saved copy"
OPEN_SAVED_COPY_CAPTION = "The current project is copied first."
OPEN_SAVED_COPY_HELP = ("A copy you saved yourself, or one the app saved "
                        "before a change. The current project is copied "
                        "first.")
CHECK_THIS_COPY = "Open this copy"

# The copies the app makes for itself, before a change that cannot be
# undone. They were files in the project folder with names like
# `burger_pre_edit` that nothing on screen ever mentioned, under a heading
# promising a list — so each is named here for what it was taken before.
SAFETY_COPIES_CAPTION = "Copies this app made before a change:"
SAFETY_COPY_REASONS = {
    'pre_edit': "Before an edit in Set up",
    # A correction to a recorded result is the copy a reader actually comes
    # back for, and five rows all reading "Before an edit" told them
    # nothing about which.
    'pre_correction': "Before correcting a recorded formulation",
    'pre_delete': "Before a deletion",
    'pre_restore': "Before opening a saved copy",
    'sample_reset': "Before Start this example over",
    'archived': "Before clearing project contents",
    'deleted': "Before deleting the project",
    # Two labels no version writes any more; a folder that met an earlier
    # one still holds them, and a copy with no name is a copy nobody dares
    # open.
    'pre_rewind': "Before a change to what was recorded",
    'pre_undo': "Before a change to what was recorded",
}
OPEN_SAFETY_COPY = "Open"


def older_copies_fold(n):
    """'Older copies (28)' — the fold the copies past the three newest sit
    in. The app makes one before every edit, so a month of ordinary work
    left thirty of them down the sidebar under a heading promising a list;
    the three that are ever wanted are the three newest."""
    return f"Older copies ({n})"


def copies_by_day(day_text):
    """'Yesterday' / '12 Sep' — the one line each day's copies sit under
    inside the fold, newest day first."""
    return day_text


COPIES_TODAY = "Today"
COPIES_YESTERDAY = "Yesterday"
DELETE_OLD_COPIES = "Delete copies older than a week"


def delete_old_copies_question(n, many=True):
    """'Delete 22 saved copies older than a week? The three newest are
    kept, and so is anything from the last seven days. Copies you
    downloaded yourself are not in this folder.'"""
    what = f"{n} saved copies" if many else "1 saved copy"
    return (f"Delete {what} older than a week? The three newest are kept, "
            "and so is anything from the last seven days. Copies you "
            "downloaded yourself are not in this folder.")


def old_copies_deleted(n, many=True):
    return f"{n} saved copies deleted." if many else "1 saved copy deleted."


NO_OLD_COPIES = "No copy here is older than a week."


def copy_when(clock, today=True):
    """'today 21:58', or '12 Sep 21:58' for a copy made on another day. The
    caller has already written the clock; the word for the day is here."""
    return f"today {clock}" if today else clock


def safety_copy_line(reason, when):
    """'Before an edit to a recorded result · today 21:58'."""
    return f"{reason} · {when}"


def prefilled_outside_caption(no):
    """'Formulation 4 was made at amounts outside today's Lowest and
    Highest. You can still add it as typed.'

    Start from the best so far types the amounts into the boxes itself, and
    the project's allowed amounts may have moved since that formulation was
    made. Being told off for numbers the app had just written was the worst
    five seconds of the cold read; the form says it plainly, before Add, and
    adding is still allowed."""
    return (f"{FORMULATION_CAP} {no} was made at amounts outside today's "
            f"{LOWEST_LABEL} and {HIGHEST_LABEL}. You can still add it as "
            "typed.")


def copy_downloaded(file_name):
    """'Saved as "burger copy 2026-09-15.json".'

    A download is the one action in the app that leaves no mark on the
    screen: the reader clicked, nothing moved, and the heading above it
    promises a list they could not see themselves in."""
    return f"Saved as '{file_name}'."
# A saved copy that will not open. ONE sentence, whatever is wrong with
# it: the reader is a food scientist whose copy will not open, and the
# stored field name, its shape and the programmer's quotation marks are
# for the log (food_bo._damaged writes them there).
COPY_DAMAGED = "This copy is damaged and cannot be opened."
NOT_A_COPY = "This file is not a Food Optimizer copy."
COPY_FROM_A_NEWER_VERSION = (
    "This copy was made with a newer version of Food Optimizer. Update the "
    "app, then try again.")



def _copy_subject(file):
    """The copy a refusal is about, as the reader knows it: the file they
    chose, or the row in Saved copies they clicked."""
    return f"**{file}**" if file else "This copy"


def copy_refused(file, reason):
    """Open a saved copy, refused: which file, that it will not open, and
    what to do. `reason` is what FoodOptimizer.validate_state raised (one of
    the constants above, or COPY_TWO_BALANCE_ROWS); anything else reads as
    damaged, which is what the reader can act on."""
    subject = _copy_subject(file)
    if reason == NOT_A_COPY:
        return f"{subject} is not a Food Optimizer copy. Choose another file."
    if reason == COPY_FROM_A_NEWER_VERSION:
        return (f"{subject} is from a newer Food Optimizer. Update the app, "
                "then open it again.")
    return (f"{subject} is damaged and can't be opened. Try another copy "
            "from your FoodOptimizer folder.")


def copy_missing(file):
    """A row in Saved copies whose file has gone from the folder."""
    return (f"{_copy_subject(file)} is no longer in your FoodOptimizer "
            "folder. Choose another copy.")


def open_saved_copy_warning(name, holds, project_name, held, open_text=""):
    """`holds` is the list of 'N formulations'/'N ingredients'/'N process
    settings' phrases the saved copy contains; `held` is the same phrase for
    what replacing it would give up. `open_text` (0.8.0 e2e fixes B3):
    open_round_lost's sentence while a Round is open."""
    lost = f"{open_text} " if open_text else ""
    return (f"This copy holds **{name}**: " + ", ".join(holds)
            + f". Replace **{project_name}**, which has {held}? " + lost + COPY_KEPT)


YES_REPLACE = "Yes, replace"


def copy_apply_failed(file, project_name):
    """Yes, replace, when the copy passed the check but would not load."""
    return (f"{_copy_subject(file)} could not be opened, and {project_name} "
            "is unchanged. Try another copy.")


def copy_reason(archived):
    """What a copy the app kept was taken before, in Saved copies' own
    words ('Before opening a saved copy'), read off the end of its file
    name — or None for a name that is not one of the app's copies."""
    name = str(archived or "")
    for suffix, reason in SAFETY_COPY_REASONS.items():
        tail = f"_{suffix}"
        stem = name.rstrip("0123456789")
        if name.endswith(tail) or (stem != name and stem.endswith(tail + "_")):
            return reason
    return None


def copy_kept_line(archived):
    """'A copy is in Saved copies: Before opening a saved copy.' — never the
    copy's file name (Yogurt pilot_pre_restore), which is machinery."""
    reason = copy_reason(archived)
    if reason is None:
        return "A copy is in Saved copies."
    return f"A copy is in Saved copies: {reason}."


def restored_flash(count_text, project_name, archived=None):
    text = f"Restored {count_text} into {project_name}."
    if archived:
        text += " " + copy_kept_line(archived)
    return text


MANAGE_PROJECT = "Manage project"

START_OVER_LABEL = "Clear project contents"
YES_START_OVER = "Yes, clear contents"

# A sample's Start over: the shipped file goes back, the reader's own version
# is kept as a copy.
START_SAMPLE_OVER_LABEL = "Start this example over"
YES_START_SAMPLE_OVER = "Yes, start over"


def start_sample_over_warning(name):
    return (f"{name} goes back to how it shipped. Your version is kept "
            "in Saved copies.")


def sample_restored(name):
    """The status after Start this example over: the one line it shows."""
    return f"{name} is back as shipped. Your version is in Saved copies."


def start_over_warning(project_name, held_text=None):
    """`held_text` is None for an already-empty project."""
    count = f"This includes its {held_text}. " if held_text else ""
    return (f"Clear all ingredients, process settings, measurements and results from **{project_name}**? "
            + count + "The project keeps its name. " + COPY_KEPT)


DELETE_PROJECT_LABEL = "Delete this project"
YES_DELETE_IT = "Yes, delete it"


def delete_project_warning(project_name, held_text=None):
    """`held_text` is None for an already-empty project. What Delete does to
    the Open project list is what Delete means; saying it again in the
    vocabulary of a list said nothing the verb had not."""
    if held_text is None:
        return (f"Delete **{project_name}**? It has no formulations yet. "
                + COPY_KEPT)
    return f"Delete **{project_name}** and its {held_text}? " + COPY_KEPT


def copy_saved_as(archived):
    return copy_kept_line(archived)


def project_deleted_flash(name, archived=None):
    if archived:
        return f"Deleted {name}. A copy was saved as {archived}."
    return f"Deleted {name}."


OPEN_BUTTON = "Open"


# ------------------------------------------------------------------ #
# Welcome panel (no project open yet).
# ------------------------------------------------------------------ #
# The welcome screen, no project open (owner, 2026-10-19): the two examples
# first, under one heading; your own project second, smaller, with its steps.
# The one line under the title on the welcome screen: what the app does.
WELCOME_TAGLINE = "Suggests what to make next, from your results."
WELCOME_HEADER = "## Start with an example"
WELCOME_OWN_HEADER = "#### Or start your own project"


def welcome_steps():
    return (
        "1. **Name a project** in the sidebar on the left.\n"
        "2. **Set up** ingredients, measurements and targets.\n"
        "3. **Generate formulations**, prepare each one, and record your measurements.\n"
        + WELCOME_REVIEW_STEP      # review guide: Review is the loop's fourth step
    )


DOWNLOAD_TEMPLATE = "Download ingredients template (Excel)"


# ------------------------------------------------------------------ #
# Load-error / save-error banners, and the discarded-batch notice.
# ------------------------------------------------------------------ #
def project_load_error_info(sample=False):
    """Under a project that will not open: the two ways on, by the names the
    sidebar gives them (a sample is started over, not cleared)."""
    start_over = START_SAMPLE_OVER_LABEL if sample else START_OVER_LABEL
    return (f"In the sidebar, use {MANAGE_PROJECT} › {OPEN_A_SAVED_COPY} or "
            f"{start_over}. The damaged file is copied first.")


SAVE_ERROR_WARNING = ("**Your last change was not saved.** Save a "
                      "copy now, then click Reload project.")
# The banner above it reads "Save a copy now, then click Reload
# project.", so the button is the first half of that sentence.
SAVE_COPY_NOW = "Save a copy now"
RELOAD_PROJECT = "Reload project"
PROJECT_RELOADED = "Project reloaded from the latest saved copy."


# What the discard notice blames when it knows. The general case is the
# tab — three ways lead there (the ingredient list, a held row, the
# allowed amounts) and naming all three in one subordinate clause was
# unreadable on one pass. The total is its own answer because it is one
# control the reader has just touched, and "your set-up changed" sent them
# looking for what else they had done.
# TOTAL_CHANGED_REASON is the other one; it lives beside the total's own
# name, further down, because it is built from it.
SETUP_CHANGED_REASON = "your set-up changed"


def batch_discarded_notice(no=None, reason=SETUP_CHANGED_REASON):
    """'Batch 2 was discarded: your set-up changed after it was made.
    Generate a new one.' — or, when the total is what did it, 'Batch 2 was
    discarded: the total of each formulation changed after it was made.'
    """
    # Every screen that flashes this knows the number. The unnumbered form
    # is the honest fallback for a batch whose number did not survive the
    # read that discarded it, and nothing reaches it today.
    who = f"{ROUND_CAP} {no}" if no is not None else f"The open {ROUND_CAP}"
    return f"{who} was discarded: {reason}. Generate a new one."


YES_SAVE_AND_DISCARD = "Yes, save and discard"


def saving_discards_round(no, held_text, own=0):
    """The question a Set-up save owes an open round before it takes it
    away: 'Saving will discard Round 2: 3 formulations, 1 of them added by
    you. A formulation you added goes with the round — a set-up change can
    make it invalid.'

    The reader typed those amounts themselves and wrote a note on them, and
    a toast AFTER the round was gone was the first they heard of it. The
    count of their own rows is said because it is the half of the round
    nothing can generate back.
    """
    line = f"Saving will discard {ROUND_CAP} {no}: {held_text}"
    if own:
        line += f", {own} of them added by you"
    line += "."
    if own:
        line += (f" A {FORMULATION} you added goes with the {ROUND_CAP} — a "
                 "set-up change can make it invalid.")
    return line


def project_created(name):
    return f"Created {name}."


def project_opened(name):
    return f"Opened {name}."


# The two samples, as their success line names them (not by file name).
YOGURT_SAMPLE_NAME = "the yogurt pilot"
BURGER_SAMPLE_NAME = "the burger example"


def sample_unreadable(name):
    """A sample the app ships, gone or damaged inside the app itself."""
    return (f"The app's file for {name} is missing or damaged. Download "
            "Food Optimizer again to replace it.")


def sample_opened(what, added=0, files=0):
    """'Opened the yogurt pilot (2 projects added).' — `added` is how many
    sample projects this click wrote into the project list, `files` how
    many results files it wrote beside them (S4: 'Opened the yogurt pilot
    (2 projects, 1 results file added).'); none is left unsaid ('Opened the
    burger example.')."""
    parts = []
    if added:
        parts.append(f"{added} project{'s' if added != 1 else ''}")
    if files:
        parts.append(f"{files} results file{'s' if files != 1 else ''}")
    if not parts:
        return f"Opened {what}."
    return f"Opened {what} ({', '.join(parts)} added)."


def batch_line_open(no, n):
    """The one line under the title on tabs 1 and 2 once a batch exists and
    still has rows without a result.

    'to record', the same word tab 3's own button uses of the same count of
    the same batch: 'N to make' and 'N to record' read as two different
    numbers, and the reader had to work out that they were one."""
    return f"{ROUND_CAP} {no} · {n} to record"


def batch_line_recorded(no):
    """Same line, once every row of that batch has been recorded (or left
    out)."""
    return f"{ROUND_CAP} {no} · Recorded"


# ------------------------------------------------------------------ #
# Tab 2 · Make a batch: generate, the batch table, downloads, record
# results, upload results.
# ------------------------------------------------------------------ #
GENERATE_FORMULATIONS_DISABLED = "Generate formulations"
BACK_TO_SETUP = "Back to Set up"

GENERATING_SPINNER = "Choosing the next formulations…"
# Every failed Generate that is not one of the named refusals: the limits
# left no room, or something broke. Both ways out fit either case.
GENERATE_FAILED = ("Could not make formulations. Ask for fewer, or widen a "
                   f"{LIMIT} in Set up.")


def repeat_of_formulation(no):
    return f"Repeat of {FORMULATION_CAP} {no}"


def batch_ready(no):
    """The flash a freshly generated batch lands on. It names the two steps
    that follow, in the order the screen puts them: a batch that is "ready"
    and nothing more left the reader looking for what to do with it."""
    return f"{ROUND_CAP} {no} is ready. Next: download the workbook."


FORMULATIONS_TO_GENERATE = "Formulations to generate"


# The expander under the Generate row (and under the batch table once one is
# open): a formulation the scientist chose, added to the batch beside the
# generated ones. It replaced the Repeat checkbox, which could only ever
# repeat the best.
ADD_OWN_EXPANDER = f"Add a {FORMULATION} to this {ROUND_CAP}"
ADD_OWN_NO_BATCH_CAPTION = "Generate first to see suggestions too."
ADD_OWN_NO_BATCH_HELP = "Want the app's suggestions too? Click Generate first, then add yours."
OWN_NOTE_PLACEHOLDER = "e.g. Repeat of 4 with more salt"
ADD_TO_THIS_BATCH = f"Add to this {ROUND_CAP}"
def enter_amounts_for(names_text, n=1, zero_allowed=True):
    """'Enter an amount for Salt and Water; type 0 to leave one out.' — a
    formulation typed in with blank boxes: Add a formulation of your own,
    a correction on Results, a past formulation. It names the boxes that
    are still empty, so the reader does not hunt for them. `n` is how many
    there are ('leave it out' for one); `zero_allowed` is False when a
    blank row's Lowest is above 0, and 0 is then not offered."""
    if not zero_allowed:
        return f"Enter an amount for {names_text}."
    which = "it" if n == 1 else "one"
    return f"Enter an amount for {names_text}; type 0 to leave {which} out."
# What the row says when the user typed no reason of their own. The note is
# part of the record, so a row on the batch table is never blank about what
# it is.
OWN_FORMULATION_NOTE = f"Own {FORMULATION}"


def own_formulation_added(no, batch_no):
    return f"{FORMULATION_CAP} {no} added to {ROUND_CAP} {batch_no}."


def generate_button_label(n):
    return f"Generate {n} formulations"


# The one line that says how formulations are chosen: the third How it works
# bullet and, word for word, the caption under Generate while fewer than five
# have results (spec 2026-09-30 §3, §6; review M2: "cover", not "spread out").
HOW_CHOSEN_2 = (f"Until five {FORMULATION}s have results, new ones cover the allowed amounts. "
                f"After that, each {ROUND_CAP} aims closer to your targets.")


def first_formulations_caption(n=5):
    """Make a round's line before `n` formulations have results (N10, ≤ 8
    words): 'First 5 formulations explore the allowed amounts.' The rest is
    its "?" (HOW_CHOSEN_2)."""
    return f"First {n} {FORMULATION}s explore the allowed amounts."


# The open batch reads as the three steps of the work, each headed with its
# number. Step 1 takes no count of its own: the title directly above it
# already says how many formulations there are.
# Unnumbered. The tab strip above them is 1 · Set up, 2 · Make a round,
# 3 · Results, and "3 · Record the results" sitting under "3 · Results"
# had the cold reader clicking the tab when they meant the section. The
# order is the order they are drawn in.
STEP_MAKE_HEADING = f"##### Make the {FORMULATION}s"
STEP_PRINT_HEADING = "##### Download the workbook"
STEP_RECORD_HEADING = "##### Record the results"


def make_these(no, n):
    """A round heading, for example 'Round 1 · Make 3 formulations'."""
    word = FORMULATION if n == 1 else FORMULATION + "s"
    return f"**{ROUND_CAP} {no} · Make {n} {word}**"


# Every row in the batch is one the user added by hand, so there is no
# Generate control on the screen and nothing else would say why.
ONLY_OWN_FORMULATIONS_CAPTION = (
    f"This {ROUND_CAP} holds only your own formulations. For the app's suggestions "
    f"too, click Generate a different {ROUND_CAP}, then Generate."
)


def batch_discarded_caption(numbers_text, many=True):
    """The numbers the regenerate retired. Where the new batch picked up is
    on the table directly below, so the line does not say it twice."""
    word = f"{FORMULATION_CAP}s" if many else FORMULATION_CAP
    verb = "were" if many else "was"
    return f"{word} {numbers_text} {verb} discarded."


def change_text(name, delta, size):
    """'Water +12.00 g' — one change in one ingredient or setting. The sign is
    the typographic minus, not a hyphen: beside a plus of the same weight a
    hyphen reads as a dash between two words."""
    return f"{name} {'+' if delta > 0 else '−'}{size}"


# What each formulation is trying. The kinds are written lower case
# because each is a fragment: compared_with_cell() capitalises the one it is
# given as the first word of the cell. 0.8.0 e2e fixes (spec 2026-10-01 A2,
# review I3): the Compared with cell and the Why fold's Role column say the
# same three words by the same distance rule, so one
# suggestion is never "trying something different" in one table and
# "refines the best" in the other. ROLE_REFINES, ROLE_BALANCED and
# ROLE_EXPLORES further down are these three.
SUGGESTION_CLOSE = "close to the best"
SUGGESTION_BALANCED = "balanced"
SUGGESTION_DIFFERENT = "explores"
SUGGESTION_SPREAD = "covering the allowed amounts"
# A formulation the user typed is not a suggestion, so it is not one of the
# three kinds: saying "spread across the allowed amounts" of a row somebody
# wrote out by hand described the app's own sampling, not their formulation.
OWN_FORMULATION_KIND = f"your own {FORMULATION}"

# The cold start has no best to compare with, so the column header names what
# the formulations ARE spread across.
COMPARED_WITH_ALLOWED = "Compared with the allowed amounts"


def compared_with_column(best_no):
    """The batch table's header for the what-is-it-trying column. It names the
    formulation the changes are measured from once, at the top, so no cell
    under it has to repeat it."""
    return f"Compared with {FORMULATION_CAP} {best_no}"


# v3 (final-v3/2-make-round.png, wording.md §2): the open Round as one
# table, ingredients down, Best so far then each formulation across. Short
# formulation names (F10), as Review writes them; ui_batch turns them into
# T10 in a study of trials.
def short_no(no):
    """'F10'."""
    return f"F{int(no)}"


def _tied_names(nos):
    """'F10'; 'F16, F17 (tied)' for a tie (ties are named, never one picked)."""
    names = ", ".join(short_no(n) for n in nos)
    return names if len(nos) < 2 else f"{names} (tied)"


def improve_on_caption(nos):
    """Under the Round title: 'Best bets to improve on F10.' (≤ 8 words)."""
    return f"Best bets to improve on {_tied_names(nos)}."


def improve_on_help(nos):
    """The caption's "?"."""
    return (f"Chosen for how much they could improve on {_tied_names(nos)}, weighing how good "
            "they look and how unsure the model is.")


# The shipped yogurt pilot's Rounds 3-10 were picked by an earlier version of
# the engine (their notes say so); a Round the app generates there is not.
CHOSEN_IN_PILOT = "Chosen by the engine in the pilot"
# During the cold start there is no best to compare with (the old
# COMPARED_WITH_ALLOWED column); the caption says what the table is instead
# ("cover", never "spread": review M2), and HOW_CHOSEN_2 is its "?".
COVER_CAPTION = "These cover your allowed amounts."


def best_column(no, score_text):
    """The table's first column after the rows: 'Best so far (F10 · 95.0)'."""
    return f"Best so far ({short_no(no)} · {score_text})"


def best_tied_help(no, others):
    """The Best so far column's "?" on a tie: the first made is shown."""
    return (f"{short_no(no)} is shown: first made; tied with "
            f"{', '.join(short_no(n) for n in others)}.")


def suggestion_column(k):
    """A suggestion not yet made: 'Suggestion 1'. Its formulation number
    stays in the workbook and the records; the table does not show it, so a
    new Round never reads like a sample's own numbers."""
    return f"Suggestion {k}"


# The key's "?" while the Round's suggestions are unmade. It lists no
# numbers (owner, v3 final cleanup): on the pilot a live Round 7 mapped its
# suggestions to F19–F21, the pilot's own numbers.
SUGGESTION_NUMBERS_HELP = (f"Suggestions get {FORMULATION} numbers when their results are "
                           "recorded.")


# A suggestion's column label when the distance rule says it explores
# (food_bo.FAR_FROM_TESTED); no label for the other two roles.
LESS_TRIED = "Less-tried area"


def less_tried_help(share_pct):
    return (f"{LESS_TRIED}: more than {share_pct} % of an ingredient's allowed range from every "
            f"{FORMULATION} made before this {ROUND_CAP}.")


AT_UPPER_LIMIT = "at upper limit"
AT_LOWER_LIMIT = "at lower limit"
TOTAL_ROW = "Total"


def total_row(lo_text=None, hi_text=None, unit=""):
    """'Total (allowed 97.69–98.69 g)' with a limit on the total; 'Total
    (set to 100.00 g)' with Set up's Total of each formulation; else
    'Total (g)'."""
    u = f" {unit}" if unit else ""
    if lo_text is not None and lo_text == hi_text:
        return f"{TOTAL_ROW} (set to {lo_text}{u})"
    if lo_text is not None and hi_text is not None:
        return f"{TOTAL_ROW} (allowed {lo_text}–{hi_text}{u})"
    if hi_text is not None:
        return f"{TOTAL_ROW} (allowed up to {hi_text}{u})"
    if lo_text is not None:
        return f"{TOTAL_ROW} (allowed from {lo_text}{u})"
    return f"{TOTAL_ROW} ({unit})" if unit else TOTAL_ROW


def compare_row_header(unit=None):
    """The rows' header: 'Ingredient (g)'; 'Ingredient or process setting'
    when the rows do not share one unit (each row then carries its own)."""
    return f"Ingredient ({unit})" if unit else "Ingredient or process setting"


# The Round table's key (owner, v3 final cleanup; it replaced the bracketed
# change and the grey shading): 'More than F10 · Less than F10 · Plain:
# changed under 5 % of its range', the first two in the direction colours.
def compare_key_more(no):
    return f"More than {short_no(no)}"


def compare_key_less(no):
    return f"Less than {short_no(no)}"


def compare_key_plain(pct):
    """N15: a plain cell moved less than `pct` % of its row's allowed range
    (Lowest–Highest) from Best so far (ui_batch.direction_of)."""
    return f"Plain: changed under {pct:g} % of its range"


def compare_key_help(no, pct, row="ingredient"):
    """The key's "?": the rule behind the colours."""
    return (f"Orange: more than {short_no(no)} by at least {pct:g} % of that {row}'s allowed "
            "range (Lowest–Highest); purple: less by as much.")


def compared_with_line(column, cell):
    """The same line on paper: 'Compared with Formulation 2: Close to the
    best · Pea protein +0.28 g'. The table writes the column header once, at
    the top, and every cell under it is read against that; a sheet carries
    one formulation and nothing else, so the line has to say what it is
    compared with itself."""
    return f"{column}: {cell}"


def compared_with_cell(kind, changes=""):
    """'Explores · Water +12.00 g, Wheat gluten −3.00 g'.

    The kind leads because it is the answer to the question the reader asks
    first — is this a small step or a new direction? — and the amounts that
    follow say which ingredients carry it. During the cold start there is
    nothing to compare with, so the kind stands alone."""
    head = kind[:1].upper() + kind[1:]
    return f"{head} · {changes}" if changes else head


def near_copy_prefix(no, names_text):
    """'Nearly the same as Formulation 20, still waiting for Cook loss. ' —
    prefixed onto a suggestion's Compared with cell (Minor 15, 2026-09-29
    review) when it sits within NEAR_COPY_SPAN of a formulation the model
    has not been told the result of yet. Screen only: the trailing space
    joins straight onto the cell's own text, which is never rewritten."""
    return f"Nearly the same as {FORMULATION_CAP} {no}, still {waiting_for(names_text)}. "


NEEDS_ONE_UNIT = (f"To make each {FORMULATION} to a {BATCH_SIZE_NOUN}, "
                  f"every {INGREDIENT} needs the same unit.")


# The round screen's own box: the weight of one formulation in the round in
# front of the bench. Set up's box (FORMULATION_TOTAL_NAME) answers the same
# question for every round still to come, and the two names say which is
# which — "Default" is the only word between them.
def batch_size_label(unit):
    """'Batch size (g)' — the round screen's box. The unit is the one the
    ingredients share."""
    return f"{BATCH_SIZE_NAME} ({unit})" if unit else BATCH_SIZE_NAME


TOTAL_LABEL = "Total"


def total_column(unit):
    """'Total (g)' — the header of the round table's last column, and of the
    same row on the sheets. A bare 'Total' where the ingredients are in more
    than one unit, because the cell then carries the units itself."""
    return f"{TOTAL_LABEL} ({unit})" if unit else TOTAL_LABEL


def batch_size_help(unit=""):
    """The help under the round screen's own box. It ties the box to the
    column that reads it back: they are one number, and the table under the
    box is the one place the reader can see it has landed. "The sheets scale
    with it" said neither, in the one word that shipped meaning two things
    last cycle."""
    return (f"Every {FORMULATION} in this {ROUND_CAP} adds up to this. Change "
            "it and the amounts are recalculated at that size; the "
            f"table's {total_column(unit)} shows it.")
# The same box's "?" while it is empty: nothing adds up to an empty box.
BATCH_SIZE_EMPTY_HELP = f"Leave empty to keep each {FORMULATION}'s own total."
# An example, not a description: the help above already says what the box is
# for.
BATCH_SIZE_PLACEHOLDER = "e.g. 100"

# Under the Batch size box while it is empty (S14): the box is optional,
# and the bare "This Round has no batch size of its own." left a new reader
# unsure whether they had to fill it.
# "scale" is a verb here (resize), in the owner's words; the vocabulary
# guard lets this one sentence through.
NO_BATCH_SIZE_OF_ITS_OWN = "Optional: scale every formulation to one batch size."


def total_mismatch_caption(no, made_text, total_text):
    """'Formulation 4 adds up to 97.00 g, not the 100 g batch size.'

    One sentence for the row that can miss the batch size: a suggestion that
    could not be moved onto it without breaking a limit. Nothing is
    rewritten to hide it, so the line says what the row adds up to and what
    it was measured against. Once the user scales the round every row lands
    on the size and the line has nothing to say."""
    return (f"{FORMULATION_CAP} {no} adds up to {made_text}, not the "
            f"{total_text} {BATCH_SIZE_NOUN}.")


def sheets_show_total_caption(total_text):
    """'Sheets show each formulation made to 150 g.' — the one line on the
    tab that names the number the files were written for."""
    return f"Sheets show each {FORMULATION} made to {total_text}."


GENERATE_DIFFERENT_BATCH = f"Discard this {ROUND_CAP} and generate again"
# After the amounts of an open Round's formulations are edited in place.
# A status line: ten words at most (owner's wording rule).
FORMULATIONS_SAVED_NEW_WORKBOOK = f"{FORMULATION_CAP}s saved. Download a new workbook."


def regenerate_warning(no, numbers_text, next_no, many=True):
    """'Discard Batch 1 and Formulations 1, 2 and 3? New formulations start
    at Formulation 4.'

    Every other confirmation in the app asks a question, and "those numbers
    will not be used again" is a release note: what the reader can act on is
    where the numbering picks up. Not "the next batch": a regenerate keeps
    this batch's own number, so only the formulation numbers move on.
    """
    word = f"{FORMULATION_CAP}s" if many else FORMULATION_CAP
    return (f"Discard {ROUND_CAP} {no} and {word} {numbers_text}? New "
            f"{FORMULATION}s start at {FORMULATION_CAP} {next_no}.")


YES_DISCARD = "Yes, discard"

RECORDED_FALLBACK = "Recorded"


# What a row is missing, named. "· partial" told the reader a word rather
# than which measurement nobody took, on a screen that had room for the
# name.
NOT_MEASURED = "not measured"


def waiting_for(names_text=""):
    """'waiting for Cook loss' — a formulation's Overall score while a
    measurement that applies to it has no value. Distinct from a Not scored
    row, which waits to be scored at all. With nothing to name (every
    measurement it was measured on has since been deleted), 'waiting for
    results'."""
    return f"waiting for {names_text or 'results'}"


# A measurement that does not apply to a formulation — added after it was
# recorded, or scored without — in the All formulations table and the
# closeness details. Never "not measured": nothing was expected.
DOES_NOT_APPLY = "—"


def not_measured_tail(names_text):
    """' · Juiciness not measured' — `names_text` is one name or a
    number_list of several."""
    return f" · {names_text} {NOT_MEASURED}"


def recorded_line_partial(line, names_text):
    tail = not_measured_tail(names_text)
    return f"{line}{tail}" if line else tail.lstrip(" ·").lstrip()


def recorded_line_note(line, note):
    return f"{line} · Note: {note}" if line else f"Note: {note}"


def formulation_heading(no):
    return f"**{FORMULATION_CAP} {no}**"


def suggestion_heading(k):
    """Make a round's record grid, for an app suggestion not yet made: the
    table's own name, '**Suggestion 1**' (suggestion_column)."""
    return f"**{suggestion_column(k)}**"


def nothing_to_save(n):
    """Every open row is ticked Not scored, so Save results has nothing to
    record. The count is the reader's own, and the way out is the tick."""
    if n == 1:
        return (f"This {FORMULATION} is ticked {NOT_SCORED}. Untick it and "
                "enter results to save.")
    return (f"All {n} {FORMULATION}s are ticked {NOT_SCORED}. Untick one "
            "and enter results to save.")
# Said by food_bo when a row arrives with nothing on it, and by the Results
# tab when a not-scored formulation is saved with every box empty. One
# sentence, so the refusal reads the same wherever the row was typed.
ENTER_A_MEASUREMENT = "Enter at least one result."
SAVE_RESULTS = "Save results"


def complete_counter(complete_n, kept_n):
    """'1 of 2 complete' — a row is complete when EVERY measurement on it has
    a number. One of three typed is not a third of a result.

    `kept_n` is every row of the batch, ticked ones included: "1 of 1
    complete · 1 not scored" made the batch look bigger than it was, and a
    reader who counted the sheets in their hand got a different answer."""
    return f"{complete_n} of {kept_n} complete"


def not_scored_counter_suffix(n):
    return f" · {n} not scored"


def partly_filled_suffix(n):
    """The rows with some measurements on them but not all. They are counted
    apart from the complete ones: folding them in told the user the batch was
    further along than it was."""
    return f" · {n} partly filled"


def could_not_save(e):
    return f"Could not save these results: {e}"


def not_scored_with_note(note):
    return f"{NOT_SCORED} · {note}"


def note_reason(note):
    """The reason out of a stored note. A row left not scored carries
    "Not scored · burner failed": the marker the screen puts in front, and
    the reason the technician typed behind it. Scoring the row later reopens
    the reason in a Note box — the marker is about to stop being true, and
    the reason is what the row has always said about itself."""
    text = str(note or "").strip()
    prefix = not_scored_with_note("")
    if text.startswith(prefix):
        return text[len(prefix):].strip()
    return "" if text == NOT_SCORED else text


def batch_recorded_flash(no):
    return f"{ROUND_CAP} {no} recorded."


UPLOAD_EXPANDER = "Or upload results from a file"
UPLOAD_HELP_CAPTION = "Upload the workbook with Results filled in."
UPLOAD_HELP_DETAIL = (
    "Enter results on the Results sheet of the workbook downloaded above, "
    "then upload it here. Formulations are matched on their number."
)
# The same door, said the other way round, once the workbook has actually
# been downloaded in this session: the fold is open and this is the line
# inside it, so the reader coming back with a filled-in file finds the door
# already ajar rather than collapsed under a heading beginning "Or".
UPLOAD_SHEETS_ARE_BACK_CAPTION = "Upload the completed workbook here."
UPLOAD_SHEETS_ARE_BACK_HELP = ("Upload the completed workbook here. The app imports entries "
                               "from the input cells.")
UPLOAD_RESULTS_FILE = "Upload results (Excel or CSV)"
DOWNLOAD_ROUND6_RESULTS = "Download the pilot's Round 6 results"
CHECK_THIS_FILE = "Check this file"
def file_unreadable(name):
    """'Could not read Round 3.xlsx. Upload the workbook downloaded above.'
    — an upload on Make a round that is not a workbook or a CSV at all."""
    return (f"Could not read {name or 'this file'}. Upload the workbook "
            "downloaded above.")


def upload_found_caption(found_n, total_n, names_text):
    return (f"Found results for {found_n} of {total_n} formulations: "
            f"{names_text}.")


SAVE_UPLOADED_RESULTS = "Save uploaded results"
# The table drawn above it, so the reader checks the numbers the app read
# rather than a count of them: a firmness of 74 written for 7.4 passed the
# count without anybody seeing it.
UPLOAD_PREVIEW_CAPTION = "Check the imported values before saving."


def upload_partial_flash(parsed_n, total_n, batch_no, left_n):
    return (f"Recorded {parsed_n} of {total_n} formulations in "
            f"{ROUND_CAP} {batch_no} · {left_n} to record.")


# ------------------------------------------------------------------ #
# Tab 1 · Set up: ingredients and process settings, measurements,
# limits, advanced (ui_setup.py).
# ------------------------------------------------------------------ #
KIND_INGREDIENT = "Ingredient"
KIND_SETTING = "Process setting"


def setting_unit_hint(rows):
    """N16: under the ingredients grid, for rows typed Ingredient whose Unit
    is a time or a temperature. `rows` is [(name, unit)]. One row: 'Mixing
    time is in min: set Type to Process setting?'. Nothing is changed
    for the reader; it only asks."""
    if len(rows) == 1:
        name, unit = rows[0]
        return f"{name} is in {unit}: set Type to {KIND_SETTING}?"
    names = ", ".join(name for name, _ in rows)
    return f"{names}: time or temperature units. Set Type to {KIND_SETTING}?"

GOAL_LABELS = {
    "max": "Prefer higher values",
    "min": "Prefer lower values",
    "target": "Target value",
}


def target_value(value):
    """'Target 6' — the Goal column's cell text for a target measurement."""
    return f"Target {float(value):g}"


def range_text(low, high):
    """'0 to 10' — the Range column's cell text, before its unit."""
    return f"{float(low):g} to {float(high):g}"


LIMIT_KEPT = (f"Formulations already made are kept. The next {ROUND_CAP} will "
             "respect this limit.")
NO_FORMULATION_FITS_LIMIT = "No formulation you have made fits this limit."


def unscaled_tail(batch_no):
    """A batch size needs one unit, and a unit change may have just taken it
    away: the round keeps the amounts it already has, but nothing on the
    round screen can change their size any more, and only this sentence says
    so. Nothing is undone — since 0.5.0 the size moves the amounts
    themselves, so there is no as-generated to go back to, and the old size
    is not worth naming: it is not a number anything can be typed back to."""
    return (f"{ROUND_CAP} {batch_no} keeps the amounts it has: your "
            "ingredients no longer share one unit.")

# The one collapsed expander that says how the app works, in the app's own
# words. It was written as the one place the specialist vocabulary was
# spoken — variable, objective, weight, constraint — and says none of those
# four words any more: the concepts are named as the screens name them
# (ingredients and settings, measurements and goals, shares, limits).
# There is therefore no glossary bridge anywhere in the app, which is a
# deliberate choice and not an oversight.
HOW_IT_WORKS = [
    "Ingredients and settings are what the app varies. Measurements "
    "and goals are what it aims for.",
    "Share of score says how much each measurement counts, out of 100. "
    "Closeness says how near a result is to its goal, from 0 to 1.",
    HOW_CHOSEN_2,
    "Generated formulations satisfy the limits you set. A "
    "formulation of your own is recorded as you typed it.",
    # v3: Make a round shows each change from Best so far; the one role
    # word left on screen is Less-tried area.
    f"Each suggestion shows its change from the best so far. One far from "
    f"every {FORMULATION} made is marked {LESS_TRIED}.",
]

# The fold directly under it, for the reader who wants the arithmetic. The
# bullets above raise the question — what is closeness, exactly? — and this
# answers it; nine bullets in one fold answered it before anyone asked.
HOW_CLOSENESS_HEADING = "**How closeness is calculated**"
HOW_CLOSENESS = [
    "Prefer higher values: closeness = (measured − lowest) ÷ (highest − lowest), "
    "so the top of your range scores 1 and the bottom scores 0.",
    "Prefer lower values: the reverse — the bottom of your range scores 1 and the "
    "top scores 0.",
    "Target value: closeness is 1 at the target and falls evenly with "
    "distance, by one point per full range; the lowest score depends on how "
    "far the target sits from the ends of your range. Because of that "
    "floor, a target measurement can affect the score less than its "
    "share suggests.",
    "The app learns the one overall score, so changing a share, a goal "
    "or a range re-scores every past formulation.",
    "A formulation of your own counts like any other. Making the best one "
    "again teaches the app how much your measurements vary.",
]

VARIABLES_HEADER = "Ingredients and process settings"


def made_before_units_caption(unit):
    return (f"These amounts were recorded before units existed, so they "
            f"are read as {unit}. Set the right unit below.")


UPLOAD_INGREDIENTS_EXPANDER = "Import ingredients"

NAME_LABEL = "Name"
TYPE_LABEL = "Type"
VARIABLE_TYPE_HELP = ("Ingredients are weighed into the formulation and "
                      "count towards its total. Process settings, such as "
                      "temperature or time, are set on the equipment.")
LOWEST_LABEL = "Lowest"
HIGHEST_LABEL = "Highest"
UNIT_LABEL = "Unit"
BASELINE_LABEL = "Baseline"
# The column arrives on its own, in the grid the reader sets their rules
# in, the moment the first round is recorded — so its one tooltip has to
# say what the number is as well as what it is for.
BASELINE_HELP = ("Only for a process setting added after results exist: enter the value "
                 "used in all earlier formulations. This is not an ingredient amount.")
def saved(name):
    """Subject first, like added() above it and every other flash on the
    tab."""
    return f"{name} saved."
# "0 if blank" and "leave a cell empty for no value" sat on one screen
# contradicting each other, and the app never knows an ingredient is
# fat-free — only that a cell was left empty.
PROPERTY_BLANK_RULE = "Blank means unknown. For a property used in a limit, fill every ingredient’s value; enter 0 only if zero."
ADD_BASELINE_ERROR = (f"Enter a {BASELINE_LABEL}: the value used in "
                      "formulations already made.")


def added(name):
    """Subject first, like every other flash on the tab: "Firmness updated.",
    "Batch 1 recorded." One shape for all of them."""
    return f"{name} added."


# The Set-up sheet's own column. The screen has no Status column any more:
# a row pinned at one amount says so in its Lowest and its Highest, which is
# where the reader already looks. The sheet keeps one, because a sheet is
# read away from the app and its two number columns are for re-importing.
STATUS_LABEL = "Status"


def fixed_status(amount_text):
    """'fixed at 0.00 g' — what the Set-up sheet writes beside a row whose
    Lowest is its Highest."""
    return f"Fixed at {amount_text}"


INGREDIENT_OR_SETTING_LABEL = "Ingredient or process setting"

# ------------------------------------------------------------------ #
# 0.5.0 · the two editable grids on tab 1.
#
# There is no add form, no control row and no per-row editor any more: a
# row is typed where it is read, and one Save changes writes the lot. So
# the words here are column headers and one pair of buttons, and the
# sentences that used to belong to six separate controls are gone with
# them.
# ------------------------------------------------------------------ #
VENDOR_LABEL = "Vendor"
SKU_LABEL = "SKU"
VENDOR_HELP = 'Supplier name, included in the workbook for reference. It does not affect suggestions.'
SKU_HELP = 'Supplier product code, included beside the vendor in the workbook. It does not affect suggestions.'
# What each measurement is worth out of 100. It moved up here from the
# measurements block below because the grid's own header is built from it,
# and a module reads top to bottom.
COL_SHARE = "Share of score"
SHARE_COLUMN = f"{COL_SHARE} (%)"
SHARE_HELP = 'This measurement’s contribution to the overall score. Changing one share adjusts the others to total 100%.'
SHARES_REBALANCED_CAPTION = "Shares adjusted to add up to 100 %."


def shares_scaled(pairs_text):
    """'Shares scaled to add up to 100 %: Crunch 66.7, Sweetness 33.3.' —
    every share typed, none adding to 100, so all of them moved: the line
    names what each one is now."""
    return f"Shares scaled to add up to 100 %: {pairs_text}."


def share_adjusted_to(name, share_text):
    """'Juiciness adjusted to 30 %' — one row that gave way."""
    return f"{name} adjusted to {share_text}"


def shares_rebalanced(named):
    """'Juiciness adjusted to 30 % so the shares add up to 100 %.' — the
    rows that gave way, named. "Shares adjusted to add up to 100 %" beside
    "Firmness saved." left the reader to find out which of the others had
    moved, in a toast that fades."""
    return f"{named} so the shares add up to 100 %."

INGREDIENT_GRID_CAPTION = "Set each lowest and highest. Equal values fix it."
INGREDIENT_GRID_HELP = (
    "Enter the lowest and highest amounts or settings to explore. "
    "Use the same number in both cells to keep a value fixed.")


DISCARD_CHANGES_BUTTON = "Discard changes"
PROPERTY_FIGURES_SET_ASIDE = (
    "Your unsaved property figures were set aside because the ingredients "
    "changed.")


def unsaved_grid_caption(heading):
    """'Ingredients and process settings — not saved yet.'

    Tab 1 has two grids, each with a banner, and one Save does not reach the
    other: a banner that named neither read as the tab's, so saving one
    table looked like saving both. The heading is the grid's own, word for
    word, so the line names something the reader can point at."""
    return f"{heading} — not saved yet."


# `row_error` — 'Row 3: Lowest cannot be above Highest.' — lives further
# down, beside the import that first needed it. Tab 1's grids say the same
# sentence about a row that cannot be saved, in the same shape.


def delete_rows_warning(names_text):
    """The question Save asks before it applies a deleted row. Spec 1.1's
    sentence, with the app's own standing promise about the copy: every
    other Delete in the app says where the copy went, and this one is no
    less permanent for arriving through a grid."""
    return (f"Delete {names_text}? Later formulations keep their numbers. "
            + COPY_KEPT)


def only_a_setting_has(column):
    """A cell filled in on a row that cannot carry it. Refused rather than
    quietly dropped: a number typed into Baseline is an answer, and throwing
    it away without a word is how a grid loses an edit. Both fixes are
    named, as actions on the two cells the reader can reach."""
    return f"Clear {column}, or change {TYPE_LABEL} to {KIND_SETTING}."


def only_an_ingredient_has(column):
    if column == MADE_AS_LABEL:
        # Preparation is a select: its "empty" is the Single ingredient
        # choice, which is what the reader has to pick.
        return (f"Set {column} to {PREMIX_MADE_AS_BOUGHT_IN}, or change "
                f"{TYPE_LABEL} to {KIND_INGREDIENT}.")
    return f"Clear {column}, or change {TYPE_LABEL} to {KIND_INGREDIENT}."


NAME_REQUIRED_ERROR = "Enter a name."
NUMBER_REQUIRED_ERROR = "Enter a number."
SHARE_REQUIRED_ERROR = f"Enter a {SHARE_COLUMN} above 0."
TYPE_LOCKED_ERROR = ("Type cannot change once results are recorded. "
                     "Delete the row and add it again.")


def _cell_text(text, limit=24):
    """What a cell held, as a refusal quotes it back: trimmed, and cut short
    so one long paste does not become the whole sentence."""
    text = str(text).strip()
    return text if len(text) <= limit else text[:limit - 1].rstrip() + "…"


def cells_need_numbers(blank=(), typed=()):
    """'Enter Lowest and Highest.' / 'Enter a number in Lowest, not “lots”.'
    — a grid row refused for a number it does not have, naming the CELL.
    "Enter a number." under a row of five number cells named none of them
    (owner, 2026-10). `blank` are the empty cells' column names; `typed` is
    [(column, text)] for cells holding something that is not a number,
    which is said first: it is the one the reader typed."""
    if typed:
        column, text = typed[0]
        return f"Enter a number in {column}, not “{_cell_text(text)}”."
    return f"Enter {_and_join(blank)}."


def type_locked(kind):
    """'Type stays Ingredient once results are recorded. …' — the type the
    row has, named, so the reader knows what to put back."""
    return (f"{TYPE_LABEL} stays {kind} once results are recorded. "
            "Delete the row and add it again.")


def share_not_above_zero(value):
    """'Share of score (%) is 0; enter more than 0.' — the number typed,
    said back."""
    return f"{SHARE_COLUMN} is {float(value):g}; enter more than 0."


NAME_ON_TWO_ROWS = "Another row has this name; rename one of them."


def name_on_two_rows(name):
    """'Butter is on two rows; rename or delete one.' — two rows of the
    ingredients grid typed with one name, neither of them a saved row."""
    return f"{name} is on two rows; rename or delete one."


def formulations_contain_none_of(names_text):
    """Said once when a grid Save adds an ingredient to a project that has
    already recorded formulations: those formulations are encoded as
    containing none of it, whatever Lowest the new row carries."""
    return f"Formulations already made contain no {names_text}."


# The properties grid: one row per ingredient, one column per property, and
# the cell is that ingredient's figure for it (spec 1.5). The row column is
# the ingredient's name, which is why "Ingredient" is a name the project
# cannot also give to an ingredient or a property.
PROPERTIES_ROW_COLUMN = "Ingredient"
PROPERTIES_NAME = "Properties"


def properties_grid_caption(per_100_already_said=False):
    """'Each ingredient's figure, per 100 g. An empty cell counts as 0 in
    any limit.'

    `per_100_already_said` drops the basis from the sentence: a project whose
    property names carry it themselves ("Fat (g per 100 g)", "Cost per 100 g")
    would otherwise say it twice in one line.
    """
    basis = "" if per_100_already_said else ", per 100 g"
    return f"Each ingredient’s property value{basis}. " + PROPERTY_BLANK_RULE


SAVE_BUTTON = "Save"
SAVE_PROPERTIES_BUTTON = "Save changes"
CLOSE_BUTTON = "Close"
PROPERTIES_SAVED = "Properties saved."
DELETE_PROPERTY_PICK_LABEL = "Property to delete"
# The small popover under the properties grid that holds the picker and its
# Delete (S11): always on screen, they read as the grid's main action.
DELETE_PROPERTY_POPOVER = "Delete a property"


def property_not_a_number(name):
    """One cell of the properties grid holding something that is not a
    number. The column is named: a grid of six properties gives the reader
    nothing else to go on."""
    return f"Enter a number in {name}, or leave it empty."


def no_such_ingredient(name):
    return f"No ingredient named {name}."


UNIT_REQUIRED_ERROR = f"Enter a {UNIT_LABEL}, such as g or ml."


def unit_changed(name, written, is_ingredient):
    """Nothing is converted and nothing is rescored, and only this sentence
    says so — so it asks for the one thing the reader can do about it."""
    if not written:
        return f"{name} is shown without a unit."
    recorded = "amounts" if is_ingredient else "numbers"
    return (f"{name} is now in {written}. The {recorded} you already recorded "
            "were not converted — check them.")


YES_DELETE = "Yes, delete"


def delete_button(name):
    return f"Delete {name}"


DELETE_VS_FIXING_CAPTION = ("Deleting takes it out of every formulation "
                            "already made; setting Lowest and Highest to the "
                            "same amount keeps the data.")
# The tick that deletes an ingredient formulations actually used. Its first
# clause is quoted back by the refusal that sends the reader to it
# (ingredient_was_used), so the two are spelled once.
DELETE_EVEN_IF_USED = "Delete even though formulations used it"
DELETE_EVEN_IF_USED_CHECKBOX = f"{DELETE_EVEN_IF_USED} — recorded amounts will also be deleted"


def deleted(name):
    return f"{name} deleted."


REPLACE_INGREDIENTS_BUTTON = "Replace ingredients"
LOAD_INGREDIENTS_BUTTON = "Load ingredients"
# The template this caption describes carries a Rule column, and a reader
# who downloaded it was told extra columns become properties — which that
# one does not (the name is reserved) — with nothing on screen to say what
# it is for.
INGREDIENTS_FILE_CAPTION = "Columns: Name, Lowest, Highest; others optional."
INGREDIENTS_FILE_HELP = ("A file with the columns Name, Lowest, Highest "
                            "and, optionally, Unit, Calculation, Part of, Preparation and Composition (%). Extra numeric columns "
                            "become properties you can set limits on.")
UPLOAD_INGREDIENTS_FILE_LABEL = "Upload ingredients (Excel or CSV)"


def blank_unit_cell_help(unit):
    return (f"Leave the {UNIT_LABEL} column blank and the app reads it "
            f"as {unit}.")


def ingredients_file_unreadable(file_name):
    """Set up's ingredient upload, refused by the file's own name."""
    return (f"{file_name} could not be opened; upload the ingredients "
            "template as .xlsx or .csv.")


FILE_UNREADABLE_RETRY = ("This file could not be read. Upload a spreadsheet "
                         "with one row per ingredient, and try again.")
FILE_ALREADY_LOADED_CAPTION = ("This file is already loaded. Choose another "
                               "to replace the ingredient list.")


def loaded(what):
    return f"Loaded {what}."


def property_limit_removed(metric):
    """A property limit is an average over the amounts, so it is the
    ingredients as a whole that stopped sharing a unit."""
    return (f"The limit on {metric} was deleted because the ingredients no "
            "longer share a unit.")


def no_longer_ingredients(names_text, many):
    return (f"{names_text} are no longer ingredients" if many
            else f"{names_text} is no longer an ingredient")


def quantity_limit_removed_missing(label, who):
    return f"The limit on {label} was deleted because {who}."


def quantity_limit_removed_unit_mismatch(label):
    return (f"The limit on {label} was deleted because those ingredients "
            "no longer share a unit.")


ALL_INGREDIENTS_LABEL = "All ingredients"
ALL_INGREDIENTS_LOWER = "all ingredients"

GOAL_LABEL = "Goal"
TARGET_LABEL = "Target"
LOWEST_MEASURABLE_LABEL = "Scale minimum"
HIGHEST_MEASURABLE_LABEL = "Scale maximum"
MEASUREMENT_EXISTS_ERROR = 'That measurement already exists. Edit its row in Measurements and targets to change it.'


def name_differs_only_by_case(stored):
    """Two rows whose names differ only in capitals are two rows with one
    name on every table in the app, and the CSV importer matches columns
    without regard to case, so the second could never be filled in."""
    return (f"{stored} already exists. Use that spelling to change it, or "
            "choose another name.")
SAVE_CHANGES_BUTTON = "Save changes"

RECALCULATED_SUFFIX = " Every overall score was recalculated."


def measurement_deleted(name):
    return f"{name} deleted."


def typed_results_deleted(names_text, n, round_no, many=False):
    """05 N1: what a measurement deleted on Set up takes from the open
    Round. Every other result typed there is kept."""
    what = "1 result" if n == 1 else f"{n} results"
    verb = "was" if n == 1 else "were"
    return (f"{what} typed for {names_text} on {ROUND_CAP} {round_no} {verb} "
            f"deleted with {'them' if many else 'it'}. The other typed "
            "results are kept.")


MEASUREMENTS_HEADER = "Measurements and targets"
MEASUREMENT_COLUMN = "Measurement"
RANGE_COLUMN = "Range"


def delete_measurement_warning(names_text, many=False):
    """The question a grid Save asks before it drops a measurement. It takes
    a list, because a grid can take two out at once, and says "them" when it
    is one."""
    return (f"Delete {names_text}? Every overall score is recalculated "
            f"without {'them' if many else 'it'}. " + COPY_KEPT)


# Where the targets came from: an optional free-text note under the
# measurements table. One label serves both the button that opens the box and
# the box itself, so the reader sees the same words twice rather than a
# button and a form asking two different questions.
# It opens a box to write in, so it says so. "Where the targets come from"
# read as an explanation the app was about to give.
TARGETS_SOURCE_BUTTON = "Edit this note"
ADD_TARGETS_SOURCE_BUTTON = "Add where the targets come from"
TARGETS_SOURCE_LABEL = "Where the targets come from"
TARGETS_SOURCE_PLACEHOLDER = "e.g. Benchmark burger, panel of 8"


def targets_from_caption(text):
    """'Targets from: Benchmark burger, panel of 8.' shown under the
    measurements table once a targets_source is set."""
    return f"{TARGETS_SOURCE_LABEL}: {text}"


HOW_IT_WORKS_HEADING = "**How it works**"

ADD_PROPERTY_LABEL = "New property"
ADD_PROPERTY_PLACEHOLDER = "e.g. Sodium mg per 100 g"
ADD_PROPERTY_BUTTON = "Add property"


def property_added(name):
    """The grid that gives it a figure is on screen with a new column for it
    the moment this lands, so the sentence points at the grid rather than
    back up the tab."""
    return f"{name} added. Enter its value for each ingredient."


def delete_property_warning(name, limits_text):
    head = (f"Delete {name} and its {limits_text}? " if limits_text
            else f"Delete {name}? ")
    return head + "Its saved values for each ingredient will also be deleted. " + COPY_KEPT


def property_deleted(name, gone_text=""):
    return f"{name} deleted.{gone_text}"


def limit_went_with_it(limits_text):
    return f" Deleted with it: {limits_text}."


FINISHED_PRODUCT_LIMIT_NAME = "Finished-product limit"
FINISHED_PRODUCT_LIMIT_HEADING = f"**{FINISHED_PRODUCT_LIMIT_NAME}**"


def per_100_caption(unit):
    return (f"Per 100 {unit} of formulation, from the properties of your "
            "ingredients.")


# The heading above it is the owner's word — Finished-product limit — and
# the grid the choices come from is headed Properties. The picker asks for
# one of those, so it is named for them.
INGREDIENT_PROPERTY_LABEL = "Property"
# Said in place of the property picker when the project has named none. The
# fold that names one is directly below this one, so the sentence points down.
NO_PROPERTIES_YET_CAPTION = "Add one under Ingredient properties below."
AT_LEAST_LABEL = "At least"
AT_MOST_LABEL = "At most"


def per_100_box_label(label, unit):
    """'At least (per 100 g)' — the property limit's own boxes. The
    ingredient limit's boxes say (g); these said nothing at all, while the
    caption above them said the basis once."""
    return f"{label} (per 100 {unit})"
NO_LIMIT_PLACEHOLDER = "No limit"
ADD_PROPERTY_LIMIT_BUTTON = "Add property limit"
# The PROPERTY limit form asks for At least and At most, so its refusal
# names those two boxes. The ingredient limit form has a Kind picker and one
# box, and says LIMIT_NUMBER_NEEDED_ERROR instead.
ENTER_LOWEST_HIGHEST_ERROR = (f"Enter a number in {AT_LEAST_LABEL}, "
                              f"{AT_MOST_LABEL}, or both.")
LIMIT_BOUNDS_ORDER_ERROR = f"{AT_LEAST_LABEL} must be less than {AT_MOST_LABEL}."


def limit_bounds_order(low, high):
    """'At least 30 is above At most 20; swap them.' — the two boxes, in
    their numbers. Equal is refused too: a limit of exactly one number is
    a range of nothing to search."""
    if float(low) == float(high):
        return (f"{AT_LEAST_LABEL} and {AT_MOST_LABEL} are both "
                f"{float(low):g}; make {AT_MOST_LABEL} larger.")
    return (f"{AT_LEAST_LABEL} {float(low):g} is above {AT_MOST_LABEL} "
            f"{float(high):g}; swap them.")


def limit_added_on(who):
    return f"Limit added on {who}."


def limit_gap_tail(name, many):
    """' · Water has no figure for it and counts as 0.' — the ingredients a
    limit is silently reading as zeroes."""
    return (f" · Missing values: {name}. Complete these before generating formulations." if many
            else f" · Missing value: {name}. Complete it before generating formulations.")


# The two tiers tab 1 folds everything optional into (spec 1.5). More
# settings holds what a project may want once — the default batch size,
# where the targets came from, the limits and the properties — and Advanced
# holds what a specialist wants at most once: the model settings and the two
# explanations. Both are collapsed, so the tab reads as its two grids.
# MORE_SETTINGS_EXPANDER is defined above SAMPLE_PROVENANCE_HELP, its first user.
ADVANCED_EXPANDER = "Advanced"

# The property rule lives here, not in the closeness fold: a property never
# touches closeness — it feeds limits and nothing else.
# The blank-figure rule is NOT repeated here. It used to be, because the
# properties were a fold somewhere else on the tab; they are now a grid in
# the fold directly below this one, under a caption that says it, and each
# limit's own line names the ingredients it is reading as zeroes.
LIMITS_CAPTION = "Generated formulations meet these limits."
LIMITS_HELP = ("Generated formulations satisfy the limits listed here. "
                  "A formulation of your own is recorded as you "
                  "typed it. A limit is on an amount you weigh out or a "
                  "property of your ingredients; measurements have goals and "
                  "targets instead.")


def old_limit_basis_caption(unit):
    return f"An older limit is now read per 100 {unit} of formulation."


def _bound_number(value):
    """A number already written as text (food_bo.limit_number) as it is;
    any other in the 'g' form."""
    return value if isinstance(value, str) else f"{value:g}"


def at_least(value, unit=""):
    text = f"at least {_bound_number(value)}"
    return f"{text} {unit}".rstrip() if unit else text


def at_most(value, unit=""):
    text = f"at most {_bound_number(value)}"
    return f"{text} {unit}".rstrip() if unit else text


# Each limit's Delete sits beside the line that names the limit, and its
# question names it again; the button naming every ingredient of a long
# limit wrapped over four lines (S12).
DELETE_THIS_LIMIT = f"Delete this {LIMIT}"


def delete_limit_warning(who):
    return (f"Delete the {LIMIT} on {who}? The next {ROUND_CAP} no longer has "
            f"to obey it. " + COPY_KEPT)


def limit_deleted(who):
    return f"Limit on {who} deleted. This limit will not apply to future suggestions."


LIMIT_ON_CHOSEN_INGREDIENTS_HEADING = "**Limit on chosen ingredients**"
INGREDIENTS_TO_LIMIT_LABEL = "Ingredients to limit together"
ADD_INGREDIENT_LIMIT_BUTTON = "Add ingredient limit"

# ---------------------------------------------------------------- #
#  Batch size, and the project's default
# ---------------------------------------------------------------- #
# The weight of ONE formulation. The round screen asks it of the round in
# front of the bench (Batch size); Set up holds the project's answer for
# every round still to come (Default batch size). Two names for two
# questions, both spelled from BATCH_SIZE_NOUN so they can never drift.


FORMULATION_TOTAL_NOUN = f"default {BATCH_SIZE_NOUN}"
FORMULATION_TOTAL_NAME = f"Default {BATCH_SIZE_NOUN}"
FORMULATION_TOTAL_LOWER = f"the {FORMULATION_TOTAL_NOUN}"
# What batch_discarded_notice blames when the default is what discarded the
# round, rather than the tab as a whole.
TOTAL_CHANGED_REASON = f"{FORMULATION_TOTAL_LOWER} changed"


def formulation_total_label(unit):
    """'Default batch size (g)'. The unit is the one the ingredients share;
    without one there is no size to ask for and no box is drawn."""
    return (f"{FORMULATION_TOTAL_NAME} ({unit})" if unit
            else FORMULATION_TOTAL_NAME)


FORMULATION_TOTAL_HELP = (f"Every {FORMULATION} adds up to this unless a "
                          f"{ROUND_CAP} sets its own {BATCH_SIZE_NOUN}.")
# An example, not a description: the help above already says what the box is
# for, and 100 g is what the sample ships with.
FORMULATION_TOTAL_PLACEHOLDER = "e.g. 100"

def total_still_holds(total_text):
    """'Each formulation still adds up to 100 g.' — the half-sentence an edit to
    the ingredient list adds to its own success line.

    The total's limit is over every ingredient, so it is rewritten on every
    such edit; this is the screen saying so. It was rewritten silently, and
    a reader who had just been told "Limits are never crossed" had no way to
    know whether the calculation they typed had survived their own step 2."""
    return f"Each {FORMULATION} still adds up to {total_text}."


def formulation_total_row(total_text):
    """'Default batch size · 100 g (set in Set up)' — the one line the
    default takes on the printed Set-up sheet.

    On paper there is no box above it and no caption beside it, so a row
    under Limits read as a limit the user had written. It is off the screen
    entirely now — the box at the top of Preparation and records is where the number
    is answered — so the sheet says where it came from."""
    return f"{FORMULATION_TOTAL_NAME} · {total_text} (set in Set up)"


def round_stays_at(size_text):
    """The second half of a refused batch size: what the round, the table and
    the sheets are still made to, so the number left in the box is never
    mistaken for the one under it."""
    return f"The {ROUND_CAP} stays at {size_text}."


def _stays(stays_text):
    """' The Round stays at 100 g.', or nothing for a Round made as
    generated, which has no size to name."""
    return " " + round_stays_at(stays_text) if stays_text else ""


def batch_size_too_big(most_text, stays_text):
    """Make a round's Batch size box, above what the allowed amounts can
    add up to: the box, the number to type, why, and what the Round is
    still made to. Set up's Default batch size box keeps its own words
    (total_not_reachable_at_most)."""
    return (f"{BATCH_SIZE_NAME}: enter {most_text} or less, the most these "
            f"{INGREDIENT}s make." + _stays(stays_text))


def batch_size_past_limits(most_text, stays_text):
    """The same box, inside the allowed amounts but above what the Limits
    let them add up to."""
    return (f"{BATCH_SIZE_NAME}: enter {most_text} or less, the most your "
            "limits allow." + _stays(stays_text))


def batch_size_under_limits(least_text, stays_text):
    return (f"{BATCH_SIZE_NAME}: enter {least_text} or more, the least your "
            "limits allow." + _stays(stays_text))


def batch_size_too_small(least_text, stays_text):
    """The same box, below what the Lowest amounts already add up to."""
    return (f"{BATCH_SIZE_NAME}: enter {least_text} or more, the least these "
            f"{INGREDIENT}s make." + _stays(stays_text))


def batch_size_leaves_rest_below_zero(name, least_text, unit, stays_text):
    """The same box with a Fill to total row: too small a size leaves that
    row with less than nothing."""
    zero = f"0 {unit}".strip()
    return (f"{BATCH_SIZE_NAME}: enter {least_text} or more, or {name} goes "
            f"below {zero}." + _stays(stays_text))


def total_not_reachable_at_most(total_text, most_text,
                                noun=FORMULATION_TOTAL_NOUN):
    """A size above everything the allowed amounts can add up to. The two
    numbers are the whole answer: nothing about the search can rescue a sum
    that has no solution, and the fix is to raise an ingredient's Highest.

    `noun` is which of the two boxes is refusing — Set up's default, or the
    round screen's own — so the sentence names the box the reader just
    typed into rather than the other one."""
    return (f"These {INGREDIENT}s make at most {most_text}; enter "
            f"{most_text} or less, or raise a {HIGHEST_LABEL}.")


def total_not_reachable_at_least(total_text, least_text,
                                 noun=FORMULATION_TOTAL_NOUN):
    return (f"These {INGREDIENT}s make at least {least_text}; enter "
            f"{least_text} or more, or lower a {LOWEST_LABEL}.")


def limits_allow_at_most(most_text):
    """Set up's default batch size, inside the allowed amounts but above
    what the Limits let them add up to ('Butter + Sugar at most 45 g')."""
    return (f"Your limits allow at most {most_text}; enter {most_text} or "
            "less, or change a limit.")


def limits_allow_at_least(least_text):
    return (f"Your limits need at least {least_text}; enter {least_text} or "
            "more, or change a limit.")


def total_not_reachable_at_all(total_text, noun=FORMULATION_TOTAL_NOUN):
    """A size of nothing. Reachable arithmetic — every amount can be 0 in a
    project with no lower bounds — and still not a formulation, so it is
    refused in the same shape as a size the amounts cannot make."""
    return f"Enter a {noun} above 0."


def no_formulation_reaches_total(total_text):
    """Generate found nothing that adds up to the default batch size. It is
    that size that is impossible, so the sentence names it and the two ways
    out, rather than talking about limits the user never wrote."""
    return (f"No formulation adds up to {total_text} within the allowed "
            f"amounts. Change {FORMULATION_TOTAL_LOWER} or widen the "
            "amounts.")


def round_uses_the_default_size(no, size_text, previous_no, previous_text):
    """'Round 2 uses the default batch size, 100 g. Round 1 was 120 g.'

    A size typed for one round belongs to that round; the next starts at the
    project's default. The cold reader set 120 g, made Round 1 at it, and
    found Round 2 quietly back at 100 with four numbers on two tabs and
    nothing saying which was which."""
    return (f"{ROUND_CAP} {no} uses the default {BATCH_SIZE_NOUN}, "
            f"{size_text}. {ROUND_CAP} {previous_no} was {previous_text}.")


def fixed_amounts_do_not_add_up(made_text, total_text):
    """Every ingredient is fixed at one amount and those amounts make the
    wrong weight. Nothing can move, so the sentence is the two numbers and
    the two things in Set up that can change."""
    return (f"Fixed amounts add up to {made_text}, not {total_text}. Change "
            f"one, or the {FORMULATION_TOTAL_NOUN}, in Set up.")


def fixing_breaks_the_total(total_text):
    """A row whose Lowest is its Highest is one amount, and one amount can
    put the batch size out of reach of the rows still moving. The two ways
    back are the size and the allowed values — not the eight ingredients the size's
    own limit happens to name."""
    return (f"Fixing these leaves no formulation at {total_text}; change "
            f"the {BATCH_SIZE_NOUN} or let more {INGREDIENT}s vary.")


def formulation_total_gone_unit(total_text):
    """The sentence a unit change owes the default batch size when it has
    just split the ingredients across units — the same debt unscaled_tail
    settles for the open round, one sentence, said once."""
    return (f"The {FORMULATION_TOTAL_NOUN} of {total_text} is gone: your "
            "ingredients no longer share one unit.")


def formulation_total_gone_unreachable(total_text):
    """...and the same when the ingredient list or its allowed amounts moved
    far enough that the sum can no longer land on that size."""
    return (f"The {FORMULATION_TOTAL_NOUN} of {total_text} is gone: the "
            "allowed amounts no longer add up to it.")


HOW_FORMULATIONS_CHOSEN_HEADING = "**How formulations are chosen**"
STANDARD_VS_EXPERT_CAPTION = (
    "Recommended settings suit most projects. Custom controls are for statistical modeling.")

HOW_FORMULATIONS_CHOSEN_LABEL = "How formulations are chosen"
STANDARD_DEFAULT_OPTION = "Recommended settings"
EXPERT_SELECTED_OPTION = "Custom model settings"
REVERT_TO_STANDARD_BUTTON = "Revert to standard settings"
USING_DEFAULT_MODEL_SETTINGS = "Using default model settings."
KERNEL_LABEL = "Kernel"
LENGTHSCALE_PRIOR_LABEL = "Lengthscale prior"
NOISE_LABEL = "Noise"
ACQUISITION_LABEL = "Acquisition"
# The four option lists. They are the specialist's own vocabulary rather than
# prose, but they are still words on screen, and food_bo.validate_bo_config
# checks a stored config against these same lists — one list each, read from
# here, so a name added to a box cannot be refused by the loader.
KERNEL_OPTIONS = ["matern52", "matern32", "rbf", "linear", "poly2"]
LENGTHSCALE_PRIOR_OPTIONS = ["default", "long", "short"]
NOISE_OPTIONS = ["default", "low", "fixed_tiny"]
ACQUISITION_OPTIONS = ["qlognei", "qlogei", "qucb"]
# No backticks: the dropdown beside this line shows the same words in plain
# text, and a caption is not a code block.
FIXED_TINY_NOISE_CAPTION = 'Use fixed_tiny only for deterministic measurements. For sensory panels, keep default unless your study requires another setting.'
APPLY_EXPERT_SETTINGS_BUTTON = "Apply expert settings"
MODEL_SETTINGS_UPDATED = "Model settings updated."
PASTE_EXPERT_SETTINGS_CHECKBOX = "Or paste expert settings as JSON"
EXPERT_SETTINGS_JSON_LABEL = "Expert settings JSON"
APPLY_PASTED_SETTINGS_BUTTON = "Apply pasted settings"


def invalid_json(e):
    """The pasted expert settings, refused where they went wrong. The
    parser's own words ("Expecting value: line 1 column 2 (char 1)") were
    the whole message before."""
    line, column = getattr(e, "lineno", None), getattr(e, "colno", None)
    if line is not None and column is not None:
        return (f"The settings stop making sense at line {line}, character "
                f"{column}; check the quotes and commas there.")
    return "Paste the settings in braces, like the example in the box."


IN_USE_PREFIX = "In use: "


# ------------------------------------------------------------------ #
# Tab 3 · Results: the best formulation, every formulation, corrections
# (ui_results.py).
# ------------------------------------------------------------------ #
# v3: the last Round's results, one row per formulation, and how one
# formulation's Overall score is worked out (final-v3 wording, 3 · Results).
def round_results_heading(no):
    """'Round 6 results'."""
    return f"{ROUND_CAP} {no} results"


def _scale_text(low, high):
    return f"{float(low):g}–{float(high):g}"


def round_results_caption(scale=None, panel_size=None):
    """Under the Round's table, before its colour key: 'Panel median (n =
    5), 0–15.' `scale`: (low, high) when every measurement shares one, else
    None (each header then names its own). `panel_size`: the panel's n, only
    when the project's provenance records how a rating was made (the median
    of a panel, sample_projects.panel_size); without it nothing is claimed
    about how a value was arrived at. '' with neither."""
    if panel_size:
        head = f"Panel median (n = {panel_size})"
        head += f", {_scale_text(*scale)}" if scale else ""
        return f"{head}."
    if scale:
        return f"Range {_scale_text(*scale)}."
    return ""


# The colour key under Round N results and Review's strip (owner, v3 final
# cleanup): a rating off its target shows its direction, above in orange,
# below in purple (ui_insights.direction_colour).
ABOVE_TARGET = "Above target"
BELOW_TARGET = "Below target"


def round_results_column(name, goal, target=None, unit="", scale=None):
    """A measurement's header in every Results table (N7: one format, the
    one the entry boxes use): 'Firmness (/10) · Target 6'; 'Juiciness ·
    Higher is better'. `unit` in the brackets, '/10' included; `scale`
    (low, high) only when measurements differ in scale and the unit does
    not say it ('/10')."""
    said = []
    if scale and not str(unit).startswith("/"):     # '/10' says the scale itself
        said.append(_scale_text(*scale))
    if unit:
        said.append(str(unit))
    head = f"{name} ({', '.join(said)})" if said else name
    if goal == "target" and target is not None:
        return f"{head} · {target_value(target)}"
    if goal == "max":
        return f"{head} · {SCORE_GOAL_HIGHER}"
    if goal == "min":
        return f"{head} · {SCORE_GOAL_LOWER}"
    return head


NEW_BEST = "new best"
NEW_BEST_TIED = "new best (tied)"


def round_results_row_label(short, best=False, tied=False):
    """'F16 · new best (tied)', 'F18'."""
    if not best:
        return short
    return f"{short} · {NEW_BEST_TIED if tied else NEW_BEST}"


# The fold under the table: one name (owner tweak to the final-v3 mock).
SCORE_WORKED_OUT_FOLD = "How the score is worked out"
SCORE_WORKED_OUT_PICK = FORMULATION_CAP


def score_point_line(span=None):
    """'Each unit off costs 1/15 of its share.' — `span` is the width of the
    measurements' scale (Scale maximum - Scale minimum) when every
    measurement shares one, else None. 'unit', not 'point': a point is what
    the score is counted in (S7)."""
    if span is None:
        return (f"Each unit off costs 1/({HIGHEST_MEASURABLE_LABEL} − "
                f"{LOWEST_MEASURABLE_LABEL}) of its share.")
    return f"Each unit off costs 1/{float(span):g} of its share."


# The fold's columns in Set up's own words (S7): a measurement, what was
# measured, its goal. "Attribute" and "Rating" were a tasting panel's words,
# and Cook loss is neither.
SCORE_ATTRIBUTE_COLUMN = MEASUREMENT_COLUMN
SCORE_RATING_COLUMN = "Measured"
SCORE_TARGET_COLUMN = "Goal"
# "Share of score", the Set up column it comes from (a bare "Share" is
# banned vocabulary; the final-v3 mock's "Share").
SCORE_SHARE_COLUMN = "Share of score"
SCORE_POINTS_COLUMN = "Points earned"
# The Goal cell of a measurement that is not a target: what scores 100 is
# its scale's end, and the cell says which way is better.
SCORE_GOAL_HIGHER = "Higher is better"
SCORE_GOAL_LOWER = "Lower is better"


# 0.8.0: a formulation teaches the suggestions only once every measurement
# that applies to it has a value. A dash in its row is a measurement it is
# scored without — added after it was recorded, or imported without it.
def complete_results_caption(compared=True):
    """What a dash means. `compared` is False while no formulation has
    every measurement (a measurement was just added): Best so far is then
    itself one with a dash, and saying dashed rows are not compared would
    contradict the star it sits under (0.8.0 review fixes, Task 1)."""
    if compared:
        return "“—”: scored without that measurement."
    return "“—”: scored without that measurement, as is Best so far."


def complete_results_help(compared=True):
    """complete_results_caption's full sentence, now its tooltip."""
    tail = (", so it is not compared for Best so far." if compared
            else "; until a formulation has every measurement, Best so far "
                 "is scored without it too.")
    return ("A formulation counts once all its results are in; “—” means it "
            "is scored without that measurement" + tail)


NO_COMPLETE_FORMULATION = "No formulation has all its results yet."


# 'Overall score' as a noun inside a sentence, and the scale it is on. The
# ceiling is a whole number — a formulation that hits every goal scores
# 100 — so it is written as one.
OVERALL_SCORE_LOWER = "overall score"


def batch_recorded_progress(no, before, now):
    """'Round 2 recorded · Best overall score 86.6 → 88.1'.

    The number had no name and no scale, beside a sibling line that did
    carry the noun. Both say the same thing now, in the same words."""
    return (f"{ROUND_CAP} {no} recorded · Best {OVERALL_SCORE_LOWER} "
            f"{score_shown(before)} → {score_shown(now)}")


def batch_recorded_no_improvement(no):
    return (f"{ROUND_CAP} {no} recorded · Best {OVERALL_SCORE_LOWER} "
            "unchanged.")


MEASURED_COLUMN = "Measured"
OFF_BY_COLUMN = "Off by"

# Directly under the best-so-far block: the one way back to Set up from
# Results, for a measurement that needs its range widened or an ingredient
# that needs a new limit once a formulation is on screen.
CHANGE_SETUP_FROM_RESULTS_BUTTON = "Change measurements or ingredients"


def change_setup_from_results(settings_only=False):
    """Results' way to Set up; a project of process settings alone names
    them (final review M-9, review I14)."""
    return ("Change measurements or settings" if settings_only
            else CHANGE_SETUP_FROM_RESULTS_BUTTON)
AMOUNTS_TO_MAKE_IT_HEADING = "**Amounts to make it**"


def amounts_to_make_it_heading(total_text=""):
    """'**Amounts to make it (150 g)**' when the batch this formulation was
    made in was printed to a total, and the plain heading otherwise. The
    stored amounts are always as generated, so a heading that did not say
    which of the two numbers was on screen showed a formulation nobody made."""
    return (f"**Amounts to make it ({total_text})**" if total_text
            else AMOUNTS_TO_MAKE_IT_HEADING)
AMOUNT_COLUMN = "Amount"
NOT_USED_PREFIX = "Not used: "


def scored_without_tail(names_text):
    """' · scored without Cook loss' — `names_text` is one name or a
    number_list of several; '' when there is nothing to name."""
    return f" · scored without {names_text}" if names_text else ""


ALL_FORMULATIONS_HEADING = "**All formulations**"
SORT_LABEL = "Sort"
# The three options, in the order the box offers them. Each is also matched
# exactly in food_bo.history_frame(order=...) — a protocol between that
# module and this one, not display prose that happens to repeat — so both
# read them from here and the two can never drift apart.
SORT_BEST_FIRST = "Best first"
SORT_NEWEST_FIRST = "Newest first"
SORT_BATCH_ORDER = f"{ROUND_CAP} order"
SORT_OPTIONS = [SORT_BEST_FIRST, SORT_NEWEST_FIRST, SORT_BATCH_ORDER]
SHOW_AMOUNTS_TOGGLE = "Show amounts"
DOWNLOAD_ALL_FORMULATIONS_BUTTON = "Download all formulations (Excel)"
DOWNLOAD_ALL_FORMULATIONS_HELP = (
    "Includes formulations, measurements and project settings, plus any saved lot numbers "
    "or additional records. Formulations marked Not scored are included."
)


CORRECT_A_FORMULATION_HEADING = f"##### Correct a {FORMULATION}"
# The same control, doing the other of its two jobs. A not-scored row has no
# result to correct — this writes its first one — and one heading over both
# made the reader work out which had happened.
SCORE_A_FORMULATION_HEADING = f"##### Score a {FORMULATION}"
CORRECT_WHICH_LABEL = f"{FORMULATION_CAP} to correct"
CHOOSE_A_FORMULATION_PLACEHOLDER = f"Choose a {FORMULATION}"


def no_formulation_to_correct_caption():
    return f"No {FORMULATION} to correct yet."


# The picker offers every number the project holds. A not-scored one has no
# result to correct — it has one to write for the first time — and this says
# so, because nothing about the box itself does.
NOT_SCORED_CAN_BE_SCORED_CAPTION = "Not-scored formulations can be scored here."
# One line above the whole form. The boxes open pre-filled, so "leave blank"
# described a state the reader was not in — and two tooltips said one thing
# two ways on one screen.
CORRECTION_CAPTION = ("Change only what is wrong. Anything you leave alone "
                      "stays as recorded.")
SAVE_CORRECTION_BUTTON = "Save correction"
SAVE_RESULT_BUTTON = "Save result"


def _waiting_tail(waiting):
    return "" if waiting is None else f" · {waiting_for(waiting)}"


def formulation_scored(no, waiting=None):
    """A formulation that was left not scored, scored later from Results. It
    is not a correction: nothing recorded changed, a result arrived.
    `waiting` names what it still waits for (None: nothing)."""
    return f"{FORMULATION_CAP} {no} scored{_waiting_tail(waiting)}."


def formulation_unchanged(no):
    return f"{FORMULATION_CAP} {no} is unchanged."


def formulation_corrected(no, waiting=None):
    """One sentence for the whole correction. A row can now change its
    amounts and its measurements in one save, and naming every number that
    moved made a flash longer than the table it described."""
    return f"{FORMULATION_CAP} {no} corrected{_waiting_tail(waiting)}."


def formulation_corrected_not_scored(no, names_text):
    """A correction to a formulation moved to Not scored that still lacks a
    value: 'Formulation 7 corrected · Not scored · Cook loss not measured.'
    — not "waiting for", which it no longer is (final review M-3)."""
    return f"{FORMULATION_CAP} {no} corrected · {not_scored_missing(names_text)}."


def saved_waiting(numbers, names_text):
    """'Formulation 12 saved · waiting for Cook loss.', or 'Formulations
    7–12 saved · …' — after a round is saved with formulations that still
    wait for the same results: one line for them all."""
    return f"{formulations_span(numbers)} saved{_waiting_tail(names_text)}."


def overall_score_waiting(names_text):
    """The score line of a formulation still waiting: 'Overall score
    waiting for Cook loss.' — never the lowered number."""
    return f"{OVERALL_SCORE_COLUMN} {waiting_for(names_text)}."


# 0.8.0 review fixes (§9 B2): a recorded formulation whose result will
# never come is moved to Not scored in place. It keeps its row, its Round
# and its partial values; entering the value scores it again.
def not_scored_missing(names_text):
    """'Not scored · Cook loss not measured' — a recorded formulation moved
    to Not scored: its Overall score cell and its line on Results. With
    nothing to name (every measurement it was measured on deleted), plain
    'Not scored'."""
    return f"{NOT_SCORED}{not_measured_tail(names_text)}" if names_text else NOT_SCORED


def saved_not_scored(no, names_text):
    """A value typed for a formulation moved to Not scored that still lacks
    another: 'Formulation 4 saved · Not scored · Cook loss not measured.'
    It is not scored yet, so the flash must not say it is."""
    return f"{FORMULATION_CAP} {no} saved · {not_scored_missing(names_text)}."


def save_missing(names_text):
    """The lit button under a formulation that waits: 'Save Cook loss'."""
    return f"Save {names_text}"


MOVE_TO_NOT_SCORED = f"Move to {NOT_SCORED}"
WAIT_AGAIN = "Wait for its results again"


def moved_to_not_scored(no):
    return f"{FORMULATION_CAP} {no} moved to {NOT_SCORED}."


def waiting_again(no, names_text):
    return f"{FORMULATION_CAP} {no} is {waiting_for(names_text)} again."


def formulation_has_all_results(no):
    """set_aside_formulation's refusal: nothing is missing."""
    return f"{FORMULATION_CAP} {no} already has all its results."


START_NEXT_BATCH = f"Start the next {ROUND_CAP}"

NO_RESULTS_YET = "No results yet."
OVERALL_SCORE_COLUMN = "Overall score"
BEST_SO_FAR_COLUMN = "Best so far"
# C12: the All formulations sheet's date column. "Recorded" was a state
# everywhere else on the screen ("Round 1 · recorded", "2 to record") and
# a date only here.
DATE_RECORDED_COLUMN = "Date recorded"


def batch_open_record_first_caption():
    """food_bo.undo_last_batch's refusal. No screen reaches it any more —
    the Delete the last batch button retired with this section — but the
    method stays for its tests and it still has to speak English."""
    return f"Record or discard the open {ROUND_CAP} first."


DELETE_FORMULATIONS_HEADING = f"##### Delete {FORMULATION}s"
FORMULATIONS_TO_DELETE_LABEL = f"{FORMULATION_CAP}s to delete"
CHOOSE_MANY_PLACEHOLDER = "Choose one or more"
# The quick pick beside the list: one batch's formulations, recorded and not
# made alike, dropped into the selection to be looked over before deleting.
# "Add a whole round to the list" read as adding a round to the project,
# under a heading about deleting them. What the picker does is choose
# one for the list of things about to go.
WHOLE_BATCH_LABEL = f"Select a whole {ROUND_CAP} for deletion"
CHOOSE_A_BATCH_PLACEHOLDER = f"Choose a {ROUND_CAP}"


def no_formulation_to_delete_caption():
    return f"No {FORMULATION} to delete yet."


def delete_formulation_button(no):
    return f"Delete {FORMULATION_CAP} {no}"


def delete_formulations_button(numbers_text, n):
    """'Delete Formulations 2 and 3' up to three, 'Delete 5 formulations'
    above that. "Delete 2 formulations" and the single row's "Delete
    Formulation 2" are one keystroke apart in meaning, so the button prints
    the numbers while they still fit on it."""
    if n > 3:
        return f"Delete {n} {FORMULATION}s"
    return f"Delete {FORMULATION_CAP}s {numbers_text}"


def delete_formulation_warning(no):
    return (f"Delete {FORMULATION_CAP} {no}? Later formulations keep their "
            "numbers. " + COPY_KEPT)


def delete_formulations_warning(numbers_text):
    return (f"Delete {FORMULATION_CAP}s {numbers_text}? Later formulations "
            "keep their numbers. " + COPY_KEPT)


def formulation_deleted(no):
    return f"{FORMULATION_CAP} {no} deleted. " + COPY_KEPT


def formulations_deleted(numbers_text):
    return f"{FORMULATION_CAP}s {numbers_text} deleted. " + COPY_KEPT


# Record, not Add: Add puts a formulation into the open batch, to be made.
# This one was made already, and the two doors sat one tab apart wearing the
# same verb.
# As the fold holding it is named ("Correct, delete or add past formulations").
ADD_PAST_FORMULATION_LABEL = f"Add a past {FORMULATION}"
ADD_PAST_FORMULATION_HEADING = "##### " + ADD_PAST_FORMULATION_LABEL
TYPE_IT_IN = "Type it in"
UPLOAD_A_FILE = "Upload a file"
# What a formulation made before this project existed is noted as, and what
# the Note box opens holding. food_bo.import_formulation defaults to the same
# word, so a row typed in and a row read off a file read alike.
IMPORTED_NOTE = "Made earlier"
ADD_THIS_FORMULATION = f"Record this {FORMULATION}"


def formulation_added(no):
    """'Formulation 6 recorded.' — the button says Record, and until now
    the line it flashed said added. One door, one verb."""
    return f"{FORMULATION_CAP} {no} recorded."


EXTRA_COLUMNS_IGNORED = " Extra columns are ignored."


def import_columns_caption(names_text):
    return (f"One row per {FORMULATION} you already made. The columns must "
            f"match these names exactly: {names_text}."
            + EXTRA_COLUMNS_IGNORED)


def import_columns_caption_empty():
    return (f"One row per {FORMULATION} you already made. Add ingredients "
            "and measurements first; the columns must match their names "
            "exactly.")


UPLOAD_FORMULATIONS_FILE_LABEL = f"Upload {FORMULATION}s (Excel or CSV)"


def missing_columns(names_text, n=1):
    """Results' Add past formulations file without a column the project
    needs. The same sentence as an uploaded results file."""
    return sheet_missing_columns(names_text, n)


def blank_amount_columns(names_text):
    """Blank amount cells in an Add past formulations file."""
    return f"Fill the blank cells in {names_text}; type 0 for none."


def past_file_unreadable(name):
    """An Add past formulations upload that is not a spreadsheet at all."""
    return (f"Could not read {name or 'this file'}. Upload a spreadsheet with one row per "
            f"{FORMULATION}.")


IMPORT_ALL_ROWS_BUTTON = "Record all rows"


def row_error(position, problem):
    """'Row 3: Lowest cannot be above Highest.' — one line per row that
    cannot be read or cannot be saved. Two places say it: the formulation
    import on tab 3, and 0.5.0's editable grids on tab 1. The number is the
    row's own, the one on screen beside it, so the reader looks in one
    place."""
    return f"Row {position}: {problem}"


def named_row_error(name, problem):
    """'Salt: this calculation could not be read.' — the ingredients grid has no
    row numbers down its left edge (Name is its first column), so a refusal
    about one of its rows names the row the way every success line already
    does. `name` is NEW_GRID_ROW for a row typed on the empty line at the
    bottom that has no name yet.

    A sentence that already opens with the row's name ("Salt is already
    the name of an ingredient.") is said as it is: "Salt: Salt is …" said
    the name twice."""
    if name and str(problem).startswith((f"{name} ", f"{name}'s ")):
        return problem
    return f"{name}: {_after_colon(problem)}"


# After "Butter:" an instruction reads on in lower case, as every other
# named refusal does ("Formulation 2: enter an amount for Protein."); a
# column name keeps its capital ("Butter: enter Lowest and Highest.").
_LOWERED_VERBS = ("Enter", "Take", "Choose", "Type", "Use", "Make", "Set",
                  "Delete", "Add", "Pick", "Give", "Raise", "Lower",
                  "Swap", "Change", "Shorten", "Start", "Fill", "Open", "Move",
                  "Leave", "Keep", "Save", "Clear")


def _after_colon(problem):
    text = str(problem)
    first = text.split(" ", 1)[0]
    return text[0].lower() + text[1:] if first in _LOWERED_VERBS else text


NEW_GRID_ROW = "The new row"


def stopped_at_row(row_no, failure):
    return f"Stopped at row {row_no}: {failure}"


def rows_before_saved(rows_text):
    """The tail on a part-way refusal: one door, one verb, so it says
    recorded like the line it rides on."""
    return f" The {rows_text} before it were recorded."


def imported(text):
    """'Recorded 3 formulations.' — the same verb the typed-in half uses.
    The radio option above it still says Upload a file, because that names
    the mechanism rather than the result."""
    return f"Recorded {text}."


def rows_with_nothing_measured(rows_text, many):
    """The tail on the import flash. A not-scored formulation is in the
    downloaded file — it has a number, its amounts and its note — but it has
    no result to teach the model, so it is left where it is rather than
    stopping the whole import."""
    return (f" {rows_text} had no measurements, so they were not recorded."
            if many
            else f" {rows_text} had no measurements, so it was not recorded.")


SET_UP_THIS_PROJECT_BUTTON = "Set up this project"
MAKE_YOUR_FIRST_BATCH_BUTTON = "Generate your first formulations"
ADD_MEASUREMENT_RESCORE_INFO = ("Add a measurement in Set up to score these "
                                "formulations again. Nothing recorded has "
                                "been lost.")


# ------------------------------------------------------------------ #
# The workbook. One Excel file carries the batch to the bench: a summary
# sheet the whole batch is weighed out from, and one sheet per
# formulation to carry, tick and write on. Every word the kitchen reads
# off the paper is here — the sheet is the one screen the app cannot see
# being used, so it says exactly what the screen says.
# ------------------------------------------------------------------ #
DOWNLOAD_BATCH_SHEETS = f"Download the {ROUND_CAP} sheets (Excel)"

# The summary sheet's own columns. The amount column carries its unit
# (`Amount (g)`), built by food_bo.label_with_unit from AMOUNT_COLUMN.
TICK_COLUMN = "Tick"
PERCENT_COLUMN = f"% of {BATCH_SIZE_NOUN}"

SETTINGS_SHEET_HEADING = "Settings"


def settings_step_note(steps_text):
    """'Settings are set to the nearest 5 s.' — what the dial can actually
    be set to, said under the block that asks for it.

    A mixing time printed as 109.04 s asks the bench to round it, and three
    benches round three ways."""
    return f"Settings are set to the nearest {steps_text}."
MEASUREMENTS_SHEET_HEADING = "Measurements"
LIMITS_SHEET_HEADING = "Limits"
# The finished-product limits, on the printed Set-up sheet. On screen they
# sit under their own heading with a caption saying what they are per; on
# paper they were rows in a Limits block that also holds the amount limits
# and the default batch size, with nothing saying which was which.
PROPERTY_LIMITS_SHEET_HEADING = f"{FINISHED_PRODUCT_LIMIT_NAME}s"
SHEET_NONE = "None"
# The same separator the screens use between a measurement and its goal
# ("Firmness (N) · target 6 N"). It was a comma on the sheet alone, which
# read as a third measurement in a list of two.
SHEET_GOAL_SEPARATOR = " · "
# A box to tick with a pen, not a run of typed underscores. The box alone
# fills the Tick column's cells, which had a header and nothing under it.
TICK_BOX = "☐"
# The label of the Not scored row, and the box that goes in the cell BESIDE
# it. The box used to be printed into the locked label, under an
# instruction to mark it — so the one cell the reader was told to write in
# was the one cell the sheet would not take a mark in.
NOT_SCORED_CHECKBOX_SHEET = NOT_SCORED

# The one line above a sheet's measurements block. On paper there is nothing
# to hover and nobody to ask, so the sheet says what mark it will read: the
# cold read put a cross in the Not scored row without knowing the app would
# take it.
SHEET_WRITE_IN_NOTE = ("Write what you measured. If you did not score it, "
                       f"tick the {NOT_SCORED} box.")

# The title row of the All formulations sheet. The amounts in it are the
# ones the project RECORDED — as generated — which are not always the ones a
# batch sheet printed, and a table of numbers with nothing saying which is
# a table nobody can weigh anything out from.
RECORDED_AMOUNTS = "Recorded amounts"
# What a not-scored row carries in that sheet's own tick column.
TICKED_BOX = "☒"
# Who made it and when. A sheet comes back from the bench days later and
# is filed; without these two blanks nothing on the page says whose work
# it was.
MADE_BY_FOOTER = "Made by ____ on ____"

# The one line under the summary's Note row. The tick and the numbers can
# both be filled in on one column, and only one of them can be true.
SUMMARY_TICK_NOTE = (f"A ticked {NOT_SCORED} box wins over numbers typed in "
                     f"that column.")

# S6: the Lot is one per ingredient for the whole round, so it lives on the
# round's own sheet. A page carried to the bench says where it is rather
# than leaving the reader to find out that this page has no such column.
def lots_are_on_the_round_sheet(batch_no):
    """'Lots: on the Round 1 sheet.'"""
    return f"{LOTS_SHEET}: on the {batch_sheet_name(batch_no)} sheet."

ALL_FORMULATIONS_SHEET = f"All {FORMULATION}s"
SET_UP_SHEET = "Set up"
INGREDIENTS_SHEET = "Ingredients"


def summary_title(batch_no, project_name, made_on, total_text=""):
    """'Batch 2 · Sample project · 2026-09-14' — the first line of the
    summary sheet. A sheet printed and carried to a bench says which batch
    of which project it is and when it was asked for; without the date, two
    printouts of the same batch number cannot be told apart.

    `total_text` puts the size on the page too: the app's own caption said
    "Sheets show each formulation made to 100 g" and that sentence was
    nowhere on the sheet the bench carried."""
    line = (f"{batch_sheet_name(batch_no)} · {project_name} · "
            f"Printed {made_on}")
    return f"{line} · Batch size {total_text}" if total_text else line


def batch_sheet_name(batch_no):
    """'Batch 2' — the summary sheet's name, and the sheet an uploaded
    workbook is read back from."""
    return f"{ROUND_CAP} {batch_no}"


def formulation_sheet_name(no):
    """'Formulation 4' — one formulation's own sheet, and the summary
    sheet's column header for it."""
    return f"{FORMULATION_CAP} {no}"


def sheet_title(no, batch_no, project_name):
    """'Formulation 4 · Batch 2 · Sample project' — the first line of one
    formulation's sheet. The project's name is on it because the sheet
    leaves the app and the bench works on more than one."""
    return f"{FORMULATION_CAP} {no} · {ROUND_CAP} {batch_no} · {project_name}"


def sheet_measurement_label(label, goal):
    """'Firmness (N) · target 6 N' — a measurement and what a good number
    looks like, as one row label on the summary sheet. An uploaded sheet
    is matched back on this exact text, so it is written once."""
    return f"{label}{SHEET_GOAL_SEPARATOR}{goal}"


def workbook_note_without_numbers(no):
    """A column with a note in it and nothing else. It is not an untouched
    formulation — somebody wrote on it — and it is not a result either, so
    it is refused rather than quietly dropped with the note it carries."""
    return (f"{FORMULATION_CAP} {no} has a note but no numbers. Tick "
            f"{NOT_SCORED} to record it, or fill in the numbers.")


def workbook_measurement_missing(name, sheet_name):
    """A measurement row whose label is gone from the sheet, with nothing
    written where the app put it. Counting rows from there would read the
    line below as this measurement's result; the sheet is refused instead,
    naming the one thing to put back."""
    return (f"{name} was not found on the {sheet_name} sheet. Keep the row "
            "labels the app wrote.")


def workbook_file_name(project_name, batch_no):
    """'Sample project · Batch 2.xlsx' — what the download is called in
    the Downloads folder, a month later, beside eleven others."""
    # A hyphen, not the app's own `·`: the browser writes the file to disk
    # with the dot dropped, and `Sample project  Round 1.xlsx` came out with
    # two spaces in the middle of it.
    return f"{project_name} - {batch_sheet_name(batch_no)}.xlsx"


def all_formulations_file_name(project_name):
    return f"{project_name} - {ALL_FORMULATIONS_SHEET.lower()}.xlsx"


INGREDIENTS_TEMPLATE_FILE_NAME = "ingredients_template.xlsx"

WORKBOOK_UNREADABLE = ("Could not read this file as a workbook. Upload the "
                       f"one downloaded for this {ROUND_CAP}.")


def workbook_unreadable(file_name):
    """'oops.xlsx is not a workbook. Upload the one downloaded for this
    Round.' — the uploaded file named; the old sentence when it has none."""
    if not file_name:
        return WORKBOOK_UNREADABLE
    return (f"{file_name} is not a workbook. Upload the one downloaded for "
            f"this {ROUND_CAP}.")


def workbook_sheet_missing(wanted):
    """The sheet named after the open batch is not in the uploaded file —
    usually last week's workbook, downloaded twice. It names the sheet it
    looked for and which workbook to upload instead."""
    return (f"This workbook has no {wanted} sheet. Upload the workbook "
            f"downloaded for {wanted}.")


def workbook_no_formulations(wanted):
    return (f"The {wanted} sheet has no formulation columns. Upload the "
            f"file you downloaded from this {ROUND_CAP}.")


def workbook_nothing_filled_in(sheet):
    """'Nothing is filled in on the Results sheet. Enter a result, or tick
    Not scored.' — `sheet` is the sheet of the uploaded file the results
    are written on: Results, or Round 2 in a printed pack."""
    return (f"Nothing is filled in on the {sheet} sheet. Enter a result, "
            f"or tick {NOT_SCORED}.")


def limit_fixed_rows_break(what, names_text, many=False):
    """A limit refused at the door it is written at, because the rows it
    reads are pinned at one amount: no formulation could ever meet it, and
    letting it in would leave Generate handing back rows the app itself
    calls invalid."""
    return (f"No formulation can meet a limit on {what} while "
            f"{names_text} {'are' if many else 'is'} fixed at one amount. "
            f"Change the limit, or give {'them' if many else 'it'} a "
            f"different {LOWEST_LABEL} and {HIGHEST_LABEL}.")


def fixed_rows_tail(names_text):
    """The tail on a refusal from a grid Save: which rows, pinned at one
    amount, are what makes the limit impossible. The refusal itself names
    the limit; without this the reader had to work out which of eight rows
    they had just fixed."""
    return f"Fixed at one amount: {names_text}."


# ------------------------------------------------------------------ #
# The four words of a rule row. They are defined here, above the limits,
# because a limit's own refusal names the Rule cell as the other way to
# say the same thing; everything else about a rule is further down, under
# "Rule cells".
# ------------------------------------------------------------------ #
FORMULA_LABEL = "Calculation"
FORMULA_IN_RANGE = "calculation"
# The word for a row filled in from its rule rather than typed by hand.
# One word, spelled once, for the range cell's own marker and every sheet
# that has to say the same thing about the same row. It is NOT the app's
# verb for "recomputed" — that is "recalculated" — so the two never meet
# on one screen meaning two things.
WORKED_OUT = "calculated"
CALCULATED_RANGE = "Calculated"
RULE_GUIDE_LABEL = "Calculation help"
INGREDIENT_MIN_HELP = "Minimum amount the app may suggest. For a calculation-based ingredient, its calculation determines the amount."
INGREDIENT_MAX_HELP = "Maximum amount the app may suggest for an ingredient without a calculation. Calculated amounts follow their calculation."
MEASUREMENT_MIN_HELP = "Lower end of the measurement scale used to score and check results. This is not a target."
MEASUREMENT_MAX_HELP = "Upper end of the measurement scale used to score and check results. Widen the scale if a genuine result falls outside it; scores will be recalculated."
OLD_VARIABLE_BLEND_MODE = "ingredients varied separately"

# '= rest' on its own is the balance of the batch size once every other row
# is filled in. The word is spelled once here so the parser and its own
# refusal can never drift apart. On screen it is always "= rest": "the
# balance" is the instrument the bench weighs on.
REST_TOKEN = "rest"
# What the same idea is called wherever a reader meets it — the grid cell,
# the calculation editor, the printed sheet. `= rest` is the stored spelling
# and what an older file carries; this is the only word said out loud.
FILL_TO_TOTAL = "Fill to total"


# ------------------------------------------------------------------ #
# 0.5.0 wave 2, "rules" (task 5): a limit over several ingredients can say
# Exactly, and can be written in the ingredients' own unit or as a % of
# the default batch size. One idea, one control each: Exactly replaces At
# least/At most rather than sitting beside them, and the % choice is a
# Unit beside the plain one, offered only while there is a default batch
# size to be a percent OF.
# ------------------------------------------------------------------ #
# One control for one idea. The form used to sit At least, At most and
# Exactly side by side in three equal boxes and refuse the combinations
# afterwards, with a "fill one of these two" sentence printed under three
# boxes. The "Limit is" picker asks the question once and shows only the
# its answer needs, so the Exactly-and-a-range refusal has gone entirely
# and ENTER_LOWEST_HIGHEST_ERROR is the property form's alone.
LIMIT_KIND_LABEL = "Limit is"
LIMIT_KIND_AT_LEAST = "At least"
LIMIT_KIND_AT_MOST = "At most"
LIMIT_KIND_BETWEEN = "Between"
LIMIT_KIND_EXACTLY = "Exactly"
LIMIT_KINDS = [LIMIT_KIND_AT_LEAST, LIMIT_KIND_AT_MOST, LIMIT_KIND_BETWEEN,
               LIMIT_KIND_EXACTLY]
# `Unit` already heads two grid columns, where it means the unit an amount
# is weighed in. This control chooses how a limit is WRITTEN, which is not
# a unit at all.
LIMIT_WRITTEN_AS_LABEL = "Write it as"
LIMIT_NUMBER_NEEDED_ERROR = "Enter a number for this limit."
PERCENT_OF_BATCH_SIZE_UNIT = f"% of {FORMULATION_TOTAL_NOUN}"

EXACTLY_ONE_INGREDIENT = (
    f"For one ingredient, give the row the same {LOWEST_LABEL} and "
    f"{HIGHEST_LABEL} in the grid, or write it as a {FORMULA_IN_RANGE} "
    f"such as = 1.5 % of {BATCH_SIZE_NOUN}.")


def exactly_one_ingredient(name):
    """An Exactly limit on one ingredient: that is the row's own Lowest and
    Highest, and the sentence sends the reader to the row by name."""
    return (f"To fix {name} at one amount, give it equal {LOWEST_LABEL} "
            f"and {HIGHEST_LABEL} in the grid.")


def limit_number_needed(kind):
    """The ingredient-limit form's one empty box, named as the box is
    labelled (At least / At most / Exactly), or both boxes for Between."""
    if kind == LIMIT_KIND_BETWEEN:
        return ENTER_LOWEST_HIGHEST_ERROR
    return f"Enter a number in {kind}."


def limit_on_worked_out_rows(names_text, many=False):
    """'Water and Salt are calculated from their calculations, so this limit
    cannot change them.' — every row of a limit filled in by a calculation leaves
    the limit nothing to act on, and the app accepted it and then let the
    calculations contradict it."""
    verb, whose = ("are", "their") if many else ("is", "its")
    return (f"{names_text} {verb} {WORKED_OUT}; limit the {INGREDIENT}s "
            f"{whose} {FORMULA_IN_RANGE} uses instead.")


def exactly(value):
    """'exactly 50' — the bound word an Exactly limit reads as, alongside
    at_least and at_most."""
    return f"exactly {value:g}"


def limit_exactly_row(who, amount_text):
    """'Water + Oil: exactly 50 g' — an Exactly limit's read-back: the
    number the reader typed, not the band it is actually enforced as (a
    continuous search cannot be held to a point, so the file gives this
    the same band it gives the total of each formulation)."""
    return f"{who}: exactly {amount_text}"


def limit_percent_row(who, percent_text, grams_text, size_text=""):
    """'Water + Oil: at most 30 % of default batch size (30 g at the
    default 100 g)' — a percent limit's read-back, in both the percent it
    is written as and the grams it means, so the number the search actually
    enforces from is never hidden behind a percentage.

    "today" read as a date. The number it meant is the default batch size,
    so the line names it."""
    tail = (f"{grams_text} at the default {size_text}" if size_text
            else grams_text)
    return (f"{who}: {percent_text} of {FORMULATION_TOTAL_NOUN} ({tail})")


def premix_limit_row(name, low_pct, high_pct, grams_text, size_text=""):
    """'Dry blend is 30 to 40 % of the default batch size (30 to 40 g at
    the default 100 g)' — a weighed pre-mix's own percent limit: it reads
    'is', not the colon a plain ingredient limit uses, and 'to' rather than
    'at least … and at most …' — the group IS a share of the batch, not an
    ingredient with a floor and a ceiling of its own."""
    if low_pct is not None and high_pct is not None:
        percent_text = f"{range_text(low_pct, high_pct)} %"
    elif low_pct is not None:
        percent_text = f"{at_least(low_pct)} %"
    else:
        percent_text = f"{at_most(high_pct)} %"
    tail = (f"{grams_text} at the default {size_text}" if size_text
            else grams_text)
    return f"{name} is {percent_text} of the {FORMULATION_TOTAL_NOUN} ({tail})"


def percent_limits_rebased(size_text):
    """'Limits written as a % of the default batch size are now calculated
    from 120 g.' — the one line a new default batch size says once,
    however many percent limits it just rewrote."""
    return (f"Limits written as a % of the {FORMULATION_TOTAL_NOUN} are "
            f"now {WORKED_OUT} from {size_text}.")


def percent_limit_removed(who):
    """'The limit on Dry blend was deleted: it was a % of the default
    batch size, and there is none now.' — clearing the default takes every
    percent limit with it: there is nothing left for it to be a percent OF.

    Same verb, same position, same noun as its sibling below. The old
    sentence said what the limit WAS and stated an unrelated fact about the
    project, and never said the limit was gone."""
    return (f"The limit on {who} was deleted: it was a % of the "
            f"{FORMULATION_TOTAL_NOUN}, and there is none now.")


def quantity_limit_removed_percent_unreachable(label, size_text):
    """'The limit on Pea protein was deleted: a default batch size of
    120 g can no longer reach it.' — a percent limit a new default
    rewrote past what the ingredients can actually make."""
    return (f"The limit on {label} was deleted: a {FORMULATION_TOTAL_NOUN} "
            f"of {size_text} can no longer reach it.")


# ------------------------------------------------------------------ #
# 0.5.0 workbook. The file is locked where the app will not read it and
# open where it will: the cells a bench writes in are unlocked and
# shaded, and the sheet says which ones they are. Two of them are new —
# the Lot a round was weighed from, and the Actual weight when the
# balance did not land on the printed number.
# ------------------------------------------------------------------ #
# One lot per ingredient per round: the summary sheet has one cell for
# it beside the amounts, not one per formulation, because a round is
# weighed out of the sacks that are open that morning.
LOT_COLUMN = "Lot"
# What was really weighed, beside what was asked for. The column header
# carries the project's unit, as the Amount column does.
ACTUAL_COLUMN = "Actual"


def upload_amounts_total(number, planned, actual):
    """'Formulation 1 total: 100.00 g printed, 99.80 g actual.' — the two
    numbers under the table that puts them side by side. The words are the
    table's own two column heads, lowercased into a sentence."""
    return (f"{FORMULATION_CAP} {number} {TOTAL_LABEL.lower()}: "
            f"{planned} printed, {actual} actual.")

# The one line under each sheet's title. The sheets are protected now,
# so this says what can be typed and where: a locked cell that refuses a
# number without saying why is the worst kind of paper.
#
# BOXED, not shaded. "Fill in the shaded cells only" was false twice over:
# every ingredient row was shaded too, and on the black-and-white printer a
# bench sheet actually goes through, the writable cream and the pastel
# bands were the same grey. Only a write-in cell is shaded now, and every
# one of them is drawn as a box — which survives a mono printer.
#
# One line per sheet, naming that sheet's own cells and no others: the
# Lot cells are on the summary and the Actual and Tick cells are on the
# pages, and a line naming a column that is not on the page in the
# reader's hand sends them hunting for it.
# Where the sheets go when they come back. The bench fills the boxed cells
# in and then has a file and no idea what to do with it: the sheet itself
# names the door, in the words the screen puts on it.
# (N17: the upload fold is on the tab itself, not under Save results.)
SHEET_RETURN_PATH = (f"Then upload this file in the app: {TAB_BATCH} \u2192 "
                     f"{UPLOAD_EXPANDER}.")
# ...and the sheets the app never reads back say so, so nobody fills one in
# and waits for it to arrive.
SHEET_IS_A_RECORD = "This sheet is a record. The app does not read it back."


def write_in_note(columns):
    """'Write in the boxed cells only: Measured, Not scored, Note, Lot. Then
    upload this file in the app: 2 · Make a round -> Save results -> Or
    upload results from a file.'

    The second half is the one thing the pack never said. A bench filled the
    boxed cells in, had a file, and nothing on the page said where it goes.
    """
    return (f"Write in the boxed cells only: {', '.join(columns)}. "
            + SHEET_RETURN_PATH)


# The summary sheet has no column headed `Measured`: the block's own header
# row heads its first column `Measurement` (and the rest by formulation
# name), so that is the word the instruction names — one word for one
# column, both read from MEASUREMENT_COLUMN. The formulation pages keep
# `Measured`, because that IS the header they write over their write-in
# column.
# Who made it and when is a cell the pen reaches too. It was printed into a
# LOCKED cell, so the blanks could not be filled in the file they were
# printed in — and the sheets now come back as a file.
MADE_BY_COLUMN = "Made by"


def summary_write_in_note(lot=True):
    """The round sheet's own line. A project that weighs nothing out has no
    Lot cells on the page, and a line naming a column that is not there
    sends the reader hunting for it."""
    columns = [MEASUREMENT_COLUMN, NOT_SCORED, NOTE]
    if lot:
        columns.append(LOT_COLUMN)
    columns.append(MADE_BY_COLUMN)
    return write_in_note(columns)


def sheet_write_in_note(actual_head=ACTUAL_COLUMN, weighs=True):
    """The formulation page's own line. `actual_head` is that page's own
    column header — 'Actual (g)' — so the line names a column the reader
    can point at rather than a shorter word beside it.

    `weighs` is False for a project of process settings alone: there is no
    amounts table on the page, so no Tick column and no Actual beside one,
    and naming either sent the reader looking for a table that is not
    there. `actual_head` of None drops the Actual cells too (they are off
    until the project asks to record them).
    """
    columns = [TICK_COLUMN] if weighs else []
    if actual_head:
        columns.append(actual_head)
    columns += [MEASURED_COLUMN, NOT_SCORED, NOTE, MADE_BY_COLUMN]
    return write_in_note(columns)


# What a formulation's note says when its amounts came back off the
# sheet rather than off the screen. It goes in front of whatever the
# bench wrote, exactly as the not-scored marker does: the row's amounts
# are no longer the ones the app suggested, and nothing else on the
# Results tab would say so.
AMOUNTS_AS_WEIGHED = "Actual amounts"


def amounts_as_weighed_note(note):
    """'Amounts as weighed · lumpy' — the marker, then what was typed."""
    text = str(note or "").strip()
    return f"{AMOUNTS_AS_WEIGHED} · {text}" if text else AMOUNTS_AS_WEIGHED


def workbook_actual_below_zero(number, name):
    """A weight written as a negative number. Nothing was ever weighed out
    of a bowl, so it is a slip of the pen or a minus sign left in front of
    a correction, and either way the sheet says which cell."""
    return (f"{FORMULATION_CAP} {number}: the {ACTUAL_COLUMN} cell for "
            f"{name} is below 0. Write what you weighed.")


def workbook_actual_not_a_number(number, name):
    """An Actual cell with something in it that is not a weight. It is
    refused rather than dropped: the whole point of the column is that
    what was weighed is not what was printed, so ignoring it would file
    the printed amount under a formulation nobody made."""
    return (f"{FORMULATION_CAP} {number}: the {ACTUAL_COLUMN} cell for "
            f"{name} is not a number. Write what you weighed, or leave it "
            "blank.")


# The sheet of the All formulations file that carries the lot numbers:
# one row per ingredient per round. They are not columns on the
# formulations table — a lot belongs to a round, not to a formulation —
# and a column per ingredient would double that table's width.
LOTS_SHEET = "Lots"


# ------------------------------------------------------------------ #
#  The model's own refusals
#
#  Every one of these reaches a screen: food_bo raises them, and app.py or
#  a tab module prints what it caught. They lived in food_bo as literals,
#  which is how a retired word went on being said for a whole cycle with
#  nobody reading it — this file says it is every word the user reads, and
#  a refusal is a word the user reads. tests/test_food_bo.py's own guard
#  now refuses a string literal at a `raise ValueError` in food_bo.py.
# ------------------------------------------------------------------ #

# Names: taken, reserved, empty.
def name_taken_by_variable(name, category):
    kind = ("an ingredient" if category == 'ingredient'
            else "a process setting")
    return f"{name} is already the name of {kind}. Choose another name."


def name_taken_by_measurement(name):
    return f"{name} is already the name of a measurement. Choose another name."


def name_taken_by_property(name):
    return f"{name} is already a property of this project."


def name_taken_by(name, what):
    """'Cocoa already exists as a process setting.' — `what` is how the
    other row was named by whoever found the clash."""
    return f"{name} already exists as {what}."


def reserved_name_message(name):
    return (f"{name} is a column name Food Optimizer uses; choose "
            f"another, such as {name}s.")


def reserved_name_short(name):
    """The same refusal for a measurement, which has no plural to suggest:
    'Firmnesss' is not a name anybody wants."""
    return (f"{name} is a column name Food Optimizer uses; choose "
            "another.")


def name_is_a_variable(name):
    return (f"{name} is already the name of an ingredient or process "
            f"setting. Choose another name for the measurement.")


MEASUREMENT_NAME_REQUIRED = "Measurement name cannot be empty."


def score_function_line(terms, ceiling):
    """'Overall score = 60 % × Firmness closeness + 40 % × Juiciness
    closeness. A formulation that hits every goal scores 100.'

    A caption drawn under the measurements grid. It was built in food_bo,
    which is how a sentence on tab 1 could drift without this file
    noticing. `terms` is the assembled left-hand side; the subject of the
    last sentence is the FORMULATION — 100 is the whole-formulation
    ceiling, and "every measurement ... scores 100" read as each one
    scoring it.
    """
    return (f"{OVERALL_SCORE_COLUMN} = {terms}. A {FORMULATION} that hits "
            f"every goal scores {ceiling:g}.")


def score_term(share_text, name):
    """'60 % × Firmness closeness' — one term of the line above."""
    return f"{share_text} × {name} closeness"


def formulation_measurement(number, name):
    """'Formulation 4: Firmness' — how one result cell is named inside the
    refusal that says its value is out of range: an uploaded sheet, and
    Save results on Make a round, which sits below every row."""
    return f"{FORMULATION_CAP} {number}: {name}"


# The two sentences the model still needs and the grid already had.
LAST_VARYING_ROW_ERROR = (
    f"Cannot delete the last {INGREDIENT} or setting that can still vary. "
    f"Give another one a different {LOWEST_LABEL} and {HIGHEST_LABEL} "
    "first.")
AMOUNTS_MISSING_DELETE_ERROR = (
    "Some results were recorded without their amounts, so this cannot be "
    "deleted. Fix it at one amount instead.")


def amounts_missing_delete(name):
    """AMOUNTS_MISSING_DELETE_ERROR, said about the row being deleted."""
    return (f"{name} cannot be deleted: some results were recorded without "
            "their amounts. Fix it at one amount instead.")

LOWEST_ABOVE_HIGHEST_ERROR = "Lowest cannot be above Highest."
RANGE_ENDS_ERROR = f"{LOWEST_MEASURABLE_LABEL} must be less than {HIGHEST_MEASURABLE_LABEL}."
TARGET_REQUIRED_ERROR = f"Enter a {TARGET_LABEL}, or change {GOAL_LABEL}."


def lowest_above_highest(low, high):
    """'Lowest 50 is above Highest 20; swap them.' — the two numbers, and
    the fix. "Lowest cannot be above Highest." said neither (owner,
    2026-10)."""
    return (f"{LOWEST_LABEL} {float(low):g} is above {HIGHEST_LABEL} "
            f"{float(high):g}; swap them.")


def scale_ends_error(low, high):
    """The measurement grid's two scale cells, refused in their numbers:
    equal is a scale of nothing, reversed is a swap."""
    if float(low) == float(high):
        return (f"{LOWEST_MEASURABLE_LABEL} and {HIGHEST_MEASURABLE_LABEL} "
                f"are both {float(low):g}; make {HIGHEST_MEASURABLE_LABEL} "
                "larger.")
    return (f"{LOWEST_MEASURABLE_LABEL} {float(low):g} is above "
            f"{HIGHEST_MEASURABLE_LABEL} {float(high):g}; swap them.")


def target_outside_message(target, low, high):
    """R13: the two controls the reader can reach, named as the grid names
    them. "the range's lowest and highest" named nothing on that screen."""
    return (f"{TARGET_LABEL} {target:g} must be between "
            f"{LOWEST_MEASURABLE_LABEL} {low:g} and "
            f"{HIGHEST_MEASURABLE_LABEL} {high:g}.")


def baseline_outside_message(value, low, high):
    return (f"{BASELINE_LABEL} {float(value):g} must be between "
            f"{LOWEST_LABEL} {low:g} and {HIGHEST_LABEL} {high:g}.")


# What cannot be done to a project whose older formulations were recorded
# without their amounts. "re-import your history" named nothing the app has:
# it has formulations already made.
CANNOT_ADD_WITHOUT_AMOUNTS = (
    "Some formulations already made were recorded without their amounts, so "
    "this cannot be added now. Clear project contents, or add those "
    "formulations again with their amounts.")
CANNOT_RELOAD_INGREDIENTS = (
    "Ingredients cannot be reloaded after results have been recorded. Use "
    "Manage project › Clear project contents, or open a saved copy.")
BASELINE_REQUIRED_FOR_A_SETTING = (
    "A process setting added now needs a baseline (the value used for every "
    "formulation already made) so those formulations are read correctly.")


# The ingredients file, row by row.
def ingredients_file_missing_columns(named):
    """'The file has no Highest column. …' — `named` is the comma-joined
    headings; the fix is the heading, or the template that has them all."""
    many = ", " in named
    return (f"The file has no {named} {'columns' if many else 'column'}. "
            f"Add {'them' if many else 'it'}, or start from the template.")


def file_row_name_blank(row_no):
    return (f"{ROW.capitalize()} {row_no}: enter a {NAME_LABEL}, or delete "
            "the row.")


def file_row_duplicate_name(row_no, name):
    return (f"{ROW.capitalize()} {row_no}: {name} is listed twice; delete "
            "one of the rows.")


def file_row_premix_in_premix(row_no, name):
    """A file row that both declares a pre-mix (its Preparation) and files
    itself under one (its Part of). Said against the row, by name."""
    return (f"{ROW.capitalize()} {row_no}: {name} is a pre-mix; clear its "
            f"{PART_OF_LABEL} cell.")


def file_row_share_not_a_number(row_no, text):
    return (f"{ROW.capitalize()} {row_no}: "
            + cells_need_numbers(typed=[(PREMIX_SHARE_LABEL, text)]))


def file_row_reserved_name(row_no, name):
    return f"{ROW.capitalize()} {row_no}: {reserved_name_message(name)}"


def file_amounts_not_numbers(name, typed=()):
    """An ingredient-file row whose Lowest or Highest holds text. `typed`
    is [(column, text)] for the cells that do, so the sentence names the
    cell and says back what is in it."""
    if not typed:
        return named_row_error(name, f"Enter numbers in {LOWEST_LABEL} and "
                                     f"{HIGHEST_LABEL}.")
    return named_row_error(name, cells_need_numbers(typed=typed))


def file_lowest_above_highest(name, low, high):
    return named_row_error(name, lowest_above_highest(low, high))


# An uploaded results sheet.
SHEET_NEEDS_A_FORMULATION_COLUMN = (
    f"The sheet needs a {FORMULATION_CAP} column with the numbers from the "
    f"{ROUND_CAP} sheets you downloaded.")
SHEET_HAS_NO_ROWS = (f"The file has no {FORMULATION}s in it. Add one row "
                     f"per {FORMULATION}.")


def sheet_missing_columns(named, n=1):
    """An uploaded results file without a column for a measurement:
    'Add a Firmness column, then upload again.'"""
    if n == 1:
        return f"Add a {named} column, then upload again."
    return f"Add columns for {named}, then upload again."


def formulation_number_not_whole(raw):
    return (f"{FORMULATION_CAP} column: {raw} is not a {FORMULATION} "
            "number. Use the numbers on the sheets.")


def formulation_not_in_round(number, round_no, holds):
    return (f"{FORMULATION_CAP} {number} is not in {ROUND_CAP} {round_no}, "
            f"which has {holds}. Check its number.")


def formulation_twice_in_the_sheet(number):
    return f"{FORMULATION_CAP} {number} is in the file twice. Keep one."


def formulation_value_not_a_number(number, name, value=""):
    """Text in a result cell: the formulation, the measurement, and what
    was typed there, cut short if it runs long."""
    shown = str(value).strip()
    if len(shown) > 30:
        shown = shown[:29] + "…"
    tail = f", not “{shown}”" if shown else ""
    return f"{FORMULATION_CAP} {number}: enter a number for {name}{tail}."


def formulation_has_no_measurements(number):
    return (f"{FORMULATION_CAP} {number} has no results. Fill them in, or "
            f"tick {NOT_SCORED}.")


def formulation_is_not_not_scored(number):
    """A number handed to Score a formulation that is not one of the rows
    waiting for a result."""
    return (f"{FORMULATION_CAP} {number} is not waiting to be scored in "
            f"this project.")


# Limits and units.
def property_limits_need_a_mass_unit(unit, fix_sentence):
    return (f"Property limits need every {INGREDIENT} in one unit; "
            f"{fix_sentence}")


def limit_needs_one_unit(fix_sentence):
    return (f"These {INGREDIENT}s need one unit for a {LIMIT}; "
            f"{fix_sentence}")


# Generating, recording, deleting.
EVERYTHING_IS_FIXED = (
    "Every row is fixed. In Set up, give one a "
    f"{HIGHEST_LABEL} above its {LOWEST_LABEL}.")
ADD_A_MEASUREMENT_FIRST = "Add a measurement in Set up before saving results."


def no_measurement_named(name):
    return f"No measurement named {name}."


def no_variable_named(name):
    """R7: the last 'variable', and the only message that printed Python's
    own quotation marks."""
    return f"No ingredient or process setting named {name}."


def no_ingredient_named(name):
    return f"No {INGREDIENT} named {name}."


def no_property_named(name):
    return f"No property named {name}."


def cannot_change(named):
    return f"Cannot change {named}."


def delete_the_process_setting_instead(name):
    return (f"{name} is a process setting. Take it out on the "
            f"{VARIABLES_HEADER} grid instead.")


# ------------------------------------------------------------------ #
#  The model's own refusals, part two: the sentences a helper builds
#
#  Each of these was assembled inside food_bo and handed to a bare
#  `raise ValueError(trouble)`, to a per-row error list or straight onto a
#  screen — which is how the first guard, which reads only the literals AT a
#  raise site, could not see them. The guard now reads every prose literal
#  in that file.
# ------------------------------------------------------------------ #

def ingredient_was_used(name, numbers_text, many=False):
    """Why an ingredient cannot simply be deleted, and the one tick that
    says delete it anyway. The tick is named as the checkbox names itself."""
    word = f"{FORMULATION_CAP}s" if many else FORMULATION_CAP
    return (f"{name} was used in {word} {numbers_text}, so it cannot be "
            f"deleted. Tick {DELETE_EVEN_IF_USED} to discard that "
            "information.")


def limit_unreachable_above(what, most_text):
    """'Fixing these would make the limit on Fat impossible to meet: the
    ingredients that can still vary only reach 10 per 100 g at most. Loosen
    the limit first.' — a limit's At least, put out of reach by the rows
    this save is about to pin."""
    return (f"Fixing these breaks the {LIMIT} on {what}: only {most_text} "
            "is reachable; loosen it first.")


def limit_unreachable_below(what, least_text):
    """The same for a limit's At most that the pinned rows already exceed."""
    return (f"Fixing these breaks the {LIMIT} on {what}: nothing gets "
            f"below {least_text}; loosen it first.")


def limit_pinned_amounts_exceed(what, pinned_text):
    """An amount limit whose At most is already passed by the amounts this
    save pins — nothing that can still vary is even involved."""
    return (f"Fixing these breaks the {LIMIT} on {what}: the fixed amounts "
            f"add up to {pinned_text}; loosen it first.")


LOOSEN_THE_LIMIT_FIRST = "Loosen the limit first."


def enter_in_this_unit(names_text, unit, instead_of=None):
    """'enter Water and Oil in g instead of ml.' — the change that would let
    a limit be written. A refusal that only says the units differ leaves the
    reader to work out which row is the odd one and what to do about it."""
    tail = "" if instead_of is None else f" instead of {instead_of}"
    return f"enter {names_text} in {unit}{tail}{'.'}"


NO_UNIT = "no unit"


def per_amount_text(unit):
    """'per 100 g' — how a limit on the finished formulation reads, in the
    unit the ingredients are written in."""
    return f"per 100 {unit}"


# How the OTHER row is named when a name is already taken by one of the two
# kinds. Two sentences of one idea, so they sit beside the kinds themselves.
AN_INGREDIENT = f"an {INGREDIENT}"
A_PROCESS_SETTING = f"a {KIND_SETTING.lower()}"

# The join between two halves of a limit ("at least 10 g and at most 40 g")
# and between the last two names of a list.
AND_JOIN = " and "

# The Off by column on the best-so-far block: how far a measurement landed
# from its target, in the measurement's own unit.
ON_TARGET = "On target"


def off_by_high(size_text):
    return f"{size_text} too high"


def off_by_low(size_text):
    return f"{size_text} too low"


# Opening a project. Both reach the screen through FoodOptimizer.load_error,
# which app.py prints as it stands.
# Each names the file, so the reader can find it in the folder; what to do
# is project_load_error_info's line, which app.py prints under it.
def project_not_found(name):
    return (f"**{name}.pkl** is no longer in your FoodOptimizer folder. It "
            "may have been renamed or deleted.")


def project_file_damaged(name):
    return f"**{name}.pkl** is damaged and can't be opened."


def project_file_unreadable(name):
    """storage.LocalStorage.load: the bytes are not a project, which is a
    damaged file or one an early version wrote (a pickle)."""
    return (f"**{name}.pkl** can't be opened: it is damaged or from an "
            "early version of Food Optimizer.")


def project_file_locked(name):
    """storage.LocalStorage.load: the file is there but will not be read."""
    return (f"**{name}.pkl** could not be read. Close any other program "
            "using it, then open the project again.")


# Saving, and the copy taken before a change, refused by the disk. `cause`
# is storage.disk_cause's word for what the disk said.
_DISK_TROUBLE = {
    'full': ("the disk is full", "Free up space, then try again."),
    'read_only': ("the FoodOptimizer folder is read-only",
                  "Allow changes to it in Finder, then try again."),
    'in_use': ("another program is using its file",
               "Close that program, then try again."),
}
_DISK_OTHER = ("the file could not be written", "Try again.")


def _disk_trouble(cause):
    """_DISK_TROUBLE's (what, fix) for `cause`, naming the file manager of
    the computer the reader is on."""
    what, fix = _DISK_TROUBLE.get(cause, _DISK_OTHER)
    if cause == 'read_only' and on_windows():
        fix = fix.replace("Finder", "File Explorer")
    return what, fix


def not_saved(name, cause=None):
    """'Oat cookie was not saved: the disk is full. Free up space, then try
    again.' — the save banner's red line."""
    what, fix = _disk_trouble(cause)
    return f"{name} was not saved: {what}. {fix}"


def nothing_changed(name, cause=None):
    """The copy taken before a change (or the rename that deletes or clears
    a project) was refused, so the change was never made."""
    what, fix = _disk_trouble(cause)
    return f"Nothing changed in {name}: {what}. {fix}"


def copy_not_deleted(cause=None):
    what, fix = _disk_trouble(cause)
    if cause == 'in_use':
        what = "another program is using it"
    return f"An old copy could not be deleted: {what}. {fix}"


def changed_elsewhere(name):
    """The save banner's red line when another window wrote the file first;
    SAVE_ERROR_WARNING under it says what to do."""
    return f"{name} was changed in another window."


# ------------------------------------------------------------------ #
# Rule cells (0.5.0 wave 2, "rules"): the Set up grid's Rule column lets an
# ingredient's amount be read off the batch size and the other rows instead
# of typed by hand.
#
# RULE, not "formula". To a food formulation scientist a formula IS the
# recipe — the thing this app calls a Formulation — and the column sat one
# cell from it on the same row. The stored field, the parser and the
# helpers keep their own names; nothing a reader sees says "formula" any
# more, and nothing else in the app is called a rule (a limit is a limit).
# ------------------------------------------------------------------ #
FORMULA_REST_ALONE = (f"Write {REST_TOKEN} on its own; it means "
                      f"{FILL_TO_TOTAL}.")
FORMULA_TWO_AMOUNTS = "Multiply an amount by a number, not by another amount."
FORMULA_DIVIDE_BY_AMOUNT = "Divide by a number, not by an amount."
FORMULA_DIVIDE_BY_ZERO = "Divide by a number other than 0."
# Every word the grammar really takes, as the fix: the reader is told
# what a calculation may hold. The keyboard spellings (- * /) and the
# rest are in the calculation guide's "?", where long reasons belong.
FORMULA_UNREADABLE = ("Use only names, numbers, batch size, %, brackets "
                      "and + − × ÷.")
RULE_NEEDS_EQUALS = "Start a calculation with =."
RULE_USES_ITS_OWN_ROW = "Take this row's own name out of its calculation."
RULE_INGREDIENTS_ONLY_PREFIX = "A calculation can use ingredients and batch size, not"


def rule_ingredients_only(name):
    """'A calculation can use ingredients and batch size, not Cook temperature.' —
    a calculation naming a process setting. Grams of salt calculated from minutes
    of cooking is arithmetic across two units that cannot be mixed, and the
    app refused a setting a calculation of its own while allowing the reverse."""
    return f"{RULE_INGREDIENTS_ONLY_PREFIX} {name}."


def formula_unknown_name(name):
    """'There is no ingredient called Sodium citrate.' — a calculation naming a
    row the project does not have."""
    return f"There is no ingredient called {name}; check the spelling."


def formula_loop(chain):
    """'A calculation cannot lead back to itself: Fat → Water → Fat.' — `chain`
    is the arrow-joined names that already spell the loop out."""
    return f"These calculations go round in a circle: {chain}. Change one."


FORMULA_NEEDS_BATCH_SIZE = (
    f"Set {FORMULATION_TOTAL_NAME} in {MORE_SETTINGS_EXPANDER} first, or "
    f"enter {LOWEST_LABEL} and {HIGHEST_LABEL}.")
PERCENT_NEEDS_BATCH_SIZE = (
    f"Set {FORMULATION_TOTAL_NAME} in {MORE_SETTINGS_EXPANDER} to limit "
    "by percent.")


# ------------------------------------------------------------------ #
# Rule rows (0.5.0 wave 2, "rules", task 2): what the model owes the
# reader once a rule row is calculated from the others instead of being
# searched. Nothing here mentions the model: a row that cannot be worked
# out is refused in the amounts and the rule the reader typed.
# ------------------------------------------------------------------ #

def formula_below_zero(name, unit):
    """'Water is below 0 g in every formulation the allowed amounts reach.
    Widen an amount, or change its calculation.' — a calculation no allowed amounts can
    ever make a real amount of.

    The ROW, not the expression it was typed as: on a grid of eight rows
    the reader had to match an expression back to a row by eye, and every
    other refusal names the row."""
    zero = f"0 {unit}".strip()
    return (f"{name} is below {zero} in every {FORMULATION} the allowed "
            f"amounts reach. Widen an amount, or change its "
            f"{FORMULA_IN_RANGE}.")


def balance_would_go_negative(name, size_text, least_text, unit="",
                              noun=None):
    """'A default batch size of 60 g leaves Water below 0 g. The other
    ingredients need at least 70 g.' — the = rest row is whatever is left
    of the batch size, and there is nothing left.

    The unit is on all three numbers: the one that needed it most was the
    one going without. And the noun is the box that asked — the Default
    batch size box on Set up, the Batch size box on Make a round — because
    wave 1 settled those as two names for two things."""
    zero = f"0 {unit}".strip()
    return (f"{(noun or FORMULATION_TOTAL_NOUN).capitalize()} {size_text} "
            f"leaves {name} below {zero}; enter at least "
            f"{least_text}.")


def formula_reads_this_row(name, row):
    """'Water is calculated from Flour. Change Water's calculation first.' —
    deleting a row another row's calculation reads would leave that calculation naming
    nothing."""
    return (f"{row} is {WORKED_OUT} from {name}. Change {row}'s "
            f"{FORMULA_IN_RANGE} first.")


COPY_TWO_BALANCE_ROWS = (f"This copy gives two rows = {REST_TOKEN}, and "
                         "only one row can take it.")


# ------------------------------------------------------------------ #
# The Rule column (0.5.0 wave 2, "rules", task 3): the one new column on
# the ingredients grid, the consequence every worked-out row owes in
# numbers, and the two refusals a rule earns over the finished grid rather
# than at one cell.
# ------------------------------------------------------------------ #
FORMULA_HELP = (
    "Optional. Calculate this ingredient's amount from other ingredients or the batch size. "
    "Leave blank to let the app choose between Lowest and Highest.")

CALCULATION_TERMS_TABLE = """| Use | Example |
|---|---|
| Ingredient names | `Flour` |
| Total for one formulation | `batch size` |
| Numbers | `2`, `0.5` |
| Add / subtract | `Flour + Starch`, `Flour - Starch` |
| Multiply / divide by a number | `2 * Flour`, `Flour / 2` |
| Percentages | `5% of batch size` |
| Parentheses | `2 * (Flour + Starch)` |"""

CALCULATION_SYNTAX_LABEL = "Syntax details"
CALCULATION_SYNTAX_DETAILS = (
    "- Select ingredient names in **Edit calculation** to avoid typing errors. Spelling and spaces must match; capitalization does not.\n"
    "- The editor supplies `=`. Both `* /` and `× ÷` work.\n"
    "- `10% of Flour` also works. `batch size - Water` subtracts only Water; **Fill to total** accounts for all other ingredients. Existing `= rest` means **Fill to total**.\n"
    "- Not supported: process settings, cell addresses, Excel functions, powers, multiplying two amounts, dividing by an amount, or circular references."
)
RULE_GUIDE = (
    "**Fill to total** adds enough of one ingredient to reach the batch size after all other ingredients.\n\n"
    + CALCULATION_TERMS_TABLE
    + "\n\nUse **Edit calculation** to insert ingredients and operations. Examples illustrate arithmetic, not recommended ratios."
)


def one_balance_only(names_text, many=False):
    """'Only one row can be = rest: Water and Salt both are.' — two rows
    each taking whatever is left of the same number is not arithmetic
    anybody can do. Said over the grid as a whole, because it is about the
    pair — and it names the pair, which the old sentence did not, leaving
    the reader to go and find the other one."""
    return (f"Only one row can use {FILL_TO_TOTAL}; {names_text} "
            f"{'all' if many else 'both'} do. Keep one.")


def worked_out_caption(name, formula_text, low_text, high_text, size_text,
                       rest=False, outside_text=""):
    """'Water is calculated as batch size − Pea protein − Salt: between
    40.00 and 62.00 g in a 100 g formulation.' — one line under the grid per
    row that is calculated rather than typed, so the calculation shows its
    consequence in numbers.

    A row that takes the remainder says so in the words it was written in:
    'Water is calculated as = rest, whatever is left of the batch size:
    ...'. The amounts are what the other rows' allowed amounts leave it,
    and `size_text` is the default batch size they are read against — blank
    while the project has none, and the sentence then stops at the amounts.

    Both branches say `calculated`, the word in the two cells beside them,
    and both quote the cell WITH its '=', so the reader can match the line
    to what they typed. `rest` is the parser's own answer to "is this the
    rest row" — passed in rather than read off the text, so '=rest' and
    '= REST' say the same thing here as they do everywhere else.

    `outside_text` is the row's own Lowest and Highest when the calculation takes
    it past them. They are dormant on a worked-out row — the calculation decides
    the amount, not the range — but a calculation that puts Salt at 16 g over a
    cap of 3 g was landing with nothing said at all.
    """
    text = str(formula_text).strip()
    if rest:
        head = (f"{name} fills to the {size_text} {FORMULATION_TOTAL_NOUN}"
                if size_text else f"{name} fills the remaining amount of the formulation")
    else:
        head = f"{name} is calculated from {text}"
    # A rule that comes to one number says that number: "between 1.50 and
    # 1.50 g" asked the reader to read a range where nothing can vary.
    span = (f"{head}: {high_text}" if low_text == high_text.split(" ")[0]
            else f"{head}: between {low_text} and {high_text}")
    if size_text and not rest:
        span = f"{span} in a {size_text} {FORMULATION}"
    if outside_text:
        return (f"{span}. Its own {LOWEST_LABEL} and {HIGHEST_LABEL} "
                f"({outside_text}) do not apply while the calculation does.")
    return f"{span}."


def rest_row_keeps_the_others(size_text):
    """'At 250 g, the other rows keep their amounts. To keep the same
    proportions, widen Lowest and Highest too.' — said under the Default
    batch size box the moment that number moves, for a project with a row
    written = rest.

    Amounts written in grams do not follow the batch size and that row
    does: a burger turns into soup one number at a time, and the only
    thing the app said about it was that a limit had been deleted. What the
    rest row now takes is the grid's own caption (worked_out_caption); it
    is not said twice (polish round: two Water captions)."""
    return (f"At {size_text}, the other rows keep their amounts. To keep the same "
            f"proportions, widen {LOWEST_LABEL} and {HIGHEST_LABEL} too.")


def formulations_keep_their_amounts(names_text, many=False):
    """'Formulations already made keep their amounts. Water is calculated
    from its formula from the next round on.' — a formula landing on a
    project that has results. What was weighed is what was weighed; the
    formula starts answering for the row from the next round."""
    if many:
        tail = (f"{names_text} are {WORKED_OUT} from their "
                f"{FORMULA_IN_RANGE}s")
    else:
        tail = f"{names_text} is {WORKED_OUT} from its {FORMULA_IN_RANGE}"
    return (f"{FORMULATION_CAP}s already made keep their amounts. "
            f"{tail} from the next {ROUND_CAP} on.")


# ------------------------------------------------------------------ #
# The workbook and the round table (0.5.0 wave 2, "rules", task 4): a
# worked-out row prints the amount it computed to, marked so a bench
# reading the page knows it was not chosen but left over, and the round
# table on tab 2 stays as it always was — read only.
# ------------------------------------------------------------------ #

def worked_out_label(name):
    """'Water · calculated' — the mark a worked-out row's own name wears on
    the summary sheet and on its own formulation page, so a bench reading
    the printed page knows this amount was not chosen, only computed."""
    return f"{name} · {WORKED_OUT}"


def worked_out_row_note(lines_text):
    """'Water is calculated: = rest. Weigh the amount printed.' — the
    summary sheet's note, naming each worked-out row's own calculation. The calculation
    is on no sheet of the round workbook, so a bench holding the page was
    told the amount came from one and had nowhere to see it."""
    return f"{lines_text} Weigh the amount printed."


def worked_out_row_rule(name, rule_text):
    """'Water is calculated: = rest.' — one row of the note above."""
    if str(rule_text).replace(" ", "").lower() == "=rest":
        return f"{name} is calculated to bring the ingredients to the formulation total."
    return f"{name} is {WORKED_OUT}: {rule_text}."


def setup_sheet_formula_text(text, rest=False):
    """'= rest (batch size − every other ingredient)' — what the row that
    takes the remainder prints in the Set-up sheet's Formula column; every
    other row prints exactly what it was typed as. `rest` comes from the
    parser, not from the spelling of the cell."""
    if rest:
        return f"{FILL_TO_TOTAL} ({BATCH_SIZE_NOUN} − every other {INGREDIENT})"
    return text


# The round table on tab 2 gains no editing of its own: a worked-out row's
# amount is what it computed to, and correcting what the bench actually
# weighed happens after the round is recorded, on the Results tab.
# The greyed boxes at the foot of `Add a formulation to this round`, and
# the one line that says why they are last: they fill in from the boxes
# above, so they cannot be anywhere else, and the form was the only table
# in the app listing the rows in a different order with nothing said.
WORKED_OUT_BOXES_CAPTION = "Calculated from the amounts above."
WORKED_OUT_BOXES_HELP = "These are calculated from the amounts above, so they come last."


# ------------------------------------------------------------------ #
# 0.7.0 wave 3, "pre-mixes": an ingredient made from its own parts.
#
# Three words, one concept each. A PRE-MIX is the ingredient; its PARTS
# are what goes into it; its MAKE-UP is what per cent of it each part is.
# The word this block never uses is the one a food scientist means by
# something else entirely, and it is banned on screen for that reason.
#
# A pre-mix is made one of two ways, and the two are named for what the
# bench does rather than for what the app does with them: one pre-mix,
# made and then portioned into every formulation, or weighed into each
# formulation part by part.
# ------------------------------------------------------------------ #
PREMIX_LABEL = "Pre-mix"
# The column that says which pre-mix a row is a PART of. It is not "Pre-mix":
# on a pre-mix's own row that cell is blank and on a part's row it is filled,
# so the one header named the thing and the relationship to the thing at once.
# `Part of` is the word the parts fold already uses, and it cannot be read as
# "this row is a pre-mix".
PART_OF_LABEL = "Part of"
MADE_AS_LABEL = "Preparation"
OLD_MADE_AS_LABEL = "Made as"
OLD_CALCULATED_LABEL = "worked out"
OLD_CALCULATED_RANGE = "From calculation"
PREMIX_SHARE_LABEL = "Composition (%)"
# The three answers to `Made as`, and the whole of what the column asks.
# Neither of the two pre-mix answers carries a comma any more: two sentences
# list both options, and "one pre-mix, portioned, or weighed into each
# formulation" reads as three things where the select offers two.
# `bought in` is the ordinary case and the default — the cell was blank, and
# a blank is not an answer a reader can recognise as the one they want.
PREMIX_MADE_AS_BOUGHT_IN = "Single ingredient"
# What the portioned answer was called before the comma came out of it, in
# both the spellings a hand-typed file uses. An ingredients file written
# then still loads.
PREMIX_MADE_AS_PORTIONED_WAS = ("portioned from one pre-mix", "one pre-mix, portioned",
                                "one premix, portioned", "Fixed-ratio pre-mix")
PREMIX_MADE_AS_PORTIONED = "Pre-mix: keep proportions fixed"
PREMIX_MADE_AS_WEIGHED = "Blend: vary each ingredient"
OLD_VARIABLE_RATIO_MODE = "variable-ratio blend"
# How a pre-mix is named when something else found the clash, alongside
# AN_INGREDIENT and A_PROCESS_SETTING above.
A_PREMIX = "a pre-mix"

MADE_AS_REQUIRED_ERROR = (
    f"Choose a preparation method: {PREMIX_MADE_AS_BOUGHT_IN}, "
    f"{PREMIX_MADE_AS_PORTIONED} or {PREMIX_MADE_AS_WEIGHED}.")
PART_SHARE_ERROR = f"Enter the {PREMIX_SHARE_LABEL} as a number, or leave it empty."
PREMIX_INSIDE_PREMIX = ("A pre-mix cannot be a part of another pre-mix. "
                       "Type an ingredient's name in the Part cell.")
PART_IS_ITS_OWN_PREMIX = PREMIX_INSIDE_PREMIX
# The same caption the measurements grid shows, in the pre-mix's own
# noun: what the reader typed did not add up to 100, and the app moved
# the rest of the column rather than refusing the save.
def shares_adjusted_premix(pairs_text):
    """'% of pre-mix adjusted to add up to 100 %: Pea protein isolate 66.00,
    Potato starch 26.00, Methylcellulose 8.00.'

    It said only that the column had been adjusted. The reader typed 20 and
    70 and the app wrote 22.22 and 77.78, and nothing on the screen said to
    what — so the numbers are in the sentence.

    The noun is the COLUMN's own name. Its sibling on the measurements grid
    is `Shares adjusted to add up to 100 %.`, over a column headed `Share of
    score (%)`; `share` on this grid is a word on no control at all.
    """
    return f"{PREMIX_SHARE_LABEL} adjusted to add up to 100 %: {pairs_text}."


def premix_amount_is_its_parts(name):
    """'Fat phase's amount is the sum of its parts. Change the parts in its
    fold.' — a number typed over the word in a weighed pre-mix's Lowest or
    Highest cell. The cell lets a pen in and the app cannot take what it
    says: the amount of a weighed pre-mix is arithmetic over its parts."""
    return (f"Its amount is the {SUM_OF_ITS_PARTS}; change them under "
            f"{BLEND_COMPOSITIONS_HEADING}.")


def premix_unknown(name):
    """'Dry blend is not a pre-mix of this project.' — the one refusal a
    caller naming a pre-mix that has been deleted, or never added, gets."""
    return f"{name} is not a pre-mix of this project."


def file_row_made_as_unknown(row_no, text):
    """Row 4 of an ingredients file says a pre-mix is made a third way.
    The sentence names the two, because they are the answer."""
    return (f"Row {row_no} says {MADE_AS_LABEL} {text}. Write "
            f"{PREMIX_MADE_AS_PORTIONED}, or {PREMIX_MADE_AS_WEIGHED}.")


def file_row_unknown_premix(row_no, name):
    """A part filed under a pre-mix no row of the file declares. The file
    is read whole, so the order of its rows is not the fault."""
    return (f"Row {row_no}: add a row for {name} with its "
            f"{MADE_AS_LABEL}.")


# The word for what per cent of a pre-mix each part is, as a sentence says
# it. The column header is PREMIX_SHARE_LABEL above; this is the noun, and
# it is the only one — a pre-mix has parts and a make-up, and the word a
# food scientist reaches for instead means something else entirely.
PREMIX_MAKE_UP = "make-up"


def premix_portioned_consequence(name):
    """'The suggestions vary how much Dry blend goes in. Its make-up stays
    the same for the whole round, so you make it once.'

    The one sentence the reader gets when they say a pre-mix is made this
    way, said once, at the choice. Both halves are consequences they can
    check: what the suggestions will move, and what the bench will do with
    the answer. Portioned, the parts are not named — they are not what
    varies, and naming them here is what made the two ways sound alike.
    """
    return (f"The suggestions vary how much {name} goes in. Its "
            f"{PREMIX_MAKE_UP} stays the same for the whole {ROUND_CAP}, so you "
            "make it once.")


def premix_weighed_consequence(names_text):
    """'The suggestions vary pea protein, fibre and salt separately. Each
    formulation gets its own amounts of them.'

    The other half of the same choice, in the same two halves. Weighed, it
    IS the parts that vary, so they are named: the reader is agreeing to a
    search over three amounts instead of one, and the list is the only
    thing on screen that says so.

    The second half says what the bench does, as the portioned sentence's
    second half does. It said "its own blend", which is the concept word
    the app does not use — and two of the sample's own pre-mixes are named
    `Dry blend` and `Seasoning blend`, so a reader who had just typed those
    read it as a statement about a row of their grid.
    """
    return (f"The suggestions vary {names_text} separately. Each "
            f"{FORMULATION} gets its own amounts of them.")


def premix_row_band(name, low_text, high_text):
    """'Dry blend goes in at 20.00 to 40.00 g.' — said beside the choice,
    so the numbers the switch produced are on the screen that made it.

    A mode switch moves which rows the suggestions hold amounts for, and
    the amounts themselves were left at 0.00 to 0.00 with nothing said: a
    round generated in that state has no protein in it at all."""
    return f"{name} goes in at {low_text} to {high_text}."


def premix_part_bands(pairs_text):
    """'Each part has its own Lowest and Highest now: Pea protein isolate
    11.00 to 22.00 g, Potato starch 4.33 to 8.67 g.' — the other side of
    the same switch, said in the same breath as the choice."""
    return (f"Each {PART_LABEL.lower()} has its own {LOWEST_LABEL} and "
            f"{HIGHEST_LABEL} now: {pairs_text}.")


def premix_parts_need_amounts(names_text, many=False):
    """'Give Pea protein isolate and Potato starch a Lowest and a Highest
    before making a round: they are at 0.00 g.' — the honest answer when a
    switch could not work the amounts out, because the pre-mix's own row
    had none or the parts' % of pre-mix add up to nothing."""
    verb = "they are" if many else "it is"
    return (f"Give {names_text} a {LOWEST_LABEL} and a {HIGHEST_LABEL} "
            f"before making a {ROUND_CAP}: {verb} at 0.00 now.")


# ------------------------------------------------------------------ #
# 0.7.0 wave 3, fix round 1: the pre-mix invariant, said in both
# directions. A name a pre-mix owns is a row of the list the suggestions
# move, or the thing such a row is made of. Nothing else in the project
# may wear it, and nothing else may take it away behind the pre-mix's
# back — so these are the sentences the other doors refuse in.
# ------------------------------------------------------------------ #

def name_taken_by_part(name, premix):
    """'Flour is already a part of Dry blend.' — an ordinary row wearing
    the name of something inside a pre-mix is the same flour in the bowl
    twice, and the sentence names the pre-mix so the reader knows where to
    look."""
    return f"{name} is already a part of {premix}."


def delete_the_premix_instead(name, premix):
    """'Water is part of Wet blend. Take it out of the pre-mix instead.'
    — deleting the row left the part behind, and the next save of the
    make-up put the row straight back at no amount at all.

    A pre-mix's own row names itself, which reads as one sentence either
    way: 'Dry blend is part of Dry blend' would not, so that case says
    what it is instead."""
    if name == premix:
        return (f"{name} is a pre-mix. Choose Single ingredient in its Preparation cell "
                "to stop treating it as a pre-mix.")
    return f"{name} is part of {premix}. Take it out of the pre-mix instead."


def part_in_two_weighed_premixes(name, first, second):
    """'Oil is already weighed into each formulation as part of Wet
    blend, so it cannot also be part of Fry blend.' — weighed, a part IS
    a row, and one row standing for two lots of mass is counted twice
    everywhere it is read."""
    return (f"{name} already has its own amount as part of "
            f"{first}, so it cannot also be part of {second}.")


# ------------------------------------------------------------------ #
# 0.7.0 wave 3, task 3: the pre-mix on the grid.
#
# The choice sits on the row — one select column right after Type — and
# the parts open in a fold directly underneath the grid. A part is read
# where it is typed, exactly as an ingredient is, and the fold shows only
# the columns the way the pre-mix is made actually needs.
# ------------------------------------------------------------------ #
MADE_AS_HELP = (
    f"**{PREMIX_MADE_AS_BOUGHT_IN}:** set the amount of one ingredient, including a purchased blend.\n\n"
    f"**{PREMIX_MADE_AS_PORTIONED}:** keep the same ingredient percentages in every formulation. "
    "The app can change how much pre-mix is used.\n\n"
    f"**{PREMIX_MADE_AS_WEIGHED}:** set Lowest and Highest for each ingredient in the blend. "
    "The app chooses their amounts separately, so their proportions and combined amount can change.\n\n"
    "Enter your own ingredient or blend name in Name.")

BLEND_OILS_EXAMPLE = (
    "Example: one formulation could use 8 g coconut oil and 4 g sunflower oil; "
    "another could use 6 g and 5 g. Fats and oils groups these two ingredients.")


# The first column of a pre-mix's own grid. The row IS the part, so the
# header is the noun and not "Name": the grid above already has a Name.
PART_LABEL = "Part"
# What a weighed pre-mix's own row says where its allowed amounts would be.
# It has none: the pre-mix is not a row of the list at all, its parts are,
# and the amount of it in a formulation is whatever they add up to.
SUM_OF_ITS_PARTS = "sum of its parts"
PARTS_ADD_TO_NOTHING = (f"{PREMIX_SHARE_LABEL} adds up to 0; give one "
                        f"{PART_LABEL.lower()} more than 0.")
PREMIX_NEEDS_A_PART = "Enter at least one name in the Part column before saving."


def premix_has_no_parts(name):
    """A limit on a weighed pre-mix that has no parts yet: nothing to add
    up. Said by name, pointing where the parts are typed."""
    return (f"{name} has no parts yet; add them under "
            f"{BLEND_COMPOSITIONS_HEADING}.")
PREMIX_QUANTITY_POSITIVE = "The smallest quantity that blends evenly must be more than nothing."


def premix_fold_caption(weighed):
    """The one line at the top of a pre-mix's parts fold: what the numbers
    in it are, and what may be a part.

    `Composition (%)` had no tooltip, no caption and no line under the grid
    — the one term on the tab that was never said — and it is the central
    number of a portioned pre-mix. One sentence each way: neither branch
    promises anything about what may be a part, which is said where a part
    is typed.
    """
    if weighed:
        return ("Set Lowest and Highest for each ingredient. The app chooses an amount for each one "
                "in every formulation. The blend total is the sum of these amounts.")
    return ("Keep these ingredient percentages the same in every formulation. "
            "The app can change how much pre-mix is used.")


def premix_grid_title(name):
    """'Dry blend · parts' — the fold under the ingredients grid, named for
    the pre-mix it belongs to and for what is inside it. The same separator
    every other two-part title in the app uses."""
    return f"{name} · Parts"


def premix_parts_total(share_text):
    """'Total 100 %' — the caption under a portioned pre-mix's parts, so the
    column the reader is typing into shows its own sum. The shares are
    scaled to 100 at the save; this is what they add up to now."""
    return f"{TOTAL_LABEL} {share_text}"


def premix_no_longer_a_premix(name, parts_text="", many=False):
    """'Dry blend will not be a pre-mix any more. Flour and Salt go with it.'

    Clearing Made as hands the row back to the ordinary list, and the parts
    have nowhere to be — so they are named before they go, the way every
    other Delete on this tab names what it takes."""
    line = f"{name} will no longer be a pre-mix."
    if parts_text:
        line += f" {parts_text} will also be deleted."
    return f"{line} {COPY_KEPT}"


PREMIX_LIMIT_ON_ITS_OWN = "Choose a pre-mix weighed into each formulation on its own to limit its total. To limit particular ingredients together, choose their names instead."


def premix_limit_on_its_own(name):
    """A pre-mix picked together with other names: its total is limited on
    its own, and its parts by name."""
    return (f"Pick {name} on its own to limit its total, or pick its parts "
            "instead.")


PREMIXES_HEADING = "Pre-mixes"
# Two headings, not one. `To weigh for this round` totalled two kinds of
# number under one instruction: a pre-mix's parts ARE weighed together at
# the number printed, and a loose row's total is three formulations added up
# and is never weighed at that number anywhere. A tired bench operator
# weighed out 167.52 g of water.
MAKE_FOR_ROUND_HEADING = "Make for this Round"
HAVE_ON_HAND_HEADING = "Have on hand"
HAVE_ON_HAND_CAPTION = (f"Across every {FORMULATION}; a {WORKED_OUT} row's total "
                        "is what the Round comes to, not one weighing.")
ROUND_TOTAL_COLUMN = "Total for this Round"
# The write-in line at the foot of a pre-mix page. The Round sheet has a Lot
# box against `Dry blend (g)` — a thing the bench made itself, which has no
# lot until somebody gives it one, and nothing anywhere assigned it one.
PREMIX_LOT_LABEL = "Pre-mix lot"
PREMIX_BLENDED_BY_LABEL = "Blended by"
PREMIX_BLENDED_ON_LABEL = "On"
PREMIX_BLEND_TIME_LABEL = "Blend time (min)"
# What the Round sheet's Lot cell says against a portioned pre-mix's row.
PREMIX_LOT_ON_ITS_PAGE = "See its preparation sheet"


def premix_write_in_note(actual_head=None, lot=True):
    """The pre-mix page's own line, naming that page's own write-in cells
    and no others."""
    columns = ([actual_head] if actual_head else []) + ([LOT_COLUMN] if lot
                                                        else [])
    if lot:
        columns.append(PREMIX_LOT_LABEL)
    columns += [PREMIX_BLENDED_BY_LABEL, PREMIX_BLENDED_ON_LABEL, PREMIX_BLEND_TIME_LABEL]
    return (f"{PREMIX_SHARE_LABEL} describes this pre-mix. Amounts are the quantities to weigh. "
            + write_in_note(columns))


def premix_sheet_title(name, total_text, need_text=""):
    """'Seasoning blend · make 100 g (this Round needs 6.60 g)'.

    `make 7.50 g` was the worst number in the workbook: you cannot blend
    7.5 g of coarse salt and four fine powders to any homogeneity, and you
    cannot portion three 2.5 g scoops out of it without the salt segregating
    to the bottom. Nor can you dispense a blend with nothing left in the
    bowl, on the paddle or on the scoop. So the page asks for a makeable
    quantity and says what the round takes out of it.
    """
    line = f"{name} · Prepare {total_text}"
    return f"{line} (this Round needs {need_text})" if need_text else line


def premix_sheet_name(name):
    return f"Pre-mix · {name}"


def premix_group_line(name):
    return f"{name} · weighed into this formulation"


def premix_group_total_line(name):
    return f"{name} · total (not weighed)"


def workbook_lot_conflict(name):
    return (f"{name} has two different lot numbers. Make them match, then "
            "upload again.")


def premix_needs_parts(name, here=False):
    """'Add at least one part to Wet blend in Set up before making a round.'
    — and, on Set up itself, 'under Blend compositions', because the tab it sent
    the reader to is the tab they are standing on."""
    where = f"under {BLEND_COMPOSITIONS_HEADING}" if here else f"in {SET_UP_SHEET}"
    return f"Add at least one {PART_LABEL.lower()} to {name} {where} before making a {ROUND_CAP}."


# 02 M6: beside Save results, "saved as you type" read as a second Save.
# The values are kept on this computer and are not results until recorded.
RESULT_DRAFT_SAVED = ("Typed values are kept if you leave this tab, but are not recorded "
                      f"until you click {SAVE_RESULTS}.")
SAVE_RESULTS_HELP = "Saves completed results. Unfinished formulations remain open."
UPLOAD_AMOUNTS_HEADING = "Amounts from the file"
UPLOAD_LOTS_HEADING = "Lot numbers from the file"


BENCH_RECORDS_SHEET = "Bench records"
BENCH_RECORD_COLUMNS = {'sheet': "Sheet", 'cell': "Cell", 'label': "Entry", 'value': "Written"}
BENCH_RECORDS_CAPTION = (
    "Filled-in boxed cells are saved with this Round and included in All formulations. "
    "Preparation amounts and notes are kept as bench records; they do not change "
    "a pre-mix's composition percentages for future Rounds."
)


def premix_page_pointer(name):
    return f"{name} is a pre-mix: see its sheet."


def premix_members_line(name, names):
    return f"{name}: {names}. These parts are weighed into each formulation."

PART_AMOUNTS_REQUIRED = "Enter a number in both Lowest and Highest for this part."


EDIT_FORMULATIONS = 'Edit formulations'
EDIT_FORMULATIONS_CAPTION = 'Formula results and totals are read-only. They update when you save.'
EDIT_FORMULATIONS_ACTIVE = '**Editing formulations** · Double-click a number to change it, then select Save changes.'
EDITABLE_TABLE_HELP = "Double-click a cell to edit, then save below."
EDITABLE_AMOUNT_HELP = 'Double-click this value to edit it. Select Save changes to apply your edits.'
READ_ONLY_AMOUNT_HELP = 'Read-only. Formula results and totals update when you save.'
PRINT_PACK_CHECKBOX = 'Include individual formulation pages for printing'
FINISH_FORMULATION_EDITS = 'Save or cancel your formulation edits to continue with preparation and results.'
COMPOSITION_AMOUNTS = 'Amounts'
SETUP_INTRO = 'Define ingredients, process settings, and measurements. Then generate your first formulations.'
COMPOSITION_AMOUNTS_HELP = 'Enter ingredient amounts. Their proportions are saved as percentages; the preview calculates amounts for the quantity below.'
COMPOSITION_ENTRY_LABEL = 'Enter ingredient amounts instead'
COMPOSITION_PERCENTAGES = 'Percentages'
def composition_basis_caption(basis):
    return f"Composition is expressed as percentages {basis}. Weighing amounts are calculated below."

def premix_quantity_label(unit):
    return f"Pre-mix quantity to prepare ({unit})"

OLD_WEIGHED_MODE = "weighed into each formulation"
OLD_SINGLE_MODE = "bought in"
OLD_COMPOSITION_LABEL = "% of pre-mix"
OLD_COMPOSITION_LABEL_UNHYPHENATED = "% of premix"
EDIT_ONLY_PENDING = "Only unrecorded formulations in this Round can be edited."
EDIT_BALANCE_HELP = " Change its amounts to match."
EDITED_FORMULATION = "Edited formulation"

def edit_calculated_error(name):
    return f"{name} is a calculation; change the amounts it uses."

def edit_number_error(number, name):
    """A cell of the Edit formulations grid left blank, or holding
    something that is not an amount: the row and the column."""
    return f"{FORMULATION_CAP} {number}: enter an amount for {name}."

def edit_formulation_error(number, error):
    return f"Formulation {number}: {error}"

def edit_negative_error(number, names_text, unit=""):
    """An edit that leaves a calculated row below nothing: which row, and
    the fix, which is in the amounts it is worked out from."""
    zero = f"0 {unit}".strip()
    return (f"{FORMULATION_CAP} {number}: {names_text} would go below "
            f"{zero}. Change the other amounts.")

def edit_breaks_limit(number, limit_text):
    """An edit that breaks a limit: the row, the limit in the words the
    Limits list writes it in, and the fix."""
    return (f"{FORMULATION_CAP} {number} breaks the {LIMIT} {limit_text}. "
            "Change its amounts.")

def workbook_result_conflict(number, key):
    return (f"{FORMULATION_CAP} {number} has two different {key} entries. "
            "Make them match, then upload again.")


SHARE_TOTAL = "Total: 100%"
SHARE_PREVIEW_HELP = "Changing one share adjusts the others to keep the total at 100%. Preview the saved shares below."
SCORING_DETAILS_CHECKBOX = "How the score is calculated"
TARGET_REFERENCES_LABEL = "Targets and references"
PROCESS_STUDY_INTRO = "Each trial is one set of process settings. Run each trial, then record its measurements."
PROCESS_SETTINGS_HEADER = "Process settings"
TRIAL_CAP = "Trial"


def for_project(opt, text):
    """Use trial terminology for a study containing only process settings."""
    if opt.has_ingredients() or not opt._process_settings():
        return text
    return (str(text).replace("Ingredients and process settings", PROCESS_SETTINGS_HEADER)
            .replace("Ingredient or process setting", "Process setting")
            .replace("Amounts to make it", PROCESS_SETTINGS_HEADER)
            .replace("weigh out the formulations", "run the trials")
            .replace("Formulations", "Trials").replace("formulations", "trials")
            .replace("Formulation", TRIAL_CAP).replace("formulation", "trial"))


SHARE_SAVED_TOTAL = "After saving: 100% total"

BLEND_COMPOSITIONS_HEADING = "Blend compositions"
INGREDIENT_LIMITS_EXPANDER = "Ingredient limits (optional)"
# The properties grid's own fold, directly under the limits (S11).
INGREDIENT_PROPERTIES_EXPANDER = "Ingredient properties (optional)"
RECORDING_GUIDANCE = "Use Notes for anything else."
RECORDING_GUIDANCE_HELP = ("Use Notes for other observations. Add outcomes you want scored to "
                           "Measurements and targets.")
RECORDS_UPDATED = "Recording options saved. Existing values are kept."

CUSTOM_ADD_LABEL = "Add your own recording field"
CUSTOM_ADD_BUTTON = "Add field"
CUSTOM_FIELD_NAME = "Field name"
CUSTOM_SCOPE_LABEL = "Record for"
# "formulation" on an ingredient project, "trial" on a project of process
# settings alone, through for_project where they are shown (answer-key
# read-through: "trial" read wrong on an ingredient project). The workbook
# import reads each cell's own marker, never these words.
CUSTOM_SCOPE_NAMES = {'formulation': "Each formulation", 'ingredient': "Each ingredient"}
CUSTOM_SECTION_NAMES = {'formulation': "Formulation records", 'ingredient': "Ingredient records"}
CUSTOM_FIELD_HELP = "For example, Operator for each formulation, or Expiry date for each ingredient. These records do not affect scores."
CUSTOM_FIELD_INVALID = "Enter a field name of 1–80 characters and choose what it records."
CUSTOM_FIELD_BUILTIN = "That recording option already exists. Select it in Additional records."
CUSTOM_FIELD_EXISTS = "A field with this name and scope already exists. Select it in Additional records."


def custom_field_builtin(name):
    return f"{name} is already a recording option; select it in Additional records."


def custom_field_exists(name, scope_name):
    """'Operator already exists for Each formulation; …' — "scope" was a
    word on no control; the picker's own choice is."""
    scope_name = scope_name[:1].lower() + scope_name[1:]
    return (f"{name} already exists for {scope_name}; select it in "
            "Additional records.")
CUSTOM_VALUES_INVALID = "Recording values must be text or numbers, up to 2,000 characters per entry."
CUSTOM_WORKBOOK_CHANGED = "These recording fields no longer match this Round. Download a new workbook and transfer your entries."


def custom_record_conflict(field, subject):
    """An Additional records field written twice, differently, in one
    uploaded workbook: which field, for which row."""
    return (f"{subject} has two different {field} entries. Make them match, "
            "then upload again.")


def custom_record_calculation(field, subject):
    """An Additional records cell holding an Excel calculation."""
    return (f"The {field} cell for {subject} is a calculation. Type the "
            "value, then upload again.")


def custom_record_too_long(field, subject):
    return (f"The {field} entry for {subject} is over 2,000 characters. "
            "Shorten it, then upload again.")
CUSTOM_RECORDS_HEADING = "Additional records"
CUSTOM_SHEET_HELP = "Optional text or numbers. These records are saved with the Round and do not affect scores."
CUSTOM_AUTOSAVE_HELP = "Double-click a cell to enter a value. Entries save automatically when you leave the cell and do not affect scores."
CUSTOM_SUBJECT = "Recorded for"
CUSTOM_VALUE = "Value"
CUSTOM_INGREDIENT_LABEL = "Ingredient"
CUSTOM_ROUND_LABEL = f"Records for {ROUND_CAP}"
CUSTOM_IMPORT_FOUND = "Additional records found. Check the entries below before saving."
CUSTOM_SAVED = "Additional records saved."

def custom_record_option(name, scope, opt=None):
    """'Operator · Each formulation': the scope in the project's words (with
    `opt`), the user's own field name exactly as typed (re-review R6)."""
    scope_text = CUSTOM_SCOPE_NAMES[scope]
    return f"{name} · {scope_text if opt is None else for_project(opt, scope_text)}"

# Additional records' options by field name only (S13: longer chips were cut
# off); where each is entered is in ALSO_RECORD_HELP.
RECORD_SCOPE_LABELS = {
    'vendor': "Vendor",
    'sku': "SKU",
    'lot': "Lot numbers",
}


def record_chip(name, scope, opt=None, name_twice=False):
    """An own record field's option: its name as typed ('Raw weight (g)'),
    with its scope only when the project has that name for both scopes."""
    return custom_record_option(name, scope, opt) if name_twice else name
ACTUAL_RECORD_OPTION = "Record changes from the planned amounts or settings"
ACTUAL_RECORD_HELP = "For each formulation, record what you actually weighed or the settings you used in the workbook. Leave unchanged values blank. Imported corrections are used when learning from your results."
LOT_ENTRY_HEADING = "Ingredient lot numbers"
LOT_LOCATION_HELP = "Enter lot numbers below the formulations in Make a round, or in the workbook."
LOT_ENTRY_HELP = "For this Round only. Double-click a Lot cell to enter a number; it saves automatically and is included in the workbook."
LIMIT_BOUND_HELP = "Leave a side empty for no limit."
LIMIT_BOUND_DETAIL = ("Leave a minimum or maximum empty for no restriction on that side. "
                      "Ingredient amounts and other limits still apply.")

def missing_property_values(metric, names):
    """Generate with a limit on a property some ingredients have no value
    for: 'Enter Fat for Butter and Oil under Ingredient properties, or
    delete the Fat limit.'"""
    return (f"Enter {metric} for {names} under Ingredient properties, or "
            f"delete the {metric} {LIMIT}.")


# Results detail browsing and ingredient composition.
RESULT_VIEW_LABEL = "View formulation"
RESULT_VIEW_HELP = "Choose a recorded formulation to see its measurements, amounts and percentages below."
RESULT_PERCENT_COLUMN = "% of formulation"
RESULT_PERCENT_PROCESS_HELP = "Percentages exclude process settings."
RESULT_PERCENT_MIXED_HELP = "No percentages: ingredient units differ."
RESULT_PERCENT_ZERO_HELP = "No percentages: ingredient amounts total zero."
RESULT_UNSCORED_HELP = "Not scored. No measurements have been saved for this formulation."


def result_heading(no, round_no=None):
    return f"{FORMULATION_CAP} {no}" + (f" ({ROUND_CAP} {round_no})" if round_no is not None else "")


def result_view_option(no, round_no=None, best=False, unscored=False):
    label = result_heading(no, round_no)
    if best:
        label = "Best so far · " + label
    if unscored:
        label += " · Not scored"
    return label

CALCULATION_EDIT_BUTTON = "Edit calculation"
CALCULATION_EDIT_HINT = "Type a calculation, or use Edit calculation."
CALCULATION_EDIT_HINT_HELP = ("Type in Calculation, or use Edit calculation for suggestions and "
                              "examples.")


# ------------------------------------------------------------------ #
# The calculation editor (calculation_editor.py) and the compact
# workbook (workbook_flow.py). Both modules arrived in wave 3 carrying
# their own screen text, where the vocabulary guard could not see it.
# Every sentence either module says out loud is here.
# ------------------------------------------------------------------ #
CALCULATION_EDITOR_TITLE = "Edit calculation"
CALCULATION_MODE_RANGE = "Vary between limits"
CALCULATION_MODE_AMOUNT = "Calculate an amount"
CALCULATION_MODE_QUESTION = "How is this amount determined?"
CALCULATION_NEEDS_AN_INGREDIENT = (
    "Add an ingredient and save it before creating a calculation.")
CALCULATION_RANGE_CAPTION = (
    f"The app chooses an amount between {LOWEST_LABEL} and {HIGHEST_LABEL}. "
    "Use the same value in both to keep it fixed.")
CALCULATION_USE_RANGE_BUTTON = "Use limits"
CALCULATION_FILL_BODY = (
    f"This {INGREDIENT} fills what remains after all other {INGREDIENT}s "
    "are added.")
CALCULATION_FILL_CAPTION = (
    f"{BATCH_SIZE_NOUN.capitalize()} is the total {INGREDIENT} amount "
    f"for one {FORMULATION}. Only one {INGREDIENT} can fill the remaining "
    "amount.")
CALCULATION_FILL_EXAMPLE = (
    f"Example: {BATCH_SIZE_NOUN} 100 g − other {INGREDIENT}s 72 g = 28 g.")
CALCULATION_USE_FILL_BUTTON = f"Use {FILL_TO_TOTAL.lower()}"
CALCULATION_BATCH_SIZE_CAPTION = (
    f"{BATCH_SIZE_NOUN.capitalize()} means the total {INGREDIENT} amount for "
    f"one {FORMULATION}, including this {INGREDIENT}.")
CALCULATION_EXAMPLES_CAPTION = (
    f"Examples explain the arithmetic, not recommended {INGREDIENT} ratios. "
    "Choose values for your own protocol.")
CALCULATION_TERMS_LABEL = "Supported calculations"
CALCULATION_EMPTY_INFO = (
    "Enter a calculation or choose a different method above.")
def calculation_staged(name):
    """'Water\'s calculation is in the table. Select Save changes to apply
    it.' — said on the tab after the panel closes, because what the panel
    did is a cell two screens up and a button that has just lit."""
    who = f"{name}'s calculation" if name else "The calculation"
    return f"{who} is in the table. Select {SAVE_CHANGES_BUTTON} to apply it."


CALCULATION_APPLY_CAPTION = (
    f"Use this choice to update the table, then select {SAVE_CHANGES_BUTTON}. "
    "Closing this panel leaves the table unchanged.")

# The expression editor's own component: every label, placeholder and
# message it draws.
CALCULATION_EXPRESSION_PLACEHOLDER = (
    f"Enter a calculation or insert an {INGREDIENT} below")
CALCULATION_EXPRESSION_HINT = (
    f"Use {INGREDIENT} amounts, {BATCH_SIZE_NOUN}, numbers and + − × ÷. "
    f"Start typing an {INGREDIENT} name for suggestions.")
CALCULATION_SUGGESTIONS_LABEL = "Ingredient suggestions"
CALCULATION_INSERT_INGREDIENT = "Insert ingredient"
CALCULATION_SEARCH_INGREDIENTS = "Search ingredients"
CALCULATION_BATCH_SIZE_BUTTON = "Batch size"
CALCULATION_BATCH_SIZE_TITLE = (
    f"Total {INGREDIENT} amount for one {FORMULATION}")
CALCULATION_OPERATORS_LABEL = "Insert an operator"
CALCULATION_PERCENT_LABEL = "Percentage"
CALCULATION_PERCENT_PLACEHOLDER = "e.g. 5"
CALCULATION_PERCENT_OF_LABEL = "Percentage of"
CALCULATION_INSERT_PERCENT = "Insert percentage"
CALCULATION_USE_BUTTON = "Use calculation"
CALCULATION_PICK_AN_INGREDIENT = "Choose an ingredient from the list."
CALCULATION_NEEDS_A_PERCENT = "Enter a percentage."


def calculated_range_caption(low_text, high_text, unit):
    """'Calculated range: 12.00–28.00 g, based on the current ingredient
    limits. This is not a measured result.' — two decimals, like every
    other amount the app prints."""
    span = f"{low_text}–{high_text} {unit}".rstrip()
    return (f"Calculated range: {span}, based on the current {INGREDIENT} "
            "limits. This is not a measured result.")


# The sheets of the compact workbook, and the lines that head them:
# Round overview and Results always, Preparation only when the Round has
# pre-mixes (v3).
OVERVIEW_SHEET = "Round overview"
PREPARATION_SHEET = "Preparation"
RESULTS_SHEET = "Results"
OVERVIEW_SHEET_INTRO = (
    f"Review the plan, prepare each {FORMULATION} separately, then enter "
    f"measurements on {RESULTS_SHEET} and upload this file in the app.")


def sheet_navigation(premixes=True):
    """The line that opens each sheet, naming the workbook's tabs in order:
    Preparation only when the Round has pre-mixes and so has that sheet."""
    tabs = ((OVERVIEW_SHEET, PREPARATION_SHEET, RESULTS_SHEET) if premixes
            else (OVERVIEW_SHEET, RESULTS_SHEET))
    where = ("Excel" if on_windows()
             else "Excel and at the top in Numbers")
    return f"Use the sheet tabs: {' → '.join(tabs)}. Tabs are at the bottom in {where}. "


PREPARATION_SHEET_TITLE = f"{PREPARATION_SHEET} — Pre-mixes for this Round"
PREPARATION_SHEET_INTRO = (
    f"Prepare pre-mixes with fixed {INGREDIENT} percentages here. For blends "
    f"where {INGREDIENT} amounts vary separately, follow the individual "
    f"amounts on {OVERVIEW_SHEET}.")
RESULTS_SHEET_TITLE = f"{RESULTS_SHEET} — Enter measurements here"
RESULTS_SHEET_INTRO = (
    f"Each {FORMULATION} has its own column. Blank means not recorded yet. "
    "Enter results, then upload this workbook, check the import and save "
    "in the app.")
ACTUAL_SHEET_HEADING = "Actual amounts and settings — optional"
ACTUAL_SHEET_HELP = ("Blank means the planned value. Enter what was actually "
                     "prepared; these values are used for learning.")
INGREDIENT_OR_SETTING_COLUMN = "Ingredient or setting"
SHEET_LINK_FALLBACK = (
    "If a link does not open, select the sheet tab with the same name.")
# The compact sheet's own spelling of PREMIX_LOT_ON_ITS_PAGE: the pointer
# is a live link to the Preparation sheet, which is where every pre-mix
# page lives once the six sheets are folded into three. The reader skips
# BOTH spellings — printed as a lot, it came back as one.
PREMIX_LOT_ON_PREPARATION = f"See {PREPARATION_SHEET}"
WORKBOOK_METADATA_ERROR = (
    f"This workbook's {ROUND_CAP} details are damaged. Download it again "
    "and copy your entries in.")


def workbook_calculation_without_value(sheet, address):
    """An Excel calculation in a write-in cell that was saved without its
    value (the file never went through Excel): the cell, and both fixes."""
    return (f"Cell {address} on {sheet} has a calculation but no value. "
            "Type the number, or resave the file in Excel.")


def workbook_wrong_round(file_round, open_round):
    """'This workbook is for Round 2; Round 3 is open. Upload Round 3's
    workbook.'"""
    return (f"This workbook is for {ROUND_CAP} {file_round}; {ROUND_CAP} "
            f"{open_round} is open. Upload {ROUND_CAP} {open_round}'s "
            "workbook.")


def workbook_formulation_changed(number):
    """A formulation of the open Round was edited after its workbook was
    downloaded: the sheet would file results under amounts nobody made."""
    return (f"{FORMULATION_CAP} {number} changed after this workbook was "
            "downloaded. Download it again and copy your results in.")


def workbook_sheet_gone(sheet):
    """The new layout's write-in sheet was deleted from the file."""
    return (f"This workbook has no {sheet} sheet. Download it again and "
            "copy your results in.")


def workbook_results_calculation(address):
    """A result cell on the Results sheet holding an Excel calculation."""
    return (f"Cell {address} on the {RESULTS_SHEET} sheet is a calculation. "
            "Type the number, then upload again.")


def workbook_actual_calculation(number, name):
    """An Actual cell holding an Excel calculation."""
    return (f"{FORMULATION_CAP} {number}: the {ACTUAL_COLUMN} cell for "
            f"{name} is a calculation. Type what you weighed.")


# ------------------------------------------------------------------ #
#  Insights (0.8.0)
# ------------------------------------------------------------------ #


TAB_INSIGHTS = "4 · Review"
SEE_WHAT_HAPPENED_NEXT = "See the pilot's Rounds 7–10"
# Where the Review's first score comes from, and the tab that takes it.
INSIGHTS_NONE = f"Score a {FORMULATION} in {TAB_RESULTS} to see the Review."


def _contiguous(rounds):
    try:
        numbers = [int(r) for r in rounds]
    except (TypeError, ValueError):
        return False
    return numbers == list(range(numbers[0], numbers[0] + len(numbers)))


def _span(word, numbers):
    numbers = [str(n) for n in numbers]
    if len(numbers) == 1:
        return f"{word} {numbers[0]}"
    if _contiguous(numbers):
        return f"{word}s {numbers[0]}–{numbers[-1]}"
    return f"{word}s {', '.join(numbers[:-1])} and {numbers[-1]}"


def formulations_span(numbers):
    """"Formulation 7"; "Formulations 7–12"; "Formulations 7, 9 and 12" —
    the same rule as rounds_span, in number order."""
    return _span(FORMULATION_CAP, sorted(int(n) for n in numbers))


CHART_FAILED = "This chart could not be drawn."


# ------------------------------------------------------------------ #
#  0.8.0 speed and simplicity, Plan B (spec 2026-09-30 §§2–6, §9 rev. 2).
#  Every new or changed line of the wave, spelled once here. The lines they
#  replace stay above until the cleanup task deletes them.
# ------------------------------------------------------------------ #

def _and_join(items):
    """'a', 'a and b', 'a, b and c'."""
    items = [str(i) for i in items]
    if len(items) <= 1:
        return "".join(items)
    return ", ".join(items[:-1]) + AND_JOIN + items[-1]


def pairs_text(pairs):
    """How many pairs of Repeats a number rests on (spec review I5, M7; 03
    §5.3: no hedge, the 3-pair rule is stated where it applies)."""
    return "1 pair" if pairs == 1 else f"{pairs} pairs"


def signed(value, decimals=1):
    """'+2.3', '−1.2', '0.0': an error or a gain with its sign (03 rule 8)."""
    text = f"{value:+.{decimals}f}"
    if float(text) == 0:
        return f"{0:.{decimals}f}"
    return text.replace("-", "−")


START_FROM_BEST_SO_FAR = "Start from Best so far"


def still_covering(k):
    return (f"Suggestions keep covering the allowed amounts until {k} {FORMULATION}s have "
            "all their results.")


def waiting_line(n, names_text, many_measurements=False):
    """One line under Generate while formulations wait (V17, M7)."""
    them = "them" if many_measurements or not names_text else "it"
    who = (f"1 {FORMULATION} is" if n == 1 else f"{n} {FORMULATION}s are")
    close = "it" if n == 1 else "them"
    return (f"{who} {waiting_for(names_text)} (enter {them} on Results). The model has not "
            f"used {close} yet, so a suggestion may nearly repeat {close}.")


WORKBOOK_TABS_HELP = ("The workbook has two sheets, Round overview and Results, and a Preparation "
                      "sheet when the Round has pre-mixes. "
                      "Tick “Include individual formulation pages” for one printable page each.")


def workbook_tabs_help(print_pack=False):
    """The download's "?", which follows the box as workbook_line does: the
    file a tick produces has no Round overview, Preparation or Results sheet."""
    if print_pack:
        return (f"The workbook has one {ROUND_CAP} sheet and one printable page per "
                f"{FORMULATION}. Without the tick it has {OVERVIEW_SHEET} and {RESULTS_SHEET}, "
                f"and {PREPARATION_SHEET} when the {ROUND_CAP} has pre-mixes.")
    return WORKBOOK_TABS_HELP


def workbook_sheets(print_pack=False, preparation=False):
    """v3: the one caption under the download (≤ 8 words); the tick is in
    its "?" (workbook_sheets_help). `preparation`: the file being offered has
    a Preparation sheet (a Round with pre-mixes), so the line names it (S6:
    it listed two sheets of a workbook that had three)."""
    if print_pack:
        return f"Sheets: {ROUND_CAP} · one page per {FORMULATION}"
    if preparation:
        return f"Sheets: {OVERVIEW_SHEET} · {PREPARATION_SHEET} · {RESULTS_SHEET}"
    return f"Sheets: {OVERVIEW_SHEET} · {RESULTS_SHEET}"


def workbook_sheets_help(print_pack=False):
    if print_pack:
        return (f"One {ROUND_CAP} sheet and one printable page per {FORMULATION}. Untick for "
                f"{OVERVIEW_SHEET} and {RESULTS_SHEET}.")
    return (f"A {PREPARATION_SHEET} sheet is added when the {ROUND_CAP} has pre-mixes. Tick "
            f"“{PRINT_PACK_CHECKBOX}” for one printable page each.")
RECORD_RESULTS_LINE = "Leave a result blank until it is in."
# Generic about how a panel's number is made (owner, v3 final cleanup): the
# pilot recorded medians, so "the panel mean" was not true of it.
RECORD_RESULTS_HELP = ("One number per measurement; where a panel rated it, the panel's score "
                       "(e.g. a median or mean). Leave blank if the result isn't in yet; you "
                       "can add it later on Results.")
NOT_SCORED_TICK_HELP = f"Tick only if this {FORMULATION} failed and will never be scored."
MORE = "More"


def more_label(not_scored=False, noted=False):
    """The record form's per-formulation fold (M11): 'More', 'More · Not
    scored', 'More · note'."""
    if not_scored:
        return f"{MORE} · {NOT_SCORED}"
    return f"{MORE} · note" if noted else MORE


# Results (§4; spec review I9–I13, M12, M18).
BEST_RULE = "Best so far: highest score with all results in."
BEST_RULE_HELP = ("Best so far: the highest Overall score among formulations with a result for "
                  "every measurement.")
BEST_RULE_WITHOUT = "Best so far: highest score; none has every measurement."
BEST_RULE_WITHOUT_HELP = ("Best so far: the highest Overall score among formulations with all "
                          "their results; none has every measurement yet.")


def score_shown(score):
    """An Overall score as every screen shows it: one decimal, '98.3'
    (owner, v3 final fixes). The data and the exports keep full precision;
    ties are found on that, never on this text."""
    return f"{float(score):.1f}"


def overall_score_line(score, ceiling, without_text=""):
    return (f"{OVERALL_SCORE_COLUMN} {score_shown(score)} of {ceiling:g}"
            f"{scored_without_tail(without_text)}.")


OVERALL_SCORE_HELP = (f"{OVERALL_SCORE_COLUMN}, 0–100: the sum over measurements of closeness "
                      "to the goal (0–1) × Share of score. Compare scores within this project "
                      "only; changing a share, a goal or a Range recalculates every score.")


def runner_up_line(no, score_text, lead_text):
    """Under Best so far: the next best different formulation, as measured
    — 'Next best: Formulation 13, 93.3 (1.7 lower).' (owner, v3 final
    cleanup: no tasting noise, no "within noise", no rise "needed"; N6: it
    sat under a view of another formulation with no label)."""
    return f"Next best: {FORMULATION_CAP} {no}, {score_text} ({lead_text} lower)."


def tie_equal(best_no, other_no, score_text):
    """An exact tie at the top (a lead of 0.00; Plan B parked item): said as
    a tie, not skipped."""
    return f"{FORMULATION_CAP}s {best_no} and {other_no} are tied at {score_text}."


MEASURED_WAITING = "waiting"


def waiting_list(names_text, numbers, round_no=None):
    """Under the Best so far line: 'Waiting for Cook loss: Formulation 22
    (Round 6).' — one line per set of missing measurements (M7)."""
    if len(numbers) == 1:
        where = f"{FORMULATION_CAP} {numbers[0]}" + (
            f", {ROUND_CAP} {round_no}" if round_no is not None else "")
    else:
        where = formulations_span(numbers)
    return f"Waiting for {names_text or 'results'}: {where}."


MOVE_TO_NOT_SCORED_HELP = ("Use when this result will never come. It stays in your records but "
                           "is not scored or used for suggestions. You can undo it.")


def move_to_not_scored_help(n):
    """The Move to Not scored help for `n` missing results (final review
    M-7): one is "this result", more are "these results"."""
    if n == 1:
        return MOVE_TO_NOT_SCORED_HELP
    return (f"Use when these results will never come. The {FORMULATION} stays in your records "
            "but is not scored or used for suggestions. You can undo it.")


def never_coming_warning(no, names_text):
    """The Move to Not scored confirmation (spec review I9)."""
    return (f"{names_text or 'The results'} for {FORMULATION_CAP} {no} will never come? It "
            "stays in your records, unscored, and stops counting in the review.")


EDIT_PAST_FOLD = f"Correct, delete or add past {FORMULATION}s"


def result_view_waiting(label, names_text):
    """A View formulation entry for a formulation still waiting."""
    return f"{label} · {waiting_for(names_text)}"


# Set up and the sidebar (§5; spec review M16, V4, V18–V21).
NEXT_MAKE_A_ROUND = f"Next: {TAB_BATCH_NAME}"
SETUP_FIRST_LINE = "Add your first ingredient in the table, or import them."


def setup_first_line(settings_only=False):
    """A new project's first step (V4). A study of process settings alone
    never adds an ingredient, so it is not told to (spec review I14)."""
    if settings_only:
        return "Add your first process setting in the table, or import them."
    return SETUP_FIRST_LINE
MEASUREMENT_GOAL_CAPTION = "Each measurement's goal and share of score."
MEASUREMENT_GOAL_HELP = "Set a goal for each measurement and how much it counts."


# ------------------------------------------------------------------ #
#  0.8.0 e2e fixes (spec 2026-10-01-e2e-fixes-design.md rev. 2; every
#  "Proposed ruling" of research/2026-10-01-e2e/spec-review.md). The lines
#  W1, W2 and W3 add, spelled once here in the 03 style guide; the lines
#  they change are edited in place above. No later task edits this file.
# ------------------------------------------------------------------ #


def best_repeats_line(best_no, score_texts, mean_text):
    """'Repeats of Formulation 14: 67.3, 66.3, 75.6. Mean of all: 75.2.'
    One decimal, as every screen shows scores (score_shown)."""
    head = "Repeat" if len(score_texts) == 1 else "Repeats"
    line = f"{head} of {FORMULATION_CAP} {best_no}: {', '.join(score_texts)}"
    # Concision round: no bracket; the mean is its own short sentence.
    return f"{line}. Mean of {'both' if len(score_texts) == 1 else 'all'}: {mean_text}."


# B1 (review I9): Fill to total on a blank project and in the ingredient
# file. The import compares a Calculation cell, stripped and lower-cased,
# with these: never reword them (a file written for them stops loading).
FILL_TO_TOTAL_IMPORT_ALIASES = (FILL_TO_TOTAL.lower(), f"= {REST_TOKEN}", f"={REST_TOKEN}")


def rest_needs_batch_size(name):
    """The Fill-to-total row's caption while there is no default batch size
    (03 M4): it replaces 'between 0.00 and −43.00 g'."""
    return (f"{name} fills the remaining amount of the {FORMULATION}. Set a "
            f"{FORMULATION_TOTAL_NOUN} in {MORE_SETTINGS_EXPANDER} to calculate its amounts.")


def import_limits_not_numbers(name, labels=(LOWEST_LABEL, HIGHEST_LABEL)):
    """An ingredient-file row whose Lowest or Highest would load empty (03:
    'Lowest/Highest NaN'), refused by name. `labels`: the cells that are
    not numbers."""
    return named_row_error(name, cells_need_numbers(blank=labels))


def import_spreadsheet_formula(name, label, text):
    """A Lowest or Highest cell holding a spreadsheet formula, read as
    written (the xlsx is read with formulas kept so '= rest' survives)."""
    return named_row_error(name, f"Replace {_cell_text(text)} in {label} "
                                 "with the number itself.")


# B3 (review I10): the restore confirmation says what an open Round loses.
def open_round_lost(round_no, n, typed=0):
    """'Round 6 is open with 4 formulations and 2 results typed; they will
    be lost.' Read before COPY_KEPT, which says a copy is saved first."""
    what = f"1 {FORMULATION}" if n == 1 else f"{n} {FORMULATION}s"
    if typed:
        what += " and 1 result typed" if typed == 1 else f" and {typed} results typed"
    return f"{ROUND_CAP} {round_no} is open with {what}; they will be lost."


# 01 O13 (review M12): the confirm button is not a second copy of the
# button that opened it.
YES_MOVE_TO_NOT_SCORED = f"Yes, move to {NOT_SCORED}"


# --- §2, R-I8, M3: the folds, named by what they hold ----------------
# Final review: one name for one thing, the header link's (METHOD_LINK).
FOLD_METHOD = "Definitions"


# --- review guide ---
# Review as step 4 of the loop (owner, 2026-10-02: "I'm not guided to see
# the review -- it leads me through the 1-3"): Set up -> Make a round ->
# Results -> Review -> Make a round. Results' one lit button after Save
# results, Make a round's one-line status from the cards, the welcome's
# fourth step.
WELCOME_REVIEW_STEP = "4. **Review each Round**: best so far, what next?"


def review_round_label(no):
    """Results' primary after Save results: 'Review Round 5'."""
    return f"Review {ROUND_CAP} {no}" if no is not None else "Review"


REVIEW_ROUND_HELP = (f"Opens {TAB_INSIGHTS}: the best so far, how it compares with the "
                     f"targets, and where the top {FORMULATION}s' amounts sit.")


# --- review plots ---
# The tooltip's expert row title, and its lines.
PLOT_BO = "BO term"
PLOT_BO_MODEL_FREE = "model-free: the measured values"


# --- review four ---
Q2_TITLE = "Are we improving?"


# Q2's x axis: Formulations in the order made, each Round's start a light rule.
Q2_X_TITLE = f"{FORMULATION_CAP} (in order made)"


def round_tick(n):
    """A Round's label above Q2's chart: 'R3'."""
    return f"R{n}"


# --- review story ---
# Review story (prototype, 2026-10-02): the Review read as a story, in the
# order a food scientist asks — Are we improving? Can we trust the
# predictions? What drives each result? What have we tried? Every number is
# read from the project; nothing compares with another method or counts
# Rounds saved. Status lines at most 10 words, captions 8, labels 5.


STORY_ROUND_AVERAGE = f"{ROUND_CAP} average"


# --- review story 2 ---
# Review story 2 (prototype, 2026-10-02; owner: "Should the plots be more
# central and higher up?" — yes): the three cards become one status strip
# above the charts — Progress · Predictions · Next, each a small label, its
# bold word and one short line, the one primary button at its right end —
# and Next becomes the closing section, "What next?", after What have we
# tried?. Status lines at most 10 words, captions 8, labels 5; the detail
# stays in each item's "?". The cards are gone, so the few tooltips that
# named "the Next card" or "the Predictions card" are re-said here (later
# assignments win; the earlier ones stay for the history).
Q5_TITLE = "What next?"


# --- persona fixes ---
# Persona fixes (prototype, 2026-10-02; two demo viewers, a senior food
# scientist and an R&D director): provenance from the note, "Called it?"
# led by the Overall score's order, plain progress and score wording, a
# Predicted column on Results, a credible "What next?", strip and
# tooltips that agree, plain tooltips with the terms of art last.

def own_source(note=None):
    """A row of the user's own, where it came from: its note when it has
    one ("Engine's pick 6.1"); else 'your own formulation' (also for
    the default note the app types itself)."""
    note = str(note or "").strip()
    if not note or note == OWN_FORMULATION_NOTE:
        return OWN_FORMULATION_KIND
    return note


# --- persona fixes 2 ---


# Provenance from the notes: a Round of entered formulations.
ENTERED_ROUND_CAPTION = (
    f"This {ROUND_CAP} holds entered {FORMULATION}s (sources in their notes). For the app's "
    f"suggestions too, click Generate a different {ROUND_CAP}, then Generate."
)


# --- tried gaps ---
# --- round 2 fixes ---


OFF_BY_TITLE = "Measurements against goals"


# Make a round (P3 #17): the button the line names is the one on screen.
ENTERED_ROUND_CAPTION = (
    f"This {ROUND_CAP} holds entered {FORMULATION}s (sources in their notes). For suggestions "
    f"too: {GENERATE_DIFFERENT_BATCH}.")
ONLY_OWN_FORMULATIONS_CAPTION = (
    f"This {ROUND_CAP} holds only your own {FORMULATION}s. For suggestions too: "
    f"{GENERATE_DIFFERENT_BATCH}.")


# One workbook name (P3 #18): the Round workbook.
STEP_PRINT_HEADING = f"##### Download the {ROUND_CAP} workbook"
DOWNLOAD_BATCH_SHEETS = f"Download the {ROUND_CAP} workbook (Excel)"


def back_to_batch_label(no, n):
    """'Back to Round 6' (at most 5 words; the count is on Make a round)."""
    return f"Back to {ROUND_CAP} {no}"


# START_NEXT_BATCH and GENERATE_DIFFERENT_BATCH stay as the owner froze
# them ("Start the next round"; tests/test_080_wording).
ENTERED_ROUND_CAPTION = (
    f"This {ROUND_CAP} holds entered {FORMULATION}s (sources in their notes). For suggestions "
    f"too: {GENERATE_DIFFERENT_BATCH}.")
ONLY_OWN_FORMULATIONS_CAPTION = (
    f"This {ROUND_CAP} holds only your own {FORMULATION}s. For suggestions too: "
    f"{GENERATE_DIFFERENT_BATCH}.")


# --- the Round panel, read in one glance (owner-approved 2026-10-14, spec §10) ---
# The score chart titled plainly; What changed as a picture, not a table: one
# row per ingredient on its own Lowest–Highest bar, a grey dot at Start, a
# blue dot at the best so far, the change in grams at the right. The exact
# amounts sit in each row's tooltip; how to read it, in the title's "?".
PANEL_SCORE_TITLE = BEST_SO_FAR_COLUMN         # the score chart's title: 'Best so far'


# --- sensory profile ---


# Layout study (owner, 2026-10-09: no scrolling while stepping through
# Rounds): one framed block holds the section's three parts at one Round,
# its control at the top; the answer line above it and What next? stay
# about the latest Round.
REPLAY_PREV = "◀"
REPLAY_NEXT = "▶"
# The button says Play: the app never uses the re-play word on screen (it
# would read as a claim about forecasts kept from the time, see
# test_review_rebuilt), and panel is the tasting panel's word.
REPLAY_PLAY = f"Play {ROUND_CAP}s"
REPLAY_STOP = "Stop"
REPLAY_BACK = "Back to latest"
REPLAY_EARLIER = f"Earlier {ROUND_CAP}"
# N13: said in the control's own label, 'Round 5 of 5 (latest)'; a lone
# grey "Latest" pill beside it explained nothing.
REPLAY_LATEST = "latest"


def replay_label(round_no, last):
    """'**Round 4** of 10' (the panel's control); '**Round 5** of 5
    (latest)' on the latest."""
    tail = f" ({REPLAY_LATEST})" if round_no == last else ""
    return f"**{ROUND_CAP} {round_no}** of {last}{tail}"


def replay_to(round_no):
    """A step button's "?": 'Round 3'."""
    return f"{ROUND_CAP} {round_no}"


# --- final demo ---


# B2 · Play.
REPLAY_STEP_TEXT = "one every 2 seconds"


def replay_play_help(first, last):
    """Play's "?": 'Round 1 to 6, one every 2 seconds.'"""
    return f"{ROUND_CAP} {first} to {last}, {REPLAY_STEP_TEXT}."


# ------------------------------------------------------------------ #
#  Review v3 (owner-approved 2026-10-16, final-v3/wording.md §4): Are we
#  improving? as measurements only — a factual headline, Best so far, Best
#  so far vs targets, Amounts in the top formulations — then What next?
#  (buttons only) and Definitions. No verdict, no forecast, no stop.
# ------------------------------------------------------------------ #

def tied_names(names):
    """'F16, F17 tied'; one name alone: 'F16'."""
    names = list(names)
    return ", ".join(names) + (" tied" if len(names) > 1 else "")


def rounds_text(rounds):
    """'Round 6'; several: 'Rounds 4, 5'."""
    rounds = [str(r) for r in rounds]
    if not rounds:
        return ""
    return (f"{ROUND_CAP} {rounds[0]}" if len(rounds) == 1
            else f"{ROUND_CAP}s {', '.join(rounds)}")


def review_headline(best, names, rounds, new=True, was=None, was_names=(), was_rounds=()):
    """The headline under Are we improving?, measured facts only: 'Best
    98.3: F16, F17 tied (Round 6). Was 95.0 (F10, Round 4).'; no new best
    in the Round on view: 'Best 99.2: F21 (Round 7). No new best since.';
    the first Round: 'Best 91.7: F3 (Round 1).'."""
    where = rounds_text(rounds)
    head = f"Best {best}: {tied_names(names)}" + (f" ({where})." if where else ".")
    if not new:
        return f"{head} No new best since."
    if was is None:
        return head
    before = ", ".join(list(was_names) + ([rounds_text(was_rounds)] if was_rounds else []))
    return f"{head} Was {was}" + (f" ({before})." if before else ".")


REVIEW_EACH_FORMULATION = f"Each {FORMULATION}"


def review_each(n):
    """The score chart's legend: 'Each formulation (18)'."""
    return f"{REVIEW_EACH_FORMULATION} ({n})"


def review_top_line(top):
    """The dotted line at the top score: '100 = every measurement on target'."""
    return f"{top} = every {MEASUREMENT} on target"


# Under the score chart, one line in every state (S8): it said "rated once
# by the panel" of a project with no panel, and changed words once a second
# Round was in. The panel, where the project records one (the yogurt pilot),
# is in its "?".
REVIEW_SCORE_CAPTION = f"Each dot: one {FORMULATION}, measured once."


def review_score_help(panel_size=None):
    """The caption's "?": why small differences may not be real, and, for a
    project that records its panel, how each result was made. The scoring
    example is the chart title's "?" (review_score_title_help; N14)."""
    text = (f"Measuring the same {FORMULATION} again could give a slightly "
            "different score, so small differences may not be real.")
    if panel_size:
        text += f" Each result is the median of a panel of {panel_size}."
    return text


def review_score_title_help(definition, example=None):
    """Best so far's "?": how the score is made (Definitions' first line)
    and what one unit off costs, from the project's own best ('Example:
    Tanginess 4 instead of 5 costs 1.7 points.'); no example when every
    measurement is on target."""
    return f"{definition} {example}" if example else definition


def review_cost_example(name, value, ideal, cost):
    """'Example: Tanginess 4 instead of 5 costs 1.7 points.'"""
    unit = "point" if cost == "1.0" else "points"
    return f"Example: {name} {value} instead of {ideal} costs {cost} {unit}."


def review_targets_title(names):
    """'Best so far vs targets (F16, F17 tied)'."""
    return f"Best so far vs targets ({tied_names(names)})"


REVIEW_PANEL_MEDIAN = "Panel median"
REVIEW_MEASURED = "Measured value"


def review_targets_axis(label, low, high):
    """The strip's axis: 'Panel median (0–15)'."""
    return f"{label} ({low}–{high})"


REVIEW_KEY_TARGET = "Target"
REVIEW_KEY_ON = "Best, on target"
REVIEW_KEY_NO_TARGET = "Best, no target"
REVIEW_HIGHER = "higher is better"
REVIEW_LOWER = "lower is better"


def review_target_value(value, target=None, gap=None, goal="target"):
    """A strip row's value: '4 (target 5, −1)', '12 (target 12)', '8
    (higher is better)'. `value` may name tied bests that differ: 'F16 4,
    F17 5'."""
    if goal == "max":
        return f"{value} ({REVIEW_HIGHER})"
    if goal == "min":
        return f"{value} ({REVIEW_LOWER})"
    return f"{value} (target {target}, {gap})" if gap else f"{value} (target {target})"


# The top formulations are the REVIEW_TOP_COUNT best so far, ties at the
# last place included (owner, v3 final cleanup; was "within 5 points").
REVIEW_TOP_COUNT = 3


def review_top_rule(count=REVIEW_TOP_COUNT):
    """'the 3 highest Overall scores so far (ties included)'."""
    return f"the {count} highest {OVERALL_SCORE_COLUMN}s so far (ties included)"


def review_amounts_title(n, count=REVIEW_TOP_COUNT):
    """'Amounts in the 3 best formulations'; with ties at the last place
    (more than `count`): 'Amounts in the top 6 formulations'; one:
    'Amounts in the best formulation'."""
    if n == 1:
        return f"Amounts in the best {FORMULATION}"
    if n <= int(count):
        return f"Amounts in the {n} best {FORMULATION}s"
    return f"Amounts in the top {n} {FORMULATION}s"


def review_amounts_help(count=REVIEW_TOP_COUNT):
    """The amounts title's "?": what Top means and how to read a row."""
    return (f"Top: {review_top_rule(count)}. Each row is one "
            "ingredient or setting between its limits: outlined, allowed but not tried; grey, "
            "the range tried; blue, the range in the top ones; the marks, the best so far.")


def review_amounts_linked(n, total, settings=False):
    """'These 9 ingredients total 100 g, so they move together.' (S9: "The
    9" named nothing.) With process settings drawn in the same chart, which
    are in no total: 'These 9 ingredients total 100 g; settings are
    separate.' (≤10 words; final walk shot 10)."""
    if settings:
        return f"These {n} {INGREDIENT}s total {total}; settings are separate."
    return f"These {n} {INGREDIENT}s total {total}, so they move together."


REVIEW_ALLOWED = "Allowed, not tried"


def review_tried(n):
    """'Tried (18)'."""
    return f"Tried ({n})"


def review_top_key(names, count=REVIEW_TOP_COUNT):
    """'Top 3: F16, F17, F10'; with ties at the last place: 'Top 6 (3
    best, with ties): F21, F16, F17, F19, F23, F26'. Every name inline,
    never cut short and no "(?)" (owner, v3 final cleanup)."""
    names = list(names)
    rule = f" ({count} best, with ties)" if len(names) > int(count) else ""
    return f"Top {len(names)}{rule}: {', '.join(names)}"


def review_best_key(name, tied=False):
    """A best's marker in the key: 'F16'; a tied one after the first: 'F17
    (tied)'."""
    return f"{name} (tied)" if tied else name


REVIEW_AT_UPPER = "at upper limit"
REVIEW_AT_LOWER = "at lower limit"


def review_limit_help(names, upper=True):
    """An 'at upper limit' tag's "?": 'F16, F17 use the most this limit
    allows. You can raise it in Set up.'"""
    names = list(names)
    verb = "uses" if len(names) == 1 else "use"
    most, change = ("most", "raise") if upper else ("least", "lower")
    return (f"{', '.join(names)} {verb} the {most} this limit allows. "
            f"You can {change} it in Set up.")


def start_round_label(n):
    """What next?'s primary: 'Start Round 7'."""
    return f"Start {ROUND_CAP} {n}"


REVIEW_REPLAY_HELP = (f"◀ ▶ step one {ROUND_CAP}; Play shows each in turn. The headline and "
                      f"charts show that {ROUND_CAP}.")


def limit_definition(upper=True):
    """Definitions' entry for the amounts chart's 'at upper limit' tag:
    what the tag says about the amount, nothing about which formulations
    are best (not every top one sits at a limit; owner, v3 fixes)."""
    if upper:
        return "At upper limit: the amount is the highest you allowed; you can raise it in Set up."
    return "At lower limit: the amount is the lowest you allowed; you can lower it in Set up."


def method_definitions(top="100", span="15", count=REVIEW_TOP_COUNT, once=True, limits=()):
    """Definitions (final-v3/wording.md §4), one line per term, from the
    project: its top score, the span its measurements' Range shares (each
    point off costs 1/span of an attribute's share — _compute_utility's
    rule; None when they differ) and how many the top formulations are.
    `once`: no Repeats on file. `limits`: the at-limit tags the amounts
    chart draws ('upper', 'lower'), each defined after the five."""
    cost = (f"1/{span}" if span
            else f"1/({HIGHEST_MEASURABLE_LABEL} − {LOWEST_MEASURABLE_LABEL})")
    best = f"the highest {OVERALL_SCORE_COLUMN} so far."
    if once:
        best += f" Each {FORMULATION} is made once and measured once."
    return (
        f"{OVERALL_SCORE_COLUMN}: {top} = every {MEASUREMENT} on target. Each {MEASUREMENT} "
        f"earns its share; each unit off costs {cost} of it.",
        f"{ROUND_CAP}: the {FORMULATION}s made and tasted together.",
        f"{BEST_SO_FAR_COLUMN}: {best}",
        f"Top {FORMULATION}s: {review_top_rule(count)}.",
        "Suggestions: best bets to improve on the best so far, weighing how good they look "
        "and how unsure the model is.",
    ) + tuple(limit_definition(kind == "upper") for kind in ("upper", "lower")
              if kind in limits)


