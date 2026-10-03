"""Every word the user reads. Change a word here and it changes everywhere;
the tests read this module too.

Use sentence case for headings, buttons and standalone statuses. Start a
separate heading or command after a middle dot with a capital letter. Keep
user-entered names, units, calculation tokens and legacy import labels unchanged.
Prefer direct instructions and current control names over idioms.
"""

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
    return f"Saved automatically at {when}, on this Mac"


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
    return f"A project named {name} already exists. Open it below."


NAME_COLLIDES_WITH_ARCHIVE = ("That name is already used by a project or an "
                              "archived copy. Choose another.")

OPEN_PROJECT_LABEL = "Open project"

TRY_SAMPLE_LABEL = "Try the sample project"
TRY_SAMPLE_HELP = (
    "Explore one guided burger example with ingredients, blends, calculated water, "
    "a process setting and measurements."
)

# The sample project's own name, so app.py and ui_setup.py agree on how it is
# recognised: by name, the same way the app has always told it apart from a
# project the user made.
SAMPLE_PROJECT_NAME = "Sample project"

# The sample's own targets_source, set once when it is built.
LEGACY_SAMPLE_TARGETS_SOURCE = (
    "Illustrative training example; not a validated formulation or recorded panel study. "
    "Firmness 6/10, juiciness 7/10 and their shares of the score are example choices. "
    "Scientific background: Peñaranda et al. (2025), Plant-Based Burgers with Reduced "
    "Texture Additives, https://doi.org/10.3390/foods14081373. That study compares "
    "binders in soybean burgers; this pea-and-soy formulation, allowed values and targets are "
    "adaptations, not values validated by that paper. Establish sensory anchors "
    "and suitable allowed values with your own pilot work. Ingredient property figures "
    "are illustrative; replace them with supplier data. The hydration ratio of 2.2 is an "
    "illustrative assumption; establish a suitable ratio for the selected protein grades in pilot trials."
)
# Where firmness 6 and juiciness 7 came from. A trained panel handed a
# 0-10 scale with no anchor and no reference product produces data that does
# not replicate between sessions, which is the failure this field exists to
# prevent — and what shipped in 0.7.1 was a disclaimer that answered "is
# this real?" and never answered "where did the numbers come from?". The
# disclaimer is kept, at the foot of the method, where a method's
# reservations belong.
SAMPLE_TARGETS_SOURCE = (
    "Scores are against an 80/20 beef control cooked to 71 °C core, which our panel rates "
    "firmness 6 and juiciness 7 on the anchored 0–10 scale (12 panellists, two sessions, "
    "February 2026). Above firmness 7 the patty eats rubbery; below juiciness 5 it eats dry "
    "and chalky. Contains soy and wheat (gluten)."
)
SAMPLE_REFERENCE = (
    "Background reading: Peñaranda et al. (2025), Plant-Based Burgers with Reduced Texture "
    "Additives, https://doi.org/10.3390/foods14081373. This paper provides background; "
    "it does not validate this example."
)


# The sample's Method: how the bench makes one formulation, in the order it
# is done. Printed on the Round sheet under the title, because three
# formulations that must be made identically except for the amounts had
# nothing on paper saying how.
SAMPLE_METHOD = (
    "1. Hydrate the textured pea and soy proteins in the Hydration water only "
    "(2.2 × their combined mass), 10 min at 45 °C, covered. Do not drain.\n"
    "2. Put the Remaining water in the bowl at ≤ 5 °C. Add the Dry blend, Wheat gluten "
    "and Seasoning blend and mix 60 s on low. (The dry blend carries the "
    "methylcellulose: it hydrates only cold.)\n"
    "3. Add the hydrated proteins. Mix 30 s on low.\n"
    "4. Add the fats and oils, coconut oil chopped to ~5 mm, and mix for this "
    "formulation's Mixing time after fat.\n"
    "5. Form 100 g patties, 100 mm across, 12 mm thick. Chill 20 min at 4 °C, covered.\n"
    "6. Griddle at 180 °C, 3 min per side, to 74 °C core. Serve within 3 min.\n"
    "Water temp at addition ____ °C (≤ 5)     Mass temp out of the bowl ____ °C (≤ 12)\n"
    "Raw weight ____ g     Cooked and blotted weight ____ g     "
    "Cook loss = (raw − cooked)/raw × 100\n"
    "Contains soy and wheat (gluten).\n"
    "Every formulation is made to the batch size printed above. These amounts and "
    "targets demonstrate the app; confirm a suitable preparation and measurement "
    "protocol before using them for real trials."
)


# The project-level Method: one text area in More settings, printed on the
# Round sheet under the title.
# The four things a project can record beside what it is asked to. All four
# are off for a new project: Vendor and SKU are specification data typed
# once and never looked at again, and Lot and Actual are two more columns
# down every printed page of a project that does not want them.
def no_such_field(name):
    """A caller asking the project to record something it has no box for."""
    return f"{name} is not an available recording field."


ALSO_RECORD_LABEL = "Additional records (optional)"
ALSO_RECORD_HELP = ("These options add entry fields in the app or "
                    "workbook. Saved values are retained when an option is "
                    "turned off.")
RECORD_FIELD_LABELS = {
    'vendor': "Vendor",
    'sku': "SKU",
    'lot': "Lot",
    'actual': "Actual amounts",
}


def record_field_on(label):
    return f"{label} recording enabled."


def record_field_off(label):
    return f"{label} recording disabled. Saved values are retained."


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
    """'Saved as "burger copy 2026-09-15.json" in your Downloads folder.'

    A download is the one action in the app that leaves no mark on the
    screen: the reader clicked, nothing moved, and the heading above it
    promises a list they could not see themselves in."""
    return f"Saved as '{file_name}' in your Downloads folder."
# A saved copy that will not open. ONE sentence, whatever is wrong with
# it: the reader is a food scientist whose copy will not open, and the
# stored field name, its shape and the programmer's quotation marks are
# for the log (food_bo._damaged writes them there).
COPY_DAMAGED = "This copy is damaged and cannot be opened."
NOT_A_COPY = "This file is not a Food Optimizer copy."
COPY_FROM_A_NEWER_VERSION = (
    "This copy was made with a newer version of Food Optimizer. Update the "
    "app, then try again.")

COPY_UNREADABLE = (
    "This file could not be read as a Food Optimizer copy. "
    "If you have another copy, try that one; recent copies are "
    "saved in your FoodOptimizer folder."
)
RECENT_COPIES_HINT = (" Recent copies of your own projects are saved in "
                      "your FoodOptimizer folder.")


def open_saved_copy_warning(name, holds, project_name, held, open_text=""):
    """`holds` is the list of 'N formulations'/'N ingredients'/'N process
    settings' phrases the saved copy contains; `held` is the same phrase for
    what replacing it would give up. `open_text` (0.8.0 e2e fixes B3):
    open_round_lost's sentence while a Round is open."""
    lost = f"{open_text} " if open_text else ""
    return (f"This copy holds **{name}**: " + ", ".join(holds)
            + f". Replace **{project_name}**, which has {held}? " + lost + COPY_KEPT)


YES_REPLACE = "Yes, replace"
COPY_APPLY_FAILED = ("This copy could not be applied. Your current "
                     "project was not changed.")


def restored_flash(count_text, project_name, archived=None):
    text = f"Restored {count_text} into {project_name}."
    if archived:
        text += f" A copy was saved as {archived}."
    return text


MANAGE_PROJECT = "Manage project"

START_OVER_LABEL = "Clear project contents"
YES_START_OVER = "Yes, clear contents"


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
    return f"A copy was saved as {archived}."


def project_deleted_flash(name, archived=None):
    if archived:
        return f"Deleted {name}. A copy was saved as {archived}."
    return f"Deleted {name}."


OPEN_BUTTON = "Open"


# ------------------------------------------------------------------ #
# Welcome panel (no project open yet).
# ------------------------------------------------------------------ #
WELCOME_HEADER = "## Create your first project"


def welcome_steps():
    return (
        "1. **Name a project** in the sidebar on the left.\n"
        "2. **Set up ingredients and process settings**, then choose measurements and targets.\n"
        "3. **Generate formulations**, prepare each one, and record your measurements.\n"
        + WELCOME_REVIEW_STEP      # review guide: Review is the loop's fourth step
    )


DOWNLOAD_TEMPLATE = "Download ingredients template (Excel)"


# ------------------------------------------------------------------ #
# Load-error / save-error banners, and the discarded-batch notice.
# ------------------------------------------------------------------ #
def project_load_error_info():
    return (
        f"This project file is damaged and cannot be edited. In the sidebar, use {OPEN_A_SAVED_COPY} "
        f"to restore a copy, or {MANAGE_PROJECT} › {START_OVER_LABEL} to clear its contents. "
        "The damaged file is copied first."
    )


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


def sample_project_rebuilt(name):
    """'Sample project put back the way it started.' — clicking Try the
    sample project again on a sample nobody has made a round from rebuilds
    it, which throws away every edit made to it. It said only "Opened
    Sample project.", so the reader\'s own calculations and amounts were gone with
    nothing on screen about it."""
    return f"{name} restored to its original settings."


def project_opened(name):
    return f"Opened {name}."


def sample_project_failed(err):
    return f"The sample project could not be created: {err}"


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
GENERATE_FAILED = 'The app could not generate formulations. Try requesting fewer formulations. If the problem persists, review recently added limits.'


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
ADD_OWN_EXPANDER = f"Add a {FORMULATION} to this {ROUND}"
ADD_OWN_NO_BATCH_CAPTION = "For suggestions too, click Generate first."
ADD_OWN_NO_BATCH_HELP = "Want the app's suggestions too? Click Generate first, then add yours."
OWN_NOTE_PLACEHOLDER = "e.g. Repeat of 4 with more salt"
ADD_TO_THIS_BATCH = f"Add to this {ROUND}"
ENTER_EVERY_AMOUNT = ("Enter every amount. Type 0 for an ingredient you "
                      "are leaving out.")
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
# same three words by the same distance rule (why_role_rule), so one
# suggestion is never "trying something different" in one table and
# "refines the best" in the other. ROLE_REFINES, ROLE_BALANCED and
# ROLE_EXPLORES further down are these three.
SUGGESTION_CLOSE = "refines the best"
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
# An example, not a description: the help above already says what the box is
# for.
BATCH_SIZE_PLACEHOLDER = "e.g. 100"

NO_BATCH_SIZE_OF_ITS_OWN = f"This {ROUND_CAP} has no {BATCH_SIZE_NOUN} of its own."


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


GENERATE_DIFFERENT_BATCH = f"Discard this {ROUND} and generate again"


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


NOTHING_TO_SAVE = "Nothing to save — at least one formulation needs results."
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
UPLOAD_HELP_CAPTION = "Upload the Round sheet with Measured cells filled."
UPLOAD_HELP_DETAIL = (
    "Fill in the Measured cells on the Round sheet you downloaded above and "
    "upload the file here. Formulations are matched on their number."
)
# The same door, said the other way round, once the workbook has actually
# been downloaded in this session: the fold is open and this is the line
# inside it, so the reader coming back with a filled-in file finds the door
# already ajar rather than collapsed under a heading beginning "Or".
UPLOAD_SHEETS_ARE_BACK_CAPTION = "Upload the completed workbook here."
UPLOAD_SHEETS_ARE_BACK_HELP = ("Upload the completed workbook here. The app imports entries "
                               "from the input cells.")
UPLOAD_RESULTS_FILE = "Upload results (Excel or CSV)"
CHECK_THIS_FILE = "Check this file"
FILE_UNREADABLE = (
    "This file could not be read. Upload the workbook you downloaded above, "
    "or a spreadsheet saved from it."
)


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
    "Each suggestion says whether it refines the best, explores, or is "
    "balanced between the two, and what it changes.",
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
ADD_BASELINE_ERROR = ("Enter the baseline: the setting you used for every "
                      "formulation already made.")


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
    it away without a word is how a grid loses an edit."""
    return f"Only a process setting has a {column}."


def only_an_ingredient_has(column):
    return f"Only an ingredient has a {column}."


NAME_REQUIRED_ERROR = "Name cannot be empty."
NUMBER_REQUIRED_ERROR = "Enter a number."
SHARE_REQUIRED_ERROR = "Enter a share above 0."
TYPE_LOCKED_ERROR = ("Type cannot change once formulations have been "
                     "recorded. Delete the row and add it again.")


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
PROPERTIES_HEADING = f"**{PROPERTIES_NAME}**"


def properties_grid_caption(per_100_already_said=False):
    """'Each ingredient's figure, per 100 g. An empty cell counts as 0 in
    any limit.'

    `per_100_already_said` drops the basis from the sentence: a project whose
    property names carry it themselves ("Fat per 100 g and Sodium per 100 g")
    would otherwise say it twice in one line.
    """
    basis = "" if per_100_already_said else ", per 100 g"
    return f"Each ingredient’s property value{basis}. " + PROPERTY_BLANK_RULE


SAVE_BUTTON = "Save"
SAVE_PROPERTIES_BUTTON = "Save changes"
CLOSE_BUTTON = "Close"
PROPERTIES_SAVED = "Properties saved."
DELETE_PROPERTY_PICK_LABEL = "Property to delete"


def property_not_a_number(name):
    """One cell of the properties grid holding something that is not a
    number. The column is named: a grid of six properties gives the reader
    nothing else to go on."""
    return f"{name} must be a number, or empty."


def no_such_ingredient(name):
    return f"No ingredient named {name}."


UNIT_REQUIRED_ERROR = "A unit is required; use g if the amount is a mass."


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


def updated(name):
    return f"{name} updated."


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
# grid that names one is below this line, so the sentence points down.
NO_PROPERTIES_YET_CAPTION = ("Name a property under Properties below to "
                             "limit it here.")
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
ENTER_LOWEST_HIGHEST_ERROR = (f"Enter {AT_LEAST_LABEL.lower()}, "
                              f"{AT_MOST_LABEL.lower()}, or both.")
LIMIT_BOUNDS_ORDER_ERROR = f"{AT_LEAST_LABEL} must be less than {AT_MOST_LABEL}."


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
MORE_SETTINGS_EXPANDER = "Preparation and records"
ADVANCED_EXPANDER = "Advanced"

LIMITS_HEADING = "**Limits (optional)**"
# The property rule lives here, not in the closeness fold: a property never
# touches closeness — it feeds limits and nothing else.
# The blank-figure rule is NOT repeated here. It used to be, because the
# properties were a fold somewhere else on the tab; they are now a grid a
# few lines below this line, under a caption that says it, and each limit's
# own line names the ingredients it is reading as zeroes.
LIMITS_CAPTION = "Generated formulations meet these limits."
LIMITS_HELP = ("Generated formulations satisfy the limits listed here. "
                  "A formulation of your own is recorded as you "
                  "typed it. A limit is on an amount you weigh out or a "
                  "property of your ingredients; measurements have goals and "
                  "targets instead.")


def old_limit_basis_caption(unit):
    return f"An older limit is now read per 100 {unit} of formulation."


def at_least(value, unit=""):
    return f"at least {value:g} {unit}".rstrip() if unit else f"at least {value:g}"


def at_most(value, unit=""):
    return f"at most {value:g} {unit}".rstrip() if unit else f"at most {value:g}"


def delete_limit_button(who):
    """'Delete limit on Fat per 100 g'. A bare `Delete limit` was drawn once
    per limit, identically, with nothing on the button to say which."""
    return f"Delete {LIMIT} on {who}"


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
    entirely now — the box at the top of More settings is where the number
    is answered — so the sheet says where it came from."""
    return f"{FORMULATION_TOTAL_NAME} · {total_text} (set in Set up)"


def round_stays_at(size_text):
    """The second half of a refused batch size: what the round, the table and
    the sheets are still made to, so the number left in the box is never
    mistaken for the one under it."""
    return f"The {ROUND_CAP} stays at {size_text}."


def total_not_reachable_at_most(total_text, most_text,
                                noun=FORMULATION_TOTAL_NOUN):
    """A size above everything the allowed amounts can add up to. The two
    numbers are the whole answer: nothing about the search can rescue a sum
    that has no solution, and the fix is to raise an ingredient's Highest.

    `noun` is which of the two boxes is refusing — Set up's default, or the
    round screen's own — so the sentence names the box the reader just
    typed into rather than the other one."""
    return (f"A {noun} of {total_text} is not reachable: the most these "
            f"ingredients can make is {most_text}.")


def total_not_reachable_at_least(total_text, least_text,
                                 noun=FORMULATION_TOTAL_NOUN):
    return (f"A {noun} of {total_text} is not reachable: the least these "
            f"ingredients can make is {least_text}.")


def total_not_reachable_at_all(total_text, noun=FORMULATION_TOTAL_NOUN):
    """A size of nothing. Reachable arithmetic — every amount can be 0 in a
    project with no lower bounds — and still not a formulation, so it is
    refused in the same shape as a size the amounts cannot make."""
    return (f"A {noun} of {total_text} is not reachable: every formulation "
            "has to add up to something.")


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
    wrong weight. Nothing can be widened and nothing can move, so the
    sentence is the two numbers and no advice."""
    return (f"The fixed amounts add up to {made_text}, not the "
            f"{total_text} {BATCH_SIZE_NOUN}.")


def fixing_breaks_the_total(total_text):
    """A row whose Lowest is its Highest is one amount, and one amount can
    put the batch size out of reach of the rows still moving. The two ways
    back are the size and the allowed values — not the eight ingredients the size's
    own limit happens to name."""
    return (f"Fixing these would leave no formulation adding up to "
            f"{total_text}. Change the {BATCH_SIZE_NOUN}, or let enough "
            f"ingredients vary again.")


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
    return f"Invalid JSON: {e}"


IN_USE_PREFIX = "In use: "


# ------------------------------------------------------------------ #
# Tab 3 · Results: the best formulation, every formulation, corrections
# (ui_results.py).
# ------------------------------------------------------------------ #
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


COMPLETE_RESULTS_CAPTION = complete_results_caption()
NO_COMPLETE_FORMULATION = "No formulation has all its results yet."


# 'Overall score' as a noun inside a sentence, and the scale it is on. The
# ceiling is a whole number — a formulation that hits every goal scores
# 100 — so it is written as one.
OVERALL_SCORE_LOWER = "overall score"


def batch_recorded_progress(no, before, now):
    """'Round 2 recorded · best overall score 86.60 → 88.10 of 100'.

    The number had no name and no scale, beside a sibling line that did
    carry the noun. Both say the same thing now, in the same words."""
    return (f"{ROUND_CAP} {no} recorded · Best {OVERALL_SCORE_LOWER} "
            f"{before:.2f} → {now:.2f}")


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


def not_scored_option(no):
    """'3 · not scored' — how a formulation with no result reads in the
    correction picker. The list is one run of numbers, and nothing on it
    said which of them were being scored for the first time."""
    return f"{no} · {NOT_SCORED}"


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


def back_to_batch_label(no, n):
    return f"Back to {ROUND_CAP} {no} · {n} to record"


START_NEXT_BATCH = f"Start the next {ROUND}"

NO_RESULTS_YET = "No results yet."
OVERALL_SCORE_COLUMN = "Overall score"
BEST_SO_FAR_COLUMN = "Best so far"
# C12: the All formulations sheet's date column. "Recorded" was a state
# everywhere else on the screen ("Round 1 · recorded", "2 to record") and
# a date only here.
DATE_RECORDED_COLUMN = "Date recorded"

# 0.8.0 review fixes (§9 I4): the Progress chart draws a formulation scored
# without a measurement (recorded before it existed) as a hollow dot; the
# caption names it only when there is one.
PROGRESS_RING = 'A hollow dot is a formulation scored without every measurement.'


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
WHOLE_BATCH_LABEL = f"Select a whole {ROUND} for deletion"
CHOOSE_A_BATCH_PLACEHOLDER = f"Choose a {ROUND}"


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


def missing_columns(names_text):
    return f"Missing columns: {names_text}"


def blank_amount_columns(names_text):
    return f"These amount columns have blank cells: {names_text}"


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
    bottom that has no name yet."""
    return f"{name}: {problem}"


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
DOWNLOAD_BATCH_SHEETS = f"Download the {ROUND} sheets (Excel)"

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

WORKBOOK_UNREADABLE = ("This file could not be read as a workbook. Upload "
                       "the file you downloaded from this " + ROUND + ".")


def workbook_sheet_missing(wanted, found_text):
    """The sheet named after the open batch is not in the uploaded file —
    usually last week's workbook, downloaded twice. It names the sheet it
    looked for and the ones it found, because both are on screen nowhere
    else."""
    return (f"This workbook has no sheet called {wanted}. It has "
            f"{found_text}. Download the sheets for {wanted} and fill "
            f"those in.")


def workbook_no_formulations(wanted):
    return (f"The {wanted} sheet has no formulation columns. Upload the "
            f"file you downloaded from this {ROUND_CAP}.")


def workbook_nothing_filled_in(wanted):
    return (f"Nothing is filled in on the {wanted} sheet. Write a number "
            f"in a Measured cell, or tick {NOT_SCORED}.")


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
RULE_GRID_CAPTION = "Calculated amounts use the Calculation column; their Lowest and Highest cells are not used."
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
EXACTLY_LABEL = "Exactly"
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


def limit_on_worked_out_rows(names_text, many=False):
    """'Water and Salt are calculated from their calculations, so this limit
    cannot change them.' — every row of a limit filled in by a calculation leaves
    the limit nothing to act on, and the app accepted it and then let the
    calculations contradict it."""
    verb, tail, them = (("are", "their calculations", "them") if many
                        else ("is", "its calculation", "it"))
    return (f"{names_text} {verb} {WORKED_OUT} from {tail}, so this limit "
            f"cannot change {them}.")


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
SHEET_RETURN_PATH = (f"Then upload this file in the app: {TAB_BATCH} \u2192 "
                     f"Save results \u2192 Or upload results from a file.")
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


SUMMARY_SHADED_NOTE = summary_write_in_note()


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


SHEET_SHADED_NOTE = sheet_write_in_note()

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
            f"{name} cannot be less than zero.")


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
    return (f"{name} is a column name Food Optimizer uses for its own "
            f"tables. Choose another name, for example {name}s.")


def reserved_name_short(name):
    """The same refusal for a measurement, which has no plural to suggest:
    'Firmnesss' is not a name anybody wants."""
    return (f"{name} is a column name Food Optimizer uses for its own "
            f"tables. Choose another name.")


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
    """'Formulation 4 Firmness' — how one cell of an uploaded sheet is named
    inside the refusal that says its value is out of range."""
    return f"{FORMULATION_CAP} {number} {name}"


# The two sentences the model still needs and the grid already had.
LAST_VARYING_ROW_ERROR = (
    f"Cannot delete the last {INGREDIENT} or setting that can still vary. "
    f"Give another one a different {LOWEST_LABEL} and {HIGHEST_LABEL} "
    "first.")
AMOUNTS_MISSING_DELETE_ERROR = (
    "Cannot delete this: some formulations were recorded without their "
    "amounts, so what was made cannot be recalculated. Fix it at one "
    "amount instead, or clear the project contents.")

LOWEST_ABOVE_HIGHEST_ERROR = "Lowest cannot be above Highest."
RANGE_ENDS_ERROR = f"{LOWEST_MEASURABLE_LABEL} must be less than {HIGHEST_MEASURABLE_LABEL}."
TARGET_REQUIRED_ERROR = (f"Enter a target value for a "
                         f"'{GOAL_LABELS['target']}' measurement.")


def target_outside_message(target, low, high):
    """R13: the two controls the reader can reach, named as the grid names
    them. "the range's lowest and highest" named nothing on that screen."""
    return (f"{TARGET_LABEL} {target:g} must be between "
            f"{LOWEST_MEASURABLE_LABEL} and {HIGHEST_MEASURABLE_LABEL} "
            f"({low:g} to {high:g}).")


def baseline_outside_message(value, low, high):
    return (f"{BASELINE_LABEL} {float(value):g} must be between {low:g} and "
            f"{high:g}.")


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
    return (f"The ingredients file is missing required column(s): {named}. "
            f"Expected columns: {NAME_LABEL}, {LOWEST_LABEL}, "
            f"{HIGHEST_LABEL} (plus an optional {UNIT_LABEL} column and "
            f"property columns like Cost or Protein).")


def file_row_name_blank(row_no):
    return f"{ROW.capitalize()} {row_no}: the {NAME_LABEL} cell is blank."


def file_row_duplicate_name(row_no, name):
    return (f"{ROW.capitalize()} {row_no}: duplicate {INGREDIENT} name "
            f"{name}.")


def file_row_reserved_name(row_no, name):
    return f"{ROW.capitalize()} {row_no}: {reserved_name_message(name)}"


def file_amounts_not_numbers(name):
    return (f"{INGREDIENT.capitalize()} {name}: {LOWEST_LABEL} and "
            f"{HIGHEST_LABEL} must be numbers. Please check that column for "
            f"text or blank cells and try again.")


def file_lowest_above_highest(name, low, high):
    return (f"{INGREDIENT.capitalize()} {name}: {LOWEST_LABEL} ({low}) "
            f"cannot be above {HIGHEST_LABEL} ({high}).")


# An uploaded results sheet.
SHEET_NEEDS_A_FORMULATION_COLUMN = (
    f"The sheet needs a {FORMULATION_CAP} column with the numbers from the "
    f"{ROUND_CAP} sheets you downloaded.")
SHEET_HAS_NO_ROWS = "The sheet has no result rows."


def sheet_missing_columns(named):
    return f"Missing columns: {named}"


def formulation_number_not_whole(raw):
    return f"{FORMULATION_CAP} number {raw} is not a whole number."


def formulation_not_in_round(number, round_no, holds):
    return (f"{FORMULATION_CAP} {number} is not in {ROUND_CAP} {round_no} "
            f"(it has {holds}).")


def formulation_twice_in_the_sheet(number):
    return (f"{FORMULATION_CAP} {number} appears more than once in the "
            f"sheet.")


def formulation_value_not_a_number(number, name):
    return f"{FORMULATION_CAP} {number} {name} is not a number."


def formulation_has_no_measurements(number):
    return f"{FORMULATION_CAP} {number} has no measurements filled in."


def formulation_is_not_not_scored(number):
    """A number handed to Score a formulation that is not one of the rows
    waiting for a result."""
    return (f"{FORMULATION_CAP} {number} is not waiting to be scored in "
            f"this project.")


# Limits and units.
def property_limits_need_a_mass_unit(unit, fix_sentence):
    return (f"Property limits are per 100 {unit}, so every {INGREDIENT} "
            f"needs a mass unit; {fix_sentence}")


def limit_needs_one_unit(fix_sentence):
    return (f"A {LIMIT} adds amounts, so these {INGREDIENT}s need one unit; "
            f"{fix_sentence}")


# Generating, recording, deleting.
EVERYTHING_IS_FIXED = (
    "Every ingredient and process setting is fixed at one amount. Give at "
    f"least one of them different {LOWEST_LABEL} and {HIGHEST_LABEL} "
    "amounts before generating formulations.")
ADD_A_MEASUREMENT_FIRST = "Add at least one measurement before saving results."


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
    return (f"Fixing these would make the {LIMIT} on {what} impossible to "
            f"meet: the {INGREDIENT}s that can still vary only reach "
            f"{most_text} at most. {LOOSEN_THE_LIMIT_FIRST}")


def limit_unreachable_below(what, least_text):
    """The same for a limit's At most that the pinned rows already exceed."""
    return (f"Fixing these would make the {LIMIT} on {what} impossible to "
            f"meet: the {INGREDIENT}s that can still vary cannot get below "
            f"{least_text}. {LOOSEN_THE_LIMIT_FIRST}")


def limit_pinned_amounts_exceed(what, pinned_text):
    """An amount limit whose At most is already passed by the amounts this
    save pins — nothing that can still vary is even involved."""
    return (f"Fixing these would make the {LIMIT} on {what} impossible to "
            f"meet: the amounts pinned already add up to {pinned_text}. "
            f"{LOOSEN_THE_LIMIT_FIRST}")


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
PROJECT_NOT_FOUND = ("This project could not be found. It may have been "
                     "renamed or archived.")
PROJECT_FILE_DAMAGED = (
    f"This project file is damaged and could not be opened. If you saved a "
    f"copy, use {OPEN_A_SAVED_COPY} in the sidebar; otherwise look in your "
    "FoodOptimizer folder for a recent copy.")


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
FORMULA_REST_ALONE = ("Write = rest on its own: it is whatever is left of "
                      "the batch size.")
FORMULA_TWO_AMOUNTS = ("A calculation can add or subtract amounts and multiply "
                       "by a number. It cannot multiply two amounts.")
FORMULA_DIVIDE_BY_AMOUNT = ("A calculation can divide by a number, not by an "
                           "amount.")
FORMULA_DIVIDE_BY_ZERO = "A calculation cannot divide by zero."
# Every word the grammar really takes, in both the alphabet the spec
# prints and the one on the keyboard. The old sentence listed four
# characters, three of them untypable, and omitted the leading '=', the
# percentage, 'batch size' and 'rest' — the four things a reader who is
# stuck most plausibly got wrong.
FORMULA_UNREADABLE = ("This calculation could not be read. Use =, + − × ÷ (or "
                      "- * /), %, numbers, brackets, batch size, rest and "
                      "ingredient names.")
RULE_NEEDS_EQUALS = "Start a calculation with =."
RULE_USES_ITS_OWN_ROW = "A calculation cannot use its own row."
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
    return f"There is no ingredient called {name}."


def formula_loop(chain):
    """'A calculation cannot lead back to itself: Fat → Water → Fat.' — `chain`
    is the arrow-joined names that already spell the loop out."""
    return f"A calculation cannot lead back to itself: {chain}."


FORMULA_NEEDS_BATCH_SIZE = (
    f"Set a {FORMULATION_TOTAL_NOUN} to calculate this amount. Set one "
    "in Preparation and records, or enter amounts without a calculation.")
PERCENT_NEEDS_BATCH_SIZE = (
    f"There is no {FORMULATION_TOTAL_NOUN} to take a percent of. Set one "
    "in Preparation and records.")


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
    return (f"A {noun or FORMULATION_TOTAL_NOUN} of {size_text} leaves "
            f"{name} below {zero}. The other {INGREDIENT}s need at least "
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
    return (f"Only one row can be = {REST_TOKEN}: {names_text} "
            f"{'all' if many else 'both'} are.")


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


FORMULA_ROW_NOTE = (f"A row marked {WORKED_OUT} is filled in from its "
                    f"{FORMULA_IN_RANGE}. Weigh the amount printed.")


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

CORRECTIONS_ON_RESULTS_CAPTION = (
    "Correct what you made in Results, after you record it.")


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
    return (f"{name}'s amount is the {SUM_OF_ITS_PARTS}. Change the "
            f"{PART_LABEL.lower()}s in its fold.")


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
    return (f"Row {row_no} puts it in {name}, and no row of this file says "
            f"how {name} is made.")


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
PARTS_ADD_TO_NOTHING = ("The composition percentages total zero. Give at least one of "
                        "them a composition percentage above zero.")
PREMIX_NEEDS_A_PART = "Enter at least one name in the Part column before saving."
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


PREMIXES_HEADING = "Pre-mixes"
# Two headings, not one. `To weigh for this round` totalled two kinds of
# number under one instruction: a pre-mix's parts ARE weighed together at
# the number printed, and a loose row's total is three formulations added up
# and is never weighed at that number anywhere. A tired bench operator
# weighed out 167.52 g of water.
MAKE_FOR_ROUND_HEADING = "Make for this round"
HAVE_ON_HAND_HEADING = "Have on hand"
HAVE_ON_HAND_CAPTION = (f"Across every {FORMULATION}; a {WORKED_OUT} row's total "
                        "is what the round comes to, not one weighing.")
ROUND_TOTAL_COLUMN = "Total for this round"
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


PREMIX_SHADED_NOTE = premix_write_in_note()


def premix_sheet_title(name, total_text, need_text=""):
    """'Seasoning blend · make 100 g (this round needs 6.60 g)'.

    `make 7.50 g` was the worst number in the workbook: you cannot blend
    7.5 g of coarse salt and four fine powders to any homogeneity, and you
    cannot portion three 2.5 g scoops out of it without the salt segregating
    to the bottom. Nor can you dispense a blend with nothing left in the
    bowl, on the paddle or on the scoop. So the page asks for a makeable
    quantity and says what the round takes out of it.
    """
    line = f"{name} · Prepare {total_text}"
    return f"{line} (this round needs {need_text})" if need_text else line


def premix_sheet_name(name):
    return f"Pre-mix · {name}"


def premix_group_line(name):
    return f"{name} · weighed into this formulation"


def premix_group_total_line(name):
    return f"{name} · total (not weighed)"


def workbook_lot_conflict(name):
    return f"{name} has different lot numbers on two sheets. Use the same lot number for it throughout this round."


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


# Round editing and compact workbook workflow.
EXAMPLE_PROJECT_LABEL = 'Example project'
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
TEACHING_EXAMPLE_CAPTION = 'Teaching example with illustrative values. See Preparation and records for the method and assumptions.'
def composition_basis_caption(basis):
    return f"Composition is expressed as percentages {basis}. Weighing amounts are calculated below."

def premix_quantity_label(unit):
    return f"Pre-mix quantity to prepare ({unit})"

OLD_WEIGHED_MODE = "weighed into each formulation"
OLD_SINGLE_MODE = "bought in"
OLD_COMPOSITION_LABEL = "% of pre-mix"
OLD_COMPOSITION_LABEL_UNHYPHENATED = "% of premix"
EDIT_ONLY_PENDING = "Only unrecorded formulations in this Round can be edited."
EDIT_BALANCE_HELP = " Adjust the amounts or use a balance ingredient."
EDITED_FORMULATION = "Edited formulation"

def edit_calculated_error(name):
    return f"{name} is calculated or unavailable; edit its inputs instead."

def edit_number_error(name, finite=False):
    return f"Enter a finite number for {name}." if finite else f"Enter a number for {name}."

def edit_formulation_error(number, error):
    return f"Formulation {number}: {error}"

def edit_negative_error(number):
    return f"Formulation {number}: the calculated ingredient amounts must be zero or greater."

def workbook_result_conflict(number, key):
    return f"Formulation {number}: conflicting {key} entries on the round and formulation sheets. Keep one value or make them agree, then upload again."


SHARE_TOTAL = "Total: 100%"
SHARE_PREVIEW_HELP = "Changing one share adjusts the others to keep the total at 100%. Preview the saved shares below."
SCORING_DETAILS_CHECKBOX = "How the score is calculated"
EXAMPLE_REFERENCES_LABEL = "Example assumptions and references"
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
CUSTOM_VALUES_INVALID = "Recording values must be text or numbers, up to 2,000 characters per entry."
CUSTOM_WORKBOOK_CHANGED = "These recording fields no longer match this Round. Download a new workbook and transfer your entries."
CUSTOM_WORKBOOK_CONFLICT = "This workbook contains conflicting entries for the same recording field. Keep one value and upload again."
CUSTOM_TEXT_ONLY = "Enter recording values as text or numbers, rather than spreadsheet formulas."
CUSTOM_RECORDS_HEADING = "Additional records"
CUSTOM_SHEET_HELP = "Optional text or numbers. These records are saved with the round and do not affect scores."
CUSTOM_AUTOSAVE_HELP = "Double-click a cell to enter a value. Entries save automatically when you leave the cell and do not affect scores."
CUSTOM_SUBJECT = "Recorded for"
CUSTOM_VALUE = "Value"
CUSTOM_INGREDIENT_LABEL = "Ingredient"
CUSTOM_ROUND_LABEL = "Round for additional records"
CUSTOM_IMPORT_FOUND = "Additional records found. Check the entries below before saving."
CUSTOM_SAVED = "Additional records saved."

def custom_record_option(name, scope, opt=None):
    """'Operator · Each formulation': the scope in the project's words (with
    `opt`), the user's own field name exactly as typed (re-review R6)."""
    scope_text = CUSTOM_SCOPE_NAMES[scope]
    return f"{name} · {scope_text if opt is None else for_project(opt, scope_text)}"

RECORD_SCOPE_LABELS = {
    'vendor': "Vendor · Each ingredient · Ingredients table",
    'sku': "SKU · Each ingredient · Ingredients table",
    'lot': "Lot numbers · Enter in Make a round",
}
ACTUAL_RECORD_OPTION = "Record changes from the planned amounts or settings"
ACTUAL_RECORD_HELP = "For each formulation, record what you actually weighed or the settings you used in the workbook. Leave unchanged values blank. Imported corrections are used when learning from your results."
LOT_ENTRY_HEADING = "Ingredient lot numbers"
LOT_LOCATION_HELP = "Enter lot numbers below the formulations in Make a round, or in the workbook."
LOT_ENTRY_HELP = "For this Round only. Double-click a Lot cell to enter a number; it saves automatically and is included in the workbook."
LIMIT_BOUND_HELP = "Leave a side empty for no limit."
LIMIT_BOUND_DETAIL = ("Leave a minimum or maximum empty for no restriction on that side. "
                      "Ingredient amounts and other limits still apply.")

def missing_property_values(metric, names):
    return f"Complete {metric} for: {names}. Open Ingredient limits → Properties and enter the missing values (0 only if zero), or delete this property limit."

BACKGROUND_REFERENCE_LABEL = "Background reference"


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


# The three sheets of the compact workbook, and the lines that head them.
OVERVIEW_SHEET = "Round overview"
PREPARATION_SHEET = "Preparation"
RESULTS_SHEET = "Results"
SHEET_NAVIGATION = (
    f"Use the sheet tabs: {OVERVIEW_SHEET} → {PREPARATION_SHEET} → "
    f"{RESULTS_SHEET}. Tabs are at the bottom in Excel and at the top in "
    "Numbers. ")
OVERVIEW_SHEET_INTRO = (
    f"Review the plan, prepare each {FORMULATION} separately, then enter "
    f"measurements on {RESULTS_SHEET} and upload this file in the app.")
PREPARATION_SHEET_TITLE = f"{PREPARATION_SHEET} — Pre-mixes for this round"
PREPARATION_SHEET_INTRO = (
    f"Prepare pre-mixes with fixed {INGREDIENT} percentages here. For blends "
    f"where {INGREDIENT} amounts vary separately, follow the individual "
    f"amounts on {OVERVIEW_SHEET}.")
PREPARATION_SHEET_EMPTY = (
    f"No pre-mixes are required. Follow the method on {OVERVIEW_SHEET}.")
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
    "This workbook is missing or has damaged Round information. Download a "
    "new workbook and transfer your entries.")
WORKBOOK_FORMULA_ERROR = (
    "A formula has no saved result. Open the workbook in Excel, recalculate "
    "and save it, then upload again; or enter the measured number directly.")
WORKBOOK_WRONG_ROUND = (
    "This workbook belongs to a different Round. Download the workbook for "
    "the open Round.")
WORKBOOK_ROUND_CHANGED = (
    f"The {FORMULATION}s changed after this workbook was downloaded. "
    "Download a new workbook and transfer your measurements before "
    "uploading.")
WORKBOOK_MISSING_SHEET = (
    "This workbook is missing a required sheet or its Round information. "
    "Download a new workbook.")
WORKBOOK_RESULTS_ARE_NUMBERS = (
    "Enter results as numbers, not Excel formulas, then upload again.")
WORKBOOK_ACTUALS_ARE_NUMBERS = (
    "Enter actual amounts as numbers, not Excel formulas.")


# ------------------------------------------------------------------ #
#  Insights (0.8.0)
# ------------------------------------------------------------------ #

def path_past_lowest(name):
    return f"{name} would go below its Lowest"


def path_past_highest(name):
    return f"{name} would go past its Highest"


def path_below_zero(name):
    return f"{name} would drop below 0"


PATH_NO_ROOM_DOWN = "the other ingredients are all at their Lowest"
PATH_NO_ROOM_UP = "the other ingredients are all at their Highest"


TAB_INSIGHTS = "4 · Review"
INSIGHTS_NONE = "The review appears once formulations are scored."
INSIGHTS_WORKING = "Preparing the insights…"
INSIGHTS_FAILED = "The review could not be prepared for this project."


def insights_gate(n):
    """Said once on the tab, under Progress, while fewer than 5
    formulations are scored (0.8.0 review fixes §6; 03 §5.1)."""
    return (f"Prediction check and Next step start at 5 scored {FORMULATION}s. "
            f"Scored so far: {n}.")


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


def rounds_span(rounds):
    """"Round 4"; "Rounds 3–5" when they run on; "Rounds 3 and 5" or
    "Rounds 3, 5 and 6" when one is skipped, so a gap is never hidden."""
    return _span(ROUND_CAP, rounds)


def formulations_span(numbers):
    """"Formulation 7"; "Formulations 7–12"; "Formulations 7, 9 and 12" —
    the same rule as rounds_span, in number order."""
    return _span(FORMULATION_CAP, sorted(int(n) for n in numbers))


def rounds_not_checked(rounds):
    one = len(rounds) == 1
    return (f"{rounds_span(rounds)} {'is' if one else 'are'} not checked: fewer than 5 "
            f"{FORMULATION}s were recorded before {'it' if one else 'them'}.")


WIDE_EVERY = ("Every 95 % interval is wide (median half-width over 25 % of the Range): "
              "values fall inside easily, so this check says little yet.")


def wide_line(names_text):
    """The 95 % intervals of some outputs are wide (§9 I8; 03 §5.2):
    `names_text` names them, 'Cook loss and the Overall score'."""
    return (f"Wide 95 % intervals for {names_text} (median half-width over 25 % of the "
            "Range): values fall inside easily, so this check says little yet.")


INGREDIENT_OR_SETTING = "Ingredient or process setting"
START_FROM = "Start from"


def start_best(n):
    return f"Best so far ({FORMULATION_CAP} {n})"


def start_formulation(n):
    return f"{FORMULATION_CAP} {n}"


def path_follows(name):
    return f"{name} follows its calculation."


def path_outside(n, amount_text):
    return f"{FORMULATION_CAP} {n} used {amount_text}, outside today's Lowest–Highest."


def path_start_breaks(n, reason):
    return f"{FORMULATION_CAP} {n} breaks today's limit: {reason}."


def path_not_checked(limit, prop):
    return f"Not checked: {limit} — some ingredients have no {prop} value."


def panel_missing(name):
    return f"{name}: not enough formulations measured yet."


def panel_failed(name):
    return f"Could not be predicted for {name}."


def change_none(settings_only=False):
    """Nothing varies: a project of process settings alone names settings."""
    what = "process setting" if settings_only else "ingredient and process setting"
    return f"Nothing to change yet: every {what} is fixed."


CHANGE_NONE = change_none()


# The kernel each model setting names (0.8.0 e2e fixes D5, 03 minor): Method
# and details says the one in use, not always the default's.
KERNEL_NAMES = {
    "matern52": "Matérn 5/2",
    "matern32": "Matérn 3/2",
    "rbf": "squared exponential",
    "linear": "linear",
    "poly2": "polynomial of degree 2",
}
_NOISE_NAMES = {"low": "noise fitted, with a prior that favours low noise",
                "fixed_tiny": "noise fixed near zero"}


def how_models(kernel=None, noise=None):
    """Method and details' model paragraph (D5): `kernel` and `noise` are
    the project's model settings (bo_config's keys), None for the
    Recommended settings, which use the library default."""
    if kernel is None:
        which = "default kernel: squared exponential"
    else:
        which = f"kernel from your model settings: {KERNEL_NAMES.get(kernel, kernel)}"
    lengths = ("" if kernel in ("linear", "poly2")
               else ", one correlation length per varying row")
    fitted = _NOISE_NAMES.get(noise, "noise fitted")
    return ("Models: one Gaussian process per measurement and one for the Overall score, "
            f"each fitted to the {FORMULATION}s that have that value ({which}{lengths}; "
            f"{fitted}). Suggestions use the Overall-score model only. It is fitted to the "
            "scores themselves, not built from the measurement models, so the two can "
            "disagree slightly.")


HOW_MODELS = how_models()


# Fix round B (statistician I2): what the 95 % intervals aim for, and how
# often they held on the app's simulated projects — one constant, set from
# fix A's re-run (fix-a-report.md, I9: Round-grouped folds): the 48
# simulated projects, all Rounds, 90.6 % of measured values and 90.0 % of
# Overall scores before the second widening (92.5 % as shown). 91 / 89
# before fix A.
SIMULATED_COVERAGE = {"measurements": 91, "score": 90}


def coverage_reading():
    """'Aims for 95 %; measured about 91 % (measurements) / 89 % (Overall
    score) on simulated projects.' The plain reading of a 95 % interval
    (never "about 19 in 20", which the record does not bear out)."""
    c = SIMULATED_COVERAGE
    return (f"Aims for 95 %; measured about {c['measurements']} % (measurements) / "
            f"{c['score']} % ({OVERALL_SCORE_COLUMN}) on simulated projects.")


HOW_RANGE = ("95 % interval: the model's 95 % prediction interval for one new measured value, "
             "mean ± 1.96 × predictive SD × calibration factor. The predictive SD includes Repeat "
             "noise, so the interval covers a single result, not the long-run mean. Its ends are "
             "cut at the measurement's Range (0–100 for the Overall score). Plain reading: "
             + coverage_reading())
HOW_RANGE_FIXED_TINY = ("95 % interval: your model settings fix Repeat noise near zero, so the "
                        "interval reflects model uncertainty only, times the calibration factor.")


# 03 §5.7, new: the verdict's rule and the noise threshold, defined once.
def how_verdict(n=35, allowed=3):
    """The verdict rule, with its example from this project's own count
    (0.8.0 e2e fixes, review M11; 01 O11): `n` values checked in the
    verdict's Rounds, `allowed` = insights.allowed_misses(n)."""
    misses = "1 miss" if allowed == 1 else f"{allowed} misses"
    return ("Verdict rule: misses are counted as a binomial count at 5 % per value. As "
            "expected: no more misses than correct 95 % intervals give at least 1 time in 10 "
            f"(for {n} values, up to {misses}). More misses than expected: rarer than 1 time "
            "in 10. Many more: rarer than 1 time in 100. No p-value is shown.")


HOW_VERDICT = how_verdict()
HOW_NOISE_THRESHOLD = ("Noise threshold: the Repeat SD of the Overall score, pooled over Repeat "
                       "groups, once it has 3 degrees of freedom and is at least 0.05; until "
                       "then, 1 point, assumed rather than measured. Rises, "
                       "ties, local effects and predicted gains are compared with it. It is a "
                       "fixed rule, not a significance test.")
# The statistics term behind Progress's Repeat line, said here only
# (answer-key refresh item 7): the main screen says it in plain words.
HOW_REGRESSION = ("Regression to the mean: Best so far is the highest of the measured Overall "
                  "scores, so its own noise is more often above zero than below. A Repeat of it "
                  "is predicted by the Overall-score model at its amounts. Progress gives that "
                  "prediction when it is below the measured score by more than the noise "
                  "threshold, and no steady bias of the Overall score explains the gap.")


def how_checks(values=False, total=True):
    """A project of process settings alone runs each trial at its values.
    `total`: the project has a Formulation total, so a formulation made off
    it is learned at it (T0 review I-1); without one, as made."""
    at = ("values it was run at" if values
          else "amounts it was made at, brought to the Formulation total when made off it"
          if total else "amounts it was made at")
    return ("Round-ahead check: each Round is predicted by models fitted only to the "
            f"{FORMULATION}s recorded before it, at the {at}, and "
            "scored with today's measurements, targets, Share of score and model settings. "
            "The verdict uses measured values from the last 3 Rounds checked, more if needed "
            "to reach 10; the Overall score's line, the mean error and the steady bias use "
            "every Round checked.")


HOW_CHECKS = how_checks()


def how_same_process(values=False):
    """0.8.0 review fixes §9 I13. `values`: a project of process settings
    alone, which has no ingredients."""
    what = "suppliers and equipment" if values else "ingredients, suppliers and equipment"
    return f"Predictions assume your {what} behave as before."


HOW_SAME_PROCESS = how_same_process()
HOW_NO_P = ("No p-values or coefficient table. A Gaussian process has no model terms to test, "
            f"and the {FORMULATION}s were chosen Round by Round from earlier results, not from a "
            "fixed design, so term tests would not keep their stated error rates. R² on "
            f"{FORMULATION}s chosen this way is misleading, so skill uses the Round-ahead "
            "baseline instead.")
DETAILS_NONE = f"This table fills in once a {ROUND_CAP} has been checked."


TRACK_PREDICTED = "Round-ahead prediction"
TRACK_MEASURED = "Measured"
TRACK_LOW = "95 % interval, lower end"
TRACK_HIGH = "95 % interval, upper end"
# The path chart's own prediction is the current model's, not Round-ahead.
PREDICTED = "Predicted"


# Insights wave B (0.8.0): predictions on Make a round, the spread of each
# ingredient or process setting, and the formulation predicted to score
# highest. Every number that is a prediction carries its likely range.
PREDICTIONS_WORKING = "Preparing the predictions…"
PREDICTED_SCORE_COLUMN = f"Predicted {OVERALL_SCORE_COLUMN} (95 % interval)"


def predicted_score_help(n=None, factor_text=None):
    """The Predicted Overall score column's "?" (03 §5.8): the basis (n
    scored formulations) and the Round-ahead widening when they are known."""
    fitted = (f"the {n} scored {FORMULATION}s" if n is not None
              else f"the scored {FORMULATION}s")
    line = ("Model mean and 95 % prediction interval for one measured Overall score, from a "
            f"Gaussian process fitted to {fitted}.")
    if factor_text is not None:
        line += f" Intervals widened ×{factor_text} by the project's Round-ahead record."
    return f"{line} {coverage_reading()}"


PREDICTED_SCORE_HELP = predicted_score_help()


def predictions_round_caption(values=False):
    """Why a suggestion can read below Best so far (review I5): the
    acquisition weighs the model's own uncertainty as well as its mean.
    `values` is kept for callers; the sentence names neither."""
    return "Exploring suggestions can predict below Best so far."


def predictions_round_help():
    """predictions_round_caption's reason, in its tooltip."""
    return (f"Suggestions favour {FORMULATION}s the model is unsure about as well as ones it "
            "predicts high, so one can be predicted below Best so far.")


PREDICTIONS_ROUND_CAPTION = predictions_round_caption()
PREDICTED_MEASUREMENTS = "Predicted measurements (95 % interval)"
# 0.8.0 review fixes §9 I13: the two sets of predictions are learned apart.
PREDICTED_MEASUREMENTS_CAPTION = "The Overall score has its own model."
PREDICTED_MEASUREMENTS_HELP = ("The Overall score has its own model, so it can differ "
                               "slightly from a score calculated from these predictions.")


def spread_from(best, n):
    """"Best so far" or "Formulation 9": the start a spread is measured from."""
    return "Best so far" if best else f"{FORMULATION_CAP} {n}"


def held_line(start_text, names):
    """Under the answer: the choices the limits box in from this start, so
    their flat line is not read as "no effect"."""
    # Not "Held": the vocabulary retired Hold/held in 0.5.0.
    return f"Kept in place by your limits from {start_text}: {', '.join(names)}."


def _choices_word(settings_only, both, settings):
    return settings if settings_only else both


def held_all(start_text, settings_only=False):
    """The answer when the limits box in every path from this start."""
    what = _choices_word(settings_only, "ingredient and process setting", "process setting")
    return f"From {start_text}, your limits keep every {what} in place."


def compare_all(has_settings, has_ingredients=True):
    if not has_ingredients:
        return "Compare all process settings"
    return "Compare all ingredients and settings" if has_settings else "Compare all ingredients"


# 0.8.0 review fixes §4: the ranking by one measurement, near-ties, and two
# ingredients the Rounds moved together.
SPREAD_ON = "Rank by"


def moved_together(first, second, r_text=None):
    """Two ingredients whose amounts moved together (§4.3; 03 §5.5):
    `r_text`, the absolute value of Pearson's r across the formulations
    with all results in (T0 review I-4: the caller passes |r|)."""
    r = f" (|r| = {r_text})" if r_text is not None else ""
    return (f"{first} and {second} are correlated across the {FORMULATION}s with all results "
            f"in{r}, so their separate effects cannot be told apart.")


def best_luck(n, mean_text, low_text, high_text, measured_text=None):
    """Under the predicted best when a Repeat of Best so far is predicted
    below its measured score (0.8.0 review fixes §3.6; 03 §5.3). Plain
    words on the main screen; the term is in Method and details
    (HOW_REGRESSION; answer-key refresh item 7)."""
    below = f", below its measured {measured_text}" if measured_text is not None else ""
    return (f"A Repeat of {FORMULATION_CAP} {n} is predicted at {mean_text} (95 % interval "
            f"{low_text}–{high_text}){below}. Best so far's measured score is partly chance, so "
            "its Repeat is predicted lower.")


BEST_ROW = "Predicted best"
BEST_FAILED = "The predicted best could not be found."


# Insights wave C (0.8.0): "More charts", the three views a reader who knows
# designed experiments expects. One heading (a question), at most two
# selectors, one chart and one caption each.
Q_PAIR = "Two measurements against each other"

CHART_FAILED = "This chart could not be drawn."
PAIR_PICK = "Two measurements"
PICK_TWO = "Choose two measurements."
PAIR_CAPTION = (f"Measured values, one dot per {FORMULATION}; lines at the targets. No dot "
                "near both targets suggests a trade-off.")


def pair_caption(targets):
    """One line, by how many of the two measurements have a target line
    drawn: `targets` names them."""
    if len(targets) >= 2:
        return PAIR_CAPTION
    if len(targets) == 1:
        return (f"Measured values, one dot per {FORMULATION}; the line marks {targets[0]}'s "
                "target, and the other axis title gives the better direction.")
    return (f"Measured values, one dot per {FORMULATION}; the axis titles give the better "
            "direction. No dot good on both suggests a trade-off.")


# The legend entry for a hollow grey dot: a formulation still waiting for
# some of its results, so it has no Overall score to colour it by.
PAIR_WAITING = "Still waiting for results"
PAIR_NONE = f"No {FORMULATION} has a value for both yet."
PAIR_BEST = "Best so far"


def pair_axis(label, goal, target=None):
    """'Cook loss (%) · lower is better'; 'Firmness (N) · target 6'."""
    if goal == "max":
        what = "higher is better"
    elif goal == "min":
        what = "lower is better"
    elif target is not None:
        what = f"target {float(target):g}"
    else:
        return label
    return f"{label} · {what}"


# Wave C: "What happens when three ingredients trade places?"
Q_MIXTURE = "Three ingredients at a fixed sum"
MIXTURE_PICK = "Three ingredients"
PICK_THREE = "Choose three ingredients."


def mixture_caption(n):
    return (f"Colour: predicted {OVERALL_SCORE_COLUMN} (model mean) as the three amounts vary at "
            f"a fixed sum; all other amounts as in {FORMULATION_CAP} {n}.")


def mixture_corner(name, amount_text):
    """A corner of the triangle: that ingredient's own amount there, where
    it takes all the room the three share (the same number as the corner's
    tooltip)."""
    return f"{name} {amount_text}"


def mixture_fills(fill):
    return f"{fill} fills to the total."


def mixture_no_room(n):
    return (f"From {FORMULATION_CAP} {n}, these three cannot trade places: together they "
            "are already at their Lowest or Highest.")


def mixture_fill_follows(calc, fill):
    return (f"{calc} follows its calculation, so {fill} cannot trade places with the other "
            "two; choose another three.")


def mixture_together(unit):
    return f"The three together ({unit})" if unit else "The three together"


MIXTURE_TESTED = "Tested near here"
MIXTURE_UNTESTED = "Not tested near here"
MIXTURE_BLOCKED = "Not allowed by your limits"


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


def points_text(text):
    """'1 point' / '1.4 points', from a number already formatted."""
    return "1 point" if text in ("1", "1.0") else f"{text} points"


def beat_by(sd_text=None, assumed=False, pairs=0, left_out=0, df=None):
    """The noise threshold's one name (spec review I5; 03 §5.3): the Repeat
    SD from 3 pairs or more, else the fixed 1 point. `assumed` (0.8.0 e2e
    fixes, review B2): the 1 point is said to be assumed, with the Repeat
    that would measure it, while there are fewer than 3 pairs (`pairs`).
    `left_out`: pairs left out because a Repeat disagrees (polish round:
    "no Repeats yet" beside Line B's one Repeat read as false). `df` (fix
    round B, statistician I11): the Repeats' degrees of freedom, said
    instead of their pairs."""
    if sd_text:
        return f"the noise threshold (Repeat SD {sd_text} points)"
    if not assumed:
        return "the 1-point noise threshold"
    if not pairs:
        basis = ("no Repeats yet" if not left_out
                 else "the only Repeat disagrees" if left_out == 1
                 else "every Repeat disagrees")
    else:
        basis = (f"{df} df of Repeats, 3 needed" if df is not None
                 else f"{pairs_text(pairs)} of Repeats, 3 needed")
        if left_out:
            basis += (f", and {left_out} left out as "
                      f"{'it disagrees' if left_out == 1 else 'they disagree'}")
    return (f"the 1-point noise threshold (assumed: {basis}; a Repeat of Best so far would "
            "measure it)")


# The header (§2.1): scored formulations, the Round, the waiting ones. The
# likely range is defined where it is first used, in the verdict.
# Fix round B (five-second re-test): the header names the waiting
# formulations up to this many; more are counted.
HEADER_NAMES_WAITING = 3


def insights_header_line(n, last_round, waiting=0, names_text="", numbers=None):
    """'Based on 24 scored formulations, up to Round 6. Formulation 25 is
    waiting for Cook loss and is not used yet.'; with none scored yet,
    'Formulations 1–3 are waiting for Cook loss.' `numbers`: the waiting
    formulations' numbers, named when there are at most
    HEADER_NAMES_WAITING of them (fix round B: "1 more is waiting" did not
    say which); otherwise, or without them, counted."""
    verb = "is" if waiting == 1 else "are"
    named = (formulations_span(numbers)
             if numbers and len(numbers) == waiting <= HEADER_NAMES_WAITING else None)
    if n == 0 and waiting:
        who = named or f"{waiting} {FORMULATION}{'s' if waiting != 1 else ''}"
        return f"{who} {verb} {waiting_for(names_text)}."
    tail = "" if last_round is None else f", up to {ROUND_CAP} {last_round}"
    who = named or f"{waiting} more"
    wait = (f" {who} {verb} {waiting_for(names_text)} and {verb} not used yet."
            if waiting else "")
    return (f"Based on {n} scored {FORMULATION}{'s' if n != 1 else ''} (all results in)"
            f"{tail}.{wait}")


# Prediction check (§2.2; spec review B1, I6, M9; 03 §5.2).
Q_TRUST = "Prediction check"


def _values(n):
    return "1 value" if n == 1 else f"{n} values"


def _allow(allowed, total):
    """'95 % intervals allow up to 3 misses in 35'."""
    misses = "1 miss" if allowed == 1 else f"{allowed} misses"
    return f"95 % intervals allow up to {misses} in {total}"


def trust_within(inside, total, rounds, measurements=False, allowed=None):
    """The As expected verdict, measurements only (03 §5.2). `allowed`: the
    most misses the rule calls as expected for `total` values
    (insights.allowed_misses). `measurements` is kept for callers: the
    line always names the measurements."""
    line = (f"Measurements, {rounds_span(rounds)}: {inside} of {total} values inside their "
            "95 % interval. As expected")
    return f"{line}: {_allow(allowed, total)}." if allowed is not None else f"{line}."


HOW_BIAS = ("Mean error: the average of measured minus predicted over every Round checked, "
            f"for the Overall score and each measurement ({FORMULATION}s with all results in, "
            "at least 5). It is a steady bias when its size exceeds the noise threshold (for a "
            "measurement, its own Repeat SD once it has 3 df, else no floor) and 2 standard "
            "errors. From 4 Rounds checked, each Round's mean error counts as one value, "
            f"because a Round's {FORMULATION}s sit close together and share the model's "
            f"error; before that, each {FORMULATION} counts. A steady bias is reported as a "
            "fact, with no advice to check the bench.")
# Review M2: the model's error runs both ways, and why.
HOW_BIAS_DIRECTION = (
    "Suggestions sit where the model predicts high or is unsure, so results can come in below "
    "prediction (its optimism at its own maximum) or above it (its mean shrinks toward the "
    "average where it has seen little).")
# Review B1: what the Round-ahead check cannot see.
HOW_CHANGE_LIMIT = (
    f"A change at the bench is learned from its first {ROUND_CAP} on, so later Round-ahead "
    f"checks cannot show it; a Repeat of an earlier {FORMULATION} in each {ROUND_CAP} would.")


def signed(value, decimals=1):
    """'+2.3', '−1.2', '0.0': an error or a gain with its sign (03 rule 8)."""
    text = f"{value:+.{decimals}f}"
    if float(text) == 0:
        return f"{0:.{decimals}f}"
    return text.replace("-", "−")


def trust_over_wide(total, rounds, median_z=None):
    """All values inside, and a median |standardised error| well under
    0.67 (03 §5.2): `median_z` is that median."""
    basis = (f", with a median absolute standardised error of {median_z:.2f} (0.67 expected)"
             if median_z is not None else "")
    return (f"Measurements, {rounds_span(rounds)}: {total} of {total} values inside their 95 % "
            f"interval{basis}: the intervals are wider than needed.")


TRUST_WIDE = ("Not judged yet: every 95 % interval is wide (median half-width over 25 % of the "
              "Range), so nearly any value falls inside.")


def trust_misses(k, rounds, inside, total, many=False, allowed=None):
    """The caution and unreliable verdicts (03 §5.2). `allowed` as for
    trust_within; `k` is kept for callers."""
    misses = total - inside
    line = (f"Measurements, {rounds_span(rounds)}: {inside} of {total} values inside their "
            "95 % interval.")
    if allowed is not None:
        count = "1 miss" if misses == 1 else f"{misses} misses"
        line += f" {count}; 95 % intervals allow up to {allowed}."
    if many:
        return (f"{line} Many more misses than expected: the intervals are too narrow for "
                "these results.")
    return f"{line} More misses than expected."


def trust_early(n):
    return f"Not judged yet: {_values(n)} checked; the verdict needs 10."


TRUST_NONE = (f"No {ROUND_CAP} checked yet. The first check is the first {ROUND_CAP} made "
              f"after 5 {FORMULATION}s are scored.")
TRUST_BEFOREHAND = f"Round-ahead check: each {ROUND_CAP} predicted from earlier {ROUND_CAP}s only."


def score_gap_line(inside, total, gap, beyond="", n=None, rounds=None):
    """The Overall score's own line (03 §5.2): its 95 % intervals and its
    mean error (measured minus predicted). `rounds` (0.8.0 e2e fixes,
    review I1): the line's own Rounds, every Round checked, named; without
    them it says "same Rounds". `beyond` (beat_by's text) says it is a
    bias: larger than the noise threshold and 2 standard errors. Under the
    e2e rulings the bias is said once, by steady_bias_line, so the caller
    leaves `beyond` empty."""
    # The 0.8.0 callers' words for the threshold, until they pass beat_by's.
    beyond = {"1 point": beat_by(), "Repeats differ": "the noise threshold"}.get(beyond, beyond)
    where = rounds_span(rounds) if rounds else f"same {ROUND_CAP}s"
    head = (f"{OVERALL_SCORE_COLUMN}, {where}: {inside} of {total} inside their 95 % "
            "interval; mean error")
    size = f"{abs(gap):.1f}"
    if size == "0.0":
        return f"{head} 0.0 points."
    if not beyond:
        return f"{head} {signed(gap)} points (measured minus predicted)."
    return (f"{head} {signed(gap)} points (n = {total if n is None else n}). This exceeds "
            f"{beyond} and 2 standard errors: Overall {bias_fact(size, gap < 0)}.")


def repeat_disagrees(later, earlier, diff_text, lower=True, half_text=None):
    """A Repeat that disagrees with its original (§2.2, spec review I4; 03
    §5.2): `half_text`, the 95 % interval for the difference of two
    results (√2 × the half-width), when known."""
    line = (f"{FORMULATION_CAP} {later}, a Repeat of {FORMULATION_CAP} {earlier}, scored "
            f"{points_text(diff_text)} {'lower' if lower else 'higher'}.")
    if half_text is not None:
        line += (f" The 95 % interval for the difference of two results is ±{half_text} "
                 "points.")
    return f"{line} {CHECK_IF_CHANGED}"


# Measured against predicted (§2.2, M4; 03 §5.2).
MVP_CAPTION = (f"One dot per {FORMULATION}: Round-ahead prediction (across) against measured value "
               "(up). Dashed diagonal: measured = predicted. Bar: 95 % interval; filled dot "
               f"inside it, hollow outside. The latest {ROUND_CAP}'s bars are highlighted.")


# Progress (§2.3, spec review I5, I7, M1; 03 §5.3).
Q_STAND = "Progress"


def stand_yes(rise_text, threshold_text=None):
    """Rising: the last rise beat the noise threshold (`threshold_text`,
    beat_by's words)."""
    return (f"Rising: up {points_text(rise_text)} in the last {ROUND_CAP}, above "
            f"{threshold_text or beat_by()}.")


def stand_barely(rise_text, sd_text=None, threshold_text=None):
    """Within noise: the last rise under what it needed (`threshold_text`,
    rise_needs' words; fix A follow-up), else under beat_by's."""
    return (f"Within noise: up {points_text(rise_text)} in the last {ROUND_CAP}, under "
            f"{threshold_text or beat_by(sd_text)}.")


def rise_factor(factor_text, q, df=None):
    """The max-of-q factor in the threshold (ftue fix A, cold read I6; fix A
    follow-up): '√2 × 2.23 (best of 4)'; on t, '√2 × 4.15 (best of 4, t on 3
    df)'; for one result, '√2 × 1.64'."""
    notes = ([f"best of {q}"] if q and q > 1 else []) + ([f"t on {df} df"] if df else [])
    return f"√2 × {factor_text}" + (f" ({', '.join(notes)})" if notes else "")


def rise_needs(need_text, factor_text, basis_text, what="needed"):
    """What a rise (or, `what` "needed to differ", a lead) must beat, with
    its make-up: 'the 3.2 points needed: √2 × 2.23 (best of 4) × the 1-point
    noise threshold'. `basis_text`: beat_by's words for the noise SD."""
    return f"the {points_text(need_text)} {what}: {factor_text} × {basis_text}"


def stand_not_lately(k):
    return f"Flat: no higher score in {k} {ROUND_CAP}s."


STAND_LATEST = f"It came from the latest {ROUND_CAP}."


def stand_since(k):
    return (f"No higher score in the {ROUND_CAP} since." if k == 1
            else f"No higher score in the {k} {ROUND_CAP}s since.")


def stand_best(score_text, n, rnd, without_text=""):
    """'Best so far: Overall score 90.65, Formulation 15 (Round 4).' — two
    decimals, as Results shows it."""
    where = (f"{FORMULATION_CAP} {n} ({ROUND_CAP} {rnd})" if rnd is not None
             else f"{FORMULATION_CAP} {n} (made earlier)")
    return (f"Best so far: {OVERALL_SCORE_COLUMN} {score_text}, {where}"
            f"{scored_without_tail(without_text)}.")


def stand_repeats(sd_text, pairs, df=None, groups=None):
    """The Repeat SD with its basis (03 §5.3; fix round B, statistician
    I11: its degrees of freedom, not its pairs): it is the noise threshold
    only from 3 df. 'Repeat SD of the Overall score: 0.8 points, 1 df (2
    results of 1 formulation). The noise threshold stays at 1 point until
    3 df.' Without `df`, the pairs as before."""
    if df is None:
        head = (f"Repeat SD of the {OVERALL_SCORE_COLUMN}: {sd_text} points from "
                f"{pairs_text(pairs)}")
        if pairs < 3:
            return f"{head}. The noise threshold stays at 1 point until 3 pairs."
        return f"{head}; this is the noise threshold."
    head = f"Repeat SD of the {OVERALL_SCORE_COLUMN}: {sd_text} points, {df} df"
    if groups:
        made = f"{FORMULATION}" if groups == 1 else f"{FORMULATION}s"
        head += f" ({df + groups} results of {groups} {made})"
    if df < 3:
        return f"{head}. The noise threshold stays at 1 point until 3 df."
    return f"{head}; this is the noise threshold."


def _where_best(close_no, values=False):
    if close_no is not None:
        return f"near {FORMULATION_CAP} {close_no}"
    return f"in untested {'values' if values else 'amounts'}"


def stand_not_higher(by_text, mean_text, low_text, high_text, close_no=None, values=False):
    """No predicted gain beyond the noise threshold (spec review I7 a, b;
    03 §5.3): `by_text` is beat_by's name for it."""
    return (f"Predicted best: {mean_text} (95 % interval {low_text}–{high_text}), "
            f"{_where_best(close_no, values)}. Predicted gain over Best so far: none beyond "
            f"{by_text}.")


def stand_withheld(mean_text, low_text, high_text, reason, close_no=None, values=False):
    """The predicted best while the model's gains are withheld (polish
    round: the A3 gate, the verdict or the noise guard, as for the expected
    improvement): no gain is claimed either way, and `reason` says why."""
    return (f"Predicted best: {mean_text} (95 % interval {low_text}–{high_text}), "
            f"{_where_best(close_no, values)}. Its gain over Best so far is not reported: "
            f"{reason}.")


def stand_higher(mean_text, low_text, high_text, close_no=None, close_score_text="",
                 gain_text=None, by_text=None, values=False):
    """A predicted gain beyond the noise threshold (spec review I7 c, d; 03
    §5.3): `gain_text`, the predicted best minus a Repeat of Best so far."""
    where = _where_best(close_no, values)
    if close_no is not None and close_score_text:
        where += f" (measured {close_score_text})"
    line = f"Predicted best: {mean_text} (95 % interval {low_text}–{high_text}), {where}."
    if gain_text is not None:
        line += (f" Predicted gain over a Repeat of Best so far: {gain_text} points, above "
                 f"{by_text or beat_by()}.")
    return line


def at_highest(name):
    return f"the Highest of {name}"


def at_lowest(name):
    return f"the Lowest of {name}"


def at_limit(text):
    return f"the limit {text}"


def stand_limits(items):
    """Line 3 (spec review I7; 03 §5.3): where the predicted best sits
    against its Lowest, Highest and limits, with the DoE term."""
    return f"Predicted best is at {_and_join(items)} (active bounds)."


def show_its_amounts(values=False):
    return "Show its settings" if values else "Show its amounts"


# Next step (§2.4; spec review I2, I3, M17; 03 §5.4).
Q_NEXT = "Next step"
# 0.8.0 e2e fixes (review B1, I1): "check" advice comes only from Repeat
# evidence, and is conditional. A steady bias is the model's, not the
# bench's, and gets no advice (next_bias, next_misses).
CHECK_IF_CHANGED = ("If something changed (panel, supplier, equipment), check it before "
                    "relying on the next suggestions.")


def _in_rounds(rounds):
    return rounds_span(rounds) if rounds else f"the verdict's {ROUND_CAP}s"


def bias_fact(gap_text, below=True):
    """'scores have come in about 4.3 points below the predictions': the
    Overall score's steady bias as a fact, with no reading clause, the one
    form the Predictions line, Next step and Make a round's track line use,
    as Insights' steady_bias_line does (polish round: it was said three
    ways; fix round B, five-second re-test: "run high" was read as "high =
    good", so the direction is said in terms of the scores)."""
    side = "below" if below else "above"
    return f"scores have come in about {points_text(gap_text)} {side} the predictions"


def next_bias(gap_text, below=True, rounds=None):
    """The steady bias of the Overall score as one sentence, no advice
    (0.8.0 e2e fixes, review B1, I1, I2): appended to next_misses when a
    bias holds. `below`: results came in below prediction, so predictions
    ran high. `rounds`: every Round checked."""
    where = f" ({rounds_span(rounds)})" if rounds else ""
    return f"Overall {bias_fact(gap_text, below)}{where}."


def next_misses(rounds=None, bias_text=""):
    """Next step's `unreliable` row (review I2): the intervals are too
    narrow, with no advice to check the bench. `bias_text`: next_bias's
    sentence when a steady bias holds."""
    line = (f"Measured values in {_in_rounds(rounds)} fell outside their 95 % interval more "
            "often than 95 % intervals allow: the intervals are too narrow here, so new "
            "predictions carry more error than their intervals show.")
    return f"{line} {bias_text}" if bias_text else line


NEXT_MISSES = next_misses()
_NOT_JUDGED = "The predictions cannot be judged yet."
NEXT_TOO_EARLY = f"Make the next {ROUND_CAP}. {_NOT_JUDGED}"


def next_untested(values=False, open_round=True, round_no=None, shown=True):
    """The predicted best is in untested amounts (03 §5.4): add it to the
    open Round (`round_no`), or to the next, as a formulation of your own.
    `shown`: "Show its amounts" is drawn (not while the noise guard's floor
    or noise trigger holds); only then is it pointed at."""
    what = "values" if values else "amounts"
    where = (f"{ROUND_CAP} {round_no}" if open_round and round_no is not None
             else f"the {ROUND_CAP}" if open_round else f"the next {ROUND_CAP}")
    # "under Progress figures" (ftue review R-I5): the fold moved below.
    pointer = (f" ({show_its_amounts(values)}, under {FOLD_PROGRESS_FIGURES_NAME})" if shown
               else "")
    return (f"The predicted best is in untested {what}{pointer}. "
            f"To test it directly, add it to {where} as your own {FORMULATION}; suggestions "
            "can also land near it.")


def raise_highest(name):
    return f"raise the Highest of {name}"


def lower_lowest(name):
    return f"lower the Lowest of {name}"


def widen_limit(text):
    return f"widen the limit {text}"


def next_close_to_top(change_text="", settings_only=False):
    """Not lately or Barely, nothing predicted clearly higher. `change_text`
    (raise_highest, lower_lowest or widen_limit) names what binds."""
    what = "process settings" if settings_only else "ingredients"
    head = f"Within these {what} and limits, no predicted gain exceeds the noise threshold."
    if change_text:
        return f"{head} To look further, {change_text}."
    new = "a new process setting" if settings_only else "a new ingredient or process setting"
    return f"{head} To look further, add {new}."


def next_explore(no=None, n=0):
    """Next step "explore" (fix round 1, ledger ruling C1): no predicted
    gain beyond the noise threshold, an expected gain above it.
    Said once, not as Progress says it (final review M-2); the number is
    Progress's."""
    why = ("the model's uncertainty, not a higher predicted score, puts their expected "
           "gain above the noise threshold.")
    if n:
        return f"{record_open_round(no, n)}: {why}"
    return (f"Make the next {ROUND_CAP}: the model's uncertainty, not a higher predicted score, "
            "puts its expected gain above the noise threshold.")


NEXT_RISING = f"Best so far is rising. Make the next {ROUND_CAP}."
NEXT_SEE = f"Make the next {ROUND_CAP} to see whether Best so far rises."


def record_open_round(no, n):
    """'Record the 4 formulations still open in Round 6' — "What to do next"
    while a Round is open, as the foot under it says (final review M-1)."""
    what = f"the {FORMULATION}" if n == 1 else f"the {n} {FORMULATION}s"
    return f"Record {what} still open in {ROUND_CAP} {no}"


def next_too_early_open(no, n):
    return f"{record_open_round(no, n)}, then make the next {ROUND_CAP}. {_NOT_JUDGED}"


def next_rising(wide_text="", no=None, n=0):
    """"Best so far is rising", with the width caution when some outputs'
    95 % intervals are still wide, then what to do: record the open Round
    first when there is one."""
    head = "Best so far is rising"
    if wide_text:
        head += f"; 95 % intervals for {wide_text} are still wide"
    tail = (f"{record_open_round(no, n)}, then make the next {ROUND_CAP}." if n
            else f"Make the next {ROUND_CAP}.")
    return f"{head}. {tail}"


def next_rising_open(no, n):
    return next_rising("", no, n)


def next_see_open(no, n):
    return f"{record_open_round(no, n)} to see whether Best so far rises."


def change_on_setup(name, highest=True):
    return f"Change {name}'s {'Highest' if highest else 'Lowest'} on Set up"


CHANGE_LIMITS_ON_SETUP = "Change limits on Set up"


# Effect of one ingredient or process setting (§2.5; spec review I8, I14,
# M3; 03 §5.5).
def near_line(name, change_text, start_text, only=False):
    if only:
        return f"Local effect of {name} from {start_text}: {points_text(change_text)}."
    return f"Largest local effect from {start_text}: {name}, {points_text(change_text)}."


def near_two(first, second, first_text, second_text, start_text):
    return (f"Largest local effects from {start_text}: {first}, {points_text(first_text)}, and "
            f"{second}, {points_text(second_text)} (within 20 % of each other).")


def near_bigger(values=False):
    what = "values" if values else "amounts"
    return f"Larger effects appear only in untested {what} (grey), where the model extrapolates."


def near_none(settings_only=False, has_settings=True):
    what = ("process setting" if settings_only
            else "ingredient or process setting" if has_settings else "ingredient")
    return f"No {what} has a local effect above the noise threshold."


def near_start_untested(start_text, nearest_no=None, share_text=None):
    """No formulation with all results in lies within 15 % of each span of
    the start (insights.CLOSE_TO_THE_BEST). It names the nearest tested one
    and its distance, the measure the Why fold's Nearest tested column uses
    (polish round: "Nothing has been tested near Formulation 22" read as
    nothing at all, beside a Why fold naming Formulation 18)."""
    line = (f"Nothing has been tested within 15 % of each span of {start_text}, so nothing "
            "is ranked.")
    if nearest_no is None:
        return line
    return (f"{line} Its nearest tested {FORMULATION}, {FORMULATION_CAP} {nearest_no}, "
            f"differs by up to {share_text} % of a span.")


def change_axis(measure=OVERALL_SCORE_COLUMN, unit="points"):
    """'Effect on predicted Overall score (points)'; 'Effect on predicted
    Firmness (N)'."""
    return f"Effect on predicted {measure} ({unit})" if unit else f"Effect on predicted {measure}"


CHANGE_AXIS = change_axis()


def change_caption(values=False, measure=OVERALL_SCORE_COLUMN):
    what = "values" if values else "amounts"
    return (f"Bar length: largest minus smallest predicted {measure} along each one-factor "
            "trace from the same start. Dark: the tested stretch. Light: the whole allowed "
            f"trace, including untested {what}.")


def path_head_fill(name, fill):
    return f"More {name} in place of {fill}"


def path_head_proportional(name, total_text):
    return (f"More {name}; the other varying ingredients shrink in proportion, so the "
            f"batch size stays {total_text}")


def path_head_free(name):
    return f"More {name}"


def path_head_process(name):
    return f"{name} changes"


def path_what(head, stops=(), fixed=()):
    """The path's first caption (§2.5, expert "Merge"): 'More Coconut oil
    in place of Water; stops at 8.04 g (Fat per 100 g). Salt fixed at 1.5
    g.' `stops`: (amount text, reason); `fixed`: (name, amount text)."""
    line = head
    if stops:
        line += "; " + AND_JOIN.join(f"stops at {a} ({r})" for a, r in stops)
    line += "."
    if fixed:
        line += " " + ", ".join(f"{n} fixed at {a}" for n, a in fixed) + "."
    return line


def path_key(n, grey=False, targets=False):
    """The path's second caption: 'Grey: not tested near here. Dotted:
    Formulation 15; dashed: target.'"""
    head = "Band: 95 % interval. " + ("Grey: not tested near here. " if grey else "")
    tail = "; dashed: target" if targets else ""
    return f"{head}Dotted: {FORMULATION_CAP} {n}{tail}."


# The Progress chart's caption (§2.3, expert #8).
def progress_caption_2(rings=False):
    line = ("Dashed line: Best so far. A flat stretch shows no higher score yet, not that no "
            f"better {FORMULATION} exists.")
    return f"{line} {PROGRESS_RING}" if rings else line


# Method and details (§2.5; spec review M5, M6, M10; 03 §5.7).
def details_window(all_rounds, k, verdict_rounds=None):
    """Both windows named (03 §5.7): the table's and the verdict's."""
    if len(all_rounds) == 1:
        return f"Table and verdict: {rounds_span(all_rounds)}."
    last = f"the last {ROUND_CAP}" if k == 1 else f"the last {k}"
    if k >= len(all_rounds):
        last = f"all {k}"
    verdict = f" ({rounds_span(verdict_rounds)})" if verdict_rounds else ""
    return (f"Table: all {ROUND_CAP}s checked ({rounds_span(all_rounds)}). "
            f"Verdict: {last}{verdict}.")


DETAILS_COLUMNS_2 = ("Measurement", "Values checked", "Inside 95 % interval",
                     "Median half-width (±)", "Median absolute error", "Calibration factor",
                     "Error ÷ Repeat noise")
# The details table's last column, from the BO-views wave (review I2).
SKILL_COLUMN = "Skill"


def widen_text(factor):
    return f"×{factor:.1f}"


HOW_WIDENED = ("Calibration factor: the root-mean-square of standardised errors (measured − "
               "predicted, ÷ predictive SD) in 5-fold cross-validation of the current model, "
               "the folds grouped by Round (a formulation and its Repeats in one fold), "
               "floored at 1. Every 95 % interval is multiplied by it. ×1.0: not widened; ×2.5: "
               "errors 2.5 times the model's own SD.")


def score_widen_text(own, top):
    """The Overall-score row of the details table (ruling M5, final review
    I-2, re-review R-1; 03 §5.7): its own fit's factor, then the second one
    the predicted best and new formulations carry — the two multiply."""
    return f"{widen_text(own)}, then {widen_text(top)} for the predicted best and new {FORMULATION}s"


HOW_WIDENED_SCORE = (f"{OVERALL_SCORE_COLUMN}, second factor: the root-mean-square of the "
                     f"standardised Round-ahead errors of earlier {ROUND_CAP}s' "
                     f"{OVERALL_SCORE_COLUMN}s (at least 5, computed after the first factor), "
                     f"floored at 1. Where {FORMULATION}s were chosen for a high predicted score "
                     "it widens their intervals, which partly restores coverage; it does not "
                     "correct the predicted value itself, which can still run high. It applies "
                     "on top of the first factor to the predicted best, Best so far's predicted "
                     f"Repeat and the Predicted {OVERALL_SCORE_COLUMN} column on "
                     f"{TAB_BATCH_NAME}.")
HOW_REPEATS_COMPARED = ("Error ÷ Repeat noise: the median absolute error divided by 0.674 × "
                        "Repeat SD, which is the median absolute error Repeat noise alone would "
                        "give. Near 1: the model misses by about as much as measurement noise. "
                        "Above about 1.5: it misses by more than noise explains (lack of fit). "
                        "Shown from 3 Repeat pairs.")
HOW_THRESHOLDS = ("Wide: the median half-width of the 95 % intervals is over 25 % of the "
                  "measurement's Range (0–100 for the Overall score). On "
                  f"{TAB_BATCH_NAME}, a predicted measurement reads “interval too wide” when "
                  "its interval covers more than 75 % of the Range.")


def repeat_compare(ratio, sd=None, pairs=0, left_out=0):
    """The details table's Error ÷ Repeat noise column (spec review I5, M6;
    03 §5.7): the ratio itself, marked lack of fit above 1.5; under 3 pairs
    it is not shown. Pairs left out because a Repeat disagreed are said as
    left out, never as "no Repeats yet" (answer-key read-through, Line B)."""
    if not pairs:
        if left_out:
            return ("1 pair left out (it disagreed)" if left_out == 1
                    else f"{left_out} pairs left out (they disagreed)")
        return "no Repeats yet"
    if pairs < 3:
        return "under 3 pairs"
    if ratio is None:
        return "Repeats identical" if sd == 0 else "no Repeats yet"
    r = round(ratio, 1)
    return f"{r:.1f}" if r <= 1.5 else f"{r:.1f} (lack of fit)"


# Make a round (§3, non-range items; spec review M2, M11, V12–V17, V31).
TOO_UNCERTAIN = "interval too wide"
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


def workbook_line(print_pack=False):
    """The one workbook caption (V12), short (concision round); the full
    one (workbook_line_full) is its tooltip."""
    return "Upload the filled workbook, or type results below."


def workbook_line_full(print_pack=False):
    """The workbook caption before the concision round."""
    where = (f"on the {ROUND_CAP} sheet or each {FORMULATION} page" if print_pack
             else "on the workbook's Results sheet")
    return f"Enter the results {where} and upload it below, or type them here."


WORKBOOK_TABS_HELP = ("The workbook has three sheets: Round overview, Preparation and Results. "
                      "Tick “Include individual formulation pages” for one printable page each.")


def workbook_tabs_help(print_pack=False):
    """The download's "?", which follows the box as workbook_line does: the
    file a tick produces has no Round overview, Preparation or Results sheet."""
    if print_pack:
        return (f"The workbook has one {ROUND_CAP} sheet and one printable page per "
                f"{FORMULATION}. Without the tick it has three sheets: {OVERVIEW_SHEET}, "
                f"{PREPARATION_SHEET} and {RESULTS_SHEET}.")
    return WORKBOOK_TABS_HELP
RECORD_RESULTS_LINE = "Leave a result blank until it is in."
RECORD_RESULTS_HELP = ("One number per measurement; the panel mean where a panel rated it. "
                       "Leave blank if the result isn't in yet; you can add it later on Results.")
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


def overall_score_line(score, ceiling, without_text=""):
    return f"{OVERALL_SCORE_COLUMN} {score:.2f} of {ceiling:g}{scored_without_tail(without_text)}."


OVERALL_SCORE_HELP = (f"{OVERALL_SCORE_COLUMN}, 0–100: the sum over measurements of closeness "
                      "to the goal (0–1) × Share of score. Compare scores within this project "
                      "only; changing a share, a goal or a Range recalculates every score.")


def tie_line(new_no, score_text, lead_text, old_no, need_text):
    """A new Best so far that leads by less than a single new result needs
    to differ (§4 ties; 03 §5.9; fix A follow-up: rise_threshold(noise SD,
    1), the rule Rising uses, so a tie never reads as Rising)."""
    return (f"Tied with {FORMULATION_CAP} {old_no}: {lead_text} apart, under the "
            f"{points_text(need_text)} needed to differ.")


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
SAMPLE_INTRO_LINE = f"A teaching example: review it, then select {NEXT_MAKE_A_ROUND}."


# Charts on Insights (Plan B Task 5): Compare all's two bar lengths.
CHANGE_NEAR = "Tested stretch"
CHANGE_WHOLE = "Whole trace"


# ------------------------------------------------------------------ #
#  0.8.0 speed and simplicity, Plan A (spec 2026-09-30 §1, §3, §10;
#  review I1, I16, V28). Plan A's own strings; Plan B's are above.
# ------------------------------------------------------------------ #

def insights_progress(done, total):
    """Insights' one progress line while the workers fit (V28): Rounds
    whose checks are all in, of the Rounds to check. With every Round
    checked, what is still running is the model of the latest results
    (after Save results), never "12 of 12 Rounds checked…" (integration
    of the 0.8.0 e2e fixes, T0′)."""
    if total and done >= total:
        head = (f"The {ROUND_CAP} is checked" if total == 1
                else f"All {total} {ROUND_CAP}s are checked")
        return f"{head}; updating the model with your latest results…"
    rounds = ROUND_CAP if total == 1 else f"{ROUND_CAP}s"
    return f"Checking predictions against your results: {done} of {total} {rounds} checked…"


# A predicted Overall score not worked out yet (review I1): the column keeps
# its width and never shows a range that would widen a moment later.
PREDICTION_PENDING = "…"


# The unreliable verdict's own fixed name on Insights (T0 review I-3).
_MEASURED_MORE = "; measured values: many more misses than expected"


def _track_head(inside, total, rounds):
    where = rounds_span(rounds) if rounds else f"the verdict's {ROUND_CAP}s"
    return (f"{OVERALL_SCORE_COLUMN}, {where}: {inside} of {total} inside their 95 % "
            "interval")


def round_track_holding(inside, total, see_insights=False, rounds=None):
    """Make a round's track-record line when the Overall score's record
    holds (spec §3, review I16; 03 §5.8). After an unreliable verdict it
    says the measurements missed more often, so it never reads as good news
    alone (final review M-8); the link under it goes to Insights."""
    line = _track_head(inside, total, rounds)
    return f"{line}{_MEASURED_MORE if see_insights else ''}."


def round_track_bias(inside, total, gap_text, below=True, see_insights=False, rounds=None):
    """The same line when the record holds but is a bias, as one fact
    (polish round: bias_fact, no reading clause)."""
    line = f"{_track_head(inside, total, rounds)}; Overall {bias_fact(gap_text, below)}"
    return f"{line}{_MEASURED_MORE if see_insights else ''}."


def round_track_rough(inside, total, gap_text=None, below=True, see_insights=False,
                      rounds=None, allowed=None):
    """The same line when the record is rough (review I16; 03 §5.8): the
    mean error when it is a bias, else the misses the 95 % rule allows."""
    line = _track_head(inside, total, rounds)
    if gap_text is not None:
        line += f"; Overall {bias_fact(gap_text, below)}. More misses than expected"
        if see_insights:
            # Said once (final review M-4: "More misses than expected;
            # measured values: many more misses than expected").
            return f"{line}; many more in the measured values."
    elif allowed is not None:
        line += f"; {_allow(allowed, total)}"
    else:
        line += "; more misses than 95 % allows"
    return f"{line}{_MEASURED_MORE if see_insights else ''}."


# Answer 2 when no amounts meet every limit together, so there is nothing
# to search (plan-a-report open question 2): said, never "could not be
# worked out". Generate stops at the same place.
def best_needs_batch_size(name):
    """The Fill to total row has no default batch size to fill (review C
    I-2: "= rest" is the stored spelling, never said out loud)."""
    return (f"There is no predicted best: {name} is set to {FILL_TO_TOTAL}, and there is "
            f"no {FORMULATION_TOTAL_NOUN} for it to fill. Set the {FORMULATION_TOTAL_NOUN} "
            f"on Set up, under {MORE_SETTINGS_EXPANDER}.")


BEST_NO_AMOUNTS = ("There is no predicted best: no amounts meet every limit together. "
                   "Check the limits on Set up.")


# ------------------------------------------------------------------ #
#  0.8.0 BO views (spec 2026-09-30-bo-views §§1-5, §8 rev. 2; review
#  B1-B4, I1-I12, M1-M16). Every line the wave adds, spelled once here,
#  in the style guide of research/2026-09-30-bo-views/03-wording-audit.md.
#  No task after this one edits this file.
# ------------------------------------------------------------------ #

# §2.1 (review I1, I2): Round-ahead skill of the Overall score, a fact of
# its own under the verdict. States (fix round 1, ledger ruling on T2 I-1):
# "no_skill" (at or below the cut c_n), "unclear" (between the cut and 0.2,
# shown with no advice), "skill" (0.2 or more).
_SKILL_READING = {
    "no_skill": "no better than the average of earlier results",
    "unclear": "not clearly better than the average of earlier results",
    "skill": "better than the average of earlier results",
}


def skill_line(value_text, n, rounds, state):
    """'Overall score, Rounds 2–6: skill 0.36 (n = 15; 0 = as good as the
    average, 1 = perfect), better than the average of earlier results.'
    (review I2: always skill, n and Rounds). The scale's two ends ride with
    the number (ftue spec §4, review R-I7: the one sanctioned edit inside
    the verbatim figures folds), so "skill 0.36" never stands alone."""
    return (f"{OVERALL_SCORE_COLUMN}, {rounds_span(rounds)}: skill {value_text} (n = {n}; "
            f"{SKILL_SCALE}), {_SKILL_READING[state]}.")


# The skill's two ends, said wherever the number is (ftue spec §4).
SKILL_SCALE = "0 = as good as the average, 1 = perfect"


def skill_cell(value_text, n):
    """The details table's Skill column: '0.36 (n = 15)'."""
    return f"{value_text} (n = {n})"


SKILL_UNDER_TEN = "under 10 values"


def how_skill(cut_text="−0.20", settings_only=False):
    """Method and details (review I2, M6; fix round 1; 0.8.0 e2e fixes D4,
    review M10): the statistic, its baseline, its pooling, its minimum n,
    the three readings, the cut stated plainly with what it costs a correct
    model at n = 10 (simulation gate, re-measured by ftue fix A: 4.1 % and
    12.5 %; 6.9 % and 15.3 % before), and the 0.20 rule
    the effect claims follow (A3)."""
    effects = "Setting effects" if settings_only else "Ingredient effects"
    return ("Skill: 1 − (sum of squared Round-ahead errors) ÷ (sum of squared errors of the "
            f"average Overall score of the {FORMULATION}s recorded before each {ROUND_CAP}), "
            f"pooled over every {ROUND_CAP} checked ({FORMULATION}s with all results in for the "
            "Overall score). 1: exact; 0: no better than that average. Shown from 10 values. "
            f"At or below {cut_text}: no better than the average; below 0.20: not clearly "
            f"better; 0.20 or more: better. The cut is {cut_text}. In 24 simulated projects "
            "whose model's errors matched its own stated uncertainty (true skill 0.20 or more), "
            "the skill read “no better than the average” in 4.1 % of windows of 10 checks "
            "(n = 10), and below 0.20 in 12.5 %; near a true skill of 0.20 it reads below "
            f"0.20 about half the time. {effects} are reported only from 10 values and skill "
            "0.20.")


def _skill_basis(value_text, n, rounds=None):
    """'skill −0.47, n = 16, Rounds 2–5' (review I2: always its Rounds, T47
    review I-3: they are not the verdict's)."""
    where = f", {rounds_span(rounds)}" if rounds else ""
    return f"skill {value_text}, n = {n}{where}"


def next_no_skill(value_text, n, no=None, left=0, rounds=None):
    """Next step row `no_skill` (review I1): the Overall score's Round-ahead
    predictions are no better than the average of earlier results."""
    head = (f"{OVERALL_SCORE_COLUMN} predictions are no better than the average of earlier "
            f"results ({_skill_basis(value_text, n, rounds)}): the predicted best and the "
            "suggestions' expected gain are not informative yet.")
    tail = (f"{record_open_round(no, left)}, then make the next {ROUND_CAP}." if left
            else f"Make the next {ROUND_CAP}.")
    return f"{head} {tail}"


def round_track_no_skill(value_text, n, rounds=None):
    """Appended to Make a round's track line when `no_skill` is set (review
    I1): shown even when coverage holds, with its own Rounds (T47 review
    I-3: every Round checked, not the verdict's)."""
    where = f", {rounds_span(rounds)}" if rounds else ""
    return (f" Skill {value_text} (n = {n}{where}): no better than the average of earlier "
            "results.")


# §2.2 (review M7): the triangle's colours against the model's own
# uncertainty. A caption only; the colours are not rescaled.
def mixture_within_noise(span_text, half_text):
    return (f"The colours span {span_text} points, less than the model's own uncertainty "
            f"here (median ±{half_text} points, 1.96 × model sd): the shading does not "
            "separate these amounts.")


# §2.3 (review I3, I4): the display-side noise guard.
GUARD_FLOOR = "the model's correlation length is at its lower limit"
GUARD_NOISE = "its fitted noise is below half the Repeat SD"


def guard_line(reasons, settings_only=False):
    """Triggers (a) and (b), with (c) or not: 'The model is fitting noise in
    these results (…), so ingredient effects are not reported.'"""
    what = "Setting" if settings_only else "Ingredient"
    return (f"The model is fitting noise in these results ({'; '.join(reasons)}), so "
            f"{what.lower()} effects are not reported.")


# The A3 gate's reason, one wording for the effect line and the path's
# caption (polish round: the caption said "do not beat the average … yet"
# under an effect line naming the 0.20 rule). Each names the rule.
A3_UNTIL = ("until Round-ahead predictions are clearly better than the average of earlier "
            "results (skill 0.20 needed)")


def a3_once_wait(n):
    return (f"once Round-ahead checks show the {OVERALL_SCORE_COLUMN}'s predictions clearly "
            f"better than the average (skill 0.20 over 10 checked {FORMULATION}s; {n} so far)")


def a3_once_unclear(value_text, n):
    return ("once Round-ahead predictions are clearly better than the average of earlier "
            f"results (skill {value_text}, n = {n}; 0.20 needed)")


def guard_skill_line(settings_only=False):
    """Trigger (c) alone (review I3): not "fitting noise"."""
    what = "Setting" if settings_only else "Ingredient"
    return f"{what} effects are not reported {A3_UNTIL}."


def why_guard_line(reasons=(), skill_only=False, verdict=False, gate_reason=None):
    """The Why fold when the model's gains are withheld (review I4; T47
    review minor 5): what it withholds, and the same reason Progress gives.
    Roles are model-free from the 0.8.0 e2e fixes (review I3) and always
    shown, so only the expected gain is withheld. `gate_reason`:
    the A3 gate's reason in Progress's own words (gain_reason_too_few,
    gain_reason_unclear or GAIN_REASON_SKILL; integration, T0′)."""
    if verdict:
        return ("Expected gain is not shown: the measurements miss their 95 % "
                "intervals more often than expected.")
    if gate_reason:
        return f"Expected gain is not shown: {gate_reason}."
    if skill_only:
        return ("Expected gain is not shown until Round-ahead predictions are "
                "clearly better than the average of earlier results.")
    return ("Expected gain is not shown: the model is fitting noise in these "
            f"results ({'; '.join(reasons)}).")


# The Why fold opened before the checks its withhold rule reads are in
# (answer-key refresh item 8): nothing is shown that might be withdrawn.
WHY_CHECKING = "Expected gain is not shown until the Prediction check is calculated."


PATH_GUARD = "The curve's bends are not reliable: the model is fitting noise in these results."
PATH_GUARD_SKILL = f"The curve's bends are not read as effects {A3_UNTIL}."


# §3.1 (review I5, I9, I12, M4, M5, M9, M15): Why these suggestions.
WHY_THESE = "Why these suggestions"
EXPECTED_IMPROVEMENT = "Expected gain (model, one formulation)"
OPTIMISTIC_SCORE = "Optimistic score (mean + 0.45 × sd)"
MODEL_SD = "Model sd (points)"
NEAREST_TESTED = "Nearest tested"
LARGEST_DIFFERENCES = "Largest differences"
ROLE = "Role"
# One vocabulary with the Compared with cell (0.8.0 e2e fixes A2, review I3).
ROLE_REFINES = SUGGESTION_CLOSE
ROLE_BALANCED = SUGGESTION_BALANCED
ROLE_EXPLORES = SUGGESTION_DIFFERENT
EXPECTED_IMPROVEMENT_HELP = (f"The model's average gain of this one {FORMULATION}'s score "
                             "(without Repeat noise) over the highest the model draws for the "
                             f"tested {FORMULATION}s, drawn afresh each time (no single "
                             "incumbent), counting no gain as zero, in points; approximate.")
EXPECTED_IMPROVEMENT_EI_HELP = (f"The model's average gain of this one {FORMULATION}'s score "
                                "over Best so far's measured score, counting no gain as zero, "
                                "in points.")
OPTIMISTIC_SCORE_HELP = ("Your model settings choose suggestions by an upper confidence "
                         "bound: the model mean plus 0.45 × model sd. It is not the upper end "
                         "of the 95 % interval.")
PENDING_MARK = "·"
# 02 M13 (review M12): a sentence, not a fragment that starts with the mark.
WHY_PENDING = ("Picks after the first, marked ·, show what each adds to the earlier picks of "
               "this Round, as they were chosen.")
def why_cold(values=False):
    """A Round chosen before there was a model (review I12); a project of
    process settings alone has values, not amounts (T0 review M-8)."""
    return f"Chosen to cover the allowed {'values' if values else 'amounts'}; no model yet."


WHY_COLD = why_cold()
WHY_FAILED = "These values could not be calculated."
WHY_WORKING = "Calculating why these were suggested…"


def why_row_label(no, own=False, edited=False):
    """'Formulation 24', '… (your own)', '… (edited)'."""
    tail = " (your own)" if own else " (edited)" if edited else ""
    return f"{FORMULATION_CAP} {no}{tail}"


def why_difference(name, diff_text):
    """'Coconut oil +1.20 g': one of the two largest differences from the
    nearest tested formulation, in the row's own unit."""
    return f"{name} {diff_text}"


# The Why map's dashed line (answer-key refresh item 6): Best so far's
# predicted Repeat, the Insights outline's reference (binding speed-spec I7).
WHY_REFERENCE = "Best so far (predicted Repeat)"


def why_chart_caption(n, line=True):
    """Concision round: the map's legend, short; the full one
    (why_chart_caption_full, with the sample's size) is its tooltip."""
    text = f"Grey: allowed. Rings: tested. Labels: this {ROUND_CAP}'s picks."
    return f"{text} Dashed: Best so far's Repeat." if line else text


def why_chart_caption_full(n, line=True):
    """One line (review M9; T47 review minor 4): what each mark is, the
    rings included, and the dashed line when it is drawn. The second fact
    is in the Model sd help."""
    text = (f"Predicted Overall score (across) against model sd (up): grey dots, {n} allowed "
            f"{FORMULATION}s sampled evenly; grey rings, tested {FORMULATION}s; labelled, this "
            f"{ROUND_CAP}'s suggestions")
    return f"{text}; dashed line, {WHY_REFERENCE}." if line else f"{text}."


def why_no_roles(n):
    """Fewer than 500 allowed formulations could be sampled (review I9).
    Retired by the 0.8.0 e2e fixes (review I3): roles no longer use the
    sample, so nothing calls this once W2's suggestion_role lands."""
    return (f"No roles: the limits leave {n} allowed {FORMULATION}s in the sample, fewer than "
            "the 500 the percentiles need.")


def why_role_rule(values=False, total=True, close_pct=15, far_pct=30):
    """Method and details: the role rule, one for the Role column and the
    Compared with cell (0.8.0 e2e fixes A2, review I3): a distance, not the
    model. `values`: a project of process settings alone; `total`: the
    project has a default batch size, the amounts' common footing;
    `close_pct`, `far_pct`: CLOSE_TO_THE_BEST and twice it, in %."""
    rows = "process settings" if values else "ingredients and process settings"
    # "Formulation total", as the rest of Method and details says it
    # (how_checks; polish round).
    footing = ("" if values or not total
               else " Amounts are compared at the Formulation total.")
    return (f"Role, and the first words of the Compared with column: how far a suggestion is "
            f"from what was made, by its largest difference over the varying {rows}, each as a "
            f"share of its {LOWEST_LABEL}–{HIGHEST_LABEL} span (calculated rows and the "
            f"{FILL_TO_TOTAL} row are left out).{footing} "
            f"{SUGGESTION_CLOSE.capitalize()}: within {close_pct} % of Best so far or a "
            f"Repeat of it. {SUGGESTION_DIFFERENT.capitalize()}: more than {far_pct} % from "
            f"every {FORMULATION} recorded before its {ROUND_CAP}, scored or not. "
            f"{SUGGESTION_BALANCED.capitalize()}: in between. The role does not use the model, "
            "so it is shown whatever the Prediction check says.")


def role_help(close_pct=15, far_pct=30):
    """The Role column's and the Compared with column's tooltip (D2, review
    M8): the three words in one line each."""
    # The Method rule's own words (polish round): "or a Repeat of it" and
    # "before its Round".
    return (f"{SUGGESTION_CLOSE.capitalize()}: every varying row within {close_pct} % of its "
            f"span of Best so far or a Repeat of it. {SUGGESTION_DIFFERENT.capitalize()}: over "
            f"{far_pct} % of a span from every {FORMULATION} recorded before its {ROUND_CAP}. "
            f"{SUGGESTION_BALANCED.capitalize()}: in between.")


# The Why fold's other two columns, kept visible at 1280 px with their full
# meaning in the tooltip (D2, review M8).
LARGEST_DIFFERENCES_HELP = ("The two largest differences from the nearest tested "
                            f"{FORMULATION}, largest share of its span first, in each row's "
                            "own unit.")


WHY_AXIS_MEAN = "Predicted Overall score (model mean)"
WHY_POOL = "Allowed formulation (sampled)"
WHY_TESTED = "Tested"
WHY_SUGGESTED = "This Round's suggestion"
WHY_RECOMPUTED = ("Recomputed from today's records: this Round's generation record is missing "
                  "or no longer matches its rows.")
# Last-fix review R-1: with the expected gain withheld, the line is
# about what the fold still shows from the recomputed model.
WHY_RECOMPUTED_WITHHELD = ("Model sd and the map are recomputed from today's records: this "
                           "Round's generation record is missing or no longer matches its rows.")


HOW_EXPECTED_IMPROVEMENT = (
    "Expected gain (model, one formulation): the suggestion model's average gain of one "
    f"{FORMULATION}'s score, without Repeat noise, over the highest latent value among the "
    f"tested {FORMULATION}s in each posterior draw (qLogNEI, the default: no single incumbent; "
    "shown as the exp of a log value, so approximate) or Best so far's measured score (qLogEI), "
    f"counting no gain as zero. It is not the gain of a whole {ROUND_CAP} and not a gain in a measured "
    "result. Model sd: the suggestion model's standard deviation for the mean, narrower than "
    "the 95 % interval.")


# §3.2 (review I5, I10, M8, M11): Expected gain by Round.
GAIN_HEADING = "Expected gain by Round"


def gain_line(value_text, round_no, threshold_text, above=False, recomputed=False,
              measured_text=None, predicted_none=False):
    """The Progress section's line for the open Round, else the last Round
    made (review I5), named by its number. `threshold_text`: beat_by's
    words; `measured_text`: Best so far's measured score, for qLogEI
    projects, whose reference it is. `predicted_none` (above the threshold
    while no predicted gain is, fix round 1, ledger ruling C1): one
    sentence says where the number comes from — the model's uncertainty,
    not a higher predicted score (final review M-2). `round_no` None: the
    next Round's figure, from today's rows (final review I-1)."""
    which = _gain_which(round_no)
    how = "above" if above else "under"
    ref = (f"Best so far's measured {measured_text}" if measured_text is not None
           else f"the highest the model draws for the tested {FORMULATION}s (approximate)")
    # 0.8.0 e2e fixes A5 (review I6): what the statistic is, never "the best
    # single new formulation beats … by", which an average with zeros is not.
    line = (f"Expected gain, {which}: {value_text} points (model, one {FORMULATION}): "
            f"the model's average gain of one new {FORMULATION} over {ref}, counting no gain "
            f"as zero; {how} {threshold_text}.")
    if above and predicted_none:
        line += " It comes from the model's uncertainty, not from a higher predicted score."
    if recomputed:
        line += " Recomputed from today's records."
    return line


def _gain_which(round_no):
    """'Round 13', or 'next Round' for the figure from today's rows."""
    return f"next {ROUND_CAP}" if round_no is None else f"{ROUND_CAP} {round_no}"


def gain_line_tiny(round_no, threshold_text, recomputed=False):
    """The Progress line when the figure rounds to 0.00 (T56 m3): said as
    under 0.01 points, with no plain reading of a machine zero."""
    line = (f"Expected gain, {_gain_which(round_no)}: under 0.01 points (model, one "
            f"{FORMULATION}), under {threshold_text}.")
    if recomputed:
        line += " Recomputed from today's records."
    return line


GAIN_UCB = ("Expected gain: your model settings choose suggestions by an upper "
            "confidence bound, which gives no expected gain.")


def gain_cold(round_no, values=False):
    what = 'values' if values else 'amounts'
    if round_no is None:
        return (f"The next {ROUND_CAP} will be chosen to cover the allowed {what}: no expected "
                "gain yet.")
    return f"{ROUND_CAP} {round_no} was chosen to cover the allowed {what}: no expected gain yet."


def gain_unreliable(reason):
    """The §3.2 number is withheld (review I4): `reason` names why."""
    return f"Expected gain is not reported: {reason}."


GAIN_REASON_VERDICT = "the measurements miss their 95 % intervals more often than expected"
GAIN_REASON_SKILL = "Round-ahead predictions are no better than the average of earlier results"
GAIN_REASON_GUARD = "the model is fitting noise in these results"
GAIN_AXIS = "Expected gain (points)"
GAIN_BEST_MEASURED = "Best so far (measured)"
GAIN_HIGHEST_PREDICTED = "Highest predicted (model mean)"
GAIN_EXACT = "As when the Round was made"
GAIN_RECOMPUTED = "Recomputed with today's measurements, targets and Share of score"
GAIN_STATUS = "How"
GAIN_CHART_CAPTION = ("Bars: the expected gain each Round's first suggestion was chosen "
                      "for (model, one formulation). No bar: that Round had no model yet, or "
                      "had only your own or past formulations.")


# §3.3 (review I6, I8, M14): layers on the triangle and the paths.
LAYER_PICK = "Show"
LAYER_SCORE = "Predicted score"
LAYER_SD = "Uncertainty"
# 0.8.0 e2e fixes A5 (review I6): the layer and its legend say one name.
LAYER_GAIN = "Expected gain"
# The option's label before the rename: a session that still carries it is
# read as LAYER_GAIN (ui_insights' layer pick maps it).
LAYER_GAIN_FORMER = "Expected improvement"   # charts proto: one name, Expected gain
# The triangle's colour under Predicted score is the model mean, not an
# interval (02 M8, 03 M5; review I13).
LAYER_SCORE_LEGEND = f"Predicted {OVERALL_SCORE_COLUMN} (model mean)"
LAYER_SD_LEGEND = "Model sd (points; narrower than the 95 % interval)"
LAYER_GAIN_LEGEND = "Expected gain (points)"
OVERLAY_NAME = "Predicted above Best so far"


def overlay_caption(ref_text):
    """The outline on the Predicted score layer (review I6; T56 m5)."""
    return (f"Outlined, {OVERLAY_NAME}: model mean − 1.96 × model sd above {ref_text} points "
            "(a Repeat of Best so far as predicted), and the predicted score above Best so far "
            "by more than the noise threshold.")


def start_suggested(no, edited=False):
    """Start from, for the Uncertainty and Expected gain layers only (review
    I8). Not "not made": _BANNED keeps that for a bowl made and never
    measured. An edited suggestion says so (T56 m8)."""
    return f"{FORMULATION_CAP} {no} ({'edited' if edited else 'suggested'}, still to make)"


def layer_caption(layer, anchor_text):
    """What a layer colours, and the slice's anchor (review I8)."""
    what = (LAYER_SD_LEGEND[0].lower() + LAYER_SD_LEGEND[1:] if layer == LAYER_SD
            else "expected gain of one new formulation (model, points)")
    return (f"Colour: {what}, as the three amounts vary at a fixed sum; all other amounts as "
            f"in {anchor_text}.")


def path_layer_caption(layer, anchor_text):
    what = ("model sd, narrower than the 95 % interval"
            if layer == LAYER_SD
            else "expected gain of one new formulation (model, points)")
    return f"Line: {what}, from {anchor_text}."


# §5 (04-navigation.md; review I7, M12, M15): links to Insights.
def round_measured_against_predicted(no):
    """E1, on Results after Save results."""
    return f"{ROUND_CAP} {no}: measured against predicted"


LINK_PREDICTION_CHECK = Q_TRUST
LINK_NEXT_STEP = Q_NEXT


def first_sentence(text):
    """E3's caption: Next step's own first sentence, cut, not reworded."""
    text = str(text)
    cut = text.find(". ")
    return text if cut < 0 else text[:cut + 1]


# ------------------------------------------------------------------ #
#  0.8.0 e2e fixes (spec 2026-10-01-e2e-fixes-design.md rev. 2; every
#  "Proposed ruling" of research/2026-10-01-e2e/spec-review.md). The lines
#  W1, W2 and W3 add, spelled once here in the 03 style guide; the lines
#  they change are edited in place above. No later task edits this file.
# ------------------------------------------------------------------ #

# A1 (review B1): the steady bias, one Prediction-check line naming every
# biased output, with no advice. Error = measured − predicted, so results
# below prediction mean predictions ran high.
def bias_item(name, size_text, unit="points"):
    """'Overall score by about 4.3 points', 'Firmness by about 3.0 N',
    'Juiciness by about 0.4' (a measurement with no unit, or a '/10' one,
    which its label carries). A panel's 'score' is points after a number
    (final review M-3: "by about 0.4 score")."""
    unit = str(unit or "").strip()
    if unit.lower() in ("points", "point", "score", "scores"):
        size = points_text(size_text)
    elif unit and not unit.startswith("/"):
        size = f"{size_text} {unit}"
    else:
        size = size_text
    return f"{name} by about {size}"


def steady_bias_line(high=(), low=(), rounds=None):
    """'Results have come in below the predictions: Overall score by about
    4.3 points, Firmness by about 3.0 N (Rounds 3–7).' `high`, `low`:
    bias_item texts of the outputs whose predictions ran high (results
    below) and low (results above); `rounds`: every Round checked. Fix
    round B: the bias in terms of the results, as bias_fact says it."""
    parts = []
    if high:
        parts.append("below the predictions: " + ", ".join(high))
    if low:
        parts.append(("above them: " if high else "above the predictions: ") + ", ".join(low))
    where = f" ({rounds_span(rounds)})" if rounds else ""
    return f"Results have come in {'; '.join(parts)}{where}."


# Method and details' two new columns (review B1, B2): the mean error over
# every Round checked, and the model's fitted noise SD, shown as context.
MEAN_ERROR_COLUMN = "Mean error"
FITTED_NOISE_COLUMN = "Fitted noise SD"
FITTED_NOISE_FIXED = "fixed near zero"


def mean_error_cell(value_text, bias=False):
    """'−4.3', or '−4.3 (bias)' when it is a steady bias. `value_text`
    comes signed (signed())."""
    return f"{value_text} (bias)" if bias else value_text


def fitted_noise_cell(value_text=None):
    """The fitted noise SD in the output's unit, or 'fixed near zero' under
    the fixed_tiny model setting (insights.fitted_noise_sd gives None)."""
    return FITTED_NOISE_FIXED if value_text is None else value_text


HOW_FITTED_NOISE = ("Fitted noise SD: each model's own estimate of Repeat noise, in the output's "
                    f"unit. It is context only: with few {FORMULATION}s it runs below the noise "
                    "Repeats show, so the noise threshold uses Repeats alone.")

# Method and details' columns, short enough to stay visible at 1280 px,
# with the full meaning in each column's tooltip (D2, review M8).
DETAILS_HELP = {
    DETAILS_COLUMNS_2[1]: "Measured values in every Round checked.",
    DETAILS_COLUMNS_2[2]: "Values inside the 95 % interval given before the Round was made.",
    DETAILS_COLUMNS_2[3]: "Median of the 95 % intervals' half-widths.",
    DETAILS_COLUMNS_2[4]: "Median of |measured − predicted|.",
    DETAILS_COLUMNS_2[5]: ("Multiplies every 95 % interval: the root-mean-square of "
                           "cross-validated standardised errors, at least 1."),
    DETAILS_COLUMNS_2[6]: ("Median absolute error ÷ (0.674 × Repeat SD). Near 1: errors as "
                             "large as Repeat noise; above 1.5: lack of fit."),
    SKILL_COLUMN: ("1 − squared Round-ahead errors ÷ those of the average of earlier results. "
                   "0.20 or more: better than that average."),
    MEAN_ERROR_COLUMN: "Average of measured − predicted over every Round checked.",
    FITTED_NOISE_COLUMN: "The model's own estimate of Repeat noise, for context.",
}


# A3 (review I4): effect claims only from 10 Overall-score checks and skill
# 0.20. Said where the claims would be (Effect of one ingredient, Compare
# all, the layers), in place of guard_skill_line.
def effects_wait_line(n, settings_only=False):
    """Fewer than 10 checks: 'Ingredient effects are reported once
    Round-ahead checks show the Overall score's predictions clearly better
    than the average (skill 0.20 over 10 checked formulations; 6 so far).'
    The rule named whole (final review M-2): after 10 checks a skill of
    0.10 "beats the average" and is still withheld."""
    what = "Setting effects" if settings_only else "Ingredient effects"
    return f"{what} are reported {a3_once_wait(n)}."


def effects_unclear_line(value_text, n, settings_only=False):
    """0 < skill < 0.20 (or the cut < skill < 0.20): not clearly better."""
    what = "Setting effects" if settings_only else "Ingredient effects"
    return f"{what} are reported {a3_once_unclear(value_text, n)}."


def gain_reason_too_few(n):
    """gain_unreliable's reason below 10 checks (A3)."""
    return f"fewer than 10 Round-ahead checks of the {OVERALL_SCORE_COLUMN} so far ({n})"


def gain_reason_unclear(value_text, n):
    """gain_unreliable's reason for skill under 0.20 (A3)."""
    return ("Round-ahead predictions are not clearly better than the average of earlier "
            f"results (skill {value_text}, n = {n})")


def path_guard_unclear(value_text, n):
    """The path's caption under the A3 gate's "unclear" state, in the effect
    line's words (polish round)."""
    return f"The curve's bends are read as effects {a3_once_unclear(value_text, n)}."


def path_guard_wait(n):
    """The same, below 10 checks."""
    return f"The curve's bends are read as effects {a3_once_wait(n)}."


# A4 (review B2, I5): while the 1-point threshold is assumed (fewer than 3
# Repeat pairs), Next step's close, explore, see and rising rows end with
# this sentence. A sentence, not a button.
NEXT_REPEAT_TO_MEASURE = (f"Make a Repeat of Best so far in the next {ROUND_CAP} to measure "
                          "the noise.")


# A6 (review I7): Results' Best so far block lists its Repeats, display
# only; Best so far is still the single highest score.
def repeat_score(score_text, disagrees=False):
    """'67.30', or '67.30 (disagrees)' for a Repeat disagreeing_repeat
    left out of the noise."""
    return f"{score_text} (disagrees)" if disagrees else score_text


def best_repeats_line(best_no, score_texts, mean_text):
    """'Repeats of Formulation 14: 67.30, 66.34, 75.61 (mean with
    Formulation 14: 75.18).' Two decimals, as Results shows scores."""
    head = "Repeat" if len(score_texts) == 1 else "Repeats"
    line = f"{head} of {FORMULATION_CAP} {best_no}: {', '.join(score_texts)}"
    # Concision round: no bracket; the mean is its own short sentence.
    return f"{line}. Mean of {'both' if len(score_texts) == 1 else 'all'}: {mean_text}."


# A7 (review I8): the callout above Set up's grid that "Change … on Set up"
# scrolls to. It clears on the next Set up save or tab move.
def setup_callout(name, value_text, highest=True):
    """'Textured pea protein: Highest 30 g. Next step suggests raising it;
    change the Highest cell in its row below.' `value_text` carries the
    unit."""
    label = HIGHEST_LABEL if highest else LOWEST_LABEL
    verb = "raising" if highest else "lowering"
    return (f"{name}: {label} {value_text}. Next step suggests {verb} it; change the {label} "
            "cell in its row below.")


def setup_callout_limit(text):
    """The same callout for a limit (CHANGE_LIMITS_ON_SETUP)."""
    return (f"Next step suggests widening the {LIMIT} {text}; change it under "
            f"{INGREDIENT_LIMITS_EXPANDER} below.")


# Final review M-6: the same callout when the limit itself is not known on
# Set up (no predicted best worked out for this data): never an empty slot.
SETUP_CALLOUT_LIMITS = (f"Next step suggests widening a {LIMIT} the predicted best sits at; "
                        f"change it under {INGREDIENT_LIMITS_EXPANDER} below.")


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
    which = _and_join(labels)
    verb = "must be numbers" if len(labels) > 1 else "must be a number"
    return named_row_error(name, f"{which} {verb}.")


def import_spreadsheet_formula(name, label, text):
    """A Lowest or Highest cell holding a spreadsheet formula, read as
    written (the xlsx is read with formulas kept so '= rest' survives)."""
    return named_row_error(name, f"{label} holds a spreadsheet formula ({text}). Type the "
                                 "number itself.")


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


# C1 (review I11): Progress's expected gain fills in after the
# first answers, never holding them up.
def gain_pending(round_no=None):
    return f"Expected gain, {_gain_which(round_no)}: calculating…"


# C2 (review M5): the indicator once the workers have broken 3 times in 10
# minutes and Insights calculates in the app's own process.
INSIGHTS_IN_PROCESS = ("Background checks stopped after repeated failures. The review now "
                       "calculates in this window, so it is slower until the app is restarted.")


# ------------------------------------------------------------------ #
#  Insights for a first-time user (spec 2026-10-01-insights-first-time-
#  design.md rev. 2; every "Proposed ruling" of research/2026-10-01-
#  insights-ftue/spec-review.md). T0: every line W1 (status logic and the
#  first screen), W2 (the expert layer) and W3 (explored and unexplored)
#  add, spelled once here in the 03 style guide. The lines they replace
#  stay above until integration deletes them; the three lines T0 edits in
#  place are skill_line, next_untested and SKILL_SCALE (above). No later
#  task edits this file: a missing line is a T0′ commit.
#
#  Trial forms. Every builder below is said through for_project, which
#  turns "formulation" into "trial" on a study of process settings alone;
#  a line that names amounts or ingredients takes `values` or
#  `settings_only` for its own form. The glossary is the one exception
#  (review M4): its two forms are written out, chosen by has_ingredients(),
#  and never go through for_project.
# ------------------------------------------------------------------ #

# --- §1, R-B1, R-I10, M7: the three status lines ---------------------

# The labels in front of the three lines. "Next" is a label with no status
# word (M7); the detail sections below keep their noun headings (Q_TRUST,
# Q_STAND, Q_NEXT: R-I8, R-I10).
STATUS_PREDICTIONS = "Predictions"
STATUS_PROGRESS = "Progress"
STATUS_NEXT = "Next"

# The owner's words for Predictions (2026-10-01), and the Progress words of
# R-I10: the existing tags at both levels, plus Too early and No new best.
# Keyed by the rule that sets them (insights.PREDICTION_ROWS, W1).
PREDICTIONS_WORDS = {"reliable": "Reliable", "not_yet": "Not yet", "off": "Off"}
PROGRESS_WORDS = {"rising": "Rising", "within_noise": "Within noise", "flat": "Flat",
                  "too_early": "Too early", "no_new_best": "No new best"}


def status_line(label, sentence, word=None):
    """'Predictions: **Reliable.** In Rounds 3–5, …'; 'Next: Make the next
    Round.' for a line with no status word (M7)."""
    if word:
        return f"{label}: **{word}.** {sentence}"
    return f"{label}: {sentence}"


# R-I8: each line's place says what it is waiting for until it fills, so
# the button below moves by at most a line's difference.
CHECKING_PREDICTIONS = "Checking predictions…"
CHECKING_PROGRESS = "Checking progress…"
PREPARING_NEXT_STEP = "Preparing the next step…"


def ceiling_text(ceiling):
    """The score's ceiling as a whole number (M1: opt.utility_ceiling(),
    100 for any project saved since 0.5.0)."""
    value = float(ceiling)
    return f"{int(value)}" if value.is_integer() else f"{value:g}"


def _out_of(score_text, ceiling):
    """'90.45 of 100': two decimals, as Results shows it (M10)."""
    return f"{score_text} of {ceiling_text(ceiling)}"


def _rounds_in(rounds):
    return f"In {rounds_span(rounds)}"


def _misses_normal(allowed):
    """'up to 3 misses are normal', inside the Reliable sentence's bracket."""
    if not allowed:
        return "no miss is normal"
    return "up to 1 miss is normal" if allowed == 1 else f"up to {allowed} misses are normal"


def _where_normal(allowed):
    """'where up to 4 are normal', after "results fell outside"."""
    if not allowed:
        return "where none is normal"
    return "where up to 1 is normal" if allowed == 1 else f"where up to {allowed} are normal"


def _sentence(head, clauses=()):
    return f"{head}{''.join(clauses)}."


# Predictions, R-B1 rows 1–9, worst first. A sentence names its cause in
# facts and never advises (R-I2): "check" is said only by Next's repeat row.
# Row 1: a disagreeing Repeat, with the coverage as expected.
PRED_OFF_REPEAT = ("A Repeat came out far from its original (see Next), so the conditions the "
                   "predictions learned from may have changed.")


def pred_off_coverage(rounds, outside, total, allowed, bias_text=None):
    """Row 2, the unreliable verdict: 'In Rounds 3–5, 10 of 36 results fell
    outside the range the app predicted, where up to 4 are normal, and
    scores have come in about 5.5 points below the predictions.' `bias_text`: bias_fact's
    words when the Overall score has a steady bias."""
    line = (f"{_rounds_in(rounds)}, {outside} of {total} results fell outside the range the app "
            f"predicted, {_where_normal(allowed)}")
    if bias_text:
        line += f", and {bias_text}"
    return f"{line}."


def pred_gate(n, needed=5):
    """Row 3, fewer than 5 scored: 'The app starts checking its predictions
    once 5 formulations have results; 3 have so far.'"""
    so_far = "none has" if not n else "1 has" if n == 1 else f"{n} have"
    return (f"The app starts checking its predictions once {needed} {FORMULATION}s have "
            f"results; {so_far} so far.")


def pred_too_few(k, needed=10, clauses=()):
    """Row 4 (no Round checked, too few values, or skill not yet computed):
    counted in formulations, the Overall-score checks the skill gate needs
    (review B1: never the 10 values of the verdict)."""
    if k:
        what = f"1 {FORMULATION}" if k == 1 else f"{k} {FORMULATION}s"
        head = (f"The app has checked its predictions against {what} and needs {needed} before "
                "it can say how far to trust them")
    else:
        head = (f"The app has not checked its predictions against any {FORMULATION} yet and "
                f"needs {needed} before it can say how far to trust them")
    return _sentence(head, clauses)


def pred_caution(rounds, outside, total, allowed, clauses=()):
    """Row 5, more misses than expected (the spec was silent; review B1)."""
    head = (f"{_rounds_in(rounds)}, more results than expected fell outside the range the app "
            f"predicted ({outside} of {total}, {_where_normal(allowed)})")
    return _sentence(head, clauses)


def pred_wide(clauses=()):
    """Row 6, every output's 95 % intervals wide (R-B2: "the predicted
    ranges" is refused by _BANNED, so this is the spec's own sentence)."""
    return _sentence("The app's predictions are still very uncertain, so results landing inside "
                     "them prove little yet", clauses)


PRED_WIDE = pred_wide()


def _score_bias(bias):
    """", and scores have come in about 4.3 points below the predictions":
    the Overall score's steady bias joined to the Score predictions
    sentence (Line C). `bias`: (gap_text, below)."""
    if not bias:
        return ""
    return f", and {bias_fact(*bias)}"


def _against_average(skill_text=None):
    """The baseline skill is measured against, said plainly (fix round B,
    five-second re-test: "the average of earlier results" read as "our
    scores beat the average"), with the skill and its meaning when known
    (the deciding number had none on the first screen)."""
    line = "simply using the average of earlier scores"
    if skill_text:
        line += f" (skill {skill_text}, where 0 matches that average and 1 is exact)"
    return line


def pred_no_skill(bias=None, clauses=(), skill_text=None):
    """Row 7, skill at or below the cut (Line C): 'Score predictions do no
    better than simply using the average of earlier scores (skill −0.47,
    where 0 matches that average and 1 is exact), and scores have come in
    about 4.3 points below the predictions; those for Firmness, Juiciness
    and Cook loss are still very uncertain.'"""
    return _sentence(f"Score predictions do no better than {_against_average(skill_text)}"
                     + _score_bias(bias), clauses)


def pred_unclear(bias=None, clauses=(), skill_text=None):
    """Row 8, skill between the cut and 0.20 (review B1: not "no better")."""
    return _sentence("Score predictions are not yet clearly better than "
                     f"{_against_average(skill_text)}" + _score_bias(bias), clauses)


def pred_reliable(rounds, inside, total, allowed, clauses=()):
    """Row 9: 'In Rounds 3–5, 32 of 35 results landed inside the range the
    app predicted (up to 3 misses are normal), and score predictions beat
    the average of earlier results.'"""
    head = (f"{_rounds_in(rounds)}, {inside} of {total} results landed inside the range the app "
            f"predicted ({_misses_normal(allowed)}), and score predictions beat the average of "
            "earlier results")
    return _sentence(head, clauses)


# The clauses rows 4–9 append, in this order (R-B1, R-I11).
def clause_wide(names, cause=False, most=None):
    """Some outputs' 95 % intervals wide, not all: names as said, the
    Overall score as "the Overall score". `cause` (fix round B, five-second
    re-test: "still very uncertain" named no cause): their models' own
    uncertainty was too small for how far results scatter (the
    calibration factor, ui_insights.wide_cause), said with `most`, the
    output it is clearly largest for."""
    line = f"; those for {_and_join(names)} are still very uncertain"
    if cause:
        line += (f", because results vary more from one {FORMULATION} to the next than the app "
                 "can explain")
        if most:
            line += f", most for {most}"
    return line


# Fix round B: said after the Predictions sentence when the wide clause
# names its cause; conditional, as Next's Repeat row is.
CHECK_SCATTER = "If the panel or method is new, check how consistently it measures."


CLAUSE_OVER_WIDE = "; the predicted range was wider than it needed to be"


def clause_bias(gap_text, below=True):
    """'; scores have come in about 4.3 points below the predictions' (bias_fact): the
    Overall score's steady bias only; other outputs' stay in the figures."""
    return f"; {bias_fact(gap_text, below)}"


def clause_guard(settings_only=False):
    """R-I11: Reliable while the noise guard holds (lengthscale at its floor,
    or fitted noise under half the Repeat SD)."""
    what = "process setting" if settings_only else "ingredient"
    return f"; the app does not yet read what each {what} does"


# Progress (R-B1, R-I9, R-I10, M6).
def vary_threshold(sd_text=None):
    """The noise threshold in plain words (M6): measured from 3 Repeat
    pairs, else the assumed 1 point."""
    if sd_text:
        return f"the {points_text(sd_text)} results vary between Repeats"
    return "the 1 point results are assumed to vary until 3 Repeats measure it"


def _tied_with(near_no, after_round=True):
    """'; tied with Formulation 18' (after a Round) or ', tied with
    Formulation 20': a runner-up within the noise threshold (fix round B,
    walkthrough ruling: Progress names ties)."""
    if near_no is None:
        return ""
    return f"{'; ' if after_round else ', '}tied with {FORMULATION_CAP} {near_no}"


def _best_is(score_text, ceiling, no, rnd=None, near_no=None):
    """'Best so far is 90.09 of 100 (Formulation 11, Round 3; tied with
    Formulation 10)'."""
    where = f"{FORMULATION_CAP} {no}" + (f", {ROUND_CAP} {rnd}" if rnd is not None else "")
    where += _tied_with(near_no, rnd is not None)
    return f"Best so far is {_out_of(score_text, ceiling)} ({where})"


# Fix round B (walkthrough ruling): No new best or Flat, with neither the
# predicted gain nor the expected improvement beyond the noise threshold.
LITTLE_MORE = "The app expects little more inside your current limits"
_LIMIT_ITEM = "your limit "


def binding_plain(kind, name, text=None, values=False):
    """One limit the best prediction sits at, plainly: 'the most Pea',
    'the least Beet juice powder', 'the highest Proof time' (a process
    setting), 'your limit Fat: at most 20 g'."""
    if kind == "limit":
        return f"{_LIMIT_ITEM}{text or name}"
    if values:
        return f"the {'highest' if kind == 'highest' else 'lowest'} {name}"
    return f"the {'most' if kind == 'highest' else 'least'} {name}"


def _little_more(little):
    """' The app expects little more inside your current limits; the best
    it predicts sits at the most Pea allowed.' `little`: binding_plain
    items (empty: nothing binds); None: not said."""
    if little is None:
        return ""
    if not little:
        return f" {LITTLE_MORE}."
    bounds = [b for b in little if not b.startswith(_LIMIT_ITEM)]
    limits = [b for b in little if b.startswith(_LIMIT_ITEM)]
    where = [f"{_and_join(bounds)} allowed"] if bounds else []
    return f" {LITTLE_MORE}; the best it predicts sits at {_and_join(where + limits)}."


def progress_tied(best_no, other_no, score_text, ceiling):
    """A tie at equal scores only (R-I9): 'Formulations 1 and 3 are tied for
    best at 96.00 of 100.'"""
    return (f"{FORMULATION_CAP}s {best_no} and {other_no} are tied for best at "
            f"{_out_of(score_text, ceiling)}.")


def progress_one_round(values=False, cold=True):
    """Too early with one scored Round: the cold start says what it was."""
    if cold:
        return (f"One {ROUND_CAP} so far, laid out to cover your allowed "
                f"{'values' if values else 'amounts'}.")
    return f"One {ROUND_CAP} so far."


def progress_too_early(score_text, ceiling, no, rnd=None, one_round=False, values=False,
                       cold=True, tie_no=None, near_no=None, without=None):
    """Too early (improvement() None: one Round, or no earlier candidate).
    Snack bar: 'One Round so far, laid out to cover your allowed amounts.
    Formulations 1 and 3 are tied for best at 96.00 of 100.' Upgrade (fix
    round B, five-second re-test: "no earlier scored formulation" beside
    "7 scored" read as a contradiction): 'Best so far is 90.50 of 100
    (Formulation 6, Round 3). Rounds 1–2 were scored without Cook loss, so
    they are not compared with it.' `without`: (the Rounds of the earlier
    scored formulations, the measurements they lack), when every earlier
    one is left out of Best so far for a missing measurement."""
    if tie_no is not None:
        best = progress_tied(no, tie_no, score_text, ceiling)
    elif one_round:
        best = f"{_best_is(score_text, ceiling, no, rnd, near_no)}."
    elif without:
        rounds, names = without
        one = len(rounds) == 1
        best = (f"{_best_is(score_text, ceiling, no, rnd, near_no)}. {rounds_span(rounds)} "
                f"{'was' if one else 'were'} scored without {names}, so "
                f"{'it is' if one else 'they are'} not compared with it.")
    else:
        best = (f"{_best_is(score_text, ceiling, no, rnd, near_no)}; no earlier scored "
                f"{FORMULATION} to compare it with.")
    if one_round:
        return f"{progress_one_round(values, cold)} {best}"
    return best


def progress_no_new_best(last_round, score_text, ceiling, no, rnd, tie_no=None, near_no=None,
                         little=None):
    """No new best (one Round since), Line A: 'Round 5 did not beat Best so
    far, 90.45 of 100 (Formulation 15, Round 4; tied with Formulation 18).
    The app expects little more inside your current limits; the best it
    predicts sits at the most Textured pea protein and the least Beet juice
    powder allowed.' `near_no`: a runner-up within the noise threshold;
    `little`: _little_more's items, None when not said."""
    if tie_no is not None:
        return (f"{ROUND_CAP} {last_round} did not beat Best so far: "
                f"{progress_tied(no, tie_no, score_text, ceiling)}{_little_more(little)}")
    where = f"{FORMULATION_CAP} {no}, {ROUND_CAP} {rnd}{_tied_with(near_no)}"
    return (f"{ROUND_CAP} {last_round} did not beat Best so far, "
            f"{_out_of(score_text, ceiling)} ({where}).{_little_more(little)}")


def progress_luck(repeat_text, of_no=None):
    """R-I9: said at level 0 only while Predictions is Reliable. `of_no`:
    the formulation, named when a tie leaves "it" unclear."""
    who = "it" if of_no is None else f"{FORMULATION_CAP} {of_no}"
    return f", probably helped by luck: a Repeat of {who} is predicted near {repeat_text}"


def progress_flat(k, score_text, ceiling, no, rnd, luck_text=None, tie_no=None, near_no=None,
                  little=None):
    """Flat: 'No higher score in 7 Rounds. Best so far is 92.50 of 100
    (Formulation 24, Round 5; tied with Formulation 3), probably helped by
    luck: a Repeat of it is predicted near 90.' `luck_text`: the predicted
    Repeat, rounded; `near_no`, `little`: as progress_no_new_best."""
    head = f"No higher score in {k} {ROUND_CAP}s."
    if tie_no is not None:
        best = progress_tied(no, tie_no, score_text, ceiling)[:-1]
        luck = progress_luck(luck_text, no) if luck_text is not None else ""
    else:
        best = _best_is(score_text, ceiling, no, rnd, near_no)
        luck = progress_luck(luck_text) if luck_text is not None else ""
    return f"{head} {best}{luck}.{_little_more(little)}"


def _raised(rnd, rise_text, score_text, ceiling, no, near_no=None):
    return (f"{ROUND_CAP} {rnd} raised Best so far by {points_text(rise_text)}, to "
            f"{_out_of(score_text, ceiling)} ({FORMULATION_CAP} {no}"
            f"{_tied_with(near_no, after_round=False)})")


def _needed(need_text):
    """'the 3.2 points needed to count as a rise' (fix A follow-up: the
    threshold the rise is judged against, Improvement.threshold, not the
    noise SD it is built from)."""
    return f"the {points_text(need_text)} needed to count as a rise"


def progress_rising(rnd, rise_text, score_text, ceiling, no, need_text, near_no=None):
    """Rising: 'Round 5 raised Best so far by 3.5 points, to 89.84 of 100
    (Formulation 18, tied with Formulation 20), more than the 3.2 points
    needed to count as a rise.'"""
    raised = _raised(rnd, rise_text, score_text, ceiling, no, near_no)
    return f"{raised}, more than {_needed(need_text)}."


def progress_within_noise(rnd, rise_text, score_text, ceiling, no, need_text, near_no=None):
    """Within noise: the same, "less than"."""
    raised = _raised(rnd, rise_text, score_text, ceiling, no, near_no)
    return f"{raised}, less than {_needed(need_text)}."


def tie_runner_up(no, score_text, threshold_text):
    """R-I9: the runner-up within the noise threshold, said in Progress
    figures only (level 0 names equal scores alone). `threshold_text`:
    beat_by's words. 'Formulation 18 (90.12) is within the 1-point noise
    threshold of Best so far, so the two are tied.'"""
    return (f"{FORMULATION_CAP} {no} ({score_text}) is within {threshold_text} of Best so far, "
            "so the two are tied.")


# Next, R-I5: one plain sentence per next_step row. Today's sentence
# follows it verbatim inside Next-step reasoning.
def next_plain_tail(no=None, left=0):
    """The early row, and the tail of the others: 'Make the next Round.' or
    'Record the 4 formulations still open in Round 6, then make the next
    Round.'"""
    if left:
        return f"{record_open_round(no, left)}, then make the next {ROUND_CAP}."
    return f"Make the next {ROUND_CAP}."


def next_plain_repeat_ask(no, open_round=False, after_results=False, values=False):
    """R-I5's Repeat sentence while the threshold is assumed: 'Include a
    Repeat of Formulation 15 (the same amounts again) to measure how much
    results vary.' `open_round`: 'In the next Round, include …' (a Round is
    open, so this one goes in the next); `after_results`: the sentence before
    already said "results" (explore: '… to measure how much they vary.')."""
    lead = f"In the next {ROUND_CAP}, include" if open_round else "Include"
    gloss = "" if after_results else f" (the same {'values' if values else 'amounts'} again)"
    what = "they" if after_results else "results"
    return f"{lead} a Repeat of {FORMULATION_CAP} {no}{gloss} to measure how much {what} vary."


def next_plain_repeat(later, earlier, diff_text, lower=True, half_text=None, values=False,
                      round_no=None):
    """Row repeat (Line B; fix round B, walkthrough ruling: the check
    first, then its evidence, and no "since Round 4", which the app cannot
    see): 'Check what changed before you make Round 6. Formulation 21, a
    Repeat of Formulation 11, scored 10.5 points lower, more than two makes
    of one formulation should differ (up to 7.4 points). Look at the panel,
    suppliers and equipment.' `round_no`: the open Round, else the next
    one. The one "check" advice on the first screen (R-I2)."""
    makes = "runs" if values else "makes"
    make = "run" if values else "make"
    which = f"{ROUND_CAP} {round_no}" if round_no is not None else f"the next {ROUND_CAP}"
    limit = f" (up to {points_text(half_text)})" if half_text is not None else ""
    return (f"Check what changed before you {make} {which}. {FORMULATION_CAP} {later}, a Repeat "
            f"of {FORMULATION_CAP} {earlier}, scored {points_text(diff_text)} "
            f"{'lower' if lower else 'higher'}, more than two {makes} of one {FORMULATION} should "
            f"differ{limit}. Look at the panel, suppliers and equipment.")


def next_plain_misses(no=None, left=0):
    """Row misses (R-I5; "treat new predictions as rough" said without the
    style guide's retired words)."""
    return ("Recent results fell outside the range the app predicted more often than they "
            "should, so new predictions can miss by more than their range shows. "
            f"{next_plain_tail(no, left)}")


def next_plain_no_skill(no=None, left=0):
    """Row no_skill (Line C; "treat the suggestions as" said without the
    retired word)."""
    return (f"{next_plain_tail(no, left)} Until predictions beat the average, the suggestions are "
            "a guided search, not a forecast.")


def next_plain_untested(round_no=None, open_round=True, values=False, shown=True):
    """Row untested: the predicted best in amounts not tried, added as the
    user's own formulation to the open Round, or to the next. `shown`: its
    amounts are drawn under Progress figures (next_untested's rule)."""
    what = "values" if values else "amounts"
    where = (f"{ROUND_CAP} {round_no}" if open_round and round_no is not None
             else f"the next {ROUND_CAP}")
    pointer = f" (its {what} are under {FOLD_PROGRESS_FIGURES_NAME})" if shown else ""
    return (f"The model's highest prediction is in {what} you have not tried yet{pointer}. To "
            f"test it directly, add it to {where} as your own {FORMULATION}.")


def next_plain_explore(gain_text, assumed=True, no=None, left=0, repeat_no=None, values=False):
    """Row explore (Timing check): 'Make the next Round. The app still
    expects about 1.1 points of gain on average from amounts it has not
    tried, more than results are assumed to vary. Include a Repeat of
    Formulation 24 to measure how much they vary.' `gain_text`: shown_gain
    to 1 decimal; `repeat_no`: Best so far, while the threshold is assumed."""
    vary = "are assumed to vary" if assumed else "vary between Repeats"
    line = (f"{next_plain_tail(no, left)} The app still expects about {points_text(gain_text)} of "
            f"gain on average from {'values' if values else 'amounts'} it has not tried, more "
            f"than results {vary}.")
    if repeat_no is not None:
        line += " " + next_plain_repeat_ask(repeat_no, open_round=bool(left), after_results=True,
                                            values=values)
    return line


def next_plain_close(change_text="", settings_only=False, repeat_no=None, values=False):
    """Row close: 'You are close to the top within these ingredients and
    limits: no formulation is predicted to beat Best so far by more than
    results vary. To look further, raise the Highest of X.' (R-I5, with
    "nothing is predicted to" said as the style guide allows). The Set up
    button under it is today's secondary."""
    what = "process settings" if settings_only else "ingredients"
    line = (f"You are close to the top within these {what} and limits: no {FORMULATION} is "
            "predicted to beat Best so far by more than results vary.")
    if change_text:
        line += f" To look further, {change_text}."
    else:
        new = "a new process setting" if settings_only else "a new ingredient or process setting"
        line += f" To look further, add {new}."
    if repeat_no is not None:
        line += " " + next_plain_repeat_ask(repeat_no, values=values)
    return line


def next_plain_see(no=None, left=0, repeat_no=None, values=False):
    """Row see (Line A): 'Record the 4 formulations still open in Round 6.
    In the next Round, include a Repeat of Formulation 15 (the same amounts
    again) to measure how much results vary.'"""
    if repeat_no is None:
        return next_plain_tail(no, left)
    if left:
        return (f"{record_open_round(no, left)}. "
                f"{next_plain_repeat_ask(repeat_no, open_round=True, values=values)}")
    return f"Make the next {ROUND_CAP}. {next_plain_repeat_ask(repeat_no, values=values)}"


def next_plain_rising(no=None, left=0, repeat_no=None, values=False):
    """Row rising: 'Best so far is rising. Make the next Round.'"""
    return f"Best so far is rising. {next_plain_see(no, left, repeat_no, values)}"


# --- §2, R-I8, M3: the folds, named by what they hold ----------------
FOLD_PREDICTION_FIGURES = "Prediction figures"
FOLD_PROGRESS_FIGURES_NAME = "Progress figures"
FOLD_PROGRESS_FIGURES = FOLD_PROGRESS_FIGURES_NAME
FOLD_NEXT_REASONING = "Next-step reasoning"
FOLD_MEASURED_CHART = "Measured against predicted"
FOLD_PROGRESS_CHART = "Progress chart"
FOLD_METHOD = "Method and definitions"
FOLD_EXPLORED = "Explored and unexplored"


def try_one_fold(has_settings, has_ingredients=True):
    """'Try one ingredient or process setting' (§2; "setting" alone is not
    the app's word)."""
    if not has_ingredients:
        return "Try one process setting"
    return "Try one ingredient or process setting" if has_settings else "Try one ingredient"


def trade_offs_fold(has_triangle=True):
    """Was More charts (02 §4): the triangle needs three ingredients."""
    return "Trade-offs"


# §7, R-I6, R-I12: each figures fold opens with its technical basis; the
# fold labels carry no tooltip, so the help sits on these lines.
BASIS_PREDICTIONS = (
    f"Basis: each {ROUND_CAP} predicted by models fitted only to the {ROUND_CAP}s before it, "
    "recomputed with today's settings (rolling-origin, pseudo-prospective validation); 95 % "
    f"posterior predictive intervals widened by 5-fold cross-validation grouped by {ROUND_CAP}; "
    "skill against the "
    f"average of the {FORMULATION}s recorded before each {ROUND_CAP}.")
BASIS_PROGRESS = (
    f"Basis: Best so far is the highest measured Overall score among {FORMULATION}s with all "
    "results in; a rise counts from the noise threshold up, the pooled Repeat SD once it has 3 "
    "degrees of freedom, else 1 point; expected gain is the suggestion model's "
    "acquisition value.")
BASIS_NEXT = ("Basis: the Next-step rules are tried in the order listed under Method and "
              "definitions, and the first that holds is shown; its full sentence follows.")


# Concision round: each basis line on screen is short; the full one above
# opens its tooltip.
BASIS_SHORT = {"predictions": f"Basis: each {ROUND_CAP} predicted from earlier {ROUND_CAP}s only.",
               "progress": "Basis: measured scores, against the noise threshold.",
               "next": "Basis: the first Next-step rule that holds."}
BASIS_EXPLORED_SHORT = "Basis: strips are your records; the map is the model."


def basis_explored(values=False, total=True):
    """The Explored and unexplored fold's first line (§7 for §8)."""
    at = ("at the values run" if values
          else "at the Formulation total" if total else "at the amounts made")
    return (f"Basis: the strips are model-free, one input at a time, {at}; the map is the "
            "suggestion model's posterior mean and sd over a sample of the allowed "
            f"{FORMULATION}s.")


# --- §3, R-I6: tooltips, the plain reading first, then the BO term ----
# One "?" per line: a line's help joins the tips of the statistics it
# names, in screen order (W2's tooltip_for). Method and definitions holds
# every one of them too (M12: hover is hard on touch and keyboard).
TIP_INTERVAL = (
    "95 % interval: the range a new result should land in about 19 times in 20. BO term: 95 % "
    "posterior predictive interval of the Gaussian process (observation noise included), "
    "widened by a calibration factor from 5-fold cross-validation and cut at the Range.")
TIP_VERDICT = (
    "As expected, More misses than expected, Many more misses than expected: whether results "
    "land outside their 95 % intervals more often than 1 in 20. BO term: a coverage check, "
    "misses counted against a binomial at 5 % per value (as expected from 1 in 10 up, many "
    "more below 1 in 100).")
TIP_NOT_JUDGED = (
    "Not judged yet: fewer than 10 values checked, or every 95 % interval so wide that nearly "
    "any result lands inside. BO term: too small a validation sample, or uninformative "
    "intervals.")
TIP_SKILL = (
    f"Skill: how much better score predictions are than the average of earlier results; "
    f"{SKILL_SCALE}, below 0 = worse. BO term: out-of-sample skill score 1 − SSE/SSE(running "
    f"mean), pooled over {ROUND_CAP}s; a Q²-type score, not R².")
TIP_ROUND_AHEAD = (
    f"Round-ahead check: each {ROUND_CAP} predicted from the {ROUND_CAP}s before it only, then "
    "compared with what was measured. BO term: rolling-origin (pseudo-prospective) validation, "
    f"recomputed with today's settings, not frozen when the {ROUND_CAP} was made.")
TIP_MEAN_ERROR = (
    "Mean error: how far results came in above (+) or below (−) prediction on average. BO "
    "term: mean prediction error (measured − predicted); a steady bias when its size exceeds "
    "the noise threshold and 2 standard errors.")
TIP_NOISE_THRESHOLD = (
    f"Noise threshold: the smallest difference counted as more than chance: how much Repeats "
    f"of one {FORMULATION} differ, or 1 point, assumed, until the Repeats give 3 degrees of "
    "freedom (3 df). BO term: "
    "pooled replicate (pure-error) SD; not the Gaussian process's fitted noise.")


def tip_repeats(values=False):
    """Repeat, pair, df."""
    made = "values run" if values else "amounts made"
    return (f"Repeat: the same {made} again; a Repeat and its original are a pair. df: the "
            f"pooled SD's degrees of freedom, results minus {FORMULATION}s made more than once. "
            "BO term: replicates.")


TIP_PROGRESS_WORDS = (
    f"Rising: Best so far went up by at least the noise threshold in the last {ROUND_CAP}. "
    f"Within noise: up by less. Flat: no higher score in 2 or more {ROUND_CAP}s. Too early: "
    f"one {ROUND_CAP}, or no earlier {FORMULATION} to compare with. No new best: the one "
    f"{ROUND_CAP} since did not beat it. BO term: improvement of the best observed value, "
    "judged against replicate noise.")


def tip_predicted_score(values=False):
    return (f"Predicted score: the model's single best guess at these "
            f"{'values' if values else 'amounts'}. BO term: posterior mean of the Gaussian "
            "process.")


TIP_MODEL_SD = (
    "Model sd: how unsure the model is about its own guess here, without the noise of one "
    "result. BO term: latent posterior standard deviation (no observation noise, no "
    "calibration), narrower than the 95 % interval.")
TIP_PREDICTED_BEST = (
    f"Predicted best: the allowed {FORMULATION} the model scores highest. BO term: maximiser of "
    "the posterior mean over the feasible region, its interval widened by the Round-ahead "
    "record.")
TIP_EXPECTED_IMPROVEMENT = (
    f"Expected gain: what one new {FORMULATION} is expected to add over the model's best, "
    "on average, counting no gain as zero; an estimate, not a promise. BO term: the acquisition "
    "value, qLogNEI by default (qLogEI: over the best measured score; qUCB gives an optimistic "
    "score instead), conditional on the model.")


def tip_untested(values=False):
    what = "values" if values else "amounts"
    return (f"Untested {what}: no {FORMULATION} with all results in lies within 15 % of each "
            "varying row's Lowest–Highest span, all rows at once. BO term: outside an L∞ "
            "neighbourhood of the design points.")


def tip_local_effect(settings_only=False):
    what = "process setting" if settings_only else "ingredient"
    return (f"Local effect: how much the predicted score moves when only this {what} changes, "
            "near what you have made. BO term: range of the posterior mean along a one-factor "
            "trace over the tested stretch; unsigned, not a coefficient.")


TIP_ACTIVE_BOUNDS = (
    "At the Highest or Lowest of a row: the predicted best is pressed against that limit "
    "(within 1 % of its span), so widening it may let it go further. BO term: active bounds of "
    "the constrained maximiser.")
# The term itself is said in Method and definitions only (HOW_REGRESSION
# and the glossary; answer-key refresh item 7), so this tip points there.
TIP_LUCK = (
    "Probably helped by luck: Best so far is the highest of many scores, each with its own "
    "noise, so a Repeat of it is predicted lower. The statistics term is in Method and "
    "definitions.")
TIP_ROLES = (
    f"Refines the best, balanced, explores: how far a suggestion is from what was made: within "
    f"15 % of Best so far, more than 30 % from every {FORMULATION} recorded, or in between. BO "
    "term: distance labels; not the acquisition's exploitation–exploration split.")
TIP_NEAREST_TESTED = (
    f"Nearest tested: the scored {FORMULATION} closest to this suggestion, by its largest "
    "difference as a share of each span. BO term: nearest design point in the L∞ distance.")


# The status lines' exact-rule tooltips (§7, R-B1, I13: each output and
# window named on its own).
def tip_coverage(names, rounds, inside, total, allowed):
    """Line A: 'Measurements (Firmness, Juiciness, Cook loss), Rounds 3–5: 32
    of 35 Round-ahead values inside their 95 % posterior predictive
    intervals (5-fold cross-validation widening); a binomial count at 5 % per
    value allows up to 3 misses in 35 at the 1-in-10 level.'"""
    misses = "1 miss" if allowed == 1 else f"{allowed} misses"
    return (f"Measurements ({', '.join(names)}), {rounds_span(rounds)}: {inside} of {total} "
            "Round-ahead values inside their 95 % posterior predictive intervals (5-fold "
            f"cross-validation widening); a binomial count at 5 % per value allows up to {misses} "
            f"in {total} at the 1-in-10 level.")


def tip_coverage_early(names, rounds, n):
    """Fewer than 10 values in the verdict's Rounds."""
    where = f", {rounds_span(rounds)}" if rounds else ""
    return (f"Measurements ({', '.join(names)}){where}: {_values(n)} checked; the coverage check "
            "needs 10.")


TIP_COVERAGE_NONE = (f"Measurements: no {ROUND_CAP} checked yet; the coverage check starts with "
                     f"the first {ROUND_CAP} made after 5 {FORMULATION}s have results.")


def tip_skill(rounds, value_text, n):
    """'Overall score, Rounds 2–5: out-of-sample skill 1 − SSE/SSE(mean of
    the rows before each Round) = 0.36 over 15 formulations, at least 0.20
    over at least 10 needed.'"""
    return (f"{OVERALL_SCORE_COLUMN}, {rounds_span(rounds)}: out-of-sample skill 1 − SSE/SSE(mean "
            f"of the rows before each {ROUND_CAP}) = {value_text} over {n} {FORMULATION}s, at "
            "least 0.20 over at least 10 needed.")


def tip_skill_too_few(n):
    return (f"{OVERALL_SCORE_COLUMN}: out-of-sample skill needs 10 Round-ahead checks; {n} so "
            "far.")


TIP_REFITTED = f"Each {ROUND_CAP} is predicted by models refitted today on earlier {ROUND_CAP}s only."
# R-I11: what "Reliable" can and cannot promise, in its own tooltip.
TIP_RELIABLE_LIMITS = (
    f"Reliable can change between {ROUND_CAP}s by chance. In 24 simulated projects whose true "
    "skill was 0.20 or more, the skill read below 0.20 in 12.5 % of windows of 10 checks "
    "(n = 10) and 11.0 % of windows of 15 (n = 15); a model whose true skill is near 0.20 reads "
    "below 0.20 about half the time. The 95 % intervals: " + coverage_reading())


def tip_repeat_off(later, earlier, diff_text, half_text):
    """Row 1's rule (ftue fix A, cold read I7, and its follow-up): until the
    Repeat SD without the pair has 3 df, both limits; from 3 df, that
    Repeat SD alone. The pair stays in the reported Repeat SD."""
    return (f"{FORMULATION_CAP} {later}, a Repeat of {FORMULATION_CAP} {earlier}, differs by "
            f"{points_text(diff_text)}: more than ±{half_text}, the 95 % limit for the "
            "difference of two results. Until the other Repeats give 3 degrees of freedom "
            "(3 df), a pair must beat both √2 × 1.96 × the larger of the model's noise SD and "
            "the assumed 1 point, and √2 × the later one's Round-ahead half-width; from 3 df, "
            "√2 × t (95 %, on its df) × the Repeat SD without this pair. The pair stays in the "
            "Repeat SD.")


def tip_bias(gap_text, n, rounds):
    """The Overall score's steady bias. `gap_text` signed (signed())."""
    return (f"{OVERALL_SCORE_COLUMN}, {rounds_span(rounds)}: mean error {gap_text} points "
            f"(measured − predicted, n = {n}), beyond the noise threshold and 2 standard errors: "
            "a steady bias.")


def tip_wide(names):
    return (f"Very uncertain: the median 95 % half-width for {_and_join(names)} is over 25 % of "
            "its Range.")


def tip_guard(reasons):
    """`reasons`: GUARD_FLOOR and or GUARD_NOISE."""
    return (f"Noise guard: {'; '.join(reasons)}. Effects and expected gain are not "
            "reported; the status itself is unchanged.")


def tip_threshold_measured(sd_text, pairs, df=None):
    """'pooled Repeat SD 1.2 points, 3 df' (fix round B, statistician I11:
    its degrees of freedom, not its pairs); pairs only without df."""
    if df is not None:
        return f"pooled Repeat SD {sd_text} points, {df} df"
    return f"pooled Repeat SD {sd_text} points from {pairs_text(pairs)}"


TIP_THRESHOLD_ASSUMED = "1 point, assumed: the Repeats give fewer than 3 degrees of freedom (3 df)"


def tip_progress(threshold_text, need=None):
    """The Progress line's rule (§7; ftue fix A, cold read I6, and its
    follow-up): the rise against noise SD × √2 × the max-of-q factor.
    `threshold_text` (the noise SD's basis) is tip_threshold_measured's or
    TIP_THRESHOLD_ASSUMED. `need`: (threshold to the hundredth,
    rise_factor's words, the noise SD) for the latest Round, said last."""
    text = (f"Rise = Best so far minus the best before the latest scored {ROUND_CAP}, compared "
            "to the hundredth with noise SD × √2 × the max-of-q factor: the one-sided 95 % "
            f"point for the best of q new results (q: the latest {ROUND_CAP}'s scored "
            f"{FORMULATION}s; z, or t on the Repeats' df once they measure the noise SD). "
            f"Noise SD: {threshold_text}. Rising at or above it, Within noise below it, Flat "
            f"after 2 or more scored {ROUND_CAP}s with no higher score.")
    if need:
        need_text, factor_text, sd_text = need
        text += f" This {ROUND_CAP}: {sd_text} × {factor_text} = {points_text(need_text)}."
    return text


def tip_next_best(score_text, ceiling, no, rnd=None):
    """Next's see and rising rows: the number Progress weighs, report
    style. 'Best so far: 90.45 of 100 (Formulation 15, Round 4).'"""
    where = f", {ROUND_CAP} {rnd}" if rnd is not None else ""
    return f"Best so far: {score_text} of {ceiling_text(ceiling)} ({FORMULATION_CAP} {no}{where})."


def tip_next_gain(gain_text, threshold_text):
    """Next's explore row: the expected gain Progress shows against
    the noise threshold. `gain_text` to 1 decimal."""
    return (f"Expected gain: {points_text(gain_text)}, above the noise threshold "
            f"({threshold_text}).")


def tip_next(status_text, rule_text):
    """The Next line's rule: which row fired, and its condition (a row of
    status_rule_rows)."""
    return (f"Next-step rule that holds: {status_text}. {rule_text} The rules are tried in "
            "order and the first that holds is shown.")


# --- §7, R-I12, R-I13, M9: Model and method --------------------------
MODEL_AND_METHOD = "Model and method"
# The one sentence that names the field (R-B2: in exactly one allowlisted
# string, the summary's first).
MODEL_SUMMARY_LEAD = ("Bayesian optimisation: a Gaussian-process surrogate model of the Overall "
                      "score chooses each Round's suggestions; this tab also fits one Gaussian "
                      "process per measurement for its predictions.")
_ACQUISITION_NAMES = {
    # Fix round B (statistician I1): qNEI has no single incumbent.
    "qlognei": ("qLogNEI (noisy expected improvement, log form: in each posterior draw, the gain "
                "of a candidate over the highest latent value among the tested {f}s, counting no "
                "gain as zero, averaged over the draws; there is no single incumbent, and from "
                "the second pick on the earlier picks count as tested; shown as the exp of the "
                "log value, so approximate)"),
    "qlogei": "qLogEI (expected improvement over the best measured Overall score, log form)",
    "qucb": "qUCB (upper confidence bound, β = {beta})",
}


def _small(value):
    """'1e-6', '0.025': a setting as its library writes it."""
    return f"{float(value):g}".replace("e-0", "e-")


def _kernel_summary(f):
    name = KERNEL_NAMES.get(f.kernel, f.kernel)
    if f.kernel == "rbf":
        name = "squared exponential (RBF)"
    source = "Recommended settings" if f.recommended else "your model settings"
    text = f"Kernel ({source}): {name}"
    if f.ard:
        text += " with one lengthscale per input (ARD)"
    if f.lengthscale_prior:
        text += f", lengthscale prior {f.lengthscale_prior}"
    if f.lengthscale_floor is not None:
        text += f", at least {f.lengthscale_floor:g}"
    if f.outputscale:
        prior = f", prior {f.outputscale_prior}" if f.outputscale_prior else ""
        text += f"; outputscale fitted{prior}"
    else:
        text += "; no outputscale"
    return f"{text}."


def _noise_summary(f):
    if f.noise == "fixed":
        return "Likelihood: Gaussian, noise fixed near zero by your model settings."
    prior = f", prior {f.noise_prior}" if f.noise_prior else ""
    floor = f", at least {_small(f.noise_floor)}" if f.noise_floor is not None else ""
    return f"Likelihood: Gaussian, noise fitted{prior}{floor}."


def _inputs_summary(f):
    if f.values:
        return ("Inputs: each process setting normalised to [0, 1] by its own Lowest–Highest (a "
                "box, not the constrained design space).")
    at = ("amounts brought to the Formulation total, then each column" if f.total
          else "amounts as made, each column")
    return (f"Inputs: {at} normalised to [0, 1] by its own Lowest–Highest (a box, not the "
            "constrained design space); choices one-hot; calculated rows and the "
            f"{FILL_TO_TOTAL} row have no column of their own.")


def model_summary(facts):
    """Method and definitions' first part (§7, R-I12, R-I13), from facts
    read off a model actually built for this project (insights.model_summary,
    W2), never from hard-coded defaults: a list of lines. At the gate (M9)
    only the cold start and the model to come."""
    f = facts
    cold = (f"Cold start: the first {f.cold_start} {FORMULATION}s with all results in come from "
            f"a space-filling design over the allowed {'values' if f.values else 'amounts'}, not "
            "from the acquisition.")
    if f.cold:
        return [cold, (f"The model to come: a Gaussian process on the Overall score, fitted once "
                       f"{f.cold_start} {FORMULATION}s have all their results; its predictions are "
                       f"checked {ROUND_CAP} by {ROUND_CAP} from then on.")]
    acq = _ACQUISITION_NAMES.get(f.acquisition, f.acquisition).format(
        f=FORMULATION, beta=f"{f.beta:g}" if f.beta is not None else "")
    q = (f"q = {f.q} per {ROUND_CAP}" if f.q is not None
         else f"q = the number of {FORMULATION}s generated")
    lines = [
        MODEL_SUMMARY_LEAD,
        (f"Surrogate: BoTorch SingleTaskGP on the Overall score, fitted to {f.scored} "
         f"{FORMULATION}s with all results in, {f.inputs} inputs; constant mean; outcome "
         f"{'standardised (mean 0, SD 1)' if f.standardised else 'as measured'}."),
        _kernel_summary(f),
        _noise_summary(f),
        _inputs_summary(f),
        ("Hyperparameters: maximum a posteriori (marginal likelihood × the priors above), point "
         "estimates, not marginalised; the suggestion model uses BoTorch's default fit, the "
         f"models on this tab L-BFGS-B with ftol {_small(f.fit_tolerance)}."),
        (f"Acquisition: {acq}; {q}, chosen one at a time (sequential greedy) under the linear "
         f"limits, {f.restarts} restarts from {f.raw_samples} raw samples, {f.mc_samples} "
         "quasi-Monte Carlo samples."),
    ]
    # Review story 3: Expected gain defined once, beside the acquisition.
    if f.acquisition == "qlognei":
        lines.append(METHOD_GAIN)
    lines.append(cold)
    if f.robust:
        lines.append("Input warping: on for the suggestion model (robust), not used by the "
                     "models on this tab.")
    lines += [
        (f"Interval calibration: each model's 95 % intervals are multiplied by the "
         f"root-mean-square of {f.folds}-fold cross-validated standardised errors (at least 1), "
         f"the folds grouped by {ROUND_CAP} with a {FORMULATION} and its Repeats always in one "
         "fold, the hyperparameters refitted in each; the Overall score's are widened again by "
         f"earlier {ROUND_CAP}s' Round-ahead record."),
        (f"Validation: rolling-origin (pseudo-prospective): each {ROUND_CAP} predicted by models "
         f"refitted on earlier {ROUND_CAP}s only, recomputed with today's measurements, targets, "
         "Share of score and model settings. Skill, coverage and bias are defined under How each "
         f"status is decided. Changing model settings re-scores every earlier {ROUND_CAP}, so a "
         "record read after choosing settings to suit it is no longer a fair test."),
    ]
    return lines


MODEL_LIMITS_HEADING = "Known limits"
# R-I12's known limits, §7's own, R-I11's two figures and §8's reason for
# drawing no projection of the whole space (02-chart-audit §6).
MODEL_LIMITS = (
    (f"Coverage: the 95 % intervals aim for 95 %; on simulated projects built by the app's "
     f"suggestions they covered about {SIMULATED_COVERAGE['measurements']} % of measured values "
     f"and {SIMULATED_COVERAGE['score']} % of Overall scores, short of 95 %."),
    # Fix round B (statistician I4): the error rates exactly, with their n.
    ("Reliable can change between Rounds by chance. In 24 simulated projects whose true skill "
     "was 0.20 or more, the skill read below 0.20 in 12.5 % of windows of 10 checks (n = 10) "
     "and 11.0 % of windows of 15 (n = 15); a model whose true skill is near 0.20 reads below 0.20 "
     "about half the time. No interval is shown for the skill."),
    # Fix round B (statistician I10).
    (f"Misses within a {ROUND_CAP}, and across the measurements of one {FORMULATION}, are not "
     "independent, so the coverage check's 1 in 10 and 1 in 100 are approximate: it flags too "
     "often."),
    # Fix round B (statistician I12).
    (f"Changing model settings re-scores every earlier {ROUND_CAP}, so a record read after "
     "choosing settings to suit it is no longer a fair test."),
    ("The models on this tab are refitted with their own tolerance, so they differ slightly "
     "from the suggestion model."),
    "Intervals are cut at 0 and the score ceiling, so coverage near 100 reads high.",
    f"{FORMULATION_CAP}s still waiting for results are left out of every model.",
    (f"Cold start: the first 5 {FORMULATION}s with all results in come from a space-filling "
     f"design, not from the model, so the first {ROUND_CAP}s show nothing of how well it "
     "chooses."),
    "Best so far is one noisy maximum, so it tends to read high.",
    ("A noise threshold from few Repeats rests on few degrees of freedom: at 3 df its 95 % "
     "interval runs from about 0.57 to 3.7 times the estimate."),
    "No correction is made for the many comparisons on this tab.",
    (f"A change at the bench shows only through a Repeat of an earlier {FORMULATION} in each "
     f"{ROUND_CAP}: the Round-ahead check learns it from its first {ROUND_CAP} on."),
    ("No 2-D map of the whole space is drawn: a principal-component projection keeps about 35 % "
     "of its variation, and 75–89 % of nearest neighbours in such a map are wrong."),
)


# --- §7, I13, R-I13: How each status is decided ----------------------
STATUS_TABLE = "How each status is decided"
# Fix round B (statistician I8): the precedence, under the table.
STATUS_TABLE_ORDER = ("Within each line the rows are tried from top to bottom; the first that "
                      "holds is the status shown.")
STATUS_TABLE_COLUMNS = ("Line", "Status", "Exact rule")


def status_rule_rows(cut_text="−0.20", settings_only=False):
    """One row per state of review B1's table and per Next row, each line in
    first-match order, with its exact condition. `cut_text`: the skill
    cut (insights.skill_cut at n = 10). Said through for_project."""
    one = "process setting" if settings_only else "ingredient"
    p, g, n = STATUS_PREDICTIONS, STATUS_PROGRESS, STATUS_NEXT
    words = PREDICTIONS_WORDS
    window = (f"the verdict's {ROUND_CAP}s (the last 3 checked, more until 10 measured values)")
    return [
        (p, f"{words['off']} (Repeat)",
         f"A Repeat differs from its original by more than √2 × the later one's Round-ahead 95 % "
         f"half-width, in {window}. Its pair is then left out of the Repeat SD."),
        (p, f"{words['off']} (coverage)",
         f"Many more misses than expected: in {window}, as many misses outside the 95 % "
         "intervals or more would happen less than 1 time in 100 under a binomial at 5 % per "
         "value, which treats misses as independent (they are not quite: see Known limits)."),
        (p, f"{words['not_yet']} (gate)", f"Fewer than 5 {FORMULATION}s with all results in."),
        (p, f"{words['not_yet']} (more misses)",
         "More misses than expected: as many misses or more would happen less than 1 time in "
         "10, and at least 1 time in 100."),
        (p, f"{words['not_yet']} (too few checks)",
         f"No {ROUND_CAP} checked, fewer than 10 measured values over every {ROUND_CAP} "
         "checked, or fewer than 10 Round-ahead checks of the Overall score for skill."),
        (p, f"{words['not_yet']} (very uncertain)",
         f"Every output's median 95 % half-width over 25 % of its Range in {window}."),
        (p, f"{words['not_yet']} (no better than the average)",
         f"Skill at or below the cut ({cut_text}), from 10 checks."),
        (p, f"{words['not_yet']} (not clearly better)", f"Skill above {cut_text} and below 0.20."),
        (p, words["reliable"],
         "As expected (as many misses or more would happen at least 1 time in 10), or wider "
         "than needed (all inside, at "
         "least 20 values, median absolute standardised error at most 0.4), and skill 0.20 or "
         "more from 10 checks."),
        (p, "Clause: still very uncertain",
         "Some outputs, not all, have a median 95 % half-width over 25 % of their Range."),
        (p, "Clause: scores have come in below or above the predictions",
         "The Overall score's mean error is beyond the noise threshold and 2 standard errors "
         f"(from 4 {ROUND_CAP}s checked, each {ROUND_CAP}'s mean error counts as one value)."),
        (p, f"Clause: does not yet read what each {one} does",
         "Reliable while the noise guard holds: the lengthscale at its lower limit, or the "
         "fitted noise under half the Repeat SD."),
        (g, PROGRESS_WORDS["rising"],
         f"Best so far came from the latest scored {ROUND_CAP} and beat the best before it by at "
         "least the noise threshold, to the hundredth."),
        (g, PROGRESS_WORDS["within_noise"], "The same, by less than the noise threshold."),
        (g, PROGRESS_WORDS["flat"], f"No higher score in the last 2 or more scored {ROUND_CAP}s."),
        (g, PROGRESS_WORDS["no_new_best"], f"No higher score in the one {ROUND_CAP} since."),
        (g, PROGRESS_WORDS["too_early"],
         f"One scored {ROUND_CAP}, or no earlier {FORMULATION} to compare Best so far with."),
        (n, "Repeat", "A Repeat disagrees, as for Off (Repeat)."),
        (n, "Misses", "Many more misses than expected."),
        (n, "Early", f"No {ROUND_CAP} checked, fewer than 10 values, or every output very "
                     "uncertain."),
        (n, "No better than the average", f"Skill at or below the cut ({cut_text})."),
        (n, "Untested",
         "Effects reported (skill 0.20 or more from 10 checks) and not More misses than "
         "expected; the predicted best beats a Repeat of Best so far by more than the noise "
         f"threshold; and no {FORMULATION} with all results in lies within 15 % of each span of "
         "it."),
        (n, "Explore",
         "Effects reported and not More misses than expected; no predicted gain beyond the "
         "noise threshold; Progress Flat or Within noise; expected gain above the noise "
         "threshold."),
        (n, "Close", "The same, with the expected gain at or under the noise threshold."),
        (n, "Rising", "Progress Rising."),
        (n, "See", "Any other case."),
    ]


# --- §3, §7, R-B2, M4, M5, M8: the glossary --------------------------
# Plain word on screen → the standard BO term → one line of meaning, in one
# canonical order (M8): the status lines, then the folds top to bottom.
# Written out in both forms (M4) and as plain literals, never f-strings: the
# few cells that say a word _BANNED keeps for everything else are allowed
# as exact strings (tests/test_food_bo._FTUE_ALLOWED), and only here.
GLOSSARY = "Glossary: word on screen → BO term"
GLOSSARY_COLUMNS = ("Word on screen", "BO term", "Meaning")
GLOSSARY_OVERALL_TERM = "objective (an additive desirability, weighted by Share of score)"
GLOSSARY_ROUND_TERM = "batch of q suggestions"
GLOSSARY_ROUND_MEANING = ("the formulations made together; in BO texts a “batch” of q generated "
                          "suggestions")
GLOSSARY_ROWS = (
    ("Reliable · Not yet · Off", "validation verdict",
     "The Predictions status: coverage of the 95 % intervals and out-of-sample skill together. "
     "Not yet covers the Prediction check's Not judged yet."),
    ("the range the app predicted · 95 % interval",
     "95 % posterior predictive interval (calibrated)",
     "Where one new measured value should land: posterior mean ± 1.96 × predictive SD × "
     "calibration factor, cut at the Range. " + coverage_reading()),
    ("score predictions beat the average · skill", "out-of-sample skill score (Q²-type, not R²)",
     "1 − SSE/SSE(running mean of earlier results), pooled over Rounds; 0 = as good as the "
     "average, 1 = perfect."),
    ("scores have come in below or above the predictions · mean error",
     "mean prediction error (bias)",
     "Average of measured − predicted; a steady bias beyond the noise threshold and 2 standard "
     "errors."),
    ("Rising · Within noise · Flat · Too early · No new best",
     "improvement of the best observed value against replicate noise",
     "The rise of Best so far in the latest Round, compared with the noise threshold."),
    ("Best so far", "best observed value",
     "The highest measured Overall score among formulations with all results in. Not what "
     "qLogNEI compares with: it has no single incumbent, but takes the highest latent value "
     "among the tested formulations in each posterior draw."),
    ("probably helped by luck", "regression to the mean",
     "A Repeat of Best so far is predicted below its measured score by more than the noise "
     "threshold."),
    ("Repeat · pair · df", "replicate · replicate pair · degrees of freedom",
     "The same amounts made again; a Repeat and its original are a pair; df = (results of "
     "formulations made more than once) − (number of such formulations)."),
    ("noise threshold", "pooled replicate (pure-error) SD",
     "Pooled SD of Repeats of the Overall score once it has 3 degrees of freedom (3 df; at "
     "least 0.05), else 1 point, assumed; not the Gaussian process's fitted noise."),
    ("Where you have tested", "marginal coverage of each input (model-free)",
     "Each recorded formulation as a tick at its amount, one input at a time; combinations can "
     "be untested where every strip looks covered."),
    ("Every allowed formulation (map)", "posterior mean against latent posterior sd",
     "Each grey dot is an allowed formulation from a hit-and-run sample of the feasible region; "
     "high = unexplored, right = predicted good."),
    ("allowed amounts", "design space; with your limits, the feasible region",
     "Every formulation within each row's Lowest–Highest that meets every limit."),
    ("Round-ahead check", "rolling-origin (pseudo-prospective) validation",
     "Each Round predicted by models refitted on earlier Rounds only, recomputed with today's "
     "settings."),
    ("As expected · More misses than expected · Many more misses than expected · Not judged yet",
     "coverage test (binomial at 5 % per value)",
     "Misses outside the 95 % intervals against a binomial at 5 %: as expected from 1 in 10 up, "
     "many more below 1 in 100; Not judged yet under 10 values or with every interval wide."),
    ("calibration factor", "cross-validated calibration factor",
     "Root-mean-square of 5-fold cross-validated standardised errors (folds grouped by Round, "
     "Repeats kept together), at least 1; every 95 % interval is multiplied by it."),
    ("predicted score · model mean", "posterior mean", "The model's single best guess."),
    ("predicted best", "maximiser of the posterior mean over the feasible region",
     "The allowed formulation the model scores highest; at a Highest or Lowest it sits on an "
     "active bound."),
    ("expected gain", "expected improvement (acquisition value; qLogNEI by default; qLogEI; qUCB)",
     "Average gain of one new formulation, counting no gain as zero: under qLogNEI, over the "
     "highest latent value among the tested formulations in each posterior draw (no single "
     "incumbent; shown as the exp of a log value, so approximate); under qLogEI, over the best "
     "measured score."),
    ("model sd · how unsure the model is · Uncertainty", "latent posterior standard deviation",
     "The model's own uncertainty about its mean, without observation noise or calibration; "
     "narrower than the 95 % interval."),
    ("tested near · untested amounts", "inside or outside an L∞ neighbourhood of the design points",
     "Within 15 % of each varying row's Lowest–Highest span, all rows at once, of a formulation "
     "with all results in."),
    ("local effect", "range of the posterior mean along a one-factor trace",
     "Largest minus smallest predicted score as one ingredient changes, over the tested "
     "stretch; unsigned, not a coefficient."),
    ("correlated (|r|)", "collinear inputs (confounded effects)",
     "|Pearson r| at least 0.8 between two ingredients across formulations with all results in."),
    ("Overall score", GLOSSARY_OVERALL_TERM,
     "Sum over measurements of closeness to the goal (0–1) × Share of score, 0–100."),
    ("measurement", "outcome (response)", "A measured result the Overall score is built from."),
    ("ingredient · process setting", "input (factor); ingredients are mixture components",
     "A column of the model, normalised by its own Lowest–Highest."),
    ("model", "surrogate model: a Gaussian process (BoTorch SingleTaskGP)",
     "Learns the Overall score from the formulations with all results in; one more per "
     "measurement for this tab."),
    ("how smoothly results change", "kernel (covariance function)",
     "Recommended settings: squared exponential (RBF) with ARD; the kernel in use is named "
     "under Model and method."),
    ("correlation length", "lengthscale (one per input: ARD)",
     "How far an input must move before results change much; at its lower limit, the model is "
     "fitting noise."),
    ("Round", GLOSSARY_ROUND_TERM, GLOSSARY_ROUND_MEANING),
    ("laid out to cover your allowed amounts", "space-filling initial design (cold start)",
     "The first 5 formulations with all results in come from a space-filling design, not the "
     "acquisition."),
    ("suggestions", "candidates that maximise the acquisition",
     "Chosen one at a time (sequential greedy), each valued with the earlier picks counted as "
     "pending."),
    ("refines the best · balanced · explores", "distance labels",
     "Within 15 % of Best so far, more than 30 % from everything recorded, or in between; not "
     "the acquisition's exploitation–exploration split."),
)

# The process-settings form (M4): the rows that name formulations, amounts
# or ingredients, rewritten; every other row is shared.
GLOSSARY_ROUND_MEANING_SETTINGS = ("the trials run together; in BO texts a “batch” of q "
                                   "generated suggestions")
_GLOSSARY_SETTINGS_FORM = {
    "Best so far": (
        "Best so far", "best observed value",
        "The highest measured Overall score among trials with all results in. Not what "
        "qLogNEI compares with: it has no single incumbent, but takes the highest latent value "
        "among the tested trials in each posterior draw."),
    "Repeat · pair · df": (
        "Repeat · pair · df", "replicate · replicate pair · degrees of freedom",
        "The same values run again; a Repeat and its original are a pair; df = (results of "
        "trials run more than once) − (number of such trials)."),
    "Where you have tested": (
        "Where you have tested", "marginal coverage of each input (model-free)",
        "Each recorded trial as a tick at its value, one process setting at a time; "
        "combinations can be untested where every strip looks covered."),
    "Every allowed formulation (map)": (
        "Every allowed trial (map)", "posterior mean against latent posterior sd",
        "Each grey dot is an allowed trial from a hit-and-run sample of the feasible region; "
        "high = unexplored, right = predicted good."),
    "allowed amounts": (
        "allowed values", "design space; with your limits, the feasible region",
        "Every trial within each setting's Lowest–Highest that meets every limit."),
    "predicted best": (
        "predicted best", "maximiser of the posterior mean over the feasible region",
        "The allowed trial the model scores highest; at a Highest or Lowest it sits on an "
        "active bound."),
    "expected gain": (
        "expected gain", "expected improvement (acquisition value; qLogNEI by default; qLogEI; qUCB)",
        "Average gain of one new trial, counting no gain as zero: under qLogNEI, over the "
        "highest latent value among the tested trials in each posterior draw (no single "
        "incumbent; shown as the exp of a log value, so approximate); under qLogEI, over the "
        "best measured score."),
    "tested near · untested amounts": (
        "tested near · untested values",
        "inside or outside an L∞ neighbourhood of the design points",
        "Within 15 % of each varying setting's Lowest–Highest span, all at once, of a trial "
        "with all results in."),
    "local effect": (
        "local effect", "range of the posterior mean along a one-factor trace",
        "Largest minus smallest predicted score as one process setting changes, over the "
        "tested stretch; unsigned, not a coefficient."),
    "correlated (|r|)": (
        "correlated (|r|)", "collinear inputs (confounded effects)",
        "|Pearson r| at least 0.8 between two process settings across trials with all results "
        "in."),
    "ingredient · process setting": (
        "process setting", "input (factor)",
        "A column of the model, normalised by its own Lowest–Highest."),
    "model": (
        "model", "surrogate model: a Gaussian process (BoTorch SingleTaskGP)",
        "Learns the Overall score from the trials with all results in; one more per "
        "measurement for this tab."),
    "Round": ("Round", GLOSSARY_ROUND_TERM, GLOSSARY_ROUND_MEANING_SETTINGS),
    "laid out to cover your allowed amounts": (
        "laid out to cover your allowed values", "space-filling initial design (cold start)",
        "The first 5 trials with all results in come from a space-filling design, not the "
        "acquisition."),
}
GLOSSARY_ROWS_SETTINGS = tuple(_GLOSSARY_SETTINGS_FORM.get(row[0], row) for row in GLOSSARY_ROWS)


def _says_trial(text):
    return any(word.strip(".,;:()“”") in ("trial", "trials") for word in text.split())


# The process-settings form's cells that type the word for themselves, as
# PROCESS_STUDY_INTRO does: the one other allowance the glossary needs.
GLOSSARY_SETTINGS_TRIAL_CELLS = tuple(cell for row in _GLOSSARY_SETTINGS_FORM.values()
                                      for cell in row if _says_trial(cell))


def glossary_rows(settings_only=False):
    """The glossary in the project's own form, chosen by has_ingredients()
    (M4), never through for_project."""
    return GLOSSARY_ROWS_SETTINGS if settings_only else GLOSSARY_ROWS


def glossary_cells():
    """Every cell of both forms, for the guard's scoped allowance."""
    return [cell for rows in (GLOSSARY_ROWS, GLOSSARY_ROWS_SETTINGS) for row in rows
            for cell in row]


# --- §8, R-I14, R-I15, R-I16: explored and unexplored ----------------
STRIPS_TITLE = "Where you have tested"


def strips_caption(settings_only=False):
    """Always under the strips (R-I15): a 1-D view hides untested
    combinations."""
    what = "process setting" if settings_only else "ingredient"
    return f"One {what} per strip; combinations may be untested."


def strips_caption_help(settings_only=False):
    """strips_caption's former sentence, now in its tooltip."""
    what = "process setting" if settings_only else "ingredient"
    return (f"Each strip shows one {what} at a time: combinations can be untested even where "
            "every strip looks covered.")


def gap_above(name, value_text):
    """'Coconut oil above 8.1 g': an end gap; `value_text` with its unit."""
    return f"{name} above {value_text}"


def gap_below(name, value_text):
    return f"{name} below {value_text}"


def gap_between(name, low_text, high_text):
    """'Salt between 0.6 and 1.0 g': `high_text` carries the unit."""
    return f"{name} between {low_text} and {high_text}"


def strip_gaps_line(items):
    """At most 2, largest first (R-I15): 'Coconut oil above 8.1 g and Salt
    below 0.6 g have never been tested.'"""
    verb = "has" if len(items) == 1 else "have"
    return f"{_and_join(items)} {verb} never been tested."


def option_never_tested(name, option):
    """A choice row's option no formulation with all results in has used."""
    return f"{name}: {option} has never been tested."


def strips_no_gap(settings_only=False, has_settings=False):
    """`has_settings`: the strips hold a process setting too (the rule
    checks them as it checks the ingredients)."""
    what = ("process setting" if settings_only
            else "ingredient or process setting" if has_settings else "ingredient")
    return f"No {what} has an untested stretch of 15 % of its span or more."


def limit_hit_line(hits, picks, round_no, limit_text):
    """R-I15: from 2 picks, when at least half sit at one row's limit.
    `limit_text`: limit_highest, limit_lowest or limit_named. '3 of the 4
    suggestions in Round 6 sit at Textured pea protein's Highest.'"""
    if hits == picks:
        who = (f"Both suggestions in {ROUND_CAP} {round_no}" if picks == 2
               else f"All {picks} suggestions in {ROUND_CAP} {round_no}")
        return f"{who} sit at {limit_text}."
    return f"{hits} of the {picks} suggestions in {ROUND_CAP} {round_no} sit at {limit_text}."


def limit_highest(name):
    return f"{name}'s {HIGHEST_LABEL}"


def limit_lowest(name):
    return f"{name}'s {LOWEST_LABEL}"


def limit_named(text):
    """A limit over several rows: 'the limit Fat per 100 g: at most 10 g'."""
    return f"the limit {text}"


# The strips' marks and tooltips.
STRIP_TESTED = "Tested (all results in)"
STRIP_WAITING = f"Waiting or {NOT_SCORED}"
STRIP_SUGGESTED = "This Round's suggestion"
STRIP_REACHABLE = "Reachable within your limits"
STRIP_LATEST = "Latest Round"
STRIP_COUNT = "Times tested"


def strip_value_title(values=False):
    """The tooltip's first field: 'Amount' or 'Value'."""
    return "Value" if values else "Amount"


def tip_strips(values=False, total=True):
    """The strips' "?" (§8 expert layer): what a tick and the shading are."""
    at = ("" if values or not total
          else ", at the Formulation total")
    return (f"Model-free: each tick is a recorded {FORMULATION}{at}. Shaded: the reachable part, "
            "the lowest to highest of this row over a hit-and-run sample of the feasible region. "
            "A gap is a reachable stretch of 15 % of the span or more with no "
            f"{FORMULATION} with all results in. One input at a time: a marginal view. BO "
            "term: one-dimensional marginals of the design points; a strip with no gap does not "
            "mean the design space is covered.")
TIP_LIMIT_HITS = (
    f"A suggestion sits at a limit when it is within 1 % of the span from it. Counted over the "
    f"generated picks of the open {ROUND_CAP}, else of the last {ROUND_CAP} the model chose; "
    f"your own {FORMULATION}s and Repeats are left out. BO term: active bounds.")

# The map (§8.2): the Why fold's chart, shared (explore_map_chart).
MAP_TITLE = f"Every allowed {FORMULATION}: predicted score vs uncertainty"
MAP_READING = "High = unexplored; right = predicted good; suggestions balance the two."
# R-I16: shown with this caption under the A3 gate or a caution or
# unreliable verdict; hidden under the noise guard.
MAP_NOT_YET = "Up: the model's uncertainty; the model is not yet reliable here."
MAP_HIDDEN = "Not drawn: the model is reading chance variation as effects."
MAP_COLD = f"The map appears at 5 scored {FORMULATION}s."
TIP_MAP = (
    "Across: posterior mean of the Overall-score suggestion model; up: latent posterior sd. "
    "Grey: a hit-and-run sample of the feasible region; rings: tested design points; "
    f"labelled: this {ROUND_CAP}'s picks; dashed: Best so far's predicted Repeat.")


def tip_design_space(n_text):
    """T0′ for W3 (§8 expert layer: "design space, feasible region sampled
    by get_polytope_samples"): added to the strips' and the map's "?".
    `n_text`: the pool's size, "3,000"."""
    return ("Lowest–Highest of every row: the design space; the part of it where every limit "
            "holds: the feasible region, sampled by hit-and-run (BoTorch get_polytope_samples, "
            f"{n_text} points).")


def explored_pointer(fold_label, settings_only=False, triangle=True):
    """§8.3: the in-context views, said once under the map, shown only when
    the Uncertainty layer is offered (R-I16). Concision round: a 5-word
    link; where it lands (explored_pointer_help) is its tooltip."""
    what = "process setting" if settings_only else "ingredient"
    return f"Show uncertainty along one {what}"


def explored_pointer_help(fold_label, settings_only=False, triangle=True):
    """The pointer's former sentence, now its tooltip."""
    what = "process setting" if settings_only else "ingredient"
    line = f"The same uncertainty along one {what}: {fold_label}"
    if triangle and not settings_only:
        line += "; across three ingredients: the triangle under Trade-offs"
    return f"{line}. Choose {LAYER_SD} under {LAYER_PICK}."


# --- §1, R-I3, R-I4, R-I14: the link labels --------------------------
def why_link(no):
    """R-I4: only with a warm Round open with at least one generated pick.
    Concision round: at most 5 words."""
    return f"Why {ROUND_CAP} {no} was suggested"


def explored_link(values=False):
    """R-I14: Next's link to the fold, priority 3 (explore and untested)."""
    return f"See untried {'values' if values else 'amounts'}"


def repeat_link(no):
    """R-I3: an owner item, ruled in by the early walkthrough (ftue fix C).
    Concision round: 'Repeat Formulation 24'."""
    return f"Repeat {FORMULATION_CAP} {no}"


# --- ftue fix C ---
# The Why fold in plain words, the BO term in each tooltip (early
# walkthrough rulings; 05-walkthrough-retest-early §5 items 2-3;
# statistician cold read I1), and the Repeat link's landing (§5 item 5).
# Every string here is new; the ones it stands in for on screen are named
# beside it and are left as they were for the wording round to retire.

# The Why fold's first line: the three role words, each read once.
WHY_ROLE_KEY = (f"{SUGGESTION_CLOSE.capitalize()}: near your best. "
                f"{SUGGESTION_DIFFERENT.capitalize()}: far from all made. "
                f"{SUGGESTION_BALANCED.capitalize()}: between.")

# Stands in for EXPECTED_IMPROVEMENT as the Why fold's column header.
EXPECTED_GAIN = "Expected gain over your best (points)"


def tip_expected_gain(kind="qlognei"):
    """The gain column's "?" (stands in for EXPECTED_IMPROVEMENT_HELP and
    EXPECTED_IMPROVEMENT_EI_HELP there): the plain reading, then the BO
    term. qLogNEI has no single incumbent (statistician I1): in each
    posterior draw the gain is over the highest latent value among the
    tested formulations. Both are shown as exp of the log value."""
    plain = (f"Expected gain over your best: what one new {FORMULATION} is expected to add to "
             "your best score, in points. It is an average that counts no gain as zero: an "
             "estimate, not a promise.")
    if kind == "qlogei":
        term = ("BO term: expected improvement (the acquisition value, qLogEI) over Best so "
                "far's measured score, averaged over 512 quasi-Monte Carlo draws; from the "
                "second pick on, valued together with the earlier picks. Shown as the "
                "exponential of the log value, so approximate.")
    else:
        term = ("BO term: expected improvement (the acquisition value, qLogNEI). In each "
                "posterior draw it is measured against the noisy incumbent, the highest latent "
                f"value among the tested {FORMULATION}s, then averaged over 512 quasi-Monte "
                "Carlo draws; there is no single incumbent, and from the second pick on the "
                "earlier picks count as tested. Shown as the exponential of the log value, so "
                "approximate.")
    return f"{plain} {term}"


def why_picked_after(numbers):
    """Stands in for WHY_PENDING: the picks marked · named by number."""
    each = "each adds" if len(numbers) > 1 else "it adds"
    return f"{formulations_span(numbers)}, marked {PENDING_MARK}: what {each} to earlier picks."


TIP_PICKED_AFTER = (
    "BO term: sequential greedy selection. Each later pick's value is computed with the earlier "
    "picks as pending points, so it is what that pick adds to them.")

# Stands in for WHY_RECOMPUTED and WHY_RECOMPUTED_WITHHELD; the reason
# moves to the tooltip.
WHY_RECALCULATED = "Figures recalculated from today's results."
TIP_RECALCULATED = (
    f"This {ROUND_CAP}'s generation record is missing or no longer matches its rows, so these "
    f"figures come from a model refitted to today's results, not from the model that chose the "
    f"{ROUND_CAP}.")

# Stands in for NEAREST_TESTED, TIP_NEAREST_TESTED and LARGEST_DIFFERENCES_HELP.
CLOSEST_MADE = f"Closest {FORMULATION} already made"
TIP_CLOSEST_MADE = (
    f"Closest {FORMULATION} already made: the scored {FORMULATION} nearest this suggestion, by "
    "its largest difference as a share of each span. It can differ from Best so far, which "
    "Compared with uses. BO term: nearest design point in the L∞ distance.")
LARGEST_DIFFERENCES_FROM_CLOSEST = (
    f"The two largest differences from the closest {FORMULATION} already made, largest share of "
    "its span first, in each row's own unit.")


def why_compared_with(best_no):
    """Under the Why table: the Round table's Compared with names Best so
    far, while the closest formulation can be another (walkthrough Line A:
    15 against 18)."""
    return f"Compared with: always Best so far, {FORMULATION_CAP} {best_no}."


def why_gain_said(line):
    """A withheld line (WHY_CHECKING, why_guard_line) in the fold's own
    column name: the gain, not the BO term."""
    # Charts proto: the withheld lines say "Expected gain" themselves now.
    return line


def repeat_after_generate(no):
    """Make a round with no Round open, reached by the Repeat link: the
    Repeat waits for Generate, so the app's suggestions still come first."""
    return f"Click Generate first. The Repeat of {FORMULATION_CAP} {no} then opens below."


def repeat_after_generate_help(no):
    """repeat_after_generate's full sentence, its tooltip."""
    return (f"Click Generate first: the Repeat of {FORMULATION_CAP} {no} then opens, filled in, "
            f"under {ADD_OWN_EXPANDER}.")


# --- charts proto ---
# 2026-10-01 charts prototype (owner: "are the plots we're showing now
# best?"): the Explored and unexplored fold holds four charts — What
# you've tried, How the app chooses, the triangle, the map. The owner's
# concision rule: a caption is about 8 words at most, a title or label 5;
# the explanation sits in the "?" (help) or the chart's own tooltip.
TRIED_TITLE = "What you've tried"
TRIED_BEST = BEST_SO_FAR_COLUMN
TRIED_WAITING = f"Waiting or {NOT_SCORED}"
TRIED_SUGGESTED = f"This {ROUND_CAP}'s suggestion"
TRIED_SCORED = f"Scored {FORMULATION}"


def tried_caption(settings_only=False):
    """Under the parallel-coordinates chart (the owner's 8-word rule)."""
    return f"One line per {FORMULATION_CAP}; colour: {OVERALL_SCORE_COLUMN}."


def tip_tried(settings_only=False):
    """The caption's "?": how to read a line."""
    what = "process setting" if settings_only else "ingredient and process setting"
    return (f"Each line is one {FORMULATION_CAP}. It crosses one axis per {what} at its "
            f"amount, Lowest at the bottom and Highest at the top, and ends at its "
            f"{OVERALL_SCORE_COLUMN} on the right (0 at the bottom), coloured by that score. "
            f"Thick: Best so far. Orange: this {ROUND_CAP}'s suggestions, not yet scored, "
            f"ending before the score axis. Grey dashed: waiting for results or {NOT_SCORED}. "
            "Lines that bunch together on an axis show the amounts tried most; empty stretches "
            "show what has not been tried. The Fill to total row is not drawn: it follows the "
            "others. BO term: parallel coordinates of the design points.")


def tried_amounts_title(values=False):
    """The line tooltip's make-up field."""
    return "Values" if values else "Amounts"


TRIED_NOTHING = "Nothing recorded yet."

CHOOSES_TITLE = "How the app chooses"
CHOOSES_CAPTION = "Suggestions go where expected gain peaks."
CHOOSES_PREDICTED = f"Predicted {OVERALL_SCORE_COLUMN}"
CHOOSES_BAND = "95 % interval"
CHOOSES_GAIN = "Expected gain (points)"
CHOOSES_TESTED = "Tested near this line"
CHOOSES_SUGGESTION = "First suggestion"
CHOOSES_COLD = f"Appears after 5 scored {FORMULATION}s."


def tip_chooses(best_no, settings_only=False, line=False, end=None):
    """The caption's "?": the owner's own reading, and what the line is."""
    what = "process setting" if settings_only else "ingredient or process setting"
    if line:
        along = (f"The line runs straight from Best so far ({FORMULATION_CAP} {best_no}) to "
                 f"{end}, every row moving together, and a little past each end where your "
                 "limits allow. ")
    else:
        along = (f"The line moves one {what} from its Lowest to its Highest, starting from "
                 f"Best so far ({FORMULATION_CAP} {best_no}); the other rows stay at Best so "
                 "far's amounts. ")
    return ("The app suggests where the expected gain (bottom) is highest — either a high "
            "prediction or high uncertainty (top). " + along +
            "Top: the suggestion model's predicted Overall score with its own 95 % band, grey "
            f"where nothing was tested near, and the {FORMULATION}s tested near the line. "
            "Bottom: the expected gain over Best so far at each point, in points. BO term: "
            "posterior mean and 95 % credible band (top); expected gain, the "
            "acquisition value (bottom).")


CHOOSES_WITHHELD = "Expected gain not shown."


def tip_chooses_withheld(reason):
    """The withheld line's "?": why."""
    return f"Expected gain is not shown: {reason}."


CHOOSES_REASON_NOISE = "the model is reading chance variation as effects"
CHOOSES_REASON_UCB = "your model settings choose suggestions by an upper bound, not a gain"
CHOOSES_REASON_COLD = (f"this {ROUND_CAP}'s suggestions were not chosen by the model (fewer "
                       f"than 5 {FORMULATION}s had all their results)")
CHOOSES_OFF_LINE = "First suggestion is off this line."


def tip_chooses_off_line(no):
    """The off-line caption's "?"."""
    return (f"This {ROUND_CAP}'s first suggestion ({FORMULATION_CAP} {no}) also changes other "
            "rows: it is not near this line. Choose another row to find it.")


PAIR_FOLD = "Two measurements"


# Demo build: the Explored fold holds the triangle alone (its colours are
# the suggestion model's predictions; the tooltip says which model).
BASIS_TRIANGLE = "Basis: the suggestion model."


def basis_charts(values=False, total=True):
    """The fold's first line with the four charts (was basis_explored)."""
    at = "at the values run" if values else "at the Formulation total" if total else "as made"
    return f"Lines: {at}. Others: the suggestion model."


# Charts proto follow-up: the straight line from Best so far to Suggestion 1
# (else the predicted best), one name for the gain, the band's width said.
CHOOSES_MODE = "Draw along"
CHOOSES_SUGGESTION_ONE = "Suggestion 1"
CHOOSES_PREDICTED_BEST = "Predicted best"
CHOOSES_AMOUNTS_HERE = "Amounts here"


def chooses_mode_option(line, has_pick=True, settings_only=False):
    """The Draw along box's two entries."""
    if line:
        end = CHOOSES_SUGGESTION_ONE if has_pick else CHOOSES_PREDICTED_BEST
        return f"Best so far → {end}"
    return "Along one process setting" if settings_only else "Along one ingredient"


def chooses_line_axis(end):
    """The straight line's x axis: 'From Best so far → Suggestion 1'."""
    return f"From Best so far → {end}"


CHOOSES_BAND_LEGEND = "Model's own band: narrower than Try one"
TIP_CHOOSES_BAND = (
    "The band is the suggestion model's own 95 % uncertainty about its mean: no Repeat noise "
    "and no widening from the Prediction check. It is therefore narrower than the 95 % "
    "interval under Try one, which is for a new measured result. BO term: latent posterior "
    "95 % credible band.")
# One name for the gain on screen (Expected gain); the BO term in tooltips.
GAIN_TOOLTIP_TITLE = "Expected gain (points) — expected improvement (acquisition value)"


# ------------------------------------------------------------------ #
#  Concision (owner, 2026-10-01: "too wordy … overwhelming to read").
#  On screen, each status line is its bold word and one short fact; the
#  full sentence (pred_*, progress_*, next_plain_*) opens its tooltip,
#  ahead of the exact rule. Numbers over prose; no clause, no aside.
# ------------------------------------------------------------------ #

def insights_header_short(n, last_round, waiting=0, names_text="", numbers=None):
    """'20 scored formulations, up to Round 5. Formulation 19 is waiting
    for Cook loss.' The full line (insights_header_line) is its tooltip."""
    verb = "is" if waiting == 1 else "are"
    named = (formulations_span(numbers)
             if numbers and len(numbers) == waiting <= HEADER_NAMES_WAITING else None)
    if n == 0 and waiting:
        who = named or f"{waiting} {FORMULATION}{'s' if waiting != 1 else ''}"
        return f"{who} {verb} {waiting_for(names_text)}."
    tail = "" if last_round is None else f", up to {ROUND_CAP} {last_round}"
    who = named or f"{waiting} more"
    wait = f" {who} {verb} {waiting_for(names_text)}." if waiting else ""
    return f"{n} scored {FORMULATION}{'s' if n != 1 else ''}{tail}.{wait}"


# Predictions.
def short_in_range(inside, total):
    """'33 of 35 results in range.' (Reliable)."""
    return f"{inside} of {total} results in range."


def short_out_of_range(outside, total, allowed):
    """'8 of 36 results out of range; up to 4 is normal.' (Off, caution)."""
    normal = "none is normal" if not allowed else f"up to {allowed} is normal"
    return f"{outside} of {total} results out of range; {normal}."


def short_gate(n, needed=5):
    """'Needs 5 scored formulations; 3 so far.'"""
    return f"Needs {needed} scored {FORMULATION}s; {n or 'none'} so far."


def short_too_few(k, needed=10):
    """'2 of 10 formulations checked.'"""
    return f"{k} of {needed} {FORMULATION}s checked."


SHORT_WIDE = "Still very uncertain."
SHORT_NO_SKILL = "No better than averaging past scores."
SHORT_UNCLEAR = "Not clearly better than averaging past scores."
SHORT_OFF_REPEAT = "A Repeat came out far from its original."
# CHECK_SCATTER, shortened; the "check" advice stays on screen.
SHORT_CHECK_SCATTER = "New panel or method? Check its consistency."


# Progress.
def short_best(score_text, ceiling, no, tie_no=None):
    """'Best so far: 90.45 of 100, Formulation 15.'; an equal-score tie
    names both."""
    who = (f"{FORMULATION_CAP}s {no} and {tie_no}" if tie_no is not None
           else f"{FORMULATION_CAP} {no}")
    return f"Best so far: {_out_of(score_text, ceiling)}, {who}."


def short_lack(rounds, names):
    """'Rounds 1–2 lack Cook loss.' (Too early, every earlier Round scored
    without a measurement)."""
    return f"{rounds_span(rounds)} {'lacks' if len(rounds) == 1 else 'lack'} {names}."


def short_flat(k):
    return f"No new best in {k} {ROUND_CAP}s."


def short_rise(rise_text, need_text=None, rising=False):
    """'Up 2.5, needs 3.2.' (Within noise) or 'Up 3.5, beats 3.2.'
    (Rising): the rise and what it is judged against, Improvement.threshold
    (fix A follow-up). The full sentence is in the tooltip."""
    if need_text is None:
        return f"Up {points_text(rise_text)}."
    if rising:
        return f"Up {rise_text}, beats {need_text}."
    return f"Up {rise_text}, needs {need_text}."


def short_rise_best(score_text, no, tie_no=None):
    """Best so far after a rise, without the ceiling, so the line keeps to
    10 words: 'Best so far: 89.84, Formulation 18.'"""
    who = (f"{FORMULATION_CAP}s {no} and {tie_no}" if tie_no is not None
           else f"{FORMULATION_CAP} {no}")
    return f"Best so far: {score_text}, {who}."


# Next.
def short_record(no, n):
    """'Record Round 6's 4 open formulations.'"""
    what = f"open {FORMULATION}" if n == 1 else f"{n} open {FORMULATION}s"
    return f"Record {ROUND_CAP} {no}'s {what}."


def short_tail(no=None, left=0):
    return short_record(no, left) if left else f"Make the next {ROUND_CAP}."


def short_repeat(repeat_no, no=None, left=0, next_no=None):
    """'Make the next Round with a Repeat of Formulation 24.'; with a Round
    open, 'Record Round 6's 4 open formulations. In Round 7, add a Repeat
    of Formulation 15.' (`next_no`: the Round after the open one)."""
    if left:
        which = f"{ROUND_CAP} {next_no}" if next_no is not None else f"the next {ROUND_CAP}"
        return (f"{short_record(no, left)} In {which}, add a Repeat of "
                f"{FORMULATION_CAP} {repeat_no}.")
    return f"Make the next {ROUND_CAP} with a Repeat of {FORMULATION_CAP} {repeat_no}."


def short_check_repeat(later, earlier, diff_text, lower=True, round_no=None):
    """The one "check" advice: 'Check panel, suppliers and equipment before
    Round 6. Formulation 11's Repeat scored 10.5 points lower.'"""
    which = f"{ROUND_CAP} {round_no}" if round_no is not None else f"the next {ROUND_CAP}"
    return (f"Check panel, suppliers and equipment before {which}. {FORMULATION_CAP} "
            f"{earlier}'s Repeat scored {points_text(diff_text)} {'lower' if lower else 'higher'}.")


SHORT_MISSES = "Predictions may miss by more than shown."
SHORT_NO_SKILL_NEXT = "Suggestions are a guided search, not a forecast."


def short_untested(round_no=None, open_round=True):
    """'Add the predicted best to Round 6 as your own formulation.'"""
    where = (f"{ROUND_CAP} {round_no}" if open_round and round_no is not None
             else f"the next {ROUND_CAP}")
    return f"Add the predicted best to {where} as your own {FORMULATION}."


def short_close(change_text="", settings_only=False):
    """'Close to the top. To look further, raise the Highest of Pea.'"""
    if not change_text:
        change_text = ("add a new process setting" if settings_only
                       else "add a new ingredient or process setting")
    return f"Close to the top. To look further, {change_text}."


def short_add_repeat(no):
    return f"Add a Repeat of {FORMULATION_CAP} {no}."


# Make a round (concision round).
WHY_GAIN_CHECKING_SHORT = "Expected gain shown once the Prediction check is done."
_WHY_GAIN_SHORT = {"verdict": "predictions miss too often",
                   "skill": "predictions not yet reliable",
                   "noise": "the model is fitting noise"}


def why_gain_short(reason):
    """The Why fold's withheld line on screen; the full one is its tooltip."""
    return f"Expected gain not shown: {_WHY_GAIN_SHORT.get(reason, _WHY_GAIN_SHORT['noise'])}."


def round_track_short(rounds, inside, total, rough=False, gap_text=None, below=True,
                      see_insights=False, no_skill=False):
    """Make a round's track line on screen: 'Overall score, Rounds 2–5: 13
    of 15 in range.', then each warning as a short sentence; the full line
    (ui_insights.track_text) opens its tooltip."""
    parts = []
    if inside is not None:
        # Review polish: with a warning after it, the Rounds go to the "?"
        # (track_text names them), so the line stays near 10 words.
        warned = rough or gap_text is not None or see_insights
        where = f", {rounds_span(rounds)}" if rounds and not warned else ""
        parts.append(f"{OVERALL_SCORE_COLUMN}{where}: {inside} of {total} in range.")
        if rough:
            parts.append("More misses than normal.")
        if gap_text is not None:
            parts.append(f"Scores {points_text(gap_text)} {'below' if below else 'above'} "
                         "predictions.")
        if see_insights:
            parts.append(TRACK_VALUES_MISS)
    if no_skill:
        parts.append(f"{OVERALL_SCORE_COLUMN} predictions no better than averaging.")
    return " ".join(parts)


# --- review cards ---
# Review-cards prototype (owner, 2026-10-01: "a clearer, less wordy way to
# present the review … understandable for someone that uses Bayesian
# Optimization … and clear for a nontechnical food scientist"): three cards
# (Predictions, Progress, Next), each a small label, one big status word,
# one number line of at most 6 words and a sparkline; the full sentence and
# the exact BO statement in the card's "?". One hero chart under them, two
# folds below. Warnings are never folded: they keep a line of their own.
CARD_PREDICTIONS = STATUS_PREDICTIONS
CARD_PROGRESS = STATUS_PROGRESS
CARD_NEXT = STATUS_NEXT


def cards_header(n, open_no=None, waiting=0, numbers=None, not_scored=0):
    """'Based on 24 formulations · Round 6 open · Formulation 25 waiting'."""
    parts = [f"Based on {n} {FORMULATION}{'s' if n != 1 else ''}"]
    if open_no is not None:
        parts.append(f"{ROUND_CAP} {open_no} open")
    if waiting:
        named = (formulations_span(numbers)
                 if numbers and len(numbers) == waiting <= HEADER_NAMES_WAITING else None)
        parts.append(f"{named or waiting} waiting")
    if not_scored:
        parts.append(f"{not_scored} {NOT_SCORED}")
    return " · ".join(parts)


# Predictions: the number line, then any warning on a line of its own.
def card_in_range(inside, total):
    return f"{inside} / {total} in range"


def card_out_of_range(outside, total, allowed):
    """'8 / 36 outside range (4 normal)'."""
    return f"{outside} / {total} outside range ({allowed or 'none'} normal)"


def card_gate(n, needed=5):
    return f"{n or 0} / {needed} {FORMULATION}s scored"


def card_checked(k, needed=10):
    return f"{k} / {needed} {FORMULATION}s checked"


CARD_WIDE = "Still very uncertain"
CARD_NO_SKILL = "No better than averaging"
CARD_UNCLEAR = "Not clearly better than averaging"
CARD_OFF_REPEAT = "A Repeat came out far off"
CARD_CHECK_SCATTER = "Check your panel's consistency"


def card_check_before(round_no=None):
    """The Off warning, under the big word Off: 'Check panel before Round 6'."""
    which = f"{ROUND_CAP} {round_no}" if round_no is not None else f"the next {ROUND_CAP}"
    return f"Check panel before {which}"


# Progress.
def card_best(score_text, tail=None):
    """'Best 90.45 · +0.2 (needs 3.2)'; 'Best 90.45 · Formulation 15'."""
    return f"Best {score_text}" + (f" · {tail}" if tail else "")


def card_rise(rise_text, need_text=None):
    """'+0.2 (needs 3.2)' (Within noise); '+3.5 (beat 3.2)' (Rising)."""
    return f"+{rise_text}" + (f" (needs {need_text})" if need_text is not None else "")


def card_rising(rise_text, need_text):
    return f"+{rise_text} (beat {need_text})"


def card_best_who(no, tie_no=None):
    """'Formulation 15'; 'Formulations 1 and 3'."""
    if tie_no is not None:
        return f"{FORMULATION_CAP}s {no} and {tie_no}"
    return f"{FORMULATION_CAP} {no}"


def card_flat(k):
    return f"same for {k} {ROUND_CAP}{'s' if k != 1 else ''}"


# Next: a verb phrase as the big word, one number line, then the Repeat
# request or a check, each a line of its own.
def card_record(no):
    return f"Record {ROUND_CAP} {no}"


def card_make(no=None):
    return f"Make {ROUND_CAP} {no}" if no is not None else f"Make the next {ROUND_CAP}"


def card_open(n):
    return f"{n} {FORMULATION}{'s' if n != 1 else ''} open"


CARD_NONE_OPEN = f"No {ROUND_CAP} open"
CARD_CHECK_FIRST = "Check panel first"


def card_before(no):
    return f"Before {ROUND_CAP} {no}"


def card_repeat_off(earlier, diff_text, lower=True):
    """'Formulation 11's Repeat: 10.5 lower'."""
    return (f"{FORMULATION_CAP} {earlier}'s Repeat: {diff_text} "
            f"{'lower' if lower else 'higher'}")


def card_repeat_ask(repeat_no, round_no=None):
    """'Round 7: Repeat Formulation 15'; with no Round open, 'With a Repeat
    of Formulation 24'."""
    if round_no is None:
        return f"With a Repeat of {FORMULATION_CAP} {repeat_no}"
    return f"{ROUND_CAP} {round_no}: Repeat {FORMULATION_CAP} {repeat_no}"


CARD_MISSES = "Predictions may miss more"
CARD_GUIDED = "Guided search, not a forecast"
CARD_TRY_BEST = "Try the predicted best"


def card_add_to(no=None):
    return f"Add it to {ROUND_CAP} {no}" if no is not None else f"Add it to the next {ROUND_CAP}"


CARD_CLOSE = "Widen a limit"
CARD_CLOSE_TOP = "Close to the top"


def card_close_line(change_text="", settings_only=False):
    """'Raise the Highest of Pea'; with nothing binding, what to add."""
    if not change_text:
        return "Add a process setting" if settings_only else "Add an ingredient or setting"
    return change_text[:1].upper() + change_text[1:]


# The hero chart: the BO figure along Best so far → Suggestion 1.
def hero_title(no=None, chose=True):
    """'How the app chose Round 6'; with no Round chosen, 'Where the app
    looks next'."""
    if chose and no is not None:
        return f"How the app chose {ROUND_CAP} {no}"
    return "Where the app looks next"


HERO_CAPTION = "Suggestions go where expected gain peaks."
HERO_COLD = f"Appears after 5 scored {FORMULATION}s."
# The withheld reason, at most 10 words; the full one in its "?".
HERO_WITHHELD = {
    "noise": "Expected gain hidden: the model may be reading noise.",
    "ucb": "Expected gain hidden: your settings use an upper bound.",
    "cold": f"Expected gain hidden: {ROUND_CAP} laid out, not modelled.",
    "verdict": "Expected gain hidden: predictions miss too often.",
    "skill": "Expected gain hidden: predictions not yet reliable.",
}

# The two folds.
FOLD_TRIED = "What you've tried"
FOLD_MORE = "More detail"
CARD_SPARK_IN_RANGE = "In range"
HERO_PREDICTED = "Predicted score"
HERO_GAIN = "Expected gain"


# --- review diagnosis ---
# Off with no disagreeing Repeat (owner, 2026-10-02): the yogurt pilot's
# Off said "Check panel" though nothing showed the panel inconsistent. Only
# a disagreeing Repeat may point at the panel, suppliers or equipment
# (spec 2026-10-01 first-time design; e2e fixes): Round-ahead errors cannot
# tell drift from a model meeting new ground. So the card says where the
# misses fell, a description, never a cause, and asks for a Repeat.
CARD_DIAG = {
    "new_ground": "Misses mostly on new ground",
    "one_way": "Each measurement misses one way",
    "unexplained": "Misses not explained yet",
}


def card_score_in_range(inside, total):
    """'Overall score: 9 / 9 in range': the score the suggestions use."""
    return f"{OVERALL_SCORE_COLUMN}: {inside} / {total} in range"


def card_noise_repeat(no=None):
    """The Next card's neutral data request: 'Repeat Formulation 10 to
    check noise'."""
    if no is None:
        return "Add a Repeat to check noise"
    return f"Repeat {FORMULATION_CAP} {no} to check noise"


_DIAG_PLAIN = {
    "new_ground": ("Most misses ({far} of {misses}) were on {f}s far from anything tested "
                   "before ({far_checked} of {checked} values checked were). That says where "
                   "they fell, not why."),
    "one_way": ("Most misses ({one_way} of {misses}) ran one way for their measurement: "
                "steadily {side} the predictions, not scattered. That says how they "
                "fell, not why."),
    "unexplained": "The misses follow no clear pattern yet.",
}


_SIDE_WORDS = {"above": "above", "below": "below", "mixed": "above or below"}


def card_diag(kind, side=None):
    """The Predictions card's diagnosis line, at most 6 words: one way says
    which way when every measurement's misses agree ('Results mostly above
    predictions'; "results", the app's word for measured values, so it is
    not read as the Overall score on the line under it)."""
    if kind == "one_way" and side in ("above", "below"):
        return f"Results mostly {side} predictions"
    return CARD_DIAG[kind]


def diag_plain(kind, far, misses, score=None, far_checked=0, checked=0, one_way=0,
               side=None):
    """The diagnosis in plain words, for the Predictions tooltip. `score`:
    (inside, total) of the Overall score when it is as expected, said
    because the suggestions are made on it."""
    line = _DIAG_PLAIN[kind].format(far=far, misses=misses, f=FORMULATION,
                                    far_checked=far_checked, checked=checked, one_way=one_way,
                                    side=_SIDE_WORDS.get(side, "above or below"))
    if score is not None:
        line += (f" The {OVERALL_SCORE_COLUMN}, which the suggestions use, was {score[0]} of "
                 f"{score[1]} in range.")
    return line


def diag_item(name, misses, checked, above, below, mean_text):
    """'Creaminess 4 of 9 (above, mean error +1.4)'; mixed sides '(3 above,
    1 below, mean error +0.8)'. Mean error: measured minus predicted, over
    the misses."""
    side = ("above" if not below else "below" if not above
            else f"{above} above, {below} below")
    return f"{name} {misses} of {checked} ({side}, mean error {mean_text})"


def diag_detail(items, rounds, far, misses, far_checked, checked):
    """More detail's one line under the verdict: the misses by measurement
    and how many fell on new ground, the base rate beside it."""
    return (f"Misses, {rounds_span(rounds)}: {'; '.join(items)}. {far} of {misses} were on "
            f"{FORMULATION}s far from anything tested before their {ROUND_CAP} ({far_checked} "
            f"of {checked} values were).")


def tip_diag(items, rounds, far, misses, far_checked, checked, far_pct, kind, score=None,
             score_rounds=None):
    """The expert line: the evidence, then the rule by name."""
    word = {"new_ground": "new ground", "one_way": "one way",
            "unexplained": "not explained yet"}[kind]
    out = (f"Misses by measurement, {rounds_span(rounds)} ({misses} of {checked} values; "
           f"mean error = measured − predicted over the misses): {'; '.join(items)}. On new "
           f"ground: {far} of {misses} misses, {far_checked} of {checked} values (farther than "
           f"{far_pct} % of each varying row's Lowest–Highest span, L∞, from every "
           f"{FORMULATION} recorded before the {ROUND_CAP}: the suggestion-role rule). "
           "Diagnosis rule, first that holds: new ground when more than half the misses are "
           "and new-ground values miss at least as often as the rest; one way when at least "
           "80 % of the misses sit on their measurement's usual side (measurements with 2 "
           f"misses or more); else not explained. Here: {word}. A description of where the "
           "misses fell, not a cause: Round-ahead errors cannot show drift; only a "
           "disagreeing Repeat can.")
    if score is not None:
        where = f", {rounds_span(score_rounds)}" if score_rounds else ""
        out += f" {OVERALL_SCORE_COLUMN}{where}: {score[0]} of {score[1]} in range."
    return out


def trust_misses_diag(rounds, inside, total, many=False, allowed=None):
    """trust_misses without its "intervals are too narrow" reading, which
    contradicted the wide-interval line beside it: the count alone; the
    diagnosis line follows."""
    misses = total - inside
    line = (f"Measurements, {rounds_span(rounds)}: {inside} of {total} values inside their "
            "95 % interval.")
    if allowed is not None:
        count = "1 miss" if misses == 1 else f"{misses} misses"
        line += f" {count}; 95 % intervals allow up to {allowed}."
    return f"{line} {'Many more' if many else 'More'} misses than expected."


# More detail's reading after the evidence: a description, not a cause.
DIAG_READING = {
    "new_ground": "Mostly on new ground: where they fell, not why.",
    "one_way": "Mostly one way per measurement: how they fell, not why.",
    "unexplained": "No clear pattern yet.",
}

_DIAG_WHERE = {"new_ground": ", mostly on new ground", "one_way": ", mostly one way",
               "unexplained": ""}


def next_misses_diag(rounds, kind, repeat_no=None, bias_text=""):
    """Next-step reasoning's misses row with no disagreeing Repeat: the fact,
    where the misses fell, and a Repeat to measure the noise; no check of
    the bench."""
    ask = (f"A Repeat of {FORMULATION_CAP} {repeat_no} would measure the noise."
           if repeat_no is not None else "A Repeat would measure the noise.")
    line = (f"Measured values in {_in_rounds(rounds)} fell outside their 95 % interval more "
            f"often than 95 % intervals allow{_DIAG_WHERE[kind]}, so new predictions can miss "
            f"by more than their intervals show. {ask}")
    return f"{line} {bias_text}" if bias_text else line


# The CV-widening check (ui_insights.wide_cause), review diagnosis ruling 2
# (2026-10-02): "check your panel" only with Repeat evidence. Without it the
# card asks for a Repeat (card_noise_repeat); with Repeats that vary a lot
# it says so, and the tooltip keeps the consistency advice, backed.
CARD_REPEATS_VARY = "Repeats vary a lot"


def repeats_vary_line(sd_text, df, spread_text):
    """The Repeat evidence behind CHECK_SCATTER, for the tooltip."""
    return (f"Repeats differ with an SD of {points_text(sd_text)} ({df} df), at least half "
            f"the SD of scores between {FORMULATION}s ({points_text(spread_text)}): "
            "results vary a lot from one make to the next.")


# --- review guide ---
# Review as step 4 of the loop (owner, 2026-10-02: "I'm not guided to see
# the review -- it leads me through the 1-3"): Set up -> Make a round ->
# Results -> Review -> Make a round. Results' one lit button after Save
# results, Make a round's one-line status from the cards, the welcome's
# fourth step.
WELCOME_REVIEW_STEP = "4. **Review each Round**: are predictions reliable? What next?"


def review_round_label(no):
    """Results' primary after Save results: 'Review Round 5'."""
    return f"Review {ROUND_CAP} {no}" if no is not None else "Review"


REVIEW_ROUND_HELP = (f"Opens {TAB_INSIGHTS}: can you trust the predictions, is Best so far "
                     f"still rising, and what to make in the next {ROUND_CAP}.")
OPEN_REVIEW = "Open Review"


REVIEW_STATUS_PREFIX = "Review"


def review_status_line(predictions, progress, extra=None):
    """Make a round's line, the cards' own words: 'Review: Predictions
    **Reliable** · Progress **Flat**' (owner, 2026-10-02: "Review:", not
    "Last Round:", for the verdict covers several recent Rounds); `extra`,
    the Next card's word when it is a warning ('Check panel first')."""
    line = (f"{REVIEW_STATUS_PREFIX}: {CARD_PREDICTIONS} **{predictions}** · "
            f"{CARD_PROGRESS} **{progress}**")
    return line + (f" · **{extra}**" if extra else "")


def review_status_warn(line):
    """The line in the cards' warning colour (Off, Check panel first)."""
    return f":red[{line}]"


REVIEW_STATUS_CHECKING = f"{REVIEW_STATUS_PREFIX}: {CHECKING_PREDICTIONS}"
REVIEW_STATUS_HELP = (f"The {TAB_INSIGHTS} cards in brief. Predictions: can you trust them? "
                      "Progress: is Best so far still rising?")


# --- review plots ---
# The plot review (2026-10-02, owner: "I want to make sure the plots are all
# interpretable"): every chart carries its own title (a plain question or
# statement, at most 6 words), a legend for each colour, shape and band, a
# "how to read" caption of at most 8 words (the full reading in its "?"),
# and a "BO term" line in its tooltip naming the exact quantity drawn.
PLOT_SPARK_IN_RANGE = f"In range, by {ROUND_CAP}"
PLOT_SPARK_BEST = f"Best so far, by {ROUND_CAP}"
PLOT_SPARK_TARGET = "95 % aim"
PLOT_SPARK_WAIT = f"Trend appears after 2 {ROUND_CAP}s."
PLOT_TRIED = "Which amounts have you tried?"
PLOT_TRIED_SETTINGS = "Which settings have you tried?"
PLOT_PROGRESS = "Is the score still rising?"
PLOT_GAIN = "Is there much left to gain?"
PLOT_MVP = "Were the predictions right?"
PLOT_CHOOSES = "Where does the app look next?"
PLOT_TRIANGLE = "Which mix of three scores best?"
PLOT_MAP = f"Which {FORMULATION}s are worth trying?"
PLOT_CHANGE = "Which change matters most?"
PLOT_PATH = "What if you change just one?"
PLOT_PAIR = "Can you have both?"
# Note added to a value axis drawn over a small part of its Range.
PLOT_ZOOMED = "zoomed"
# Axis titles, with units.
PLOT_SCORE_AXIS = "Predicted score (of 100)"
PLOT_GAIN_AXIS = "Expected gain (points)"
PLOT_MEASURED_SCORE = "Overall score (of 100)"
# Legend entries.
PLOT_BAND = "Likely range of the mean, 95 %"
PLOT_INTERVAL = "Likely range of a result, 95 %"
PLOT_MEAN = "Predicted (model mean)"
PLOT_UNTESTED = "Not tested near here"
PLOT_TESTED = f"Tested {FORMULATION}"
PLOT_TESTED_NEAR = "Tested near this line"
PLOT_ALLOWED = f"Allowed {FORMULATION} (sample)"
PLOT_SUGGESTION = f"This {ROUND_CAP}'s suggestion"
PLOT_START = "Start"
PLOT_TARGET = "Target"
PLOT_SCORE_EACH = f"Each {FORMULATION}'s score"
PLOT_INSIDE = "Inside its 95 % range"
PLOT_OUTSIDE = "Outside its 95 % range"
PLOT_LATEST = f"Latest {ROUND_CAP}"
PLOT_EARLIER = f"Earlier {ROUND_CAP}s"
PLOT_EXACT = "Measured = predicted"
PLOT_NEAR = "Near tested amounts"
PLOT_WHOLE = "Whole allowed stretch"
# The tooltip's expert row title, and its lines.
PLOT_BO = "BO term"
PLOT_BO_MODEL_FREE = "model-free: the measured values"
PLOT_BO_COVERAGE = ("share of Round-ahead values inside their 95 % posterior predictive "
                    "intervals")
PLOT_BO_BEST = "incumbent: highest measured Overall score so far"


def plot_bo_curve(mean, sd, gain=None):
    """The hero's and How the app chooses' expert line at one point."""
    text = f"GP posterior mean {mean:.2f}, sd {sd:.2f} (latent, no noise)"
    return text if gain is None else f"{text}; expected gain = qLogNEI value {gain:.3f}"


def plot_bo_interval(mean, low, high):
    return (f"posterior mean {mean:.2f}; 95 % posterior predictive interval "
            f"{low:.2f} to {high:.2f}, calibrated")


def plot_bo_map(mean, sd):
    return f"GP posterior mean {mean:.2f}, latent sd {sd:.2f}"


PLOT_BO_GAIN = "qLogNEI value of the Round's first suggestion (exp of the log value)"
PLOT_BO_HIGHEST = "maximum of the GP posterior mean over the allowed region"
PLOT_BO_CHANGE = "range of the GP posterior mean along a one-factor trace"
PLOT_BO_TRIANGLE = "GP posterior mean on a simplex slice, other inputs fixed"
PLOT_BO_ROUND_AHEAD = "rolling-origin posterior predictive mean and 95 % interval"
# Captions under the charts: at most 8 words each.
PLOT_CAPTION_PROGRESS = "Flat line: no higher score yet."
PLOT_CAPTION_GAIN = "Bars near zero: little left to gain."
PLOT_CAPTION_MVP = "Near the diagonal: predicted well."
PLOT_CAPTION_CHANGE = "Longer bar: bigger predicted effect."
PLOT_CAPTION_PATH = "Wider band: less certain."
PLOT_CAPTION_MAP = "Up: less known. Right: better predicted."
PLOT_CAPTION_PAIR = "Dots near both dashed lines hit both."
PLOT_CAPTION_PAIR_NONE = "Axis titles say which way is better."


def plot_caption_triangle(layer=None):
    if layer == LAYER_SD:
        return "Darker: the model is less sure."
    if layer == LAYER_GAIN:
        return "Darker: more expected gain."
    return "Darker: higher predicted score."


# Withheld, at most 10 words.
PLOT_GAIN_WITHHELD = "Expected gain chart hidden: predictions not yet reliable."
PLOT_TOO_FEW_LINES = f"Lines appear once a {FORMULATION} is recorded."


def at_limit_fact(name, end):
    """Try one's answer line when the choice is already at a limit (demo
    build, owner 2026-10-02): 'Tap water: already at its Highest', so the
    largest local effect is not read as advice; agrees with
    path_head_at_limit, the chart's heading."""
    return f"{name}: already at its {end}"


def path_head_at_limit(name, end, amount_text):
    """Try one's first caption when the start already sits at a limit (the
    yogurt pilot: water at its 42 g Highest read "More Tap water")."""
    other = "less" if end == "Highest" else "more"
    return f"{name} is at its {end} ({amount_text}); the line shows {other}"


def plot_at_end(amount_text, n, total):
    """What you've tried: an axis end where most lines sit."""
    return f"{amount_text} ({n} of {total})"


# --- review polish ---
# Review polish (2026-10-02, demo prep for food scientists and BO /
# statistics readers): the header says what the Review is about; an Off
# card names which numbers missed and which held; each card's "?" opens
# with its plain sentence, then one compact evidence line (n, method,
# rule); a hidden expected gain reads as a choice with its next step; the
# method is one click from the Review. Wording and layout only: no rule
# and no suggestion changes.
def review_header(n, last_round=None, open_no=None, waiting=0, numbers=None, not_scored=0):
    """'Review after Round 5 · 20 formulations · Round 6 open · Formulation
    19 waiting'; the scored count is the header's one count."""
    lead = f"Review after {ROUND_CAP} {last_round}" if last_round is not None else "Review"
    parts = [lead, f"{n} {FORMULATION}{'s' if n != 1 else ''}"]
    if open_no is not None:
        parts.append(f"{ROUND_CAP} {open_no} open")
    if waiting:
        named = (formulations_span(numbers)
                 if numbers and len(numbers) == waiting <= HEADER_NAMES_WAITING else None)
        parts.append(f"{named or waiting} waiting")
    if not_scored:
        parts.append(f"{not_scored} {NOT_SCORED}")
    return " · ".join(parts)


REVIEW_ABOUT = ("Three questions after each Round: can you trust the predictions, is the best "
                "still rising, and what to make next.")


# The Off card: the measurements that missed, named, beside the Overall
# score the suggestions use. Same counts as before, scoped.
def card_values_missed(outside, total, allowed):
    """'Measurements: 9 / 36 missed (4 normal)'."""
    return f"Measurements: {outside} / {total} missed ({allowed or 'none'} normal)"


# The "?" of each card: the plain sentence, then this line, then the exact rule.
def evidence(parts):
    """'Evidence: n = 36 values, Rounds 4–6 · …'."""
    return "Evidence: " + " · ".join(p.rstrip(".") for p in parts if p) + "."


def ev_values(total, rounds):
    return f"n = {total} measured values, {rounds_span(rounds)}"


EV_ROUND_AHEAD = ("each predicted from earlier Rounds only (Round-ahead 95 % posterior "
                  "predictive coverage)")


def ev_misses(misses, allowed):
    return f"{misses} outside, up to {allowed} expected (binomial, 5 % each)"


def ev_score(inside, total, skill_text=None, n=None, rounds=None):
    """'Overall score, Rounds 3–6: 10 of 12 in range, skill 0.69 against the
    running mean (n = 12)': its own Rounds, as More detail names them."""
    where = f", {rounds_span(rounds)}" if rounds else ""
    out = f"{OVERALL_SCORE_COLUMN}{where}: {inside} of {total} in range"
    if skill_text is not None:
        out += f", skill {skill_text} against the running mean (n = {n})"
    return out


def ev_checks(k, needed):
    return f"{k} of {needed} Overall-score checks needed so far"


def ev_scored(n, needed):
    return f"{n} of {needed} {FORMULATION}s scored; checks start at {needed}"


EV_MODEL_FREE = f"model-free: measured {OVERALL_SCORE_COLUMN}s only"


def ev_rise(rise_text, need_text):
    return f"rise {rise_text} against {need_text} needed"


def ev_noise(sd_text, assumed, df=0):
    """'noise SD 1 point, assumed (0 Repeat df)'; measured, '(4 Repeat df)'."""
    how = "assumed" if assumed else "from Repeats"
    return f"noise SD {points_text(sd_text)}, {how} ({df} Repeat df)"


def ev_flat(k):
    return f"no higher score in {k} {ROUND_CAP}{'s' if k != 1 else ''}"


def ev_rule(name):
    """'rule: Misses (first Next-step rule that holds)'."""
    return f"rule: {name} (first Next-step rule that holds)"


_ACQ_SHORT = {"qlognei": "qLogNEI", "qlogei": "qLogEI", "qucb": "qUCB"}


def ev_suggestions(acquisition="qlognei"):
    """'suggestions: Gaussian process on the Overall score, qLogNEI'."""
    name = _ACQ_SHORT.get(acquisition, acquisition)
    return f"suggestions: Gaussian process on the {OVERALL_SCORE_COLUMN}, {name}"


# Review four: the method link sits under the cards now.
EV_METHOD = "Kernel, priors and every check: How the app knows, under the cards."

# The one-click way to the method (the Method and definitions fold).
METHOD_LINK = "How the app knows"
METHOD_LINK_HELP = ("Opens Method and definitions: the Gaussian-process model read from this "
                    "project, the acquisition (qLogNEI by default), the Round-ahead checks, "
                    "the noise threshold and the known limits.")

# The hero: what the x axis is, and a hidden gain as a choice with a next step.
def hero_axis(values=False):
    """The hero's x axis: every row moved together in a straight line."""
    return f"Every {'setting' if values else 'amount'} moved in a straight line"


HERO_HIDDEN_ICON = ":material/visibility_off:"
# At most 10 words each; the full reason in its "?".
HERO_HIDDEN = {
    "noise": "Expected gain hidden: may be noise. See Next.",
    "ucb": "Expected gain not drawn: your settings use an upper bound.",
    "cold": f"Expected gain not drawn: {ROUND_CAP} laid out, not modelled.",
    "verdict": "Expected gain hidden while predictions miss. See Next.",
    "skill": "Expected gain hidden until predictions beat averaging. See Next.",
}


TIP_HERO_TOP = ("The line runs straight from Best so far, every row moving together. Line: the "
                "suggestion model's predicted Overall score; band: its own 95 % range; grey: "
                f"nothing tested near; rings: {FORMULATION}s tested near the line. BO term: "
                "posterior mean and 95 % credible band.")


def tip_hero_hidden(reason):
    """The hidden line's "?": why, and that suggestions are still made."""
    return (f"Expected gain is not drawn: {reason}. Suggestions are still made the same "
            "way; the Next card says what would bring the gain back.")


# Make a round: the three lines the review cards left over the limit.
WHY_ROLE_KEY_SHORT = (f"{SUGGESTION_CLOSE.capitalize()}: near. "
                      f"{SUGGESTION_DIFFERENT.capitalize()}: far. "
                      f"{SUGGESTION_BALANCED.capitalize()}: between.")


def next_misses_short(repeat_no=None):
    """Make a round's misses line above Generate, at most 10 words."""
    if repeat_no is None:
        return "Predictions miss more than expected: add a Repeat."
    return f"Predictions miss more than expected: repeat {FORMULATION_CAP} {repeat_no}."


TRACK_VALUES_MISS = "Measurements miss more."


SETUP_BUTTON_SHORT = "Change on Set up"

# --- review profiler ---
# The prediction profiler (prototype, 2026-10-02): one small panel per
# ingredient or process setting (columns) and per result (rows), each line
# moving that one row from its Lowest to its Highest through the reference
# formulation. It replaces Try one's path and Compare all.
PLOT_PROFILER = "What drives each result?"
PROFILER_CAPTION = "Steeper line: bigger effect. Band: uncertainty."
PROFILER_THROUGH = "Lines drawn through"
PROFILER_REF_BEST = BEST_SO_FAR_COLUMN
PROFILER_REF_PICK = "Suggestion 1"
PROFILER_TESTED = "Tested near here"
PROFILER_GAIN_ROW = "Expected gain"
PROFILER_GAIN_AXIS = "Expected gain (points)"
PROFILER_SCORE_ROW = "Overall score (of 100)"
PROFILER_STOPPED = "limit"


def profiler_more(n):
    """The multiselect for the panels left out, at most 5 words."""
    return f"+{n} more"


def profiler_more_help(kind_settings=False, rows=False):
    if rows:
        return ("The rows shown are the measurements with the largest Share of score; "
                "pick more to add their rows.")
    what = "process settings" if kind_settings else "ingredients or process settings"
    return (f"The columns shown are the {what} that move the predicted Overall score "
            "most; pick more to add their columns.")


def profiler_fill_note(fill):
    """Under the chart, at most 8 words: the Fill to total row has no column."""
    return f"{fill} fills to the total: no column."


def profiler_at_end(end):
    """A panel's note when the reference sits at a limit of its row."""
    return f"at its {end}"


def profiler_row_target(target_text):
    return f"target {target_text}"


PROFILER_ROW_LOWER = "lower is better"
PROFILER_ROW_HIGHER = "higher is better"


def profiler_help(ref, ref_no, fill=None, fixed=(), gain=True, settings_only=False):
    """The caption's "?": how to read every panel."""
    what = "process setting" if settings_only else "ingredient or process setting"
    lines = [f"Each column moves one {what} from its Lowest to its Highest, starting from "
             f"{ref} ({FORMULATION_CAP} {ref_no}); everything else stays as it is there.",
             "Each row is one result: the blue line is the predicted value, the light band "
             "its 95 % likely range of a result. A steep line means that row matters; a flat "
             "one, that it hardly does here.",
             f"The vertical line marks {ref}'s own amount; a dashed line, the measurement's "
             "target. Grey: nothing tested near here, so the line is a guess there. Grey "
             f"rings: {FORMULATION}s tested near the line, at their measured Overall score."]
    if fill:
        lines.append(f"{fill} fills to the Formulation total, so it moves with each column "
                     "and has none of its own.")
    if fixed:
        lines.append("Fixed rows have no column: " + ", ".join(fixed) + ".")
    if gain:
        lines.append("Bottom row: the expected gain along each line, the quantity the app "
                     "chooses suggestions by (as in the chart at the top).")
    return " ".join(lines)


def profiler_bo(mean, low, high):
    """The tooltip's expert line at one point of a panel."""
    return (f"GP posterior mean {mean:.2f}, 95 % posterior predictive interval "
            f"{low:.2f} to {high:.2f} (calibrated); one-factor slice")


def profiler_bo_gain(gain):
    return f"qLogNEI acquisition value {gain:.3f} along a one-factor slice"


# --- review four ---
# Review four (owner, 2026-10-02: "clean up the Review tab to cleanly (and
# visually) answer these 4 questions"): under the three cards, four
# question sections, always open, each a small heading, one answer line of
# at most 10 words opening with its card's bold status word, one chart and
# its caption of at most 8 words; the evidence in the answer line's "?".
Q1_TITLE = "Can we trust the predictions?"
Q2_TITLE = "Are we improving?"
Q3_TITLE = "What drives each result?"
Q4_TITLE = "What have we tried?"
SEE_CHART = "See chart ↓"
SEE_CHART_HELP = "Scrolls to the chart that answers this card's question."
Q_WAIT = f"Appears after 5 scored {FORMULATION}s."
Q3_CAPTION_PICK = "Suggestion 1 sits where expected gain peaks."
PROFILER_REF_HELP = ("Draw every line through Suggestion 1 (the open Round's first suggestion) "
                     "or through Best so far.")


def answer_line(word, text):
    """A section's answer: '**Rising.** Best 98.33 · up 3.3 points.'"""
    text = text.strip()
    if text and text[-1] not in ".!?":
        text += "."
    return f"**{word}.** {text}" if text else f"**{word}.**"


def q3_answer(name):
    """Q3's answer after the Next card's word: what moves the score most."""
    return f"{name} moves the score most"


Q3_FLAT = "No single change clearly leads"


def q4_word(n):
    """Q4's status word: '30 tested'."""
    return f"{n} tested"


def q4_gap(stretch):
    """'Tap water below 31 g untested'."""
    return f"{stretch} untested"


Q4_NO_GAP = "No large untested stretch"


def gap_span(name, low_text, high_text):
    """'Salt 0.6–1.0 g': an interior gap, compact."""
    return f"{name} {low_text}–{high_text}"

# Ruling 1 (owner, 2026-10-02): the Predictions word reads the Overall
# score's Round-ahead record (the suggestions are made from its model); the
# measurements' misses are a secondary, non-red line.
CARD_MEASURES_MISS = "Measurements miss more than expected"


def card_score_missed(outside, total):
    """'Overall score: 4 / 16 missed' (Off, More misses than expected)."""
    return f"{OVERALL_SCORE_COLUMN}: {outside} / {total} missed"


def on_score(sentence):
    """A Predictions sentence's counts are the Overall score's: 'results'
    becomes 'Overall scores' the first time it is said."""
    return sentence.replace(" results ", f" {OVERALL_SCORE_COLUMN}s ", 1)


def measures_sentence(inside, total, allowed, miss=False):
    """The full sentence's second part: the measurements' own record."""
    if miss:
        return (f" Measurements miss more than expected: {inside} of {total} inside their "
                f"range, up to {allowed} misses normal.")
    return f" Measurements: {inside} of {total} inside their range."


def ev_predictions(inside, total, skill_text=None, n=None, m_inside=None, m_total=None):
    """'Overall score 10/12 in range, skill 0.69 (n = 12); measurements 27/36'."""
    out = f"{OVERALL_SCORE_COLUMN} {inside}/{total} in range"
    if skill_text is not None:
        out += f", skill {skill_text} (n = {n})"
    if m_total:
        out += f"; measurements {m_inside}/{m_total}"
    return out


# Ruling 2: Expected gain is never hidden. Where the old rules withheld it,
# it is drawn in grey with one line saying why; suggestions are made the
# same way regardless. At most 10 words each; the full reason in its "?".
# Review four polish: greyed by the Predictions card's own word (ruling 1,
# the Overall score's record), never the measurements' verdict, so the line
# never contradicts the card: "not_yet" and "off" are its two words short of
# Reliable; "noise" the noise guard.
GAIN_GREYED = {
    "noise": "Expected gain greyed: may be noise. See Next.",
    "not_yet": "Expected gain greyed: predictions not yet reliable.",
    "off": "Expected gain greyed: predictions are Off.",
}
GAIN_REASON_NOT_YET = "the Predictions card does not read Reliable yet"
GAIN_REASON_OFF = ("the Predictions card reads Off: the Overall score misses its 95 % range "
                   "more often than expected")


def tip_gain_greyed(reason):
    """The greyed gain's "?": why, and that suggestions are still made."""
    return (f"Expected gain is drawn in grey: {reason}. Suggestions are made the same way "
            "regardless; the Next card says what would make the gain count again.")


def why_gain_greyed(reason):
    """Make a round's Why these line when the gain is greyed: the profiler's
    own line (GAIN_GREYED), so the two never differ."""
    return GAIN_GREYED.get(reason, GAIN_GREYED["noise"])


def why_greyed_line(reason, detail=None):
    """Why these' greyed column's "?" for the Predictions card's word;
    `detail`: the skill gate's reason in Progress's words, when it holds."""
    why = GAIN_REASON_OFF if reason == "off" else GAIN_REASON_NOT_YET
    return f"Expected gain is greyed: {why}" + (f" ({detail})." if detail else ".")


def ev_profiler(n, ref, ref_no, acquisition="qlognei"):
    """Q3's evidence: the models, the slice and the acquisition."""
    name = _ACQ_SHORT.get(acquisition, acquisition)
    return (f"n = {n} {FORMULATION}s with all results; one Gaussian process per result "
            f"(posterior mean, 95 % predictive interval); one-factor slices through {ref} "
            f"({FORMULATION_CAP} {ref_no}); Expected gain is the suggestion model's {name} "
            "value along each slice")


def ev_tried(n):
    """Q4's evidence: model-free."""
    return (f"model-free: {n} {FORMULATION}s with all results; untested = no {FORMULATION} "
            "within 15 % of a row's span")


def why_gain_greyed_said(line):
    """Ruling 2: a withheld line (why_gain_said) as the greyed column's "?":
    shown greyed, never hidden, and the suggestions made the same way."""
    line = line.replace("Expected gain is not shown", "Expected gain is greyed", 1)
    return f"{line} Suggestions are made the same way regardless."


# Review four polish: Q4 coloured by role, not by score — the owner found
# the score colours "all look very similar". The ink line is Best so far,
# mid-grey the next best few, light grey every other tested one.
TRIED_NEXT_BEST = "Next best"


def tried_caption_roles():
    """Q4's caption: what the empty space means."""
    return f"One line per {FORMULATION_CAP}; gaps are untested."


def tip_tried_roles(settings_only=False):
    """Q4's caption "?": how to read a line, by its colour."""
    what = "process setting" if settings_only else "ingredient and process setting"
    return (f"Each line is one {FORMULATION_CAP}. It crosses one axis per {what} at its "
            f"amount, Lowest at the bottom and Highest at the top, and ends at its "
            f"{OVERALL_SCORE_COLUMN} on the right (0 at the bottom). Thick ink: Best so far. "
            f"Mid-grey: the next best few. Light grey: every other tested {FORMULATION}. "
            f"Orange: this {ROUND_CAP}'s suggestions, not yet scored, ending before the score "
            f"axis. Grey dashed: waiting for results or {NOT_SCORED}. Empty stretches of an "
            "axis show amounts not tried yet. The Fill to total row is not drawn: it follows "
            "the others. BO term: parallel coordinates of the design points.")


# Q2's x axis: Formulations in the order made, each Round's start a light rule.
Q2_X_TITLE = f"{FORMULATION_CAP} (in order made)"


def round_tick(n):
    """A Round's label above Q2's chart: 'R3'."""
    return f"R{n}"


# The profiler's "+N more" pickers: a label and a placeholder each, at
# most 5 and 4 words (Streamlit's own placeholder read "Choose options").
def profiler_more_label(kind):
    """'More ingredients' / 'More settings' / 'More ingredients or settings'
    / 'More measurements'."""
    return {"ingredients": "More ingredients", "settings": "More settings",
            "both": "More ingredients or settings", "rows": "More measurements"}[kind]


def profiler_more_placeholder(kind):
    return {"ingredients": "Add an ingredient", "settings": "Add a setting",
            "both": "Add a column", "rows": "Add a measurement"}[kind]



# --- review story ---
# Review story (prototype, 2026-10-02): the Review read as a story, in the
# order a food scientist asks — Are we improving? Can we trust the
# predictions? What drives each result? What have we tried? Every number is
# read from the project; nothing compares with another method or counts
# Rounds saved. Status lines at most 10 words, captions 8, labels 5.

def card_progress(start_text, best_text, tail=None):
    """The Progress card's line, start to best: '87.8 → 90.5 · same for 2
    Rounds' (the sparkline under it is titled Best so far, and the card
    line keeps to 6 words); 'Best 96.0 · …' when there is nothing to go
    from."""
    head = f"Best {best_text}" if start_text is None else f"{start_text} → {best_text}"
    return head + (f" · {tail}" if tail else "")


def card_checked_so_far(inside, total, needed):
    """Not yet, too few checked: '7 / 7 in range · needs 10' (it read
    '7 / 10 formulations checked', which hid that all 7 landed)."""
    return f"{inside} / {total} in range · needs {needed}"


# Are we improving? — (a) the best so far, each Round's average, the start
# and the best named; (b) closer to target, one row per measurement.
STORY_START = "Start"
STORY_BEST = "Best"


def story_point(label, score_text, round_no):
    """'Start: 87.8 (Round 1)'; 'Best: 90.5 (Round 4)'."""
    return f"{label}: {score_text} ({ROUND_CAP} {round_no})"


STORY_ROUND_AVERAGE = f"{ROUND_CAP} average"
STORY_EACH = f"Each {FORMULATION}"
STORY_IMPROVING_CAPTION = "Grey dots: each measured score."


def story_improving_help(first_avg, last_avg, first_round, last_round):
    """Chart (a)'s "?": how to read it, and the Round averages."""
    avg = ""
    if first_avg is not None and last_avg is not None and first_round != last_round:
        avg = (f" {ROUND_CAP} average: {first_avg} ({ROUND_CAP} {first_round}) → {last_avg} "
               f"({ROUND_CAP} {last_round}).")
    return (f"Grey dots: each {FORMULATION}'s measured {OVERALL_SCORE_COLUMN}. Ink step: "
            f"{BEST_SO_FAR_COLUMN}. Short grey bars: each {ROUND_CAP}'s average, so one lucky "
            f"{FORMULATION} does not read as progress.{avg} Model-free: measured values only.")


def story_target_start(round_no):
    """Chart (b)'s hollow dot: 'Round 1's best'."""
    return f"{ROUND_CAP} {round_no}'s best"


STORY_TARGET_BEST = BEST_SO_FAR_COLUMN
STORY_TARGET = "Target"
STORY_HIGHER = "Higher is better →"
STORY_LOWER = "← Lower is better"
STORY_TARGET_TITLE = "Closer to target"
STORY_TARGET_CAPTION = "Hollow: start. Ink: best so far."


def story_target_row(name, obj_goal, target=None):
    """A row's label: 'Firmness · target 6'; 'Cook loss · lower'."""
    if obj_goal == "target" and target is not None:
        return f"{name} · target {float(target):g}"
    return f"{name} · {'lower' if obj_goal == 'min' else 'higher'}"


def story_target_help(start_no, best_no, start_round):
    """Chart (b)'s "?": what each mark is."""
    return (f"One row per measurement, across its own Range. Hollow grey: {FORMULATION_CAP} "
            f"{start_no}, the best of {ROUND_CAP} {start_round}. Ink: {BEST_SO_FAR_COLUMN}, "
            f"{FORMULATION_CAP} {best_no}. Dashed tick: the target; an arrow where higher or "
            "lower is better. Model-free: measured values only.")


# Can we trust the predictions? — forecast against actual, over time.
STORY_FORECAST = "Predicted 95 % range"
STORY_LANDED_IN = "Landed in range"
STORY_LANDED_OUT = "Landed outside"
STORY_NO_FORECAST = "No forecast yet"
STORY_TRUST_CAPTION = "Blue: predicted before tasting. Dot: measured."


def story_trust_answer(inside, total):
    """'13 of 15 landed in their predicted range'."""
    return f"{inside} of {total} landed in their predicted range"


def story_trust_so_far(inside, total, needed):
    """Not yet, too few: '7 of 7 landed in range; 10 needed to judge'."""
    return f"{inside} of {total} landed in range; {needed} needed to judge"


def story_misses(above, below):
    """The misses' side, for the "?": 'Both misses scored higher than predicted.'"""
    n = above + below
    if n == 0:
        return ""
    if below == 0:
        who = "Both misses" if n == 2 else ("The miss" if n == 1 else f"All {n} misses")
        return f"{who} scored higher than predicted."
    if above == 0:
        who = "Both misses" if n == 2 else ("The miss" if n == 1 else f"All {n} misses")
        return f"{who} scored lower than predicted."
    return f"{above} scored higher than predicted, {below} lower."


def story_trust_help(inside, total, misses_text=""):
    """The forecast chart's "?": how to read it, then the BO term."""
    return (f"Each light blue bar is the 95 % range the model predicted for a {FORMULATION}'s "
            f"{OVERALL_SCORE_COLUMN} before it was made, from the earlier {ROUND_CAP}s only; the "
            "blue tick, the predicted value. The "
            f"dot is what was measured: filled inside the range, hollow outside. {inside} of "
            f"{total} landed inside. {misses_text} About 1 in 20 is expected outside. "
            "BO term: Round-ahead (rolling-origin) posterior predictive coverage.").replace(
                "  ", " ")


# What drives each result? — a finding, from the profiler.
def story_driver(name):
    """'Methylcellulose matters most'."""
    return f"{name} matters most"


def story_driver_end(ref_label, end):
    """'Best so far uses its Highest'."""
    return f"{ref_label} uses its {end}"


def story_driver_then(name):
    """'Then Textured soy protein'."""
    return f"Then {name}"


STORY_NO_DRIVER = "No single change clearly leads"


def story_driver_help(ref_label):
    """The finding's "?": what "matters most" means."""
    return (f"The largest predicted change in {OVERALL_SCORE_COLUMN} when one row moves alone "
            f"across the amounts tested near {ref_label}, the others as they are. A model reading, "
            "not a measured result.")


# What have we tried? — the search path.
STORY_PATH_CAPTION = f"One dot per {FORMULATION}; gaps are untested."
STORY_SHOW_LINES = "Show as lines"
STORY_SHOW_LINES_HELP = (f"Draw each {FORMULATION} as one line across every row, to see "
                         "combinations.")
STORY_PATH_TESTED = f"Tested {FORMULATION}"


def story_path_help(settings_only=False):
    """The search path's "?": how to read a panel."""
    what = "process setting" if settings_only else "ingredient or process setting"
    return (f"One small chart per {what}: across, its allowed amounts from Lowest to Highest; down, "
            f"the {ROUND_CAP}s, the first at the top. Each dot is one {FORMULATION}; the grey "
            f"bar joins a {ROUND_CAP}'s lowest and highest amounts. Ink ring: "
            f"{BEST_SO_FAR_COLUMN}. Orange: this {ROUND_CAP}'s suggestions. Empty stretches "
            "were not tried. Bars that shorten show the search settling. BO term: the design "
            "points by iteration, per input dimension.")
STORY_VALUE = "Value"


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


def score_of(text, ceiling):
    """'96.4 of 100' when the score is on 0–100 (owner: "Flat" at 96.4 then
    reads as converged, not stuck); the number alone on any other scale."""
    try:
        on_100 = ceiling is not None and abs(float(ceiling) - 100.0) < 1e-6
    except (TypeError, ValueError):
        on_100 = False
    return f"{text} of 100" if on_100 else str(text)


def strip_progress(start_text, best_text):
    """The strip's Progress line: '88.4 → 96.4 of 100'; 'Best 96.0 of 100'."""
    return f"Best {best_text}" if start_text is None else f"{start_text} → {best_text}"


def strip_short(text):
    """A card line cut for the strip: 'Overall score: 13 / 15 in range' ->
    '13 / 15 in range' (the label already says Predictions)."""
    return text.removeprefix(f"{OVERALL_SCORE_COLUMN}: ")


def strip_jump_help(question):
    """A strip item's click: 'Scrolls to: Are we improving?'."""
    return f"Scrolls to: {question}"


# The cards are gone: what named them now names the strip or What next?.
SEE_CHART_HELP = "Scrolls to the section that answers it."
EV_METHOD = "Kernel, priors and every check: How the app knows, top right."
REVIEW_STATUS_HELP = (f"The {TAB_INSIGHTS} status line in brief. Predictions: can you trust "
                      "them? Progress: is Best so far still rising?")
GAIN_REASON_NOT_YET = "Predictions do not read Reliable yet"
GAIN_REASON_OFF = ("Predictions read Off: the Overall score misses its 95 % range more often "
                   "than expected")


def tip_hero_hidden(reason):
    """The hidden line's "?": why, and that suggestions are still made."""
    return (f"Expected gain is not drawn: {reason}. Suggestions are still made the same "
            f"way; {Q5_TITLE} says what would bring the gain back.")


def tip_gain_greyed(reason):
    """The greyed gain's "?": why, and that suggestions are still made."""
    return (f"Expected gain is drawn in grey: {reason}. Suggestions are made the same way "
            f"regardless; {Q5_TITLE} says what would make the gain count again.")


def story_trust_so_far(inside, total, needed):
    """Not yet, too few: '7 of 7 landed in range; needs 10' (review story
    2: it read '…; 10 needed to judge', 12 words with its bold word, over
    the 10 a status line may take)."""
    return f"{inside} of {total} landed in range; needs {needed}"


# --- review story 3 ---
# Review story 3 (prototype, 2026-10-02; owner: demo the yogurt pilot only,
# and fade the profiler's untested stretches so a model curve there does
# not read as a finding): the line thin and dashed, the band lighter, over
# amounts not tested near; the finding from tested stretches only; a note
# when two rows moved together; and Expected gain defined once.
PROFILER_UNTESTED_DASHED = "Not tested near here (dashed)"


def story_no_leader(kind="ingredients"):
    """'No single ingredient clearly leads yet' (`kind`, _more_kind's)."""
    what = {"settings": "setting", "both": "ingredient or setting"}.get(kind, "ingredient")
    return f"No single {what} clearly leads yet"


def story_leader_rule(noise_text):
    """The finding's "?", its rule: tested stretches only."""
    return (f"Counted over tested amounts only: the dashed, untested stretches are the model's "
            f"guess. A clear lead moves the predicted {OVERALL_SCORE_COLUMN} by at least "
            f"{noise_text} points (one tasting's noise) and by a quarter more than the next row.")


def moved_note(first, second):
    """'Tap water and Coconut cream moved together: effects mixed.'"""
    return f"{first} and {second} moved together: effects mixed."


def moved_r(r):
    """'r = −0.70' (Pearson's r, signed, two decimals)."""
    return f"r = {r:.2f}".replace("-", "−")


def moved_help(first, second, r, n, others=()):
    """The note's "?": the exact r, why it matters, and any other pairs."""
    text = (f"{first} and {second} changed together across the {n} tested {FORMULATION}s "
            f"({moved_r(r)}). No model can separate their effects: what the chart credits to "
            f"one may belong to the other. Noted from |r| ≥ 0.7. To separate them, test "
            f"{FORMULATION}s that change one without the other.")
    if others:
        text += " Also moved together: " + "; ".join(
            f"{a} and {b} ({moved_r(x)})" for a, b, x in others) + "."
    return text


METHOD_GAIN = ("Expected gain: expected improvement in the Overall score (qLogNEI), in points: "
               "how much a formulation is expected to add above the model's estimate of the best "
               "tested formulation. Not information gain.")


# --- review learned ---
# Review learned (prototype, 2026-10-02; owner: the Review "doesn't inspire
# confidence in the tool"): lead with honest, checkable evidence of what
# the model learned. Each measurement's within-Round order, forecast before
# tasting, against chance; its verdict word; the effects that held across
# the last three refits; "Called it?" for the latest Round; the open
# Round's forecast on Make a round; and What next?'s preview of the next
# suggestions.
LEARNED_TITLE = "What has the model learned?"
LEARNED_STRIP_LABEL = "Model"
LEARNED_WORDS = {"predictable": "Predictable", "partly": "Partly", "not_yet": "Not yet"}
LEARNED_REASONS = {"flat": "barely varies", "few": "too few pairs",
                   "chance": "no better than chance"}
LEARNED_COL_RESULT = "Result"
LEARNED_COL_VERDICT = "Verdict"
LEARNED_COL_PAIRS = "Ordered right before tasting"
LEARNED_COL_TOP = "Picked the top one"
LEARNED_COL_EFFECTS = "Learned effects"
LEARNED_CHART_X = "Pairs ordered right (%)"
LEARNED_CHANCE = "Chance"
LEARNED_CAPTION = "Dot: share right. Bar: 95 % interval."
LEARNED_EFFECTS_CAPTION = "Effects named only when steady 3 Rounds."
LEARNED_NONE_YET = "Nothing to check yet: needs a scored Round."
LEARNED_FORECAST_TITLE = "Forecast before tasting"
LEARNED_FORECAST_NONE = "No measurement predictable yet."
LEARNED_PREVIEW_TITLE = "Next suggestions"
LEARNED_PREVIEW_SPINNER = "Previewing the next suggestions…"
LEARNED_PREVIEW_CAPTION = "Generate on Make a round makes these."
LEARNED_PREVIEW_FAILED = "Preview not available. Generate on Make a round."
NOT_PREDICTABLE_YET = "Not predictable yet"


def learned_strip(predictable, partly):
    """The strip's Model item, (word, line) at most 6 words: ('Predicts',
    'Consistency, Tangy'); ('Partly', 'Consistency'); ('Not yet', '')."""
    names = predictable or partly
    if not names:
        return "Not yet", "predictable"
    shown = ", ".join(names[:2]) + (f" +{len(names) - 2}" if len(names) > 2 else "")
    return ("Predicts" if predictable else "Partly"), shown


def learned_answer(predictable, partly, not_yet):
    """The section's answer: '**Predicts Consistency, Tangy.** Not yet:
    Creaminess.'"""
    if predictable:
        head = f"Predicts {', '.join(predictable)}"
    elif partly:
        head = f"Partly predicts {', '.join(partly)}"
    else:
        head = "Not predictable yet"
    rest = []
    if predictable and partly:
        rest.append(f"Partly: {', '.join(partly)}")
    if not_yet and (predictable or partly):
        rest.append(f"Not yet: {', '.join(not_yet)}")
    return f"**{head}.**" + (" " + ". ".join(rest) + "." if rest else "")


def learned_verdict(word, reason=""):
    """'Predictable'; 'Not yet (barely varies)'."""
    return f"{word} ({reason})" if reason else word


def learned_pairs(right, pairs, chance):
    """'16 of 18 pairs (chance 9)'."""
    if not pairs:
        return "no pairs yet"
    c = f"{chance:g}" if float(chance).is_integer() else f"{chance:.1f}"
    return f"{right} of {pairs} pairs (chance {c})"


def learned_top(right, rounds, chance):
    """'7 of 8 Rounds (chance 2.7)'."""
    if not rounds:
        return "—"
    return f"{right} of {rounds} Rounds (chance {chance:.1f})"


def learned_effects(raised, lowered):
    """'Raised by Tapioca starch, Xanthan gum. Lowered by Salt.'; '—'."""
    parts = []
    if raised:
        parts.append(f"Raised by {', '.join(raised)}")
    if lowered:
        parts.append(f"Lowered by {', '.join(lowered)}")
    return ". ".join(parts) + "." if parts else "—"


def learned_effect_tip(var, name, lo, hi, mean, low, high):
    """One effect's numbers, for the "?"."""
    sign = lambda v: f"{v:+.2f}".replace("-", "−")
    return (f"{var} from {lo} to {hi} (its tested stretch): {name} {sign(mean)} "
            f"(95 % interval {sign(low)} to {sign(high)}).")


def learned_headline(var, name, up):
    """The profiler's finding from a stable effect: 'Tapioca starch raises
    Consistency'."""
    return f"{var} {'raises' if up else 'lowers'} {name}"


def learned_not_predictable(names):
    """'Not predictable yet: Creaminess, Thickness uniformity.'"""
    return f"{NOT_PREDICTABLE_YET}: {', '.join(names)}."


def learned_help(pairs_rule, skill_line):
    """The section answer's "?": the rule, then the Overall score's expert
    line."""
    return "\n\n".join(t for t in (pairs_rule, skill_line) if t)


LEARNED_RULE = ("Each Round's formulations are forecast before tasting, from the formulations "
                "scored before that Round. Within each Round, every pair whose measured values "
                "differ counts as ordered right when the forecast put them in the same order; "
                "a pair measured equal is not counted. Guessing gets half right. Predictable: "
                "right clearly more often than chance (one-sided binomial test, p < 0.05, at "
                "least 5 pairs). Partly: right in more than half, p < 0.20. Not yet: otherwise, "
                "or when more than half of its pairs measured equal (it barely varies).")
LEARNED_TOP_RULE = ("Picked the top one: the formulation forecast highest was measured highest "
                    "(the Overall score: the Round's best), in Rounds where not all measured "
                    "equal. Chance: the share a random pick would get.")


def learned_skill_line(inside, total, vs_mean, vs_best, n):
    """The Overall score's expert line: coverage and skill against both
    baselines."""
    if not total:
        return ""
    line = f"{OVERALL_SCORE_COLUMN}: {inside} of {total} inside the 95 % predicted range."
    if vs_mean is not None and vs_best is not None:
        line += (f" Skill over {n} checks: {vs_mean:+.2f} against the average of earlier "
                 f"scores, {vs_best:+.2f} against Best so far (1 − squared forecast error ÷ "
                 f"squared error of the baseline; 0 is no better than the baseline).")
    elif vs_mean is not None:
        line += f" Skill against the average of earlier scores: {vs_mean:+.2f}."
    else:
        line += " Skill needs 10 checks."
    return line.replace("−0.", "−0.").replace("-", "−")


LEARNED_STABLE_RULE = ("An effect is named only when it is steady: moving that ingredient or setting "
                       "from the 10th to the 90th percentile of its tested amounts, the other "
                       "amounts as tested, changes the prediction with a 95 % posterior interval "
                       "that excludes 0, and in the same direction under the models fitted "
                       "through each of the last 3 Rounds.")


def called_title(round_no):
    """'Round 6: called it?'"""
    return f"Round {round_no}: called it?"


def called_pairs(name, right, counted, total):
    """'Consistency: 2 of 2 pairs in the right order.'; all equal: 'Tangy:
    all measured the same, nothing to order.'"""
    if not counted:
        return f"{name}: all measured the same, nothing to order." if total else f"{name}: —"
    return f"{name}: {right} of {counted} {'pair' if counted == 1 else 'pairs'} in the right order."


def called_pair(high, low, right):
    """'F18 > F16 ✓'; a measured tie: 'F16 = F17 (tie)'."""
    if right is None:
        return f"F{min(high, low)} = F{max(high, low)} (tie)"
    return f"F{high} > F{low} {'✓' if right else '✗'}"


def called_top(top, measured_top, right):
    """'Forecast highest: F18 ✓'; 'Forecast highest: F17 ✗ (measured: F16)'."""
    if right is None:
        return ""
    if right:
        return f"Forecast highest: F{top} ✓"
    return f"Forecast highest: F{top} ✗ (measured: {', '.join(f'F{n}' for n in measured_top)})"


def called_range(inside, above, below):
    """The Overall score's landing: '1 of 3 in range, 2 better than
    forecast.' Above the range is better than forecast, not a miss."""
    total = inside + above + below
    parts = [f"{inside} of {total} in range"]
    if above:
        parts.append(f"{above} better than forecast")
    if below:
        parts.append(f"{below} below forecast")
    return ", ".join(parts) + "."


def forecast_line(name, cells, top):
    """'Consistency: F16 13.5 (11.1–≥ 15) · F17 … Highest: F18.'"""
    line = f"{name}: " + " · ".join(cells) + "."
    return line + (f" Highest: F{top}." if top is not None else "")


def forecast_help(n_rows, round_no):
    """The forecast panel's "?"."""
    return (f"Forecast from the {n_rows} {FORMULATION}s scored before Round {round_no}: the "
            f"same forecast the Review checks once results are saved (Called it?). Only "
            f"measurements the model has ordered right before (Predictable) are shown. A value "
            f"at the top of its Range reads ≥ that top.")


PREVIEW_COLUMNS = ("#", "Role", PREDICTED_SCORE_COLUMN, "Changes vs Best so far")


def preview_line(k, role, score_text, changes):
    """'1 · refines the best · Overall score 96 (92–99+) · Salt −0.10 g'."""
    parts = [f"{k}", role, f"{OVERALL_SCORE_COLUMN} {score_text}"]
    if changes:
        parts.append(changes)
    return " · ".join(parts)


def preview_help(n, best_no):
    return (f"The {n} {FORMULATION}s Generate would make now, previewed on a copy of the project "
            f"(nothing is saved). Changes are against Best so far ({FORMULATION_CAP} {best_no}). "
            f"Predicted {OVERALL_SCORE_COLUMN} with its 95 % interval; ≥ 99 is the top of the "
            f"Range.")


METHOD_LEARNED_TITLE = "What the model learned: definitions"
METHOD_FORECAST_CHART = "Overall score forecast against measured, Round by Round"


def review_status_learned(model, progress, extra=None):
    """Make a round's line with the Review's Model word: 'Review: Model
    **Predicts** · Progress **Flat**'."""
    line = (f"{REVIEW_STATUS_PREFIX}: {LEARNED_STRIP_LABEL} **{model}** · "
            f"{CARD_PROGRESS} **{progress}**")
    return line + (f" · **{extra}**" if extra else "")


def learned_evidence(rows, skill_line=""):
    """The Model item's evidence: 'Evidence: pairs ordered right before
    tasting — Overall score 14/21, Firmness 16/23. Overall score: 13 of 15
    inside the 95 % predicted range. …'"""
    pairs = ", ".join(f"{name} {right}/{n}" for name, right, n in rows if n)
    line = f"Evidence: pairs ordered right before tasting — {pairs or 'none yet'}."
    return f"{line} {skill_line}".strip()


LEARNED_RANGE_RECORD = "The Overall score's Round-ahead record:"
