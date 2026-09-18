"""Every word the user reads. Change a word here and it changes everywhere;
the tests read this module too."""

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
STARTING_APP = (f"Starting {APP_TITLE}… loading its components. "
                "This takes a few seconds.")

# The three tabs, in loop order. The separator is U+00B7 MIDDLE DOT.
TAB_SETUP = "1 · Set up"
TAB_BATCH = f"2 · Make a {ROUND}"
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
# screen: the grid's columns are Lowest measurable and Highest measurable.
# HIGHEST_MEASURABLE_LABEL is the same words, defined with the grid's other
# column headers further down; this one cannot read it because it is needed
# above them. The guard below the grid pins the two together.
WIDEN_RANGE_HINT = " Raise Highest measurable in Set up, or check the value."


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
    is a hard rule: the sheet that breaks one has to say which, in the same
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
    return (f"Best moved from {FORMULATION_CAP} {before} to "
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
    "A ready-made plant-based burger: six rows, three pre-mixes, a mixing "
    "time, a fat limit, and firmness, juiciness and cook loss. Try it "
    "before setting up your own."
)

# The sample project's own name, so app.py and ui_setup.py agree on how it is
# recognised: by name, the same way the app has always told it apart from a
# project the user made.
SAMPLE_PROJECT_NAME = "Sample project"

# The sample's own targets_source, set once when it is built.
SAMPLE_TARGETS_SOURCE = (
    "Illustrative targets, not recorded panel results: scores are against "
    "an 80/20 beef control cooked to 74 °C core, which "
    "the example panel rates firmness 6 and juiciness 7 out of 10, against written "
    "anchors (12 panellists, two sessions, February 2026). Above firmness 7 "
    "the patty eats rubbery; below juiciness 5 it eats dry and chalky."
)

# The sample's Method: how the bench makes one formulation, in the order it
# is done. Printed on the Round sheet under the title, because three
# formulations that must be made identically except for the amounts had
# nothing on paper saying how.
SAMPLE_METHOD = (
    "Hydrate the textured pea protein in 2.2 x as much of the water, "
    "10 min at 45 °C. Add the dry blend and gluten to the rest of the water "
    "and mix 60 s. Add the fat phase and mix for the mixing time. Form "
    "100 g patties, 100 mm x 12 mm. Chill 20 min at 4 °C. Griddle at "
    "180 °C, 3 min per side, to 74 °C core; serve within 3 min."
)

# Tab 1's two-line welcome for the sample project, shown only before its
# first formulation is scored; the second sentence names the lit button so a
# first-time visitor knows what to do next.
SAMPLE_TAB1_DESCRIPTION = (
    "A plant-based burger study with ingredients, pre-mixes and mixing time. "
    "Next: make a round."
)

# The project-level Method: one text area in More settings, printed on the
# Round sheet under the title.
# The four things a project can record beside what it is asked to. All four
# are off for a new project: Vendor and SKU are specification data typed
# once and never looked at again, and Lot and Actual are two more columns
# down every printed page of a project that does not want them.
def no_such_field(name):
    """A caller asking the project to record something it has no box for."""
    return f"{name} is not one of the things this project can record."


ALSO_RECORD_LABEL = "Also record:"
ALSO_RECORD_HELP = ("Each one adds a column to the grid or a cell to the "
                    "printed pages. Nothing already recorded is lost by "
                    "turning one off.")
RECORD_FIELD_LABELS = {
    'vendor': "Vendor",
    'sku': "SKU",
    'lot': "Lot",
    'actual': "Actual amounts",
}


def record_field_on(label):
    return f"{label} is recorded now."


def record_field_off(label):
    return f"{label} is not recorded any more."


METHOD_LABEL = "Method"
METHOD_HELP = "How the formulation is made, in the order the bench does it."
METHOD_PLACEHOLDER = ("e.g. Mix the dry blend into the water, 60 s. Add the "
                      "fat phase. Form 100 g patties.")
METHOD_SAVED = "Method saved."
METHOD_CLEARED = "Method cleared."
# The heading it prints under on the Round sheet: every formulation of a
# round is made the same way, and the sheet says so once.
METHOD_SHEET_HEADING = "Method — the same for every formulation"

PROJECT_LOAD_ERROR_SIDEBAR_NOTE = ("This project could not be opened. The "
                                   "main screen says why.")

SAVED_COPIES_HEADING = "**Saved copies**"
SAVED_COPIES_CAPTION = ("A copy holds everything: ingredients, "
                        "measurements, every formulation and result.")

COPY_UNAVAILABLE = ("Saving a copy is unavailable while the project "
                    "file cannot be read.")
SAVE_A_COPY = "Save a copy of this project"

OPEN_A_SAVED_COPY = "Open a saved copy"
OPEN_SAVED_COPY_CAPTION = ("A copy you saved yourself, or one the app saved "
                           "before a change. The current project is copied "
                           "first.")
CHECK_THIS_COPY = "Open this copy"

# The copies the app makes for itself, before a change that cannot be
# undone. They were files in the project folder with names like
# `burger_pre_edit` that nothing on screen ever mentioned, under a heading
# promising a list — so each is named here for what it was taken before.
SAFETY_COPIES_CAPTION = "Copies this app made before a change:"
SAFETY_COPY_REASONS = {
    'pre_edit': "Before an edit to a recorded result",
    'pre_delete': "Before a deletion",
    'pre_restore': "Before opening a saved copy",
    'archived': "Before starting this project over",
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


def open_saved_copy_warning(name, holds, project_name, held):
    """`holds` is the list of 'N formulations'/'N ingredients'/'N process
    settings' phrases the saved copy contains; `held` is the same phrase for
    what replacing it would give up."""
    return (f"This copy holds **{name}**: " + ", ".join(holds)
            + f". Replace **{project_name}**, which has {held}? " + COPY_KEPT)


YES_REPLACE = "Yes, replace"
COPY_APPLY_FAILED = ("This copy could not be applied. Your current "
                     "project was not changed.")


def restored_flash(count_text, project_name, archived=None):
    text = f"Restored {count_text} into {project_name}."
    if archived:
        text += f" A copy was saved as {archived}."
    return text


MANAGE_PROJECT = "Manage project"

START_OVER_LABEL = "Start this project over"
YES_START_OVER = "Yes, start over"


def start_over_warning(project_name, held_text=None):
    """`held_text` is None for an already-empty project."""
    if held_text is None:
        return f"Start **{project_name}** over? It becomes empty. " + COPY_KEPT
    return (f"Start **{project_name}** over? Its {held_text}, ingredients "
            f"and measurements all go. " + COPY_KEPT)


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
        "2. **Add ingredients** and the measurements you will record.\n"
        f"3. **Make a {ROUND}**, weigh out the formulations, and record "
        "what you measured."
    )


DOWNLOAD_TEMPLATE = "Download ingredients template (Excel)"


# ------------------------------------------------------------------ #
# Load-error / save-error banners, and the discarded-batch notice.
# ------------------------------------------------------------------ #
def project_load_error_info():
    return (
        "This project file is damaged, so editing is off. Two ways out, both "
        f"in the sidebar: {OPEN_A_SAVED_COPY}, if you saved one. Or "
        f"{MANAGE_PROJECT} › {START_OVER_LABEL} — the damaged file is copied "
        "first."
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
    who = f"{ROUND_CAP} {no}" if no is not None else f"The open {ROUND}"
    return (f"{who} was discarded: {reason} after it was made. "
            "Generate a new one.")


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
        line += (f" A {FORMULATION} you added goes with the {ROUND} — a "
                 "set-up change can make it invalid.")
    return line


def project_created(name):
    return f"Created {name}."


def sample_project_rebuilt(name):
    """'Sample project put back the way it started.' — clicking Try the
    sample project again on a sample nobody has made a round from rebuilds
    it, which throws away every edit made to it. It said only "Opened
    Sample project.", so the reader\'s own rules and amounts were gone with
    nothing on screen about it."""
    return f"{name} put back the way it started."


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
    return f"{ROUND_CAP} {no} · recorded"


# ------------------------------------------------------------------ #
# Tab 2 · Make a batch: generate, the batch table, downloads, record
# results, upload results.
# ------------------------------------------------------------------ #
GENERATE_FORMULATIONS_DISABLED = "Generate formulations"
BACK_TO_SETUP = "Back to set up"

GENERATING_SPINNER = "Choosing the next formulations…"
GENERATE_FAILED = (
    "The app could not choose formulations this time. Try again with fewer. "
    "If it keeps happening, loosen any limit you added recently."
)


def repeat_of_formulation(no):
    return f"Repeat of {FORMULATION_CAP} {no}"


def batch_ready(no):
    """The flash a freshly generated batch lands on. It names the two steps
    that follow, in the order the screen puts them: a batch that is "ready"
    and nothing more left the reader looking for what to do with it."""
    return (f"{ROUND_CAP} {no} is ready to make. Print the sheets, then "
            "record the results below when you have them.")


FORMULATIONS_TO_GENERATE = "Formulations to generate"


# The expander under the Generate row (and under the batch table once one is
# open): a formulation the scientist chose, added to the batch beside the
# generated ones. It replaced the Repeat checkbox, which could only ever
# repeat the best.
ADD_OWN_EXPANDER = f"Add a {FORMULATION} to this {ROUND}"
ADD_OWN_NO_BATCH_CAPTION = ("Want the app's suggestions too? Click "
                            "Generate first, then add yours.")
OWN_NOTE_PLACEHOLDER = "e.g. Repeat of 4 with more salt"
START_FROM_BEST = "Start from the best so far"
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


# The one line that says how formulations are chosen. It is the third How it
# works bullet and, word for word, the caption under Generate — before and
# after the fifth formulation alike. Two captions for the two halves of one
# rule made a reader who had seen only one of them think there were two.
HOW_CHOSEN = ("Until five formulations have results, new ones are spread out "
              f"to cover the allowed amounts. After that, each {ROUND} aims "
              "closer to your targets.")


# The open batch reads as the three steps of the work, each headed with its
# number. Step 1 takes no count of its own: the title directly above it
# already says how many formulations there are.
# Unnumbered. The tab strip above them is 1 · Set up, 2 · Make a round,
# 3 · Results, and "3 · Record the results" sitting under "3 · Results"
# had the cold reader clicking the tab when they meant the section. The
# order is the order they are drawn in.
STEP_MAKE_HEADING = f"##### Make the {FORMULATION}s"
STEP_PRINT_HEADING = "##### Print the sheets"
STEP_RECORD_HEADING = "##### Record the results"


def make_these(no, n):
    """'Batch 1 · make this 1 formulation' / '... make these 3
    formulations'."""
    word = FORMULATION if n == 1 else FORMULATION + "s"
    return (f"**{ROUND_CAP} {no} · make "
            f"{'this' if n == 1 else 'these'} {n} {word}**")


# Every row in the batch is one the user added by hand, so there is no
# Generate control on the screen and nothing else would say why.
ONLY_OWN_FORMULATIONS_CAPTION = (
    "This round holds only your own formulations. For the app's suggestions "
    f"too, click Generate a different {ROUND}, then Generate."
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


# What each formulation is trying. The three kinds are written lower case
# because each is a fragment: compared_with_cell() capitalises the one it is
# given as the first word of the cell, and nothing says them anywhere else.
SUGGESTION_CLOSE = "close to the best"
SUGGESTION_DIFFERENT = "trying something different"
SUGGESTION_SPREAD = "spread across the allowed amounts"
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
    """'Trying something different · Water +12.00 g, Wheat gluten −3.00 g'.

    The kind leads because it is the answer to the question the reader asks
    first — is this a small step or a new direction? — and the amounts that
    follow say which ingredients carry it. During the cold start there is
    nothing to compare with, so the kind stands alone."""
    head = kind[:1].upper() + kind[1:]
    return f"{head} · {changes}" if changes else head


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
    return (f"Every {FORMULATION} in this {ROUND} adds up to this. Change "
            "it and the amounts are recalculated at that size; the "
            f"table's {total_column(unit)} shows it.")
# An example, not a description: the help above already says what the box is
# for.
BATCH_SIZE_PLACEHOLDER = "e.g. 100"

NO_BATCH_SIZE_OF_ITS_OWN = f"This {ROUND} has no {BATCH_SIZE_NOUN} of its own."


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


GENERATE_DIFFERENT_BATCH = f"Throw this {ROUND} away and generate again"


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

RECORDED_FALLBACK = "recorded"


# What a row is missing, named. "· partial" told the reader a word rather
# than which measurement nobody took, on a screen that had room for the
# name.
NOT_MEASURED = "not measured"


def not_measured_tail(names_text):
    """' · Juiciness not measured' — `names_text` is one name or a
    number_list of several."""
    return f" · {names_text} {NOT_MEASURED}"


def recorded_line_partial(line, names_text):
    tail = not_measured_tail(names_text)
    return f"{line}{tail}" if line else tail.lstrip(" ·").lstrip()


def recorded_line_note(line, note):
    return f"{line} · Note: {note}" if line else f"Note: {note}"


RECORD_RESULTS_CAPTION = ("One number per measurement; the panel mean where "
                          "a panel rated it. Leave blank if it was not "
                          "measured.")


def formulation_heading(no):
    return f"**{FORMULATION_CAP} {no}**"


NOT_SCORED_HELP = (f"Ticked wins over any number typed in this {ROW}. Say "
                   f"why in {NOTE}; it stays with the formulation.")

NOTHING_TO_SAVE = "Nothing to save — at least one formulation needs results."
# Said by food_bo when a row arrives with nothing on it, and by the Results
# tab when a not-scored formulation is saved with every box empty. One
# sentence, so the refusal reads the same wherever the row was typed.
ENTER_A_MEASUREMENT = "Enter a value for at least one measurement."
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
UPLOAD_HELP_CAPTION = (
    "Fill in the Measured cells on the round sheet you downloaded above and "
    "upload the file here. Formulations are matched on their number."
)
# The same door, said the other way round, once the workbook has actually
# been downloaded in this session: the fold is open and this is the line
# inside it, so the reader coming back with a filled-in file finds the door
# already ajar rather than collapsed under a heading beginning "Or".
UPLOAD_SHEETS_ARE_BACK_CAPTION = (
    "When the sheets come back, upload the file here. The app reads what "
    "you wrote in the boxed cells."
)
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
UPLOAD_PREVIEW_CAPTION = "What the file says. Check it before saving."


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
    "max": "Higher is better",
    "min": "Lower is better",
    "target": "Hit a target",
}


def target_value(value):
    """'Target 6' — the Goal column's cell text for a target measurement."""
    return f"Target {float(value):g}"


def range_text(low, high):
    """'0 to 10' — the Range column's cell text, before its unit."""
    return f"{float(low):g} to {float(high):g}"


LIMIT_KEPT = (f"Formulations already made are kept. The next {ROUND} will "
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
    HOW_CHOSEN,
    "Limits are never crossed. A "
    "formulation of your own is recorded as you typed it.",
    "Each suggestion says whether it stays close to the best or tries "
    "something different, and what it changes.",
]

# The fold directly under it, for the reader who wants the arithmetic. The
# bullets above raise the question — what is closeness, exactly? — and this
# answers it; nine bullets in one fold answered it before anyone asked.
HOW_CLOSENESS_HEADING = "**How closeness is calculated**"
HOW_CLOSENESS = [
    "Higher is better: closeness = (measured − lowest) ÷ (highest − lowest), "
    "so the top of your range scores 1 and the bottom scores 0.",
    "Lower is better: the reverse — the bottom of your range scores 1 and the "
    "top scores 0.",
    "Hit a target: closeness is 1 at the target and falls evenly with "
    "distance, by one point per full range; the lowest score depends on how "
    "far the target sits from the ends of your range. Because of that "
    "floor, a target measurement sways the score a little less than its "
    "share suggests.",
    "The app learns the one overall score, so changing a share, a goal "
    "or a range re-scores every past formulation.",
    "A formulation of your own counts like any other. Making the best one "
    "again teaches the app how noisy your measurements are.",
]

VARIABLES_HEADER = "Ingredients and process settings"


def made_before_units_caption(unit):
    return (f"These amounts were recorded before units existed, so they "
            f"are read as {unit}. Set the right unit below.")


UPLOAD_INGREDIENTS_EXPANDER = "Or upload an ingredients file"

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
BASELINE_HELP = ("The amounts of the first formulation recorded. Later "
                 "suggestions are read against it, so those results still "
                 "count.")
def saved(name):
    """Subject first, like added() above it and every other flash on the
    tab."""
    return f"{name} saved."
# "0 if blank" and "leave a cell empty for no value" sat on one screen
# contradicting each other, and the app never knows an ingredient is
# fat-free — only that a cell was left empty.
PROPERTY_BLANK_RULE = "An empty cell counts as 0 in any limit."
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
    return f"fixed at {amount_text}"


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
VENDOR_HELP = ("Printed on the sheets so the bench knows what to reach "
               "for. The app never reads it.")
SKU_HELP = ("The supplier's own code for it, printed on the sheets beside "
            "the vendor. The app never reads it.")
# What each measurement is worth out of 100. It moved up here from the
# measurements block below because the grid's own header is built from it,
# and a module reads top to bottom.
COL_SHARE = "Share of score"
SHARE_COLUMN = f"{COL_SHARE} (%)"
SHARE_HELP = ("What this measurement is worth out of 100. Change one and "
              "the others move to keep the column adding up to 100.")
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

INGREDIENT_GRID_CAPTION = (f"One row per {INGREDIENT} or process setting; "
                           f"Type says which ({KIND_INGREDIENT} or "
                           f"{KIND_SETTING}). Type a new one on the empty "
                           "row at the bottom. Type the same number in "
                           f"{LOWEST_LABEL} and {HIGHEST_LABEL} to fix an "
                           "amount.")
MEASUREMENT_GRID_CAPTION = ("One row per measurement. Share of score says "
                            "what each one is worth out of 100.")

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
    return f"Each ingredient's figure{basis}. " + PROPERTY_BLANK_RULE


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
DELETE_EVEN_IF_USED_CHECKBOX = f"{DELETE_EVEN_IF_USED} — those amounts go too"


def deleted(name):
    return f"{name} deleted."


REPLACE_INGREDIENTS_BUTTON = "Replace ingredients"
LOAD_INGREDIENTS_BUTTON = "Load ingredients"
# The template this caption describes carries a Rule column, and a reader
# who downloaded it was told extra columns become properties — which that
# one does not (the name is reserved) — with nothing on screen to say what
# it is for.
INGREDIENTS_FILE_CAPTION = ("A file with the columns Name, Lowest, Highest "
                            "and, optionally, Unit, Rule, Part of, Made as and % of pre-mix. Extra numeric columns "
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
LOWEST_MEASURABLE_LABEL = "Lowest measurable"
HIGHEST_MEASURABLE_LABEL = "Highest measurable"
MEASUREMENT_EXISTS_ERROR = ("That measurement already exists. Use Edit on "
                            "its row to change it.")


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
    return f"{name} added. Give each ingredient a figure for it."


def delete_property_warning(name, limits_text):
    head = (f"Delete {name} and its {limits_text}? " if limits_text
            else f"Delete {name}? ")
    return head + "Each ingredient's figure for it goes too. " + COPY_KEPT


def property_deleted(name, gone_text=""):
    return f"{name} deleted.{gone_text}"


def limit_went_with_it(limits_text):
    return f" Its {limits_text} went with it."


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
NO_LIMIT_PLACEHOLDER = "no limit"
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
    return (f" · {name} have no figure for it and count as 0." if many
            else f" · {name} has no figure for it and counts as 0.")


# The two tiers tab 1 folds everything optional into (spec 1.5). More
# settings holds what a project may want once — the default batch size,
# where the targets came from, the limits and the properties — and Advanced
# holds what a specialist wants at most once: the model settings and the two
# explanations. Both are collapsed, so the tab reads as its two grids.
MORE_SETTINGS_EXPANDER = "More settings"
ADVANCED_EXPANDER = "Advanced"

LIMITS_HEADING = "**Limits (optional)**"
# The property rule lives here, not in the closeness fold: a property never
# touches closeness — it feeds limits and nothing else.
# The blank-figure rule is NOT repeated here. It used to be, because the
# properties were a fold somewhere else on the tab; they are now a grid a
# few lines below this line, under a caption that says it, and each limit's
# own line names the ingredients it is reading as zeroes.
LIMITS_CAPTION = ("Every formulation the app suggests keeps every limit "
                  "here. A formulation of your own is recorded as you "
                  "typed it. A limit is on an amount you weigh out or a "
                  "property of your ingredients; measurements have goals and "
                  "targets instead.")


def old_limit_basis_caption(unit):
    return f"An older limit is now read per 100 {unit} of formulation."


def at_least(value):
    return f"at least {value:g}"


def at_most(value):
    return f"at most {value:g}"


def delete_limit_button(who):
    """'Delete limit on Fat per 100 g'. A bare `Delete limit` was drawn once
    per limit, identically, with nothing on the button to say which."""
    return f"Delete {LIMIT} on {who}"


def delete_limit_warning(who):
    return (f"Delete the {LIMIT} on {who}? The next {ROUND} no longer has "
            f"to obey it. " + COPY_KEPT)


def limit_deleted(who):
    return f"Limit on {who} deleted. The next {ROUND} no longer has to obey it."


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
                          f"{ROUND} sets its own {BATCH_SIZE_NOUN}.")
# An example, not a description: the help above already says what the box is
# for, and 100 g is what the sample ships with.
FORMULATION_TOTAL_PLACEHOLDER = "e.g. 100"

def total_still_holds(total_text):
    """'Each formulation still adds up to 100 g.' — the half-sentence an edit to
    the ingredient list adds to its own success line.

    The total's limit is over every ingredient, so it is rewritten on every
    such edit; this is the screen saying so. It was rewritten silently, and
    a reader who had just been told "Limits are never crossed" had no way to
    know whether the rule they typed had survived their own step 2."""
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
    return f"The {ROUND} stays at {size_text}."


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
    back are the size and the ranges — not the eight ingredients the size's
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
STANDARD_VS_EXPERT_CAPTION = ("Standard uses tested defaults and fits most "
                              "projects. Expert-selected lets a specialist "
                              "choose the model's kernel, prior, noise "
                              "handling and acquisition. These cannot be "
                              "changed for the life of the project.")
HOW_FORMULATIONS_CHOSEN_LABEL = "How formulations are chosen"
STANDARD_DEFAULT_OPTION = "Standard (default)"
EXPERT_SELECTED_OPTION = "Expert-selected"
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
FIXED_TINY_NOISE_CAPTION = ("fixed_tiny noise suits a deterministic "
                            "measurement, not a sensory panel — keep "
                            "default unless you have a specific reason.")
APPLY_EXPERT_SETTINGS_BUTTON = "Apply expert settings"
MODEL_SETTINGS_UPDATED = "Model settings updated."
PASTE_EXPERT_SETTINGS_CHECKBOX = "Or paste expert settings as JSON"
EXPERT_SETTINGS_JSON_LABEL = "Expert settings JSON"
APPLY_PASTED_SETTINGS_BUTTON = "Apply pasted settings"


def invalid_json(e):
    return f"Invalid JSON: {e}"


IN_USE_PREFIX = "In use: "

NEXT_MAKE_BATCH_BUTTON = f"Next: make a {ROUND}"


# ------------------------------------------------------------------ #
# Tab 3 · Results: the best formulation, every formulation, corrections
# (ui_results.py).
# ------------------------------------------------------------------ #
PARTIAL_SCORES_CAPTION = ("A formulation missing a measurement scores it as "
                          "zero, so its overall score is low. Record the "
                          "missing number to fix it.")


# 'Overall score' as a noun inside a sentence, and the scale it is on. The
# ceiling is a whole number — a formulation that hits every goal scores
# 100 — so it is written as one.
OVERALL_SCORE_LOWER = "overall score"


def batch_recorded_progress(no, before, now):
    """'Round 2 recorded · best overall score 86.60 → 88.10 of 100'.

    The number had no name and no scale, beside a sibling line that did
    carry the noun. Both say the same thing now, in the same words."""
    return (f"{ROUND_CAP} {no} recorded · best {OVERALL_SCORE_LOWER} "
            f"{before:.2f} → {now:.2f}")


def batch_recorded_no_improvement(no):
    return (f"{ROUND_CAP} {no} recorded · best {OVERALL_SCORE_LOWER} "
            "unchanged.")


def best_so_far_heading(no, batch_no=None):
    heading = f"Best so far: {FORMULATION_CAP} {no}"
    if batch_no is not None:
        heading += f" ({ROUND_CAP} {batch_no})"
    return heading


MEASURED_COLUMN = "Measured"
OFF_BY_COLUMN = "Off by"

# Directly under the best-so-far block: the one way back to Set up from
# Results, for a measurement that needs its range widened or an ingredient
# that needs a new limit once a formulation is on screen.
CHANGE_SETUP_FROM_RESULTS_BUTTON = "Change a measurement or an ingredient"
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


def overall_score_caption(score, ceiling, missing_text=""):
    """`missing_text` names the measurements nobody took, or is empty. A goal
    re-scores every formulation exactly as a share of the score or a range
    does, so it is named here with them."""
    return (f"{OVERALL_SCORE_COLUMN} {score:.2f} of {ceiling:g}"
            + (not_measured_tail(missing_text) if missing_text else "")
            + ". Scores only compare within this project. Change a share, a "
            "goal or a range and every score is recalculated.")


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
DOWNLOAD_ALL_FORMULATIONS_HELP = ("One row per formulation, with the same "
                                  "units the screen shows, and a second "
                                  "sheet holding the set-up they were made "
                                  "under. Formulations marked not scored are "
                                  "included, with their measurements blank.")

# ---------------------------------------------------------------- #
# `Edit past formulations`: one collapsed section for every way the
# record is fixed after the fact. It replaced three separate controls
# — a correction picker, a delete section and an import expander —
# the last of which hid the only way to enter work done before the
# project existed.
# ---------------------------------------------------------------- #
EDIT_PAST_FORMULATIONS_EXPANDER = f"Edit past {FORMULATION}s"

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
    return f"{no} · {NOT_SCORED.lower()}"


def formulation_scored(no):
    """A formulation that was left not scored, scored later from Results. It
    is not a correction: nothing recorded changed, a result arrived."""
    return f"{FORMULATION_CAP} {no} scored."


def formulation_unchanged(no):
    return f"{FORMULATION_CAP} {no} is unchanged."


def formulation_corrected(no):
    """One sentence for the whole correction. A row can now change its
    amounts and its measurements in one save, and naming every number that
    moved made a flash longer than the table it described."""
    return f"{FORMULATION_CAP} {no} corrected."


def back_to_batch_label(no, n):
    return f"Back to {ROUND_CAP} {no} · {n} to record"


START_NEXT_BATCH = f"Start the next {ROUND}"

PROGRESS_CHART_EXPANDER = "Progress chart"
NO_RESULTS_YET = "No results yet."
OVERALL_SCORE_COLUMN = "Overall score"
BEST_SO_FAR_COLUMN = "Best so far"
# C12: the All formulations sheet's date column. "Recorded" was a state
# everywhere else on the screen ("Round 1 · recorded", "2 to record") and
# a date only here.
DATE_RECORDED_COLUMN = "Date recorded"
PROGRESS_CHART_CAPTION = ("The top line only rises. A few flat "
                          f"{ROUND}s are normal; a long flat stretch "
                          "suggests this ingredient list is close to the "
                          "best it can do.")

def batch_open_record_first_caption():
    """food_bo.undo_last_batch's refusal. No screen reaches it any more —
    the Delete the last batch button retired with this section — but the
    method stays for its tests and it still has to speak English."""
    return f"Record or discard the open {ROUND} first."


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
ADD_PAST_FORMULATION_LABEL = "Add a past result"
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
    """'Salt: this rule could not be read.' — the ingredients grid has no
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
MAKE_YOUR_FIRST_BATCH_BUTTON = f"Make your first {ROUND}"
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
            f"printed {made_on}")
    return f"{line} · made to {total_text}" if total_text else line


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
            f"file you downloaded from this {ROUND}.")


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
FORMULA_LABEL = "Rule"
FORMULA_IN_RANGE = "rule"
# The word for a row filled in from its rule rather than typed by hand.
# One word, spelled once, for the range cell's own marker and every sheet
# that has to say the same thing about the same row. It is NOT the app's
# verb for "recomputed" — that is "recalculated" — so the two never meet
# on one screen meaning two things.
WORKED_OUT = "worked out"

# '= rest' on its own is the balance of the batch size once every other row
# is filled in. The word is spelled once here so the parser and its own
# refusal can never drift apart. On screen it is always "= rest": "the
# balance" is the instrument the bench weighs on.
REST_TOKEN = "rest"


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
    """'Water and Salt are worked out from their rules, so this limit
    cannot change them.' — every row of a limit filled in by a rule leaves
    the limit nothing to act on, and the app accepted it and then let the
    rules contradict it."""
    verb, tail, them = (("are", "their rules", "them") if many
                        else ("is", "its rule", "it"))
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
    """'Limits written as a % of the default batch size are now worked out
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
    f"Cannot delete the last {INGREDIENT} or setting that can still move. "
    f"Give another one a different {LOWEST_LABEL} and {HIGHEST_LABEL} "
    "first.")
AMOUNTS_MISSING_DELETE_ERROR = (
    "Cannot delete this: some formulations were recorded without their "
    "amounts, so what was made cannot be recalculated. Fix it at one "
    "amount instead, or start this project over.")

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
    "this cannot be added now. Start this project over, or add those "
    "formulations again with their amounts.")
CANNOT_RELOAD_INGREDIENTS = (
    "Ingredients cannot be reloaded after results have been recorded. Use "
    "Manage project › Start this project over, or open a saved copy.")
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
    f"{ROUND} sheets you downloaded.")
SHEET_HAS_NO_ROWS = "The sheet has no result rows."


def sheet_missing_columns(named):
    return f"Missing columns: {named}"


def formulation_number_not_whole(raw):
    return f"{FORMULATION_CAP} number {raw} is not a whole number."


def formulation_not_in_round(number, round_no, holds):
    return (f"{FORMULATION_CAP} {number} is not in {ROUND} {round_no} "
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
FORMULA_TWO_AMOUNTS = ("A rule can add or subtract amounts and multiply "
                       "by a number. It cannot multiply two amounts.")
FORMULA_DIVIDE_BY_AMOUNT = ("A rule can divide by a number, not by an "
                           "amount.")
FORMULA_DIVIDE_BY_ZERO = "A rule cannot divide by zero."
# Every word the grammar really takes, in both the alphabet the spec
# prints and the one on the keyboard. The old sentence listed four
# characters, three of them untypable, and omitted the leading '=', the
# percentage, 'batch size' and 'rest' — the four things a reader who is
# stuck most plausibly got wrong.
FORMULA_UNREADABLE = ("This rule could not be read. Use =, + − × ÷ (or "
                      "- * /), %, numbers, brackets, batch size, rest and "
                      "ingredient names.")
RULE_NEEDS_EQUALS = "Start a rule with =."
RULE_USES_ITS_OWN_ROW = "A rule cannot use its own row."
RULE_INGREDIENTS_ONLY_PREFIX = "A rule can use ingredients and batch size, not"


def rule_ingredients_only(name):
    """'A rule can use ingredients and batch size, not Cook temperature.' —
    a rule naming a process setting. Grams of salt worked out from minutes
    of cooking is arithmetic across two units that cannot be mixed, and the
    app refused a setting a rule of its own while allowing the reverse."""
    return f"{RULE_INGREDIENTS_ONLY_PREFIX} {name}."


def formula_unknown_name(name):
    """'There is no ingredient called Sodium citrate.' — a rule naming a
    row the project does not have."""
    return f"There is no ingredient called {name}."


def formula_loop(chain):
    """'A rule cannot lead back to itself: Fat → Water → Fat.' — `chain`
    is the arrow-joined names that already spell the loop out."""
    return f"A rule cannot lead back to itself: {chain}."


FORMULA_NEEDS_BATCH_SIZE = (
    f"There is no {FORMULATION_TOTAL_NOUN} to work this out from. Set one "
    "in More settings, or write the amounts instead.")
PERCENT_NEEDS_BATCH_SIZE = (
    f"There is no {FORMULATION_TOTAL_NOUN} to take a percent of. Set one "
    "in More settings.")


# ------------------------------------------------------------------ #
# Rule rows (0.5.0 wave 2, "rules", task 2): what the model owes the
# reader once a rule row is worked out from the others instead of being
# searched. Nothing here mentions the model: a row that cannot be worked
# out is refused in the amounts and the rule the reader typed.
# ------------------------------------------------------------------ #

def formula_below_zero(name, unit):
    """'Water is below 0 g in every formulation the allowed amounts reach.
    Widen an amount, or change its rule.' — a rule no allowed amounts can
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
    """'Water is worked out from Flour. Change Water's rule first.' —
    deleting a row another row's rule reads would leave that rule naming
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
    "Write what this ingredient is, in terms of the others: = batch size − "
    "Water − Salt makes this row whatever those two leave. Write = rest for the row that takes whatever is left. "
    "Leave it blank to give the row its own Lowest and Highest.")

# The one line under the grid while no row has a rule: the column arrived
# with no header tooltip anybody reads, no placeholder and no mention in
# the caption, and everything a cold reader learned about it they learned
# from refusals. It stands down the moment a rule exists — the worked-out
# captions take its place.
RULE_HINT = (f"To write a {FORMULA_IN_RANGE} for a row, type it in its "
             f"{FORMULA_LABEL} cell: = {BATCH_SIZE_NOUN} − Water, or "
             f"= {REST_TOKEN}.")


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
    """'Water is worked out as batch size − Pea protein − Salt: between
    40.00 and 62.00 g in a 100 g formulation.' — one line under the grid per
    row that is worked out rather than typed, so the rule shows its
    consequence in numbers.

    A row that takes the remainder says so in the words it was written in:
    'Water is worked out as = rest, whatever is left of the batch size:
    ...'. The amounts are what the other rows' allowed amounts leave it,
    and `size_text` is the default batch size they are read against — blank
    while the project has none, and the sentence then stops at the amounts.

    Both branches say `worked out`, the word in the two cells beside them,
    and both quote the cell WITH its '=', so the reader can match the line
    to what they typed. `rest` is the parser's own answer to "is this the
    rest row" — passed in rather than read off the text, so '=rest' and
    '= REST' say the same thing here as they do everywhere else.

    `outside_text` is the row's own Lowest and Highest when the rule takes
    it past them. They are dormant on a worked-out row — the rule decides
    the amount, not the range — but a rule that puts Salt at 16 g over a
    cap of 3 g was landing with nothing said at all.
    """
    text = str(formula_text).strip()
    if rest:
        head = (f"{name} is {WORKED_OUT} as = {REST_TOKEN}, whatever is "
                f"left of the {BATCH_SIZE_NOUN}")
    else:
        head = f"{name} is {WORKED_OUT} as {text}"
    # A rule that comes to one number says that number: "between 1.50 and
    # 1.50 g" asked the reader to read a range where nothing can vary.
    span = (f"{head}: {high_text}" if low_text == high_text.split(" ")[0]
            else f"{head}: between {low_text} and {high_text}")
    if size_text:
        span = f"{span} in a {size_text} {FORMULATION}"
    if outside_text:
        return (f"{span}. Its own {LOWEST_LABEL} and {HIGHEST_LABEL} "
                f"({outside_text}) do not apply while the rule does.")
    return f"{span}."


def rest_row_takes_the_difference(name, low_text, high_text, size_text):
    """'Water takes up the difference: between 152.00 and 220.00 g in a
    250 g formulation. To keep the same proportions, widen Lowest and
    Highest too.' — said under the Default batch size box the moment that
    number moves, for a project with a row written = rest.

    Amounts written in grams do not follow the batch size and that row
    does: a burger turns into soup one number at a time, and the only
    thing the app said about it was that a limit had been deleted."""
    return (f"{name} takes up the difference: between {low_text} and "
            f"{high_text} in a {size_text} {FORMULATION}. To keep the same "
            f"proportions, widen {LOWEST_LABEL} and {HIGHEST_LABEL} too.")


def formulations_keep_their_amounts(names_text, many=False):
    """'Formulations already made keep their amounts. Water is worked out
    from its formula from the next round on.' — a formula landing on a
    project that has results. What was weighed is what was weighed; the
    formula starts answering for the row from the next round."""
    if many:
        tail = (f"{names_text} are {WORKED_OUT} from their "
                f"{FORMULA_IN_RANGE}s")
    else:
        tail = f"{names_text} is {WORKED_OUT} from its {FORMULA_IN_RANGE}"
    return (f"{FORMULATION_CAP}s already made keep their amounts. "
            f"{tail} from the next {ROUND} on.")


# ------------------------------------------------------------------ #
# The workbook and the round table (0.5.0 wave 2, "rules", task 4): a
# worked-out row prints the amount it computed to, marked so a bench
# reading the page knows it was not chosen but left over, and the round
# table on tab 2 stays as it always was — read only.
# ------------------------------------------------------------------ #

def worked_out_label(name):
    """'Water · worked out' — the mark a worked-out row's own name wears on
    the summary sheet and on its own formulation page, so a bench reading
    the printed page knows this amount was not chosen, only computed."""
    return f"{name} · {WORKED_OUT}"


FORMULA_ROW_NOTE = (f"A row marked {WORKED_OUT} is filled in from its "
                    f"{FORMULA_IN_RANGE}. Weigh the amount printed.")


def worked_out_row_note(lines_text):
    """'Water is worked out: = rest. Weigh the amount printed.' — the
    summary sheet's note, naming each worked-out row's own rule. The rule
    is on no sheet of the round workbook, so a bench holding the page was
    told the amount came from one and had nowhere to see it."""
    return f"{lines_text} Weigh the amount printed."


def worked_out_row_rule(name, rule_text):
    """'Water is worked out: = rest.' — one row of the note above."""
    return f"{name} is {WORKED_OUT}: {rule_text}."


def setup_sheet_formula_text(text, rest=False):
    """'= rest (batch size − every other ingredient)' — what the row that
    takes the remainder prints in the Set-up sheet's Formula column; every
    other row prints exactly what it was typed as. `rest` comes from the
    parser, not from the spelling of the cell."""
    if rest:
        return f"= {REST_TOKEN} ({BATCH_SIZE_NOUN} − every other {INGREDIENT})"
    return text


# The round table on tab 2 gains no editing of its own: a worked-out row's
# amount is what it computed to, and correcting what the bench actually
# weighed happens after the round is recorded, on the Results tab.
# The greyed boxes at the foot of `Add a formulation to this round`, and
# the one line that says why they are last: they fill in from the boxes
# above, so they cannot be anywhere else, and the form was the only table
# in the app listing the rows in a different order with nothing said.
WORKED_OUT_BOXES_CAPTION = ("These are worked out from the amounts above, "
                            "so they come last.")

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
MADE_AS_LABEL = "Made as"
PREMIX_SHARE_LABEL = "% of pre-mix"
# The three answers to `Made as`, and the whole of what the column asks.
# Neither of the two pre-mix answers carries a comma any more: two sentences
# list both options, and "one pre-mix, portioned, or weighed into each
# formulation" reads as three things where the select offers two.
# `bought in` is the ordinary case and the default — the cell was blank, and
# a blank is not an answer a reader can recognise as the one they want.
PREMIX_MADE_AS_BOUGHT_IN = "bought in"
# What the portioned answer was called before the comma came out of it, in
# both the spellings a hand-typed file uses. An ingredients file written
# then still loads.
PREMIX_MADE_AS_PORTIONED_WAS = ("one pre-mix, portioned",
                                "one premix, portioned")
PREMIX_MADE_AS_PORTIONED = "portioned from one pre-mix"
PREMIX_MADE_AS_WEIGHED = "weighed into each formulation"
# How a pre-mix is named when something else found the clash, alongside
# AN_INGREDIENT and A_PROCESS_SETTING above.
A_PREMIX = "a pre-mix"

MADE_AS_REQUIRED_ERROR = (
    f"Say how this row is made: {PREMIX_MADE_AS_BOUGHT_IN}, "
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
            f"{PREMIX_MAKE_UP} stays the same for the whole {ROUND}, so you "
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
            f"before making a {ROUND}: {verb} at 0.00 now.")


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
        return (f"{name} is a pre-mix. Choose bought in in its Made as cell "
                "to take it apart.")
    return f"{name} is part of {premix}. Take it out of the pre-mix instead."


def part_in_two_weighed_premixes(name, first, second):
    """'Oil is already weighed into each formulation as part of Wet
    blend, so it cannot also be part of Fry blend.' — weighed, a part IS
    a row, and one row standing for two lots of mass is counted twice
    everywhere it is read."""
    return (f"{name} is already {PREMIX_MADE_AS_WEIGHED} as part of "
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
    f"{PREMIX_MADE_AS_BOUGHT_IN} — you weigh it straight in, and the "
    f"suggestions move that one amount. "
    f"{PREMIX_MADE_AS_PORTIONED} — one lot made for the whole {ROUND}, and "
    f"the suggestions move how much of it goes in. "
    f"{PREMIX_MADE_AS_WEIGHED} — the parts go in one by one, and the "
    "suggestions move each of them. The parts of either open in a fold "
    "under the grid.")
# The first column of a pre-mix's own grid. The row IS the part, so the
# header is the noun and not "Name": the grid above already has a Name.
PART_LABEL = "Part"
# What a weighed pre-mix's own row says where its allowed amounts would be.
# It has none: the pre-mix is not a row of the list at all, its parts are,
# and the amount of it in a formulation is whatever they add up to.
SUM_OF_ITS_PARTS = "sum of its parts"
PARTS_ADD_TO_NOTHING = ("The parts add up to nothing. Give at least one of "
                        "them a % of pre-mix above zero.")
PREMIX_NEEDS_A_PART = "Enter at least one name in the Part column before saving."


def premix_fold_caption(weighed):
    """The one line at the top of a pre-mix's parts fold: what the numbers
    in it are, and what may be a part.

    `% of pre-mix` had no tooltip, no caption and no line under the grid —
    the one term on the tab that was never said — and it is the central
    number of a portioned pre-mix. The second sentence answers the other
    thing nothing said: an ingredient already on the grid can be a part of
    a pre-mix too.
    """
    if weighed:
        return (f"Parts weighed into each {FORMULATION}, each with its own "
                f"{LOWEST_LABEL} and {HIGHEST_LABEL}. An ingredient already "
                "on the grid can be a part too.")
    return (f"Parts and their {PREMIX_SHARE_LABEL}, adding up to 100 %. An "
            "ingredient already on the grid can be a part too.")


def premix_grid_title(name):
    """'Dry blend · parts' — the fold under the ingredients grid, named for
    the pre-mix it belongs to and for what is inside it. The same separator
    every other two-part title in the app uses."""
    return f"{name} · parts"


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
    line = f"{name} will not be a pre-mix any more."
    if parts_text:
        line += f" {parts_text} {'go' if many else 'goes'} with it."
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
HAVE_ON_HAND_CAPTION = ("Across every formulation; a worked-out row's total "
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
PREMIX_LOT_ON_ITS_PAGE = "see its page"


def premix_write_in_note(actual_head=None, lot=True):
    """The pre-mix page's own line, naming that page's own write-in cells
    and no others."""
    columns = ([actual_head] if actual_head else []) + ([LOT_COLUMN] if lot
                                                        else [])
    if lot:
        columns.append(PREMIX_LOT_LABEL)
    columns += [PREMIX_BLENDED_BY_LABEL, PREMIX_BLENDED_ON_LABEL, PREMIX_BLEND_TIME_LABEL]
    return ("% of pre-mix is the percentage of this pre-mix, not of the formulation. "
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
    line = f"{name} · make {total_text}"
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
    — and, on Set up itself, 'in the fold below', because the tab it sent
    the reader to is the tab they are standing on."""
    where = "in the fold below" if here else f"in {SET_UP_SHEET}"
    return f"Add at least one {PART_LABEL.lower()} to {name} {where} before making a {ROUND}."


RESULT_DRAFT_SAVED = "Measurements and notes are saved as you type."
SAVE_RESULTS_HELP = "Records the formulations you have filled in; the rest stay to record."
UPLOAD_AMOUNTS_HEADING = "Amounts from the file"
UPLOAD_LOTS_HEADING = "Lot numbers from the file"


BENCH_RECORDS_SHEET = "Bench records"
BENCH_RECORD_COLUMNS = {'sheet': "Sheet", 'cell': "Cell", 'label': "Entry", 'value': "Written"}
BENCH_RECORDS_CAPTION = (
    "Filled-in boxed cells are saved with this round and included in All formulations. "
    "Preparation amounts and notes are kept as bench records; they do not change "
    "a pre-mix's % of pre-mix for future rounds."
)


def premix_page_pointer(name):
    return f"{name} is a pre-mix: see its sheet."


def premix_members_line(name, names):
    return f"{name}: {names}. These parts are weighed into each formulation."

PART_AMOUNTS_REQUIRED = "Enter a number in both Lowest and Highest for this part."
