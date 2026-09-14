"""Every word the user reads. Change a word here and it changes everywhere;
the tests read this module too."""

# ------------------------------------------------------------------ #
# Concepts. The set of formulations issued together is called a "batch"
# on screen — the word the owner's formulation team already uses for a
# round of formulations. Every batch-naming sentence below is built from
# these two constants, so a name change is a one-line edit.
# ------------------------------------------------------------------ #
BATCH = "batch"
BATCH_CAP = "Batch"
FORMULATION = "formulation"
FORMULATION_CAP = "Formulation"

# Plain nouns handed to ui_helpers.plural(n, word) at more than one call
# site, so the word is spelled once.
INGREDIENT = "ingredient"
PROCESS_SETTING = "process setting"
LIMIT = "limit"
ROW = "row"

# "Note" is both a table column header (food_bo's own dataframes carry the
# same header) and a form field label; one spelling serves both.
NOTE = "Note"


# ------------------------------------------------------------------ #
# Shell: the page title, the three tabs, the sentence every irreversible
# action's confirmation ends with, the project-name rule.
# ------------------------------------------------------------------ #
APP_TITLE = "Food Optimizer"

# The three tabs, in loop order. The separator is U+00B7 MIDDLE DOT.
TAB_SETUP = "1 · Set up"
TAB_BATCH = f"2 · Make a {BATCH}"
TAB_RESULTS = "3 · Results"

# Archived copies are written beside the project's own file, which on the
# desktop app is the FoodOptimizer folder. Every tab and the sidebar use it,
# so the sentence exists once.
COPY_KEPT = ("A copy is saved first. Restore from backup can bring it "
             "back.")

NAME_RULE = "Up to 64 characters. Start with a letter or a number."


# ------------------------------------------------------------------ #
# ui_helpers: generic building blocks used across every tab.
# ------------------------------------------------------------------ #
YES_CONTINUE = "Yes, continue"
CANCEL = "Cancel"

NEED_A_VARIABLE = "Add at least one ingredient or process setting."
NEED_A_MEASUREMENT = "Add at least one measurement."

# scale_error's and bounds_warning's arguments to outside_message: what a
# value is outside of, and the hint scale_error appends that bounds_warning
# does not.
YOUR_RANGE = "your range"
ALLOWED_AMOUNTS = "its allowed amounts"
# The half of the scaled caution that names what was exceeded. One spelling,
# whether the line lists the ingredients or counts them.
AMOUNTS_YOU_ALLOWED = "the amounts you allowed"
WIDEN_RANGE_HINT = " Widen the range in Set up, or check the value."


SMALLER_TOTAL_HINT = "Print at a smaller total, or widen them in Set up."


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
# Sidebar: projects, backup and restore, manage project.
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
    "A ready-made plant-based burger: eight ingredients, and panel scores "
    "for juiciness and firmness. Try it before setting up your own."
)

PROJECT_LOAD_ERROR_SIDEBAR_NOTE = ("This project could not be opened. The "
                                   "main screen says why.")

BACKUP_UNAVAILABLE = ("Backup download is unavailable while the project "
                      "file cannot be read.")
DOWNLOAD_PROJECT_BACKUP = "Download project backup"

RESTORE_FROM_BACKUP = "Restore from backup"
RESTORE_CAPTION = ("Choose a project backup (.json) you downloaded "
                   "earlier. Your current project is copied first.")
CHECK_THIS_BACKUP = "Check this backup"
BACKUP_UNREADABLE = (
    "This file could not be read as a Food Optimizer backup. "
    "If you have another copy, try that one; recent copies are "
    "saved in your FoodOptimizer folder."
)
RECENT_COPIES_HINT = (" Recent copies of your own projects are saved in "
                      "your FoodOptimizer folder.")


def restore_backup_warning(name, holds, project_name, held):
    """`holds` is the list of 'N formulations'/'N ingredients'/'N process
    settings' phrases the backup contains; `held` is the same phrase for
    what replacing it would give up."""
    return (f"This backup holds **{name}**: " + ", ".join(holds)
            + f". Replace **{project_name}**, which has {held}? " + COPY_KEPT)


YES_REPLACE = "Yes, replace"
BACKUP_APPLY_FAILED = ("This backup could not be applied. Your current "
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
        "1. **Name a project** and click Create project.\n"
        "2. **Add ingredients** and the measurements you will record.\n"
        f"3. **Make a {BATCH}**, weigh out the formulations, and record "
        "what you measured."
    )


DOWNLOAD_CSV_TEMPLATE = "Download CSV template"


# ------------------------------------------------------------------ #
# Load-error / save-error banners, and the discarded-batch notice.
# ------------------------------------------------------------------ #
def project_load_error_info():
    return (
        "This project file is damaged, so editing is off. Two ways out, both "
        f"in the sidebar: {RESTORE_FROM_BACKUP}, if you downloaded one. Or "
        f"{MANAGE_PROJECT} › {START_OVER_LABEL} — the damaged file is copied "
        "first."
    )


SAVE_ERROR_WARNING = ("**Your last change was not saved.** Download a "
                      "backup now, then click Reload project.")
DOWNLOAD_BACKUP = "Download backup"
RELOAD_PROJECT = "Reload project"
PROJECT_RELOADED = "Project reloaded from the latest saved copy."


def batch_discarded_notice(no=None):
    """'Batch 2 was discarded: your set-up changed after it was made.
    Generate a new one.'

    Three ways lead here — the ingredient list, a paused ingredient or
    setting, the allowed amounts — and naming all three in one subordinate
    clause was unreadable on one pass and still never said what to do. "Your
    set-up" is the tab all three live on.
    """
    # Every screen that flashes this knows the number. The unnumbered form
    # is the honest fallback for a batch whose number did not survive the
    # read that discarded it, and nothing reaches it today.
    who = f"{BATCH_CAP} {no}" if no is not None else f"The open {BATCH}"
    return (f"{who} was discarded: your set-up changed after it was made. "
            "Generate a new one.")


def project_created(name):
    return f"Created {name}."


def project_opened(name):
    return f"Opened {name}."


def sample_project_failed(err):
    return f"The sample project could not be created: {err}"


def batch_line_open(no, n):
    """The one line under the title on tabs 1 and 2 once a batch exists and
    still has rows to make."""
    return f"{BATCH_CAP} {no} · {n} to make"


def batch_line_recorded(no):
    """Same line, once every row of that batch has been recorded (or left
    out)."""
    return f"{BATCH_CAP} {no} · recorded"


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
    return (f"{BATCH_CAP} {no} is ready to make. Print the sheets, then "
            "record the results below when you have them.")


FORMULATIONS_TO_GENERATE = "Formulations to generate"


# The expander under the Generate row (and under the batch table once one is
# open): a formulation the scientist chose, added to the batch beside the
# generated ones. It replaced the Repeat checkbox, which could only ever
# repeat the best.
ADD_OWN_EXPANDER = f"Add a {FORMULATION} of your own"
ADD_OWN_NO_BATCH_CAPTION = ("Want the app's suggestions too? Click "
                            "Generate first, then add yours.")
OWN_NOTE_PLACEHOLDER = "e.g. Repeat of 4 with more salt"
START_FROM_BEST = "Start from the best so far"
ADD_TO_THIS_BATCH = f"Add to this {BATCH}"
ENTER_EVERY_AMOUNT = ("Enter every amount. Type 0 for an ingredient you "
                      "are leaving out.")
# What the row says when the user typed no reason of their own. The note is
# part of the record, so a row on the batch table is never blank about what
# it is.
OWN_FORMULATION_NOTE = f"Own {FORMULATION}"


def own_formulation_added(no, batch_no):
    return f"{FORMULATION_CAP} {no} added to {BATCH_CAP} {batch_no}."


def generate_button_label(n):
    return f"Generate {n} formulations"


# The one line that says how formulations are chosen. It is the third How it
# works bullet and, word for word, the caption under Generate — before and
# after the fifth formulation alike. Two captions for the two halves of one
# rule made a reader who had seen only one of them think there were two.
HOW_CHOSEN = ("Until five formulations have results, new ones are spread out "
              f"to learn the space. After that, each {BATCH} aims closer to "
              "your targets.")


# The open batch reads as the three steps of the work, each headed with its
# number. Step 1 takes no count of its own: the title directly above it
# already says how many formulations there are.
STEP_MAKE_HEADING = f"##### 1 · Make the {FORMULATION}s"
STEP_PRINT_HEADING = "##### 2 · Print the sheets"
STEP_RECORD_HEADING = "##### 3 · Record the results"


def make_these(no, n):
    """'Batch 1 · make this 1 formulation' / '... make these 3
    formulations'."""
    word = FORMULATION if n == 1 else FORMULATION + "s"
    return (f"**{BATCH_CAP} {no} · make "
            f"{'this' if n == 1 else 'these'} {n} {word}**")


# Every row in the batch is one the user added by hand, so there is no
# Generate control on the screen and nothing else would say why.
ONLY_OWN_FORMULATIONS_CAPTION = (
    "This batch holds only your own formulations. For the app's suggestions "
    f"too, click Generate a different {BATCH}, then Generate."
)


def batch_discarded_caption(numbers_text, many=True):
    """The numbers the regenerate retired. Where the new batch picked up is
    on the table directly below, so the line does not say it twice."""
    word = f"{FORMULATION_CAP}s" if many else FORMULATION_CAP
    verb = "were" if many else "was"
    return f"{word} {numbers_text} {verb} discarded."


def biggest_changes_caption(row_no, best_no, parts):
    return (f"Biggest changes in {FORMULATION_CAP} {row_no} from "
            f"{FORMULATION_CAP} {best_no}: {parts}.")


NEEDS_ONE_UNIT = ("To make each formulation to a set amount, every "
                  "ingredient needs the same unit.")


def batch_total_label(unit):
    """The box that says how big to make each formulation. It asks the
    question the bench asks — how much do I make? — rather than naming a
    quantity ("Formulation total") the reader then has to decode."""
    return (f"Make each formulation to ({unit})" if unit
            else "Make each formulation to")


BATCH_TOTAL_HELP = (
    "How much of each formulation to make. Leave blank to use the amounts in "
    "the table."
)
# An example, not a description: a blank box means "the amounts in the table",
# and the help says so once.
BATCH_TOTAL_PLACEHOLDER = "e.g. 150"

SHEET_GOAL_SEPARATOR = ", "
TOTAL_PREFIX = "Total: "
# The label above the ruled area a technician writes the note in. Nothing on
# a sheet is a run of typed underscores any more: the rules are drawn in CSS,
# so they stay straight however long the label beside them is.
NOTE_SHEET_LABEL = f"{NOTE}:"


def note_line(note):
    return f"{NOTE}: {note}"


NOT_SCORED_CHECKBOX_SHEET = "Not scored [  ]"

DOWNLOAD_BENCH_SHEET = "Download the batch to fill in (CSV)"
DOWNLOAD_FORMULATION_SHEETS = "Download one sheet per formulation (to print)"
PREVIEW_SHEETS = "Preview the printed sheets"


def sheets_show_total_caption(total_text):
    """'Sheets show each formulation made to 150 g.' — the one line on the
    tab that names the number the files were written for."""
    return f"Sheets show each {FORMULATION} made to {total_text}."


GENERATE_DIFFERENT_BATCH = f"Generate a different {BATCH}"


def regenerate_warning(no, numbers_text, next_no, many=True):
    """'Discard Batch 1 and Formulations 1, 2 and 3? New formulations start
    at Formulation 4.'

    Every other confirmation in the app asks a question, and "those numbers
    will not be used again" is a release note: what the reader can act on is
    where the numbering picks up. Not "the next batch": a regenerate keeps
    this batch's own number, so only the formulation numbers move on.
    """
    word = f"{FORMULATION_CAP}s" if many else FORMULATION_CAP
    return (f"Discard {BATCH_CAP} {no} and {word} {numbers_text}? New "
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
                          "a panel scored it. Leave blank if it was not "
                          "measured.")


def formulation_heading(no):
    return f"**{FORMULATION_CAP} {no}**"


# A formulation with no result yet: never made, or made and never measured.
# It can be scored later from the Results tab.
NOT_SCORED = "Not scored"
NOT_SCORED_HELP = f"Say why in {NOTE}. It stays with the formulation."

NOTHING_TO_SAVE = "Nothing to save — at least one formulation needs results."
# Said by food_bo when a row arrives with nothing on it, and by the Results
# tab when a not-scored formulation is saved with every box empty. One
# sentence, so the refusal reads the same wherever the row was typed.
ENTER_A_MEASUREMENT = "Enter a value for at least one measurement."
SAVE_RESULTS = "Save results"


def complete_counter(complete_n, kept_n):
    """'1 of 2 complete' — a row is complete when EVERY measurement on it has
    a number. One of three typed is not a third of a result."""
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


def batch_recorded_flash(no):
    return f"{BATCH_CAP} {no} recorded."


UPLOAD_EXPANDER = "Or upload results from a CSV"
UPLOAD_HELP_CAPTION = (
    "Fill in the CSV you downloaded above — one column per measurement — "
    "and upload it here. Rows are matched on Formulation number."
)
UPLOAD_RESULTS_CSV = "Upload results CSV"
CHECK_THIS_FILE = "Check this file"
CSV_UNREADABLE = (
    "This file could not be read as a CSV. If it came "
    "from Excel, use File > Save As and pick CSV format."
)


def upload_found_caption(found_n, total_n, names_text):
    return (f"Found results for {found_n} of {total_n} formulations: "
            f"{names_text}.")


SAVE_UPLOADED_RESULTS = "Save uploaded results"


def upload_partial_flash(parsed_n, total_n, batch_no, left_n):
    return (f"Recorded {parsed_n} of {total_n} formulations in "
            f"{BATCH_CAP} {batch_no} · {left_n} to make.")


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


LIMIT_KEPT = (f"Formulations already made are kept. The next {BATCH} will "
             "respect this limit.")
NO_FORMULATION_FITS_LIMIT = "No formulation you have made fits this limit."


def unscaled_tail(batch_no, total_text):
    """Scaling needs one unit, and a unit change may have just taken it away:
    the open batch is back to as-generated, and only this sentence says so."""
    return (f"{BATCH_CAP} {batch_no} is back to its own amounts: your "
            "ingredients no longer share one unit.")

# The one collapsed expander that maps the words on this tab to the words a
# specialist would use. Every optimization term the app otherwise refuses to
# say — variable, objective, weight, constraint — is said here and only here,
# one line each, so the mapping exists exactly once. The vocabulary guard
# reads this list by name and allows what is in it.
HOW_IT_WORKS = [
    "Ingredients and settings are what the model varies. Measurements "
    "and goals are what it aims for.",
    "Importance says how much each measurement counts. Closeness is a 0 to 1 "
    "score for how near a result is to its goal.",
    HOW_CHOSEN,
    "Limits are hard rules the model never breaks.",
]

# The fold directly under it, for the reader who wants the arithmetic. The
# four lines above raise the question — what is closeness, exactly? — and
# this answers it; nine bullets in one fold answered it before anyone asked.
HOW_CLOSENESS_EXPANDER = "How closeness is calculated"
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
    "The model learns the one overall score, so changing an importance, a "
    "goal or a range re-scores every past formulation.",
    "A formulation of your own counts like any other. Making the best one "
    "again teaches the model how noisy your measurements are.",
]

VARIABLES_HEADER = "Ingredients and process settings"


def made_before_units_caption(unit):
    return (f"These amounts were recorded before units existed, so they "
            f"are read as {unit}. Set the right unit below.")


UPLOAD_INGREDIENTS_EXPANDER = "Or upload an ingredients CSV"

NAME_LABEL = "Name"
SETTING_NAME_PLACEHOLDER = "e.g. Cook temperature"
INGREDIENT_NAME_PLACEHOLDER = "e.g. Water"
TYPE_LABEL = "Type"
VARIABLE_TYPE_HELP = ("Ingredients are weighed into the formulation and "
                      "count towards its total. Process settings, such as "
                      "temperature or time, are set on the equipment.")
LOWEST_LABEL = "Lowest"
HIGHEST_LABEL = "Highest"
NEW_INGREDIENT_FIXED_LOW_HELP = ("Fixed at 0: every formulation you have "
                                 "already made contains none of it.")
UNIT_LABEL = "Unit"
INGREDIENT_UNIT_PLACEHOLDER = "e.g. g"
SETTING_UNIT_PLACEHOLDER = "e.g. °C"
BASELINE_LABEL = "Baseline"
BASELINE_ADD_LABEL = f"{BASELINE_LABEL} (required)"
BASELINE_PLACEHOLDER = "e.g. 180"
BASELINE_HELP = ("The setting you used for every formulation already made, "
                 "so those results still count.")
ADD_VARIABLE_BUTTON = "Add ingredient or setting"
# An example, not a rule: "0 if blank" and "leave a box empty for no
# value" sat on one screen contradicting each other, and the app never knows
# an ingredient is fat-free — only that a box was left empty.
PROPERTY_PLACEHOLDER = "e.g. 2"
PROPERTY_BLANK_RULE = "An empty box counts as 0 in any limit."
PROPERTY_BOX_HELP = "Per 100 g of this ingredient. " + PROPERTY_BLANK_RULE
ADD_BASELINE_ERROR = ("Enter the baseline: the setting you used for every "
                      "formulation already made.")


def added(name):
    """Subject first, like every other flash on the tab: "Firmness updated.",
    "Batch 1 recorded." One shape for all of them."""
    return f"{name} added."


STATUS_LABEL = "Status"
ACTIVE_STATUS = "active"


def paused_status(held_text):
    return f"paused · held at {held_text}"


INGREDIENT_OR_SETTING_LABEL = "Ingredient or process setting"
NEW_UNIT_LABEL = "New unit"
SET_UNIT_BUTTON = "Set unit"


# A project's property names are its own and can be long ("Sodium mg per
# 100 g"), so the button stays the short, stable label and the dialog it
# opens does the naming.
SET_PROPERTIES_BUTTON = "Set properties"


ONLY_INGREDIENT_HAS_PROPERTIES = "Only an ingredient has properties."


def properties_for_caption(names_text, name, per_100_already_said=False):
    """'Fat and sodium in Pea protein isolate, per 100 g. An empty box
    counts as 0 in any limit.'

    `per_100_already_said` drops the basis from the sentence: a project whose
    property names carry it themselves ("Fat per 100 g and Sodium per 100 g")
    would otherwise say it three times in one line.
    """
    basis = "" if per_100_already_said else ", per 100 g"
    return f"{names_text} in {name}{basis}. " + PROPERTY_BLANK_RULE


SAVE_BUTTON = "Save"
CLOSE_BUTTON = "Close"


def properties_saved(names_text, name):
    return f"Saved {names_text} for {name}."


RESUME_BUTTON = "Resume"
RESUME_HELP = "Use this ingredient in new formulations again."
PAUSE_BUTTON = "Pause"
PAUSE_DISABLED_HELP = ("At least two ingredients or settings must stay "
                       "active before one can be paused.")
PAUSE_HELP = ("New formulations will not use this ingredient. Results "
             "already recorded are kept.")


def resumed(name):
    return f"{name} resumed."


def paused(name):
    return f"{name} paused."


UNIT_REQUIRED_ERROR = "A unit is required; use g if the amount is a mass."


def unit_changed(name, written, is_ingredient):
    """Nothing is converted and nothing is rescored, and only this sentence
    says so — so it asks for the one thing the reader can do about it."""
    if not written:
        return f"{name} is shown without a unit."
    held = "amounts" if is_ingredient else "numbers"
    return (f"{name} is now in {written}. The {held} you already recorded "
            "were not converted — check them.")


YES_DELETE = "Yes, delete"


def delete_button(name):
    return f"Delete {name}"


def delete_variable_warning(name, is_ingredient):
    head = (f"Delete {name} from this project permanently? " if is_ingredient
            else f"Delete {name}? ")
    return (head + "Formulations you already recorded keep their amounts. "
            + COPY_KEPT)


DELETE_VS_PAUSE_CAPTION = ("Deleting takes it out of every formulation "
                           "already made; pausing keeps the data.")
DELETE_EVEN_IF_USED_CHECKBOX = ("Delete even though formulations used it "
                                "— those amounts go too")


def deleted(name):
    return f"{name} deleted."


REPLACE_INGREDIENTS_BUTTON = "Replace ingredients"
LOAD_INGREDIENTS_BUTTON = "Load ingredients"
INGREDIENTS_CSV_CAPTION = ("A CSV with the columns Name, Lowest, Highest "
                           "and, optionally, Unit. Extra columns become "
                           "properties you can set limits on.")
UPLOAD_INGREDIENTS_CSV_LABEL = "Upload ingredients CSV"


def blank_unit_cell_help(unit):
    return (f"Leave the {UNIT_LABEL} column blank and the app reads it "
            f"as {unit}.")


CSV_UNREADABLE_RETRY = ("This file could not be read as a CSV. If it came "
                        "from Excel, use File > Save As and pick CSV "
                        "format, then try again.")
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

MEASUREMENT_NAME_PLACEHOLDER = "e.g. Firmness"
MEASUREMENT_UNIT_PLACEHOLDER = "e.g. N or /10"
GOAL_LABEL = "Goal"
TARGET_LABEL = "Target"
RANGE_HEADING = "**Range**"
LOWEST_MEASURABLE_LABEL = "Lowest measurable"
HIGHEST_MEASURABLE_LABEL = "Highest measurable"
RANGE_HINT_CAPTION = "The ends of your range, not the numbers you expect."
IMPORTANCE_LABEL = "Importance"
IMPORTANCE_HELP = "Any positive number. 2 counts twice as much as 1."
MEASUREMENT_EXISTS_ERROR = ("That measurement already exists. Use Edit on "
                            "its row to change it.")


def name_differs_only_by_case(stored):
    """Two rows whose names differ only in capitals are two rows with one
    name on every table in the app, and the CSV importer matches columns
    without regard to case, so the second could never be filled in."""
    return (f"{stored} already exists. Use that spelling to change it, or "
            "choose another name.")
ADD_MEASUREMENT_BUTTON = "Add measurement"
SAVE_CHANGES_BUTTON = "Save changes"

RECALCULATED_SUFFIX = " Every overall score was recalculated."


def importance_changed(name, value):
    return f"{name} importance is now {float(value):.1f}."


def updated(name):
    return f"{name} updated."


def measurement_deleted(name):
    return f"{name} deleted."


MEASUREMENTS_HEADER = "Measurements and targets"
ADD_A_MEASUREMENT_EXPANDER = "Add a measurement"
MEASUREMENT_COLUMN = "Measurement"
RANGE_COLUMN = "Range"
COL_SHARE = "Share of score"


def edit_button(name):
    return f"Edit {name}"


def edit_measurement_heading(name):
    """The open editor's own title. Without it the form was six unlabelled
    boxes with a greyed Name at the top, and nothing said which measurement
    Save changes would change."""
    return f"##### Edit {name}"


def delete_measurement_warning(name):
    return (f"Delete {name}? Every overall score is recalculated without "
            "it. " + COPY_KEPT)


HOW_IT_WORKS_EXPANDER = "How it works"

ADD_PROPERTY_LABEL = "Add a property"
ADD_PROPERTY_PLACEHOLDER = "e.g. Sodium mg per 100 g"
ADD_PROPERTY_BUTTON = "Add property"


def property_added(name):
    return (f"{name} added. Set it for each ingredient in "
            f"{VARIABLES_HEADER}.")


def delete_property_warning(name, limits_text):
    head = (f"Delete {name} and its {limits_text}? " if limits_text
            else f"Delete {name}? ")
    return head + "Each ingredient's figure for it goes too. " + COPY_KEPT


def property_deleted(name, gone_text=""):
    return f"{name} deleted.{gone_text}"


def limit_went_with_it(limits_text):
    return f" Its {limits_text} went with it."


FINISHED_PRODUCT_LIMIT_HEADING = "**Finished-product limit**"
PER_100G_UNRESOLVED_CAPTION = ("Per 100 g of formulation once every "
                               "ingredient is in one mass unit.")


def per_100_caption(unit):
    return (f"Per 100 {unit} of formulation, from the properties of your "
            "ingredients.")


INGREDIENT_PROPERTY_LABEL = "Ingredient property"
AT_LEAST_LABEL = "At least"
AT_MOST_LABEL = "At most"
NO_LIMIT_PLACEHOLDER = "no limit"
ADD_PROPERTY_LIMIT_BUTTON = "Add property limit"
# The two limit forms ask for At least and At most, so their refusals name
# those two boxes. (The ingredient form's own Lowest/Highest keeps its own.)
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


LIMITS_EXPANDER = "Limits (optional)"
# The property rule lives here, not in the closeness fold: a property never
# touches closeness — it feeds limits and nothing else.
LIMITS_CAPTION = ("Limits hold every new formulation to an amount you weigh "
                  "out or a property of your ingredients. Measurements have "
                  "goals and targets instead. An ingredient with no figure "
                  "for a property counts as 0 in any limit on it.")


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
    return (f"Delete the {LIMIT} on {who}? The next {BATCH} is no longer "
            f"held to it. " + COPY_KEPT)


def limit_deleted(who):
    return f"Limit on {who} deleted. The next {BATCH} is no longer held to it."


LIMIT_ON_CHOSEN_INGREDIENTS_HEADING = "**Limit on chosen ingredients**"
INGREDIENTS_TO_LIMIT_LABEL = "Ingredients to limit together"
ADD_INGREDIENT_LIMIT_BUTTON = "Add ingredient limit"

HOW_FORMULATIONS_CHOSEN_EXPANDER = "How formulations are chosen (advanced)"
STANDARD_VS_EXPERT_CAPTION = ("Standard uses tested defaults and fits most "
                              "projects. Expert-selected lets a specialist "
                              "choose the model's kernel, prior, noise "
                              "handling and acquisition. These are fixed for "
                              "the life of the project.")
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

NEXT_MAKE_BATCH_BUTTON = f"Next: make a {BATCH}"


# ------------------------------------------------------------------ #
# Tab 3 · Results: the best formulation, every formulation, corrections
# (ui_results.py).
# ------------------------------------------------------------------ #
PARTIAL_SCORES_CAPTION = ("A formulation missing a measurement scores it as "
                          "zero, so its overall score is low. Record the "
                          "missing number to fix it.")


def batch_recorded_progress(no, before, now):
    return (f"{BATCH_CAP} {no} recorded · best improved "
            f"{before:.2f} → {now:.2f}")


def batch_recorded_no_improvement(no):
    return f"{BATCH_CAP} {no} recorded · best score unchanged."


def best_so_far_heading(no, batch_no=None):
    heading = f"Best so far: {FORMULATION_CAP} {no}"
    if batch_no is not None:
        heading += f" ({BATCH_CAP} {batch_no})"
    return heading


MEASURED_COLUMN = "Measured"
OFF_BY_COLUMN = "Off by"
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
    re-scores every formulation exactly as an importance or a range does, so
    it is named here with them."""
    return (f"Overall score {score:.2f} of {ceiling:.2f}"
            + (not_measured_tail(missing_text) if missing_text else "")
            + ". Scores only compare within this project. Change an "
            "importance, a goal or a range and every score is worked out "
            "again.")


ALL_FORMULATIONS_HEADING = "**All formulations**"
SORT_LABEL = "Sort"
# The three options, in the order the box offers them. Each is also matched
# exactly in food_bo.history_frame(order=...) — a protocol between that
# module and this one, not display prose that happens to repeat — so both
# read them from here and the two can never drift apart.
SORT_BEST_FIRST = "Best first"
SORT_NEWEST_FIRST = "Newest first"
SORT_BATCH_ORDER = f"{BATCH_CAP} order"
SORT_OPTIONS = [SORT_BEST_FIRST, SORT_NEWEST_FIRST, SORT_BATCH_ORDER]
SHOW_AMOUNTS_TOGGLE = "Show amounts"
DOWNLOAD_ALL_FORMULATIONS_BUTTON = "Download all formulations (CSV)"
DOWNLOAD_ALL_FORMULATIONS_HELP = ("One row per formulation, with the same "
                                  "units the screen shows. Formulations "
                                  "marked not scored are included, with "
                                  "their measurements blank.")

# ---------------------------------------------------------------- #
# `Edit past formulations`: one collapsed section for every way the
# record is fixed after the fact. It replaced three separate controls
# — a correction picker, a delete section and an import expander —
# the last of which hid the only way to enter work done before the
# project existed.
# ---------------------------------------------------------------- #
EDIT_PAST_FORMULATIONS_EXPANDER = f"Edit past {FORMULATION}s"

CORRECT_A_FORMULATION_HEADING = f"##### Correct a {FORMULATION}"
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
    return f"Back to {BATCH_CAP} {no} · {n} to record"


START_NEXT_BATCH = f"Start the next {BATCH}"

PROGRESS_CHART_EXPANDER = "Progress chart"
NO_RESULTS_YET = "No results yet."
OVERALL_SCORE_COLUMN = "Overall score"
BEST_SO_FAR_COLUMN = "Best so far"
PROGRESS_CHART_CAPTION = ("The top line only rises. A few flat "
                          f"{BATCH}es are normal; a long flat stretch "
                          "suggests this ingredient list is close to the "
                          "best it can do.")

def batch_open_record_first_caption():
    """food_bo.undo_last_batch's refusal. No screen reaches it any more —
    the Delete the last batch button retired with this section — but the
    method stays for its tests and it still has to speak English."""
    return f"Record or discard the open {BATCH} first."


DELETE_FORMULATIONS_HEADING = f"##### Delete {FORMULATION}s"
FORMULATIONS_TO_DELETE_LABEL = f"{FORMULATION_CAP}s to delete"
CHOOSE_MANY_PLACEHOLDER = "Choose one or more"
# The quick pick beside the list: one batch's formulations, recorded and not
# made alike, dropped into the selection to be looked over before deleting.
WHOLE_BATCH_LABEL = f"Add a whole {BATCH} to the list"
CHOOSE_A_BATCH_PLACEHOLDER = f"Choose a {BATCH}"


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
ADD_PAST_FORMULATION_LABEL = f"Record a {FORMULATION} you already made"
ADD_PAST_FORMULATION_HEADING = "##### " + ADD_PAST_FORMULATION_LABEL
TYPE_IT_IN = "Type it in"
UPLOAD_A_CSV = "Upload a CSV"
# What a formulation made before this project existed is noted as, and what
# the Note box opens holding. food_bo.import_formulation defaults to the same
# word, so a row typed in and a row read off a CSV read alike.
IMPORTED_NOTE = "Made earlier"
ADD_THIS_FORMULATION = f"Record this {FORMULATION}"


def formulation_added(no):
    return f"{FORMULATION_CAP} {no} added."


EXTRA_COLUMNS_IGNORED = " Extra columns are ignored."


def import_columns_caption(names_text):
    return (f"One row per {FORMULATION} you already made. The columns must "
            f"match these names exactly: {names_text}."
            + EXTRA_COLUMNS_IGNORED)


def import_columns_caption_empty():
    return (f"One row per {FORMULATION} you already made. Add ingredients "
            "and measurements first; the columns must match their names "
            "exactly.")


UPLOAD_FORMULATIONS_CSV_LABEL = f"Upload {FORMULATION}s CSV"


def missing_columns(names_text):
    return f"Missing columns: {names_text}"


def blank_amount_columns(names_text):
    return f"These amount columns have blank cells: {names_text}"


IMPORT_ALL_ROWS_BUTTON = "Import all rows"


def row_error(position, problem):
    return f"Row {position}: {problem}"


def stopped_at_row(row_no, failure):
    return f"Stopped at row {row_no}: {failure}"


def rows_before_saved(rows_text):
    return f" The {rows_text} before it were imported and saved."


def imported(text):
    return f"Imported {text}."


def rows_with_nothing_measured(rows_text, many):
    """The tail on the import flash. A not-scored formulation is in the
    downloaded file — it has a number, its amounts and its note — but it has
    no result to teach the model, so it is left where it is rather than
    stopping the whole import."""
    return (f" {rows_text} had no measurements, so they were not imported."
            if many
            else f" {rows_text} had no measurements, so it was not imported.")


SET_UP_THIS_PROJECT_BUTTON = "Set up this project"
MAKE_YOUR_FIRST_BATCH_BUTTON = f"Make your first {BATCH}"
ADD_MEASUREMENT_RESCORE_INFO = ("Add a measurement in Set up to score these "
                                "formulations again. Nothing recorded has "
                                "been lost.")
