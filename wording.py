"""Every word the user reads. Change a word here and it changes everywhere;
the tests read this module too."""

# ------------------------------------------------------------------ #
# Concepts. Today the set of formulations issued together is called a
# "trial" on screen; a future wave flips these two constants to
# "batch"/"Batch", and because every trial-naming sentence below is built
# from them, that flip is a one-line change.
# ------------------------------------------------------------------ #
BATCH = "trial"
BATCH_CAP = "Trial"
FORMULATION = "formulation"
FORMULATION_CAP = "Formulation"

# Plain nouns handed to ui_helpers.plural(n, word) at more than one call
# site, so the word is spelled once.
INGREDIENT = "ingredient"
PROCESS_SETTING = "process setting"

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
COPY_KEPT = "A copy is saved in your FoodOptimizer folder first."

NAME_RULE = ("Use 1 to 64 letters, numbers, spaces, hyphens, underscores or "
            "periods, starting with a letter or number.")


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
WIDEN_RANGE_HINT = " Widen the range in Set up, or check the value."


def saved_line(when):
    """'Saved 14:32 · automatically, to this Mac' — `when` is the already
    formatted time or date+time."""
    return f"Saved {when} · automatically, to this Mac"


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
    "Opens a ready-made plant-based burger project with eight "
    "ingredients and two trained-panel scores, Juiciness and "
    "Firmness, each with a target intensity, so you can explore "
    "before setting up your own."
)

PROJECT_LOAD_ERROR_SIDEBAR_NOTE = ("This project could not be opened. See "
                                   "the message on the right.")

BACKUP_UNAVAILABLE = ("Backup download is unavailable while the project "
                      "file cannot be read.")
DOWNLOAD_PROJECT_BACKUP = "Download project backup"

RESTORE_FROM_BACKUP = "Restore from backup"
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
    return (f"This backup contains project **{name}** with "
            + ", ".join(holds[:-1]) + f" and {holds[-1]}. Replace "
            f"**{project_name}** ({held})? " + COPY_KEPT)


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
    return (f"Start **{project_name}** over? Its {held_text} and set-up go, "
            f"and the project becomes empty. " + COPY_KEPT)


DELETE_PROJECT_LABEL = "Delete this project"
YES_DELETE_IT = "Yes, delete it"


def delete_project_warning(project_name, held_text=None):
    """`held_text` is None for an already-empty project."""
    if held_text is None:
        return (f"Delete **{project_name}**? It has no formulations yet, "
                f"and it leaves this list. " + COPY_KEPT)
    return (f"Delete **{project_name}** and its {held_text}? It leaves "
            f"this list. " + COPY_KEPT)


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
# Load-error / save-error banners, and the discarded-trial notice.
# ------------------------------------------------------------------ #
def project_load_error_info():
    return (
        "To protect the original file, editing is paused. In the sidebar on "
        "the left you can: restore a backup you downloaded earlier "
        "(Restore from backup), or start this project over "
        "(Manage project › Start this project over — a copy of the damaged "
        "file is saved first)."
    )


SAVE_ERROR_WARNING = ("**Your last change was not saved.** Download a "
                      "backup now, then click Reload project.")
DOWNLOAD_BACKUP = "Download backup"
RELOAD_PROJECT = "Reload project"
PROJECT_RELOADED = "Project reloaded from the latest saved copy."


def batch_discarded_notice():
    return (f"The open {BATCH} was discarded because the ingredient list or "
            "its allowed amounts changed since it was generated.")


def project_created(name):
    return f"Created {name}."


def project_opened(name):
    return f"Opened {name}."


def sample_project_failed(err):
    return f"The sample project could not be created: {err}"


def batch_line_open(no, n):
    """The one line under the title on tabs 1 and 2 once a trial exists and
    still has rows to make."""
    return f"{BATCH_CAP} {no} · {n} to make"


def batch_line_recorded(no):
    """Same line, once every row of that trial has been recorded (or left
    out)."""
    return f"{BATCH_CAP} {no} · recorded"


# ------------------------------------------------------------------ #
# Tab 2 · Make a trial: generate, the trial table, downloads, record
# results, upload results.
# ------------------------------------------------------------------ #
GENERATE_FORMULATIONS_DISABLED = "Generate formulations"
BACK_TO_SETUP = "Back to set up"

GENERATING_SPINNER = "Choosing the next formulations…"
GENERATE_FAILED = (
    "The app could not choose formulations this time. Try "
    "again with fewer formulations. If it keeps happening, "
    "loosen any limits you added recently, or open "
    "Help › Get Help."
)


def repeat_of_formulation(no):
    return f"Repeat of {FORMULATION_CAP} {no}"


def batch_ready(no):
    return f"{BATCH_CAP} {no} is ready to make."


NEW_FORMULATIONS_IN_BATCH = f"New formulations in this {BATCH}"


def repeat_checkbox_label(no):
    return f"Repeat {FORMULATION_CAP} {no} (best so far)"


REPEAT_HELP = ("A second reading of the same formulation, alongside the "
               "new ones.")


def generate_button_label(n):
    return f"Generate {n} formulations"


FIRST_FIVE_SPREAD = (
    "The first five formulations are spread across the "
    "allowed amounts; later trials aim closer to your "
    "targets."
)
EACH_BATCH_AIMS_CLOSER = f"Each {BATCH} aims closer to your targets."


def make_these(no, n):
    """'Trial 1 · make this 1 formulation' / '... make these 3
    formulations'."""
    word = FORMULATION if n == 1 else FORMULATION + "s"
    return (f"**{BATCH_CAP} {no} · make "
            f"{'this' if n == 1 else 'these'} {n} {word}**")


def batch_discarded_caption(numbers_text):
    return (f"Formulations {numbers_text} were discarded and their numbers "
            "will not be used again.")


def biggest_changes_caption(row_no, best_no, parts):
    return (f"Biggest changes in {FORMULATION_CAP} {row_no} from "
            f"{FORMULATION_CAP} {best_no}: {parts}.")


NEEDS_ONE_UNIT = "A formulation total needs all ingredients in one unit."


def batch_total_label(unit):
    return f"Formulation total ({unit})" if unit else "Formulation total"


BATCH_TOTAL_HELP = (
    "Leave this empty to weigh out the amounts as they were "
    "generated. Type a total and every formulation is scaled to it, "
    "on screen and on the sheets you print."
)
AS_GENERATED_PLACEHOLDER = "as generated"
AMOUNTS_SHOWN_FOR_TOTAL = "Amounts shown for this total."

MEASURED_PREFIX = "Measured "
TOTAL_PREFIX = "Total: "
NOTE_BLANK_LINE = "Note: ______________________________________________"


def note_line(note):
    return f"Note: {note}"


NOT_MADE_CHECKBOX_SHEET = "Not made [  ]"

DOWNLOAD_BENCH_SHEET = "Download the bench sheet (CSV)"
DOWNLOAD_FORMULATION_SHEETS = "Download formulation sheets (to print)"
PREVIEW_SHEETS = "Preview formulation sheets"


def sheets_use_total_caption(total_text):
    return f"Sheets use a formulation total of {total_text}."


GENERATE_DIFFERENT_BATCH = f"Generate a different {BATCH}"


def regenerate_warning(no, numbers_text):
    return (f"{BATCH_CAP} {no} (Formulations {numbers_text}) will be "
            "discarded. Those numbers will not be used again.")


YES_DISCARD = "Yes, discard"

RECORDED_FALLBACK = "recorded"


def recorded_line_partial(line):
    return f"{line} · partial" if line else "partial"


def recorded_line_note(line, note):
    return f"{line} · Note: {note}" if line else f"Note: {note}"


RECORD_RESULTS_HEADER = "Record results"
RECORD_RESULTS_CAPTION = ("Enter your panel mean. Leave a measurement "
                          "blank if it could not be scored.")


def formulation_heading(no):
    return f"**{FORMULATION_CAP} {no}**"


NOT_MADE_LABEL = "Not made"
NOT_MADE_HELP = "Type why in Note; it is kept with the formulation."

NOTHING_TO_SAVE = "Nothing to save — at least one formulation needs results."
SAVE_RESULTS = "Save results"


def filled_in_counter(entered, kept_n):
    """'2 of 3 filled in' — "filled in", not "to record": this counts rows
    that HAVE a value."""
    return f"{entered} of {kept_n} filled in"


def not_made_counter_suffix(n):
    return f" · {n} not made"


SAVED_WHEN_SUFFIX = " · saved when you press Save results"


def could_not_save(e):
    return f"Could not save these results: {e}"


NOT_MADE_NOTE_PREFIX = "Not made"


def not_made_with_note(note):
    return f"Not made · {note}"


def batch_recorded_flash(no):
    return f"{BATCH_CAP} {no} recorded."


UPLOAD_EXPANDER = "Or upload results from a CSV"
UPLOAD_HELP_CAPTION = (
    "Download the bench sheet above, fill in one column per "
    "measurement, and upload it here. Rows are matched by "
    "Formulation number."
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
    return (f"Recorded {parsed_n} of {total_n} formulations in {BATCH} "
            f"{batch_no} · {left_n} to make.")
