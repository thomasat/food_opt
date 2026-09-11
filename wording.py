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
    f"allowed amounts; later {BATCH}s aim closer to your "
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


NOT_MADE = "Not made"
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
    the open trial is back to as-generated, and only this sentence says so."""
    return (f"{BATCH_CAP} {batch_no} is no longer shown at a formulation "
            f"total of {total_text}; a formulation total needs all "
            "ingredients in one unit.")

# The one collapsed expander that maps the words on this tab to the words a
# specialist would use. Every optimization term the app otherwise refuses to
# say — variable, objective, weight, constraint — is said here and only here,
# one line each, so the mapping exists exactly once. The vocabulary guard
# reads this list by name and allows what is in it.
HOW_IT_WORKS = [
    "Ingredients and process settings are the variables; measurements with "
    "their goals are the objectives.",
    "Importance is each measurement's weight; closeness is its score between "
    "0 and 1 (1 at the goal).",
    "Higher is better: closeness = (measured − lowest) ÷ (highest − lowest), "
    "so the top of your range scores 1 and the bottom scores 0.",
    "Lower is better: the reverse — the bottom of your range scores 1 and the "
    "top scores 0.",
    "Hit a target: closeness is 1 at the target and falls evenly with "
    "distance, by one point per full range; the lowest score depends on how "
    "far the target sits from the ends of your range.",
    "The overall score is the weighted sum of closeness. The model learns "
    "this one number, so changing an importance or a range re-scores every "
    "past formulation.",
    "Limits are hard constraints applied when formulations are generated; an "
    "ingredient with no value for a property counts as containing none.",
    "The first five formulations are spread across the allowed amounts; "
    f"later {BATCH}s are chosen together — one set, chosen jointly, the "
    "optimizer's batch — from what the results suggest, some to test an idea "
    "rather than beat the best.",
    "A repeat is a second reading of one formulation; it teaches the model "
    "how noisy your measurements are.",
]
# Which of the nine lines above are the three goal lines nested under the
# second bullet, rather than bullets of their own.
HOW_IT_WORKS_NESTED = (2, 3, 4)

VARIABLES_HEADER = "Ingredients and process settings"


def made_before_units_caption(unit):
    return (f"Made before units were recorded; amounts are in {unit}. Set "
            "each unit below if that is wrong.")


UPLOAD_INGREDIENTS_EXPANDER = "Or upload an ingredients CSV"

NAME_LABEL = "Name"
SETTING_NAME_PLACEHOLDER = "e.g. Cook temperature"
INGREDIENT_NAME_PLACEHOLDER = "e.g. Water"
TYPE_LABEL = "Type"
VARIABLE_TYPE_HELP = ("Ingredients are weighed into the formulation and "
                      "count towards its total. Process settings, such as "
                      "temperature or time, are dialled in.")
LOWEST_LABEL = "Lowest"
HIGHEST_LABEL = "Highest"
NEW_INGREDIENT_FIXED_LOW_HELP = ("A new ingredient starts at 0 in every "
                                 "formulation already made, so its lowest "
                                 "is fixed at 0 for now.")
UNIT_LABEL = "Unit"
VARIABLE_UNIT_PLACEHOLDER = "°C, min, %"
BASELINE_LABEL = "Baseline"
BASELINE_REQUIRED_PLACEHOLDER = "required"
BASELINE_HELP = ("The setting you used for every formulation already made, "
                 "so those results still count.")
ADD_VARIABLE_BUTTON = "Add ingredient or setting"
NO_VALUE_PLACEHOLDER = "no value"
ADD_BASELINE_ERROR = ("Enter the baseline: the setting you used for every "
                      "formulation already made.")


def added(name):
    return f"Added {name}."


STATUS_LABEL = "Status"
ACTIVE_STATUS = "active"


def paused_status(held_text):
    return f"paused · held at {held_text}"


INGREDIENT_OR_SETTING_LABEL = "Ingredient or process setting"
NEW_UNIT_LABEL = "New unit"
SET_UNIT_BUTTON = "Set unit"
SET_PROPERTY_VALUES_BUTTON = "Set property values"
ONLY_INGREDIENT_HAS_PROPERTIES = "Only an ingredient carries property values."


def values_for_caption(name):
    return f"Values for {name}. Leave a box empty for no value."


SAVE_VALUES_BUTTON = "Save values"
CLOSE_BUTTON = "Close"


def property_values_saved(name):
    return f"Property values saved for {name}."


RESUME_BUTTON = "Resume"
RESUME_HELP = "Put it back into new formulations."
PAUSE_BUTTON = "Pause"
PAUSE_DISABLED_HELP = ("At least two ingredients or settings must stay "
                       "active before one can be paused.")
PAUSE_HELP = ("New formulations will not use it. Results already recorded "
             "are kept.")


def resumed(name):
    return f"Resumed {name}."


def paused(name):
    return f"Paused {name}."


UNIT_REQUIRED_ERROR = "A unit is required; use g if the amount is a mass."


def unit_changed(name, written, is_ingredient):
    """What changed is how the number is written, not the number: nothing is
    converted and nothing is rescored, and only this sentence says so."""
    held = "amounts were" if is_ingredient else "value was"
    if written:
        return f"{name} is now written in {written}. The {held} not converted."
    return f"{name} is shown without a unit."


YES_DELETE = "Yes, delete"


def delete_button(name):
    return f"Delete {name}"


def delete_variable_warning(name, is_ingredient):
    head = (f"Delete {name} from this project permanently? " if is_ingredient
            else f"Delete {name}? ")
    return (head + "Formulations already made will be recorded without it. "
            + COPY_KEPT)


DELETE_VS_PAUSE_CAPTION = ("Deleting takes it out of every formulation "
                           "already made; pausing keeps the data.")
DELETE_EVEN_IF_USED_CHECKBOX = ("Delete even if it was used (discards that "
                                "information)")


def deleted(name):
    return f"Deleted {name}."


REPLACE_INGREDIENTS_BUTTON = "Replace ingredients"
LOAD_INGREDIENTS_BUTTON = "Load ingredients"
INGREDIENTS_CSV_CAPTION = ("A CSV with the columns Name, Lowest, Highest "
                           "and, optionally, Unit. Extra columns become "
                           "properties you can set limits on.")
UPLOAD_INGREDIENTS_CSV_LABEL = "Upload ingredients CSV"


def blank_unit_cell_help(unit):
    return f"A blank Unit cell is in {unit}."


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
MEASUREMENT_UNIT_PLACEHOLDER = "e.g. N"
GOAL_LABEL = "Goal"
GOAL_SELECT_HELP = ("Whether you want this measurement higher, lower, or "
                    "at a target.")
TARGET_LABEL = "Target"
RANGE_HEADING = "**Range**"
LOWEST_MEASURABLE_LABEL = "Lowest measurable"
HIGHEST_MEASURABLE_LABEL = "Highest measurable"
RANGE_HINT_CAPTION = "The ends of your range, not the values you expect."
IMPORTANCE_LABEL = "Importance"
IMPORTANCE_HELP = "Any positive number. 2 counts twice as much as 1."
MEASUREMENT_EXISTS_ERROR = ("That measurement already exists. Use Edit on "
                            "its row to change it.")
ADD_MEASUREMENT_BUTTON = "Add measurement"
SAVE_CHANGES_BUTTON = "Save changes"

RECALCULATED_SUFFIX = " Every overall score was recalculated."


def importance_changed(name, value):
    return f"{name} importance changed to {float(value):.1f}."


def updated(name):
    return f"Updated {name}."


def measurement_deleted(name):
    return f"{name} deleted."


MEASUREMENTS_HEADER = "Measurements and targets"
ADD_A_MEASUREMENT_EXPANDER = "Add a measurement"
MEASUREMENT_COLUMN = "Measurement"
RANGE_COLUMN = "Range"


def edit_button(name):
    return f"Edit {name}"


def delete_measurement_warning(name):
    return (f"Delete {name}? Every overall score is recalculated without "
            "it. " + COPY_KEPT)


HOW_IT_WORKS_EXPANDER = "How it works"

ADD_PROPERTY_LABEL = "Add a property, such as Sodium per 100 g"
ADD_PROPERTY_PLACEHOLDER = "e.g. Sodium mg per 100 g"
ADD_PROPERTY_BUTTON = "Add property"


def property_added(name):
    return (f"Added {name}. Give each ingredient a value for it in "
            f"{VARIABLES_HEADER}.")


def delete_property_warning(name, limits_text):
    head = (f"Delete {name} and its {limits_text}? " if limits_text
            else f"Delete {name}? ")
    return head + "Ingredient values for it go too. " + COPY_KEPT


def property_deleted(name, gone_text=""):
    return f"Deleted {name}.{gone_text}"


def limit_went_with_it(limits_text):
    return f" Its {limits_text} went with it."


FINISHED_PRODUCT_LIMIT_HEADING = "**Finished-product limit**"
PER_100G_UNRESOLVED_CAPTION = ("Per 100 g of formulation once every "
                               "ingredient is in one mass unit.")


def per_100_caption(unit):
    return (f"Per 100 {unit} of formulation, from each ingredient's "
            "property values.")


INGREDIENT_PROPERTY_LABEL = "Ingredient property"
AT_LEAST_LABEL = "At least"
AT_MOST_LABEL = "At most"
NO_LIMIT_PLACEHOLDER = "no limit"
ADD_PROPERTY_LIMIT_BUTTON = "Add property limit"
ENTER_LOWEST_HIGHEST_ERROR = "Enter a lowest, a highest, or both."


def limit_added_on(who):
    return f"Limit added on {who}."


def limit_gap_tail(name, many):
    """' · Water has no value and counts as 0.' — the ingredients a limit is
    silently reading as zeroes."""
    return (f" · {name} have no value and count as 0." if many
            else f" · {name} has no value and counts as 0.")


LIMITS_EXPANDER = "Limits (optional)"
LIMITS_CAPTION = ("Limits hold every new formulation to an amount you weigh "
                  "out or a property of your ingredients. Measurements are "
                  "aimed at with targets, not limited.")


def old_limit_basis_caption(unit):
    return (f"A limit set before this version is now read per 100 {unit} "
            "of formulation.")


def at_least(value):
    return f"at least {value:g}"


def at_most(value):
    return f"at most {value:g}"


DELETE_LIMIT_BUTTON = "Delete limit"


def limit_deleted(who):
    return f"Limit on {who} deleted. The next {BATCH} is no longer held to it."


LIMIT_ON_CHOSEN_INGREDIENTS_HEADING = "**Limit on chosen ingredients**"
INGREDIENTS_TO_LIMIT_LABEL = "Ingredients to limit together"
ADD_INGREDIENT_LIMIT_BUTTON = "Add ingredient limit"

HOW_FORMULATIONS_CHOSEN_EXPANDER = "How formulations are chosen (advanced)"
STANDARD_VS_EXPERT_CAPTION = ("Standard uses tested defaults and fits most "
                              "projects. Expert-selected lets a specialist "
                              "set the model's kernel, prior, noise "
                              "handling and acquisition once at the start.")
HOW_FORMULATIONS_CHOSEN_LABEL = "How formulations are chosen"
STANDARD_DEFAULT_OPTION = "Standard (default)"
EXPERT_SELECTED_OPTION = "Expert-selected"
REVERT_TO_STANDARD_BUTTON = "Revert to standard settings"
USING_DEFAULT_MODEL_SETTINGS = "Using default model settings."
KERNEL_LABEL = "Kernel"
LENGTHSCALE_PRIOR_LABEL = "Lengthscale prior"
NOISE_LABEL = "Noise"
ACQUISITION_LABEL = "Acquisition"
FIXED_TINY_NOISE_CAPTION = ("`fixed_tiny` noise suits a deterministic "
                            "measurement, not a sensory panel — keep "
                            "`default` unless you have a specific reason.")
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
PARTIAL_SCORES_CAPTION = ("Partial scores are missing a measurement, which "
                          "counts as zero, so they are low and the model "
                          "treats them that way.")


def batch_recorded_progress(no, before, now):
    return (f"{BATCH_CAP} {no} recorded · best improved "
            f"{before:.2f} → {now:.2f}")


def batch_recorded_no_improvement(no):
    return f"{BATCH_CAP} {no} recorded · no improvement."


def best_so_far_heading(no, batch_no=None):
    heading = f"Best so far: {FORMULATION_CAP} {no}"
    if batch_no is not None:
        heading += f" ({BATCH} {batch_no})"
    return heading


MEASURED_COLUMN = "Measured"
OFF_BY_COLUMN = "Off by"
AMOUNTS_TO_MAKE_IT_HEADING = "**Amounts to make it**"
AMOUNT_COLUMN = "Amount"
NOT_USED_PREFIX = "Not used: "


def overall_score_caption(score, ceiling, partial):
    return (f"Overall score {score:.2f} of {ceiling:.2f}"
            + (" · partial" if partial else "")
            + ". Scores compare only within this project, and only until "
            "you change an importance or a range.")


ALL_FORMULATIONS_HEADING = "**All formulations**"
SORT_LABEL = "Sort"
SHOW_AMOUNTS_TOGGLE = "Show amounts"
DOWNLOAD_ALL_FORMULATIONS_BUTTON = "Download all formulations (CSV)"
DOWNLOAD_ALL_FORMULATIONS_HELP = ("Amounts are unitless in this file so it "
                                  "can be imported back; units are shown on "
                                  "screen. Formulations that were not made "
                                  "are not included.")

CORRECT_A_RESULT_LABEL = "Correct a result"
FORMULATIONS_NOT_MADE_NO_RESULT_CAPTION = ("Formulations that were not made "
                                           "have no result to correct.")
ADD_MEASUREMENT_BEFORE_CORRECTING_INFO = ("Add a measurement in Set up "
                                          "before correcting a result.")
LEAVE_BLANK_KEEP_VALUE_HELP = "Leave blank to keep the value already recorded."
SAVE_CORRECTION_BUTTON = "Save correction"
ENTER_VALUE_AT_LEAST_ONE_ERROR = "Enter a value for at least one measurement."


def formulation_unchanged(no):
    return f"{FORMULATION_CAP} {no} is unchanged."


def formulation_corrected(no, name, was_text, now_text):
    return f"{FORMULATION_CAP} {no} {name} corrected {was_text} → {now_text}."


def formulation_recorded_as(no, name, now_text):
    return f"{FORMULATION_CAP} {no} {name} recorded as {now_text}."


def back_to_batch_label(no, n):
    return f"Back to {BATCH} {no} · {n} to record"


START_NEXT_BATCH = f"Start the next {BATCH}"

PROGRESS_CHART_EXPANDER = "Progress chart"
NO_RESULTS_YET = "No results yet."
OVERALL_SCORE_COLUMN = "Overall score"
BEST_SO_FAR_COLUMN = "Best so far"
PROGRESS_CHART_CAPTION = ("Each formulation's overall score, and the best "
                          "so far. When the top line stops rising, you are "
                          "close to the best this ingredient list can do.")

DELETE_BATCH_OR_FORMULATION_EXPANDER = f"Delete a {BATCH} or a {FORMULATION}"
BATCH_NOT_RECORDED_BEFORE_VERSION_CAPTION = (
    f"{BATCH_CAP}s were not recorded before this version. You can delete one "
    "formulation at a time below.")


def no_batch_to_delete_caption():
    return f"No {BATCH} to delete yet."


def batch_open_record_first_caption():
    return f"Record or discard the open {BATCH} first."


DELETE_LAST_BATCH_BUTTON = f"Delete the last {BATCH}"


def delete_last_batch_warning(no, formulations_text):
    return f"Deletes {BATCH} {no} and its {formulations_text}. " + COPY_KEPT


def batch_deleted(no):
    return f"{BATCH_CAP} {no} deleted. " + COPY_KEPT


FORMULATION_TO_DELETE_LABEL = f"{FORMULATION_CAP} to delete"


def no_formulation_to_delete_caption():
    return f"No {FORMULATION} to delete yet."


def delete_formulation_button(no):
    return f"Delete {FORMULATION_CAP} {no}"


def delete_formulation_warning(no):
    return (f"Delete {FORMULATION_CAP} {no}? Later formulations keep their "
            "numbers. " + COPY_KEPT)


def formulation_deleted(no):
    return f"{FORMULATION_CAP} {no} deleted. " + COPY_KEPT


IMPORT_FORMULATIONS_EXPANDER = f"Import past {FORMULATION}s from a CSV"


def import_columns_caption(names_text):
    return (f"One row per {FORMULATION} you already made. The columns must "
            f"match these names exactly: {names_text}.")


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


SET_UP_THIS_PROJECT_BUTTON = "Set up this project"
MAKE_YOUR_FIRST_BATCH_BUTTON = f"Make your first {BATCH}"
ADD_MEASUREMENT_RESCORE_INFO = ("Add a measurement in Set up to score these "
                                "formulations again. Nothing recorded has "
                                "been lost.")
