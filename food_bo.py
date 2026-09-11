import json
import re
from datetime import datetime, timezone

import numpy as np
import pandas as pd
import torch
from torch.quasirandom import SobolEngine

from botorch.acquisition import (
    qLogExpectedImprovement,
    qLogNoisyExpectedImprovement,
)
from botorch.acquisition.monte_carlo import qUpperConfidenceBound
from botorch.fit import fit_gpytorch_mll
from botorch.models import SingleTaskGP
from botorch.models.transforms.input import Warp
from botorch.models.transforms.outcome import Standardize
from botorch.optim import optimize_acqf
from botorch.sampling.normal import SobolQMCNormalSampler
from botorch.utils.transforms import normalize, unnormalize
from gpytorch.kernels import (
    LinearKernel,
    MaternKernel,
    PolynomialKernel,
    RBFKernel,
    ScaleKernel,
)
from gpytorch.mlls import ExactMarginalLogLikelihood
from gpytorch.priors import GammaPrior

from storage import LocalStorage, StorageError
import wording


# Column names history_frame() (and any future export) reserves for itself;
# a variable with one of these names would silently overwrite that column.
RESERVED_VARIABLE_NAMES = {
    "Experiment", "Date", "Overall Score", "Recipe",
    "Formulation", "Batch", "Trial", "Overall score", "Note", "Recorded",
    "Best", "Total",
}

# The batch table's own total column carries the unit it is summing —
# 'Total (g)', 'Total (ml)' — so an ingredient named 'Total (g)' collides with
# it just as plainly as one named 'Total'. Two columns of the same name break
# the batch table outright and put the formulation total on the sheet where
# that ingredient's own amount belongs.
_TOTAL_COLUMN_RE = re.compile(r"^total(\s*\(.*\))?$", re.IGNORECASE)


# What a missing ingredient-file column is called on screen. The file may
# head its columns Min and Max, or Lowest and Highest; the boxes on the screen
# say Lowest and Highest, so that is what any message about them says.
_FILE_COLUMNS = {'Name': 'Name', 'Min': 'Lowest', 'Max': 'Highest'}


def is_reserved_name(name):
    """True when a variable name would overwrite a column the app owns.
    Capitalisation is ignored: 'total' and 'Total (g)' are the same column."""
    name = str(name).strip()
    return (name.lower() in {r.lower() for r in RESERVED_VARIABLE_NAMES}
            or bool(_TOTAL_COLUMN_RE.match(name)))


def join_unit(text, unit):
    """'6 N', '7/10' — a unit that starts with a slash joins tight, everything
    else takes one space. Every screen goes through this, so a project whose
    panel scores are '/10' never reads '7 /10'."""
    unit = str(unit or "")
    if not unit:
        return str(text)
    return f"{text}{unit}" if unit.startswith("/") else f"{text} {unit}"


def unit_after_number(unit):
    """The unit as it is written after a number. A "/"-style unit (such as
    /10) is shown once, in the measurement's label or column header, so after
    a number it is nothing at all: '5', not '5/10'."""
    unit = str(unit or "")
    return "" if unit.startswith("/") else unit


def label_with_unit(name, unit):
    """'Firmness (/10)' — the one place a "/"-style unit is written. Every
    other unit rides after its number, so the label stays bare ('Firmness',
    measured '6 N')."""
    unit = str(unit or "")
    return f"{name} ({unit})" if unit.startswith("/") else str(name)


def outside_message(name, value, low, high, unit, what, tail=""):
    """'Firmness 12 N is outside your range of 0 to 10 N.' — the one builder
    for every out-of-bounds line, so a measurement typed into the grid, one
    read off an uploaded sheet and an amount imported from a CSV are refused
    in the same words. `what` names the bounds, `tail` is any sentence that
    follows. A "/"-style unit stays off the numbers, as it does everywhere."""
    unit = unit_after_number(unit)
    return (join_unit(f"{name} {float(value):g}", unit)
            + f" is outside {what} of "
            + join_unit(f"{float(low):g} to {float(high):g}", unit)
            + "." + tail)


def local_date(ts):
    """The date a stored moment fell on where the user is standing. Results
    are stamped in UTC, so slicing the first ten characters off the stamp
    dated a batch recorded at 23:25 as tomorrow."""
    if not isinstance(ts, str) or not ts:
        return ""
    try:
        moment = datetime.fromisoformat(ts)
    except ValueError:
        return ts[:10]
    if moment.tzinfo is None:
        moment = moment.replace(tzinfo=timezone.utc)
    return moment.astimezone().strftime("%Y-%m-%d")


def number_list(numbers):
    """'1', '1 and 2', '7, 8 and 9'. Lives here because refusals raised by the
    model name formulations too, and ui_helpers already imports from this
    module."""
    items = [str(n) for n in numbers]
    if len(items) <= 1:
        return "".join(items)
    return ", ".join(items[:-1]) + " and " + items[-1]


def goal_line(obj):
    """The half of an input label that says what a good number looks like:
    'target 6 N', 'lower is better', 'higher is better'."""
    if obj.get('goal') == 'target' and obj.get('target') is not None:
        return join_unit(f"target {float(obj['target']):g}",
                         unit_after_number(obj.get('unit')))
    return "lower is better" if obj.get('goal') == 'min' else "higher is better"


def _fmt_weight(w):
    """'1', '1.5', '2.25' — a whole importance drops its decimal now that
    its share of the score sits right beside it in parentheses; anything
    with a fraction still shows one."""
    txt = f"{float(w):.2f}".rstrip("0").rstrip(".")
    return txt


# --------------------------------------------------------------------------- #
#  Expert-selectable BO hyperparameters (optional "arm 3").
#  bo_config == None  =>  library defaults, i.e. byte-identical to the standard
#  non-adaptive arm. A validated dict swaps in the expert's chosen kernel /
#  lengthscale prior / noise handling / acquisition. Chosen ONCE at project
#  start (not per iteration): the GP's lengthscale/noise VALUES still refit from
#  data each iteration via MLE; only the structural config is fixed a priori.
# --------------------------------------------------------------------------- #
_KERNELS = {"matern52", "matern32", "rbf", "linear", "poly2"}
_LENGTHSCALE = {"default", "long", "short"}
_NOISE = {"default", "low", "fixed_tiny"}
_ACQ = {"qlognei", "qlogei", "qucb"}
_LS_PRIORS = {"default": (3.0, 6.0), "long": (3.0, 1.0), "short": (3.0, 12.0)}
DEFAULT_BO_CONFIG = {
    "kernel": "matern52", "lengthscale_prior": "default",
    "noise": "default", "acquisition": "qlognei",
}


def validate_bo_config(spec):
    """Coerce a raw config dict to valid values. Returns None for an empty/None
    spec (None => library defaults, identical to the standard arm)."""
    if not spec:
        return None

    def pick(key, allowed, default):
        v = str(spec.get(key, default)).strip().lower()
        return v if v in allowed else default

    return {
        "kernel": pick("kernel", _KERNELS, "matern52"),
        "lengthscale_prior": pick("lengthscale_prior", _LENGTHSCALE, "default"),
        "noise": pick("noise", _NOISE, "default"),
        "acquisition": pick("acquisition", _ACQ, "qlognei"),
    }


def _build_covar(cfg, dim):
    kernel = cfg.get("kernel", "matern52")
    if kernel in ("matern52", "matern32"):
        nu = 2.5 if kernel == "matern52" else 1.5
        conc, rate = _LS_PRIORS.get(cfg.get("lengthscale_prior", "default"), _LS_PRIORS["default"])
        base = MaternKernel(nu=nu, ard_num_dims=dim, lengthscale_prior=GammaPrior(conc, rate))
    elif kernel == "rbf":
        conc, rate = _LS_PRIORS.get(cfg.get("lengthscale_prior", "default"), _LS_PRIORS["default"])
        base = RBFKernel(ard_num_dims=dim, lengthscale_prior=GammaPrior(conc, rate))
    elif kernel == "linear":
        base = LinearKernel()
    elif kernel == "poly2":
        base = PolynomialKernel(power=2)
    else:
        base = MaternKernel(nu=2.5, ard_num_dims=dim)
    return ScaleKernel(base)


class FoodOptimizer:
    CLASS_VERSION = 8  # bump when adding methods/attrs to force session refresh

    def __init__(self, project_name="experiment", robust=False, storage=None):
        """Initialize or load a food optimization project.

        Args:
            project_name: Name used for the .pkl save file.
            robust: If True, uses Input Warping for cliffs/traps.
                    If False (default), uses standard GP for smooth problems.
            storage: Persistence backend. Defaults to LocalStorage() (the
                     desktop app's on-disk .pkl files); a cloud backend can
                     be injected instead.
        """
        self.project_name = project_name
        self.robust = robust
        self.filename = f"{project_name}.pkl"   # informational; kept for compat
        self.storage = storage if storage is not None else LocalStorage()

        self.variables = []
        self.objectives = []
        self.ingredient_properties = {}
        # Properties named in the app rather than in an ingredient file. The
        # ordered union of these and the columns a file brought in is what
        # properties() answers, and what the screen lists.
        self.property_names = []
        self.constraints = []
        self.quantity_constraints = []
        self.screening_model = None
        self.bo_config = None  # None => library defaults; dict => expert-chosen (arm 3)

        self.X_history = []
        self.Y_history = []
        self.recipe_history = []
        self.results_history = []
        self.timestamps_history = []  # UTC ISO per tell(); parallel to results_history
        # Identity. A formulation's number is permanent and never reissued:
        # numbers are drawn at generation from next_formulation_no, and a
        # discarded, deleted or undone number simply retires.
        self.formulation_ids = []     # global formulation number per scored row
        self.batch_history = []       # batch number per scored row, or None
        self.notes_history = []       # the lab's note per scored row, or ""
        self.skipped = []             # generated but never scored: dicts with
                                      # formulation / batch / recipe / note
        self.next_formulation_no = 1
        # One unit for every amount in the project. Grams is the default a
        # food scientist expects; a blank unit made every amount ambiguous.
        self.amount_unit = "g"
        self.amount_unit_backfilled = False  # True only for a file with no unit
        self.pending_batch = None     # the open batch: [{'formulation', 'recipe'}]
        self.pending_batch_no = None  # its batch number
        self.pending_batch_created = None   # ISO date it was generated, for the sheets
        self.pending_batch_discarded = []   # numbers a regenerate retired, for one caption
        self.load_error = None  # set to a plain-language string if load() fails
        self.save_error = None  # set when a cloud save fails; cleared on success
        self.last_saved_at = None  # records successful save time; timezone-aware datetime or None

        try:
            exists = self.storage.exists(project_name)
        except StorageError as e:
            # Backend down: must NOT save (would overwrite a real project
            # with a blank one). Surface as a load error instead.
            self.load_error = str(e)
        else:
            if exists:
                self.load()
            elif self.storage.persist_empty_on_init:
                self.save()

    # ------------------------------------------------------------------ #
    #  Setup: Ingredients & Process Parameters
    # ------------------------------------------------------------------ #

    def _check_new_variable(self, name, min_val, max_val, category):
        """Shared validation for add_ingredient / add_process_parameter.
        Returns the stripped name. Same-name same-category is allowed (the
        caller updates bounds); a clash with the other category is an error."""
        name = str(name).strip()
        if not name:
            raise ValueError("Name cannot be empty.")
        if is_reserved_name(name):
            raise ValueError(
                f"{name} is a column name Food Optimizer uses for its own "
                f"tables. Choose another name, for example {name}s."
            )
        if any(name.lower() == obj['name'].lower() for obj in self.objectives):
            raise ValueError(
                f"{name} is already the name of a measurement. Choose another name."
            )
        if float(min_val) >= float(max_val):
            raise ValueError("Lowest must be less than Highest.")
        for v in self.variables:
            if v['name'].lower() == name.lower() and v.get('category', 'ingredient') != category:
                other = v.get('category', 'ingredient')
                other_label = "an ingredient" if other == 'ingredient' else "a process setting"
                raise ValueError(f"{v['name']} already exists as {other_label}.")
        return name

    def add_ingredient(self, name, min_val, max_val, unit=None):
        """Add a single ingredient. Safe to call mid-run (adaptive EGBO): the
        ingredient is treated as absent (=0) in every prior recipe, and the
        encoded history is rebuilt so the GP stays dimensionally consistent.

        `unit` is this ingredient's own unit — the water may be in ml while
        the powders are in g. None means "no unit of its own": the ingredient
        follows the project's default (`amount_unit`), which is what a 0.2.x
        project's ingredients do and what a blank CSV cell means.

        Returns the amount limits this change emptied of meaning (see
        prune_amount_limits), as every other unit edit does: adding an
        ingredient that already exists is one of the ways a unit changes, and
        a limit that then adds grams to millilitres is a number of nothing.
        """
        name = self._check_new_variable(name, min_val, max_val, 'ingredient')
        min_val, max_val = float(min_val), float(max_val)
        for var in self.variables:
            if var['name'] == name:
                var['bounds'] = (min_val, max_val)
                if unit is not None:
                    var['unit'] = str(unit).strip()
                removed = self.prune_amount_limits()
                self._drop_pending_batch()
                self.save()
                return removed
        if self.X_history:
            if len(self.recipe_history) != len(self.X_history):
                raise ValueError(
                    "Cannot add this now: some formulations were recorded without their "
                    "amounts. Start a fresh project or re-import your history."
                )
            min_val = 0.0  # absent-in-past encodes as 0; it must be within bounds
        var = {
            'name': name,
            'type': 'continuous',
            'bounds': (min_val, max_val),
            'category': 'ingredient',
            'active': True,
        }
        if unit is not None:
            var['unit'] = str(unit).strip()
        self.variables.append(var)
        if self.X_history:
            self._reencode_history()
        removed = self.prune_amount_limits()
        self._drop_pending_batch()
        self.save()
        return removed

    def load_ingredients_from_csv(self, df):
        """Bulk-load ingredients from a DataFrame (used by the app).

        Raises ValueError if experiments already exist, since reloading
        would invalidate encoded history vectors. Returns the amount limits
        the new file emptied of meaning (see prune_amount_limits).
        """
        if self.X_history:
            raise ValueError(
                "Cannot reload ingredients after results have been recorded. "
                "Use Manage project > Start this project over, or restore "
                "from a backup."
            )

        # Accept any capitalization/whitespace for the required headers, and
        # fail with a plain-language error (the app shows ValueError text to
        # the user) instead of a KeyError when one is missing.
        # Lowest/Highest are what every box on the screen says; Min/Max are
        # what a sheet written for an older version carries. Both are read,
        # and neither is mentioned to the user in the other's place.
        canonical = {'name': 'Name', 'min': 'Min', 'max': 'Max', 'type': 'Type',
                     'unit': 'Unit', 'lowest': 'Min', 'highest': 'Max'}
        df = df.rename(columns={
            c: canonical[c.strip().lower()]
            for c in df.columns if c.strip().lower() in canonical
        })
        missing = [c for c in ('Name', 'Min', 'Max') if c not in df.columns]
        if missing:
            raise ValueError(
                f"The ingredients file is missing required column(s): "
                f"{', '.join(_FILE_COLUMNS[c] for c in missing)}. Expected "
                f"columns: Name, Lowest, Highest (plus an optional Unit "
                f"column and property columns like Cost or Protein)."
            )

        process_vars = [v for v in self.variables if v.get('category') == 'process']
        self.variables = []
        self.ingredient_properties = {}

        standard_cols = {'Name', 'Min', 'Max', 'Type', 'Unit'}
        prop_cols = [c for c in df.columns if c not in standard_cols]

        seen_names = set()
        for i, (_, row) in enumerate(df.iterrows()):
            raw_name = row.get('Name')
            name = "" if raw_name is None or (isinstance(raw_name, float) and np.isnan(raw_name)) else str(raw_name).strip()
            if not name:
                raise ValueError(f"Row {i + 2}: the Name cell is blank.")
            if name.lower() in seen_names:
                raise ValueError(f"Row {i + 2}: duplicate ingredient name {name}.")
            if is_reserved_name(name):
                raise ValueError(
                    f"Row {i + 2}: {name} is a column name Food Optimizer uses "
                    f"for its own tables. Choose another name, for example {name}s."
                )
            seen_names.add(name.lower())
            try:
                min_val, max_val = float(row['Min']), float(row['Max'])
            except (ValueError, TypeError):
                raise ValueError(
                    f"Ingredient '{row['Name']}': Lowest and Highest must be "
                    f"numbers. "
                    f"Please check that column for text or blank cells and try "
                    f"again."
                )
            if min_val >= max_val:
                raise ValueError(
                    f"Ingredient '{name}': Lowest ({min_val}) must be less "
                    f"than Highest ({max_val})"
                )
            var = {
                'name': name,
                'type': 'continuous',
                'bounds': (min_val, max_val),
                'category': 'ingredient',
                'active': True,
            }
            # A blank Unit cell means "the project's default", not a blank
            # unit: a file listing ml against the water alone should leave
            # every other row in whatever the project is set to.
            raw_unit = row.get('Unit') if 'Unit' in df.columns else None
            if raw_unit is not None and not (isinstance(raw_unit, float)
                                             and np.isnan(raw_unit)):
                unit = str(raw_unit).strip()
                if unit:
                    var['unit'] = unit
            self.variables.append(var)

            props = {}
            for col in prop_cols:
                try:
                    val = float(row[col])
                    if not pd.isna(val):
                        # The file's own capitalisation, kept: the picker and
                        # the limits list show this name, and 'Fat per 100 g'
                        # lower-cased read as a different column from the one
                        # the caption above it names. Matching ignores case.
                        props[str(col).strip()] = val
                except (ValueError, TypeError):
                    pass
            self.ingredient_properties[name] = props

        self.variables.extend(process_vars)
        # A column of the file is a property of this project from now on, and
        # it keeps its place in the file's own order. A property named in the
        # app earlier stays named: the file replaces the values, not the list.
        for col in prop_cols:
            self._remember_property(str(col).strip())
        # The new file can rename every unit and drop ingredients outright,
        # so the limits are re-checked against it and the caller is told.
        removed = self.prune_amount_limits()
        self._drop_pending_batch()
        self.save()
        return removed

    def add_process_parameter(self, name, min_val, max_val, baseline=None,
                              unit=""):
        """Add a process parameter (e.g. baking temperature, mixing time).

        Added mid-run it requires `baseline` — the value used in ALL prior
        batches — because past experiments ran at a fixed setting, not at 0.
        History then encodes at that baseline (its 'absent' value), and min is
        NOT forced to 0 (unlike an ingredient). `baseline` must lie in [min, max].

        `unit` is the setting's own unit (°C, min, rpm). A setting is not an
        amount, so it never wears the project's amount unit; without one of
        its own a sheet printed a bare "Cook temperature: 175".
        """
        name = self._check_new_variable(name, min_val, max_val, 'process')
        min_val, max_val = float(min_val), float(max_val)
        unit = str(unit or "").strip()
        for var in self.variables:
            if var['name'] == name:
                var['bounds'] = (min_val, max_val)
                var['unit'] = unit
                self._drop_pending_batch()
                self.save()
                return
        var = {
            'name': name,
            'type': 'continuous',
            'bounds': (min_val, max_val),
            'category': 'process',
            'active': True,
            'unit': unit,
        }
        if self.X_history:
            if len(self.recipe_history) != len(self.X_history):
                raise ValueError(
                    "Cannot add this now: some formulations were recorded without their "
                    "amounts. Start a fresh project or re-import your history."
                )
            if baseline is None:
                raise ValueError(
                    "A process setting added now needs a baseline (the value used for "
                    "every formulation already made) so those formulations encode correctly."
                )
            baseline = float(baseline)
            if not (min_val <= baseline <= max_val):
                raise ValueError(
                    f"Baseline {baseline:g} must be between {min_val:g} and {max_val:g}."
                )
            var['_absent_value'] = baseline
        self.variables.append(var)
        if self.X_history:
            self._reencode_history()
        self._drop_pending_batch()
        self.save()

    def remove_process_parameter(self, name):
        """Remove a process parameter by name."""
        self.variables = [
            v for v in self.variables
            if not (v['name'] == name and v.get('category') == 'process')
        ]
        self._reencode_history()
        self._drop_pending_batch()
        self.save()

    def load_screening_model(self, model_obj):
        self.screening_model = model_obj

    # ------------------------------------------------------------------ #
    #  Setup: Objectives
    # ------------------------------------------------------------------ #

    def add_objective(self, name, weight, goal='max', target=None,
                      min_val=None, max_val=None, unit=""):
        """Add or replace an objective. Returns True if an objective of the
        same name was replaced. Stored scores are recomputed either way so the
        history and the model never disagree with the current weights."""
        name = str(name).strip()
        if not name:
            raise ValueError("Measurement name cannot be empty.")
        if any(name.lower() == v['name'].lower() for v in self.variables):
            raise ValueError(
                f"{name} is already the name of an ingredient or process "
                f"setting. Choose another name for the measurement."
            )
        if is_reserved_name(name):
            raise ValueError(
                f"{name} is a column name Food Optimizer uses for its own "
                f"tables. Choose another name."
            )
        weight = float(weight)
        if weight <= 0:
            raise ValueError("Importance must be greater than 0.")
        min_val = float(min_val) if min_val is not None else 0.0
        max_val = float(max_val) if max_val is not None else 10.0
        if min_val >= max_val:
            raise ValueError("Range lowest must be less than range highest.")
        if goal == 'target':
            if target is None:
                raise ValueError("Enter a target value for a 'Hit a target' measurement.")
            target = float(target)
            if not (min_val <= target <= max_val):
                raise ValueError(
                    f"Target {target:g} must be between the range's lowest "
                    f"and highest ({min_val:g} to {max_val:g})."
                )
        else:
            target = None
        replaced = any(obj['name'] == name for obj in self.objectives)
        self.objectives = [obj for obj in self.objectives if obj['name'] != name]
        self.objectives.append({
            'name': name, 'weight': weight, 'goal': goal,
            'target': target, 'min_val': min_val, 'max_val': max_val,
            'unit': str(unit or "").strip(),
        })
        self._recompute_utilities()
        self.save()
        return replaced

    def remove_objective(self, name):
        """Remove an objective and recalculate stored utility scores."""
        self.objectives = [obj for obj in self.objectives if obj['name'] != name]
        self._recompute_utilities()
        self.save()

    def set_amount_unit(self, unit):
        """The default unit a new ingredient starts in — 'g', '%', 'ml'. It is
        also the unit every ingredient without one of its own is written in,
        which is what makes it the one place a 0.2.x project (whose
        ingredients predate per-ingredient units) can be corrected in one
        move — and what lets it split an amount limit's ingredients apart, so
        it prunes them like every other unit change and returns what went."""
        self.amount_unit = str(unit or "").strip()
        self.amount_unit_backfilled = False
        removed = self.prune_amount_limits()
        self.save()
        return removed

    def prune_amount_limits(self):
        """Drop every limit that has stopped meaning anything, and return what
        was dropped so the screen can name it.

        A limit is arithmetic, not a label: an amount limit holds the next
        batch to the SUM of some ingredients, and a property limit to their
        mass-weighted AVERAGE. Three edits can leave either meaningless — a
        unit set on one ingredient, a new default unit (every ingredient
        without one of its own follows it), and a reloaded ingredient file,
        which can do both at once and drop ingredients outright. All three
        call this, so a limit that survives one edit is one that still means
        something.

        An amount-limit entry is {'ingredients', 'min', 'max', 'reason'} with
        reason 'missing' (it names an ingredient this project no longer has,
        listed in 'missing') or 'unit' (its ingredients no longer share one);
        a property-limit entry carries 'metric' instead of 'ingredients', and
        only ever the 'unit' reason. Formulations already made are
        untouched."""
        names = {v['name'] for v in self.variables
                 if v.get('category', 'ingredient') == 'ingredient'}
        kept, removed = [], []
        for qc in self.quantity_constraints:
            gone = [n for n in qc['ingredients'] if n not in names]
            if gone:
                removed.append(dict(qc, reason='missing', missing=gone))
            elif len({self.unit_of(n) for n in qc['ingredients']}) > 1:
                removed.append(dict(qc, reason='unit'))
            else:
                kept.append(qc)
        self.quantity_constraints = kept
        # A property limit is read per 100 g of the finished formulation —
        # an average over the same amounts — so it needs one unit just as a
        # sum does, and there is no per-limit question to ask: once the
        # ingredients differ, every one of them has stopped meaning anything.
        if self.constraints and len(self.ingredient_units()) > 1:
            removed += [dict(c, reason='unit') for c in self.constraints]
            self.constraints = []
        return removed

    def set_variable_unit(self, name, unit):
        """The unit one row of What you can vary is written in, whichever kind
        it is. A process setting carries its own unit — a cook temperature is
        in °C — and is part of no sum and no average, so setting one can break
        no limit; an ingredient's goes the long way round, through the pruning
        every unit change owes the limits."""
        var = next((v for v in self.variables if v['name'] == name), None)
        if var is None:
            raise ValueError(f"No ingredient or setting named {name}.")
        if var.get('category', 'ingredient') == 'ingredient':
            return self.set_ingredient_unit(name, unit)
        var['unit'] = str(unit or "").strip()
        self.save()
        return []

    def set_ingredient_unit(self, name, unit):
        """The unit one ingredient's amounts are written in. Nothing is
        rescored and the open batch stands: a unit is how a number is
        written, not the number. Returns the amount limits this change
        emptied of meaning (see prune_amount_limits)."""
        var = next((v for v in self.variables
                    if v['name'] == name
                    and v.get('category', 'ingredient') == 'ingredient'), None)
        if var is None:
            raise ValueError(f"No ingredient named {name}.")
        var['unit'] = str(unit or "").strip()
        removed = self.prune_amount_limits()
        self.save()
        return removed

    def unit_of(self, name):
        """The unit one variable's amount is written in. Every screen that
        prints an amount asks here, so the batch table, the printable sheets,
        the amounts table and the CSV always agree."""
        return self._unit_of(next((v for v in self.variables
                                   if v['name'] == name), None))

    def _unit_of(self, var):
        """The same answer for a variable dict. A process setting has its own
        unit or none — a cook temperature is never 200 g. An ingredient with
        no unit of its own follows the project's default: that is how a 0.2.x
        project's ingredients (and a blank Unit cell in a CSV) behave."""
        if var is None:
            return ""
        if var.get('category', 'ingredient') == 'process':
            return str(var.get('unit', "") or "")
        unit = var.get('unit')
        return str(self.amount_unit or "") if unit is None else str(unit or "")

    def properties(self):
        """Every property this project knows, in order: the ones named in the
        app first, then any column an ingredient file brought in.

        A property is an attribute of an ingredient — Sodium per 100 g, Cost —
        that a limit on the finished formulation is written against. It used
        to arrive only as an extra column in an ingredient CSV; it can now be
        named on the Limits section and given a value per ingredient, which is
        why this is the one list the screen reads."""
        names, seen = [], set()
        for name in (getattr(self, 'property_names', None) or []):
            key = str(name).strip().lower()
            if key and key not in seen:
                seen.add(key)
                names.append(str(name).strip())
        for var in self.variables:
            if var.get('category', 'ingredient') != 'ingredient':
                continue
            for prop in (self.ingredient_properties.get(var['name']) or {}):
                key = str(prop).strip().lower()
                if key and key not in seen:
                    seen.add(key)
                    names.append(str(prop).strip())
        return names

    def _remember_property(self, name):
        """Record a property name in property_names if it is not there yet."""
        name = str(name).strip()
        if not name:
            return
        stored = getattr(self, 'property_names', None) or []
        if any(str(p).strip().lower() == name.lower() for p in stored):
            return
        self.property_names = list(stored) + [name]

    def _known_property(self, metric):
        """The stored spelling of a property, or None. Matching ignores
        capitals, exactly as property_value does."""
        wanted = str(metric).strip().lower()
        return next((p for p in self.properties()
                     if str(p).strip().lower() == wanted), None)

    def add_property(self, name):
        """Name a property in the app, with no ingredient file at all. Returns
        the stored name.

        A property is a column of the app's own tables once it exists, so its
        name is checked against everything else the project names: a property
        called Water beside an ingredient called Water is two things with one
        name on one screen."""
        name = str(name).strip()
        if not name:
            raise ValueError("Name cannot be empty.")
        if is_reserved_name(name):
            raise ValueError(
                f"{name} is a column name Food Optimizer uses for its own "
                f"tables. Choose another name, for example {name}s."
            )
        lowered = name.lower()
        for var in self.variables:
            if var['name'].lower() == lowered:
                kind = ("an ingredient"
                        if var.get('category', 'ingredient') == 'ingredient'
                        else "a process setting")
                raise ValueError(
                    f"{var['name']} is already the name of {kind}. Choose "
                    f"another name.")
        for obj in self.objectives:
            if obj['name'].lower() == lowered:
                raise ValueError(
                    f"{obj['name']} is already the name of a measurement. "
                    f"Choose another name.")
        if self._known_property(name) is not None:
            raise ValueError(f"{name} is already a property of this project.")
        self._remember_property(name)
        self.save()
        return name

    def remove_property(self, name):
        """Remove a property, the values every ingredient holds for it and any
        limit written against it. Returns the limits that went, so the screen
        can name them."""
        stored = self._known_property(name)
        if stored is None:
            raise ValueError(f"No property named {name}.")
        lowered = stored.lower()
        self.property_names = [
            p for p in (getattr(self, 'property_names', None) or [])
            if str(p).strip().lower() != lowered
        ]
        for props in self.ingredient_properties.values():
            for key in [k for k in list(props or {})
                        if str(k).strip().lower() == lowered]:
                props.pop(key, None)
        removed = [c for c in self.constraints
                   if str(c['metric']).strip().lower() == lowered]
        self.constraints = [c for c in self.constraints
                            if str(c['metric']).strip().lower() != lowered]
        self.save()
        return removed

    def set_property_value(self, ingredient, metric, value):
        """One ingredient's value for one property. `None` clears it, and an
        ingredient with no value counts as 0 in the average — which is what
        the limit line says on screen."""
        var = next((v for v in self.variables
                    if v['name'] == ingredient
                    and v.get('category', 'ingredient') == 'ingredient'), None)
        if var is None:
            raise ValueError(f"No ingredient named {ingredient}.")
        stored = self._known_property(metric)
        if stored is None:
            raise ValueError(f"No property named {metric}.")
        props = self.ingredient_properties.setdefault(ingredient, {})
        for key in [k for k in list(props)
                    if str(k).strip().lower() == stored.lower()]:
            props.pop(key, None)
        if value is not None:
            props[stored] = float(value)
        self.save()

    def has_property_value(self, name, metric):
        """True when this ingredient has a value for this property. A 0 is a
        value; a blank is not, and the two must not read alike."""
        props = self.ingredient_properties.get(name, {}) or {}
        wanted = str(metric).strip().lower()
        return any(str(key).strip().lower() == wanted for key in props)

    def ingredients_without_property(self, metric):
        """The ingredients that have no value for this property, in order.
        They count as 0 in the per-100 average, and every limit on it says so
        in as many words."""
        return [v['name'] for v in self.variables
                if v.get('category', 'ingredient') == 'ingredient'
                and not self.has_property_value(v['name'], metric)]

    def property_value(self, name, metric):
        """One ingredient's value for a property — 0.0 when it has none.

        Matched without regard to capitalisation: the stored key carries the
        file's own capitalisation ('Fat per 100 g'), while a limit written
        against an older project stored it lower-cased, and both must find
        the same column."""
        props = self.ingredient_properties.get(name, {}) or {}
        if metric in props:
            return float(props[metric])
        lowered = str(metric).strip().lower()
        for key, value in props.items():
            if str(key).strip().lower() == lowered:
                return float(value)
        return 0.0

    def per_amount_text(self):
        """'per 100 g' — how a limit on the finished formulation reads, in the
        unit the ingredients are written in."""
        return "per 100 " + (self.one_amount_unit() or "g")

    def property_per_100(self, recipe_dict, metric):
        """One property of a finished formulation, per 100 g of it: the
        mass-weighted average of its ingredients' own per-100 g values. None
        when the formulation weighs nothing, which has no average.

        A process setting is not part of the mass and takes no part."""
        total = weighted = 0.0
        for var in self.variables:
            if var.get('category', 'ingredient') != 'ingredient':
                continue
            amount = float(recipe_dict.get(var['name'], 0.0) or 0.0)
            total += amount
            weighted += amount * self.property_value(var['name'], metric)
        if total <= 0:
            return None
        return weighted / total

    def _property_coeff(self, metric, limit):
        """The coefficient each variable carries in Σ amount × (property −
        limit): the per-100 g average rearranged so that it stays linear in
        the amounts — ≤ 0 is 'at most the limit', ≥ 0 is 'at least'.

        Linear is what BoTorch can be given, and giving the screen the same
        form is what keeps the two agreeing to the last decimal."""
        limit = float(limit)
        names = {v['name'] for v in self.variables
                 if v.get('category', 'ingredient') == 'ingredient'}
        return lambda name: ((self.property_value(name, metric) - limit)
                             if name in names else 0.0)

    def _property_residual(self, recipe_dict, metric, limit):
        """Σ amount × (property − limit) for one formulation."""
        coeff = self._property_coeff(metric, limit)
        return sum(coeff(name) * float(amount or 0.0)
                   for name, amount in recipe_dict.items())

    def majority_amount_unit(self, names=None):
        """The unit most of the ingredients are already in — the one the
        refusals ask for the odd ones out to be re-entered in."""
        counted = {}
        for var in self.variables:
            if var.get('category', 'ingredient') != 'ingredient':
                continue
            if names is not None and var['name'] not in names:
                continue
            counted.setdefault(self._unit_of(var), []).append(var['name'])
        if not counted:
            return str(self.amount_unit or "")
        order = list(counted)
        return sorted(order, key=lambda u: (-len(counted[u]), u == "",
                                            order.index(u)))[0]

    def unit_fix_sentence(self, names=None):
        """'enter Water in g instead of ml.' — the change that would let a
        limit be written, or '' when the ingredients already share a unit.

        A refusal that only says the units differ leaves the user to work out
        which ingredient is the odd one and what to do about it. `names`
        limits the question to one group of ingredients (an amount limit); the
        default asks it of every ingredient (a property limit)."""
        counted = {}
        for var in self.variables:
            if var.get('category', 'ingredient') != 'ingredient':
                continue
            if names is not None and var['name'] not in names:
                continue
            counted.setdefault(self._unit_of(var), []).append(var['name'])
        if len(counted) <= 1:
            return ""
        order = list(counted)
        # The unit most of them are already in, preferring a real unit to a
        # blank one and, on a tie, the one that appears first.
        target = self.majority_amount_unit(names)
        odd = [name for unit in order if unit != target
               for name in counted[unit]]
        others = [unit for unit in order if unit != target]
        tail = (f" instead of {others[0] or 'no unit'}"
                if len(others) == 1 else "")
        return (f"enter {number_list(odd)} in {target or 'no unit'}{tail}.")

    def ingredient_units(self):
        """Every unit the ingredients are written in, in ingredient order."""
        units = []
        for var in self.variables:
            if var.get('category', 'ingredient') != 'ingredient':
                continue
            unit = self._unit_of(var)
            if unit not in units:
                units.append(unit)
        return units

    def one_amount_unit(self):
        """The unit every ingredient shares, or None when they differ. What
        the screen asks before offering to scale a batch to a total, and what
        an amount limit needs: adding 25 g of powder to 40 ml of water gives
        a number of nothing."""
        units = self.ingredient_units()
        if not units:
            return str(self.amount_unit or "")
        return units[0] if len(units) == 1 else None

    def unit_totals(self, recipe):
        """The formulation total per unit: [(unit, total), ...] in ingredient order.
        A total that added grams to millilitres was a number of nothing."""
        totals, order = {}, []
        for var in self.variables:
            if var.get('category', 'ingredient') != 'ingredient':
                continue
            unit = self._unit_of(var)
            if unit not in totals:
                totals[unit] = 0.0
                order.append(unit)
            totals[unit] += float(recipe.get(var['name'], 0.0))
        return [(unit, totals[unit]) for unit in order]

    def total_text(self, recipe):
        """'340.00 g · 60.00 ml' — the total of a formulation whose
        ingredients are not all in one unit, written as one cell. A unit that
        adds up to nothing is left out; two decimals, as every other amount."""
        groups = self.unit_totals(recipe)
        shown = [(u, t) for u, t in groups if round(float(t), 2) != 0] or groups[:1]
        return " · ".join(join_unit(f"{float(t):.2f}", u) for u, t in shown)

    def has_ingredients(self):
        """True when anything is weighed out. A project of process settings
        alone — incubation temperature, time, culture dose — has no amounts,
        so it has no total to show and nothing to scale to one."""
        return any(v.get('category', 'ingredient') == 'ingredient'
                   for v in self.variables)

    def total_column(self):
        """The header of the total column: 'Total (g)' while every ingredient
        shares one unit, a bare 'Total' when they do not, because the cell
        then carries the units itself. None when nothing is weighed out."""
        if not self.has_ingredients():
            return None
        unit = self.one_amount_unit()
        if unit is None:
            return "Total"
        return f"Total ({unit})" if unit else "Total"

    def _total_cell(self, recipe):
        """What goes in that column: a number while there is one unit (the
        screen rounds it itself), the written-out per-unit total otherwise."""
        if self.one_amount_unit() is None:
            return self.total_text(recipe)
        return self.ingredient_total(recipe)

    def total_csv_columns(self):
        """The total columns of the downloaded batch sheet, as
        [(unit, header), ...]. A spreadsheet cannot add up '10.00 g · 40.00
        ml', so where the screen shows one written-out cell the sheet gives
        each unit a column of numbers: `Total (g)`, `Total (ml)`."""
        if not self.has_ingredients():
            return []
        if self.one_amount_unit() is not None:
            return [(self.one_amount_unit(), self.total_column())]
        return [(unit, f"Total ({unit})" if unit else "Total")
                for unit in self.ingredient_units()]

    def update_objective(self, name, /, **fields):
        """Change a measurement in place. Its name is fixed (renaming would
        orphan every stored result), everything else can change, and every
        stored overall score is recalculated. The open batch is untouched:
        formulations do not depend on measurements."""
        obj = next((o for o in self.objectives if o['name'] == name), None)
        if obj is None:
            raise ValueError(f"No measurement named {name}.")
        allowed = {'weight', 'goal', 'target', 'min_val', 'max_val', 'unit'}
        unknown = sorted(set(fields) - allowed)
        if unknown:
            raise ValueError(f"Cannot change {', '.join(unknown)}.")
        merged = dict(obj)
        merged.update(fields)
        weight = float(merged.get('weight', 1.0))
        if weight <= 0:
            raise ValueError("Importance must be greater than 0.")
        min_val = float(merged.get('min_val', 0.0))
        max_val = float(merged.get('max_val', 10.0))
        if min_val >= max_val:
            raise ValueError("Range lowest must be less than range highest.")
        goal = merged.get('goal', 'max')
        if goal == 'target':
            if merged.get('target') is None:
                raise ValueError("Enter a target value for a 'Hit a target' measurement.")
            target = float(merged['target'])
            if not (min_val <= target <= max_val):
                raise ValueError(
                    f"Target {target:g} must be between the range's lowest "
                    f"and highest ({min_val:g} to {max_val:g})."
                )
        else:
            target = None
        obj.update({
            'weight': weight, 'goal': goal, 'target': target,
            'min_val': min_val, 'max_val': max_val,
            'unit': str(merged.get('unit', "") or "").strip(),
        })
        self._recompute_utilities()
        self.save()
        return obj

    def measurements_by_importance(self):
        """Measurements as every screen orders them: most important first,
        ties in the order they were added."""
        return sorted(self.objectives, key=lambda o: -float(o['weight']))

    def share_of_score(self, name):
        """This measurement's importance as a fraction of the sum, in
        [0, 1]. Reads as the measurement's share of the score, not of a
        target's distance: two of the three goals have no target."""
        obj = next((o for o in self.objectives if o['name'] == name), None)
        if obj is None:
            raise ValueError(f"No measurement named {name}.")
        total = sum(float(o['weight']) for o in self.objectives)
        return float(obj['weight']) / total if total else 0.0

    def share_text(self, name):
        """'60 %' — this measurement's share of the score, as a whole
        percent."""
        return join_unit(f"{round(100 * self.share_of_score(name)):d}", "%")

    def score_function_line(self):
        """The one line under the measurements table that writes the score
        out, each importance beside its share of the total."""
        if not self.objectives:
            return ""
        terms = " + ".join(
            f"{_fmt_weight(o['weight'])} ({self.share_text(o['name'])}) "
            f"× {o['name']} closeness"
            for o in self.measurements_by_importance()
        )
        # How closeness is worked out belongs in the expander below this
        # line, per goal: two of the three goals have no target at all, so a
        # sentence about distance from one was wrong on most screens.
        return (f"Overall score = {terms}. Every measurement at its goal "
                f"scores {self.utility_ceiling():.2f}.")

    def closeness_details(self, index):
        """The best-formulation table, most important first: one dict per
        measurement with 'name', 'goal', 'measured' and 'off_by' as the
        strings the screen shows.

        Off by is only meaningful against a target. A 'higher is better'
        measurement has no target, so quoting its distance from the top of the
        range would read a good result as a failure; those rows show '—'.
        No formulation at `index` (None, negative, or past the end) returns
        an empty list rather than wrapping around or raising."""
        if index is None or index < 0:
            return []
        results = self.results_history[index] if index < len(self.results_history) else {}
        rows = []
        for obj in self.measurements_by_importance():
            # A "/10" is written once, on the measurement's own row label;
            # every other unit follows each number. Measured and Off by share
            # one number format, so 5 and 1 never read as 5 and 1.0.
            unit = unit_after_number(obj.get('unit'))
            name = label_with_unit(obj['name'], obj.get('unit'))
            is_target = obj['goal'] == 'target'
            if is_target:
                goal_text = join_unit(f"Target {float(obj['target']):g}", unit)
            elif obj['goal'] == 'min':
                goal_text = "Lower is better"
            else:
                goal_text = "Higher is better"
            raw = results.get(obj['name'])
            if raw is None:
                rows.append({'name': name, 'goal': goal_text,
                             'measured': "not scored",
                             'off_by': "not scored" if is_target else "—"})
                continue
            val = float(raw)
            measured = join_unit(f"{val:g}", unit)
            if not is_target:
                rows.append({'name': name, 'goal': goal_text,
                             'measured': measured, 'off_by': "—"})
                continue
            delta = val - float(obj['target'])
            if abs(delta) < 1e-9:
                off_by = "On target"
            else:
                size = join_unit(f"{abs(delta):g}", unit)
                off_by = f"{size} too high" if delta > 0 else f"{size} too low"
            rows.append({'name': name, 'goal': goal_text,
                         'measured': measured, 'off_by': off_by})
        return rows

    def biggest_changes(self, recipe, ref_recipe, n=2):
        """The n largest amount changes from ref_recipe to recipe, largest
        first, as (name, change) pairs. Amounts that did not move are left out.

        Ingredients only. A process setting is not an amount: reporting a cook
        temperature as "+198.65 g" priced a setting in grams, and against a
        formulation made before the setting existed the change was the whole
        baseline rather than anything the batch actually moved."""
        pairs = []
        for var in self.variables:
            if var.get('category', 'ingredient') != 'ingredient':
                continue
            name = var['name']
            try:
                delta = float(recipe.get(name, 0.0)) - float(ref_recipe.get(name, 0.0))
            except (TypeError, ValueError):
                continue
            if abs(delta) < 0.005:
                continue
            pairs.append((name, delta))
        pairs.sort(key=lambda kv: abs(kv[1]), reverse=True)
        return pairs[:n]

    def ingredient_total(self, recipe):
        """The formulation total: process settings are not amounts and are excluded."""
        return sum(float(recipe.get(v['name'], 0.0)) for v in self.variables
                   if v.get('category', 'ingredient') == 'ingredient')

    def scaled_recipe(self, recipe, scale_to=None):
        """The same formulation written for a different formulation total. Every
        screen, sheet and download that shows a scaled amount goes through
        this, so what is printed always equals what is displayed."""
        if scale_to is None:
            return dict(recipe)
        total = self.ingredient_total(recipe)
        if total <= 0:
            return dict(recipe)
        factor = float(scale_to) / total
        out = dict(recipe)
        for var in self.variables:
            value = float(recipe.get(var['name'], 0.0))
            out[var['name']] = (value * factor
                                if var.get('category', 'ingredient') == 'ingredient'
                                else value)
        return out

    def _recompute_utilities(self):
        for i, results_dict in enumerate(self.results_history):
            if i < len(self.Y_history):
                self.Y_history[i] = self._compute_utility(results_dict)

    def utility_ceiling(self):
        """The Overall Score a perfect recipe would get: the sum of weights."""
        return float(sum(obj['weight'] for obj in self.objectives))

    def best_index(self):
        """0-based index of the highest-scoring experiment, or None."""
        if not self.Y_history:
            return None
        return int(max(range(len(self.Y_history)), key=lambda i: self.Y_history[i]))

    def best_so_far(self):
        """Running maximum of the Overall Score, one value per experiment."""
        out, cur = [], float('-inf')
        for y in self.Y_history:
            cur = max(cur, float(y))
            out.append(cur)
        return out

    def _measurement_column(self, obj):
        unit = str(obj.get('unit', "") or "")
        return f"{obj['name']} ({unit})" if unit else obj['name']

    def _amount_column(self, name):
        """The table header for one variable, carrying that variable's own
        unit: `Water (ml)` beside `Pea protein (g)`, and a process setting
        with its own unit or none — a cook temperature must never read
        "Cook temperature (g)"."""
        unit = self.unit_of(name)
        return f"{name} ({unit})" if unit else name

    def _amount_columns(self, recipe):
        return {self._amount_column(v['name']): recipe.get(v['name'])
                for v in self.variables}

    def history_frame(self, order="Best first", include_amounts=False):
        """Every formulation — scored and left out — as the All formulations
        table shows them. `order` is 'Best first', 'Newest first' or
        wording.SORT_BATCH_ORDER's value. Batch is a string in every row: a
        project that predates batches has blanks, and a mixed int/blank
        column renders inconsistently."""
        objs = self.measurements_by_importance()
        best_i = self.best_index()
        rows = []
        for i in range(len(self.X_history)):
            results = self.results_history[i] if i < len(self.results_history) else {}
            partial = any(o['name'] not in results for o in objs)
            ts = self.timestamps_history[i] if i < len(self.timestamps_history) else None
            batch = self.batch_history[i] if i < len(self.batch_history) else None
            row = {
                "Best": "★" if i == best_i else "",
                wording.BATCH_CAP: "" if batch is None else str(int(batch)),
                "Formulation": int(self.formulation_ids[i]),
                "_score": float(self.Y_history[i]),
                "_seq": i,
                "_batch": 0 if batch is None else int(batch),
            }
            for obj in objs:
                row[self._measurement_column(obj)] = results.get(obj['name'])
            # '2.30 · partial', in the separator the rest of the app reads
            # a list with: a partial score is missing a measurement and is not
            # the same number as a complete one.
            row["Overall score"] = (f"{float(self.Y_history[i]):.2f}"
                                    + (" · partial" if partial else ""))
            row["Recorded"] = local_date(ts)
            row["Note"] = self.notes_history[i] if i < len(self.notes_history) else ""
            if include_amounts:
                row.update(self._amount_columns(self._decode(self.X_history[i])))
            rows.append(row)
        for k, s in enumerate(self.skipped):
            batch = s.get('batch')
            row = {
                # A row nobody made has no score to be best; saying so in the
                # Best column is what stops it reading as the worst.
                "Best": "not made",
                wording.BATCH_CAP: "" if batch is None else str(int(batch)),
                "Formulation": int(s['formulation']),
                "_score": float('-inf'),
                "_seq": len(self.X_history) + k,
                "_batch": 0 if batch is None else int(batch),
            }
            for obj in objs:
                row[self._measurement_column(obj)] = None
            row["Overall score"] = ""
            row["Recorded"] = ""
            row["Note"] = s.get('note') or "Not made"
            if include_amounts:
                row.update(self._amount_columns(s.get('recipe', {})))
            rows.append(row)
        columns = (["Best", wording.BATCH_CAP, "Formulation"]
                   + [self._measurement_column(o) for o in objs]
                   + ["Overall score", "Recorded", "Note"])
        if include_amounts:
            columns += [self._amount_column(v['name']) for v in self.variables]
        if not rows:
            return pd.DataFrame(columns=columns)
        df = pd.DataFrame(rows)
        if order == "Newest first":
            df = df.sort_values("_seq", ascending=False)
        elif order == wording.SORT_BATCH_ORDER:
            df = df.sort_values(["_batch", "Formulation"], ascending=[True, True])
        else:
            df = df.sort_values(["_score", "_seq"], ascending=[False, True])
        return df[columns].reset_index(drop=True)

    def batch_frame(self, batch, scale_to=None):
        """The open batch as the make-these table: one row per formulation, one
        column per ingredient and setting carrying its own unit, and the total
        of the ingredients — one number while they share a unit, `340.00 g ·
        60.00 ml` when they do not. `scale_to` rewrites it for a different
        formulation total — display only, the stored row never changes."""
        total_col = self.total_column()
        # The total closes the amounts you weigh out, so it sits with them,
        # before the settings you dial in — the order the sheet is filled in.
        ingredients = [v for v in self.variables
                       if v.get('category', 'ingredient') == 'ingredient']
        process = [v for v in self.variables if v.get('category') == 'process']
        rows = []
        read = self._batch_rows(batch)
        noted = any(r.get('note') for r in read)
        for row in read:
            recipe = self.scaled_recipe(row['recipe'], scale_to)
            item = {"Formulation": int(row['formulation'])}
            for var in ingredients:
                item[self._amount_column(var['name'])] = float(
                    recipe.get(var['name'], 0.0))
            if total_col is not None:
                item[total_col] = self._total_cell(recipe)
            for var in process:
                item[self._amount_column(var['name'])] = float(
                    recipe.get(var['name'], 0.0))
            if noted:
                item["Note"] = row.get('note', "")
            rows.append(item)
        columns = (["Formulation"]
                   + [self._amount_column(v['name']) for v in ingredients]
                   + ([total_col] if total_col is not None else [])
                   + [self._amount_column(v['name']) for v in process]
                   + (["Note"] if noted else []))
        return pd.DataFrame(rows, columns=columns)

    def recipe_lines(self, recipe, limit=None):
        """Ingredient/setting amounts for display: largest first, zero amounts
        omitted, as (name, amount) pairs. `limit` keeps the first N (the rest are
        summarised by the caller)."""
        pairs = []
        for k, v in recipe.items():
            try:
                amount = float(v)
            except (TypeError, ValueError):
                continue
            if amount != amount or amount == 0.0:   # NaN or exactly zero: not shown
                continue
            pairs.append((k, amount))
        items = sorted(pairs, key=lambda kv: kv[1], reverse=True)
        return items if limit is None else items[:limit]

    def batch_csv(self, batch, scale_to=None):
        """The sheet the lab fills in: the global Formulation numbers, the
        amounts to weigh out with their units in the headers and rounded as
        the screen rounds them, the same total the screen shows, one blank
        column per measurement, and a Note column. Its columns run in the
        order the batch table's do, so the sheet reads like the screen.
        `scale_to` must match what the screen shows, or the lab weighs out
        amounts nobody saw.

        The measurement columns stay bare: they are the ones an uploaded
        sheet is matched by, and a "/10" belongs on a label, not in a
        header the parser reads back. Where the screen writes one total cell
        across units, the sheet gives each unit its own column of numbers —
        a spreadsheet cannot add up "10.00 g · 40.00 ml"."""
        objs = self.measurements_by_importance()
        ingredients = [v for v in self.variables
                       if v.get('category', 'ingredient') == 'ingredient']
        process = [v for v in self.variables if v.get('category') == 'process']
        total_cols = self.total_csv_columns()
        rows = []
        for row in self._batch_rows(batch):
            recipe = self.scaled_recipe(row['recipe'], scale_to)
            item = {"Formulation": int(row['formulation'])}
            for var in ingredients:
                item[self._amount_column(var['name'])] = round(
                    float(recipe.get(var['name'], 0.0)), 2)
            totals = dict(self.unit_totals(recipe))
            for unit, column in total_cols:
                item[column] = round(float(totals.get(unit, 0.0)), 2)
            for var in process:
                item[self._amount_column(var['name'])] = round(
                    float(recipe.get(var['name'], 0.0)), 2)
            for obj in objs:
                item[obj['name']] = ""
            item["Note"] = row.get('note', "")
            rows.append(item)
        columns = (["Formulation"]
                   + [self._amount_column(v['name']) for v in ingredients]
                   + [column for _, column in total_cols]
                   + [self._amount_column(v['name']) for v in process]
                   + [o['name'] for o in objs] + ["Note"])
        return pd.DataFrame(rows, columns=columns).to_csv(index=False)

    def parse_batch_results(self, df, batch):
        """Match an uploaded results sheet to the open batch.

        The sheet needs a Formulation column holding the global numbers from
        the downloaded bench sheet. `Recipe` and `Experiment` are accepted as
        legacy headers and read as 1-based positions in the batch. Every
        measurement needs its own column — an absent column is refused
        outright (a typo'd header would otherwise silently drop that
        measurement from every row). A blank cell in a column that IS present
        means that one result could not be scored, so the row is stored as a
        partial result; a row with nothing filled in is refused. Returns
        [(formulation number, {measurement: value}, note), ...].
        """
        rows = self._batch_rows(batch)
        numbers = [r['formulation'] for r in rows]
        norm = {str(c).strip().lower(): c for c in df.columns}
        if "formulation" in norm:
            key_col, legacy = norm["formulation"], False
        elif "recipe" in norm:
            key_col, legacy = norm["recipe"], True
        elif "experiment" in norm:
            key_col, legacy = norm["experiment"], True
        else:
            raise ValueError(
                "The sheet needs a Formulation column with the numbers from "
                "the downloaded bench sheet."
            )
        col_for, missing = {}, []
        for obj in self.objectives:
            key = obj['name'].strip().lower()
            if key in norm:
                col_for[obj['name']] = norm[key]
            else:
                missing.append(obj['name'])
        if missing:
            raise ValueError("Missing columns: " + ", ".join(missing))
        note_col = norm.get("note")
        if len(df) == 0:
            raise ValueError("The sheet has no result rows.")
        in_batch = ", ".join(str(n) for n in numbers)
        parsed, seen = [], set()
        for _, sheet_row in df.iterrows():
            raw_no = sheet_row[key_col]
            try:
                as_float = float(raw_no)
            except (TypeError, ValueError):
                raise ValueError(f"Formulation number {raw_no!s} is not a whole number.")
            if not as_float.is_integer():
                raise ValueError(f"Formulation number {raw_no!s} is not a whole number.")
            number = int(as_float)
            if legacy:
                if not (1 <= number <= len(rows)):
                    raise ValueError(
                        f"Formulation {number} is not in batch "
                        f"{self.pending_batch_no} (it has {in_batch})."
                    )
                number = numbers[number - 1]
            elif number not in numbers:
                raise ValueError(
                    f"Formulation {number} is not in batch "
                    f"{self.pending_batch_no} (it has {in_batch})."
                )
            if number in seen:
                raise ValueError(
                    f"Formulation {number} appears more than once in the sheet."
                )
            seen.add(number)
            results = {}
            for name, col in col_for.items():
                val = sheet_row[col]
                if (val is None
                        or (isinstance(val, float) and np.isnan(val))
                        or str(val).strip() == ""):
                    continue
                try:
                    val = float(val)
                except (TypeError, ValueError):
                    raise ValueError(f"Formulation {number} {name} is not a number.")
                obj = next(o for o in self.objectives if o['name'] == name)
                if not (obj['min_val'] <= val <= obj['max_val']):
                    # The same sentence the results grid refuses with.
                    raise ValueError(outside_message(
                        f"Formulation {number} {name}", val,
                        obj['min_val'], obj['max_val'], obj.get('unit'),
                        "your range",
                        " Widen the range in Set up, or check the value."))
                results[name] = val
            if not results:
                raise ValueError(
                    f"Formulation {number} has no measurements filled in."
                )
            note = ""
            if note_col is not None:
                raw_note = sheet_row[note_col]
                if raw_note is not None and not (isinstance(raw_note, float)
                                                 and np.isnan(raw_note)):
                    note = str(raw_note).strip()
            parsed.append((number, results, note))
        return parsed

    def history_csv(self):
        """Every scored formulation as CSV, with the plain column names
        'Import past formulations from a CSV' expects, plus Formulation, Batch
        and Note. Skipped (Not made) formulations carry no results at all, so
        they are left out entirely: a blank measurement cell would be refused
        by the importer's own "these columns have blank cells" check, and a
        formulation that was never made has nothing to import.

        Amount columns come from the re-encoded history (as history_frame
        does), not raw recipe_history: an ingredient added mid-project is
        backfilled to 0 in X_history for earlier rows, while recipe_history is
        never backfilled and would export a blank there.
        """
        rows = []
        for i, x in enumerate(self.X_history):
            ts = self.timestamps_history[i] if i < len(self.timestamps_history) else None
            batch = self.batch_history[i] if i < len(self.batch_history) else None
            row = {
                "Formulation": int(self.formulation_ids[i]),
                wording.BATCH_CAP: "" if batch is None else int(batch),
                "Recorded": local_date(ts),
                # Two decimals, as the screen shows it: a file that says
                # 2.625 where the table says 2.62 reads as a third number.
                "Overall score": round(float(self.Y_history[i]), 2),
            }
            row.update(self._decode(x))
            row.update(self.results_history[i] if i < len(self.results_history) else {})
            row["Note"] = self.notes_history[i] if i < len(self.notes_history) else ""
            rows.append(row)
        return pd.DataFrame(rows).to_csv(index=False)

    # ------------------------------------------------------------------ #
    #  Setup: Constraints
    # ------------------------------------------------------------------ #

    def add_constraint(self, metric, min_val=None, max_val=None):
        """A limit on the finished formulation: one property per 100 g of what
        you make, worked out as the mass-weighted average of the ingredients'
        own values. Replaces any existing limit on the same property.

        Per 100 g of formulation, not as a total: a total grew with the batch,
        so the same formulation passed at 100 g and failed at 1 kg."""
        if min_val is not None and max_val is not None and float(min_val) >= float(max_val):
            raise ValueError("Lowest must be less than Highest.")
        # An average over the amounts, so the amounts must be in one unit:
        # 25 g of powder and 40 ml of water share no 100 g to be measured per.
        units = self.ingredient_units()
        if len(units) > 1:
            raise ValueError(
                f"Property limits are per 100 "
                f"{self.majority_amount_unit() or 'g'}, so every "
                f"ingredient needs a mass unit; " + self.unit_fix_sentence()
            )
        # Same property, whatever its capitalisation: two limits on 'Fat' and
        # 'fat' would both be enforced against the same column.
        self.constraints = [c for c in self.constraints
                            if str(c['metric']).strip().lower()
                            != str(metric).strip().lower()]
        self.constraints.append({
            'metric': metric,
            'min': float(min_val) if min_val is not None else None,
            'max': float(max_val) if max_val is not None else None,
            # What this limit was written to mean. A limit from an older file
            # has no basis at all, and the screen says once that it is now
            # read per 100 g.
            'basis': 'per_100',
        })
        self.save()

    def remove_constraint(self, index):
        """Remove a property constraint by index."""
        if 0 <= index < len(self.constraints):
            self.constraints.pop(index)
            self.save()

    def add_quantity_constraint(self, ingredients, min_val=None, max_val=None):
        """Add a constraint on the sum of selected ingredient quantities.
        Replaces any existing constraint on the same set of ingredients.

        Args:
            ingredients: List of ingredient names whose quantities to sum.
            min_val: Minimum allowed sum (or None for no lower bound).
            max_val: Maximum allowed sum (or None for no upper bound).
        """
        if min_val is not None and max_val is not None and float(min_val) >= float(max_val):
            raise ValueError("Lowest must be less than Highest.")
        # A limit is a sum, and a sum across units is a number of nothing:
        # 25 g of powder plus 40 ml of water is neither 65 g nor 65 ml.
        if len({self.unit_of(name) for name in ingredients}) > 1:
            raise ValueError(
                "A limit adds amounts, so these ingredients need one "
                "unit; " + self.unit_fix_sentence(list(ingredients)))
        ingredient_set = set(ingredients)
        self.quantity_constraints = [
            qc for qc in self.quantity_constraints if set(qc['ingredients']) != ingredient_set
        ]
        self.quantity_constraints.append({
            'ingredients': list(ingredients),
            'min': float(min_val) if min_val is not None else None,
            'max': float(max_val) if max_val is not None else None,
        })
        self.save()

    def add_total_mass_constraint(self, min_val=None, max_val=None):
        """The limit over every ingredient — what the one Limit on chosen
        ingredients control writes when its picker is left on All ingredients.

        Refused in the same words as any other such limit while the
        ingredients are not all in one unit: the refusal names the ingredient
        to re-enter and the unit to enter it in, which is the same answer
        whether the group is two ingredients or all of them."""
        all_ingredients = [
            v['name'] for v in self.variables
            if v.get('category', 'ingredient') == 'ingredient'
        ]
        self.add_quantity_constraint(all_ingredients, min_val, max_val)

    def remove_quantity_constraint(self, index):
        """Remove a quantity constraint by index."""
        if 0 <= index < len(self.quantity_constraints):
            self.quantity_constraints.pop(index)
            self.save()

    # ------------------------------------------------------------------ #
    #  Utility Scoring
    # ------------------------------------------------------------------ #

    def _compute_utility(self, results_dict):
        """Compute a weighted utility score from raw results."""
        total_utility = 0.0
        for obj in self.objectives:
            raw_val = results_dict.get(obj['name'])
            if raw_val is None:
                continue
            val = float(raw_val)

            min_v = obj.get('min_val', 0.0)
            max_v = obj.get('max_val', 10.0)
            rng = max_v - min_v
            if rng == 0:
                rng = 1.0

            norm_val = max(0.0, min(1.0, (val - min_v) / rng))

            if obj['goal'] == 'max':
                utility = norm_val
            elif obj['goal'] == 'min':
                utility = 1.0 - norm_val
            elif obj['goal'] == 'target':
                targ_val = obj.get('target', (max_v + min_v) / 2)
                norm_targ = (targ_val - min_v) / rng
                utility = max(0.0, 1.0 - abs(norm_val - norm_targ))
            else:
                utility = 0.0

            total_utility += obj['weight'] * utility
        return total_utility

    # ------------------------------------------------------------------ #
    #  History Editing
    # ------------------------------------------------------------------ #

    def edit_result(self, index, new_results_dict):
        """Edit a previously saved result and recalculate its utility."""
        if index < 0 or index >= len(self.Y_history):
            raise IndexError("There is no formulation at that position.")
        if index < len(self.results_history):
            self.results_history[index] = dict(new_results_dict)
        self.Y_history[index] = self._compute_utility(new_results_dict)
        self.save()

    def edit_amounts(self, index, recipe_dict):
        """Correct the amounts a past formulation was actually made with.

        The encoded row goes with them: the model reads X_history, so a
        recipe fixed on screen and left un-encoded would keep steering the
        next batch towards a formulation nobody made."""
        if index < 0 or index >= len(self.recipe_history):
            raise IndexError("There is no formulation at that position.")
        self.recipe_history[index] = dict(recipe_dict)
        self.X_history[index] = self._encode(recipe_dict)
        self.save()

    def _drop_result(self, index):
        """Take one scored formulation out of every parallel history. No save:
        the caller decides when the file is written, so several deletions can
        share one."""
        if index < 0 or index >= len(self.X_history):
            raise IndexError("There is no formulation at that position.")
        self.X_history.pop(index)
        self.Y_history.pop(index)
        for lst in (self.recipe_history, self.results_history,
                    self.timestamps_history, self.formulation_ids,
                    self.batch_history, self.notes_history):
            if index < len(lst):
                lst.pop(index)

    def delete_result(self, index):
        """Delete one formulation. Later formulations keep their numbers."""
        self._drop_result(index)
        self.save()

    def rewind_to(self, index):
        """Keep only formulations 0..index (inclusive), discard the rest."""
        if index < 0 or index >= len(self.X_history):
            raise IndexError("There is no formulation at that position.")
        keep = index + 1
        self.X_history = self.X_history[:keep]
        self.Y_history = self.Y_history[:keep]
        self.recipe_history = self.recipe_history[:keep]
        self.results_history = self.results_history[:keep]
        self.timestamps_history = self.timestamps_history[:keep]
        self.formulation_ids = self.formulation_ids[:keep]
        self.batch_history = self.batch_history[:keep]
        self.notes_history = self.notes_history[:keep]
        # Left-out formulations belong to their batch. A skipped row from a
        # batch that no longer has any scored rows would otherwise survive as
        # an orphan — it would still be offered for deletion, still count
        # towards the last batch, and still show in All formulations.
        kept = [int(b) for b in self.batch_history if b is not None]
        cut = max(kept) if kept else None
        self.skipped = [s for s in self.skipped
                        if s.get('batch') is None
                        or (cut is not None and int(s['batch']) <= cut)]
        self._drop_pending_batch()
        self.save()

    # ------------------------------------------------------------------ #
    #  Internal: Encoding / Decoding / Bounds
    # ------------------------------------------------------------------ #

    def _encode(self, recipe_dict):
        """Encode a recipe dict into a flat numeric vector.

        Missing keys default to 0.0 / absent so that a variable added mid-run
        (adaptive EGBO) re-encodes prior recipes correctly: the new variable was
        at zero concentration in every past mixture, which is exactly EGBO's
        'earlier observations remain valid' property.
        """
        vector = []
        for var in self.variables:
            if var['type'] == 'continuous':
                # Missing key => the variable's 'absent' value: 0 for an
                # ingredient (not in the recipe), or a process parameter's
                # baseline (prior batches ran at a fixed setting, not 0).
                absent = var.get('_absent_value', 0.0)
                vector.append(float(recipe_dict.get(var['name'], absent)))
            elif var['type'] == 'categorical':
                chosen = recipe_dict.get(var['name'], None)
                for opt in var['options']:
                    vector.append(1.0 if opt == chosen else 0.0)
        return vector

    def _decode(self, vector):
        """Decode a flat numeric vector back into a recipe dict."""
        recipe = {}
        idx = 0
        for var in self.variables:
            if var['type'] == 'continuous':
                recipe[var['name']] = float(vector[idx])
                idx += 1
            elif var['type'] == 'categorical':
                n_opts = len(var['options'])
                one_hot_segment = vector[idx:idx + n_opts]
                best_idx = np.argmax(one_hot_segment)
                recipe[var['name']] = var['options'][best_idx]
                idx += n_opts
        return recipe

    def _get_bounds(self):
        """Return a (2, dim) tensor of [mins, maxs] for all variables."""
        bounds_min, bounds_max = [], []
        for var in self.variables:
            if var['type'] == 'continuous':
                bounds_min.append(var['bounds'][0])
                bounds_max.append(var['bounds'][1])
            elif var['type'] == 'categorical':
                for _ in var['options']:
                    bounds_min.append(0.0)
                    bounds_max.append(1.0)
        return torch.tensor([bounds_min, bounds_max], dtype=torch.double)

    # ------------------------------------------------------------------ #
    #  Internal: Constraint Helpers
    # ------------------------------------------------------------------ #

    def _get_botorch_constraints(self):
        """Build BoTorch inequality constraints from property + quantity constraints."""
        constraints_list = []
        var_indices = {
            var['name']: i
            for i, var in enumerate(self.variables)
            if var['type'] == 'continuous'
        }

        # Property limits, per 100 g of the finished formulation. The average
        # Σ a·p / Σ a is not linear, but "at most L" rearranges to the linear
        # Σ a × (p − L) ≤ 0, which is what BoTorch can be handed. Every
        # ingredient carries a coefficient, including one with none of the
        # property at all: adding water is how a formulation is diluted.
        for constr in self.constraints:
            metric = constr['metric']
            for bound, sense in (('min', 1.0), ('max', -1.0)):
                if constr[bound] is None:
                    continue
                coeff_of = self._property_coeff(metric, constr[bound])
                indices, coeffs, offset = [], [], 0.0
                for var_name, idx in var_indices.items():
                    coeff = coeff_of(var_name)
                    if coeff == 0:
                        continue
                    var_def = next(v for v in self.variables if v['name'] == var_name)
                    v_min, v_max = float(var_def['bounds'][0]), float(var_def['bounds'][1])
                    if v_max != v_min:
                        indices.append(idx)
                        coeffs.append(sense * coeff * (v_max - v_min))
                    offset += coeff * v_min

                if not indices:
                    continue
                constraints_list.append((
                    torch.tensor(indices, dtype=torch.long),
                    torch.tensor(coeffs, dtype=torch.double),
                    -sense * offset,
                ))

        # Quantity constraints (direct sum of ingredient quantities)
        for qc in getattr(self, 'quantity_constraints', []):
            indices, coeffs = [], []
            offset = 0.0

            for ing_name in qc['ingredients']:
                if ing_name in var_indices:
                    idx = var_indices[ing_name]
                    var_def = next(v for v in self.variables if v['name'] == ing_name)
                    v_min, v_max = var_def['bounds']
                    indices.append(idx)
                    coeffs.append(v_max - v_min)
                    offset += v_min

            if not indices:
                continue
            t_idx = torch.tensor(indices, dtype=torch.long)
            t_coeffs = torch.tensor(coeffs, dtype=torch.double)

            if qc['min'] is not None:
                constraints_list.append((t_idx, t_coeffs, qc['min'] - offset))
            if qc['max'] is not None:
                constraints_list.append((t_idx, -t_coeffs, -(qc['max'] - offset)))

        return constraints_list

    def _check_constraints(self, recipe_dict):
        """Return True if a recipe satisfies all constraints."""
        # A property limit is read per 100 g of the finished formulation, in
        # the linear form the optimizer is given: Σ amount × (property − limit).
        # A tolerance rides on the size of the numbers, so a candidate that
        # lands exactly on the limit is not rejected by the last bit of a float.
        scale = 1.0 + sum(abs(float(a or 0.0)) for a in recipe_dict.values())
        for constr in self.constraints:
            metric = constr['metric']
            if constr['min'] is not None:
                slack = 1e-9 * scale * (1.0 + abs(float(constr['min'])))
                if self._property_residual(recipe_dict, metric,
                                           constr['min']) < -slack:
                    return False
            if constr['max'] is not None:
                slack = 1e-9 * scale * (1.0 + abs(float(constr['max'])))
                if self._property_residual(recipe_dict, metric,
                                           constr['max']) > slack:
                    return False

        # Quantity constraints
        for qc in getattr(self, 'quantity_constraints', []):
            total_val = sum(recipe_dict.get(name, 0.0) for name in qc['ingredients'])
            if qc['min'] is not None and total_val < qc['min']:
                return False
            if qc['max'] is not None and total_val > qc['max']:
                return False

        return True

    # ------------------------------------------------------------------ #
    #  Core Loop: Ask / Tell
    # ------------------------------------------------------------------ #

    def ask(self, n_suggestions=1, n_init_random=5):
        """Suggest the next batch of recipes to try.

        Inactive variables (see deactivate_variable) are held at their frozen
        value: the search runs over the active set only, while the surrogate
        still sees every past observation.
        """
        if not self.active_variables():
            raise ValueError(
                "Everything is paused — resume at least one ingredient before "
                "generating formulations."
            )
        bounds_tensor = self._get_bounds()
        dim = bounds_tensor.shape[1]

        # Cold start: space-filling Sobol sequence
        if len(self.X_history) < n_init_random:
            recipes = self._ask_cold_start(n_suggestions, bounds_tensor, dim)
        else:
            # Warm: GP-based Bayesian optimization
            recipes = self._ask_optimize(n_suggestions, bounds_tensor, dim)

        # Numbers are issued here, at generation, and never reissued.
        self.set_pending_batch(recipes)
        return recipes

    def _ask_seed(self):
        """The seed both regimes draw from. It is the next formulation number,
        NOT the length of the history: numbers are issued at generation and
        never reissued, so it advances on every generate — including one whose
        formulations were discarded without a result. Seeding on the history
        length made `Generate a different batch` hand back the batch it had
        just discarded, byte for byte, in both the cold and the warm regime.
        The same project in the same state still generates the same
        formulations twice: the number has not moved."""
        return int(self.next_formulation_no)

    def _ask_cold_start(self, n_suggestions, bounds_tensor, dim):
        """Generate initial recipes using Sobol sampling."""
        print(f"DEBUG: cold start, Sobol design of {n_suggestions}...")
        sobol = SobolEngine(dimension=dim, scramble=True, seed=self._ask_seed())
        pool_norm = sobol.draw(2048).double()

        # Pin inactive variables so the Sobol design also lives in X_S.
        for col, z in self._get_fixed_features().items():
            pool_norm[:, col] = z

        candidates = [
            self._decode(unnormalize(pool_norm[i], bounds_tensor).numpy().flatten())
            for i in range(2048)
        ]

        # Optional screening model
        if self.screening_model is not None:
            scored = []
            for rec in candidates:
                if self._check_constraints(rec):
                    try:
                        if hasattr(self.screening_model, 'predict'):
                            score = self.screening_model.predict([list(rec.values())])[0]
                        else:
                            score = self.screening_model(rec)
                        scored.append((score, rec))
                    except Exception:
                        pass
            scored.sort(key=lambda x: x[0], reverse=True)
            return [x[1] for x in scored[:n_suggestions]]

        # Standard: pick first feasible candidates
        results = []
        for candidate in candidates:
            if self._check_constraints(candidate):
                results.append(candidate)
            if len(results) >= n_suggestions:
                break

        if not results:
            raise ValueError(
                "No valid formulations found — your limits may be too restrictive. "
                "Try widening the allowed amounts or relaxing limits."
            )
        return results

    def _ask_optimize(self, n_suggestions, bounds_tensor, dim):
        """Generate recipes using a GP + the configured acquisition (default qLogNEI)."""
        print(f"DEBUG: suggestion step, q={n_suggestions}...")
        torch.manual_seed(self._ask_seed())

        train_X = torch.tensor(self.X_history, dtype=torch.double)
        train_Y = torch.tensor(self.Y_history, dtype=torch.double).unsqueeze(-1)
        train_X_norm = normalize(train_X, bounds_tensor)

        input_tf = Warp(d=dim, indices=list(range(dim))) if self.robust else None

        gp = self._build_gp(train_X_norm, train_Y, dim, input_tf)
        mll = ExactMarginalLogLikelihood(gp.likelihood, gp)
        fit_gpytorch_mll(mll)

        acq_func = self._build_acqf(gp, train_X_norm, train_Y)

        # fixed_features drops the inactive columns from the optimization outright
        # (and folds them into the linear constraints), so the candidate is the true
        # argmax over the restricted domain X_S rather than a full-space argmax with
        # coordinates zeroed after the fact.
        candidate_norm, _ = optimize_acqf(
            acq_function=acq_func,
            bounds=torch.stack([torch.zeros(dim), torch.ones(dim)]).double(),
            q=n_suggestions,
            num_restarts=20,
            raw_samples=1024,
            sequential=True,
            inequality_constraints=self._get_botorch_constraints(),
            fixed_features=self._get_fixed_features() or None,
        )

        return [
            self._decode(unnormalize(candidate_norm[i], bounds_tensor).detach().numpy().flatten())
            for i in range(n_suggestions)
        ]

    def tell(self, recipe_dict, results_dict, formulation_no=None,
             batch_no=None, note=None):
        """Record a formulation's amounts and results.

        A measurement that could not be scored is left out (None or absent):
        the formulation is stored as a partial result and its overall score is
        computed from what was measured. A row with nothing measured at all is
        refused. `formulation_no` defaults to the next global number, and
        `batch_no` to the open batch's number. A number passed in explicitly
        still retires, so no later ask() can hand it out again.
        """
        if not self.objectives:
            raise ValueError("Add at least one measurement before saving results.")
        kept = {k: v for k, v in results_dict.items() if v is not None}
        if not any(obj['name'] in kept for obj in self.objectives):
            raise ValueError("Enter a value for at least one measurement.")

        if formulation_no is None:
            formulation_no = self._issue_formulation_no()
        else:
            self._retire_formulation_no(formulation_no)
        if batch_no is None:
            batch_no = self.pending_batch_no

        self.X_history.append(self._encode(recipe_dict))
        self.Y_history.append(self._compute_utility(kept))
        self.recipe_history.append(dict(recipe_dict))
        self.results_history.append(kept)
        self.timestamps_history.append(datetime.now(timezone.utc).isoformat())
        self.formulation_ids.append(int(formulation_no))
        self.batch_history.append(None if batch_no is None else int(batch_no))
        self.notes_history.append("" if note is None else str(note))
        self.save()

    # ------------------------------------------------------------------ #
    #  Identity: formulation numbers, batch numbers, the open batch
    # ------------------------------------------------------------------ #

    def _drop_pending_batch(self):
        """Forget the open batch. The numbers it used retire — they are never
        reissued, so a discarded batch can never be confused with a later one."""
        self.pending_batch = None
        self.pending_batch_no = None
        self.pending_batch_created = None
        self.pending_batch_discarded = []

    def _issue_formulation_no(self):
        n = int(self.next_formulation_no)
        self.next_formulation_no = n + 1
        return n

    def _retire_formulation_no(self, no):
        """Make sure a number handed in from outside can never be issued again."""
        self.next_formulation_no = max(int(self.next_formulation_no), int(no) + 1)

    def next_batch_no(self):
        """The number the next batch will carry: one past the highest in use."""
        seen = [int(b) for b in self.batch_history if b is not None]
        seen += [int(s['batch']) for s in self.skipped if s.get('batch') is not None]
        if self.pending_batch_no is not None:
            seen.append(int(self.pending_batch_no))
        return (max(seen) + 1) if seen else 1

    def _batch_rows(self, batch):
        """Read a batch in either shape without issuing numbers: the stored
        [{'formulation': n, 'recipe': {...}}, ...] or a bare list of recipes
        (which a 0.2.x file's pending_batch is)."""
        rows = []
        for k, item in enumerate(batch or []):
            if isinstance(item, dict) and 'recipe' in item and 'formulation' in item:
                rows.append(self._batch_row(int(item['formulation']),
                                            item['recipe'], item.get('note')))
            else:
                rows.append(self._batch_row(k + 1, item, None))
        return rows

    @staticmethod
    def _batch_row(number, recipe, note=None):
        """One row of the open batch. The note is kept only when there is one:
        it is how a repeat of the best formulation says so, on the make-these
        table and in the note stored with the result."""
        row = {'formulation': int(number), 'recipe': dict(recipe)}
        if note:
            row['note'] = str(note)
        return row

    def _number_batch(self, batch):
        """Like _batch_rows, but rows that carry no number draw one."""
        if batch is None:
            return None
        rows = []
        for item in batch:
            if isinstance(item, dict) and 'recipe' in item and 'formulation' in item:
                rows.append(self._batch_row(int(item['formulation']),
                                            item['recipe'], item.get('note')))
            else:
                rows.append(self._batch_row(self._issue_formulation_no(),
                                            item, None))
        return rows

    def add_to_pending_batch(self, recipe, note=""):
        """Append one more formulation to the open batch (a formulation of the
        user's own) and return the global number it was given. `note` marks
        what the row is, so the extra formulation is not an unexplained row.

        With no batch open this opens one, exactly as ask() does: a
        formulation of your own can be the first in a batch, and the date it
        was opened on is what the printable sheets carry."""
        rows = list(self.pending_batch or [])
        number = self._issue_formulation_no()
        rows.append(self._batch_row(number, recipe, note))
        self.pending_batch = rows
        if self.pending_batch_no is None:
            self.pending_batch_no = self.next_batch_no()
        if self.pending_batch_created is None:
            self.pending_batch_created = datetime.now().astimezone().strftime("%Y-%m-%d")
        self.save()
        return number

    def index_of_formulation(self, no):
        """Position of formulation `no` in the scored history, or None."""
        try:
            return [int(i) for i in self.formulation_ids].index(int(no))
        except (ValueError, TypeError):
            return None

    def record_skipped(self, formulation_no, batch_no, recipe, note="Not made"):
        """Store a formulation that was generated but never scored. It keeps
        its number and amounts, and stays out of the scored history and the
        model."""
        self.skipped.append({
            'formulation': int(formulation_no),
            'batch': None if batch_no is None else int(batch_no),
            'recipe': dict(recipe),
            'note': str(note) if note else "Not made",
        })
        self._retire_formulation_no(formulation_no)
        self.save()

    def import_formulation(self, recipe_dict, results_dict, note="Imported"):
        """Record a formulation made before this project existed. It draws the
        next global number, its batch stays blank (it belongs to no batch this
        project generated), and the note says where it came from."""
        self.tell(recipe_dict, results_dict, note=note)
        self.batch_history[-1] = None    # tell() would inherit the open batch
        self.save()

    def delete_formulation(self, no):
        """Delete one formulation by its global number, scored or left out.
        Later formulations keep their numbers; the deleted number retires.
        Returns True when something was deleted."""
        index = self.index_of_formulation(no)
        if index is not None:
            self.delete_result(index)
            return True
        before = len(self.skipped)
        self.skipped = [s for s in self.skipped
                        if int(s['formulation']) != int(no)]
        if len(self.skipped) != before:
            self.save()
            return True
        return False

    def delete_formulations(self, numbers):
        """Delete several formulations by number, recorded or not made, in one
        save. Returns how many went.

        The scored rows go highest position first: every list here is
        parallel, so deleting position 1 before position 3 would shift the
        row out from under the second delete and take a formulation nobody
        picked."""
        wanted = {int(n) for n in numbers}
        positions = sorted(
            (i for i in (self.index_of_formulation(n) for n in wanted)
             if i is not None), reverse=True)
        gone = 0
        for index in positions:
            self._drop_result(index)
            gone += 1
        before = len(self.skipped)
        self.skipped = [s for s in self.skipped
                        if int(s['formulation']) not in wanted]
        gone += before - len(self.skipped)
        if gone:
            self.save()
        return gone

    def last_batch_no(self):
        """The highest batch number in the recorded history. Left-out
        formulations count: a batch nobody managed to make is still the last
        batch, and undo has to be able to reach it."""
        seen = [int(b) for b in self.batch_history if b is not None]
        seen += [int(s['batch']) for s in self.skipped
                 if s.get('batch') is not None]
        return max(seen) if seen else None

    def undo_last_batch(self):
        """Remove the most recently recorded batch: its scored rows and its
        left-out formulations. Returns (batch number, rows removed), or None
        when no batch is numbered. next_formulation_no is NOT wound back, so
        the numbers retire rather than coming back on the next batch.

        Refused while a batch is open: taking that batch down as a side effect
        would retire numbers the user never asked to discard."""
        if self.pending_batch is not None:
            raise ValueError(wording.batch_open_record_first_caption())
        last = self.last_batch_no()
        if last is None:
            return None
        keep = [i for i, b in enumerate(self.batch_history) if b != last]
        removed = len(self.batch_history) - len(keep)
        self.X_history = [self.X_history[i] for i in keep]
        self.Y_history = [self.Y_history[i] for i in keep]
        self.recipe_history = [self.recipe_history[i] for i in keep]
        self.results_history = [self.results_history[i] for i in keep]
        self.timestamps_history = [self.timestamps_history[i] for i in keep]
        self.formulation_ids = [self.formulation_ids[i] for i in keep]
        self.notes_history = [self.notes_history[i] for i in keep]
        self.batch_history = [self.batch_history[i] for i in keep]
        removed += sum(1 for s in self.skipped if s.get('batch') == last)
        self.skipped = [s for s in self.skipped if s.get('batch') != last]
        self.save()
        return last, removed

    def _backfill_identity(self):
        """Give a 0.2.x project the identity it never had: its rows are
        numbered 1..n in the order they were recorded, their batch is blank,
        and the counter starts after the highest number in use."""
        n = len(self.X_history)
        while len(self.formulation_ids) < n:
            highest = max([int(i) for i in self.formulation_ids] or [0])
            self.formulation_ids.append(highest + 1)
        del self.formulation_ids[n:]
        while len(self.batch_history) < n:
            self.batch_history.append(None)
        del self.batch_history[n:]
        while len(self.notes_history) < n:
            self.notes_history.append("")
        del self.notes_history[n:]
        used = [int(i) for i in self.formulation_ids]
        used += [int(s['formulation']) for s in self.skipped
                 if s.get('formulation') is not None]
        used += [int(r['formulation']) for r in (self.pending_batch or [])
                 if isinstance(r, dict) and 'formulation' in r]
        highest = max(used) if used else 0
        self.next_formulation_no = max(int(self.next_formulation_no or 1), highest + 1)

    def set_pending_batch(self, batch_or_none, batch_no=None, discarded=None):
        """Persist (or clear) the open batch so a user who closes the window
        mid-batch finds their formulations on return. Rows without a number
        draw one. Only one batch is ever open. `discarded` records the numbers
        a regenerate retired, so the screen can say so once."""
        if batch_or_none is None:
            self._drop_pending_batch()
        else:
            rows = self._number_batch(batch_or_none)
            if batch_no is not None:
                self.pending_batch_no = int(batch_no)
            elif self.pending_batch_no is None:
                self.pending_batch_no = self.next_batch_no()
            if self.pending_batch_created is None:
                self.pending_batch_created = datetime.now().astimezone().strftime("%Y-%m-%d")
            if discarded is not None:
                self.pending_batch_discarded = [int(n) for n in discarded]
            self.pending_batch = rows
        self.save()

    # ------------------------------------------------------------------ #
    #  Adaptivity + expert BO config (optional arms 2 & 3)
    # ------------------------------------------------------------------ #

    def _reencode_history(self):
        """Rebuild X_history from recipe_history under the current variable set.
        Call after any change to self.variables. recipe_history holds the raw
        dicts, so this stays lossless for prior experiments."""
        if self.recipe_history:
            self.X_history = [self._encode(r) for r in self.recipe_history]

    # ------------------------------------------------------------------ #
    #  Non-monotone active set: deactivate / reactivate / remove
    #
    #  Standard EGBO grows the active set monotonically (S_1 <= S_2 <= ...),
    #  which means expert false positives accumulate and never leave: the
    #  expected active-set size tends to the full ambient dimension as the
    #  number of expert rounds grows, so adaptive EGBO degenerates towards
    #  vanilla BO. Allowing the expert to prune gives the active set a bounded
    #  steady state instead.
    #
    #  Pruning is implemented as *deactivation*, not deletion:
    #    - the variable keeps its column in the encoded history, so every past
    #      observation stays in the GP (a recorded experiment is still a valid
    #      observation of f — it is only outside the current search domain);
    #    - the acquisition function is maximized over the active set only;
    #    - reactivation is free, which is what a non-monotone active set needs.
    # ------------------------------------------------------------------ #

    def active_variables(self):
        """Variables in the current active set S_r."""
        return [v for v in self.variables if v.get('active', True)]

    def inactive_variables(self):
        """Variables pruned from S_r but still carried in the history/GP."""
        return [v for v in self.variables if not v.get('active', True)]

    def _var_by_name(self, name):
        for var in self.variables:
            if var['name'] == name:
                return var
        raise ValueError(f"No variable named {name!r}.")

    def _frozen_value(self, var):
        """The value an inactive variable is held at during search."""
        lo, hi = float(var['bounds'][0]), float(var['bounds'][1])
        for key in ('_frozen_at', '_absent_value'):
            if key in var:
                return float(var[key])
        if var.get('category') == 'process':
            return lo  # a process parameter has no meaningful 'off' state
        return min(max(0.0, lo), hi)

    def _achievable_range(self, coeff_of, pinned):
        """Range of sum_i coeff_i * x_i attainable when `pinned` variables are held
        fixed and the rest range over their bounds. Handles negative coefficients."""
        lo = hi = 0.0
        for var in self.variables:
            if var['type'] != 'continuous':
                continue
            coeff = coeff_of(var['name'])
            if coeff == 0:
                continue
            if var['name'] in pinned:
                lo += coeff * pinned[var['name']]
                hi += coeff * pinned[var['name']]
            else:
                a = coeff * float(var['bounds'][0])
                b = coeff * float(var['bounds'][1])
                lo += min(a, b)
                hi += max(a, b)
        return lo, hi

    def _achievable_property(self, metric, pinned):
        """The lowest and the highest one property can be per 100 g of the
        finished formulation, with `pinned` variables held fixed and the rest
        free within their allowed amounts.

        An average is a ratio, so it is not read off the bounds the way a sum
        is. It is found by asking of each candidate value whether any
        formulation sits under it — Σ amount × (property − value) < 0, the
        same linear form the limit itself takes — and narrowing the answer.
        Every achievable average lies between the lowest and the highest
        property value on the shelf, which is where the search starts."""
        values = [self.property_value(v['name'], metric) for v in self.variables
                  if v.get('category', 'ingredient') == 'ingredient']
        if not values:
            return 0.0, 0.0
        low, high = min(values), max(values)
        if high - low <= 1e-12:
            return low, high

        def span(value):
            return self._achievable_range(self._property_coeff(metric, value),
                                          pinned)

        # The lowest achievable average: the smallest value some formulation
        # can sit under.
        if span(high)[0] >= 0:
            lowest = high
        else:
            under, over = high, low        # under: reachable, over: not
            for _ in range(60):
                mid = (under + over) / 2.0
                if span(mid)[0] < 0:
                    under = mid
                else:
                    over = mid
            lowest = under
        # ...and the highest: the largest value some formulation can sit over.
        if span(low)[1] <= 0:
            highest = low
        else:
            over, under = low, high
            for _ in range(60):
                mid = (over + under) / 2.0
                if span(mid)[1] > 0:
                    over = mid
                else:
                    under = mid
            highest = over
        return lowest, highest

    def _assert_constraints_satisfiable(self, pinned):
        """Raise if holding `pinned` ({name: value}) fixed makes any constraint
        unsatisfiable. Pinning an ingredient to 0 can strand a lower bound that
        the ingredient was carrying, which would otherwise surface later as an
        opaque acquisition-optimization failure."""
        per = self.per_amount_text()
        for constr in self.constraints:
            metric = constr['metric']
            lo, hi = self._achievable_property(metric, pinned)
            if constr['min'] is not None and hi < constr['min']:
                raise ValueError(
                    f"Pausing these would make the limit on {metric} impossible "
                    f"to meet: the remaining active ingredients can only reach "
                    f"{hi:.4g} {per} at most. Loosen the limit first."
                )
            if constr['max'] is not None and lo > constr['max']:
                raise ValueError(
                    f"Pausing these would make the limit on {metric} impossible "
                    f"to meet: the remaining active ingredients cannot get "
                    f"below {lo:.4g} {per}. Loosen the limit first."
                )

        for i, qc in enumerate(getattr(self, 'quantity_constraints', [])):
            names = set(qc['ingredients'])
            label = " + ".join(qc['ingredients'])
            lo, hi = self._achievable_range(lambda n: 1.0 if n in names else 0.0, pinned)
            if qc['min'] is not None and hi < qc['min']:
                raise ValueError(
                    f"Pausing these would make the limit on {label} impossible "
                    f"to meet: the remaining active ingredients can only reach "
                    f"{hi:.4g} at most. Loosen the limit first."
                )
            if qc['max'] is not None and lo > qc['max']:
                raise ValueError(
                    f"Pausing these would make the limit on {label} impossible "
                    f"to meet: the paused items alone add up to {lo:.4g}. "
                    f"Loosen the limit first."
                )

    def _get_fixed_features(self):
        """{column: normalized_value} for the inactive columns, in the [0,1]^d frame
        that `ask` hands to optimize_acqf. Returns {} when everything is active, so
        the standard monotone behavior is byte-identical."""
        fixed = {}
        col = 0
        for var in self.variables:
            if var['type'] == 'continuous':
                if not var.get('active', True):
                    lo, hi = float(var['bounds'][0]), float(var['bounds'][1])
                    span = hi - lo
                    z = 0.0 if span <= 0 else (self._frozen_value(var) - lo) / span
                    fixed[col] = float(min(max(z, 0.0), 1.0))
                col += 1
            elif var['type'] == 'categorical':
                n_opts = len(var['options'])
                if not var.get('active', True):
                    for j in range(n_opts):
                        fixed[col + j] = 0.0
                col += n_opts
        return fixed

    def deactivate_variable(self, name, value=None):
        """Prune `name` from the active set, pinning it at `value` (default: 0 for
        an ingredient, the lower bound or stored baseline for a process parameter).

        Nothing is destroyed: past experiments keep contributing to the GP, and
        reactivate_variable puts the variable back. Raises if pruning would make a
        constraint unsatisfiable or would empty the active set.
        """
        var = self._var_by_name(name)
        if not var.get('active', True):
            return
        if len(self.active_variables()) <= 1:
            raise ValueError(
                "At least one ingredient or process setting must stay active."
            )
        frozen = self._frozen_value(var) if value is None else float(value)
        lo, hi = float(var['bounds'][0]), float(var['bounds'][1])
        if not (lo <= frozen <= hi):
            raise ValueError(
                f"Frozen value {frozen} for '{name}' must lie within [{lo}, {hi}]."
            )

        pinned = {v['name']: self._frozen_value(v) for v in self.inactive_variables()}
        pinned[name] = frozen
        self._assert_constraints_satisfiable(pinned)

        var['active'] = False
        var['_frozen_at'] = frozen
        self._drop_pending_batch()
        self.save()

    def reactivate_variable(self, name):
        """Return a pruned variable to the active set (S_r is non-monotone, so a
        variable removed in one round may be re-added in a later one)."""
        var = self._var_by_name(name)
        if var.get('active', True):
            return
        var['active'] = True
        var.pop('_frozen_at', None)
        self._drop_pending_batch()
        self.save()

    def remove_ingredient(self, name, force=False):
        """Permanently delete an ingredient and drop its column from the history.

        Refuses by default if the ingredient was ever used at a nonzero amount,
        because dropping its column silently rewrites those experiments into
        recipes that were never run. Prefer deactivate_variable, which keeps the
        data. force=True deletes anyway and discards that information.
        """
        var = self._var_by_name(name)
        if var.get('category', 'ingredient') != 'ingredient':
            raise ValueError(
                f"'{name}' is a process setting. Use Delete next to the "
                f"process setting instead."
            )
        if self.X_history and len(self.recipe_history) != len(self.X_history):
            raise ValueError(
                "Cannot delete this: some formulations were recorded without "
                "their amounts, so the history cannot be rebuilt. Pause it "
                "instead, or start a fresh project."
            )

        used = [
            i for i, r in enumerate(self.recipe_history)
            if float(r.get(name, 0.0)) != 0.0
        ]
        if used and not force:
            # Name the formulations, not row indexes: a formulation number is
            # what the user wrote on the sheet, and #0 is not a thing they own.
            numbers = [int(self.formulation_ids[i]) if i < len(self.formulation_ids)
                       else i + 1 for i in used]
            word = "Formulation" if len(numbers) == 1 else "Formulations"
            raise ValueError(
                f"'{name}' was used in {word} {number_list(numbers)}, so it "
                f"cannot be deleted. Tick 'Delete even if it was used' to "
                f"discard that information."
            )

        remaining = [v for v in self.variables if v['name'] != name]
        if not any(v.get('active', True) for v in remaining):
            raise ValueError("Cannot delete the last active variable.")

        self.variables = remaining
        self.ingredient_properties.pop(name, None)
        for recipe in self.recipe_history:
            recipe.pop(name, None)

        kept = []
        for qc in getattr(self, 'quantity_constraints', []):
            qc['ingredients'] = [n for n in qc['ingredients'] if n != name]
            if qc['ingredients']:
                kept.append(qc)
        self.quantity_constraints = kept

        self._reencode_history()
        self._drop_pending_batch()
        self.save()

    def set_bo_config(self, spec):
        """Set expert-selected BO hyperparameters (arm 3). Pass None/{} for the
        library defaults (arm 1). Chosen once at project start, not per iteration."""
        self.bo_config = validate_bo_config(spec)
        self.save()

    def fork(self, new_project_name):
        """Branch the current state into a new project, saved under a new name.
        Used to split one run into adaptive vs non-adaptive at the first re-query:
        both share an identical pre-fork history."""
        clone = FoodOptimizer(new_project_name, storage=self.storage)
        state = self.export_json()
        state['project_name'] = new_project_name
        clone.import_json(state)
        clone.screening_model = None
        clone.save()
        return clone

    def export_trajectory(self):
        """Human-readable optimization trajectory for an expert re-query (adaptive
        arm). The app appends the 'available to add' pool (full CSV minus active)."""
        if not self.Y_history:
            return "No formulations recorded yet."
        best_i = int(np.argmax(self.Y_history))
        active = [v['name'] for v in self.active_variables()]
        inactive = [
            f"{v['name']} (fixed at {self._frozen_value(v):.3g})"
            for v in self.inactive_variables()
        ]
        all_names = [v['name'] for v in self.variables]
        lines = [f"In play ({len(active)}): {', '.join(active)}"]
        if inactive:
            lines.append(f"Paused ({len(inactive)}): {', '.join(inactive)}")
        lines.append("")
        for i, y in enumerate(self.Y_history):
            rec = self.recipe_history[i] if i < len(self.recipe_history) else {}
            # Show every variable that was actually used, including since-paused
            # ones, so the expert can see what a paused variable contributed.
            comp = ", ".join(f"{k}={rec[k]:.3g}" for k in all_names if rec.get(k))
            res = self.results_history[i] if i < len(self.results_history) else {}
            attrs = ", ".join(f"{k}={v:.3g}" for k, v in res.items())
            mark = "*" if i == best_i else " "
            lines.append(
                f"{mark} Formulation {self.formulation_ids[i]}: score {y:.3f} | "
                f"{comp} | results: {attrs}"
            )
        return "\n".join(lines)

    def _build_gp(self, train_X_norm, train_Y, dim, input_tf):
        """Build the GP. With bo_config == None this is byte-identical to the
        original default GP; a config swaps in the expert's kernel/noise."""
        cfg = getattr(self, 'bo_config', None)
        if not cfg:
            return SingleTaskGP(
                train_X_norm, train_Y,
                outcome_transform=Standardize(m=1), input_transform=input_tf,
            )
        kwargs = dict(
            outcome_transform=Standardize(m=1), input_transform=input_tf,
            covar_module=_build_covar(cfg, dim),
        )
        if cfg.get('noise') == 'fixed_tiny':
            yvar = float(max(1e-8, 1e-6 * (train_Y.var().item() + 1e-12)))
            return SingleTaskGP(
                train_X_norm, train_Y,
                train_Yvar=torch.full_like(train_Y, yvar), **kwargs,
            )
        gp = SingleTaskGP(train_X_norm, train_Y, **kwargs)
        if cfg.get('noise') == 'low':
            gp.likelihood.noise_covar.register_prior(
                'noise_prior', GammaPrior(1.1, 50.0), 'raw_noise',
            )
        return gp

    def _build_acqf(self, gp, train_X_norm, train_Y):
        cfg = getattr(self, 'bo_config', None)
        acq = (cfg or {}).get('acquisition', 'qlognei')
        sampler = SobolQMCNormalSampler(sample_shape=torch.Size([512]))
        if acq == 'qlogei':
            return qLogExpectedImprovement(
                model=gp, best_f=float(train_Y.max().item()), sampler=sampler,
            )
        if acq == 'qucb':
            return qUpperConfidenceBound(model=gp, beta=0.2, sampler=sampler)
        return qLogNoisyExpectedImprovement(
            model=gp, X_baseline=train_X_norm, sampler=sampler,
        )

    # ------------------------------------------------------------------ #
    #  Persistence: Save / Load / Export / Import
    # ------------------------------------------------------------------ #

    def save(self):
        # Delegates to the storage backend. StorageError (a cloud backend
        # failure) is swallowed into save_error so the user's in-memory work
        # survives; the app shows it as a banner. Local I/O errors propagate.
        try:
            self.storage.save(self.project_name, self.export_json())
        except StorageError as e:
            self.save_error = str(e)
        else:
            self.save_error = None
            self.last_saved_at = datetime.now().astimezone()

    def load(self):
        """Load the project via the storage backend. Sets self.load_error to a
        plain-language message on failure instead of silently producing a
        blank project. Returns True on success, False on failure."""
        self.load_error = None
        try:
            state = self.storage.load(self.project_name)
        except StorageError as e:
            self.load_error = str(e)
            return False
        if state is None:
            self.load_error = (
                "This project could not be found. It may have been renamed "
                "or archived."
            )
            return False
        try:
            self.import_json(state)
        except Exception:
            self.load_error = (
                "This project file is damaged and could not be opened. If you "
                "have a backup, use Restore from backup; otherwise look in your "
                "FoodOptimizer folder for a recent copy."
            )
            return False
        # Re-save only when the file is behind the current CLASS_VERSION, so an
        # older JSON file is brought up to date once. Up-to-date files are not
        # rewritten: rewriting on every open would bump the mtime and make
        # other open windows see a false conflict. Legacy pickle files are no
        # longer readable (LocalStorage.load refuses them).
        _ver = state.get('CLASS_VERSION', 0)
        if not isinstance(_ver, int):
            _ver = 0
        if self.storage.persist_after_load and _ver < self.CLASS_VERSION:
            self.save()
        return True

    def export_json(self):
        """Export full project state as a JSON-serializable dict."""

        def _make_serializable(obj):
            if isinstance(obj, tuple):
                return list(obj)
            if isinstance(obj, np.integer):
                return int(obj)
            if isinstance(obj, np.floating):
                return float(obj)
            if isinstance(obj, np.ndarray):
                return obj.tolist()
            return obj

        state = {
            'project_name': self.project_name,
            'variables': self.variables,
            'objectives': self.objectives,
            'ingredient_properties': self.ingredient_properties,
            'property_names': getattr(self, 'property_names', None) or [],
            'constraints': self.constraints,
            'quantity_constraints': self.quantity_constraints,
            'robust': self.robust,
            'X_history': self.X_history,
            'Y_history': self.Y_history,
            'recipe_history': self.recipe_history,
            'results_history': self.results_history,
            'timestamps_history': self.timestamps_history,
            'formulation_ids': self.formulation_ids,
            'batch_history': self.batch_history,
            'notes_history': self.notes_history,
            'skipped': self.skipped,
            'next_formulation_no': self.next_formulation_no,
            'pending_batch_no': self.pending_batch_no,
            'pending_batch_created': self.pending_batch_created,
            'pending_batch_discarded': self.pending_batch_discarded,
            'amount_unit': self.amount_unit,
            'pending_batch': self.pending_batch,
            'bo_config': self.bo_config,
            'CLASS_VERSION': self.CLASS_VERSION,
        }
        return json.loads(json.dumps(state, default=_make_serializable))

    @staticmethod
    def validate_state(state):
        """Check a backup dict before importing it. Returns a summary dict
        (name, experiments, ingredients, version) or raises ValueError with a
        message suitable for the UI. import_json assigns attributes one by
        one, so validating first is what keeps a bad file from leaving the
        optimizer half-mutated."""
        bad = "This file is not a Food Optimizer backup."
        if not isinstance(state, dict):
            raise ValueError(bad)
        required = {
            'variables': list, 'objectives': list,
            'recipe_history': list, 'results_history': list,
        }
        if not all(k in state for k in required):
            raise ValueError(bad)
        for key, typ in required.items():
            if not isinstance(state[key], typ):
                raise ValueError(f"This backup's '{key}' section has the wrong shape.")
        for key in ('variables', 'objectives'):
            for item in state[key]:
                if not isinstance(item, dict) or not isinstance(item.get('name'), str):
                    raise ValueError(f"This backup's '{key}' section has the wrong shape.")
        for key in ('recipe_history', 'results_history'):
            for item in state[key]:
                if not isinstance(item, dict):
                    raise ValueError(f"This backup's '{key}' section has the wrong shape.")
        version = state.get('CLASS_VERSION')
        if not isinstance(version, int):
            raise ValueError(bad)
        if version > FoodOptimizer.CLASS_VERSION:
            raise ValueError(
                "This backup was made with a newer version of Food Optimizer. "
                "Update the app, then try again."
            )
        if len(state['recipe_history']) != len(state['results_history']):
            raise ValueError("This backup is inconsistent: formulations and results differ in count.")
        # The 0.3.0 identity lists. They are optional (a 0.2.x file has none),
        # but a present-and-malformed one must be refused here: import_json
        # assigns attributes one by one, so a TypeError raised halfway through
        # would leave the optimizer wearing half of a bad backup.
        def _whole(x):
            return isinstance(x, int) and not isinstance(x, bool)

        def _number(x):
            """A formulation number: whole and at least 1. This app has never
            issued a 0 or a −1, and a row numbered −1 renders as
            'Formulation -1' for ever after."""
            return _whole(x) and x >= 1

        def _left_out(x):
            """A left-out formulation carries its number, its batch, its
            amounts and its note. history_frame reads all four on every
            render, so one missing key would load, save, and then break the
            All formulations table for good."""
            return (isinstance(x, dict)
                    and _number(x.get('formulation'))
                    and (x.get('batch') is None or _whole(x.get('batch')))
                    and isinstance(x.get('recipe'), dict)
                    and isinstance(x.get('note', ""), str))

        identity = {
            'formulation_ids': _number,
            'batch_history': lambda x: x is None or _whole(x),
            # None is accepted as an ELEMENT: import_json coerces it to "",
            # so an older or hand-edited backup with a blank note restores.
            'notes_history': lambda x: x is None or isinstance(x, str),
            'skipped': _left_out,
        }
        for key, ok in identity.items():
            if key not in state:
                continue
            # A section that is present but null is refused, not skipped:
            # import_json iterates it and would raise a TypeError halfway
            # through, leaving the optimizer wearing half of a bad backup.
            if not isinstance(state[key], list) or not all(ok(i) for i in state[key]):
                raise ValueError(f"This backup's '{key}' section has the wrong shape.")
        if state.get('next_formulation_no') is not None and not _whole(
                state['next_formulation_no']):
            raise ValueError(
                "This backup's 'next_formulation_no' section has the wrong shape.")
        # Properties named in the app. A malformed list would reach the
        # property picker and the limits list, so it is refused here.
        names = state.get('property_names')
        if names is not None and (not isinstance(names, list)
                                  or not all(isinstance(n, str) for n in names)):
            raise ValueError(
                "This backup's 'property_names' section has the wrong shape.")
        pending = state.get('pending_batch')
        if pending is not None and not isinstance(pending, list):
            raise ValueError("This backup's 'pending_batch' section has the wrong shape.")
        for item in pending or []:
            if (isinstance(item, dict) and item.get('formulation') is not None
                    and not _number(item['formulation'])):
                raise ValueError(
                    "This backup's 'pending_batch' section has the wrong shape.")
        # A formulation's number is permanent and never reissued. A file that
        # numbers two rows the same breaks that for good — index_of_formulation
        # finds only the first, so deleting one leaves the others behind — and
        # a number the counter never reached would be handed out again.
        stored = [int(n) for n in state.get('formulation_ids') or []]
        stored += [int(s['formulation']) for s in state.get('skipped') or []]
        # The open batch is not a third list of formulations: a batch recorded
        # one sheet at a time keeps its recorded rows, so a pending row may
        # legitimately carry a number the history already holds. Only repeats
        # WITHIN the batch are a fault.
        in_batch = [int(r['formulation']) for r in pending or []
                    if isinstance(r, dict) and r.get('formulation') is not None]
        if len(set(stored)) != len(stored) or len(set(in_batch)) != len(in_batch):
            raise ValueError(
                "This backup gives two formulations the same number, and a "
                "formulation number is permanent. It cannot be restored.")
        counter = state.get('next_formulation_no')
        if _whole(counter) and any(n >= counter for n in stored + in_batch):
            raise ValueError(
                "This backup holds a formulation number its own counter never "
                "issued. It cannot be restored.")
        ingredients = sum(
            1 for v in state['variables']
            if isinstance(v, dict) and v.get('category', 'ingredient') == 'ingredient'
        )
        settings = sum(
            1 for v in state['variables']
            if isinstance(v, dict) and v.get('category') == 'process'
        )
        return {
            'name': state.get('project_name', '(unnamed)'),
            'experiments': len(state['recipe_history']),
            # Everything the backup holds a number for. A left-out formulation
            # keeps its number, its amounts and its note, so a warning that
            # counts only the scored ones undercounts what it is offering.
            'formulations': len(state['recipe_history'])
                            + len(state.get('skipped') or []),
            'ingredients': ingredients,
            'settings': settings,
            'version': version,
        }

    def import_json(self, state):
        """Restore project state from a JSON dict (as produced by export_json)."""
        self.project_name = state.get('project_name', self.project_name)
        self.filename = f"{self.project_name}.pkl"
        self.variables = state.get('variables', [])
        self.objectives = state.get('objectives', [])
        self.ingredient_properties = state.get('ingredient_properties', {})
        self.property_names = [str(p).strip()
                               for p in (state.get('property_names') or [])]
        self.constraints = state.get('constraints', [])
        self.quantity_constraints = state.get('quantity_constraints', [])
        self.robust = state.get('robust', False)
        self.bo_config = validate_bo_config(state.get('bo_config', None))
        self.recipe_history = state.get('recipe_history', [])
        self.results_history = state.get('results_history', [])
        self.timestamps_history = state.get('timestamps_history', [])
        self.formulation_ids = [int(i) for i in state.get('formulation_ids', [])]
        self.batch_history = [None if b is None else int(b)
                              for b in state.get('batch_history', [])]
        self.notes_history = ["" if t is None else str(t)
                              for t in state.get('notes_history', [])]
        self.skipped = [dict(s) for s in state.get('skipped', [])]
        self.next_formulation_no = int(state.get('next_formulation_no', 1) or 1)
        # A 0.2.x file has no unit at all: it backfills to g rather than
        # reopening as a project whose amounts mean nothing, and says so once
        # on tab 1. A file that HOLDS a blank unit is a deliberate blank and
        # is kept — the key's presence, not its truthiness, is the question.
        self.amount_unit_backfilled = 'amount_unit' not in state
        self.amount_unit = ("g" if self.amount_unit_backfilled
                            else str(state.get('amount_unit') or ""))
        self.pending_batch = state.get('pending_batch', None)
        self.pending_batch_no = state.get('pending_batch_no', None)
        self.pending_batch_created = state.get('pending_batch_created', None)
        self.pending_batch_discarded = [
            int(n) for n in (state.get('pending_batch_discarded') or [])
        ]
        while len(self.timestamps_history) < len(self.results_history):
            self.timestamps_history.append(None)  # pre-feature files/backups

        for var in self.variables:
            if 'bounds' in var and isinstance(var['bounds'], list):
                var['bounds'] = tuple(var['bounds'])
            var.setdefault('category', 'ingredient')
            var.setdefault('active', True)
            if var['category'] == 'process':
                # A 0.2.x setting was stored before settings could carry a
                # unit; blank is what it had, and blank is what it keeps.
                var.setdefault('unit', "")

        for obj in self.objectives:
            obj.setdefault('unit', "")

        # Rebuild encoded vectors and utility scores from raw data
        if self.recipe_history and self.variables:
            self.X_history = [self._encode(r) for r in self.recipe_history]
        else:
            self.X_history = state.get('X_history', [])

        if self.results_history and self.objectives:
            self.Y_history = [self._compute_utility(r) for r in self.results_history]
        else:
            self.Y_history = state.get('Y_history', [])

        # A successful import means the in-memory state is valid again, so a
        # restore-from-backup clears any earlier damaged-file error.
        self.load_error = None

        # Identity last: the counter must be correct before a 0.2.x pending
        # batch of bare recipes draws its numbers, or it would reuse numbers
        # the backfilled history already owns.
        self._backfill_identity()
        self.pending_batch = self._number_batch(self.pending_batch)
        if self.pending_batch and self.pending_batch_no is None:
            self.pending_batch_no = self.next_batch_no()
        if self.pending_batch_no is not None:
            self.pending_batch_no = int(self.pending_batch_no)
