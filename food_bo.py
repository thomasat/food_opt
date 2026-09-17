import contextlib
import hashlib
import io
import json
import logging
import re
from collections import namedtuple
from datetime import datetime, timezone

import numpy as np
import pandas as pd
import torch
from openpyxl import Workbook
from openpyxl.styles import (Alignment, Border, Font, PatternFill, Protection,
                             Side)
from openpyxl.utils import get_column_letter
from openpyxl.worksheet.properties import PageSetupProperties
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


# The stored name of the round a recorded formulation belongs to. It is
# spelled the old way — `batch` — because stored field names never change, and
# it lives here rather than as a bare literal in every file that reads the
# record: a one-word "batch" or "Batch" in a screen module is a label, and the
# vocabulary guard refuses one there. EVERY reader uses it, this file
# included: spelled two ways, a grep for the key stopped finding half of
# the code that reads it.
ROUND_FIELD = "batch"


# Column names history_frame() (and any future export) reserves for itself;
# a variable with one of these names would silently overwrite that column.
RESERVED_VARIABLE_NAMES = {
    "Experiment", "Date", "Overall Score", "Recipe",
    "Formulation", "Round", "Batch", "Trial", wording.OVERALL_SCORE_COLUMN, "Note",
    # Both spellings of the two columns that were renamed in 0.5.0: a
    # project made before the rename may hold a variable of the old name,
    # and the new name has to be reserved from today on.
    "Recorded", wording.DATE_RECORDED_COLUMN,
    "Best", wording.BEST_SO_FAR_COLUMN, "Total",
    # The workbook's own row labels. An ingredient named "Not scored" put an
    # amount on the row the upload reads the tick off, and the whole
    # formulation came back as a row nobody scored.
    wording.NOT_SCORED, wording.MEASURED_COLUMN,
    # The properties grid's row column (spec 1.5). Its other columns are the
    # project's own property names, so a property — or an ingredient — of
    # this name would put two columns of one name on one grid.
    wording.PROPERTIES_ROW_COLUMN,
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


# The what-is-it-trying column is headed with the best formulation's number,
# or with the allowed amounts while the cold start is still spreading them
# out. An ingredient of either name collides with that column exactly as one
# named 'Total (g)' collides with the total, so both headers are read off the
# wording that writes them rather than spelled out a second time here.
_COMPARED_COLUMN_RE = re.compile(
    "^(" + re.escape(wording.compared_with_column("").strip()) + r"\s*\d+"
    + "|" + re.escape(wording.COMPARED_WITH_ALLOWED) + ")$", re.IGNORECASE)


def is_reserved_name(name):
    """True when a variable name would overwrite a column the app owns.
    Capitalisation is ignored: 'total' and 'Total (g)' are the same column."""
    name = str(name).strip()
    return (name.lower() in {r.lower() for r in RESERVED_VARIABLE_NAMES}
            or bool(_TOTAL_COLUMN_RE.match(name))
            or bool(_COMPARED_COLUMN_RE.match(name)))


# ------------------------------------------------------------------ #
# Formula cells (0.5.0 wave 2, "rules"): the Set up grid's Formula column
# lets an ingredient's amount be read off the batch size and the other rows
# instead of typed by hand. parse_formula turns a cell's text into a
# LinearForm; nothing here touches the grid or the model — that is later
# work, not this one's.
# ------------------------------------------------------------------ #

class FormulaError(ValueError):
    """A Formula cell's text could not be turned into a LinearForm. Its one
    argument is already the sentence to show at the cell, worded in
    wording.py — food_bo never writes one of its own."""


class LinearForm:
    """What a formula cell's text means, once it has been read: a constant,
    a multiple of the batch size, and a multiple of each other row it
    names. `terms` maps a name to what it is multiplied by; a name absent
    from it counts as a multiple of 0, so two forms that only differ by a
    zero term still combine and compare as if they agreed.

    `rest` marks the one shape that is not a sum of multiples at all:
    '= rest' on its own, standing for whatever is left of the batch size
    once every other row is filled in. A LinearForm with rest set carries
    no const, batch or terms — the grammar never combines it with
    anything else, since parse_formula refuses that at the cell."""

    def __init__(self, const=0.0, batch=0.0, terms=None, rest=False):
        self.const = float(const)
        self.batch = float(batch)
        self.terms = {name: float(v) for name, v in (terms or {}).items()}
        self.rest = bool(rest)

    def is_constant(self):
        """True for a plain number: no batch size, no name, not the rest.
        The multiply and divide rules below are the only place this is
        asked, since those are the only operators a formula may not use
        between two amounts."""
        return not self.rest and self.batch == 0.0 and not self.terms

    def names(self):
        """The other rows this formula reads (not the batch size, which is
        its own field, and not the rest, which reads no row at all)."""
        return set(self.terms)

    def scaled(self, k):
        """This form multiplied through by the plain number k: how a
        leading '−' and a '× number' / '÷ number' are all carried out."""
        return LinearForm(self.const * k, self.batch * k,
                          {name: v * k for name, v in self.terms.items()})

    def __add__(self, other):
        terms = dict(self.terms)
        for name, v in other.terms.items():
            terms[name] = terms.get(name, 0.0) + v
        return LinearForm(self.const + other.const, self.batch + other.batch,
                          terms)

    def __sub__(self, other):
        return self + other.scaled(-1)

    def __eq__(self, other):
        return (isinstance(other, LinearForm) and self.rest == other.rest
                and self.const == other.const and self.batch == other.batch
                and self.terms == other.terms)

    def __repr__(self):
        return "LinearForm(const=%r,batch=%r,terms=%r,rest=%r)" % (
            self.const, self.batch, self.terms, self.rest)


# Both alphabets a formula may be typed in, mapped to the one symbol the
# parser below reads: the typographic row the spec prints (−×÷) and the
# ASCII row the bench types (-*/) mean the same four operators.
#
# And the characters a keyboard produces without being asked. macOS smart
# dashes, Word and Excel all turn a typed '-' into an en or em dash, which
# looks like a minus sign and is not one; '·' is the multiplication dot a
# scientist writes by hand. Refusing them named a character the reader
# could not tell apart from the one that works.
_FORMULA_OPS = {'+': '+', '-': '-', '−': '-', '–': '-', '—': '-',
               '*': '*', '×': '*', '·': '*', '/': '/', '÷': '/'}


def _formula_word_ends(body, end):
    """True when a token matched up to `end` is not immediately followed by
    more of the same word — so a name that is a prefix of a longer,
    unlisted word (or 'batch size' inside a longer phrase) is never
    mistaken for a match."""
    return end >= len(body) or not body[end].isalnum()


def _tokenize_formula(body, names, has_batch_size):
    """The text after the Formula cell's leading '=' has been stripped, cut
    into the grammar's tokens: numbers, 'batch size', the given names
    (longest first, case-insensitive), brackets, the four operators and the
    two words of a percentage ('%' and 'of').

    Raises FormulaError as soon as a token cannot be placed: a character
    the grammar has no use for, a name nothing in the project wears, or
    'batch size' asked of a project with no default to read it from."""
    ordered_names = sorted({str(nm) for nm in names}, key=len, reverse=True)
    batch_phrase = wording.BATCH_SIZE_NOUN
    rest_word = wording.REST_TOKEN
    lowered = body.lower()
    tokens = []
    i, n = 0, len(body)
    while i < n:
        ch = body[i]
        if ch.isspace():
            i += 1
        elif ch in _FORMULA_OPS:
            tokens.append((_FORMULA_OPS[ch], None))
            i += 1
        elif ch == '%':
            # A percentage is the wording the Limits box above the grid
            # uses for the same idea, so the cell reads it too: '1.5 % of
            # batch size' is '× 0.015'.
            tokens.append(('%', None))
            i += 1
        elif ch in '()':
            tokens.append((ch, None))
            i += 1
        elif ch.isdigit():
            j = i
            while j < n and body[j].isdigit():
                j += 1
            if j < n and body[j] == '.' and j + 1 < n and body[j + 1].isdigit():
                j += 1
                while j < n and body[j].isdigit():
                    j += 1
            tokens.append(('NUMBER', float(body[i:j])))
            i = j
        elif ch.isalpha():
            end = i + len(batch_phrase)
            if (lowered.startswith(batch_phrase.lower(), i)
                    and _formula_word_ends(body, end)):
                if not has_batch_size:
                    raise FormulaError(wording.FORMULA_NEEDS_BATCH_SIZE)
                tokens.append(('BATCH_SIZE', None))
                i = end
                continue
            matched = None
            for nm in ordered_names:
                end = i + len(nm)
                if (lowered.startswith(nm.lower(), i)
                        and _formula_word_ends(body, end)):
                    matched = nm
                    break
            if matched is not None:
                tokens.append(('NAME', matched))
                i += len(matched)
                continue
            # The second half of a percentage, and only ever that: a row
            # really called 'of' is matched above, as a name.
            if lowered.startswith('of', i) and _formula_word_ends(body, i + 2):
                tokens.append(('OF', None))
                i += 2
                continue
            end = i + len(rest_word)
            if (lowered.startswith(rest_word.lower(), i)
                    and _formula_word_ends(body, end)):
                raise FormulaError(wording.FORMULA_REST_ALONE)
            j = i
            while j < n and (body[j].isalnum() or body[j] in " '-"):
                j += 1
            raise FormulaError(wording.formula_unknown_name(body[i:j].strip()))
        else:
            raise FormulaError(wording.FORMULA_UNREADABLE)
    return tokens


class _FormulaParser:
    """expr := term (('+'|'−') term)* · term := factor (('×'|'÷') factor |
    '%' 'of' factor)* · factor := ['−'] (number | name | 'batch size' |
    '(' expr ')') — the grammar in task-1-brief.md, read over the token
    list _tokenize_formula produced, plus the percentage the Limits box
    above the grid writes the same idea in."""

    def __init__(self, tokens):
        self.tokens = tokens
        self.pos = 0

    def _peek(self):
        if self.pos < len(self.tokens):
            return self.tokens[self.pos]
        return (None, None)

    def _advance(self):
        token = self.tokens[self.pos]
        self.pos += 1
        return token

    def parse(self):
        value = self._expr()
        if self.pos != len(self.tokens):
            raise FormulaError(wording.FORMULA_UNREADABLE)
        return value

    def _expr(self):
        value = self._term()
        while self._peek()[0] in ('+', '-'):
            op, _ = self._advance()
            rhs = self._term()
            value = value + rhs if op == '+' else value - rhs
        return value

    def _term(self):
        value = self._factor()
        while self._peek()[0] in ('*', '/', '%'):
            op, _ = self._advance()
            if op == '%':
                # 'N % of X' is 'X × N/100'. It binds like a multiplication
                # because that is what it is, and the number has to be a
                # plain one for the same reason a multiplier does.
                if self._peek()[0] != 'OF':
                    raise FormulaError(wording.FORMULA_UNREADABLE)
                self._advance()
                rhs = self._factor()
                if not value.is_constant():
                    raise FormulaError(wording.FORMULA_TWO_AMOUNTS)
                value = rhs.scaled(value.const / 100.0)
                continue
            rhs = self._factor()
            if op == '*':
                if value.is_constant():
                    value = rhs.scaled(value.const)
                elif rhs.is_constant():
                    value = value.scaled(rhs.const)
                else:
                    raise FormulaError(wording.FORMULA_TWO_AMOUNTS)
            else:
                if not rhs.is_constant():
                    raise FormulaError(wording.FORMULA_DIVIDE_BY_AMOUNT)
                if rhs.const == 0:
                    raise FormulaError(wording.FORMULA_DIVIDE_BY_ZERO)
                value = value.scaled(1.0 / rhs.const)
        return value

    def _factor(self):
        negate = False
        if self._peek()[0] == '-':
            negate = True
            self._advance()
        kind, val = self._peek()
        if kind == 'NUMBER':
            self._advance()
            value = LinearForm(const=val)
        elif kind == 'NAME':
            self._advance()
            value = LinearForm(terms={val: 1.0})
        elif kind == 'BATCH_SIZE':
            self._advance()
            value = LinearForm(batch=1.0)
        elif kind == '(':
            self._advance()
            value = self._expr()
            if self._peek()[0] != ')':
                raise FormulaError(wording.FORMULA_UNREADABLE)
            self._advance()
        else:
            raise FormulaError(wording.FORMULA_UNREADABLE)
        return value.scaled(-1) if negate else value


def parse_formula(text, names, has_batch_size):
    """Read a Formula cell's text and return the LinearForm it means.

    `names` are every OTHER row's name this formula may reference (an
    ingredient or a process setting; not the row the formula is on, which
    is not this function's business to know). `has_batch_size` says
    whether the project has a default batch size for 'batch size' to mean;
    without one that token is refused here, at the cell, rather than left
    for a later screen to trip over.

    Raises FormulaError(sentence) — one sentence, worded in wording.py —
    the moment the text cannot be read as this grammar:

        expr := term (('+'|'−') term)*
        term := factor (('×'|'÷') factor | '%' 'of' factor)*
        factor := ['−'] (number | name | 'batch size' | '(' expr ')')

    '= rest' (wording.REST_TOKEN, case-insensitive, alone in the cell) is
    the one exception: not an expression at all, but the balance of the
    batch size, returned as LinearForm(rest=True). Combined with anything
    else, 'rest' is refused rather than read as an unknown name."""
    if not isinstance(text, str):
        raise FormulaError(wording.FORMULA_UNREADABLE)
    if not text.strip().startswith('='):
        # One character short of a good rule is not gibberish, and the
        # generic sentence never said which character was missing.
        raise FormulaError(wording.RULE_NEEDS_EQUALS)
    body = text.strip()[1:]
    if body.strip().lower() == wording.REST_TOKEN.lower():
        return LinearForm(rest=True)
    tokens = _tokenize_formula(body, names, has_batch_size)
    return _FormulaParser(tokens).parse()


def formula_is_rest(text):
    """True when this cell says '= rest' and nothing else.

    Through the parser, never a string compare: '=rest', '= REST' and
    '=  rest ' are all the same cell to parse_formula, and every reader of
    a Formula cell — the grid, the CSV loader, the Set-up sheet — has to
    agree with it or a project loads with half the app thinking it has a
    rest row and half thinking it has none. `names` is empty on purpose:
    '= rest' is decided before a single name is looked up, and anything
    else is not the rest whatever it reads."""
    try:
        return parse_formula(text, (), True).rest
    except FormulaError:
        return False


def _amount(value):
    """One cell of a formulation as a number: a blank, a missing row or a
    value that is not a number at all counts as nothing. A formula reads
    the other rows through this, so one odd cell cannot break a whole
    round."""
    try:
        return float(value or 0.0)
    except (TypeError, ValueError):
        return 0.0


def _renamed_in_formula(text, renames, names):
    """`text` with every row `renames` moves rewritten, and every other name
    in it left exactly as it was.

    `renames` is the whole {old name: new name} map of one save, applied in
    ONE pass: each name in the cell is looked up once, in the spelling the
    project had when the cell was typed, and the answer is written straight
    out. Applying them one after another would read the second rename
    against the text the first one just wrote, so a chain (Sugar→Butter,
    Flour→Sugar) would rewrite the Butter it had only just produced.

    Longest name first, the same order the tokenizer reads them in, so
    renaming Cream in a project that also has Cream cheese leaves the
    longer row alone. Capitals are ignored on the way in and the new name
    is written as the project spells it."""
    known = sorted({n for n in names if n}, key=len, reverse=True)
    moves = {str(old).lower(): new for old, new in (renames or {}).items()}
    if not known or not moves:
        return text
    pattern = re.compile("|".join(re.escape(n) for n in known), re.I)
    out, at = [], 0
    for match in pattern.finditer(text):
        start, end = match.span()
        if start < at:
            continue
        if ((start and text[start - 1].isalnum())
                or (end < len(text) and text[end].isalnum())):
            continue
        out.append(text[at:start])
        out.append(moves.get(match.group().lower(), match.group()))
        at = end
    out.append(text[at:])
    return "".join(out)


# How many results the cold start spreads across the allowed amounts before
# the model starts aiming, and how large a change — as a fraction of a
# variable's own allowed range — still counts as staying close to the best.
COLD_START_RUNS = 5

# How narrow a fixed column's recorded history may be before _search_bounds
# stops treating it as a frame at all. Relative to the numbers themselves:
# 175 °C and 0.0001 g are not near each other on one absolute scale.
_FIXED_SPAN_FLOOR = 1e-9

# `tell(batch_no=NO_BATCH)`: this formulation belongs to no batch this
# project generated. None cannot say it — None means "the open batch" — and
# patching batch_history afterwards left the open batch's own total written
# against a formulation made before the project existed.
NO_BATCH = object()
CLOSE_TO_THE_BEST = 0.15
# How many process settings one what-is-it-trying line names, on its own so
# that a project of eight settings does not bury the amounts under them. The
# ingredients have their own count, which vs_best_text takes as an argument.
SETTINGS_SHOWN = 3


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


def fmt_amount(value, unit="", decimals=2):
    """An amount as prose: '12.50 g', '0.30 g', '' for a missing value.

    Always two decimals. A weighing sheet that mixes '0.3 g', '33.9 g' and
    '11.88 g' cannot be read down the column, and 0.30 g is the precision a
    balance works to. A process setting is not an amount and does not come
    through here: a cook temperature is 180 °C, never 180.00 °C."""
    if value is None:
        return ""
    txt = f"{float(value):.{decimals}f}"
    if float(txt) == 0:
        txt = f"{0.0:.{decimals}f}"     # never '-0.00'
    return join_unit(txt, unit)


def fmt_setting(value, unit=""):
    """A process setting as prose: '188.49 °C', '180 °C', '' for a missing
    value. A setting is dialled in, not weighed: at most two decimals, and no
    trailing zeros, because 188.494 is a precision no oven dial has and
    180.00 is a precision nobody typed. Every screen that shows a setting —
    the batch table, the printable sheets, the amounts table — goes through
    here, so the three always agree."""
    if value is None:
        return ""
    txt = f"{float(value):.2f}".rstrip("0").rstrip(".")
    if txt in ("", "-0"):
        txt = "0"
    return join_unit(txt, unit)


def amount_range_placeholder(low, high):
    """'0–60', and just '20' when a row is FIXED. A box whose placeholder
    reads '20–20' asks the reader to work out that the two ends are the same
    number; the one number says it outright."""
    if float(low) == float(high):
        return f"{float(low):g}"
    return f"{float(low):g}–{float(high):g}"


def outside_message(name, value, low, high, unit, what, tail=""):
    """'Firmness 12 N is outside your range of 0 to 10 N.' — the one builder
    for every out-of-bounds line, so a measurement typed into the grid, one
    read off an uploaded sheet and an amount imported from a file are refused
    in the same words. `what` names the bounds, `tail` is any sentence that
    follows. A "/"-style unit stays off the numbers, as it does everywhere."""
    unit = unit_after_number(unit)
    # A fixed row has one allowed amount, and "0 to 0" said it twice.
    reach = (f"{float(low):g}" if float(low) == float(high)
             else f"{float(low):g} to {float(high):g}")
    return (join_unit(f"{name} {float(value):g}", unit)
            + f" is outside {what} of "
            + join_unit(reach, unit)
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
    return ", ".join(items[:-1]) + wording.AND_JOIN + items[-1]


# ------------------------------------------------------------------ #
#  0.5.0 · reading an edited grid
#
#  `st.data_editor` hands back a DataFrame: the rows the user left alone,
#  the ones they changed, and the ones they typed on the empty line at the
#  bottom, with no way of telling them apart. GRID_ID is how they are told
#  apart — a hidden column carrying the name each row is filed under today,
#  so a name typed over it is a RENAME of that row and a row that arrives
#  without one is new.
#
#  Every cell is read through the three helpers below, because a grid cell
#  is never quite a value: a number box left empty comes back as NaN, a text
#  box as None or "", and a column the frame does not have at all is simply
#  missing.
# ------------------------------------------------------------------ #
GRID_ID = "_id"


def _grid_frame(data, columns):
    """A grid frame with its rows numbered from 1, so that "Row 3" under the
    grid names the third row the reader can see."""
    frame = pd.DataFrame(data, columns=columns)
    frame.index = pd.RangeIndex(start=1, stop=len(frame) + 1)
    return frame


def _grid_rows(frame):
    """(row number, row) for every row, numbered as the grid shows them —
    by position, never by the frame's own index, which an edited frame may
    have renumbered."""
    return [(i, row) for i, (_, row)
            in enumerate(frame.iterrows(), start=1)]


def _cell(row, column, default=None):
    """One cell, or `default` for one that is empty, missing or NaN."""
    if column not in row:
        return default
    value = row[column]
    if value is None:
        return default
    try:
        if pd.isna(value):
            return default
    except (TypeError, ValueError):
        pass
    return value


def _text_cell(row, column):
    return str(_cell(row, column, "")).strip()


def _number_cell(row, column):
    """(number, True) for a cell that holds one, (None, True) for an empty
    one, and (None, False) for a cell holding something that is not a number
    at all — which is a refusal, not a blank."""
    value = _cell(row, column)
    if value is None or (isinstance(value, str) and not value.strip()):
        return None, True
    try:
        return float(value), True
    except (TypeError, ValueError):
        return None, False


def _file_text(row, column, present=True):
    """One text cell of an ingredient file, as the grid would read it: the
    text, stripped, and "" for a blank, a missing column or a NaN."""
    if not present:
        return ""
    value = row.get(column)
    if value is None:
        return ""
    try:
        if pd.isna(value):
            return ""
    except (TypeError, ValueError):
        pass
    return str(value).strip()


def _file_range(row):
    """(Lowest, Highest) for a row of an ingredient file that carries a
    formula: the two cells when both hold numbers, and (0, 0) when either is
    blank. Dormant either way — a worked-out row's range is not enforced —
    so a cell that cannot be read is not worth a refusal here."""
    out = []
    for column in ('Min', 'Max'):
        try:
            out.append(float(_file_text(row, column)))
        except (TypeError, ValueError):
            return 0.0, 0.0
    return out[0], out[1]


def _range_from_cells(row):
    """(Lowest, Highest, whether both cells hold something readable) for one
    row of the ingredients grid.

    The two cells are text now (see _range_cell), so they are read back
    through the same _number_cell every other number on a grid goes through:
    a cell holding anything but a number is still a refusal. The one word
    that is not a refusal is the app's own `worked out`, which is what the
    cells of a formula row say — it reads as no number at all, and a row
    with no formula to work it out from then asks for one."""
    low, low_ok = _number_cell(row, wording.LOWEST_LABEL)
    high, high_ok = _number_cell(row, wording.HIGHEST_LABEL)
    for column, ok in ((wording.LOWEST_LABEL, low_ok),
                       (wording.HIGHEST_LABEL, high_ok)):
        if not ok and _text_cell(row, column) == wording.WORKED_OUT:
            if column == wording.LOWEST_LABEL:
                low, low_ok = None, True
            else:
                high, high_ok = None, True
    return low, high, (low_ok and high_ok)


def grid_signature(frame):
    """One comparable value per row, for asking whether a grid still says
    what the project says. Compared cell by cell rather than frame by frame:
    an editor hands back its own dtypes (an int typed into a float column, a
    blank as NaN), and two frames that read the same on screen are not equal
    to pandas."""
    return [tuple(_signature_cell(row.get(column)) for column in frame.columns)
            for _, row in frame.iterrows()]


def _signature_cell(value):
    """One cell, as the thing it means: a blank, a number, or text."""
    if value is None:
        return None
    try:
        if pd.isna(value):
            return None
    except (TypeError, ValueError):
        pass
    if isinstance(value, bool):
        return value
    try:
        return round(float(value), 9)
    except (TypeError, ValueError):
        return str(value).strip()


def _grid_ids(frame):
    return [_text_cell(row, GRID_ID) for _, row in frame.iterrows()]


def _row_is_blank(row):
    """True for the empty line at the bottom of a dynamic grid, clicked and
    then left alone. It is not an addition and it is not an error.

    A row that carries an `_id` is never blank however empty its cells look:
    it is a row of the project with its visible answers rubbed out, and
    skipping it would have taken it off the grid — a deletion, applied with
    no question asked and no copy kept. It falls through to the row reader
    and is refused there for having no name."""
    if _text_cell(row, GRID_ID):
        return False
    return not any(str(_cell(row, c, "")).strip() for c in row.index
                   if c != GRID_ID)


def _share_to_total(free, room_of, need):
    """`free` moved so that it adds up to `need`, every part of it kept
    between 0 and its own `room_of`.

    The correction is shared out in proportion to what each part can still
    take, which is what stops an amount pinned at its end from swallowing
    it and the loop from stalling. The numbers are contributions to the
    total, not amounts: the caller works in what each part of an amount is
    WORTH and divides the coefficient back out afterwards.
    """
    free = list(free)
    for _ in range(40):
        residual = need - sum(free)
        if abs(residual) <= 1e-9:
            break
        headroom = ([c - f for f, c in zip(free, room_of)] if residual > 0
                    else list(free))
        share = sum(headroom)
        if share <= 1e-12:
            break
        for k, head in enumerate(headroom):
            free[k] = min(room_of[k],
                          max(0.0, free[k] + residual * head / share))
    return free


def _rename_order(rows):
    """The order to write the existing rows in so that no rename ever lands
    on a name another row is still wearing.

    Renaming A to B while B is still on the grid under another row is only
    safe once B has moved out of the way. Anything whose target is free goes
    first, which frees more names, and so on. What is left when nothing can
    move is a swap — A to B and B to A — and that is handed back to be
    refused by name rather than half-applied.
    """
    pending = [(row_no, spec) for row_no, spec in rows
               if spec['id'] is not None]
    held = {spec['id'] for _, spec in pending}
    order = []
    while pending:
        movable = [item for item in pending
                   if item[1]['name'] not in (held - {item[1]['id']})]
        if not movable:
            return order, pending[0]
        for item in movable:
            order.append(item[0])
            held.discard(item[1]['id'])
            held.add(item[1]['name'])
            pending.remove(item)
    return order, None


def _proposed_variable(spec):
    """One row of the finished grid as a variable dict, for the feasibility
    question. A copy: the real row is not touched until the answer is in."""
    var = dict(spec['var']) if spec['var'] is not None else {
        'type': 'continuous'}
    var['name'] = spec['name']
    var['category'] = spec['category']
    var['bounds'] = (float(spec['low']), float(spec['high']))
    var['type'] = var.get('type', 'continuous')
    var['unit'] = spec['unit']
    # As typed, not as stored: this copy is what a question about the
    # finished grid — a loop, a formula that can never be a real amount — is
    # asked against, and the cell the reader just edited is the answer.
    var['formula'] = spec['formula']
    var['balance'] = spec['balance']
    return var


def _duplicate_name_errors(rows):
    """Two rows of one grid wearing one name, blamed on the row that MOVED.

    One name, one thing: two rows with the same name are two columns of one
    name on every table in the app, and a second spelling of it is the same
    collision under another face. Which row to say it against is the whole
    question — the reader typed into one of them, and blaming the other asks
    them to fix a row they never touched. The row that moved is the one
    whose Name no longer matches the identity it arrived with; when both
    moved (or both are new) the later one is the one they just typed.
    """
    errors = []
    first = {}
    for row_no, spec in rows:
        lowered = spec['name'].lower()
        twin = first.get(lowered)
        if twin is None:
            first[lowered] = (row_no, spec)
            continue
        blamed, other = (twin, spec) if _row_moved(twin[1]) and not _row_moved(
            spec) else ((row_no, spec), twin[1])
        errors.append((blamed[0],
                       _name_taken_message(other['name'], blamed[1]['category'])
                       if other['name'] == blamed[1]['name']
                       else wording.name_differs_only_by_case(other['name'])))
    return errors


def _row_moved(spec):
    """True when this row's name is not the one it arrived under — a rename,
    or a row typed on the empty line at the bottom."""
    return spec['id'] is None or spec['id'] != spec['name']


def _formula_after_renames(text, renames, spelled):
    """One formula cell with every name this save is renaming rewritten,
    and everything else left exactly as it was.

    The cell was typed against the names the project had when it was
    written, so a rename of a row it names leaves it spelling a row that no
    longer exists. rename_variable rewrites it after the fact through the
    same _renamed_in_formula; this is the same answer BEFORE the save, so
    the grid never refuses an untouched cell for a row the reader can see.

    Every rename of the save at once, read against the spelling the project
    had before any of them landed: a chain (Sugar→Butter, Flour→Sugar) means
    what it says on the grid the reader is looking at, not what one pass
    would leave for the next to read.
    """
    return _renamed_in_formula(text, renames, spelled)


def _formula_cell_moved(spec):
    """True when this row's Formula cell says something different from what
    the project has filed under it — a formula given, changed or rubbed
    out. A rename written into an untouched cell is not a change, and a new
    row that arrives without one has not changed anything either."""
    was = (("", False) if spec['var'] is None
           else (str(spec['var'].get('formula') or ""),
                 bool(spec['var'].get('balance'))))
    return was != (spec.get('formula_typed', spec['formula']),
                   spec['balance'])


def _renames(rows):
    """{old name: new name} for the rows of a finished grid that moved."""
    return {spec['id']: spec['name'] for _, spec in rows
            if spec['id'] and spec['id'] != spec['name']}


def _limits_over(limits, names, renamed=None, deleted=()):
    """The amount limits that would survive the finished grid: one that has
    lost an ingredient is pruned, one whose ingredients were renamed is
    rewritten (rename_variable does exactly that), and the total's own limit
    is over whatever the ingredient list now is.

    `deleted` is applied FIRST, which is the order the save itself keeps. A
    limit on a Salt this save deletes is gone; without that step, a Water
    renamed to Salt in the same save would have left the limit reading as a
    limit on the new row."""
    renamed, gone = renamed or {}, set(deleted)
    kept = []
    for qc in limits:
        if qc.get('source') == 'formulation_total':
            kept.append(dict(qc, ingredients=list(names)))
            continue
        if any(n in gone for n in qc['ingredients']):
            continue
        moved = [renamed.get(n, n) for n in qc['ingredients']]
        if all(n in names for n in moved):
            kept.append(dict(qc, ingredients=moved))
    return kept


def _properties_over(properties, renamed, deleted=()):
    """An ingredient's property figures follow its name, as rename_variable
    moves them. Without this a row renamed in the same save reads as having
    no figure for anything, and a property limit looks broken when it is
    not.

    The deleted rows go first, for the same reason and in the same order the
    save keeps: delete Salt and rename Water to Salt, and mapping the two
    together would have given the new Salt the old one's figures — or, worse,
    whichever of the two the dict happened to write last."""
    gone = set(deleted)
    return {renamed.get(name, name): values
            for name, values in (properties or {}).items()
            if name not in gone}


def _what_moved(before, after):
    """Which part of a grid row changed, as a set of "unit", "supplier" and
    "other".

    They are kept apart because they cost different things. "other" — the
    allowed amounts, the type, a setting's baseline — is a change to the
    question the model is being asked, and retires the open round. A unit is
    how a number is written, not the number, and has a sentence of its own
    to say. A vendor and an SKU are printed on the sheets and read by
    nothing. A FORMULA is "other": the row leaves the search vector, so what
    the model is being asked is a different question. The row's NAME is not
    in here at all: a rename goes through rename_variable, which moves
    everything filed under it.
    """
    if before == after:
        return set()
    if before is None:
        return {'other'}                 # a row that was not there before
    moved = set()
    if before[3] != after[3]:
        moved.add('unit')
    if before[4:6] != after[4:6]:
        moved.add('supplier')
    if (before[1], before[2], before[6:9]) != (after[1], after[2],
                                              after[6:9]):
        moved.add('other')
    return moved


def _shares_moved(typed, final):
    """True when the column on screen will not come back holding what was
    typed into it."""
    return any(abs(float(typed[name]) - float(final.get(name, 0)))
               > 0.5 for name in typed)


# The refusals the grid has to give BEFORE it writes anything, so they are
# written once and raised from two places: the model path that discovers
# them after the fact, and the grid's own validation pass.
LAST_VARYING_ROW_ERROR = wording.LAST_VARYING_ROW_ERROR
AMOUNTS_MISSING_DELETE_ERROR = wording.AMOUNTS_MISSING_DELETE_ERROR


# Every sentence below is wording's; these names are what the rest of this
# file already reads them by.
_reserved_name_message = wording.reserved_name_message
_name_taken_message = wording.name_taken_by_variable
_target_outside_message = wording.target_outside_message
_baseline_outside_message = wording.baseline_outside_message
LOWEST_ABOVE_HIGHEST_ERROR = wording.LOWEST_ABOVE_HIGHEST_ERROR
RANGE_ENDS_ERROR = wording.RANGE_ENDS_ERROR
TARGET_REQUIRED_ERROR = wording.TARGET_REQUIRED_ERROR


def _objective_scoring_state(obj):
    """The fields that feed closeness. A change to any of them recalculates
    every overall score already stored, which is what decides whether a save
    keeps a copy first and says it recalculated."""
    return (obj['goal'],
            None if obj.get('target') is None else float(obj['target']),
            float(obj.get('min_val', 0.0)), float(obj.get('max_val', 10.0)))


def _objective_state(obj):
    """Everything one row of the measurements grid says about a measurement,
    as one comparable value — the share apart, which is set for the column
    as a whole. The unit is in here and NOT in the scoring state above: it
    is how a number is written, not the number."""
    return _objective_scoring_state(obj) + (str(obj.get('unit', "") or ""),)


def _used_ingredient_message(name, numbers):
    return wording.ingredient_was_used(name, number_list(numbers),
                                       many=len(numbers) != 1)


def goal_text(obj):
    """'Target 6 N', 'Higher is better', 'Lower is better' — what a good
    number looks like, in ONE rendering.

    It was written three ways for one cell: 'Hit a target' beside a Target
    of 6 on the grid, 'Target 6' in the Results table, and 'target 6' in
    lower case on the round sheets and the record-results labels. A bench
    worker holding the sheet and a scientist reading Results were reading
    the same fact in different words and different case. A '/10' rides on
    the measurement's own name instead of on every number in its row.
    """
    if obj.get('goal') == 'target' and obj.get('target') is not None:
        return join_unit(wording.target_value(obj['target']),
                         unit_after_number(obj.get('unit')))
    return wording.GOAL_LABELS.get(obj.get('goal'),
                                   wording.GOAL_LABELS['max'])


# The old name for the same sentence, kept because three screens and two
# sheets already ask for it by this one.
goal_line = goal_text


def measurement_range_text(obj):
    """'0 to 10 N' — the Range cell, in the measurement's own unit."""
    return join_unit(wording.range_text(obj['min_val'], obj['max_val']),
                     unit_after_number(obj.get('unit')))


# --------------------------------------------------------------------- #
#  The workbook: what the bench carries away from the screen.
# --------------------------------------------------------------------- #
# What an uploaded workbook came back with. The frame is the rows
# parse_batch_results reads, and beside it travel the two things a sheet
# says that are not results: the amounts somebody wrote in the Actual
# cells, and the lot numbers, which belong to the round rather than to any
# one formulation. They are not columns of the frame — its columns are the
# shape the parser reads, and a lot number is not a result — and they are
# not smuggled on the frame either: something that has to survive a trip
# through session state should be visible in the signature that hands it
# over.
UploadedWorkbook = namedtuple("UploadedWorkbook", "frame actual lots")


def uploaded_parts(upload):
    """(rows, what was weighed, the lots) out of whatever the upload step
    left behind. A comma-separated file — and a frame put straight into the
    app's own state — carries the rows and nothing else."""
    if isinstance(upload, UploadedWorkbook):
        return upload.frame, dict(upload.actual or {}), dict(upload.lots or {})
    return upload, {}, {}

# What a browser is told a workbook is. Not screen text: the one string
# every download button hands to Streamlit.
WORKBOOK_MIME = ("application/vnd.openxmlformats-officedocument"
                 ".spreadsheetml.sheet")

_BOLD = Font(bold=True)
_TITLE_FONT = Font(bold=True, size=14)
# A cell to write in is a box on paper. Nothing on a sheet is a run of
# typed underscores: a rule drawn by the spreadsheet stays straight.
_THIN = Side(style="thin", color="FF999999")
_WRITE_IN = Border(left=_THIN, right=_THIN, top=_THIN, bottom=_THIN)
_TWO_DP = "0.00"
_ONE_DP = "0.0"
# The one shade a write-in cell wears, on every sheet of every workbook.
# The sheets are protected, so the shading is the only thing that says in
# advance which cells will take a number: a technician who finds out by
# being refused has already lost the line they were typing.
_WRITE_IN_FILL = PatternFill("solid", fgColor="FFF2CC")
# Vendor and SKU are printed, never read. On a formulation page they sit
# under the ingredient's own name in this grey, so the name stays the
# thing the eye lands on.
_QUIET_FONT = Font(color="FF808080", size=9)
# Unlocked, for the cells the app will read back. Locked is openpyxl's
# default, so every other cell is already closed once the sheet is
# protected.
_UNLOCKED = Protection(locked=False)


# What a technician's tick looks like once a spreadsheet has read it: a
# cross, a tick, a letter, TRUE. A cell holding 0, "no" or "false" is
# somebody answering the question rather than leaving it blank — and so is
# the empty box the sheet prints into that cell, which is what an untouched
# Not scored cell comes back holding.
_NOT_TICKED = {"", "0", "0.0", "no", "n", "false", "none", "-",
               wording.TICK_BOX}


def _is_ticked(value):
    """True when the Not scored box on a sheet has been marked."""
    if value is None or (isinstance(value, float) and np.isnan(value)):
        return False
    return str(value).strip().lower() not in _NOT_TICKED


_log = logging.getLogger(__name__)


def _damaged(detail):
    """The ValueError a copy that cannot be opened is refused with, and the
    one place its reason is written down.

    ONE sentence reaches the screen. The old refusals named the stored
    field — "This copy's 'recipe_history' section has the wrong shape." —
    which is four of the words the app retired, programmer punctuation and
    a shape, shown to a food scientist whose saved copy will not open. The
    detail is what an engineer reading the log needs, and it is the only
    place it belongs.
    """
    _log.warning("saved copy refused: %s", detail)
    return ValueError(wording.COPY_DAMAGED)


def _write_cell(sheet, row, column, value, bold=False, fill=None,
                number_format=None, border=False, wrap=False):
    """One cell, with the furniture the sheets use over and over."""
    cell = sheet.cell(row=row, column=column, value=value)
    if bold:
        cell.font = _BOLD
    if fill is not None:
        cell.fill = fill
    if number_format is not None:
        cell.number_format = number_format
    if border:
        cell.border = _WRITE_IN
    if wrap:
        cell.alignment = Alignment(wrap_text=True, vertical="top")
    return cell


def _write_in_cell(sheet, row, column, value=None, wrap=False):
    """A cell the bench fills in: boxed, shaded and unlocked, so it is the
    one kind of cell a protected sheet still takes a number in. Every cell
    written this way is a cell the upload reads back, and no other cell is
    unlocked — the sheet is closed exactly where the app is deaf."""
    cell = _write_cell(sheet, row, column, value, fill=_WRITE_IN_FILL,
                       border=True, wrap=wrap)
    cell.protection = _UNLOCKED
    return cell


def _write_banner(sheet, row, column, text, last_column):
    """One line of instruction across the width of the sheet: merged,
    wrapped and given the room its second line needs. Left in a single
    column it is cut off at the print edge, and an instruction the printed
    page ends halfway through is worse than none."""
    cell = _write_cell(sheet, row, column, text, wrap=True)
    if last_column > column:
        sheet.merge_cells(start_row=row, start_column=column,
                          end_row=row, end_column=last_column)
    sheet.row_dimensions[row].height = 30
    return cell


def _protect(sheet):
    """Lock the sheet around its write-in cells. A summary sheet whose
    amounts were overtyped on the way to the bench came back as results for
    a formulation the app had never suggested, and nothing in the file said
    so."""
    sheet.protection.sheet = True
    return sheet


def _set_widths(sheet, widths):
    for i, width in enumerate(widths, start=1):
        sheet.column_dimensions[get_column_letter(i)].width = width


def _fit_to_page(sheet, last_row, last_column, landscape=False):
    """One page wide, portrait unless the batch is too wide for it. A sheet
    that prints its last two ingredients on a second page is a sheet the
    bench weighs out wrong."""
    sheet.page_setup.orientation = ("landscape" if landscape else "portrait")
    sheet.sheet_properties.pageSetUpPr = PageSetupProperties(fitToPage=True)
    sheet.page_setup.fitToWidth = 1
    sheet.page_setup.fitToHeight = 0 if landscape else 1
    sheet.print_area = f"A1:{get_column_letter(max(1, last_column))}{max(1, last_row)}"


def _write_frame(sheet, frame, two_decimals=(), freeze=True, title=None):
    """A DataFrame onto a sheet: bold headers, the values below, columns as
    wide as their longest cell, and two decimals on the columns that hold
    amounts — a downloaded 11.875 beside a screen that says 11.88 reads as a
    third number.

    `title` puts one line above the header, saying what the table is."""
    columns = list(frame.columns)
    widest = [len(str(name)) for name in columns]
    top = 1 if title is None else 2
    if title is not None:
        _write_cell(sheet, 1, 1, title).font = _TITLE_FONT
    for c, name in enumerate(columns, start=1):
        _write_cell(sheet, top, c, str(name), bold=True)
    for r, (_, row) in enumerate(frame.iterrows(), start=top + 1):
        for c, name in enumerate(columns, start=1):
            value = row[name]
            if value is None or (isinstance(value, float) and np.isnan(value)):
                continue
            if isinstance(value, (np.integer,)):
                value = int(value)
            elif isinstance(value, (np.floating,)):
                value = float(value)
            widest[c - 1] = max(widest[c - 1], len(str(value)))
            _write_cell(sheet, r, c, value,
                        number_format=(_TWO_DP if name in two_decimals
                                       else None))
    # Four characters of padding, and never wider than a column a reader can
    # take in: a note of three sentences would otherwise push everything
    # after it off the page.
    _set_widths(sheet, [min(60, max(10, width + 4)) for width in widest])
    if freeze and columns:
        sheet.freeze_panes = f"A{top + 1}"
    return sheet


def frame_workbook(sheets):
    """One workbook from {sheet name: DataFrame}. The ingredients template
    goes out this way, so the file a project starts from is the same kind of
    file every other download is."""
    book = Workbook()
    first = True
    for name, frame in sheets.items():
        sheet = book.active if first else book.create_sheet()
        sheet.title = name
        _write_frame(sheet, frame)
        first = False
    buffer = io.BytesIO()
    book.save(buffer)
    return buffer.getvalue()


def ingredients_template_workbook(path):
    """The ingredients template as a workbook: the column headers and ONE
    example row, in the shape the uploader reads back. One file to fill in
    and hand back, rather than a comma-separated file to be talked through a
    spreadsheet's import dialog.

    One row, not eight. A file arriving with a full ingredient list already
    in it is an export, and the reader who downloaded a "template" then has
    to work out which lines are theirs and which the app's."""
    return frame_workbook(
        {wording.INGREDIENTS_SHEET: pd.read_csv(path).head(1)})


# --------------------------------------------------------------------------- #
#  Expert-selectable BO hyperparameters (optional "arm 3").
#  bo_config == None  =>  library defaults, i.e. byte-identical to the standard
#  non-adaptive arm. A validated dict swaps in the expert's chosen kernel /
#  lengthscale prior / noise handling / acquisition. Chosen ONCE at project
#  start (not per iteration): the GP's lengthscale/noise VALUES still refit from
#  data each iteration via MLE; only the structural config is fixed a priori.
# --------------------------------------------------------------------------- #
# The same four lists the expert boxes on Set up offer, read from wording so
# a value the screen can pick can never be one the loader refuses.
_KERNELS = set(wording.KERNEL_OPTIONS)
_LENGTHSCALE = set(wording.LENGTHSCALE_PRIOR_OPTIONS)
_NOISE = set(wording.NOISE_OPTIONS)
_ACQ = set(wording.ACQUISITION_OPTIONS)
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
    CLASS_VERSION = 12  # bump when adding methods/attrs to force session refresh

    # How far a suggested formulation may sit from the total it was asked
    # for. A total is an equality, and an equality is not something a
    # continuous search can be held to exactly; half a percent is narrower
    # than any bench scale and wide enough that the search always has room.
    FORMULATION_TOTAL_TOLERANCE = 0.005

    # How many space-filling points the opening looks at before it gives up
    # and asks the model. A batch that cannot be filled from the first pool
    # is one the limits have made narrow, and a wider pool is cheap.
    COLD_START_POOLS = (2048, 8192)

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
        # The same rule for batches: a batch number is permanent. Deriving it
        # from the batches still on file handed batch 2's number to the next
        # batch as soon as batch 2 was deleted, so two different sets of
        # formulations wore one number in the same project's records.
        self.next_batch_number = 1
        # The scramble the space-filling opening is drawn from, fixed once
        # per project (see _sobol_seed). None means "derive it from the name".
        self.sobol_seed = None
        # One unit for every amount in the project. Grams is the default a
        # food scientist expects; a blank unit made every amount ambiguous.
        self.amount_unit = "g"
        self.amount_unit_backfilled = False  # True only for a file with no unit
        # A free-text note on where the measurement targets came from — a
        # benchmark product, a published panel, a brief from marketing. Blank
        # says nothing was recorded; shown as a caption once it is set.
        self.targets_source = ""
        self.pending_batch = None     # the open batch: [{'formulation', 'recipe'}]
        self.pending_batch_no = None  # its batch number
        self.pending_batch_created = None   # ISO date it was generated, for the sheets
        self.pending_batch_discarded = []   # numbers a regenerate retired, for one caption
        # The total each formulation in the open batch is made to, or None for
        # "as generated". The amounts stored with a result are always as
        # generated, so without this the number the bench actually weighed out
        # was lost the moment the batch closed.
        self.pending_batch_total = None
        self.batch_totals = {}        # batch number -> the total it was made to
        # Lot numbers written on the sheets and read back: round number ->
        # {ingredient: the lot it was weighed from}. Empty until a sheet
        # comes back carrying them.
        self.lots = {}
        # The total every SUGGESTED formulation adds up to, or None for "any
        # total the allowed amounts reach". Unlike pending_batch_total, which
        # records what one batch was weighed out to after the fact, this is
        # part of the question the model is asked: it writes the limit over
        # every ingredient that both the space-filling opening and the model
        # obey, so a batch comes off the bench at the size the mixer or the
        # panel needs.
        self.formulation_total = None
        # The formula rows written out over the columns the search moves,
        # with the variable list they were worked out from. Rebuilt the
        # moment that list, a formula or the balance changes; see
        # _resolved_forms.
        self._forms_cache = None
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

    def _name_is_free(self, name, skip=None):
        """Raise unless `name` is free for something in this project to wear.

        One name, one thing: an ingredient, a process setting, a measurement
        and a property all head columns of the same tables, so no two of them
        may share one. The three doors that name something — adding a
        variable, renaming one, naming a property — ask here, so they refuse
        the same clashes in the same words. `skip` is the row already wearing
        the name and entitled to keep it (the one being renamed).

        The variable pass is a plain refusal, which is the answer at two of
        those doors. Adding is the third and has its own answers first:
        re-adding a name the project already has is an EDIT, and a second
        spelling of one has a sentence of its own — so add calls this with
        the row it is editing as `skip`, once those have had their say.
        """
        self._name_is_free_of_variables(name, skip=skip)
        self._name_is_free_of_measurements(name, skip=skip)
        self._name_is_free_of_properties(name)

    # Three passes, asked together everywhere but on a grid.
    #
    # A grid is saved whole, so a name another ROW of it is giving up in the
    # same save is free by the time the save lands — Flour renamed to Barley
    # leaves Flour for the next row to take, and a deleted Salt leaves Salt
    # — and the grid's own pass over its finished names is what catches a
    # real collision. So each grid skips the pass about its own rows and
    # asks the other two, which name things it cannot move.

    def _name_is_free_of_variables(self, name, skip=None):
        lowered = str(name).strip().lower()
        for var in self.variables:
            if var is skip or var['name'].lower() != lowered:
                continue
            raise ValueError(_name_taken_message(
                var['name'], var.get('category', 'ingredient')))

    def _name_is_free_of_measurements(self, name, skip=None):
        lowered = str(name).strip().lower()
        for obj in self.objectives:
            if obj is not skip and obj['name'].lower() == lowered:
                raise ValueError(
                    wording.name_taken_by_measurement(obj['name']))

    def _name_is_free_of_properties(self, name):
        if self._known_property(name) is not None:
            raise ValueError(wording.name_taken_by_property(name))

    def _check_new_variable(self, name, min_val, max_val, category):
        """Shared validation for add_ingredient / add_process_parameter.
        Returns the stripped name. Same-name same-category is allowed (the
        caller updates bounds); a clash with the other category is an error."""
        name = str(name).strip()
        if not name:
            raise ValueError(wording.NAME_REQUIRED_ERROR)
        if is_reserved_name(name):
            raise ValueError(_reserved_name_message(name))
        # Equal is allowed, and is how a row is FIXED: one amount, in every
        # formulation. Only Lowest ABOVE Highest is a range with nothing in
        # it, and that is what is refused.
        if float(min_val) > float(max_val):
            raise ValueError(LOWEST_ABOVE_HIGHEST_ERROR)
        for v in self.variables:
            if v['name'].lower() == name.lower() and v.get('category', 'ingredient') != category:
                other = v.get('category', 'ingredient')
                other_label = (wording.AN_INGREDIENT if other == 'ingredient'
                               else wording.A_PROCESS_SETTING)
                raise ValueError(wording.name_taken_by(v['name'],
                                                      other_label))
        for v in self.variables:
            # Same name, same category, different capitals. The exact name is
            # an EDIT (the caller updates the bounds); a second spelling of it
            # would be a second row, with the same name on every table.
            if v['name'] != name and v['name'].lower() == name.lower():
                raise ValueError(wording.name_differs_only_by_case(v['name']))
        # The row this add is really an edit of, if there is one: it is
        # allowed to go on wearing its own name.
        editing = next((v for v in self.variables if v['name'] == name), None)
        self._name_is_free(name, skip=editing)
        return name

    def add_ingredient(self, name, min_val, max_val, unit=None,
                       keep_lowest=False):
        """Add a single ingredient. Safe to call mid-run (adaptive EGBO): the
        ingredient is treated as absent (=0) in every prior recipe, and the
        encoded history is rebuilt so the GP stays dimensionally consistent.

        `keep_lowest` keeps the Lowest the caller typed even mid-run. The old
        add form disabled that box and this method forced it to 0, because
        absent-in-past encodes as 0 and a bound that excluded it looked like
        a contradiction. The grid lifts it: "5 g of salt in every formulation
        from here on" is a thing a formulator asks for, and the formulations
        already made are still honest observations of a formulation with none
        of it in. The screen says so once, in the save flash.

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
                self._check_fixed_feasible(name, min_val, max_val, 'ingredient')
                var['bounds'] = (min_val, max_val)
                if unit is not None:
                    var['unit'] = str(unit).strip()
                removed = self.prune_amount_limits()
                self._drop_pending_batch()
                self.save()
                return removed
        if self.X_history:
            if len(self.recipe_history) != len(self.X_history):
                raise ValueError(wording.CANNOT_ADD_WITHOUT_AMOUNTS)
            if not keep_lowest:
                min_val = 0.0   # absent-in-past encodes as 0
        self._check_fixed_feasible(name, min_val, max_val, 'ingredient')
        var = {
            'name': name,
            'type': 'continuous',
            'bounds': (min_val, max_val),
            'category': 'ingredient',
            # Printed on the sheets, never read by the model. Task 3 fills
            # them in from the grid; every row carries them from the start so
            # the shape on disk is one shape.
            'vendor': "",
            'sku': "",
            # Worked out rather than searched, and the one row that takes
            # whatever is left of the batch size. Blank and False are what
            # a row typed by hand holds, and what import_json backfills, so
            # a row added today and the same row restored from a copy are
            # the same dict.
            'formula': "",
            'balance': False,
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
            raise ValueError(wording.CANNOT_RELOAD_INGREDIENTS)

        # Accept any capitalization/whitespace for the required headers, and
        # fail with a plain-language error (the app shows ValueError text to
        # the user) instead of a KeyError when one is missing.
        # Lowest/Highest are what every box on the screen says; Min/Max are
        # what a sheet written for an older version carries. Both are read,
        # and neither is mentioned to the user in the other's place.
        canonical = {'name': 'Name', 'min': 'Min', 'max': 'Max', 'type': 'Type',
                     'unit': 'Unit', 'lowest': 'Min', 'highest': 'Max',
                     # Both spellings: the column is Rule now, and a file
                     # written by 0.5.0 before the rename carries Formula.
                     'rule': wording.FORMULA_LABEL,
                     'formula': wording.FORMULA_LABEL}
        df = df.rename(columns={
            c: canonical[c.strip().lower()]
            for c in df.columns if c.strip().lower() in canonical
        })
        missing = [c for c in ('Name', 'Min', 'Max') if c not in df.columns]
        if missing:
            raise ValueError(wording.ingredients_file_missing_columns(
                ", ".join(_FILE_COLUMNS[c] for c in missing)))

        process_vars = [v for v in self.variables if v.get('category') == 'process']
        # Kept whole until the file is known to be readable: a refusal
        # writes nothing, here as everywhere, and a file refused half way
        # down used to leave the rows it had already read standing.
        variables_before = self.variables
        properties_before = self.ingredient_properties
        try:
            self.variables = []
            self.ingredient_properties = {}

            # Rule among them, so a column of rules is read as what the
            # grid's own Rule cell holds and never as a property of every
            # ingredient.
            standard_cols = {'Name', 'Min', 'Max', 'Type', 'Unit',
                             wording.FORMULA_LABEL}
            prop_cols = [c for c in df.columns if c not in standard_cols]

            seen_names = set()
            for i, (_, row) in enumerate(df.iterrows()):
                raw_name = row.get('Name')
                blank = raw_name is None or (isinstance(raw_name, float)
                                             and np.isnan(raw_name))
                name = "" if blank else str(raw_name).strip()
                if not name:
                    raise ValueError(wording.file_row_name_blank(i + 2))
                if name.lower() in seen_names:
                    raise ValueError(
                        wording.file_row_duplicate_name(i + 2, name))
                if is_reserved_name(name):
                    raise ValueError(
                        wording.file_row_reserved_name(i + 2, name))
                seen_names.add(name.lower())
                # A formula is data a bench can bring in a file: the cell is
                # read exactly as the grid's own is, and a row that carries one
                # has no Lowest and no Highest to fill in.
                formula = _file_text(row, wording.FORMULA_LABEL,
                                     wording.FORMULA_LABEL in df.columns)
                if formula:
                    # A worked-out row may still carry a range, dormant: the
                    # grid shows the word instead of it, and rubbing the
                    # formula out gives the row its amounts back rather than
                    # a row pinned at nothing. A file that leaves the two
                    # cells blank leaves the row with none.
                    min_val, max_val = _file_range(row)
                else:
                    try:
                        min_val, max_val = float(row['Min']), float(row['Max'])
                    except (ValueError, TypeError):
                        raise ValueError(
                            wording.file_amounts_not_numbers(row['Name']))
                    if min_val > max_val:
                        raise ValueError(wording.file_lowest_above_highest(
                            name, min_val, max_val))
                var = {
                    'name': name,
                    'type': 'continuous',
                    'bounds': (min_val, max_val),
                    'category': 'ingredient',
                    'vendor': "",
                    'sku': "",
                    'formula': formula,
                    # Through the parser, never a string compare: '=rest'
                    # and '= REST' are the rest to every other reader of
                    # this cell, and a file that set the flag by spelling
                    # left the project with a rest row half the app could
                    # not see.
                    'balance': formula_is_rest(formula),
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
                            # The file's own capitalisation, kept: the
                            # picker and the limits list show this name,
                            # and 'Fat per 100 g' lower-cased read as a
                            # different column from the one the caption
                            # above it names. Matching ignores case.
                            props[str(col).strip()] = val
                    except (ValueError, TypeError):
                        pass
                self.ingredient_properties[name] = props

            # The settings go back BEFORE the formulas are read: a formula may
            # name one, and a project asked about its own rows without them
            # called them unknown.
            self.variables.extend(process_vars)
            self._check_file_formulas()
        except Exception:
            self.variables = variables_before
            self.ingredient_properties = properties_before
            raise
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

    def _check_file_formulas(self):
        """Refuse a file whose Formula column cannot be read, before it is
        saved.

        The same two questions the grid asks over a finished grid, asked of
        the rows the file just built: one row at most takes the rest, and no
        formula leads back to itself. A formula is read with `batch size`
        allowed whatever the project's default is now — a file is loaded
        before the size is set, and a project with none answers 0 — so the
        one thing not asked here is a default a later save will supply.

        Every formula row is parsed, the rest ones included: skipping the
        rows a flag called the rest meant the flag was trusted to say what
        the text says, and the two could differ.
        """
        if not any(self.has_formula(v) for v in self.variables):
            return
        balance = [v for v in self.variables
                   if formula_is_rest(v.get('formula') or "")]
        if len(balance) > 1:
            raise ValueError(wording.one_balance_only(
                number_list([v['name'] for v in balance]),
                many=len(balance) > 2))
        for var in self.variables:
            if not var.get('formula'):
                continue
            form = parse_formula(var['formula'],
                                 [v['name'] for v in self.variables], True)
            if var['name'] in form.names():
                raise ValueError(wording.RULE_USES_ITS_OWN_ROW)
            trouble = self._rule_names_a_setting(form)
            if trouble:
                raise ValueError(trouble)
        self._formula_order()

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
                # A setting that carries a baseline can have it corrected:
                # it is what every formulation already made is read at, so
                # the encoded history moves with it. And it has to stay
                # inside the allowed amounts either way — a baseline outside
                # them is an amount the history encodes at and the setting
                # says it cannot take.
                stored = var.get('_absent_value')
                carried = (float(baseline) if baseline is not None
                           and stored is not None else stored)
                # FIXING a setting is the one case where the baseline is
                # allowed to sit outside: the range is then a decision about
                # the next round, while the baseline is a fact about bakes
                # already done at another setting. "Baseline 175 must be
                # between 190 and 190" refused a thing the user is entitled
                # to ask for, in a sentence that reads as a fault.
                #
                # "No baseline was passed" is not the test — the editor
                # prefilled the stored one and the grid sends it straight
                # back, so the screen never took this door. The test is
                # whether the baseline is being MOVED.
                fixing = min_val == max_val and (
                    baseline is None or stored is None
                    or float(baseline) == float(stored))
                if (carried is not None and not fixing
                        and not (min_val <= float(carried) <= max_val)):
                    raise ValueError(_baseline_outside_message(
                        carried, min_val, max_val))
                self._check_fixed_feasible(name, min_val, max_val, 'process')
                var['bounds'] = (min_val, max_val)
                var['unit'] = unit
                if carried is not None and float(carried) != float(stored):
                    var['_absent_value'] = float(carried)
                    self._reencode_history()
                self._drop_pending_batch()
                self.save()
                return
        self._check_fixed_feasible(name, min_val, max_val, 'process')
        var = {
            'name': name,
            'type': 'continuous',
            'bounds': (min_val, max_val),
            'category': 'process',
            'unit': unit,
        }
        if self.X_history:
            if len(self.recipe_history) != len(self.X_history):
                raise ValueError(wording.CANNOT_ADD_WITHOUT_AMOUNTS)
            if baseline is None:
                raise ValueError(wording.BASELINE_REQUIRED_FOR_A_SETTING)
            baseline = float(baseline)
            if not (min_val <= baseline <= max_val):
                raise ValueError(_baseline_outside_message(
                    baseline, min_val, max_val))
            var['_absent_value'] = baseline
        self.variables.append(var)
        if self.X_history:
            self._reencode_history()
        self._drop_pending_batch()
        self.save()

    def remove_process_parameter(self, name):
        """Remove a process parameter by name.

        Refused while another row's formula reads it: a cook temperature
        can be half of what a row is worked out from, and deleting it would
        leave that formula naming nothing."""
        if any(v['name'] == name and v.get('category') == 'process'
               for v in self.variables):
            trouble = self._formula_reads_refusal(name)
            if trouble:
                raise ValueError(trouble)
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
            raise ValueError(wording.MEASUREMENT_NAME_REQUIRED)
        if any(name.lower() == v['name'].lower() for v in self.variables):
            raise ValueError(wording.name_is_a_variable(name))
        # Exact name replaces (this method is add-or-replace); a second
        # spelling of it would be a second measurement with one name.
        if any(obj['name'] != name and obj['name'].lower() == name.lower()
               for obj in self.objectives):
            raise ValueError(wording.MEASUREMENT_EXISTS_ERROR)
        if is_reserved_name(name):
            raise ValueError(wording.reserved_name_short(name))
        weight = float(weight)
        if weight <= 0:
            raise ValueError(wording.SHARE_REQUIRED_ERROR)
        min_val = float(min_val) if min_val is not None else 0.0
        max_val = float(max_val) if max_val is not None else 10.0
        if min_val >= max_val:
            raise ValueError(RANGE_ENDS_ERROR)
        if goal == 'target':
            if target is None:
                raise ValueError(TARGET_REQUIRED_ERROR)
            target = float(target)
            if not (min_val <= target <= max_val):
                raise ValueError(_target_outside_message(target, min_val,
                                                         max_val))
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

    def _check_rename_objective(self, name, new_name):
        """The stripped name `rename_objective` would give this measurement,
        or a ValueError saying why it cannot have it. Separate from the
        rename itself so a grid Save can refuse the whole thing before it
        writes a word."""
        obj = next((o for o in self.objectives if o['name'] == name), None)
        if obj is None:
            raise ValueError(wording.no_measurement_named(name))
        new_name = str(new_name).strip()
        if not new_name:
            raise ValueError(wording.NAME_REQUIRED_ERROR)
        if new_name == name:
            return name
        if is_reserved_name(new_name):
            raise ValueError(_reserved_name_message(new_name))
        # A measurement heads a column of the same tables an ingredient
        # does, so it is refused for the same clashes in the same words.
        self._name_is_free(new_name, skip=obj)
        return new_name

    def rename_objective(self, name, new_name):
        """Give one measurement a different name, keeping every result
        recorded under the old one.

        A measurement's name is a KEY: every row of results_history is a
        dict filed under it, and the utility of every formulation is worked
        out by looking it up. So the rename moves the objective and every
        one of those keys together, and the scores do not move at all —
        which is why nothing is recalculated here and no copy is kept.

        Where the targets came from is a note about the project, not about
        this measurement, and is left exactly as it was.
        """
        obj = next((o for o in self.objectives if o['name'] == name), None)
        new_name = self._check_rename_objective(name, new_name)
        if new_name == name:
            return
        obj['name'] = new_name
        for results in self.results_history:
            if name in results:
                results[new_name] = results.pop(name)
        self.save()

    def remove_objective(self, name):
        """Remove an objective and recalculate stored utility scores. What
        is left shares the whole 100 between them."""
        self.objectives = [obj for obj in self.objectives if obj['name'] != name]
        self._shares_to_100()
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
            if qc.get('source') == 'formulation_total':
                # Not pruned like the others: the total's limit is over every
                # ingredient by definition, so it is kept here and rewritten
                # (or dropped, once, with its own reason) below.
                kept.append(qc)
                continue
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
        # The Total of each formulation limit is not pruned like the others:
        # it is over every ingredient by definition, so the same edits that
        # empty a chosen-ingredients limit of meaning simply move it. It is
        # rewritten over the list as it now stands, or it goes and says why.
        removed += self._sync_formulation_total()
        return removed

    def set_variable_unit(self, name, unit):
        """The unit one row of What you can vary is written in, whichever kind
        it is. A process setting carries its own unit — a cook temperature is
        in °C — and is part of no sum and no average, so setting one can break
        no limit; an ingredient's goes the long way round, through the pruning
        every unit change owes the limits."""
        var = next((v for v in self.variables if v['name'] == name), None)
        if var is None:
            raise ValueError(wording.no_variable_named(name))
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
            raise ValueError(wording.no_ingredient_named(name))
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

    def grid_properties(self):
        """`properties()`, minus the properties grid's own row-column name.

        A property called "Ingredient" can only ever have arrived as a
        column of an old ingredient file (`add_property` refuses the name,
        it being reserved) and cannot be drawn as a column of a grid whose
        row column already carries it — so it is kept out of the grid, the
        limit picker built from it, and the values `Add a property` writes,
        while `properties()` itself keeps naming it so it can still be
        found and deleted.
        """
        return [p for p in self.properties() if p != wording.PROPERTIES_ROW_COLUMN]

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
            raise ValueError(wording.NAME_REQUIRED_ERROR)
        if is_reserved_name(name):
            raise ValueError(wording.reserved_name_message(name))
        self._name_is_free(name)
        self._remember_property(name)
        self.save()
        return name

    def remove_property(self, name):
        """Remove a property, the values every ingredient holds for it and any
        limit written against it. Returns the limits that went, so the screen
        can name them."""
        stored = self._known_property(name)
        if stored is None:
            raise ValueError(wording.no_property_named(name))
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
            raise ValueError(wording.no_ingredient_named(ingredient))
        stored = self._known_property(metric)
        if stored is None:
            raise ValueError(wording.no_property_named(metric))
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
        return wording.per_amount_text(self.one_amount_unit() or "g")

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
        return wording.enter_in_this_unit(
            number_list(odd), target or wording.NO_UNIT,
            instead_of=(others[0] or wording.NO_UNIT
                        if len(others) == 1 else None))

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

    def batch_total_text(self, total):
        """'150 g' — the total a batch is made to, written out. Tab 2's
        caption under the downloads and tab 3's `Amounts to make it` heading
        name the same number, so it is written in one place. Blank for a
        batch made as generated, which has no total to name."""
        if total is None:
            return ""
        return join_unit(f"{float(total):g}", self.one_amount_unit() or "")

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
        return wording.total_column("" if unit is None else unit)

    def _total_cell(self, recipe):
        """What goes in that column: a number while there is one unit, the
        written-out per-unit total otherwise.

        Rounded to the two decimals every amount on screen and on paper is
        written to. Left raw it read `250.00000000000003` — a sum of
        rounded-looking numbers that was not the number underneath them."""
        if self.one_amount_unit() is None:
            return self.total_text(recipe)
        return round(float(self.ingredient_total(recipe)), 2)

    def update_objective(self, name, /, **fields):
        """Change a measurement in place. Its name is fixed (renaming would
        orphan every stored result), everything else can change, and every
        stored overall score is recalculated. The open batch is untouched:
        formulations do not depend on measurements."""
        obj = next((o for o in self.objectives if o['name'] == name), None)
        if obj is None:
            raise ValueError(wording.no_measurement_named(name))
        allowed = {'weight', 'goal', 'target', 'min_val', 'max_val', 'unit'}
        unknown = sorted(set(fields) - allowed)
        if unknown:
            raise ValueError(wording.cannot_change(", ".join(unknown)))
        merged = dict(obj)
        merged.update(fields)
        weight = float(merged.get('weight', 1.0))
        if weight <= 0:
            raise ValueError(wording.SHARE_REQUIRED_ERROR)
        min_val = float(merged.get('min_val', 0.0))
        max_val = float(merged.get('max_val', 10.0))
        if min_val >= max_val:
            raise ValueError(RANGE_ENDS_ERROR)
        goal = merged.get('goal', 'max')
        if goal == 'target':
            if merged.get('target') is None:
                raise ValueError(TARGET_REQUIRED_ERROR)
            target = float(merged['target'])
            if not (min_val <= target <= max_val):
                raise ValueError(_target_outside_message(target, min_val,
                                                         max_val))
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

    def _shares_to_100(self):
        """Hold the one invariant 0.5.0 rests on: what each measurement is
        worth is its SHARE of the score, and the shares add up to 100.

        Before 0.5.0 an importance was any positive number and the ceiling
        was whatever they summed to — 2.50 for the sample's 1.5 and 1. The
        column on the grid is typed in percent, so there is one scale now
        and everything that can change the set of measurements comes
        through here: adding one, deleting one, typing the column, and
        opening a file written before the rule existed.

        `add_objective` is deliberately NOT one of them. Its `weight` is a
        number on whatever scale the caller is using, and a series of adds
        has no last one the model can recognise; normalizing after each
        would measure the second against a first already rewritten as 100.
        So a project assembled in memory keeps the scale it was assembled
        on, and is brought onto this one the moment it is opened — which is
        how every screen sees it, because every screen reads the project
        back off the file.

        It is a change of units and nothing else: every ratio, every
        closeness and therefore the ORDER of every formulation is untouched,
        and the score each one reads moves by the same factor. Idempotent,
        so a project already at 100 is left alone and its mtime with it.
        """
        total = sum(float(o['weight']) for o in self.objectives)
        if not self.objectives or total <= 0 or abs(total - 100.0) <= 1e-9:
            return
        for obj in self.objectives:
            obj['weight'] = float(obj['weight']) * 100.0 / total

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
            raise ValueError(wording.no_measurement_named(name))
        total = sum(float(o['weight']) for o in self.objectives)
        return float(obj['weight']) / total if total else 0.0

    def share_percents(self):
        """Every measurement's share of the score as whole percents that add
        up to 100, keyed by name.

        Largest remainder, not one round() each: three equally important
        measurements are 33.33 % apiece, and rounding them one at a time put
        "33 %" three times in a column headed Share of score — a column the
        reader adds up. The odd point goes to the measurement the screens
        list first, so the table reads top-heavy rather than arbitrarily."""
        ordered = self.measurements_by_importance()
        if not ordered:
            return {}
        exact = [100 * self.share_of_score(o['name']) for o in ordered]
        whole = [int(v) for v in exact]          # floor: every share is >= 0
        left = 100 - sum(whole)
        # Biggest fraction first; a tie goes to the one listed first.
        order = sorted(range(len(ordered)),
                       key=lambda i: (-(exact[i] - whole[i]), i))
        for i in order[:max(0, left)]:
            whole[i] += 1
        return {o['name']: whole[i] for i, o in enumerate(ordered)}

    def share_text(self, name):
        """'60 %' — this measurement's share of the score, as a whole
        percent."""
        shares = self.share_percents()
        if name not in shares:
            # Keeps share_of_score's own refusal for an unknown name.
            self.share_of_score(name)
        return join_unit(f"{shares[name]:d}", "%")

    def score_function_line(self):
        """The one line under the measurements grid that writes the score
        out, in the shares the reader typed and nothing else.

        It used to carry the importance as well — "1.5 (60 %) × Firmness" —
        because the two were different numbers. Since 0.5.0 they are one
        number, so the line says it once."""
        if not self.objectives:
            return ""
        # How closeness is worked out belongs in the expander below this
        # line, per goal: two of the three goals have no target at all, so a
        # sentence about distance from one was wrong on most screens. The
        # sentence itself is in wording, like every other word on a screen;
        # what is assembled here is the list of terms.
        terms = " + ".join(
            wording.score_term(self.share_text(o['name']), o['name'])
            for o in self.measurements_by_importance()
        )
        return wording.score_function_line(terms, self.utility_ceiling())

    def set_targets_source(self, text):
        """Remember where the measurement targets came from. Written only on
        a change: the box posts back on every render while it is open, and a
        save with nothing new would bump the file's mtime and make another
        open window see a false conflict."""
        value = "" if text is None else str(text).strip()
        if value == getattr(self, 'targets_source', ""):
            return
        self.targets_source = value
        self.save()

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
            goal = goal_text(obj)
            raw = results.get(obj['name'])
            if raw is None:
                rows.append({'name': name, 'goal': goal,
                             'measured': wording.NOT_MEASURED,
                             'off_by': (wording.NOT_MEASURED if is_target
                                        else "—")})
                continue
            val = float(raw)
            measured = join_unit(f"{val:g}", unit)
            if not is_target:
                rows.append({'name': name, 'goal': goal,
                             'measured': measured, 'off_by': "—"})
                continue
            delta = val - float(obj['target'])
            if abs(delta) < 1e-9:
                off_by = wording.ON_TARGET
            else:
                size = join_unit(f"{abs(delta):g}", unit)
                off_by = (wording.off_by_high(size) if delta > 0
                          else wording.off_by_low(size))
            rows.append({'name': name, 'goal': goal,
                         'measured': measured, 'off_by': off_by})
        return rows

    def _variable_deltas(self, recipe, ref_recipe, category=None, floor=0.005):
        """Every change from ref_recipe to recipe — largest first, as
        (name, change) pairs. `category` keeps to the ingredients or to the
        process settings; None takes both. A change smaller than `floor` is
        left out, and so is a held variable: it is held at one value in
        every new formulation, so it cannot be a change this batch made;
        naming it as the biggest one pointed at the row nobody moved.

        Amounts and settings are asked for separately wherever they are
        written out, because they are not comparable numbers: reporting a
        cook temperature as "+198.65 g" priced an oven in grams.

        A key missing from either side is read as the variable's 'absent'
        value — 0 for an ingredient, its baseline for a process setting —
        which is the rule _encode follows, so a setting added mid-run is not
        reported as having moved by the whole of that baseline. A categorical
        variable has no change to subtract and falls out here.

        The sort is stable, so two changes of the same size stay in set-up
        order rather than swapping between two runs of the same batch."""
        pairs = []
        for var in self.variables:
            if category is not None and \
                    var.get('category', 'ingredient') != category:
                continue
            if not self.has_formula(var) and self.is_fixed(var):
                continue
            name = var['name']
            absent = var.get('_absent_value', 0.0)
            try:
                delta = (float(recipe.get(name, absent))
                         - float(ref_recipe.get(name, absent)))
            except (TypeError, ValueError):
                continue
            if abs(delta) < floor:
                continue
            pairs.append((name, delta))
        pairs.sort(key=lambda kv: abs(kv[1]), reverse=True)
        return pairs

    def best_formulation_no(self):
        """The number of the highest-scoring formulation, or None. Several
        screens name it, and they must all mean the same formulation."""
        i = self.best_index()
        return None if i is None else int(self.formulation_ids[i])

    def _best_recipe_index(self):
        """The best formulation's position while its amounts are on file, so
        that there is something to subtract from. None otherwise."""
        best = self.best_index()
        if best is None or best >= len(self.recipe_history):
            return None
        return best

    def _best_to_compare(self):
        """The best formulation's position while it is what the next batch is
        measured against. During the cold start it is not: those formulations
        are spread across the allowed amounts rather than stepped away from a
        best, so there is a best-so-far but nothing to compare with."""
        if len(self.X_history) < COLD_START_RUNS:
            return None
        return self._best_recipe_index()

    def vs_best_text(self, recipe, n=3, scale_to=None):
        """What this formulation changes from the best one so far: the `n`
        largest amount changes among the ingredients, largest first, then the
        SETTINGS_SHOWN largest among the process settings — the settings have
        a count of their own, so widening the amounts does not also lengthen
        the list of dials. '' while there is no best.

        Amounts and settings are never ranked against each other — a change
        of 12 g and a change of 12 °C are not comparable numbers — and the
        amounts come first because they are weighed out before anything is
        dialled in. Each number wears its own variable's unit; an amount is
        written to the two decimals a balance works to and a setting is not,
        because no oven dial reads 180.00.

        `scale_to` rewrites this formulation AND the best one to that total
        before the two are compared, so a change read beside a scaled table
        is the difference between two numbers the screen actually shows."""
        best = self._best_recipe_index()
        if best is None:
            return ""
        mine = self.scaled_recipe(recipe, scale_to)
        ref = self.scaled_recipe(self.recipe_history[best], scale_to)
        parts = [
            wording.change_text(name, delta,
                                fmt_amount(abs(delta), self.unit_of(name)))
            for name, delta in self._variable_deltas(mine, ref, 'ingredient')[:n]
        ]
        parts += [
            wording.change_text(name, delta,
                                fmt_setting(abs(delta), self.unit_of(name)))
            for name, delta
            in self._variable_deltas(mine, ref, 'process')[:SETTINGS_SHOWN]
        ]
        return ", ".join(parts)

    def suggestion_kind(self, recipe):
        """Whether this formulation stays near the best one or strikes out:
        `close to the best` while the largest change is at most
        CLOSE_TO_THE_BEST of that variable's own allowed range, and
        `trying something different` otherwise.

        Normalised, because the raw numbers are not comparable: 5 g of salt
        out of an allowed 0 to 10 g is a bold move, 5 °C out of an allowed
        20 to 200 °C is not. Until the cold start is over there is nothing to
        be close to — the formulations are spread out to learn the space —
        so every row says that instead.

        The amounts are the ones the project stores, never a scaled copy: the
        allowed amounts this is a fraction of are the project's own."""
        best = self._best_to_compare()
        if best is None:
            return wording.SUGGESTION_SPREAD
        # Every kind of variable at once, and no floor: a move of half a gram
        # is too small to be worth naming on the sheet but not too small to
        # decide what this formulation is, if half a gram is what it is
        # allowed to move at all.
        deltas = self._variable_deltas(recipe, self.recipe_history[best],
                                       floor=0.0)
        spans = {v['name']: float(v['bounds'][1]) - float(v['bounds'][0])
                 for v in self.variables if v['type'] == 'continuous'}
        largest = max((abs(delta) / spans[name] for name, delta in deltas
                       if spans.get(name)), default=0.0)
        return (wording.SUGGESTION_CLOSE if largest <= CLOSE_TO_THE_BEST
                else wording.SUGGESTION_DIFFERENT)

    def compared_with_column(self):
        """The header of the what-is-it-trying column: the best formulation's
        number, or — while the cold start is still spreading formulations out
        — the allowed amounts they are spread across."""
        if self._best_to_compare() is None:
            return wording.COMPARED_WITH_ALLOWED
        return wording.compared_with_column(self.best_formulation_no())

    def compared_with_text(self, recipe, scale_to=None, own=False):
        """One cell of the batch table, and the same line on the sheet: what
        kind of formulation this is, and the amounts that carry it.

        During the cold start nothing is listed. There is a best-so-far from
        the very first result, but these formulations were not stepped away
        from it — they are spread across the allowed amounts — so naming the
        amounts they happen to differ by would claim a reason nobody had.

        `own` marks a formulation the user typed. It is not a suggestion, so
        it has no kind: the app describing its own sampling ('spread across
        the allowed amounts') on a row somebody wrote out by hand was the
        app taking credit for their bench standard. The changes from the best
        still follow — those are a fact about the amounts."""
        if own:
            return wording.compared_with_cell(
                wording.OWN_FORMULATION_KIND,
                self.vs_best_text(recipe, scale_to=scale_to))
        kind = self.suggestion_kind(recipe)
        changes = ("" if kind == wording.SUGGESTION_SPREAD
                   else self.vs_best_text(recipe, scale_to=scale_to))
        return wording.compared_with_cell(kind, changes)

    def ingredient_total(self, recipe):
        """The formulation total: process settings are not amounts and are excluded."""
        return sum(float(recipe.get(v['name'], 0.0)) for v in self.variables
                   if v.get('category', 'ingredient') == 'ingredient')

    def scaled_recipe(self, recipe, scale_to=None):
        """The same formulation written for a different formulation total. Every
        screen, sheet and download that shows a scaled amount goes through
        this, so what is printed always equals what is displayed.

        A row that is WORKED OUT is not scaled, it is worked out again at
        the new size: multiplying `= 5 + 0.1 × Flour` by two makes 14 where
        the formula says 9, and the row would then be the one thing on the
        sheet that does not do what the Formula cell says. The rows the
        search moves are scaled, `batch size` reads the new size, and the
        formulas are filled in after — which is what puts a balance row
        exactly on the size that was asked for.

        Which means the factor is not the ratio of the two sizes. A
        formula's constant term does not scale — `= 5 + 0.1 × Flour` keeps
        its 5 — so scaling the other rows by size/total lands the round
        NEXT to the size it was asked for, and every row on the sheet then
        wears a caption apologising for it. The factor is solved instead:
        the sum is linear in it, so there is exactly one that puts the
        total on the size, and it is the plain ratio whenever no formula
        holds a constant.
        """
        if scale_to is None:
            return dict(recipe)
        total = self.ingredient_total(recipe)
        if total <= 0:
            return dict(recipe)
        factor = self._scale_factor(recipe, float(scale_to), total)
        out = dict(recipe)
        for var in self.variables:
            if self.has_formula(var):
                continue
            value = float(recipe.get(var['name'], 0.0))
            out[var['name']] = (value * factor
                                if var.get('category', 'ingredient') == 'ingredient'
                                else value)
        return self.fill_formulas(out, batch=scale_to)

    def _scale_factor(self, recipe, scale_to, total):
        """What to multiply the rows the search moves by, so that the
        formulation adds up to `scale_to` once the formulas are worked out
        again at that size.

        The sum is linear in the factor: each row the search moves is worth
        its own gram plus whatever every formula reads it at, and what a
        formula holds that does NOT move with the rows — its constant, its
        `batch size` term, a process setting it reads — is the same number
        whatever the factor is. So one division answers it. The plain ratio
        comes back when nothing holds a constant, which is every project
        without a formula and every proportional formula there is.
        """
        names = {v['name'] for v in self.variables
                 if v.get('category', 'ingredient') == 'ingredient'}
        try:
            form = self._weighted_form(
                lambda name: 1.0 if name in names else 0.0)
        except (FormulaError, ValueError):
            return scale_to / total
        fixed_part = form.const + form.batch * scale_to
        moving = 0.0
        by_name = self._by_name()
        for name, coeff in form.terms.items():
            var = by_name.get(name)
            value = _amount(recipe.get(name))
            if var is not None and var.get('category',
                                           'ingredient') != 'ingredient':
                # A cook temperature is not an amount of anything and is
                # not scaled, so a formula reading one reads the same
                # number whatever the factor is.
                fixed_part += coeff * value
            else:
                moving += coeff * value
        if abs(moving) <= 1e-12:
            # Nothing the factor touches reaches the total — a balance row
            # makes every factor land on the size — so the rows keep the
            # proportions they came with.
            return scale_to / total
        factor = (scale_to - fixed_part) / moving
        return factor if factor > 0 else scale_to / total

    def _rewrites_amounts(self, own=False):
        """Whether a row is rewritten for a size on the way to the screen.

        Under a PROJECT default every suggestion is BUILT to the size — the
        cold start projects onto it and a warm round is snapped onto it — so
        there is nothing to rewrite, and a row that could not be snapped
        without breaking a limit stands at the band edge and says so. A
        formulation of the user's own is never rewritten either: the sheet
        told the bench to weigh out 20.62 g while the model learned the
        20.00 g that was typed, and the two disagreed about what was made.

        For the round on the bench this is now a NO-OP, and deliberately
        left standing rather than deleted. Until 0.5.0 the Batch size box
        scaled the picture and the stored rows kept the amounts the model
        proposed; `scale_round` moves the amounts themselves, so a sized
        round already sums to its size and rescaling it to that size is the
        identity (TestTheDisplayRescaleIsOnlyForOlderRounds pins it). What
        still needs the rescale is a round RECORDED before 0.5.0: its rows
        are as generated and its stored size is what its sheet was printed
        to, and tab 3 hands back what the bench weighed out.
        """
        return not own and not self.has_formulation_total()

    def shown_recipe(self, row, total):
        """(amounts, basis) — what one row of a batch is SHOWN and PRINTED
        at, and the total its `%` column is a share of.

        One accessor for the table, the summary sheet and the formulation
        sheets, so the paper in the technician's hand can never carry
        different numbers from the screen it was downloaded from. `basis` is
        None for a row shown at its own sum: each share is then of that
        formulation's own total, and the column still adds to 100.
        """
        recipe = row['recipe'] if isinstance(row, dict) and 'recipe' in row else row
        own = bool(isinstance(row, dict) and row.get('note'))
        if total is not None and self._rewrites_amounts(own):
            return self.scaled_recipe(recipe, total), float(total)
        return dict(recipe), None

    def total_mismatch(self, number, recipe, total):
        """'Formulation 4 adds up to 97.00 g, not the 100 g total.', or ''
        when it lands on it.

        Nothing is rescaled to hide the difference, so this line is the only
        thing that says it — under the batch table, and on the row's own
        sheet."""
        if total is None or self.one_amount_unit() is None:
            return ""
        made = self.ingredient_total(recipe)
        if round(made, 2) == round(float(total), 2):
            return ""
        return wording.total_mismatch_caption(
            number, join_unit(f"{made:.2f}", self.one_amount_unit() or ""),
            self.batch_total_text(total))

    def total_mismatch_lines(self, rows, total):
        """One line per row of a batch that does not add up to the total, in
        the order the table shows them."""
        lines = [self.total_mismatch(row['formulation'],
                                     self.shown_recipe(row, total)[0], total)
                 for row in self._batch_rows(rows)]
        return [line for line in lines if line]

    def _recompute_utilities(self):
        for i, results_dict in enumerate(self.results_history):
            if i < len(self.Y_history):
                self.Y_history[i] = self._compute_utility(results_dict)

    def utility_ceiling(self):
        """The overall score a formulation that hits every goal would get:
        the sum of the shares, which is 100 for any project saved since
        0.5.0 (see set_shares and _rescale_shares_to_100)."""
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

    def _amount_column(self, name, mark=False):
        """The table header for one variable, carrying that variable's own
        unit: `Water (ml)` beside `Pea protein (g)`, and a process setting
        with its own unit or none — a cook temperature must never read
        "Cook temperature (g)".

        `mark` adds the worked-out mark the printed sheets already carry:
        `Water · worked out (g)`. The mark lived only where the reader
        could not ask a question about it — on paper — and was missing from
        the one screen a click from Set up, so the round table and the
        sheets printed different names for the same row."""
        shown = (wording.worked_out_label(name)
                 if mark and self.has_formula(self._by_name().get(name, {}))
                 else name)
        unit = self.unit_of(name)
        return f"{shown} ({unit})" if unit else shown

    def _recorded_recipe(self, index):
        """What formulation `index` was actually made to.

        A formula row has no column in the encoded history — it left the
        search vector — so decoding alone would work it out AGAIN, from
        today's formula and today's batch size. That is what the NEXT
        formulation will hold, not what the bench weighed out last week.
        recipe_history keeps the amounts as they were recorded, so they are
        the truth wherever it has them and the decode fills in only the
        rest."""
        recipe = self._decode(self.X_history[index])
        recorded = (self.recipe_history[index]
                    if index < len(self.recipe_history) else {})
        for var in self._formula_rows():
            if var['name'] in recorded:
                recipe[var['name']] = recorded[var['name']]
        return recipe

    def _amount_columns(self, recipe):
        return {self._amount_column(v['name']): recipe.get(v['name'])
                for v in self.variables}

    def history_frame(self, order=wording.SORT_BEST_FIRST,
                      include_amounts=False):
        """Every formulation — scored and left out — as the All formulations
        table shows them. `order` is 'Best first', 'Newest first' or
        wording.SORT_BATCH_ORDER's value. Round is a string in every row: a
        project that predates batches has blanks, and a mixed int/blank
        column renders inconsistently."""
        objs = self.measurements_by_importance()
        best_i = self.best_index()
        rows = []
        for i in range(len(self.X_history)):
            results = self.results_history[i] if i < len(self.results_history) else {}
            unmeasured = [o['name'] for o in objs if o['name'] not in results]
            ts = self.timestamps_history[i] if i < len(self.timestamps_history) else None
            batch = self.batch_history[i] if i < len(self.batch_history) else None
            row = {
                wording.BEST_SO_FAR_COLUMN: "★" if i == best_i else "",
                wording.ROUND_CAP: "" if batch is None else str(int(batch)),
                "Formulation": int(self.formulation_ids[i]),
                "_score": float(self.Y_history[i]),
                "_seq": i,
                "_batch": 0 if batch is None else int(batch),
            }
            for obj in objs:
                row[self._measurement_column(obj)] = results.get(obj['name'])
            # '2.30 · Juiciness not measured', in the separator the rest of
            # the app reads a list with. The measurement is NAMED: a score
            # missing one is not the same number as a complete one, and
            # "partial" made the reader go and find out which.
            row[wording.OVERALL_SCORE_COLUMN] = (
                f"{float(self.Y_history[i]):.2f}"
                + (wording.not_measured_tail(number_list(unmeasured))
                   if unmeasured else ""))
            row[wording.DATE_RECORDED_COLUMN] = local_date(ts)
            row["Note"] = self.notes_history[i] if i < len(self.notes_history) else ""
            if include_amounts:
                row.update(self._amount_columns(self._recorded_recipe(i)))
            rows.append(row)
        for k, s in enumerate(self.skipped):
            batch = s.get(ROUND_FIELD)
            row = {
                # Best is a star or nothing. "Not scored" belongs in the
                # Note column, which already carries it, and a Best column
                # with words in it read as a third kind of score.
                wording.BEST_SO_FAR_COLUMN: "",
                wording.ROUND_CAP: "" if batch is None else str(int(batch)),
                "Formulation": int(s['formulation']),
                "_score": float('-inf'),
                "_seq": len(self.X_history) + k,
                "_batch": 0 if batch is None else int(batch),
            }
            for obj in objs:
                row[self._measurement_column(obj)] = None
            row[wording.OVERALL_SCORE_COLUMN] = ""
            row[wording.DATE_RECORDED_COLUMN] = ""
            row["Note"] = s.get('note') or wording.NOT_SCORED
            if include_amounts:
                row.update(self._amount_columns(s.get('recipe', {})))
            rows.append(row)
        columns = ([wording.BEST_SO_FAR_COLUMN, wording.ROUND_CAP,
                    "Formulation"]
                   + [self._measurement_column(o) for o in objs]
                   + [wording.OVERALL_SCORE_COLUMN, wording.DATE_RECORDED_COLUMN,
                      "Note"])
        if include_amounts:
            columns += [self._amount_column(v['name']) for v in self.variables]
        if not rows:
            return pd.DataFrame(columns=columns)
        df = pd.DataFrame(rows)
        if order == wording.SORT_NEWEST_FIRST:
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
        # What each formulation is trying, last: it is the only column of
        # words among the numbers, and it reads as the answer to the row
        # rather than another figure to weigh out.
        #
        # During the cold start there is nothing to compare with, and a
        # column headed 'Compared with the allowed amounts' whose every cell
        # repeated it word for word said one thing twice and told the reader
        # nothing about the row. The caption under the table says the first
        # five are spread out; the column comes back when it has a
        # formulation to name.
        trying = self.compared_with_column()
        if trying == wording.COMPARED_WITH_ALLOWED:
            trying = None
        for row in read:
            own = bool(row.get('note'))
            recipe, _ = self.shown_recipe(row, scale_to)
            item = {"Formulation": int(row['formulation'])}
            # Rounded in the FRAME, not only in the formatting: the reader
            # met 18.11529942207362 and a Total (g) of 250.00000000000003 on
            # the screen the bench weighs from, and a balance reads to two
            # decimals. A process setting keeps its own precision — it is
            # dialled in, not weighed.
            for var in ingredients:
                item[self._amount_column(var['name'], mark=True)] = round(
                    float(recipe.get(var['name'], 0.0)), 2)
            if total_col is not None:
                item[total_col] = self._total_cell(recipe)
            for var in process:
                item[self._amount_column(var['name'])] = float(
                    recipe.get(var['name'], 0.0))
            if noted:
                item["Note"] = row.get('note', "")
            if trying is not None:
                # The stored amounts, with the table's own total handed on:
                # the changes are then between two numbers the screen shows.
                item[trying] = self.compared_with_text(
                    row['recipe'], scale_to=scale_to, own=own)
            rows.append(item)
        columns = (["Formulation"]
                   + [self._amount_column(v['name'], mark=True)
                      for v in ingredients]
                   + ([total_col] if total_col is not None else [])
                   + [self._amount_column(v['name']) for v in process]
                   + (["Note"] if noted else [])
                   + ([trying] if trying is not None else []))
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

    def parse_batch_results(self, df, batch, with_skipped=False, weighed=None):
        """Match an uploaded results sheet to the open batch.

        The sheet needs a Formulation column holding the global numbers from
        the downloaded workbook (results_from_workbook transposes the summary
        sheet into exactly this shape). `Recipe` and `Experiment` are
        accepted as legacy headers and read as 1-based positions in the
        batch. Every measurement needs its own column — an absent column is
        refused outright (a typo'd header would otherwise silently drop that
        measurement from every row). A blank cell in a column that IS present
        means that one result could not be scored, so the row is stored as a
        partial result; a row with nothing filled in is refused.

        A `Not scored` column is the sheet's own tick box: a row marked there
        is not a row missing its numbers, it is a formulation nobody scored,
        and it is kept apart rather than refused. Returns [(formulation
        number, {measurement: value}, note), ...], or that and the not-scored
        rows as [(number, note), ...] when `with_skipped` is set.

        `weighed` is what an uploaded workbook's Actual cells said, keyed by
        formulation number (UploadedWorkbook.actual). A row that has any is
        a row whose amounts are no longer the ones the app suggested, and
        its note says so in front of whatever the bench wrote.
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
            raise ValueError(wording.SHEET_NEEDS_A_FORMULATION_COLUMN)
        col_for, missing = {}, []
        for obj in self.objectives:
            key = obj['name'].strip().lower()
            if key in norm:
                col_for[obj['name']] = norm[key]
            else:
                missing.append(obj['name'])
        if missing:
            raise ValueError(wording.sheet_missing_columns(
                ", ".join(missing)))
        note_col = norm.get("note")
        skipped_col = norm.get(wording.NOT_SCORED.lower())
        if len(df) == 0:
            raise ValueError(wording.SHEET_HAS_NO_ROWS)
        in_batch = ", ".join(str(n) for n in numbers)
        # Which formulations came back with amounts of their own.
        weighed = {int(k): v for k, v in (weighed or {}).items()}
        parsed, skipped, seen = [], [], set()
        for _, sheet_row in df.iterrows():
            raw_no = sheet_row[key_col]
            try:
                as_float = float(raw_no)
            except (TypeError, ValueError):
                raise ValueError(
                    wording.formulation_number_not_whole(raw_no))
            if not as_float.is_integer():
                raise ValueError(
                    wording.formulation_number_not_whole(raw_no))
            number = int(as_float)
            if legacy:
                if not (1 <= number <= len(rows)):
                    raise ValueError(wording.formulation_not_in_round(
                        number, self.pending_batch_no, in_batch))
                number = numbers[number - 1]
            elif number not in numbers:
                raise ValueError(wording.formulation_not_in_round(
                    number, self.pending_batch_no, in_batch))
            if number in seen:
                raise ValueError(
                    wording.formulation_twice_in_the_sheet(number))
            seen.add(number)
            note = ""
            if note_col is not None:
                raw_note = sheet_row[note_col]
                if raw_note is not None and not (isinstance(raw_note, float)
                                                 and np.isnan(raw_note)):
                    note = str(raw_note).strip()
            if int(number) in weighed:
                note = wording.amounts_as_weighed_note(note)
            if skipped_col is not None and _is_ticked(sheet_row[skipped_col]):
                # The box was ticked on the sheet: no result to read, and
                # nothing wrong with the row.
                skipped.append((number, note))
                continue
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
                    raise ValueError(
                        wording.formulation_value_not_a_number(number, name))
                obj = next(o for o in self.objectives if o['name'] == name)
                if not (obj['min_val'] <= val <= obj['max_val']):
                    # The same sentence the results grid refuses with.
                    raise ValueError(outside_message(
                        wording.formulation_measurement(number, name), val,
                        obj['min_val'], obj['max_val'], obj.get('unit'),
                        wording.YOUR_RANGE, wording.WIDEN_RANGE_HINT))
                results[name] = val
            if not results:
                raise ValueError(
                    wording.formulation_has_no_measurements(number))
            parsed.append((number, results, note))
        return (parsed, skipped) if with_skipped else parsed

    def history_csv(self):
        """Every formulation the project holds as CSV — the ones with results
        and the not-scored ones — with the identity columns (Formulation,
        Round, Recorded, Overall score) and the Note.

        The download is a workbook now (all_formulations_workbook), and both
        write the one table history_export_frame builds. This is still the
        shape `Record a formulation you already made` accepts from a bench
        that keeps its own spreadsheet, and what those tests hand it.

        Amount and measurement columns carry their own units, exactly as the
        All formulations table and the batch sheets write them: a file whose
        "Water" column meant millilitres while the screen said "Water (ml)"
        was the one place in the app an amount had no unit on it. Measurements
        run by importance, as they do on every screen.

        A not-scored formulation is here too, with its amounts, its note and
        blank measurement cells: it has a number and it is part of the record,
        and leaving it out made the file disagree with the table it was
        downloaded from.

        Amount columns come from the re-encoded history (as history_frame
        does), not raw recipe_history: an ingredient added mid-project is
        backfilled to 0 in X_history for earlier rows, while recipe_history is
        never backfilled and would export a blank there.
        """
        return self.history_export_frame().to_csv(index=False)

    # ------------------------------------------------------------------ #
    #  The workbook
    # ------------------------------------------------------------------ #

    def _ingredients(self):
        return [v for v in self.variables
                if v.get('category', 'ingredient') == 'ingredient']

    def ingredient_names(self):
        """Every ingredient's name, in set-up order.

        The screens ask this three times over — the properties grid, the
        amount-limit picker and tab 3's "not used" line — and each had
        written the category filter out again beside an accessor that
        already knew it."""
        return [v['name'] for v in self._ingredients()]

    def _process_settings(self):
        return [v for v in self.variables if v.get('category') == 'process']

    def _percent_of(self, recipe, var, total):
        """`%` — one amount as a share of the formulation total, to one
        decimal. The share is of the total the sheet was written to, so the
        column adds up to 100 for the numbers printed beside it. Where the
        ingredients are in more than one unit there is no one total to be a
        share of, so each amount is a share of its own unit's total."""
        amount = float(recipe.get(var['name'], 0.0))
        if total is not None and self.one_amount_unit() is not None:
            base = float(total)
        else:
            base = dict(self.unit_totals(recipe)).get(self.unit_of(var['name']), 0.0)
        if not base:
            return None
        return round(amount / float(base) * 100.0, 1)

    def _measurement_sheet_label(self, obj):
        """'Firmness, target 6 N' — how a measurement heads its row on the
        summary sheet and its line on a formulation's own sheet. An uploaded
        workbook is matched back on this exact text."""
        return wording.sheet_measurement_label(
            label_with_unit(obj['name'], obj.get('unit')), goal_line(obj))

    def workbook_bytes(self, batch, total=None, sized=False):
        """The open round as one Excel file: a summary sheet the whole round
        is weighed out from, and one sheet per formulation to carry, tick and
        write on.

        `total` is the batch size every formulation is made to — the
        project's default, or the one typed on the round screen — and the
        amounts are written for it, so the file and the screen can never show
        different numbers. With no size the amounts are as generated and each
        `%` is a share of that formulation's own sum. `sized` says the box
        has already moved these rows onto `total` (see _rewritten); it only
        decides which cautions the sheets carry.

        The file comes back the same way: the summary sheet's Measured cells
        are read straight back off it by results_from_workbook.
        """
        rows = self._batch_rows(batch)
        book = Workbook()
        summary = book.active
        summary.title = wording.batch_sheet_name(self.pending_batch_no)
        self._write_summary_sheet(summary, rows, total, sized)
        for row in rows:
            self._write_formulation_sheet(
                book.create_sheet(
                    wording.formulation_sheet_name(row['formulation'])),
                row, total, sized)
        buffer = io.BytesIO()
        book.save(buffer)
        return buffer.getvalue()

    def _sheet_date(self):
        """The date the sheets are for: when the batch was generated, when
        its first formulation was recorded if it has been closed since, and
        today for a batch with neither. It is on the summary's title line
        because two printouts of Round 2 cannot otherwise be told apart."""
        created = getattr(self, 'pending_batch_created', None)
        if created:
            return str(created)
        for i, batch in enumerate(self.batch_history):
            if (batch == self.pending_batch_no
                    and i < len(self.timestamps_history)):
                return local_date(self.timestamps_history[i])
        return datetime.now().astimezone().strftime("%Y-%m-%d")

    def _shows_shares(self):
        """Whether the sheets carry a `%` column. An amount is a share of the
        formulation total, and a project whose ingredients are in more than
        one unit has no one total to be a share of — its Total cell reads
        '50.00 ml · 25.00 g', and a column of percentages beside it would be
        arithmetic nobody can check."""
        return self.one_amount_unit() is not None

    def _variable_column_head(self):
        """What heads the column of names on the summary sheet: a project
        with process settings in it has them in that column too, and filing
        a cook temperature under 'Ingredient' made the sheet disagree with
        every screen that names the pair."""
        return (wording.INGREDIENT_OR_SETTING_LABEL
                if self._process_settings() else wording.KIND_INGREDIENT)

    def _sheet_ingredient_label(self, name):
        """How an ingredient is named on a sheet. With one unit the column
        header carries it ('Amount (g)') and the row stays bare; with two the
        unit has to ride on every row, or 40 of water and 25 of powder read
        as one column of numbers."""
        return name if self._shows_shares() else self._amount_column(name)

    def _actual_column_head(self):
        """'Actual (g)' — the header of the column the bench writes what it
        really weighed into, carrying the unit the Amount column beside it
        carries. Over the settings block it is a bare 'Actual': a cook
        temperature and a proving time share no unit, and each setting's
        row says its own."""
        unit = self.one_amount_unit()
        return (f"{wording.ACTUAL_COLUMN} ({unit})" if unit
                else wording.ACTUAL_COLUMN)

    @staticmethod
    def _vendor_line(var):
        """'Acme · PP-80' — where an ingredient was bought, printed under
        its name on the page the bench carries. Blank when the project
        never said."""
        return " · ".join(part for part in (str(var.get('vendor') or "").strip(),
                                            str(var.get('sku') or "").strip())
                          if part)

    def _write_summary_sheet(self, sheet, rows, total, sized=False):
        """One column per formulation, one row per ingredient: the sheet a
        bench weighs a whole batch out from, and the one it writes the
        results back onto.

        Row 1 says which batch of which project this is and when it was
        asked for; row 2 says which cells can be written in; row 3 is the
        header the upload finds the formulations by.

        Beside the amounts sit the Lot cells — one per ingredient, not one
        per formulation: a round is weighed out of the sacks that are open
        that morning — and, where the project says so, the vendor and SKU
        that name what to reach for.
        """
        ingredients, process = self._ingredients(), self._process_settings()
        objs = self.measurements_by_importance()
        # What each column is weighed out at, and the total its % is a share
        # of. A row shown at its own sum has no total to be a share of, so
        # its column is a share of that formulation instead.
        shown = [self.shown_recipe(row, total) for row in rows]
        recipes = [recipe for recipe, _ in shown]
        bases = [basis for _, basis in shown]
        shares = self._shows_shares()
        stride = 2 if shares else 1

        def column(j, offset=0):
            return 2 + stride * j + offset

        # The write-in columns, after the last formulation: the Lot wherever
        # anything is weighed out, and the vendor and the SKU only where the
        # project holds them. A column of blanks on every sheet is a column
        # nobody reads, and a project of process settings alone weighs
        # nothing out of any sack.
        lot_column = 2 + stride * len(rows) if ingredients else None
        vendor_column = (lot_column + 1
                         if lot_column and any(str(v.get('vendor') or "").strip()
                                               for v in ingredients) else None)
        sku_column = ((vendor_column or lot_column) + 1
                      if lot_column and any(str(v.get('sku') or "").strip()
                                            for v in ingredients) else None)
        last_column = (sku_column or vendor_column or lot_column
                       or 1 + stride * len(rows))

        title = _write_cell(sheet, 1, 1, wording.summary_title(
            self.pending_batch_no, self.project_name, self._sheet_date(),
            self.batch_total_text(total)))
        title.font = _TITLE_FONT
        # Under the title, because the Lot cells are up here in the amounts
        # and the Measured cells are pages below: one line about the whole
        # sheet belongs where the sheet starts. It names this sheet's own
        # cells; the Actual cells are named on the pages that carry them.
        _write_banner(sheet, 2, 1, wording.SUMMARY_SHADED_NOTE, last_column)

        # The first column carries the settings too when the project has
        # any: they were filed silently under "Ingredient".
        _write_cell(sheet, 3, 1, self._variable_column_head(), bold=True)
        for j, row in enumerate(rows):
            _write_cell(sheet, 3, column(j),
                        wording.formulation_sheet_name(row['formulation']),
                        bold=True)
            if shares:
                _write_cell(sheet, 3, column(j, 1), wording.PERCENT_COLUMN,
                            bold=True)
        if lot_column:
            _write_cell(sheet, 3, lot_column, wording.LOT_COLUMN, bold=True)
        if vendor_column:
            _write_cell(sheet, 3, vendor_column, wording.VENDOR_LABEL,
                        bold=True)
        if sku_column:
            _write_cell(sheet, 3, sku_column, wording.SKU_LABEL, bold=True)

        r = 4
        for var in ingredients:
            label = self._amount_column(var['name'])
            if self.has_formula(var):
                label = wording.worked_out_label(label)
            _write_cell(sheet, r, 1, label, bold=True)
            for j, recipe in enumerate(recipes):
                _write_cell(sheet, r, column(j),
                            round(float(recipe.get(var['name'], 0.0)), 2),
                            number_format=_TWO_DP)
                if shares:
                    _write_cell(sheet, r, column(j, 1),
                                self._percent_of(recipe, var, bases[j]),
                                number_format=_ONE_DP)
            # One lot for the round, written once on the row it belongs to.
            if lot_column:
                _write_in_cell(sheet, r, lot_column)
            if vendor_column:
                _write_cell(sheet, r, vendor_column,
                            str(var.get('vendor') or "").strip() or None)
            if sku_column:
                _write_cell(sheet, r, sku_column,
                            str(var.get('sku') or "").strip() or None)
            r += 1
        if ingredients:
            _write_cell(sheet, r, 1, self.total_column(), bold=True)
            for j, recipe in enumerate(recipes):
                cell = _write_cell(sheet, r, column(j),
                                   self._total_cell(recipe), bold=True)
                if shares:
                    cell.number_format = _TWO_DP
                    _write_cell(sheet, r, column(j, 1), 100.0, bold=True,
                                number_format=_ONE_DP)
            r += 1
        # The settings are on the sheet too: they are dialled in, not weighed
        # out, so they carry no share of the total and no colour.
        for var in process:
            _write_cell(sheet, r, 1, self._amount_column(var['name']), bold=True)
            for j, recipe in enumerate(recipes):
                # A setting is dialled in, not weighed: two decimals at most,
                # as fmt_setting writes it on every screen.
                _write_cell(sheet, r, column(j),
                            round(float(recipe.get(var['name'], 0.0)), 2))
            r += 1
        # The caution belongs with the amounts it is about, directly under
        # them — not at the foot of the sheet, under the signature line. A
        # row that does not add up to the total says so on its own line: the
        # bench weighs out what is printed above, and nothing else on the
        # page would say the column is not the total in the title.
        for line in (self.scaled_cautions(rows, total, sized)
                     + self.total_mismatch_lines(rows, total)):
            _write_cell(sheet, r, 1, line)
            r += 1
        # Said once, only when a row on the sheet is one: the rest of a
        # project with no formula has nothing worked out to explain.
        if self._formula_rows():
            _write_cell(sheet, r, 1, self.worked_out_note())
            r += 1
        r += 1   # a blank line: what to make above it, what to write below

        # The block is headed in the word the app uses for it everywhere
        # else — the same heading the formulation pages give it — so
        # "write in the Measured cells" names something the reader can see
        # on the sheet, and the line under it says what mark the app will
        # read, which is the one thing the paper cannot be asked.
        _write_cell(sheet, r, 1, wording.MEASUREMENTS_SHEET_HEADING,
                    bold=True)
        r += 1
        _write_banner(sheet, r, 1, wording.SHEET_WRITE_IN_NOTE, last_column)
        r += 1
        # The formulation names again, directly above the cells they are
        # the heading for. They are otherwise seven rows up with a % column
        # in between, and the bench had to count columns back up the page
        # to know which formulation a firmness belonged to.
        _write_cell(sheet, r, 1, wording.MEASUREMENT_COLUMN, bold=True)
        for j, row in enumerate(rows):
            _write_cell(sheet, r, column(j),
                        wording.formulation_sheet_name(row['formulation']),
                        bold=True)
        r += 1
        for obj in objs:
            _write_cell(sheet, r, 1, self._measurement_sheet_label(obj))
            for j in range(len(rows)):
                _write_in_cell(sheet, r, column(j))
            r += 1
        # The label is the words; the BOX is in the cell the pen can reach.
        # Printed into the locked label, the one thing the instruction asked
        # the reader to mark was the one cell the sheet would not take a
        # mark in.
        _write_cell(sheet, r, 1, wording.NOT_SCORED_CHECKBOX_SHEET)
        for j in range(len(rows)):
            _write_in_cell(sheet, r, column(j), wording.TICK_BOX)
        r += 1
        _write_cell(sheet, r, 1, wording.NOTE)
        for j, row in enumerate(rows):
            _write_in_cell(sheet, r, column(j), row.get('note') or None,
                           wrap=True)
        r += 1
        _write_cell(sheet, r, 1, wording.SUMMARY_TICK_NOTE)
        r += 2
        # This is the page the whole round is weighed out from and the page
        # the Lot numbers are written on, and it came back from the bench
        # with nothing on it to say whose work it was.
        _write_cell(sheet, r, 1, wording.MADE_BY_FOOTER)

        # The Lot is written in, so it gets a hand's width; the vendor and
        # the SKU are printed, so they get the width of what they say —
        # capped, because one long supplier name must not push the sheet
        # onto a second page.
        def printed_width(key):
            longest = max([len(str(v.get(key) or "").strip())
                           for v in ingredients] or [0])
            return min(30, max(12, longest + 4))

        extra = []
        if lot_column:
            extra.append(14)
        if vendor_column:
            extra.append(printed_width('vendor'))
        if sku_column:
            extra.append(printed_width('sku'))
        _set_widths(sheet, [34] + ([14, 7] if shares else [18])
                    * max(1, len(rows)) + extra)
        sheet.freeze_panes = "B4"
        # The title and the header ride on every printed page: page two of a
        # wide batch is a grid of numbers with nothing to read it by.
        sheet.print_title_rows = "$1:$3"
        # A batch of six formulations is twelve columns wide; portrait would
        # print it in slices.
        _fit_to_page(sheet, r, last_column, landscape=len(rows) > 2)
        _protect(sheet)

    def _write_formulation_sheet(self, sheet, row, total, sized=False):
        """One formulation, as the page a technician carries to the bench:
        what it is trying, what to weigh out in the order it is set up, what
        to dial in, what to measure, and room to sign it."""
        recipe, basis = self.shown_recipe(row, total)
        ingredients, process = self._ingredients(), self._process_settings()
        unit = self.one_amount_unit()
        shares = self._shows_shares()
        # Tick, name, amount and Actual, with the share behind them where
        # the project has one: what the page is printed to, and what an
        # instruction line is merged across.
        page_width = 5 if shares else 4

        title = _write_cell(sheet, 1, 1,
                            wording.sheet_title(row['formulation'],
                                                self.pending_batch_no,
                                                self.project_name))
        title.font = _TITLE_FONT
        # What this formulation is trying. During the cold start the column
        # header and the cell say the same thing, and "Compared with the
        # allowed amounts: Spread across the allowed amounts" is that
        # sentence twice.
        own = bool(row.get('note'))
        column_head = self.compared_with_column()
        cell_text = self.compared_with_text(row['recipe'], scale_to=total,
                                            own=own)
        _write_cell(sheet, 2, 1,
                    cell_text if (own
                                  or column_head == wording.COMPARED_WITH_ALLOWED)
                    else wording.compared_with_line(column_head, cell_text))
        # The page is protected, so it says up front which cells still take
        # a number — the Actual cells are in the table below, a long way
        # from the Measured ones. Across the page, because a sentence left
        # in the first column is cut off where the printed page ends.
        _write_banner(sheet, 3, 1,
                      wording.sheet_write_in_note(self._actual_column_head()),
                      page_width)

        r = 5
        if ingredients:
            amount_header = (f"{wording.AMOUNT_COLUMN} ({unit})" if unit
                             else wording.AMOUNT_COLUMN)
            # Actual sits directly beside Amount, because that is the
            # comparison the balance makes: read the printed number, write
            # what the pan said. The share is arithmetic about the printed
            # number and follows behind them both.
            headers = [wording.TICK_COLUMN, wording.KIND_INGREDIENT,
                       amount_header, self._actual_column_head()]
            if shares:
                headers.append(wording.PERCENT_COLUMN)
            for c, name in enumerate(headers, start=1):
                _write_cell(sheet, r, c, name, bold=True)
            r += 1
            for var in ingredients:
                    # The box is drawn, not left as an empty bordered cell: the
                # column had a header and nothing under it to put a mark in.
                # It is a write-in cell like any other — a sheet filled in
                # on a screen has to be tickable there too — so it wears
                # the write-in shade rather than the ingredient's colour.
                _write_in_cell(sheet, r, 1, wording.TICK_BOX)
                label = self._sheet_ingredient_label(var['name'])
                if self.has_formula(var):
                    label = wording.worked_out_label(label)
                _write_cell(sheet, r, 2, label, bold=True)
                _write_cell(sheet, r, 3,
                            round(float(recipe.get(var['name'], 0.0)), 2),
                            number_format=_TWO_DP)
                _write_in_cell(sheet, r, 4)
                if shares:
                    _write_cell(sheet, r, 5,
                                self._percent_of(recipe, var, basis),
                                number_format=_ONE_DP)
                r += 1
                # Where it was bought, under the name and in grey: a column
                # for it would push a page that already gained Actual into
                # landscape, and the vendor is read once, at the shelf.
                bought = self._vendor_line(var)
                if bought:
                    _write_cell(sheet, r, 2, bought).font = _QUIET_FONT
                    r += 1
            _write_cell(sheet, r, 2, wording.TOTAL_LABEL, bold=True)
            cell = _write_cell(sheet, r, 3, self._total_cell(recipe), bold=True)
            if shares:
                cell.number_format = _TWO_DP
                _write_cell(sheet, r, 5, 100.0, bold=True,
                            number_format=_ONE_DP)
            r += 1
            # Directly under the amounts it is about: the bench reads down
            # the table and stops at the line that says these numbers are
            # outside what the project allows, or that they do not add up to
            # the total the title names.
            for line in (self.scaled_cautions([row], total, sized)
                         + self.total_mismatch_lines([row], total)):
                _write_cell(sheet, r, 2, line)
                r += 1
            if self._formula_rows():
                _write_cell(sheet, r, 2, self.worked_out_note())
                r += 1
            r += 1

        if process:
            _write_cell(sheet, r, 2, wording.SETTINGS_SHEET_HEADING, bold=True)
            _write_cell(sheet, r, 4, wording.ACTUAL_COLUMN, bold=True)
            r += 1
            for var in process:
                _write_cell(sheet, r, 2, self._amount_column(var['name']))
                _write_cell(sheet, r, 3,
                            round(float(recipe.get(var['name'], 0.0)), 2))
                # A setting is dialled in, and the dial lands where it
                # lands: 188 °C for the 188.49 the sheet asked for is the
                # same correction as a gram weighed heavy.
                _write_in_cell(sheet, r, 4)
                r += 1
            r += 1

        _write_cell(sheet, r, 2, wording.MEASUREMENTS_SHEET_HEADING, bold=True)
        r += 1
        # What mark the app will read, said on the page that asks for it:
        # the summary sheet carried this and the pages the bench actually
        # writes on carried nothing.
        _write_banner(sheet, r, 2, wording.SHEET_WRITE_IN_NOTE, page_width)
        r += 1
        _write_cell(sheet, r, 2, wording.MEASUREMENT_COLUMN, bold=True)
        # Goal for the words, Target for the number: this column holds
        # "higher is better" as often as it holds a 6, and "Target: higher
        # is better" was printed on every sheet.
        _write_cell(sheet, r, 3, wording.GOAL_LABEL, bold=True)
        _write_cell(sheet, r, 4, wording.MEASURED_COLUMN, bold=True)
        r += 1
        for obj in self.measurements_by_importance():
            _write_cell(sheet, r, 2, label_with_unit(obj['name'], obj.get('unit')))
            _write_cell(sheet, r, 3, goal_line(obj))
            _write_in_cell(sheet, r, 4)
            r += 1
        r += 1
        _write_cell(sheet, r, 2, wording.NOT_SCORED_CHECKBOX_SHEET)
        # The box is in the cell a pen can reach, not in the locked label
        # beside it. This is also what the upload reads when the summary
        # sheet was left empty.
        _write_in_cell(sheet, r, 4, wording.TICK_BOX)
        r += 1
        _write_cell(sheet, r, 2, wording.NOTE)
        note = str(row.get('note') or "").strip()
        # Both note cells are open: the app reads the printed one back as
        # what it said itself, and a technician who corrects it there is
        # not writing into a locked sheet to no effect.
        _write_in_cell(sheet, r, 3, note or None, wrap=True)
        _write_in_cell(sheet, r, 4)
        r += 2
        # One lot per ingredient for the whole round, so it is recorded
        # once, on the round's own sheet. The page says where rather than
        # leaving the bench to discover that this one has no such column.
        if ingredients:
            _write_cell(sheet, r, 2, wording.lots_are_on_the_round_sheet(
                self.pending_batch_no))
            r += 1
        # The sheet leaves the app and comes back days later: without these
        # two blanks nothing on the page says whose work it was.
        _write_cell(sheet, r, 2, wording.MADE_BY_FOOTER)
        # Five columns whatever the amounts table holds: the measurements
        # below it are Measurement, Target and Measured, beside the tick,
        # and the share rides at the end of the amounts table alone.
        _set_widths(sheet, [6, 34, 14, 14] + ([7] if shares else []))
        _fit_to_page(sheet, r, page_width)
        _protect(sheet)

    def results_from_workbook(self, source, batch_no=None):
        """A filled-in workbook read back as one row per formulation, in the
        shape parse_batch_results reads.

        The summary sheet is a column per formulation — that is what a bench
        can write on — and the parser wants a row per formulation, so it is
        transposed here. The sheet is found by the batch's own name: last
        week's workbook, downloaded twice, is the mistake this catches. A
        workbook whose summary sheet was left empty is read off the
        formulation sheets instead, because a bench that prints one sheet
        per bowl writes on the sheet in its hand.
        """
        wanted = wording.batch_sheet_name(
            self.pending_batch_no if batch_no is None else batch_no)
        try:
            book = pd.ExcelFile(source)
        except Exception:
            raise ValueError(wording.WORKBOOK_UNREADABLE)
        with book:
            if wanted not in book.sheet_names:
                raise ValueError(wording.workbook_sheet_missing(
                    wanted, number_list(book.sheet_names)))
            summary = book.parse(wanted, header=None)
            rows, numbers = self._transpose_batch_sheet(summary, wanted)
            if not numbers:
                raise ValueError(wording.workbook_no_formulations(wanted))
            if not rows:
                rows = self._read_formulation_sheets(book, numbers)
            lots = self._lots_from_summary(summary)
            actual = self._actual_from_sheets(book, numbers)
        if not rows:
            raise ValueError(wording.workbook_nothing_filled_in(wanted))
        columns_out = (["Formulation"] + [o['name'] for o in self.objectives]
                       + [wording.NOT_SCORED, wording.NOTE])
        # The rows, and beside them the two things the sheet says that are
        # not results. See UploadedWorkbook: they travel in the open, named
        # in the signature, rather than smuggled on the frame.
        return UploadedWorkbook(pd.DataFrame(rows, columns=columns_out),
                                actual, lots)

    def _lots_from_summary(self, frame):
        """{ingredient: lot} off the summary sheet's Lot column.

        One lot per ingredient for the whole round — a round is weighed out
        of the sacks that are open that morning — so the column is one cell
        wide and sits beside the amounts. A workbook written before the
        column existed has no such header and answers with nothing.
        """
        grid = frame.values.tolist()
        header = next((i for i, row in enumerate(grid)
                       if any(self._formulation_column_number(cell) is not None
                              for cell in row[1:])), None)
        if header is None:
            return {}
        column = next((c for c, cell in enumerate(grid[header])
                       if str(cell).strip() == wording.LOT_COLUMN), None)
        if column is None:
            return {}
        wanted = {}
        for var in self._ingredients():
            labels = [self._amount_column(var['name']), var['name']]
            if self.has_formula(var):
                # A worked-out row's printed name carries the mark the
                # summary sheet wrote it with — the same name it would
                # otherwise never match.
                labels.append(wording.worked_out_label(
                    self._amount_column(var['name'])))
            for label in labels:
                wanted[str(label).strip().lower()] = var['name']
        lots = {}
        for row in grid[header + 1:]:
            label = (str(row[0]).strip().lower()
                     if row and row[0] is not None else "")
            name = wanted.get(label)
            value = self._cell(row, column)
            if name is not None and value is not None:
                lots[name] = str(value).strip()
        return lots

    def _actual_from_sheets(self, book, numbers):
        """{formulation number: {name: what was weighed}} off the Actual
        column of each formulation's own page.

        A blank Actual cell means "as printed", so only the cells somebody
        wrote in come back; a page with none of them filled in is not in the
        answer at all. Something that is not a weight is refused rather than
        dropped: the whole point of the column is that what was made is not
        what was printed.
        """
        labels = {}
        for var in self.variables:
            name_labels = [self._sheet_ingredient_label(var['name']),
                          self._amount_column(var['name']), var['name']]
            if self.has_formula(var):
                # The formulation page prints this row's name with the
                # worked-out mark on it; without this alternative that
                # printed name never matches and the Actual it carries is
                # dropped with no error.
                name_labels.append(wording.worked_out_label(
                    self._sheet_ingredient_label(var['name'])))
            for label in name_labels:
                labels[str(label).strip().lower()] = var['name']
        headers = {self._actual_column_head().strip().lower(),
                   wording.ACTUAL_COLUMN.strip().lower()}
        out = {}
        for number in numbers:
            name = wording.formulation_sheet_name(number)
            if name not in book.sheet_names:
                continue
            grid = book.parse(name, header=None).values.tolist()
            column = next((c for row in grid for c, cell in enumerate(row)
                           if str(cell).strip().lower() in headers), None)
            if column is None:
                continue        # a workbook written before the column
            weighed = {}
            for row in grid:
                label = (str(row[1]).strip().lower()
                         if len(row) > 1 and row[1] is not None else "")
                variable = labels.get(label)
                value = self._cell(row, column)
                if variable is None or value is None:
                    continue
                try:
                    amount = float(value)
                except (TypeError, ValueError):
                    raise ValueError(wording.workbook_actual_not_a_number(
                        number, variable))
                if amount < 0:
                    raise ValueError(wording.workbook_actual_below_zero(
                        number, variable))
                weighed[variable] = amount
            if weighed:
                out[int(number)] = weighed
        return out

    @staticmethod
    def amounts_as_weighed(recipe, weighed=None):
        """What to record as one formulation's amounts: what the bench wrote
        in the Actual cells, over the amounts it was given, or the amounts
        unchanged where nothing was written.

        Only the cells somebody filled in move — a blank Actual cell means
        the printed amount was weighed out — and the sheets print the stored
        amounts, so what comes back sits on the basis the model reads.
        """
        out = dict(recipe)
        out.update(weighed or {})
        return out

    def store_lots(self, batch_no, lots):
        """Keep the lot numbers an uploaded workbook came back with against
        the round they were weighed for. They are never read by the model:
        they are what a formulation is traced back through six months
        later."""
        if not lots or batch_no is None:
            return {}
        if not isinstance(getattr(self, 'lots', None), dict):
            self.lots = {}
        kept = dict(self.lots.get(int(batch_no)) or {})
        kept.update({str(k): str(v) for k, v in lots.items()})
        self.lots[int(batch_no)] = kept
        self.save()
        return kept

    def _result_item(self, number, measured, ticked, note, prefill=None):
        """One formulation's row of an uploaded sheet, in the shape
        parse_batch_results reads, or None when nothing was written on it —
        half a batch measured today and the rest tomorrow is how a bench
        works, so an untouched formulation is left where it is.

        `prefill` is the note the app itself printed on that sheet. The
        write-in cell wins over it: the app's note says what the formulation
        was FOR, and a technician who writes beside it is saying what
        happened.

        A note and nothing else is refused, in workbook_note_without_numbers.
        Somebody wrote on that column — it is not an untouched formulation —
        and importing it as a blank would file the one sentence anybody wrote
        about the bowl under a formulation with no result to hold it.
        """
        written = None if note is None else str(note).strip()
        if written and written == str(prefill or "").strip():
            written = ""        # the app's own note, printed and handed back
        if all(value is None for value in measured.values()) and not _is_ticked(ticked):
            if written:
                raise ValueError(
                    wording.workbook_note_without_numbers(number))
            return None
        item = {"Formulation": number}
        item.update(measured)
        item[wording.NOT_SCORED] = "" if ticked is None else str(ticked)
        item[wording.NOTE] = written or str(prefill or "").strip()
        return item

    @staticmethod
    def _cell(values, index):
        """One cell of an uploaded sheet, with a blank of any spelling read
        as nothing at all."""
        if index is None or index < 0 or index >= len(values):
            return None
        value = values[index]
        if value is None or (isinstance(value, float) and np.isnan(value)):
            return None
        if isinstance(value, str) and not value.strip():
            return None
        return value

    def _transpose_batch_sheet(self, frame, sheet_name=""):
        """The summary sheet's columns turned back into rows, as (rows,
        formulation numbers).

        The sheet is read with no header row of its own: it opens with a
        title line, and a workbook written before that line existed opens
        with the header. Whichever it is, the header is the row carrying the
        `Formulation N` cells, and everything below it is the labels in the
        first column.

        The measurement rows are found by their labels, last match first.
        The tick and the note are found by their labels too, but only among
        the rows BELOW the last measurement, falling back to the positions
        the app writes them at: an ingredient named `Not scored` would
        otherwise hijack the tick row and take its column's results with it
        (that name is reserved now; a project saved before it was is not),
        while counting positions alone lost them to any row inserted in the
        block.
        """
        grid = frame.values.tolist()
        header = next((i for i, row in enumerate(grid)
                       if any(self._formulation_column_number(cell) is not None
                              for cell in row[1:])), None)
        if header is None:
            return [], []
        labels = [str(row[0]).strip() if row and row[0] is not None else ""
                  for row in grid[header + 1:]]
        lowered = [label.lower() for label in labels]

        by_measurement, last_row, guessed = {}, -1, {}
        # The block's heading. `Measurements` is what it says now, on both
        # kinds of sheet; `Measured` is what the summary sheet said before
        # the two were brought into line, and a workbook downloaded then is
        # still on somebody's bench.
        headings = {wording.MEASUREMENTS_SHEET_HEADING.lower(),
                    wording.MEASURED_COLUMN.lower()}
        heading = next((i for i in range(len(lowered) - 1, -1, -1)
                        if lowered[i] in headings), None)
        # Where the write-in block's FIRST measurement sits, which is not
        # the row under the heading: between them sit one line of
        # instruction ("Write what you measured…") and the row of
        # formulation names that heads the block's columns. A fallback that
        # counted from the heading landed rows too high — rename
        # Juiciness's label and its 8.0 came back as Firmness's 5.5,
        # silently. A sheet written before either row existed has neither,
        # so each is skipped only when it is there.
        skippable = {wording.SHEET_WRITE_IN_NOTE.lower(),
                     wording.MEASUREMENT_COLUMN.lower()}
        first_measurement = None
        if heading is not None:
            first_measurement = heading + 1
            while (first_measurement < len(lowered)
                    and lowered[first_measurement] in skippable):
                first_measurement += 1
        for position, obj in enumerate(self.measurements_by_importance()):
            names = {self._measurement_sheet_label(obj).lower(),
                     label_with_unit(obj['name'], obj.get('unit')).lower(),
                     str(obj['name']).strip().lower()}
            index = next((i for i in range(len(lowered) - 1, -1, -1)
                          if lowered[i] in names), None)
            if index is None and first_measurement is not None:
                # The label was retyped on the sheet; inside the write-in
                # block the measurements are still in importance order.
                # A guess, and recorded as one: if nothing was written where
                # it points while the block holds numbers elsewhere, it is
                # pointing at somebody else's row and the sheet is refused.
                index = first_measurement + position
                guessed[obj['name']] = index
            if index is not None and index < len(labels):
                by_measurement[obj['name']] = index
                last_row = max(last_row, index)

        columns = []
        for c, cell in enumerate(grid[header]):
            if c == 0:
                continue
            number = self._formulation_column_number(cell)
            if number is not None:
                columns.append((number, c))
        numbers = [number for number, _ in columns]
        if not by_measurement:
            # Not one measurement row was found, so there is no block to read
            # the tick and the note from the foot of. Counting positions from
            # nowhere would have read the first ingredient's amount as a
            # ticked box and lost the whole batch to it.
            return [], numbers

        def under_the_measurements(label_texts, fallback):
            """The row carrying one of these labels somewhere BELOW the last
            measurement, or the position the app writes it at.

            By label, so a row inserted into the block — a second note line,
            a blank line a technician left — does not shift the tick onto
            the note. Below the measurements, so an ingredient row of the
            same name on a sheet written before that name was reserved still
            cannot hijack it. Several spellings, because the tick row is
            'Not scored ☐' on a sheet this version wrote and a bare 'Not
            scored' on one written before the box was drawn into the label.
            """
            wanted = {text.lower() for text in label_texts}
            for i in range(last_row + 1, len(lowered)):
                if lowered[i] in wanted:
                    return i
            return fallback

        not_scored_row = under_the_measurements(
            (wording.NOT_SCORED_CHECKBOX_SHEET, wording.NOT_SCORED),
            last_row + 1)
        note_row = under_the_measurements((wording.NOTE,), last_row + 2)

        # What the app printed in each Note cell, so a sheet handed straight
        # back is not read as a technician's own note.
        prefills = {int(r['formulation']): str(r.get('note') or "").strip()
                    for r in self._batch_rows(self.pending_batch or [])}
        rows = []
        for number, c in columns:
            values = [row[c] if c < len(row) else None
                      for row in grid[header + 1:]]
            measured = {name: self._cell(values, index)
                        for name, index in by_measurement.items()}
            if any(value is not None for value in measured.values()):
                for name, index in guessed.items():
                    if self._cell(values, index) is None:
                        raise ValueError(
                            wording.workbook_measurement_missing(
                                name, sheet_name))
            item = self._result_item(number, measured,
                                     self._cell(values, not_scored_row),
                                     self._cell(values, note_row),
                                     prefill=prefills.get(number))
            if item is not None:
                rows.append(item)
        return rows, numbers

    def _read_formulation_sheets(self, book, numbers):
        """The per-formulation sheets read for their Measured cells, when the
        summary sheet came back empty. Each sheet is `Ingredient | Amount`
        under a tick column, so a label sits in the second column and what
        was written beside it in the fourth."""
        measurement_names = {}
        for obj in self.objectives:
            for label in (label_with_unit(obj['name'], obj.get('unit')),
                          str(obj['name'])):
                measurement_names[label.strip().lower()] = obj['name']
        rows = []
        for number in numbers:
            name = wording.formulation_sheet_name(number)
            if name not in book.sheet_names:
                continue
            grid = book.parse(name, header=None).values.tolist()
            measured = {obj['name']: None for obj in self.objectives}
            ticked, note, prefill = None, None, None
            for row in grid:
                label = str(row[1]).strip().lower() if len(row) > 1 and row[1] is not None else ""
                written = self._cell(row, 3)
                if label in measurement_names:
                    measured[measurement_names[label]] = written
                elif label == wording.NOT_SCORED_CHECKBOX_SHEET.lower():
                    ticked = written
                elif label == wording.NOTE.lower():
                    # Two cells on this row: the app's own note, printed
                    # beside the label, and the box the bench writes in.
                    # What the bench wrote wins — the printed one says what
                    # the formulation was FOR, and somebody who writes
                    # beside it is saying what happened.
                    note, prefill = written, self._cell(row, 2)
            item = self._result_item(number, measured, ticked, note,
                                     prefill=prefill)
            if item is not None:
                rows.append(item)
        return rows

    @staticmethod
    def _formulation_column_number(column):
        """The number out of a `Formulation 4` column header, or None for
        any other column (the `%` columns pandas names %, %.1, %.2 included)."""
        match = re.fullmatch(
            re.escape(wording.FORMULATION_CAP) + r"\s+(\d+)(\.\d+)?",
            str(column).strip())
        return int(match.group(1)) if match else None

    def history_export_frame(self):
        """Every formulation the project holds as one table — the download on
        tab 3, whether it leaves as a workbook or as a comma-separated file.
        See history_csv for what each column is. The Total closes the
        amounts, as it does on every other table: the cold read had to sum
        eight columns by hand to find out that three rows were 100.06, 97
        and 102 g. `Not scored` is the tick the screen shows in the Note
        column, as its own column."""
        objs = self.measurements_by_importance()
        total_col = self.total_column()
        rows = []
        for i in range(len(self.X_history)):
            ts = self.timestamps_history[i] if i < len(self.timestamps_history) else None
            batch = self.batch_history[i] if i < len(self.batch_history) else None
            results = self.results_history[i] if i < len(self.results_history) else {}
            row = {
                "Formulation": int(self.formulation_ids[i]),
                wording.ROUND_CAP: "" if batch is None else int(batch),
                wording.DATE_RECORDED_COLUMN: local_date(ts),
                # Two decimals, as the screen shows it: a file that says
                # 2.625 where the table says 2.62 reads as a third number.
                wording.OVERALL_SCORE_COLUMN: round(float(self.Y_history[i]), 2),
            }
            recipe = self._recorded_recipe(i)
            row.update(self._amount_columns(recipe))
            if total_col is not None:
                row[total_col] = self._total_cell(recipe)
            for obj in objs:
                row[self._measurement_column(obj)] = results.get(obj['name'])
            row[wording.NOT_SCORED] = ""
            row["Note"] = self.notes_history[i] if i < len(self.notes_history) else ""
            rows.append(row)
        for left_out in self.skipped:
            batch = left_out.get(ROUND_FIELD)
            row = {
                "Formulation": int(left_out['formulation']),
                wording.ROUND_CAP: "" if batch is None else int(batch),
                wording.DATE_RECORDED_COLUMN: "",
                wording.OVERALL_SCORE_COLUMN: "",
            }
            recipe = left_out.get('recipe', {})
            row.update(self._amount_columns(recipe))
            if total_col is not None:
                row[total_col] = self._total_cell(recipe)
            for obj in objs:
                row[self._measurement_column(obj)] = None
            row[wording.NOT_SCORED] = wording.TICKED_BOX
            row["Note"] = left_out.get('note') or wording.NOT_SCORED
            rows.append(row)
        columns = (["Formulation", wording.ROUND_CAP,
                    wording.DATE_RECORDED_COLUMN,
                    wording.OVERALL_SCORE_COLUMN]
                   + [self._amount_column(v['name']) for v in self.variables]
                   + ([total_col] if total_col is not None else [])
                   + [self._measurement_column(o) for o in objs]
                   + [wording.NOT_SCORED, "Note"])
        return pd.DataFrame(rows, columns=columns)

    def all_formulations_workbook(self):
        """Everything the project holds, in one file to send on: every
        formulation with its amounts and results, and the set-up they were
        made under. A table of numbers with nothing saying what the targets
        were is a table nobody can read six months later."""
        book = Workbook()
        sheet = book.active
        sheet.title = wording.ALL_FORMULATIONS_SHEET
        two_dp = {self._amount_column(v['name'])
                  for v in self._ingredients()}
        if self.total_column() is not None:
            two_dp.add(self.total_column())
        _write_frame(sheet, self.history_export_frame(), two_decimals=two_dp,
                     title=wording.RECORDED_AMOUNTS)
        # One row per ingredient per round, on a sheet of its own. A lot
        # belongs to a round, not to a formulation, so a column per
        # ingredient on the table above would say the same thing six times
        # and double its width. The sheet arrives with the first lot
        # anybody wrote down.
        lots = self._lots_frame()
        if lots is not None:
            _write_frame(book.create_sheet(wording.LOTS_SHEET), lots)
        self._write_setup_sheet(book.create_sheet(wording.SET_UP_SHEET))
        buffer = io.BytesIO()
        book.save(buffer)
        return buffer.getvalue()

    def _lots_frame(self):
        """Round, ingredient and lot, in round order and then in set-up
        order: the sheet somebody reads when one round came out wrong and
        the question is which sack it was weighed from. None when the
        project holds no lot at all."""
        stored = getattr(self, 'lots', None) or {}
        order = [v['name'] for v in self.variables]
        rows = []
        for batch_no in sorted(stored, key=lambda k: int(k)):
            written = stored.get(batch_no) or {}
            for name in sorted(written,
                               key=lambda n: (order.index(n) if n in order
                                              else len(order), str(n))):
                rows.append({wording.ROUND_CAP: int(batch_no),
                             wording.KIND_INGREDIENT: str(name),
                             wording.LOT_COLUMN: str(written[name])})
        if not rows:
            return None
        return pd.DataFrame(rows, columns=[wording.ROUND_CAP,
                                           wording.KIND_INGREDIENT,
                                           wording.LOT_COLUMN])

    def _write_setup_sheet(self, sheet):
        """The project as it stands: what can be changed and between which
        amounts, what is measured and what a good number is, the limits, the
        total, and where the targets came from."""
        r = 1
        _write_cell(sheet, r, 1, wording.VARIABLES_HEADER, bold=True)
        r += 1
        # The Status column arrives with the first fixed row and not before:
        # a column that says nothing on every row of a project where nothing
        # is fixed is a column of noise. It is what the screen's own Status
        # column used to say, now that the screen reads a fixed row off its
        # one amount instead.
        # A worked-out row counts too now: Status is where the sheet says
        # what a row the search does not move is doing, and it says it of
        # both kinds in the one column.
        any_fixed = any(self.has_formula(v) or self.is_fixed(v)
                        for v in self.variables)
        # Vendor and SKU are printed so the bench knows what to reach for;
        # like Status, they arrive with the first row that has one.
        any_supplier = any(v.get('vendor') or v.get('sku')
                           for v in self.variables)
        # And a Formula column with the first row that is worked out. The
        # sheet is what the bench reads: a row whose amount is arithmetic
        # over the others has to say so where its two amounts would be.
        any_formula = any(self.has_formula(v) for v in self.variables)
        headers = [wording.NAME_LABEL, wording.TYPE_LABEL,
                   wording.LOWEST_LABEL, wording.HIGHEST_LABEL,
                   wording.UNIT_LABEL, wording.BASELINE_LABEL]
        if any_formula:
            headers.append(wording.FORMULA_LABEL)
        if any_supplier:
            headers += [wording.VENDOR_LABEL, wording.SKU_LABEL]
        if any_fixed:
            headers.append(wording.STATUS_LABEL)
        for c, name in enumerate(headers, start=1):
            _write_cell(sheet, r, c, name, bold=True)
        r += 1
        for var in self.variables:
            ingredient = var.get('category', 'ingredient') == 'ingredient'
            worked_out = self.has_formula(var)
            _write_cell(sheet, r, 1, var['name'])
            _write_cell(sheet, r, 2, wording.KIND_INGREDIENT if ingredient
                        else wording.KIND_SETTING)
            # The same two cells the screen shows: the word for a row that
            # is worked out, its own two numbers for every other.
            low, high = self._range_cell(var)
            # A worked-out row's own two cells go BLANK here, not to the
            # word: the Status column below says what the row is doing, in
            # the same column that says it of a fixed row, and a reader
            # scanning Status for the pinned rows found one of the two
            # kinds and an empty cell for the other.
            _write_cell(sheet, r, 3, None if worked_out
                        else float(var['bounds'][0]))
            _write_cell(sheet, r, 4, None if worked_out
                        else float(var['bounds'][1]))
            _write_cell(sheet, r, 5, self.unit_of(var['name']) or None)
            baseline = var.get('_absent_value')
            _write_cell(sheet, r, 6, None if baseline is None else float(baseline))
            c = 7
            if any_formula:
                _write_cell(sheet, r, c,
                            wording.setup_sheet_formula_text(
                                self._formula_text(var),
                                rest=bool(var.get('balance')))
                            if worked_out else None)
                c += 1
            if any_supplier:
                _write_cell(sheet, r, c, var.get('vendor') or None)
                _write_cell(sheet, r, c + 1, var.get('sku') or None)
                c += 2
            if any_fixed:
                _write_cell(sheet, r, c,
                            wording.WORKED_OUT if worked_out
                            else (wording.fixed_status(self.fixed_at_text(var))
                                  if self.is_fixed(var) else None))
            r += 1
        r += 1

        _write_cell(sheet, r, 1, wording.MEASUREMENTS_HEADER, bold=True)
        r += 1
        # No Importance column: 0.5.0 makes Share of score the number the
        # reader types and the importance behind it derived, so a sheet that
        # printed both printed one fact twice — in two scales.
        # One Goal cell, not a Goal beside a Target: 'Target 6 N' is the
        # one rendering every other surface uses, and the sheet was the odd
        # one out.
        for c, name in enumerate((wording.MEASUREMENT_COLUMN,
                                  wording.GOAL_LABEL,
                                  wording.RANGE_COLUMN,
                                  wording.SHARE_COLUMN), start=1):
            _write_cell(sheet, r, c, name, bold=True)
        r += 1
        for obj in self.measurements_by_importance():
            _write_cell(sheet, r, 1, label_with_unit(obj['name'], obj.get('unit')))
            _write_cell(sheet, r, 2, goal_text(obj))
            _write_cell(sheet, r, 3, measurement_range_text(obj))
            _write_cell(sheet, r, 4, self.share_text(obj['name']))
            r += 1
        r += 1

        # Two blocks, as the screen has two: the amount limits (and the
        # default batch size, which is written as one) under Limits, and
        # the finished-product limits under the owner's own name for them.
        # One block held both, so on paper "Fat: at most 15" and "Water +
        # Oil: at most 40 g" read as one kind of rule; they are per 100 g
        # of what you make and a weight in the bowl.
        _write_cell(sheet, r, 1, wording.LIMITS_SHEET_HEADING, bold=True)
        r += 1
        amounts = [self.limit_text(qc)
                   for qc in getattr(self, "quantity_constraints", [])]
        for line in amounts or [wording.SHEET_NONE]:
            _write_cell(sheet, r, 1, line)
            r += 1
        if self.formulation_total is None:
            _write_cell(sheet, r, 1, wording.FORMULATION_TOTAL_NAME)
            _write_cell(sheet, r, 2, wording.SHEET_NONE)
            r += 1
        r += 1

        _write_cell(sheet, r, 1, wording.PROPERTY_LIMITS_SHEET_HEADING,
                    bold=True)
        r += 1
        _write_cell(sheet, r, 1, self.per_amount_text())
        r += 1
        for line in ([self.property_limit_text(c) for c in self.constraints]
                     or [wording.SHEET_NONE]):
            _write_cell(sheet, r, 1, line)
            r += 1
        r += 1

        _write_cell(sheet, r, 1, wording.TARGETS_SOURCE_LABEL, bold=True)
        r += 1
        _write_cell(sheet, r, 1,
                    getattr(self, "targets_source", "") or wording.SHEET_NONE,
                    wrap=True)
        _set_widths(sheet, [34, 18, 12, 12, 10, 12, 14, 14, 14])
        _fit_to_page(sheet, r, 9)

    def limit_label(self, qc):
        """How one ingredient limit is named — 'Total of each formulation',
        'All ingredients' or 'Water + Oil'. The list under Limits, the line
        that reports a limit removed and the workbook's Set-up sheet all read
        from here, so they name it alike.

        The total's own limit is named for the box that wrote it, not for the
        ingredients it happens to cover: it is over all of them by
        definition, and 'All ingredients' would read as something the user
        typed into the picker below."""
        if qc.get('source') == 'formulation_total':
            return wording.FORMULATION_TOTAL_NAME
        names = [v['name'] for v in self._ingredients()]
        if names and set(qc['ingredients']) == set(names):
            return wording.ALL_INGREDIENTS_LABEL
        return " + ".join(qc['ingredients'])

    @staticmethod
    def _bound_words(min_v, max_v, unit):
        """'at least 10 g and at most 40 g' — the two bound words a range
        limit reads as, in the unit it is written in, joined the way every
        limit line joins them."""
        bounds = ([join_unit(wording.at_least(min_v), unit)]
                  if min_v is not None else [])
        bounds += ([join_unit(wording.at_most(max_v), unit)]
                   if max_v is not None else [])
        return wording.AND_JOIN.join(bounds)

    @staticmethod
    def _bound_amounts(min_v, max_v, unit):
        """'10 g and 40 g' — the same two numbers with no bound word, for
        the grams a percent limit's read-back names beside the percent."""
        amounts = ([join_unit(f"{min_v:g}", unit)] if min_v is not None else [])
        amounts += ([join_unit(f"{max_v:g}", unit)] if max_v is not None else [])
        return wording.AND_JOIN.join(amounts)

    def limit_text(self, qc):
        """One limit as one line: 'Water + Oil: at least 10 g and at most
        40 g'; an Exactly limit as the number typed, not the band it is
        enforced as ('Water + Oil: exactly 50 g'); a percent limit in both
        the percent it is written as and the grams it means today ('Water +
        Oil: at most 30 % of batch size (30 g today)'); or the total
        written as the one number the user typed."""
        if qc.get('source') == 'formulation_total':
            return wording.formulation_total_row(
                self.batch_total_text(self.formulation_total))
        limited = {self.unit_of(n) for n in qc['ingredients']}
        unit = limited.pop() if len(limited) == 1 else ""
        who = self.limit_label(qc)
        percent = qc.get('percent')
        if percent:
            if percent.get('exactly') is not None:
                percent_text = join_unit(wording.exactly(percent['exactly']), '%')
                grams_text = join_unit(f"{qc['exactly']:g}", unit)
            else:
                percent_text = self._bound_words(percent['min'], percent['max'], '%')
                grams_text = self._bound_amounts(qc['min'], qc['max'], unit)
            return wording.limit_percent_row(
                who, percent_text, grams_text,
                self.batch_total_text(self.formulation_total))
        if qc.get('exactly') is not None:
            return wording.limit_exactly_row(
                who, join_unit(f"{qc['exactly']:g}", unit))
        return f"{who}: {self._bound_words(qc['min'], qc['max'], unit)}"

    def property_limit_text(self, constraint):
        """A property limit, per 100 of the amount unit, as one line."""
        bounds = ([wording.at_least(constraint['min'])]
                  if constraint.get('min') is not None else [])
        bounds += ([wording.at_most(constraint['max'])]
                   if constraint.get('max') is not None else [])
        return f"{constraint['metric']}: {wording.AND_JOIN.join(bounds)}"

    def bounds_caution(self, name, value):
        """The line for an amount outside what the project allows, or '' when
        it fits."""
        var = next((v for v in self.variables if v['name'] == name), None)
        if var is None or value is None:
            return ""
        low, high = (float(b) for b in var['bounds'])
        if low <= float(value) <= high:
            return ""
        return outside_message(name, value, low, high, self.unit_of(name),
                               wording.ALLOWED_AMOUNTS)

    def scaled_caution(self, recipes, total, sized=False):
        """The one line for the ingredients whose amounts fall outside what
        the project allows once these formulations are made to `total`, or ""
        when they all fit. Up to three it names them; above that it counts
        them.

        The stored amounts were chosen inside the project's own Lowest and
        Highest; a batch size they were never chosen for moves them past it,
        and the sheets are made from those numbers — so the bench weighs out
        an amount the project says it does not allow. The round screen's box,
        tab 3's amounts table and the workbook's own sheets all say so in
        these words, from here, so the three can never drift apart.

        `recipes` may be plain amounts or whole rows. `sized` is the caller
        saying these rows are an open round scale_round has already made to
        `total` — see _rewritten.
        """
        scaled = self._rewritten(recipes, total, sized)
        if not scaled:
            return ""
        ingredients = [var['name'] for var in self._ingredients()]
        names = [name for name in ingredients
                 if any(self.bounds_caution(name, recipe.get(name))
                        for recipe in scaled)]
        if not names:
            return ""
        return wording.scaled_amounts_caution(
            self.batch_total_text(total),
            names_text=number_list(names) if len(names) <= 3 else "",
            n_outside=len(names), n_total=len(ingredients))

    def _rewritten(self, recipes, total, sized=False):
        """The amounts a batch size actually put where the model did not
        choose them, ready to be checked against the project's own rules.
        Empty when nothing was moved.

        Two ways an amount gets here. A round SHOWN at a size it was not
        built to is rewritten on the way to the screen, and shown_recipe says
        so by handing back a basis. And since 0.5.0 the Batch size box moves
        the stored amounts themselves — every row of the open round, the
        bench's own rows included, and whether or not the project has a
        default — so once that round has a size of its own every one of its
        rows is checked, however it is displayed.

        `sized` is that second case, and it is the CALLER's to answer: only
        the round screen and the workbook it prints know that these rows are
        the open round and that scale_round has been over them. Working it
        out here meant matching formulation numbers against pending_batch,
        which guessed at what the caller already knew.
        """
        if total is None:
            return []
        out = []
        for row in recipes:
            recipe, basis = self.shown_recipe(row, total)
            if basis is not None or sized:
                out.append(recipe)
        return out

    def scaled_limit_caution(self, recipes, total, sized=False):
        """The line for a limit the batch size broke on its way past it, or ""
        when they all hold.

        An amount still inside its own Lowest and Highest can carry a limit
        over — 'Pea protein isolate + Wheat gluten at most 20 g' became
        20.32 g when the round was printed at 150 g — and a limit is
        documented as a hard rule. One line, naming the first limit that does
        not hold, in the words the Limits list writes it in."""
        for recipe in self._rewritten(recipes, total, sized):
            for qc in getattr(self, 'quantity_constraints', []):
                if qc.get('source') == 'formulation_total':
                    continue     # the total is the thing being asked about
                value = sum(float(recipe.get(n, 0.0))
                            for n in qc['ingredients'])
                if ((qc['min'] is not None and value < qc['min'])
                        or (qc['max'] is not None and value > qc['max'])):
                    return wording.scaled_limit_caution(
                        self.batch_total_text(total), self.limit_text(qc))
            for constraint in self.constraints:
                if not self._property_limit_holds(recipe, constraint):
                    return wording.scaled_limit_caution(
                        self.batch_total_text(total),
                        self.property_limit_text(constraint))
        return ""

    def _property_limit_holds(self, recipe, constraint):
        """One property limit, read the way _check_constraints reads it."""
        metric = constraint['metric']
        if (constraint['min'] is not None
                and self._property_residual(recipe, metric,
                                            constraint['min']) < -1e-9):
            return False
        if (constraint['max'] is not None
                and self._property_residual(recipe, metric,
                                            constraint['max']) > 1e-9):
            return False
        return True

    def scaled_cautions(self, recipes, total, sized=False):
        """Every line a re-sized round owes the bench: the amounts pushed past
        what the project allows, and the limit the size broke. Callers draw
        them in order — the screen as captions, the sheets as rows — so one
        list is the whole answer. `sized` is passed straight through to
        _rewritten, which says what it means."""
        return [line
                for line in (self.scaled_caution(recipes, total, sized),
                             self.scaled_limit_caution(recipes, total, sized))
                if line]

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
            raise ValueError(wording.LIMIT_BOUNDS_ORDER_ERROR)
        # An average over the amounts, so the amounts must be in one unit:
        # 25 g of powder and 40 ml of water share no 100 g to be measured per.
        units = self.ingredient_units()
        if len(units) > 1:
            raise ValueError(wording.property_limits_need_a_mass_unit(
                self.majority_amount_unit() or "g",
                self.unit_fix_sentence()))
        # Same property, whatever its capitalisation: two limits on 'Fat' and
        # 'fat' would both be enforced against the same column.
        self.constraints = [c for c in self.constraints
                            if str(c['metric']).strip().lower()
                            != str(metric).strip().lower()]
        kept = list(self.constraints)
        entry = {
            'metric': metric,
            'min': float(min_val) if min_val is not None else None,
            'max': float(max_val) if max_val is not None else None,
            # What this limit was written to mean. A limit from an older file
            # has no basis at all, and the screen says once that it is now
            # read per 100 g.
            'basis': 'per_100',
        }
        self.constraints.append(entry)
        self._refuse_limit_the_fixed_rows_break(
            metric, self._property_limit_refusal(entry), None,
            lambda: setattr(self, 'constraints', kept))
        self.save()

    def remove_constraint(self, index):
        """Remove a property constraint by index."""
        if 0 <= index < len(self.constraints):
            self.constraints.pop(index)
            self.save()

    def add_quantity_constraint(self, ingredients, min_val=None, max_val=None,
                                source=None, exactly=None, percent=None):
        """Add a constraint on the sum of selected ingredient quantities.
        Replaces any existing constraint on the same set of ingredients that
        was written the same way.

        Args:
            ingredients: List of ingredient names whose quantities to sum.
            min_val: Minimum allowed sum (or None for no lower bound).
            max_val: Maximum allowed sum (or None for no upper bound).
            source: What wrote it. None is a limit the user typed into the
                Limits section; 'formulation_total' is the one the Total of
                each formulation box owns, which is why replacement matches
                on the source as well as on the ingredients: a user who
                limits every ingredient by hand must not silently take the
                total's limit away, and both then hold.
            exactly: A single number instead of a range — the typed amount
                (or, with `percent`, the typed percent). Stored as the
                number itself and enforced as that number's own band, the
                same answer this file already gives the total of each
                formulation, and for the same reason: a continuous search
                cannot be held to a point. Refused together with min_val or
                max_val (one idea, one control) and over a single
                ingredient (its own Lowest and Highest already say that).
            percent: True when min_val, max_val and exactly are percentages
                of the default batch size rather than an amount. The typed
                percentages are kept as `entry['percent']` alongside the
                grams they come to today, so a later change of default can
                rewrite the grams without losing what was actually asked
                for.
        """
        # No refusal for Exactly-and-a-range: the Kind picker on screen is
        # one control for one idea and cannot produce the combination, and
        # a caller that sends both gets the well-defined answer below —
        # Exactly wins and writes its own band.
        if exactly is not None and len(ingredients) == 1:
            raise ValueError(wording.EXACTLY_ONE_INGREDIENT)
        # A limit can only change what the search moves. Every row of this
        # one worked out from a rule means the limit has nothing to act on:
        # it was accepted, listed, and then quietly contradicted by the
        # rules that fix those rows.
        if source is None and ingredients and all(
                self.has_formula(self._var_by_name(name))
                for name in ingredients):
            raise ValueError(wording.limit_on_worked_out_rows(
                number_list(list(ingredients)),
                many=len(ingredients) != 1))
        if min_val is not None and max_val is not None and float(min_val) >= float(max_val):
            raise ValueError(wording.LIMIT_BOUNDS_ORDER_ERROR)
        # A limit is a sum, and a sum across units is a number of nothing:
        # 25 g of powder plus 40 ml of water is neither 65 g nor 65 ml.
        if len({self.unit_of(name) for name in ingredients}) > 1:
            raise ValueError(wording.limit_needs_one_unit(
                self.unit_fix_sentence(list(ingredients))))
        ingredient_set = set(ingredients)
        kept = list(self.quantity_constraints)
        self.quantity_constraints = [
            qc for qc in self.quantity_constraints
            if set(qc['ingredients']) != ingredient_set
            or qc.get('source') != source
        ]
        percent_block = None
        if percent:
            # Offered on screen only while there is a default to be a
            # percent OF. A caller reaching this without one is refused in
            # the LIMIT form's own words: the Rule cell's sentence names
            # writing the amounts instead, which is an act on another
            # screen with nothing to do with what this reader was doing.
            total = getattr(self, 'formulation_total', None)
            if total is None:
                raise ValueError(wording.PERCENT_NEEDS_BATCH_SIZE)
            percent_block = {
                'min': float(min_val) if min_val is not None else None,
                'max': float(max_val) if max_val is not None else None,
                'exactly': float(exactly) if exactly is not None else None,
            }
            to_grams = lambda p: None if p is None else p / 100.0 * total
            min_val = to_grams(percent_block['min'])
            max_val = to_grams(percent_block['max'])
            exactly = to_grams(percent_block['exactly'])
        entry = {'ingredients': list(ingredients)}
        if exactly is not None:
            entry['exactly'] = float(exactly)
            entry['min'], entry['max'] = self._formulation_total_bounds(exactly)
        else:
            entry['min'] = float(min_val) if min_val is not None else None
            entry['max'] = float(max_val) if max_val is not None else None
        if percent_block is not None:
            entry['percent'] = percent_block
        if source is not None:
            entry['source'] = source
        self.quantity_constraints.append(entry)
        if source is None:
            # The batch size's own limit is exempt: set_formulation_total
            # asks the reach question first, in the two numbers the box was
            # typed into, and its answer is the better one.
            self._refuse_limit_the_fixed_rows_break(
                " + ".join(ingredients), self._quantity_limit_refusal(entry),
                ingredients,
                lambda: setattr(self, 'quantity_constraints', kept))
        self.save()

    def add_total_mass_constraint(self, min_val=None, max_val=None,
                                  source=None):
        """The limit over every ingredient — what the Total of each
        formulation box writes (tagged 'formulation_total'), and what a
        project saved before 0.4.0 may already hold untagged from the days
        when the Limits picker's empty state meant all of them.

        Refused in the same words as any other such limit while the
        ingredients are not all in one unit: the refusal names the ingredient
        to re-enter and the unit to enter it in, which is the same answer
        whether the group is two ingredients or all of them."""
        all_ingredients = [
            v['name'] for v in self.variables
            if v.get('category', 'ingredient') == 'ingredient'
        ]
        self.add_quantity_constraint(all_ingredients, min_val, max_val,
                                     source=source)

    def remove_quantity_constraint(self, index):
        """Remove a quantity constraint by index. Taking out the one the
        Total of each formulation box owns takes the total with it: the
        number on tab 1 says the suggestions add up to it, and a number
        nothing enforces would be a lie.

        And every limit written as a % of the default batch size goes with
        it, one line each — there is nothing left for one to be a percent
        OF. The third door onto that, beside clear_formulation_total and
        _sync_formulation_total; what comes back is the notice the screen
        prints, in the shape both of them hand it."""
        if 0 <= index < len(self.quantity_constraints):
            gone = self.quantity_constraints.pop(index)
            if gone.get('source') == 'formulation_total':
                self.formulation_total = None
                removed = self._drop_percent_limits()
                self.save()
                return self.limit_removed_messages(
                    [dict(qc, reason='no_default') for qc in removed])
            self.save()
        return []

    # ------------------------------------------------------------------ #
    #  Total of each formulation
    # ------------------------------------------------------------------ #

    def fixed_ingredient_total(self):
        """What the ingredients add up to when every one of them is fixed,
        or None while any of them can still move.

        The one number a project with nothing to vary in the bowl can be
        refused in: there is no range to widen and no amount to move, so
        the refusal names what the amounts make and what was asked for."""
        total = 0.0
        for var in self.variables:
            if var.get('category', 'ingredient') != 'ingredient':
                continue
            # A row worked out from the others is not pinned at anything,
            # and the sum of the pinned rows would leave it out: there is
            # no one number to name.
            if self.has_formula(var) or not self.is_fixed(var):
                return None
            total += self._fixed_value(var)
        return total

    def total_reach(self):
        """(lowest, highest) — the totals the allowed amounts can add up to.

        The sum of every ingredient's Lowest and the sum of every ingredient's
        Highest. A total outside that pair is not a tight fit, it is
        arithmetic that has no answer, and the refusal says so in those two
        numbers rather than letting the search fail later with nothing to
        show for it.

        A FIXED ingredient needs no special case: its Lowest is its Highest,
        so it adds the same amount at both ends — exactly as _snap_to_total
        takes its amount off the target before moving anything.

        A FORMULA row does need one, and gets it from _achievable_range: the
        sum is read off the rows each formula is worked out FROM, so a
        balance row does not leave the reach reading as the two ends of
        rows that can never both be at them.

        A BALANCE row is the one case with no top at all. It takes whatever
        is left of the batch size, so every formulation adds up to the size
        asked for however large that is: there is no size these ingredients
        cannot make, only a size too small for the other rows to fit
        inside. So the reach is (what the other rows need at least,
        unbounded) — and the low end is refused in the balance's own words,
        not as "the least these ingredients can make". Reading the batch
        size at both ends said the same number was the floor and the
        ceiling, and the box then accepted nothing at all.
        """
        names = {v['name'] for v in self.variables
                 if v.get('category', 'ingredient') == 'ingredient'}
        balance = self._balance_row()
        if balance is not None:
            others = names - {balance['name']}
            least = self._form_reach(
                self._weighted_form(
                    lambda name: 1.0 if name in others else 0.0),
                self._batch_size())[0]
            return (least, float('inf'))
        return self._achievable_range(
            lambda name: 1.0 if name in names else 0.0)

    def _formulation_total_index(self):
        """Where the Total of each formulation box's own limit sits in
        quantity_constraints, or None. It is found by its tag, never by its
        ingredients: a limit the user typed on every ingredient by hand is a
        different limit that happens to cover the same names."""
        for i, qc in enumerate(getattr(self, 'quantity_constraints', [])):
            if qc.get('source') == 'formulation_total':
                return i
        return None

    def has_formulation_total(self):
        """True while the project says how big a formulation is. The one
        question four screens ask — tab 1's box, tab 2's hidden box, the
        seed and the store — so it is answered in one place."""
        return getattr(self, 'formulation_total', None) is not None

    def _snap_to_total(self, recipe, total):
        """Move one candidate onto the total, or None if it cannot get there.

        A total is an equality on a sum, and rejection sampling is hopeless
        against one: near the ends of what the allowed amounts reach, almost
        no random point lands in the band, and the opening batch failed with
        a sentence about limits being too restrictive. So the space-filling
        point is not tested against the total, it is PROJECTED onto it — the
        part of each amount above its Lowest is rescaled so the sum comes
        out right, clipped back into the allowed amounts, and the leftover
        redistributed among the amounts that still have room. The design
        stays spread out; it now spreads across the face of the box the
        total cuts, which is the only place a valid formulation lives.

        A fixed ingredient takes no part: its Lowest is its Highest, so its
        amount comes off the target first and never moves.

        A formula row takes no part either, but it changes what moving
        another row is WORTH. Each row the search moves carries an effective
        coefficient — 1 for itself, plus whatever every formula reads it at
        — so putting a gram into it may add more than a gram to the total,
        or less, or nothing. The rescaling is done in those coefficients; a
        row worth nothing or less takes no share of a correction. When every
        coefficient is 0, which is exactly what a balance row makes true,
        the total holds wherever the point already is and the candidate
        comes back as it went in, with the formulas written in."""
        form = LinearForm()
        for var in self.variables:
            if var.get('category', 'ingredient') != 'ingredient':
                continue
            form = form + self._linear_form(var['name'])
        constant = form.const + form.batch * self._batch_size()
        by_name = self._by_name()
        weights = {}
        for name, coeff in form.terms.items():
            var = by_name.get(name)
            if var is None:
                continue
            if self.is_fixed(var):
                constant += coeff * self._fixed_value(var)
            elif var.get('category', 'ingredient') != 'ingredient':
                # A process setting a formula reads. The search moves it,
                # but not onto the total: it is whatever this candidate
                # already says it is.
                constant += coeff * _amount(recipe.get(name))
            else:
                weights[name] = coeff

        names, lows, caps, start = [], [], [], []
        for var in self._free_ingredients():
            low = float(var['bounds'][0])
            high = float(var['bounds'][1])
            value = _amount(recipe.get(var['name']))
            names.append(var['name'])
            lows.append(low)
            caps.append(max(0.0, high - low))
            start.append(min(max(value - low, 0.0), max(0.0, high - low)))
        coeffs = [weights.get(name, 0.0) for name in names]

        base = constant + sum(c * low for c, low in zip(coeffs, lows))
        base += sum(c * s for c, s in zip(coeffs, start) if c <= 0)
        movable = [i for i, c in enumerate(coeffs) if c > 0]
        sinks = [i for i, c in enumerate(coeffs) if c < 0]
        need = float(total) - base
        snapped = dict(recipe)
        for i, name in enumerate(names):
            snapped[name] = lows[i] + start[i]

        if not movable and not sinks:
            # Nothing the search moves changes the total. Either the point
            # is already on it — every ingredient fixed and adding up, or a
            # balance row making it hold identically — or nothing can put it
            # there. The same slack _check_constraints allows the total's
            # own limit, so the two agree about one candidate.
            if abs(need) > 1e-6 * (1.0 + abs(float(total))):
                return None
            for var in self.variables:
                if (var.get('category', 'ingredient') == 'ingredient'
                        and not self.has_formula(var) and self.is_fixed(var)):
                    snapped[var['name']] = self._fixed_value(var)
            return self.fill_formulas(snapped)

        # Which way each row moves the total, and how far. A row worth more
        # than nothing raises it by rising; a row worth LESS than nothing —
        # one a formula subtracts more of than it adds — raises it by
        # falling, and lowers it by rising. Both are real ways onto the
        # total, and leaving the second out is what made total_reach and
        # this disagree about the same project: the reach counted a row the
        # projection would not move.
        room = sum(coeffs[i] * caps[i] for i in movable)
        lift = sum(-coeffs[i] * start[i] for i in sinks)
        drop = sum(-coeffs[i] * (caps[i] - start[i]) for i in sinks)
        if need < -drop - 1e-9 or need > room + lift + 1e-9:
            return None         # the ingredients that can move cannot reach it
        # The rows worth more than nothing go first, so a point a one-sided
        # projection already handled comes out of this exactly as it did
        # before; what is left over is what the rows worth less than nothing
        # have to answer for.
        up_need = min(max(need, 0.0), room)
        sink_need = need - up_need
        # Worked in what each part of an amount is WORTH to the total, so
        # the share-out below is the one it always was and the coefficients
        # are divided back out at the end.
        free = _share_to_total([coeffs[i] * start[i] for i in movable],
                               [coeffs[i] * caps[i] for i in movable],
                               up_need)
        for k, i in enumerate(movable):
            snapped[names[i]] = lows[i] + free[k] / coeffs[i]
        if sinks:
            # Read from the row's Highest down, so the same share-out reads
            # a rising number as a rising total here too.
            headroom = [-coeffs[i] * (caps[i] - start[i]) for i in sinks]
            fell = _share_to_total(
                headroom, [-coeffs[i] * caps[i] for i in sinks],
                sum(headroom) + sink_need)
            for k, i in enumerate(sinks):
                snapped[names[i]] = (lows[i] + caps[i]
                                     + fell[k] / coeffs[i])
        return self.fill_formulas(snapped)

    def _formulation_total_bounds(self, total):
        """The band a total is enforced as: the total, give or take
        FORMULATION_TOTAL_TOLERANCE."""
        value = float(total)
        return (value * (1.0 - self.FORMULATION_TOTAL_TOLERANCE),
                value * (1.0 + self.FORMULATION_TOTAL_TOLERANCE))

    def set_formulation_total(self, total):
        """Every suggested formulation adds up to `total`.

        Stored as a number in its own right AND written as the limit over
        every ingredient, because the limit is what the cold start and the
        model already obey — there is no second mechanism to keep in step.
        The band never makes a reachable total infeasible: the total itself
        is a point the allowed amounts reach, and the band is centred on it.

        Refused, in the numbers, when the allowed amounts cannot add up to it
        at all, and refused in the usual words while the ingredients are not
        all in one unit — a sum across units is a number of nothing."""
        value = float(total)
        balance = self._balance_row()
        if balance is None:
            lowest, highest = self.total_reach()
            if value > highest:
                raise ValueError(wording.total_not_reachable_at_most(
                    self.batch_total_text(value),
                    self.batch_total_text(highest)))
            if value < lowest:
                raise ValueError(wording.total_not_reachable_at_least(
                    self.batch_total_text(value),
                    self.batch_total_text(lowest)))
        else:
            # With a balance row every formulation adds up to the batch size
            # by construction, so the reach is that number at both ends and
            # the two sentences above would compare it with itself. The
            # question that is left is whether there is anything for the
            # balance to be: the other ingredients have a lowest sum, and a
            # batch size under it leaves the balance below nothing.
            others = {v['name'] for v in self.variables
                      if v.get('category', 'ingredient') == 'ingredient'
                      and v['name'] != balance['name']}
            least = self._form_reach(
                self._weighted_form(
                    lambda name: 1.0 if name in others else 0.0), value)[0]
            if value < least - 1e-9:
                raise ValueError(wording.balance_would_go_negative(
                    balance['name'], self.batch_total_text(value),
                    self.batch_total_text(least),
                    unit=self._unit_of(balance)))
        # A project every one of whose amounts can be 0 reaches 0, so the
        # sentence above lets a total of nothing through. It is refused in
        # the same shape rather than as the band's own "At least must be less
        # than At most", which named two boxes the user never saw.
        if value <= 0:
            raise ValueError(wording.total_not_reachable_at_all(
                self.batch_total_text(value)))
        low, high = self._formulation_total_bounds(value)
        index = self._formulation_total_index()
        if (value == getattr(self, 'formulation_total', None)
                and index is not None):
            return                      # a rerun, not a change: no save
        # Written before it is stored: add_total_mass_constraint refuses a
        # set of ingredients that share no unit, and a stored total whose
        # limit was refused would say the suggestions add up to something
        # nothing holds them to.
        previous = (self.quantity_constraints.pop(index)
                    if index is not None else None)
        try:
            self.add_total_mass_constraint(low, high,
                                           source='formulation_total')
        except ValueError:
            # Refused (the ingredients share no unit): the project is left
            # exactly as it was, rather than holding a total nothing enforces.
            if previous is not None:
                self.quantity_constraints.insert(index, previous)
            raise
        self._keep_limit_position(index)
        self.formulation_total = value
        # The open batch was generated under the old answer: its rows were
        # built to a total that no longer holds, and its sheets name it. It
        # goes the way every other set-up change sends it, with the same
        # notice, rather than sitting on screen as a batch nothing on tab 1
        # describes.
        self._drop_pending_batch()
        self.save()
        # Every limit written as a % of batch size is a percent OF this
        # number, so a new one rewrites them all in the one place that
        # knows the new figure.
        return self._resync_percent_limits()

    def _keep_limit_position(self, index):
        """Put the limit just appended back where the old one stood. A limit
        that jumped to the bottom of the Limits list every time the ingredient
        list was touched read as a new limit the user had not written."""
        if index is None or index >= len(self.quantity_constraints) - 1:
            return
        self.quantity_constraints.insert(index, self.quantity_constraints.pop())

    def _drop_percent_limits(self):
        """Every limit written as a % of batch size, taken off the project
        and handed back plain (no 'reason' on it yet). There is nothing
        left for one to be a percent OF the moment the default that wrote
        it is gone — whichever door took the default away: the box
        cleared, a unit split across the ingredients, or amounts that no
        longer reach it."""
        dropped = [qc for qc in self.quantity_constraints if qc.get('percent')]
        if dropped:
            self.quantity_constraints = [
                qc for qc in self.quantity_constraints if not qc.get('percent')]
        return dropped

    def clear_formulation_total(self):
        """Back to "any total the allowed amounts reach": the number goes and
        so does the limit it wrote — and every limit written as a % of
        batch size goes with it, one line each: there is nothing left for
        it to be a percent OF. A no-op when there is nothing to clear, so a
        rerun does not bump the file's mtime.

        Returns the one-line notice for every percent limit taken with it,
        as (kind, message) pairs — the shape every other door onto
        limit_removed_messages hands the screen."""
        index = self._formulation_total_index()
        if index is None and getattr(self, 'formulation_total', None) is None:
            return []
        if index is not None:
            self.quantity_constraints.pop(index)
        removed = self._drop_percent_limits()
        self.formulation_total = None
        self._drop_pending_batch()
        self.save()
        return self.limit_removed_messages(
            [dict(qc, reason='no_default') for qc in removed])

    def _resync_percent_limits(self):
        """Rewrite every limit written as a % of batch size against the
        CURRENT default, and drop the ones the new number leaves nothing
        can meet — named through limit_removed_messages like every other
        limit an edit empties of meaning (reason 'percent_unreachable').

        Runs at the end of set_formulation_total. A percent limit stores
        the percent it was written as, not only the grams it came to that
        day, exactly so a later default can rewrite it here instead of
        stranding it at yesterday's number.

        Returns the (kind, message) pairs the screen owes: one line saying
        the rewrite happened, at all, whenever a percent limit exists to
        rewrite, and then one per limit it had to drop."""
        total = getattr(self, 'formulation_total', None)
        if total is None or not any(
                qc.get('percent') for qc in self.quantity_constraints):
            return []
        to_grams = lambda p: None if p is None else p / 100.0 * total
        kept, removed = [], []
        for qc in self.quantity_constraints:
            percent = qc.get('percent')
            if not percent:
                kept.append(qc)
                continue
            rewritten = dict(qc)
            if percent.get('exactly') is not None:
                rewritten['exactly'] = to_grams(percent['exactly'])
                rewritten['min'], rewritten['max'] = \
                    self._formulation_total_bounds(rewritten['exactly'])
            else:
                rewritten['min'] = to_grams(percent['min'])
                rewritten['max'] = to_grams(percent['max'])
            if self._quantity_limit_refusal(rewritten):
                removed.append(dict(rewritten, reason='percent_unreachable'))
            else:
                kept.append(rewritten)
        self.quantity_constraints = kept
        self.save()
        messages = [("info", wording.percent_limits_rebased(
            self.batch_total_text(total)))]
        messages += self.limit_removed_messages(removed)
        return messages

    def _sync_formulation_total(self):
        """Keep the total's limit true to the ingredient list, and return
        what was dropped so the screen can say so.

        The limit is over EVERY ingredient, so every edit to the list moves
        it: an ingredient added is one more the total has to cover, one
        deleted is one fewer, and a unit set on one of them can leave the sum
        adding grams to millilitres. Three answers, in order — rewrite it,
        drop it because the amounts can no longer reach the total, drop it
        because there is no one unit to add them in.

        Either drop takes every % of batch size limit with it: an edit that
        blanks the default the same way clear_formulation_total's own box
        does must own the same consequence, or a percent limit is left
        naming a batch size that no longer exists — and still enforced at
        yesterday's grams, silently."""
        total = getattr(self, 'formulation_total', None)
        index = self._formulation_total_index()
        if total is None:
            if index is not None:
                self.quantity_constraints.pop(index)
            return []
        if index is not None:
            self.quantity_constraints.pop(index)
        # The unit is carried out with it: by the time the screen names the
        # total that went, the ingredients may share no unit for it to look
        # up, and a bare '100' names no amount at all.
        gone = {'ingredients': [], 'source': 'formulation_total',
                'total': float(total),
                'unit': self.one_amount_unit() or self.majority_amount_unit()}
        if self.has_ingredients() and len(self.ingredient_units()) > 1:
            self.formulation_total = None
            return [dict(gone, reason='unit')] + [
                dict(qc, reason='no_default')
                for qc in self._drop_percent_limits()]
        lowest, highest = self.total_reach()
        # Nothing weighed out reaches (0, 0), so a project whose last
        # ingredient has just gone falls through to the same answer as one
        # whose amounts no longer add up: the total is unreachable.
        if not self.has_ingredients() or not lowest <= float(total) <= highest:
            self.formulation_total = None
            return [dict(gone, reason='unreachable')] + [
                dict(qc, reason='no_default')
                for qc in self._drop_percent_limits()]
        low, high = self._formulation_total_bounds(total)
        self.add_total_mass_constraint(low, high, source='formulation_total')
        self._keep_limit_position(index)
        return []

    def recorded_total(self, batch_no):
        """What batch `batch_no` was actually made to, for a batch already
        recorded: its own stored total, and the project's only for a batch
        made before totals were stored at all.

        The opposite order to open_round_size, and deliberately: a round on
        the bench is being made NOW, to whatever the bench or the project
        says; a round in the records was made once, to a number that cannot
        change afterwards because someone later typed a different default on
        tab 1.

        None means "as generated", whether the batch recorded that answer
        itself or predates the record being kept at all. Falling back to the
        project's current total put a number on a batch nobody made to it.
        """
        stored = self.batch_total(batch_no)
        return None if stored is None else float(stored)

    def open_round_size(self):
        """The batch size the OPEN round is being made to: the size the bench
        typed on the round screen if there is one, else the project's
        default, else None for as generated.

        The round's own answer wins. Until 0.5.0 the project's default hid
        tab 2's box altogether, so one accessor (`sheet_total`) could put the
        project first and be right for the bench as well as for the records;
        now the box is always there, scaling the round is how a bench makes
        one round bigger than the default, and the two questions have
        opposite answers. They are two accessors accordingly: this one for
        the round on the bench, `recorded_total` for a round in the records,
        and nothing has to pick between them.
        """
        stored = getattr(self, 'pending_batch_total', None)
        if stored is not None:
            return float(stored)
        project_total = getattr(self, 'formulation_total', None)
        return None if project_total is None else float(project_total)

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

    def edit_note(self, index, note):
        """Correct one formulation's note. It is part of the record — which
        bowl it was, what went wrong — and a correction that could change
        every number on the row but not the sentence beside them left the
        reader with a note about a formulation that had moved."""
        if index < 0 or index >= len(self.notes_history):
            raise IndexError("There is no formulation at that position.")
        self.notes_history[index] = "" if note is None else str(note)
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
                        if s.get(ROUND_FIELD) is None
                        or (cut is not None and int(s[ROUND_FIELD]) <= cut)]
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

        A FORMULA row has no column: its amount is worked out from the rows
        that do, so a column for it would say the same thing twice and the
        GP would be handed a coordinate nothing can choose.
        """
        vector = []
        for var in self.variables:
            if self.has_formula(var):
                continue
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
        """Decode a flat numeric vector back into a recipe dict.

        The formula rows are written in at the end, from the rows the
        vector holds: they left the search, so they are worked out rather
        than read off it."""
        recipe = {}
        idx = 0
        for var in self.variables:
            if self.has_formula(var):
                continue
            if var['type'] == 'continuous':
                recipe[var['name']] = float(vector[idx])
                idx += 1
            elif var['type'] == 'categorical':
                n_opts = len(var['options'])
                one_hot_segment = vector[idx:idx + n_opts]
                best_idx = np.argmax(one_hot_segment)
                recipe[var['name']] = var['options'][best_idx]
                idx += n_opts
        return self.fill_formulas(recipe)

    def _search_bounds(self):
        """[(low, high)] per column of the [0,1]^d frame the search runs in.

        A FIXED row is one point, and a point has no frame: normalizing by a
        zero span is a division by zero, and it is the encoded history that
        is handed through that normalization to the GP. So a fixed column is
        widened — to whatever the recorded formulations already span, so the
        history still lands inside [0, 1], and to one unit when they span
        nothing at all. The row itself does not move: _get_fixed_features
        pins its coordinate inside the widened frame.

        A FORMULA row has no column here either: _encode left it out, so
        the frame has nothing to place."""
        spans = []
        col = 0
        for var in self.variables:
            if self.has_formula(var):
                continue
            if var['type'] == 'continuous':
                lo, hi = float(var['bounds'][0]), float(var['bounds'][1])
                # A row whose ABSENT value sits outside its allowed
                # amounts: an ingredient added mid-run with a Lowest above 0
                # (0.5.0 lets the grid ask for that), or a setting fixed away
                # from the baseline its past bakes ran at. Those formulations
                # really are encoded at that value, so the frame has to reach
                # it or the GP is handed training rows outside its own [0, 1]
                # box.
                #
                # Only when some recipe actually LACKS the row, which is the
                # only way the absent value reaches the encoding. An
                # ingredient that has been there all along was recorded at
                # its own amounts, and widening its frame down to 0 would
                # change what the GP sees for every project that has one.
                absent = var.get(
                    '_absent_value',
                    0.0 if var.get('category', 'ingredient') == 'ingredient'
                    else None)
                if absent is not None and any(var['name'] not in recipe
                                              for recipe in self.recipe_history):
                    lo, hi = min(lo, float(absent)), max(hi, float(absent))
                if hi <= lo:
                    seen = [float(row[col]) for row in self.X_history
                            if col < len(row)]
                    lo, hi = min([lo] + seen), max([hi] + seen)
                    # A floor, not just "wider than zero": a history that
                    # moved by 1e-13 is a span the normalization would
                    # divide by, and the frame would blow up rather than
                    # break. Scaled by the numbers themselves, because a
                    # setting at 175 °C and an amount at 0.0001 g are not
                    # near each other on the same absolute scale.
                    if hi - lo <= _FIXED_SPAN_FLOOR * max(1.0, abs(lo)):
                        hi = lo + 1.0
                spans.append((lo, hi))
                col += 1
            elif var['type'] == 'categorical':
                for _ in var['options']:
                    spans.append((0.0, 1.0))
                    col += 1
        return spans

    def _get_bounds(self):
        """Return a (2, dim) tensor of [mins, maxs] for all variables."""
        spans = self._search_bounds()
        return torch.tensor([[lo for lo, _ in spans], [hi for _, hi in spans]],
                            dtype=torch.double)

    # ------------------------------------------------------------------ #
    #  Internal: Constraint Helpers
    # ------------------------------------------------------------------ #

    def _get_botorch_constraints(self):
        """Build BoTorch inequality constraints from property + quantity
        constraints, and the floor every formula row carries.

        Every limit is written out through _linear_form before it meets a
        column, so a limit that names a row which is worked out becomes
        coefficients on the rows it is worked out FROM, plus an offset. A
        limit that comes out a constant is dropped when the constant meets
        it and refused when it does not — the same branch a limit only
        fixed rows feed has always taken."""
        constraints_list = []
        columns = self._search_columns()
        known = {var['name'] for var in self.variables
                 if var['type'] == 'continuous'}

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
                form = self._weighted_form(
                    self._property_coeff(metric, constr[bound]))
                indices, coeffs, offset = self._form_over_columns(form,
                                                                  columns)
                if not indices:
                    # Every ingredient carrying this property is FIXED or
                    # worked out, so the average is one number and there is
                    # nothing for the solver to choose. Dropping the row
                    # here let `ask` hand back formulations the app's own
                    # _check_constraints then calls invalid, with nothing on
                    # screen to say why.
                    self._refuse_unreachable_limit(
                        self._property_limit_refusal(constr))
                    continue
                constraints_list.append((
                    torch.tensor(indices, dtype=torch.long),
                    torch.tensor([sense * c for c in coeffs],
                                 dtype=torch.double),
                    -sense * offset,
                ))

        # Quantity constraints (direct sum of ingredient quantities)
        for qc in getattr(self, 'quantity_constraints', []):
            form = LinearForm()
            for ing_name in qc['ingredients']:
                if ing_name in known:
                    form = form + self._linear_form(ing_name)
            indices, coeffs, offset = self._form_over_columns(form, columns)

            if not indices:
                # The sum is the constant `offset`: every row this limit
                # reads is FIXED, or worked out from rows that are. Either
                # the constant meets the limit or it cannot — and "cannot"
                # is a refusal, not a row to drop silently.
                self._refuse_unreachable_limit(self._quantity_limit_refusal(qc))
                continue
            t_idx = torch.tensor(indices, dtype=torch.long)
            t_coeffs = torch.tensor(coeffs, dtype=torch.double)

            if qc['min'] is not None:
                constraints_list.append((t_idx, t_coeffs, qc['min'] - offset))
            if qc['max'] is not None:
                constraints_list.append((t_idx, -t_coeffs, -(qc['max'] - offset)))

        # The floor every formula row carries. An amount worked out from the
        # others is still an amount: below zero it is nothing anybody can
        # weigh, and the search has to be told so rather than shown it
        # afterwards.
        for var in self._formula_rows():
            indices, coeffs, offset = self._form_over_columns(
                self._linear_form(var['name']), columns)
            if not indices:
                if offset < 0:
                    self._refuse_unreachable_limit(wording.formula_below_zero(
                        var['name'], self._unit_of(var)))
                continue
            constraints_list.append((
                torch.tensor(indices, dtype=torch.long),
                torch.tensor(coeffs, dtype=torch.double),
                -offset,
            ))

        return constraints_list

    def _fixed_ingredients_among(self, names=None):
        """The fixed ingredients out of `names`, in project order — or every
        fixed ingredient when `names` is None, which is what a property
        limit reads (adding water is how a formulation is diluted, so every
        ingredient carries one)."""
        wanted = None if names is None else set(names)
        return [v['name'] for v in self.variables
                if not self.has_formula(v) and self.is_fixed(v)
                and v.get('category', 'ingredient') == 'ingredient'
                and (wanted is None or v['name'] in wanted)]

    def _refuse_limit_the_fixed_rows_break(self, what, refusal, names,
                                           put_back):
        """Refuse a limit at the door it is written at when the rows it reads
        are already pinned at one amount.

        The far end of the same problem the round guards: a limit only fixed
        rows feed is a constant, and a constant the limit does not hold is a
        question with no answer. Caught here it names the rows; let through,
        it surfaced as Generate quietly handing back formulations the app's
        own check calls invalid.

        Only when a fixed row is why. A limit nothing can meet for its own
        sake — a minimum above every ingredient's figure — is refused the
        way it always was, by Generate, because that is a limit the reader
        may still be part way through writing."""
        if not refusal:
            return
        fixed = self._fixed_ingredients_among(names)
        if not fixed:
            return
        put_back()
        raise ValueError(wording.limit_fixed_rows_break(
            what, number_list(fixed), len(fixed) > 1))

    @staticmethod
    def _refuse_unreachable_limit(message):
        """Raise `message` if there is one. The one line between "this limit
        is a constant that happens to hold" and "this limit is a constant
        that does not"."""
        if message:
            raise ValueError(message)

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

        # Quantity constraints, with the same relative slack the property
        # loop above uses: a sum reached the long way round (as
        # _snap_to_total reaches it, a share at a time) lands a few bits
        # above the limit it was built to sit exactly on, and a candidate
        # rejected by the last bit of a float is a suggestion the bench
        # never sees.
        for qc in getattr(self, 'quantity_constraints', []):
            total_val = sum(recipe_dict.get(name, 0.0) for name in qc['ingredients'])
            slack = 1e-6 * (1.0 + abs(float(total_val)))
            if qc['min'] is not None and total_val < float(qc['min']) - slack:
                return False
            if qc['max'] is not None and total_val > float(qc['max']) + slack:
                return False

        # The floor every formula row carries, checked the way the model was
        # told it. An amount worked out from the others is still an amount,
        # and a formulation that asks the bench for less than none of
        # something is not one.
        for form in self._resolved_forms().values():
            value = self._form_value(form, recipe_dict)
            if value < -1e-6 * (1.0 + abs(value)):
                return False

        return True

    # ------------------------------------------------------------------ #
    #  Core Loop: Ask / Tell
    # ------------------------------------------------------------------ #

    def ask(self, n_suggestions=1, n_init_random=COLD_START_RUNS, batch_no=None,
            discarded=None):
        """Suggest the next batch of recipes to try.

        A fixed variable (its Lowest is its Highest) is pinned at that one
        amount: the search runs over the rest, while the surrogate still
        sees every past observation.

        `batch_no` is for `Generate a different batch`, which keeps the number
        the batch it replaces was wearing; `discarded` are the formulation
        numbers that regenerate retired, for the one caption that says so.
        Passing the number HERE rather than re-stamping the batch afterwards
        is what stops a regenerate spending a batch number nobody ever saw.
        """
        if not self.varying_variables():
            raise ValueError(wording.EVERYTHING_IS_FIXED)
        bounds_tensor = self._get_bounds()
        dim = bounds_tensor.shape[1]

        # Cold start: space-filling Sobol sequence
        if len(self.X_history) < n_init_random:
            recipes = self._ask_cold_start(n_suggestions, bounds_tensor, dim)
        else:
            # Warm: GP-based Bayesian optimization
            recipes = self._ask_optimize(n_suggestions, bounds_tensor, dim)
            if self.has_formulation_total():
                recipes = [self._snapped_if_it_still_fits(rec)
                           for rec in recipes]

        # Numbers are issued here, at generation, and never reissued.
        self.set_pending_batch(recipes, batch_no=batch_no, discarded=discarded)
        return recipes

    def _snapped_if_it_still_fits(self, recipe):
        """One suggestion of a warm batch, moved onto the project's total —
        but only while it keeps every other limit.

        The total is enforced on the optimiser as a band (the limit is
        ±0.5 %), so the model can land at 99.5 g where the project says 100,
        and the sheets then carry an amount the caution has to apologise
        for. Projecting it onto the total is the same move the cold start
        makes — except that the cold start's candidates are projected BEFORE
        anything is checked, while these have already been chosen inside
        every limit the user wrote. Moving 0.5 g back into the amounts can
        push a limit of its own over ('Pea protein isolate + Wheat gluten at
        most 20 g' became 20.32 g, silently), so the projected row is used
        only if it still satisfies them all. When it does not, the
        optimiser's own row stands: it is inside the band, and the caution
        on the sheets says the total it was made to.
        """
        snapped = self._snap_to_total(recipe, self.formulation_total)
        if snapped is None or not self._check_constraints(snapped):
            return recipe
        return snapped

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

    def _sobol_seed(self):
        """The scramble this project's space-filling design is drawn from.
        Fixed once per project, and stored: a seed derived from anything that
        moves would give the project a different sequence every time it is
        asked for one.

        A file written before the seed was stored has none, so it is derived
        from the project's own name — deterministic, and the same on every
        machine that opens the file."""
        seed = getattr(self, 'sobol_seed', None)
        if seed is None:
            digest = hashlib.sha256(str(self.project_name).encode('utf-8'))
            seed = int(digest.hexdigest()[:8], 16)
            self.sobol_seed = seed
        return int(seed)

    def _ask_cold_start(self, n_suggestions, bounds_tensor, dim):
        """The space-filling opening: the next points of ONE Sobol sequence.

        Not a fresh sequence per call. A scrambled Sobol design is only
        space-filling as a whole, and reseeding it on every generate made
        five formulations asked for as 3 then 2 five points from two
        unrelated scrambles — clustered exactly where the caption promised
        they would not be. The project draws one sequence and fast-forwards
        past the points it has already issued, so 3 + 2 lands on the same
        five points as 5 in one go, and a regenerate still differs because
        the formulation numbers have moved on."""
        for size in self.COLD_START_POOLS:
            candidates = self._cold_start_pool(size, bounds_tensor, dim)

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
                results = [x[1] for x in scored[:n_suggestions]]
            else:
                # Standard: pick first feasible candidates
                results = []
                for candidate in candidates:
                    if self._check_constraints(candidate):
                        results.append(candidate)
                    if len(results) >= n_suggestions:
                        break

            if len(results) >= n_suggestions:
                return results

        # A wider pool did not fill the batch. With results already in, the
        # model can be asked instead — it is handed the same limits as
        # inequalities and solves them rather than sampling for them.
        if self.X_history:
            try:
                chosen = self._ask_optimize(n_suggestions, bounds_tensor, dim)
            except Exception:
                chosen = []
            if chosen:
                return chosen
        if results:
            return results

        if self.has_formulation_total():
            # The total is one number the user typed, and it is what nothing
            # could satisfy: the refusal names it rather than talking about
            # limits the user never wrote.
            fixed_sum = self.fixed_ingredient_total()
            if fixed_sum is not None:
                # Every ingredient is fixed, so there is nothing to widen
                # and nothing to move: two numbers is the whole answer.
                raise ValueError(wording.fixed_amounts_do_not_add_up(
                    self.batch_total_text(fixed_sum),
                    self.batch_total_text(self.formulation_total)))
            raise ValueError(wording.no_formulation_reaches_total(
                self.batch_total_text(self.formulation_total)))
        # One sentence for a failed Generate, whichever path failed: the
        # screen used to show this one or wording.GENERATE_FAILED depending
        # on which internal step gave up, in two registers.
        raise ValueError(wording.GENERATE_FAILED)

    def _cold_start_pool(self, size, bounds_tensor, dim):
        """`size` points of this project's ONE Sobol sequence, decoded, and —
        while the project has a total — projected onto it.

        A fresh engine each time, fast-forwarded the same way, so a larger
        pool opens with exactly the points the smaller one held: growing the
        pool can only add candidates after the ones already considered, never
        renumber them."""
        sobol = SobolEngine(dimension=dim, scramble=True,
                            seed=self._sobol_seed())
        # _ask_seed is the NEXT formulation number, so one less is how many
        # points of this sequence the project has already spent.
        already = max(0, int(self._ask_seed()) - 1)
        if already:
            sobol.fast_forward(already)
        pool_norm = sobol.draw(size).double()

        # Pin inactive variables so the Sobol design also lives in X_S.
        for col, z in self._get_fixed_features().items():
            pool_norm[:, col] = z

        candidates = [
            self._decode(unnormalize(pool_norm[i], bounds_tensor).numpy().flatten())
            for i in range(size)
        ]
        if not self.has_formulation_total():
            return candidates
        snapped = (self._snap_to_total(rec, self.formulation_total)
                   for rec in candidates)
        return [rec for rec in snapped if rec is not None]

    def _ask_optimize(self, n_suggestions, bounds_tensor, dim):
        """Generate recipes using a GP + the configured acquisition (default qLogNEI)."""
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
            raise ValueError(wording.ADD_A_MEASUREMENT_FIRST)
        kept = {k: v for k, v in results_dict.items() if v is not None}
        if not any(obj['name'] in kept for obj in self.objectives):
            raise ValueError(wording.ENTER_A_MEASUREMENT)

        if formulation_no is None:
            formulation_no = self._issue_formulation_no()
        else:
            self._retire_formulation_no(formulation_no)
        if batch_no is None:
            batch_no = self.pending_batch_no
        elif batch_no is NO_BATCH:
            batch_no = None          # belongs to no batch this project made
        else:
            self._retire_batch_no(batch_no)

        self.X_history.append(self._encode(recipe_dict))
        self.Y_history.append(self._compute_utility(kept))
        self.recipe_history.append(dict(recipe_dict))
        self.results_history.append(kept)
        self.timestamps_history.append(datetime.now(timezone.utc).isoformat())
        self.formulation_ids.append(int(formulation_no))
        self.batch_history.append(None if batch_no is None else int(batch_no))
        self.notes_history.append("" if note is None else str(note))
        # The total the open batch's sheets were printed to belongs with the
        # batch number, not with the row: the amounts stored are as generated,
        # and only this says what the bench weighed out. It is written when a
        # result arrives, because the open batch is cleared straight after.
        if batch_no is not None and batch_no == self.pending_batch_no:
            # open_round_size, not the box's own number: a round the bench
            # never re-sized was printed to the project's default, and the
            # box's value must not be read as an answer it never gave.
            #
            # Written even when it is None. "This round was made as
            # generated" is an answer, and the only place it is kept: with
            # the key absent, a default typed on tab 1 months later was read
            # back as the size this round had been made to.
            made_to = self.open_round_size()
            self._batch_totals()[int(batch_no)] = (
                None if made_to is None else float(made_to))
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
        self.pending_batch_total = None

    def _issue_formulation_no(self):
        n = int(self.next_formulation_no)
        self.next_formulation_no = n + 1
        return n

    def _retire_formulation_no(self, no):
        """Make sure a number handed in from outside can never be issued again."""
        self.next_formulation_no = max(int(self.next_formulation_no), int(no) + 1)

    def next_batch_no(self):
        """The number the next batch will carry. It is stored and only ever
        goes up, exactly as next_formulation_no does: read off the batches
        still on file, it came back down the moment one was deleted, and the
        next batch generated then reused a number the project's own records
        had already spent."""
        return int(self.next_batch_number)

    def _issue_batch_no(self):
        n = int(self.next_batch_number)
        self.next_batch_number = n + 1
        return n

    def _retire_batch_no(self, no):
        """Make sure a batch number handed in from outside can never be
        issued again."""
        if no is None:
            return
        self.next_batch_number = max(int(self.next_batch_number), int(no) + 1)

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
        # A formulation of the reader's own names the rows they typed; a row
        # that is worked out is not one of them, and the bench still has to
        # weigh it. Filled here, at the one door such a formulation comes
        # in by, so the table, the sheets and tell() all read the same
        # amounts a generated formulation would have carried.
        rows.append(self._batch_row(number, self.fill_formulas(recipe), note))
        self.pending_batch = rows
        if self.pending_batch_no is None:
            self.pending_batch_no = self._issue_batch_no()
        self._date_pending_batch()
        self.save()
        return number

    def index_of_formulation(self, no):
        """Position of formulation `no` in the scored history, or None."""
        try:
            return [int(i) for i in self.formulation_ids].index(int(no))
        except (ValueError, TypeError):
            return None

    def record_skipped(self, formulation_no, batch_no, recipe,
                       note=wording.NOT_SCORED):
        """Store a formulation that was generated but never scored. It keeps
        its number and amounts, and stays out of the scored history and the
        model. score_skipped() moves it into the scored history if a result
        turns up later."""
        self.skipped.append({
            'formulation': int(formulation_no),
            ROUND_FIELD: None if batch_no is None else int(batch_no),
            'recipe': dict(recipe),
            'note': str(note) if note else wording.NOT_SCORED,
        })
        self._retire_formulation_no(formulation_no)
        self._retire_batch_no(batch_no)
        self.save()

    def score_skipped(self, formulation_no, results_dict, note=None):
        """Score a formulation that was recorded as not scored: the bowl was
        made after all, or measured late.

        It keeps the number and the batch it was generated with — both
        retired the day the row was recorded, so tell() issues nothing — and
        its stored amounts are the ones recorded. The row leaves `skipped`
        only if tell() accepts the result: a refusal must not delete the one
        record the project holds of that formulation.
        """
        position = next((k for k, s in enumerate(self.skipped)
                         if int(s['formulation']) == int(formulation_no)), None)
        if position is None:
            raise ValueError(
                wording.formulation_is_not_not_scored(formulation_no))
        row = self.skipped.pop(position)
        try:
            self.tell(dict(row.get('recipe') or {}), results_dict,
                      formulation_no=int(row['formulation']),
                      batch_no=row.get(ROUND_FIELD),
                      note="" if note is None else str(note))
        except Exception:
            self.skipped.insert(position, row)
            raise

    def import_formulation(self, recipe_dict, results_dict,
                           note=wording.IMPORTED_NOTE):
        """Record a formulation made before this project existed. It draws the
        next global number, its batch stays blank (it belongs to no batch this
        project generated), and the note says where it came from.

        NO_BATCH, not a blank patched over afterwards: tell() would otherwise
        inherit the open batch and write the total that batch is being made
        to against a formulation made before the project existed."""
        self.tell(recipe_dict, results_dict, batch_no=NO_BATCH, note=note)

    def delete_formulation(self, no):
        """Delete one formulation by its global number, scored or left out.
        Later formulations keep their numbers; the deleted number retires.
        Returns True when something was deleted."""
        index = self.index_of_formulation(no)
        if index is not None:
            self.delete_result(index)
            self._prune_round_records()
            self.save()
            return True
        before = len(self.skipped)
        self.skipped = [s for s in self.skipped
                        if int(s['formulation']) != int(no)]
        if len(self.skipped) != before:
            self._prune_round_records()
            self.save()
            return True
        return False

    def delete_formulations(self, numbers):
        """Delete several formulations by number, scored or not, in one
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
            self._prune_round_records()
            self.save()
        return gone

    def _prune_round_records(self):
        """Forget what was kept against a round that has no rows left: the
        size it was made to, and the lot numbers it was weighed from. A
        round number is never reissued, so either one left behind could only
        ever be read against a round nobody can see any more — and the Lots
        sheet went on printing 'Round 1 · Water · L-1' for a round undo had
        taken away."""
        totals = self._batch_totals()
        lots = getattr(self, 'lots', None)
        if not isinstance(lots, dict):
            lots = self.lots = {}
        if not totals and not lots:
            return
        live = {int(b) for b in self.batch_history if b is not None}
        live |= {int(s[ROUND_FIELD]) for s in self.skipped
                 if s.get(ROUND_FIELD) is not None}
        if self.pending_batch_no is not None:
            live.add(int(self.pending_batch_no))
        for no in [n for n in totals if n not in live]:
            del totals[no]
        for no in [n for n in lots if int(n) not in live]:
            del lots[no]

    def last_batch_no(self):
        """The highest batch number in the recorded history. Left-out
        formulations count: a batch nobody managed to make is still the last
        batch, and undo has to be able to reach it."""
        seen = [int(b) for b in self.batch_history if b is not None]
        seen += [int(s[ROUND_FIELD]) for s in self.skipped
                 if s.get(ROUND_FIELD) is not None]
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
        removed += sum(1 for s in self.skipped if s.get(ROUND_FIELD) == last)
        self.skipped = [s for s in self.skipped if s.get(ROUND_FIELD) != last]
        self._prune_round_records()
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
        # The batch counter, the same way: a file from before it was stored
        # carries no number, so it starts one past the highest batch in it.
        batches = [int(b) for b in self.batch_history if b is not None]
        batches += [int(s[ROUND_FIELD]) for s in self.skipped
                    if s.get(ROUND_FIELD) is not None]
        if self.pending_batch_no is not None:
            batches.append(int(self.pending_batch_no))
        highest_batch = max(batches) if batches else 0
        self.next_batch_number = max(int(getattr(self, 'next_batch_number', 1)
                                         or 1), highest_batch + 1)
        # A file written before a batch could carry a total has none, and
        # none is exactly right: those batches were made as generated.
        self.pending_batch_total = getattr(self, 'pending_batch_total', None)
        self._batch_totals()
        # A file written before this was stored has no field at all; blank is
        # exactly right — nothing was ever said about where the targets
        # came from.
        self.targets_source = getattr(self, 'targets_source', "") or ""
        # The same for the total every suggested formulation is built to: a
        # file from before it existed asked nothing of the sum, and the limit
        # it would have written is not there either.
        self.formulation_total = getattr(self, 'formulation_total', None)

    def _date_pending_batch(self):
        """Stamp the open batch with the day it was opened, once. Both ways a
        batch opens — ask() through set_pending_batch, and a formulation of
        the user's own through add_to_pending_batch — come through here, so
        the date the printable sheets carry is the same one either way, and a
        row added later never re-dates the batch it joined."""
        if self.pending_batch_created is None:
            self.pending_batch_created = datetime.now().astimezone().strftime("%Y-%m-%d")

    def _batch_totals(self):
        """The totals by batch number, created for a session object made
        before they were stored."""
        totals = getattr(self, 'batch_totals', None)
        if not isinstance(totals, dict):
            totals = self.batch_totals = {}
        return totals

    def batch_total(self, batch_no):
        """The total batch `batch_no` was made to, or None for as generated."""
        if batch_no is None:
            return None
        return self._batch_totals().get(int(batch_no))

    def set_pending_batch_total(self, total):
        """Remember the total the open batch is being made to. Writes only on
        a change: the screen sets this on every rerun, and a save per rerun
        would bump the file's mtime and make another open window see a false
        conflict."""
        value = None if total is None else float(total)
        if value == getattr(self, 'pending_batch_total', None):
            return
        self.pending_batch_total = value
        self.save()

    def scale_round(self, batch_size):
        """Make every formulation in the open round to `batch_size`.

        The round screen's Batch size box calls this. Until 0.5.0 that box
        scaled the PICTURE — `scaled_recipe` rewrote the table and the sheets
        on the way out while the stored rows kept their own amounts — so a
        round printed at 150 g was recorded at the 50 g the model had
        proposed, and the bench's own row was not rewritten at all. Here the
        amounts move: the table, the workbook and what `tell` records are one
        set of numbers, because they are the same numbers.

        Proportional, from what the rows hold NOW, so scaling to 100 and then
        to 50 lands on 50. Process settings do not scale — a cook temperature
        is not an amount of anything — and a row that adds up to nothing has
        no factor that reaches the size, so it is left as it is.

        The round stays open and keeps its number: this is a change of size,
        not a regenerate. `pending_batch_total` (the stored name, unchanged)
        remembers the size, so a reopened window and the Results tab both
        still know what the bench weighed out. With no round open there is
        nothing to size and nothing to remember it by, so this does nothing.

        A size of nothing is not a size. None and anything at or below zero
        mean the same thing here — the round has no size of its own — and
        both leave the amounts alone: the box moved them once and there is no
        going back to what the model proposed.
        """
        if not self.pending_batch:
            return          # no round to size; nothing to remember it by
        size = None if batch_size is None else float(batch_size)
        if size is not None and size <= 0:
            size = None
        rows = self._batch_rows(self.pending_batch)
        if size is not None:
            rows = [self._batch_row(row['formulation'],
                                    self.scaled_recipe(row['recipe'], size),
                                    row.get('note'))
                    for row in rows]
        self.pending_batch = rows
        self.pending_batch_total = size
        self.save()

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
                # A regenerate keeps its own number; the counter still has to
                # know it is spent.
                self.pending_batch_no = int(batch_no)
                self._retire_batch_no(batch_no)
            elif self.pending_batch_no is None:
                self.pending_batch_no = self._issue_batch_no()
            self._date_pending_batch()
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
    #  Non-monotone active set: a row is FIXED when Lowest equals Highest
    #
    #  Standard EGBO grows the active set monotonically (S_1 <= S_2 <= ...),
    #  which means expert false positives accumulate and never leave: the
    #  expected active-set size tends to the full ambient dimension as the
    #  number of expert rounds grows, so adaptive EGBO degenerates towards
    #  vanilla BO. Allowing the expert to prune gives the active set a bounded
    #  steady state instead.
    #
    #  Pruning is not deletion, and as of 0.5.0 it is not a flag either: the
    #  active set is read off the allowed amounts. A row whose Lowest is its
    #  Highest is one number, so there is nothing for the search to choose;
    #  everything else is in S_r. That is the same domain restriction as
    #  before, said in the two boxes the user already types into:
    #    - the variable keeps its column in the encoded history, so every past
    #      observation stays in the GP (a recorded experiment is still a valid
    #      observation of f — it is only outside the current search domain);
    #    - the acquisition function is maximized over the varying set only;
    #    - widening the range again is free, which is what a non-monotone
    #      active set needs.
    # ------------------------------------------------------------------ #

    @staticmethod
    def is_fixed(var):
        """True for a row pinned at one amount: its Lowest is its Highest.

        A categorical variable has options rather than a range and can never
        be fixed this way."""
        if var.get('type') != 'continuous':
            return False
        return float(var['bounds'][0]) == float(var['bounds'][1])

    @staticmethod
    def has_formula(var):
        """True for a row that is worked out rather than searched: it holds
        a formula, or it is the balance of the batch size.

        Asked BEFORE is_fixed everywhere a row's part in the search is
        decided. A constant formula (an ingredient at 0.15 of the batch
        size) has a Lowest equal to its Highest and would otherwise read as
        a fixed row pinned inside the frame the search runs in. It is
        neither pinned nor searched: it has no column at all."""
        return bool(var.get('formula') or var.get('balance'))

    @staticmethod
    def _migrate_fixed(var):
        """A project written before 0.5.0 spelled a fixed row as a row marked
        inactive and pinned at `_frozen_at`. It becomes the range it was
        pinned at — Lowest and Highest both that amount — which is the same
        question asked in the two boxes the user already reads.

        The two old keys are read here and nowhere else, ever again: a file
        still carrying them opens, and is rewritten without them the first
        time it is saved. A row this cannot migrate — a categorical has
        options rather than a range, so there is no pair of numbers to write
        the pin into — keeps what it had rather than being quietly stripped
        of it."""
        if var.get('type', 'continuous') != 'continuous':
            return
        was_held = not var.pop('active', True)
        frozen = var.pop('_frozen_at', None)
        if not was_held:
            return
        lo, hi = float(var['bounds'][0]), float(var['bounds'][1])
        if frozen is None:
            # What _frozen_value worked out for a row held without a number
            # of its own: a process setting has no 'off', so it sat at its
            # Lowest; an ingredient sat at nothing.
            frozen = (float(var['_absent_value']) if '_absent_value' in var
                      else (lo if var.get('category') == 'process'
                            else min(max(0.0, lo), hi)))
        value = float(frozen)
        var['bounds'] = (value, value)

    def varying_variables(self):
        """Variables the search may move: everything neither worked out from
        the other rows nor pinned at one amount."""
        return [v for v in self.variables
                if not self.has_formula(v) and not self.is_fixed(v)]

    def fixed_variables(self):
        """Variables pinned at one amount, still carried in the history/GP.
        A formula row is not one: it is not pinned at anything, it is worked
        out afresh for every formulation."""
        return [v for v in self.variables
                if not self.has_formula(v) and self.is_fixed(v)]

    def _var_by_name(self, name):
        for var in self.variables:
            if var['name'] == name:
                return var
        raise ValueError(wording.no_variable_named(name))

    def _fixed_value(self, var):
        """The one amount a fixed variable takes in every formulation."""
        return float(var['bounds'][0])

    def fixed_at_text(self, var):
        """'20.00 g', '175 °C' — the one amount a fixed row is at, written in
        its own unit. A cook temperature is dialled in and an ingredient is
        weighed out, and a setting written as '175 g' priced it in grams."""
        unit = self._unit_of(var)
        value = self._fixed_value(var)
        if var.get('category') == 'process':
            return fmt_setting(value, unit)
        return fmt_amount(value, unit)

    # ------------------------------------------------------------------ #
    #  0.5.0 · formula rows
    #
    #  A formula row's amount is not searched, it is worked out: the row
    #  leaves the search vector altogether and its coefficients are folded
    #  into every limit that reads it. Substitution, not an equality band —
    #  three of this file's own mechanisms say why.
    #
    #    * the opening samples space-filling points and REJECTS what does
    #      not fit (_snap_to_total's note), and rejection against an
    #      equality is hopeless. A formula is an equality;
    #    * a band lets the search land half a percent off the formula, so
    #      the row would have to be corrected afterwards, and
    #      _snapped_if_it_still_fits exists because such a correction can
    #      break another limit. A correction that cannot be declined — the
    #      formula must hold — has nowhere to fall back to;
    #    * _check_constraints would have to pass a row it knows is wrong by
    #      the width of the band.
    #
    #  Substituted, the formula holds exactly at every point the app ever
    #  produces, and the column that left carries nothing the rest of the
    #  vector did not already say, so the GP loses nothing.
    #
    #  Order everywhere: decode the rows the search moves, snap onto the
    #  batch size, fill the formulas in, then check the limits.
    # ------------------------------------------------------------------ #

    def _batch_size(self):
        """What `batch size` is worth inside a formula: the project's
        default, or nothing while it has none."""
        total = getattr(self, 'formulation_total', None)
        return 0.0 if total is None else float(total)

    def _formula_rows(self):
        """Every row that is worked out rather than searched, in project
        order."""
        return [v for v in self.variables if self.has_formula(v)]

    def _balance_row(self):
        """The one row that takes whatever is left of the batch size, or
        None. One row at a time: set_balance is what enforces that."""
        return next((v for v in self.variables if v.get('balance')), None)

    def _formula_names(self, skip=None):
        """The names a formula on row `skip` may read: every other row of
        the project, ingredient or process setting."""
        return [v['name'] for v in self.variables if v['name'] != skip]

    def _formula_text(self, var):
        """The text a formula row's cell holds, as typed. A row given the
        balance without one reads as the balance."""
        return str(var.get('formula') or "") or "= " + wording.REST_TOKEN

    def _raw_form(self, var):
        """One row's formula as it was typed, before the rows it names are
        themselves written out.

        Read with `batch size` allowed whatever the project's default is
        now: the text went through parse_formula's own gate at save, and a
        default cleared afterwards must not turn every later question into
        a refusal. _batch_size answers 0 while there is none."""
        if var.get('balance'):
            return LinearForm(rest=True)
        return parse_formula(self._formula_text(var),
                             self._formula_names(var['name']), True)

    def _formula_order(self):
        """The formula rows in the order they can be worked out: every row
        after the rows it reads.

        Raises FormulaError naming the loop — joined by arrows, the chain
        the reader has to break — when a formula leads back to itself. The
        balance reads every other ingredient, so it is in the graph like
        any other row: a formula that names the balance row, and a balance
        row that must wait for it, is the same loop."""
        forms = {v['name']: self._raw_form(v) for v in self._formula_rows()}
        ingredients = [v['name'] for v in self.variables
                       if v.get('category', 'ingredient') == 'ingredient']
        order, done, open_rows, chain = [], set(), set(), []

        def reads(name):
            form = forms[name]
            named = ([n for n in ingredients if n != name] if form.rest
                     else form.names())
            return [n for n in named if n in forms]

        def visit(name):
            open_rows.add(name)
            chain.append(name)
            for other in reads(name):
                if other in open_rows:
                    loop = chain[chain.index(other):] + [other]
                    raise FormulaError(wording.formula_loop(" → ".join(loop)))
                if other not in done:
                    visit(other)
            chain.pop()
            open_rows.discard(name)
            done.add(name)
            order.append(name)

        for var in self._formula_rows():
            if var['name'] not in done:
                visit(var['name'])
        return order

    def _resolved_forms(self):
        """{name: LinearForm} for every formula row, each written out over
        the rows the search actually moves.

        One pass in _formula_order, so a formula that reads another formula
        reads the finished answer. Kept against the variable list it was
        worked out from: the forms are asked for once per candidate and
        once per limit, and re-reading every cell each time made the
        opening's own pool the slowest thing in the app."""
        rows = self._formula_rows()
        if not rows:
            return {}
        key = tuple((v['name'], str(v.get('formula') or ""),
                     bool(v.get('balance')),
                     v.get('category', 'ingredient'))
                    for v in self.variables)
        cached = getattr(self, '_forms_cache', None)
        if cached is not None and cached[0] == key:
            return cached[1]
        done = {}
        for name in self._formula_order():
            raw = self._raw_form(self._var_by_name(name))
            if raw.rest:
                form = LinearForm(batch=1.0)
                for var in self.variables:
                    if var.get('category', 'ingredient') != 'ingredient':
                        continue
                    if var['name'] == name:
                        continue
                    form = form - done.get(
                        var['name'], LinearForm(terms={var['name']: 1.0}))
            else:
                form = LinearForm(raw.const, raw.batch)
                for term, coeff in raw.terms.items():
                    form = form + done.get(
                        term, LinearForm(terms={term: 1.0})).scaled(coeff)
            done[name] = form
        self._forms_cache = (key, done)
        return done

    def _linear_form(self, name):
        """What this row's amount is, written over the rows the search
        moves: {name: 1} for a row typed by hand, the formula itself for a
        formula row, and the batch size less every other ingredient for the
        balance."""
        var = self._var_by_name(name)
        if not self.has_formula(var):
            return LinearForm(terms={name: 1.0})
        return self._resolved_forms()[name]

    def _form_value(self, form, recipe, batch=None):
        """One linear form, worked out over the amounts of one formulation.
        `batch` is what `batch size` reads, for the one caller that is
        writing a formulation for a size other than the project's."""
        size = self._batch_size() if batch is None else float(batch)
        value = form.const + form.batch * size
        for name, coeff in form.terms.items():
            value += coeff * _amount(recipe.get(name))
        return value

    def fill_formulas(self, recipe, batch=None):
        """`recipe` with every formula row written in.

        The rows are written in _formula_order, each from the rows it
        names, with the project's default batch size standing in for
        `batch size` — or `batch`, when a round is being written for a size
        of its own. The one door: every formulation the app hands out —
        decoded, snapped, or scaled — comes through here, so no screen and
        no sheet ever shows a formula row that does not hold."""
        forms = self._resolved_forms()
        if not forms:
            return recipe
        filled = dict(recipe)
        for name, form in forms.items():
            filled[name] = self._form_value(form, filled, batch)
        return filled

    def worked_out_captions(self):
        """One line per worked-out row, in project order — what the rule
        comes to, in numbers.

        Every rule shows its consequence: a formula is arithmetic the reader
        cannot do in their head over eight rows, so the line says what the
        other rows' allowed amounts leave this one, read against the default
        batch size. Empty for a project with no formula, which is what keeps
        the space under the grid blank until there is something to say.

        A row whose formula cannot be read gets no line rather than taking
        the screen down with it. validate_state refuses such a copy at the
        door, but this is drawn ABOVE the Save button, so a project that
        reached the grid by any other route has to leave the reader a way
        to edit their way out.

        The low end is what the app will ALLOW, not what the arithmetic
        reaches: no amount is weighed out below nothing, and the model and
        _check_constraints both hold a worked-out row at or above 0. The
        line said "between -5.00 and 55.00 g" where the app would never
        suggest one of those numbers, and a negative gram is not a
        consequence anybody can act on.

        And where the rule takes the row past the Lowest and Highest it was
        given, the line says so. Those two are dormant on a worked-out row
        — the rule decides the amount — but a rule that put Salt at 16 g
        over its own 3 g cap landed with nothing on screen but "Salt
        saved.", and the two numbers it overrode were replaced by a word.
        Information, not enforcement: nothing here refuses the save.
        """
        lines = []
        for var in self._formula_rows():
            name = var['name']
            try:
                low, high = self._achievable_range(
                    lambda n, row=name: 1.0 if n == row else 0.0)
            except (FormulaError, ValueError):
                continue
            low = max(low, 0.0)
            unit = self._unit_of(var)
            lines.append(wording.worked_out_caption(
                name, self._formula_text(var), f"{low:.2f}",
                join_unit(f"{high:.2f}", unit),
                self.batch_total_text(self.formulation_total),
                rest=bool(var.get('balance')),
                outside_text=self._outside_its_own_range(var, low, high)))
        return lines

    def _outside_its_own_range(self, var, low, high):
        """'0.00 to 3.00 g' when the rule takes this row outside the
        allowed amounts its own two cells still hold, else "".

        A row that never had a range of its own — one that arrived already
        worked out — has nothing to be outside of, and says nothing."""
        kept = tuple(float(b) for b in var['bounds'])
        if kept[0] == kept[1] == 0.0:
            return ""
        if kept[0] - 1e-9 <= low and high <= kept[1] + 1e-9:
            return ""
        unit = self._unit_of(var)
        return join_unit(f"{kept[0]:.2f} to {kept[1]:.2f}", unit)

    def _rule_names_a_setting(self, form, rows=None):
        """Why this rule cannot be saved for naming a process setting, or
        None.

        A rule may use ingredients and the batch size, and nothing else.
        The app refused a setting a rule of its own and then let a setting
        drive an ingredient's weight — grams of salt worked out from
        minutes of cooking, in two units that cannot be mixed. `rows` is
        the finished grid's names when there is one: a row this same save
        is adding has no variable yet, and a name the grid does not know is
        already refused by the parser.
        """
        by_name = self._by_name()
        for term in form.names():
            var = by_name.get(term)
            if var is None:
                continue
            if var.get('category', 'ingredient') != 'ingredient':
                return wording.rule_ingredients_only(var['name'])
        return None

    def worked_out_note(self):
        """The note the round sheets carry under the amounts, naming every
        worked-out row's own rule: 'Water is worked out: = rest. Weigh the
        amount printed.'

        The rule is on no sheet of the ROUND workbook — the Set-up sheet
        that carries the Rule column is in the All formulations download —
        so the old note told a bench holding the page that the amount came
        from a rule and gave it nowhere to see one. And the bench's real
        question is not where the number came from but whether they still
        weigh it, which the last sentence answers."""
        rows = self._formula_rows()
        if not rows:
            return ""
        return wording.worked_out_row_note(" ".join(
            wording.worked_out_row_rule(
                var['name'],
                wording.setup_sheet_formula_text(
                    self._formula_text(var),
                    rest=bool(var.get('balance'))))
            for var in rows))

    def batch_size_consequence(self):
        """What the row written = rest comes to at the default batch size,
        or "" when there is no such row, as one line for the box that has
        just moved that number.

        Amounts written in grams do not follow the batch size; a rest row
        does. Nothing said so, and a reader who changed 100 to 250 found a
        burger that was 86 % water with no comment but a deleted limit."""
        row = self._balance_row()
        if row is None or not self.has_formulation_total():
            return ""
        name = row['name']
        try:
            low, high = self._achievable_range(
                lambda n: 1.0 if n == name else 0.0)
        except (FormulaError, ValueError):
            return ""
        unit = self._unit_of(row)
        return wording.rest_row_takes_the_difference(
            name, f"{max(low, 0.0):.2f}", join_unit(f"{high:.2f}", unit),
            self.batch_total_text(self.formulation_total))

    def balance_row_name(self):
        """The name of the row written = rest, or None. The screen's own
        way of asking: a project with one has no largest batch size, and
        the floor it does have is said in that row's name."""
        row = self._balance_row()
        return None if row is None else row['name']

    def _formula_reads_refusal(self, name):
        """Why this row cannot be deleted while another row's formula reads
        it by name, or None.

        Both doors ask — an ingredient's and a process setting's — because
        a formula may read either, and a formula left naming a row the
        project no longer has refuses every later round instead of the one
        save that caused it. The balance reads whatever is left rather than
        any row by name, so it rebalances around a deletion and has nothing
        to say here. Force is no answer either: it is about amounts already
        recorded, not about a formula that would stop reading."""
        for var in self.variables:
            if (var['name'] == name or var.get('balance')
                    or not self.has_formula(var)):
                continue
            try:
                reads = self._raw_form(var).names()
            except FormulaError:
                continue
            if name in reads:
                return wording.formula_reads_this_row(name, var['name'])
        return None

    def _by_name(self):
        """{name: variable} for one pass of work. A formula's reach and its
        columns are asked for once per candidate of a 2048-point pool, and
        a scan of the variable list per term made that quadratic in the
        size of the project."""
        return {var['name']: var for var in self.variables}

    def _free_ingredients(self):
        """The ingredients the search really moves: neither worked out from
        the other rows nor pinned at one amount."""
        return [v for v in self.variables
                if v.get('category', 'ingredient') == 'ingredient'
                and not self.has_formula(v) and not self.is_fixed(v)]

    def _search_columns(self):
        """{name: column} for the rows the search vector holds. A formula
        row is not one of them — it left the vector — so a limit that reads
        it is written out over the rows it is worked out from instead."""
        columns, col = {}, 0
        for var in self.variables:
            if self.has_formula(var):
                continue
            if var['type'] == 'continuous':
                columns[var['name']] = col
                col += 1
            elif var['type'] == 'categorical':
                col += len(var['options'])
        return columns

    def _weighted_form(self, coeff_of):
        """Σ coeff_of(row) × that row's own form — the one place a limit
        stops naming rows and starts naming the columns the search has.
        With nothing worked out it is the limit itself, coefficient for
        coefficient, in project order."""
        form = LinearForm()
        for var in self.variables:
            if var['type'] != 'continuous':
                continue
            coeff = coeff_of(var['name'])
            if coeff == 0:
                continue
            form = form + self._linear_form(var['name']).scaled(coeff)
        return form

    def _form_reach(self, form, batch, by_name=None):
        """(lowest, highest) one linear form can come to while every row it
        names ranges over its allowed amounts. Handles negative
        coefficients."""
        by_name = self._by_name() if by_name is None else by_name
        lo = hi = form.const + form.batch * float(batch)
        for name, coeff in form.terms.items():
            if coeff == 0:
                continue
            var = by_name.get(name)
            if var is None or var['type'] != 'continuous':
                continue
            a = coeff * float(var['bounds'][0])
            b = coeff * float(var['bounds'][1])
            lo += min(a, b)
            hi += max(a, b)
        return lo, hi

    def _form_over_columns(self, form, columns, by_name=None):
        """(columns, coefficients, offset) — one linear form as the model
        is handed it, in the [0, 1] frame the search runs in. A fixed row
        is the same number at both ends, so it moves the offset rather than
        riding in the form; so do the constant and the batch size."""
        by_name = self._by_name() if by_name is None else by_name
        indices, coeffs = [], []
        offset = form.const + form.batch * self._batch_size()
        for name, coeff in form.terms.items():
            if coeff == 0:
                continue
            var = by_name.get(name)
            if var is None or var['type'] != 'continuous':
                continue
            v_min = float(var['bounds'][0])
            v_max = float(var['bounds'][1])
            if v_max != v_min and name in columns:
                indices.append(columns[name])
                coeffs.append(coeff * (v_max - v_min))
            offset += coeff * v_min
        return indices, coeffs, offset

    def _refuse_without_amounts(self):
        """A project whose recorded formulations have lost their amounts
        cannot have its history rebuilt, and changing which rows are
        searched rebuilds it. The same refusal deleting an ingredient
        gives, for the same reason."""
        if self.X_history and len(self.recipe_history) != len(self.X_history):
            raise ValueError(AMOUNTS_MISSING_DELETE_ERROR)

    def set_balance(self, name):
        """Give one row the balance of the batch size, or none (None).

        One row at a time: giving it to a second row takes it off the
        first, because two rows each taking whatever is left of the same
        number is not arithmetic anybody can do. Internal — the grid writes
        the text into a Formula cell and set_formula lands here."""
        for var in self.variables:
            wanted = name is not None and var['name'] == name
            if wanted or var.get('balance'):
                var['balance'] = wanted

    def set_formula(self, name, text):
        """Work this row's amount out from the others instead of searching
        it, from the text as typed. THE one door a formula is written
        through; the grid's Save comes through here too.

        The text is stored as typed, so the cell reads back the way it was
        written. Refused — with nothing written — for a formula that cannot
        be read, one that leads back to itself, and one no allowed amounts
        can ever make a real amount of. Those three questions are about ONE
        row, so a grid Save has already asked them over the finished grid
        and they are held back for the length of it (`_in_grid_apply`), the
        way _check_fixed_feasible and remove_ingredient hold back theirs:
        asking again half way through refuses a save for a state nothing
        ever sees.

        What is NOT held back is what a formula costs the project as a
        whole. The open round goes the way every other set-up change sends
        it; the history is re-encoded, because the row has left the search
        vector and every recorded formulation is one column shorter; and
        the total's own limit is re-asked, because a formula folds a whole
        row's coefficients into the sum and can put the default batch size
        out of reach. What that drops comes back here, the way
        add_ingredient and remove_ingredient hand theirs back, so the
        screen says it at the Save rather than leaving Generate to refuse.
        """
        var = self._var_by_name(name)
        text = str(text).strip()
        in_grid = getattr(self, '_in_grid_apply', False)
        if not in_grid:
            self._refuse_without_amounts()
            form = parse_formula(text, [v['name'] for v in self.variables],
                                 self.has_formulation_total())
            if name in form.names():
                raise FormulaError(wording.RULE_USES_ITS_OWN_ROW)
            trouble = self._rule_names_a_setting(form)
            if trouble:
                raise FormulaError(trouble)
            if form.rest and not self.has_formulation_total():
                raise FormulaError(wording.FORMULA_NEEDS_BATCH_SIZE)
            rest = form.rest
        else:
            rest = formula_is_rest(text)
        before = [(v, v.get('formula'), v.get('balance'))
                  for v in self.variables]

        def put_back():
            for row, formula, balance in before:
                row['formula'] = formula
                row['balance'] = balance
                if formula is None:
                    row.pop('formula')
                if balance is None:
                    row.pop('balance')

        var['formula'] = text
        if rest:
            self.set_balance(name)
        else:
            var['balance'] = False
        if not in_grid:
            try:
                self._formula_order()  # refuses a loop, naming the chain
                if self._form_reach(self._linear_form(name),
                                    self._batch_size())[1] < 0:
                    raise ValueError(wording.formula_below_zero(
                        name, self._unit_of(var)))
            except ValueError:
                put_back()
                raise
        return self._after_formula_written()

    def clear_formula(self, name):
        """Type this row's amounts by hand again. A no-op on a row that has
        no formula, so a rerun does not bump the file's mtime. The other
        half of the one door: rubbing a formula out moves the sum exactly
        as writing one does, and owes the same consequences."""
        var = self._var_by_name(name)
        if not self.has_formula(var):
            return []
        if not getattr(self, '_in_grid_apply', False):
            self._refuse_without_amounts()
        var['formula'] = ""
        var['balance'] = False
        return self._after_formula_written()

    def _after_formula_written(self):
        """What every formula write owes the project, in one place: the
        re-encoded history, the open round, the total's own limit, and the
        file. What the total dropped comes back, for the screen to say."""
        self._reencode_history()
        removed = self._sync_formulation_total()
        self._drop_pending_batch()
        self.save()
        return removed

    def _achievable_range(self, coeff_of):
        """Range of sum_i coeff_i * x_i attainable while every variable ranges
        over its allowed amounts. Handles negative coefficients.

        A fixed variable needs no special case: its Lowest is its Highest, so
        it adds the same number at both ends. Neither does a formula row:
        _weighted_form has already written it out over the rows it is
        worked out from, so what is read off the bounds here is only ever a
        row the search really moves."""
        return self._form_reach(self._weighted_form(coeff_of),
                                self._batch_size())

    def _achievable_property(self, metric):
        """The lowest and the highest one property can be per 100 g of the
        finished formulation, with every variable free within its allowed
        amounts (a fixed one has only the one).

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
            return self._achievable_range(self._property_coeff(metric, value))

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

    def _property_limit_refusal(self, constr):
        """Why this property limit cannot be met by any formulation the
        allowed amounts describe, or None while one can."""
        metric = constr['metric']
        per = self.per_amount_text()
        lo, hi = self._achievable_property(metric)
        if constr['min'] is not None and hi < constr['min']:
            return wording.limit_unreachable_above(metric, f"{hi:.4g} {per}")
        if constr['max'] is not None and lo > constr['max']:
            return wording.limit_unreachable_below(metric, f"{lo:.4g} {per}")
        return None

    def _quantity_limit_refusal(self, qc):
        """The same question of one amount limit. The batch size's own limit
        answers in its own words: it is one number the user typed, not a rule
        about a list, and naming its eight ingredients helped nobody."""
        names = set(qc['ingredients'])
        lo, hi = self._achievable_range(lambda n: 1.0 if n in names else 0.0)
        broken = ((qc['min'] is not None and hi < qc['min'])
                  or (qc['max'] is not None and lo > qc['max']))
        if qc.get('source') == 'formulation_total':
            if broken:
                return wording.fixing_breaks_the_total(
                    self.batch_total_text(self.formulation_total))
            return None
        label = " + ".join(qc['ingredients'])
        if qc['min'] is not None and hi < qc['min']:
            return wording.limit_unreachable_above(label, f"{hi:.4g}")
        if qc['max'] is not None and lo > qc['max']:
            return wording.limit_pinned_amounts_exceed(label, f"{lo:.4g}")
        return None

    @staticmethod
    def _limit_key(qc):
        """One amount limit's identity, stable across an edit so a BEFORE and
        an AFTER can be lined up limit by limit. Not its index: a save that
        deletes an ingredient prunes the list and every index after it
        shifts."""
        if qc.get('source') == 'formulation_total':
            return ('total',)
        return ('quantity', tuple(qc['ingredients']))

    def _limit_refusals(self):
        """{key: sentence} for every limit the allowed amounts, as they
        stand, cannot meet.

        Per limit rather than first-one-wins, because the question every
        caller really asks is "did THIS save break something": a limit
        stranded by a narrowing last week is not this save's doing, and
        refusing for it would leave the reader with no way to edit their way
        back out."""
        refusals = {}
        for constr in self.constraints:
            message = self._property_limit_refusal(constr)
            if message:
                refusals[('property', constr['metric'])] = message
        for qc in getattr(self, 'quantity_constraints', []):
            message = self._quantity_limit_refusal(qc)
            if message:
                refusals[self._limit_key(qc)] = message
        return refusals

    def _broken_by_this_save(self, before, after, fixed_names=()):
        """The one sentence a save owes when it breaks a limit that was fine
        before it, or None. `fixed_names` are the rows this save pinned at
        one amount, named at the end so the reader knows which of eight rows
        the refusal is about."""
        newly = [message for key, message in after.items() if key not in before]
        if not newly:
            return None
        if not fixed_names:
            return newly[0]
        return newly[0] + " " + wording.fixed_rows_tail(
            number_list(list(fixed_names)))

    def _check_fixed_feasible(self, name, min_val, max_val, category):
        """Refuse a save that would fix `name` at one amount where nothing
        could then be made.

        Asked before the write, with the amounts the form is proposing put in
        place for the length of the question and then taken out again: the
        refusal has to name the amounts the user just typed, and a project
        left half-written by a refusal is worse than the refusal.

        Only a save that FIXES a row is asked, and only a limit that passes
        BEFORE and fails AFTER earns the refusal. Narrowing a range is the
        user's own business and has always been allowed to strand a limit; a
        save blamed for one it did not break is a save with no way back.
        """
        if float(min_val) != float(max_val):
            return
        # A grid Save asks this question ONCE, over the finished grid, and
        # compares the answer with the one the project already gives (see
        # _grid_feasibility_error). Asking it again per row would refuse a
        # save for a half-applied state nothing ever sees.
        if getattr(self, '_in_grid_apply', False):
            return
        before = self._limit_refusals()
        index = next((i for i, v in enumerate(self.variables)
                      if v['name'] == name), None)
        added = index is None
        if added:
            # Appended, and removed again BY INDEX: two rows of a project can
            # hold equal dicts, and list.remove would take the first of them.
            index = len(self.variables)
            self.variables.append({
                'name': name, 'type': 'continuous',
                'bounds': (float(min_val), float(max_val)),
                'category': category})
            was = None
        else:
            was = self.variables[index]['bounds']
            self.variables[index]['bounds'] = (float(min_val), float(max_val))
        try:
            after = self._limit_refusals()
        finally:
            if added:
                self.variables.pop(index)
            else:
                self.variables[index]['bounds'] = was
        message = self._broken_by_this_save(before, after, [name])
        if message is not None:
            raise ValueError(message)

    def _get_fixed_features(self):
        """{column: normalized_value} for the fixed columns, in the [0,1]^d frame
        that `ask` hands to optimize_acqf. Returns {} when everything varies, so
        the standard monotone behavior is byte-identical.

        The frame is _search_bounds, not the row's own Lowest and Highest: a
        fixed row's are the same number, and the frame it is placed in has
        been widened so it has a width at all."""
        fixed = {}
        spans = self._search_bounds()
        col = 0
        for var in self.variables:
            if self.has_formula(var):
                continue
            if var['type'] == 'continuous':
                if self.is_fixed(var):
                    lo, hi = spans[col]
                    span = hi - lo
                    z = 0.0 if span <= 0 else (self._fixed_value(var) - lo) / span
                    fixed[col] = float(min(max(z, 0.0), 1.0))
                col += 1
            elif var['type'] == 'categorical':
                col += len(var['options'])
        return fixed

    def _check_rename(self, name, new_name):
        """The stripped name `rename_variable` would give this row, or a
        ValueError saying why it cannot have it. Separate from the rename
        itself so a caller with other writes to make can ask first and refuse
        the whole edit, rather than renaming and then failing."""
        var = self._var_by_name(name)
        new_name = str(new_name).strip()
        if not new_name:
            raise ValueError(wording.NAME_REQUIRED_ERROR)
        if new_name == name:
            return name
        if is_reserved_name(new_name):
            raise ValueError(_reserved_name_message(new_name))
        self._name_is_free(new_name, skip=var)
        return new_name

    def rename_variable(self, name, new_name):
        """Give one ingredient or process setting a different name, keeping
        everything recorded under the old one.

        A name is a KEY here, not a label: the amounts of every formulation
        are stored against it, the open batch and every not-scored row hold
        it, an amount limit lists it, the total's own limit lists it, and an
        ingredient's property values are filed under it. Renaming rewrites
        all six and re-encodes the history, so the project after the rename
        holds exactly what it held before, under the new name.

        Refused for a name that is empty, reserved, or already the name of
        something else in this project — the same refusals adding one gives,
        in the same words. Nothing is written until every one of them has
        passed, and _check_rename answers the same question without writing
        anything, so a screen can refuse before it starts.
        """
        var = self._var_by_name(name)
        new_name = self._check_rename(name, new_name)
        if new_name == name:
            return

        # Read before the rename: a formula names the rows the way the
        # project spelled them when it was typed, longest name first, so a
        # rename of Cream leaves a Cream cheese in the same cell alone.
        spelled = [v['name'] for v in self.variables]
        for row in self.variables:
            if row.get('formula'):
                row['formula'] = _renamed_in_formula(
                    row['formula'], {name: new_name}, spelled)
        var['name'] = new_name
        for recipe in self.recipe_history:
            if name in recipe:
                recipe[new_name] = recipe.pop(name)
        for row in (self.pending_batch or []):
            recipe = row.get('recipe', row) if isinstance(row, dict) else row
            if isinstance(recipe, dict) and name in recipe:
                recipe[new_name] = recipe.pop(name)
        for row in self.skipped:
            recipe = row.get('recipe') or {}
            if name in recipe:
                recipe[new_name] = recipe.pop(name)
        for qc in getattr(self, 'quantity_constraints', []):
            # The total's own limit is in here too: it lists every ingredient
            # by name, so it is rewritten with the rest rather than dropped
            # and rebuilt.
            qc['ingredients'] = [new_name if n == name else n
                                 for n in qc['ingredients']]
        if name in self.ingredient_properties:
            self.ingredient_properties[new_name] = \
                self.ingredient_properties.pop(name)
        # The lot numbers are filed per round, per ingredient, under the
        # name as well: left alone, the Lots sheet printed a name the
        # project no longer has.
        for written in (getattr(self, 'lots', None) or {}).values():
            if isinstance(written, dict) and name in written:
                written[new_name] = written.pop(name)
        # The columns are in the same order and hold the same numbers, but
        # encoding reads the recipes by name: a history left keyed to the old
        # name would encode every amount as absent.
        self._reencode_history()
        self.save()

    def remove_ingredient(self, name, force=False):
        """Permanently delete an ingredient and drop its column from the history.

        Refuses by default if the ingredient was ever used at a nonzero amount,
        because dropping its column silently rewrites those experiments into
        recipes that were never run. Prefer fixing it (Lowest = Highest),
        which keeps the data. force=True deletes anyway and discards that
        information.
        """
        var = self._var_by_name(name)
        if var.get('category', 'ingredient') != 'ingredient':
            raise ValueError(
                wording.delete_the_process_setting_instead(name))
        trouble = self._ingredient_delete_refusal(name, force)
        if trouble:
            raise ValueError(trouble)

        remaining = [v for v in self.variables if v['name'] != name]
        # Held back for the length of a grid Save: the finished grid was
        # asked this question as a whole, and a row half way through it is
        # not a state the project ever reaches.
        if (not getattr(self, '_in_grid_apply', False)
                and not any(not self.is_fixed(v) and not self.has_formula(v)
                            for v in remaining)):
            raise ValueError(LAST_VARYING_ROW_ERROR)

        self.variables = remaining
        self.ingredient_properties.pop(name, None)
        for written in (getattr(self, 'lots', None) or {}).values():
            if isinstance(written, dict):
                written.pop(name, None)
        for recipe in self.recipe_history:
            recipe.pop(name, None)

        kept = []
        for qc in getattr(self, 'quantity_constraints', []):
            qc['ingredients'] = [n for n in qc['ingredients'] if n != name]
            if qc['ingredients']:
                kept.append(qc)
        self.quantity_constraints = kept

        self._reencode_history()
        # The total is over every ingredient, and there is one fewer now. It
        # is handed back the way add_ingredient hands back what a unit change
        # emptied: the screen owes the same one-line notice either way.
        removed = self._sync_formulation_total()
        self._drop_pending_batch()
        self.save()
        return removed

    # ------------------------------------------------------------------ #
    #  0.5.0 · the two editable grids
    #
    #  Tab 1 has no add form, no control row and no per-row editor: the
    #  ingredients and the measurements are typed where they are read, and
    #  one `Save changes` writes the lot. That work lives here rather than
    #  on the screen for two reasons. The difference between what is on the
    #  grid and what is in the project is arithmetic over the model's own
    #  objects; and `streamlit.testing`'s AppTest cannot click a cell, so
    #  this is the layer the behaviour can be pinned at.
    #
    #  The contract both halves keep:
    #    * every row is validated FIRST and nothing is written until they
    #      all pass, so a refusal leaves the project exactly as it was;
    #    * the fixed-feasibility question is asked ONCE, over the finished
    #      grid, and blames this save only for a limit it NEWLY breaks;
    #    * the writes go through the same paths the old form used —
    #      add_ingredient, add_process_parameter, rename_variable,
    #      remove_ingredient, set_variable_unit — so every consequence (the
    #      default batch size kept or dropped, limits pruned, the open round
    #      discarded, results kept) fires exactly once and is said once.
    # ------------------------------------------------------------------ #

    def grid_variables(self):
        """Ingredients first, then process settings, each in the order they
        were added. The grid and the Set-up sheet read the same way down."""
        return self._ingredients() + self._process_settings()

    def ingredient_grid_frame(self):
        """What the ingredients grid opens holding.

        The hidden `_id` column is the row's identity: it carries the name
        the row is filed under TODAY, so a name typed over it is a rename of
        that row rather than a new row beside a deleted one. A row typed on
        the empty line at the bottom comes back without one, which is what
        makes it an addition.

        Baseline is a column only once results exist (spec 1.1): it is the
        value every formulation already made is read at, and a project with
        nothing recorded has nothing to read.

        Formula is last and always there: one column for one idea, and the
        idea is at the end of the row because it is the answer to the two
        columns before it. A row that carries one has no Lowest and no
        Highest of its own — both cells read `worked out` instead, which is
        why they are text and not numbers.
        """
        columns = [GRID_ID, wording.NAME_LABEL, wording.TYPE_LABEL,
                   wording.LOWEST_LABEL, wording.HIGHEST_LABEL,
                   wording.UNIT_LABEL, wording.VENDOR_LABEL,
                   wording.SKU_LABEL]
        if self.X_history:
            columns.append(wording.BASELINE_LABEL)
        columns.append(wording.FORMULA_LABEL)
        data = []
        for var in self.grid_variables():
            ingredient = var.get('category', 'ingredient') == 'ingredient'
            low, high = self._range_cell(var)
            row = {
                GRID_ID: var['name'],
                wording.NAME_LABEL: var['name'],
                wording.TYPE_LABEL: (wording.KIND_INGREDIENT if ingredient
                                     else wording.KIND_SETTING),
                wording.LOWEST_LABEL: low,
                wording.HIGHEST_LABEL: high,
                wording.UNIT_LABEL: self.unit_of(var['name']) or "",
                wording.VENDOR_LABEL: str(var.get('vendor', "") or ""),
                wording.SKU_LABEL: str(var.get('sku', "") or ""),
                wording.FORMULA_LABEL: (self._formula_text(var)
                                        if self.has_formula(var) else ""),
            }
            if self.X_history:
                baseline = var.get('_absent_value')
                row[wording.BASELINE_LABEL] = (None if baseline is None
                                               else float(baseline))
            data.append(row)
        return _grid_frame(data, columns)

    def _range_cell(self, var):
        """(Lowest, Highest) as the grid holds them: the word for a row that
        is worked out, and two-decimal text for every other.

        Text, not numbers, and this is the one reason why. A row with a
        rule has no range of its own to show — showing the numbers it
        happens to still carry would invite the reader to type into them —
        and `st.data_editor` will not put a word in a number column. A fixed
        row keeps showing the same number twice, as wave 1 left it.

        The word goes in ONE of the two cells. Both of them said it, which
        is one word doing one job twice on two cells side by side; Lowest
        is left blank and Highest carries the mark, so the pair reads as
        one fact about the row rather than two."""
        if self.has_formula(var):
            return "", wording.WORKED_OUT
        return (f"{float(var['bounds'][0]):.2f}",
                f"{float(var['bounds'][1]):.2f}")

    def measurement_grid_frame(self):
        """What the measurements grid opens holding, most important first —
        the order every screen lists them in. Share of score is the column
        that is typed into; the importance behind it is derived from that and
        is on no screen any more (spec 1.3)."""
        columns = [GRID_ID, wording.MEASUREMENT_COLUMN, wording.GOAL_LABEL,
                   wording.TARGET_LABEL, wording.LOWEST_MEASURABLE_LABEL,
                   wording.HIGHEST_MEASURABLE_LABEL, wording.UNIT_LABEL,
                   wording.SHARE_COLUMN]
        shares = self.share_percents()
        data = [{
            GRID_ID: obj['name'],
            wording.MEASUREMENT_COLUMN: obj['name'],
            wording.GOAL_LABEL: wording.GOAL_LABELS.get(obj['goal'],
                                                        obj['goal']),
            wording.TARGET_LABEL: (None if obj.get('target') is None
                                   else float(obj['target'])),
            wording.LOWEST_MEASURABLE_LABEL: float(obj.get('min_val', 0.0)),
            wording.HIGHEST_MEASURABLE_LABEL: float(obj.get('max_val', 10.0)),
            wording.UNIT_LABEL: str(obj.get('unit', "") or ""),
            wording.SHARE_COLUMN: float(shares.get(obj['name'], 0)),
        } for obj in self.measurements_by_importance()]
        return _grid_frame(data, columns)

    def ingredient_grid_deletions(self, frame):
        """The rows the grid has taken out: every variable whose `_id` is no
        longer anywhere on it. Asked by the screen BEFORE Save, because a
        deletion is confirmed by name first."""
        kept = {i for i in _grid_ids(frame) if i}
        return [v['name'] for v in self.grid_variables()
                if v['name'] not in kept]

    def measurement_grid_deletions(self, frame):
        kept = {i for i in _grid_ids(frame) if i}
        return [o['name'] for o in self.measurements_by_importance()
                if o['name'] not in kept]

    # -- the ingredients grid ------------------------------------------- #

    def apply_ingredient_grid(self, frame, force=()):
        """Write the ingredients grid to the project.

        Returns `(errors, messages)`. `errors` is a list of `(row, message)`
        — the row being the number the grid shows down its left edge, or
        None for a refusal about the grid as a whole — and while it is not
        empty NOTHING has been written. `messages` is a list of
        `(kind, sentence)` for the screen to flash, one per consequence.

        `force` names the ingredients the reader has ticked to delete even
        though formulations used them.
        """
        errors, plan = self._plan_ingredient_grid(frame, force)
        if errors:
            return errors, []
        return [], self._apply_ingredient_plan(plan, force)

    def _plan_ingredient_grid(self, frame, force=()):
        """Read the grid, refuse everything that cannot be saved, and hand
        back what to do. Nothing here writes."""
        errors, rows = [], []
        by_id = {v['name']: v for v in self.variables}
        ids_used = set()
        # The names the finished grid will carry, and what this save is
        # renaming — both read before any row is, so a formula may name a
        # row this same save adds, and an untouched one may still spell a
        # row it renames.
        names, renames, kept = [], {}, set()
        for _, row in _grid_rows(frame):
            if _row_is_blank(row):
                continue
            was, now = _text_cell(row, GRID_ID), _text_cell(row,
                                                            wording.NAME_LABEL)
            names.append(now)
            if was:
                kept.add(was)
            if was and now and was != now:
                renames[was] = now
        # A row another row's rule reads cannot go, and that is said FIRST —
        # before a single row is read. The per-row parse would otherwise get
        # there first and say "there is no ingredient called Salt" against
        # the rule's own row, which is true of the name list it is asked
        # against and flatly untrue of the grid the reader is looking at.
        # The same helper the process-setting door has always used.
        reads = [(None, self._formula_reads_refusal(v['name']))
                 for v in self.grid_variables() if v['name'] not in kept]
        reads = [pair for pair in reads if pair[1]]
        if reads:
            return reads, None
        for row_no, row in _grid_rows(frame):
            if _row_is_blank(row):
                # The empty line at the bottom of a dynamic grid, clicked and
                # then left alone. Not an addition, and not an error.
                continue
            spec, trouble = self._read_ingredient_row(row, by_id, ids_used,
                                                      names, renames)
            if trouble:
                errors.append((row_no, trouble))
                continue
            rows.append((row_no, spec))
        if errors:
            return errors, None

        errors += _duplicate_name_errors(rows)
        # A name may clash with something that is not on this grid at all —
        # a measurement, a property. The VARIABLES are not asked: a name
        # another row is giving up in this same save is free by the time the
        # save lands, and the pass above is what catches a real collision.
        for row_no, spec in rows:
            try:
                self._name_is_free_of_measurements(spec['name'])
                self._name_is_free_of_properties(spec['name'])
            except ValueError as e:
                errors.append((row_no, str(e)))
        if errors:
            return sorted(errors, key=lambda e: e[0]), None

        deleted = [v['name'] for v in self.grid_variables()
                   if v['name'] not in ids_used]
        errors += self._check_grid_deletions(deleted, rows, force)
        # A row that ARRIVES is one more column in the search vector, so
        # the history has to be rebuilt to hold it — and a project whose
        # recorded amounts have gone cannot be rebuilt. The same gate a
        # deletion and a formula cell are asked, asked here too: without
        # it add_ingredient raised out of the middle of the save, past the
        # grid's promise that nothing is written until every row passes,
        # and reached the browser as a traceback.
        if (self.X_history and len(self.recipe_history) != len(self.X_history)
                and any(spec['var'] is None for _, spec in rows)):
            errors += [(row_no, wording.CANNOT_ADD_WITHOUT_AMOUNTS)
                       for row_no, spec in rows if spec['var'] is None]
        if errors:
            return errors, None

        # The order the rows that STAY are written in. It is not cosmetic:
        # a rename onto a name another row is still wearing is refused by
        # the model, so the renames have to go in an order that frees each
        # name before it is taken.
        order, stuck = _rename_order(rows)
        if stuck is not None:
            errors.append((stuck[0], _name_taken_message(
                stuck[1]['name'], stuck[1]['category'])))
            return errors, None

        errors += self._formula_grid_errors(rows, deleted)
        if errors:
            return errors, None

        trouble = self._grid_feasibility_error(rows, deleted)
        if trouble:
            return [(None, trouble)], None
        return [], {'rows': rows, 'deleted': deleted, 'rename_order': order}

    def _formula_grid_errors(self, rows, deleted=()):
        """What the formulas on the finished grid cannot be, asked once over
        the grid as a whole.

        Three questions no single cell can answer. Two rows each taking
        whatever is left of the same number is not arithmetic anybody can
        do; a formula that leads back to itself is a loop between cells, and
        the chain is what the reader has to break; and a formula no allowed
        amounts can ever make a real amount of is refused at the save that
        strands it rather than at the Generate that cannot answer it — which
        is why it is asked here, over the rows as they will be, and not only
        at the one door that writes a formula.
        """
        errors = []
        balance = [(row_no, spec) for row_no, spec in rows if spec['balance']]
        if len(balance) > 1:
            errors.append((None, wording.one_balance_only(
                number_list([spec['name'] for _, spec in balance]),
                many=len(balance) > 2)))
        # Changing which rows are searched rebuilds the history, and a
        # project whose recorded amounts have gone cannot be rebuilt. Either
        # way round: giving a row a formula takes a column out of the
        # history, and rubbing one out puts a column back. A row that
        # arrives with no formula changes nothing about which rows are
        # searched and owes this nothing.
        if any(_formula_cell_moved(spec) for _, spec in rows):
            if self.X_history and (
                    len(self.recipe_history) != len(self.X_history)):
                errors.append((None, AMOUNTS_MISSING_DELETE_ERROR))
        if errors:
            return errors
        with self._as_proposed(rows, deleted):
            try:
                self._formula_order()
            except FormulaError as e:
                return [(None, str(e))]
            for row_no, spec in rows:
                if not spec['formula']:
                    continue
                reach = self._form_reach(self._linear_form(spec['name']),
                                         self._batch_size())
                if reach[1] < 0:
                    errors.append((row_no, wording.formula_below_zero(
                        spec['name'], spec['unit'])))
        return errors

    @contextlib.contextmanager
    def _as_proposed(self, rows, deleted=()):
        """The project with the finished grid in place, and put back exactly
        as it was afterwards. Nothing is written: this is how a question
        about the grid AS A WHOLE — a loop, a formula that can never be a
        real amount — is asked before the first row is saved."""
        variables = self.variables
        limits = self.quantity_constraints
        properties = self.ingredient_properties
        renamed = _renames(rows)
        try:
            self.variables = [_proposed_variable(spec) for _, spec in rows]
            self.ingredient_properties = _properties_over(properties, renamed,
                                                          deleted)
            self.quantity_constraints = _limits_over(
                limits, [v['name'] for v in self.variables
                         if v.get('category', 'ingredient') == 'ingredient'],
                renamed, deleted)
            yield
        finally:
            self.variables = variables
            self.quantity_constraints = limits
            self.ingredient_properties = properties

    def ingredient_grid_retires_round(self, frame, force=()):
        """The open round's number when saving this grid would take it away,
        else None. Nothing is written to find out.

        A screen has to be able to ASK before it saves — the round the save
        retires holds formulations the bench may already have made, and one
        of them may be the reader's own, typed in by hand. The answer is
        read off the same plan the save itself runs: a row that goes, a row
        that arrives, or a row whose allowed amounts, type or baseline move
        ('other'). A rename or a corrected unit, vendor or SKU does not
        retire it, and neither does a save that will be refused.
        """
        if self.pending_batch_no is None:
            return None
        errors, plan = self._plan_ingredient_grid(frame, force)
        if errors or plan is None:
            return None                  # nothing is going to be written
        if plan['deleted']:
            return self.pending_batch_no
        for _, spec in plan['rows']:
            var = spec['var']
            if var is None or 'other' in _what_moved(
                    self._row_state(var),
                    self._proposed_row_state(var, spec)):
                return self.pending_batch_no
        return None

    def _read_ingredient_row(self, row, by_id, ids_used, names=None,
                             renames=None):
        """One row of the grid as a plain dict, or (None, why it cannot be
        saved). Every refusal here is about this row on its own.

        `names` are the names the FINISHED grid will carry, so a formula may
        read a row this same save is adding or renaming. Without them a
        formula naming a row typed on the line above would be refused for a
        row the reader can see. `renames` is what this save is renaming: a
        cell nobody typed into still spells the row the way the project did
        when it was written, and is rewritten before it is read."""
        row_id = _text_cell(row, GRID_ID)
        var = by_id.get(row_id)
        if var is not None:
            ids_used.add(row_id)
        name = _text_cell(row, wording.NAME_LABEL)
        if not name:
            return None, wording.NAME_REQUIRED_ERROR
        if is_reserved_name(name):
            return None, _reserved_name_message(name)
        kind = _text_cell(row, wording.TYPE_LABEL) or wording.KIND_INGREDIENT
        category = 'process' if kind == wording.KIND_SETTING else 'ingredient'
        # The formula first, because it decides whether the two cells after
        # it are read at all: a row that is worked out has no Lowest and no
        # Highest of its own.
        typed = _text_cell(row, wording.FORMULA_LABEL)
        formula = typed
        if (typed and renames and var is not None
                and typed == str(var.get('formula') or "")):
            # Untouched: the reader renamed a row this formula names and
            # left the formula alone, which is the rename meaning exactly
            # what it says. The cell follows the name.
            formula = _formula_after_renames(
                typed, renames, [v['name'] for v in self.variables])
        form = None
        if formula:
            if category == 'process':
                # It works — the setting leaves the search vector and is
                # worked out like any other row — but a setting is in no
                # sum, so there is nothing for it to be the rest of and
                # nothing the grid's own captions could say about it.
                return None, wording.only_an_ingredient_has(
                    wording.FORMULA_LABEL)
            # The row's OWN name is in the list it is read against, so a
            # cell naming it is refused for what it is rather than for
            # naming an ingredient that is right there on the grid.
            available = list(self._formula_names(name)
                             if names is None else names)
            try:
                form = parse_formula(formula, available,
                                     self.has_formulation_total())
            except FormulaError as e:
                return None, str(e)
            if name in form.names():
                return None, wording.RULE_USES_ITS_OWN_ROW
            trouble = self._rule_names_a_setting(form, rows=names)
            if trouble:
                return None, trouble
            if form.rest and not self.has_formulation_total():
                # The file door (_check_file_formulas) is deliberately
                # looser about this one: a file is loaded before a size can
                # be set at all — there are no ingredients to reach one —
                # so it lets '= rest' in and the reader sets the size
                # after. The grid is the other way round: by the time
                # anyone types in it the project can have a size, and a
                # rest row without one has nothing to be the rest OF.
                return None, wording.FORMULA_NEEDS_BATCH_SIZE
        low, high, range_ok = _range_from_cells(row)
        kept = None if var is None else tuple(float(b) for b in var['bounds'])
        if formula:
            # Not read at all, whatever they hold: the row's amount is its
            # formula, and the two cells are rewritten to the word on the
            # next render. It keeps the range it had, so clearing the
            # formula gives the row its amounts back.
            low, high = (0.0, 0.0) if kept is None else kept
        else:
            if not range_ok:
                return None, wording.NUMBER_REQUIRED_ERROR
            if kept is not None and self.has_formula(var):
                # The rule has just been rubbed out, and the two cells still
                # hold what the app wrote into them — the word in Highest,
                # nothing in Lowest. That is not the reader failing to type
                # a number: it is the row coming back to the amounts it had
                # before it was worked out.
                if low is None and _text_cell(
                        row, wording.LOWEST_LABEL) in ("", wording.WORKED_OUT):
                    low = kept[0]
                if high is None and _text_cell(
                        row, wording.HIGHEST_LABEL) in ("", wording.WORKED_OUT):
                    high = kept[1]
            if low is None or high is None:
                return None, wording.NUMBER_REQUIRED_ERROR
            # Equal is allowed, and is how a row is FIXED at one amount.
            # Only Lowest ABOVE Highest is a range with nothing in it.
            if low > high:
                return None, LOWEST_ABOVE_HIGHEST_ERROR
        unit = _text_cell(row, wording.UNIT_LABEL)
        if category == 'ingredient' and not unit:
            # An ingredient's amounts are added up, averaged and printed. A
            # blank cell there is not "no unit", it is a sum of nothing. A
            # process setting may have none: a mixer speed of 3 is a 3.
            return None, wording.UNIT_REQUIRED_ERROR
        vendor = _text_cell(row, wording.VENDOR_LABEL)
        sku = _text_cell(row, wording.SKU_LABEL)
        if category == 'process' and (vendor or sku):
            return None, wording.only_an_ingredient_has(
                wording.VENDOR_LABEL if vendor else wording.SKU_LABEL)
        baseline, baseline_ok = _number_cell(row, wording.BASELINE_LABEL)
        if not baseline_ok:
            return None, wording.NUMBER_REQUIRED_ERROR
        if baseline is not None and category == 'ingredient':
            return None, wording.only_a_setting_has(wording.BASELINE_LABEL)
        if (var is not None and self.X_history
                and var.get('category', 'ingredient') != category):
            return None, wording.TYPE_LOCKED_ERROR
        if var is None and category == 'process' and self.X_history and (
                baseline is None):
            return None, wording.ADD_BASELINE_ERROR
        # A baseline the reader has MOVED is a fact about bakes already
        # done, and has to sit inside the range those bakes are read
        # against. One they have left alone does not: fixing a setting at
        # 190 is a decision about the next round, and refusing it because a
        # past bake ran at 175 refuses something they are entitled to ask
        # for (see add_process_parameter).
        stored = None if var is None else var.get('_absent_value')
        moved = (baseline is not None
                 and (stored is None or float(stored) != baseline))
        if moved and category == 'process' and not (low <= baseline <= high):
            return None, _baseline_outside_message(baseline, low, high)
        return {
            'var': var, 'id': row_id if var is not None else None,
            'name': name, 'category': category, 'low': low, 'high': high,
            'unit': unit, 'vendor': vendor, 'sku': sku, 'baseline': baseline,
            'baseline_moved': moved, 'formula': formula,
            # What the CELL holds, before the renames were written into it.
            # A rename is not an edit: the row keeps the open round, and the
            # history is not rebuilt for a spelling.
            'formula_typed': typed,
            'balance': bool(form is not None and form.rest),
        }, None

    def _check_grid_deletions(self, deleted, rows, force):
        """Refuse a deletion before anything is written, in the words
        remove_ingredient would have used after the fact.

        A row another row's rule reads is refused earlier still, in
        _plan_ingredient_grid, before the rows are read at all — and for
        every category, because a rule may read a process setting too."""
        errors = []
        for name in deleted:
            if self._var_by_name(name).get('category',
                                           'ingredient') != 'ingredient':
                continue
            trouble = self._ingredient_delete_refusal(name, name in force)
            if trouble:
                errors.append((None, trouble))
        if deleted and not any(spec['low'] < spec['high']
                               for _, spec in rows if not spec['formula']):
            errors.append((None, LAST_VARYING_ROW_ERROR))
        return errors

    def _ingredient_delete_refusal(self, name, force):
        """Why this ingredient cannot be deleted, or None. The two questions
        remove_ingredient asks, asked without writing — so the grid can
        refuse the whole save rather than half of it."""
        if self.X_history and len(self.recipe_history) != len(self.X_history):
            return AMOUNTS_MISSING_DELETE_ERROR
        trouble = self._formula_reads_refusal(name)
        if trouble:
            return trouble
        used = [i for i, r in enumerate(self.recipe_history)
                if float(r.get(name, 0.0)) != 0.0]
        if used and not force:
            # Name the formulations, not row indexes: a formulation number is
            # what the reader wrote on the sheet, and #0 is not theirs.
            numbers = [int(self.formulation_ids[i])
                       if i < len(self.formulation_ids) else i + 1
                       for i in used]
            return _used_ingredient_message(name, numbers)
        return None

    def _grid_feasibility_error(self, rows, deleted=()):
        """The fixed-feasibility question, asked ONCE over the finished grid.

        Two rulings live in these lines. It is asked once rather than per
        row, because a grid is saved whole and a row half way through it is
        not a state the project ever sits in. And it compares BEFORE with
        AFTER, limit by limit, through the same `_broken_by_this_save` the
        single-row door uses: a limit that was already impossible — after a
        range narrowed last week, which has always been allowed — is not this
        save's doing, and refusing the save for it would leave the reader
        with no way to edit their way back out.
        """
        # A worked-out row is neither pinned nor searched: its Lowest and
        # Highest are not enforced at all, so it is no more fixed than the
        # formula that fills it in.
        fixed = [spec['name'] for _, spec in rows
                 if not spec['formula'] and spec['low'] == spec['high']]
        if not fixed:
            return None
        return self._broken_by_this_save(
            self._limit_refusals(), self._refusals_with(rows, deleted), fixed)

    def _refusals_with(self, rows, deleted=()):
        """The same question with the finished grid in place, and the project
        put back exactly as it was afterwards.

        Everything a limit reads through a NAME moves with the rows: the
        variables, the ingredients each amount limit lists, and the property
        figures filed per ingredient. Leaving the figures behind made a
        renamed row read as having none, which refused a save that was
        perfectly legal."""
        with self._as_proposed(rows, deleted):
            return self._limit_refusals()

    def _apply_ingredient_plan(self, plan, force=()):
        """Write the plan through the paths the form used, in the one order
        that never trips over a name: what goes, goes first; then the rows
        that stay, in an order that frees and takes names safely; then the
        new ones.

        The per-row fixed check and the last-varying-row refusal stand down
        for the length of this. Both were asked over the finished grid
        already, and asking them again half way through would refuse a save
        for a state nothing ever sees.
        """
        rows, deleted = plan['rows'], plan['deleted']
        by_row = dict(rows)
        round_before = self.pending_batch_no
        removed, changed, added, units = [], [], [], []
        self._in_grid_apply = True
        try:
            for name in deleted:
                if self._var_by_name(name).get('category',
                                               'ingredient') == 'ingredient':
                    removed += self.remove_ingredient(
                        name, force=name in force) or []
                else:
                    self.remove_process_parameter(name)
            for row_no in plan['rename_order']:
                spec = by_row[row_no]
                moved = self._write_ingredient_row(spec)
                removed += spec.pop('_removed', [])
                if spec['id'] != spec['name']:
                    self.rename_variable(spec['id'], spec['name'])
                    moved = moved | {'other'}
                if 'unit' in moved:
                    units.append(spec['name'])
                if moved - {'unit'}:
                    changed.append(spec['name'])
            for _, spec in rows:
                if spec['var'] is not None:
                    continue
                self._write_ingredient_row(spec)
                removed += spec.pop('_removed', [])
                added.append(spec['name'])
            worked_out, dropped = self._write_grid_formulas(rows)
            removed += dropped
        finally:
            self._in_grid_apply = False
        return self._ingredient_grid_messages(added, changed, deleted, units,
                                              removed, round_before,
                                              worked_out)

    def _write_grid_formulas(self, rows):
        """The formula cells, written LAST — the names of the rows that
        newly have one, and whatever the writes cost the limits.

        Last, because a formula may name a row this same save adds, and a
        row is added after the rows that stay are written. Each cell goes
        through set_formula / clear_formula, the one door a formula is
        written through: the per-row refusals that door asks are held back
        for the length of a grid Save (`_in_grid_apply`), because the grid
        has already asked them over the finished grid, and everything the
        door owes the project as a whole — the re-encoded history, the open
        round, the total's own limit — happens there, once, for both
        callers.
        """
        new, removed = [], []
        for _, spec in rows:
            var = self._var_by_name(spec['name'])
            was = (str(var.get('formula') or ""), bool(var.get('balance')))
            now = (spec['formula'], spec['balance'])
            if was == now:
                continue
            if spec['formula']:
                removed += self.set_formula(spec['name'],
                                            spec['formula']) or []
                new.append(spec['name'])
            else:
                removed += self.clear_formula(spec['name']) or []
        return new, removed

    def _write_ingredient_row(self, spec):
        """One row's answers, through the narrowest door that carries them.

        What comes back is WHAT moved — its unit, its supplier, everything
        else, or nothing — and the door is chosen by that, because the doors
        differ in what they cost. A changed range retires the open round;
        a unit does not (it is how a number is written, not the number), and
        a vendor printed on a sheet does not either. Routing every row
        through add_ingredient would have retired the round for a corrected
        SKU.
        """
        var = spec['var']
        name = spec['id'] or spec['name']
        if var is None:
            self._add_grid_row(spec, spec['name'])
            return {'other'}
        moved = _what_moved(self._row_state(var),
                            self._proposed_row_state(var, spec))
        if not moved:
            # Nothing about the question the model is being asked has moved,
            # so there is nothing for the default batch size, the limits or
            # the open round to answer for. This is also what lets a pure
            # rename keep the open round: rename_variable rewrites its rows
            # in place, where add_ingredient would retire it.
            return moved
        if 'other' in moved:
            if var.get('category', 'ingredient') != spec['category']:
                self.set_variable_type(name, spec['category'])
            self._add_grid_row(spec, name)
        elif 'unit' in moved:
            spec['_removed'] = self.set_variable_unit(name, spec['unit']) or []
        if 'supplier' in moved:
            var = self._var_by_name(name)
            var['vendor'], var['sku'] = spec['vendor'], spec['sku']
            self.save()
        return moved

    def _add_grid_row(self, spec, name):
        """The add path, which is also the edit path: re-adding a name the
        project already has updates its allowed amounts and its unit, so
        every consequence the screen owes fires exactly once."""
        if spec['category'] == 'process':
            self.add_process_parameter(
                name, spec['low'], spec['high'],
                # Only a baseline the reader moved is passed on: an
                # unchanged one is already where it belongs, and handing it
                # back would take the range-fixing path's exemption away.
                baseline=spec['baseline'] if spec['baseline_moved'] else None,
                unit=spec['unit'])
            return
        spec['_removed'] = self.add_ingredient(
            name, spec['low'], spec['high'], unit=spec['unit'],
            keep_lowest=True) or []
        var = self._var_by_name(name)
        if (str(var.get('vendor', "")), str(var.get('sku', ""))) != (
                spec['vendor'], spec['sku']):
            var['vendor'], var['sku'] = spec['vendor'], spec['sku']
            self.save()

    def _proposed_row_state(self, var, spec):
        """What _row_state would say about this row once the grid is saved.
        A baseline the grid does not ask for is the one the row already
        carries."""
        baseline = (var.get('_absent_value') if spec['baseline'] is None
                    else float(spec['baseline']))
        return (spec['name'], spec['category'],
                (float(spec['low']), float(spec['high'])), spec['unit'],
                spec['vendor'], spec['sku'], baseline,
                spec.get('formula_typed', spec['formula']), spec['balance'])

    def _row_state(self, var):
        """Everything one row of the grid says about a variable, as one
        comparable value. None for a row that was not there before."""
        if var is None:
            return None
        return (var['name'], var.get('category', 'ingredient'),
                tuple(float(b) for b in var['bounds']), self._unit_of(var),
                str(var.get('vendor', "")), str(var.get('sku', "")),
                var.get('_absent_value'), str(var.get('formula') or ""),
                bool(var.get('balance')))

    def _ingredient_grid_messages(self, added, changed, deleted, units,
                                  removed, round_before, worked_out=()):
        """One line per consequence, each said once however many rows caused
        it — which is the whole point of applying a grid in one go."""
        messages = []
        new_ingredients = [n for n in added if self._var_by_name(n).get(
            'category', 'ingredient') == 'ingredient']
        if added:
            line = wording.added(number_list(added))
            if new_ingredients and self.has_formulation_total():
                line += " " + wording.total_still_holds(
                    self.batch_total_text(self.formulation_total))
            messages.append(("success", line))
        if changed:
            messages.append(("success", wording.saved(number_list(changed))))
        for name in units:
            # Its own sentence, per row: nothing was converted and nothing
            # was rescored, and this is the only line that says so.
            var = self._var_by_name(name)
            messages.append(("success", wording.unit_changed(
                name, self._unit_of(var),
                var.get('category', 'ingredient') == 'ingredient')))
        if deleted:
            messages.append(("success", wording.deleted(number_list(deleted))))
        if new_ingredients and self.recipe_history:
            messages.append(("info", wording.formulations_contain_none_of(
                number_list(new_ingredients))))
        if worked_out and self.recipe_history:
            # What was weighed is what was weighed: a formula arriving today
            # answers for the row from the next round on, and nothing it
            # says rewrites a formulation the bench already made.
            messages.append(("info", wording.formulations_keep_their_amounts(
                number_list(list(worked_out)), len(worked_out) > 1)))
        messages += self.limit_removed_messages(removed)
        if round_before is not None and self.pending_batch_no is None:
            messages.append(("info",
                             wording.batch_discarded_notice(round_before)))
        return messages

    def limit_removed_messages(self, removed):
        """Name every limit an edit has just emptied of meaning, one line
        each. It lives here rather than on the screen because four doors
        reach it — a unit set, a new default unit, a reloaded ingredient
        file and now a grid Save — and all four owe the same sentence."""
        messages = []
        for qc in removed or []:
            if 'metric' in qc:
                # A property limit is an average over the amounts, so it is
                # the ingredients as a whole that stopped sharing a unit.
                messages.append(("warning",
                                 wording.property_limit_removed(qc['metric'])))
            elif qc.get('source') == 'formulation_total':
                total_text = join_unit(f"{float(qc.get('total')):g}",
                                       qc.get('unit') or "")
                messages.append((
                    "warning",
                    wording.formulation_total_gone_unit(total_text)
                    if qc.get('reason') == 'unit'
                    else wording.formulation_total_gone_unreachable(
                        total_text)))
            elif qc.get('reason') == 'percent_unreachable':
                messages.append((
                    "warning",
                    wording.quantity_limit_removed_percent_unreachable(
                        self.limit_label(qc),
                        self.batch_total_text(self.formulation_total))))
            elif qc.get('reason') == 'no_default':
                # The default itself is gone by the time this is read —
                # cleared on purpose, or blanked by a unit split or by
                # amounts that no longer reach it — so there is nothing
                # left for a percent limit to be a percent OF.
                messages.append(("warning", wording.percent_limit_removed(
                    self.limit_label(qc))))
            elif qc.get('reason') == 'missing':
                gone = qc.get('missing') or []
                many = len(gone) > 1
                messages.append(("warning",
                                 wording.quantity_limit_removed_missing(
                                     self.limit_label(qc),
                                     wording.no_longer_ingredients(
                                         number_list(gone) if many else gone[0],
                                         many))))
            else:
                messages.append((
                    "warning", wording.quantity_limit_removed_unit_mismatch(
                        self.limit_label(qc))))
        return messages

    def set_variable_type(self, name, category):
        """Move one row between Ingredient and Process setting.

        Allowed only while nothing has been recorded. A name is the key every
        formulation's amounts are filed under, and an ingredient's column is
        a mass where a setting's is a temperature: swapping them under a
        history would rewrite bakes nobody made. With no history there is
        nothing to rewrite, so the row simply changes its mind — and the
        amount limits are pruned, because a setting is in no sum.
        """
        var = self._var_by_name(name)
        if var.get('category', 'ingredient') == category:
            return []
        if self.X_history:
            raise ValueError(wording.TYPE_LOCKED_ERROR)
        var['category'] = category
        if category == 'process':
            var.pop('vendor', None)
            var.pop('sku', None)
            self.ingredient_properties.pop(name, None)
            var['unit'] = str(var.get('unit') or "")
        else:
            var.setdefault('vendor', "")
            var.setdefault('sku', "")
        removed = self.prune_amount_limits()
        self._drop_pending_batch()
        self.save()
        return removed

    # -- the measurements grid ------------------------------------------ #

    def set_shares(self, shares):
        """Say what each measurement is worth out of 100, and derive the
        importances from that (spec 1.3).

        The shares ARE the importances. They are scaled proportionally so
        the column adds up to 100 — the reader is typing into a column whose
        sum they can see, and a column of 30, 30, 30 has to mean something —
        and that sum, 100, is then the ceiling every overall score is
        written against: a formulation that hits every goal scores 100, and
        one that is most of the way there reads 88 of 100.

        One scale, everywhere: no second number behind the column, and
        nothing on screen the reader did not type. `True` comes back when
        the scaling moved what was handed in, which is what the caption
        under the grid is about.
        """
        if not self.objectives:
            return False
        values = {}
        for obj in self.objectives:
            value = shares.get(obj['name'])
            if value is None or float(value) <= 0:
                raise ValueError(wording.SHARE_REQUIRED_ERROR)
            values[obj['name']] = float(value)
        total = sum(values.values())
        rebalanced = abs(total - 100.0) > 1e-9
        for obj in self.objectives:
            obj['weight'] = values[obj['name']]
        self._shares_to_100()
        self._recompute_utilities()
        self.save()
        return rebalanced

    def apply_measurement_grid(self, frame, archive=None):
        """Write the measurements grid to the project. The same contract as
        apply_ingredient_grid: `(errors, messages)`, and nothing is written
        while there is an error.

        `archive` is called once, after the grid reads clean and before the
        first write, when this save would take something away — a deleted
        measurement, or a change that recalculates every overall score
        already stored. It is the screen's own "keep a copy first", handed
        in rather than asked for, so the grid is read ONCE per save: asking
        the model whether it would rescore and then telling it to save
        planned the whole thing twice, and the second plan is the one that
        counts. Anything it raises comes back out before a word is
        written."""
        errors, plan = self._plan_measurement_grid(frame)
        if errors:
            return errors, []
        if archive is not None and (plan['deleted']
                                    or (self.Y_history
                                        and self._plan_rescores(plan))):
            archive()
        return [], self._apply_measurement_plan(plan)

    def _plan_rescores(self, plan):
        if plan['deleted'] or any(spec['obj'] is None
                                  for _, spec in plan['rows']):
            return True
        stored = self.share_percents()
        for _, spec in plan['rows']:
            if _objective_scoring_state(spec['obj']) != (
                    spec['goal'],
                    None if spec['target'] is None else float(spec['target']),
                    float(spec['min_val']), float(spec['max_val'])):
                return True
            # Under the name it is filed under today: a rename moves the
            # results with it, so it recalculates nothing.
            if abs(float(stored.get(spec['id'] or spec['name'], 0))
                   - float(spec['share_of_score'])) > 1e-9:
                return True
        return False

    def _plan_measurement_grid(self, frame):
        errors, rows, seen = [], [], {}
        by_id = {o['name']: o for o in self.objectives}
        ids_used = set()
        for row_no, row in _grid_rows(frame):
            if _row_is_blank(row):
                continue
            spec, trouble = self._read_measurement_row(row, by_id, ids_used)
            if trouble:
                errors.append((row_no, trouble))
                continue
            twin = seen.get(spec['name'].lower())
            if twin is not None:
                # Blamed on the row that moved, as on the grid above: the
                # reader typed into one of the two, and asking them to fix
                # the other is asking them to fix a row they never touched.
                blamed = (twin if _row_moved(twin[1]) and not _row_moved(spec)
                          else (row_no, spec))
                errors.append((blamed[0], wording.MEASUREMENT_EXISTS_ERROR))
                continue
            seen[spec['name'].lower()] = (row_no, spec)
            rows.append((row_no, spec))
        if errors:
            return errors, None
        # An ingredient or a property is not on this grid and cannot move
        # under it; another MEASUREMENT can, so a name one row is giving up
        # in this same save is free for the next row to take.
        for row_no, spec in rows:
            try:
                self._name_is_free_of_variables(spec['name'])
                self._name_is_free_of_properties(spec['name'])
            except ValueError as e:
                errors.append((row_no, str(e)))
        if errors:
            return errors, None
        deleted = [o['name'] for o in self.measurements_by_importance()
                   if o['name'] not in ids_used]
        # ...which the write order has to make true: a rename onto a name
        # another row is still wearing is refused by the model, so the
        # renames go in an order that frees each name before it is taken.
        order, stuck = _rename_order(rows)
        if stuck is not None:
            return [(stuck[0], wording.MEASUREMENT_EXISTS_ERROR)], None
        return [], {'rows': rows, 'deleted': deleted, 'rename_order': order}

    def _read_measurement_row(self, row, by_id, ids_used):
        row_id = _text_cell(row, GRID_ID)
        obj = by_id.get(row_id)
        if obj is not None:
            ids_used.add(row_id)
        name = _text_cell(row, wording.MEASUREMENT_COLUMN)
        if not name:
            return None, wording.NAME_REQUIRED_ERROR
        if obj is not None and name != obj['name']:
            # A name typed over another is a rename of THAT measurement:
            # every result already recorded is filed under it, and
            # rename_objective moves them together. Asked here, without
            # writing, so a name that is taken refuses the whole save — but
            # not against the OTHER measurements, which are this grid's own
            # rows and may be giving the name up in this same save. The
            # grid's pass over its finished names catches a real collision.
            if is_reserved_name(name):
                return None, _reserved_name_message(name)
            try:
                self._name_is_free_of_variables(name)
                self._name_is_free_of_properties(name)
            except ValueError as e:
                return None, str(e)
        goal_label = _text_cell(row, wording.GOAL_LABEL)
        goal = next((k for k, v in wording.GOAL_LABELS.items()
                     if v == goal_label), goal_label or 'max')
        low, low_ok = _number_cell(row, wording.LOWEST_MEASURABLE_LABEL)
        high, high_ok = _number_cell(row, wording.HIGHEST_MEASURABLE_LABEL)
        if not low_ok or not high_ok or low is None or high is None:
            return None, wording.NUMBER_REQUIRED_ERROR
        if low >= high:
            return None, RANGE_ENDS_ERROR
        target, target_ok = _number_cell(row, wording.TARGET_LABEL)
        if not target_ok:
            return None, wording.NUMBER_REQUIRED_ERROR
        if goal == 'target':
            if target is None:
                return None, TARGET_REQUIRED_ERROR
            if not (low <= target <= high):
                return None, _target_outside_message(target, low, high)
        share, share_ok = _number_cell(row, wording.SHARE_COLUMN)
        if not share_ok or share is None or share <= 0:
            return None, wording.SHARE_REQUIRED_ERROR
        return {
            'obj': obj, 'id': row_id if obj is not None else None,
            'name': name, 'goal': goal,
            'target': target if goal == 'target' else None,
            'min_val': low, 'max_val': high,
            'unit': _text_cell(row, wording.UNIT_LABEL),
            # Spelled out: a bare 'share' literal is a column header the
            # vocabulary guard refuses, and the guard is right to — the
            # column is "Share of score", never "Share".
            'share_of_score': share,
        }, None

    def _apply_measurement_plan(self, plan):
        messages, rows, deleted = [], plan['rows'], plan['deleted']
        # What the grid was SHOWING before this save: the basis both for
        # which shares the reader moved and for how far the rest give way.
        stored_shares = self.share_percents()
        rescored = self._plan_rescores(plan)
        for name in deleted:
            self.remove_objective(name)
        added, changed = [], []
        by_row = dict(rows)
        # The rows that stay, in an order that frees a name before it is
        # taken; then the new ones, which can then take a name a rename has
        # just given up. The same order the ingredients grid keeps, and for
        # the same reason.
        for row_no in plan['rename_order']:
            spec = by_row[row_no]
            obj = spec['obj']
            # Everything here acts on the row as it is filed TODAY; the
            # rename follows, once it has gone through.
            before = _objective_state(obj)
            self.update_objective(spec['id'], goal=spec['goal'],
                                  target=spec['target'],
                                  min_val=spec['min_val'],
                                  max_val=spec['max_val'], unit=spec['unit'])
            moved_name = spec['id'] != spec['name']
            if moved_name:
                self.rename_objective(spec['id'], spec['name'])
            if moved_name or before != _objective_state(obj):
                changed.append(spec['name'])
        for _, spec in rows:
            if spec['obj'] is not None:
                continue
            self.add_objective(spec['name'], 1.0, spec['goal'],
                               target=spec['target'],
                               min_val=spec['min_val'],
                               max_val=spec['max_val'], unit=spec['unit'])
            added.append(spec['name'])
        typed = {spec['name']: spec['share_of_score'] for _, spec in rows}
        # A renamed row is the same row: its share is compared against what
        # it was showing under its old name, not read as a new one.
        was_called = {spec['name']: (spec['id'] or spec['name'])
                      for _, spec in rows}
        if typed:
            # A row the reader typed a share into is one they have
            # answered — and a row they have just added is always one of
            # those, because there was nothing there to leave alone.
            moved = {name for name, share in typed.items()
                     if was_called[name] not in stored_shares
                     or abs(stored_shares[was_called[name]] - share) > 1e-9}
            # Keyed by the name each row wears NOW, because that is how
            # `typed` is keyed; a renamed row keeps the share it was
            # showing under its old name.
            was_showing = {name: float(stored_shares.get(old, 0.0))
                           for name, old in was_called.items()}
            self.set_shares(self._rebalanced_shares(typed, moved,
                                                    was_showing))
            final = self.share_percents()
            if _shares_moved(typed, final):
                # Named, not counted: the reader typed one number and two
                # other rows moved, and a line that said only "Shares
                # adjusted" left them to find out which — in a toast.
                gave_way = [wording.share_adjusted_to(name,
                                                      self.share_text(name))
                            for name in typed
                            if name not in moved
                            and abs(final.get(name, 0.0)
                                    - was_showing.get(name, 0.0)) >= 0.5]
                messages.append(("info",
                                 wording.shares_rebalanced(
                                     number_list(gave_way))
                                 if gave_way
                                 else wording.SHARES_REBALANCED_CAPTION))
            # A share moved is a row saved: without this a save that changed
            # nothing but the column of shares had no green line at all, and
            # the sentences that ride on it — the rescore, the best moving,
            # the copy — had nowhere to land.
            changed += [n for n in moved
                        if n not in changed and n not in added]
        if added:
            messages.append(("success", wording.added(number_list(added))))
        if changed:
            messages.append(("success", wording.saved(number_list(changed))))
        if deleted:
            messages.append(("success", wording.measurement_deleted(
                number_list(deleted))))
        if rescored and self.Y_history:
            # Said once, on the last green line, whatever mix of edits
            # caused it: a rescore is one thing that happened, not four.
            green = [i for i, (kind, _) in enumerate(messages)
                     if kind == "success"]
            if green:
                kind, line = messages[green[-1]]
                messages[green[-1]] = (kind, line + wording.RECALCULATED_SUFFIX)
        return messages

    def _rebalanced_shares(self, typed, moved, stored):
        """What the column should hold once the reader has moved part of it.

        A share they typed is an answer and is kept; the rest give way
        proportionally to make room for it, which is what "rebalances the
        others" means with more than two rows on the grid. When everything
        moved — or when what moved already asks for 100 or more — there is
        nothing left to give way, and the whole column is scaled instead.

        `stored` is what the grid was SHOWING before this save, taken by
        the caller before a word was written. Read off the project instead,
        it would be the column contaminated by the placeholder weight of a
        row this same save has already added: two rows at 50 % each plus a
        new row at 20 came out 41 / 39 / 20, and the reader watched two
        identical rows separate in the one column whose point is that they
        can add it up.
        """
        others = [n for n in typed if n not in moved]
        kept = sum(typed[n] for n in moved)
        if not moved or not others or kept >= 100:
            return dict(typed)
        pool = sum(max(float(stored.get(n, 0.0)), 0.0) for n in others)
        remainder = 100.0 - kept
        out = {n: float(typed[n]) for n in moved}
        for n in others:
            out[n] = (remainder * float(stored.get(n, 0.0)) / pool if pool
                      else remainder / len(others))
        if any(v <= 0 for v in out.values()):
            return dict(typed)
        return out

    # -- the properties grid -------------------------------------------- #
    #
    #  Rows are the ingredients, columns are the properties, and a cell is
    #  that ingredient's figure for that property per 100 g. It is the third
    #  grid on tab 1 and the smallest: it has no rows of its own to add or
    #  delete (the ingredients grid above owns the rows, and `Add a property`
    #  owns the columns), so there is nothing here to confirm and no
    #  deletion to keep a copy before.
    # ------------------------------------------------------------------ #

    def property_grid_frame(self):
        """What the properties grid opens holding.

        No hidden `_id`: a row of this grid is an ingredient the grid cannot
        rename, add or take away, so its name IS its identity. An empty cell
        is an ingredient with no figure, which is not the same as a 0 — the
        caption above the grid says what the app does with one.
        """
        key = wording.PROPERTIES_ROW_COLUMN
        # A property that arrived as a CSV column may be called anything at
        # all, the row column's own label included; two columns of one name
        # is a frame nothing can read a cell out of. `add_property` refuses
        # the name (it is reserved), so this only ever catches a file.
        properties = self.grid_properties()
        data = []
        for var in self._ingredients():
            row = {key: var['name']}
            for prop in properties:
                row[prop] = (float(self.property_value(var['name'], prop))
                             if self.has_property_value(var['name'], prop)
                             else None)
            data.append(row)
        return _grid_frame(data, [key] + properties)

    def apply_property_grid(self, frame):
        """Write the properties grid. `(errors, messages)`, and while there
        is an error NOTHING has been written — the same contract the two
        grids above keep.

        Every figure goes through `set_property_value`, and only the cells
        that actually moved do: the door is what keeps a property's stored
        capitalisation and what a cleared cell means, and writing every cell
        of an eight-by-six grid on every save would save the project
        forty-eight times to change one number.
        """
        key = wording.PROPERTIES_ROW_COLUMN
        properties = self.grid_properties()
        known = {v['name'] for v in self._ingredients()}
        errors, writes = [], []
        for row_no, row in _grid_rows(frame):
            name = _text_cell(row, key)
            if not name:
                continue                  # the editor's own empty line
            if name not in known:
                errors.append((row_no, wording.no_such_ingredient(name)))
                continue
            for prop in properties:
                value, is_number = _number_cell(row, prop)
                if not is_number:
                    errors.append((row_no,
                                   wording.property_not_a_number(prop)))
                    continue
                writes.append((name, prop, value))
        if errors:
            return errors, []
        moved = False
        for name, prop, value in writes:
            stored = (self.property_value(name, prop)
                      if self.has_property_value(name, prop) else None)
            if stored is None and value is None:
                continue
            if (stored is not None and value is not None
                    and abs(stored - value) < 1e-9):
                continue
            self.set_property_value(name, prop, value)
            moved = True
        if not moved:
            return [], []
        return [], [("success", wording.PROPERTIES_SAVED)]

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
        active = [v['name'] for v in self.varying_variables()]
        inactive = [
            f"{v['name']} (fixed at {self._fixed_value(v):.3g})"
            for v in self.fixed_variables()
        ]
        all_names = [v['name'] for v in self.variables]
        lines = [f"In play ({len(active)}): {', '.join(active)}"]
        if inactive:
            lines.append(f"Fixed ({len(inactive)}): {', '.join(inactive)}")
        lines.append("")
        for i, y in enumerate(self.Y_history):
            rec = self.recipe_history[i] if i < len(self.recipe_history) else {}
            # Show every variable that was actually used, including ones
            # since fixed, so the expert can see what it contributed.
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
            self.load_error = wording.PROJECT_NOT_FOUND
            return False
        try:
            self.import_json(state)
        except Exception:
            self.load_error = wording.PROJECT_FILE_DAMAGED
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
            'next_batch_number': self.next_batch_number,
            'sobol_seed': getattr(self, 'sobol_seed', None),
            'pending_batch_no': self.pending_batch_no,
            'pending_batch_created': self.pending_batch_created,
            'pending_batch_discarded': self.pending_batch_discarded,
            'pending_batch_total': getattr(self, 'pending_batch_total', None),
            'formulation_total': getattr(self, 'formulation_total', None),
            # Round number -> {ingredient: the lot it was weighed from}.
            'lots': {str(k): {str(i): str(v) for i, v in (row or {}).items()}
                     for k, row in (getattr(self, 'lots', None) or {}).items()},
            # A None value is kept: it is this batch's own record that it
            # was made as generated, which is not the same as no record.
            'batch_totals': {str(k): (None if v is None else float(v))
                             for k, v in self._batch_totals().items()},
            'amount_unit': self.amount_unit,
            'targets_source': getattr(self, 'targets_source', "") or "",
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
        bad = wording.NOT_A_COPY
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
                raise _damaged(f"'{key}' section has the wrong shape")
        for key in ('variables', 'objectives'):
            for item in state[key]:
                if not isinstance(item, dict) or not isinstance(item.get('name'), str):
                    raise _damaged(f"'{key}' section has the wrong shape")
        # A row worked out from the others carries the text as it was typed
        # and, for the one row that takes whatever is left of the batch
        # size, a flag. import_json reads both straight back into the search
        # vector, so a copy that holds anything else is refused here.
        balanced = []
        copy_names = [item['name'] for item in state['variables']]
        for item in state['variables']:
            if not isinstance(item.get('formula', ""), str):
                raise _damaged("'variables' section has the wrong shape")
            if not isinstance(item.get('balance', False), bool):
                raise _damaged("'variables' section has the wrong shape")
            if item.get('balance'):
                balanced.append(item['name'])
            # Readable, not just a string: every screen that draws the grid
            # asks the parser what this cell means, and ui_setup calls
            # worked_out_captions BEFORE it draws the Save button. A copy
            # holding '= Nonexistent' rendered the grid, raised under it,
            # and left the reader with no Save to edit their way out with.
            if item.get('formula'):
                try:
                    parse_formula(item['formula'],
                                  [n for n in copy_names if n != item['name']],
                                  True)
                except FormulaError:
                    raise _damaged("'variables' section has the wrong shape")
        if len(balanced) > 1:
            raise ValueError(wording.COPY_TWO_BALANCE_ROWS)
        for key in ('recipe_history', 'results_history'):
            for item in state[key]:
                if not isinstance(item, dict):
                    raise _damaged(f"'{key}' section has the wrong shape")
        # An amount limit's own 'percent' block is optional (0.5.0 wave 2,
        # task 5) — a limit written as a % of batch size rather than a
        # plain amount — but a present one must carry only its own three
        # numbers-or-None. A malformed one would divide the wrong thing by
        # the batch size on the very first render.
        for qc in state.get('quantity_constraints') or []:
            if not isinstance(qc, dict):
                raise _damaged("'quantity_constraints' section has the "
                               "wrong shape")
            # Exactly is the same kind of number as the two beside it, and
            # limit_text prints it with a 'g' format code: a copy holding
            # 'twelve' raised on every render of the Limits section and of
            # the Set-up sheet.
            exactly = qc.get('exactly')
            if exactly is not None and (isinstance(exactly, bool)
                                        or not isinstance(exactly,
                                                          (int, float))):
                raise _damaged("'quantity_constraints' section has the "
                               "wrong shape")
            percent = qc.get('percent')
            if percent is None:
                continue
            if (not isinstance(percent, dict)
                    or set(percent) - {'min', 'max', 'exactly'}
                    or not all(v is None or (isinstance(v, (int, float))
                                             and not isinstance(v, bool))
                              for v in percent.values())):
                raise _damaged("'quantity_constraints' section has the "
                               "wrong shape")
        version = state.get('CLASS_VERSION')
        if not isinstance(version, int):
            raise ValueError(bad)
        if version > FoodOptimizer.CLASS_VERSION:
            raise ValueError(wording.COPY_FROM_A_NEWER_VERSION)
        if len(state['recipe_history']) != len(state['results_history']):
            raise _damaged("formulations and results differ in count")
        # The 0.3.0 identity lists. They are optional (a 0.2.x file has none),
        # but a present-and-malformed one must be refused here: import_json
        # assigns attributes one by one, so a TypeError raised halfway through
        # would leave the optimizer wearing half of a bad backup.
        def _whole(x):
            return isinstance(x, int) and not isinstance(x, bool)

        def _as_int(x):
            """A JSON object's key is a string; 'two' is not a batch number."""
            try:
                return int(x)
            except (TypeError, ValueError):
                return None

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
                    and (x.get(ROUND_FIELD) is None
                         or _whole(x.get(ROUND_FIELD)))
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
                raise _damaged(f"'{key}' section has the wrong shape")
        if state.get('next_formulation_no') is not None and not _whole(
                state['next_formulation_no']):
            raise _damaged("'next_formulation_no' section has the wrong shape")
        # Properties named in the app. A malformed list would reach the
        # property picker and the limits list, so it is refused here.
        names = state.get('property_names')
        if names is not None and (not isinstance(names, list)
                                  or not all(isinstance(n, str) for n in names)):
            raise _damaged("'property_names' section has the wrong shape")
        # Where the targets came from: optional, but a present value must be
        # text — import_json would otherwise store a number or a list as the
        # caption the measurements table shows.
        targets_source = state.get('targets_source')
        if targets_source is not None and not isinstance(targets_source, str):
            raise _damaged("'targets_source' section has the wrong shape")
        # The total a batch was made to. A bad one would silently rewrite
        # every amount tab 3 shows for the best formulation.
        def _total(x):
            return (isinstance(x, (int, float)) and not isinstance(x, bool)
                    and float(x) > 0)

        open_total = state.get('pending_batch_total')
        if open_total is not None and not _total(open_total):
            raise _damaged("'pending_batch_total' section has the wrong shape")
        # The total every suggested formulation is built to. A bad one would
        # be written straight back out as the limit the next batch is held to.
        project_total = state.get('formulation_total')
        if project_total is not None and not _total(project_total):
            raise _damaged("'formulation_total' section has the wrong shape")
        totals = state.get('batch_totals')
        if totals is not None and not isinstance(totals, dict):
            raise _damaged("'batch_totals' section has the wrong shape")
        for key, value in (totals or {}).items():
            # None is a value here: 'this batch was made as generated'.
            if not _whole(key if isinstance(key, int) else _as_int(key)) \
                    or (value is not None and not _total(value)):
                raise _damaged("'batch_totals' section has the wrong shape")
        # The lot numbers: {round: {ingredient: lot}}. import_json walks it
        # and would raise halfway through a restore, leaving the optimizer
        # wearing half of a bad copy.
        lots = state.get('lots')
        if lots is not None and not isinstance(lots, dict):
            raise _damaged("'lots' section has the wrong shape")
        # The names the copy's own variable list holds. A lot filed against
        # an ingredient the copy does not have is a lot nothing can ever
        # show: the Lots sheet would print a name the project never had.
        named = {v['name'] for v in state['variables']}
        for key, written in (lots or {}).items():
            if not _whole(key if isinstance(key, int) else _as_int(key)) \
                    or not isinstance(written, dict) \
                    or not all(isinstance(name, str) and isinstance(lot, str)
                               for name, lot in written.items()):
                raise _damaged("'lots' section has the wrong shape")
            if not set(written) <= named:
                raise _damaged("'lots' section has the wrong shape")
        pending = state.get('pending_batch')
        if pending is not None and not isinstance(pending, list):
            raise _damaged("'pending_batch' section has the wrong shape")
        for item in pending or []:
            if (isinstance(item, dict) and item.get('formulation') is not None
                    and not _number(item['formulation'])):
                raise _damaged("'pending_batch' section has the wrong shape")
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
            raise _damaged("two formulations share one number")
        counter = state.get('next_formulation_no')
        if _whole(counter) and any(n >= counter for n in stored + in_batch):
            raise _damaged("a formulation number the counter never issued")
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
        # A file written before batch numbers were stored has none; it is
        # computed from the batches the file holds, in _backfill_identity.
        try:
            self.next_batch_number = int(state.get('next_batch_number') or 1)
        except (TypeError, ValueError):
            self.next_batch_number = 1
        try:
            stored_seed = state.get('sobol_seed')
            self.sobol_seed = None if stored_seed is None else int(stored_seed)
        except (TypeError, ValueError):
            self.sobol_seed = None
        # A 0.2.x file has no unit at all: it backfills to g rather than
        # reopening as a project whose amounts mean nothing, and says so once
        # on tab 1. A file that HOLDS a blank unit is a deliberate blank and
        # is kept — the key's presence, not its truthiness, is the question.
        self.amount_unit_backfilled = 'amount_unit' not in state
        self.amount_unit = ("g" if self.amount_unit_backfilled
                            else str(state.get('amount_unit') or ""))
        # A file written before this was stored has no key at all; blank is
        # backfilled below in _backfill_identity.
        self.targets_source = str(state.get('targets_source') or "").strip()
        self.pending_batch = state.get('pending_batch', None)
        self.pending_batch_no = state.get('pending_batch_no', None)
        self.pending_batch_created = state.get('pending_batch_created', None)
        self.pending_batch_discarded = [
            int(n) for n in (state.get('pending_batch_discarded') or [])
        ]
        stored_total = state.get('pending_batch_total')
        self.pending_batch_total = (None if stored_total is None
                                    else float(stored_total))
        # A file written before a project could carry a total has no key at
        # all, and None is exactly right: nothing was ever asked of the sum.
        project_total = state.get('formulation_total')
        self.formulation_total = (None if project_total is None
                                  else float(project_total))
        self.batch_totals = {int(k): (None if v is None else float(v))
                             for k, v
                             in (state.get('batch_totals') or {}).items()}
        self.lots = {int(k): {str(i): str(v) for i, v in (row or {}).items()}
                     for k, row in (state.get('lots') or {}).items()}
        while len(self.timestamps_history) < len(self.results_history):
            self.timestamps_history.append(None)  # pre-feature files/backups

        for var in self.variables:
            if 'bounds' in var and isinstance(var['bounds'], list):
                var['bounds'] = tuple(var['bounds'])
            var.setdefault('category', 'ingredient')
            self._migrate_fixed(var)
            if var['category'] == 'process':
                # A 0.2.x setting was stored before settings could carry a
                # unit; blank is what it had, and blank is what it keeps.
                var.setdefault('unit', "")
            else:
                # 0.5.0: where an ingredient was bought, printed on the
                # sheets and never read by the model. Blank is what every
                # older project had.
                var.setdefault('vendor', "")
                var.setdefault('sku', "")
                # 0.5.0 wave 2: the Formula cell as it was typed, and the
                # one row that takes whatever is left of the batch size.
                # Blank and False are what every older project had, which
                # is a project where every amount is searched.
                var.setdefault('formula', "")
                var.setdefault('balance', False)

        for obj in self.objectives:
            obj.setdefault('unit', "")
        self._shares_to_100()

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
            self.pending_batch_no = self._issue_batch_no()
        if self.pending_batch_no is not None:
            self.pending_batch_no = int(self.pending_batch_no)
