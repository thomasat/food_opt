"""Save and load: the Optimizer's state as plain JSON."""
from __future__ import annotations

import copy
import json
from dataclasses import asdict

from egbo._limits import PropertyLimit, QuantityLimit
from egbo._score import Objective
from egbo._space import Variable

FORMAT = "egbo"
FORMAT_VERSION = 1


def to_dict(opt) -> dict:
    return {
        "format": FORMAT,
        "format_version": FORMAT_VERSION,
        "sobol_seed": int(opt.sobol_seed),
        "robust": bool(opt.robust),
        "bo_config": copy.deepcopy(opt.bo_config),
        "n_init": int(opt.n_init),
        "variables": [asdict(v) for v in opt.variables],
        "objectives": [asdict(o) for o in opt.objectives],
        "quantity_limits": [asdict(q) for q in opt.quantity_limits],
        "property_limits": [asdict(p) for p in opt.property_limits],
        "properties": copy.deepcopy(opt.properties),
        "formulation_total": opt.formulation_total,
        "next_formulation_no": int(opt.next_formulation_no),
        "recipes": copy.deepcopy(opt.recipes),
        "results": copy.deepcopy(opt.results),
        "scores": list(opt.scores),
        "formulation_ids": list(opt.formulation_ids),
        "open_rows": copy.deepcopy(opt.open_rows),
        "retired": copy.deepcopy(opt.retired),
    }


def from_dict(cls, d: dict):
    if not isinstance(d, dict) or d.get("format") != FORMAT:
        raise ValueError("This is not an egbo file")
    version = d.get("format_version")
    if not isinstance(version, int) or version > FORMAT_VERSION:
        raise ValueError(f"This egbo file is version {version}; this egbo "
                         f"reads up to version {FORMAT_VERSION}")
    opt = cls(seed=d["sobol_seed"], robust=d["robust"],
              bo_config=copy.deepcopy(d["bo_config"]), n_init=d["n_init"])
    opt.variables = [Variable(**v) for v in d["variables"]]
    opt.objectives = [Objective(**o) for o in d["objectives"]]
    opt.quantity_limits = [QuantityLimit(**q) for q in d["quantity_limits"]]
    opt.property_limits = [PropertyLimit(**p) for p in d["property_limits"]]
    opt.properties = copy.deepcopy(d["properties"])
    opt.formulation_total = d["formulation_total"]
    opt.next_formulation_no = int(d["next_formulation_no"])
    opt.recipes = copy.deepcopy(d["recipes"])
    opt.results = copy.deepcopy(d["results"])
    opt.scores = [float(s) for s in d["scores"]]
    opt.formulation_ids = [int(i) for i in d["formulation_ids"]]
    opt.open_rows = copy.deepcopy(d["open_rows"])
    opt.retired = copy.deepcopy(d["retired"])
    return opt


def save(opt, path) -> None:
    with open(path, "w", encoding="utf-8") as fh:
        json.dump(to_dict(opt), fh, indent=2)


def load(cls, path):
    with open(path, encoding="utf-8") as fh:
        return from_dict(cls, json.load(fh))
