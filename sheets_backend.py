"""Google Sheets persistence backend for FoodOptimizer.

Replaces pickle-on-disk with a Google Sheet per project.
Each spreadsheet has two worksheets:
  - 'config'  : JSON blob of project settings (variables, objectives, constraints)
  - 'history' : one row per experiment (ingredient values + raw scores)

Requires:
  - gspread + google-auth
  - A service account JSON key (stored in Streamlit secrets or as a file)
"""

import json
import gspread
import pandas as pd
from google.oauth2.service_account import Credentials


SCOPES = [
    "https://www.googleapis.com/auth/spreadsheets",
    "https://www.googleapis.com/auth/drive",
]


class SheetsBackend:
    """Read/write FoodOptimizer state to Google Sheets."""

    def __init__(self, credentials_info: dict, folder_id: str | None = None):
        """
        Args:
            credentials_info: service-account JSON dict (from st.secrets or file)
            folder_id: optional Google Drive folder ID to create spreadsheets in
        """
        creds = Credentials.from_service_account_info(credentials_info, scopes=SCOPES)
        self.gc = gspread.authorize(creds)
        self.folder_id = folder_id

    # ------------------------------------------------------------------
    # Public API (mirrors pickle save/load)
    # ------------------------------------------------------------------

    def save(self, project_name: str, state: dict) -> None:
        """Persist full optimizer state to a Google Sheet."""
        spreadsheet = self._get_or_create_spreadsheet(project_name)

        # --- config tab ---
        config_ws = self._get_or_create_worksheet(spreadsheet, "config")
        config_data = {
            "project_name": state.get("project_name", project_name),
            "filename": state.get("filename", ""),
            "variables": state.get("variables", []),
            "objectives": state.get("objectives", []),
            "ingredient_properties": state.get("ingredient_properties", {}),
            "constraints": state.get("constraints", []),
            "quantity_constraints": state.get("quantity_constraints", []),
            "robust": state.get("robust", False),
            "CLASS_VERSION": state.get("CLASS_VERSION", 2),
        }
        config_ws.clear()
        config_ws.update("A1", [["config_json"], [json.dumps(config_data)]])

        # --- history tab ---
        history_ws = self._get_or_create_worksheet(spreadsheet, "history")
        history_ws.clear()

        recipe_history = state.get("recipe_history", [])
        results_history = state.get("results_history", [])
        y_history = state.get("Y_history", [])

        if not recipe_history:
            history_ws.update("A1", [["(no experiments yet)"]])
            return

        # Build column headers from first experiment
        ing_cols = list(recipe_history[0].keys())
        res_cols = list(results_history[0].keys()) if results_history else []
        headers = ["exp_index"] + ing_cols + res_cols + ["utility"]

        rows = [headers]
        for i, recipe in enumerate(recipe_history):
            row = [i]
            row.extend(recipe.get(c, "") for c in ing_cols)
            if i < len(results_history):
                row.extend(results_history[i].get(c, "") for c in res_cols)
            else:
                row.extend("" for _ in res_cols)
            row.append(y_history[i] if i < len(y_history) else "")
            rows.append(row)

        history_ws.update("A1", rows)

    def load(self, project_name: str) -> dict | None:
        """Load optimizer state from a Google Sheet.  Returns None if not found."""
        try:
            spreadsheet = self.gc.open(project_name)
        except gspread.SpreadsheetNotFound:
            return None

        # --- config ---
        try:
            config_ws = spreadsheet.worksheet("config")
            config_json = config_ws.acell("A2").value
            config = json.loads(config_json) if config_json else {}
        except (gspread.WorksheetNotFound, json.JSONDecodeError):
            config = {}

        # --- history ---
        recipe_history = []
        results_history = []
        X_history = []
        Y_history = []

        try:
            history_ws = spreadsheet.worksheet("history")
            all_values = history_ws.get_all_values()
            if len(all_values) > 1 and all_values[0][0] != "(no experiments yet)":
                headers = all_values[0]
                # Figure out which columns are ingredients vs results
                variables = config.get("variables", [])
                objectives = config.get("objectives", [])
                ing_names = [v["name"] for v in variables]
                obj_names = [o["name"] for o in objectives]

                for row in all_values[1:]:
                    row_dict = dict(zip(headers, row))
                    recipe = {}
                    for name in ing_names:
                        if name in row_dict and row_dict[name] != "":
                            recipe[name] = float(row_dict[name])
                    results = {}
                    for name in obj_names:
                        if name in row_dict and row_dict[name] != "":
                            results[name] = float(row_dict[name])

                    recipe_history.append(recipe)
                    results_history.append(results)
                    if "utility" in row_dict and row_dict["utility"] != "":
                        Y_history.append(float(row_dict["utility"]))
        except gspread.WorksheetNotFound:
            pass

        # Reconstruct state dict (same shape as pickle)
        state = {
            **config,
            "recipe_history": recipe_history,
            "results_history": results_history,
            "X_history": X_history,  # Will be rebuilt by _encode() on load
            "Y_history": Y_history,
        }
        return state

    def list_projects(self) -> list[str]:
        """List all project spreadsheets the service account can see."""
        files = self.gc.list_spreadsheet_files()
        return [f["name"] for f in files]

    # ------------------------------------------------------------------
    # Internals
    # ------------------------------------------------------------------

    def _get_or_create_spreadsheet(self, name: str):
        try:
            return self.gc.open(name)
        except gspread.SpreadsheetNotFound:
            return self.gc.create(name, folder_id=self.folder_id)

    def _get_or_create_worksheet(self, spreadsheet, title: str):
        try:
            return spreadsheet.worksheet(title)
        except gspread.WorksheetNotFound:
            return spreadsheet.add_worksheet(title=title, rows=1000, cols=50)
