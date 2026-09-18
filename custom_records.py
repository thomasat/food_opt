"""Optional, unscored records with stable field IDs and explicit scope."""
import copy
import hashlib
import json
import uuid

import pandas as pd
from openpyxl import load_workbook
from openpyxl.comments import Comment
from openpyxl.utils import get_column_letter
from openpyxl.styles import Font, PatternFill, Protection, Alignment

import wording

SCOPES = ('formulation', 'ingredient')
MARKER = 'food-opt-custom-record:'


def empty():
    return {'fields': [], 'values': {}}


def state(opt):
    return getattr(opt, 'custom_records', None) or empty()


def fields(opt, scope=None, enabled=True):
    return [f for f in state(opt)['fields']
            if (scope is None or f['scope'] == scope) and (not enabled or f['enabled'])]


def valid(data):
    if not isinstance(data, dict) or set(data) != {'fields', 'values'}:
        return False
    if not isinstance(data['fields'], list) or not isinstance(data['values'], dict):
        return False
    known, labels = {}, set()
    for f in data['fields']:
        if (not isinstance(f, dict) or set(f) != {'id', 'name', 'scope', 'enabled'}
                or not isinstance(f['id'], str) or not f['id'] or f['id'] in known
                or not isinstance(f['name'], str) or not 1 <= len(f['name'].strip()) <= 80
                or f['scope'] not in SCOPES or type(f['enabled']) is not bool):
            return False
        key = (f['scope'], f['name'].strip().casefold())
        if key in labels:
            return False
        labels.add(key)
        known[f['id']] = f
    for number, scopes in data['values'].items():
        if not str(number).isdigit() or not isinstance(scopes, dict) or any(s not in SCOPES for s in scopes):
            return False
        for scope, subjects in scopes.items():
            if not isinstance(subjects, dict):
                return False
            for subject, values in subjects.items():
                if (not isinstance(subject, str) or not subject
                        or (scope == 'formulation' and not subject.isdigit())
                        or not isinstance(values, dict)):
                    return False
                for key, value in values.items():
                    if (key not in known or known[key]['scope'] != scope
                            or not isinstance(value, str) or len(value) > 2000):
                        return False
    return True


def add_field(opt, name, scope):
    name = str(name).strip()
    if not name or len(name) > 80 or scope not in SCOPES:
        raise ValueError(wording.CUSTOM_FIELD_INVALID)
    if name.casefold() in {str(v).casefold() for v in wording.RECORD_FIELD_LABELS.values()}:
        raise ValueError(wording.CUSTOM_FIELD_BUILTIN)
    data = copy.deepcopy(state(opt))
    if any(f['name'].casefold() == name.casefold() and f['scope'] == scope for f in data['fields']):
        raise ValueError(wording.CUSTOM_FIELD_EXISTS)
    key = uuid.uuid4().hex
    data['fields'].append({'id': key, 'name': name, 'scope': scope, 'enabled': True})
    opt.custom_records = data
    opt.save()
    return key


def set_enabled(opt, selected):
    selected = set(selected)
    data = copy.deepcopy(state(opt))
    for f in data['fields']:
        f['enabled'] = f['id'] in selected
    if data != state(opt):
        opt.custom_records = data
        opt.save()


def ingredient_names(opt, round_no=None):
    names = [v['name'] for v in opt._ingredients()]
    for group in opt.premixes:
        names += [p['name'] for p in opt.premix_parts(group, round_no)]
    return list(dict.fromkeys(names))


def values(opt, round_no, scope, subject):
    return state(opt)['values'].get(str(round_no), {}).get(scope, {}).get(str(subject), {})


def save_values(opt, round_no, entries):
    """Validate the complete update before one save; values never enter scoring."""
    entries = [(scope, str(subject), key, str(value).strip()) for scope, subject, key, value in entries
               if values(opt, round_no, scope, subject).get(key, '') != str(value).strip()]
    if not entries:
        return False
    data = copy.deepcopy(state(opt))
    target = data['values'].setdefault(str(round_no), {})
    for scope, subject, key, value in entries:
        target.setdefault(scope, {}).setdefault(subject, {})[key] = value
    if not valid(data):
        raise ValueError(wording.CUSTOM_VALUES_INVALID)
    opt.custom_records = data
    opt.save()
    return True


def rename_ingredient(opt, old, new):
    for scopes in state(opt)['values'].values():
        subjects = scopes.get('ingredient', {})
        if old in subjects:
            subjects[new] = subjects.pop(old)


def export_frame(opt):
    definitions = {f['id']: f for f in fields(opt, enabled=False)}
    rows = []
    live = set(opt.formulation_ids) | {r['formulation'] for r in opt.skipped}
    live |= {r['formulation'] for r in opt.pending_batch or []}
    live_rounds = set(opt.batch_history) | {r.get('batch') for r in opt.skipped} | {opt.pending_batch_no}
    for number, scopes in state(opt)['values'].items():
        if int(number) not in live_rounds:
            continue
        for scope, subjects in scopes.items():
            for subject, entries in subjects.items():
                if scope == 'formulation' and int(subject) not in live:
                    continue
                for key, value in entries.items():
                    if value:
                        f = definitions[key]
                        rows.append({wording.ROUND_CAP: int(number), wording.CUSTOM_SCOPE_LABEL: wording.CUSTOM_SCOPE_NAMES[scope],
                                     wording.CUSTOM_SUBJECT: subject, wording.CUSTOM_FIELD_NAME: f['name'],
                                     wording.CUSTOM_VALUE: value})
    return pd.DataFrame(rows)


def append_workbook(book, opt, rows, print_pack):
    """Write both optional scopes on an existing sheet, never add a round tab."""
    if not fields(opt):
        return
    sheet = book[wording.batch_sheet_name(opt.pending_batch_no)] if print_pack else book['Results']
    r = sheet.max_row + 3
    for scope, subjects in [('formulation', [str(row['formulation']) for row in rows]),
                             ('ingredient', ingredient_names(opt, opt.pending_batch_no))]:
        definitions = fields(opt, scope)
        if not definitions or not subjects:
            continue
        cell = sheet.cell(r, 1, wording.CUSTOM_SECTION_NAMES[scope]); cell.font = Font(bold=True)
        r += 1
        sheet.cell(r, 1, wording.CUSTOM_SHEET_HELP)
        r += 1
        # Unhide dedicated columns without affecting the earlier Results matrix.
        columns = [c for c in range(2, sheet.max_column + len(definitions) * 2 + 3)
                   if not sheet.column_dimensions[get_column_letter(c)].hidden][:len(definitions)]
        sheet.cell(r, 1, wording.for_project(opt, wording.FORMULATION_CAP) if scope == 'formulation' else wording.CUSTOM_INGREDIENT_LABEL)
        for col, f in zip(columns, definitions):
            cell = sheet.cell(r, col, f['name']); cell.data_type = 's'; cell.font = Font(bold=True)
        r += 1
        for subject in subjects:
            cell = sheet.cell(r, 1, subject); cell.data_type = 's'
            for col, f in zip(columns, definitions):
                cell = sheet.cell(r, col, values(opt, opt.pending_batch_no, scope, subject).get(f['id'], ''))
                cell.data_type = 's'
                cell.number_format = '@'
                cell.protection = Protection(locked=False)
                cell.fill = PatternFill('solid', fgColor='FFF2CC')
                cell.alignment = Alignment(wrap_text=True)
                cell.comment = Comment(MARKER + json.dumps({'round': opt.pending_batch_no, 'scope': scope,
                                                            'subject': subject, 'field': f['id']}), 'Food Opt')
            r += 1
        r += 2
    sheet.print_area = f'A1:{get_column_letter(sheet.max_column)}{sheet.max_row}'


def read_workbook(source, opt, round_no):
    """Read only marked custom cells; validate scope and subject before import."""
    if hasattr(source, 'seek'):
        source.seek(0)
    if not fields(opt, enabled=False):
        return []
    book = load_workbook(source, data_only=False)
    known = {f['id']: f for f in fields(opt, enabled=False)}
    allowed = {'formulation': {str(row['formulation']) for row in opt.pending_batch or []},
               'ingredient': set(ingredient_names(opt, round_no))}
    entries, seen = [], {}
    for sheet in book:
        for row in sheet:
            for cell in row:
                if not cell.comment or not cell.comment.text.startswith(MARKER):
                    continue
                try:
                    meta = json.loads(cell.comment.text[len(MARKER):])
                except (ValueError, TypeError):
                    raise ValueError(wording.CUSTOM_WORKBOOK_CHANGED)
                if (not isinstance(meta, dict)
                        or not all(isinstance(meta.get(k), str) for k in ('scope', 'field', 'subject'))
                        or meta.get('round') != round_no or meta.get('scope') not in SCOPES
                        or meta.get('field') not in known
                        or known[meta['field']]['scope'] != meta['scope']
                        or meta.get('subject') not in allowed[meta['scope']]):
                    raise ValueError(wording.CUSTOM_WORKBOOK_CHANGED)
                if cell.data_type == 'f':
                    raise ValueError(wording.CUSTOM_TEXT_ONLY)
                value = '' if cell.value is None else str(cell.value).strip()
                if not value:
                    continue  # empty imported cells never erase an existing record
                if len(value) > 2000:
                    raise ValueError(wording.CUSTOM_VALUES_INVALID)
                key = (meta['scope'], meta['subject'], meta['field'])
                if key in seen and seen[key] != value:
                    raise ValueError(wording.CUSTOM_WORKBOOK_CONFLICT)
                seen[key] = value
    entries = [(*key, value) for key, value in seen.items()]
    if hasattr(source, 'seek'):
        source.seek(0)
    return entries


def setup(opt):
    import streamlit as st
    from ui_helpers import saved_ok
    with st.popover(wording.CUSTOM_ADD_LABEL):
        st.caption(wording.CUSTOM_FIELD_HELP)
        name = st.text_input(wording.CUSTOM_FIELD_NAME, key='custom_field_name', max_chars=80)
        scope = st.selectbox(wording.CUSTOM_SCOPE_LABEL, SCOPES, format_func=wording.CUSTOM_SCOPE_NAMES.get,
                             key='custom_field_scope')
        if st.button(wording.CUSTOM_ADD_BUTTON, key='custom_field_add'):
            try:
                add_field(opt, name, scope)
            except ValueError as e:
                st.error(str(e))
            else:
                if saved_ok(opt):
                    st.session_state.pop('record_fields', None)
                    st.rerun()


def editor(opt, round_no, formulation_numbers, prefix='round'):
    import streamlit as st
    from ui_helpers import saved_ok
    if (not fields(opt) and not opt.records('lot')) or round_no is None:
        return
    with st.expander(wording.CUSTOM_RECORDS_HEADING):
        st.caption(wording.CUSTOM_AUTOSAVE_HELP)
        if opt.records('lot'):
            st.caption(wording.LOT_ENTRY_HELP)
            stored_lots = opt.lots.get(int(round_no), {})
            lot_frame = pd.DataFrame([{wording.CUSTOM_INGREDIENT_LABEL: n, wording.LOT_COLUMN: stored_lots.get(n, '')}
                                     for n in ingredient_names(opt, round_no)])
            if not lot_frame.empty:
                mark = hashlib.sha256(lot_frame.to_json().encode()).hexdigest()[:12]
                edited_lots = st.data_editor(lot_frame, hide_index=True, disabled=[wording.CUSTOM_INGREDIENT_LABEL],
                    key=f'custom_lots_{opt.project_name}_{round_no}_{prefix}_{mark}',
                    column_config={wording.LOT_COLUMN: st.column_config.TextColumn(wording.LOT_COLUMN, max_chars=2000)},
                    use_container_width=True)
                updates = {row[wording.CUSTOM_INGREDIENT_LABEL]: ('' if pd.isna(row[wording.LOT_COLUMN]) else str(row[wording.LOT_COLUMN]).strip())
                           for _, row in edited_lots.iterrows()}
                if any(stored_lots.get(n, '') != v for n, v in updates.items()):
                    opt.store_lots(round_no, updates)
                    if saved_ok(opt):
                        st.rerun()
        for scope, subjects in [('formulation', list(map(str, formulation_numbers))),
                                ('ingredient', ingredient_names(opt, round_no))]:
            definitions = fields(opt, scope)
            if not definitions or not subjects:
                continue
            st.markdown(wording.CUSTOM_SECTION_NAMES[scope])
            label = wording.for_project(opt, wording.FORMULATION_CAP) if scope == 'formulation' else wording.CUSTOM_INGREDIENT_LABEL
            frame = pd.DataFrame([{label: subject, **{f['id']: values(opt, round_no, scope, subject).get(f['id'], '')
                                                     for f in definitions}} for subject in subjects])
            mark = hashlib.sha256(frame.to_json().encode()).hexdigest()[:12]
            key = f'custom_values_{opt.project_name}_{round_no}_{prefix}_{scope}_{mark}'
            edited = st.data_editor(frame, hide_index=True, num_rows='fixed', disabled=[label], key=key,
                                    column_config={f['id']: st.column_config.TextColumn(f['name'], max_chars=2000)
                                                   for f in definitions}, use_container_width=True)
            entries = [(scope, str(row[label]), f['id'], '' if pd.isna(row[f['id']]) else str(row[f['id']]))
                       for _, row in edited.iterrows() for f in definitions]
            if save_values(opt, round_no, entries) and saved_ok(opt):
                st.rerun()


def history(opt):
    import streamlit as st
    frame = export_frame(opt)
    if frame.empty and not fields(opt) and not opt.records('lot'):
        return
    rounds = sorted({int(n) for n in opt.batch_history if n is not None}
                    | {int(r['batch']) for r in opt.skipped if r.get('batch') is not None}, reverse=True)
    if not rounds:
        return
    number = st.selectbox(wording.CUSTOM_ROUND_LABEL, rounds, key='custom_history_round')
    numbers = [int(n) for n, b in zip(opt.formulation_ids, opt.batch_history) if b == number]
    numbers += [int(r['formulation']) for r in opt.skipped if r.get('batch') == number]
    editor(opt, number, numbers, prefix='history')
    if not frame.empty:
        st.dataframe(frame, hide_index=True)
