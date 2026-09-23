"""Compact calculation display and an accessible, cursor-aware expression editor.

The editor stages a canonical ingredient grid. Only the existing grid Save flow
writes projects or retires a pending round; preview validation never writes.
"""
import hashlib
import pandas as pd
import streamlit as st
import streamlit.components.v2 as components

import wording
from food_bo import GRID_ID, formula_is_rest

# The dialog is drawn while this is set, not only on the run its button
# was pressed. Every keystroke in the expression box reruns the app, and a
# dialog rendered inside `if st.button(...)` is gone on the first of them —
# taking the refusal it had just drawn with it.
EDITOR_OPEN = '_calculation_editor_open'
# What the tab says after the panel has staged a calculation: the grid shows
# it and Save changes has lit, and neither of those is a sentence.
EDITOR_STAGED = '_calculation_editor_staged'
FILL = wording.FILL_TO_TOTAL
# Handed to the component, which cannot import wording of its own.
_MESSAGES = {'ingredient': wording.CALCULATION_PICK_AN_INGREDIENT,
             'percentage': wording.CALCULATION_NEEDS_A_PERCENT,
             'operator': wording.CALCULATION_OPERATORS_LABEL.split()[0]}
MODES = [wording.CALCULATION_MODE_RANGE, FILL, wording.CALCULATION_MODE_AMOUNT]


def display_calculation(value):
    text = '' if value is None or pd.isna(value) else str(value).strip()
    if not text:
        return ''
    if formula_is_rest(text):
        return FILL
    return text.removeprefix('=').strip()


def canonical_calculation(value):
    text = '' if value is None or pd.isna(value) else str(value).strip()
    if not text:
        return ''
    if text.casefold() in (FILL.casefold(), 'fill to batch size'):
        return f'= {wording.REST_TOKEN}'
    return text if text.startswith('=') else '= ' + text


def display_frame(frame):
    result = frame.copy()
    result[wording.FORMULA_LABEL] = result[wording.FORMULA_LABEL].map(display_calculation)
    return result


def canonical_frame(frame, opening):
    result = frame.copy()
    # Preserve untouched source text exactly, including spacing, so a rename
    # can still update its references through the existing model transaction.
    originals = {row[GRID_ID]: row[wording.FORMULA_LABEL] for _, row in opening.iterrows()}
    for index, row in result.iterrows():
        value = row.get(wording.FORMULA_LABEL)
        original = originals.get(row.get(GRID_ID), '')
        result.at[index, wording.FORMULA_LABEL] = (
            original if str(value or '') == display_calculation(original)
            else canonical_calculation(value))
    return result


_EXPRESSION = components.component(
    'calculation_expression',
    html=f'''<div class="editor">
<label for="expression">{wording.FORMULA_LABEL}</label>
<div class="entry"><span aria-hidden="true">=</span><textarea id="expression" aria-label="{wording.FORMULA_LABEL}" rows="3" spellcheck="false" aria-describedby="hint" placeholder="{wording.CALCULATION_EXPRESSION_PLACEHOLDER}"></textarea></div>
<div id="suggestions" role="group" aria-label="{wording.CALCULATION_SUGGESTIONS_LABEL}"></div>
<p id="hint">{wording.CALCULATION_EXPRESSION_HINT}</p>
<div class="insert"><label for="ingredient">{wording.CALCULATION_INSERT_INGREDIENT}</label><input id="ingredient" aria-label="{wording.CALCULATION_INSERT_INGREDIENT}" type="search" list="ingredients" placeholder="{wording.CALCULATION_SEARCH_INGREDIENTS}"><datalist id="ingredients"></datalist><button id="insert">{wording.CALCULATION_INSERT_INGREDIENT}</button><button id="batch" title="{wording.CALCULATION_BATCH_SIZE_TITLE}">{wording.CALCULATION_BATCH_SIZE_BUTTON}</button></div>
<div id="operators" role="group" aria-label="{wording.CALCULATION_OPERATORS_LABEL}"></div>
<div class="percent"><label for="percent">{wording.CALCULATION_PERCENT_LABEL}</label><input id="percent" aria-label="{wording.CALCULATION_PERCENT_LABEL}" type="number" placeholder="{wording.CALCULATION_PERCENT_PLACEHOLDER}" step="any"><span> % of </span><select id="of" aria-label="{wording.CALCULATION_PERCENT_OF_LABEL}"></select><button id="percentage">{wording.CALCULATION_INSERT_PERCENT}</button></div>
<p class="notice" role="status" aria-live="polite"></p>
<button id="use" class="primary">{wording.CALCULATION_USE_BUTTON}</button>
</div>''',
    css='''
*{box-sizing:border-box}
.editor{font:inherit;color:var(--st-text-color)} label{display:block;font-weight:600;margin-bottom:6px} .entry{display:flex;gap:8px;align-items:baseline} textarea,input,select,button{font:inherit;color:inherit;border:1px solid #8886;border-radius:6px;background:var(--st-background-color);padding:8px} textarea{width:100%;resize:vertical} input,select{min-width:0;max-width:100%} button{cursor:pointer;white-space:normal} button:hover{border-color:var(--st-primary-color)} button:focus-visible,input:focus-visible,textarea:focus-visible,select:focus-visible{outline:2px solid var(--st-primary-color);outline-offset:2px} p{font-size:.88rem;margin:8px 0 12px} .insert,.percent,#operators,#suggestions{display:flex;gap:6px;flex-wrap:wrap;align-items:center;margin:10px 0}.insert label{width:100%}.insert input{flex:1}.percent input{width:75px}.primary{border-color:var(--st-primary-color);font-weight:600}.notice:empty{display:none}''',
    js=r'''
export default function(component) {
 const {parentElement: root, data, setStateValue, setTriggerValue} = component;
 const $ = s => root.querySelector(s);
 const input = $('#expression');
 if (input.dataset.ready) return;
 input.dataset.ready = '1'; input.value = data.expression;
 let start=input.value.length, end=start, timer;
 const list=$('#ingredients'), target=$('#of');
 function option(parent,text){const node=document.createElement('option');node.value=text;node.textContent=text;parent.appendChild(node);}
 data.names.forEach(n=>option(list,n)); option(target,'batch size'); data.names.forEach(n=>option(target,n));
 const remember=()=>{start=input.selectionStart;end=input.selectionEnd;};
 const publish=()=>{clearTimeout(timer);setStateValue('expression',input.value);};
 function suggest(){
   const box=$('#suggestions');box.replaceChildren();
   const prefix=input.value.slice(0,input.selectionStart);
   const match=prefix.match(/(?:^|[+*\/()=−×÷]|\s-\s)\s*([^+*\/()=−×÷]*)$/);
   const query=match ? match[1].trimStart().split(/%\s+of\s+/i).pop() : ''; 
   if(query.length<2) return;
   const matches=data.names.filter(n=>n.toLowerCase().startsWith(query.toLowerCase()) && n.toLowerCase()!==query.toLowerCase()).slice(0,5);
   matches.forEach(name=>{const b=document.createElement('button');b.textContent=name;b.type='button';b.onmousedown=e=>e.preventDefault();b.onclick=()=>{start=input.selectionStart-query.length;end=input.selectionStart;insert(name);};box.appendChild(b);});
 }
 function insert(text){input.setRangeText(text,start,end,'end');remember();input.focus();suggest();publish();}
 input.onselect=remember; input.onkeyup=()=>{remember();suggest();}; input.onclick=remember;
 input.oninput=()=>{remember();suggest();clearTimeout(timer);timer=setTimeout(publish,650);};
 input.onblur=remember;
 $('#insert').onclick=()=>{const name=$('#ingredient').value;if(data.names.includes(name)){insert(name);$('.notice').textContent='';}else{$('.notice').textContent=data.messages.ingredient;}};
 $('#batch').onclick=()=>insert('batch size');
 for(const [label,text] of [['+',' + '],['−',' - '],['×',' * '],['÷',' / '],['(','('],[')',')']]){
   const b=document.createElement('button');b.textContent=label;b.type='button';b.setAttribute('aria-label',data.messages.operator+' '+label);b.onclick=()=>insert(text);$('#operators').appendChild(b);
 }
 $('#percentage').onclick=()=>{const value=$('#percent').value;if(value!=='' && Number.isFinite(Number(value))){insert(value+'% of '+target.value);}else{$('.notice').textContent=data.messages.percentage;}};
 $('#use').onclick=()=>{clearTimeout(timer);setTriggerValue('apply',input.value);};
}
''')


def _validate(opt, frame):
    errors, plan = opt._plan_ingredient_grid(frame)
    return errors, plan


def _preview(opt, plan, name):
    """A range, not invented measured values; use the model's own expansion."""
    if plan is None:
        return
    with opt._as_proposed(plan['rows'], plan['deleted']):
        var = opt._var_by_name(name)
        if not opt.has_formula(var):
            return
        low, high = opt._form_reach(opt._linear_form(name), opt._batch_size())
        unit = opt.unit_of(name)
        st.caption(wording.calculated_range_caption(f"{low:.2f}", f"{high:.2f}", unit))


def apply_calculation(opt, frame, index, text):
    """One row given a calculation, checked before anything is staged.

    Hands back `(errors, candidate)`: the candidate grid when the project
    can take it, and None with the refusals when it cannot. A function
    rather than a branch inside the dialog, because the dialog is redrawn
    from scratch on every rerun and what it decided has to survive that.
    """
    candidate = frame.copy()
    candidate.at[index, wording.FORMULA_LABEL] = canonical_calculation(text)
    errors, _ = _validate(opt, candidate)
    return (errors, None) if errors else ([], candidate)


def _stage(frame, name=""):
    from ui_helpers import ING_GRID_KEY, park_grid, rekey_grid
    park_grid(ING_GRID_KEY, frame)
    rekey_grid(ING_GRID_KEY)
    st.session_state.pop(EDITOR_OPEN, None)
    st.session_state[EDITOR_STAGED] = wording.calculation_staged(name)
    st.rerun(scope='app')


def _close_editor():
    """The ✕, or a click outside. The panel is drawn while EDITOR_OPEN is
    set, so dismissing it has to put that down — otherwise the next rerun
    opens it again."""
    st.session_state.pop(EDITOR_OPEN, None)


@st.dialog(wording.CALCULATION_EDITOR_TITLE, width='large',
           on_dismiss=_close_editor)
def open_editor(opt, frame):
    st.session_state[EDITOR_OPEN] = True
    # Aggregate blends and process settings cannot have an amount calculation.
    eligible = [i for i, row in frame.iterrows()
                if str(row.get(wording.NAME_LABEL) or '').strip()
                and row.get(wording.TYPE_LABEL) != wording.KIND_SETTING
                and row.get(wording.MADE_AS_LABEL) != wording.PREMIX_MADE_AS_WEIGHED]
    if not eligible:
        st.info(wording.CALCULATION_NEEDS_AN_INGREDIENT)
        return
    index = st.selectbox(wording.KIND_INGREDIENT, eligible,
                         format_func=lambda i: str(frame.at[i, wording.NAME_LABEL]),
                         key=f'calculation_row_{opt.project_name}')
    row = frame.loc[index]
    name = str(row[wording.NAME_LABEL])
    source = str(row.get(wording.FORMULA_LABEL) or '')
    key = 'calculation-' + hashlib.sha256(f'{opt.project_name}|{index}|{source}'.encode()).hexdigest()[:20]
    mode = st.radio(wording.CALCULATION_MODE_QUESTION, MODES,
                    index=1 if formula_is_rest(source) else 2 if source else 0,
                    key=key+'_mode', horizontal=True)
    candidate = frame.copy()
    apply = False
    if mode == MODES[0]:
        st.caption(wording.CALCULATION_RANGE_CAPTION)
        var = next((v for v in opt.variables if v['name']==row.get(GRID_ID)), None)
        bounds = var['bounds'] if var else (0, 100)
        def initial(column, fallback):
            try: return float(row[column])
            except (ValueError, TypeError): return float(fallback)
        left, right = st.columns(2)
        low = left.number_input(wording.LOWEST_LABEL, value=initial(wording.LOWEST_LABEL,bounds[0]),key=key+'_low')
        high = right.number_input(wording.HIGHEST_LABEL, value=initial(wording.HIGHEST_LABEL,bounds[1]),key=key+'_high')
        candidate.at[index, wording.FORMULA_LABEL] = ''
        candidate.at[index, wording.LOWEST_LABEL] = str(low)
        candidate.at[index, wording.HIGHEST_LABEL] = str(high)
        apply = st.button(wording.CALCULATION_USE_RANGE_BUTTON, key=key+'_use_limits')
    elif mode == MODES[1]:
        st.write(wording.CALCULATION_FILL_BODY)
        st.caption(wording.CALCULATION_FILL_CAPTION)
        st.caption(wording.CALCULATION_FILL_EXAMPLE)
        candidate.at[index, wording.FORMULA_LABEL] = f'= {wording.REST_TOKEN}'
        apply = st.button(wording.CALCULATION_USE_FILL_BUTTON, key=key+'_use_fill')
    else:
        names = [str(r[wording.NAME_LABEL]) for i,r in frame.iterrows()
                 if i!=index and r.get(wording.TYPE_LABEL)!=wording.KIND_SETTING
                 and r.get(wording.MADE_AS_LABEL)!=wording.PREMIX_MADE_AS_WEIGHED
                 and str(r.get(wording.NAME_LABEL) or '').strip()]
        # Parts of variable-composition blends are ingredients too.
        for group in opt.premixes.values():
            if group['mode']=='weighed':
                names.extend(p['name'] for p in group['parts'] if p['name']!=name)
        result = _EXPRESSION(data={'expression': '' if formula_is_rest(source) else display_calculation(source),
                                   'names': sorted(set(names)),
                                   'messages': _MESSAGES},key=key+'_expression',
                             on_expression_change=lambda: None,on_apply_change=lambda: None)
        text = result.expression if result.expression is not None else ('' if formula_is_rest(source) else display_calculation(source))
        if result.apply is not None:
            text = result.apply
            apply = True
        candidate.at[index, wording.FORMULA_LABEL] = canonical_calculation(text)
        st.caption(wording.CALCULATION_BATCH_SIZE_CAPTION)
        st.caption(wording.CALCULATION_EXAMPLES_CAPTION)
        with st.expander(wording.CALCULATION_TERMS_LABEL):
            st.markdown(wording.CALCULATION_TERMS_TABLE)
        with st.expander(wording.CALCULATION_SYNTAX_LABEL):
            st.markdown(wording.CALCULATION_SYNTAX_DETAILS)
        if not str(text).strip():
            st.info(wording.CALCULATION_EMPTY_INFO)
            return
    errors, plan = _validate(opt, candidate)
    for row_no, message in errors:
        st.error((str(frame.at[row_no, wording.NAME_LABEL])+': ' if row_no in frame.index else '')+message)
    if not errors:
        _preview(opt, plan, name)
    st.caption(wording.CALCULATION_APPLY_CAPTION)
    if apply and not errors:
        _stage(candidate, name)
