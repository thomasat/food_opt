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

FILL = "Fill to total"
MODES = ["Vary between limits", "Fill to total", "Calculate an amount"]


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
        return '= rest'
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
    html='''<div class="editor">
<label for="expression">Calculation</label>
<div class="entry"><span aria-hidden="true">=</span><textarea id="expression" aria-label="Calculation" rows="3" spellcheck="false" aria-describedby="hint" placeholder="Enter a calculation or insert an ingredient below"></textarea></div>
<div id="suggestions" role="group" aria-label="Ingredient suggestions"></div>
<p id="hint">Use ingredient amounts, batch size, numbers and + − × ÷. Start typing an ingredient name for suggestions.</p>
<div class="insert"><label for="ingredient">Insert ingredient</label><input id="ingredient" aria-label="Insert ingredient" type="search" list="ingredients" placeholder="Search ingredients"><datalist id="ingredients"></datalist><button id="insert">Insert ingredient</button><button id="batch" title="Total ingredient amount for one formulation">Batch size</button></div>
<div id="operators" role="group" aria-label="Insert an operator"></div>
<div class="percent"><label for="percent">Percentage</label><input id="percent" aria-label="Percentage" type="number" placeholder="e.g. 5" step="any"><span> % of </span><select id="of" aria-label="Percentage of"></select><button id="percentage">Insert percentage</button></div>
<p class="notice" role="status" aria-live="polite"></p>
<button id="use" class="primary">Use calculation</button>
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
 $('#insert').onclick=()=>{const name=$('#ingredient').value;if(data.names.includes(name)){insert(name);$('.notice').textContent='';}else{$('.notice').textContent='Choose an ingredient from the list.';}};
 $('#batch').onclick=()=>insert('batch size');
 for(const [label,text] of [['+',' + '],['−',' - '],['×',' * '],['÷',' / '],['(','('],[')',')']]){
   const b=document.createElement('button');b.textContent=label;b.type='button';b.setAttribute('aria-label','Insert '+label);b.onclick=()=>insert(text);$('#operators').appendChild(b);
 }
 $('#percentage').onclick=()=>{const value=$('#percent').value;if(value!=='' && Number.isFinite(Number(value))){insert(value+'% of '+target.value);}else{$('.notice').textContent='Enter a percentage.';}};
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
        st.caption(f"Calculated range: {low:g}–{high:g} {unit}, based on the current ingredient limits. This is not a measured result.")


def _stage(frame):
    from ui_helpers import ING_GRID_KEY, park_grid, rekey_grid
    park_grid(ING_GRID_KEY, frame)
    rekey_grid(ING_GRID_KEY)
    st.rerun(scope='app')


@st.dialog('Edit calculation', width='large')
def open_editor(opt, frame):
    # Aggregate blends and process settings cannot have an amount calculation.
    eligible = [i for i, row in frame.iterrows()
                if str(row.get(wording.NAME_LABEL) or '').strip()
                and row.get(wording.TYPE_LABEL) != wording.KIND_SETTING
                and row.get(wording.MADE_AS_LABEL) != wording.PREMIX_MADE_AS_WEIGHED]
    if not eligible:
        st.info('Add an ingredient and save it before creating a calculation.')
        return
    index = st.selectbox('Ingredient', eligible,
                         format_func=lambda i: str(frame.at[i, wording.NAME_LABEL]),
                         key=f'calculation_row_{opt.project_name}')
    row = frame.loc[index]
    name = str(row[wording.NAME_LABEL])
    source = str(row.get(wording.FORMULA_LABEL) or '')
    key = 'calculation-' + hashlib.sha256(f'{opt.project_name}|{index}|{source}'.encode()).hexdigest()[:20]
    mode = st.radio('How is this amount determined?', MODES,
                    index=1 if formula_is_rest(source) else 2 if source else 0,
                    key=key+'_mode', horizontal=True)
    candidate = frame.copy()
    apply = False
    if mode == MODES[0]:
        st.caption('The app chooses an amount between Lowest and Highest. Use the same value in both to keep it fixed.')
        var = next((v for v in opt.variables if v['name']==row.get(GRID_ID)), None)
        bounds = var['bounds'] if var else (0, 100)
        def initial(column, fallback):
            try: return float(row[column])
            except (ValueError, TypeError): return float(fallback)
        left, right = st.columns(2)
        low = left.number_input('Lowest', value=initial(wording.LOWEST_LABEL,bounds[0]),key=key+'_low')
        high = right.number_input('Highest', value=initial(wording.HIGHEST_LABEL,bounds[1]),key=key+'_high')
        candidate.at[index, wording.FORMULA_LABEL] = ''
        candidate.at[index, wording.LOWEST_LABEL] = str(low)
        candidate.at[index, wording.HIGHEST_LABEL] = str(high)
        apply = st.button('Use limits', key=key+'_use_limits')
    elif mode == MODES[1]:
        st.write('This ingredient fills what remains after all other ingredients are added.')
        st.caption('Batch size is the total ingredient amount for one formulation. Only one ingredient can fill the remaining amount.')
        st.caption('Example: batch size 100 g − other ingredients 72 g = 28 g.')
        candidate.at[index, wording.FORMULA_LABEL] = '= rest'
        apply = st.button('Use fill to total', key=key+'_use_fill')
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
                                   'names': sorted(set(names))},key=key+'_expression',
                             on_expression_change=lambda: None,on_apply_change=lambda: None)
        text = result.expression if result.expression is not None else ('' if formula_is_rest(source) else display_calculation(source))
        if result.apply is not None:
            text = result.apply
            apply = True
        candidate.at[index, wording.FORMULA_LABEL] = canonical_calculation(text)
        st.caption('Batch size means the total ingredient amount for one formulation, including this ingredient.')
        st.caption('Examples explain the arithmetic, not recommended ingredient ratios. Choose values for your own protocol.')
        with st.expander('Supported calculations'):
            st.markdown(wording.CALCULATION_TERMS_TABLE)
        with st.expander(wording.CALCULATION_SYNTAX_LABEL):
            st.markdown(wording.CALCULATION_SYNTAX_DETAILS)
        if not str(text).strip():
            st.info('Enter a calculation or choose a different method above.')
            return
    errors, plan = _validate(opt, candidate)
    for row_no, message in errors:
        st.error((str(frame.at[row_no, wording.NAME_LABEL])+': ' if row_no in frame.index else '')+message)
    if not errors:
        _preview(opt, plan, name)
    st.caption('Use this choice to update the table, then select Save changes. Closing this panel leaves the table unchanged.')
    if apply and not errors:
        _stage(candidate)
