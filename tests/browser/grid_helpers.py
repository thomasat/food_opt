"""Canvas-grid interactions shared by the three v0.7.0 browser walkthroughs."""
import os, time
from playwright.sync_api import sync_playwright

URL = os.environ.get("APP_URL", "http://127.0.0.1:18724")
ROW_H = 35
# The ingredients grid, left to right, as the app draws it:
# Name Calculation Lowest Highest Unit Type Preparation [Vendor SKU]
NAME, CALCULATION, LOWEST, HIGHEST, UNIT, TYPE, MADE_AS = 0, 1, 2, 3, 4, 5, 6
PORTIONED = "Pre-mix: keep proportions fixed"
WEIGHED = "Blend: vary each ingredient"
GRID_HEAD = ["Name", "Calculation", "Lowest", "Highest", "Unit", "Type",
             "Preparation"]
# The sample, row by row, in the order the method adds them.
SAMPLE_ROWS = ["Textured pea protein", "Textured soy protein",
               "Hydration water", "Remaining water", "Dry blend",
               "Wheat gluten", "Seasoning blend", "Fats and oils",
               "Mixing time after fat"]


SET_UP_TAB = "1 · Set up"
ROUND_TAB = "2 · Make a round"
RESULTS_TAB = "3 · Results"


def ensure_round(p):
    """A round on the bench, generated if there is not one already. The
    walkthroughs share one served project, so each says what it needs
    rather than assuming what the one before it left."""
    tab(p, ROUND_TAB)
    if p.get_by_role("button", name="Download the round sheets (Excel)",
                     exact=True).count():
        return
    tab(p, SET_UP_TAB)
    click(p, "Generate formulations")
    generate = next(x for x in labels(p)
                    if x.startswith("Generate") and "different" not in x)
    click(p, generate, wait=10)
    tab(p, ROUND_TAB)


def open_sample(pw, viewport=None):
    """A browser on the served copy with the sample project open. Returns
    (browser, page); the caller closes the browser."""
    browser = pw.chromium.launch(channel="chrome", headless=True)
    page = browser.new_context(
        viewport=viewport or {"width": 1280, "height": 860},
        accept_downloads=True).new_page()
    page.goto(URL, wait_until="domcontentloaded")
    settle(page, 5)
    if "Try the sample project" in labels(page):
        click(page, "Try the sample project", wait=6)
    assert not page.locator('[data-testid="stException"]').count()
    return browser, page
rows = []


def check(section, condition, message, observed=""):
    rows.append((section, bool(condition), message, observed))
    print(("ok   " if condition else "FAIL ") + f"[{section}] {message}"
          + (f"  |  observed: {observed!r}" if observed else ""), flush=True)


def note(section, observed):
    print(f"..   [{section}] {observed}", flush=True)


def settle(p, s=1.5):
    time.sleep(s)
    try:
        p.locator('[data-testid="stStatusWidget"]').wait_for(state="hidden",
                                                             timeout=30000)
    except Exception:
        pass
    time.sleep(0.6)


def click(p, name, wait=2.0):
    """One button by its words. A button with a material icon carries the
    ligature text in its name, so an exact match on the words alone finds
    nothing — the words are matched inside the name instead."""
    button = p.get_by_role("button", name=name, exact=True)
    if not button.count():
        button = p.get_by_role("button", name=name, exact=False)
    button.first.click()
    settle(p, wait)


def tick(p, label):
    """One checkbox by its label. Streamlit hides the input itself behind a
    styled label, so the label is what a person clicks and what Playwright
    has to click too."""
    p.locator('[data-testid="stCheckbox"]').filter(
        has_text=label).locator("label").first.click()
    settle(p, 1.5)


def tab(p, name, wait=3.0):
    p.get_by_role("tab", name=name).first.click()
    settle(p, wait)


def labels(p):
    return p.eval_on_selector_all(
        "button", "els=>els.filter(e=>e.offsetParent!==null)"
                  ".map(e=>(e.innerText||'').trim()).filter(Boolean)")


def captions(p):
    loc = p.locator('[data-testid="stCaptionContainer"]')
    return [loc.nth(i).inner_text().strip() for i in range(loc.count())]


def alerts(p, kind):
    return [t.strip() for t in p.locator(
        f'[data-testid="stAlertContent{kind}"]').all_inner_texts()]


def grid(p, n):
    return p.locator('[data-testid="stDataFrame"]').nth(n)


def grid_rows(p, n):
    return p.evaluate("""n => {
      const g = document.querySelectorAll('[data-testid="stDataFrame"]')[n];
      return [...g.querySelectorAll('tbody tr')].map(
          r => [...r.querySelectorAll('td')].map(c => c.innerText));
    }""", n)


def grid_head(p, n):
    return p.evaluate("""n => {
      const g = document.querySelectorAll('[data-testid="stDataFrame"]')[n];
      return [...g.querySelectorAll('thead th')].map(c => c.innerText);
    }""", n)


def grid_named(p, first_heading):
    """The index of the visible grid whose first column is `first_heading` —
    how the parts grid is found, since it is never the nth of anything."""
    return p.evaluate("""heading => {
      const all = [...document.querySelectorAll('[data-testid="stDataFrame"]')];
      for (let i = 0; i < all.length; i++) {
        if (all[i].offsetParent === null) continue;
        const head = [...all[i].querySelectorAll('thead th')].map(
            c => c.innerText);
        if (head[0] === heading) return i;
      }
      return -1;
    }""", first_heading)


def table(p, heading):
    return p.evaluate("""heading => {
      for (const g of document.querySelectorAll('[data-testid="stDataFrame"]')) {
        if (g.offsetParent === null) continue;
        const head = [...g.querySelectorAll('thead th')].map(c => c.innerText);
        if (head[0] !== heading) continue;
        return {head: head,
                body: [...g.querySelectorAll('tbody tr')].map(
                    r => [...r.querySelectorAll('td')].map(c => c.innerText))};
      }
      return {head: [], body: []};
    }""", heading)


def fold(p, title):
    return p.locator('[data-testid="stExpander"]').filter(has_text=title)


def fold_titles(p):
    """What every fold on the page is called. The summary carries the
    chevron's own ligature text, which is not part of the title."""
    loc = p.locator('[data-testid="stExpander"] summary')
    out = []
    for i in range(loc.count()):
        text = loc.nth(i).inner_text().strip().splitlines()[-1]
        for icon in ("keyboard_arrow_right", "keyboard_arrow_down"):
            text = text.replace(icon, "")
        out.append(text.strip())
    return out


def fold_is_open(p, title):
    return fold(p, title).first.evaluate(
        "e => { const d = e.matches('details') ? e : e.querySelector('details');"
        "       return d ? d.open : false; }")


def open_fold(p, title):
    """Open one fold, and leave an open one alone — the summary is a
    toggle, and clicking one that is already open shuts it."""
    if not fold_is_open(p, title):
        fold(p, title).first.locator("summary").first.click()
        settle(p, 1.5)


def fold_click(p, title, name, wait=3.0):
    fold(p, title).first.get_by_role("button", name=name,
                                     exact=True).first.click()
    settle(p, wait)


def to_top(p, n):
    """One grid scrolled to its own first row, and left on screen. The wheel
    is what walks a canvas grid back up; bringing the grid into view again
    afterwards is what keeps a fold half way down the page reachable."""
    g = grid(p, n)
    g.scroll_into_view_if_needed()
    time.sleep(0.4)
    box = g.bounding_box()
    p.mouse.move(box["x"] + 200, box["y"] + min(60, box["height"] / 2))
    p.mouse.wheel(0, -4000)
    time.sleep(0.4)
    g.scroll_into_view_if_needed()
    time.sleep(0.4)


def selection(p, n):
    ids = p.evaluate("""n => {
      const g = document.querySelectorAll('[data-testid="stDataFrame"]')[n];
      return [...g.querySelectorAll('td[aria-selected=\"true\"]')]
             .map(e => e.dataset.testid);
    }""", n)
    if not ids:
        return None
    _, _, column, row = ids[0].split("-")
    return int(row), int(column) - 1


# Where each column of each grid sits, as an offset from the grid's left
# edge. The canvas has no DOM geometry to read — every cell measures 0 by 0
# — and a grid drawn inside a fold takes no keyboard at all, so a cell is
# reached the one way that always works: clicked. The offsets are found the
# way a person would find them, by clicking along a row and seeing which
# column lights up, and they are found once per grid shape.
_COLUMNS = {}


def _measure_columns(p, n):
    to_top(p, n)
    box = grid(p, n).bounding_box()
    found, x = {}, 20.0
    while x < box["width"] - 8:
        p.mouse.click(box["x"] + x, box["y"] + ROW_H * 1.5)
        time.sleep(0.12)
        here = selection(p, n)
        if here is not None and here[1] not in found:
            found[here[1]] = x
        x += 30
    return found


def column_x(p, n, column):
    key = (n, tuple(grid_head(p, n)))
    if key not in _COLUMNS:
        _COLUMNS[key] = _measure_columns(p, n)
    xs = _COLUMNS[key]
    if column not in xs:
        raise AssertionError(f"grid {n}: column {column} is not on screen; "
                             f"found {sorted(xs)}")
    return xs[column]


def walk(p, key, times):
    for _ in range(times):
        p.keyboard.press(key)
        time.sleep(0.25)


def editor_open(p):
    return p.locator("#portal [id^=gdg-overlay]").count() > 0


def walk_to(p, n, row, column):
    """Reach a cell the way a person does — click the first one and walk —
    and say whether it worked. It does not work inside a fold: a grid drawn
    there never takes the keyboard."""
    to_top(p, n)
    box = grid(p, n).bounding_box()
    p.mouse.click(box["x"] + 60, box["y"] + ROW_H * 1.5)
    time.sleep(0.6)
    here = selection(p, n)
    if here is None:
        return False
    walk(p, "ArrowDown", max(0, row - here[0]))
    walk(p, "ArrowUp", max(0, here[0] - row))
    walk(p, "ArrowRight", max(0, column - here[1]))
    walk(p, "ArrowLeft", max(0, here[1] - column))
    return selection(p, n) == (row, column)


def open_cell(p, n, row, column):
    """Select one cell and open its own editor.

    Both ways a person opens one, in the order a person tries them: walk to
    the cell and press Enter, and failing that click it and double click it
    open. The second is what a grid drawn inside a fold needs, because it
    never takes the keyboard; the first is what a select cell needs, whose
    editor a double click opens and shuts again in one gesture."""
    if walk_to(p, n, row, column):
        p.keyboard.press("Enter")
        time.sleep(0.8)
        if editor_open(p):
            return
    to_top(p, n)
    x = column_x(p, n, column)
    box = grid(p, n).bounding_box()
    p.mouse.click(box["x"] + x, box["y"] + ROW_H * (row + 1.5))
    time.sleep(0.4)
    landed = selection(p, n)
    if landed != (row, column):
        raise AssertionError(f"grid {n}: wanted cell {(row, column)}, "
                             f"landed on {landed}")
    p.mouse.dblclick(box["x"] + x, box["y"] + ROW_H * (row + 1.5))
    time.sleep(1.0)
    if not editor_open(p):
        raise AssertionError(f"grid {n}: cell {(row, column)} would not "
                             f"open its editor")


def set_cell(p, n, row, column, value):
    """Type `value` into one cell of grid `n`, by row and column number."""
    open_cell(p, n, row, column)
    p.locator("#portal input, #portal textarea").first.fill(str(value))
    time.sleep(0.3)
    p.keyboard.press("Enter")
    settle(p, 1.2)


def cell_options(p, n, row, column):
    """What a select cell offers, without choosing any of it."""
    open_cell(p, n, row, column)
    out = [t.strip() for t in p.locator('#portal [role="option"]'
                                        ).all_inner_texts()]
    p.keyboard.press("Escape")
    settle(p, 1.0)
    return out


def set_select(p, n, row, column, value):
    """Pick `value` out of a select cell's own list."""
    open_cell(p, n, row, column)
    options = p.locator('#portal [role="option"]')
    for i in range(options.count()):
        if options.nth(i).inner_text().strip() == value:
            options.nth(i).click()
            settle(p, 1.5)
            return
    raise AssertionError(f"grid {n}: {value!r} is not on offer; "
                         f"{options.all_inner_texts()}")


def add_row(p, n):
    """Click the empty line at the bottom of a dynamic grid. Returns its
    row number."""
    before = len(grid_rows(p, n))
    to_top(p, n)
    box = grid(p, n).bounding_box()
    p.mouse.click(box["x"] + 60, box["y"] + ROW_H * (before + 1.5))
    settle(p, 1.2)
    after = len(grid_rows(p, n))
    if after != before + 1:
        raise AssertionError(f"grid {n}: clicking the empty line added "
                             f"{after - before} rows")
    return before


def row_of(p, n, name):
    return [r[0] for r in grid_rows(p, n)].index(name)


