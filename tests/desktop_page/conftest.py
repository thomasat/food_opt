"""Drive desktop/shared/pages/status.html in Chromium (as WebView2 runs it)
and WebKit (as WKWebView runs it), with the page's clock under the test's
control. Skipped, never errored, when Playwright or a browser is missing."""
import json

import pytest

from tests.shared_texts import REPO, texts

PAGE_URL = (REPO / "desktop" / "shared" / "pages" / "status.html").as_uri()


@pytest.fixture(scope="package")
def _playwright():
    pytest.importorskip("playwright.sync_api")
    from playwright.sync_api import sync_playwright
    with sync_playwright() as pw:
        yield pw


# Package scope, not session: the sync Playwright loop must be gone before
# the rest of the suite (Streamlit's AppTest runs its own asyncio loop).
@pytest.fixture(scope="package", params=["chromium", "webkit"])
def browser(request, _playwright):
    try:
        b = getattr(_playwright, request.param).launch()
    except Exception as err:   # the browser executable is not installed
        pytest.skip(f"{request.param} is not available: {err}")
    yield b
    b.close()


_BRIDGE = {
    "chromium": "window.chrome = {webview: {postMessage: function (m) { window.__posted.push(m); }}};",
    "webkit": ('Object.defineProperty(window, "webkit", {configurable: true, value: '
               '{messageHandlers: {foodopt: {postMessage: function (m) { window.__posted.push(m); }}}}});'),
}
_NO_BRIDGE = ('try { delete window.chrome; } catch (e) {}'
              'Object.defineProperty(window, "webkit", {configurable: true, value: undefined});')


@pytest.fixture
def open_page(browser):
    contexts = []

    def _open(os_name="mac", view=None, version="0.9.0", bridge=True):
        ctx = browser.new_context()
        contexts.append(ctx)
        page = ctx.new_page()
        page.clock.install(time=0)
        page.clock.pause_at(0)   # time moves only when a test moves it
        boot = {"messages": texts(os_name), "version": version, "os": os_name}
        if view is not None:
            boot["view"] = view
        script = f"window.FOODOPT_BOOT = {json.dumps(boot)}; window.__posted = [];"
        script += _BRIDGE[browser.browser_type.name] if bridge else _NO_BRIDGE
        page.add_init_script(script)
        page.errors = []
        page.on("pageerror", lambda err: page.errors.append(err))
        page.goto(PAGE_URL)
        return page

    yield _open
    for ctx in contexts:
        ctx.close()


def visible(page, sel) -> bool:
    return page.is_visible(sel)


def text(page, sel) -> str:
    return page.eval_on_selector(sel, "e => e.textContent.trim()")


def call(page, js):
    return page.evaluate(f"FoodOpt.{js}")
