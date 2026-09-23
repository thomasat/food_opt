"""The browser walkthroughs: collected like every other test, and skipped —
never errored — unless a served copy of the app is named in APP_URL.

    APP_URL=http://127.0.0.1:18724 pytest tests/browser

They drive the real canvas grids in Chrome, which a Streamlit AppTest
cannot do at all, so they are the only checks that see what a reader sees.
Without APP_URL, and on a machine with no Playwright, `pytest` still runs
the whole suite green.
"""
import os
import sys

import pytest

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))


@pytest.fixture(scope="session")
def playwright_session():
    pytest.importorskip("playwright.sync_api")
    from playwright.sync_api import sync_playwright
    with sync_playwright() as pw:
        yield pw
