"""Tests of the shared Python launcher (desktop/shared/launcher). The
package is imported from the repo, so `import launcher` works here."""
import os
import sys

import pytest

from tests.shared_texts import REPO

SHARED = REPO / "desktop" / "shared"
if str(SHARED) not in sys.path:
    sys.path.insert(0, str(SHARED))

slow = pytest.mark.skipif(os.environ.get("FOODOPT_SLOW_TESTS") != "1",
                          reason="set FOODOPT_SLOW_TESTS=1")
