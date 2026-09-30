"""Test-wide settings.

Insights' worker processes are off in the test suite (FOODOPT_INSIGHTS_WORKERS
0): every screen then fits in the test's own process, as 0.8.0 did, so a
test sees the finished tab on its first run. The tests of the workers
start their own (tests/test_080_insights_worker.py and neighbours).

The cache beside the project is off too (FOODOPT_INSIGHTS_CACHE 0), so one
test's app never reads what another wrote; tests/test_080_insights_cache.py
and the speed UI tests turn it on."""
import os

os.environ.setdefault("FOODOPT_INSIGHTS_WORKERS", "0")
os.environ.setdefault("FOODOPT_INSIGHTS_CACHE", "0")
