#!/usr/bin/env bash
# run_dairy_overnight.sh
#
# Run all 10 dairy category experiments overnight with budget=50.
# Logs stdout/stderr to a timestamped file and prints a summary on completion.
#
# Usage:
#   nohup ./experiments/run_dairy_overnight.sh &
#   # or simply:
#   ./experiments/run_dairy_overnight.sh

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
ROOT_DIR="$(dirname "$SCRIPT_DIR")"
LOG_DIR="${ROOT_DIR}/logs"
mkdir -p "$LOG_DIR"

TIMESTAMP="$(date +%Y%m%d_%H%M%S)"
LOG_FILE="${LOG_DIR}/dairy_overnight_${TIMESTAMP}.log"

BUDGET=50
SEEDS=5
N_INIT=5

CATEGORIES=(
    "barista milk"
    "butter"
    "cheddar"
    "coffee creamer"
    "cream cheese"
    "ice cream"
    "milk"
    "mozzarella"
    "sour cream"
    "yogurt"
)

echo "============================================================"
echo "  Dairy Experiments – Overnight Run"
echo "  Budget:     ${BUDGET}"
echo "  Seeds:      ${SEEDS}"
echo "  Categories: ${#CATEGORIES[@]}"
echo "  Log file:   ${LOG_FILE}"
echo "  Started:    $(date)"
echo "============================================================"

# Run each category sequentially so a failure in one does not skip the rest.
# Results are appended to the same JSON via the main script when run with --categories.
FAILED=()
for cat in "${CATEGORIES[@]}"; do
    echo ""
    echo ">>> Starting: ${cat}  ($(date))"
    if python "${SCRIPT_DIR}/run_dairy_experiments.py" \
        --budget "$BUDGET" \
        --seeds "$SEEDS" \
        --n-init "$N_INIT" \
        --categories "$cat" \
        2>&1 | tee -a "$LOG_FILE"; then
        echo ">>> Finished: ${cat}  ($(date))"
    else
        echo ">>> FAILED:   ${cat}  ($(date))" | tee -a "$LOG_FILE"
        FAILED+=("$cat")
    fi
done

echo ""
echo "============================================================"
echo "  Overnight run complete: $(date)"
echo "  Log: ${LOG_FILE}"
if [ ${#FAILED[@]} -eq 0 ]; then
    echo "  All 10 categories succeeded."
else
    echo "  FAILED categories (${#FAILED[@]}):"
    for f in "${FAILED[@]}"; do
        echo "    - $f"
    done
fi
echo "============================================================"
