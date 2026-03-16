#!/usr/bin/env bash

set -euo pipefail

# Thin wrapper for running the Python smoke test with one command.
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
exec python3 "${SCRIPT_DIR}/../tests/smoke/test_train_roll_infer.py" "$@"
