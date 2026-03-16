"""
Test configuration for router unit tests.

Pytest can collect tests from `router/tests/` without always putting the router
project root on `sys.path`, so add it explicitly to keep imports stable.
"""

from __future__ import annotations

import sys
from pathlib import Path


ROUTER_DIR = Path(__file__).resolve().parents[1]

if str(ROUTER_DIR) not in sys.path:
    sys.path.insert(0, str(ROUTER_DIR))
