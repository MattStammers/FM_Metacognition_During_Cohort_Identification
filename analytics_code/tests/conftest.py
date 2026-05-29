"""Pytest configuration for analytics_code tests.

Forces matplotlib to use the non-interactive Agg backend so tests can run
in headless environments (no display / Tk required).
"""

from __future__ import annotations

import matplotlib

matplotlib.use("Agg")
