"""Analytics pipeline package.

This package implements the post-hoc analysis pipeline that consumes the
batched LLM outputs produced by ``python_client`` and turns them into the
figures and tables used for downstream reporting.

The public entry point is the command-line interface in :mod:`analytics_code.cli`,
which orchestrates the five-stage pipeline (``data_prep``, ``dropout_analysis``,
``missingness_threshold``, ``full_performance``, ``narrative_analysis``).
"""

__all__ = ["__version__"]

__version__ = "0.1.0"
