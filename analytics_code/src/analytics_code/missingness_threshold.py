"""Missingness, threshold and calibration analysis.

Reads the per-row merged outputs from :mod:`data_prep` and emits the
artefacts organised under ``missingness_threshold``:

- ``descriptive_stats/``       missingness + grouped numeric summaries + dropout bar plots
- ``macro_f1_calibration/``    per-model threshold sweep tables + curves
- ``confidence_complexity_plots/``  accuracy vs certainty/complexity by bucket
- ``calibration_per_model/``   reliability diagrams + Brier/ECE/MCE summary
"""

from __future__ import annotations

import logging
import shutil
from pathlib import Path

from analytics_code._missingness_threshold_calibration import _per_model_calibration
from analytics_code._missingness_threshold_descriptive import (
    _confidence_complexity,
    _descriptive_stats,
)
from analytics_code._missingness_threshold_shared import (
    _detect_truth_column,
    _load_frame,
)
from analytics_code._missingness_threshold_thresholds import (
    _basic_threshold_tables,
    _macro_f1_calibration,
)
from analytics_code.common import ensure_dir, remove_tree
from analytics_code.config import AnalysisConfig

LOGGER = logging.getLogger("analytics_code.missingness_threshold")


def _clear_truth_dependent_outputs(stage_dir: Path) -> None:
    """Remove stale truth-dependent artefacts when calibration is skipped."""
    for name in (
        "basic_thresholds",
        "macro_f1_calibration",
        "confidence_complexity_plots",
        "calibration_per_model",
    ):
        target = stage_dir / name
        if target.exists():
            remove_tree(target)


def run_missingness_threshold(config: AnalysisConfig) -> dict[str, Path]:
    """Run the missingness, threshold and calibration stage.

    Loads the per-row dataframe written by ``data_prep`` and emits
    descriptive statistics, macro-F1 threshold curves, accuracy /
    confidence / complexity tables, and per-model reliability
    diagrams. Stages that depend on the ground-truth column are
    skipped silently when no truth column is available.

    Parameters
    ----------
    config:
        The active :class:`AnalysisConfig`.

    Returns
    -------
    dict[str, pathlib.Path]
        Mapping with at least ``stage_dir``.
    """
    stage_dir = ensure_dir(config.paths["output_root"] / "missingness_threshold")
    frame, prepared_truth_col, require_prepared_truth = _load_frame(config)
    if frame.empty:
        return {"stage_dir": stage_dir}

    truth_col = _detect_truth_column(
        frame,
        prepared_truth_col,
        require_prepared_truth=require_prepared_truth,
    )
    if truth_col is None:
        LOGGER.warning("No truth column found; skipping calibration stages")
        _clear_truth_dependent_outputs(stage_dir)

    _descriptive_stats(frame, stage_dir)
    if truth_col:
        _basic_threshold_tables(frame, truth_col, stage_dir)
        _macro_f1_calibration(frame, truth_col, stage_dir)
        _confidence_complexity(frame, truth_col, stage_dir)
        _per_model_calibration(frame, truth_col, stage_dir)

    return {"stage_dir": stage_dir}
