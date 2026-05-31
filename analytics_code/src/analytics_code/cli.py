"""Command-line interface for the analytics pipeline.

Exposes three sub-commands:

* ``validate-config`` -- load and validate a JSON configuration file.
* ``run-all`` -- execute every stage in :data:`PIPELINE` in order.
* ``run-stage --stage NAME`` -- execute a single stage by name.

Stages are declared in :data:`PIPELINE`. The keys double as user-facing
stage names *and* as output sub-directory names beneath
``config.paths['output_root']``.
"""

from __future__ import annotations

import argparse
from copy import deepcopy
from pathlib import Path

from analytics_code.common import setup_logging
from analytics_code.config import load_config
from analytics_code.data_prep import run_data_prep
from analytics_code.dropout_analysis import run_dropout_analysis
from analytics_code.full_performance import run_full_performance
from analytics_code.missingness_threshold import run_missingness_threshold
from analytics_code.narrative_analysis import run_narrative_analysis
from analytics_code.tiered_performance import run_tiered_performance

# Ordered pipeline of stages. The keys are the user-facing CLI stage names
# and double as the output sub-directory names under ``paths.output_root``.
PIPELINE = (
    ("data_prep", run_data_prep),
    ("dropout_analysis", run_dropout_analysis),
    ("missingness_threshold", run_missingness_threshold),
    ("full_performance", run_full_performance),
    ("narrative_analysis", run_narrative_analysis),
)
STAGE_FUNCTIONS = dict(PIPELINE)
STAGE_FUNCTIONS["tiered_performance"] = run_tiered_performance


def _document_level_config(
    config, *, mode: str = "document", suffix: str = "_document_level"
):
    """Clone ``config`` for a document-level analytics run in a sibling output root."""
    raw = deepcopy(config.raw)
    analysis = dict(raw.get("analysis", {}))
    analysis["truth_mode"] = mode
    raw["analysis"] = analysis

    paths = dict(raw.get("paths", {}))
    output_root = Path(paths["output_root"])
    paths["output_root"] = str(output_root.parent / f"{output_root.name}{suffix}")
    raw["paths"] = paths
    return type(config)(raw=raw, config_path=config.config_path)


def _validation_view_config(
    config,
    *,
    level: str,
    top_level_folder: str,
    truth_mode_value: str,
):
    """Clone ``config`` so the pipeline runs into one validation-level subtree.

    The root output tree remains the user's original dummy-analysis
    folder. Each validation view gets its own top-level directory
    beneath it (``Document``, ``Cumulative``, ``Final``,
    ``Doc2Patient``), and inside that directory the original stage
    folders (``data_prep``, ``full_performance``, etc.) are recreated.
    """
    raw = deepcopy(config.raw)
    analysis = dict(raw.get("analysis", {}))
    analysis["validation_level"] = level
    analysis["truth_mode"] = truth_mode_value
    raw["analysis"] = analysis

    paths = dict(raw.get("paths", {}))
    output_root = Path(paths["output_root"])
    paths["output_root"] = str(output_root / top_level_folder)
    raw["paths"] = paths
    return type(config)(raw=raw, config_path=config.config_path)


def build_parser() -> argparse.ArgumentParser:
    """Build the top-level :class:`argparse.ArgumentParser` for the pipeline.

    Returns
    -------
    argparse.ArgumentParser
        Parser configured with the ``validate-config``, ``run-all`` and
        ``run-stage`` sub-commands. Each requires a ``--config`` path.
    """
    parser = argparse.ArgumentParser(description="Run analysis-code workflows")
    subparsers = parser.add_subparsers(dest="command", required=True)

    validate = subparsers.add_parser("validate-config", help="Validate a config file")
    validate.add_argument("--config", required=True)

    run_all = subparsers.add_parser("run-all", help="Run all analysis stages")
    run_all.add_argument("--config", required=True)

    run_doc_all = subparsers.add_parser(
        "run-document-level-all",
        help="Run all analysis stages with document-level truth labels",
    )
    run_doc_all.add_argument("--config", required=True)

    run_doc_complete = subparsers.add_parser(
        "run-document-level-complete-all",
        help=(
            "Run all analysis stages with the document-level "
            "complete-marker sensitivity variant (restrict to rows whose "
            "relevant document markers are all present)"
        ),
    )
    run_doc_complete.add_argument("--config", required=True)

    run_validation_views = subparsers.add_parser(
        "run-validation-views-all",
        help=(
            "Run the existing analysis pipeline into top-level Document, "
            "Cumulative, Final, and Doc2Patient folders beneath the "
            "configured output root"
        ),
    )
    run_validation_views.add_argument("--config", required=True)

    run_stage = subparsers.add_parser("run-stage", help="Run one analysis stage")
    run_stage.add_argument("--config", required=True)
    run_stage.add_argument("--stage", choices=sorted(STAGE_FUNCTIONS), required=True)

    return parser


def main(argv: list[str] | None = None) -> int:
    """Run the analytics pipeline command-line interface.

    Parameters
    ----------
    argv:
        Optional list of arguments to parse. When ``None`` (the default),
        the arguments are read from :data:`sys.argv`.

    Returns
    -------
    int
        Process exit status. ``0`` on success, ``2`` for an unknown
        command (although argparse normally exits before reaching that
        path).
    """
    setup_logging()
    parser = build_parser()
    args = parser.parse_args(argv)
    config = load_config(args.config)

    if args.command == "validate-config":
        print(f"Config valid: {Path(args.config).resolve()}")
        return 0

    if args.command == "run-all":
        for _, fn in PIPELINE:
            fn(config)
        return 0

    if args.command == "run-document-level-all":
        document_config = _document_level_config(config)
        for _, fn in PIPELINE:
            fn(document_config)
        return 0

    if args.command == "run-document-level-complete-all":
        document_config = _document_level_config(
            config,
            mode="document_complete",
            suffix="_document_level_complete",
        )
        for _, fn in PIPELINE:
            fn(document_config)
        return 0

    if args.command == "run-validation-views-all":
        validation_runs = (
            ("document", "Document", "document"),
            ("cumulative", "Cumulative", "document"),
            ("final", "Final", "patient"),
            ("doc2patient", "Doc2Patient", "patient"),
        )
        for level, folder, mode in validation_runs:
            derived = _validation_view_config(
                config,
                level=level,
                top_level_folder=folder,
                truth_mode_value=mode,
            )
            for _, fn in PIPELINE:
                fn(derived)
        return 0

    if args.command == "run-stage":
        STAGE_FUNCTIONS[args.stage](config)
        return 0

    parser.error("Unknown command")
    return 2
