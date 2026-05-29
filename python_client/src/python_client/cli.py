"""Command-line interface for the chronological experiment client.

Provides two sub-commands: ``run`` to execute the configured
chronological matrix and ``validate-config`` to verify a configuration
file without dispatching any requests.
"""

from __future__ import annotations

import argparse
import logging
from pathlib import Path

from python_client.config import ConfigError, load_config
from python_client.experiment_runner import ensure_dir, run_chronological_experiments


def setup_logging(
    log_dir: str | Path = "logs", log_file: str = "chronology_client.log"
) -> None:
    """Configure root logging with file + console handlers.

    Parameters
    ----------
    log_dir:
        Directory in which the log file is created (created on demand).
    log_file:
        Name of the log file inside ``log_dir``.
    """
    log_dir = ensure_dir(log_dir)
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)
    if logger.handlers:
        return

    file_handler = logging.FileHandler(log_dir / log_file, encoding="utf-8")
    console_handler = logging.StreamHandler()
    formatter = logging.Formatter("%(asctime)s - %(levelname)s - %(message)s")

    file_handler.setFormatter(formatter)
    console_handler.setFormatter(formatter)
    file_handler.setLevel(logging.INFO)
    console_handler.setLevel(logging.INFO)

    logger.addHandler(file_handler)
    logger.addHandler(console_handler)


def _build_parser() -> argparse.ArgumentParser:
    """Construct the argument parser for the client CLI."""
    parser = argparse.ArgumentParser(
        description="Chronological experiment runner for Gradio-hosted APIs"
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    run_parser = subparsers.add_parser("run", help="Run chronological experiments")
    run_parser.add_argument(
        "--config", required=True, help="Path to the JSON config file"
    )

    validate_parser = subparsers.add_parser(
        "validate-config", help="Validate a config file"
    )
    validate_parser.add_argument(
        "--config", required=True, help="Path to the JSON config file"
    )

    return parser


def main() -> None:
    """Run the chronological client CLI.

    Reads command-line arguments, configures logging, loads and
    validates the requested config, and either reports validation
    success or dispatches the chronological experiment run.
    """
    parser = _build_parser()
    args = parser.parse_args()
    setup_logging()

    try:
        config = load_config(args.config)
    except ConfigError as exc:
        logging.error("Configuration error: %s", exc)
        raise SystemExit(2) from exc

    if args.command == "validate-config":
        logging.info("Configuration is valid: %s", Path(args.config).resolve())
        return

    logging.info(
        "Starting chronological experiment run using %s", Path(args.config).resolve()
    )
    run_chronological_experiments(config)
    logging.info("Chronological experiment run completed.")
