# analytics_code

Multi-stage analysis pipeline that consumes the batched outputs produced
by [`python_client/`](../python_client/) and reproduces the tables and
figures.

## Pipeline stages

| Stage | Module | Description |
| ----- | ------ | ----------- |
| 1 | `analytics_code.data_prep` | Total study cohort |
| 2 | `analytics_code.dropout_analysis` | Response dropout |
| 3 | `analytics_code.missingness_threshold` | Threshold establishment and Grey-zone performance |
| 4 | `analytics_code.full_performance` | Performance variance by factor |
| 5 | `analytics_code.narrative_analysis` | Narrative analysis |

Each stage reads its configuration from a single JSON file and writes
its outputs to a stage-specific subdirectory under the configured output
root. Stages can be run individually or as a single end-to-end pipeline.

Demographic comparisons (chi-squared and Mann-Whitney U tests against
the physician-validated patient cohort) are not part of this public
pipeline: those analyses were performed against patient and document
level clinical data and the inputs required to reproduce them are not
shipped with the demo.

## Layout

```text
analytics_code/
├── configs/
│   ├── analysis_config.example.json
│   ├── all_models_dummy_analysis.json
│   ├── pseudonymised_dummy_reference.csv
│   └── pseudonymised_dummy_reference_document_flags.csv
├── outputs/                        # generated artefacts (kept out of git)
├── pyproject.toml
├── requirements.txt
└── src/
    └── analytics_code/
        ├── cli.py
        ├── common.py
        ├── config.py
        ├── data_prep.py
        ├── dropout_analysis.py
        ├── full_performance.py
        ├── metrics.py
        ├── missingness_threshold.py
        └── narrative_analysis.py
```

## Installation

The package is plain Python and is not packaged as a Docker image.
Install it into the same virtual environment used for `python_client`:

Linux / macOS

```bash
source .venv/bin/activate
python -m pip install -e analytics_code
```

Windows (PowerShell)

```powershell
.\.venv\Scripts\Activate.ps1
python -m pip install -e analytics_code
```

## Command-line interface

Validate a configuration:

```bash
python -m analytics_code validate-config --config configs/analysis_config.example.json
```

Run the full pipeline (all stages):

```bash
python -m analytics_code run-all --config configs/all_models_dummy_analysis.json
```

Run the document-level variant of the full pipeline (primary rule:
positive if any relevant document marker is positive; negative if any
relevant marker is known and none are positive):

```bash
python -m analytics_code run-document-level-all --config configs/all_models_dummy_analysis.json
```

Run the document-level complete-marker sensitivity variant
(restricts the analysis to rows whose relevant document markers are
*all* present):

```bash
python -m analytics_code run-document-level-complete-all --config configs/all_models_dummy_analysis.json
```

Run a single stage:

```bash
python -m analytics_code run-stage --config configs/analysis_config.example.json --stage full_performance
```

Valid stage names are `data_prep`, `dropout_analysis`,
`missingness_threshold`, `full_performance`, and `narrative_analysis`.

The document-level runner keeps the original patient-level pipeline unchanged and
writes a sibling output tree whose name ends with `_document_level`
(primary rule) or `_document_level_complete` (complete-marker
sensitivity variant). In both modes truth labels are derived from the
boolean document markers present in the merged reference data,
restricted to the document types actually included by each
`report_sequence_name`.

## Configuration

The configuration file controls every input and output path for the
dummy workflow shipped with this repository. The minimum required fields are:

```json
{
  "paths": {
    "llm_batch_root": "../../python_client/outputs/chronology_runs/<version>",
    "human_reference": "./pseudonymised_dummy_reference_document_flags.csv",
    "output_root": "../outputs/<analysis_name>"
  },
  "data_prep": {
    "batch_glob": ["**/batch_*.xlsx", "**/batch_*.parquet", "**/batch_*.csv"],
    "id_column": "patient_id"
  },
  "analysis": { "decision_threshold": 5, "bootstrap_iterations": 1000, "random_seed": 42 },
  "runner_metadata": { "<runner_key>": { "display_name": "...", "temperature": 0.6 } },
  "model_mapping":   { "<display_name>": "<canonical_model>" }
}
```

`runner_metadata` and `model_mapping` together describe how raw runner
folder names should be aggregated into the canonical model identifiers
used in the figures.

### Temperature factor

The production matrix crossed each model with a single serving
temperature, so the pooled `temperature` factor is collinear with
`model` and is suppressed by default in `full_performance`. Per-model
temperature variants (e.g. Mistral-7B at 0.75 and 1.0) are retained and
reported as configuration sensitivity analyses. To re-enable the pooled
temperature contrast for a sensitivity run, set:

```json
{ "analysis": { "enable_temperature_analysis": true } }
```

Results from that variant should be interpreted only as sensitivity
analyses, not as causal estimates of temperature effect.

## Outputs

The pipeline writes one folder per stage under
`paths.output_root`:

```text
<output_root>/
├── data_prep/
├── dropout_analysis/
├── missingness_threshold/
├── full_performance/
└── narrative_analysis/
```

Within each stage folder the artefacts include:

- summary CSVs (cohort, JSON parsing, merge, dropout, calibration);
- metric tables for Brier score, macro F1, MCC, recall, specificity,
  precision, NPV, and accuracy with bootstrapped confidence intervals;
- per-model and per-factor figures rendered as PNG, PDF, and SVG;
- narrative analysis tables covering n-gram contrasts, topic models,
  per-model error profiles, and reasoning-cue audits.

All figures are saved at 600 DPI for publication-quality output.

## Reproducing the dummy analysis

The default configuration `configs/all_models_dummy_analysis.json`
points at the multi-model dummy matrix produced by the runner in
[`python_client/`](../python_client/). After the matrix has been
generated, the analysis can be reproduced from the repository root with:

Linux / macOS

```bash
scripts/run_analytics_pipeline.sh
```

Windows (cmd)

```bat
scripts\run_analytics_pipeline.bat
```

Both scripts default to the bundled dummy analysis configuration.

## Tests

```bash
python -m pytest analytics_code/tests
```

The tests cover the configuration loader, the metric helpers, and the
data-preparation stage, and run without requiring any model output.
