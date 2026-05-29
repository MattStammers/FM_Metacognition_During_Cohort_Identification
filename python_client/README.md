# python_client

Chronological experiment runner that drives the Gradio model servers in
[`gradio_api/`](../gradio_api/). The runner merges three streams of
patient documents in chronological order, builds prompts according to the
configured shot type and document sequence, and submits the resulting
batches to one or more model endpoints.

## Scope

The package implements the experiment design:

- chronological linking of primary, secondary, and contextual documents
  per patient
- zero-, single-, and dual-shot prompt templates
- a configurable set of document-order sequences
- parallel dispatch to up to four runner endpoints
- resumable per-batch exports written as `.xlsx`, with automatic
  promotion to `.parquet` above 10 MB; the downstream analytics pipeline
  concatenates these files transparently

The runner is deliberately configuration-driven: report sources are
described in the configuration file rather than hard-coded, which keeps
the bundled dummy workflow easy to reproduce and inspect.

## Layout

```text
python_client/
├── configs/
│   ├── chronological_config.example.json    # general template
│   ├── all_models_dummy_mixtral.json         # one per gradio service
│   ├── all_models_dummy_mixtral_t1.json      # Mixtral T=1.0 sensitivity
│   ├── all_models_dummy_m42.json
│   ├── all_models_dummy_m42_t1.json          # Med42 T=1.0 sensitivity
│   ├── all_models_dummy_deepseek14.json
│   ├── all_models_dummy_deepseek14_t050.json # DeepSeek-14B T=0.5 sensitivity
│   ├── all_models_dummy_deepseek32.json
│   ├── all_models_dummy_qwen32.json
│   └── all_models_dummy_gemma4_31b.json
├── data/                                    # dummy CSV inputs
├── logs/
├── outputs/
├── prompt_templates/
│   ├── zero_shot.txt
│   ├── single_shot.txt
│   └── dual_shot.txt
├── pyproject.toml
├── requirements.txt
└── src/
    └── python_client/
        ├── api.py
        ├── cli.py
        ├── config.py
        ├── data_processing.py
        ├── experiment_runner.py
        └── prompts.py
```

## Installation

The package is plain Python and is not packaged as a Docker image. Install
it into a virtual environment from the repository root:

Linux / macOS

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -e python_client
```

Windows (PowerShell)

```powershell
py -3.12 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -e python_client
```

The console entry point `chronology-client` and the `python -m
python_client` module entry point both invoke the CLI defined in
`src/python_client/cli.py`.

## Configuration model

Every experiment is fully described by a JSON configuration file. The
canonical template is
[`configs/chronological_config.example.json`](configs/chronological_config.example.json).
The file has four sections:

1. `general` — batch size, output version, retry behaviour, token
   budget, and the map from runner key to API base URL.
2. `data_sources` — the three input CSV files and the column names
   used to identify patients, dates, and free-text content.
3. `prompt_templates` — paths to the zero-, single-, and dual-shot
   templates.
4. `report_sequences` — named ordered lists describing each
   document-arrangement experiment.

The matching block (`max_days_between_primary_and_secondary`) controls
the chronological linkage between the primary and secondary report
streams. Context notes are attached as the latest preceding and earliest
following entries within the configured tolerance.

## Data inputs

Three CSV files are required:

| Stream | Default path | Role |
| ------ | ------------ | ---- |
| Primary report | `data/histopathology_filtered_dummy.csv` | Anchor document |
| Secondary report | `data/endoscopy_filtered_dummy.csv` | Linked to primary by date proximity |
| Context notes | `data/clinical_letters_filtered_dummy.csv` | Surrounding clinical narrative |

The dummy CSV files distributed with the repository are deterministic and
contain pseudonymised synthetic content. They reproduce the schema used
in the original study without exposing any clinical text.

## Command-line interface

Validate a configuration without executing it:

```bash
python -m python_client validate-config --config configs/chronological_config.example.json
```

Run the configured experiments:

```bash
python -m python_client run --config configs/chronological_config.example.json
```

The runner is resumable: each invocation skips batches that already exist
on disk for the given runner and experiment, so an interrupted run can be
restarted with the same command.

## Output structure

```text
outputs/
└── chronology_runs/
    └── <version>/
        └── <runner_name>/
            └── <experiment_name>/
                ├── batch_0.xlsx
                ├── batch_1.xlsx
                └── ...
```

`<version>` is taken from `general.version` in the configuration file.
The downstream analysis pipeline expects this exact layout.

## Reproducing the multi-model dummy matrix

Nine per-service configurations under `configs/all_models_dummy_*.json`
together cover every model family, temperature variant, prompt type,
and document sequence in the matrix. Each configuration writes
into the shared output version `all_models_realistic_dummy_v1`, so the
nine runs accumulate into a single matrix.

> **Note on demo limits.** These public dummy configurations use
> reduced values (e.g. `batch_size: 20`, `max_cases: 60`) so that the
> smoke matrix completes quickly on a single workstation. A
> production-style / redacted example documented elsewhere uses larger
> values (for example `batch_size: 5`, `max_cases: 1000`); the schema
> is identical, only the limits differ.

Because the model services are GPU-resident and not all of them fit in
memory simultaneously, the orchestration scripts in `scripts/` start one
service at a time, run the corresponding client configuration, and stop
the service before moving to the next:

Linux / macOS

```bash
../scripts/run_all_models_dummy_matrix.sh
```

Windows (cmd)

```bat
..\scripts\run_all_models_dummy_matrix.bat
```

Both scripts read `HF_ACCESS_TOKEN` from a `.env` file at the repository
root and assume that Docker is available on the host.

## Tests

```bash
python -m pytest python_client/tests
```

The tests cover configuration loading, prompt assembly, document
processing, and the batched experiment runner. They do not require a
running model service.
