# FM Metacognition During a Cohort Identification Task

### Subtitle: IBD_NLP_Cohort_Identification_IC-IBD_Part_3

## By Matt Stammers

### Final Code Completed: 31/05/2026

## Purpose

Accompanyment to papers and other related artefacts to maximise reproducibility - (associated artefacts to follow).

## Core Question?

How do LLMs or Foundation Models (FMs) think when performing clinical tasks?

## What is this repo?

IBD FM metacognition - a self-contained, runnable demo of the
end-to-end evaluation workflow: containerised open-weight model
servers, a chronological experiment runner, and a multi-stage
analytics pipeline. The code is designed to run standalone on the
bundled synthetic data so that the workflow can be inspected and
reproduced without access to any clinical data. It has been tested on
Ubuntu Linux (preferred) and on Windows.

## At a glance
- Python difficulty: intermediate to advanced; familiarity with virtual
  environments and the command line is assumed.
- Primary purpose: a transparent, reproducible reference implementation
  that others can rerun, adapt, and extend.

## How to Use Yourself

For a basic primer on using Python and setting it up for the first time: [Python Starter Guide](https://mattstammers.github.io/hdruk_avoidable_admissions_collaboration_docs/how_to_guides/new_to_python).

Analysts must prepare the environment appropriately. If you are new to
Python and working in a healthcare context, see this short
introduction: [NHS BI Analysts Python for Data Science Intro](https://github.com/MattStammers/Community_Of_Practice_Session_Two).

The repository is organised into three independent components:

| Folder | Purpose |
| ------ | ------- |
| [`gradio_api/`](gradio_api/) | Containerised Gradio model servers (one per open-weight LLM) |
| [`python_client/`](python_client/) | Chronological experiment runner that calls the model servers |
| [`analytics_code/`](analytics_code/) | Multi-stage analysis pipeline (data prep, dropout, missingness threshold, full performance, narrative analysis) that consumes the runner outputs |

Each component has its own README with detailed usage. The sections below
describe how the pieces fit together and how to reproduce the end-to-end
workflow on a fresh machine.

## Models evaluated

The code targets six open-weight model families. Two of the non-thinking
families (Mistral-7B and Med42-8B) are evaluated at two temperatures each
and DeepSeek-14B is also evaluated at two temperatures, giving nine
runner endpoints in total:

| Runner key | Hugging Face model | Temperature | Default port |
| ---------- | ------------------ | ----------- | ------------ |
| `mixtral_runner` | `mistralai/Mistral-7B-Instruct-v0.3` | 0.75 | 9001 |
| `mixtral_t1_runner` | `mistralai/Mistral-7B-Instruct-v0.3` | 1.00 | 9011 |
| `m42_runner` | `m42-health/Llama3-Med42-8B` | 0.75 | 9002 |
| `m42_t1_runner` | `m42-health/Llama3-Med42-8B` | 1.00 | 9012 |
| `deepseek14_runner` | `deepseek-ai/DeepSeek-R1-Distill-Qwen-14B` | 0.75 | 9003 |
| `deepseek14_t05_runner` | `deepseek-ai/DeepSeek-R1-Distill-Qwen-14B` | 0.50 | 9013 |
| `deepseek32_runner` | `deepseek-ai/DeepSeek-R1-Distill-Qwen-32B` | 0.60 | 9004 |
| `qwen32_runner` | `Qwen/Qwen3-32B` | 0.60 | 9005 |
| `gemma4_31b_runner` | `google/gemma-4-31b-it` | 0.60 | 9006 |

All models are loaded in FP16 (`torch.float16`, `device_map="auto"`) with
no quantisation applied. The actual model identifiers used at load time
are defined in the corresponding script under
[`gradio_api/src/`](gradio_api/src/).

## Repository layout

```text
.
├── README.md                  # this file
├── gradio_api/                # Docker-based Gradio model servers
├── python_client/             # chronological experiment runner (no Docker)
├── analytics_code/            # analysis pipeline (no Docker)
├── scripts/                   # cross-platform orchestration wrappers
│   ├── run_all_models_dummy_matrix.sh
│   ├── run_all_models_dummy_matrix.bat
│   ├── run_analytics_pipeline.sh
│   ├── run_analytics_pipeline.bat
│   └── generate_dummy_pipeline_data.py
└── archive/                   # historical outputs (excluded from packaging)
```

Only the model servers are containerised. The Python client and the
analytics pipeline are standard Python packages and run in a local virtual
environment on Linux, macOS, and Windows.

## End-to-end workflow

1. Start one or more model servers from `gradio_api/` using Docker
   Compose.
2. Configure the Python client to point at the running endpoints and run
   the chronological experiment matrix.
3. Point the analytics pipeline at the resulting batch files to produce
   the figures and tables.

## 1. Environment

Python 3.12 is recommended. From the repository root:

Linux / macOS

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -e python_client
python -m pip install -e analytics_code
pre-commit install
```

Windows (PowerShell)

```powershell
py -3.12 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -e python_client
python -m pip install -e analytics_code
```

The Gradio servers run inside Docker and do not depend on this virtual
environment. The client and analytics packages are installed in editable
mode so that changes propagate without reinstalling.

## 2. Hugging Face access token

A Hugging Face access token is required for any gated weights. Create a
file named `.env` at the repository root with at least:

```env
HF_ACCESS_TOKEN=hf_your_token_here
```

This file is read by `docker compose` when starting the model servers and
by the orchestration scripts in `scripts/`. It is excluded from version
control by `.gitignore`.

## 3. Starting the model servers

See [`gradio_api/README.md`](gradio_api/README.md) for the full set of
options. The minimal command from the repository root is:

```bash
cd gradio_api
cp .env.example .env          # then set HF_ACCESS_TOKEN
docker compose --profile mixtral up --build -d
```

Each configured service profile (`mixtral`, `mixtral_t1`, `m42`, `m42_t1`,
`deepseek14`, `deepseek14_t05`, `deepseek32`, `qwen32`, `gemma4_31b`)
starts one server on the port listed in the table above. Servers bind to
`127.0.0.1` by default.

## 4. Running the chronological experiment

See [`python_client/README.md`](python_client/README.md) for full
configuration details. To reproduce the multi-model dummy matrix used as a
smoke test:

Linux / macOS

```bash
scripts/run_all_models_dummy_matrix.sh
```

Windows (cmd)

```bat
scripts\run_all_models_dummy_matrix.bat
```

The script starts each Docker profile in turn, runs the corresponding
client configuration once the endpoint is reachable, then shuts the
service down before moving to the next phase. All nine phases
(`mixtral` T=0.75, `mixtral_t1` T=1.0, `m42` T=0.75, `m42_t1` T=1.0,
`deepseek14` T=0.75, `deepseek14_t05` T=0.5, `deepseek32` T=0.6,
`qwen32` T=0.6, `gemma4_31b` T=0.6) accumulate into
`python_client/outputs/chronology_runs/all_models_realistic_dummy_v1/`.

## 5. Running the analysis pipeline

See [`analytics_code/README.md`](analytics_code/README.md) for stage
descriptions and CLI options. Both the patient-level (`run-all`) and
document-level (`run-document-level-all`) passes are executed
sequentially, writing to sibling output directories.

### Easiest: containerised (no local Python required)

If you have Docker installed, the quickstart wrapper runs everything
inside a self-contained image, sidestepping local venv setup,
OneDrive sync issues, and Windows MAX_PATH problems:

Linux / macOS

```bash
scripts/quickstart_analytics.sh
```

Windows (cmd / PowerShell)

```bat
scripts\quickstart_analytics.bat
```

The wrapper auto-detects Docker and falls back to the native venv
runner below if Docker is unavailable.

You can also drive the container directly:

```bash
docker compose -f scripts/docker-compose.analytics.yml run --rm analytics
```

#### Zero-Python beginner walkthrough (Windows + WSL2)

This path requires no prior Python knowledge and no local Python
installation. Everything runs inside a Linux container. The full
analytics pipeline (data prep, dropout, missingness/threshold, full
performance with primary-estimand and strict-marker sensitivity passes,
and narrative analysis) executes end to end and writes results back to
`analytics_code/outputs/` on your Windows drive.

Validated on Windows 11 with WSL2 (Ubuntu 24.04) and Docker Engine
29.1.3 installed inside the WSL distro; total wall time on a powerful
workstation is roughly fifteen minutes for the full dummy matrix.

1. **Install WSL2 and an Ubuntu distribution.** Open *PowerShell as
   administrator* and run:

   ```powershell
   wsl --install Ubuntu-24.04
   ```

   Reboot if prompted. On first launch Ubuntu will ask you to create a
   username and password; pick anything you like.

2. **Enable WSL mirrored networking** (so the container can reach
   Docker Hub through your normal corporate or home network). Create the
   file `%USERPROFILE%\.wslconfig` with this content:

   ```ini
   [wsl2]
   networkingMode=mirrored
   dnsTunneling=true
   autoProxy=true
   firewall=true
   ```

   Then shut WSL down so the new settings take effect:

   ```powershell
   wsl --shutdown
   ```

3. **Install Docker Engine inside Ubuntu.** Open the Ubuntu shell (just
   type `ubuntu` in the Start menu) and run:

   ```bash
   sudo apt-get update
   sudo apt-get install -y docker.io
   sudo service docker start
   sudo usermod -aG docker "$USER"   # optional: avoids needing sudo
   ```

   Log out of the Ubuntu shell and back in once for the group change to
   take effect. Verify the engine is up:

   ```bash
   docker version
   ```

4. **(Corporate networks only)** If your organisation inspects HTTPS
   traffic, import your Windows certificate trust store into Ubuntu so
   image pulls succeed:

   ```powershell
   # In PowerShell on the host
   $pem = "$env:TEMP\corp-roots.pem"
   Get-ChildItem Cert:\LocalMachine\Root, Cert:\LocalMachine\AuthRoot |
     ForEach-Object {
       "-----BEGIN CERTIFICATE-----`n" +
       [Convert]::ToBase64String($_.RawData, 'InsertLineBreaks') +
       "`n-----END CERTIFICATE-----`n"
     } | Set-Content -Encoding ASCII $pem
   ```

   ```bash
   # Then in the Ubuntu shell
   sudo mkdir -p /usr/local/share/ca-certificates/corp
   sudo cp /mnt/c/Users/$USER/AppData/Local/Temp/corp-roots.pem \
           /usr/local/share/ca-certificates/corp/roots.crt
   sudo update-ca-certificates
   sudo service docker restart
   ```

5. **Run the pipeline.** From the Ubuntu shell, navigate to wherever
   you cloned this repository (Windows paths appear under `/mnt/c/`)
   and run:

   ```bash
   cd "/mnt/c/path/to/llm_metacognition_study"
   docker build -f scripts/Dockerfile.analytics -t llm-metacognition-analytics .
   docker run --rm -v "$PWD:/workspace" llm-metacognition-analytics
   ```

   The first build takes a few minutes (it downloads a Python base
   image and installs the analytics dependencies); subsequent runs
   reuse the cached image and complete the full dummy pipeline end to
   end in roughly fifteen minutes.

6. **Inspect the results** in Windows Explorer under:

   - `analytics_code/outputs/all_models_realistic_dummy_analysis/`
     (patient-level)
   - `analytics_code/outputs/all_models_realistic_dummy_analysis_document_level/`
     (document-level, lenient truth)
   - `analytics_code/outputs/all_models_realistic_dummy_analysis_document_level_complete/`
     (document-level, strict-marker sensitivity)

   Each tree contains stage-specific CSVs, PNG/PDF/SVG figures at
   600 DPI, and the bootstrap confidence intervals described in
   [`analytics_code/README.md`](analytics_code/README.md).

Common gotchas:

- *"Cannot connect to the Docker daemon"* — the daemon stopped when the
  Ubuntu shell closed. Run `sudo service docker start` and retry.
- *Pulls hang or get reset by the firewall* — confirm `.wslconfig`
  contains `networkingMode=mirrored` and you ran `wsl --shutdown`. On
  very restrictive networks, you may also need step 4.
- *"x509: certificate signed by unknown authority"* — repeat step 4 to
  install your corporate root certificates inside Ubuntu.

### Native venv runner

Linux / macOS

```bash
scripts/run_analytics_pipeline.sh
```

Windows (cmd)

```bat
scripts\run_analytics_pipeline.bat
```

By default this consumes the dummy matrix output and writes results to
`analytics_code/outputs/all_models_realistic_dummy_analysis/`
(patient-level) and `..._document_level/` (document-level).

### Warning

If you run this in a Windows environment sometimes the long paths restriction can be an issue with some of the lower level outputs. The Windows runner automatically maps the repo onto a short drive letter to mitigate this; if it can't, prefer the Docker quickstart above.

## Synthetic data for reproducibility

The public repository reproduces the open-weight model-runner matrix
using dummy data. It deliberately excludes patient-level data and the
local production deployment details (hospital server inventory, internal
networking, governance configuration). The original clinical evaluation
was executed against physician-validated patient and document-level labels
on the hospital platform;
the synthetic corpus shipped here is sufficient to exercise every
pipeline stage end to end as a demo deployment.

The repository ships pseudonymised dummy CSV inputs under
`python_client/data/` together with balanced reference cohorts at
`analytics_code/configs/pseudonymised_dummy_reference.csv` and
`analytics_code/configs/pseudonymised_dummy_reference_document_flags.csv`.
The primary demo config now uses the enriched document-flags reference so
patient-level and document-level analytics both work after regeneration.
These files let new users exercise the full pipeline without access to
clinical data. The corpus can be regenerated deterministically with:

```bash
python scripts/generate_dummy_pipeline_data.py --n-patients 10 --seed 42
```

### Pinned dependency versions

The demo environment pins newer versions than the original production
deployment to match the late-evaluation Gemma 4 server requirement.
Key demo pins:

- `torch==2.8.0`, `transformers==5.5.3`, `accelerate==1.6.0`,
  `gradio==5.8.0`, `huggingface-hub==1.10.1` (model servers)
- `gradio_client==1.5.1`, `tiktoken==0.9.0`, `pandas==2.2.3` (client and
  analytics)

The production deployment used older pins (notably
`transformers==4.55.0` for the pre-Gemma-4 models and
`huggingface-hub==0.34.4`); this is intentional and reflects the
Gemma 4 toolchain requirements at the time the public demo was
finalised.

## Privacy and security

- Gradio services bind to `127.0.0.1` by default. Override the bind
  address explicitly only when remote access is required.
- Hugging Face tokens are read from environment variables (`.env` files
  or process environment) and are never committed to source.
- Logs and generated outputs remain inside the workspace and are listed
  in `.gitignore`.

## Corrections

31/05/2026: added tiered validation matrix missing from original demo.

### Contributing

If you would like to contribute further to this project you can do so by submitting a pull request to this repo. If you remix or fork the project please attribute appropriately.

### Licence

This project and the associated models are Attribution-NonCommercial 4.0 International Licensed. The copyright holder is Matt Stammers and University Hospital Southampton NHS Foundation Trust.

Shield: [![CC BY-NC 4.0][cc-by-nc-shield]][cc-by-nc]

This work is licensed under a
[Creative Commons Attribution-NonCommercial 4.0 International License][cc-by-nc].

[![CC BY-NC 4.0][cc-by-nc-image]][cc-by-nc]

[cc-by-nc]: https://creativecommons.org/licenses/by-nc/4.0/
[cc-by-nc-image]: https://licensebuttons.net/l/by-nc/4.0/88x31.png
[cc-by-nc-shield]: https://img.shields.io/badge/License-CC%20BY--NC%204.0-lightgrey.svg

### Disclaimer

No guarantee is given of model performance outside this research workflow. These models should be used in full accordance with the EU AI Act - Regulation 2024/1689. These are not CE marked medical devices and are suitable at this point only for research and development / experimentation. They can be improved but any improvements should be published openly and shared openly with the community. UHSFT and the author own the copyright and are choosing to share them freely under a CC BY-NC 4.0 Licence for the benefit of the wider research community.

## Citation

If you use this code, please cite this repository (author: Matt
Stammers, University Hospital Southampton NHS Foundation Trust). A
`CITATION.cff` file is included for tooling that consumes citation
metadata.
