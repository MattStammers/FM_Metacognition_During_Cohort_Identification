#!/usr/bin/env python3
"""Generate a richer synthetic dataset for the analytics pipeline.

This script produces a coherent dummy corpus that exercises every stage of
``analytics_code`` end-to-end:

* ``analytics_code/configs/pseudonymised_dummy_reference.csv``
    Stratified pseudonymised reference (default 60 patients, balanced
    ``ground_truth`` 50/50, age/sex/site distributed).
* ``analytics_code/configs/pseudonymised_dummy_reference_document_flags.csv``
    Same cohort enriched with explicit document-level IBD marker labels so
    document-truth analytics can be regenerated from the demo data too.
* ``python_client/data/{clinical_letters,endoscopy,histopathology}_filtered_dummy.csv``
    Source documents for the bundled dummy Gradio matrix run.
* ``python_client/outputs/chronology_runs/all_models_realistic_dummy_v1/<runner>/<config>/batch_0.xlsx``
  One synthetic batch per (model, config) folder. Likelihood scores are
  drawn so that the **median across the merged cohort sits at 5** (the
  decision threshold), guaranteeing healthy grey-zone coverage and a mix
  of true-positive / true-negative / false-positive / false-negative
  predictions across every analytics stage.

Run::

    python scripts/generate_dummy_pipeline_data.py            # 60 patients, seed 42
    python scripts/generate_dummy_pipeline_data.py --n-patients 100 --seed 7
"""

from __future__ import annotations

import argparse
import json
import random
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
REFERENCE_CSV = (
    ROOT / "analytics_code" / "configs" / "pseudonymised_dummy_reference.csv"
)
DOCUMENT_FLAGS_REFERENCE_CSV = (
    ROOT
    / "analytics_code"
    / "configs"
    / "pseudonymised_dummy_reference_document_flags.csv"
)
SOURCE_DIR = ROOT / "python_client" / "data"
RUNS_ROOT = (
    ROOT
    / "python_client"
    / "outputs"
    / "chronology_runs"
    / "all_models_realistic_dummy_v1"
)

AGE_GROUPS = ["18-39", "40-59", "60+"]
SEXES = ["F", "M"]
SITES = ["SiteA", "SiteB", "SiteC"]

MODEL_OFFSETS = {
    "mixtral_runner": -0.4,
    "mixtral_t1_runner": -0.5,
    "m42_runner": +0.3,
    "m42_t1_runner": +0.2,
    "deepseek14_runner": -0.1,
    "deepseek14_t05_runner": -0.2,
    "deepseek32_runner": +0.6,
    "qwen32_runner": +0.2,
    "gemma4_31b_runner": +0.4,
}

# Light per-shot-type bias: dual-shot tends to be slightly more confident.
SHOT_OFFSETS = {"zero": -0.2, "single": 0.0, "dual": +0.2}

# ---------------------------------------------------------------------------
# Patient synthesis
# ---------------------------------------------------------------------------

POSITIVE_HISTO = [
    "Terminal ileum biopsies show patchy chronic active ileitis with cryptitis and a poorly formed granuloma.",
    "Right colonic biopsies show chronic active colitis with crypt distortion, basal plasmacytosis, and Paneth-cell metaplasia.",
    "Sigmoid biopsies demonstrate severe active colitis with crypt abscesses; features compatible with ulcerative colitis.",
    "Ascending colon biopsies show chronic active colitis with non-caseating granulomas; suspicious for Crohn's.",
]
NEGATIVE_HISTO = [
    "Mapping biopsies show normal colonic mucosa with no evidence of inflammation, dysplasia, or malignancy.",
    "Left colonic biopsies are histologically unremarkable with preserved crypt architecture.",
    "No acute or chronic inflammation identified; features within normal limits.",
    "Biopsies show only mild reactive changes without IBD-defining features.",
]
POSITIVE_ENDO = [
    "Ileocolonoscopy demonstrated discontinuous ulceration with skip lesions and rectal sparing.",
    "Pancolitis with friability, contact bleeding, and loss of vascular pattern from rectum to caecum.",
    "Cobblestoning and aphthous ulcers in the terminal ileum; biopsies obtained.",
    "Severe left-sided colitis with confluent ulceration; mucosa friable on contact.",
]
NEGATIVE_ENDO = [
    "Normal colonoscopy to caecum and terminal ileum. No inflammation seen.",
    "Diverticulosis only. No mucosal inflammation; biopsies for surveillance.",
    "Internal haemorrhoids identified; otherwise unremarkable colonoscopy.",
    "Mild non-specific erythema in the rectum; no ulceration or friability.",
]
POSITIVE_PRECEDING = [
    "Gastroenterology referral: 6-month bloody diarrhoea, weight loss, raised faecal calprotectin (>500). Colonoscopy to exclude IBD.",
    "Outpatient assessment: nocturnal diarrhoea, urgency, abdominal pain, anaemia. High clinical suspicion of IBD.",
    "Persistent diarrhoea with mucus and tenesmus; calprotectin 820. Lower GI endoscopy arranged.",
]
NEGATIVE_PRECEDING = [
    "Clinic letter: bowel habit change attributed to IBS; calprotectin normal. Colonoscopy for reassurance.",
    "Polyp surveillance review; asymptomatic. Routine colonoscopy planned.",
    "Iron deficiency anaemia work-up; no GI symptoms. Lower GI endoscopy organised.",
]
POSITIVE_FOLLOWING = [
    "Post-procedure clinic: endoscopic and histological findings consistent with Crohn's disease. MDT planned.",
    "Follow-up: ulcerative colitis confirmed. Mesalazine commenced; biologic discussion if no response.",
    "Diagnosis of IBD established; gastroenterology follow-up in 6 weeks.",
]
NEGATIVE_FOLLOWING = [
    "Follow-up clinic: histology did not confirm IBD. Conservative management; repeat calprotectin in 3 months.",
    "Outpatient review: no IBD identified; symptoms attributed to functional bowel disease.",
    "Reassurance given; discharged back to GP.",
]


def build_patients(n_patients: int, rng: random.Random) -> pd.DataFrame:
    """Generate ``n_patients`` synthetic patients with a balanced ground truth.

    The cohort is exactly 50/50 positive/negative for ``ground_truth``;
    age group, sex and site are drawn uniformly from the module-level
    pools. Stable ``patient_id`` identifiers ``PSN001..`` are assigned
    after a shuffle so the order is randomised but reproducible for a
    given ``rng`` state.
    """
    rows = []
    for i in range(1, n_patients + 1):
        gt = i % 2  # exactly balanced 50/50
        rows.append(
            {
                "patient_id": f"PSN{i:03d}",
                "ground_truth": gt,
                "age_group": rng.choice(AGE_GROUPS),
                "sex": rng.choice(SEXES),
                "site": rng.choice(SITES),
            }
        )
    rng.shuffle(rows)  # randomise order while preserving 50/50 balance
    # re-issue stable IDs
    for idx, row in enumerate(rows, start=1):
        row["patient_id"] = f"PSN{idx:03d}"
    return pd.DataFrame(rows)


def build_document_flag_reference(
    patients: pd.DataFrame, rng: random.Random
) -> pd.DataFrame:
    """Attach explicit document-level IBD marker labels to the dummy cohort.

    The column names intentionally match the labels used by the document-level
    analytics logic so the bundled demo can exercise patient, document, and
    complete-marker truth modes from a single regenerated reference file.
    """
    rows: list[dict[str, object]] = []
    for _, patient in patients.iterrows():
        gt = int(patient["ground_truth"])
        if gt == 1:
            flags = {
                "preceding_clinic_ibd_flag": int(rng.random() < 0.8),
                "following_clinic_ibd_flag": int(rng.random() < 0.7),
                "endoscopy_ibd_flag": int(rng.random() < 0.75),
                "histology_ibd_flag": int(rng.random() < 0.7),
            }
            if not any(flags.values()):
                flags[rng.choice(tuple(flags))] = 1
        else:
            flags = {
                "preceding_clinic_ibd_flag": int(rng.random() < 0.1),
                "following_clinic_ibd_flag": int(rng.random() < 0.08),
                "endoscopy_ibd_flag": int(rng.random() < 0.12),
                "histology_ibd_flag": int(rng.random() < 0.1),
            }

        row = patient.to_dict()
        row.update(flags)
        rows.append(row)
    return pd.DataFrame(rows)


def build_source_documents(
    patients: pd.DataFrame, rng: random.Random
) -> dict[str, pd.DataFrame]:
    """Build the per-patient source CSVs (histopathology, endoscopy, clinic letters).

    The text content is drawn from the positive/negative document
    pools so the corpus is internally consistent with each patient's
    ``ground_truth``.
    """
    histo, endo, clinic = [], [], []
    for _, p in patients.iterrows():
        pid = p["patient_id"]
        gt = int(p["ground_truth"])
        histo.append(
            {
                "patient_id": pid,
                "sample_received_date": "2025-01-10",
                "result_report": rng.choice(POSITIVE_HISTO if gt else NEGATIVE_HISTO),
            }
        )
        endo.append(
            {
                "patient_id": pid,
                "procedure_date": "2025-01-09",
                # The python_client config maps text_column -> "Combined_Content";
                # keep the same column name in the source CSV so it matches.
                "Combined_Content": rng.choice(POSITIVE_ENDO if gt else NEGATIVE_ENDO),
            }
        )
        # Clinic-letter source is read in long format (one row per letter)
        # with text_column="clean_content" and date_column="date_creation".
        # Date format must match the config: "%d-%b-%Y %H:%M:%S".
        clinic.append(
            {
                "patient_id": pid,
                "date_creation": "07-Jan-2025 09:15:00",
                "clean_content": rng.choice(
                    POSITIVE_PRECEDING if gt else NEGATIVE_PRECEDING
                ),
            }
        )
        clinic.append(
            {
                "patient_id": pid,
                "date_creation": "13-Jan-2025 14:30:00",
                "clean_content": rng.choice(
                    POSITIVE_FOLLOWING if gt else NEGATIVE_FOLLOWING
                ),
            }
        )
    return {
        "histopathology_filtered_dummy.csv": pd.DataFrame(histo),
        "endoscopy_filtered_dummy.csv": pd.DataFrame(endo),
        "clinical_letters_filtered_dummy.csv": pd.DataFrame(clinic),
    }


# ---------------------------------------------------------------------------
# LLM response synthesis
# ---------------------------------------------------------------------------

TITLES_POS = [
    "Crohn's disease",
    "Ulcerative colitis",
    "Inflammatory bowel disease",
    "Suspected Crohn's",
    "IBD - probable UC",
]
TITLES_NEG = [
    "Normal mucosa",
    "No evidence of IBD",
    "Functional bowel disease",
    "IBS - no IBD features",
    "Polyp surveillance",
]
FEATURES_POS = [
    "Patchy chronic active ileocolitis",
    "Cryptitis with crypt abscesses",
    "Granuloma identified",
    "Skip lesions on endoscopy",
    "Raised faecal calprotectin",
    "Rectal sparing with right-sided inflammation",
]
FEATURES_NEG = [
    "Normal colonic mapping biopsies",
    "No active inflammation",
    "Preserved crypt architecture",
    "Reassuring clinic letter",
    "Calprotectin within normal range",
    "Polyp surveillance only",
]


def _draw_likelihood(
    gt: int, model_bias: float, shot_bias: float, config_bias: float, rng: random.Random
) -> int:
    """Sample a synthetic likelihood score in ``[0, 10]`` for one row.

    The base score is centred at 7 for positives and 3 for negatives;
    Gaussian noise plus the supplied biases produce realistic spread
    around the decision threshold.
    """
    base = 7.0 if gt == 1 else 3.0
    noise = rng.gauss(0.0, 1.6)
    raw = base + model_bias + shot_bias + config_bias + noise
    return int(np.clip(round(raw), 0, 10))


def _make_json_payload(gt: int, likelihood: int, rng: random.Random) -> dict:
    """Build a JSON response payload mimicking a real LLM output.

    The certainty scaling is intentionally tied to the distance from
    the decision threshold so that high-certainty + extreme-likelihood
    + wrong-direction rows occur (the "catastrophic failure" pattern
    used by the narrative-analysis stage).
    """
    title_pool = TITLES_POS if likelihood >= 5 else TITLES_NEG
    feat_pool = FEATURES_POS if likelihood >= 5 else FEATURES_NEG
    n_feats = rng.randint(2, 4)
    # Cocky-wrong: high certainty + extreme likelihood + wrong prediction
    certainty_base = 6 + (abs(likelihood - 5) // 2)
    certainty = int(np.clip(certainty_base + rng.randint(-1, 2), 1, 10))
    complexity = int(np.clip(rng.randint(3, 8), 1, 10))
    return {
        "Title": rng.choice(title_pool),
        "Features": rng.sample(feat_pool, k=min(n_feats, len(feat_pool))),
        "Likelihood of IBD": likelihood,
        "Certainty Level": certainty,
        "Complexity of Case": complexity,
    }


def _full_response(payload: dict, gt: int) -> str:
    """Render the full free-text response with CLUES/REASONING/OUTCOME sections."""
    clues_pos = "Endoscopic and histological findings suspicious for IBD."
    clues_neg = "Reports lack convincing IBD-defining features."
    reasoning_pos = "Combination of clinical, endoscopic, and histological findings supports an IBD diagnosis."
    reasoning_neg = (
        "Absence of inflammation and reassuring clinic letters argue against IBD."
    )
    return (
        "**CLUES:**\n"
        f"{clues_pos if gt == 1 else clues_neg}\n\n"
        "**REASONING:**\n"
        f"{reasoning_pos if gt == 1 else reasoning_neg}\n\n"
        "**OUTCOME:**\n"
        "```json\n"
        f"{json.dumps(payload, indent=2)}\n"
        "```\n"
    )


def synthesize_batch(
    patients: pd.DataFrame,
    runner: str,
    config_name: str,
    rng: random.Random,
) -> pd.DataFrame:
    """Synthesise a single batch dataframe for one (runner, config) folder.

    Models a realistic mix of clean responses (~88%), responses
    needing JSON repair (~7%) and missing responses (~5%), plus a
    small fraction of ``Truncated`` flags.
    """
    shot_type = config_name.split("_", 1)[0]
    shot_bias = SHOT_OFFSETS.get(shot_type, 0.0)
    model_bias = MODEL_OFFSETS.get(runner, 0.0)
    config_bias = (
        hash(config_name) % 7 - 3
    ) * 0.15  # deterministic small drift per config

    rows = []
    json_col = f"{runner}_Json_Response_{config_name}"
    full_col = f"{runner}_Full_Response_{config_name}"
    payload_col = f"{runner}_Payload_{config_name}"
    truncated_col = f"Truncated_{config_name}"

    for _, p in patients.iterrows():
        gt = int(p["ground_truth"])
        likelihood = _draw_likelihood(gt, model_bias, shot_bias, config_bias, rng)
        payload = _make_json_payload(gt, likelihood, rng)
        json_text = json.dumps(payload)
        full_text = _full_response(payload, gt)

        # ~5% no-response, ~7% needs JSON repair (extra text around JSON)
        roll = rng.random()
        truncated = False
        if roll < 0.05:
            json_text = ""
            full_text = ""
        elif roll < 0.12:
            # corrupt JSON slightly so the parser's repair path is exercised
            json_text = "Note: " + json_text + "\n--end--"
            truncated = rng.random() < 0.4

        rows.append(
            {
                "patient_id": p["patient_id"],
                "procedure_date": "2025-01-09",
                "sample_received_date": "2025-01-10",
                "result_report": "(synthetic placeholder)",
                "Combined_Content": "(synthetic placeholder)",
                "preceding_clinic_letter": "(synthetic placeholder)",
                "following_clinic_letter": "(synthetic placeholder)",
                "preceding_clinic_date": "2025-01-07 09:15",
                "following_clinic_date": "2025-01-13 14:30",
                "preceding_clinic_time_diff": 1,
                "following_clinic_time_diff": 4,
                "date_diff": 1,
                full_col: full_text,
                json_col: json_text,
                payload_col: "(synthetic payload)",
                truncated_col: truncated,
            }
        )
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Driver
# ---------------------------------------------------------------------------


def main() -> int:
    """Generate the synthetic reference, source documents and per-runner batches.

    Returns the process exit code (always ``0`` on success).
    """
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--n-patients", type=int, default=60)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    rng = random.Random(args.seed)
    np.random.seed(args.seed)

    patients = build_patients(args.n_patients, rng)
    document_flag_reference = build_document_flag_reference(patients, rng)
    REFERENCE_CSV.parent.mkdir(parents=True, exist_ok=True)
    patients.to_csv(REFERENCE_CSV, index=False)
    print(f"Wrote reference: {REFERENCE_CSV} ({len(patients)} patients)")
    document_flag_reference.to_csv(DOCUMENT_FLAGS_REFERENCE_CSV, index=False)
    print(
        "Wrote reference: "
        f"{DOCUMENT_FLAGS_REFERENCE_CSV} ({len(document_flag_reference)} patients)"
    )

    sources = build_source_documents(patients, rng)
    SOURCE_DIR.mkdir(parents=True, exist_ok=True)
    for name, frame in sources.items():
        path = SOURCE_DIR / name
        frame.to_csv(path, index=False)
        print(f"Wrote source: {path} ({len(frame)} rows)")

    if not RUNS_ROOT.exists():
        print(f"WARNING: {RUNS_ROOT} does not exist; skipping batch synthesis.")
        return 0

    runners = sorted(p.name for p in RUNS_ROOT.iterdir() if p.is_dir())
    total_batches = 0
    all_likelihoods: list[int] = []
    for runner in runners:
        runner_dir = RUNS_ROOT / runner
        configs = sorted(p.name for p in runner_dir.iterdir() if p.is_dir())
        for cfg in configs:
            cfg_dir = runner_dir / cfg
            batch_rng = random.Random(args.seed ^ hash((runner, cfg)) & 0xFFFFFFFF)
            batch = synthesize_batch(patients, runner, cfg, batch_rng)
            out = cfg_dir / "batch_0.xlsx"
            batch.to_excel(out, index=False, engine="openpyxl")
            total_batches += 1
            json_col = f"{runner}_Json_Response_{cfg}"
            for txt in batch[json_col]:
                if not txt:
                    continue
                try:
                    payload = json.loads(
                        txt.split("Note: ", 1)[-1].split("\n--end--", 1)[0]
                    )
                    all_likelihoods.append(int(payload["Likelihood of IBD"]))
                except Exception:
                    pass
    print(f"Wrote {total_batches} batch xlsx files across {len(runners)} runners")

    if all_likelihoods:
        arr = np.array(all_likelihoods)
        print(
            f"Likelihood distribution -- n={arr.size} median={np.median(arr):.1f} "
            f"mean={arr.mean():.2f} q1={np.quantile(arr, 0.25):.1f} q3={np.quantile(arr, 0.75):.1f}"
        )
        positive_share = float((arr >= 5).mean())
        print(f"Predicted-positive share (likelihood>=5): {positive_share:.1%}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
