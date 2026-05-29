from __future__ import annotations

import importlib.util
from pathlib import Path


def _load_dummy_generator_module():
    root = Path(__file__).resolve().parents[2]
    script_path = root / "scripts" / "generate_dummy_pipeline_data.py"
    spec = importlib.util.spec_from_file_location(
        "generate_dummy_pipeline_data", script_path
    )
    module = importlib.util.module_from_spec(spec)
    assert spec is not None and spec.loader is not None
    spec.loader.exec_module(module)
    return module


def test_build_document_flag_reference_emits_expected_marker_columns():
    generator = _load_dummy_generator_module()
    rng = generator.random.Random(42)
    patients = generator.build_patients(12, rng)

    reference = generator.build_document_flag_reference(patients, rng)

    expected_columns = {
        "patient_id",
        "ground_truth",
        "preceding_clinic_ibd_flag",
        "following_clinic_ibd_flag",
        "endoscopy_ibd_flag",
        "histology_ibd_flag",
    }
    assert expected_columns.issubset(reference.columns)
    for column in expected_columns - {"patient_id", "ground_truth"}:
        assert set(reference[column].dropna().unique()).issubset({0, 1})
    assert (
        reference.loc[
            reference["ground_truth"] == 1,
            [
                "preceding_clinic_ibd_flag",
                "following_clinic_ibd_flag",
                "endoscopy_ibd_flag",
                "histology_ibd_flag",
            ],
        ]
        .sum(axis=1)
        .ge(1)
        .all()
    )
