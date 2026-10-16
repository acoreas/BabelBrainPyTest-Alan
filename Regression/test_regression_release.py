"""
Major release regression tests (BabelBrain_Pytest_Usage.md, Table 10).

Each regression test comes in two parts:

  * test_generate_<name>  runs the pipeline for every case and saves the outputs in
                          <gen_output_folder>/<suite>/<case_id> (marked complete only
                          once the run succeeded). Pytest runs all of these first.
  * test_<name>           compares the saved outputs with the reference outputs in
                          <ref_output_folder_1>/<suite>/<case_id> at 0/1/5% tolerance.
                          It generates the case itself if needed, so it can also be
                          selected on its own.

Saved outputs from the same BabelBrain commit are reused across tests and sessions
(set BABEL_PYTEST_REGENERATE=1 to force a rerun). To create reference outputs, check
out the reference BabelBrain version (v0.8.1), point gen_output_folder at the
reference location and run only the generate tests:

    pytest Tests/Regression -m generate_outputs
"""
import pytest

from Tests.parameters import (CT_TYPES, DEFAULT_MULTIPOINT_PROFILE,
                              FULL_PIPELINE_INTENSIVE_TRANSDUCER,
                              INTENSIVE_THERMAL_PROFILE,
                              FULL_PIPELINE_TRANSDUCERS, RELEASE_BACKENDS,
                              SKULL_MASK_TRANSDUCERS)
from Tests.pipeline_helpers import PipelineCase, case_param, find_reference_dir

# ================================================================================================================================
# CASES
# ================================================================================================================================
FULL_PIPELINE_SUITE = 'regression_full_pipeline'
FULL_PIPELINE_INTENSIVE_SUITE = 'regression_full_pipeline_intensive'
SKULL_MASK_SUITE = 'regression_GetSkullMaskFromSimbNIBSSTL'

FULL_PIPELINE_CASES = [
    case_param(PipelineCase(dataset='ernie', transducer=tx, freq_khz=freq, backend=backend),
               marks=[pytest.mark.slow])
    for backend in RELEASE_BACKENDS
    for tx, freq in FULL_PIPELINE_TRANSDUCERS
]

FULL_PIPELINE_INTENSIVE_CASES = [
    case_param(PipelineCase(dataset='ID_0082', transducer=FULL_PIPELINE_INTENSIVE_TRANSDUCER[0],
                            freq_khz=FULL_PIPELINE_INTENSIVE_TRANSDUCER[1], backend=backend,
                            target='Deep_Target_Left', ct_type='CT', multipoint_profile=DEFAULT_MULTIPOINT_PROFILE,
                            thermal_profile=INTENSIVE_THERMAL_PROFILE),
               marks=[pytest.mark.slow])
    for backend in RELEASE_BACKENDS
]

SKULL_MASK_CASES = [
    case_param(PipelineCase(dataset='ID_0082', transducer=tx, freq_khz=freq, backend=backend,
                            ct_type=ct_type, segment_soft_tissue=segment, last_step=1),
               marks=[pytest.mark.step1])
    for backend in RELEASE_BACKENDS
    for ct_type in CT_TYPES
    for tx, freq in SKULL_MASK_TRANSDUCERS
    for segment in [False, True]
]


# ================================================================================================================================
# GENERATE (runs first)
# ================================================================================================================================
@pytest.mark.generate_outputs
@pytest.mark.parametrize('case', FULL_PIPELINE_CASES)
def test_generate_regression_full_pipeline(pipeline, case):
    pipeline.ensure_outputs(FULL_PIPELINE_SUITE, case)


@pytest.mark.generate_outputs
@pytest.mark.parametrize('case', FULL_PIPELINE_INTENSIVE_CASES)
def test_generate_regression_full_pipeline_intensive(pipeline, case):
    pipeline.ensure_outputs(FULL_PIPELINE_INTENSIVE_SUITE, case)


@pytest.mark.generate_outputs
@pytest.mark.parametrize('case', SKULL_MASK_CASES)
def test_generate_regression_GetSkullMaskFromSimbNIBSSTL(pipeline, case):
    pipeline.ensure_outputs(SKULL_MASK_SUITE, case)


# ================================================================================================================================
# COMPARE AGAINST REFERENCE
# ================================================================================================================================
@pytest.mark.parametrize('case', FULL_PIPELINE_CASES)
def test_regression_full_pipeline(pipeline, tolerance, case):
    ref_dir = find_reference_dir(FULL_PIPELINE_SUITE, case)
    test_dir = pipeline.ensure_outputs(FULL_PIPELINE_SUITE, case)
    pipeline.assert_thermal_outputs_match(ref_dir, test_dir, tolerance)


@pytest.mark.parametrize('case', FULL_PIPELINE_INTENSIVE_CASES)
def test_regression_full_pipeline_intensive(pipeline, tolerance, case):
    ref_dir = find_reference_dir(FULL_PIPELINE_INTENSIVE_SUITE, case)
    test_dir = pipeline.ensure_outputs(FULL_PIPELINE_INTENSIVE_SUITE, case)
    pipeline.assert_thermal_outputs_match(ref_dir, test_dir, tolerance)


@pytest.mark.parametrize('case', SKULL_MASK_CASES)
def test_regression_GetSkullMaskFromSimbNIBSSTL(pipeline, tolerance, case):
    ref_dir = find_reference_dir(SKULL_MASK_SUITE, case)
    test_dir = pipeline.ensure_outputs(SKULL_MASK_SUITE, case)
    pipeline.assert_all_outputs_match(ref_dir, test_dir, tolerance)
