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

from Tests.parameters import REGRESSION_CASES
from Tests.pipeline_helpers import PipelineCase, case_param, find_reference_dir

# ================================================================================================================================
# CASES (parameters in parameters.REGRESSION_CASES)
# ================================================================================================================================
FULL_PIPELINE_SUITE = 'regression_full_pipeline'
FULL_PIPELINE_INTENSIVE_SUITE = 'regression_full_pipeline_intensive'
SKULL_MASK_SUITE = 'regression_GetSkullMaskFromSimbNIBSSTL'


def regression_cases(name, marks):
    """pytest params for every combination of a REGRESSION_CASES entry."""
    spec = REGRESSION_CASES[name]
    return [case_param(PipelineCase(dataset=spec['dataset'], transducer=tx, freq_khz=freq, backend=backend,
                                    target=spec['target'], trajectory_type=spec['trajectory_type'], ct_type=ct_type,
                                    thermal_profile=spec['thermal_profile'],
                                    multipoint_profile=spec['multipoint_profile'],
                                    segment_soft_tissue=segment, last_step=spec['last_step']),
                       marks=marks)
            for backend in spec['backends']
            for ct_type in spec['ct_types']
            for tx, freq in spec['transducers']
            for segment in spec['soft_tissue_segmentation']]


FULL_PIPELINE_CASES = regression_cases('regression_test_full_pipeline', [pytest.mark.slow])
FULL_PIPELINE_INTENSIVE_CASES = regression_cases('regression_test_full_pipeline_intensive', [pytest.mark.slow])
SKULL_MASK_CASES = regression_cases('regression_test_GetSkullMaskFromSimbNIBSSTL', [pytest.mark.step1])


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
