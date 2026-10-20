"""
Test collection for the whole suite, called from the conftest.py hooks: default parametrization of
the test inputs (dataset, transducer, ...) and the markers added from each test's parameters.
"""
from pathlib import Path

import pytest

from Tests.helpers.parameters import (CONFIGURATION_MARKERS, COMPUTING_BACKENDS, CT_TYPES, DATASETS,
                                      RELEASE_SPATIAL_STEPS, SPATIAL_STEPS, SYNTHETIC_DATASET,
                                      TESTED_INDIVIDUAL_TARGETS, TRAJECTORY_TYPES, TESTS_DIR, TRANSDUCERS,
                                      marker_selects_transducer)

UNIT_TESTS_DIR = TESTS_DIR / 'Unit'
CONSISTENCY_TESTS_DIR = TESTS_DIR / 'Consistency'

RES_MARKS = {'Low_Res': [pytest.mark.low_res],
             'Med_Res': [pytest.mark.medium_res],
             'High_Res': [pytest.mark.slow, pytest.mark.high_res],
             'Stress_Res': [pytest.mark.slow, pytest.mark.stress_res]}


# ================================================================================================================================
# PARAMETRIZATION
# ================================================================================================================================
def _explicitly_parametrized(metafunc, name):
    """True when the test parametrizes `name` itself (@pytest.mark.parametrize), so the default
    parametrization below doesn't apply."""
    for marker in metafunc.definition.iter_markers('parametrize'):
        argnames = marker.args[0]
        if isinstance(argnames, str):
            argnames = [n.strip() for n in argnames.split(',')]
        if name in argnames:
            return True
    return False


def default_parametrize(metafunc):
    """Default parametrization for tests that request these fixtures. A test narrows any of them by
    parametrizing it explicitly, e.g. @pytest.mark.parametrize('trajectory', INVALID_TARGETS)."""
    def wants(name):
        return name in metafunc.fixturenames and not _explicitly_parametrized(metafunc, name)

    dataset_with_spatial_step = wants('dataset') and wants('spatial_step')
    if dataset_with_spatial_step:
        # GPU unit tests: every spatial step on the small synthetic head and the real datasets, marked
        # minor_release / major_release for the resolutions in each release set (RELEASE_SPATIAL_STEPS)
        def release_marks(kind, key):
            return [getattr(pytest.mark, release) for release, steps in RELEASE_SPATIAL_STEPS.items() if key in steps[kind]]
        params = [pytest.param(SYNTHETIC_DATASET, value, id=f"synthetic-{key}",
                               marks=RES_MARKS.get(key, []) + release_marks('synthetic', key))
                  for key, value in SPATIAL_STEPS.items()]
        for ds_id, ds in DATASETS.items():
            for key, value in SPATIAL_STEPS.items():
                params.append(pytest.param(ds, value, id=f"{ds_id}-{key}",
                                           marks=RES_MARKS.get(key, []) + release_marks('real', key)))
        metafunc.parametrize('dataset,spatial_step', params)
    elif wants('dataset'):
        metafunc.parametrize('dataset', list(DATASETS.values()), ids=list(DATASETS))

    if wants('trajectory_type'):
        metafunc.parametrize('trajectory_type', list(TRAJECTORY_TYPES))

    if wants('trajectory'):
        metafunc.parametrize('trajectory', TESTED_INDIVIDUAL_TARGETS)

    for name in ('scan_type', 'second_scan_type'):
        if wants(name):
            metafunc.parametrize(name, list(CT_TYPES), ids=[f"{item}_Extra_Scan" for item in CT_TYPES])

    if wants('transducer'):
        if 'frequency' in metafunc.fixturenames:
            # Parametrize both transducer and freq
            params = [pytest.param(tx, freq, id=f"{tx['name']}-{int(freq/1000)}kHz")
                      for tx in TRANSDUCERS for freq in tx['freqs']]
            metafunc.parametrize("transducer,frequency", params)
        else:
            metafunc.parametrize("transducer", [pytest.param(tx, id=tx['name']) for tx in TRANSDUCERS])

    if wants('computing_backend'):
        params = [pytest.param(cb, id=cb['type'], marks=pytest.mark.gpu) for cb in COMPUTING_BACKENDS]
        metafunc.parametrize("computing_backend", params)

    if wants('spatial_step') and not dataset_with_spatial_step:
        metafunc.parametrize('spatial_step', [pytest.param(value, id=key, marks=RES_MARKS.get(key, []))
                                              for key, value in SPATIAL_STEPS.items()])

    if wants('tolerance'):
        metafunc.parametrize('tolerance',
                             [pytest.param(0, marks=pytest.mark.tol_0, id="0%_tolerance"),
                              pytest.param(0.01, marks=pytest.mark.tol_1, id="1%_tolerance"),
                              pytest.param(0.05, marks=pytest.mark.tol_5, id="5%_tolerance")])


# ================================================================================================================================
# MARKERS
# ================================================================================================================================
def _params(item):
    return getattr(getattr(item, 'callspec', None), 'params', {})


def _in(item, folder):
    return Path(item.path).is_relative_to(folder)


def _uses_real_data(item):
    """Unit test reading the real datasets: parametrized with one of DATASETS or marked real_data."""
    if item.get_closest_marker('real_data'):
        return True
    dataset = _params(item).get('dataset')
    return isinstance(dataset, dict) and dataset.get('id') in DATASETS


def _add_release_markers(item):
    """minor_release / major_release markers for the unit tests and suite consistency checks in each
    release test set (BabelBrain_Pytest_Usage.md, Tables 3 and 7). GPU unit tests get them from their
    spatial step (default_parametrize); every other unit test is in the major set, and in the minor
    set too unless it's slow."""
    if not (_in(item, UNIT_TESTS_DIR) or _in(item, CONSISTENCY_TESTS_DIR)):
        return
    if 'spatial_step' in _params(item):
        return
    item.add_marker(pytest.mark.major_release)
    if not item.get_closest_marker('slow'):
        item.add_marker(pytest.mark.minor_release)


def _add_configuration_markers(item):
    """*_babelbrain_params selection markers based on the test's parameter values
    (transducer/frequency choices are in parameters.MARKER_TRANSDUCERS)."""
    params = _params(item)
    if not {'transducer', 'frequency', 'dataset', 'trajectory', 'trajectory_type', 'scan_type'} <= params.keys():
        return
    tx, freq = params['transducer']['name'], params['frequency']
    dataset, target = params['dataset']['id'], params['trajectory']
    traj_type, scan = params['trajectory_type'], params['scan_type']

    def selects(marker):
        return marker_selects_transducer(marker, tx, freq)

    backend = params.get('computing_backend', {}).get('type')
    for marker, config in CONFIGURATION_MARKERS.items():
        if (dataset == config['dataset'] and scan in config['ct_types']
                and traj_type == config['trajectory_type'] and target == config['target']
                and (tx, int(freq/1000)) in config['transducers'] and backend in config['backends']):
            item.add_marker(getattr(pytest.mark, marker))

    if dataset == 'ID_0082' and traj_type == 'brainsight':
        if selects('all_targets_babelbrain_params'):
            if scan in ('NONE', 'CT', 'ZTE'):
                item.add_marker(pytest.mark.all_targets_babelbrain_params)
            if scan in ('NONE', 'CT'):
                item.add_marker(pytest.mark.ct_targets_babelbrain_params)
        if target == 'Deep_Target_Left' and scan == 'NONE' and selects('bare_min_babelbrain_params'):
            item.add_marker(pytest.mark.bare_min_babelbrain_params)

    if (target == 'Superficial_Target_Left' and dataset != 'ID_0082' and tx == 'Single'
            and traj_type == 'slicer' and scan in ('CT', 'NONE')):
        item.add_marker(pytest.mark.orig_paper_params)


def add_markers(items):
    """Add the real_data marker to unit tests using the real datasets (deselect them with
    -m "not real_data"), the minor_release / major_release markers, and the *_babelbrain_params
    selection markers. E2E, regression, ... tests always use real data and aren't marked real_data."""
    for item in items:
        if _in(item, UNIT_TESTS_DIR) and _uses_real_data(item):
            item.add_marker(pytest.mark.real_data)
        _add_release_markers(item)
        _add_configuration_markers(item)
