"""
Checks that the test parameters in parameters.py still match the BabelBrain checkout and the test
data, so a new transducer, GUI option or custom transducer template (or a renamed data file) shows
up as a clear failure here instead of as silently skipped or untested cases.

These need no GPU and run in seconds:

    pytest Tests/test_suite_consistency.py
"""
import os

import pytest
import yaml

from Tests import parameters as P

# ================================================================================================================================
# APP OPTIONS
# ================================================================================================================================
GUI_OPTION_LABELS = {
    'CTTypecomboBox': P.CT_TYPE_LABELS,
    'TrajectoryTypecomboBox': P.TRAJECTORY_TYPE_LABELS,
    'SimbNIBSTypecomboBox': P.SIMNIBS_TYPE_LABELS,
    'CoregCTcomboBox': P.COREGISTRATION_LABELS,
}


@pytest.mark.parametrize('combo_name', GUI_OPTION_LABELS)
def test_gui_options_match_test_labels(combo_name):
    gui_items = P.selfiles_combo_items(combo_name)
    labels = GUI_OPTION_LABELS[combo_name]
    assert gui_items, f"{combo_name} not found in SelFiles/form.ui"

    untested = [item for item in gui_items if item not in labels.values()]
    assert not untested, f"New {combo_name} option(s) {untested} have no test name in parameters.py"

    # Options missing from the app (older version, or removed) just aren't parametrized; a rename
    # is caught above as an untested option
    missing = [label for label in labels.values() if label not in gui_items]
    if missing:
        pytest.skip(f"{combo_name} option(s) {missing} are not in BabelBrain {P.BABELBRAIN_VERSION}, "
                    "tests using them are not generated")


# ================================================================================================================================
# TRANSDUCERS
# ================================================================================================================================
def _referenced_transducers():
    """(where, transducer, kHz) for every transducer/frequency the tests name explicitly."""
    refs = [('FULL_PIPELINE_TRANSDUCERS', tx, f) for tx, f in P.FULL_PIPELINE_TRANSDUCERS]
    refs.append(('FULL_PIPELINE_INTENSIVE_TRANSDUCER', *P.FULL_PIPELINE_INTENSIVE_TRANSDUCER))
    refs += [('SKULL_MASK_TRANSDUCERS', tx, f) for tx, f in P.SKULL_MASK_TRANSDUCERS]
    refs += [('CUSTOM_TX_EQUIVALENTS', eq['transducer'], eq['freq_khz']) for eq in P.CUSTOM_TX_EQUIVALENTS.values()]
    for marker, rule in P.MARKER_TRANSDUCERS.items():
        for tx, freqs in rule['frequencies'].items():
            refs += [(marker, tx, f) for f in (freqs or [None])]
    return refs


@pytest.mark.parametrize('where,transducer,freq_khz', _referenced_transducers(),
                         ids=lambda v: str(v))
def test_referenced_transducers_exist(where, transducer, freq_khz):
    assert transducer in P.TRANSDUCERS_BY_NAME, \
        f"{where} names transducer {transducer}, which is not in SelFiles/transducer_list.yaml"
    if freq_khz is not None:
        available = [int(f / 1000) for f in P.TRANSDUCERS_BY_NAME[transducer]['freqs']]
        assert freq_khz in available, \
            f"{where} uses {transducer} at {freq_khz} kHz, available frequencies are {available}"


def test_every_transducer_type_has_full_pipeline_regression():
    if not P.TRANSDUCER_TYPES:
        pytest.skip(f"BabelBrain {P.BABELBRAIN_VERSION} doesn't define transducer geometry types")
    covered = {P.TRANSDUCERS_BY_NAME[tx]['transducer_type'] for tx, _ in P.FULL_PIPELINE_TRANSDUCERS
               if tx in P.TRANSDUCERS_BY_NAME}
    missing = sorted(set(P.TRANSDUCER_TYPES) - covered)
    assert not missing, f"Transducer type(s) {missing} have no transducer in FULL_PIPELINE_TRANSDUCERS"


@P.requires_feature('custom_transducers')
def test_every_custom_template_has_builtin_equivalent():
    templates, equivalents = set(P.CUSTOM_TX_TEMPLATES), set(P.CUSTOM_TX_EQUIVALENTS)
    assert not templates - equivalents, \
        f"Custom transducer template(s) {sorted(templates - equivalents)} have no entry in CUSTOM_TX_EQUIVALENTS"
    assert not equivalents - templates, \
        f"CUSTOM_TX_EQUIVALENTS lists template(s) {sorted(equivalents - templates)} that no longer exist"


@P.requires_feature('custom_transducers')
@pytest.mark.parametrize('template', sorted(P.CUSTOM_TX_EQUIVALENTS))
def test_custom_template_matches_its_equivalent(template):
    path = P.CUSTOM_TX_TEMPLATES_DIR / f"{template}.yaml"
    if not path.exists():
        pytest.skip(f"{path} does not exist (reported by test_every_custom_template_has_builtin_equivalent)")
    with open(path) as f:
        config = yaml.safe_load(f)
    equivalent = P.CUSTOM_TX_EQUIVALENTS[template]

    tx = P.TRANSDUCERS_BY_NAME.get(equivalent['transducer'])
    assert tx is not None, f"{equivalent['transducer']} is not a built-in transducer"
    assert tx['transducer_type'] == config['geometry_type'], \
        f"{template} is a {config['geometry_type']} template but {tx['name']} is a {tx['transducer_type']}"
    assert equivalent['freq_khz'] * 1000 in config['frequencies'], \
        f"{template} template doesn't include {equivalent['freq_khz']} kHz ({config['frequencies']})"


# ================================================================================================================================
# TEST DATA
# ================================================================================================================================
def _required_inputs():
    """Inputs the release test matrix needs."""
    files = []
    for ds in P.DATASETS.values():
        files += [ds['m2m_folder_path'], ds['T1_path']]
        for trajectory_type in ('brainsight', 'slicer'):
            for target in P.TESTED_INDIVIDUAL_TARGETS:
                files.append(str(P.trajectory_file(trajectory_type, ds['id'], target)))
    files += [str(P.THERMAL_PROFILES_DIR / P.DEFAULT_THERMAL_PROFILE),
              str(P.THERMAL_PROFILES_DIR / P.INTENSIVE_THERMAL_PROFILE),
              str(P.MULTIPOINT_PROFILES_DIR / P.DEFAULT_MULTIPOINT_PROFILE),
              os.path.join(P.DATASETS['ID_0082']['folder_path'], P.CT_TYPE_FILES['CT'])]
    return files


def _optional_inputs():
    """Inputs for parameter values that are skipped when their file is missing."""
    files = []
    for ds in P.DATASETS.values():
        for trajectory_type in P.TRAJECTORY_TYPES:
            for target in list(P.VALID_INDIVIDUAL_TARGETS) + list(P.VALID_COMBINED_TARGETS) + list(P.INVALID_TARGETS):
                files.append(str(P.trajectory_file(trajectory_type, ds['id'], target)))
        files += [os.path.join(ds['folder_path'], f) for f in P.CT_TYPE_FILES.values()]
    return sorted(set(files) - set(_required_inputs()))


def test_required_test_inputs_exist():
    missing = [f for f in _required_inputs() if not os.path.exists(f)]
    assert not missing, "Missing test inputs:\n" + "\n".join(missing)


def test_optional_test_inputs_exist():
    missing = [f for f in _optional_inputs() if not os.path.exists(f)]
    if missing:
        pytest.skip(f"{len(missing)} optional inputs are missing, the tests using them are skipped:\n"
                    + "\n".join(missing))
