"""
Single source of truth for the parameters the test suite runs with (see BabelBrain_Pytest_Usage.md,
Table 2). conftest.py, pipeline_helpers.py and the tests all read from here.

Anything that describes the app (transducers, their frequencies and geometry types, custom
transducer templates, the order of the options in SelFiles' drop-downs) is read from the
BabelBrain checkout under test, so new transducers/options are picked up without editing the
tests. Anything the tests choose (datasets, targets, profiles, the release test matrix) is
declared here, and Tests/test_suite_consistency.py fails when those choices no longer match
the app or the test data.
"""
import configparser
import os
import xml.etree.ElementTree as ET
from pathlib import Path

import pytest
import yaml

TESTS_DIR = Path(__file__).resolve().parent.parent
BABELBRAIN_ROOT = TESTS_DIR.parent
BABELBRAIN_SRC = BABELBRAIN_ROOT / 'BabelBrain'

# ================================================================================================================================
# CONFIG.INI
# ================================================================================================================================
config = configparser.ConfigParser()
config.read(TESTS_DIR / 'config.ini')
gpu_device = config['GPU']['device_name']                   # GPU device used for test
test_data_folder = config['Paths']['data_folder_path']      # Folder containing input test data
ref_output_dir = config['Paths']['ref_output_folder_1']     # Previously generated outputs, used in regression tests
ref_output_dir_2 = config['Paths']['ref_output_folder_2']   # Second set of outputs, used in test_full_pipeline_two_outputs
gen_output_dir = config['Paths']['gen_output_folder']       # Where generate_outputs tests save their outputs
simnibs_root_path = config['Paths'].get('simnibs_root_path', '')  # SimNIBS install (soft tissue segmentation, PlanTUS)

# ================================================================================================================================
# APP OPTIONS (read from the BabelBrain checkout)
# ================================================================================================================================
def selfiles_combo_items(combo_name):
    """Option labels of a SelFiles drop-down, in GUI order, read from SelFiles/form.ui."""
    root = ET.parse(BABELBRAIN_SRC / 'SelFiles' / 'form.ui').getroot()
    widget = root.find(f".//widget[@name='{combo_name}']")
    if widget is None:
        return []
    return [item.findtext("property[@name='text']/string") for item in widget.findall('item')]


def _option_indices(combo_name, labels):
    """{test name: drop-down index} for the options of `labels` ({test name: GUI label}) that exist."""
    items = selfiles_combo_items(combo_name)
    return {key: items.index(label) for key, label in labels.items() if label in items}


# Test name -> GUI label of each option. Indices come from the GUI, so reordering the options
# in BabelBrain doesn't silently select the wrong one.
CT_TYPE_LABELS = {'NONE': 'NO', 'CT': 'real CT', 'ZTE': 'ZTE', 'PETRA': 'PETRA', 'Density': 'Density'}
TRAJECTORY_TYPE_LABELS = {'brainsight': 'Brainsight', 'slicer': 'Slicer', 'localite': 'Localite'}
SIMNIBS_TYPE_LABELS = {'charm': 'charm', 'headreco': 'headreco'}
COREGISTRATION_LABELS = {'no': 'NO', 'yes': 'CT to MRI'}

CT_TYPES = _option_indices('CTTypecomboBox', CT_TYPE_LABELS)  # index is also BabelBrain's CTType code
TRAJECTORY_TYPES = _option_indices('TrajectoryTypecomboBox', TRAJECTORY_TYPE_LABELS)
SIMNIBS_TYPES = _option_indices('SimbNIBSTypecomboBox', SIMNIBS_TYPE_LABELS)
COREGISTRATION = _option_indices('CoregCTcomboBox', COREGISTRATION_LABELS)

# File name of each extra scan in a dataset folder
CT_TYPE_FILES = {'CT': 'CT.nii.gz', 'ZTE': 'ZTE.nii.gz', 'PETRA': 'PETRA.nii.gz', 'Density': 'Density.nii.gz'}


def _load_transducers_pre_v0_8_2():
    """Built-in transducers of versions before SelFiles/transducer_list.yaml existed (v0.8.1 and
    earlier): names come from the SelFiles drop-down and each transducer lives in Babel_<module>/.
    Geometry types weren't defined yet, so transducer_type is None."""
    # Names whose module differs, mirroring the idimport table in BabelBrain.py of those versions
    modules = {'CTX_500': 'CTX500', 'CTX_250': 'CTX250', 'CTX_250_2ch': 'CTX250_2ch', 'DPX_500': 'DPX500',
               'DPXPC_300': 'DPXPC300', 'Single': 'SingleTx', 'BSonix': 'SingleTx'}
    transducers = []
    for index, name in enumerate(selfiles_combo_items('TransducerTypecomboBox')):
        module = modules.get(name, name)
        default_yaml = BABELBRAIN_SRC / f"Babel_{module}" / ('defaultBSonix.yaml' if name == 'BSonix' else 'default.yaml')
        assert default_yaml.is_file(), f"missing yaml for {name}: {default_yaml}"
        with open(default_yaml) as f:
            freqs = yaml.safe_load(f)['USFrequencies']
        transducers.append({'name': name, 'module_name': module, 'transducer_type': None,
                            'dropdown_index': index, 'freqs': freqs})
    return transducers


def _load_transducers():
    """Built-in transducers from SelFiles/transducer_list.yaml, with the frequencies from each
    transducer's default.yaml."""
    if not (BABELBRAIN_SRC / 'SelFiles' / 'transducer_list.yaml').is_file():
        return _load_transducers_pre_v0_8_2()
    with open(BABELBRAIN_SRC / 'SelFiles' / 'transducer_list.yaml') as f:
        transducer_list = yaml.safe_load(f)

    transducers = []
    for index, tx in enumerate(transducer_list):
        default_yaml = BABELBRAIN_SRC / 'babel_transducers' / tx['transducer_type'] / tx['module_name'] / 'default.yaml'
        assert default_yaml.is_file(), f"missing yaml for {tx['name']}: {default_yaml}"
        with open(default_yaml) as f:
            freqs = yaml.safe_load(f)['USFrequencies']
        transducers.append({**tx, 'dropdown_index': index, 'freqs': freqs})
    return transducers


TRANSDUCERS = _load_transducers()
TRANSDUCERS_BY_NAME = {tx['name']: tx for tx in TRANSDUCERS}
TRANSDUCER_TYPES = sorted({tx['transducer_type'] for tx in TRANSDUCERS if tx['transducer_type']})  # empty before v0.8.2

CUSTOM_TX_TEMPLATES_DIR = BABELBRAIN_ROOT / 'custom_tx_templates'
CUSTOM_TX_TEMPLATES = sorted(p.stem for p in CUSTOM_TX_TEMPLATES_DIR.glob('*.yaml'))

# ================================================================================================================================
# FEATURES (so tests for features an older BabelBrain version doesn't have are skipped)
# ================================================================================================================================
with open(BABELBRAIN_SRC / 'version-gui.txt') as f:
    BABELBRAIN_VERSION = f.readline().strip()

FEATURES = {
    'remote_servers': (BABELBRAIN_SRC / 'RemoteServers.py').is_file(),
    'custom_transducers': ((BABELBRAIN_SRC / 'CreateTransducers' / 'transducer_creator.py').is_file()
                           and CUSTOM_TX_TEMPLATES_DIR.is_dir()),
    'plantus': ((BABELBRAIN_SRC / 'PlanTUSViewer' / 'RunPlanTUS.py').is_file()
                and (BABELBRAIN_SRC / 'ExternalBin' / 'PlanTUS' / 'PlanTUS' / 'PlanTUS_wrapper.py').is_file()),
}


def requires_feature(feature):
    """Marker skipping a test when the BabelBrain version under test doesn't have `feature`."""
    return pytest.mark.skipif(not FEATURES[feature],
                              reason=f"BabelBrain {BABELBRAIN_VERSION} doesn't have the {feature} feature")


# ================================================================================================================================
# TEST INPUTS (chosen by the tests)
# ================================================================================================================================
def _dataset(id, folder, m2m, t1):
    m2m_folder = os.path.join(test_data_folder, m2m) + os.sep
    return {
        'id': id,
        'folder_path': os.path.join(test_data_folder, folder) + os.sep,
        'm2m_folder_path': m2m_folder,
        'T1_path': os.path.join(test_data_folder, t1),
        'simbNIBS_type': 'charm' if os.path.exists(m2m_folder + 'charm_log.html') else 'headreco',
    }


DATASETS = {
    'ID_0082': _dataset('ID_0082', 'ID_0082', os.path.join('ID_0082', 'm2m_ID_0082'), os.path.join('ID_0082', 'T1W.nii.gz')),
    'ernie': _dataset('ernie', 'ernie', os.path.join('ernie', 'm2m_ernie'), os.path.join('ernie', 'm2m_ernie', 'T1.nii.gz')),
}

# Small synthetic head (synthetic_dataset.py) for the GPU unit tests, generated at the start of each
# session in the gitignored Tests/.cache folder. GPU unit tests run every spatial step on both the
# synthetic head and the real datasets.
SYNTHETIC_DATASET_DIR = TESTS_DIR / '.cache' / 'synthetic_gpu_dataset'
SYNTHETIC_DATASET = {
    'id': 'synthetic',
    'folder_path': str(SYNTHETIC_DATASET_DIR) + os.sep,
    'm2m_folder_path': str(SYNTHETIC_DATASET_DIR / 'm2m_synthetic') + os.sep,
    'T1_path': str(SYNTHETIC_DATASET_DIR / 'T1W.nii.gz'),
    'simbNIBS_type': 'charm',
}


# Targets: test name -> trajectory file stem in <data>/Trajectories/<trajectory type>_<dataset>_<stem>.<ext>
TRAJECTORIES_DIR = Path(test_data_folder) / 'Trajectories'
VALID_INDIVIDUAL_TARGETS = {  # one target per file
    'Superficial_Target_Left': 'superficial_target_left',
    'Deep_Target_Left': 'deep_target_left',
    'Superficial_Target_Right': 'superficial_target_right',
    'Deep_Target_Right': 'deep_target_right',
}
TESTED_INDIVIDUAL_TARGETS = ['Superficial_Target_Left', 'Deep_Target_Left']  # default `trajectory` parametrization (left side)
VALID_COMBINED_TARGETS = {  # several targets in one file (left and right)
    'Superficial_Targets': 'superficial_targets',
    'Deep_Targets': 'deep_targets',
}
INVALID_TARGETS = {
    'Outside_Target': 'outside_target',
}
DEFAULT_TARGET = 'Superficial_Target_Left'

# Extension of the trajectory files for each type; combined Slicer targets are a yaml file
# listing the individual Slicer files
TRAJECTORY_EXTENSIONS = {'brainsight': '.txt', 'slicer': '.txt', 'localite': '.xml'}
COMBINED_TRAJECTORY_EXTENSIONS = {**TRAJECTORY_EXTENSIONS, 'slicer': '.yaml'}


def trajectory_file(trajectory_type, dataset_id, target):
    if target in VALID_COMBINED_TARGETS:
        stem, ext = VALID_COMBINED_TARGETS[target], COMBINED_TRAJECTORY_EXTENSIONS[trajectory_type]
    else:
        stem = {**VALID_INDIVIDUAL_TARGETS, **INVALID_TARGETS}[target]
        ext = TRAJECTORY_EXTENSIONS[trajectory_type]
    return TRAJECTORIES_DIR / f"{trajectory_type}_{dataset_id}_{stem}{ext}"


# Thermal profiles kept in the pytest repo
THERMAL_PROFILES_DIR = TESTS_DIR / 'Profiles'
DEFAULT_THERMAL_PROFILE = 'pytest_thermal_profile_1.yaml'    # quick sims
INTENSIVE_THERMAL_PROFILE = 'pytest_thermal_profile_2.yaml'  # long sims, repetitions and grouped sonications

# Multipoint profiles shipped with BabelBrain
MULTIPOINT_PROFILES_DIR = BABELBRAIN_ROOT / 'Profiles'
DEFAULT_MULTIPOINT_PROFILE = 'MultiFocus_Profile1.yaml'

# Computing backends. supported_os is only used by the older check_os fixture, the pipeline
# helpers skip based on the engines SelFiles actually lists on this machine.
COMPUTING_BACKENDS = [
    {'type': 'OpenCL', 'supported_os': ['Windows', 'Linux']},
    {'type': 'CUDA',   'supported_os': ['Windows', 'Linux']},
    {'type': 'Metal',  'supported_os': ['Mac']},
    {'type': 'MLX',    'supported_os': ['Mac']},
]
RELEASE_BACKENDS = ['CUDA', 'OpenCL', 'Metal']  # backends in the release test matrix

SPATIAL_STEPS = {
    'Low_Res': 0.919,     # 200 kHz,   6 PPW
    'Med_Res': 0.306,     # 600 kHz,   6 PPW
    'High_Res': 0.184,    # 1000 kHz,  6 PPW
    'Stress_Res': 0.092,  # 1000 kHz, 12 PPW
}

# GPU unit test spatial steps in each release test set (markers minor_release / major_release, see
# conftest.py). The synthetic head runs every resolution; the full head real datasets only a
# cheaper one, since their high resolutions take very long.
RELEASE_SPATIAL_STEPS = {
    'minor_release': {'synthetic': list(SPATIAL_STEPS), 'real': ['Low_Res']},
    'major_release': {'synthetic': list(SPATIAL_STEPS), 'real': ['Med_Res']},
}

# ================================================================================================================================
# RELEASE TEST MATRIX (BabelBrain_Pytest_Usage.md, Tables 9 and 10)
# ================================================================================================================================
# Basic configuration: the standard full pipeline set up (dataset, target, one transducer per
# geometry type, ...). Run as regression_test_full_pipeline, and selected in the older
# E2E/generate-outputs tests by the basic_babelbrain_params marker. Fields as in REGRESSION_CASES.
BASIC_PARAMS = {
    'dataset': 'ernie',
    'trajectory_type': 'brainsight',
    'target': DEFAULT_TARGET,
    'ct_types': ['NONE'],
    # One transducer per general geometry type (+ DomeTx); test_suite_consistency.py fails if a
    # new geometry type is added to BabelBrain without a representative here
    'transducers': [
        ('Single', 500),    # simple focused
        ('CTX_500', 500),   # focused annular array
        ('H246', 500),      # flat annular array
        ('REMOPD', 490),    # flat array 2D
        ('H317', 250),      # focused array
        ('DomeTx', 220),    # focused array
    ],
    'thermal_profile': DEFAULT_THERMAL_PROFILE,
    'multipoint_profile': None,
    'soft_tissue_segmentation': [False],
    'last_step': 3,
    'backends': RELEASE_BACKENDS,
    'require_ct_files': True,
}

# Intensive configuration: a high frequency phased array with CT, multipoint focusing and long
# sonications. Run as regression_test_full_pipeline_intensive, and selected in the older
# E2E/generate-outputs tests by the intensive_babelbrain_params marker. Fields as in REGRESSION_CASES.
INTENSIVE_PARAMS = {
    'dataset': 'ID_0082',
    'trajectory_type': 'brainsight',
    'target': 'Deep_Target_Left',
    'ct_types': ['CT'],
    'transducers': [('H317', 825)],
    'thermal_profile': INTENSIVE_THERMAL_PROFILE,
    'multipoint_profile': DEFAULT_MULTIPOINT_PROFILE,
    'soft_tissue_segmentation': [False],
    'last_step': 3,
    'backends': RELEASE_BACKENDS,
    'require_ct_files': True,
}

# Major release regression tests, one entry per row of Table 10. Each case runs every combination of
# backends x ct_types x transducers x soft_tissue_segmentation (Regression/test_regression_release.py).
#   dataset, trajectory_type, target  inputs (see DATASETS, trajectory_file)
#   ct_types                          extra scan types (CT_TYPES keys)
#   transducers                       (transducer, kHz) pairs
#   thermal_profile                   file in Tests/Profiles
#   multipoint_profile                file in BabelBrain/Profiles, or None for a single focus
#   soft_tissue_segmentation          values of Step 1's soft tissue segmentation option
#   last_step                         last pipeline step run (1 = Step 1 only)
#   require_ct_files                  False when the case covers scan types a dataset may not have
#                                     (those combinations are skipped instead of required)
REGRESSION_CASES = {
    'regression_test_full_pipeline': BASIC_PARAMS,
    'regression_test_full_pipeline_intensive': INTENSIVE_PARAMS,
    'regression_test_GetSkullMaskFromSimbNIBSSTL': {
        'dataset': 'ID_0082',
        'trajectory_type': 'brainsight',
        'target': DEFAULT_TARGET,
        'ct_types': list(CT_TYPES),     # every scan type the GUI offers
        'transducers': [('Single', 500), ('Single', 250)],
        'thermal_profile': DEFAULT_THERMAL_PROFILE,
        'multipoint_profile': None,
        'soft_tissue_segmentation': [False, True],
        'last_step': 1,
        'backends': RELEASE_BACKENDS,
        'require_ct_files': False,
    },
}

# Markers selecting a configuration in the older E2E/generate-outputs tests (applied in
# conftest.pytest_collection_modifyitems). Marked tests also run with the configuration's thermal
# and multipoint profiles.
CONFIGURATION_MARKERS = {
    'basic_babelbrain_params': BASIC_PARAMS,
    'intensive_babelbrain_params': INTENSIVE_PARAMS,
}

# Custom transducer template -> built-in transducer it replicates (see custom_tx_templates/)
CUSTOM_TX_EQUIVALENTS = {
    'simple_focused':        {'transducer': 'Single',  'freq_khz': 500},
    'flat_annular_array':    {'transducer': 'H246',    'freq_khz': 500},
    'focused_annular_array': {'transducer': 'CTX_500', 'freq_khz': 500},
    'flat_array_2D':         {'transducer': 'REMOPD',  'freq_khz': 490},
    'focused_array':         {'transducer': 'H317',    'freq_khz': 250},
}

# ================================================================================================================================
# MARKER SELECTIONS (applied in conftest.pytest_collection_modifyitems)
# ================================================================================================================================
# Transducer frequencies (kHz) each marker selects. include_others=True means every other
# transducer is selected at all its frequencies, so newly added transducers are covered
# automatically; an empty list excludes a transducer.

MARKER_TRANSDUCERS = {
    'all_targets_babelbrain_params': {
        'include_others': True,
        'frequencies': {'Single': [500], 'CTX_500': [500], 'H317': [250], 'REMOPD': [490],
                        'CTX_250': [], 'CTX_250_2ch': []},
    },
    'bare_min_babelbrain_params': {
        'include_others': False,
        'frequencies': {'H317': [250], 'Single': [250], 'CTX_250': None, 'H246': None, 'BSonix': None,
                        'REMOPD': [300]},  # None = all frequencies
    },
}


def marker_selects_transducer(marker, tx_name, freq_hz):
    rule = MARKER_TRANSDUCERS[marker]
    if tx_name not in rule['frequencies']:
        return rule['include_others']
    freqs = rule['frequencies'][tx_name]
    return freqs is None or int(freq_hz / 1000) in freqs
