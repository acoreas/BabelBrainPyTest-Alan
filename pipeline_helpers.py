"""
Shared helpers to set up, run and compare BabelBrain pipelines in tests (used through the
`pipeline` fixture in conftest.py, and by the release test matrix described in
BabelBrain_Pytest_Usage.md).

The pipeline is driven through the SelFiles/BabelBrain widgets (same approach as
E2E/test_full_pipeline.py) instead of BabelBrain/scripting.py, so the same tests
can be pointed at an older BabelBrain checkout (e.g. v0.8.1) to generate the
reference outputs.

Output persistence
------------------
Regression/E2E cases write their outputs to

    <gen_output_folder>/<suite>/<case_id>/

and only drop a completion marker (PYTEST_MARKER) once every requested step
finished without error. A later test (or a later pytest session) that needs the
same case reuses those outputs instead of rerunning the simulation, as long as
they were produced by the same BabelBrain commit, trajectory and profile files. A folder without
a marker is a run that died halfway: it is deleted and the case is rerun.

    BABEL_PYTEST_REGENERATE=1   ignore saved outputs and rerun every case

Reference outputs are looked up in <ref_output_folder_1>/<suite>/<case_id>/,
falling back to the same case produced with a different computing backend.
"""
import datetime
import glob
import hashlib
import logging
import os
import re
import shutil
import subprocess
import sys
from dataclasses import asdict, dataclass, replace
from pathlib import Path

import nibabel
import numpy as np
import pytest
import trimesh
import yaml
from PySide6.QtWidgets import QMessageBox

from Tests.parameters import (BABELBRAIN_ROOT, COREGISTRATION, CT_TYPE_FILES,
                              CT_TYPES, DATASETS, DEFAULT_TARGET,
                              DEFAULT_THERMAL_PROFILE, MULTIPOINT_PROFILES_DIR,
                              RELEASE_BACKENDS, SIMNIBS_TYPES,
                              THERMAL_PROFILES_DIR, TRAJECTORY_TYPES,
                              gen_output_dir, gpu_device, ref_output_dir,
                              simnibs_root_path, trajectory_file)

# ================================================================================================================================
# SETTINGS
# ================================================================================================================================
PYTEST_MARKER = 'pytest_case_complete.yaml'
REGENERATE_ENV = 'BABEL_PYTEST_REGENERATE'

# Timeouts for each pipeline step (ms)
STEP_TIMEOUTS_MS = {1: 30 * 60 * 1000, 2: 3 * 60 * 60 * 1000, 3: 60 * 60 * 1000}

# Ignore failures to generate outputs that were already reported this session, so the
# same broken case isn't rerun for every tolerance/comparison test that needs it
_SESSION_GENERATION_FAILURES = {}


class PipelineError(RuntimeError):
    pass


# ================================================================================================================================
# CASE DEFINITION
# ================================================================================================================================
@dataclass(frozen=True)
class PipelineCase:
    dataset: str
    transducer: str
    freq_khz: int
    backend: str
    target: str = DEFAULT_TARGET
    trajectory_type: str = 'brainsight'
    ct_type: str = 'NONE'
    thermal_profile: str = DEFAULT_THERMAL_PROFILE  # file name in Tests/Profiles
    multipoint_profile: str = None      # file name in <BabelBrain>/Profiles, e.g. 'MultiFocus_Profile1.yaml'
    segment_soft_tissue: bool = False
    custom_tx_template: str = None      # template name in custom_tx_templates/, transducer is then the generated custom tx
    last_step: int = 3
    ppw: int = 6

    @property
    def case_id(self):
        tx = f"CustomTx-{self.custom_tx_template}" if self.custom_tx_template else self.transducer
        parts = [self.dataset, self.trajectory_type, f"CT-{self.ct_type}", self.target, tx,
                 f"Freq-{self.freq_khz}kHz", self.backend]
        if self.thermal_profile != DEFAULT_THERMAL_PROFILE:
            parts.append('TP-' + Path(self.thermal_profile).stem)
        if self.multipoint_profile:
            parts.append('MP-' + Path(self.multipoint_profile).stem)
        if self.segment_soft_tissue:
            parts.append('SoftTissueSeg')
        if self.last_step < 3:
            parts.append(f"Step{self.last_step}")
        return '_'.join(parts)


def case_param(case, marks=()):
    """pytest.param for a case, with the backend marks every GPU case needs."""
    return pytest.param(case, id=case.case_id, marks=[pytest.mark.gpu, *marks])


# ================================================================================================================================
# INPUT RESOLUTION
# ================================================================================================================================
def dataset_paths(case):
    ds = DATASETS[case.dataset]
    for key in ('m2m_folder_path', 'T1_path'):
        if not os.path.exists(ds[key]):
            pytest.skip(f"Dataset {case.dataset} is missing {ds[key]}")
    return ds


def trajectory_path(case):
    traj = trajectory_file(case.trajectory_type, case.dataset, case.target)
    if not traj.exists():
        pytest.skip(f"Trajectory file not found: {traj}")
    return traj


def ct_path(case):
    if case.ct_type == 'NONE':
        return None
    ct = Path(DATASETS[case.dataset]['folder_path']) / CT_TYPE_FILES[case.ct_type]
    if not ct.exists():
        pytest.skip(f"Dataset {case.dataset} does not possess a {case.ct_type} file ({ct})")
    return ct


def thermal_profile_path(case):
    profile = THERMAL_PROFILES_DIR / case.thermal_profile
    if not profile.exists():
        pytest.skip(f"Thermal profile not found: {profile}")
    return profile


def multipoint_profile_path(case):
    profile = MULTIPOINT_PROFILES_DIR / case.multipoint_profile
    if not profile.exists():
        pytest.skip(f"Multipoint profile not found: {profile}")
    return profile


def simnibs_root():
    """SimNIBS install folder from config.ini, needed for soft tissue segmentation and PlanTUS."""
    if not simnibs_root_path or not os.path.isdir(os.path.join(simnibs_root_path, 'simnibs_env')):
        pytest.skip("simnibs_root_path in Tests/config.ini is not set to a SimNIBS install (folder containing simnibs_env)")
    return simnibs_root_path


def babelbrain_commit():
    """Current BabelBrain git commit (with '-dirty' suffix for uncommitted changes)."""
    try:
        commit = subprocess.run(['git', 'rev-parse', 'HEAD'], cwd=BABELBRAIN_ROOT,
                                capture_output=True, text=True, check=True).stdout.strip()
        dirty = subprocess.run(['git', 'status', '--porcelain', '--untracked-files=no'], cwd=BABELBRAIN_ROOT,
                               capture_output=True, text=True, check=True).stdout.strip()
        return commit + ('-dirty' if dirty else '')
    except (OSError, subprocess.CalledProcessError):
        return 'unknown'


def babelbrain_version():
    with open(BABELBRAIN_ROOT / 'BabelBrain' / 'version-gui.txt') as f:
        return f.readline().strip()


# ================================================================================================================================
# DIALOGS
# ================================================================================================================================
def auto_answer_dialogs(monkeypatch):
    """Make modal QMessageBoxes return immediately so an unattended run never stalls.
    Questions answer Yes so cached results are always recalculated."""
    monkeypatch.setattr(QMessageBox, 'question', staticmethod(lambda *a, **k: QMessageBox.Yes))
    monkeypatch.setattr(QMessageBox, 'warning', staticmethod(lambda *a, **k: QMessageBox.Ok))
    monkeypatch.setattr(QMessageBox, 'critical', staticmethod(lambda *a, **k: QMessageBox.Ok))
    monkeypatch.setattr(QMessageBox, 'information', staticmethod(lambda *a, **k: QMessageBox.Ok))
    monkeypatch.setattr(QMessageBox, 'exec', lambda self, *a, **k: QMessageBox.Ok)
    monkeypatch.setenv('BABEL_PYTEST', '1')


# ================================================================================================================================
# BABELBRAIN SETUP
# ================================================================================================================================
def select_computing_engine(sf, backend, remote_server_name=None):
    """Select the requested backend (or remote server) in SelFiles, skipping the test when
    this machine doesn't offer it."""
    combo = sf.ui.ComputingEnginecomboBox
    labels = [combo.itemText(i) for i in range(combo.count())]

    if remote_server_name is not None:
        wanted = [i for i, label in enumerate(labels) if label.startswith(f"Remote: {remote_server_name} ")]
    else:
        backend_rows = [i for i, label in enumerate(labels) if label.endswith(f"-- {backend}")]
        # Prefer the GPU named in config.ini, otherwise take the first device with that backend
        wanted = [i for i in backend_rows if gpu_device in labels[i]] or backend_rows

    if not wanted:
        target = f"remote server '{remote_server_name}'" if remote_server_name else f"{backend} backend"
        pytest.skip(f"{target} is not available on this machine (available: {labels})")
    combo.setCurrentIndex(wanted[0])


def select_transducer(sf, name, is_custom=False):
    if is_custom:
        from GUIComponents.custom_transducer_dialog import CUSTOM_TRANSDUCER_PREFIX
        name = CUSTOM_TRANSDUCER_PREFIX + name
    index = sf.ui.TransducerTypecomboBox.findText(name)
    if index < 0:
        available = [sf.ui.TransducerTypecomboBox.itemText(i) for i in range(sf.ui.TransducerTypecomboBox.count())]
        pytest.fail(f"Transducer {name} not found in transducer list: {available}")
    sf.ui.TransducerTypecomboBox.setCurrentIndex(index)


def validate_selfiles(sf):
    """Mirror SelFiles.Continue's checks without opening its modal dialog."""
    sf.msgDetails = ""
    checks = [(sf.ValidTrajectory(), 'trajectory'),
              (sf.ValidSimNIBS(), 'SimNIBS folder'),
              (sf.ValidThermalProfile(), 'thermal profile'),
              (os.path.isfile(sf.ui.T1WlineEdit.text()), 'T1W file')]
    if hasattr(sf, 'ValidateMultiPointProfile'):
        checks.append((sf.ValidateMultiPointProfile(), 'multipoint profile'))
    if sf.ui.CTTypecomboBox.currentIndex() > 0:
        checks.append((os.path.isfile(sf.ui.CTlineEdit.text()), 'CT/ZTE/PETRA/Density file'))
    invalid = [name for ok, name in checks if not ok]
    if invalid:
        pytest.fail(f"Invalid SelFiles input(s): {', '.join(invalid)}. {sf.msgDetails}")


def build_babelbrain(qtbot, case, output_folder, custom_tx_name=None, remote_server_name=None):
    """Create a BabelBrain widget for `case` that writes everything to `output_folder`."""
    from BabelBrain.BabelBrain import BabelBrain
    from BabelBrain.SelFiles.SelFiles import SelFiles

    ds = dataset_paths(case)
    trajectory_file = trajectory_path(case)
    ct_file = ct_path(case)
    thermal_profile = thermal_profile_path(case)
    output_folder = str(output_folder)
    os.makedirs(output_folder, exist_ok=True)

    # Input selection
    sf = SelFiles()
    qtbot.addWidget(sf)
    sf.ui.TrajectoryTypecomboBox.setCurrentIndex(TRAJECTORY_TYPES[case.trajectory_type])
    sf.ui.TrajectorylineEdit.setText(str(trajectory_file))
    sf.ui.SimbNIBSTypecomboBox.setCurrentIndex(SIMNIBS_TYPES[ds['simbNIBS_type']])
    sf.ui.SimbNIBSlineEdit.setText(ds['m2m_folder_path'])
    sf.ui.T1WlineEdit.setText(ds['T1_path'])
    sf.ui.CTTypecomboBox.setCurrentIndex(CT_TYPES[case.ct_type])
    if ct_file is not None:
        sf.ui.CoregCTcomboBox.setCurrentIndex(COREGISTRATION['yes'])
        sf.ui.CTlineEdit.setText(str(ct_file))
    sf.ui.ThermalProfilelineEdit.setText(str(thermal_profile))
    select_transducer(sf, custom_tx_name or case.transducer, is_custom=custom_tx_name is not None)
    select_computing_engine(sf, case.backend, remote_server_name)
    if case.multipoint_profile:
        if not sf.ui.MultiPointTypecomboBox.isEnabled():
            pytest.fail(f"Multipoint is not available for transducer {case.transducer}")
        sf.ui.MultiPointTypecomboBox.setCurrentIndex(1)
        sf.ui.MultiPointlineEdit.setText(str(multipoint_profile_path(case)))
    elif sf.ui.MultiPointTypecomboBox.isEnabled():
        sf.ui.MultiPointTypecomboBox.setCurrentIndex(0)
    validate_selfiles(sf)

    bb = BabelBrain(sf, AltOutputFilesPath=output_folder)
    bb.show()
    qtbot.addWidget(bb)

    # Advanced options start from defaults, not from whatever the last GUI session saved
    try:
        from Options.Options import ApplyAdvancedConfig
        ApplyAdvancedConfig(bb.Config, bb._AllTransducers, force=True)
    except ImportError:
        logging.warning("ApplyAdvancedConfig not available in this BabelBrain version, using saved advanced options")
    if case.segment_soft_tissue:
        bb.Config['bSegmentBrainTissue'] = True
        bb.Config['SimbNINBSRoot'] = simnibs_root()

    redirect_inputs_to_output_folder(bb, output_folder, ct_file, copy_mesh=case.segment_soft_tissue)

    # Simulation parameters
    freq_index = bb.Widget.USMaskkHzDropDown.findText(str(case.freq_khz))
    if freq_index < 0:
        available = [bb.Widget.USMaskkHzDropDown.itemText(i) for i in range(bb.Widget.USMaskkHzDropDown.count())]
        pytest.fail(f"{case.freq_khz} kHz is not available for {case.transducer}: {available}")
    bb.Widget.USMaskkHzDropDown.setCurrentIndex(freq_index)
    bb.Widget.USPPWSpinBox.setProperty('UserData', case.ppw)
    if case.ct_type != 'NONE':
        bb.Widget.HUThresholdSpinBox.setValue(300)

    return bb


def redirect_inputs_to_output_folder(bb, output_folder, ct_file, copy_mesh=False):
    """Copy the inputs BabelBrain writes next to into the output folder, so the test data
    folder is never modified and every generated file lands in the case folder."""
    shutil.copy(bb.Config['T1W'], os.path.join(output_folder, os.path.basename(bb.Config['T1W'])))
    if ct_file is not None:
        shutil.copy(ct_file, os.path.join(output_folder, os.path.basename(ct_file)))

    m2m_src = os.path.normpath(bb.Config['simbnibs_path'])
    m2m_dst = os.path.join(output_folder, os.path.basename(m2m_src))
    os.makedirs(m2m_dst, exist_ok=True)
    m2m_files = ['final_tissues.nii.gz'] if bb.Config['SimbNIBSType'] == 'charm' else ['skin.nii.gz']
    if copy_mesh:  # soft tissue segmentation reads the charm mesh
        m2m_files += [os.path.basename(f) for f in glob.glob(os.path.join(m2m_src, '*.msh'))]
    for fname in m2m_files:
        shutil.copy(os.path.join(m2m_src, fname), os.path.join(m2m_dst, fname))

    trajectory_new_file = os.path.join(output_folder, os.path.basename(bb.Config['Mat4Trajectory']))
    shutil.copy(bb.Config['Mat4Trajectory'], trajectory_new_file)

    # Config['ID'] is a plain string up to v0.8.x and a list (one entry per trajectory) after
    # the dual-Tx refactor. Keep whatever shape this version uses.
    new_ID = os.path.splitext(os.path.basename(trajectory_new_file))[0]
    if isinstance(bb.Config['ID'], str):
        bb.Config['ID'] = new_ID
    else:
        n_traj = len(bb.Config['ID'])
        bb.Config['ID'] = [new_ID] if n_traj == 1 else [f"{new_ID}_{i}" for i in range(n_traj)]

    bb.Config['Mat4Trajectory'] = trajectory_new_file
    bb.Config['T1WIso'] = os.path.join(output_folder, os.path.basename(bb.Config['T1WIso']))
    bb.Config['simbnibs_path'] = m2m_dst
    if bb.Config['bUseCT']:
        bb.Config['CT_or_ZTE_input'] = os.path.join(output_folder, os.path.basename(bb.Config['CT_or_ZTE_input']))


# ================================================================================================================================
# PIPELINE EXECUTION
# ================================================================================================================================
def _wait_for_step(qtbot, bb, step, screenshots, image_to_base64):
    try:
        qtbot.waitUntil(bb.Widget.tabWidget.isEnabled, timeout=STEP_TIMEOUTS_MS[step])
    except qtbot.TimeoutError as error:
        raise PipelineError(f"Step {step} did not finish within {STEP_TIMEOUTS_MS[step] / 60000:.0f} min") from error
    qtbot.wait(1000)  # Wait for plots to display
    if screenshots is not None and image_to_base64 is not None:
        screenshots.append(image_to_base64(qtbot.screenshot(bb)))
    if bb.testing_error:
        raise PipelineError(f"BabelBrain reported an error in step {step}")


def _set_cone_distance(bb):
    """Transducers with a cone and Z steering: place the focus at the skin-to-target distance,
    steering for whatever the cone can't cover."""
    if not hasattr(bb.AcSim.Widget, 'DistanceConeToFocusSpinBox'):
        return
    cone = bb.AcSim.Widget.DistanceConeToFocusSpinBox
    skin_to_target = bb.AcSim.Widget.DistanceSkinLabel.property('UserData')
    z_steering = 0.0
    if cone.minimum() <= skin_to_target <= cone.maximum():
        cone.setValue(skin_to_target)
    elif skin_to_target > cone.maximum():
        cone.setValue(cone.maximum())
        z_steering = skin_to_target - cone.maximum()
    else:
        cone.setValue(cone.minimum())
        z_steering = skin_to_target - cone.minimum()
    bb.AcSim.Widget.ZSteeringSpinBox.setValue(z_steering)


def run_pipeline(qtbot, bb, last_step=3, screenshots=None, image_to_base64=None, export_maps=True):
    """Run BabelBrain steps 1..last_step (plus exports after step 3), adding a screenshot of the GUI
    after each step to `screenshots`. Raises PipelineError on failure."""
    bb.testing_error = False
    bb.Widget.CalculatePlanningMask.click()
    _wait_for_step(qtbot, bb, 1, screenshots, image_to_base64)
    if last_step == 1:
        return

    bb.Widget.tabWidget.setCurrentIndex(1)
    _set_cone_distance(bb)
    bb.AcSim.Widget.CalculateAcField.click()
    _wait_for_step(qtbot, bb, 2, screenshots, image_to_base64)
    if last_step == 2:
        return

    bb.Widget.tabWidget.setCurrentIndex(2)
    bb.ThermalSim.Widget.CalculateThermal.click()
    _wait_for_step(qtbot, bb, 3, screenshots, image_to_base64)

    with qtbot.captureExceptions() as exceptions:
        bb.ThermalSim.Widget.ExportSummary.click()
        if export_maps:
            bb.ThermalSim.Widget.ExportMaps.click()
    if exceptions:
        raise PipelineError(f"Error exporting results: {exceptions}")


def thermal_outputs(folder):
    return sorted(glob.glob(os.path.join(str(folder), '**', '*DataForSim-ThermalField-Duration*.h5'), recursive=True))


# ================================================================================================================================
# OUTPUT PERSISTENCE
# ================================================================================================================================
def case_output_dir(suite, case):
    return Path(gen_output_dir) / suite / case.case_id


def read_marker(folder):
    marker = Path(folder) / PYTEST_MARKER
    if not marker.exists():
        return None
    with open(marker) as f:
        return yaml.safe_load(f) or {}


def input_hashes(case):
    """SHA-256 of the small input files a case reads (trajectory and profiles), so editing one
    invalidates saved outputs."""
    files = [trajectory_path(case), thermal_profile_path(case)]
    if case.multipoint_profile:
        files.append(multipoint_profile_path(case))
    return {f.name: hashlib.sha256(f.read_bytes()).hexdigest() for f in files}


def saved_outputs_reusable(suite, case):
    """True when complete outputs for `case` from the current BabelBrain commit and the current
    trajectory/profile files are saved."""
    marker = read_marker(case_output_dir(suite, case))
    if marker is None or os.environ.get(REGENERATE_ENV) == '1':
        return False
    if marker.get('babelbrain_commit') != babelbrain_commit():
        logging.info(f"Saved outputs for {case.case_id} were generated by BabelBrain commit "
                     f"{marker.get('babelbrain_commit')}, they will be regenerated")
        return False
    if marker.get('input_hashes') != input_hashes(case):
        logging.info(f"Input files changed since the saved outputs for {case.case_id} were generated, "
                     "they will be regenerated")
        return False
    return True



def ensure_case_outputs(qtbot, monkeypatch, suite, case, screenshots=None, image_to_base64=None,
                        custom_tx_name=None, remote_server_name=None):
    """Return the folder holding the outputs of `case`, running the pipeline only if no
    complete outputs from this BabelBrain commit were saved before.

    Outputs are written straight to <gen_output_folder>/<suite>/<case_id> and marked complete
    as soon as the pipeline finishes, before anything is compared, so a later failure (or a
    crashed session) never throws away a finished simulation."""
    out_dir = case_output_dir(suite, case)
    commit = babelbrain_commit()

    if case.case_id in _SESSION_GENERATION_FAILURES:
        pytest.fail(f"Generating outputs for {case.case_id} already failed this session: "
                    f"{_SESSION_GENERATION_FAILURES[case.case_id]}")

    if saved_outputs_reusable(suite, case):
        logging.info(f"Reusing saved outputs for {case.case_id} in {out_dir}")
        return out_dir

    # The case folder only holds copied inputs and outputs, so a rerun starts from an empty one
    # (otherwise BabelBrain's own cache would reload half-written or outdated step results)
    if out_dir.exists():
        shutil.rmtree(out_dir)

    auto_answer_dialogs(monkeypatch)
    started = datetime.datetime.now()
    try:
        bb = build_babelbrain(qtbot, case, out_dir, custom_tx_name=custom_tx_name,
                              remote_server_name=remote_server_name)
        run_pipeline(qtbot, bb, last_step=case.last_step, screenshots=screenshots, image_to_base64=image_to_base64)
        if case.last_step == 3 and not thermal_outputs(out_dir):
            raise PipelineError(f"Pipeline finished but no thermal output was saved in {out_dir}")
    except Exception as error:  # pytest skip/fail outcomes are BaseExceptions and pass through
        _SESSION_GENERATION_FAILURES[case.case_id] = f"{type(error).__name__}: {error}"
        pytest.fail(f"Failed to generate outputs for {case.case_id}: {type(error).__name__}: {error}")

    with open(out_dir / PYTEST_MARKER, 'w') as f:
        yaml.safe_dump({'case': asdict(case),
                        'case_id': case.case_id,
                        'babelbrain_version': babelbrain_version(),
                        'babelbrain_commit': commit,
                        'input_hashes': input_hashes(case),
                        'python': sys.version.split()[0],
                        'started': started.isoformat(timespec='seconds'),
                        'finished': datetime.datetime.now().isoformat(timespec='seconds')}, f, sort_keys=False)
    return out_dir


def find_reference_dir(suite, case):
    """Reference outputs for `case`: same case id, else the same case from another backend."""
    suite_dir = Path(ref_output_dir) / suite
    exact = suite_dir / case.case_id
    if exact.is_dir():
        return exact
    for backend in RELEASE_BACKENDS + ['MLX']:
        if backend == case.backend:
            continue
        candidate = suite_dir / replace(case, backend=backend).case_id
        if candidate.is_dir():
            logging.warning(f"No {case.backend} reference for {case.case_id}, using {candidate.name}")
            return candidate
    pytest.skip(f"No reference outputs for {case.case_id} in {suite_dir}")


# ================================================================================================================================
# OUTPUT COMPARISON
# ================================================================================================================================
def _thermal_key(fname):
    """Part of a thermal output name that doesn't depend on transducer/trajectory naming."""
    base = os.path.basename(fname)
    return base[base.index('DataForSim-ThermalField'):]


def compare_thermal_outputs(ref_folder, test_folder, tolerance, screenshots, compare_data):
    """Compare every thermal h5 output in test_folder with its counterpart in ref_folder.
    Returns (all_match, problems)."""
    ref_files = {_thermal_key(f): f for f in thermal_outputs(ref_folder)}
    test_files = {_thermal_key(f): f for f in thermal_outputs(test_folder)}
    problems = []
    if not ref_files:
        problems.append(f"No thermal outputs in reference folder {ref_folder}")
    if not test_files:
        problems.append(f"No thermal outputs in test folder {test_folder}")
    problems += [f"Missing in test outputs: {k}" for k in sorted(set(ref_files) - set(test_files))]
    problems += [f"Missing in reference outputs: {k}" for k in sorted(set(test_files) - set(ref_files))]

    matches = []
    for key in sorted(set(ref_files) & set(test_files)):
        matches.append(compare_data['h5_data'](ref_files[key], test_files[key], screenshots,
                                               tolerance=tolerance, label=key))
    return (all(matches) and not problems), problems


# Step 1 outputs compared by compare_all_outputs, and files never worth comparing
_ALL_OUTPUT_PATTERNS = ['*.nii.gz', '*.nii', '*.npy', '*.stl', '*.h5']
_IGNORED_OUTPUTS = re.compile(r'(ExecutionTimes|pytest_case_complete)')


def _relative_outputs(folder):
    files = set()
    for pattern in _ALL_OUTPUT_PATTERNS:
        for f in glob.glob(os.path.join(str(folder), '**', pattern), recursive=True):
            if not _IGNORED_OUTPUTS.search(os.path.basename(f)):
                files.add(os.path.relpath(f, str(folder)))
    return files


def _compare_arrays(a, b, tolerance):
    if a.shape != b.shape:
        return False, f"shape {a.shape} vs {b.shape}"
    if not (np.issubdtype(a.dtype, np.number) or a.dtype == bool):  # e.g. FileManager hash files
        ok = np.array_equal(a, b)
        return ok, "identical" if ok else f"{a} vs {b}"
    a64 = a.astype(np.float64)
    b64 = b.astype(np.float64)
    ok = np.allclose(a64, b64, rtol=tolerance, atol=0, equal_nan=True)
    diff = np.abs(a64 - b64)
    return ok, f"max abs diff {np.nanmax(diff) if diff.size else 0:.6g}, mismatched voxels {int(np.sum(~np.isclose(a64, b64, rtol=tolerance, atol=0, equal_nan=True)))}"


def _compare_file(ref, test, tolerance, screenshots, compare_data):
    if ref.endswith('.h5'):
        return compare_data['h5_data'](ref, test, screenshots, tolerance=tolerance, label=os.path.basename(ref)), 'see h5 table'
    if ref.endswith('.npy'):
        return _compare_arrays(np.load(ref), np.load(test), tolerance)
    if ref.endswith('.stl'):
        ref_mesh, test_mesh = trimesh.load(ref), trimesh.load(test)
        if len(ref_mesh.vertices) != len(test_mesh.vertices) and tolerance == 0:
            return False, f"vertex count {len(ref_mesh.vertices)} vs {len(test_mesh.vertices)}"
        area_error = compare_data['stl_area'](test_mesh, ref_mesh)
        return area_error <= tolerance, f"area relative error {area_error:.6g}"
    # NIfTI
    ref_nii, test_nii = nibabel.load(ref), nibabel.load(test)
    if not np.allclose(ref_nii.affine, test_nii.affine, rtol=1e-5, atol=1e-5):
        return False, "affines differ"
    return _compare_arrays(np.asanyarray(ref_nii.dataobj), np.asanyarray(test_nii.dataobj), tolerance)


def compare_all_outputs(ref_folder, test_folder, tolerance, screenshots, compare_data):
    """Compare every NIfTI/npy/stl/h5 output of a case with its reference. Returns (all_match, problems)."""
    ref_files = _relative_outputs(ref_folder)
    test_files = _relative_outputs(test_folder)
    problems = [f"Missing in test outputs: {f}" for f in sorted(ref_files - test_files)]
    for f in sorted(test_files - ref_files):
        logging.warning(f"Output not present in reference, not compared: {f}")
    if not ref_files:
        problems.append(f"No outputs in reference folder {ref_folder}")

    rows = []
    for rel in sorted(ref_files & test_files):
        logging.info(f"Comparing {rel}")
        ok, detail = _compare_file(os.path.join(str(ref_folder), rel), os.path.join(str(test_folder), rel),
                                   tolerance, screenshots, compare_data)
        rows.append((rel, ok, detail))
        if not ok:
            problems.append(f"{rel}: {detail}")

    table = ''.join(f"<tr style='background-color:{'#c6f6c6' if ok else '#f6c6c6'}'><td>{rel}</td><td>{detail}</td></tr>"
                    for rel, ok, detail in rows)
    screenshots.append({'kind': 'html', 'html': (
        f"<h3>Output comparison (rtol={tolerance})</h3>"
        f"<div style='font-size:12px;color:#555'>Reference: {ref_folder}<br>Test: {test_folder}</div>"
        "<table border='1' style='border-collapse:collapse;font-size:12px'>"
        f"<tr><th>File</th><th>Result</th></tr>{table}</table>")})
    return not problems, problems
