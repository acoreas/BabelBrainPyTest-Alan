"""
Fixtures and hooks for the whole BabelBrain test suite. The code behind them lives in Tests/helpers:
parameters.py (what the tests run with), collection.py (default parametrization and markers),
comparisons.py, plots.py and report_viewers.py (report entries), pipeline_helpers.py (running
BabelBrain), mesh_checks.py, blender_view.py and synthetic_dataset.py.
"""
import datetime
import logging
import os
import platform
import shutil
import sys

import nibabel
import numpy as np
import pytest
import pytest_html
import SimpleITK as sitk
import trimesh
from nibabel import affines, nifti1, processing

sys.path.append('./BabelBrain/')
np.random.seed(42) # RNG is same every time
from BabelBrain.FileManager import FileManager
import BabelBrain.BabelBrain as bb_module # Needed to get monkeypatch working

# ================================================================================================================================
# PARAMETERS (defined in helpers/parameters.py)
# ================================================================================================================================
from Tests.helpers import collection, comparisons, plots
from Tests.helpers.parameters import (CONFIGURATION_MARKERS, CT_TYPE_FILES, CT_TYPES, SYNTHETIC_DATASET_DIR,
                                      gen_output_dir, gpu_device, ref_output_dir, ref_output_dir_2,
                                      test_data_folder)
from Tests.helpers.report_viewers import report_entries_html

print('Using GPU device: ',gpu_device)

# ================================================================================================================================
# PYTEST FIXTURES
# ================================================================================================================================
@pytest.fixture(scope="session", autouse=True)
def synthetic_dataset():
    """Generate the synthetic GPU unit test dataset (instant, reused when unchanged)."""
    from Tests.helpers.synthetic_dataset import ensure_synthetic_dataset
    ensure_synthetic_dataset(str(SYNTHETIC_DATASET_DIR))

@pytest.fixture()
def mock_confirm_pseudoct(monkeypatch):
    monkeypatch.setattr(bb_module,"ConfirmPseudoCT", lambda *args, **kwargs: True)

@pytest.fixture()
def check_files_exist():

    def _check_files_exist(fnames):
        missing_files = []
        for file in fnames:
            if not os.path.exists(file):
                missing_files.append(file)

        if missing_files:
            return False, missing_files
        else:
            return True, ""

    return _check_files_exist

@pytest.fixture()
def load_files(check_files_exist):
    
    def _load_files(fnames,nifti_load_method='nibabel',skip_test=True):

        if isinstance(fnames,dict):
            datas = fnames.copy()
            fnames_list = fnames.values()
        else:
            datas = []
            fnames_list = fnames

        # Check files exist
        files_exist, missing_files = check_files_exist(fnames_list)

        if not files_exist:
            if skip_test:
                logging.warning(f"Following files are missing: {', '.join(missing_files)}")
                pytest.skip(f"Skipping test because the following files are missing: {', '.join(missing_files)}")
            else:
                raise FileNotFoundError(f"Following files are missing: {', '.join(missing_files)}")

        # Load files based on their extensions
        if isinstance(datas, dict):
            # Iterate over dictionary keys and values directly
            for key, fname in fnames.items():
                datas[key] = _load_file(fname, nifti_load_method)
        else:
            # For lists, just iterate over file names
            for fname in fnames_list:
                datas.append(_load_file(fname,nifti_load_method))

        return datas
    
    def _load_file(fname, nifti_load_method='nibabel'):
        """Helper function to load a single file based on its extension."""
        
        # Get file extension type
        base, ext = os.path.splitext(fname)
        
        # Repeat for compressed files
        if ext == '.gz':
            base, ext = os.path.splitext(base)

        # Load file using appropriate method
        if ext == '.npy':
            return np.load(fname)
        elif ext == '.stl':
            return trimesh.load(fname)
        elif ext == '.nii':
            if nifti_load_method == 'nibabel':
                return nibabel.load(fname)
            elif nifti_load_method == 'sitk':
                return sitk.ReadImage(fname)
            else:
                raise ValueError(f"Invalid nifti load method specified: {nifti_load_method}")
        elif ext == '.txt':
            with open(fname, 'r') as file:
                content = file.read()
                return content
        else:
            logging.warning(f"Unsupported file extension, {fname} not loaded")

    return _load_files

@pytest.fixture()
def check_os(computing_backend):
    sys_os = None
    sys_platform = platform.platform(aliased=True)
    
    if 'macOS' in sys_platform:
        sys_os = 'Mac'
    elif 'Windows' in sys_platform:
        sys_os = 'Windows'
    elif 'Linux' in sys_platform:
        sys_os = 'Linux'
    else:
        logging.warning("No idea what os you're using")

    if sys_os not in computing_backend['supported_os']:
        pytest.skip("Skipping test because the selected computing backend is not available on this system")

@pytest.fixture(scope="session")
def get_gpu_device():
    return gpu_device

@pytest.fixture(scope="session")
def get_config_dirs():
    config_dirs = {}
    config_dirs["test_data_dir"] = test_data_folder
    config_dirs["ref_dir_1"] = ref_output_dir
    config_dirs["ref_dir_2"] = ref_output_dir_2
    config_dirs["gen_output_dir"] = gen_output_dir
    return config_dirs

@pytest.fixture()
def get_rmse():
    return comparisons.rmse


@pytest.fixture()
def resample_grid():
    def _resample_grid(nifti,spatial_step):
        """Affine and shape of `nifti` resampled to isotropic `spatial_step` mm voxels."""
        zooms = np.asarray(nifti.header.get_zooms())[:3]
        new_zooms = np.full(3,spatial_step)
        logging.info(f"Original zooms: {zooms}")
        logging.info(f"New zooms: {new_zooms}")
        new_shape = tuple(int(nifti.shape[i]/(new_zooms[i]/zooms[i])) for i in range(3))
        new_affine = affines.rescale_affine(nifti.affine.copy(),nifti.shape[:3],new_zooms,new_shape)
        return new_affine, new_shape

    return _resample_grid

@pytest.fixture()
def cached_cpu_truth():
    def _cached_cpu_truth(fname,compute,affine=None,dtype=np.float32):
        """CPU reference output saved in `fname` (NIfTI or .npy), computed with compute() and
        saved the first time so later runs skip the slow CPU calculation. NIfTI data is saved
        as `dtype` (None keeps the computed dtype)."""
        if os.path.exists(fname):
            logging.info(f"Reloading CPU truth from {fname}")
            return np.load(fname) if fname.endswith('.npy') else nibabel.load(fname).get_fdata()

        logging.info('Generating CPU truth')
        data = compute()
        logging.info(f"Saving CPU truth to {fname} for future use")
        if fname.endswith('.npy'):
            np.save(fname,data)
        else:
            data_to_save = np.asarray(data) if dtype is None else np.asarray(data).astype(dtype)
            nibabel.save(nibabel.Nifti1Image(data_to_save,affine),fname)
        return data

    return _cached_cpu_truth

@pytest.fixture()
def get_resampled_input(load_files,resample_grid):
    def _get_resampled_input(input,new_zoom,output_fname):

        if input.ndim > 3:
            tmp_data = input.get_fdata()[:,:,:,0]
            tmp_affine = input.affine
            input = nifti1.Nifti1Image(tmp_data,tmp_affine)

        # Create output with the new dimensions and affine
        new_affine, new_shape = resample_grid(input,new_zoom)
        output_data = np.zeros(new_shape,dtype=np.uint8)
        output_nifti = nifti1.Nifti1Image(output_data,new_affine)
        logging.info(f"New Dimensions: {output_data.shape}")
        logging.info(f"New Size: {output_data.size}")

        if os.path.exists(output_fname):
            logging.info('Reloading resampled input')
            resampled_nifti = nibabel.load(output_fname)
            resampled_data = resampled_nifti.get_fdata()
        else:
            logging.info('Generating resampled input')
            resampled_nifti = processing.resample_from_to(input,output_nifti,mode='constant',order=0,cval=input.get_fdata().min()) # Truth method
            resampled_data = resampled_nifti.get_fdata()
            logging.info('Saving file for future use')
            nibabel.save(resampled_nifti,output_fname)

        # Check data is contiguous
        if not resampled_data.flags.contiguous:
            logging.info("Changing resampled input data to be a contiguous array")
            resampled_data = np.ascontiguousarray(resampled_data)

        return resampled_nifti, resampled_data
    
    return _get_resampled_input

@pytest.fixture()
def check_data():
    def isometric_check(nifti):
        logging.info('Running isometric check')
        zooms = nifti.header.get_zooms()
        logging.info(f"Zooms: {zooms}")
        diffs = np.abs(np.subtract.outer(zooms, zooms))
        isometric = np.all(diffs <= 1e-6)

        return isometric

    # Return the fixture object with the specified attribute
    return {'isometric': isometric_check}


@pytest.fixture()
def compare_data():
    """Comparison functions by name (see helpers/comparisons.py)."""
    return comparisons.COMPARISONS

@pytest.fixture()
def extract_nib_info():
    def _extract_nib_info(nifti_nib):
        zooms = np.asarray(nifti_nib.header.get_zooms())[:3]
        affine = nifti_nib.affine
        data = np.squeeze(nifti_nib.get_fdata())

        return zooms, affine, data
    
    return _extract_nib_info

@pytest.fixture()
def extract_sitk_info():
    def _extract_sitk_info(nifti_sitk):
        spacing = np.asarray(nifti_sitk.GetSpacing())
        direction = np.asarray(nifti_sitk.GetDirection())
        origin = np.asarray(nifti_sitk.GetOrigin())
        data = sitk.GetArrayFromImage(nifti_sitk)

        return spacing, direction, origin, data
    
    return _extract_sitk_info

@pytest.fixture()
def image_to_base64():
    return plots.image_to_base64

@pytest.fixture()
def get_mpl_plot():
    """Slider viewer of volume slices (see helpers/plots.py mpl_plot)."""
    return plots.mpl_plot

@pytest.fixture()
def gpu_vs_cpu_plot():
    """Slider viewer of the inputs, CPU and GPU outputs and |GPU - CPU| (see helpers/plots.py)."""
    return plots.gpu_vs_cpu_plot

@pytest.fixture
def get_pyvista_plot():
    """3D mesh and voxel plots by name, each returning camera orbit frames (see helpers/plots.py)."""
    return plots.MESH_PLOTS


@pytest.fixture()
def get_extra_scan_file():
    def _get_extra_scan_file(extra_scan_type,ds_folder_path):
        scan_file_path = ""

        if extra_scan_type != 'NONE':
            scan_file_path = ds_folder_path + os.sep + CT_TYPE_FILES[extra_scan_type]

            if not os.path.exists(scan_file_path):
                pytest.skip(f"{ds_folder_path} does not possess a {extra_scan_type} file")

        return scan_file_path
    
    return _get_extra_scan_file

@pytest.fixture()
def pipeline_case(request, trajectory_type, scan_type, trajectory, dataset, transducer, frequency, computing_backend):
    """PipelineCase for this test's parameters. Tests marked with a configuration marker
    (parameters.CONFIGURATION_MARKERS, e.g. intensive_babelbrain_params) also get that configuration's
    thermal and multipoint profiles, so they run it exactly."""
    import Tests.helpers.pipeline_helpers as rh
    profiles = {}
    for marker, config in CONFIGURATION_MARKERS.items():
        if request.node.get_closest_marker(marker):
            profiles = {'thermal_profile': config['thermal_profile'], 'multipoint_profile': config['multipoint_profile']}
            break
    return rh.PipelineCase(dataset=dataset['id'],
                           transducer=transducer['name'],
                           freq_khz=int(frequency/1000),
                           backend=computing_backend['type'],
                           target=trajectory,
                           trajectory_type=trajectory_type,
                           ct_type=scan_type,
                           **profiles)

@pytest.fixture()
def babelbrain_widget(qtbot, monkeypatch, pipeline_case, tmp_path):
    """Factory returning a BabelBrain widget set up for this test's parameters (pipeline_case), writing
    its outputs to tmp_path (or gen_output_folder when generate_outputs=True)."""
    import Tests.helpers.pipeline_helpers as rh

    def _babelbrain_widget(generate_outputs=False):
        case = pipeline_case
        output_folder = os.path.join(gen_output_dir if generate_outputs else str(tmp_path), case.case_id)
        rh.auto_answer_dialogs(monkeypatch)
        return rh.build_babelbrain(qtbot, case, output_folder)

    yield _babelbrain_widget

    if tmp_path.exists():
        shutil.rmtree(tmp_path) # Remove all files created in tmp folder

@pytest.fixture()
def pipeline(qtbot, monkeypatch, request, compare_data, image_to_base64, tmp_path):
    """Runs and compares BabelBrain pipelines (see pipeline_helpers.py).
    Screenshots and comparison tables go to this test's html report entry."""
    import Tests.helpers.pipeline_helpers as rh

    request.node.screenshots = []
    screenshots = request.node.screenshots

    class _Pipeline:
        work_dir = tmp_path

        @staticmethod
        def auto_answer_dialogs():
            rh.auto_answer_dialogs(monkeypatch)

        @staticmethod
        def ensure_outputs(suite, case, custom_tx_name=None):
            """Folder of saved outputs for `case`, running the pipeline only if needed."""
            return rh.ensure_case_outputs(qtbot, monkeypatch, suite, case, screenshots, image_to_base64,
                                          custom_tx_name=custom_tx_name)

        @staticmethod
        def run_widget(bb, last_step=3, export_maps=True):
            """Run steps 1..last_step (and exports) on an already set up BabelBrain widget."""
            try:
                rh.run_pipeline(qtbot, bb, last_step=last_step, screenshots=screenshots,
                                image_to_base64=image_to_base64, export_maps=export_maps)
            except rh.PipelineError as error:
                pytest.fail(str(error))

        @staticmethod
        def run(case, output_dir, remote_server_name=None):
            """Run `case` into output_dir, without saving it for reuse. Returns the BabelBrain widget."""
            rh.auto_answer_dialogs(monkeypatch)
            bb = rh.build_babelbrain(qtbot, case, output_dir, remote_server_name=remote_server_name)
            _Pipeline.run_widget(bb, last_step=case.last_step)
            return bb

        @staticmethod
        def assert_thermal_outputs_match(ref_dir, test_dir, tolerance, message="Outputs differ from reference"):
            match, problems = rh.compare_thermal_outputs(ref_dir, test_dir, tolerance, screenshots, compare_data)
            assert match, f"{message}:\n" + "\n".join(problems or ["see h5 tables in report"])

        @staticmethod
        def assert_all_outputs_match(ref_dir, test_dir, tolerance, message="Outputs differ from reference"):
            match, problems = rh.compare_all_outputs(ref_dir, test_dir, tolerance, screenshots, compare_data)
            assert match, f"{message}:\n" + "\n".join(problems)

    return _Pipeline

@pytest.fixture()
def set_up_file_manager(load_files,tmpdir,get_example_data,get_extra_scan_file):

    def existing_dataset(ds,extra_scan_type="NONE",HUT=300.0,pCT_range=(0.1,0.6)):
        T1_iso_path = ds['folder_path'] + f"T1W-isotropic.nii.gz"
        extra_scan_path = get_extra_scan_file(extra_scan_type,ds['folder_path'])
        prefix = ""

        # Instantiate FileManager class object
        file_manager = FileManager(ds['m2m_folder_path'],
                                   ds['simbNIBS_type'],
                                   ds['T1_path'],
                                   T1_iso_path,
                                   extra_scan_path,
                                   prefix,
                                   CT_TYPES[extra_scan_type],
                                   current_HUT=HUT,
                                   current_pCT_range=pCT_range)
        
        # Load T1 using nibabel and save to file manager
        file_manager.saved_objects['T1_nib'] = load_files([ds['T1_path']],nifti_load_method='nibabel')[0]

        # Load T1 using sitk and save to file manager
        file_manager.saved_objects['T1_sitk'] = load_files([ds['T1_path']],nifti_load_method='sitk')[0]

        return file_manager
    
    def blank(CT_type='NONE',HUT=300.0,pCT_range=(0.1,0.6)):

        # Instantiate blank FileManager class object
        file_manager = FileManager(simNIBS_dir="",
                                   simbNIBS_type="",
                                   T1_fname="",
                                   T1_iso_fname="",
                                   extra_scan_fname="",
                                   prefix="",
                                   current_CT_type=CT_TYPES[CT_type],
                                   current_HUT=HUT,
                                   current_pCT_range=pCT_range)
        
        # Set file paths
        input_1_path = str(tmpdir.join('input1.npy'))
        output_1_path = str(tmpdir.join('output1.nii.gz'))
        output_2_path = str(tmpdir.join('output2.nii.gz'))
        input_fnames = {'input1': input_1_path}
        output_fnames = {'output1': output_1_path,'output2': output_2_path}

        # Get random example data
        example_input_data = get_example_data['numpy']()

        # Save input files
        if not os.path.exists(input_1_path):
            np.save(input_1_path,example_input_data)

        return file_manager, input_fnames, output_fnames
    
    return {'existing_dataset':existing_dataset,'blank':blank}

@pytest.fixture()
def get_example_data():
    def numpy_data(dims = (4,4)):
        return np.random.random(dims)
    
    def nifti_nib_data(dims=(16,16,16)):
        # Small on purpose: the FileManager tests only hash and compare these files, and gzip-saving
        # a full-size volume of random (incompressible) data took ~1.5 s per file
        affine = np.random.rand(4,4)
        data = np.random.random(dims)
        nibabel_nifti = nibabel.nifti1.Nifti1Image(data,affine)
        nibabel_nifti.header.set_zooms(np.random.rand(3))
        
        return nibabel_nifti
    
    # Return the fixture object with the specified attribute
    return {'numpy': numpy_data,
            'nifti_nib':nifti_nib_data}
        
# ================================================================================================================================
# PYTEST HOOKS
# ================================================================================================================================
def pytest_generate_tests(metafunc):
    collection.default_parametrize(metafunc)


def pytest_collection_modifyitems(config, items):
    collection.add_markers(items)


@pytest.hookimpl(hookwrapper=True)
def pytest_runtest_makereport(item,call):
    outcome = yield
    report = outcome.get_result()
    
    if report.when == 'call':
        extras = getattr(report, 'extras', [])

        # Add saved screenshots and plots to html report
        if hasattr(item, 'screenshots'):
            extras.append(pytest_html.extras.html(report_entries_html(item.screenshots)))

        report.extras = extras


@pytest.hookimpl()
def pytest_terminal_summary(terminalreporter, exitstatus, config):
    """Hook to modify inline final report"""
    terminalreporter.write_line(f"Total tests run: {terminalreporter._numcollected}")
    terminalreporter.write_line(f"Total failures: {len(terminalreporter.stats.get('failed', []))}")
    terminalreporter.write_line(f"Total passes: {len(terminalreporter.stats.get('passed', []))}")

    if os.path.isfile(os.path.join('PyTest_Reports','report.html')):
    # Change report name to include time of completion
        report_name = f"report_{datetime.datetime.now().strftime('%Y-%m-%d_%H-%M-%S')}.html"
        os.rename(os.path.join('PyTest_Reports','report.html'), os.path.join('PyTest_Reports',report_name))
        print(f"Report saved as {report_name}")
