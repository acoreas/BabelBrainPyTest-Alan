
import base64
import configparser
import datetime
import logging
import os
import platform
import re
import shutil
import sys
from io import BytesIO
from pathlib import Path
from pprint import pprint

import h5py
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import nibabel
import numpy as np
import pytest
import pytest_html
import pyvista as pv
import SimpleITK as sitk
import trimesh
import yaml
import uuid
from BabelViscoFDTD.H5pySimple import ReadFromH5py
from nibabel import affines, nifti1, processing
from PIL import Image
from PySide6.QtCore import Qt
from skimage.metrics import (
    mean_squared_error,
    normalized_root_mse,
    structural_similarity,
)

sys.path.append('./BabelBrain/')
matplotlib.use('Agg')  # Use the 'Agg' backend, which is noninteractive
np.random.seed(42) # RNG is same every time
_IS_MAC = platform.system() == 'Darwin'

def resource_path():  # needed for bundling
    """Get absolute path to resource, works for dev and for PyInstaller"""
    if not _IS_MAC:
        return os.path.split(Path(__file__))[0]

    if getattr(sys, 'frozen', False) and hasattr(sys, '_MEIPASS'):
        bundle_dir = Path(sys._MEIPASS)
    else:
        bundle_dir = Path(__file__).parent

    return bundle_dir

from BabelBrain.BabelBrain import BabelBrain
from BabelBrain.SelFiles.SelFiles import SelFiles
from BabelBrain.FileManager import FileManager
import BabelBrain.BabelBrain as bb_module # Needed to get monkeypatch working

# ================================================================================================================================
# FOLDER/FILE PATHS
# ================================================================================================================================
config = configparser.ConfigParser()
config.read('Tests' + os.sep + 'config.ini')
gpu_device = config['GPU']['device_name']               # GPU device used for test
print('Using GPU device: ',gpu_device)
test_data_folder = config['Paths']['data_folder_path']  # Folder containing input test data
ref_output_dir = config['Paths']['ref_output_folder_1']   # Folder containing previously generated BabelBrain outputs. Used in regression tests
ref_output_dir_2 = config['Paths']['ref_output_folder_2']   # Folder containing previously generated BabelBrain outputs. Used in test_full_pipeline_two_outputs tests
gen_output_dir = config['Paths']['gen_output_folder']   # Folder to store newly generated BabelBrain outputs. Used for generate_outputs "test"
REPORTS_DIR = "PyTest_Reports"

# ================================================================================================================================
# PARAMETERS
# ================================================================================================================================
test_trajectory_type = {
    'brainsight': 0,
    'slicer': 1
}
valid_trajectories = [
    'Superficial_Target',
    'Deep_Target',
    'Skull_Target',
]
invalid_trajectories = [
    'Outside_Target'
]
SimNIBS_type = {
    'charm': 0
}
test_datasets = [
    {'id': 'SDR_0p31','folder_path': test_data_folder + os.sep + 'SDR_0p31' + os.sep},
    {'id': 'SDR_0p42','folder_path': test_data_folder + os.sep + 'SDR_0p42' + os.sep},
    {'id': 'SDR_0p55','folder_path': test_data_folder + os.sep + 'SDR_0p55' + os.sep},    
    {'id': 'SDR_0p67','folder_path': test_data_folder + os.sep + 'SDR_0p67' + os.sep},     
    {'id': 'SDR_0p79','folder_path': test_data_folder + os.sep + 'SDR_0p79' + os.sep},
    {'id': 'ID_0082' ,'folder_path': test_data_folder + os.sep + 'ID_0082'  + os.sep}
]
for ds in test_datasets:
    ds['m2m_folder_path'] = ds['folder_path'] + f"m2m_{ds['id']}" + os.sep
    ds['T1_path'] = ds['folder_path'] + "T1W.nii.gz"
    ds['T1_iso_path'] = ds['folder_path'] + "T1W-isotropic.nii.gz"

    if os.path.exists(ds['m2m_folder_path'] + 'charm_log.html'):
        ds['simbNIBS_type'] = 'charm'
    else:
        ds['simbNIBS_type'] = 'headreco'

CT_types = {
    'NONE': 0, # T1W Only
    'CT': 1,
    'ZTE': 2,
    'PETRA': 3
}
coregistration = {
    'no': 0,
    'yes': 1
}
thermal_profiles = {
    'thermal_profile_1': test_data_folder + os.sep + 'Profiles' + os.sep + 'Thermal_Profile_1.yaml',
    'thermal_profile_2': test_data_folder + os.sep + 'Profiles' + os.sep + 'Thermal_Profile_2.yaml',
    'thermal_profile_3': test_data_folder + os.sep + 'Profiles' + os.sep + 'Thermal_Profile_3.yaml'
}

#we build the Tx list using TRANSDUCER_LIST and Tx yaml files
transducer_list_yaml = os.path.join(resource_path(),'..','BabelBrain','SelFiles','transducer_list.yaml')
with open(transducer_list_yaml,'r') as ftxlist:
    TRANSDUCER_LIST = yaml.load(ftxlist,yaml.SafeLoader)

transducers=[]
for n,tx in enumerate(TRANSDUCER_LIST):
    name=tx['name']
    selYaml=os.path.join(resource_path(),'..','BabelBrain','babel_transducers',tx['transducer_type'],tx['module_name'],'default.yaml')
    assert(os.path.isfile(selYaml)), f"missing yaml for {name}: {selYaml}"
    with open(selYaml,'r') as ftx:
        TxConfig=yaml.load(ftx,yaml.SafeLoader)
    if name =='Single':
        freqs=[200000.0,300000.0,400000.0,500000.0,600000.0,700000.0,800000.0,900000.0,1000000.0]
    else:
        freqs=TxConfig['USFrequencies']
    entry={}
    entry['name']=name
    entry['dropdown_index']=n
    entry['diameter']=0
    entry['freqs']=freqs
    transducers.append(entry)

pprint(transducers)

computing_backends = [
    # {'type': 'CPU','supported_os': ['Mac','Windows','Linux']},
    {'type': 'OpenCL','supported_os': ['Windows','Linux']},
    {'type': 'CUDA',  'supported_os': ['Windows','Linux']},
    {'type': 'Metal', 'supported_os': ['Mac']},
    {'type': 'MLX',   'supported_os': ['Mac']} # Linux too?
]
spatial_step = {
    'Low_Res': 0.919,  # 200 kHz,   6 PPW
    'Med_Res': 0.306,  # 600 kHz,   6 PPW
    'High_Res': 0.184,  # 1000 kHz,  6 PPW
    'Stress_Res': 0.092,  # 1000 kHz, 12 PPW
}

# ================================================================================================================================
# PYTEST FIXTURES
# ================================================================================================================================
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
    def _get_rmse(output_points, truth_points):
        rmse = np.sqrt(np.mean((output_points - truth_points) ** 2))
        data_range = np.max(truth_points) - np.min(truth_points)
        norm_rmse = rmse / data_range

        return rmse, data_range, norm_rmse
        
    return _get_rmse

@pytest.fixture()
def get_resampled_input(load_files):
    def _get_resampled_input(input,new_zoom,output_fname):

        if input.ndim > 3:
            tmp_data = input.get_fdata()[:,:,:,0]
            tmp_affine = input.affine
            input = nifti1.Nifti1Image(tmp_data,tmp_affine)

        # Determine new output dimensions and affine
        zooms = np.asarray(input.header.get_zooms())
        new_zooms = np.full(3,new_zoom)
        logging.info(f"Original zooms: {zooms}")
        logging.info(f"New zooms: {new_zooms}")
        new_x_dim = int(input.shape[0]/(new_zooms[0]/zooms[0]))
        new_y_dim = int(input.shape[1]/(new_zooms[1]/zooms[1]))
        new_z_dim = int(input.shape[2]/(new_zooms[2]/zooms[2]))
        new_affine = affines.rescale_affine(input.affine.copy(),
                                                input.shape,
                                                new_zooms,
                                                (new_x_dim,new_y_dim,new_z_dim))

        # Create output
        output_data = np.zeros((new_x_dim,new_y_dim,new_z_dim),dtype=np.uint8)
        output_nifti = nifti1.Nifti1Image(output_data,new_affine)
        logging.info(f"New Dimensions: {output_data.shape}")
        logging.info(f"New Size: {output_data.size}")

        try:
            logging.info('Reloading resampled input')
            resampled_nifti = load_files([output_fname],skip_test=False)[0]
            resampled_data = resampled_nifti.get_fdata()
        except:
            logging.info("File doesn't exist")
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
def compare_data(get_rmse):

    def array_data(output_array,truth_array):
        logging.info('Calculating root mean square error')

        array_rmse = array_range = array_norm_rmse = None

        # Check array size
        if len(output_array) == len(truth_array):
            logging.info(f"Number of array points are equal: {len(output_array)}")
            array_length_same = True

            array_rmse, array_range, array_norm_rmse = get_rmse(output_array,truth_array)
            if array_norm_rmse > 0:
                logging.warning(f"Array had a root mean square error of {array_rmse}, range of {array_range}, and a normalized RMSE of {array_norm_rmse}")
        else:
            logging.error(f"# of array points in output ({len(output_array)}) vs truth ({len(truth_array)})")
            array_length_same = False
        
        return array_length_same, array_norm_rmse
    
    def bhattacharyya_coefficient(arr1,arr2,num_bins=None):

        # Check arrays are not empty
        if arr1.size == 0 or arr2.size == 0:
            pytest.fail("One or both arrays are empty")

        # Determine range of values. We extended the range slightly so bins are divided at 0.5 marks 
        # instead of 1.0 (e.g. -0.5, 0.5, 1.5,...) as array values are more likely to exist at integer 
        # values and helps prevent errors when values lie exactly at bin edge
        min_val = int(np.floor(min(arr1.min(),arr2.min()))) - 0.5
        max_val = int(np.ceil(max(arr1.max(),arr2.max()))) + 0.5
        logging.debug(f"Using {min_val} to {max_val} range for bhatt coeff calculation")
        
        
        # Determine number of bins if argument is not supplied
        if num_bins is None:
            num_bins = int(max_val - min_val)
        logging.debug(f"Using {num_bins} bins for bhatt coeff calculation")
        
        # Get and normalize histograms
        hist1,_ = np.histogram(arr1,bins=num_bins,range=(min_val,max_val))
        hist2,_ = np.histogram(arr2,bins=num_bins,range=(min_val,max_val))
        norm_hist1 = hist1 / np.sum(hist1)
        norm_hist2 = hist2 / np.sum(hist2)

        # Compute Bhattacharyya coefficient
        logging.info('Calculating Bhattacharyya Coefficient')
        bhatt_coefficent = np.sum(np.sqrt(norm_hist1 * norm_hist2))
        logging.info(f"Bhattacharyya coefficient : {bhatt_coefficent}")

        return bhatt_coefficent

    def dice_coefficient(output_array,truth_array,abs_tolerance=1e-6,rel_tolerance=0):
        logging.info('Calculating dice coefficient')

        if output_array.size != truth_array.size:
            pytest.fail(f"Array sizes don't match: {output_array.size} vs {truth_array.size}")

        if output_array.size == 0:
            pytest.fail("Arrays are empty")
        
        if output_array.dtype == bool:
            matches = output_array == truth_array
        else:
            matches = np.isclose(output_array,truth_array,atol=abs_tolerance,rtol=rel_tolerance)
        matches_count = len(matches[matches==True])

        dice_coeff = 2 * matches_count / (output_array.size + truth_array.size)
        logging.info(f"DICE Coefficient: {dice_coeff}")
        return dice_coeff
    
    def h5_data(h5_ref_path,h5_test_path,node_screenshots,tolerance=0,label=None):
        mismatches = []
        scalar_results = []
        array_results = []

        VIEWER_MAX_FRAMES = 24

        # Units for the fields saved in DataForSim-ThermalField-Duration*.h5 (see
        # ThermalModeling/CalculateTemperatureEffects.py SaveDict); unknown fields fall
        # back to no unit rather than guessing.
        FIELD_UNITS = {
            'p_map': 'Pa',
            'TempEndFUS': '°C',
            'FinalTemp': '°C',
            'TI': '°C',
            'TIC': '°C',
            'TIS': '°C',
            'DoseEndFUS': 'CEM43 min',
            'FinalDose': 'CEM43 min',
            'CEMBrain': 'CEM43 min',
            'CEMSkin': 'CEM43 min',
            'CEMSkull': 'CEM43 min',
            'Isppa': 'W/cm$^2$',
            'Ispta': 'W/cm$^2$',
            'MaxIsppa': 'W/cm$^2$',
            'MaxIspta': 'W/cm$^2$',
            'MaterialMap': 'tissue ID',
        }

        def field_unit(name):
            return FIELD_UNITS.get(name.split('/')[-1], '')

        def make_volume_slice_viewer(vol1,vol2,diff_vol,title,dlabel,unit='',diff_unit=None):
            if diff_unit is None:
                diff_unit = f'Diff{dlabel}'
            nz = vol1.shape[0]
            n_frames = min(nz, VIEWER_MAX_FRAMES)
            frame_indices = np.unique(np.linspace(0, nz - 1, n_frames).astype(int))

            vmin=min(np.nanmin(vol1),np.nanmin(vol2))
            vmax=max(np.nanmax(vol1),np.nanmax(vol2))
            finite_diff = diff_vol[np.isfinite(diff_vol)]
            dvmin = finite_diff.min() if finite_diff.size else 0
            dvmax = finite_diff.max() if finite_diff.size else 1

            # Degenerate/near-uniform ranges (e.g. a diff volume of near-identical floats)
            # make matplotlib's default formatter print full-precision floats, which are
            # long enough to get clipped off the saved image; cap tick labels to 3 sig figs.
            tick_formatter = mticker.FormatStrFormatter('%.3g')

            fig, axs = plt.subplots(1,3,figsize=(9,3),dpi=70)
            im0=axs[0].imshow(vol1[frame_indices[0]],vmin=vmin,vmax=vmax)
            axs[0].set_title('Reference')
            cb0=plt.colorbar(im0,ax=axs[0],fraction=0.046)
            cb0.ax.yaxis.set_major_formatter(tick_formatter)
            cb0.set_label(unit)
            im1=axs[1].imshow(vol2[frame_indices[0]],vmin=vmin,vmax=vmax)
            axs[1].set_title('Test')
            cb1=plt.colorbar(im1,ax=axs[1],fraction=0.046)
            cb1.ax.yaxis.set_major_formatter(tick_formatter)
            cb1.set_label(unit)
            im2=axs[2].imshow(diff_vol[frame_indices[0]],vmin=dvmin,vmax=dvmax)
            axs[2].set_title(f'Diff{dlabel}')
            cb2=plt.colorbar(im2,ax=axs[2],fraction=0.046)
            cb2.ax.yaxis.set_major_formatter(tick_formatter)
            cb2.set_label(diff_unit)
            for ax in axs:
                ax.set_xlabel('X (voxels)')
                ax.set_ylabel('Y (voxels)')
            suptitle=fig.suptitle(f'{title}\nSlice {frame_indices[0]+1}/{nz}')
            plt.tight_layout()

            frames_b64 = []
            for z in frame_indices:
                im0.set_data(vol1[z])
                im1.set_data(vol2[z])
                im2.set_data(diff_vol[z])
                suptitle.set_text(f'{title}\nSlice {z+1}/{nz}')
                buffer = BytesIO()
                fig.savefig(buffer,format='webp')
                buffer.seek(0)
                frames_b64.append(base64.b64encode(buffer.getvalue()).decode('utf-8'))
            plt.close(fig)

            wid = f"volviewer_{uuid.uuid4().hex[:8]}"
            frames_json = '[' + ','.join(f'"{b}"' for b in frames_b64) + ']'
            # pytest-html injects "extra" html via innerHTML at runtime, so <script>
            # tags here would never execute. Everything below is wired through inline
            # on*="" attributes instead, which the browser still activates on elements
            # created via innerHTML. Frame data rides in a data-* attribute (single-quoted,
            # so the embedded base64/JSON double quotes need no escaping) and is parsed
            # once, then cached on the container element to avoid re-parsing per drag tick.
            get_frames_js = (
                f"var c=document.getElementById('{wid}');"
                "if(!c._frames){c._frames=JSON.parse(c.dataset.frames);}"
                "var frames=c._frames;"
            )
            show_slice_js = (
                "var i=parseInt(this.value,10);"
                f"document.getElementById('{wid}_img').src='data:image/webp;base64,'+frames[i];"
                f"document.getElementById('{wid}_label').textContent='Slice '+(i+1)+'/'+frames.length;"
            )
            slider_oninput = get_frames_js + show_slice_js
            play_onclick = (
                get_frames_js +
                f"var slider=document.getElementById('{wid}_slider');"
                f"var img=document.getElementById('{wid}_img');"
                f"var label=document.getElementById('{wid}_label');"
                "if(c._timer){clearInterval(c._timer);c._timer=null;this.textContent='Play';return;}"
                "this.textContent='Pause';"
                "c._timer=setInterval(function(){"
                "var v=(parseInt(slider.value,10)+1)%frames.length;"
                "slider.value=v;"
                "img.src='data:image/webp;base64,'+frames[v];"
                "label.textContent='Slice '+(v+1)+'/'+frames.length;"
                "},150);"
            )
            # The <img>'s onload fires every time its src is swapped (i.e. every frame),
            # so it's guarded to only kick off autoplay once, the first time the widget's
            # initial frame finishes loading (which happens even though it was inserted
            # via innerHTML, same as the on*="" attributes above).
            autoplay_onload = (
                f"var c=document.getElementById('{wid}');"
                "if(!c._autoplayStarted){"
                "c._autoplayStarted=true;"
                f"document.getElementById('{wid}_play').click();"
                "}"
            )
            return f"""
<div id='{wid}' data-frames='{frames_json}' style='text-align:center;margin:4px 0 12px 0'>
  <img id='{wid}_img' src='data:image/webp;base64,{frames_b64[0]}' width='900' onload="{autoplay_onload}"><br>
  <input type='range' id='{wid}_slider' min='0' max='{len(frames_b64)-1}' value='0' style='width:900px'
    oninput="{slider_oninput}"><br>
  <span id='{wid}_label'>Slice 1/{len(frames_b64)}</span>
  <button type='button' id='{wid}_play' style='margin-left:8px' onclick="{play_onclick}">Play</button>
</div>
"""

        section_title = label if label else os.path.basename(os.path.dirname(h5_ref_path))
        node_screenshots.append({
            'kind': 'html',
            'html': (
                f"<h3>{section_title}</h3>"
                f"<div style='font-size:12px;color:#555'>"
                f"Reference file: {h5_ref_path}<br>"
                f"Test file: {h5_test_path}<br>"
                f"Tolerance (rtol): {tolerance}"
                f"</div>"
            )
        })

        def compare_items(name, obj1):
            logging.info(f"Comparing {name}")
            if name not in f2:
                logging.warning(f"{name} missing in test file")
                mismatches.append(name)
                return
            obj2 = f2[name]
            if isinstance(obj1, h5py.Dataset):
                data1, data2 = obj1[()], obj2[()]
                bCorrectShape=True
                if not np.isscalar(data1):
                    if np.any(np.array(data1.shape)!=np.array(data2.shape)):
                        bCorrectShape=False
                        logging.warning(f"Dataset {name} differs in data shapes with {data1.shape} and {data2.shape}")

                tolerancepass=False
                if bCorrectShape:
                    if type(data1) == bytes or type(data2)==bytes:
                        def _as_float(v):
                            try:
                                return float(v.decode() if isinstance(v, bytes) else v)
                            except (TypeError, ValueError):
                                return None
                        num1, num2 = _as_float(data1), _as_float(data2)
                        if num1 is not None and num2 is not None:
                            tolerancepass = np.isclose(num1, num2, rtol=tolerance, atol=0, equal_nan=True)
                        else:
                            tolerancepass = data1==data2
                    else:
                        if np.isdtype(data1.dtype, "unsigned integer"):
                            data1 = data1.astype(np.float32)
                            data2 = data2.astype(np.float32)
                        tolerancepass=np.allclose(data1, data2, rtol=tolerance, atol=0, equal_nan=True)

                # Stats are always collected (for every variable, matched or not) so the
                # report can show a complete side-by-side table, not just the mismatches.
                is_bytes = isinstance(data1, bytes) or isinstance(data2, bytes)
                if not is_bytes and data1.size > 1:
                    if len(data1.shape)==3: #we save an animated scan through the volume of the error
                        if bCorrectShape:
                            unit = field_unit(name)
                            if np.issubdtype(data1.dtype, np.integer):
                                diff3d=np.abs(data2.astype(np.float64)-data1.astype(np.float64))
                                dlabel=''
                                diff_unit = f'{unit} (abs diff)' if unit else 'abs diff'
                                mean_abs_diff=diff3d.mean()
                                array_results.append((name, str(data1.shape), str(data2.shape), diff3d.max(), mean_abs_diff, None, None, tolerancepass))
                                if not tolerancepass:
                                    logging.warning(f"Dataset {name} differs with maximal diff of {diff3d.max()} (int)")
                            else:
                                nmrse=normalized_root_mse(data1,data2,normalization='min-max')
                                diff3d=np.abs(data2-data1)
                                diffMax=diff3d.max()
                                mean_abs_diff=diff3d.mean()
                                diff3d[data1==0.0]=0
                                diff3d[data1!=0.0]/=np.abs(data1[data1!=0])
                                max_rel_diff_pct=diff3d.max()*100
                                dlabel=' %'
                                array_results.append((name, str(data1.shape), str(data2.shape), diffMax, mean_abs_diff, max_rel_diff_pct, nmrse, tolerancepass))
                                if not tolerancepass:
                                    logging.warning(f"Dataset {name} differs with maximal diff of {diffMax} ({max_rel_diff_pct} %) and NRMSE {nmrse}")
                                if (diff3d.max()-diff3d.min())>100:
                                    diff3d[diff3d!=0]=np.log10(diff3d[diff3d!=0])
                                    diff3d[diff3d==0]=np.nan
                                    diff_unit = 'log10(% diff)'
                                else:
                                    diff3d*=100
                                    diff_unit = '% diff'

                            if not tolerancepass:
                                try:
                                    fn=h5_ref_path.split(os.sep)[-2].split('CT-')[1].split('kHz')[0]
                                except IndexError:
                                    fn=h5_ref_path.split(os.sep)[-2]

                                viewer_html = make_volume_slice_viewer(data1,data2,diff3d,title=f'{fn}\n{name}  Tol={tolerance}',dlabel=dlabel,unit=unit,diff_unit=diff_unit)
                                node_screenshots.append({'kind': 'html','html': f"<div><strong>{name}</strong> &mdash; drag the slider or press Play to scan through the volume</div>{viewer_html}"})
                        else:
                            logging.warning(f"Dataset {name} differs in shape: {data1.shape} vs {data2.shape}")
                            array_results.append((name, str(data1.shape), str(data2.shape), None, None, None, None, False))
                        if name =='TempEndFUS' and not tolerancepass:
                            Location =refH5Simple1['TargetLocation']
                            MTT1=data1[Location[0],Location[1],Location[2]]
                            Location =refH5Simple2['TargetLocation']
                            MTT2=data2[Location[0],Location[1],Location[2]]
                            logging.warning(f'MTT: {MTT1} vs {MTT2}')
                        if name == 'p_map' and not tolerancepass:
                            Location =refH5Simple1['TargetLocation']
                            p1=data1[Location[0],Location[1],Location[2]]
                            Location =refH5Simple2['TargetLocation']
                            p2=data2[Location[0],Location[1],Location[2]]
                            logging.warning(f'Pressure at target: {p1} vs {p2}')

                    else:
                        if bCorrectShape:
                            diff_arr = np.abs(data2.astype(np.float64) - data1.astype(np.float64))
                            max_abs_diff = diff_arr.max()
                            mean_abs_diff = diff_arr.mean()
                            nonzero = data1 != 0
                            max_rel_diff_pct = float(np.abs(diff_arr[nonzero] / data1[nonzero]).max() * 100) if np.any(nonzero) else None
                            try:
                                nrmse = normalized_root_mse(data1, data2, normalization='min-max')
                            except Exception:
                                nrmse = None
                            if not tolerancepass:
                                logging.warning(
                                    f"Dataset {name} differs with maximal diff of {max_abs_diff:.6g}"
                                    + (f" ({max_rel_diff_pct:.4g}%)" if max_rel_diff_pct is not None else "")
                                    + (f" and NRMSE {nrmse:.6g}" if nrmse is not None else "")
                                )
                            array_results.append((name, str(data1.shape), str(data2.shape), max_abs_diff, mean_abs_diff, max_rel_diff_pct, nrmse, tolerancepass))
                        else:
                            logging.warning(f"Dataset {name} differs in shape: {data1.shape} vs {data2.shape}")
                            array_results.append((name, str(data1.shape), str(data2.shape), None, None, None, None, False))
                else:
                    abs_diff = None
                    rel_diff_pct = None
                    try:
                        abs_diff = abs(float(data2) - float(data1))
                        if float(data1) != 0:
                            rel_diff_pct = abs_diff / abs(float(data1)) * 100
                    except (TypeError, ValueError):
                        pass
                    if not tolerancepass:
                        logging.warning(f"Dataset {name} differs: {data1} vs {data2}"
                                         + (f" (abs diff={abs_diff:.6g}"
                                            + (f", rel diff={rel_diff_pct:.4g}%)" if rel_diff_pct is not None else ")")
                                            if abs_diff is not None else ""))
                    scalar_results.append((name, data1, data2, abs_diff, rel_diff_pct, tolerancepass))

                if not tolerancepass:
                    mismatches.append(name)
                else:
                    logging.info(f"{name} matches")
            elif isinstance(obj1, h5py.Group):
                pass  # groups are containers, children checked recursively

        refH5Simple1=ReadFromH5py(h5_ref_path)
        refH5Simple2=ReadFromH5py(h5_test_path)

        with h5py.File(h5_ref_path, "r") as f1, h5py.File(h5_test_path, "r") as f2:
            exact_match = f1.visititems(lambda name, obj: compare_items(name, obj1=obj))

        def _fmt(v, spec='.6g'):
            return format(v, spec) if v is not None else 'n/a'

        def _row_style(matched):
            color = '#c6f6c6' if matched else '#f6c6c6'
            return f"style='background-color:{color}'"

        if scalar_results:
            rows = ''.join(
                f"<tr {_row_style(matched)}>"
                f"<td>{name}</td><td>{ref_val}</td><td>{test_val}</td>"
                f"<td>{_fmt(abs_diff)}</td>"
                f"<td>{_fmt(rel_diff_pct, '.4g') + '%' if rel_diff_pct is not None else 'n/a'}</td>"
                "</tr>"
                for name, ref_val, test_val, abs_diff, rel_diff_pct, matched in scalar_results
            )
            table_html = (
                "<table border='1' style='border-collapse:collapse;font-size:12px'>"
                "<tr><th>Variable</th><th>Reference</th><th>Test</th><th>Abs diff</th><th>Rel diff</th></tr>"
                f"{rows}</table>"
            )
            node_screenshots.append({'kind': 'html', 'html': table_html})

        if array_results:
            rows = ''.join(
                f"<tr {_row_style(matched)}>"
                f"<td>{name}</td><td>{ref_shape}</td><td>{test_shape}</td>"
                f"<td>{_fmt(max_abs_diff)}</td>"
                f"<td>{_fmt(mean_abs_diff)}</td>"
                f"<td>{_fmt(max_rel_diff_pct, '.4g') + '%' if max_rel_diff_pct is not None else 'n/a'}</td>"
                f"<td>{_fmt(nrmse)}</td>"
                "</tr>"
                for name, ref_shape, test_shape, max_abs_diff, mean_abs_diff, max_rel_diff_pct, nrmse, matched in array_results
            )
            table_html = (
                "<table border='1' style='border-collapse:collapse;font-size:12px'>"
                "<tr><th>Variable</th><th>Reference shape</th><th>Test shape</th>"
                "<th>Max abs diff</th><th>Mean abs diff</th><th>Max rel diff</th><th>NRMSE</th></tr>"
                f"{rows}</table>"
            )
            node_screenshots.append({'kind': 'html', 'html': table_html})

        if len(mismatches) == 0:
            node_screenshots.append({'kind': 'html', 'html': "<div style='color:green'>All datasets match within tolerance</div>"})

        return len(mismatches) == 0
    
    def mse(output_array,truth_array):
        logging.info('Calculating mean square error')

        if output_array.size != truth_array.size:
            pytest.fail(f"Array sizes don't match: {output_array.size} vs {truth_array.size}")

        if output_array.size == 0:
            pytest.fail("Arrays are empty")
        
        mean_square_error = mean_squared_error(output_array, truth_array)
        return mean_square_error
    
    def ssim(output_array,truth_array,win_size=7,data_range=None):
        logging.info('Calculating structural similarity')

        if output_array.size != truth_array.size:
            pytest.fail(f"Array sizes don't match: {output_array.size} vs {truth_array.size}")

        if output_array.size == 0:
            pytest.fail("Arrays are empty")

        score = structural_similarity(output_array, truth_array, win_size=win_size,data_range=data_range)
        return score
    
    def stl_area(output_stl,truth_stl):
        logging.info('Calculating percent error of stl area')
        
        # Check STL area
        percent_error_area = abs((output_stl.area - truth_stl.area)/truth_stl.area)
        if percent_error_area > 0:
            logging.warning(f"STL area had a percent error of {percent_error_area*100}%")
        else:
            logging.info(f"STL area is identical ({output_stl.area})")
        
        return percent_error_area

    # Return the fixture object with the specified attribute
    return {'array_data': array_data,'bhatt_coeff': bhattacharyya_coefficient,'dice_coefficient': dice_coefficient,'h5_data': h5_data,'mse': mse,'ssim': ssim,'stl_area': stl_area}

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
    def _image_to_base64(image_path):
        # Ensure the file exists
        if not image_path.exists() or not image_path.is_file():
            raise FileNotFoundError(f"File {image_path} does not exist.")
        

        # Open the PNG image
        im = Image.open(image_path)
        # Convert and save as WEBP
        buffer = BytesIO()
        # Save the image in WebP format into the buffer
        im.save(buffer, format="WEBP", quality=90)
        buffer.seek(0)
                                
        # Encode the image data as base64 string
        base64_string = base64.b64encode(buffer.getvalue()).decode('utf-8')
        
        
        return base64_string
    
    return _image_to_base64

@pytest.fixture()
def get_mpl_plot():
    def _get_mpl_plot(datas,axes_num=1,titles=None, color_map='viridis'):

        data_num = len(datas)
        fig, axs = plt.subplots(axes_num, data_num, figsize = (data_num * 2.5, axes_num * 2.5))

        for axis in range(axes_num):
            for num in range(data_num):
                midpoint = datas[num].shape[axis]//2

                try:
                    if axis == 0:
                        axs[axis,num].imshow(np.rot90(datas[num][midpoint,:,:]), cmap=color_map)
                    elif axis == 1:
                        axs[axis,num].imshow(np.rot90(datas[num][:,midpoint,:]), cmap=color_map)
                    else:
                        axs[axis,num].imshow(datas[num][:,:,midpoint], cmap=color_map)

                    if titles is not None and axis == 0:
                        axs[axis,num].set_title(titles[num])
                except:
                    if axis == 0:
                        axs[num].imshow(np.rot90(datas[num][midpoint,:,:]), cmap=color_map)
                    elif axis == 1:
                        axs[num].imshow(np.rot90(datas[num][:,midpoint,:]), cmap=color_map)
                    else:
                        axs[num].imshow(datas[num][:,:,midpoint], cmap=color_map)

                    if titles is not None and axis == 0:
                        axs[num].set_title(titles[num],fontsize=14)

        # Adjust plots
        plt.subplots_adjust(left=0.1, right=0.9, bottom=0.1, top=0.9, wspace=0.5, hspace=0.5)
        plt.tight_layout()

        # Save the plot to a BytesIO object
        buffer = BytesIO()
        plt.savefig(buffer, format='webp')
        buffer.seek(0)
        
        # Encode the image data as base64 string
        base64_plot = base64.b64encode(buffer.getvalue()).decode('utf-8')
        
        return base64_plot
    
    return _get_mpl_plot

@pytest.fixture
def get_pyvista_plot():

    def intersection_plot(mesh1,mesh2,mesh3):
        # Create pyvista plot
        plotter = pv.Plotter(window_size=(400, 400),off_screen=True)
        plotter.background_color = 'white'
        plotter.add_mesh(mesh1, opacity=0.2)
        plotter.add_mesh(mesh2, opacity=0.2)
        plotter.add_mesh(mesh3, opacity=0.5, color='red')

        # Save the plot to a BytesIO object
        buffer = BytesIO()
        plotter.show(screenshot=buffer)
        
        # Encode the image data as base64 string
        base64_plot = base64.b64encode(buffer.getvalue()).decode('utf-8')
        
        return base64_plot
    
    def mesh_plot(meshes,title=''):

        # Create pyvista plot
        plotter = pv.Plotter(window_size=(500, 500),off_screen=True)
        plotter.background_color = 'white'
        for mesh in meshes:
            plotter.add_mesh(pv.wrap(mesh),opacity=0.5)
        plotter.add_title(title, font_size=12)
        
        # Save the plot to a BytesIO object
        buffer = BytesIO()
        plotter.show(screenshot=buffer)

        # Encode the image data as base64 string
        base64_plot = base64.b64encode(buffer.getvalue()).decode('utf-8')

        return base64_plot
    
    def voxel_plot(mesh,Points,title=''):
        # Create points mesh
        step = Points.shape[0]//1000000 # Plot 1000000 points
        points_mesh =  pv.PolyData(Points[::step,:])

        # Create pyvista plot
        plotter = pv.Plotter(window_size=(500, 500),off_screen=True)
        plotter.background_color = 'white'
        plotter.add_mesh(pv.wrap(mesh),opacity=0.5)
        plotter.add_mesh(points_mesh,color='blue',opacity=0.1)
        plotter.add_title(title,font_size=12)
        
        # Save the plot to a BytesIO object
        buffer = BytesIO()
        plotter.show(screenshot=buffer)

        # Encode the image data as base64 string
        base64_plot = base64.b64encode(buffer.getvalue()).decode('utf-8')

        return base64_plot
    
    # Return the fixture object with the specified attribute
    return {'intersection_plot': intersection_plot,'mesh_plot': mesh_plot,'voxel_plot': voxel_plot}

@pytest.fixture()
def get_freq():
    def _get_default_freq(tx):
        tx = tx['name']
        if tx == 'Single':
            freq = '400'
        elif tx in ['CTX_500','DPX_500','H246','R15148','IGT64_500']:
            freq = '500'
        elif tx in ['CTX_250','CTX_250_2ch']:
            freq = '250'
        elif tx == 'H317':
            freq = '250'
        elif tx in ['BSonix','I12378','R15646']:
            freq = '650'
        elif tx in ['ATAC']:
            freq = '1000'
        elif tx in ['REMOPD','R15287','DPXPC_300','R15473']:
            freq = '300'
        return freq
    
    def _get_low_freq(tx):
        return tx['freqs'][0]
        
    def _get_high_freq(tx):
        return tx['freqs'][-1]

    return {'default': _get_default_freq,'low': _get_low_freq,'high': _get_high_freq}

@pytest.fixture()
def get_extra_scan_file():
    def _get_extra_scan_file(extra_scan_type,ds_folder_path):
        scan_file_path = ""

        if extra_scan_type != 'NONE':
            scan_file_path = ds_folder_path + os.sep + extra_scan_type + '.nii.gz'

            if not os.path.exists(scan_file_path):
                pytest.skip(f"{ds_folder_path} does not possess a {extra_scan_type} file")

        return scan_file_path
    
    return _get_extra_scan_file

@pytest.fixture()
def selfiles_widget(qtbot):

    sf_widget = SelFiles()
    sf_widget.show()
    qtbot.addWidget(sf_widget) # qtbot will handle sf_widget teardown

    yield sf_widget
    
    sf_widget.close()
    sf_widget.deleteLater()

@pytest.fixture()
def babelbrain_widget(request,qtbot,
                      trajectory_type,
                      scan_type,
                      trajectory,
                      dataset,
                      transducer,
                      selfiles_widget,
                      frequency,
                      get_extra_scan_file,
                      computing_backend,
                      load_files,
                      tmp_path):
    created_widgets = []
    
    def _babelbrain_widget(generate_outputs=False):
        
        # Convert frequency to string
        freq = str(int(frequency/1000)
                   )
        # Folder paths
        input_folder = dataset['folder_path']
        simNIBS_folder = dataset['m2m_folder_path']
        trajectory_folder = input_folder + 'Trajectories' + os.sep
        if generate_outputs:
            os.makedirs(gen_output_dir,exist_ok = True)
            output_folder = gen_output_dir + f"{os.sep}{dataset['id']}_{trajectory_type}_CT-{scan_type}_{trajectory}_{transducer['name']}_Freq-{freq}kHz_{computing_backend['type']}{os.sep}"
        else:
            output_folder = str(tmp_path) + f"{os.sep}{dataset['id']}_{trajectory_type}_CT-{scan_type}_{trajectory}_{transducer['name']}_Freq-{freq}kHz_{computing_backend['type']}{os.sep}"

        os.makedirs(output_folder,exist_ok = True)
        # Filenames
        T1W_file = dataset['T1_path']
        if scan_type != 'NONE':
            CT_file = get_extra_scan_file(scan_type,input_folder)
        thermal_profile_file = thermal_profiles['thermal_profile_1']
        trajectory_file = trajectory_folder + f"{trajectory_type}_{dataset['id']}_{trajectory}.txt"
        load_files([trajectory_file]) # Use to ensure trajectory file exists otherwise skip

        # Set SelFiles Parameters
        selfiles_widget.ui.TrajectoryTypecomboBox.setCurrentIndex(test_trajectory_type[trajectory_type])
        selfiles_widget.ui.TrajectorylineEdit.setText(trajectory_file)
        selfiles_widget.ui.SimbNIBSTypecomboBox.setCurrentIndex(SimNIBS_type['charm'])
        selfiles_widget.ui.SimbNIBSlineEdit.setText(simNIBS_folder)
        selfiles_widget.ui.T1WlineEdit.setText(T1W_file)
        selfiles_widget.ui.CTTypecomboBox.setCurrentIndex(CT_types[scan_type])
        if scan_type != 'NONE':
            selfiles_widget.ui.CoregCTcomboBox.setCurrentIndex(coregistration['yes'])
            selfiles_widget.ui.CTlineEdit.setText(CT_file)
        selfiles_widget.ui.ThermalProfilelineEdit.setText(thermal_profile_file)
        selfiles_widget.ui.TransducerTypecomboBox.setCurrentIndex(transducer['dropdown_index'])
        cb_index = selfiles_widget.ui.ComputingEnginecomboBox.findText(computing_backend['type'],Qt.MatchContains)
        selfiles_widget.ui.ComputingEnginecomboBox.setCurrentIndex(cb_index)
        if selfiles_widget.ui.MultiPointTypecomboBox.isEnabled():
            selfiles_widget.ui.MultiPointTypecomboBox.setCurrentIndex(0) # Only single focus
            # selfiles_widget.ui.MultiPointlineEdit.setText() # Fill out when implementing test for multi-point sims
        selfiles_widget.ui.ContinuepushButton.click()

        # Create BabelBrain widget
        os.environ['BABEL_PYTEST']='1'
        bb_widget = BabelBrain(selfiles_widget,AltOutputFilesPath=str(output_folder))
        bb_widget.show()
        qtbot.addWidget(bb_widget) # qtbot will handle bb_widget teardown
        created_widgets.append(bb_widget)

        # Copy T1W file and additional scan over to output folder
        # Not needed?
        shutil.copy(bb_widget.Config['T1W'],os.path.join(output_folder,os.path.basename(bb_widget.Config['T1W'])))
        if scan_type != 'NONE':
            shutil.copy(CT_file,os.path.join(output_folder,os.path.basename(CT_file)))

        # Copy SimbNIBs input file over to output folder
        if bb_widget.Config['SimbNIBSType'] == 'charm':
            SimbNIBSInput = bb_widget.Config['simbnibs_path'] + 'final_tissues.nii.gz'
        else:
            SimbNIBSInput = bb_widget.Config['simbnibs_path'] + 'skin.nii.gz'
        os.makedirs(os.path.join(output_folder,os.path.basename(os.path.dirname(bb_widget.Config['simbnibs_path']))),exist_ok=True)
        shutil.copy(SimbNIBSInput,os.path.join(output_folder,re.search('m2m.*',SimbNIBSInput)[0]))

        # Copy Trajectory file over to output folder
        trajectory_new_file = os.path.join(output_folder,os.path.basename(bb_widget.Config['Mat4Trajectory']))
        shutil.copy(bb_widget.Config['Mat4Trajectory'],trajectory_new_file)
        # Affects trajectory naming in output files. BabelBrain normalises Config['ID'] itself
        # when it reads the trajectory, and the shape differs between versions: a plain string
        # up to v0.8.x, a list (one entry per trajectory/transducer) from the dual-Tx refactor on.
        # Preserve whatever shape the app produced - assigning a bare string to a list-based
        # version makes it iterate over the characters of the name and rerun Step 1 once per letter.
        new_ID = os.path.splitext(os.path.basename(bb_widget.Config['Mat4Trajectory']))[0]
        if isinstance(bb_widget.Config['ID'], str):
            bb_widget.Config['ID'] = new_ID
        else:
            n_traj = len(bb_widget.Config['ID'])
            bb_widget.Config['ID'] = [new_ID] if n_traj == 1 else [f"{new_ID}_{i}" for i in range(n_traj)]

        # Edit file paths so new data is saved in output folder
        bb_widget.Config['Mat4Trajectory'] = trajectory_new_file
        bb_widget.Config['T1WIso'] = os.path.join(output_folder,os.path.basename(bb_widget.Config['T1WIso']))
        bb_widget.Config['simbnibs_path'] = os.path.join(output_folder,os.path.split(os.path.split(bb_widget.Config['simbnibs_path'])[0])[1])
        if bb_widget.Config['bUseCT']:
            bb_widget.Config['CT_or_ZTE_input'] = os.path.join(output_folder,os.path.basename(bb_widget.Config['CT_or_ZTE_input']))

        # Set Sim Parameters
        freq_index = bb_widget.Widget.USMaskkHzDropDown.findText(freq)

        bb_widget.Widget.USMaskkHzDropDown.setCurrentIndex(freq_index)
        bb_widget.Widget.USPPWSpinBox.setProperty('UserData',6) # 6 PPW
        if scan_type != 'NONE':
            bb_widget.Widget.HUThresholdSpinBox.setValue(300)
        
        return bb_widget

    yield _babelbrain_widget

    for w in created_widgets:
        w.close()
        w.deleteLater()
    
    if tmp_path.exists():
        shutil.rmtree(tmp_path) # Remove all files created in tmp folder
    
    if 'BABEL_PYTEST' in os.environ:
        os.environ.pop('BABEL_PYTEST')

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
                                   CT_types[extra_scan_type],
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
                                   current_CT_type=CT_types[CT_type],
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
    
    def nifti_nib_data(dims=(256,256,128)):
        affine = np.random.rand(4,4)
        data = np.random.random(dims)
        nibabel_nifti = nibabel.nifti1.Nifti1Image(data,affine)
        nibabel_nifti.header.set_zooms(np.random.rand(3))
        
        return nibabel_nifti
    
    def nifti_sitk_data():
        data = np.random.rand(256,256,128)
        nibabel_sitk = sitk.GetImageFromArray(data)

        # Set the spacing, direction, and origin in the SimpleITK image
        nibabel_sitk.SetSpacing(np.random.rand(3))
        nibabel_sitk.SetDirection(np.random.rand(3,3))
        nibabel_sitk.SetOrigin(np.random.rand(3))
        
        return nibabel_sitk
    
    # Return the fixture object with the specified attribute
    return {'numpy': numpy_data,
            'nifti_nib':nifti_nib_data,
            'nifti_sitk':nifti_sitk_data}
        
# ================================================================================================================================
# PYTEST HOOKS
# ================================================================================================================================
def pytest_generate_tests(metafunc):
    # Parametrize + mark tests based on fixtures used
    if 'trajectory_type' in metafunc.fixturenames:
        metafunc.parametrize('trajectory_type', tuple(test_trajectory_type)) 

    if 'scan_type' in metafunc.fixturenames:
        metafunc.parametrize('scan_type',tuple(CT_types.keys()))

    if 'second_scan_type' in metafunc.fixturenames:
        metafunc.parametrize('second_scan_type',tuple(CT_types.keys()))

    if 'trajectory' in metafunc.fixturenames and 'invalid' in metafunc.function.__name__:
        metafunc.parametrize('trajectory', tuple(invalid_trajectories))
    elif 'trajectory' in metafunc.fixturenames and ('valid' in metafunc.function.__name__ or
                                                    'normal' in metafunc.function.__name__):
        metafunc.parametrize('trajectory', tuple(valid_trajectories)) 
    
    if 'dataset' in metafunc.fixturenames:
        metafunc.parametrize('dataset',tuple(test_datasets),ids=tuple(ds['id'] for ds in test_datasets))
    
    if 'transducer' in metafunc.fixturenames:
        if 'frequency' in metafunc.fixturenames:
            # Parametrize both transducer and freq
            params = []
            for tx in transducers:
                for freq in tx['freqs']:
                    params.append(pytest.param(tx, freq, id=f"{tx['name']}-{int(freq/1000)}kHz"))
            metafunc.parametrize("transducer,frequency", params)
        else:
            # Only parametrize transducer
            metafunc.parametrize(
                "transducer",
                [pytest.param(tx, id=tx['name']) for tx in transducers]
            )
    
    if 'computing_backend' in metafunc.fixturenames:
        params = [pytest.param(cb, id=cb['type'], marks=pytest.mark.gpu) for cb in computing_backends]
        metafunc.parametrize("computing_backend", params)

    if 'spatial_step' in metafunc.fixturenames:
        # metafunc.parametrize('spatial_step',tuple(spatial_step.values()),ids=tuple(spatial_step.keys()))
        params = []
        for ss_key,ss_value in spatial_step.items():
            if "low" in ss_key.lower():
                params.append(pytest.param(ss_value, id=ss_key, marks=pytest.mark.low_res))
            elif "med" in ss_key.lower():
                params.append(pytest.param(ss_value, id=ss_key, marks=pytest.mark.medium_res))
            elif "high" in ss_key.lower():
                params.append(pytest.param(ss_value, id=ss_key, marks=[pytest.mark.slow,pytest.mark.high_res]))
            elif "stress" in ss_key.lower():
                params.append(pytest.param(ss_value, id=ss_key, marks=[pytest.mark.slow,pytest.mark.stress_res]))
            else:
                params.append(pytest.param(ss_value, id=ss_key))
        metafunc.parametrize('spatial_step',params)
        
    if 'tolerance' in metafunc.fixturenames:
        metafunc.parametrize('tolerance',
                             [pytest.param(0, marks=pytest.mark.tol_0, id="0%_tolerance"),
                              pytest.param(0.01, marks=pytest.mark.tol_1, id="1%_tolerance"),
                              pytest.param(0.05, marks=pytest.mark.tol_5, id="5%_tolerance")])

def pytest_collection_modifyitems(config, items):
    for item in items:
        # Add markers for basic babelbrain param tests
        if "Deep_Target" in item.name and \
            "ID_0082" in item.name and (\
            ("H317" in item.name and ("250kHz" in item.name or "825kHz" in item.name)) or \
            ("Single" in item.name and "500kHz" in item.name) or \
            ("CTX_500" in item.name and "500kHz" in item.name) or \
            "CTX_250" in item.name or \
            "DPX_500" in item.name or \
            "ATAC" in item.name or \
            "DPXPC_300" in item.name or \
            "H246" in item.name or \
            "BSonix" in item.name or \
            ("REMOPD" in item.name and "490kHz" in item.name) or \
            "I12378" in item.name or \
            "R15148" in item.name or \
            "R15287" in item.name or \
            "R15473" in item.name or \
            "R15646" in item.name or \
            "IGT64_500" in item.name) and \
            "PETRA" not in item.name and \
            "brainsight" in item.name and \
            ("NONE" in item.name or "CT" in item.name or "ZTE" in item.name):
            item.add_marker(pytest.mark.basic_babelbrain_params)

        if "ID_0082" in item.name and (\
            ("H317" in item.name and ("250kHz" in item.name )) or \
            ("Single" in item.name and "500kHz" in item.name) or \
            ("CTX_500" in item.name and "500kHz" in item.name) or \
            "DPX_500" in item.name or \
            "ATAC" in item.name or \
            "DPXPC_300" in item.name or \
            "H246" in item.name or \
            "BSonix" in item.name or \
            ("REMOPD" in item.name and "490kHz" in item.name) or \
            "I12378" in item.name or \
            "R15148" in item.name or \
            "R15287" in item.name or \
            "R15473" in item.name or \
            "R15646" in item.name or \
            "IGT64_500" in item.name) and \
            "PETRA" not in item.name and \
            "brainsight" in item.name:
            if ("NONE" in item.name or "CT" in item.name or "ZTE" in item.name):
                item.add_marker(pytest.mark.all_targets_babelbrain_params)
            if "NONE" in item.name or "CT" in item.name:
                item.add_marker(pytest.mark.ct_targets_babelbrain_params)
           
        if "Superficial_Target" in item.name and \
            "ID_0082" not  in item.name and \
            "Single" in item.name and \
            "slicer" in item.name and \
            ("CT" in item.name or "NONE" in item.name):
            item.add_marker(pytest.mark.orig_paper_params)

        if "Deep_Target" in item.name and \
            "ID_0082" in item.name and (\
            ("H317" in item.name and ("250kHz" in item.name )) or \
            ("Single" in item.name and "250kHz" in item.name) or \
            "CTX_250" in item.name or \
            "H246" in item.name or \
            "BSonix" in item.name or \
            ("REMOPD" in item.name and "300kHz" in item.name)) or \
            "PETRA" not in item.name and \
            "brainsight" in item.name and \
            ("NONE" in item.name):
            item.add_marker(pytest.mark.bare_min_babelbrain_params)

@pytest.hookimpl(hookwrapper=True)
def pytest_runtest_makereport(item,call):
    outcome = yield
    report = outcome.get_result()
    
    if report.when == 'call':
        extras = getattr(report, 'extras', [])

        # Add saved screenshots to html report
        if hasattr(item, 'screenshots'):
            html_parts = []
            row_tags = ''

            def flush_row():
                nonlocal row_tags
                if row_tags:
                    html_parts.append(f"<tr>{row_tags}</tr>")
                    row_tags = ''

            for entry in item.screenshots:
                if isinstance(entry, dict):
                    kind = entry.get('kind', 'image')
                    if kind == 'html':
                        flush_row()
                        html_parts.append(f"<tr><td colspan='10'>{entry['html']}</td></tr>")
                    else:
                        caption = entry.get('caption')
                        caption_html = f"<div style='text-align:center;font-size:12px'>{caption}</div>" if caption else ''
                        width = entry.get('width', 500)
                        mime = entry.get('mime', 'webp')
                        row_tags += f"<td><img src='data:image/{mime};base64,{entry['image']}' width='{width}'>{caption_html}</td>"
                else:
                    # Legacy entries: plain base64-encoded webp strings
                    row_tags += "<td><img src='data:image/webp;base64,{}' width='500'></td>".format(entry)
            flush_row()
            extras.append(pytest_html.extras.html(''.join(html_parts)))

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