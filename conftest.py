
import base64
import datetime
import logging
import os
import platform
import shutil
import sys
from io import BytesIO

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
from BabelViscoFDTD.H5pySimple import ReadFromH5py
from nibabel import affines, nifti1, processing
from PIL import Image
from skimage.metrics import normalized_root_mse

sys.path.append('./BabelBrain/')
matplotlib.use('Agg')  # Use the 'Agg' backend, which is noninteractive
np.random.seed(42) # RNG is same every time
from BabelBrain.FileManager import FileManager
import BabelBrain.BabelBrain as bb_module # Needed to get monkeypatch working

# ================================================================================================================================
# PARAMETERS (defined in parameters.py)
# ================================================================================================================================
from Tests.parameters import (COMPUTING_BACKENDS, CT_TYPE_FILES, CT_TYPES,
                              DATASETS, RELEASE_REAL_DATASET_SPATIAL_STEPS,
                              SPATIAL_STEPS, SYNTHETIC_DATASET,
                              SYNTHETIC_DATASET_DIR, TRAJECTORY_TYPES,
                              TESTED_INDIVIDUAL_TARGETS, TRANSDUCERS,
                              gen_output_dir, gpu_device,
                              marker_selects_transducer, ref_output_dir,
                              ref_output_dir_2, test_data_folder)
from Tests.report_viewers import slider_viewer_entry, slider_viewer_html

print('Using GPU device: ',gpu_device)

# ================================================================================================================================
# PYTEST FIXTURES
# ================================================================================================================================
@pytest.fixture(scope="session", autouse=True)
def synthetic_dataset():
    """Generate the synthetic GPU unit test dataset (instant, reused when unchanged)."""
    from Tests.synthetic_dataset import ensure_synthetic_dataset
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
    def _get_rmse(output_points, truth_points):
        rmse = np.sqrt(np.mean((output_points - truth_points) ** 2))
        data_range = np.max(truth_points) - np.min(truth_points)
        norm_rmse = rmse / data_range

        return rmse, data_range, norm_rmse
        
    return _get_rmse

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
            im2=axs[2].imshow(diff_vol[frame_indices[0]],vmin=dvmin,vmax=dvmax,cmap=plt.cm.jet)
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

            return slider_viewer_html([('', frames_b64)], frame_names=[f'Slice {z+1}/{nz}' for z in frame_indices],
                                      start=len(frame_indices)//2, width=630)

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
                                node_screenshots.append({'kind': 'html','html': f"<div><strong>{name}</strong> &mdash; drag the slider to scan through the volume</div>{viewer_html}"})
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
            f1.visititems(lambda name, obj: compare_items(name, obj1=obj))

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
    return {'array_data': array_data,'bhatt_coeff': bhattacharyya_coefficient,'dice_coefficient': dice_coefficient,'h5_data': h5_data,'stl_area': stl_area}

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
    def _get_mpl_plot(datas,axes_num=1,titles=None,color_map='viridis',colorbars=False,zero_based=None,n_positions=24):
        """Report entry with a slider over slice position: one row per axis (the first `axes_num`),
        one column per volume, each row showing the slice at the slider's position along that axis.
        color_map may be one map or one per volume; zero_based (one bool per volume) starts that
        volume's colour scale at 0 (e.g. absolute differences, so no difference is the lowest colour).
        Colour scales are fixed per volume so they don't change between slices."""
        data_num = len(datas)
        color_maps = color_map if isinstance(color_map,(list,tuple)) else [color_map]*data_num
        zero_based = zero_based or [False]*data_num
        datas = [np.asarray(d) for d in datas]
        limits = []
        for num, data in enumerate(datas):
            vmax = float(np.nanmax(data)) if data.size else 1.0
            vmin = 0.0 if zero_based[num] else (float(np.nanmin(data)) if data.size else 0.0)
            if vmax <= vmin:  # constant volume, e.g. no differences at all
                vmax = vmin + 1.0
            limits.append((vmin, vmax))

        def slice_of(data, axis, fraction):
            index = int(round(fraction*(data.shape[axis]-1)))
            image = np.take(data, index, axis=axis).astype(np.float64)
            return (np.rot90(image) if axis < 2 else image), index

        positions = np.linspace(0, 1, n_positions)
        fig, axs = plt.subplots(axes_num, data_num, figsize=(data_num * 2.5, axes_num * 2.5), squeeze=False, dpi=80)
        images, row_labels = [], []
        for axis in range(axes_num):
            row = []
            for num in range(data_num):
                image, _ = slice_of(datas[num], axis, positions[0])
                im = axs[axis,num].imshow(image, cmap=color_maps[num], vmin=limits[num][0], vmax=limits[num][1])
                if colorbars:
                    plt.colorbar(im, ax=axs[axis,num], fraction=0.046)
                if titles is not None and axis == 0:
                    axs[axis,num].set_title(titles[num], fontsize=10)
                row.append(im)
            images.append(row)
            # Widest label text up front so tight_layout leaves room for it
            row_labels.append(axs[axis,0].set_ylabel(f"axis {axis}, slice {datas[0].shape[axis]}/{datas[0].shape[axis]}"))
        plt.tight_layout()

        frames = []
        for fraction in positions:
            for axis in range(axes_num):
                for num in range(data_num):
                    image, index = slice_of(datas[num], axis, fraction)
                    images[axis][num].set_data(image)
                    if num == 0:
                        row_labels[axis].set_text(f"axis {axis}, slice {index+1}/{datas[0].shape[axis]}")
            buffer = BytesIO()
            fig.savefig(buffer, format='webp')
            frames.append(base64.b64encode(buffer.getvalue()).decode('utf-8'))
        plt.close(fig)

        return slider_viewer_entry([('', frames)], frame_label='Slice position', start=n_positions//2)

    return _get_mpl_plot

@pytest.fixture()
def gpu_vs_cpu_plot(get_mpl_plot):
    def _gpu_vs_cpu_plot(plots,titles,cpu_output,gpu_output,color_map='gray'):
        """`plots` (inputs, CPU and GPU outputs) plus a |GPU - CPU| panel, all sliced together. The
        difference panel's title gives the number of differing voxels in the whole volume."""
        diff = np.abs(np.asarray(gpu_output,dtype=np.float64) - np.asarray(cpu_output,dtype=np.float64))
        n_diff = int(np.count_nonzero(diff))
        return get_mpl_plot(list(plots) + [diff],
                            axes_num=3,
                            titles=list(titles) + [f"|GPU - CPU|\n{n_diff} voxels differ"],
                            color_map=[color_map]*len(plots) + [plt.cm.jet],
                            colorbars=True,
                            zero_based=[False]*len(plots) + [True])

    return _gpu_vs_cpu_plot

@pytest.fixture
def get_pyvista_plot():
    """3D plots of meshes and voxels for the html report. Every plot returns the frames of a camera
    orbit around its reference mesh; show them with report_viewers.slider_viewer_entry, where the
    slider moves the camera (several plots of the same mesh line up frame by frame)."""

    def _plotter(size=500):
        plotter = pv.Plotter(window_size=(size, size),off_screen=True)
        plotter.background_color = 'white'
        return plotter

    def _orbit_frames(plotter,reference_mesh,n_frames):
        """Frames (base64 WebP) of the camera making a full turn around the scene while sweeping from
        60 degrees above to 60 degrees below it, so every side can be reached with the slider. The camera is fitted to reference_mesh rather than to
        the scene, so plots sharing that mesh are framed identically and their frames line up."""
        plotter.show(auto_close=False)
        bounds = np.asarray(pv.wrap(reference_mesh).bounds).reshape(3, 2)
        focal = bounds.mean(axis=1)
        radius = np.linalg.norm(bounds[:, 1] - bounds[:, 0]) / 2
        distance = 1.1 * radius / np.tan(np.radians(plotter.camera.view_angle) / 2)
        frames = []
        for i in range(n_frames):
            azimuth = 2*np.pi*i/n_frames
            elevation = np.radians(60)*np.sin(azimuth)
            direction = np.array([np.cos(elevation)*np.cos(azimuth), np.cos(elevation)*np.sin(azimuth), np.sin(elevation)])
            plotter.camera.position = focal + distance*direction
            plotter.camera.focal_point = focal
            plotter.camera.up = (0, 0, 1)
            plotter.render()
            buffer = BytesIO()
            Image.fromarray(plotter.screenshot(return_img=True)).save(buffer,format='WEBP',quality=70)
            frames.append(base64.b64encode(buffer.getvalue()).decode('utf-8'))
        plotter.close()
        return frames

    def _render(plotter,reference_mesh,n_frames=36):
        return _orbit_frames(plotter,reference_mesh,n_frames)

    def _add_points(plotter,points,max_points,**kwargs):
        step = max(1, len(points)//max_points) # Plot at most ~max_points points
        plotter.add_mesh(pv.PolyData(np.asarray(points[::step],dtype=np.float64)),**kwargs)

    def intersection_plot(mesh1,mesh2,mesh3):
        plotter = _plotter(400)
        plotter.add_mesh(mesh1, opacity=0.2)
        plotter.add_mesh(mesh2, opacity=0.2)
        plotter.add_mesh(mesh3, opacity=0.5, color='red')
        return _render(plotter,mesh1)

    def mesh_plot(meshes,title=''):
        plotter = _plotter()
        for mesh in meshes:
            plotter.add_mesh(pv.wrap(mesh),opacity=0.5,show_scalar_bar=False)
        plotter.add_title(title, font_size=12)
        return _render(plotter,meshes[0])

    def voxel_plot(mesh,Points,title=''):
        plotter = _plotter()
        plotter.add_mesh(pv.wrap(mesh),opacity=0.5,show_scalar_bar=False)
        _add_points(plotter,Points,1000000,color='blue',opacity=0.1)
        plotter.add_title(title,font_size=12)
        return _render(plotter,mesh)

    def difference_plot(mesh,only_gpu_points,only_cpu_points,title=''):
        """Skin mesh with the voxels found by only one of the GPU (red) and CPU (blue) calculations."""
        plotter = _plotter()
        plotter.add_mesh(pv.wrap(mesh),opacity=0.2,color='lightgray',show_scalar_bar=False)
        for points, color, label in ((only_gpu_points,'red','GPU only'),(only_cpu_points,'blue','CPU only')):
            if len(points):
                _add_points(plotter,points,200000,color=color,point_size=4,label=f"{label} ({len(points)})")
        if len(only_gpu_points) or len(only_cpu_points):
            plotter.add_legend(loc='lower left',size=(0.4,0.1),bcolor='white')
        plotter.add_title(title,font_size=10)
        return _render(plotter,mesh)

    # Return the fixture object with the specified attribute
    return {'intersection_plot': intersection_plot,'mesh_plot': mesh_plot,'voxel_plot': voxel_plot,
            'difference_plot': difference_plot}


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
def babelbrain_widget(qtbot,
                      monkeypatch,
                      trajectory_type,
                      scan_type,
                      trajectory,
                      dataset,
                      transducer,
                      frequency,
                      computing_backend,
                      tmp_path):
    """Factory returning a BabelBrain widget set up for this test's parameters, writing its
    outputs to tmp_path (or gen_output_folder when generate_outputs=True)."""
    import Tests.pipeline_helpers as rh

    def _babelbrain_widget(generate_outputs=False):
        case = rh.PipelineCase(dataset=dataset['id'],
                               transducer=transducer['name'],
                               freq_khz=int(frequency/1000),
                               backend=computing_backend['type'],
                               target=trajectory,
                               trajectory_type=trajectory_type,
                               ct_type=scan_type)
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
    import Tests.pipeline_helpers as rh

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
    
    def nifti_nib_data(dims=(256,256,128)):
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


def pytest_generate_tests(metafunc):
    """Default parametrization for tests that request these fixtures. A test narrows any of them by
    parametrizing it explicitly, e.g. @pytest.mark.parametrize('trajectory', INVALID_TARGETS)."""
    def wants(name):
        return name in metafunc.fixturenames and not _explicitly_parametrized(metafunc, name)

    dataset_with_spatial_step = wants('dataset') and wants('spatial_step')
    if dataset_with_spatial_step:
        # GPU unit tests: every spatial step on the small synthetic head and the real datasets. Real
        # dataset resolutions that release testing skips (they take very long) are marked release_exclude
        params = [pytest.param(SYNTHETIC_DATASET, value, id=f"synthetic-{key}", marks=res_marks.get(key, []))
                  for key, value in SPATIAL_STEPS.items()]
        for ds_id, ds in DATASETS.items():
            for key, value in SPATIAL_STEPS.items():
                marks = list(res_marks.get(key, []))
                if key not in RELEASE_REAL_DATASET_SPATIAL_STEPS:
                    marks.append(pytest.mark.release_exclude)
                params.append(pytest.param(ds, value, id=f"{ds_id}-{key}", marks=marks))
        metafunc.parametrize('dataset,spatial_step', params)
    elif wants('dataset'):
        metafunc.parametrize('dataset', list(DATASETS.values()), ids=list(DATASETS))
    
    if wants('trajectory_type'):
        metafunc.parametrize('trajectory_type', list(TRAJECTORY_TYPES))

    if wants('trajectory'):
        metafunc.parametrize('trajectory', TESTED_INDIVIDUAL_TARGETS)
        
    for name in ('scan_type', 'second_scan_type'):
        if wants(name):
            metafunc.parametrize(name, [item + '_Extra_Scan' for item in list(CT_TYPES)])

    res_marks = {'Low_Res': [pytest.mark.low_res],
                 'Med_Res': [pytest.mark.medium_res],
                 'High_Res': [pytest.mark.slow, pytest.mark.high_res],
                 'Stress_Res': [pytest.mark.slow, pytest.mark.stress_res]}

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
        metafunc.parametrize('spatial_step', [pytest.param(value, id=key, marks=res_marks.get(key, []))
                                              for key, value in SPATIAL_STEPS.items()])

    if wants('tolerance'):
        metafunc.parametrize('tolerance',
                             [pytest.param(0, marks=pytest.mark.tol_0, id="0%_tolerance"),
                              pytest.param(0.01, marks=pytest.mark.tol_1, id="1%_tolerance"),
                              pytest.param(0.05, marks=pytest.mark.tol_5, id="5%_tolerance")])


def pytest_collection_modifyitems(config, items):
    """Add the *_babelbrain_params selection markers based on each test's parameter values
    (transducer/frequency choices are in parameters.MARKER_TRANSDUCERS)."""
    for item in items:
        params = getattr(getattr(item, 'callspec', None), 'params', {})
        if not {'transducer', 'frequency', 'dataset', 'trajectory', 'trajectory_type', 'scan_type'} <= params.keys():
            continue
        tx, freq = params['transducer']['name'], params['frequency']
        dataset, target = params['dataset']['id'], params['trajectory']
        traj_type, scan = params['trajectory_type'], params['scan_type']

        def selects(marker):
            return marker_selects_transducer(marker, tx, freq)

        if dataset == 'ID_0082' and traj_type == 'brainsight':
            if target == 'Deep_Target_Left' and scan in ('NONE', 'CT', 'ZTE') and selects('basic_babelbrain_params'):
                item.add_marker(pytest.mark.basic_babelbrain_params)
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