"""
Comparisons of test outputs against reference outputs (arrays, histograms, masks, meshes and the h5
files BabelBrain saves). Available to tests through the compare_data fixture.
"""
import logging
import os

import h5py
import numpy as np
import pytest
from BabelViscoFDTD.H5pySimple import ReadFromH5py
from skimage.metrics import normalized_root_mse

from Tests.helpers.plots import volume_diff_viewer


def rmse(output_points, truth_points):
    """Root mean square error, the truth's data range and the RMSE normalized by that range."""
    error = np.sqrt(np.mean((output_points - truth_points) ** 2))
    data_range = np.max(truth_points) - np.min(truth_points)
    return error, data_range, error / data_range


def array_data(output_array,truth_array):
    logging.info('Calculating root mean square error')

    array_rmse = array_range = array_norm_rmse = None

    # Check array size
    if len(output_array) == len(truth_array):
        logging.info(f"Number of array points are equal: {len(output_array)}")
        array_length_same = True

        array_rmse, array_range, array_norm_rmse = rmse(output_array,truth_array)
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


def stl_area(output_stl,truth_stl):
    logging.info('Calculating percent error of stl area')

    # Check STL area
    percent_error_area = abs((output_stl.area - truth_stl.area)/truth_stl.area)
    if percent_error_area > 0:
        logging.warning(f"STL area had a percent error of {percent_error_area*100}%")
    else:
        logging.info(f"STL area is identical ({output_stl.area})")

    return percent_error_area


# ================================================================================================================================
# H5 OUTPUTS
# ================================================================================================================================
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


def _fmt(v, spec='.6g'):
    return format(v, spec) if v is not None else 'n/a'


def _row_style(matched):
    color = '#c6f6c6' if matched else '#f6c6c6'
    return f"style='background-color:{color}'"


def h5_data(h5_ref_path,h5_test_path,node_screenshots,tolerance=0,label=None):
    """True when every dataset in h5_test_path matches h5_ref_path within `tolerance` (rtol). Adds
    tables of all datasets, and a slice viewer of every mismatched volume, to node_screenshots."""
    mismatches = []
    scalar_results = []
    array_results = []

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

                            viewer_html = volume_diff_viewer(data1,data2,diff3d,title=f'{fn}\n{name}  Tol={tolerance}',dlabel=dlabel,unit=unit,diff_unit=diff_unit)
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


COMPARISONS = {'array_data': array_data,'bhatt_coeff': bhattacharyya_coefficient,'dice_coefficient': dice_coefficient,
               'h5_data': h5_data,'stl_area': stl_area}
