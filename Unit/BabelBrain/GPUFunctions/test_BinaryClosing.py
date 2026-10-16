import logging
import re

import numpy as np
import pytest
from scipy import ndimage

from BabelBrain.GPUFunctions.GPUBinaryClosing import BinaryClosing

def binary_closing_box(mask,size):
    """scipy binary_closing with a box structuring element of `size`, done as one 1D dilation and
    erosion per axis. The result is identical to ndimage.binary_closing(mask, np.ones(size)) but
    the cost grows with the box side instead of its volume, which keeps the CPU reference practical
    at high resolutions (e.g. a 54^3 box at Stress_Res)."""
    out = mask
    for operation in (ndimage.binary_dilation, ndimage.binary_erosion):
        for axis in range(3):
            line = [1, 1, 1]
            line[axis] = size[axis]
            out = operation(out, structure=np.ones(line, dtype=bool))
    return out

@pytest.mark.step1
def test_BinaryClosing_vs_CPU(computing_backend,dataset,spatial_step,check_os,get_gpu_device,load_files,get_resampled_input,cached_cpu_truth,gpu_vs_cpu_plot,compare_data,request):
        
    # Parameters
    input_folder = dataset['folder_path']
    input_fnames = {'CT': input_folder + 'CT.nii.gz',}
    spatial_step_text = re.sub("\\.","_",str(spatial_step))
    output_fnames = {
        'Resampled_Input': input_folder + f"CT_cpu_resampled_spatial_step_{spatial_step_text}.nii.gz",
        'Output_Truth': input_folder + f"CT_cpu_binary_closing_spatial_step_{spatial_step_text}.nii.gz"
    }

    # Initialize GPU Backend
    gpu_device = get_gpu_device
    BinaryClosing.InitBinaryClosing(gpu_device,computing_backend['type'])

    # Load inputs
    input_data = load_files(input_fnames)

    # Get resampled input for specified spatial step
    resampled_nifti, resampled_data = get_resampled_input(input_data['CT'],spatial_step,output_fnames['Resampled_Input'])

    # Create bone mask
    HUCapThreshold = 2100.0
    HUThreshold = 300.0
    resampled_data[resampled_data>HUCapThreshold] = HUCapThreshold
    bone_mask = resampled_data > HUThreshold

    # Determine binary closing filter size
    filter_size = np.round((np.ones(3)*5)/resampled_nifti.header.get_zooms()).astype(int)
    logging.info(f"Binary Closing Filter Size: {filter_size}")

    # Run binary closing step
    logging.info('Running binary closing step via GPU')
    data_binary_closing_gpu = BinaryClosing.BinaryClose(bone_mask, structure=np.ones(filter_size,dtype=int), GPUBackend=computing_backend['type'])
    data_binary_closing_cpu = cached_cpu_truth(output_fnames['Output_Truth'],
                                               lambda: binary_closing_box(bone_mask,filter_size),
                                               affine=resampled_nifti.affine)

    # Save plot screenshot to be added to html report later
    request.node.screenshots = []
    plots = [resampled_data,bone_mask,data_binary_closing_cpu,data_binary_closing_gpu]
    plot_names = ['Resampled Data', 'Bone Mask', 'CPU Binary Closing\nOutput (Truth)', 'GPU Binary\nClosing Output']
    screenshot = gpu_vs_cpu_plot(plots,plot_names,data_binary_closing_cpu,data_binary_closing_gpu)
    request.node.screenshots.append(screenshot)
    
    # Calculate dice coefficient
    calc_dice_func = compare_data['dice_coefficient']
    dice_coeff = calc_dice_func(data_binary_closing_gpu,data_binary_closing_cpu)

    assert dice_coeff > 0.99, f"Dice coefficient is less than 0.99 ({dice_coeff})"