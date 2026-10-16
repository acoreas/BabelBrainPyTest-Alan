import logging
import re

import pytest
from skimage.measure import label

from BabelBrain.GPUFunctions.GPULabel import LabelImage

@pytest.mark.step1
def test_LabelImage_vs_CPU(computing_backend,dataset,spatial_step,check_os,get_gpu_device,load_files,get_resampled_input,cached_cpu_truth,gpu_vs_cpu_plot,compare_data,request):
        
    # Parameters
    input_folder = dataset['folder_path']
    input_fnames = {'CT': input_folder + 'CT.nii.gz',}
    spatial_step_text = re.sub("\\.","_",str(spatial_step))
    output_fnames = {
        'Resampled_Input': input_folder + f"CT_cpu_resampled_spatial_step_{spatial_step_text}.nii.gz",
        'Output_Truth': input_folder + f"CT_cpu_label_spatial_step_{spatial_step_text}.nii.gz"
    }

    # Initialize GPU Backend
    gpu_device = get_gpu_device
    LabelImage.InitLabel(gpu_device,computing_backend['type'])

    # Load inputs
    input_data = load_files(input_fnames)

    # Get resampled input for specified spatial step
    resampled_nifti, resampled_data = get_resampled_input(input_data['CT'],spatial_step,output_fnames['Resampled_Input'])

    # Create bone mask
    HUCapThreshold = 2100.0
    HUThreshold = 300.0
    resampled_data[resampled_data>HUCapThreshold] = HUCapThreshold
    bone_mask = resampled_data > HUThreshold

    # Run label step
    logging.info('Running label step via GPU')
    data_label_gpu = LabelImage.LabelImage(bone_mask, GPUBackend=computing_backend['type'])
    data_label_cpu = cached_cpu_truth(output_fnames['Output_Truth'],lambda: label(bone_mask),affine=resampled_nifti.affine)

    # Save plot screenshot to be added to html report later
    request.node.screenshots = []
    plots = [resampled_data,bone_mask,data_label_cpu,data_label_gpu]
    plot_names = ['Resampled Data', 'Bone Mask', 'CPU Label\nOutput (Truth)', 'GPU Label\nOutput']
    screenshot = gpu_vs_cpu_plot(plots,plot_names,data_label_cpu,data_label_gpu)
    request.node.screenshots.append(screenshot)
    
    # Calculate dice coefficient
    calc_dice_func = compare_data['dice_coefficient']
    dice_coeff = calc_dice_func(data_label_gpu,data_label_cpu)

    assert dice_coeff > 0.99, f"Dice coefficient is less than 0.99 ({dice_coeff})"