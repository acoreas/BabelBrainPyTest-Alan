import logging
import re

import numpy as np
import pytest

from BabelBrain.GPUFunctions.GPUMapping import MappingFilter

@pytest.mark.step1
def test_MappingFilter_vs_CPU(computing_backend,dataset,spatial_step,check_os,get_gpu_device,load_files,get_resampled_input,cached_cpu_truth,gpu_vs_cpu_plot,compare_data,request):
        
    # Parameters
    input_folder = dataset['folder_path']
    input_files = {
        'T1W': dataset['T1_path'],
        'Final_Tissues': dataset['m2m_folder_path'] + 'final_tissues.nii.gz'
    }
    spatial_step_text = re.sub("\\.","_",str(spatial_step))
    output_fnames = {
        'Resampled_T1W': input_folder + f"T1W_cpu_resampled_spatial_step_{spatial_step_text}.nii.gz",
        'Resampled_Final_Tissues': input_folder + f"final_tissues_cpu_resampled_spatial_step_{spatial_step_text}.nii.gz",
        'Output_Truth': input_folder + f"CT_cpu_mapping_spatial_step_{spatial_step_text}.nii.gz"
    }

    # Initialize GPU Backend
    gpu_device = get_gpu_device
    MappingFilter.InitMapFilter(gpu_device,computing_backend['type'])

    # Load inputs
    input_data = load_files(input_files)

    # Get resampled inputs for specified spatial step
    resampled_nifti_1, resampled_data_1 = get_resampled_input(input_data['T1W'],spatial_step,output_fnames['Resampled_T1W'])
    _, resampled_data_2 = get_resampled_input(input_data['Final_Tissues'],spatial_step,output_fnames['Resampled_Final_Tissues'])

    resampled_data_1 = resampled_data_1.astype(np.float32)
    resampled_data_2 = resampled_data_2.astype(np.float32)

    # Create bone mask
    SelBone = (resampled_data_2==7)
    logging.info(f"SelBone shape: {SelBone.shape}")
    logging.info(f"Resampled T1W shape: {resampled_data_1.shape}")

    # Collect uniqye values
    UniqueHU = np.unique(resampled_data_1[SelBone])
    logging.info(f"UniqueHU shape: {UniqueHU.shape}")
    logging.info(f"# of UniqueHU values: {len(UniqueHU)}")

    # Run mapping step
    logging.info('Running map filter via GPU')
    data_map_gpu = MappingFilter.MapFilter(resampled_data_1,SelBone.astype(np.uint8),UniqueHU,GPUBackend=computing_backend['type'])
    def map_on_cpu():
        # Index of each bone voxel's value in UniqueHU (sorted), 0 outside bone. One sorted lookup
        # instead of a pass over the volume per unique value, which is very slow at high resolutions
        data_map_cpu=np.zeros(resampled_data_1.shape,np.uint32)
        data_map_cpu[SelBone]=np.searchsorted(UniqueHU,resampled_data_1[SelBone])
        return data_map_cpu
    data_map_cpu = cached_cpu_truth(output_fnames['Output_Truth'],map_on_cpu,affine=resampled_nifti_1.affine)

    # Save plot screenshot to be added to html report later
    request.node.screenshots = []
    plots = [resampled_data_1,SelBone,data_map_cpu,data_map_gpu]
    plot_names = ['Resampled Data', 'Bone Mask', 'CPU Map\nOutput (Truth)', 'GPU Map\nOutput']
    screenshot = gpu_vs_cpu_plot(plots,plot_names,data_map_cpu,data_map_gpu,color_map='viridis')
    request.node.screenshots.append(screenshot)
    
    # Calculate dice coefficient
    calc_dice_func = compare_data['dice_coefficient']
    dice_coeff = calc_dice_func(data_map_gpu,data_map_cpu)

    assert dice_coeff > 0.99, f"Dice coefficient is less than 0.99 ({dice_coeff})"