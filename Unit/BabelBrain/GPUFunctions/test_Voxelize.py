import logging
import os
import re

import numpy as np
import pytest

from BabelBrain.GPUFunctions.GPUVoxelize import Voxelize
from BabelBrain.BabelDatasetPreps import FixMesh
from Tests.conftest import synced_frames_viewer_html

@pytest.mark.step1
def test_Voxelize_vs_CPU(computing_backend,dataset,spatial_step,check_os,get_gpu_device,load_files,resample_grid,cached_cpu_truth,get_pyvista_plot,request):
        
     # Parameters
     input_folder = dataset['folder_path']
     input_files = {
          'T1W': dataset['T1_path'],
          'skin': dataset['m2m_folder_path'] + os.sep + 'skin.stl',
     }
     spatial_step_text = re.sub("\\.","_",str(spatial_step))
     output_fnames = {
          'Output_Truth': input_folder + f"T1W_cpu_voxelized_spatial_step_{spatial_step_text}.npy",
     }

     # Initialize GPU Backend
     gpu_device = get_gpu_device
     Voxelize.InitVoxelize(gpu_device,computing_backend['type'])

     # Load inputs
     input_data = load_files(input_files)

     # Fix mesh if needed (in memory only, the test data is left untouched)
     if input_data['skin'].body_count != 1:
          logging.info("Fixing mesh file")
          input_data['skin'] = FixMesh(input_data['skin'])

     # Resampled grid
     new_affine, _ = resample_grid(input_data['T1W'],spatial_step)

     # Run voxelization step
     logging.info('Running Voxelization via GPU')
     points_gpu = Voxelize.Voxelize(input_data['skin'],targetResolution=spatial_step,GPUBackend=computing_backend['type'])
     points_cpu = cached_cpu_truth(output_fnames['Output_Truth'],
                                   lambda: input_data['skin'].voxelized(spatial_step,max_iter=30).fill().points)

     # Save plot screenshot to be added to html report later
     save_voxel_plot = get_pyvista_plot['voxel_plot']
     save_mesh_plot = get_pyvista_plot['mesh_plot']
     request.node.screenshots = []
     # Orbit frames of every view, played together further down so the mesh and voxels can be
     # compared from every side
     orbit_panels = [("Skin mesh", save_mesh_plot([input_data['skin']], title="Skin Mesh", video=True)),
                     ("CPU voxelization (truth)", save_voxel_plot(input_data['skin'],points_cpu, title="Skin Mesh Voxelized - Truth", video=True)),
                     ("GPU voxelization (test)", save_voxel_plot(input_data['skin'],points_gpu, title="Skin Mesh Voxelized - Test", video=True))]
     
     # Convert voxels back to integer 3D indices
     inv_affine = np.linalg.inv(new_affine)
     ijk = np.round(points_gpu @ inv_affine[:3,:3].T + inv_affine[:3,3]).astype(np.int64)
     ijk_truth = np.round(points_cpu @ inv_affine[:3,:3].T + inv_affine[:3,3]).astype(np.int64)

     # Encode each voxel as one integer so matching is vectorized (sets of tuples are far too slow
     # for the tens of millions of voxels at high resolutions)
     lo = np.minimum(ijk.min(axis=0), ijk_truth.min(axis=0))
     dims = np.maximum(ijk.max(axis=0), ijk_truth.max(axis=0)) - lo + 1
     keys_gpu = np.unique(np.ravel_multi_index((ijk - lo).T, dims))
     keys_cpu = np.unique(np.ravel_multi_index((ijk_truth - lo).T, dims))
     logging.info(f"Number of GPU indexes: {keys_gpu.size}")
     logging.info(f"Number of CPU indexes: {keys_cpu.size}")
     match_count = np.intersect1d(keys_gpu, keys_cpu, assume_unique=True).size

     # Plot the voxels only one of GPU/CPU found, back in world coordinates
     def to_world(keys):
          ijk_keys = np.stack(np.unravel_index(keys, dims), axis=1) + lo
          return ijk_keys @ new_affine[:3,:3].T + new_affine[:3,3]
     only_gpu = np.setdiff1d(keys_gpu, keys_cpu, assume_unique=True)
     only_cpu = np.setdiff1d(keys_cpu, keys_gpu, assume_unique=True)
     orbit_panels.append(("GPU only (red) / CPU only (blue)",
                          get_pyvista_plot['difference_plot'](input_data['skin'],to_world(only_gpu),to_world(only_cpu),
                                                              title="Voxelization differences", video=True)))
     request.node.screenshots.append({'kind': 'html', 'html': synced_frames_viewer_html(orbit_panels, frame_label='View')})

     # Calculate dice coefficient
     dice_coeff = 2 * match_count / (keys_gpu.size + keys_cpu.size)
     logging.info(f"Dice coefficient: {dice_coeff}")

     assert dice_coeff > 0.95, f"Dice coefficient is less than 0.95 ({dice_coeff})"
