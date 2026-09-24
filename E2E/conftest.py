import glob
import logging
import os
from pathlib import Path
import re

import pytest

def _discover_output_subfolders(base_dir):
    """Immediate subfolder names under base_dir that contain a
    DataForSim-ThermalField-Duration h5 output. Empty if base_dir has the
    h5 files directly inside it (single-folder layout, nothing to split)."""
    pattern = "**/*DataForSim-ThermalField-Duration*.h5"
    h5_files = glob.glob(os.path.join(base_dir, pattern), recursive=True)
    subfolders = set()
    for f in h5_files:
        parts = os.path.relpath(f, base_dir).split(os.sep)
        if len(parts) > 1:
            subfolders.add(parts[0])
    return subfolders

def pytest_generate_tests(metafunc):
    # Turns test_full_pipeline_two_outputs (which used to compare every subfolder
    # of ref_dir_1 vs ref_dir_2 inside one pytest test, showing up as one row in
    # the html report) into one parametrized test per subfolder, so each folder's
    # pass/fail shows up as its own row.
    if "output_subfolder" in metafunc.fixturenames:
        from Tests.conftest import ref_output_dir
        subfolders = sorted(_discover_output_subfolders(ref_output_dir))
        metafunc.parametrize("output_subfolder", subfolders or [None])

@pytest.fixture
def compare_BabelBrain_Outputs(compare_data):
    def _compare_BabelBrain_Outputs(ref_folder,test_folder,tolerance,node_screenshots):
        
        # Find h5 files in folders
        pattern = "**/*DataForSim-ThermalField-Duration*.h5"

        h5_refs = glob.glob(os.path.join(ref_folder, pattern),recursive=True)
        h5_tests = glob.glob(os.path.join(test_folder, pattern),recursive=True)
        
        # Check for presence of files
        if len(h5_refs) == 0:
            pytest.skip(f"Files not found in {ref_folder}")
        if len(h5_tests) == 0:
            pytest.skip(f"Files not found in {test_folder}")
        
        def grab_folder_file_name(file, base_folder, no_gpu=False):
            # Path of the file relative to its base folder, so this works whether
            # base_folder directly contains the h5 files (single folder) or nests
            # them inside per-run subfolders (directory of subfolders)
            base_file_path = os.path.relpath(file, base_folder)
            if no_gpu:
                base_file_path = re.sub("(CUDA|OpenCL|Metal|MLX)","",base_file_path)

            return base_file_path

        # Build lookup for reference files
        ref_lookup = {grab_folder_file_name(h5, ref_folder): h5 for h5 in h5_refs}
        ref_lookup_no_gpu = {grab_folder_file_name(h5, ref_folder, no_gpu=True): h5 for h5 in h5_refs} # Used as backup in case exact file doesn't exist

        # Compare output against reference outputs
        compare_h5 = compare_data["h5_data"]
        matches = []
        missing_ref_files = []
        logging.info(f'TOLERANCE = {tolerance}')
        for h5_test in h5_tests:
            test_base = grab_folder_file_name(h5_test, test_folder)
            subfolder_label = os.path.dirname(test_base) or test_base
            h5_ref = ref_lookup.get(test_base)
            if h5_ref:
                logging.info('\n' + '*'*100 + f"\n\nCOMPARING\n{h5_test} FILE\nTO\n{h5_ref} FILE\n\n" + '*'*100)
                matches.append(compare_h5(h5_ref, h5_test, node_screenshots,tolerance=tolerance,label=subfolder_label))
            else:
                # See if there is another reference file that uses different gpu
                test_base_no_gpu = grab_folder_file_name(h5_test, test_folder, no_gpu=True)
                h5_ref = ref_lookup_no_gpu.get(test_base_no_gpu)
                if h5_ref:
                    logging.info('\n' + '*'*100 + f"\n\nCOMPARING\n{h5_test} FILE\nTO\n{h5_ref} FILE\n\n" + '*'*100)
                    matches.append(compare_h5(h5_ref, h5_test, node_screenshots,tolerance=tolerance,label=subfolder_label))
                else:
                    missing_ref_files.append(h5_test)
        
        if len(missing_ref_files) > 0:
            files = '\n'.join(str(f) for f in missing_ref_files)
            pytest.skip(f"The following files are missing equivalent files in {ref_folder}:\n{files}")
        
        # Check that all matches are True
        outputs_match = all(matches) 
            
        return outputs_match
    
    return _compare_BabelBrain_Outputs