import os

def test_full_pipeline_normal(babelbrain_widget,pipeline,check_os,transducer):
    pipeline.run_widget(babelbrain_widget())

def test_full_pipeline_regression_normal(babelbrain_widget,pipeline,check_os,compare_BabelBrain_Outputs,request,tmp_path,tolerance,get_config_dirs):
    pipeline.run_widget(babelbrain_widget())

    # Compare output against reference outputs
    bb_outputs_match = compare_BabelBrain_Outputs(ref_folder = get_config_dirs['ref_dir_1'], test_folder = tmp_path, tolerance=tolerance, node_screenshots=request.node.screenshots)

    assert bb_outputs_match == True, "There were differences between the h5 files, see logged warnings for more details"

def test_full_pipeline_two_outputs(compare_BabelBrain_Outputs,tolerance,get_config_dirs,request,output_subfolder):
    # Save plot screenshot to be added to html report later
    request.node.screenshots = []
    config_dirs = get_config_dirs
    ref_output_dir = config_dirs['ref_dir_1']
    ref_output_dir_2 = config_dirs['ref_dir_2']

    # output_subfolder is None when ref_dir_1 has the h5 files directly in it
    # (single-folder layout); otherwise each subfolder is its own parametrized
    # test, so it shows up as its own row in the html report.
    if output_subfolder is not None:
        ref_output_dir = os.path.join(ref_output_dir, output_subfolder)
        ref_output_dir_2 = os.path.join(ref_output_dir_2, output_subfolder)

    bb_outputs_match = compare_BabelBrain_Outputs(ref_folder = ref_output_dir, test_folder = ref_output_dir_2,tolerance=tolerance,node_screenshots=request.node.screenshots)

    assert bb_outputs_match == True, "There were differences between the h5 files, see logged warnings for more details"