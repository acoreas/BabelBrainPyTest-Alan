#!/bin/zsh
CONDA_BASE=$(conda info --base)
source "$CONDA_BASE/etc/profile.d/conda.sh"
cd BabelBrain
# git checkout main
# conda activate BabelBrain #activating 1st reference 
# sed -i '' "/gen_output_folder/c\\
# gen_output_folder = /Users/spichardo/Documents/BabelBrainRegression/Generated_Outputs/v0.4.3
# " Tests/config.ini
# pytest -k "test_generate_valid_outputs and Metal" -m "basic_babelbrain_params" 
# conda deactivate
# git checkout main
# conda activate BabelBrain #activating 1st reference 
# sed -i '' "/gen_output_folder/c\\
# gen_output_folder = /Users/spichardo/Documents/BabelBrainRegression/Generated_Outputs/v0.4.3
# " Tests/config.ini
# pytest -k "test_generate_valid_outputs and Metal" -m "all_targets_babelbrain_params" 
# conda deactivate
# # #########################
# git checkout rc0.8.0
# conda activate BabelMLX
# # # setting output directory
# sed -i '' "/gen_output_folder/c\\
# gen_output_folder = /Users/spichardo/Documents/BabelBrainRegression/Generated_Outputs/v0.8.0
# " Tests/config.ini
# pytest -k "test_generate_valid_outputs and MLX and H317 and NONE" -m "basic_babelbrain_params" 
# conda deactivate
# git checkout rc0.8.0
# conda activate BabelMLX
# # # setting output directory
# sed -i '' "/gen_output_folder/c\\
# gen_output_folder = /Users/spichardo/Documents/BabelBrainRegression/Generated_Outputs/v0.8.0_Metal
# " Tests/config.ini
# pytest -k "test_generate_valid_outputs and Metal" -m "basic_babelbrain_params" 
# conda deactivate

# git checkout rc0.8.0
# conda activate BabelMLX
# # # setting output directory
# sed -i '' "/gen_output_folder/c\\
# gen_output_folder = /Users/spichardo/Documents/BabelBrainRegression/Generated_Outputs/v0.8.0_Metal
# " Tests/config.ini
# pytest -k "test_generate_valid_outputs and Metal" -m "all_targets_babelbrain_params"
# conda deactivate

# git checkout rc0.8.0
# conda activate BabelMLX  
# # # setting output directory
# sed -i '' "/gen_output_folder/c\\
# gen_output_folder = /Users/spichardo/Documents/BabelBrainRegression/Generated_Outputs/v0.8.0_NoAir
# " Tests/config.ini
# pytest -k "test_generate_valid_outputs and Metal" -m "all_targets_babelbrain_params"
# conda deactivate
# # ###########################

# git checkout NewConeSelection
# conda activate BabelMLX  
# # # setting output directory
# sed -i '' "/gen_output_folder/c\\
# gen_output_folder = /Users/spichardo/Documents/BabelBrainRegression/Generated_Outputs/v0.8.0_ROIFix_and_Att
# " Tests/config.ini
# pytest -k "test_generate_valid_outputs and Metal" -m "all_targets_babelbrain_params"
# conda deactivate
# git checkout NewConeSelection
# conda activate BabelMLX  
# # # setting output directory
# sed -i '' "/gen_output_folder/c\\
# gen_output_folder = /Users/spichardo/Documents/BabelBrainRegression/Generated_Outputs/v0.8.0_NO_DISPERSION_CORRECT
# " Tests/config.ini
# pytest -k "test_generate_valid_outputs and Metal" -m "ct_targets_babelbrain_params"
# conda deactivate

git checkout rc0.8.0
conda activate BabelMLX  
# # setting output directory
sed -i '' "/gen_output_folder/c\\
gen_output_folder = /Users/spichardo/Documents/BabelBrainRegression/Generated_Outputs/v0.8.0_NoAir
" Tests/config.ini
pytest -k "test_generate_valid_outputs and Metal" -m "all_targets_babelbrain_params"
conda deactivate

# # ###########################
# #setting comparison directories
conda activate BabelMLX 
# sed -i '' "/ref_output_folder_1/c\\
# ref_output_folder_1 = /Users/spichardo/Documents/BabelBrainRegression/Generated_Outputs/v0.4.3
# " Tests/config.ini
# sed -i '' "/ref_output_folder_2/c\\
# ref_output_folder_2 = /Users/spichardo/Documents/BabelBrainRegression/Generated_Outputs/v0.8.0_Metal
# " Tests/config.ini
# # run comparison test
# pytest -k "test_full_pipeline_two_outputs"

# sed -i '' "/ref_output_folder_1/c\\
# ref_output_folder_1 = /Users/spichardo/Documents/BabelBrainRegression/Generated_Outputs/v0.4.3
# " Tests/config.ini
# sed -i '' "/ref_output_folder_2/c\\
# ref_output_folder_2 = /Users/spichardo/Documents/BabelBrainRegression/Generated_Outputs/v0.8.0
# " Tests/config.ini
# # run comparison test
# pytest -k "test_full_pipeline_two_outputs"

# sed -i '' "/ref_output_folder_1/c\\
# ref_output_folder_1 = /Users/spichardo/Documents/BabelBrainRegression/Generated_Outputs/v0.8.0_Metal
# " Tests/config.ini
# sed -i '' "/ref_output_folder_2/c\\
# ref_output_folder_2 = /Users/spichardo/Documents/BabelBrainRegression/Generated_Outputs/v0.8.0
# " Tests/config.ini
# # run comparison test
# pytest -k "test_full_pipeline_two_outputs"
# conda deactivate

# sed -i '' "/ref_output_folder_1/c\\
# ref_output_folder_1 = /Users/spichardo/Documents/BabelBrainRegression/Generated_Outputs/v0.8.0_Metal
# " Tests/config.ini
# sed -i '' "/ref_output_folder_2/c\\
# ref_output_folder_2 = /Users/spichardo/Documents/BabelBrainRegression/Generated_Outputs/v0.8.0_MetalMaximize
# " Tests/config.ini
# # run comparison test
# pytest -k "test_full_pipeline_two_outputs"
# conda deactivate

# sed -i '' "/ref_output_folder_1/c\\
# ref_output_folder_1 = /Users/spichardo/Documents/BabelBrainRegression/Generated_Outputs/v0.8.0_Metal
# " Tests/config.ini
# sed -i '' "/ref_output_folder_2/c\\
# ref_output_folder_2 = /Users/spichardo/Documents/BabelBrainRegression/Generated_Outputs/v0.8.0_MetalMaximizeV2
# " Tests/config.ini
# # run comparison test
# pytest -k "test_full_pipeline_two_outputs"

# sed -i '' "/ref_output_folder_1/c\\
# ref_output_folder_1 = /Users/spichardo/Documents/BabelBrainRegression/Generated_Outputs/v0.8.0_Metal
# " Tests/config.ini
# sed -i '' "/ref_output_folder_2/c\\
# ref_output_folder_2 = /Users/spichardo/Documents/BabelBrainRegression/Generated_Outputs/v0.8.0_NoAir
# " Tests/config.ini
# # run comparison test
# pytest -k "test_full_pipeline_two_outputs"

# sed -i '' "/ref_output_folder_1/c\\
# ref_output_folder_1 = /Users/spichardo/Documents/BabelBrainRegression/Generated_Outputs/v0.4.3
# " Tests/config.ini
# sed -i '' "/ref_output_folder_2/c\\
# ref_output_folder_2 = /Users/spichardo/Documents/BabelBrainRegression/Generated_Outputs/v0.8.0_NoAir
# " Tests/config.ini
# # run comparison test
# pytest -k "test_full_pipeline_two_outputs"

# sed -i '' "/ref_output_folder_1/c\\
# ref_output_folder_1 = /Users/spichardo/Documents/BabelBrainRegression/Generated_Outputs/v0.4.3
# " Tests/config.ini
# sed -i '' "/ref_output_folder_2/c\\
# ref_output_folder_2 = /Users/spichardo/Documents/BabelBrainRegression/Generated_Outputs/v0.8.0_ROIFix_and_Att
# " Tests/config.ini
# # run comparison test
# pytest -k "test_full_pipeline_two_outputs"


sed -i '' "/ref_output_folder_1/c\\
ref_output_folder_1 = /Users/spichardo/Documents/BabelBrainRegression/Generated_Outputs/v0.4.3
" Tests/config.ini
sed -i '' "/ref_output_folder_2/c\\
ref_output_folder_2 = /Users/spichardo/Documents/BabelBrainRegression/Generated_Outputs/v0.8.0_NoAir
" Tests/config.ini
# run comparison test
pytest -k "test_full_pipeline_two_outputs"

conda deactivate
cd ..


