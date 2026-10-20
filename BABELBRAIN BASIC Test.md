# Examples of basic tests

Create in the `Tests` directory a `config.ini` file by copying [example_config.ini](example_config.ini) and updating its entries, e.g.

```
[Paths]
data_folder_path = /Users/spichardo/Documents/TempForSim/
ref_output_folder_1 = /Users/spichardo/Documents/BabelBrainRegression/Generated_Outputs/v0.8.1
ref_output_folder_2 = /Users/spichardo/Documents/BabelBrainRegression/Generated_Outputs/v0.8.8
gen_output_folder = /Users/spichardo/Documents/BabelBrainRegression/Generated_Outputs/new
[GPU]
device_name = M3
```

Then check the test parameters still match BabelBrain and the test data is in place (no GPU needed):

`pytest Tests/test_suite_consistency.py`

## Minimal test per release
The test below should the bare minimum at every release.\
It will test that no execution errors occurs running the full pipeline on the ernie dataset (MRI only, superficial target) with one transducer of each geometry type: the basic configuration (see `basic_babelbrain_params` in [BABELBRAIN_PYTEST.md](BABELBRAIN_PYTEST.md)), which the major release `regression_test_full_pipeline` also runs. Add `--collect-only` to see how many simulations it selects.

`pytest Tests -k 'test_full_pipeline_normal and Metal' -m basic_babelbrain_params`

(replace Metal with the backend of your machine)

if on macOS/Linux.

In Windows, run first a BAT generator for the test with

`python genPytestWin.py`

them run 

`runPytestWin.bat`

It will run one test each time as GPU code has tendency to crash in Windows if running too many in a row. It has been difficult to predict