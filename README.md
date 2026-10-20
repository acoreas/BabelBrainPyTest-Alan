# Pytest Setup and Usage Guide

This guide provides instructions for setting up and running pytest tests for BabelBrain, along with additional instructions for using it in Visual Studio Code.

# Installation
Before you begin, ensure you have Pytest installed in your environment.

``` bash
pip install pytest
```

The following packages are also required for BabelBrain pytests:
- pytest-html (Generating html test reports)
- pytest-metadata 
- pytest-qt (Handling QT applications)

```bash
pip install pytest-html, pytest-metadata, pytest-qt, pytest-xdist
```

These optional packages may also be installed:
- pytest-xdist (Parallelization of tests to speed up test run time, avoid using for GPU tests)
- pytest-benchmark (Evaluate code performance)
- pytest-profiling (Useful for determing code bottlenecks)
- pytest-sugar (Improve readability of pytest output)

```bash
pip install pytest-benchmark, pytest-profiling, pytest-sugar
```

# Writing Tests

Pytest allows you to write tests using Python's built-in `assert` statement. Test files/tests should be named with the prefix `test_` so that pytest can automatically discover them.

Here's an example of a few simple pytest test functions:

```python
# test_example.py

def test_addition():
    x = 1
    y = 2

    assert x + y == 3, "message shown if assertion fails"

@pytest.mark.custom_marker_1 # example of a marker
def test_subtraction():
    x = 1
    y = 2

    assert y - x == 1, "message shown if assertion fails"
```

Note that the folder structure of tests should parallel to that of the main directory

<pre>
BabelBrain/
├── pytest.ini
├── BabelBrain/
│     ├── BabelBrain.py
│     └── BabelDatasetPreps.py
├── TranscranialModeling/
│     ├── BabelIntegrationSingle.py
│     └── ...
├── Tests/
│     ├── config.ini
│     ├── conftest.py               (fixtures and hooks, delegating to helpers/)
│     ├── helpers/                  (code the tests import, never run directly)
│     │    ├── parameters.py        (datasets, targets, profiles, transducers, release test matrix)
│     │    ├── collection.py        (default parametrization and markers)
│     │    ├── pipeline_helpers.py  (set up, run and compare BabelBrain pipelines)
│     │    ├── comparisons.py       (array, mask, mesh and h5 output comparisons)
│     │    ├── plots.py             (report plots: volume slices, mesh views)
│     │    ├── report_viewers.py    (slider viewer and report entry html)
│     │    ├── mesh_checks.py       (point-in-mesh checks for intersections)
│     │    ├── blender_view.py      (Blender's own boolean result and .blend scenes)
│     │    └── synthetic_dataset.py (small synthetic head for the GPU unit tests)
│     ├── Consistency/              (checks parameters.py still matches BabelBrain and the test data)
│     ├── Profiles/                 (pytest thermal profiles)
│     ├── scripts/                  (run by hand around pytest, never imported by tests)
│     │    ├── isolate_pytest.py    (one pytest process per test)
│     │    ├── combine_html_reports.py
│     │    └── examples/            (example regression shell scripts)
│     ├── E2E/
│     ├── Generate_Outputs/
│     ├── Regression/
│     ├── Unit/                     (mirrors the BabelBrain repo: <source folder>/test_<module>.py)
│     │    ├── BabelBrain/
│     │    │    ├── test_BabelBrain.py
│     │    │    ├── test_BabelDatasetPreps.py
│     │    │    ├── test_FileManager.py
│     │    │    ├── GPUFunctions/
│     │    │    │    ├── GPUBinaryClosing/test_BinaryClosing.py
│     │    │    │    ├── GPUResample/test_Resample.py
│     │    │    │    └── ...
│     │    │    └── PlanTUSViewer/
│     │    │         └── test_RunPlanTUS.py
│     │    └── TranscranialModeling/
│     │         └── babel_integration/
│     │              └── test_babel_integration_helpers.py
│     └── ...
└── ...
</pre>

Unit tests mirror the BabelBrain repo: the tests for `BabelBrain/GPUFunctions/GPUResample/Resample.py` are in `Tests/Unit/BabelBrain/GPUFunctions/GPUResample/test_Resample.py`. Put a new unit test in the folder matching its module's path.

`helpers/` holds code that pytest imports through `conftest.py` and the test files, so the suite breaks without it; `scripts/` holds command line tools you run yourself before or after a pytest run (e.g. `python Tests/scripts/combine_html_reports.py ./reports`), and no test imports them. New code a test or fixture needs goes in `helpers/`; anything you run by hand goes in `scripts/`.

# Types of Tests
**Unit Test:**  
As the name implies, these tests are for specific functions or "units" of BabelBrain. Simplest test to ensure individual components work properly.

**Integration Test:**  
Tests to ensure multiple components work together. None are implemented yet; they would go in a `Tests/Integration/` folder.

**E2E (End-to-End) Test:**  
Tests to ensure the entire pipeline works as expected.

**Regression Test:**  
Tests to ensure consistency to BabelBrain outputs following major changes. Requires previously generated outputs to use as truth data.

**Generate_Outputs Test:**  
A "Test" to automatically batch run and save BabelBrain outputs to be used in future regression tests.

# Running Tests
## In terminal

In the highest level directory of BabelBrain, simply execute the `pytest` command in your terminal. This command will discover and run **ALL** tests. Note that pytest will automatically be configured based on parameters specified in the `pytest.ini` file. 

A timestamped report of all tests ran will automatically be saved to the `Pytest_Reports` folder while the individual tests ran will also be saved to `Pytest_Reports/individual_tests/`

Specific tests can be ran by searching for test files/names using the -k commandline argument and/or searching for specific test markers uing the -m option. Some example searches include:

```bash
# Retrieves all test inside test_example.py (i.e. test_addition and test_subtraction)
pytest -k "test_example.py" 

# Retrieves test containing addition OR subtraction in the test filepath. Achieves same result as above command
pytest -k "addition or subtraction"

# Retrieves test containing BOTH addition and subtraction in the test filepath. Returns no tests in this case.
pytest -k "addition and subtraction"

# Retrieves any test marked with custom_marker_1 (i.e. test_subtraction)
pytest -m "custom_marker_1"
```

Note that a full list of available markers can be found in the [pytest.ini](pytest.ini) file.

If you want to see what tests are collected without running them, you can add the `--collect-only` option

```bash
pytest -m "custom_marker_1" --collect-only
```

More information on other types of pytest arguments can be found [here](https://docs.pytest.org/en/6.2.x/usage.html).

## In Visual Studio Code

Open your project in Visual Studio Code. Install Python and Test Explorer UI extensions for Visual Studio Code.

Open the Command Palette (Ctrl+Shift+P or Cmd+Shift+P on macOS) and select "Open Workspace Settings (JSON)". Then set up the following parameters:

```json
{
    "python.testing.unittestEnabled": false,
    "python.testing.pytestEnabled": true,
    "python.testing.pytestPath": "<path_to_your_pytest>"
}
```
Example: For a conda environment, the pytest path looks something like 

`<user>/miniconda3/envs/<unit_testing_env>/Lib/site-packages/pytest`

Visual Studio Code will now discover pytest tests and display them in the testing tab; You can run them using the UI. 

If they do not appear, open the Command Palette again and select "Python: Select Interpreter" and choose the one in the same environment as your pytest package.

Note that the commandline arguments are still specified in the [pytest.ini](pytest.ini) file and are run automatically.

Alternatively, tests can be run from the integrated terminal similar to previous section.


# Additional Setup for BabelBrain testing
Certain tests require a directory containing test data, GPU device name, etc. These parameters are specified in the `config.ini` which is an untracked file as it is user-specific. If it's your first time running tests, you should duplicate the [example_config.ini](example_config.ini) file, update variables to match your setup, then rename the file to `config.ini`. This file possesses roughly the following format:

```ini
[Paths]
data_folder_path  = <path to test data>
ref_output_folder_1 = <path to reference BabelBrain output data set 1>
ref_output_folder_2 = <path to reference BabelBrain output data set 2>
gen_output_folder = <path to folder to store outputs from generate_outputs tests>
simnibs_root_path = <path to SimNIBS install (optional)>

[GPU]
device_name = <GPU device name> 

[Remote]   # optional, see example_config.ini
host = <server address>
```

Where the

`data_folder_path` folder contains test data including image sets

`ref_output_folder_1` folder contains previously generated BabelBrain outputs to be use in regression testing (i.e. test current setup against this reference).

`ref_output_folder_2` folder contains another previously generated BabelBrain outputs. Used when you want to compare two already generated ouput folders.

`gen_output_folder` folder is where output data generated from the generate_outputs tests will be stored

`simnibs_root_path` is the SimNIBS install folder (the one containing `simnibs_env`). Only needed by the soft tissue segmentation regression cases and the PlanTUS test, which are skipped without it

The optional `[Remote]` section points the remote server E2E test at an existing BabelBrain job server; without it the test starts a local one

Note that your `data_folder_path` folder should have the following structure. We are still in the process of sharing our test data, therefore you will need to create your own to run the tests.

<pre>
BabelBrain_Test_Data/
├── ID_0082/
│     ├── m2m_ID_0082/            (charm output, incl. final_tissues.nii.gz and the .msh mesh)
│     ├── T1W.nii.gz
│     ├── CT.nii.gz
│     ├── ZTE.nii.gz
│     └── ...                     (PETRA.nii.gz / Density.nii.gz if available)
├── ernie/
│     └── m2m_ernie/              (charm output, T1 is m2m_ernie/T1.nii.gz)
└── Trajectories/
      ├── &lt;type&gt;_&lt;dataset&gt;_superficial_target_left.txt
      ├── &lt;type&gt;_&lt;dataset&gt;_deep_target_left.txt
      ├── &lt;type&gt;_&lt;dataset&gt;_superficial_target_right.txt
      ├── &lt;type&gt;_&lt;dataset&gt;_deep_target_right.txt
      ├── &lt;type&gt;_&lt;dataset&gt;_superficial_targets.txt   (combined, .yaml for slicer)
      └── &lt;type&gt;_&lt;dataset&gt;_deep_targets.txt         (combined, .yaml for slicer)
</pre>

where `<type>` is brainsight, slicer or localite (.xml). The GPU unit tests additionally use a small synthetic head that is generated automatically (see [synthetic_dataset.py](helpers/synthetic_dataset.py)), so it doesn't need to be in the data folder. The thermal profiles used by the tests are kept in this repo's [Profiles](Profiles) folder and the multipoint profile comes from BabelBrain's own `Profiles` folder.

Datasets, targets, profiles and the release test matrix are defined in [parameters.py](helpers/parameters.py). Transducers, their frequencies and the drop-down options are read from the BabelBrain checkout under test, so new ones are picked up automatically. Run `pytest Tests/Consistency` (no GPU needed, a few seconds) to check that the test parameters still match BabelBrain and that the test data is in place.

# Further Reading
For more information about pytest and its features, check out the [pytest documentation](https://docs.pytest.org/en/latest/).

For examples of specific BabelBrain pytests, see [BABELBRAIN_PYTEST](BABELBRAIN_PYTEST.md) file
