# Running BabelBrain Pytests
## Description
This document covers the how to run the various types of BabelBrain tests. For details on setting up pytest for BabelBrain, see [README.md](README.md)

## Running a BabelBrain Unit Test
Simplest case, which for the most part can be run simply by calling the test name

**Example:**  
After making changes to generate_hash function in FileManager.py file, the following pytest command 
```bash
pytest -k "test_generate_hash"
```
will collect and run the following 3 tests to ensure proper operation of generate_hash following changes
- test_generate_hash
- test_generate_hash_file_not_found
- test_generate_hash_read_error

If multiple changes to FileManager.py were made instead, you could run

```bash
pytest -k "test_FileManager.py"
```
or
```bash
pytest Tests/Unit/BabelBrain/test_FileManager.py
```
which would run available tests for all the functions inside FileManager.py (assuming the tests exist)

**Important to note that some unit tests are parameterized** 

**Example**  
test_nibabel_to_sitk relies on T1 data and thus is parameterized for image datasets (e.g. SDR_0p31, SDR_0p42, etc.) Runnning

```bash
pytest -k "test_nibabel_to_sitk"
```
will collect/run for all datasets so tests ran would be:
- test_nibabel_to_sitk[SDR_0p31]
- test_nibabel_to_sitk[SDR_0p42]
- etc.

Therefore to run a specific dataset, simply specify the dataset like so
```bash
pytest -k "test_nibabel_to_sitk[SDR_0p31]"
```
or
```bash
pytest -k "test_nibabel_to_sitk and SDR_0p31"
```

In general, running all the different parameters will not take too long to run. The exception being pytests for GPU functions.

### Running GPU unit tests
GPU unit tests are parameterized for spatial step
- low_res
- medium_res
- high_res
- stress_res

where high_res and stress_res are **much slower** to run due to higher resolution sims. Note that tests for GPU function are marked with the "gpu" marker. That is, to collect/run all GPU unit tests, run the following command:

```bash
pytest -k "unit" -m "gpu"
```
To avoid slow gpu tests, run
```bash
pytest -k "unit" -m "gpu and (low_res or med_res)"
```
or
```bash
pytest -k "unit" -m "gpu and not slow"
```
where slow is another marker.

### Running step specific unit tests
Unit tests specific to BabelBrain steps can be run as follows

**Step 1 Unit Tests**
```bash
pytest -k "unit" -m "step1"
```
This will collect/run tests for **ANY** function called during step 1 (Assuming the appropriate test exists).
Replace the `step1` marker with `step2` and `step3` for step 2 and step 3 unit tests respectively

## Running a BabelBrain Integration Test
Integration tests are broken into BabelBrain's 3 major steps therefore to run integration tests for step 1 simply type:
```bash
pytest -k "integration_step1.py"
```
or
```bash
pytest -k "integration" -m "step1"
```
Similar to above, replace "step1" with the appropriate tag for whatever step you wish to test.

## Running a BabelBrain E2E Test
### Background
End-to-End Tests run the full BabelBrain pipeline and are parametrized based on:
- Trajectory type
    - Slicer
    - Brainsight
- Target Location
    - Deep_Target
    - Superficial_Target
    - Skull_Target
- CT Scan file
    - NONE
    - CT
    - ZTE
    - PETRA
- Patient Dataset
    - SDR_0p31
    - SDR_0p42
    - ...
- Transducer
    - Single
    - CTX_500
    - ...
- Frequency
    - Dependent on Transducer
- GPU Backend
    - OpenCL
    - CUDA
    - Metal
    - MLX

**Note:** Tolerance is another parameter used when comparing BabelBrain outputs in the test.
- 0% tolerance
- 1% tolerance
- 5% tolerance

Therefore, a single BabelBrain full pipeline test could have upwards of 100000+ versions and would not be feasible to run all of them. The user can narrow down what set of tests they want to run by specifying the parameters when calling the test  

**Example**  
A test_full_pipeline_normal test focused only on a deep target created using slicer for patient SDR_0p31 while using no CT data/substitute, a single tx @ 200 kHz, and a CUDA backend could be specified as follows:

```bash
pytest -k "test_full_pipeline_normal and slicer and Deep_Target and SDR_0p31 and NONE and Single and 200kHz and CUDA"
```

While this works, we've introduced the `basic_babelbrain_params` marker which grabs tests with the following parameter combinations:
- Target Location
    - Deep_Target
- CT Scan file
    - NONE
    - CT
    - ZTE
- Patient Dataset
    - ID_0082 (contains both CT and ZTE data)
- Transducer
    - H317
- Frequency
    - 250 kHz (low frequency)
    - 825 kHz (high frequency)

Which provides good coverage of BabelBrain user cases while being more manageable to run. Note that the trajectory type and GPU Backend still need to be specified by the user or can be left empty and tests will be skipped if they cannot be run on the user's current system.

There are 3 E2E test that can be performed:
### test_full_pipeline_normal
This test is used to simply check if BabelBrain will run to completion without any errors. It saves plots, but does not check outputs therefore it is up to the user to review the results to verify they make sense visually. It can be run with the following command:
```bash
pytest -k "test_full_pipeline_normal" -m "basic_babelbrain_params"
```

### test_full_pipeline_regression_normal
This test compares the results from the current setup to a reference folder containing previously generated outputs. To create a reference folder with data, see the [Running a BabelBrain Generating_Outputs "Test"](#running-a-babelbrain-generating_outputs-test) below. 

Before running this type of test, ensure the `ref_output_folder_1` in your [config.ini](config.ini) file is set to the folder you want to use as reference. 

**Example**  
You're interested in testing your current setup of BabelBrain against BabelBrain v0.4.2 results then your [config.ini](config.ini) file should be specified something like
```ini
[Paths]
...
ref_output_folder_1 = /<User>/BabelBrain_Unit_Testing/Generated_Outputs/BabelBrain_0_4_2
```

You can then call the following command:
```bash
pytest -k "test_full_pipeline_regression_normal" -m "basic_babelbrain_params"
```

This will also run tests for different tolerances (e.g. 0%,1%,5%). If you're only interested in one type of tolerance, you can run:

```bash
pytest -k "test_full_pipeline_regression_normal and MLX" -m "basic_babelbrain_params and tol_0"
```
or replace tol_0 with tol_1 or tol_5

### test_full_pipeline_two_outputs
This test is similar to above, except it doesn't run your current setup and depends solely on previously generated output folders. 

Before running this type of test, ensure both  `ref_output_folder_1` and `ref_output_folder_2` in your [config.ini](config.ini) file are set to the folders you want to use. 

**Example**  
You're interested in testing MLX outputs against Metal outputs then your [config.ini](config.ini) file should be specified something like
```ini
[Paths]
...
ref_output_folder_1 = /<User>/BabelBrain_Unit_Testing/Generated_Outputs/METAL
ref_output_folder_2 = /<User>/BabelBrain_Unit_Testing/Generated_Outputs/MLX
```

You can then call the following command:
```bash
pytest -k "test_full_pipeline_two_outputs"
```

This will also run tests for different tolerances (e.g. 0%,1%,5%) which you can specify (same as previous section). Note that we don't include the "basic_babelbrain_params" marker since it should be used when generating the outputs beforehand.

The html report breaks the comparison down by matching subfolder (i.e. one section per compared parameter set/device), each showing the reference and test file paths and tolerance used. For any mismatched 3D field, it embeds an interactive slice viewer (reference, test, and diff side by side, sharing a fixed color scale across slices) with a slider you can drag to any slice, plus a Play/Pause button to auto-scan through the volume — all self-contained in the html file, no server needed. To keep report size/generation time reasonable, each viewer samples up to 24 slices spread evenly across the volume rather than every slice. Mismatched single-value variables (e.g. scalar summary metrics) are listed in a table with their reference/test values and absolute/relative differences.

## Running a BabelBrain Generating_Outputs "Test"
The test_generate_valid_outputs "test" is used not to check BabelBrain proper functionality, but to leverage pytest and qtbot to automatically generate outputs for various BabelBrain parameters to be used later in regression tests.

Before running this type of "test", ensure `gen_output_folder` in your [config.ini](config.ini) file is set to the folder where you want to store your outputs. The folder name should be descriptive of what you're generating. Some examples include

Outputs for BabelBrain v0.4.2
```ini
[Paths]
...
gen_output_folder = /<user>/BabelBrain_Unit_Testing/Generated_Outputs/BabelBrain_v0_4_2
```

Outputs for latest BabelBrain version using only Metal GPU backend
```ini
[Paths]
...
gen_output_folder = /<user>/BabelBrain_Unit_Testing/Generated_Outputs/Metal
```

Outputs for latest BabelBrain version using newer version of cupy
```ini
[Paths]
...
gen_output_folder = /<user>/BabelBrain_Unit_Testing/Generated_Outputs/Cupy_v13_6
```

This "test" can be run with the following command:
```bash
pytest -k "test_generate_valid_outputs" -m "basic_babelbrain_params"
```

## Isolating a pytest run (crash and memory triage)

A large selection is a single long-lived process: `pytest Tests -m "basic_babelbrain_params"` collects
~1300 cases and runs them all in one interpreter. When that process dies half way through — killed by a
signal, or by the OS after memory creeps up — the terminal scrollback is all you get, the html report is
never written, and there is no record of which case was running.

[Tools/isolate_pytest.py](Tools/isolate_pytest.py) takes the same arguments you would give pytest, asks
pytest which cases they select, and runs **each case in its own pytest process**. A process per case is
the only bullet-proof deallocation: when it exits, every host buffer, Metal/CUDA/OpenCL context and Qt
object is released back to the OS, so anything that still grows across cases is on disk or in the GPU
driver rather than a leak that teardown would fix.

```bash
# generate a per-case runner script, run nothing
python Tests/Tools/isolate_pytest.py -- -s -m "basic_babelbrain_params"

# collect and run now, one case per process
python Tests/Tools/isolate_pytest.py --run -- -s -m "basic_babelbrain_params"

# rerun only the cases that did not pass
python Tests/Tools/isolate_pytest.py --run --from-file PyTest_Reports/isolated/<run>/failed_nodeids.txt
```

Everything after `--` goes to pytest verbatim. With no path argument the whole `Tests` tree is selected.
The tool works from any directory: it finds the BabelBrain checkout holding this `Tests` clone from its
own location and runs pytest from there (`--root` points it at a different checkout).

### What you get

Artifacts are written to `<BabelBrain>/PyTest_Reports/isolated/<timestamp>/`:

| file | contents |
| --- | --- |
| `nodeids.txt` | every collected case, one per line |
| `run_isolated.sh` / `.bat` | one `run_case` line per case — edit or comment out lines freely |
| `logs/NNNN_<case>.log` | full output of that case, flushed line by line so a hard kill still leaves a readable file |
| `summary.csv` | status, exit code, duration, peak RSS and free memory per case |
| `html/NNNN_<case>.html` | that case's pytest-html report, with the GUI screenshots embedded |
| `failed_nodeids.txt` | feed straight back in with `--from-file` |

Exit codes are decoded per case: `PASS`, `FAIL`, `CRASH` with the signal name (`SIGSEGV`, `SIGBUS`,
`SIGABRT`, `SIGKILL` — the last usually meaning the OS ran out of memory), `TIMEOUT`, `MEMLIMIT`, and
`NO_TESTS_RUN`. Child processes run with `PYTHONFAULTHANDLER=1`, so a native segfault prints the Python
line that caused it instead of dying silently.

### Chasing a memory problem

Every row of `summary.csv` records peak RSS of the whole process tree plus free system memory before,
during and after the case.

```bash
# is one case at fault, or is it accumulation? passes at 1, dies at 20 => accumulation
python Tests/Tools/isolate_pytest.py --run --group 20 -- -m "basic_babelbrain_params"

# does one case leak? run it 5x inside a single process and watch peak RSS
python Tests/Tools/isolate_pytest.py --run --repeat-each 5 --group 5 -- -k "H317 and Metal and NONE" -m basic_babelbrain_params

# stop a runaway before the OS kills your session, and bound a hang
python Tests/Tools/isolate_pytest.py --run --max-rss-mb 60000 --timeout 3600 -- -m "basic_babelbrain_params"
```

Other useful options: `--stop-on-crash`, `--start-at`/`--max-cases` to resume a long run, `--cooldown`
to let the GPU driver settle between cases, `--no-html` to skip the per-case html report (keep it on if a
test takes screenshots: `conftest.py` attaches them through `pytest_html.extras`, which only exists while
the html plugin is loaded), `--quiet-child` to keep the console tidy (the logs are unaffected), and
`--dry-run` to see the commands. Long GPU steps
produce no output for minutes, so the runner prints a heartbeat with elapsed time and current RSS while a
case is silent (`--heartbeat SEC`, `0` disables). Run `--help` for the full list.
