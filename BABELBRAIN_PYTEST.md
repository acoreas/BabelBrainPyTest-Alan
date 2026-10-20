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

**Unit tests and real data**

Unit tests that read the real datasets (ID_0082, ernie) get the `real_data` marker automatically. To run only the unit tests that use data they generate themselves (e.g. the synthetic head), which is quicker and needs no test data, add `-m "not real_data"`:

```bash
pytest -k "unit" -m "not real_data"           # unit tests without real data
pytest -k "unit"                              # all unit tests
pytest -k "unit" -m "real_data"               # only the unit tests using real data
pytest -k "unit" -m "gpu and not real_data"   # combined with other markers
```

Only unit tests are marked: E2E, integration, regression and generate_outputs tests always use the real datasets.

**Important to note that some unit tests are parameterized** 

**Example**  
test_nibabel_to_sitk relies on T1 data and thus is parameterized for image datasets (ID_0082 and ernie), so it's marked `real_data`. Runnning

```bash
pytest -k "test_nibabel_to_sitk"
```
will collect/run for all datasets so tests ran would be:
- test_nibabel_to_sitk[ID_0082]
- test_nibabel_to_sitk[ernie]

Therefore to run a specific dataset, simply specify the dataset like so
```bash
pytest -k "test_nibabel_to_sitk[ID_0082]"
```
or
```bash
pytest -k "test_nibabel_to_sitk and ID_0082"
```

In general, running all the different parameters will not take too long to run. The exception being pytests for GPU functions.

### Running GPU unit tests
GPU unit tests are parameterized for spatial step
- low_res
- medium_res
- high_res
- stress_res

where high_res and stress_res are **much slower** to run due to higher resolution sims. Every spatial step runs on the real datasets (ID_0082, ernie) and on a small synthetic head (dataset id `synthetic`, generated automatically by [synthetic_dataset.py](helpers/synthetic_dataset.py) into the gitignored `Tests/.cache` folder), which makes the high resolutions practical. The release test sets add the real datasets at one resolution only: low_res for minor releases and medium_res for major releases, with the synthetic head at every resolution in both (`RELEASE_SPATIAL_STEPS` in [parameters.py](helpers/parameters.py)). Unit tests and suite consistency checks are marked `minor_release` / `major_release` accordingly, so

```bash
pytest -m "minor_release" -k "not gpu or Metal"   # minor release set on this machine's backend
pytest -m "major_release"                         # major release set, all backends
```

run the release sets (see [BabelBrain_Pytest_Usage.md](BabelBrain_Pytest_Usage.md)). The real datasets at high resolution can still be selected explicitly, e.g. `pytest Tests/Unit -k "ID_0082-High_Res and Metal"`, and `-m "not real_data"` keeps only the synthetic head. Each GPU test's report entry shows the inputs, the CPU and GPU outputs and a |GPU - CPU| difference panel (with the number of differing voxels in the whole volume), with a slider that moves the slice position through all three axes; the voxelization test shows the skin mesh, both voxelizations and the voxels only one of them found, with a slider that moves the camera around the mesh.

All plots in the reports use the same viewer ([report_viewers.py](helpers/report_viewers.py)): drag the slider (slice position, camera position, ...) or press Play to step through. All panels of a viewer stay on the same frame, and nothing plays until you press Play. Screenshots of the GUI (e.g. after each step of the full pipeline tests) are shown as plain images.

For a quick check on your machine's backend:

```bash
pytest Tests/Unit -m "gpu and not slow" -k "synthetic and Metal"
``` Note that tests for GPU function are marked with the "gpu" marker. That is, to collect/run all GPU unit tests, run the following command:

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

## Running a BabelBrain E2E Test
### Background
End-to-End Tests run the full BabelBrain pipeline and are parametrized based on (values defined in [parameters.py](helpers/parameters.py); options that the BabelBrain version under test doesn't offer are left out):
- Trajectory type
    - brainsight
    - slicer
    - localite
- Target Location
    - Superficial_Target_Left
    - Deep_Target_Left
- CT Scan file
    - NONE
    - CT
    - ZTE
    - PETRA
    - Density
- Patient Dataset
    - ID_0082
    - ernie
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
A test_full_pipeline_normal test focused only on a deep target created using slicer for patient ernie while using no CT data/substitute, a single tx @ 200 kHz, and a CUDA backend could be specified as follows:

```bash
pytest -k "test_full_pipeline_normal and slicer and Deep_Target_Left and ernie and NONE and Single and 200kHz and CUDA"
```

While this works, the `basic_babelbrain_params` marker grabs the basic configuration, the standard full pipeline set up:
- Patient Dataset
    - ernie
- CT Scan file
    - NONE
- Trajectory type / Target
    - brainsight / Superficial_Target_Left
- Transducer / Frequency
    - Single 500 kHz, CTX_500 500 kHz, H246 500 kHz, REMOPD 490 kHz, H317 250 kHz, DomeTx 220 kHz (one per transducer geometry type)
- GPU Backend
    - CUDA, OpenCL, Metal (backends not available on your system are skipped)

It's defined by `BASIC_PARAMS` in [parameters.py](helpers/parameters.py). The major release `regression_test_full_pipeline` (see [BabelBrain_Pytest_Usage.md](BabelBrain_Pytest_Usage.md), Table 10) runs this same configuration: its `REGRESSION_CASES` entry is `BASIC_PARAMS`, so changing the configuration there updates both. For the release test sets use the `minor_release` / `major_release` markers (see [BabelBrain_Pytest_Usage.md](BabelBrain_Pytest_Usage.md)); the other `*_babelbrain_params` markers choose their transducers/frequencies with `MARKER_TRANSDUCERS` in the same file.

Similarly, the `intensive_babelbrain_params` marker grabs the intensive configuration: ID_0082 with CT, Deep_Target_Left, H317 at 825 kHz on CUDA, OpenCL and Metal. Tests with this marker also run with the configuration's multipoint profile (MultiFocus_Profile1) and intensive thermal profile, so they run it exactly. It's defined by `INTENSIVE_PARAMS` in [parameters.py](helpers/parameters.py), and the major release `regression_test_full_pipeline_intensive` runs this same configuration.

```bash
pytest -k "test_full_pipeline_normal" -m "intensive_babelbrain_params"
```

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

The html report breaks the comparison down by matching subfolder (i.e. one section per compared parameter set/device), each showing the reference and test file paths and tolerance used. For any mismatched 3D field, it embeds a slice viewer (reference, test, and diff side by side, sharing a fixed color scale across slices) with a slider you can drag to any slice and a Play button to step through the volume, all self-contained in the html file, no server needed. To keep report size/generation time reasonable, each viewer samples up to 24 slices spread evenly across the volume rather than every slice. Mismatched single-value variables (e.g. scalar summary metrics) are listed in a table with their reference/test values and absolute/relative differences.

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

Each case is saved in `<gen_output_folder>/<case id>` together with a `pytest_case_complete.yaml` marker written once the run succeeded. A later run skips cases already generated by the same BabelBrain commit with the same trajectory and thermal profile files, and reruns (from an empty folder) anything else, including folders from a run that crashed halfway. Set `BABEL_PYTEST_REGENERATE=1` to rerun every case.

## Release testing
The tests required for minor and major releases (regression, E2E and the PlanTUS unit test) are described in [BabelBrain_Pytest_Usage.md](BabelBrain_Pytest_Usage.md). The regression tests in [Regression](Regression) generate their outputs first (`test_generate_*`, marked `generate_outputs`) and then compare them against `<ref_output_folder_1>/<suite>/<case id>`, so a failed comparison never throws away a finished simulation. To create references, check out the reference BabelBrain version, point `gen_output_folder` at the reference location and run:
```bash
pytest Tests/Regression -m generate_outputs
```
The parameters of each regression test (one entry per row of Table 10: dataset, trajectory, target, scan types, transducers, profiles, ...) are in `REGRESSION_CASES` in [parameters.py](helpers/parameters.py); change a row there and the regression cases and the suite consistency checks follow. `regression_test_full_pipeline` and `regression_test_full_pipeline_intensive` use the basic and intensive configurations, `BASIC_PARAMS` and `INTENSIVE_PARAMS`, so they're edited there and the `basic_babelbrain_params` / `intensive_babelbrain_params` markers follow too. [test_suite_consistency.py](Consistency/test_suite_consistency.py) checks that every dataset, target, scan type, transducer/frequency and input file a case names exists.

Tests for features an older BabelBrain version doesn't have (remote servers, custom transducers, PlanTUS) are skipped automatically.

## Isolating a pytest run (crash and memory triage)

A large selection is a single long-lived process: `pytest Tests -m "basic_babelbrain_params"` collects
~1300 cases and runs them all in one interpreter. When that process dies half way through — killed by a
signal, or by the OS after memory creeps up — the terminal scrollback is all you get, the html report is
never written, and there is no record of which case was running.

[scripts/isolate_pytest.py](scripts/isolate_pytest.py) takes the same arguments you would give pytest, asks
pytest which cases they select, and runs **each case in its own pytest process**. A process per case is
the only bullet-proof deallocation: when it exits, every host buffer, Metal/CUDA/OpenCL context and Qt
object is released back to the OS, so anything that still grows across cases is on disk or in the GPU
driver rather than a leak that teardown would fix.

```bash
# generate a per-case runner script, run nothing
python Tests/scripts/isolate_pytest.py -- -s -m "basic_babelbrain_params"

# collect and run now, one case per process
python Tests/scripts/isolate_pytest.py --run -- -s -m "basic_babelbrain_params"

# rerun only the cases that did not pass
python Tests/scripts/isolate_pytest.py --run --from-file PyTest_Reports/isolated/<run>/failed_nodeids.txt
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
python Tests/scripts/isolate_pytest.py --run --group 20 -- -m "basic_babelbrain_params"

# does one case leak? run it 5x inside a single process and watch peak RSS
python Tests/scripts/isolate_pytest.py --run --repeat-each 5 --group 5 -- -k "H317 and Metal and NONE" -m basic_babelbrain_params

# stop a runaway before the OS kills your session, and bound a hang
python Tests/scripts/isolate_pytest.py --run --max-rss-mb 60000 --timeout 3600 -- -m "basic_babelbrain_params"
```

Other useful options: `--stop-on-crash`, `--start-at`/`--max-cases` to resume a long run, `--cooldown`
to let the GPU driver settle between cases, `--no-html` to skip the per-case html report (keep it on if a
test takes screenshots: `conftest.py` (through `helpers/report_viewers.py`) attaches them through `pytest_html.extras`, which only exists while
the html plugin is loaded), `--quiet-child` to keep the console tidy (the logs are unaffected), and
`--dry-run` to see the commands. Long GPU steps
produce no output for minutes, so the runner prints a heartbeat with elapsed time and current RSS while a
case is silent (`--heartbeat SEC`, `0` disables). Run `--help` for the full list.
