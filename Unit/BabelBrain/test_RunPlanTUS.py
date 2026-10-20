"""
Unit test for PlanTUSViewer.RunPlanTUS.RunPlanTUSBackground (BabelBrain_Pytest_Usage.md, Table 7).

Runs PlanTUS (bundled in BabelBrain/ExternalBin/PlanTUS) for a single element transducer on a
copy of the ID_0082 charm folder, only checking that it completes and writes its outputs. PlanTUS
runs on the CPU (through SimNIBS), so the test doesn't depend on a GPU or computing backend.
Requires simnibs_root_path in Tests/config.ini.
"""
import glob
import os
import shutil
import time
from multiprocessing import Process, Queue
from pathlib import Path
from queue import Empty

import nibabel
import numpy as np
import pytest

from Tests.helpers.parameters import (BABELBRAIN_ROOT, DATASETS, DEFAULT_TARGET,
                              requires_feature, trajectory_file)
from Tests.helpers.pipeline_helpers import simnibs_root

PLANTUS_TIMEOUT_S = 60 * 60
PLANTUS_DATASET = 'ID_0082'


@requires_feature('plantus')
@pytest.mark.real_data  # runs PlanTUS on the ID_0082 charm folder
@pytest.mark.slow
def test_RunPlanTUSBackground(tmp_path):
    from ConvMatTransform import ReadTrajectoryBrainsight
    from CreateVoxelMask import create_target_mask
    from PlanTUSViewer.RunPlanTUS import (FindTPOEquivalent, PlanTUSTxConfig,
                                          RunPlanTUSBackground)

    simnibs = simnibs_root()

    ds = DATASETS[PLANTUS_DATASET]
    m2m_src, t1_src = Path(ds['m2m_folder_path']), Path(ds['T1_path'])
    trajectory = trajectory_file('brainsight', PLANTUS_DATASET, DEFAULT_TARGET)
    for path in (m2m_src, t1_src, trajectory):
        if not path.exists():
            pytest.skip(f"Missing PlanTUS input {path}")

    # PlanTUS writes next to the mesh, so it works on a copy of the charm folder
    m2m = tmp_path / m2m_src.name
    shutil.copytree(m2m_src, m2m, ignore=shutil.ignore_patterns('PlanTUS'))
    t1_path = tmp_path / t1_src.name
    shutil.copy(t1_src, t1_path)
    msh_files = glob.glob(str(m2m / '*.msh'))
    assert len(msh_files) == 1, f"Expected one .msh file in {m2m}, found {msh_files}"

    # Target mask around the trajectory target (2 mm radius), as RunPlanTUS does
    traj_copy = tmp_path / trajectory.name
    shutil.copy(trajectory, traj_copy)
    RMat = ReadTrajectoryBrainsight(str(traj_copy))
    if RMat.ndim == 3:
        RMat = RMat[:, :, 0]
    mask_path = tmp_path / (traj_copy.stem + '_PlanTUSMask_T0.nii.gz')
    raddi = tuple(np.ceil(2 / np.array(nibabel.load(str(t1_path)).header.get_zooms()[:3])).astype(int))
    create_target_mask(str(t1_path), RMat[:3, 3], str(mask_path), raddi=raddi)

    # Single element transducer at 500 kHz (Single transducer GUI defaults)
    freq, diameter, focal_length = 500e3, 0.05, 0.05
    plane_offset, tpo_equivalent, flhm = FindTPOEquivalent(freq, diameter, focal_length)
    distance = float(tpo_equivalent * 1e3)
    tx_config_path = tmp_path / 'PlanTUSTxConfig.yaml'
    PlanTUSTxConfig(transducer_diameter=diameter * 1e3,
                    min_distance=distance,
                    max_distance=distance,
                    optimal_distance=distance,
                    max_angle=10.0,
                    plane_offset=float(plane_offset * 1e3),
                    additional_offset=0.0,
                    focal_distance_list=[distance],
                    flhm_list=[float(flhm * 1e3)],
                    IDTarget=traj_copy.stem,
                    bUseGenericTransducerModel=True).ExportYAML(str(tx_config_path))

    scriptbase = str(BABELBRAIN_ROOT / 'BabelBrain' / 'ExternalBin' / 'PlanTUS') + os.sep
    plantus_root = os.path.join(scriptbase, 'PlanTUS')
    assert os.path.isfile(os.path.join(plantus_root, 'PlanTUS_wrapper.py')), f"PlanTUS not bundled at {plantus_root}"

    # Run in a separate process like BabelBrain does (RunPlanTUSBackground redirects stdout to the queue).
    # The queue is drained while waiting, otherwise the child blocks on exit flushing it.
    queue = Queue()
    process = Process(target=RunPlanTUSBackground,
                      args=(queue, scriptbase, simnibs, plantus_root, str(t1_path), msh_files[0],
                            str(mask_path), str(tx_config_path)))
    process.start()
    messages = []
    deadline = time.time() + PLANTUS_TIMEOUT_S
    while process.is_alive() or not queue.empty():
        try:
            messages.append(queue.get(timeout=1))
        except Empty:
            pass
        if time.time() > deadline:
            process.terminate()
            pytest.fail(f"PlanTUS did not finish within {PLANTUS_TIMEOUT_S / 60:.0f} min")
    process.join()
    log = ''.join(messages)

    assert '--Babel-Brain-Success' in log, f"PlanTUS failed:\n{log[-5000:]}"
    output_dir = m2m / 'PlanTUS' / mask_path.name.replace('.nii.gz', '')
    assert (output_dir / 'distances_skin.func.gii').exists(), f"PlanTUS outputs missing in {output_dir}\n{log[-5000:]}"
