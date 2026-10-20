"""
Major release end-to-end tests (BabelBrain_Pytest_Usage.md, Table 9).

e2e_test_remote_full_pipeline
    Full pipeline offloaded to a BabelBrain job server. Uses the server in the optional
    [Remote] section of Tests/config.ini (name, host, port, token, https), otherwise starts
    a local headless server (BabelBrain.py --serve). Only checks that the run completes.

e2e_test_custom_tx_full_pipeline
    Creates a custom transducer from each template in custom_tx_templates/ and compares
    its full pipeline outputs against the built-in transducer the template replicates.
    Both runs are saved in <gen_output_folder>/e2e_custom_tx_full_pipeline/ before being
    compared (see pipeline_helpers.ensure_case_outputs).
"""
import os
import signal
import socket
import subprocess
import sys
import time
from dataclasses import replace
from pathlib import Path

import pytest
import yaml

from Tests.helpers.parameters import (BABELBRAIN_ROOT, CUSTOM_TX_EQUIVALENTS,
                              CUSTOM_TX_TEMPLATES_DIR, RELEASE_BACKENDS, config,
                              requires_feature)
from Tests.helpers.pipeline_helpers import (PipelineCase, case_param,
                                    saved_outputs_reusable, thermal_outputs)

pytestmark = pytest.mark.major_release

CUSTOM_TX_SUITE = 'e2e_custom_tx_full_pipeline'
REMOTE_SERVER_NAME = 'pytest-server'
SERVER_START_TIMEOUT_S = 180

# ================================================================================================================================
# CASES
# ================================================================================================================================
REMOTE_CASES = [
    pytest.param(PipelineCase(dataset='ernie', transducer=tx, freq_khz=freq, backend='Remote'),
                 id=f"ernie_{tx}_{freq}kHz", marks=[pytest.mark.gpu, pytest.mark.slow])
    for tx, freq in [('Single', 500), ('H317', 250)]
]

# Slicer targets for every pair so REMOPD isn't flipped and compares 1 to 1 with its custom version
CUSTOM_TX_CASES = [
    case_param(PipelineCase(dataset='ernie', trajectory_type='slicer', backend=backend,
                            custom_tx_template=template, **equivalent),
               marks=[pytest.mark.slow])
    for backend in RELEASE_BACKENDS
    for template, equivalent in CUSTOM_TX_EQUIVALENTS.items()
]


# ================================================================================================================================
# FIXTURES
# ================================================================================================================================
def _free_port():
    with socket.socket() as s:
        s.bind(('127.0.0.1', 0))
        return s.getsockname()[1]


@pytest.fixture(scope='module')
def remote_server(tmp_path_factory):
    """Remote server registered in an isolated servers.json (the user's list is left alone)."""
    import RemoteServers

    servers_dir = tmp_path_factory.mktemp('remote_servers')
    process = None
    log_file = None

    if config.has_section('Remote') and config['Remote'].get('host'):
        remote = config['Remote']
        server = {'name': remote.get('name', REMOTE_SERVER_NAME), 'host': remote['host'],
                  'port': remote.getint('port', 8760), 'token': remote.get('token') or None,
                  'https': remote.getboolean('https', False)}
    else:
        server = {'name': REMOTE_SERVER_NAME, 'host': '127.0.0.1', 'port': _free_port(), 'token': None}
        log_path = servers_dir / 'server.log'
        log_file = open(log_path, 'w')
        process = subprocess.Popen([sys.executable, os.path.join('BabelBrain', 'BabelBrain.py'), '--serve', '--headless',
                                    '--serve-host', server['host'], '--serve-port', str(server['port'])],
                                   cwd=BABELBRAIN_ROOT, stdout=log_file, stderr=subprocess.STDOUT)

    with pytest.MonkeyPatch.context() as mp:
        mp.setattr(RemoteServers, 'CONFIG_DIR', str(servers_dir))
        mp.setattr(RemoteServers, 'SERVERS_FILE', str(servers_dir / 'servers.json'))
        RemoteServers.save_servers([server])

        deadline = time.time() + SERVER_START_TIMEOUT_S
        ok, info = RemoteServers.test_connection(server)
        while not ok and time.time() < deadline:
            if process is not None and process.poll() is not None:
                break
            time.sleep(2)
            ok, info = RemoteServers.test_connection(server)

        if not ok:
            _stop_server(process, log_file)
            details = f" Server log: {servers_dir / 'server.log'}" if process is not None else ""
            pytest.fail(f"Remote server {server['host']}:{server['port']} not reachable: {info}.{details}")

        yield server

    _stop_server(process, log_file)


def _stop_server(process, log_file):
    if process is not None and process.poll() is None:
        process.send_signal(signal.SIGINT)
        try:
            process.wait(timeout=60)
        except subprocess.TimeoutExpired:
            process.kill()
    if log_file is not None:
        log_file.close()


@pytest.fixture(scope='module')
def custom_tx_root(tmp_path_factory):
    """Isolated folder for custom transducers so the user's saved ones are never touched."""
    from CreateTransducers import transducer_creator

    root = tmp_path_factory.mktemp('custom_transducers')
    with pytest.MonkeyPatch.context() as mp:
        mp.setenv('BABEL_CUSTOM_TX_ROOT', str(root))
        # CustomTransducer writes to its own module constant rather than custom_transducers_root()
        mp.setattr(transducer_creator, 'CUSTOM_TRANSDUCERS_FOLDER', root)
        yield {'root': root, 'created': {}}
    sys.path[:] = [p for p in sys.path if p != str(root)]


def _create_custom_tx(pipeline, custom_tx_root, case):
    """Create (once per module) the custom transducer for case.custom_tx_template on case.backend."""
    from CreateTransducers.transducer_creator import CustomTransducer, get_class_name
    from BabelBrain.SelFiles.SelFiles import SelFiles

    key = (case.custom_tx_template, case.backend)
    if key in custom_tx_root['created']:
        return custom_tx_root['created'][key]

    devices = [dev for dev, backend in SelFiles.GetAvailableGPUs(None) if backend == case.backend]
    if not devices:
        pytest.skip(f"{case.backend} backend is not available on this machine")

    # Unique name per template/backend so each one gets its own package
    with open(CUSTOM_TX_TEMPLATES_DIR / f"{case.custom_tx_template}.yaml") as f:
        template = f.read()
    tx_name = f"Pytest_{case.custom_tx_template}_{case.backend}"
    config_file = Path(pipeline.work_dir) / f"{tx_name}.yaml"
    with open(config_file, 'w') as f:
        f.write(_replace_yaml_name(template, tx_name))

    pipeline.auto_answer_dialogs()
    with open(BABELBRAIN_ROOT / 'BabelBrain' / 'version-gui.txt') as f:
        bb_version = f.readline().strip()
    tx = CustomTransducer(bb_version=bb_version, transducer_yaml=str(config_file),
                          gpu=devices[0], computing_backend=case.backend)
    custom_tx_root['created'][key] = get_class_name(tx.name)
    return custom_tx_root['created'][key]


def _replace_yaml_name(template_text, new_name):
    lines = template_text.splitlines(keepends=True)
    for i, line in enumerate(lines):
        if line.startswith('name:'):
            lines[i] = f"name: {new_name}\n"
            break
    else:
        raise ValueError("Template has no 'name:' entry")
    assert yaml.safe_load(''.join(lines))['name'] == new_name
    return ''.join(lines)


# ================================================================================================================================
# TESTS
# ================================================================================================================================
@requires_feature('remote_servers')
@pytest.mark.parametrize('case', REMOTE_CASES)
def test_e2e_remote_full_pipeline(pipeline, remote_server, case):
    output_dir = pipeline.work_dir / case.case_id
    bb = pipeline.run(case, output_dir, remote_server_name=remote_server['name'])

    assert bb.Config['ComputingBackend'] == 5, "Remote server was not selected as the computing engine"
    assert thermal_outputs(output_dir), "Remote run finished but no thermal outputs were returned"


def _custom_tx_case_outputs(pipeline, custom_tx_root, case):
    """Saved outputs of the built-in transducer and of its custom replica (the custom transducer
    is only created when its outputs have to be generated)."""
    builtin_dir = pipeline.ensure_outputs(CUSTOM_TX_SUITE, replace(case, custom_tx_template=None))
    custom_name = None
    if not saved_outputs_reusable(CUSTOM_TX_SUITE, case):
        custom_name = _create_custom_tx(pipeline, custom_tx_root, case)
    custom_dir = pipeline.ensure_outputs(CUSTOM_TX_SUITE, case, custom_tx_name=custom_name)
    return builtin_dir, custom_dir


@requires_feature('custom_transducers')
@pytest.mark.generate_outputs
@pytest.mark.parametrize('case', CUSTOM_TX_CASES)
def test_generate_e2e_custom_tx_full_pipeline(pipeline, custom_tx_root, case):
    _custom_tx_case_outputs(pipeline, custom_tx_root, case)


@requires_feature('custom_transducers')
@pytest.mark.parametrize('case', CUSTOM_TX_CASES)
def test_e2e_custom_tx_full_pipeline(pipeline, custom_tx_root, tolerance, case):
    builtin_dir, custom_dir = _custom_tx_case_outputs(pipeline, custom_tx_root, case)
    pipeline.assert_thermal_outputs_match(
        builtin_dir, custom_dir, tolerance,
        message=f"Custom {case.custom_tx_template} transducer differs from built-in {case.transducer}")
