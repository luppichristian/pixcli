import os
from pathlib import Path
import shutil
import subprocess

import pytest


POWERSHELL = shutil.which('powershell.exe') if os.name == 'nt' else None
pytestmark = pytest.mark.skipif(not POWERSHELL, reason='Windows PowerShell required')
PROJECT = Path(__file__).resolve().parents[1]


@pytest.fixture(scope='module')
def fake_executable(tmp_path_factory):
    directory = tmp_path_factory.mktemp('setup-fake-exe')
    source = directory / 'stub.cs'
    executable = directory / 'stub.exe'
    source.write_text('''
using System;
using System.IO;
class Stub {
    static int Main(string[] args) {
        string name = Path.GetFileNameWithoutExtension(Environment.GetCommandLineArgs()[0]);
        string call = name + " " + string.Join(" ", args);
        File.AppendAllText(Environment.GetEnvironmentVariable("PIXCLI_TEST_LOG"), call + Environment.NewLine);
        return call == Environment.GetEnvironmentVariable("PIXCLI_TEST_FAIL") ? 7 : 0;
    }
}
''', encoding='utf-8')
    env = os.environ.copy()
    env.update(PIXCLI_TEST_SOURCE=str(source), PIXCLI_TEST_EXE=str(executable))
    result = subprocess.run(
        [POWERSHELL, '-NoProfile', '-NonInteractive', '-Command',
         'Add-Type -Path $env:PIXCLI_TEST_SOURCE -OutputAssembly $env:PIXCLI_TEST_EXE -OutputType ConsoleApplication'],
        env=env, capture_output=True, text=True,
    )
    assert result.returncode == 0, result.stderr
    return executable


def run_setup(tmp_path, fake_executable, failure=''):
    project = tmp_path / 'project'
    scripts = project / 'scripts'
    binary = project / '.venv' / 'Scripts'
    fake_bin = tmp_path / 'bin'
    for directory in (scripts, binary, fake_bin):
        directory.mkdir(parents=True)
    shutil.copy2(PROJECT / 'setup.ps1', project / 'setup.ps1')
    shutil.copy2(fake_executable, binary / 'pixcli.exe')
    for name in ('uv.exe', 'git.exe'):
        shutil.copy2(fake_executable, fake_bin / name)
    (scripts / 'setup-animation.ps1').write_text('''
Add-Content -LiteralPath $env:PIXCLI_TEST_LOG -Value 'backend'
if ($env:PIXCLI_TEST_FAIL -eq 'backend') { throw 'simulated backend failure' }
''', encoding='utf-8')
    log = tmp_path / 'calls.txt'
    env = os.environ.copy()
    env.update(PATH=str(fake_bin) + os.pathsep + env['PATH'],
               PIXCLI_TEST_LOG=str(log), PIXCLI_TEST_FAIL=failure)
    result = subprocess.run([POWERSHELL, '-NoProfile', '-NonInteractive', '-File', str(project / 'setup.ps1')],
                            cwd=project, env=env, capture_output=True, text=True)
    calls = log.read_text().splitlines() if log.exists() else []
    return result, calls


CALLS = [
    'uv sync --locked --python 3.11',
    'backend',
    'pixcli models download -profile image',
    'pixcli models download -profile animation',
]


def test_setup_prepares_every_feature(tmp_path, fake_executable):
    result, calls = run_setup(tmp_path, fake_executable)
    assert result.returncode == 0, result.stderr
    assert calls == CALLS


@pytest.mark.parametrize('failure_index', range(len(CALLS)))
def test_setup_stops_on_failure(tmp_path, fake_executable, failure_index):
    result, calls = run_setup(tmp_path, fake_executable, CALLS[failure_index])
    assert result.returncode != 0, result.stdout
    assert calls == CALLS[:failure_index + 1]
