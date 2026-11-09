import os
from pathlib import Path
import subprocess
import sys

import psutil
import pytest


def test_probe_preserves_output_and_process_errors():
    from pixcli.comfy_backend import _run_supervised_probe
    result = _run_supervised_probe([sys.executable, '-c', 'print("ready")'], timeout=10)
    assert result.stdout.strip() == 'ready'
    with pytest.raises(subprocess.CalledProcessError) as error:
        _run_supervised_probe([sys.executable, '-c', 'import sys; print("bad", file=sys.stderr); sys.exit(3)'], timeout=10)
    assert error.value.returncode == 3 and 'bad' in error.value.stderr


def test_probe_enforces_system_commit_reserve_before_start(monkeypatch, tmp_path):
    from pixcli import comfy_backend
    monkeypatch.setattr(comfy_backend, '_free_commit_bytes', lambda: 1024**3)
    marker = tmp_path / 'started'
    with pytest.raises(MemoryError, match='commit memory'):
        comfy_backend._run_supervised_probe([sys.executable, '-c', 'import pathlib,sys; pathlib.Path(sys.argv[1]).touch()', str(marker)],
                                            timeout=10, min_free_commit_gib=2)
    assert not marker.exists()


def test_probe_stops_child_when_command_tree_exceeds_budget(tmp_path):
    from pixcli.comfy_backend import _run_supervised_probe, _process_tree_resident_bytes
    parent = psutil.Process()
    budget = (_process_tree_resident_bytes(parent) + 70 * 1024**2) / 1024**3
    marker = tmp_path / 'pid'
    code = ('import pathlib,sys,time,os; pathlib.Path(sys.argv[1]).write_text(str(os.getpid())); '
            'data=bytearray(192*1024**2); time.sleep(5)')
    with pytest.raises(MemoryError, match='RAM memory budget'):
        _run_supervised_probe([sys.executable, '-c', code, str(marker)], timeout=10, ram_budget_gib=budget)
    if marker.exists():
        assert not psutil.pid_exists(int(marker.read_text()))


def test_probe_timeout_terminates_descendants(tmp_path):
    from pixcli.comfy_backend import _run_supervised_probe
    marker = tmp_path / 'child_pid'
    code = ('import pathlib,subprocess,sys,time; '
            'child=subprocess.Popen([sys.executable,"-c","import time; time.sleep(30)"]); '
            'pathlib.Path(sys.argv[1]).write_text(str(child.pid)); time.sleep(30)')
    with pytest.raises(subprocess.TimeoutExpired):
        _run_supervised_probe([sys.executable, '-c', code, str(marker)], timeout=1)
    assert marker.exists()
    assert not psutil.pid_exists(int(marker.read_text()))


@pytest.mark.skipif(sys.platform != 'win32', reason='Windows hard working-set limit')
def test_python_probe_caps_worker_before_executing_code():
    from pixcli.comfy_backend import _run_python_probe, _process_tree_resident_bytes
    parent = _process_tree_resident_bytes(psutil.Process())
    budget = (parent + 512 * 1024**2) / 1024**3
    code = ('import ctypes; k=ctypes.WinDLL("kernel32"); k.GetCurrentProcess.restype=ctypes.c_void_p; '
            'k.GetProcessWorkingSetSizeEx.argtypes=(ctypes.c_void_p,ctypes.POINTER(ctypes.c_size_t),'
            'ctypes.POINTER(ctypes.c_size_t),ctypes.POINTER(ctypes.c_ulong)); '
            'a=ctypes.c_size_t(); b=ctypes.c_size_t(); f=ctypes.c_ulong(); '
            'assert k.GetProcessWorkingSetSizeEx(k.GetCurrentProcess(),ctypes.byref(a),ctypes.byref(b),ctypes.byref(f)); '
            'print(b.value, f.value)')
    result = _run_python_probe(sys.executable, code, timeout=10, ram_budget_gib=budget)
    maximum, flags = map(int, result.stdout.strip().split())
    assert flags & 4
    assert maximum <= 384 * 1024**2
