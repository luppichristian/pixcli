import json
import os
import subprocess
import sys

import pytest


@pytest.mark.skipif(sys.platform != 'win32', reason='Windows working-set limits')
def test_residency_limit_pages_file_backed_memory_and_restores(tmp_path):
    script = r'''
import ctypes, json, mmap, os, psutil, sys
from pixcli.residency import resident_memory_limit

kernel = ctypes.WinDLL('kernel32', use_last_error=True)
kernel.GetCurrentProcess.restype = ctypes.c_void_p
kernel.GetProcessWorkingSetSizeEx.argtypes = (ctypes.c_void_p, ctypes.POINTER(ctypes.c_size_t), ctypes.POINTER(ctypes.c_size_t), ctypes.POINTER(ctypes.c_ulong))
def settings():
    low, high, flags = ctypes.c_size_t(), ctypes.c_size_t(), ctypes.c_ulong()
    if not kernel.GetProcessWorkingSetSizeEx(kernel.GetCurrentProcess(), ctypes.byref(low), ctypes.byref(high), ctypes.byref(flags)):
        raise ctypes.WinError(ctypes.get_last_error())
    return low.value, high.value, flags.value

before = settings()
with open(sys.argv[1], 'w+b') as handle:
    handle.truncate(128 * 1024**2)
    with mmap.mmap(handle.fileno(), 0, access=mmap.ACCESS_READ) as mapped:
        with resident_memory_limit(64 / 1024) as applied:
            with resident_memory_limit(128 / 1024):
                inner_maximum = settings()[1]
            for offset in range(0, len(mapped), 4096):
                mapped[offset]
            resident = psutil.Process().memory_info().rss
after = settings()
print(json.dumps({'applied': applied, 'resident': resident, 'inner_maximum': inner_maximum, 'before': before, 'after': after}))
'''
    result = subprocess.run([sys.executable, '-c', script, str(tmp_path / 'mapped.bin')],
                            capture_output=True, text=True, cwd=os.getcwd(), timeout=30)
    assert result.returncode == 0, result.stderr
    values = json.loads(result.stdout)
    assert values['applied'] is True
    assert values['inner_maximum'] <= 64 * 1024**2
    assert values['resident'] <= 66 * 1024**2
    assert values['before'] == values['after']


def test_no_budget_does_not_change_residency():
    from pixcli.residency import resident_memory_limit
    with resident_memory_limit(None) as applied:
        assert applied is False


@pytest.mark.skipif(sys.platform != 'win32', reason='Windows working-set limits')
def test_other_process_residency_limit_reduces_its_working_set():
    from pixcli.residency import lower_process_residency_limit
    script = r'''
import os, sys, time
data = bytearray(96 * 1024**2)
print(os.getpid(), flush=True)
while sys.stdin.read(1):
    for offset in range(0, len(data), 4096):
        data[offset] = 1
    print('touched', flush=True)
'''
    process = subprocess.Popen([sys.executable, '-c', script], stdin=subprocess.PIPE,
                               stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    try:
        pid = int(process.stdout.readline())
        import psutil
        created = psutil.Process(pid).create_time()
        with pytest.raises(RuntimeError, match='identity'):
            lower_process_residency_limit(pid, 48 * 1024**2, expected_create_time=created - 60)
        assert lower_process_residency_limit(pid, 48 * 1024**2, expected_create_time=created)
        process.stdin.write('x')
        process.stdin.flush()
        assert process.stdout.readline().strip() == 'touched'
        assert psutil.Process(pid).memory_info().rss <= 50 * 1024**2
    finally:
        process.stdin.close()
        process.wait(timeout=15)
