import subprocess
import sys
import time
import psutil
from pixcli.processes import terminate_process_tree


def test_terminates_child_processes_before_return():
    child_code = 'import time; time.sleep(120)'
    parent_code = 'import subprocess,sys,time; subprocess.Popen([sys.executable,"-c",sys.argv[1]]); time.sleep(120)'
    proc = subprocess.Popen([sys.executable, '-c', parent_code, child_code])
    try:
        parent = psutil.Process(proc.pid)
        deadline = time.monotonic() + 10
        descendants = []
        while time.monotonic() < deadline:
            descendants = parent.children(recursive=True)
            if descendants:
                break
            time.sleep(.05)
        assert descendants, 'child did not start'
        terminate_process_tree(proc)
        assert proc.poll() is not None
        assert all(not p.is_running() for p in descendants)
    finally:
        if proc.poll() is None:
            proc.kill()
            proc.wait()

def test_terminates_previously_observed_child_after_parent_exit():
    child_code = 'import time; time.sleep(120)'
    parent_code = 'import subprocess,sys; p=subprocess.Popen([sys.executable,"-c",sys.argv[1]]); print(p.pid,flush=True)'
    proc = subprocess.Popen([sys.executable, '-c', parent_code, child_code], stdout=subprocess.PIPE, text=True)
    child = None
    try:
        child = psutil.Process(int(proc.stdout.readline().strip()))
        proc.wait(timeout=10)
        terminate_process_tree(proc, tracked=[child])
        assert not child.is_running()
    finally:
        if child is not None and child.is_running():
            child.kill()
        if proc.poll() is None:
            proc.kill()
            proc.wait()
