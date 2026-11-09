"""Bound a job's resident working set where the operating system supports it."""
from contextlib import contextmanager
import math
import sys


def lower_process_residency_limit(pid: int, maximum_bytes: int, *, expected_create_time: float | None = None) -> int | None:
    """Lower another Windows process's hard working-set cap; never raise it.

    The caller must verify the process belongs to its job immediately before
    calling. The handle is closed before returning. Returns the applied cap,
    or None when the platform cannot apply this kind of limit.
    """
    if type(pid) is not int or pid <= 0 or type(maximum_bytes) is not int or maximum_bytes <= 0:
        raise ValueError('A positive process ID and byte limit are required')
    if sys.platform != 'win32':
        return None
    import ctypes
    kernel = ctypes.WinDLL('kernel32', use_last_error=True)
    kernel.OpenProcess.argtypes = (ctypes.c_ulong, ctypes.c_int, ctypes.c_ulong)
    kernel.OpenProcess.restype = ctypes.c_void_p
    kernel.CloseHandle.argtypes = (ctypes.c_void_p,)
    kernel.CloseHandle.restype = ctypes.c_int
    kernel.GetProcessWorkingSetSizeEx.argtypes = (ctypes.c_void_p, ctypes.POINTER(ctypes.c_size_t),
                                                   ctypes.POINTER(ctypes.c_size_t), ctypes.POINTER(ctypes.c_ulong))
    kernel.GetProcessWorkingSetSizeEx.restype = ctypes.c_int
    kernel.SetProcessWorkingSetSizeEx.argtypes = (ctypes.c_void_p, ctypes.c_size_t, ctypes.c_size_t, ctypes.c_ulong)
    kernel.SetProcessWorkingSetSizeEx.restype = ctypes.c_int
    class FileTime(ctypes.Structure):
        _fields_ = [('low', ctypes.c_ulong), ('high', ctypes.c_ulong)]
    kernel.GetProcessTimes.argtypes = (ctypes.c_void_p, *([ctypes.POINTER(FileTime)] * 4))
    kernel.GetProcessTimes.restype = ctypes.c_int
    handle = kernel.OpenProcess(0x0100 | 0x0400, False, pid)  # SET_QUOTA | QUERY_INFORMATION
    if not handle:
        raise ctypes.WinError(ctypes.get_last_error())
    try:
        if expected_create_time is not None:
            created, exited, kernel_time, user_time = (FileTime() for _ in range(4))
            if not kernel.GetProcessTimes(handle, ctypes.byref(created), ctypes.byref(exited),
                                          ctypes.byref(kernel_time), ctypes.byref(user_time)):
                raise ctypes.WinError(ctypes.get_last_error())
            actual = ((created.high << 32) | created.low) / 10_000_000 - 11_644_473_600
            if abs(actual - expected_create_time) > 0.02:
                raise RuntimeError('Backend process identity changed before adjusting its RAM limit')
        minimum, maximum, flags = ctypes.c_size_t(), ctypes.c_size_t(), ctypes.c_ulong()
        if not kernel.GetProcessWorkingSetSizeEx(handle, ctypes.byref(minimum), ctypes.byref(maximum), ctypes.byref(flags)):
            raise ctypes.WinError(ctypes.get_last_error())
        applied = min(maximum_bytes, maximum.value) if flags.value & 0x4 else maximum_bytes
        if not kernel.SetProcessWorkingSetSizeEx(handle, min(minimum.value, applied), applied, (flags.value & 0x3) | 0x4):
            raise ctypes.WinError(ctypes.get_last_error())
        return applied
    finally:
        kernel.CloseHandle(handle)


@contextmanager
def resident_memory_limit(budget_gib: float | None):
    """Temporarily cap this process's Windows working set; yield whether applied.

    File-backed pages may be evicted and faulted back in. This does not limit
    committed virtual memory, GPU allocations, or memory in other processes.
    On other operating systems, callers retain their sampled memory guard.
    """
    if budget_gib is None:
        yield False
        return
    if not math.isfinite(budget_gib) or budget_gib <= 0:
        raise ValueError('RAM budget must be a positive finite number of GiB')
    if sys.platform != 'win32':
        yield False
        return

    import ctypes

    kernel = ctypes.WinDLL('kernel32', use_last_error=True)
    kernel.GetCurrentProcess.restype = ctypes.c_void_p
    kernel.GetProcessWorkingSetSizeEx.argtypes = (
        ctypes.c_void_p, ctypes.POINTER(ctypes.c_size_t),
        ctypes.POINTER(ctypes.c_size_t), ctypes.POINTER(ctypes.c_ulong))
    kernel.GetProcessWorkingSetSizeEx.restype = ctypes.c_int
    kernel.SetProcessWorkingSetSizeEx.argtypes = (
        ctypes.c_void_p, ctypes.c_size_t, ctypes.c_size_t, ctypes.c_ulong)
    kernel.SetProcessWorkingSetSizeEx.restype = ctypes.c_int
    process = kernel.GetCurrentProcess()
    old_minimum, old_maximum, old_flags = ctypes.c_size_t(), ctypes.c_size_t(), ctypes.c_ulong()
    if not kernel.GetProcessWorkingSetSizeEx(process, ctypes.byref(old_minimum),
                                             ctypes.byref(old_maximum), ctypes.byref(old_flags)):
        raise ctypes.WinError(ctypes.get_last_error())
    maximum = int(budget_gib * 1024**3)
    # Never relax a stricter hard limit imposed by an enclosing caller.
    if old_flags.value & 0x4:
        maximum = min(maximum, old_maximum.value)
    minimum = min(old_minimum.value, maximum)
    # Keep the previous minimum policy and enable the hard maximum.
    flags = (old_flags.value & 0x3) | 0x4
    if not kernel.SetProcessWorkingSetSizeEx(process, minimum, maximum, flags):
        raise ctypes.WinError(ctypes.get_last_error())
    try:
        yield True
    finally:
        if not kernel.SetProcessWorkingSetSizeEx(process, old_minimum.value, old_maximum.value,
                                                 old_flags.value):
            raise ctypes.WinError(ctypes.get_last_error())
