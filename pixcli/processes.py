"""Manage the local backend and any interpreter-launcher child processes."""
from __future__ import annotations

import subprocess
import psutil


def terminate_process_tree(proc: subprocess.Popen, timeout: float = 10, *, tracked=()) -> None:
    """Stop descendants before the launcher, then wait for every known process."""
    try:
        parent = psutil.Process(proc.pid)
    except psutil.NoSuchProcess:
        parent = None
    seeds = ([parent] if parent is not None else []) + list(tracked)
    targets = []
    seen = set()
    for seed in seeds:
        try:
            candidates = list(reversed(seed.children(recursive=True))) + [seed]
        except psutil.NoSuchProcess:
            candidates = []
        for process in candidates:
            if process not in seen:
                targets.append(process)
                seen.add(process)
    for process in targets:
        try:
            process.terminate()
        except psutil.NoSuchProcess:
            pass
    _, alive = psutil.wait_procs(targets, timeout=timeout)
    for process in alive:
        try:
            process.kill()
        except psutil.NoSuchProcess:
            pass
    if alive:
        _, alive = psutil.wait_procs(alive, timeout=timeout)
        if alive:
            raise RuntimeError('Backend processes did not stop: ' + ', '.join(str(p.pid) for p in alive))
    try:
        proc.wait(timeout=timeout)
    except subprocess.TimeoutExpired:
        proc.kill()
        proc.wait(timeout=timeout)


def tree_private_bytes(process: psutil.Process) -> int:
    """Count the interpreter launcher and its descendants, without double counting."""
    try:
        processes = [process, *process.children(recursive=True)]
    except psutil.NoSuchProcess:
        return 0
    total = 0
    for member in processes:
        try:
            memory = member.memory_info()
            total += getattr(memory, 'private', memory.rss)
        except psutil.NoSuchProcess:
            continue
    return total
