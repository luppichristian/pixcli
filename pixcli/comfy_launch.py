"""Start the pinned ComfyUI with a bounded PyTorch CUDA allocator."""
from __future__ import annotations


def configure_cuda_budget(torch, budget_gib):
    from pixcli.comfy_backend import _vram_reserve_flags
    total = int(torch.cuda.get_device_properties(0).total_memory)
    if total <= 0:
        raise RuntimeError('Cannot determine CUDA device memory for VRAM budget')
    fraction = min(1.0, budget_gib * 1024**3 / total)
    torch.cuda.set_per_process_memory_fraction(fraction, 0)
    observed_fraction = torch.cuda.get_per_process_memory_fraction(0)
    if abs(observed_fraction - fraction) > 1e-6:
        raise RuntimeError('CUDA allocator did not apply the requested VRAM budget')
    return _vram_reserve_flags(total, budget_gib), {
        'allocator_fraction': fraction, 'device_total_bytes': total, 'budget_gib': budget_gib,
    }


def main():
    import json
    import os
    from pixcli.residency import resident_memory_limit

    ram_value = os.environ.get('PIXCLI_RAM_BUDGET_GIB')
    ram_budget = float(ram_value) if ram_value is not None else None
    # Allow some of the process-tree budget for the Windows venv launcher.
    process_budget = max(ram_budget * 0.5, ram_budget - 0.125) if ram_budget is not None else None
    process_limit = os.environ.get('PIXCLI_RAM_PROCESS_LIMIT_BYTES')
    if process_limit is not None:
        if ram_budget is None or int(process_limit) <= 0:
            raise ValueError('A positive process limit requires a RAM budget')
        process_budget = min(process_budget, int(process_limit) / 1024**3)
    with resident_memory_limit(process_budget) as applied:
        if ram_budget is not None:
            print('PIXCLI_RAM_BUDGET ' + json.dumps({'budget_gib': ram_budget,
                  'process_budget_gib': process_budget, 'hard_limit_applied': applied,
                  'pid': os.getpid()}), flush=True)
        _run_backend()


def _run_backend():
    import json
    import os
    import runpy
    import sys
    from pathlib import Path

    value = os.environ.get('PIXCLI_VRAM_BUDGET_GIB')
    sys.argv = sys.argv[1:]
    sys.path[0] = str(Path(sys.argv[0]).resolve().parent)
    if value is None:
        runpy.run_path(sys.argv[0], run_name='__main__')
        return
    budget = float(value)
    if '--reserve-vram' in sys.argv:
        raise ValueError('VRAM reservation is managed by the pixcli budget launcher')
    # ComfyUI caches parsed args when cuda_malloc is imported. Supply a
    # placeholder first, then replace both cached and command-line values.
    sys.argv.extend(('--reserve-vram', '0'))
    import comfy.options
    comfy.options.enable_args_parsing()
    from comfy.cli_args import args
    import cuda_malloc  # noqa: F401 -- sets allocator config before torch import

    # Match the pinned ComfyUI main's device visibility policy before CUDA is
    # initialized, including its Windows default of one visible GPU.
    if (os.name == 'nt' and args.cuda_device is None and args.default_device is None
            and os.environ.get('CUDA_VISIBLE_DEVICES') is None):
        os.environ['CUDA_VISIBLE_DEVICES'] = '0'
    if args.default_device is not None and args.cuda_device != 'all':
        devices = list(range(32))
        devices.remove(args.default_device)
        devices.insert(0, args.default_device)
        visible = ','.join(map(str, devices))
        os.environ['CUDA_VISIBLE_DEVICES'] = visible
        os.environ['HIP_VISIBLE_DEVICES'] = visible
    if args.cuda_device is not None and args.cuda_device != 'all':
        os.environ['CUDA_VISIBLE_DEVICES'] = str(args.cuda_device)
        os.environ['HIP_VISIBLE_DEVICES'] = str(args.cuda_device)
        os.environ['ASCEND_RT_VISIBLE_DEVICES'] = str(args.cuda_device)

    import torch
    flags, report = configure_cuda_budget(torch, budget)
    sys.argv[-1] = flags[1]
    args.reserve_vram = float(flags[1])
    print('PIXCLI_CUDA_BUDGET ' + json.dumps(report), flush=True)
    runpy.run_path(sys.argv[0], run_name='__main__')


if __name__ == '__main__':
    main()
