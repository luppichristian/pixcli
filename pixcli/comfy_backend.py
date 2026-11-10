"""Experimental ComfyUI graph for the author's two-expert Wan workflow."""
from __future__ import annotations


def _local_opener():
    """Connect directly to the loopback backend even when proxies are configured."""
    import urllib.request
    return urllib.request.build_opener(urllib.request.ProxyHandler({}))


def _complete_log_lines(pending: str, chunk: str) -> tuple[list[str], str]:
    parts = (pending + chunk).splitlines(keepends=True)
    remaining = ''
    if parts and not parts[-1].endswith(('\n', '\r')):
        remaining = parts.pop()
    return [line.rstrip('\r\n') for line in parts], remaining


def _history_or_pending(get, prompt_id: str):
    """Keep polling when a busy local backend temporarily misses a request."""
    import urllib.error
    try:
        return get('/history/' + prompt_id).get(prompt_id)
    except urllib.error.HTTPError:
        raise
    except (urllib.error.URLError, OSError, TimeoutError):
        return None


def _require_cuda_device(stats: dict) -> dict:
    for device in stats.get('devices', []):
        if device.get('type') == 'cuda':
            return device
    raise RuntimeError('The animation backend did not detect a CUDA device. Run pixcli doctor and check the NVIDIA driver; CPU fallback is disabled.')


def _require_backend_device(stats: dict, device: str) -> dict:
    for item in stats.get('devices', []):
        if item.get('type') == device:
            return item
    raise RuntimeError(f'The animation backend did not detect the requested {device} device')


def _resolve_animation_device(requested: str, cuda_available: bool, vram_budget_gib: float | None,
                              preset: str, adapter_device: str) -> str:
    if requested not in ('auto', 'cpu', 'cuda'):
        raise ValueError('animation device must be auto, cpu or cuda')
    chosen = ('cuda' if cuda_available else 'cpu') if requested == 'auto' else requested
    if chosen == 'cuda' and not cuda_available:
        raise RuntimeError('CUDA requested but unavailable in the animation backend')
    if chosen == 'cpu':
        if vram_budget_gib is not None:
            raise ValueError('VRAM budget requires CUDA; no GPU is used in CPU mode')
        if preset != 'low-vram' or adapter_device != 'cpu':
            raise ValueError('CPU animation requires low-vram preset and CPU adapter')
    return chosen


def _backend_cuda_available(python, *, ram_budget_gib=None, min_free_commit_gib=2.0,
                            min_free_physical_gib=2.0) -> bool:
    import subprocess
    try:
        result = _run_python_probe(python, 'import torch; print(int(torch.cuda.is_available()))', timeout=45,
                                   ram_budget_gib=ram_budget_gib, min_free_commit_gib=min_free_commit_gib,
                                   min_free_physical_gib=min_free_physical_gib)
        last = result.stdout.strip().splitlines()[-1]
        if last not in ('0', '1'):
            raise ValueError('unexpected availability response')
        return last == '1'
    except (OSError, subprocess.SubprocessError, IndexError, ValueError) as exc:
        raise RuntimeError(f'Could not detect CUDA in the animation backend: {exc}') from exc


def _backend_preset_flags(preset: str, text_encoder: str, *, device: str = 'cuda') -> tuple[str, ...]:
    modes = {'low-vram': '--lowvram', 'gpu-resident': '--highvram'}
    if preset not in modes:
        raise ValueError('Unsupported animation preset; choose low-vram or gpu-resident')
    if preset == 'gpu-resident' and text_encoder != 'fp8':
        raise ValueError('The gpu-resident preset requires the fp8 text encoder on CPU')
    if device not in ('cuda', 'cpu'):
        raise ValueError('Unsupported animation device')
    return ('--cpu' if device == 'cpu' else modes[preset], '--disable-dynamic-vram', '--disable-pinned-memory', '--cache-none')


def _verify_backend_identity(stats: dict, port: int, user_directory: str, preset: str = 'low-vram', *, reserve_flags=(), device='cuda') -> None:
    args = stats.get('system', {}).get('argv', [])
    pairs = list(zip(args, args[1:]))
    expected_flags = _backend_preset_flags(preset, 'fp8', device=device)
    modes = {'--lowvram', '--highvram', '--normalvram', '--novram', '--cpu', '--gpu-only'}
    if (('--port', str(port)) not in pairs or ('--user-directory', user_directory) not in pairs
            or any(args.count(flag) != 1 for flag in expected_flags)
            or modes.intersection(args) != {expected_flags[0]}
            or (reserve_flags and (args.count('--reserve-vram') != 1 or tuple(reserve_flags) not in pairs))):
        raise RuntimeError('Local backend identity does not match the process started for this job; refusing to submit the prompt.')


def _coverage_complete(reports: list[dict], adapter_device: str = 'cpu') -> bool:
    return (len(reports) == 2 and {r.get('expert') for r in reports} == {'high', 'low'}
            and all(r.get('targets') == 400 and r.get('tensors') == 800 and r.get('unmatched') == 0
                    and r.get('adapter_device') == adapter_device for r in reports))


def _assert_lora_log_clean(content: str) -> None:
    """Reject patches ComfyUI logged and skipped as well as unmatched keys."""
    import re
    for line in content.splitlines():
        if re.search(r'\bERROR\s+lora\s', line, flags=re.IGNORECASE):
            raise RuntimeError(f'ComfyUI failed to apply a LoRA patch: {line.strip()}')
        if 'lora key not loaded:' in line or 'NOT LOADED ' in line:
            raise RuntimeError('ComfyUI reported unmatched LoRA keys; refusing to report success')


def _device_memory_scope(device: str, sample: int | None) -> str | None:
    return ('physical GPU index 0, device-wide, sampled approximately every two seconds'
            if device == 'cuda' and sample is not None else None)


def _gpu_memory_used(*, min_free_commit_gib=2.0, min_free_physical_gib=2.0) -> int | None:
    """Sample device-wide GPU usage without creating another CUDA context."""
    import subprocess
    try:
        value = _run_supervised_probe(
            ['nvidia-smi', '--id=0', '--query-gpu=memory.used', '--format=csv,noheader,nounits'],
            timeout=2, min_free_commit_gib=min_free_commit_gib,
            min_free_physical_gib=min_free_physical_gib).stdout
        return int(value.strip()) * 1024**2
    except (OSError, ValueError, subprocess.SubprocessError):
        return None


def _optional_gpu_memory_used(device: str, ram_budget_gib: float | None,
                              min_free_commit_gib: float, min_free_physical_gib: float) -> int | None:
    # Device-wide telemetry is optional; don't let its transient subprocess
    # consume a tight command RAM budget or distort the residency measurement.
    if device != 'cuda' or ram_budget_gib is not None:
        return None
    return _gpu_memory_used(min_free_commit_gib=min_free_commit_gib,
                            min_free_physical_gib=min_free_physical_gib)


def validate_memory_budget(max_private_gib: float | None, min_free_commit_gib: float, min_free_physical_gib: float,
                           ram_budget_gib: float | None = None, vram_budget_gib: float | None = None) -> None:
    import math
    if max_private_gib is not None and (not math.isfinite(max_private_gib) or max_private_gib <= 0):
        raise ValueError('maximum process private memory must be a finite positive GiB value')
    if any(not math.isfinite(value) or value < 1 for value in (min_free_commit_gib, min_free_physical_gib)):
        raise ValueError('minimum available commit and physical memory must each be at least 1 GiB')
    if any(value is not None and (not math.isfinite(value) or value <= 0)
           for value in (ram_budget_gib, vram_budget_gib)):
        raise ValueError('RAM and VRAM memory budgets must be finite positive GiB values')


def _vram_reserve_flags(total_bytes: int, budget_gib: float) -> tuple[str, str]:
    # ComfyUI uses this as a reserve during weight placement, not a hard cap.
    reserve = max(0.0, total_bytes / 1024**3 - budget_gib)
    return ('--reserve-vram', format(reserve, '.12g'))


def _process_tree_resident_bytes(process) -> int:
    import psutil
    try:
        members = [process, *process.children(recursive=True)]
        total = 0
        for member in members:
            try:
                total += member.memory_info().rss
            except psutil.NoSuchProcess:
                continue
        return total
    except psutil.Error as exc:
        raise RuntimeError('RAM memory measurement unavailable for backend process tree') from exc


def _backend_ram_limit_bytes(budget_gib: float, parent_resident_bytes: int) -> int:
    """Reserve command-parent and launcher headroom before starting ComfyUI."""
    available = int(budget_gib * 1024**3) - parent_resident_bytes - 128 * 1024**2
    if available <= 0:
        raise MemoryError('RAM budget is too small for the command parent and backend launcher')
    return available


def _command_resident_sample(parent, backend_pid: int | None) -> tuple[int, int | None, float | None]:
    """Sum each command process once and identify the reported backend child."""
    import psutil
    try:
        members = [parent, *parent.children(recursive=True)]
        total, backend_resident, backend_created = 0, None, None
        seen = set()
        for member in members:
            if member.pid in seen:
                continue
            seen.add(member.pid)
            try:
                resident = member.memory_info().rss
                total += resident
                if member.pid == backend_pid:
                    backend_resident = resident
                    backend_created = member.create_time()
            except psutil.NoSuchProcess:
                continue
        return total, backend_resident, backend_created
    except psutil.Error as exc:
        raise RuntimeError('RAM memory measurement unavailable for command process tree') from exc


def _fit_command_residency(parent, report, budget_gib, expected_created, current_limit):
    """Account for the parent and trim the verified backend if helpers grow."""
    pid = report.get('pid') if report else None
    if report and (type(pid) is not int or pid <= 0):
        raise RuntimeError('Invalid backend RAM-budget process identity')
    total, backend_resident, created = _command_resident_sample(parent, pid)
    if report:
        if backend_resident is None or created is None:
            raise RuntimeError('Backend RAM-budget process identity is not in the command tree')
        if expected_created is not None and abs(created - expected_created) > 0.02:
            raise RuntimeError('Backend RAM-budget process identity changed')
        expected_created = created
        if report.get('hard_limit_applied'):
            from .residency import lower_process_residency_limit
            if current_limit is None:
                current_limit = int(report['process_budget_gib'] * 1024**3)
            desired = int(budget_gib * 1024**3) - (total - backend_resident) - 32 * 1024**2
            if desired <= 0:
                raise MemoryError('RAM budget is too small for the command parent and backend helpers')
            if desired < current_limit:
                current_limit = lower_process_residency_limit(pid, desired, expected_create_time=created)
                total, backend_resident, observed_created = _command_resident_sample(parent, pid)
                if backend_resident is None or observed_created != created:
                    raise RuntimeError('Backend RAM-budget process identity changed')
    _check_sampled_budgets(total, None, budget_gib, None)
    return total, expected_created, current_limit


def _check_sampled_budgets(ram_bytes, vram_bytes, ram_budget_gib, vram_budget_gib) -> None:
    for label, sample, budget in (('RAM', ram_bytes, ram_budget_gib), ('VRAM', vram_bytes, vram_budget_gib)):
        if budget is None:
            continue
        if sample is None:
            raise RuntimeError(f'{label} memory measurement unavailable for this backend; cannot enforce requested budget')
        if sample > budget * 1024**3:
            raise MemoryError(f'{label} memory budget exceeded: sampled {sample/1024**3:.2f} GiB, budget {budget:g} GiB')


def _effective_private_budget(max_private_gib: float | None, ram_budget_gib: float | None) -> float | None:
    if max_private_gib is None and ram_budget_gib is None:
        return 14.0
    return max_private_gib


def _validate_animation_headroom(physical: int, commit: int | None, min_free_commit_gib: float,
                                 min_free_physical_gib: float, weight_storage: str) -> None:
    required_physical = max(4.0, min_free_physical_gib) if weight_storage == 'normal' else min_free_physical_gib
    required_commit = max(6.0, min_free_commit_gib) if weight_storage == 'normal' else min_free_commit_gib
    if physical < required_physical * 1024**3:
        raise RuntimeError(f'Less than {required_physical:g} GiB physical RAM is available; close other applications before animation')
    if commit is not None and commit < required_commit * 1024**3:
        raise RuntimeError(f'Only {commit / 1024**3:.1f} GiB commit is available; animation requires at least {required_commit:g} GiB free before startup')


def memory_guard_violation(private: int, physical: int, commit: int | None, max_private_gib: float | None,
                           min_free_commit_gib: float, min_free_physical_gib: float) -> str | None:
    gib = 1024**3
    if max_private_gib is not None and private > max_private_gib * gib:
        return 'process private memory'
    if physical < min_free_physical_gib * gib:
        return 'physical memory'
    if commit is not None and commit < min_free_commit_gib * gib:
        return 'commit memory'
    return None


def _run_supervised_probe(command, *, timeout, ram_budget_gib=None,
                          min_free_commit_gib=2.0, min_free_physical_gib=2.0, env=None):
    """Run a short setup probe while monitoring the complete command tree."""
    import subprocess
    import tempfile
    import time
    import psutil
    from .processes import terminate_process_tree

    parent = psutil.Process()
    def guard():
        if ram_budget_gib is not None:
            _check_sampled_budgets(_process_tree_resident_bytes(parent), None, ram_budget_gib, None)
        violation = memory_guard_violation(0, psutil.virtual_memory().available, _free_commit_bytes(), None,
                                            min_free_commit_gib, min_free_physical_gib)
        if violation:
            raise MemoryError(f'Animation stopped by memory guard ({violation}) during backend checks')

    guard()
    with tempfile.TemporaryFile(mode='w+', encoding='utf-8') as stdout, tempfile.TemporaryFile(mode='w+', encoding='utf-8') as stderr:
        proc = subprocess.Popen(command, stdout=stdout, stderr=stderr, env=env,
                                creationflags=getattr(subprocess, 'CREATE_NO_WINDOW', 0))
        known = set()
        started = time.monotonic()
        try:
            while True:
                try:
                    known.update(psutil.Process(proc.pid).children(recursive=True))
                except psutil.NoSuchProcess:
                    pass
                guard()
                status = proc.poll()
                if status is not None:
                    break
                if time.monotonic() - started > timeout:
                    raise subprocess.TimeoutExpired(command, timeout)
                time.sleep(0.05)
        except BaseException:
            terminate_process_tree(proc, tracked=known)
            raise
        stdout.seek(0)
        stderr.seek(0)
        out, err = stdout.read(), stderr.read()
        if status:
            raise subprocess.CalledProcessError(status, command, output=out, stderr=err)
        return subprocess.CompletedProcess(command, status, stdout=out, stderr=err)


def _run_python_probe(python, code, *, timeout, ram_budget_gib=None,
                      min_free_commit_gib=2.0, min_free_physical_gib=2.0):
    """Install a worker working-set limit before running Python probe code."""
    import os
    from pathlib import Path
    import psutil
    env = os.environ.copy()
    root = str(Path(__file__).resolve().parent.parent)
    env['PYTHONPATH'] = root + (os.pathsep + env['PYTHONPATH'] if env.get('PYTHONPATH') else '')
    if ram_budget_gib is None:
        command = [str(python), '-s', '-c', code]
    else:
        limit = _backend_ram_limit_bytes(ram_budget_gib, _process_tree_resident_bytes(psutil.Process()))
        bootstrap = ('import sys\nfrom pixcli.residency import resident_memory_limit\n'
                     'with resident_memory_limit(float(sys.argv[1])):\n'
                     ' exec(sys.argv[2], {"__name__": "__main__"})\n')
        command = [str(python), '-s', '-c', bootstrap, str(limit / 1024**3), code]
    return _run_supervised_probe(command, timeout=timeout, ram_budget_gib=ram_budget_gib,
                                 min_free_commit_gib=min_free_commit_gib,
                                 min_free_physical_gib=min_free_physical_gib, env=env)


def _oom_explanation(message: str, width: int, height: int, frames: int, *,
                     seed: int | None = None, guidance: float | None = None,
                     lora_strength: float | None = None, text_encoder: str | None = None,
                     expert_quant: str | None = None, preset: str = 'low-vram', adapter_device: str = 'cpu') -> str | None:
    lower = message.lower()
    if 'out of memory' not in lower and 'outofmemoryerror' not in lower:
        return None
    area = width * height
    if area > 384 * 384:
        advice = 'retry with -inference-width 384 -inference-height 384 -frames 5'
    elif area > 256 * 256:
        advice = 'retry with -inference-width 256 -inference-height 256 -frames 5'
    elif frames > 5:
        advice = 'retry with -frames 5'
    else:
        advice = 'reduce the inference dimensions if possible'
    return (f'Animation ran out of memory at {width}x{height}, {frames} frames, '
            f'4 steps, seed={seed}, guidance={guidance}, pixel LoRA strength={lora_strength}, '
            f'backend=comfyui-gguf, preset={preset}, text encoder={text_encoder}, expert quantization={expert_quant}, adapter device={adapter_device}. '
            f'Close other GPU applications and {advice}. No automatic fallback was used. '
            f'Backend error: {message}')


def validate_animation(width: int, height: int, frames: int) -> None:
    if any(type(value) is not int for value in (width, height, frames)):
        raise ValueError('Video inference dimensions and frame count must be integers')
    if width < 16 or height < 16 or width % 16 or height % 16:
        raise ValueError('Video inference width and height must be positive multiples of 16')
    if width > 832 or height > 832 or width * height > 832 * 480:
        raise ValueError('The experimental low-VRAM preset allows at most 832x480 pixels of inference area')
    if frames < 5 or frames > 81 or (frames - 1) % 4:
        raise ValueError('Video frame count must be 4n+1 between 5 and 81')
    latent_bytes = 4 * 16 * ((frames - 1) // 4 + 1) * (height // 8) * (width // 8)
    conditioning_bytes = frames * height * width * 3 * 4
    if latent_bytes + conditioning_bytes > 512 * 1024**2:
        raise ValueError('The latent and image conditioning allocation exceeds the 512 MiB limit')


def _text_encoder_choice(text_encoder: str) -> tuple[str, str]:
    choices = {
        'fp8': ('wan_text', 'umt5_xxl_fp8_e4m3fn_scaled.safetensors'),
        'q3': ('wan_text_q3', 'umt5-xxl-encoder-Q3_K_S.gguf'),
    }
    try:
        return choices[text_encoder]
    except (KeyError, TypeError) as exc:
        raise ValueError('Unsupported text encoder; choose fp8 or q3') from exc


def _expert_choice(expert_quant: str) -> tuple[str, str, str]:
    choices = {'q3': ('wan_high', 'wan_low', 'Q3_K_S'),
               'q2': ('wan_high_q2', 'wan_low_q2', 'Q2_K')}
    try:
        return choices[expert_quant]
    except (KeyError, TypeError) as exc:
        raise ValueError('Unsupported expert quantization; choose q3 or q2') from exc


def validate_model_locations(paths: dict, *, text_encoder: str = 'fp8', expert_quant: str = 'q3') -> None:
    """Reject same-name files in another selected directory in the same category."""
    from pathlib import Path
    text_key, text_filename = _text_encoder_choice(text_encoder)
    high_key, low_key, suffix = _expert_choice(expert_quant)
    groups = ((high_key, low_key), (text_key,), ('wan_vae',),
              ('wan_distill_high', 'wan_distill_low', 'wan_pixel'))
    names = {
        high_key: f'wan2.2_i2v_high_noise_14B_{suffix}.gguf',
        low_key: f'wan2.2_i2v_low_noise_14B_{suffix}.gguf',
        text_key: text_filename,
        'wan_vae': 'wan_2.1_vae.safetensors',
        'wan_distill_high': 'wan2.2_i2v_lightx2v_4steps_lora_v1_high_noise.safetensors',
        'wan_distill_low': 'wan2.2_i2v_lightx2v_4steps_lora_v1_low_noise.safetensors',
        'wan_pixel': 'wan2.2_animate_adapter_model.safetensors',
    }
    for group in groups:
        folders = {Path(paths[key]).parent.resolve() for key in group}
        for key in group:
            path = Path(paths[key])
            if path.name != names[key]:
                raise ValueError(f'Unexpected filename for {key}: {path.name}')
            for folder in folders:
                candidate = folder / names[key]
                if candidate.exists() and candidate.resolve() != path.resolve():
                    raise ValueError(f'Model filename shadow detected for {key}: {candidate}')


def validate_ready(marker, lock, node, commits: dict, *, mmap_node=None, chunked_node=None, fp8_node=None) -> None:
    """Require a successful setup marker matching current checked-in artifacts."""
    import hashlib
    import json
    from pathlib import Path
    try:
        data = json.loads(Path(marker).read_text(encoding='utf-8-sig'))
        lock_hash = hashlib.sha256(Path(lock).read_bytes()).hexdigest()
        node_hash = hashlib.sha256(Path(node).read_bytes()).hexdigest()
        mmap_hash = hashlib.sha256(Path(mmap_node).read_bytes()).hexdigest() if mmap_node is not None else None
        chunked_hash = hashlib.sha256(Path(chunked_node).read_bytes()).hexdigest() if chunked_node is not None else None
        fp8_hash = hashlib.sha256(Path(fp8_node).read_bytes()).hexdigest() if fp8_node is not None else None
    except (OSError, ValueError) as exc:
        raise RuntimeError('Animation backend setup is incomplete; run setup.ps1') from exc
    if (data.get('schema_version') != 1 or data.get('lock_sha256') != lock_hash
            or data.get('node_sha256') != node_hash or data.get('commits') != commits
            or (mmap_node is not None and data.get('mmap_node_sha256') != mmap_hash)
            or (chunked_node is not None and data.get('chunked_node_sha256') != chunked_hash)
            or (fp8_node is not None and data.get('fp8_node_sha256') != fp8_hash)):
        raise RuntimeError('Animation backend setup drift detected; rerun setup.ps1')


def build_graph(image_name: str, prompt: str, negative: str, seed: int,
                width: int, height: int, frames: int, guidance: float, lora_strength: float = 1.0,
                text_encoder: str = 'fp8', expert_quant: str = 'q3', adapter_device: str = 'cpu', weight_storage: str = 'normal') -> dict:
    validate_animation(width, height, frames)
    _, text_filename = _text_encoder_choice(text_encoder)
    _, _, expert_suffix = _expert_choice(expert_quant)
    if adapter_device not in ('cpu', 'cuda'):
        raise ValueError('Unsupported adapter device; choose cpu or cuda')
    if weight_storage not in ('normal', 'mmap'):
        raise ValueError('Unsupported weight storage; choose normal or mmap')
    if not prompt.strip() or not 0 <= seed <= (2**64 - 1) or not 0 <= guidance <= 100:
        raise ValueError('Invalid prompt, seed or guidance')
    if not 0.01 <= lora_strength <= 2.0:
        raise ValueError('Pixel LoRA strength must be between 0.01 and 2')
    def node(kind, **inputs):
        return {'class_type': kind, 'inputs': inputs}
    result = {
        'image': node('LoadImage', image=image_name),
        'text': (node('PixcliMappedFP8UMT5Loader' if weight_storage == 'mmap' else 'CLIPLoader', clip_name=text_filename, type='wan', device='cpu') if text_encoder == 'fp8'
                 else node('PixcliMappedUMT5Loader' if weight_storage == 'mmap' else 'CLIPLoaderGGUF', clip_name=text_filename, type='wan')),
        'positive': node('CLIPTextEncode', clip=['text', 0], text=prompt),
        'negative': node('CLIPTextEncode', clip=['text', 0], text=negative),
        'vae': node('VAELoader', vae_name='wan_2.1_vae.safetensors'),
        'condition': node('PainterI2V', positive=['positive', 0], negative=['negative', 0],
                          vae=['vae', 0], width=width, height=height, length=frames, batch_size=1,
                          motion_amplitude=1.1, start_image=['image', 0]),
    }
    for expert in ('high', 'low'):
        result[expert + '_model'] = node('PixcliMappedGGUFLoader' if weight_storage == 'mmap' else 'UnetLoaderGGUF', unet_name=f'wan2.2_i2v_{expert}_noise_14B_{expert_suffix}.gguf')
        result[expert + '_distill'] = node('PixcliMappedDistillLoRA' if weight_storage == 'mmap' else 'LoraLoaderModelOnly', model=[expert + '_model', 0],
                                           lora_name=f'wan2.2_i2v_lightx2v_4steps_lora_v1_{expert}_noise.safetensors', strength_model=1.0)
        result[expert + '_pixel'] = node('PixcliCheckedPixelLoRA', model=[expert + '_distill', 0],
                                         lora_name='wan2.2_animate_adapter_model.safetensors', strength=lora_strength, expert=expert,
                                         adapter_device=adapter_device, weight_storage=weight_storage)
        result[expert + '_sampling'] = node('ModelSamplingSD3', model=[expert + '_pixel', 0], shift=5.0)
    common = dict(noise_seed=seed, steps=4, cfg=guidance, sampler_name='ddim', scheduler='simple',
                  positive=['condition', 0], negative=['condition', 1])
    result['high_sample'] = node('KSamplerAdvanced', model=['high_sampling', 0], add_noise='enable',
                                 latent_image=['condition', 2], start_at_step=0, end_at_step=2,
                                 return_with_leftover_noise='enable', **common)
    result['low_sample'] = node('KSamplerAdvanced', model=['low_sampling', 0], add_noise='disable',
                                latent_image=['high_sample', 0], start_at_step=2, end_at_step=4,
                                return_with_leftover_noise='disable', **common)
    result['decode'] = node('VAEDecode', samples=['low_sample', 0], vae=['vae', 0])
    result['save'] = node('SaveImage', images=['decode', 0], filename_prefix='pixcli_frame')
    return result


def _free_commit_bytes() -> int | None:
    """Available commit on Windows; return None on other platforms."""
    import ctypes
    import sys
    if sys.platform != 'win32':
        return None
    class MemoryStatus(ctypes.Structure):
        _fields_ = [('length', ctypes.c_ulong), ('load', ctypes.c_ulong),
                    ('total_phys', ctypes.c_ulonglong), ('avail_phys', ctypes.c_ulonglong),
                    ('total_page', ctypes.c_ulonglong), ('avail_page', ctypes.c_ulonglong),
                    ('total_virtual', ctypes.c_ulonglong), ('avail_virtual', ctypes.c_ulonglong),
                    ('avail_extended', ctypes.c_ulonglong)]
    status = MemoryStatus()
    status.length = ctypes.sizeof(status)
    if not ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(status)):
        return None
    return int(status.avail_page)


def run_comfy(image_path, prompt: str, negative: str, seed: int, width: int, height: int,
              frames: int, guidance: float, model_paths: dict, *,
              backend_dir=None, background=(255, 255, 255), lora_strength=1.0,
              timeout_seconds=None, max_private_gib=None, min_free_commit_gib=2.0, min_free_physical_gib=2.0,
              progress=None, text_encoder='fp8', expert_quant='q3', preset='low-vram', adapter_device='cpu', weight_storage='normal',
              ram_budget_gib=None, vram_budget_gib=None, device='auto'):
    """Run the experimental pinned local ComfyUI graph; return frames and audit.

    It uses a fresh localhost-only process and temporary directories. The resource
    guard aborts on low available memory or high process private memory, but a
    polling guard cannot guarantee prevention of every transient allocation spike.
    """
    import json
    import hashlib
    import os
    from pathlib import Path
    import shutil
    import socket
    import subprocess
    import tempfile
    import time
    import urllib.error
    import urllib.request
    import psutil
    from PIL import Image, ImageOps
    from .processes import terminate_process_tree, tree_private_bytes

    _backend_preset_flags(preset, text_encoder, device='cpu' if device == 'cpu' else 'cuda')
    if device == 'cpu':
        _resolve_animation_device(device, False, vram_budget_gib, preset, adapter_device)
    elif device not in ('auto', 'cuda'):
        raise ValueError('animation device must be auto, cpu or cuda')
    validate_memory_budget(max_private_gib, min_free_commit_gib, min_free_physical_gib, ram_budget_gib, vram_budget_gib)
    max_private_gib = _effective_private_budget(max_private_gib, ram_budget_gib)
    if vram_budget_gib is not None and preset != 'low-vram':
        raise ValueError('VRAM memory budget requires the low-vram preset')
    if adapter_device == 'cuda' and preset != 'gpu-resident':
        raise ValueError('CUDA adapter device requires the gpu-resident preset')
    graph = build_graph('source.png', prompt, negative, seed, width, height, frames, guidance, lora_strength, text_encoder, expert_quant, adapter_device, weight_storage)
    text_key, _ = _text_encoder_choice(text_encoder)
    high_key, low_key, _ = _expert_choice(expert_quant)
    if timeout_seconds is not None:
        import math
        if not math.isfinite(timeout_seconds) or timeout_seconds <= 0:
            raise ValueError('animation timeout must be a positive finite number of seconds')
    required = (high_key, low_key, text_key, 'wan_vae', 'wan_distill_high', 'wan_distill_low', 'wan_pixel')
    missing = [key for key in required if key not in model_paths or not Path(model_paths[key]).is_file()]
    if missing:
        from .models import animation_profile
        profile = animation_profile(text_encoder, expert_quant)
        raise FileNotFoundError(f'Missing animation models: {missing}. Run pixcli models download -profile {profile}')
    validate_model_locations(model_paths, text_encoder=text_encoder, expert_quant=expert_quant)
    _validate_animation_headroom(psutil.virtual_memory().available, _free_commit_bytes(),
                                 min_free_commit_gib, min_free_physical_gib, weight_storage)
    root = Path(backend_dir or Path(__file__).resolve().parent.parent / '.backend')
    comfy = root / 'ComfyUI'
    py = root / '.venv' / ('Scripts/python.exe' if os.name == 'nt' else 'bin/python')
    if not py.is_file() or not (comfy / 'main.py').is_file():
        raise RuntimeError('Local ComfyUI backend is not installed; run the animation backend setup script')
    cuda_available = (_backend_cuda_available(py, ram_budget_gib=ram_budget_gib,
                                               min_free_commit_gib=min_free_commit_gib,
                                               min_free_physical_gib=min_free_physical_gib)
                      if device != 'cpu' else False)
    execution_device = _resolve_animation_device(device, cuda_available, vram_budget_gib, preset, adapter_device)
    preset_flags = _backend_preset_flags(preset, text_encoder, device=execution_device)
    if (comfy / 'extra_model_paths.yaml').exists():
        raise RuntimeError('Unexpected ComfyUI extra_model_paths.yaml could shadow pinned models; remove it from the isolated backend')
    pins = {
        'ComfyUI': (comfy, '63e5bf594339f676b4b15e072529873e3448409e'),
        'ComfyUI-GGUF': (comfy/'custom_nodes'/'ComfyUI-GGUF', '6ea2651e7df66d7585f6ffee804b20e92fb38b8a'),
        'ComfyUI-PainterI2V': (comfy/'custom_nodes'/'ComfyUI-PainterI2V', '714044a6536ad37293f0658e247c5c6be1622745'),
    }
    for name, (checkout, expected) in pins.items():
        try:
            actual = _run_supervised_probe(['git', '-C', str(checkout), 'rev-parse', 'HEAD'], timeout=10,
                                           ram_budget_gib=ram_budget_gib, min_free_commit_gib=min_free_commit_gib,
                                           min_free_physical_gib=min_free_physical_gib).stdout.strip()
            dirty = _run_supervised_probe(['git', '-C', str(checkout), 'status', '--porcelain', '--untracked-files=no'], timeout=10,
                                          ram_budget_gib=ram_budget_gib, min_free_commit_gib=min_free_commit_gib,
                                          min_free_physical_gib=min_free_physical_gib).stdout.strip()
        except (OSError, subprocess.SubprocessError) as exc:
            raise RuntimeError(f'Cannot inspect pinned backend checkout {name}: {exc}') from exc
        if actual != expected:
            raise RuntimeError(f'{name} is at {actual}, expected {expected}; rerun backend setup')
        if dirty:
            raise RuntimeError(f'{name} has modified tracked files; refusing an unpinned backend')
    source_node = Path(__file__).resolve().parent/'comfy_custom_nodes'/'pixcli_wan'/'__init__.py'
    installed_node = comfy/'custom_nodes'/'pixcli_wan'/'__init__.py'
    source_mmap_node = source_node.with_name('mmap_loader.py')
    installed_mmap_node = installed_node.with_name('mmap_loader.py')
    source_chunked_node = source_node.with_name('chunked_text.py')
    installed_chunked_node = installed_node.with_name('chunked_text.py')
    source_fp8_node = source_node.with_name('fp8_text.py')
    installed_fp8_node = installed_node.with_name('fp8_text.py')
    validate_ready(root/'ready.json', Path(__file__).resolve().parent.parent/'backend'/'requirements.lock',
                   source_node, {name: revision for name, (_, revision) in pins.items()}, mmap_node=source_mmap_node,
                   chunked_node=source_chunked_node, fp8_node=source_fp8_node)
    if not installed_node.is_file() or hashlib.sha256(installed_node.read_bytes()).digest() != hashlib.sha256(source_node.read_bytes()).digest():
        raise RuntimeError('The checked LoRA loader is missing or out of date; rerun backend setup')
    if not installed_mmap_node.is_file() or hashlib.sha256(installed_mmap_node.read_bytes()).digest() != hashlib.sha256(source_mmap_node.read_bytes()).digest():
        raise RuntimeError('The mapped GGUF loader is missing or out of date; rerun backend setup')
    if not installed_chunked_node.is_file() or hashlib.sha256(installed_chunked_node.read_bytes()).digest() != hashlib.sha256(source_chunked_node.read_bytes()).digest():
        raise RuntimeError('The mapped text loader is missing or out of date; rerun backend setup')
    if not installed_fp8_node.is_file() or hashlib.sha256(installed_fp8_node.read_bytes()).digest() != hashlib.sha256(source_fp8_node.read_bytes()).digest():
        raise RuntimeError('The mapped FP8 text loader is missing or out of date; rerun backend setup')
    version_script = ("import importlib.metadata as m,json; "
                      "names=['torch','torchvision','gguf','transformers','safetensors','comfyui-frontend-package']; "
                      "print(json.dumps({n:m.version(n) for n in names}))")
    try:
        version_run = _run_python_probe(py, version_script, timeout=30, ram_budget_gib=ram_budget_gib,
                                        min_free_commit_gib=min_free_commit_gib, min_free_physical_gib=min_free_physical_gib)
        versions = json.loads(version_run.stdout)
    except (OSError, subprocess.SubprocessError, ValueError) as exc:
        raise RuntimeError(f'Backend dependency inventory failed; rerun backend setup: {exc}') from exc
    info = progress or (lambda message: None)
    with tempfile.TemporaryDirectory(prefix='pixcli-wan-') as tmp:
        work = Path(tmp)
        for name in ('input', 'output', 'temp', 'user', 'models'):
            (work / name).mkdir()
        with Image.open(image_path) as image:
            image = image.convert('RGBA')
            canvas = Image.new('RGBA', image.size, (*background, 255))
            image = Image.alpha_composite(canvas, image).convert('RGB')
            image = ImageOps.contain(image, (width, height), method=Image.Resampling.NEAREST)
            canvas = Image.new('RGB', (width, height), background)
            canvas.paste(image, ((width-image.width)//2, (height-image.height)//2))
            canvas.save(work / 'input' / 'source.png')
        # JSON is valid YAML. Multiple paths are newline-delimited in ComfyUI.
        cfg = {'pixcli': {
            'diffusion_models': '\n'.join(sorted({str(Path(model_paths[k]).parent.resolve()) for k in (high_key, low_key)})),
            'text_encoders': str(Path(model_paths[text_key]).parent.resolve()),
            'vae': str(Path(model_paths['wan_vae']).parent.resolve()),
            'loras': '\n'.join(sorted({str(Path(model_paths[k]).parent.resolve()) for k in ('wan_distill_high', 'wan_distill_low', 'wan_pixel')})),
        }}
        config = work / 'extra_model_paths.yaml'
        config.write_text(json.dumps(cfg), encoding='utf-8')
        with socket.socket() as sock:
            sock.bind(('127.0.0.1', 0))
            port = sock.getsockname()[1]
        url = f'http://127.0.0.1:{port}'
        opener = _local_opener()
        launch = [str(py), '-s']
        if vram_budget_gib is not None or ram_budget_gib is not None:
            launch.append(str(Path(__file__).with_name('comfy_launch.py')))
        command = [*launch, str(comfy / 'main.py'), '--listen', '127.0.0.1', '--port', str(port),
                   '--offline', '--disable-auto-launch', *preset_flags,
                   '--disable-all-custom-nodes', '--whitelist-custom-nodes', 'ComfyUI-GGUF', 'ComfyUI-PainterI2V', 'pixcli_wan',
                   '--extra-model-paths-config', str(config), '--input-directory', str(work/'input'),
                   '--output-directory', str(work/'output'), '--temp-directory', str(work/'temp'),
                   '--user-directory', str(work/'user'), '--models-directory', str(work/'models')]
        env = os.environ.copy()
        env.pop('PIXCLI_RAM_BUDGET_GIB', None)
        env.pop('PIXCLI_RAM_PROCESS_LIMIT_BYTES', None)
        env.pop('PIXCLI_VRAM_BUDGET_GIB', None)
        env.update({'HF_HUB_OFFLINE': '1', 'TRANSFORMERS_OFFLINE': '1', 'PYTHONPATH': str(Path(__file__).resolve().parent.parent)})
        if vram_budget_gib is not None:
            env['PIXCLI_VRAM_BUDGET_GIB'] = str(vram_budget_gib)
        if ram_budget_gib is not None:
            env['PIXCLI_RAM_BUDGET_GIB'] = str(ram_budget_gib)
            env['PIXCLI_RAM_PROCESS_LIMIT_BYTES'] = str(
                _backend_ram_limit_bytes(ram_budget_gib, _process_tree_resident_bytes(psutil.Process())))
        flags = getattr(subprocess, 'CREATE_NO_WINDOW', 0)
        logfile = work / 'comfy.log'
        started = time.monotonic()
        with logfile.open('w+', encoding='utf-8') as log:
            proc = subprocess.Popen(command, cwd=comfy, env=env, stdout=log, stderr=subprocess.STDOUT, creationflags=flags)
            process = psutil.Process(proc.pid)
            last_offset = 0
            pending_line = ''
            peak_private = 0
            peak_resident = 0
            budget_report = None
            ram_budget_report = None
            command_process = psutil.Process()
            backend_created = None
            backend_limit = None
            known_children = set()
            peak_device_used = _optional_gpu_memory_used(execution_device, ram_budget_gib,
                                                        min_free_commit_gib, min_free_physical_gib)
            last_gpu_sample = time.monotonic()
            def drain_log():
                nonlocal last_offset, pending_line, budget_report, ram_budget_report
                log.flush()
                with logfile.open(encoding='utf-8', errors='replace') as reader:
                    reader.seek(last_offset)
                    text = reader.read()
                    last_offset = reader.tell()
                lines, pending_line = _complete_log_lines(pending_line, text)
                for line in lines:
                    if line.startswith('PIXCLI_CUDA_BUDGET '):
                        budget_report = json.loads(line.removeprefix('PIXCLI_CUDA_BUDGET '))
                    if line.startswith('PIXCLI_RAM_BUDGET '):
                        ram_budget_report = json.loads(line.removeprefix('PIXCLI_RAM_BUDGET '))
                    if 'Executing' in line or 'PIXCLI_LORA_COVERAGE' in line or 'PIXCLI_MEMORY_PHASE' in line or '%' in line:
                        info(line.rstrip())

            def guard():
                nonlocal peak_private, peak_device_used, last_gpu_sample, peak_resident, backend_created, backend_limit
                try:
                    known_children.update(process.children(recursive=True))
                except psutil.Error:
                    pass
                drain_log()
                if proc.poll() is not None:
                    log.flush()
                    raise RuntimeError(f'ComfyUI exited with code {proc.returncode}, preset={preset}; log: {logfile.read_text(encoding="utf-8", errors="replace")[-5000:]}')
                try:
                    private = tree_private_bytes(process)
                    peak_private = max(peak_private, private)
                except psutil.Error:
                    private = 0
                if ram_budget_gib is not None:
                    resident, backend_created, backend_limit = _fit_command_residency(
                        command_process, ram_budget_report, ram_budget_gib, backend_created, backend_limit)
                    peak_resident = max(peak_resident, resident)
                free = psutil.virtual_memory().available
                avail_commit = _free_commit_bytes()
                violation = memory_guard_violation(private, free, avail_commit, max_private_gib, min_free_commit_gib, min_free_physical_gib)
                if violation:
                    raise MemoryError(f'Animation stopped by memory guard ({violation}) at {width}x{height}, {frames} frames, '
                                      f'seed={seed}, guidance={guidance}, pixel LoRA strength={lora_strength}, preset={preset}, text encoder={text_encoder}, expert quantization={expert_quant}, adapter device={adapter_device}: '
                                      f'process private={private/1024**3:.2f} GiB, '
                                      f'physical available={free/1024**3:.2f} GiB, commit available='
                                      f'{avail_commit/1024**3 if avail_commit is not None else "unknown"} GiB. '
                                      f'Limits: private={str(max_private_gib) + " GiB maximum" if max_private_gib is not None else "unlimited"}, physical={min_free_physical_gib:g} GiB minimum, commit={min_free_commit_gib:g} GiB minimum. '
                                      'Close other applications to increase available memory. Smaller dimensions or fewer frames may reduce sampling memory but may not help model loading; no fallback was used.')
                if timeout_seconds is not None and time.monotonic() - started > timeout_seconds:
                    raise TimeoutError('Animation timed out; no fallback was used')
                if peak_device_used is not None and time.monotonic() - last_gpu_sample >= 2:
                    sample = _optional_gpu_memory_used(execution_device, ram_budget_gib,
                                                       min_free_commit_gib, min_free_physical_gib)
                    if sample is not None:
                        peak_device_used = max(peak_device_used, sample)
                    last_gpu_sample = time.monotonic()
            def get(endpoint):
                with opener.open(url + endpoint, timeout=2) as response:
                    return json.load(response)
            try:
                info(f'Starting local ComfyUI animation backend, preset={preset}, adapter device={adapter_device}')
                while True:
                    guard()
                    try:
                        stats = get('/system_stats')
                        reserve_flags = ()
                        if vram_budget_gib is not None:
                            if not budget_report or budget_report.get('budget_gib') != vram_budget_gib:
                                raise RuntimeError('VRAM allocator budget confirmation unavailable from backend')
                            reserve_flags = _vram_reserve_flags(budget_report['device_total_bytes'], vram_budget_gib)
                        if ram_budget_gib is not None and (not ram_budget_report or ram_budget_report.get('budget_gib') != ram_budget_gib):
                            raise RuntimeError('RAM budget confirmation unavailable from backend')
                        _verify_backend_identity(stats, port, str(work / 'user'), preset, reserve_flags=reserve_flags, device=execution_device)
                        backend_device = _require_backend_device(stats, execution_device)
                        break
                    except (urllib.error.URLError, OSError, TimeoutError):
                        if time.monotonic() - started > 120:
                            raise TimeoutError('ComfyUI failed to start within 120 seconds')
                        time.sleep(0.25)
                request = urllib.request.Request(url + '/prompt', data=json.dumps({'prompt': graph}).encode(),
                                                 headers={'Content-Type': 'application/json'}, method='POST')
                try:
                    with opener.open(request, timeout=15) as response:
                        queued = json.load(response)
                except urllib.error.HTTPError as exc:
                    raise RuntimeError(f'ComfyUI rejected workflow: {exc.read().decode(errors="replace")}') from exc
                prompt_id = queued['prompt_id']
                info(f'Animation submitted: {prompt_id}')
                while True:
                    guard()
                    history = _history_or_pending(get, prompt_id)
                    if history:
                        status = history.get('status', {})
                        if status.get('status_str') == 'error':
                            raise RuntimeError(f'Animation failed, adapter device={adapter_device}: {status.get("messages", [])}')
                        if status.get('completed'):
                            break
                    time.sleep(0.5)
                guard()
                log.flush()
                content = logfile.read_text(encoding='utf-8', errors='replace')
                reports = []
                for line in content.splitlines():
                    if 'PIXCLI_LORA_COVERAGE ' in line:
                        reports.append(json.loads(line.split('PIXCLI_LORA_COVERAGE ', 1)[1]))
                if not _coverage_complete(reports, adapter_device):
                    raise RuntimeError('ComfyUI did not report complete pixel LoRA coverage for both experts')
                _assert_lora_log_clean(content)
                records = history['outputs']['save']['images']
                if len(records) != frames:
                    raise RuntimeError(f'Expected {frames} output frames, got {len(records)}')
                output = []
                for record in records:
                    path = (work/'output'/record.get('subfolder', '')/record['filename']).resolve()
                    if not path.is_relative_to((work/'output').resolve()):
                        raise RuntimeError('Unexpected path returned by local backend')
                    with Image.open(path) as frame:
                        output.append(frame.copy())
                return output, {'backend': 'comfyui-gguf-experimental', 'preset': preset, 'text_encoder': text_encoder, 'expert_quant': expert_quant, 'adapter_device': adapter_device, 'weight_storage': weight_storage,
                                'memory_guard_gib': {'max_private': max_private_gib, 'min_free_commit': min_free_commit_gib, 'min_free_physical': min_free_physical_gib},
                                'adapter_coverage': reports, 'cuda_device': backend_device if execution_device == 'cuda' else None,
                                'execution_device': execution_device, 'backend_device': backend_device,
                                'backend_commits': {k: v[1] for k, v in pins.items()},
                                'backend_versions': versions,
                                'latent_bytes': 4 * 16 * ((frames - 1) // 4 + 1) * (height // 8) * (width // 8),
                                'peak_process_private_bytes': peak_private,
                                'memory_budgets': {'ram_gib': ram_budget_gib, 'vram_gib': vram_budget_gib,
                                                   'ram_scope': 'sampled sum of the command parent and descendant resident set sizes; shared pages may be counted more than once',
                                                   'vram_scope': 'backend PyTorch CUDA caching allocator limit; excludes CUDA context and non-PyTorch allocations',
                                                   'ram_peak_bytes': peak_resident if ram_budget_gib is not None else None,
                                                   'ram_residency': ram_budget_report,
                                                   'backend_final_residency_limit_bytes': backend_limit,
                                                   'vram_allocator': budget_report,
                                                   'vram_reserve_flags': list(reserve_flags),
                                                   'limits': 'RAM is sampled and can exceed the budget between checks; VRAM limit applies only to the PyTorch allocator'},
                                'sampled_peak_device_used_bytes': peak_device_used,
                                'device_memory_scope': _device_memory_scope(execution_device, peak_device_used),
                                'elapsed_seconds': time.monotonic()-started, 'fallbacks': []}
            except Exception as exc:
                explanation = _oom_explanation(str(exc), width, height, frames, seed=seed,
                                               guidance=guidance, lora_strength=lora_strength, text_encoder=text_encoder, expert_quant=expert_quant, preset=preset, adapter_device=adapter_device)
                if explanation is not None:
                    raise RuntimeError(explanation) from exc
                raise
            finally:
                try:
                    terminate_process_tree(proc, tracked=known_children)
                finally:
                    drain_log()
