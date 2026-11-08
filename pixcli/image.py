"""Local SDXL + Pixel Art XL image inference and image job output."""
from __future__ import annotations

from dataclasses import asdict, dataclass
import gc
import hashlib
from importlib.metadata import PackageNotFoundError, version
import json
import math
import os
from pathlib import Path
import struct
import sys
import tempfile
import time
import traceback
from typing import Any

from PIL import Image, ImageColor
from .processing import finish_edit, pixelate, prepare_edit


@dataclass
class ImageSettings:
    prompt: str
    out: Path
    negative_prompt: str = ''
    seed: int = 42
    steps: int = 30
    guidance: float = 7.0
    lora_strength: float = 1.0
    inference_width: int = 1024
    inference_height: int = 1024
    pixel_width: int = 128
    pixel_height: int = 128
    input_path: Path | None = None
    mask_path: Path | None = None
    strength: float = 0.3
    background: str = '#ffffff'
    save_original: bool = False
    preview_scale: int = 1
    palette_colors: int | None = None
    overwrite: bool = False


def validate_image(s: ImageSettings) -> None:
    """Reject invalid settings before loading large model files."""
    if not s.prompt.strip():
        raise ValueError('prompt cannot be empty')
    if Path(s.out).suffix.lower() != '.png':
        raise ValueError('output must have a .png extension')
    if not 0 <= s.seed < 2**63:
        raise ValueError('seed must be between 0 and 2^63-1')
    if not 1 <= s.steps <= 150:
        raise ValueError('steps must be between 1 and 150')
    if not math.isfinite(s.guidance) or not 0 <= s.guidance <= 30:
        raise ValueError('guidance must be between 0 and 30')
    if not math.isfinite(s.lora_strength) or not 0 < s.lora_strength <= 3:
        raise ValueError('LoRA strength must be greater than zero and at most 3')
    if any(x < 512 or x > 2048 or x % 8 for x in (s.inference_width, s.inference_height)):
        raise ValueError('inference dimensions must be multiples of 8 between 512 and 2048')
    if min(s.pixel_width, s.pixel_height) < 1 or max(s.pixel_width, s.pixel_height) > 2048:
        raise ValueError('pixel dimensions must be between 1 and 2048')
    if not s.input_path and s.pixel_width * s.inference_height != s.pixel_height * s.inference_width:
        raise ValueError('generation pixel dimensions and inference dimensions must have the same aspect ratio; choose matching sizes to avoid stretching')
    if not 1 <= s.preview_scale <= 16:
        raise ValueError('preview scale must be between 1 and 16')
    if s.palette_colors is not None and not 2 <= s.palette_colors <= 256:
        raise ValueError('palette colors must be between 2 and 256')
    if s.mask_path and not s.input_path:
        raise ValueError('mask requires an input image')
    if s.input_path and (not math.isfinite(s.strength) or not 0 < s.strength <= 1 or int(s.steps * s.strength) < 1):
        raise ValueError('edit strength must be in (0, 1] and produce at least one denoising step')
    if s.input_path and s.palette_colors is not None:
        raise ValueError('palette reduction is unavailable for edits because it changes preserved pixels')
    try:
        rgb = ImageColor.getrgb(s.background)
        if len(rgb) != 3:
            raise ValueError()
    except ValueError as exc:
        raise ValueError('background must be an RGB color such as #ffffff') from exc
    out = Path(s.out)
    targets = [out, out.with_suffix('.json')]
    if s.save_original:
        targets.append(out.with_name(out.stem + '.original.png'))
    if s.preview_scale > 1:
        targets.append(out.with_name(out.stem + '.preview.png'))
    if len({path.resolve() for path in targets}) != len(targets):
        raise ValueError('output paths collide')
    for path in targets:
        if path.exists() and not s.overwrite:
            raise FileExistsError(f'Output already exists: {path}. Use --overwrite to replace it.')
        if not s.overwrite and any(path.resolve() == Path(p).resolve() for p in (s.input_path, s.mask_path) if p):
            raise FileExistsError(f'Output would overwrite an input: {path}. Use --overwrite to replace it.')
    if s.input_path:
        with Image.open(s.input_path) as source:
            source_size = source.size
        if s.mask_path:
            with Image.open(s.mask_path) as mask:
                if mask.size != source_size:
                    raise ValueError('mask dimensions must match the input image')


def _hash_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open('rb') as handle:
        for data in iter(lambda: handle.read(1024 * 1024), b''):
            digest.update(data)
    return digest.hexdigest()


def _write_atomic(path: Path, data: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = None
    try:
        with tempfile.NamedTemporaryFile(dir=path.parent, prefix='.' + path.name, suffix='.tmp', delete=False) as f:
            temp = Path(f.name)
            f.write(data)
        os.replace(temp, path)
    finally:
        if temp is not None:
            temp.unlink(missing_ok=True)


def _image_bytes(image: Image.Image) -> bytes:
    import io
    buffer = io.BytesIO()
    image.save(buffer, format='PNG')
    return buffer.getvalue()


def _assert_lora_coverage(pipe: Any, adapter: Path) -> dict[str, int]:
    """Require every Kohya UNet LoRA pair to produce one active PEFT layer."""
    with adapter.open('rb') as file:
        raw = file.read(8)
        if len(raw) != 8:
            raise RuntimeError('Invalid LoRA safetensors header')
        length = struct.unpack('<Q', raw)[0]
        if not 0 < length <= 16 * 1024 * 1024:
            raise RuntimeError('Invalid LoRA safetensors header size')
        header = json.loads(file.read(length))
    keys = set(header) - {'__metadata__'}
    downs = {key[:-len('.lora_down.weight')] for key in keys if key.endswith('.lora_down.weight')}
    ups = {key[:-len('.lora_up.weight')] for key in keys if key.endswith('.lora_up.weight')}
    allowed = {base + suffix for base in downs for suffix in ('.lora_down.weight', '.lora_up.weight', '.alpha')}
    if not downs or downs != ups or keys - allowed or any(not key.startswith('lora_unet_') for key in downs):
        raise RuntimeError('Unexpected Pixel Art XL LoRA tensors or unpaired layers')
    applied = sum(1 for _, layer in pipe.unet.named_modules()
                  if 'pixel' in getattr(layer, 'lora_A', {}) and 'pixel' in getattr(layer, 'lora_B', {}))
    if applied != len(downs):
        raise RuntimeError(f'Pixel Art XL LoRA coverage mismatch: expected {len(downs)} paired layers, loaded {applied}; refusing to continue.')
    return {'expected_layer_pairs': len(downs), 'applied_layer_pairs': applied}


def _memory_status() -> dict[str, int | None]:
    import psutil
    result: dict[str, int | None] = {'physical_available': psutil.virtual_memory().available, 'commit_available': None}
    if sys.platform == 'win32':
        import ctypes
        class PerformanceInformation(ctypes.Structure):
            _fields_ = [('cb', ctypes.c_ulong)] + [(name, ctypes.c_size_t) for name in (
                'CommitTotal', 'CommitLimit', 'CommitPeak', 'PhysicalTotal', 'PhysicalAvailable',
                'SystemCache', 'KernelTotal', 'KernelPaged', 'KernelNonpaged', 'PageSize')] + [
                ('HandleCount', ctypes.c_ulong), ('ProcessCount', ctypes.c_ulong), ('ThreadCount', ctypes.c_ulong)]
        info = PerformanceInformation()
        info.cb = ctypes.sizeof(info)
        fn = ctypes.windll.psapi.GetPerformanceInfo
        fn.argtypes = [ctypes.POINTER(PerformanceInformation), ctypes.c_ulong]
        fn.restype = ctypes.c_int
        if not fn(ctypes.byref(info), info.cb):
            raise RuntimeError('Could not determine Windows commit headroom before loading SDXL.')
        result['commit_available'] = max(0, (info.CommitLimit - info.CommitTotal) * info.PageSize)
    return result


def _memory_preflight(model_paths: dict[str, Path]) -> dict[str, int | None]:
    """Estimate checkpoint size plus 4 GiB and require physical headroom."""
    from .models import list_models
    seen: set[Path] = set()
    for key, model in list_models('image').items():
        for item in model['files']:
            if not item['filename'].endswith('.safetensors'):
                continue
            path = Path(model_paths[key]) if key == 'lora' else Path(model_paths[key]) / item['filename']
            if not path.is_file():
                raise RuntimeError(f'Missing selected {key} checkpoint {path}. Run pixcli models verify.')
            seen.add(path.resolve())
    weights = sum(path.stat().st_size for path in seen)
    required_commit = max(8 * 1024**3, weights + 4 * 1024**3)
    required_physical = 8 * 1024**3
    status = _memory_status()
    commit = status['commit_available']
    if commit is not None and commit < required_commit:
        raise RuntimeError(f'Insufficient Windows commit headroom for SDXL: {commit / 1024**3:.1f} GiB available; estimated {required_commit / 1024**3:.1f} GiB needed (checkpoint files plus 4 GiB overhead). Close memory-heavy applications or increase the Windows page-file limit, then retry; no model was loaded.')
    physical = status['physical_available']
    if physical is not None and physical < required_physical:
        raise RuntimeError(f'Insufficient available physical RAM for SDXL: {physical / 1024**3:.1f} GiB available; estimated at least 8 GiB needed. Close memory-heavy applications and retry; no model was loaded.')
    return {**status, 'checkpoint_bytes': weights, 'estimated_commit_needed': required_commit, 'estimated_physical_needed': required_physical}


class _ImageMemoryGuard:
    """Sample host headroom between model-loading phases and denoising steps."""
    def __init__(self) -> None:
        self.minimum_physical: int | None = None
        self.minimum_commit: int | None = None

    def sample(self) -> None:
        status = _memory_status()
        physical, commit = status['physical_available'], status['commit_available']
        if physical is not None:
            self.minimum_physical = physical if self.minimum_physical is None else min(self.minimum_physical, physical)
        if commit is not None:
            self.minimum_commit = commit if self.minimum_commit is None else min(self.minimum_commit, commit)
        if (physical is not None and physical < 2 * 1024**3) or (commit is not None and commit < 2 * 1024**3):
            raise MemoryError('Image memory guard stopped the job because available physical or commit memory fell below 2 GiB')

    def callback(self, pipe: Any, step: int, timestep: Any, values: dict) -> dict:
        self.sample()
        return values


def _oom_message(s: ImageSettings) -> str:
    if s.inference_width == s.inference_height and s.inference_width > 512:
        advice = 'Try --inference-width 512 --inference-height 512 and close other GPU applications.'
    else:
        advice = 'Close other GPU applications; if possible, reduce the inference dimensions while keeping the required aspect ratio.'
    return (f'CUDA out of memory at {s.inference_width}x{s.inference_height}, {s.steps} steps, '
            f'seed={s.seed}, guidance={s.guidance}, LoRA strength={s.lora_strength}, '
            f'edit strength={s.strength}. {advice} No settings were changed automatically.')


def _process_memory_peaks() -> dict[str, int | None]:
    import psutil
    try:
        memory = psutil.Process().memory_info()
    except (OSError, psutil.Error):
        return {'peak_working_set_bytes': None, 'peak_commit_bytes': None}
    return {'peak_working_set_bytes': getattr(memory, 'peak_wset', None),
            'peak_commit_bytes': getattr(memory, 'peak_pagefile', None)}


def _infer(s: ImageSettings, model_paths: dict[str, Path], image: Image.Image | None) -> tuple[Image.Image, dict[str, Any]]:
    """Run SDXL with offline flags scoped to this inference call."""
    keys = ('HF_HUB_OFFLINE', 'TRANSFORMERS_OFFLINE')
    previous = {key: os.environ.get(key) for key in keys}
    try:
        for key in keys:
            os.environ[key] = '1'
        return _infer_offline(s, model_paths, image)
    finally:
        for key, value in previous.items():
            if value is None:
                os.environ.pop(key, None)
            else:
                os.environ[key] = value


def _infer_offline(s: ImageSettings, model_paths: dict[str, Path], image: Image.Image | None) -> tuple[Image.Image, dict[str, Any]]:
    """Run one SDXL job from local snapshots, using CPU model offload."""
    try:
        import torch
        from diffusers import AutoencoderKL, StableDiffusionXLPipeline, StableDiffusionXLImg2ImgPipeline
    except ImportError as exc:
        raise RuntimeError('Image dependencies missing; run the project setup script.') from exc
    if not torch.cuda.is_available():
        raise RuntimeError('CUDA is unavailable. Install the project CUDA dependencies and use an NVIDIA GPU; run pixcli doctor.')
    for key in ('base', 'vae', 'lora'):
        if key not in model_paths or not Path(model_paths[key]).exists():
            raise RuntimeError(f'Missing local {key} model. Run pixcli models download --profile image.')
    memory = _memory_preflight(model_paths)
    guard = _ImageMemoryGuard()
    pipe = None
    vae = None
    oom_error = None
    torch.cuda.reset_peak_memory_stats()
    try:
        vae = AutoencoderKL.from_pretrained(str(model_paths['vae']), torch_dtype=torch.float16, local_files_only=True, use_safetensors=True)
        guard.sample()
        cls = StableDiffusionXLImg2ImgPipeline if image is not None else StableDiffusionXLPipeline
        pipe = cls.from_pretrained(str(model_paths['base']), vae=vae, torch_dtype=torch.float16, variant='fp16', local_files_only=True, use_safetensors=True, low_cpu_mem_usage=True)
        guard.sample()
        adapter = Path(model_paths['lora'])
        pipe.load_lora_weights(str(adapter.parent), weight_name=adapter.name, adapter_name='pixel', local_files_only=True)
        pipe.set_adapters('pixel', adapter_weights=s.lora_strength)
        if 'pixel' not in pipe.get_active_adapters():
            raise RuntimeError('Pixel Art XL adapter did not activate; refusing to generate without it.')
        coverage = _assert_lora_coverage(pipe, adapter)
        guard.sample()
        pipe.enable_vae_slicing()
        pipe.enable_vae_tiling()
        pipe.enable_model_cpu_offload()
        args: dict[str, Any] = dict(prompt=s.prompt, negative_prompt=s.negative_prompt, num_inference_steps=s.steps,
                                    guidance_scale=s.guidance, generator=torch.Generator(device='cpu').manual_seed(s.seed),
                                    callback_on_step_end=guard.callback)
        if image is None:
            args.update(width=s.inference_width, height=s.inference_height)
        else:
            args.update(image=image, strength=s.strength)
        with torch.inference_mode():
            output = pipe(**args).images[0]
        return output, {'peak_cuda_allocated_bytes': torch.cuda.max_memory_allocated(),
                        'peak_cuda_reserved_bytes': torch.cuda.max_memory_reserved(),
                        'gpu_name': torch.cuda.get_device_name(), 'precision': 'float16',
                        'scheduler': type(pipe.scheduler).__name__, 'scheduler_config': dict(pipe.scheduler.config),
                        'quantization': None, 'memory_mode': 'model_cpu_offload', 'adapter_active': True,
                        'adapter_coverage': coverage, 'memory_preflight': memory,
                        'minimum_sampled_physical_available_bytes': guard.minimum_physical,
                        'minimum_sampled_commit_available_bytes': guard.minimum_commit,
                        'process_memory_peaks': _process_memory_peaks()}
    except torch.cuda.OutOfMemoryError as exc:
        # Traceback frames from model loading or execution may themselves own
        # tensors. Clear them before collecting and emptying the CUDA cache.
        traceback.clear_frames(exc.__traceback__)
        exc.__traceback__ = None
        oom_error = _oom_message(s)
    except MemoryError as exc:
        message = str(exc) or 'Host memory allocation failed'
        traceback.clear_frames(exc.__traceback__)
        exc.__traceback__ = None
        oom_error = (f'{message} at {s.inference_width}x{s.inference_height}, {s.steps} steps, seed={s.seed}. '
                     'Close memory-heavy applications and retry; no settings were changed automatically.')
    finally:
        pipe = None
        vae = None
        gc.collect()
        torch.cuda.empty_cache()
    if oom_error:
        raise RuntimeError(oom_error) from None


def run_image(s: ImageSettings, model_paths: dict[str, Path]) -> dict[str, Any]:
    """Write a PNG, JSON sidecar, and requested optional files. Returns metadata.

    Model paths must be local pinned base and VAE snapshot directories and the
    local adapter safetensors file, under keys ``base``, ``vae``, and ``lora``.
    """
    validate_image(s)
    out = Path(s.out)
    sidecar = out.with_suffix('.json')
    original_path = out.with_name(out.stem + '.original.png')
    preview_path = out.with_name(out.stem + '.preview.png')
    source, mask, inference_input, box = None, None, None, None
    input_hashes = {}
    if s.input_path:
        path = Path(s.input_path)
        with Image.open(path) as im:
            source = im.copy()
        input_hashes['input_sha256'] = _hash_file(path)
        inference_input, box = prepare_edit(source, (s.inference_width, s.inference_height), s.background)
        if s.mask_path:
            mask_path = Path(s.mask_path)
            with Image.open(mask_path) as im:
                mask = im.convert('L')
            if mask.size != source.size:
                raise ValueError('mask dimensions must match the input image')
            input_hashes['mask_sha256'] = _hash_file(mask_path)
    started = time.perf_counter()
    generated, runtime = _infer(s, model_paths, inference_input)
    if source is None:
        result = pixelate(generated, (s.pixel_width, s.pixel_height), s.palette_colors)
        processing = ['nearest-neighbor resize to final dimensions']
        if s.palette_colors:
            processing.append(f'palette reduction to {s.palette_colors} colors, no dithering')
    else:
        result = finish_edit(source, generated, box, mask)
        processing = ['alpha composite source onto inference background', f'aspect-preserving nearest-neighbor fit and letterbox {box}',
                      'crop inference padding and nearest-neighbor resize to source dimensions', 'retain source alpha']
        if mask is not None:
            processing.append('composite edit over original using grayscale mask; black pixels unchanged')
    if s.preview_scale > 1:
        preview = result.resize((result.width * s.preview_scale, result.height * s.preview_scale), Image.Resampling.NEAREST)
        processing.append(f'nearest-neighbor preview scaled {s.preview_scale}x')
    from .models import list_models
    versions = {}
    for name in ('torch', 'diffusers', 'transformers', 'accelerate', 'peft', 'Pillow', 'huggingface-hub', 'safetensors'):
        try:
            versions[name] = version(name)
        except PackageNotFoundError:
            versions[name] = None
    settings = {k: str(v) if isinstance(v, Path) else v for k, v in asdict(s).items()}
    models = {key: {'repo_id': data['repo_id'], 'revision': data['revision']} for key, data in list_models('image').items()}
    metadata = {'schema_version': 1, 'operation': 'edit' if source else 'gen', 'settings': settings, 'offline': True,
                'input_hashes': input_hashes, 'models': models, 'backend': 'diffusers', 'runtime': runtime,
                'dependency_versions': versions, 'inference_dimensions': [s.inference_width, s.inference_height],
                'final_dimensions': list(result.size), 'processing': processing,
                'elapsed_seconds': round(time.perf_counter() - started, 3), 'fallbacks': []}
    _write_atomic(out, _image_bytes(result))
    if s.save_original:
        _write_atomic(original_path, _image_bytes(generated))
    if s.preview_scale > 1:
        _write_atomic(preview_path, _image_bytes(preview))
    _write_atomic(sidecar, (json.dumps(metadata, indent=2, ensure_ascii=False) + '\n').encode('utf-8'))
    return metadata
