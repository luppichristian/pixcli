"""Animation job validation, local inference, export and metadata."""
from __future__ import annotations

from dataclasses import asdict, dataclass
import hashlib
from importlib.metadata import PackageNotFoundError, version
import json
import math
from pathlib import Path
import time
from typing import Callable

from PIL import Image, ImageColor
from .animation_export import export_animation
from .comfy_backend import build_graph, run_comfy, validate_animation, validate_memory_budget


@dataclass
class AnimationSettings:
    input_path: Path
    prompt: str
    out: Path
    negative_prompt: str = ''
    seed: int = 42
    steps: int = 4
    guidance: float = 1.0
    lora_strength: float = 1.0
    inference_width: int = 256
    inference_height: int = 256
    frames: int = 5
    fps: float = 8.0
    pixel_width: int = 128
    pixel_height: int = 128
    preset: str = 'low-vram'
    backend: str = 'comfyui-gguf'
    text_encoder: str = 'q3'
    expert_quant: str = 'q2'
    adapter_device: str = 'cpu'
    weight_storage: str = 'mmap'
    max_private_gib: float = 14.0
    min_free_commit_gib: float = 2.0
    min_free_physical_gib: float = 2.0
    palette_colors: int | None = None
    palette_image: Path | None = None
    background: str = '#ffffff'
    mp4: bool = True
    overwrite: bool = False


def _check_mp4_dependencies() -> None:
    try:
        import imageio.v2  # noqa: F401
        import imageio_ffmpeg
        import numpy  # noqa: F401
        imageio_ffmpeg.get_ffmpeg_exe()
    except (ImportError, RuntimeError) as exc:
        raise RuntimeError('MP4 export needs imageio[ffmpeg] and numpy; run setup.ps1 or specify --no-mp4.') from exc


def validate_animation_job(s: AnimationSettings) -> None:
    validate_animation(s.inference_width, s.inference_height, s.frames)
    validate_memory_budget(s.max_private_gib, s.min_free_commit_gib, s.min_free_physical_gib)
    if not s.prompt.strip():
        raise ValueError('prompt cannot be empty')
    if not 0 <= s.seed <= 2**64 - 1:
        raise ValueError('seed must be in [0, 2^64-1]')
    if s.steps != 4:
        raise ValueError('This distilled workflow supports exactly 4 steps')
    if s.preset not in ('low-vram', 'gpu-resident') or s.backend != 'comfyui-gguf':
        raise ValueError('Use preset low-vram or gpu-resident with backend comfyui-gguf')
    if s.text_encoder not in ('fp8', 'q3'):
        raise ValueError('text encoder must be fp8 or q3')
    if s.expert_quant not in ('q2', 'q3'):
        raise ValueError('expert quantization must be q2 or q3')
    if s.preset == 'gpu-resident' and s.text_encoder != 'fp8':
        raise ValueError('gpu-resident requires the CPU-pinned fp8 text encoder')
    if s.adapter_device not in ('cpu', 'cuda'):
        raise ValueError('adapter device must be cpu or cuda')
    if s.weight_storage not in ('normal', 'mmap'):
        raise ValueError('weight storage must be normal or mmap')
    if s.adapter_device == 'cuda' and s.preset != 'gpu-resident':
        raise ValueError('CUDA adapter residency requires the gpu-resident preset')
    if not math.isfinite(s.guidance) or not 0 <= s.guidance <= 100:
        raise ValueError('guidance must be between 0 and 100')
    if not math.isfinite(s.lora_strength) or not 0.01 <= s.lora_strength <= 2:
        raise ValueError('Pixel LoRA strength must be between 0.01 and 2')
    if max(s.inference_width, s.inference_height) > 1024 or s.frames > 129:
        raise ValueError('inference dimensions may not exceed 1024 and frames may not exceed 129')
    if min(s.pixel_width, s.pixel_height) < 1 or max(s.pixel_width, s.pixel_height) > 2048:
        raise ValueError('pixel dimensions must be between 1 and 2048')
    if not math.isfinite(s.fps) or not 0 < s.fps <= 100:
        raise ValueError('fps must be greater than zero and at most 100')
    if s.palette_colors is not None and not 2 <= s.palette_colors <= 256:
        raise ValueError('palette colors must be between 2 and 256')
    if s.palette_image is not None and s.palette_colors is None:
        raise ValueError('palette image requires --palette-colors')
    try:
        if len(ImageColor.getrgb(s.background)) != 3:
            raise ValueError()
    except ValueError as exc:
        raise ValueError('background must be an RGB color such as #ffffff') from exc
    source, out = Path(s.input_path), Path(s.out)
    if source.resolve().is_relative_to(out.resolve()):
        raise ValueError('input must not be inside the output directory')
    with Image.open(source) as image:
        image.verify()
    if s.palette_image is not None:
        if Path(s.palette_image).resolve().is_relative_to(out.resolve()):
            raise ValueError('palette input must not be inside the output directory')
        with Image.open(s.palette_image) as image:
            image.verify()
    if s.mp4:
        _check_mp4_dependencies()
    if out.is_symlink():
        raise ValueError('output may not be a symbolic link')
    if out.exists():
        if not s.overwrite:
            raise FileExistsError(f'Output already exists: {out}; use --overwrite')
        if not out.is_dir() or not (out / '.pixcli-animation').is_file():
            raise FileExistsError(f'Refusing to replace a directory not created by pixcli: {out}')


def run_animation(s: AnimationSettings, model_paths: dict, progress: Callable[[str], None] | None = None) -> dict:
    validate_animation_job(s)
    source = Path(s.input_path)
    sha = hashlib.sha256()
    with source.open('rb') as f:
        for block in iter(lambda: f.read(1024 * 1024), b''):
            sha.update(block)
    started = time.perf_counter()
    frames, audit = run_comfy(source, s.prompt, s.negative_prompt, s.seed, s.inference_width,
                              s.inference_height, s.frames, s.guidance, model_paths,
                              background=ImageColor.getrgb(s.background), lora_strength=s.lora_strength,
                              text_encoder=s.text_encoder, expert_quant=s.expert_quant, preset=s.preset,
                              adapter_device=s.adapter_device, weight_storage=s.weight_storage, max_private_gib=s.max_private_gib,
                              min_free_commit_gib=s.min_free_commit_gib, min_free_physical_gib=s.min_free_physical_gib,
                              progress=progress)
    palette = None
    palette_sha256 = None
    if s.palette_image is not None:
        palette_path = Path(s.palette_image)
        with Image.open(palette_path) as image:
            palette = image.copy()
        palette_hash = hashlib.sha256()
        with palette_path.open('rb') as file:
            for block in iter(lambda: file.read(1024 * 1024), b''):
                palette_hash.update(block)
        palette_sha256 = palette_hash.hexdigest()
    result = export_animation(frames, s.out, final_size=(s.pixel_width, s.pixel_height), fps=s.fps,
                              palette_colors=s.palette_colors, palette_image=palette, background=ImageColor.getrgb(s.background),
                              overwrite=s.overwrite, mp4=s.mp4)
    from .models import animation_profile, list_models
    profile = animation_profile(s.text_encoder, s.expert_quant)
    models = {key: {'repo_id': value['repo_id'], 'revision': value['revision'],
                    'files': [item['filename'] for item in value['files']]}
              for key, value in list_models(profile).items()}
    versions = {}
    for package in ('torch', 'Pillow', 'psutil', 'imageio', 'imageio-ffmpeg'):
        try:
            versions[package] = version(package)
        except PackageNotFoundError:
            versions[package] = None
    graph = build_graph('source.png', s.prompt, s.negative_prompt, s.seed, s.inference_width,
                        s.inference_height, s.frames, s.guidance, s.lora_strength, text_encoder=s.text_encoder,
                        expert_quant=s.expert_quant, adapter_device=s.adapter_device, weight_storage=s.weight_storage)
    high, low = graph['high_sample']['inputs'], graph['low_sample']['inputs']
    workflow = {'sampler': high['sampler_name'], 'scheduler': high['scheduler'],
                'shift': graph['high_sampling']['inputs']['shift'], 'steps': high['steps'],
                'high_noise_steps': [high['start_at_step'], high['end_at_step']],
                'low_noise_steps': [low['start_at_step'], low['end_at_step']],
                'distillation_lora_strength': graph['high_distill']['inputs']['strength_model'],
                'motion_amplitude': graph['condition']['inputs']['motion_amplitude']}
    metadata = {'schema_version': 1, 'operation': 'animate', 'offline': True, 'settings': {k: str(v) if isinstance(v, Path) else v for k, v in asdict(s).items()},
                'input_sha256': sha.hexdigest(), 'palette_sha256': palette_sha256, 'models': models, 'backend': s.backend,
                'model_profile': profile,
                'runtime': audit, 'dependency_versions': versions, 'precision': 'mixed',
                'quantization': f"GGUF {'Q2_K' if s.expert_quant == 'q2' else 'Q3_K_S'} experts; {'Q3_K_S' if s.text_encoder == 'q3' else 'FP8'} text",
                'workflow': workflow,
                'inference_dimensions': [s.inference_width, s.inference_height], 'final_dimensions': [s.pixel_width, s.pixel_height],
                'processing': ['aspect-preserving nearest-neighbor fit and letterbox', 'shared sequence transform and GIF palette', 'playback loop only'],
                'elapsed_seconds': round(time.perf_counter() - started, 3), 'fallbacks': []}
    from .image import _write_atomic
    _write_atomic(Path(s.out) / 'metadata.json', (json.dumps(metadata, indent=2) + '\n').encode('utf-8'))
    return result
