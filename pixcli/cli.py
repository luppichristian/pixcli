"""Command-line interface for local pixel-art inference."""
from __future__ import annotations

import json
import os
import platform
import shutil
import sys
from pathlib import Path
from typing import Optional

import typer
from typer.completion import install_callback, show_callback

app = typer.Typer(no_args_is_help=True, add_completion=False, context_settings={'help_option_names': ['-help']}, help="Generate, edit, and animate pixel art locally.")
models_app = typer.Typer(no_args_is_help=True, add_completion=False, context_settings={'help_option_names': ['-help']}, help="Manage pinned local model files.")
app.add_typer(models_app, name="models")


@app.callback()
def main(
    install_completion: bool = typer.Option(None, '-install-completion', callback=install_callback, expose_value=False, is_eager=True, help='Install completion for the current shell.'),
    show_completion: bool = typer.Option(None, '-show-completion', callback=show_callback, expose_value=False, is_eager=True, help='Show completion for the current shell.'),
):
    pass


def _print_report(value, indent=0):
    """Print nested reports in a readable form for interactive use."""
    prefix = ' ' * indent
    if isinstance(value, dict):
        for key, item in value.items():
            if isinstance(item, (dict, list)):
                typer.echo(f'{prefix}{key}:')
                _print_report(item, indent + 2)
            else:
                typer.echo(f'{prefix}{key}: {item}')
    elif isinstance(value, list):
        for item in value:
            if isinstance(item, (dict, list)):
                _print_report(item, indent + 2)
            else:
                typer.echo(f'{prefix}- {item}')
    else:
        typer.echo(f'{prefix}{value}')


def _output(value, as_json):
    if as_json:
        typer.echo(json.dumps(value, indent=2, default=str))
    else:
        _print_report(value)


def fail(exc: Exception) -> None:
    typer.echo(f"Error: {exc}", err=True)
    raise typer.Exit(1)


def _os_description() -> str:
    description = platform.platform()
    if sys.platform == "win32":
        try:
            if int(platform.version().split(".")[-1]) >= 22000:
                return f"Windows 11 ({description})"
        except ValueError:
            pass
    return description


@app.command()
def doctor(cache_dir: Optional[Path] = typer.Option(None, '-cache-dir', help="Hugging Face Hub cache directory."), json_output: bool = typer.Option(False, '-json', help='Print JSON output.')):
    """Report hardware, environment and offline model readiness."""
    from . import models
    import importlib.metadata as md
    import psutil
    report = {"os": _os_description(), "python": sys.version.split()[0], "cache": str(models.cache_location(cache_dir)), "system_ram_total_bytes": psutil.virtual_memory().total, "system_ram_available_bytes": psutil.virtual_memory().available}
    try:
        from .image import _memory_status
        report["memory_headroom"] = _memory_status()
    except Exception as exc:
        report["memory_headroom_error"] = str(exc)
    path = models.cache_location(cache_dir)
    while not path.exists() and path != path.parent:
        path = path.parent
    report["disk_free_bytes"] = shutil.disk_usage(path).free
    for package in ("torch", "diffusers", "transformers", "accelerate", "huggingface-hub", "peft", "Pillow", "imageio-ffmpeg"):
        try:
            report[package] = md.version(package)
        except md.PackageNotFoundError:
            report[package] = "missing"
    try:
        import torch
        report["cuda_available"] = torch.cuda.is_available()
        report["torch_cuda"] = torch.version.cuda
        if torch.cuda.is_available():
            free, total = torch.cuda.mem_get_info()
            report["gpu"] = torch.cuda.get_device_name()
            report["vram_free_bytes"] = free
            report["vram_total_bytes"] = total
    except Exception as exc:
        report["cuda_error"] = str(exc)
    try:
        for key in ("base", "vae", "lora"):
            try:
                report[f"model_{key}"] = str(models.model_path(key, cache_dir))
            except models.ModelError as exc:
                report[f"model_{key}"] = str(exc)
    except Exception as exc:
        report["models_error"] = str(exc)
    for profile in ('animation', 'animation-q3-fp8', 'animation-q3-text', 'animation-q2-experts', 'animation-q2-q3-text'):
        try:
            entries = {}
            for key in models.list_models(profile):
                try:
                    entries[key] = str(models.model_path(key, cache_dir))
                except models.ModelError as exc:
                    entries[key] = str(exc)
            report[f'models_{profile}'] = entries
        except models.ModelError as exc:
            report[f'models_{profile}'] = str(exc)
    root = Path(__file__).resolve().parent.parent / '.backend'
    python = root / '.venv' / ('Scripts/python.exe' if os.name == 'nt' else 'bin/python')
    installed = (root / 'ready.json').is_file() and python.is_file() and (root / 'ComfyUI' / 'main.py').is_file() and (root / 'ComfyUI' / 'custom_nodes' / 'pixcli_wan' / '__init__.py').is_file()
    report["animation_backend"] = "installed, runtime unverified" if installed else "setup incomplete, runtime unverified"
    _output(report, json_output)


@models_app.command("list")
def model_list(json_output: bool = typer.Option(False, '-json', help='Print JSON output.')):
    """Show pinned repositories, revisions and required files."""
    from . import models
    _output(models.list_models(), json_output)


@models_app.command("download")
def model_download(profile: str = typer.Option("image", '-profile'), cache_dir: Optional[Path] = typer.Option(None, '-cache-dir'), offline: bool = typer.Option(False, '-offline', help="Disallow network access."), json_output: bool = typer.Option(False, '-json', help='Print JSON output.')):
    """Download the selected pinned profile into the Hub cache."""
    if offline:
        fail(ValueError("Offline mode forbids downloads. Run models verify to inspect cached files."))
    from . import models
    try:
        result = models.download(profile, cache_dir)
        if json_output:
            _output(result, True)
        else:
            typer.echo(f"Verified {len(result)} files.")
    except Exception as exc:
        fail(exc)


@models_app.command("verify")
def model_verify(profile: str = typer.Option("image", '-profile'), cache_dir: Optional[Path] = typer.Option(None, '-cache-dir'), json_output: bool = typer.Option(False, '-json', help='Print JSON output.')):
    """Verify pinned file sizes and hashes without network access."""
    from . import models
    try:
        result = models.verify(profile, cache_dir)
        if json_output:
            _output(result, True)
        else:
            for record in result:
                typer.echo(f"{record['key']}: {record['filename']}: {record['detail']}")
        if not all(record['ok'] for record in result):
            raise typer.Exit(1)
    except typer.Exit:
        raise
    except Exception as exc:
        fail(exc)


def _run_image(settings, cache_dir):
    from . import models
    from .image import run_image, validate_image
    try:
        validate_image(settings)
        paths = {key: models.model_path(key, cache_dir) for key in ("base", "vae", "lora")}
        result = run_image(settings, paths)
        if settings.save_json:
            _output(result, True)
        else:
            typer.echo(f'Saved image: {settings.out}')
    except Exception as exc:
        fail(exc)


def _style_settings(name: str | None, palette: str | None) -> dict:
    from .styles import load_style
    try:
        style = load_style(name)
        return {'style': name, 'style_prompt': style.prompt if style else None,
                'palette': palette if palette is not None else (style.palette if style else None)}
    except ValueError as exc:
        fail(exc)


@app.command()
def gen(
    prompt: str = typer.Option(..., '-prompt'), out: Path = typer.Option(..., '-out'),
    style: Optional[str] = typer.Option(None, '-style', help='Style name from styles/NAME.txt in the current directory.'),
    palette: Optional[str] = typer.Option(None, '-palette', help='Palette name from palettes/ in the current directory.'),
    negative_prompt: str = typer.Option("", '-negative-prompt'), seed: int = typer.Option(42, '-seed'), steps: int = typer.Option(30, '-steps'),
    guidance: float = typer.Option(7.0, '-guidance'), lora_strength: float = typer.Option(1.0, '-lora-strength'),
    inference_width: int = typer.Option(1024, '-inference-width'), inference_height: int = typer.Option(1024, '-inference-height'),
    pixel_width: int = typer.Option(128, '-pixel-width'), pixel_height: int = typer.Option(128, '-pixel-height'),
    preview_scale: int = typer.Option(1, '-preview-scale'), palette_colors: Optional[int] = typer.Option(None, '-palette-colors'),
    save_original: bool = typer.Option(False, '-save-original'), overwrite: bool = typer.Option(False, '-overwrite'),
    ram_budget_gib: Optional[float] = typer.Option(None, '-ram-budget-gib', help='Sampled process resident RAM budget in GiB.'),
    vram_budget_gib: Optional[float] = typer.Option(None, '-vram-budget-gib', help='PyTorch CUDA allocator budget in GiB.'),
    min_free_commit_gib: float = typer.Option(2.0, '-min-free-commit-gib', help='Minimum sampled free Windows commit in GiB (at least 1).'),
    min_free_physical_gib: float = typer.Option(2.0, '-min-free-physical-gib', help='Minimum sampled available physical RAM in GiB (at least 1).'),
    device: str = typer.Option('auto', '-device', help='Compute device: auto, cuda, or cpu.'),
    weight_storage: str = typer.Option('auto', '-weight-storage', help='Image weights: auto, memory or disk-backed offload.'),
    cache_dir: Optional[Path] = typer.Option(None, '-cache-dir'), offline: bool = typer.Option(False, '-offline', help="Explicit offline mode (inference is always offline)."),
    json_output: bool = typer.Option(False, '-json', help='Print JSON and save metadata files.'),
):
    """Generate an image from text, entirely offline."""
    from .image import ImageSettings
    _run_image(ImageSettings(prompt=prompt, out=out, save_json=json_output, **_style_settings(style, palette), negative_prompt=negative_prompt, seed=seed, steps=steps, guidance=guidance, lora_strength=lora_strength, inference_width=inference_width, inference_height=inference_height, pixel_width=pixel_width, pixel_height=pixel_height, preview_scale=preview_scale, palette_colors=palette_colors, save_original=save_original, overwrite=overwrite, ram_budget_gib=ram_budget_gib, vram_budget_gib=vram_budget_gib, min_free_commit_gib=min_free_commit_gib, min_free_physical_gib=min_free_physical_gib, weight_storage=weight_storage, device=device), cache_dir)


@app.command()
def edit(
    input: Path = typer.Option(..., '-input'), prompt: str = typer.Option(..., '-prompt'), out: Path = typer.Option(..., '-out'),
    style: Optional[str] = typer.Option(None, '-style', help='Style name from styles/NAME.txt in the current directory.'),
    palette: Optional[str] = typer.Option(None, '-palette', help='Palette name from palettes/ in the current directory.'),
    mask: Optional[Path] = typer.Option(None, '-mask'), strength: float = typer.Option(0.3, '-strength'),
    negative_prompt: str = typer.Option("", '-negative-prompt'), seed: int = typer.Option(42, '-seed'), steps: int = typer.Option(30, '-steps'),
    guidance: float = typer.Option(7.0, '-guidance'), lora_strength: float = typer.Option(1.0, '-lora-strength'),
    inference_width: int = typer.Option(1024, '-inference-width'), inference_height: int = typer.Option(1024, '-inference-height'),
    background: str = typer.Option("#ffffff", '-background'), preview_scale: int = typer.Option(1, '-preview-scale'),
    palette_colors: Optional[int] = typer.Option(None, '-palette-colors'), save_original: bool = typer.Option(False, '-save-original'),
    overwrite: bool = typer.Option(False, '-overwrite'), cache_dir: Optional[Path] = typer.Option(None, '-cache-dir'),
    ram_budget_gib: Optional[float] = typer.Option(None, '-ram-budget-gib', help='Sampled process resident RAM budget in GiB.'),
    vram_budget_gib: Optional[float] = typer.Option(None, '-vram-budget-gib', help='PyTorch CUDA allocator budget in GiB.'),
    min_free_commit_gib: float = typer.Option(2.0, '-min-free-commit-gib', help='Minimum sampled free Windows commit in GiB (at least 1).'),
    min_free_physical_gib: float = typer.Option(2.0, '-min-free-physical-gib', help='Minimum sampled available physical RAM in GiB (at least 1).'),
    device: str = typer.Option('auto', '-device', help='Compute device: auto, cuda, or cpu.'),
    weight_storage: str = typer.Option('auto', '-weight-storage', help='Image weights: auto, memory or disk-backed offload.'),
    offline: bool = typer.Option(False, '-offline', help="Explicit offline mode (inference is always offline)."),
    json_output: bool = typer.Option(False, '-json', help='Print JSON and save metadata files.'),
):
    """Edit an image; white mask pixels are editable and black are preserved."""
    from .image import ImageSettings
    _run_image(ImageSettings(prompt=prompt, out=out, save_json=json_output, **_style_settings(style, palette), input_path=input, mask_path=mask, strength=strength, negative_prompt=negative_prompt, seed=seed, steps=steps, guidance=guidance, lora_strength=lora_strength, inference_width=inference_width, inference_height=inference_height, background=background, preview_scale=preview_scale, palette_colors=palette_colors, save_original=save_original, overwrite=overwrite, ram_budget_gib=ram_budget_gib, vram_budget_gib=vram_budget_gib, min_free_commit_gib=min_free_commit_gib, min_free_physical_gib=min_free_physical_gib, weight_storage=weight_storage, device=device), cache_dir)


@app.command()
def animate(
    input: Path = typer.Option(..., '-input'), prompt: str = typer.Option(..., '-prompt'), out: Path = typer.Option(..., '-out'),
    style: Optional[str] = typer.Option(None, '-style', help='Style name from styles/NAME.txt in the current directory.'),
    palette: Optional[str] = typer.Option(None, '-palette', help='Palette name from palettes/ in the current directory.'),
    negative_prompt: str = typer.Option("", '-negative-prompt'), seed: int = typer.Option(42, '-seed'), steps: int = typer.Option(4, '-steps'),
    guidance: float = typer.Option(1.0, '-guidance'), lora_strength: float = typer.Option(1.0, '-lora-strength'), preset: str = typer.Option("low-vram", '-preset'),
    backend: str = typer.Option("comfyui-gguf", '-backend'), text_encoder: str = typer.Option("q3", '-text-encoder'), expert_quant: str = typer.Option("q2", '-expert-quant'), adapter_device: str = typer.Option("cpu", '-adapter-device'), weight_storage: str = typer.Option("mmap", '-weight-storage'), inference_width: int = typer.Option(256, '-inference-width'),
    inference_height: int = typer.Option(256, '-inference-height'), frames: int = typer.Option(5, '-frames'), fps: float = typer.Option(8.0, '-fps'),
    pixel_width: int = typer.Option(128, '-pixel-width'), pixel_height: int = typer.Option(128, '-pixel-height'),
    palette_colors: Optional[int] = typer.Option(None, '-palette-colors'), palette_image: Optional[Path] = typer.Option(None, '-palette-image'), background: str = typer.Option("#ffffff", '-background'),
    mp4: bool = typer.Option(True, '-mp4/-no-mp4'), overwrite: bool = typer.Option(False, '-overwrite'),
    ram_budget_gib: Optional[float] = typer.Option(None, '-ram-budget-gib', help='Command process-tree resident RAM budget in GiB.'),
    vram_budget_gib: Optional[float] = typer.Option(None, '-vram-budget-gib', help='Backend PyTorch CUDA allocator budget in GiB.'),
    device: str = typer.Option('auto', '-device', help='Compute device: auto, cuda, or cpu.'),
    max_private_gib: Optional[float] = typer.Option(None, '-max-private-gib', help='Backend private-memory limit in GiB; defaults to 14 without a RAM budget.'),
    timeout_seconds: Optional[float] = typer.Option(None, '-timeout-seconds', help='Optional overall animation timeout in seconds.'),
    min_free_commit_gib: float = typer.Option(2.0, '-min-free-commit-gib', help='Minimum sampled free Windows commit in GiB (at least 1).'),
    min_free_physical_gib: float = typer.Option(2.0, '-min-free-physical-gib', help='Minimum sampled available physical RAM in GiB (at least 1).'),
    cache_dir: Optional[Path] = typer.Option(None, '-cache-dir'), offline: bool = typer.Option(False, '-offline', help="Explicit offline mode (inference is always offline)."),
    json_output: bool = typer.Option(False, '-json', help='Print JSON and save metadata files.'),
):
    """Animate with the experimental local Wan2.2 GGUF backend."""
    from .animation import AnimationSettings, run_animation, validate_animation_job
    from . import models
    try:
        settings = AnimationSettings(input_path=input, prompt=prompt, out=out, save_json=json_output, **_style_settings(style, palette), negative_prompt=negative_prompt,
                                     seed=seed, steps=steps, guidance=guidance, lora_strength=lora_strength, preset=preset, backend=backend, text_encoder=text_encoder, expert_quant=expert_quant, adapter_device=adapter_device, weight_storage=weight_storage,
                                     inference_width=inference_width, inference_height=inference_height, frames=frames, fps=fps,
                                     pixel_width=pixel_width, pixel_height=pixel_height, palette_colors=palette_colors, palette_image=palette_image,
                                     background=background, mp4=mp4, overwrite=overwrite, max_private_gib=max_private_gib, ram_budget_gib=ram_budget_gib, vram_budget_gib=vram_budget_gib, device=device, timeout_seconds=timeout_seconds,
                                     min_free_commit_gib=min_free_commit_gib, min_free_physical_gib=min_free_physical_gib)
        validate_animation_job(settings)
        profile = models.animation_profile(text_encoder, expert_quant)
        paths = {key: models.model_path(key, cache_dir, profile=profile) for key in models.list_models(profile)}
        typer.echo("Local Wan backend: memory requirements and output quality depend on the selected settings.", err=True)
        result = run_animation(settings, paths, progress=lambda message: typer.echo(message, err=True))
        if json_output:
            _output(result, True)
        else:
            typer.echo(f'Saved animation: {out}')
    except typer.Exit:
        raise
    except Exception as exc:
        fail(exc)


if __name__ == "__main__":
    app()

