import json

import pytest
from PIL import Image
from typer.testing import CliRunner

from pixcli.cli import app


@pytest.mark.parametrize('command', ['gen', 'edit', 'animate'])
def test_style_option_loads_prompt_for_each_command(tmp_path, monkeypatch, command):
    from pixcli import animation, cli, models

    monkeypatch.chdir(tmp_path)
    (tmp_path / 'styles').mkdir()
    (tmp_path / 'styles' / 'retro.txt').write_text('limited colors,\nsharp outlines\n', encoding='utf-8')
    source = tmp_path / 'source.png'
    Image.new('RGB', (8, 8)).save(source)
    seen = []
    monkeypatch.setattr(cli, '_run_image', lambda settings, cache: seen.append(settings))
    monkeypatch.setattr(animation, 'run_animation', lambda settings, paths, progress: seen.append(settings) or {})
    monkeypatch.setattr(models, 'list_models', lambda profile=None: {})
    extra = [] if command == 'gen' else ['--input', str(source)]
    if command == 'animate':
        extra += ['--no-mp4']
    result = CliRunner().invoke(app, [command, *extra, '--prompt', 'a knight', '--out', str(tmp_path / ('out' if command == 'animate' else 'out.png')), '--style', 'retro'])
    assert result.exit_code == 0, result.output
    assert seen[0].prompt == 'a knight'
    assert seen[0].style == 'retro'
    assert seen[0].style_prompt == 'limited colors,\nsharp outlines'


@pytest.mark.parametrize('name', ['missing', '../escape', 'empty'])
def test_style_rejects_missing_invalid_or_empty_files_before_model_loading(tmp_path, monkeypatch, name):
    monkeypatch.chdir(tmp_path)
    (tmp_path / 'styles').mkdir()
    (tmp_path / 'styles' / 'empty.txt').write_text(' \n', encoding='utf-8')
    result = CliRunner().invoke(app, ['gen', '--prompt', 'cat', '--out', str(tmp_path / 'out.png'), '--style', name])
    assert result.exit_code != 0
    assert 'style' in result.output.lower()


def test_animation_style_reaches_backend_and_is_recorded(tmp_path, monkeypatch):
    from pixcli.animation import AnimationSettings, run_animation
    source = tmp_path / 'source.png'
    Image.new('RGB', (8, 8)).save(source)
    seen = []
    def infer(*args, **kwargs):
        seen.append(args[1])
        return [Image.new('RGB', (8, 8))] * 5, {}
    monkeypatch.setattr('pixcli.animation.run_comfy', infer)
    settings = AnimationSettings(input_path=source, prompt='walk', out=tmp_path / 'out',
                                 style='retro', style_prompt='sharp outlines', mp4=False)
    run_animation(settings, {})
    metadata = json.loads((settings.out / 'metadata.json').read_text(encoding='utf-8'))
    assert seen == ['walk, sharp outlines']
    assert metadata['settings']['prompt'] == 'walk'
    assert metadata['settings']['style_prompt'] == 'sharp outlines'


@pytest.mark.parametrize('editing', [False, True])
def test_image_style_reaches_the_model(tmp_path, monkeypatch, editing):
    import contextlib
    import sys
    from types import SimpleNamespace
    from pixcli.image import ImageSettings, _infer_offline

    seen = []
    class Pipeline:
        scheduler = SimpleNamespace(config={})
        @classmethod
        def from_pretrained(cls, *args, **kwargs): return cls()
        def load_lora_weights(self, *args, **kwargs): pass
        def set_adapters(self, *args, **kwargs): pass
        def get_active_adapters(self): return ['pixel']
        def enable_vae_slicing(self): pass
        def enable_vae_tiling(self): pass
        def to(self, device): pass
        def __call__(self, **kwargs):
            seen.append(kwargs['prompt'])
            return SimpleNamespace(images=[Image.new('RGB', (2, 2))])
    class VAE:
        @classmethod
        def from_pretrained(cls, *args, **kwargs): return cls()
    class Generator:
        def __init__(self, **kwargs): pass
        def manual_seed(self, seed): return self
    class Guard:
        minimum_physical = minimum_commit = None
        def __init__(self, *args): pass
        def sample(self): pass
        def callback(self, *args): pass
    class FakeOOM(Exception): pass
    torch = SimpleNamespace(cuda=SimpleNamespace(is_available=lambda: False, OutOfMemoryError=FakeOOM),
                            float32=object(), Generator=Generator, inference_mode=contextlib.nullcontext)
    diffusers = SimpleNamespace(AutoencoderKL=VAE, StableDiffusionXLPipeline=Pipeline,
                                StableDiffusionXLImg2ImgPipeline=Pipeline)
    monkeypatch.setitem(sys.modules, 'torch', torch)
    monkeypatch.setitem(sys.modules, 'diffusers', diffusers)
    monkeypatch.setattr('pixcli.image._memory_preflight', lambda *args, **kwargs: {})
    monkeypatch.setattr('pixcli.image._ImageMemoryGuard', Guard)
    monkeypatch.setattr('pixcli.image._assert_lora_coverage', lambda *args: {})
    settings = ImageSettings(prompt='a knight', out=tmp_path / 'out.png', style='retro', style_prompt='sharp outlines', weight_storage='memory')
    _infer_offline(settings, {key: tmp_path for key in ('base', 'vae', 'lora')},
                   Image.new('RGB', (2, 2)) if editing else None)
    assert seen == ['a knight, sharp outlines']

@pytest.mark.parametrize('command', ['gen', 'edit', 'animate'])
@pytest.mark.parametrize('override', [False, True])
def test_style_palette_selection_and_explicit_override(tmp_path, monkeypatch, command, override):
    from pixcli import animation, cli, models

    monkeypatch.chdir(tmp_path)
    (tmp_path / 'styles').mkdir()
    (tmp_path / 'styles' / 'retro.txt').write_text('palette: soft.hex\n\nsharp outlines\nmuted shading\n', encoding='utf-8')
    (tmp_path / 'palettes').mkdir()
    (tmp_path / 'palettes' / 'soft.hex').write_text('000000\nffffff\n')
    (tmp_path / 'palettes' / 'custom.hex').write_text('ff0000\n0000ff\n')
    source = tmp_path / 'source.png'
    Image.new('RGB', (8, 8)).save(source)
    seen = []
    monkeypatch.setattr(cli, '_run_image', lambda settings, cache: seen.append(settings))
    monkeypatch.setattr(animation, 'run_animation', lambda settings, paths, progress: seen.append(settings) or {})
    monkeypatch.setattr(models, 'list_models', lambda profile=None: {})
    extra = [] if command == 'gen' else ['--input', str(source)]
    if command == 'animate':
        extra += ['--no-mp4']
    if override:
        extra += ['--palette', 'custom']
    out = tmp_path / ('out' if command == 'animate' else 'out.png')
    result = CliRunner().invoke(app, [command, *extra, '--prompt', 'a knight', '--out', str(out), '--style', 'retro'])
    assert result.exit_code == 0, result.output
    assert seen[0].style_prompt == 'sharp outlines\nmuted shading'
    assert seen[0].palette == ('custom' if override else 'soft.hex')


@pytest.mark.parametrize('content', ['palette: \nsharp outlines', 'palette: soft\n\n'])
def test_style_rejects_empty_palette_reference_or_prompt(tmp_path, monkeypatch, content):
    monkeypatch.chdir(tmp_path)
    (tmp_path / 'styles').mkdir()
    (tmp_path / 'styles' / 'bad.txt').write_text(content, encoding='utf-8')
    result = CliRunner().invoke(app, ['gen', '--prompt', 'x', '--out', 'out.png', '--style', 'bad'])
    assert result.exit_code != 0
    assert 'style' in result.output.lower()

@pytest.mark.parametrize('command', ['gen', 'edit', 'animate'])
def test_missing_style_palette_fails_before_models(tmp_path, monkeypatch, command):
    from pixcli import models
    monkeypatch.chdir(tmp_path)
    (tmp_path / 'styles').mkdir()
    (tmp_path / 'styles' / 'retro.txt').write_text('palette: missing\nsharp outlines\n', encoding='utf-8')
    source = tmp_path / 'source.png'
    Image.new('RGB', (8, 8)).save(source)
    monkeypatch.setattr(models, 'model_path', lambda *a, **k: pytest.fail('model lookup should not occur'))
    extra = [] if command == 'gen' else ['--input', str(source)]
    if command == 'animate':
        extra += ['--no-mp4']
    result = CliRunner().invoke(app, [command, *extra, '--prompt', 'x', '--out', str(tmp_path / ('out' if command == 'animate' else 'out.png')), '--style', 'retro'])
    assert result.exit_code != 0
    assert 'palette not found' in result.output.lower()


def test_explicit_palette_override_does_not_require_style_default(tmp_path, monkeypatch):
    from pixcli import cli
    monkeypatch.chdir(tmp_path)
    (tmp_path / 'styles').mkdir()
    (tmp_path / 'styles' / 'retro.txt').write_text('palette: missing\r\nsharp outlines\r\n', encoding='utf-8', newline='')
    (tmp_path / 'palettes').mkdir()
    (tmp_path / 'palettes' / 'valid.hex').write_text('000000\nffffff\n')
    seen = []
    monkeypatch.setattr(cli, '_run_image', lambda settings, cache: seen.append(settings))
    result = CliRunner().invoke(app, ['gen', '--prompt', 'x', '--out', 'out.png', '--style', 'retro', '--palette', 'valid'])
    assert result.exit_code == 0, result.output
    assert seen[0].palette == 'valid'
    assert seen[0].style_prompt == 'sharp outlines'


@pytest.mark.parametrize('reference', ['../outside', 'foo.png', 'foo/bar'])
def test_style_rejects_malformed_palette_name(tmp_path, monkeypatch, reference):
    monkeypatch.chdir(tmp_path)
    (tmp_path / 'styles').mkdir()
    (tmp_path / 'styles' / 'bad.txt').write_text(f'palette: {reference}\nsharp outlines\n', encoding='utf-8')
    result = CliRunner().invoke(app, ['gen', '--prompt', 'x', '--out', 'out.png', '--style', 'bad'])
    assert result.exit_code != 0
    assert 'invalid palette name' in result.output.lower()


@pytest.mark.parametrize('command,conflict', [('gen', ['--palette-colors', '2']), ('animate', ['--palette-colors', '2']), ('animate', ['--palette-image', 'other.png'])])
def test_style_palette_conflicts_with_other_palette_controls(tmp_path, monkeypatch, command, conflict):
    from pixcli import models
    monkeypatch.chdir(tmp_path)
    (tmp_path / 'styles').mkdir()
    (tmp_path / 'styles' / 'retro.txt').write_text('palette: soft\nsharp outlines\n')
    (tmp_path / 'palettes').mkdir()
    (tmp_path / 'palettes' / 'soft.hex').write_text('000000\nffffff\n')
    source = tmp_path / 'source.png'
    Image.new('RGB', (8, 8)).save(source)
    monkeypatch.setattr(models, 'model_path', lambda *a, **k: pytest.fail('model lookup should not occur'))
    extra = [] if command == 'gen' else ['--input', str(source), '--no-mp4']
    result = CliRunner().invoke(app, [command, *extra, *conflict, '--prompt', 'x', '--out', str(tmp_path / ('out' if command == 'animate' else 'out.png')), '--style', 'retro'])
    assert result.exit_code != 0
    assert 'palette' in result.output.lower()


def test_style_palette_applies_to_generated_image_and_is_recorded(tmp_path, monkeypatch):
    from pixcli import models
    monkeypatch.chdir(tmp_path)
    (tmp_path / 'styles').mkdir()
    (tmp_path / 'styles' / 'retro.txt').write_text('palette: soft\nsharp outlines\n')
    (tmp_path / 'palettes').mkdir()
    (tmp_path / 'palettes' / 'soft.hex').write_text('ff0000\n0000ff\n')
    seen = []
    def infer(settings, paths, image):
        seen.append(settings.style_prompt)
        return Image.new('RGB', (8, 8), (240, 10, 10)), {}
    monkeypatch.setattr('pixcli.image._infer', infer)
    monkeypatch.setattr(models, 'model_path', lambda *args, **kwargs: tmp_path)
    result = CliRunner().invoke(app, ['gen', '--prompt', 'x', '--out', 'sprite.png', '--style', 'retro'])
    assert result.exit_code == 0, result.output
    assert seen == ['sharp outlines']
    assert Image.open('sprite.png').convert('RGB').getpixel((0, 0)) == (255, 0, 0)
    metadata = json.loads((tmp_path / 'sprite.json').read_text(encoding='utf-8'))
    assert metadata['settings']['palette'] == 'soft'
    assert metadata['settings']['style_prompt'] == 'sharp outlines'
