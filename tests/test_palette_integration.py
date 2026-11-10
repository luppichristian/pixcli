import json

import pytest
from PIL import Image
from typer.testing import CliRunner

from pixcli.animation import AnimationSettings, run_animation, validate_animation_job
from pixcli.cli import app
from pixcli.image import ImageSettings, run_image, validate_image


@pytest.fixture
def palette_dir(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    (tmp_path / 'palettes').mkdir()
    (tmp_path / 'palettes' / 'two.hex').write_text('ff0000\n0000ff\n', encoding='utf-8')
    return tmp_path


def test_generation_and_masked_edit_apply_named_palette(palette_dir, monkeypatch):
    monkeypatch.setattr('pixcli.image._infer', lambda settings, paths, image: (Image.new('RGB', image.size if image else (8, 8), (240, 10, 10)), {}))
    out = palette_dir / 'gen.png'
    meta = run_image(ImageSettings(prompt='x', out=out, pixel_width=2, pixel_height=2, palette='two'), {})
    assert Image.open(out).convert('RGB').getpixel((0, 0)) == (255, 0, 0)
    assert meta['settings']['palette'] == 'two'
    source = palette_dir / 'source.png'
    mask = palette_dir / 'mask.png'
    Image.new('RGBA', (2, 1), (12, 34, 56, 80)).save(source)
    m = Image.new('L', (2, 1), 0)
    m.putpixel((1, 0), 255)
    m.save(mask)
    out = palette_dir / 'edit.png'
    run_image(ImageSettings(prompt='x', out=out, input_path=source, mask_path=mask, palette='two'), {})
    result = Image.open(out)
    assert result.getpixel((0, 0)) == (12, 34, 56, 80)
    assert result.getpixel((1, 0)) == (255, 0, 0, 80)


def test_animation_fixed_palette_applies_to_frames_and_sheet(palette_dir, monkeypatch):
    source = palette_dir / 'source.png'
    Image.new('RGB', (2, 2)).save(source)
    monkeypatch.setattr('pixcli.animation.run_comfy', lambda *a, **k: ([Image.new('RGB', (2, 2), (240, 10, 10))] * 5, {}))
    settings = AnimationSettings(input_path=source, prompt='walk', out=palette_dir / 'anim', pixel_width=2, pixel_height=2, mp4=False, palette='two')
    run_animation(settings, {})
    assert Image.open(settings.out / 'frames/000000.png').convert('RGB').getpixel((0, 0)) == (255, 0, 0)
    assert Image.open(settings.out / 'sheet.png').getpixel((0, 0)) == (255, 0, 0)
    assert Image.open(settings.out / 'original_frames/000000.png').getpixel((0, 0)) == (240, 10, 10)
    assert json.loads((settings.out / 'metadata.json').read_text())['settings']['palette'] == 'two'


def test_palette_validation_and_conflicts_before_inference(palette_dir):
    with pytest.raises((ValueError, FileNotFoundError)):
        validate_image(ImageSettings(prompt='x', out=palette_dir / 'x.png', palette='missing'))
    with pytest.raises(ValueError, match='palette'):
        validate_image(ImageSettings(prompt='x', out=palette_dir / 'x.png', palette='two', palette_colors=2))
    source = palette_dir / 'source.png'
    Image.new('RGB', (2, 2)).save(source)
    with pytest.raises(ValueError, match='palette'):
        validate_animation_job(AnimationSettings(input_path=source, prompt='x', out=palette_dir / 'out', palette='two', palette_colors=2, mp4=False))


@pytest.mark.parametrize('command', ['gen', 'edit', 'animate'])
def test_cli_passes_palette_option(palette_dir, monkeypatch, command):
    from pixcli import animation, cli, models
    seen = []
    monkeypatch.setattr(cli, '_run_image', lambda settings, cache: seen.append(settings))
    monkeypatch.setattr(animation, 'run_animation', lambda settings, paths, progress: seen.append(settings) or {})
    monkeypatch.setattr(models, 'list_models', lambda profile=None: {})
    source = palette_dir / 'source.png'
    Image.new('RGB', (2, 2)).save(source)
    extra = [] if command == 'gen' else ['--input', str(source)]
    if command == 'animate':
        extra += ['--no-mp4']
    output = palette_dir / ('out' if command == 'animate' else 'out.png')
    result = CliRunner().invoke(app, [command, *extra, '--prompt', 'x', '--out', str(output), '--palette', 'two'])
    assert result.exit_code == 0, result.output
    assert seen[0].palette == 'two'


@pytest.mark.parametrize('kind', ['palette', 'style'])
def test_animation_does_not_replace_selected_customization_directory(palette_dir, kind):
    source = palette_dir / 'source.png'
    Image.new('RGB', (2, 2)).save(source)
    directory = palette_dir / ('palettes' if kind == 'palette' else 'styles')
    directory.mkdir(exist_ok=True)
    (directory / '.pixcli-animation').write_text('pixcli animation output')
    kwargs = {'palette': 'two'} if kind == 'palette' else {'style': 'retro', 'style_prompt': 'retro'}
    with pytest.raises(ValueError, match='output directory'):
        validate_animation_job(AnimationSettings(input_path=source, prompt='x', out=directory, mp4=False, overwrite=True, **kwargs))
