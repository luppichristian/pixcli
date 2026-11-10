import json

import pytest
from PIL import Image
from typer.testing import CliRunner

from pixcli.cli import app


@pytest.mark.parametrize('command', ['gen', 'edit', 'animate'])
def test_style_palette_affects_output_and_metadata(tmp_path, monkeypatch, command):
    from pixcli import models

    monkeypatch.chdir(tmp_path)
    (tmp_path / 'styles').mkdir()
    (tmp_path / 'palettes').mkdir()
    (tmp_path / 'styles' / 'retro.txt').write_text('palette: red\n\ncrisp pixels\n', encoding='utf-8')
    (tmp_path / 'palettes' / 'red.hex').write_text('ff0000\n', encoding='utf-8')
    source = tmp_path / 'source.png'
    Image.new('RGB', (2, 2), 'blue').save(source)
    monkeypatch.setattr(models, 'model_path', lambda *args, **kwargs: tmp_path)
    monkeypatch.setattr(models, 'list_models', lambda *args, **kwargs: {})
    monkeypatch.setattr('pixcli.image._infer', lambda s, paths, image: (Image.new('RGB', image.size if image else (8, 8), (240, 20, 10)), {}))
    monkeypatch.setattr('pixcli.animation.run_comfy', lambda *a, **k: ([Image.new('RGB', (2, 2), (240, 20, 10))] * 5, {}))
    output = tmp_path / ('out' if command == 'animate' else 'out.png')
    extra = [] if command == 'gen' else ['--input', str(source)]
    if command == 'animate':
        extra += ['--no-mp4', '--pixel-width', '2', '--pixel-height', '2']
    result = CliRunner().invoke(app, [command, *extra, '--prompt', 'a knight', '--out', str(output), '--style', 'retro'])
    assert result.exit_code == 0, result.output
    image_path = output / 'frames/000000.png' if command == 'animate' else output
    meta_path = output / 'metadata.json' if command == 'animate' else output.with_suffix('.json')
    assert Image.open(image_path).convert('RGB').getpixel((0, 0)) == (255, 0, 0)
    metadata = json.loads(meta_path.read_text(encoding='utf-8'))
    assert metadata['settings']['palette'] == 'red'
    assert metadata['settings']['style_prompt'] == 'crisp pixels'
    assert metadata['fixed_palette'] == [[255, 0, 0]]
