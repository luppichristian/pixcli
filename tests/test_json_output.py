from PIL import Image
import pytest

from pixcli.image import ImageSettings, run_image, validate_image
from pixcli.animation import AnimationSettings, run_animation


@pytest.mark.parametrize('save_json', [False, True])
def test_image_json_is_opt_in(tmp_path, monkeypatch, save_json):
    output = tmp_path / 'out.png'
    settings = ImageSettings(prompt='sprite', out=output, save_json=save_json)
    monkeypatch.setattr('pixcli.image._infer', lambda *args: (Image.new('RGB', (8, 8)), {}))
    run_image(settings, {})
    assert output.exists()
    assert output.with_suffix('.json').exists() is save_json


def test_unrequested_sidecar_does_not_block_or_change_image_output(tmp_path, monkeypatch):
    output = tmp_path / 'out.png'
    sidecar = output.with_suffix('.json')
    sidecar.write_text('keep', encoding='utf-8')
    with pytest.raises(FileExistsError):
        validate_image(ImageSettings(prompt='sprite', out=output, save_json=True))
    monkeypatch.setattr('pixcli.image._infer', lambda *args: (Image.new('RGB', (8, 8)), {}))
    run_image(ImageSettings(prompt='sprite', out=output), {})
    assert sidecar.read_text(encoding='utf-8') == 'keep'


@pytest.mark.parametrize('save_json', [False, True])
def test_animation_json_is_opt_in(tmp_path, monkeypatch, save_json):
    source = tmp_path / 'source.png'
    Image.new('RGB', (8, 8)).save(source)
    settings = AnimationSettings(input_path=source, prompt='walk', out=tmp_path / 'out',
                                 frames=5, pixel_width=8, pixel_height=8, mp4=False, save_json=save_json)
    monkeypatch.setattr('pixcli.animation.run_comfy', lambda *args, **kwargs: ([Image.new('RGB', (8, 8))] * 5, {}))
    result = run_animation(settings, {})
    assert (settings.out / 'sheet.png').exists()
    assert (settings.out / 'metadata.json').exists() is save_json
    assert (settings.out / 'sheet.json').exists() is save_json
    assert (result['sheet_json'] is not None) is save_json
