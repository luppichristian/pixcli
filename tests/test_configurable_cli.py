"""End-to-end selection through the public CLI without large model weights."""
import json

from PIL import Image
from typer.testing import CliRunner

from pixcli.cli import app


runner = CliRunner()


def _config(tmp_path):
    first = tmp_path / 'image-one'
    second = tmp_path / 'image-two'
    first.mkdir()
    second.mkdir()
    (first / 'model_index.json').write_text('{}', encoding='utf-8')
    (second / 'model_index.json').write_text('{}', encoding='utf-8')
    (first / 'weights.safetensors').write_bytes(b'first')
    (second / 'weights.safetensors').write_bytes(b'second')
    config = tmp_path / 'models.toml'
    config.write_text('''version = 1
[defaults]
generation = "first"
editing = "second"
animation = "default-animation"
[models.first]
backend = "diffusers"
[models.first.source]
type = "local"
path = "image-one"
[models.second]
backend = "diffusers"
[models.second.source]
type = "local"
path = "image-two"
''', encoding='utf-8')
    return config, first, second


def test_gen_edit_and_override_select_configured_local_models(tmp_path, monkeypatch):
    from pixcli import image

    config, first, second = _config(tmp_path)
    source = tmp_path / 'input.png'
    Image.new('RGB', (8, 8)).save(source)
    calls = []

    def run(settings, paths, model_spec=None):
        calls.append((settings.input_path, paths['base'], model_spec['name']))
        return {'ok': True}

    monkeypatch.setattr(image, 'run_image', run)
    common = ['-prompt', 'sprite', '-out', str(tmp_path / 'out.png'), '-config', str(config)]
    for args in (['gen', *common], ['edit', '-input', str(source), *common],
                 ['gen', *common, '-model', 'second']):
        result = runner.invoke(app, args)
        assert result.exit_code == 0, result.output
    assert calls == [(None, first, 'first'), (source, second, 'second'), (None, second, 'second')]


def test_configured_status_and_uninstall_preserve_external_models(tmp_path):
    config, first, _ = _config(tmp_path)
    marker = first / 'keep.txt'
    marker.write_text('owned by user', encoding='utf-8')
    common = ['-config', str(config), '-cache-dir', str(tmp_path / 'cache')]
    result = runner.invoke(app, ['models', 'status', 'first', *common, '-json'])
    assert result.exit_code == 0, result.output
    assert json.loads(result.stdout)['name'] == 'first'
    result = runner.invoke(app, ['models', 'uninstall', 'first', *common])
    assert result.exit_code == 0, result.output
    assert marker.read_text(encoding='utf-8') == 'owned by user'


def test_install_of_available_local_model_can_run_offline(tmp_path):
    config, first, _ = _config(tmp_path)
    result = runner.invoke(app, ['models', 'install', 'first', '-offline', '-config', str(config),
                                 '-cache-dir', str(tmp_path / 'cache'), '-json'])
    assert result.exit_code == 0, result.output
    report = json.loads(result.stdout)
    assert report['installed'] and not report['managed']
    assert first.exists()


def test_unknown_model_fails_before_inference(tmp_path):
    config, _, _ = _config(tmp_path)
    result = runner.invoke(app, ['gen', '-prompt', 'sprite', '-out', str(tmp_path / 'out.png'),
                                 '-config', str(config), '-model', 'missing'])
    assert result.exit_code != 0
    assert 'missing' in result.output.lower()


def test_obsolete_managed_installations_are_listed_and_removable(tmp_path, monkeypatch):
    from pixcli import model_store

    removed = []
    monkeypatch.setattr(model_store, 'installed', lambda cache: [
        {'id': 'a' * 64, 'name': 'previous-definition', 'size_bytes': 123},
    ])
    monkeypatch.setattr(model_store, 'uninstall_id', lambda ident, cache: removed.append(ident) or {'removed': True})
    cache = str(tmp_path / 'cache')
    result = runner.invoke(app, ['models', 'status', '-installed', '-cache-dir', cache, '-json'])
    assert result.exit_code == 0, result.output
    assert json.loads(result.stdout)[0]['name'] == 'previous-definition'
    result = runner.invoke(app, ['models', 'uninstall-id', 'a' * 64, '-cache-dir', cache])
    assert result.exit_code == 0, result.output
    assert removed == ['a' * 64]


def test_animation_forwards_selected_wan_model_and_checks_quantization(tmp_path, monkeypatch):
    from pixcli import animation, model_store

    source = tmp_path / 'source.png'
    Image.new('RGB', (8, 8)).save(source)
    (tmp_path / 'high.gguf').touch()
    config = tmp_path / 'models.toml'
    config.write_text('''version = 1
[defaults]
animation = "custom-video"
[models.custom-video]
backend = "wan-i2v"
text_encoder = "fp8"
expert_quant = "q3"
[models.custom-video.components.wan_high]
type = "local"
path = "high.gguf"
''', encoding='utf-8')
    seen = {}

    def resolve(selected, cache_dir):
        seen['selection'] = selected
        return {}, {'sources': {'wan_high': 'custom'}}

    def run(settings, paths, progress, model_spec=None):
        seen['settings'], seen['spec'] = settings, model_spec
        return {'ok': True}

    monkeypatch.setattr(model_store, 'resolve', resolve)
    monkeypatch.setattr(animation, 'run_animation', run)
    args = ['animate', '-input', str(source), '-prompt', 'move', '-out', str(tmp_path / 'out'),
            '-no-mp4', '-config', str(config)]
    result = runner.invoke(app, args)
    assert result.exit_code == 0, result.output
    assert (seen['settings'].text_encoder, seen['settings'].expert_quant) == ('fp8', 'q3')
    assert seen['spec']['name'] == 'custom-video'
    assert seen['spec']['provenance']['sources']['wan_high'] == 'custom'
    result = runner.invoke(app, [*args, '-text-encoder', 'q3'])
    assert result.exit_code != 0
    assert 'conflict' in result.output.lower()
