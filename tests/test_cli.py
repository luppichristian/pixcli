from typer.testing import CliRunner
from pixcli.cli import app

runner = CliRunner()

def test_image_cli_forwards_memory_budgets(tmp_path, monkeypatch):
    from PIL import Image
    from pixcli import cli
    source = tmp_path / 'input.png'
    Image.new('RGB', (8, 8)).save(source)
    seen = []
    monkeypatch.setattr(cli, '_run_image', lambda settings, cache: seen.append(settings))
    for command, extra in [('gen', []), ('edit', ['--input', str(source)])]:
        result = runner.invoke(app, [command, *extra, '--prompt', 'cat', '--out', str(tmp_path / 'out.png'),
                                     '--ram-budget-gib', '8', '--vram-budget-gib', '4', '--weight-storage', 'disk',
                                     '--min-free-commit-gib', '1.25', '--min-free-physical-gib', '1.5'])
        assert result.exit_code == 0, result.output
    assert [(s.ram_budget_gib, s.vram_budget_gib) for s in seen] == [(8, 4), (8, 4)]
    assert [s.weight_storage for s in seen] == ['disk', 'disk']
    assert [(s.min_free_commit_gib, s.min_free_physical_gib) for s in seen] == [(1.25, 1.5), (1.25, 1.5)]

def test_image_cli_forwards_explicit_cpu_device(tmp_path, monkeypatch):
    from PIL import Image
    from pixcli import cli
    source = tmp_path / 'input.png'
    Image.new('RGB', (8, 8)).save(source)
    seen = []
    monkeypatch.setattr(cli, '_run_image', lambda settings, cache: seen.append(settings))
    for command, extra in [('gen', []), ('edit', ['--input', str(source)])]:
        result = runner.invoke(app, [command, *extra, '--prompt', 'cat', '--out', str(tmp_path / 'out.png'), '--device', 'cpu'])
        assert result.exit_code == 0, result.output
    assert [s.device for s in seen] == ['cpu', 'cpu']

def test_animation_cli_forwards_explicit_cpu_device(tmp_path, monkeypatch):
    from PIL import Image
    from pixcli import animation, models
    source = tmp_path / 'input.png'
    Image.new('RGB', (8, 8)).save(source)
    seen = []
    monkeypatch.setattr(animation, 'run_animation', lambda settings, paths, progress: seen.append(settings) or {})
    monkeypatch.setattr(models, 'list_models', lambda profile=None: {})
    result = runner.invoke(app, ['animate', '--input', str(source), '--prompt', 'walk', '--out', str(tmp_path / 'out'), '--no-mp4', '--device', 'cpu', '--timeout-seconds', '7200'])
    assert result.exit_code == 0, result.output
    assert seen[0].device == 'cpu'
    assert seen[0].timeout_seconds == 7200

def test_animation_cli_forwards_resource_budgets(tmp_path, monkeypatch):
    from PIL import Image
    from pixcli import animation, models
    source = tmp_path / 'input.png'
    Image.new('RGB', (8, 8)).save(source)
    seen = []
    monkeypatch.setattr(animation, 'run_animation', lambda settings, paths, progress: seen.append(settings) or {})
    monkeypatch.setattr(models, 'list_models', lambda profile=None: {})
    result = runner.invoke(app, ['animate', '--input', str(source), '--prompt', 'walk', '--out', str(tmp_path/'out'),
                                 '--no-mp4', '--ram-budget-gib', '10', '--vram-budget-gib', '5'])
    assert result.exit_code == 0, result.output
    assert (seen[0].ram_budget_gib, seen[0].vram_budget_gib) == (10, 5)

def test_gen_rejects_invalid_size_before_loading_models(tmp_path):
    result = runner.invoke(app, ['gen', '--prompt', 'cat', '--out', str(tmp_path / 'cat.png'), '--inference-width', '127'])
    assert result.exit_code != 0
    assert 'multiples of 8' in result.output.lower()

def test_edit_requires_existing_source(tmp_path):
    result = runner.invoke(app, ['edit', '--input', str(tmp_path / 'missing.png'), '--prompt', 'cat', '--out', str(tmp_path / 'cat.png')])
    assert result.exit_code != 0
    assert 'missing.png' in result.output.lower()

def test_offline_download_is_rejected():
    result = runner.invoke(app, ['models', 'download', '--profile', 'image', '--offline'])
    assert result.exit_code != 0
    assert 'offline' in result.output.lower()



def test_windows_11_build_is_reported_as_windows_11(monkeypatch):
    from pixcli import cli
    monkeypatch.setattr(cli.sys, 'platform', 'win32')
    monkeypatch.setattr(cli.platform, 'version', lambda: '10.0.26200')
    monkeypatch.setattr(cli.platform, 'platform', lambda: 'Windows-10-10.0.26200-SP0')
    assert 'Windows 11' in cli._os_description()

def test_animation_rejects_unsupported_steps_before_model_lookup(tmp_path):
    from PIL import Image
    source = tmp_path / 'input.png'
    Image.new('RGB', (8, 8)).save(source)
    result = runner.invoke(app, ['animate', '--input', str(source), '--prompt', 'walk', '--out', str(tmp_path / 'out'), '--steps', '5'])
    assert result.exit_code != 0
    assert 'exactly 4 steps' in result.output

def test_animation_rejects_unknown_text_encoder_before_model_loading(tmp_path):
    from PIL import Image
    source = tmp_path / 'input.png'
    Image.new('RGB', (8, 8)).save(source)
    result = runner.invoke(app, ['animate', '--input', str(source), '--prompt', 'walk', '--out', str(tmp_path/'out'), '--text-encoder', 'q4'])
    assert result.exit_code != 0
    assert 'text encoder must be fp8 or q3' in result.output

def test_animation_cli_forwards_memory_thresholds(tmp_path, monkeypatch):
    from PIL import Image
    from pixcli import animation, models
    source = tmp_path / 'input.png'
    Image.new('RGB', (8, 8)).save(source)
    seen = {}
    def fake_run(settings, paths, progress):
        seen.update(max_private=settings.max_private_gib, commit=settings.min_free_commit_gib,
                    physical=settings.min_free_physical_gib)
        return {'ok': True}
    monkeypatch.setattr(animation, 'run_animation', fake_run)
    monkeypatch.setattr(models, 'list_models', lambda profile=None: {})
    result = runner.invoke(app, ['animate', '--input', str(source), '--prompt', 'walk', '--out', str(tmp_path/'out'),
                                 '--no-mp4', '--max-private-gib', '16', '--min-free-commit-gib', '3',
                                 '--min-free-physical-gib', '4'])
    assert result.exit_code == 0, result.output
    assert seen == {'max_private': 16, 'commit': 3, 'physical': 4}

def test_animation_cli_rejects_too_low_memory_reserve(tmp_path):
    from PIL import Image
    source = tmp_path / 'input.png'
    Image.new('RGB', (8, 8)).save(source)
    result = runner.invoke(app, ['animate', '--input', str(source), '--prompt', 'walk', '--out', str(tmp_path/'out'),
                                 '--min-free-commit-gib', '0.5'])
    assert result.exit_code != 0
    assert 'memory' in result.output.lower()

def test_animation_cli_defaults_match_verified_profile(tmp_path, monkeypatch):
    from PIL import Image
    from pixcli import animation, models
    source = tmp_path / 'input.png'
    Image.new('RGB', (8, 8)).save(source)
    seen = {}
    def fake_run(settings, paths, progress):
        seen['settings'] = settings
        return {'ok': True}
    def fake_list(profile=None):
        seen['profile'] = profile
        return {}
    monkeypatch.setattr(animation, 'run_animation', fake_run)
    monkeypatch.setattr(models, 'list_models', fake_list)
    result = runner.invoke(app, ['animate', '--input', str(source), '--prompt', 'walk', '--out', str(tmp_path/'out'), '--no-mp4'])
    assert result.exit_code == 0, result.output
    settings = seen['settings']
    assert (settings.text_encoder, settings.expert_quant, settings.weight_storage) == ('q3', 'q2', 'mmap')
    assert (settings.inference_width, settings.inference_height, settings.frames, settings.max_private_gib) == (256, 256, 5, None)
    assert seen['profile'] == 'animation'

def test_default_animation_profile_matches_verified_models():
    from pixcli import models
    assert set(models.list_models('animation')) == set(models.list_models('animation-q2-q3-text'))
    assert models.animation_profile('q3', 'q2') == 'animation'
    assert set(models.list_models('animation-q3-fp8')) == {'wan_high', 'wan_low', 'wan_vae', 'wan_text', 'wan_distill_high', 'wan_distill_low', 'wan_pixel'}
