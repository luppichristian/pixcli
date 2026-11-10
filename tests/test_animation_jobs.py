import json
from pathlib import Path
from PIL import Image
import pytest
from pixcli.animation import AnimationSettings, run_animation, validate_animation_job


def test_job_exports_frames_and_metadata(tmp_path, monkeypatch):
    source = tmp_path / 'source.png'
    Image.new('RGB', (8, 8), 'red').save(source)
    settings = AnimationSettings(input_path=source, prompt='walk', out=tmp_path / 'walk', frames=5, pixel_width=8, pixel_height=8, mp4=False, save_json=True)
    monkeypatch.setattr('pixcli.animation.run_comfy', lambda *a, **kw: ([Image.new('RGB', (8, 8), 'red')] * 5, {'adapter_coverage': []}))
    result = run_animation(settings, {})
    metadata = json.loads((settings.out / 'metadata.json').read_text(encoding='utf-8'))
    assert metadata['settings']['prompt'] == 'walk'
    assert metadata['final_dimensions'] == [8, 8]
    assert metadata['input_sha256']
    assert Path(result['sheet']).exists()
    with pytest.raises(FileExistsError):
        validate_animation_job(settings)


def test_animation_applies_residency_limit_to_complete_job(tmp_path, monkeypatch):
    from contextlib import contextmanager
    source = tmp_path / 'source.png'
    Image.new('RGB', (8, 8)).save(source)
    settings = AnimationSettings(input_path=source, prompt='walk', out=tmp_path/'out',
                                 frames=5, pixel_width=8, pixel_height=8, mp4=False, ram_budget_gib=8, save_json=True)
    active = []
    @contextmanager
    def limit(value):
        active.append(value)
        try:
            yield True
        finally:
            active.pop()
    monkeypatch.setattr('pixcli.residency.resident_memory_limit', limit)
    def infer(*args, **kwargs):
        assert active == [8]
        return [Image.new('RGB', (8, 8))] * 5, {}
    monkeypatch.setattr('pixcli.animation.run_comfy', infer)
    run_animation(settings, {})
    assert not active
    metadata = json.loads((settings.out/'metadata.json').read_text())
    assert metadata['runtime']['command_residency_limit_applied'] is True


def test_animation_rejects_invalid_frames_and_input_inside_output(tmp_path):
    source = tmp_path / 'source.png'
    Image.new('RGB', (8, 8)).save(source)
    with pytest.raises(ValueError, match='4n'):
        validate_animation_job(AnimationSettings(input_path=source, prompt='x', out=tmp_path / 'out', frames=6))
    with pytest.raises(ValueError, match='input'):
        validate_animation_job(AnimationSettings(input_path=source, prompt='x', out=tmp_path, overwrite=True))

def test_supplied_palette_requires_color_count(tmp_path):
    source = tmp_path / 'source.png'
    palette = tmp_path / 'palette.png'
    Image.new('RGB', (8, 8)).save(source)
    Image.new('RGB', (2, 2), 'blue').save(palette)
    with pytest.raises(ValueError, match='palette'):
        validate_animation_job(AnimationSettings(input_path=source, prompt='x', out=tmp_path / 'out', palette_image=palette))

def test_animation_records_supplied_palette_hash(tmp_path, monkeypatch):
    import hashlib
    source, palette = tmp_path / 'source.png', tmp_path / 'palette.png'
    Image.new('RGB', (8, 8), 'red').save(source)
    Image.new('RGB', (4, 4), 'blue').save(palette)
    settings = AnimationSettings(input_path=source, prompt='walk', out=tmp_path / 'out', frames=5,
                                 pixel_width=8, pixel_height=8, palette_colors=4, palette_image=palette, mp4=False, save_json=True)
    monkeypatch.setattr('pixcli.animation.run_comfy', lambda *a, **kw: ([Image.new('RGB', (8, 8), 'red')] * 5, {}))
    run_animation(settings, {})
    metadata = json.loads((settings.out / 'metadata.json').read_text(encoding='utf-8'))
    assert metadata['palette_sha256'] == hashlib.sha256(palette.read_bytes()).hexdigest()
    assert json.loads((settings.out / 'sheet.json').read_text())['palette_source'] == 'image'

def test_animation_exports_mp4_by_default(tmp_path):
    settings = AnimationSettings(input_path=tmp_path / 'source.png', prompt='walk', out=tmp_path / 'out')
    assert settings.mp4 is True

def test_mp4_dependency_failure_occurs_before_inference(tmp_path, monkeypatch):
    from pixcli import animation
    source = tmp_path / 'source.png'
    Image.new('RGB', (8, 8)).save(source)
    monkeypatch.setattr(animation, '_check_mp4_dependencies', lambda: (_ for _ in ()).throw(RuntimeError('ffmpeg unavailable')))
    with pytest.raises(RuntimeError, match='ffmpeg unavailable'):
        validate_animation_job(AnimationSettings(input_path=source, prompt='walk', out=tmp_path / 'out'))

def test_animation_metadata_records_effective_workflow(tmp_path, monkeypatch):
    source = tmp_path / 'source.png'
    Image.new('RGB', (8, 8)).save(source)
    settings = AnimationSettings(input_path=source, prompt='walk', out=tmp_path / 'out', frames=5, pixel_width=8, pixel_height=8, mp4=False, save_json=True)
    monkeypatch.setattr('pixcli.animation.run_comfy', lambda *a, **kw: ([Image.new('RGB', (8, 8), 'red')] * 5, {}))
    run_animation(settings, {})
    metadata = json.loads((settings.out / 'metadata.json').read_text(encoding='utf-8'))
    assert metadata['offline'] is True
    assert metadata['workflow']['sampler'] == 'ddim'
    assert metadata['workflow']['scheduler'] == 'simple'
    assert metadata['workflow']['high_noise_steps'] == [0, 2]
    assert metadata['workflow']['low_noise_steps'] == [2, 4]

def test_q3_text_encoder_is_explicitly_routed_and_recorded(tmp_path, monkeypatch):
    import pixcli.models
    source = tmp_path / 'source.png'
    Image.new('RGB', (8, 8)).save(source)
    settings = AnimationSettings(input_path=source, prompt='walk', out=tmp_path/'out', frames=5,
                                 pixel_width=8, pixel_height=8, mp4=False, text_encoder='q3', expert_quant='q3', save_json=True)
    seen = {}
    def fake_infer(*args, **kwargs):
        seen['encoder'] = kwargs['text_encoder']
        return [Image.new('RGB', (8, 8))] * 5, {}
    def fake_models(profile=None):
        seen['profile'] = profile
        return {}
    monkeypatch.setattr('pixcli.animation.run_comfy', fake_infer)
    monkeypatch.setattr(pixcli.models, 'list_models', fake_models)
    run_animation(settings, {})
    metadata = json.loads((settings.out / 'metadata.json').read_text(encoding='utf-8'))
    assert seen == {'encoder': 'q3', 'profile': 'animation-q3-text'}
    assert metadata['settings']['text_encoder'] == 'q3'
    assert 'Q3_K_S text' in metadata['quantization']

def test_q2_expert_selection_is_explicit_and_recorded(tmp_path, monkeypatch):
    import pixcli.models
    source = tmp_path / 'source.png'
    Image.new('RGB', (8, 8)).save(source)
    settings = AnimationSettings(input_path=source, prompt='walk', out=tmp_path/'out', frames=5,
                                 pixel_width=8, pixel_height=8, mp4=False, expert_quant='q2', text_encoder='fp8', save_json=True)
    seen = {}
    def fake_infer(*args, **kwargs):
        seen['expert_quant'] = kwargs['expert_quant']
        return [Image.new('RGB', (8, 8))] * 5, {}
    def fake_profile(text_encoder, expert_quant):
        seen['profile_args'] = (text_encoder, expert_quant)
        return 'animation-q2-experts'
    monkeypatch.setattr('pixcli.animation.run_comfy', fake_infer)
    monkeypatch.setattr(pixcli.models, 'animation_profile', fake_profile, raising=False)
    monkeypatch.setattr(pixcli.models, 'list_models', lambda profile=None: {})
    run_animation(settings, {})
    metadata = json.loads((settings.out/'metadata.json').read_text(encoding='utf-8'))
    assert seen == {'expert_quant': 'q2', 'profile_args': ('fp8', 'q2')}
    assert metadata['settings']['expert_quant'] == 'q2'
    assert metadata['model_profile'] == 'animation-q2-experts'
    assert 'Q2_K experts' in metadata['quantization']

def test_gpu_resident_requires_cpu_pinned_fp8_text(tmp_path):
    source = tmp_path / 'source.png'
    Image.new('RGB', (8, 8)).save(source)
    with pytest.raises(ValueError, match='fp8'):
        validate_animation_job(AnimationSettings(input_path=source, prompt='walk', out=tmp_path/'out', preset='gpu-resident', text_encoder='q3', mp4=False))
    validate_animation_job(AnimationSettings(input_path=source, prompt='walk', out=tmp_path/'out', preset='gpu-resident', text_encoder='fp8', mp4=False))

def test_cuda_adapter_requires_explicit_gpu_resident_preset(tmp_path):
    source = tmp_path / 'source.png'
    Image.new('RGB', (8, 8)).save(source)
    with pytest.raises(ValueError, match='gpu-resident'):
        validate_animation_job(AnimationSettings(input_path=source, prompt='walk', out=tmp_path/'out', adapter_device='cuda', mp4=False))
    validate_animation_job(AnimationSettings(input_path=source, prompt='walk', out=tmp_path/'out', preset='gpu-resident', text_encoder='fp8', adapter_device='cuda', mp4=False))

def test_memory_budget_passes_to_backend_and_metadata(tmp_path, monkeypatch):
    source = tmp_path / 'source.png'
    Image.new('RGB', (8, 8)).save(source)
    settings = AnimationSettings(input_path=source, prompt='walk', out=tmp_path/'out', frames=5,
                                 pixel_width=8, pixel_height=8, mp4=False, max_private_gib=16,
                                 min_free_commit_gib=3, min_free_physical_gib=4,
                                 ram_budget_gib=9, vram_budget_gib=6, timeout_seconds=5000, save_json=True)
    seen = {}
    def fake_infer(*args, **kwargs):
        seen.update(kwargs)
        return [Image.new('RGB', (8, 8))] * 5, {}
    monkeypatch.setattr('pixcli.animation.run_comfy', fake_infer)
    run_animation(settings, {})
    recorded = json.loads((settings.out/'metadata.json').read_text())['settings']
    for key, value in [('max_private_gib', 16), ('min_free_commit_gib', 3), ('min_free_physical_gib', 4),
                       ('ram_budget_gib', 9), ('vram_budget_gib', 6), ('timeout_seconds', 5000)]:
        assert seen[key] == value
        assert recorded[key] == value

@pytest.mark.parametrize('key,value', [
    ('max_private_gib', 0), ('max_private_gib', float('nan')),
    ('min_free_commit_gib', 0.5), ('min_free_physical_gib', float('inf')),
    ('ram_budget_gib', 0), ('vram_budget_gib', float('nan')),
])
def test_animation_rejects_invalid_memory_budgets(tmp_path, key, value):
    source = tmp_path / 'source.png'
    Image.new('RGB', (8, 8)).save(source)
    with pytest.raises(ValueError, match='memory'):
        validate_animation_job(AnimationSettings(input_path=source, prompt='walk', out=tmp_path/'out', mp4=False, **{key: value}))


@pytest.mark.parametrize('timeout', [0, -1, float('nan'), float('inf')])
def test_animation_rejects_invalid_timeout(tmp_path, timeout):
    source = tmp_path / 'source.png'
    Image.new('RGB', (8, 8)).save(source)
    with pytest.raises(ValueError, match='timeout'):
        validate_animation_job(AnimationSettings(input_path=source, prompt='walk', out=tmp_path/'out', mp4=False, timeout_seconds=timeout))
def test_animation_checks_free_commit_before_inference_even_without_ram_budget(tmp_path, monkeypatch):
    from types import SimpleNamespace
    from PIL import Image
    from pixcli import animation, comfy_backend
    import psutil
    source = tmp_path / 'source.png'
    Image.new('RGB', (8, 8)).save(source)
    settings = animation.AnimationSettings(input_path=source, prompt='walk', out=tmp_path / 'out', mp4=False)
    monkeypatch.setattr(psutil, 'virtual_memory', lambda: SimpleNamespace(available=8 * 1024**3))
    monkeypatch.setattr(comfy_backend, '_free_commit_bytes', lambda: 1024**3)
    monkeypatch.setattr(animation, 'run_comfy', lambda *args, **kwargs: (_ for _ in ()).throw(AssertionError('inference started')))
    with pytest.raises(MemoryError, match='commit memory'):
        animation.run_animation(settings, {})


def test_animation_checks_free_commit_during_export_without_ram_budget(tmp_path, monkeypatch):
    from types import SimpleNamespace
    from PIL import Image
    from pixcli import animation, comfy_backend
    import psutil
    source = tmp_path / 'source.png'
    Image.new('RGB', (8, 8)).save(source)
    settings = animation.AnimationSettings(input_path=source, prompt='walk', out=tmp_path / 'out',
                                           pixel_width=8, pixel_height=8, mp4=False)
    monkeypatch.setattr(psutil, 'virtual_memory', lambda: SimpleNamespace(available=8 * 1024**3))
    values = iter([8 * 1024**3, 8 * 1024**3, 1024**3])
    monkeypatch.setattr(comfy_backend, '_free_commit_bytes', lambda: next(values))
    monkeypatch.setattr(animation, 'run_comfy', lambda *args, **kwargs: ([Image.new('RGB', (8, 8))] * 5, {}))
    with pytest.raises(MemoryError, match='commit memory'):
        animation.run_animation(settings, {})
    assert not (settings.out / 'metadata.json').exists()
