import pytest

from pixcli.image import ImageSettings, validate_image


@pytest.mark.parametrize('field,value', [
    ('min_free_commit_gib', 0.5), ('min_free_commit_gib', float('nan')),
    ('min_free_physical_gib', 0), ('min_free_physical_gib', float('inf')),
])
def test_image_rejects_invalid_system_reserves(tmp_path, field, value):
    settings = ImageSettings(prompt='x', out=tmp_path / 'out.png', **{field: value})
    with pytest.raises(ValueError, match='memory'):
        validate_image(settings)


def test_streaming_preflight_uses_selected_reserves(tmp_path, monkeypatch):
    from pixcli import image, models
    monkeypatch.setattr(models, 'list_models', lambda profile: {key: {'files': []} for key in ('base', 'vae', 'lora')})
    status = {'physical_available': int(1.5 * 1024**3), 'commit_available': int(1.25 * 1024**3)}
    monkeypatch.setattr(image, '_memory_status', lambda: status)
    paths = {'base': tmp_path, 'vae': tmp_path, 'lora': tmp_path}
    result = image._memory_preflight(paths, streaming=True, min_free_commit_gib=1.25, min_free_physical_gib=1.5)
    assert result['estimated_commit_needed'] == int(1.25 * 1024**3)
    assert result['estimated_physical_needed'] == int(1.5 * 1024**3)
    status['commit_available'] -= 1
    with pytest.raises(RuntimeError, match='commit'):
        image._memory_preflight(paths, streaming=True, min_free_commit_gib=1.25, min_free_physical_gib=1.5)
    status['commit_available'] += 1
    status['physical_available'] -= 1
    with pytest.raises(RuntimeError, match='physical'):
        image._memory_preflight(paths, streaming=True, min_free_commit_gib=1.25, min_free_physical_gib=1.5)


def test_image_guard_uses_selected_reserves(monkeypatch):
    from pixcli import image
    status = {'physical_available': int(1.5 * 1024**3), 'commit_available': int(1.25 * 1024**3)}
    monkeypatch.setattr(image, '_memory_status', lambda: status)
    guard = image._ImageMemoryGuard(min_free_commit_gib=1.25, min_free_physical_gib=1.5)
    guard.sample()
    status['commit_available'] -= 1
    with pytest.raises(MemoryError, match='memory guard'):
        guard.sample()
    status['commit_available'] += 1
    status['physical_available'] -= 1
    with pytest.raises(MemoryError, match='memory guard'):
        guard.sample()


def test_resident_preflight_honors_higher_requested_reserves(tmp_path, monkeypatch):
    from pixcli import image, models
    monkeypatch.setattr(models, 'list_models', lambda profile: {key: {'files': []} for key in ('base', 'vae', 'lora')})
    monkeypatch.setattr(image, '_memory_status', lambda: {'physical_available': 10 * 1024**3, 'commit_available': 10 * 1024**3})
    with pytest.raises(RuntimeError, match='commit'):
        image._memory_preflight({'base': tmp_path, 'vae': tmp_path, 'lora': tmp_path}, min_free_commit_gib=12)


def test_image_job_checks_system_reserves_without_ram_budget(tmp_path, monkeypatch):
    from PIL import Image
    from pixcli import image
    monkeypatch.setattr(image, '_memory_status', lambda: {'physical_available': int(1.5 * 1024**3), 'commit_available': int(1.5 * 1024**3)})
    monkeypatch.setattr(image, '_infer', lambda *args: (Image.new('RGB', (8, 8)), {
        'minimum_sampled_physical_available_bytes': 4 * 1024**3,
        'minimum_sampled_commit_available_bytes': 4 * 1024**3,
    }))
    settings = ImageSettings(prompt='x', out=tmp_path/'out.png', pixel_width=8, pixel_height=8,
                             min_free_commit_gib=1, min_free_physical_gib=1)
    result = image.run_image(settings, {})
    assert result['settings']['min_free_commit_gib'] == 1
    assert result['settings']['min_free_physical_gib'] == 1
    assert result['runtime']['minimum_sampled_physical_available_bytes'] == int(1.5 * 1024**3)
    assert result['runtime']['minimum_sampled_commit_available_bytes'] == int(1.5 * 1024**3)
    assert (tmp_path/'out.png').exists()
