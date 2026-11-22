from pathlib import Path
from types import SimpleNamespace

import pytest


def local_selection(path):
    return {'name': 'local', 'backend': 'diffusers', 'source': {'type': 'local', 'path': str(path)}}


def remote_selection():
    return {'name': 'cloud', 'backend': 'diffusers',
            'source': {'type': 'huggingface', 'repo_id': 'owner/model', 'revision': 'main'}}


def fake_hub(tmp_path, monkeypatch):
    from pixcli import models

    calls = []
    sha = 'a' * 40

    class API:
        def model_info(self, repo_id, revision):
            calls.append(('info', repo_id, revision))
            return SimpleNamespace(sha=sha)

    def download(repo_id, *, revision, local_dir, **kwargs):
        calls.append(('snapshot', repo_id, revision))
        folder = Path(local_dir)
        folder.mkdir(parents=True, exist_ok=True)
        (folder / 'model_index.json').write_text('{}', encoding='utf-8')
        (folder / 'weights.safetensors').write_bytes(b'weights')
        return str(folder)

    hub = SimpleNamespace(HfApi=API, snapshot_download=download)
    monkeypatch.setattr(models, '_hub', lambda: hub)
    return hub, calls, sha


def test_local_model_is_usable_without_install_and_never_deleted(tmp_path):
    from pixcli import model_store

    model_dir = tmp_path / 'external'
    model_dir.mkdir()
    (model_dir / 'model_index.json').write_text('{}', encoding='utf-8')
    (model_dir / 'weights.safetensors').write_bytes(b'weights')
    selection = local_selection(model_dir)
    paths, provenance = model_store.resolve(selection, tmp_path / 'cache')
    assert paths['base'] == model_dir
    assert provenance['sources']['base']['type'] == 'local'
    assert model_store.status(selection, tmp_path / 'cache')['installed']
    assert all(item['ok'] for item in model_store.verify(selection, tmp_path / 'cache'))
    assert model_store.uninstall(selection, tmp_path / 'cache')['removed'] is False
    assert (model_dir / 'model_index.json').exists()


def test_local_diffusers_status_rejects_a_directory_without_weights(tmp_path):
    from pixcli import model_store

    model_dir = tmp_path / 'incomplete'
    model_dir.mkdir()
    (model_dir / 'model_index.json').write_text('{}', encoding='utf-8')
    result = model_store.status(local_selection(model_dir), tmp_path / 'cache')
    assert not result['installed']
    assert 'safetensors' in result['detail']


def test_remote_install_pins_revision_verifies_and_uninstalls_only_owned_files(tmp_path, monkeypatch):
    from pixcli import model_store

    hub, calls, sha = fake_hub(tmp_path, monkeypatch)
    cache = tmp_path / 'cache'
    cache.mkdir()
    shared = cache / 'shared-file'
    shared.write_text('another app owns this', encoding='utf-8')
    selection = remote_selection()
    assert not model_store.status(selection, cache)['installed']
    assert model_store.install(selection, cache)['installed']
    assert ('snapshot', 'owner/model', sha) in calls
    # Once installed, neither resolution nor verification may contact the Hub.
    monkeypatch.setattr(hub, 'HfApi', lambda: pytest.fail('unexpected network access'))
    paths, provenance = model_store.resolve(selection, cache)
    assert paths['base'].is_dir()
    assert provenance['sources']['base']['resolved_revision'] == sha
    assert all(item['ok'] for item in model_store.verify(selection, cache))
    (paths['base'] / 'weights.safetensors').write_bytes(b'corrupt')
    assert any(not item['ok'] for item in model_store.verify(selection, cache))
    assert model_store.uninstall(selection, cache)['removed'] is True
    assert not paths['base'].exists()
    assert shared.read_text(encoding='utf-8') == 'another app owns this'


def test_interrupted_remote_install_is_not_visible_until_completed(tmp_path, monkeypatch):
    from pixcli import model_store

    hub, _, _ = fake_hub(tmp_path, monkeypatch)
    good_download = hub.snapshot_download

    def fail_download(*args, local_dir, **kwargs):
        Path(local_dir).mkdir(parents=True, exist_ok=True)
        (Path(local_dir) / 'partial').write_text('partial')
        raise OSError('interrupted')

    monkeypatch.setattr(hub, 'snapshot_download', fail_download)
    selection = remote_selection()
    with pytest.raises(Exception, match='interrupted'):
        model_store.install(selection, tmp_path / 'cache')
    assert not model_store.status(selection, tmp_path / 'cache')['installed']
    with pytest.raises(Exception, match='install'):
        model_store.resolve(selection, tmp_path / 'cache')
    monkeypatch.setattr(hub, 'snapshot_download', good_download)
    assert model_store.install(selection, tmp_path / 'cache')['installed']


def test_remote_snapshot_without_supported_weights_is_not_marked_installed(tmp_path, monkeypatch):
    from pixcli import model_store

    hub, _, _ = fake_hub(tmp_path, monkeypatch)
    def download(*args, local_dir, **kwargs):
        target = Path(local_dir)
        target.mkdir(parents=True, exist_ok=True)
        (target / 'model_index.json').write_text('{}', encoding='utf-8')
        (target / 'weights.bin').write_bytes(b'not supported')
    monkeypatch.setattr(hub, 'snapshot_download', download)
    selection = remote_selection()
    with pytest.raises(Exception, match='safetensors'):
        model_store.install(selection, tmp_path / 'cache')
    assert not model_store.status(selection, tmp_path / 'cache')['installed']


def test_wan_override_uses_external_component_and_pinned_fallback(tmp_path, monkeypatch):
    from pixcli import models, model_store

    cache = tmp_path / 'cache'
    cache.mkdir()
    high = tmp_path / 'custom-high.gguf'
    high.write_bytes(b'model')
    calls = []

    def pinned(key, cache_dir, *, profile=None):
        calls.append((key, profile))
        result = cache / key
        result.write_bytes(b'pinned')
        return result

    monkeypatch.setattr(models, 'model_path', pinned)
    selection = {'name': 'motion', 'backend': 'wan-i2v', 'text_encoder': 'q3', 'expert_quant': 'q2',
                 'components': {'wan_high_q2': {'type': 'local', 'path': str(high)}}}
    paths, provenance = model_store.resolve(selection, cache)
    assert paths['wan_high_q2'] == high
    assert paths['wan_low_q2'] == cache / 'wan_low_q2'
    assert all(key != 'wan_high_q2' for key, _ in calls)
    assert all(profile == 'animation' for _, profile in calls)
    assert provenance['sources']['wan_high_q2']['type'] == 'local'


def test_builtin_install_is_owned_and_uninstall_preserves_legacy_cache(tmp_path, monkeypatch):
    from pixcli import models, model_store

    legacy = tmp_path / 'cache'
    legacy.mkdir()
    (legacy / 'sentinel').write_text('shared')
    selection = {'name': 'default-image', 'backend': 'builtin-image'}

    def download(profile, cache_dir):
        assert profile == 'image'
        Path(cache_dir).mkdir(parents=True, exist_ok=True)
        (Path(cache_dir) / 'sentinel').write_text('managed')
        return []

    def model_path(key, cache_dir, *, profile=None):
        target = Path(cache_dir)
        if not (target / 'sentinel').exists():
            raise models.ModelError('missing')
        return target / key

    monkeypatch.setattr(models, 'download', download)
    monkeypatch.setattr(models, 'model_path', model_path)
    paths, _ = model_store.resolve(selection, legacy)
    assert paths['base'] == legacy / 'base'
    assert model_store.install(selection, legacy)['installed']
    paths, _ = model_store.resolve(selection, legacy)
    assert paths['base'] != legacy / 'base'
    assert model_store.uninstall(selection, legacy)['removed']
    assert (legacy / 'sentinel').read_text() == 'shared'
    paths, _ = model_store.resolve(selection, legacy)
    assert paths['base'] == legacy / 'base'


def test_store_does_not_follow_tampered_marker_paths(tmp_path, monkeypatch):
    import json
    from pixcli import model_store

    fake_hub(tmp_path, monkeypatch)
    selection = remote_selection()
    cache = tmp_path / 'cache'
    model_store.install(selection, cache)
    marker = next(cache.rglob('installed.json'))
    data = json.loads(marker.read_text(encoding='utf-8'))
    data['sources']['base']['path'] = '../../outside'
    marker.write_text(json.dumps(data), encoding='utf-8')
    with pytest.raises(Exception, match='(invalid|unsafe|install)'):
        model_store.resolve(selection, cache)


def test_old_definitions_can_be_listed_and_uninstalled_by_id(tmp_path, monkeypatch):
    from pixcli import model_store

    fake_hub(tmp_path, monkeypatch)
    cache = tmp_path / 'cache'
    old = remote_selection()
    model_store.install(old, cache)
    previous = model_store.installed(cache)
    assert len(previous) == 1 and previous[0]['name'] == 'cloud'
    assert previous[0]['bytes'] > 0
    revised = remote_selection()
    revised['source']['revision'] = 'other-branch'
    model_store.install(revised, cache)
    assert len(model_store.installed(cache)) == 2
    assert model_store.uninstall_id(previous[0]['id'], cache)['removed']
    assert model_store.status(revised, cache)['installed']
    assert len(model_store.installed(cache)) == 1


@pytest.mark.parametrize('bad', ['..', '../outside', 'f' * 63, 'F' * 64, 'a' * 64 + '/more'])
def test_uninstall_id_rejects_invalid_ids(tmp_path, bad):
    from pixcli import model_store

    with pytest.raises(Exception, match='(invalid|Invalid)'):
        model_store.uninstall_id(bad, tmp_path / 'cache')


def test_builtin_animation_effective_defaults_share_installation_id(tmp_path):
    from pixcli import model_store

    selected = {'name': 'default-animation', 'backend': 'builtin-animation'}
    effective = {**selected, 'text_encoder': 'q3', 'expert_quant': 'q2'}
    assert model_store._fingerprint(selected) == model_store._fingerprint(effective)


def test_reinstall_does_not_damage_existing_good_model_when_network_fails(tmp_path, monkeypatch):
    from pixcli import model_store

    hub, _, _ = fake_hub(tmp_path, monkeypatch)
    selection = remote_selection()
    cache = tmp_path / 'cache'
    model_store.install(selection, cache)
    original, _ = model_store.resolve(selection, cache)
    monkeypatch.setattr(hub, 'HfApi', lambda: (_ for _ in ()).throw(OSError('offline')))
    assert model_store.install(selection, cache)['installed']
    assert (original['base'] / 'weights.safetensors').read_bytes() == b'weights'


def test_wan_install_reuses_valid_legacy_pinned_components(tmp_path, monkeypatch):
    from pixcli import model_store, models

    cache = tmp_path / 'cache'
    cache.mkdir()
    roles = models.list_models('animation')
    files = {}
    fake_models = {}
    for key in roles:
        file = cache / f'{key}.weights'
        file.write_bytes(key.encode())
        files[key] = file
        fake_models[key] = {'repo_id': f'owner/{key}', 'revision': 'a' * 40,
                            'files': [{'filename': file.name, 'size': file.stat().st_size,
                                       'sha256': __import__('hashlib').sha256(file.read_bytes()).hexdigest()}]}
    monkeypatch.setattr(models, 'list_models', lambda profile: fake_models)
    monkeypatch.setattr(models, '_cached', lambda data, filename, cache_dir: cache / filename
                        if Path(cache_dir) == cache else None)
    monkeypatch.setattr(models, 'model_path', lambda key, cache_dir, profile=None: files[key]
                        if Path(cache_dir) == cache else (_ for _ in ()).throw(models.ModelError('missing')))
    monkeypatch.setattr(models, '_hub', lambda: pytest.fail('valid shared components should not download'))
    selection = {'name': 'motion', 'backend': 'wan-i2v', 'text_encoder': 'q3', 'expert_quant': 'q2', 'components': {}}
    report = model_store.install(selection, cache)
    assert report['installed'] and not report['managed']
    assert all(source['storage'] == 'shared' for source in report['provenance']['sources'].values())
