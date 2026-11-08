"""Offline model inventory checks; no weights or network required."""
import hashlib

import pytest

from pixcli import models


def test_missing_model_is_actionable_and_offline(tmp_path, monkeypatch):
    class OfflineHub:
        @staticmethod
        def try_to_load_from_cache(*args, **kwargs):
            return None

        @staticmethod
        def hf_hub_download(*args, **kwargs):
            raise AssertionError('offline lookup attempted a download')

    monkeypatch.setattr(models, '_hub', lambda: OfflineHub)
    assert all(not item['ok'] and item['detail'] == 'missing' for item in models.verify('image', tmp_path))
    with pytest.raises(models.ModelError, match='models download --profile image'):
        models.model_path('base', tmp_path)


def test_verify_checks_lfs_sha256_and_git_blob_id(tmp_path, monkeypatch):
    content = b'example model data'
    lfs = tmp_path / 'weights.safetensors'
    small = tmp_path / 'config.json'
    lfs.write_bytes(content)
    small.write_bytes(content)
    sha256 = hashlib.sha256(content).hexdigest()
    git_id = hashlib.sha1(f'blob {len(content)}\0'.encode() + content).hexdigest()
    sample = {'repo_id': 'test/model', 'revision': 'test', 'files': [
        {'filename': lfs.name, 'size': len(content), 'sha256': sha256},
        {'filename': small.name, 'size': len(content), 'git_blob_sha1': git_id},
    ]}
    monkeypatch.setattr(models, 'list_models', lambda profile=None: {'test': sample})
    monkeypatch.setattr(models, '_cached', lambda model, filename, cache_dir: tmp_path / filename)
    assert all(result['ok'] for result in models.verify())
    lfs.write_bytes(b'X' * len(content))
    results = models.verify()
    assert not results[0]['ok'] and results[0]['detail'] == 'hash or size mismatch'
    assert results[1]['ok']

def test_animation_profiles_select_exact_experts_and_text():
    assert models.animation_profile('fp8','q3') == 'animation-q3-fp8'
    assert models.animation_profile('q3','q3') == 'animation-q3-text'
    assert models.animation_profile('fp8','q2') == 'animation-q2-experts'
    assert models.animation_profile('q3','q2') == 'animation'
    assert set(models.list_models('animation-q2-q3-text')) == {'wan_high_q2','wan_low_q2','wan_vae','wan_text_q3','wan_distill_high','wan_distill_low','wan_pixel'}
    with pytest.raises(models.ModelError):
        models.animation_profile('fp8','q4')


def test_missing_selected_q2_model_reports_selected_profile(tmp_path, monkeypatch):
    monkeypatch.setattr(models, '_cached', lambda *args: None)
    with pytest.raises(models.ModelError, match='animation-q2-q3-text'):
        models.model_path('wan_high_q2', tmp_path, profile='animation-q2-q3-text')
    with pytest.raises(models.ModelError, match='not in'):
        models.model_path('wan_high', tmp_path, profile='animation-q2-experts')
