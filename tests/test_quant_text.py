from pathlib import Path
import pytest

from pixcli import models
from pixcli.comfy_backend import build_graph, run_comfy, validate_model_locations


def test_q3_profile_reuses_wan_models_and_selects_only_quantized_encoder():
    fp8 = models.list_models('animation-q3-fp8')
    q3 = models.list_models('animation-q3-text')
    assert set(q3) == (set(fp8) - {'wan_text'}) | {'wan_text_q3'}
    model = q3['wan_text_q3']
    assert model['repo_id'] == 'city96/umt5-xxl-encoder-gguf'
    assert model['revision'] == 'b535255bee98c2b0a59ea7c0ae2dcd0c6657b3b7'
    assert model['files'][0]['filename'] == 'umt5-xxl-encoder-Q3_K_S.gguf'
    assert model['files'][0]['size'] == 2858489696
    assert model['files'][0]['sha256'] == 'f64f8d6dc4d8a24276df69d0ccea789aae686f7417950a41e6568c30cb478a5c'


def test_graph_selects_explicit_q3_encoder_and_keeps_expert_routing():
    default = build_graph('source.png', 'walk', '', 42, 384, 384, 5, 1.0)
    quant = build_graph('source.png', 'walk', '', 42, 384, 384, 5, 1.0, text_encoder='q3')
    assert default['text']['class_type'] == 'CLIPLoader'
    assert default['text']['inputs']['device'] == 'cpu'
    assert quant['text'] == {'class_type': 'CLIPLoaderGGUF', 'inputs': {'clip_name': 'umt5-xxl-encoder-Q3_K_S.gguf', 'type': 'wan'}}
    assert quant['high_pixel'] == default['high_pixel']
    assert quant['low_pixel'] == default['low_pixel']
    assert quant['high_sample'] == default['high_sample']
    assert quant['low_sample'] == default['low_sample']


def test_invalid_text_encoder_is_rejected_before_model_loading(tmp_path):
    with pytest.raises(ValueError, match='text encoder'):
        build_graph('source.png', 'walk', '', 42, 384, 384, 5, 1.0, text_encoder='wrong')
    with pytest.raises(ValueError, match='text encoder'):
        run_comfy(tmp_path/'x.png', 'walk', '', 42, 384, 384, 5, 1.0, {}, text_encoder='wrong')


def test_missing_q3_model_names_explicit_profile(tmp_path, monkeypatch):
    class EmptyHub:
        @staticmethod
        def try_to_load_from_cache(*args, **kwargs):
            return None
    monkeypatch.setattr(models, '_hub', lambda: EmptyHub)
    with pytest.raises(models.ModelError, match='models download --profile animation'):
        models.model_path('wan_text_q3', tmp_path)
    with pytest.raises(FileNotFoundError, match='models download --profile animation-q3-text'):
        run_comfy(tmp_path/'x.png', 'walk', '', 42, 384, 384, 5, 1.0, {}, text_encoder='q3')


def test_location_check_uses_selected_q3_filename(tmp_path):
    names = {
        'wan_high': 'wan2.2_i2v_high_noise_14B_Q3_K_S.gguf',
        'wan_low': 'wan2.2_i2v_low_noise_14B_Q3_K_S.gguf',
        'wan_text_q3': 'umt5-xxl-encoder-Q3_K_S.gguf',
        'wan_vae': 'wan_2.1_vae.safetensors',
        'wan_distill_high': 'wan2.2_i2v_lightx2v_4steps_lora_v1_high_noise.safetensors',
        'wan_distill_low': 'wan2.2_i2v_lightx2v_4steps_lora_v1_low_noise.safetensors',
        'wan_pixel': 'wan2.2_animate_adapter_model.safetensors',
    }
    paths = {key: tmp_path/name for key, name in names.items()}
    for path in paths.values():
        path.touch()
    validate_model_locations(paths, text_encoder='q3')
    paths['wan_text_q3'] = tmp_path/'different.gguf'
    with pytest.raises(ValueError, match='filename'):
        validate_model_locations(paths, text_encoder='q3')
