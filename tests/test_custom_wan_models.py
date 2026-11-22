import json
from pathlib import Path

import pytest
from PIL import Image

from pixcli.animation import AnimationSettings, run_animation
from pixcli.comfy_backend import build_graph, validate_model_locations
from test_checked_lora_device import checked_node  # reuse the ComfyUI test fixture


NAMES = {
    'wan_high_q2': 'custom-high.gguf',
    'wan_low_q2': 'custom-low.gguf',
    'wan_text_q3': 'custom-text.gguf',
    'wan_vae': 'custom-vae.safetensors',
    'wan_distill_high': 'custom-high-lora.safetensors',
    'wan_distill_low': 'custom-low-lora.safetensors',
    'wan_pixel': 'custom-pixel.safetensors',
}


def _graph(**kwargs):
    return build_graph('source.png', 'walk', '', 42, 256, 256, 5, 1.0,
                       text_encoder='q3', expert_quant='q2', **kwargs)


def test_custom_filenames_are_routed_to_both_experts_and_every_loader():
    graph = _graph(custom_names=NAMES)
    assert graph['high_model']['inputs']['unet_name'] == 'custom-high.gguf'
    assert graph['low_model']['inputs']['unet_name'] == 'custom-low.gguf'
    assert graph['text']['inputs']['clip_name'] == 'custom-text.gguf'
    assert graph['vae']['inputs']['vae_name'] == 'custom-vae.safetensors'
    assert graph['high_distill']['inputs']['lora_name'] == 'custom-high-lora.safetensors'
    assert graph['low_distill']['inputs']['lora_name'] == 'custom-low-lora.safetensors'
    assert graph['high_pixel']['inputs']['lora_name'] == 'custom-pixel.safetensors'
    assert graph['low_pixel']['inputs']['lora_name'] == 'custom-pixel.safetensors'
    assert graph['high_sample']['inputs']['end_at_step'] == 2
    assert graph['low_sample']['inputs']['start_at_step'] == 2


def test_custom_model_names_must_cover_the_whole_workflow():
    partial = {key: value for key, value in NAMES.items() if key != 'wan_pixel'}
    with pytest.raises(ValueError, match='wan_pixel'):
        _graph(custom_names=partial)


def test_custom_filenames_reject_same_category_collisions_and_shadows(tmp_path):
    paths = {}
    for index, (key, name) in enumerate(NAMES.items()):
        folder = tmp_path / str(index)
        folder.mkdir()
        paths[key] = folder / name
        paths[key].write_bytes(b'model')
    validate_model_locations(paths, text_encoder='q3', expert_quant='q2', custom_names=NAMES)

    duplicate = dict(paths)
    duplicate['wan_low_q2'] = paths['wan_high_q2']
    duplicate_names = dict(NAMES, wan_low_q2=NAMES['wan_high_q2'])
    with pytest.raises(ValueError, match='duplicate|ambiguous'):
        validate_model_locations(duplicate, text_encoder='q3', expert_quant='q2', custom_names=duplicate_names)

    (paths['wan_low_q2'].parent / NAMES['wan_high_q2']).write_bytes(b'imposter')
    with pytest.raises(ValueError, match='shadow'):
        validate_model_locations(paths, text_encoder='q3', expert_quant='q2', custom_names=NAMES)


def test_custom_animation_metadata_records_its_real_model_provenance(tmp_path, monkeypatch):
    source = tmp_path / 'source.png'
    Image.new('RGB', (8, 8)).save(source)
    settings = AnimationSettings(input_path=source, prompt='walk', out=tmp_path/'out',
                                 pixel_width=8, pixel_height=8, mp4=False, save_json=True)
    paths = {key: tmp_path / name for key, name in NAMES.items()}
    spec = {'name': 'my-wan', 'backend': 'wan-i2v', 'provenance': {
        'name': 'my-wan', 'backend': 'wan-i2v', 'profile': 'animation',
        'sources': {'wan_high_q2': {'repo_id': 'author/custom', 'revision': 'abc123', 'files': ['custom-high.gguf']}}
    }}
    observed = {}
    def fake_infer(*args, **kwargs):
        observed.update(kwargs)
        return [Image.new('RGB', (8, 8))] * 5, {}
    monkeypatch.setattr('pixcli.animation.run_comfy', fake_infer)
    monkeypatch.setattr('pixcli.comfy_backend._free_commit_bytes', lambda: 16 * 1024**3)
    run_animation(settings, paths, model_spec=spec)
    metadata = json.loads((settings.out / 'metadata.json').read_text(encoding='utf-8'))
    assert metadata['model_profile'] == 'my-wan'
    assert metadata['models'] == spec['provenance']['sources']
    assert metadata['quantization'] != 'GGUF Q2_K experts; Q3_K_S text'
    assert observed['custom_names'] == NAMES


def test_mapped_custom_pixel_file_requires_valid_adapter_schema(checked_node, monkeypatch):
    from types import SimpleNamespace
    from pixcli import mapped_safetensors
    node, _, fake_torch = checked_node
    fake_torch.bfloat16 = object()
    tensors = {}
    for index in range(400):
        prefix = f'diffusion_model.blocks.{index}.attention'
        tensors[prefix + '.lora_A.weight'] = SimpleNamespace(dtype=fake_torch.bfloat16, shape=(2, 4))
        tensors[prefix + '.lora_B.weight'] = SimpleNamespace(dtype=fake_torch.bfloat16, shape=(4, 2))
    monkeypatch.setattr(mapped_safetensors, 'load_readonly_safetensors', lambda path: (tensors, {}))
    assert node._load('other-pixel.safetensors', 'cpu', 'mmap') is tensors
    del tensors[next(iter(tensors))]
    with pytest.raises(ValueError, match='adapter|schema|pinned'):
        node._load('bad-pixel.safetensors', 'cpu', 'mmap')


def test_mapped_custom_distillation_file_requires_full_key_and_dtype_schema(checked_node, monkeypatch):
    from types import SimpleNamespace
    from pixcli import mapped_safetensors
    node, _, fake_torch = checked_node
    fake_torch.float32, fake_torch.int64 = object(), object()
    tensors = {}
    for index in range(400):
        prefix = f'diffusion_model.blocks.{index}.attention'
        tensors[prefix + '.lora_up.weight'] = SimpleNamespace(dtype=fake_torch.float32, shape=(4, 2))
        tensors[prefix + '.lora_down.weight'] = SimpleNamespace(dtype=fake_torch.float32, shape=(2, 4))
        tensors[prefix + '.alpha'] = SimpleNamespace(dtype=fake_torch.int64, shape=())
    monkeypatch.setattr(mapped_safetensors, 'load_readonly_safetensors', lambda path: (tensors, {}))
    assert node._load_distill('other-distill.safetensors')[0] is tensors
    del tensors[next(iter(tensors))]
    with pytest.raises(ValueError, match='keys|schema|adapter'):
        node._load_distill('bad-distill.safetensors')


def _setup_fp8_node(checked_node, monkeypatch, *, bad_shape=False):
    import sys
    from types import ModuleType, SimpleNamespace
    from pixcli import mapped_safetensors
    node, comfy, fake_torch = checked_node
    fake_torch.float16, fake_torch.float8_e4m3fn = object(), object()
    state = {
        'shared.weight': SimpleNamespace(shape=(1, 4) if bad_shape else (256384, 4096), dtype=fake_torch.float16),
        'scaled_fp8': SimpleNamespace(dtype=fake_torch.float8_e4m3fn),
        'encoder.final_layer_norm.weight': SimpleNamespace(dtype=fake_torch.float16),
        **{f'extra-{index}': object() for index in range(409)},
    }
    monkeypatch.setattr(mapped_safetensors, 'load_readonly_safetensors', lambda path: (state, {}))
    comfy.ops, comfy.sd = ModuleType('comfy.ops'), ModuleType('comfy.sd')
    comfy.ops.mixed_precision_ops = lambda *args, **kwargs: SimpleNamespace(Embedding=object)
    comfy.sd.CLIPType = SimpleNamespace(WAN='wan')
    comfy.sd.load_text_encoder_state_dicts = lambda *args, **kwargs: 'loaded encoder'
    comfy.utils.convert_old_quants = lambda state, **kwargs: (state, {})
    comfy.utils.detect_layer_quantization = lambda *args: {'quantized': True}
    monkeypatch.setitem(sys.modules, 'comfy.ops', comfy.ops)
    monkeypatch.setitem(sys.modules, 'comfy.sd', comfy.sd)
    module = sys.modules[node.PixcliMappedFP8UMT5Loader.__module__]
    monkeypatch.setattr(module, 'torch', fake_torch)
    return module


def test_renamed_fp8_text_encoder_loads_with_the_same_structure_checks(checked_node, monkeypatch):
    module = _setup_fp8_node(checked_node, monkeypatch)
    assert module.load_mapped_fp8_text('my-encoder.safetensors') == 'loaded encoder'


def test_renamed_fp8_text_encoder_still_rejects_wrong_embedding_shape(checked_node, monkeypatch):
    module = _setup_fp8_node(checked_node, monkeypatch, bad_shape=True)
    with pytest.raises(ValueError, match='structure'):
        module.load_mapped_fp8_text('my-encoder.safetensors')
