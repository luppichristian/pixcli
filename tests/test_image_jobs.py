import json
from pathlib import Path
from PIL import Image
import pytest
from pixcli.image import ImageSettings, run_image, validate_image


def test_image_job_emits_metadata_and_refuses_existing_output(tmp_path, monkeypatch):
    output = tmp_path / 'sprite.png'
    settings = ImageSettings(prompt='a knight', out=output, pixel_width=3, pixel_height=2, inference_width=768, inference_height=512)
    monkeypatch.setattr('pixcli.image._infer', lambda *a: (Image.new('RGB', (768, 512), 'red'), {}))
    metadata = run_image(settings, {})
    assert Image.open(output).size == (3, 2)
    assert json.loads(output.with_suffix('.json').read_text(encoding='utf-8'))['settings']['prompt'] == 'a knight'
    assert metadata['final_dimensions'] == [3, 2]
    with pytest.raises(FileExistsError):
        run_image(settings, {})


def test_validation_rejects_too_small_img2img_strength_and_nonmultiple_dimensions(tmp_path):
    with pytest.raises(ValueError):
        validate_image(ImageSettings(prompt='x', out=tmp_path / 'x.png', inference_width=1000, inference_height=999))
    with pytest.raises(ValueError):
        validate_image(ImageSettings(prompt='x', out=tmp_path / 'x.png', input_path=tmp_path / 'x.png', strength=0.001))


def test_masked_job_preserves_source_pixels_and_alpha(tmp_path, monkeypatch):
    source = tmp_path / 'source.png'
    mask_path = tmp_path / 'mask.png'
    Image.new('RGBA', (4, 2), (17, 31, 47, 83)).save(source)
    mask = Image.new('L', (4, 2), 0)
    mask.putpixel((1, 0), 255)
    mask.save(mask_path)
    output = tmp_path / 'edited.png'
    monkeypatch.setattr('pixcli.image._infer', lambda settings, paths, image: (Image.new('RGB', image.size, 'red'), {}))
    run_image(ImageSettings(prompt='red', out=output, input_path=source, mask_path=mask_path), {})
    result = Image.open(output)
    assert result.size == (4, 2)
    assert result.getpixel((0, 0)) == (17, 31, 47, 83)
    assert result.getpixel((1, 0)) == (255, 0, 0, 83)

def test_validation_checks_input_and_mask_before_models(tmp_path):
    source = tmp_path / 'source.png'
    mask = tmp_path / 'mask.png'
    settings = ImageSettings(prompt='x', out=tmp_path / 'out.png', input_path=source)
    with pytest.raises(FileNotFoundError):
        validate_image(settings)
    Image.new('RGB', (4, 2)).save(source)
    Image.new('L', (2, 2)).save(mask)
    settings.mask_path = mask
    with pytest.raises(ValueError, match='mask dimensions'):
        validate_image(settings)


def test_validation_checks_sidecar_collision_before_models(tmp_path):
    output = tmp_path / 'sprite.png'
    output.with_suffix('.json').write_text('old', encoding='utf-8')
    with pytest.raises(FileExistsError):
        validate_image(ImageSettings(prompt='x', out=output))

def test_adapter_coverage_rejects_unapplied_layers(tmp_path):
    import struct
    from types import SimpleNamespace
    from pixcli.image import _assert_lora_coverage
    header = json.dumps({
        'lora_unet_a.lora_down.weight': {'dtype': 'F16', 'shape': [1], 'data_offsets': [0, 2]},
        'lora_unet_a.lora_up.weight': {'dtype': 'F16', 'shape': [1], 'data_offsets': [2, 4]},
        'lora_unet_b.lora_down.weight': {'dtype': 'F16', 'shape': [1], 'data_offsets': [4, 6]},
        'lora_unet_b.lora_up.weight': {'dtype': 'F16', 'shape': [1], 'data_offsets': [6, 8]},
    }).encode('utf-8')
    adapter = tmp_path / 'adapter.safetensors'
    adapter.write_bytes(struct.pack('<Q', len(header)) + header + b'\0' * 8)
    layer = SimpleNamespace(lora_A={'pixel': object()}, lora_B={'pixel': object()})
    unet = SimpleNamespace(named_modules=lambda: iter([('one', layer)]))
    with pytest.raises(RuntimeError, match='coverage'):
        _assert_lora_coverage(SimpleNamespace(unet=unet), adapter)

def test_memory_preflight_rejects_insufficient_commit(tmp_path, monkeypatch):
    from pixcli.image import _memory_preflight
    from pixcli import models
    monkeypatch.setattr(models, 'list_models', lambda profile: {key: {'files': []} for key in ('base', 'vae', 'lora')})
    monkeypatch.setattr('pixcli.image._memory_status', lambda: {'physical_available': 16 * 1024**3, 'commit_available': 2 * 1024**3})
    with pytest.raises(RuntimeError, match='commit'):
        _memory_preflight({'base': tmp_path, 'vae': tmp_path, 'lora': tmp_path / 'adapter'})

def test_generation_rejects_aspect_ratio_stretch(tmp_path):
    with pytest.raises(ValueError, match='aspect ratio'):
        validate_image(ImageSettings(prompt='x', out=tmp_path / 'sprite.png', pixel_width=128, pixel_height=64))

def test_memory_preflight_counts_only_selected_pinned_files(tmp_path, monkeypatch):
    from pixcli.image import _memory_preflight
    from pixcli import models
    base = tmp_path / 'base'
    vae = tmp_path / 'vae'
    base.mkdir()
    vae.mkdir()
    (base / 'selected.safetensors').write_bytes(b'a' * 10)
    (base / 'unrelated-fp32.safetensors').write_bytes(b'x' * 100)
    (vae / 'selected.safetensors').write_bytes(b'b' * 20)
    lora = tmp_path / 'adapter.safetensors'
    lora.write_bytes(b'c' * 30)
    monkeypatch.setattr(models, 'list_models', lambda profile: {
        'base': {'files': [{'filename': 'selected.safetensors'}]},
        'vae': {'files': [{'filename': 'selected.safetensors'}]},
        'lora': {'files': [{'filename': 'adapter.safetensors'}]},
    })
    monkeypatch.setattr('pixcli.image._memory_status', lambda: {'physical_available': 64 * 1024**3, 'commit_available': 64 * 1024**3})
    result = _memory_preflight({'base': base, 'vae': vae, 'lora': lora})
    assert result['checkpoint_bytes'] == 60

def test_oom_releases_models_and_traceback_before_emptying_cuda_cache(tmp_path, monkeypatch):
    import sys
    import types
    import weakref
    from pixcli.image import _infer
    references = []
    class FakeOOM(Exception):
        pass
    class Allocation:
        pass
    class FakeVAE:
        @staticmethod
        def from_pretrained(*args, **kwargs):
            vae = Allocation()
            references.append(weakref.ref(vae))
            return vae
    class FakePipeline:
        @staticmethod
        def from_pretrained(*args, **kwargs):
            temporary_gpu_allocation = Allocation()
            references.append(weakref.ref(temporary_gpu_allocation))
            raise FakeOOM('allocation failed')
    released_at_empty_cache = []
    cuda = types.SimpleNamespace(is_available=lambda: True, reset_peak_memory_stats=lambda: None,
        OutOfMemoryError=FakeOOM, empty_cache=lambda: released_at_empty_cache.append(all(ref() is None for ref in references)))
    monkeypatch.setitem(sys.modules, 'torch', types.SimpleNamespace(cuda=cuda, float16=object()))
    monkeypatch.setitem(sys.modules, 'diffusers', types.SimpleNamespace(AutoencoderKL=FakeVAE, StableDiffusionXLPipeline=FakePipeline, StableDiffusionXLImg2ImgPipeline=FakePipeline))
    monkeypatch.setattr('pixcli.image._memory_preflight', lambda paths, **kwargs: {})
    with pytest.raises(RuntimeError, match='out of memory'):
        _infer(ImageSettings(prompt='x', out=tmp_path / 'out.png'), {'base': tmp_path, 'vae': tmp_path, 'lora': tmp_path}, None)
    assert released_at_empty_cache == [True]

def test_inference_sets_offline_flags_before_import_and_restores_them(tmp_path, monkeypatch):
    import builtins
    import os
    from pixcli.image import _infer
    original_import = builtins.__import__
    monkeypatch.setenv('HF_HUB_OFFLINE', '0')
    monkeypatch.delenv('TRANSFORMERS_OFFLINE', raising=False)
    seen = []
    def intercept(name, *args, **kwargs):
        if name == 'torch':
            seen.append((os.environ.get('HF_HUB_OFFLINE'), os.environ.get('TRANSFORMERS_OFFLINE')))
            raise ImportError('simulated missing torch')
        return original_import(name, *args, **kwargs)
    monkeypatch.setattr(builtins, '__import__', intercept)
    with pytest.raises(RuntimeError, match='dependencies missing'):
        _infer(ImageSettings(prompt='x', out=tmp_path / 'out.png'), {}, None)
    assert seen == [('1', '1')]
    assert os.environ['HF_HUB_OFFLINE'] == '0'
    assert 'TRANSFORMERS_OFFLINE' not in os.environ

def test_oom_advice_does_not_recommend_the_failed_canvas(tmp_path):
    from pixcli.image import _oom_message
    message = _oom_message(ImageSettings(prompt='x', out=tmp_path / 'x.png', inference_width=768, inference_height=768, seed=123))
    assert '768x768' in message and 'seed=123' in message
    assert '--inference-width 512 --inference-height 512' in message

def test_oom_advice_mentions_requested_vram_budget(tmp_path):
    from pixcli.image import _oom_message
    message = _oom_message(ImageSettings(prompt='x', out=tmp_path / 'x.png', vram_budget_gib=0.5))
    assert '--vram-budget-gib' in message
    assert '0.5' in message

def test_image_process_peak_measurements_report_available_windows_fields(monkeypatch):
    import psutil
    from types import SimpleNamespace
    from pixcli.image import _process_memory_peaks
    monkeypatch.setattr(psutil, 'Process', lambda: SimpleNamespace(memory_info=lambda: SimpleNamespace(peak_wset=1234, peak_pagefile=5678)))
    assert _process_memory_peaks() == {'peak_working_set_bytes': 1234, 'peak_commit_bytes': 5678}

def test_image_memory_guard_stops_when_commit_is_low(monkeypatch):
    from pixcli.image import _ImageMemoryGuard
    import pixcli.image as image
    values = iter([
        {'physical_available': 10 * 1024**3, 'commit_available': 6 * 1024**3},
        {'physical_available': 8 * 1024**3, 'commit_available': 1024**3},
    ])
    monkeypatch.setattr(image, '_memory_status', lambda: next(values))
    guard = _ImageMemoryGuard()
    guard.sample()
    with pytest.raises(MemoryError, match='memory guard'):
        guard.sample()
    assert guard.minimum_commit == 1024**3

def test_image_metadata_records_offline_execution(tmp_path, monkeypatch):
    out = tmp_path / 'sprite.png'
    monkeypatch.setattr('pixcli.image._infer', lambda *args: (Image.new('RGB', (8, 8), 'red'), {}))
    metadata = run_image(ImageSettings(prompt='x', out=out, pixel_width=4, pixel_height=4), {})
    assert metadata['offline'] is True

@pytest.mark.parametrize('field,value', [('ram_budget_gib', 0), ('ram_budget_gib', float('nan')),
                                        ('vram_budget_gib', -1), ('vram_budget_gib', float('inf'))])
def test_image_rejects_invalid_memory_budgets(tmp_path, field, value):
    settings = ImageSettings(prompt='x', out=tmp_path / 'out.png', **{field: value})
    with pytest.raises(ValueError, match='budget'):
        validate_image(settings)

def test_image_memory_guard_stops_when_resident_budget_exceeded(monkeypatch):
    from pixcli.image import _ImageMemoryGuard
    monkeypatch.setattr('pixcli.image._memory_status', lambda: {'physical_available': 8 * 1024**3, 'commit_available': 8 * 1024**3})
    monkeypatch.setattr('pixcli.image._resident_memory_bytes', lambda: 2 * 1024**3)
    guard = _ImageMemoryGuard(ram_budget_gib=1)
    with pytest.raises(MemoryError, match='RAM budget'):
        guard.sample()

def test_image_offload_uses_per_layer_streaming_for_a_budget():
    from pixcli.image import _configure_image_offload
    calls = []
    class Pipeline:
        def enable_model_cpu_offload(self): calls.append('model')
        def enable_sequential_cpu_offload(self): calls.append('sequential')
    assert _configure_image_offload(Pipeline(), None, None) == 'model_cpu_offload'
    assert _configure_image_offload(Pipeline(), 4, None) == 'sequential_cpu_offload'
    assert _configure_image_offload(Pipeline(), None, 4) == 'sequential_cpu_offload'
    assert calls == ['model', 'sequential', 'sequential']

def test_cuda_budget_fraction_preserves_a_stricter_existing_limit():
    from types import SimpleNamespace
    from pixcli.image import _set_cuda_budget
    fractions = []
    cuda = SimpleNamespace(get_device_properties=lambda device: SimpleNamespace(total_memory=16 * 1024**3),
                           get_per_process_memory_fraction=lambda device: 0.5,
                           set_per_process_memory_fraction=lambda fraction, device: fractions.append((fraction, device)))
    assert _set_cuda_budget(cuda, 4) == 0.5
    assert fractions == [(0.25, 0)]
    fractions.clear()
    assert _set_cuda_budget(cuda, 12) == 0.5
    assert fractions == [(0.5, 0)]

def test_image_rejects_unknown_weight_storage(tmp_path):
    with pytest.raises(ValueError, match='weight storage'):
        validate_image(ImageSettings(prompt='x', out=tmp_path / 'out.png', weight_storage='invalid'))

def test_disk_offload_executes_real_layer_and_preserves_output(tmp_path):
    import torch
    from types import SimpleNamespace
    from pixcli.image import _configure_image_offload
    layer = torch.nn.Linear(3, 2).eval()
    inputs = torch.tensor([[1., 2., 3.]])
    expected = layer(inputs).detach()
    pipe = SimpleNamespace(components={'unet': layer}, _exclude_from_cpu_offload=[])
    mode = _configure_image_offload(pipe, None, None, 'disk', tmp_path, device='cpu')
    assert mode == 'disk_offload'
    assert next(layer.parameters()).device.type == 'meta'
    assert torch.allclose(layer(inputs), expected)
    assert (tmp_path / 'unet' / 'index.json').exists()

def test_auto_image_storage_selects_disk_only_for_ram_budget(tmp_path):
    from pixcli.image import _image_weight_storage
    settings = ImageSettings(prompt='x', out=tmp_path / 'out.png')
    assert _image_weight_storage(settings) == 'memory'
    settings.ram_budget_gib = 8
    assert _image_weight_storage(settings) == 'disk'
    settings.weight_storage = 'memory'
    assert _image_weight_storage(settings) == 'memory'

def test_image_checks_resident_budget_after_final_decode(tmp_path, monkeypatch):
    import contextlib
    import sys
    from types import SimpleNamespace
    from pixcli.image import _infer
    samples = []
    class Guard:
        minimum_physical = minimum_commit = None
        def __init__(self, *args, **kwargs): pass
        def sample(self):
            samples.append('sample')
            if len(samples) == 6:
                raise MemoryError('Image RAM budget exceeded after decode')
        def callback(self, *args): pass
    class Pipeline:
        scheduler = SimpleNamespace(config={})
        @classmethod
        def from_pretrained(cls, *args, **kwargs): return cls()
        def load_lora_weights(self, *args, **kwargs): pass
        def set_adapters(self, *args, **kwargs): pass
        def get_active_adapters(self): return ['pixel']
        def enable_vae_slicing(self): pass
        def enable_vae_tiling(self): pass
        def enable_model_cpu_offload(self): pass
        def __call__(self, **kwargs): return SimpleNamespace(images=[Image.new('RGB', (1, 1))])
    class VAE:
        @classmethod
        def from_pretrained(cls, *args, **kwargs): return cls()
    class Generator:
        def __init__(self, **kwargs): pass
        def manual_seed(self, seed): return self
    class FakeOOM(Exception): pass
    cuda = SimpleNamespace(is_available=lambda: True, reset_peak_memory_stats=lambda: None,
                           empty_cache=lambda: None, OutOfMemoryError=FakeOOM,
                           max_memory_allocated=lambda: 0, max_memory_reserved=lambda: 0,
                           get_device_name=lambda: 'test')
    torch = SimpleNamespace(cuda=cuda, float16=object(), Generator=Generator,
                            inference_mode=contextlib.nullcontext)
    diffusers = SimpleNamespace(AutoencoderKL=VAE, StableDiffusionXLPipeline=Pipeline,
                                StableDiffusionXLImg2ImgPipeline=Pipeline)
    monkeypatch.setitem(sys.modules, 'torch', torch)
    monkeypatch.setitem(sys.modules, 'diffusers', diffusers)
    monkeypatch.setattr('pixcli.image._memory_preflight', lambda paths, **kwargs: {})
    monkeypatch.setattr('pixcli.image._ImageMemoryGuard', Guard)
    monkeypatch.setattr('pixcli.image._assert_lora_coverage', lambda *args: {})
    with pytest.raises(RuntimeError, match='RAM budget exceeded after decode'):
        _infer(ImageSettings(prompt='x', out=tmp_path / 'out.png'),
               {'base': tmp_path, 'vae': tmp_path, 'lora': tmp_path}, None)
