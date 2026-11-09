import importlib.util
import sys
from pathlib import Path
from types import ModuleType, SimpleNamespace

import pytest


@pytest.fixture
def checked_node(monkeypatch):
    comfy = ModuleType('comfy')
    comfy.__path__ = []
    comfy.lora = ModuleType('comfy.lora')
    comfy.model_management = ModuleType('comfy.model_management')
    comfy.model_management.current_loaded_models = []
    comfy.utils = ModuleType('comfy.utils')
    folder_paths = ModuleType('folder_paths')
    folder_paths.get_full_path_or_raise = lambda kind, name: name
    fake_torch = ModuleType('torch')
    fake_torch.device = lambda value: SimpleNamespace(type=value)
    fake_torch.cuda = SimpleNamespace(is_available=lambda: True)
    for name, module in [('comfy', comfy), ('comfy.lora', comfy.lora),
                         ('comfy.model_management', comfy.model_management),
                         ('comfy.utils', comfy.utils), ('folder_paths', folder_paths),
                         ('torch', fake_torch)]:
        monkeypatch.setitem(sys.modules, name, module)
    source = Path(__file__).resolve().parents[1] / 'pixcli/comfy_custom_nodes/pixcli_wan/__init__.py'
    spec = importlib.util.spec_from_file_location('test_pixcli_wan_node', source, submodule_search_locations=[str(source.parent)])
    module = importlib.util.module_from_spec(spec)
    monkeypatch.setitem(sys.modules, 'test_pixcli_wan_node', module)
    spec.loader.exec_module(module)
    monkeypatch.setattr(module, 'report_memory_phase', lambda *args: None)
    return module, comfy, fake_torch


def test_cache_reuses_only_same_path_and_device(checked_node):
    node, comfy, _ = checked_node
    calls = []
    comfy.utils.load_torch_file = lambda path, **kw: calls.append((path, kw)) or object()
    cpu = node._load('adapter', 'cpu')
    assert node._load('adapter', 'cpu') is cpu
    gpu = node._load('adapter', 'cuda')
    assert node._load('adapter', 'cuda') is gpu
    assert [call[1]['device'].type for call in calls] == ['cpu', 'cuda']
    assert all(call[1]['safe_load'] for call in calls)


def test_mapped_cpu_pixel_adapter_uses_read_only_loader(checked_node, monkeypatch):
    node, comfy, _ = checked_node
    from pixcli import mapped_safetensors
    calls = []
    monkeypatch.setattr(mapped_safetensors, 'load_readonly_safetensors',
                        lambda path: (calls.append(path) or {'pixel': 1}, {'format': 'pt'}))
    comfy.utils.load_torch_file = lambda *args, **kwargs: pytest.fail('mapped adapter used ordinary loader')
    path = 'wan2.2_animate_adapter_model.safetensors'
    assert node._load(path, 'cpu', 'mmap') == {'pixel': 1}
    assert node._load(path, 'cpu', 'mmap') == {'pixel': 1}
    assert calls == [path]
    with pytest.raises(ValueError, match='pinned'):
        node._load('untrusted.safetensors', 'cpu', 'mmap')


def test_mapped_distillation_validates_schema_and_preserves_metadata(checked_node, monkeypatch):
    node, comfy, fake_torch = checked_node
    from pixcli import mapped_safetensors
    fake_torch.float32, fake_torch.int64 = object(), object()
    entries = {}
    for index in range(400):
        prefix = f'diffusion_model.blocks.{index}.attention'
        entries[prefix + '.lora_up.weight'] = SimpleNamespace(dtype=fake_torch.float32)
        entries[prefix + '.lora_down.weight'] = SimpleNamespace(dtype=fake_torch.float32)
        entries[prefix + '.alpha'] = SimpleNamespace(dtype=fake_torch.int64)
    metadata = {'source': 'example'}
    monkeypatch.setattr(mapped_safetensors, 'load_readonly_safetensors', lambda path: (entries, metadata))
    comfy.sd = ModuleType('comfy.sd')
    seen = []
    comfy.sd.load_lora_for_models = lambda model, clip, weights, strength, strength_clip, lora_metadata: (
        seen.append((model, weights, strength, lora_metadata)) or 'patched', None)
    monkeypatch.setitem(sys.modules, 'comfy.sd', comfy.sd)
    name = 'wan2.2_i2v_lightx2v_4steps_lora_v1_high_noise.safetensors'
    model = object()
    assert node.PixcliMappedDistillLoRA().load_lora_model_only(model, name, .5) == ('patched',)
    assert seen == [(model, entries, .5, metadata)]
    entries.pop(next(iter(entries)))
    with pytest.raises(ValueError, match='keys'):
        node._load_distill('wan2.2_i2v_lightx2v_4steps_lora_v1_low_noise.safetensors')


def test_node_rejects_unavailable_cuda_and_device_mismatch(checked_node):
    node, comfy, torch = checked_node
    loader = node.PixcliCheckedPixelLoRA()
    torch.cuda.is_available = lambda: False
    with pytest.raises(RuntimeError, match='CUDA.*unavailable'):
        loader.load(None, 'adapter', 1, 'high', 'cuda')
    torch.cuda.is_available = lambda: True
    comfy.utils.load_torch_file = lambda *a, **kw: {'x': SimpleNamespace(device=SimpleNamespace(type='cpu'))}
    with pytest.raises(RuntimeError, match='device mismatch'):
        loader.load(None, 'adapter', 1, 'high', 'cuda')
