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
