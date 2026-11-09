import gc
import hashlib

import torch
from safetensors.torch import save_file


def test_read_only_mapping_preserves_float8_e4m3_bytes(tmp_path):
    from pixcli.mapped_safetensors import load_readonly_safetensors
    path = tmp_path / 'fp8.safetensors'
    data = torch.tensor([0.0, 1.0, -2.0], dtype=torch.float8_e4m3fn)
    save_file({'weight': data}, str(path))
    tensors, _ = load_readonly_safetensors(path)
    assert tensors['weight'].dtype == torch.float8_e4m3fn
    assert torch.equal(tensors['weight'].view(torch.uint8), data.view(torch.uint8))


def test_read_only_mapping_preserves_tensors_after_mapping_owner_is_dropped(tmp_path):
    from pixcli.mapped_safetensors import load_readonly_safetensors
    path = tmp_path / 'adapter.safetensors'
    expected = {'a': torch.arange(12, dtype=torch.float32).reshape(3, 4),
                'b': torch.tensor([[1, -2], [3, 4]], dtype=torch.bfloat16),
                'alpha': torch.tensor(4, dtype=torch.int64)}
    save_file(expected, str(path), metadata={'kind': 'test'})
    before = hashlib.sha256(path.read_bytes()).digest()
    tensors, metadata = load_readonly_safetensors(path)
    assert metadata == {'kind': 'test'}
    assert all(torch.equal(tensors[key], value) for key, value in expected.items())
    a = tensors['a']
    del tensors
    gc.collect()
    assert torch.equal(a, expected['a'])
    assert hashlib.sha256(path.read_bytes()).digest() == before


def test_read_only_mapping_rejects_invalid_tensor_ranges(tmp_path):
    from pixcli.mapped_safetensors import load_readonly_safetensors
    path = tmp_path / 'invalid.safetensors'
    path.write_bytes((2).to_bytes(8, 'little') + b'{}')
    tensors, _ = load_readonly_safetensors(path)
    assert tensors == {}
    path.write_bytes((64).to_bytes(8, 'little') + b'{}')
    import pytest
    with pytest.raises(RuntimeError, match='header'):
        load_readonly_safetensors(path)
