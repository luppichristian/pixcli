"""Small checks for bounded row-wise UMT5 embedding conversion."""
import importlib.util
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch


def load_module():
    path = Path(__file__).resolve().parents[1] / 'pixcli/comfy_custom_nodes/pixcli_wan/chunked_text.py'
    spec = importlib.util.spec_from_file_location('pixcli_chunked_text_test', path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_dequantizes_embedding_rows_in_bounded_chunks():
    module = load_module()
    data = torch.arange(35, dtype=torch.uint8).reshape(7, 5)
    source = SimpleNamespace(shape=(7, 5), data=data, tensor_type='fake')
    seen = []

    def construct(chunk, *, tensor_type, tensor_shape):
        assert tensor_type == 'fake'
        seen.append(tensor_shape)
        return chunk

    def dequantize(chunk, *, dtype):
        return chunk.to(dtype)

    result = module.chunk_dequantize_embedding(source, construct, dequantize, rows_per_chunk=3)
    assert seen == [(3, 5), (3, 5), (1, 5)]
    assert result.dtype == torch.float16
    assert torch.equal(result, data.to(torch.float16))


@pytest.mark.parametrize('rows', [0, -1])
def test_rejects_invalid_chunk_size(rows):
    module = load_module()
    source = SimpleNamespace(shape=(2, 2), data=torch.zeros(2, 2), tensor_type='fake')
    with pytest.raises(ValueError, match='chunk'):
        module.chunk_dequantize_embedding(source, lambda *a, **kw: None, lambda *a, **kw: None, rows_per_chunk=rows)
