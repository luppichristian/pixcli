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


def test_mapped_embedding_dequantizes_only_requested_rows():
    module = load_module()
    packed = torch.arange(35, dtype=torch.uint8).reshape(7, 5)
    source = SimpleNamespace(shape=(7, 5), data=packed, tensor_type='fake')
    seen = []
    def construct(chunk, *, tensor_type, tensor_shape):
        seen.append((chunk.clone(), tensor_shape))
        return chunk
    def dequantize(chunk, *, dtype):
        return chunk.to(dtype)
    tokens = torch.tensor([[6, 1, 6], [0, 3, 1]])
    result = module.gather_dequantize_embedding(source, tokens, construct, dequantize, dtype=torch.float32)
    assert torch.equal(result, packed[tokens].to(torch.float16).to(torch.float32))
    assert seen[0][0].shape == (6, 5)
    assert seen[0][1] == (6, 5)


def test_mapped_embeddings_assign_nonquantized_state_without_meta_copy():
    module = load_module()
    class BaseEmbedding(torch.nn.Embedding):
        def ggml_load_from_state_dict(self, state_dict, prefix, local_metadata, strict,
                                      missing_keys, unexpected_keys, error_msgs):
            self.weight = torch.nn.Parameter(state_dict[prefix + 'weight'], requires_grad=False)
    class Ops:
        Embedding = BaseEmbedding
    custom = module.mapped_umt5_ops(Ops, None, None)
    embedding = custom.Embedding(3, 2)
    weights = torch.arange(6, dtype=torch.float32).reshape(3, 2)
    embedding.load_state_dict({'weight': weights})
    assert embedding.weight.device.type == 'cpu'
    assert torch.equal(embedding.weight, weights)
    embedding._apply(lambda weight: (_ for _ in ()).throw(AssertionError('weight must remain mapped on CPU')))
    assert torch.equal(embedding.weight, weights)
