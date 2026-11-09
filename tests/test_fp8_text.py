"""Small tests for the mapped FP8 UMT5 text encoder components."""
import importlib.util
from pathlib import Path
from types import SimpleNamespace

import torch


def _module():
    path = Path(__file__).resolve().parents[1] / 'pixcli/comfy_custom_nodes/pixcli_wan/fp8_text.py'
    spec = importlib.util.spec_from_file_location('pixcli_fp8_text_test', path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_mapped_embedding_gathers_the_same_rows_and_stays_on_cpu():
    module = _module()
    ops = SimpleNamespace(Embedding=torch.nn.Embedding)
    cls = module.mapped_fp16_embedding_type(ops.Embedding)
    embedding = cls(8, 3, dtype=torch.float16)
    values = torch.arange(24, dtype=torch.float16).reshape(8, 3)
    embedding.load_state_dict({'weight': values})
    indices = torch.tensor([[3, 1, 3]])
    expected = torch.nn.functional.embedding(indices, values).to(torch.float32)
    embedding.to('meta')
    assert embedding.weight.device.type == 'cpu'
    assert torch.equal(embedding(indices, out_dtype=torch.float32), expected)


def test_mapped_embedding_rejects_unsupported_mutable_lookup():
    import pytest
    module = _module()
    cls = module.mapped_fp16_embedding_type(torch.nn.Embedding)
    with pytest.raises(ValueError, match='max_norm'):
        cls(8, 3, max_norm=1.0)


def test_fp8_graph_uses_mapped_loader_only_for_mapped_weights():
    from pixcli.comfy_backend import build_graph
    args = ('source.png', 'walk', '', 42, 256, 256, 5, 1.0)
    mapped = build_graph(*args, text_encoder='fp8', weight_storage='mmap')
    ordinary = build_graph(*args, text_encoder='fp8', weight_storage='normal')
    assert mapped['text']['class_type'] == 'PixcliMappedFP8UMT5Loader'
    assert ordinary['text']['class_type'] == 'CLIPLoader'


def test_readiness_detects_fp8_loader_drift(tmp_path):
    import hashlib
    import json
    import pytest
    from pixcli.comfy_backend import validate_ready
    lock, node, fp8_node = (tmp_path / name for name in ('lock', 'node', 'fp8_text.py'))
    for path in (lock, node, fp8_node):
        path.write_text(path.name)
    digest = lambda p: hashlib.sha256(p.read_bytes()).hexdigest()
    marker = tmp_path / 'ready.json'
    marker.write_text(json.dumps({'schema_version': 1, 'lock_sha256': digest(lock),
                                  'node_sha256': digest(node), 'fp8_node_sha256': digest(fp8_node), 'commits': {}}))
    validate_ready(marker, lock, node, {}, fp8_node=fp8_node)
    fp8_node.write_text('changed')
    with pytest.raises(RuntimeError, match='drift'):
        validate_ready(marker, lock, node, {}, fp8_node=fp8_node)
