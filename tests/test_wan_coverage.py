import pytest
from pixcli.wan_coverage import adapter_targets


def test_rejects_unmatched_adapter_tensors():
    adapter = {'diffusion_model.blocks.0.self_attn.q.lora_A.weight': (256, 5120),
               'diffusion_model.blocks.0.self_attn.q.lora_B.weight': (5120, 256),
               'unexpected': (1,)}
    with pytest.raises(ValueError, match='unexpected'):
        adapter_targets(adapter, {'diffusion_model.blocks.0.self_attn.q.weight': (5120, 5120)})


def test_rejects_missing_model_targets_and_incompatible_shapes():
    adapter = {'diffusion_model.blocks.0.self_attn.q.lora_A.weight': (256, 5120),
               'diffusion_model.blocks.0.self_attn.q.lora_B.weight': (5120, 256)}
    with pytest.raises(ValueError, match='Missing'):
        adapter_targets(adapter, {})
    with pytest.raises(ValueError, match='shape'):
        adapter_targets(adapter, {'diffusion_model.blocks.0.self_attn.q.weight': (4096, 5120)})


def test_returns_exact_expected_model_keys():
    adapter = {'diffusion_model.blocks.0.self_attn.q.lora_A.weight': (256, 5120),
               'diffusion_model.blocks.0.self_attn.q.lora_B.weight': (5120, 256)}
    assert adapter_targets(adapter, {'diffusion_model.blocks.0.self_attn.q.weight': (5120, 5120)}) == {'diffusion_model.blocks.0.self_attn.q.weight'}

