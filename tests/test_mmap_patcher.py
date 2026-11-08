"""Small, model-free checks of the opt-in GGUF mmap-preserving patcher."""
import importlib.util
from pathlib import Path

import pytest


def load_module():
    path = Path(__file__).resolve().parents[1] / 'pixcli/comfy_custom_nodes/pixcli_wan/mmap_loader.py'
    spec = importlib.util.spec_from_file_location('pixcli_mmap_loader_test', path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class BasePatcher:
    def load(self, *args, **kwargs):
        self.events.append(('base_load', args, kwargs))

    def pin_weight_to_device(self, key):
        self.events.append(('base_pin', key))


class GGUFParent(BasePatcher):
    def __init__(self):
        self.events = []
        self.weight_inplace_update = False

    def load(self, *args, **kwargs):
        raise AssertionError('upstream copies mapped weights')

    def pin_weight_to_device(self, key):
        raise AssertionError('upstream copies mapped weights')

    def clone(self):
        # Like pinned GGUFModelPatcher.clone, this returns the upstream class.
        clone = GGUFParent()
        clone.events = self.events
        return clone


def test_mmap_patcher_survives_clone_and_uses_base_load_and_pin():
    module = load_module()
    patcher = GGUFParent()
    patcher.__class__ = module.mmap_patcher_type(GGUFParent, BasePatcher)
    clone = patcher.clone().clone()
    clone.load('cuda', lowvram_model_memory=1, force_patch_weights=False)
    clone.pin_weight_to_device('weight')
    assert clone.events == [('base_load', ('cuda',), {'lowvram_model_memory': 1, 'force_patch_weights': True}),
                            ('base_pin', 'weight')]


def test_mmap_patcher_rejects_in_place_updates():
    module = load_module()
    patcher = GGUFParent()
    patcher.__class__ = module.mmap_patcher_type(GGUFParent, BasePatcher)
    patcher.weight_inplace_update = True
    with pytest.raises(RuntimeError, match='in.place'):
        patcher.load('cuda')
    patcher.weight_inplace_update = False
    with pytest.raises(RuntimeError, match='in.place'):
        patcher.patch_weight_to_device('weight', inplace_update=True)
